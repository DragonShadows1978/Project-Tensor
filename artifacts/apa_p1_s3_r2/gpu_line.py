"""Lead-only test dispatch and isolated derivative mutant builds.
Prior art: CMake (Kitware, 2000), pytest, P1 isolated builds (2026), DeMillo,
Lipton, Sayward (1978) registered mutation testing. No numerical changes to
correct sources. This driver never treats arbitrary errors as mutant kills.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / 'artifacts/apa_p1_s3_r2'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mutant_build(case):
    dest = A / 'mutants' / case
    dest.mkdir(parents=True, exist_ok=True)
    receipt = dest / 'BUILD.json'
    if receipt.exists():
        r = json.loads(receipt.read_text())
        assert r['status'] == 'PASS' and sha(r['module']) == r['module_sha256']
        return Path(r['refs'])
    src = dest / 'src'
    shutil.copytree(ROOT / 'tensor_cuda/src', src, ignore=shutil.ignore_patterns('__pycache__'))
    defines = []
    if case == 'eb3_early':
        defines = ['APA_EB3_MUTANT=3']
    elif case == 'eb4_early':
        p = src / 'apa_spo/eb4.cuh'; text = p.read_text()
        # EB4's analogue is NOT APA_EB4_MUTANT=3 (that drops V scale).
        start = text.index(' select_mixed(')
        i = text.index('last_s=prev==(unsigned)(NC-1);', start)
        text = text[:i] + text[i:].replace('last_s=prev==(unsigned)(NC-1);', 'last_s=prev==(unsigned)(NC-2);', 1)
        p.write_text(text)
    elif case == 'eb4_reverse_tie':
        defines = ['APA_EB4_MUTANT=7']
    else:
        raise ValueError(case)
    original = (A / 'cmake_source/CMakeLists.txt').read_text()
    cmake = original.replace(str(ROOT / 'tensor_cuda/src'), str(src)).replace(str(A / 'build/lib'), str(dest / 'build/lib'))
    if defines: cmake += '\ntarget_compile_definitions(_tensor_cuda PRIVATE ' + ' '.join(defines) + ')\n'
    cmake_src = dest / 'cmake_source'; cmake_src.mkdir()
    (cmake_src / 'CMakeLists.txt').write_text(cmake)
    refs = dest / 'refs'; (refs / 'tensor_cuda').mkdir(parents=True)
    for p in (A / 'refs/after/tensor_cuda').rglob('*.py'):
        out = refs / 'tensor_cuda' / p.relative_to(A / 'refs/after/tensor_cuda')
        out.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, out)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', TMPDIR='/tmp/apap1s3r2_mutant_' + case)
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    commands = [
        ['cmake', '-S', str(cmake_src), '-B', str(dest / 'build'), '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_ARCHITECTURES=89', '-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc', '-DCUDAToolkit_ROOT=/usr/local/cuda-12.6', '-DFETCHCONTENT_SOURCE_DIR_PYBIND11=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src', '-DFETCHCONTENT_FULLY_DISCONNECTED=ON', '-DTC_APA_PACKED=ON'],
        ['cmake', '--build', str(dest / 'build'), '-j2']]
    for i, cmd in enumerate(commands):
        with (dest / f'build_{i}.log').open('x') as log:
            subprocess.run(['timeout', '60s' if i == 0 else '440s', *cmd], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    module = next((dest / 'build/lib').glob('_tensor_cuda*.so'))
    (refs / 'tensor_cuda' / module.name).symlink_to(module)
    r = dict(status='PASS',case=case,module=str(module),module_sha256=sha(module),refs=str(refs),defines=defines,cuda_visible_devices='',commands=commands)
    receipt.write_text(json.dumps(r,indent=2)+'\n')
    return refs


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--case', required=True)
    args = parser.parse_args(); case = args.case
    # Called only by the lead inside a real forge-gpu-run lease. CPU dry runs
    # stop at forge-gpu-run with 78, before this program executes.
    if os.environ.get('CUDA_VISIBLE_DEVICES') == '':
        print('FIRST UNMET NEED: GPU visibility is empty', file=sys.stderr); return 2
    expected = json.loads((A / 'EXPECTED.json').read_text())
    for p, digest in expected['code_sha256'].items(): assert sha(ROOT / p) == digest, ('source drift', p)
    record = next(r for r in expected['lines'] if r['case'] == case)
    refs = mutant_build(case) if record['mutant'] else A / 'refs/after'
    module = next((refs / 'tensor_cuda').glob('_tensor_cuda*.so'))
    if not record['mutant']:
        assert sha(module) == json.loads((A / 'BUILD_COMPLETED.json').read_text())['module_sha256']
    out = A / 'lead_gpu' / case; out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, APA_P1_GPU='1', APA_P1_MUTANT=case, TC_APA_SP='1', APA_FXTC1_VARIANT='2', PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONPATH=str(refs)+':'+str(ROOT/'tensor_cuda/tests'), TMPDIR='/tmp/apap1s3r2_gpu_'+case, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    cmd = [sys.executable, '-B', str(ROOT/'artifacts/apa_p1_a1/pytest_receipt.py'), str(out/'pytest.json'), '-q', '--tb=short', '-ra', '-p', 'no:cacheprovider', '--basetemp', env['TMPDIR']+'/pytest', *record['tests']]
    with (out / 'pytest.log').open('x') as log:
        result = subprocess.run(['timeout','160s',*cmd], env=env, stdout=log, stderr=subprocess.STDOUT)
    p = json.loads((out/'pytest.json').read_text())
    calls = [r for rows in p['reports'].values() for r in rows if r['when']=='call']
    failures = [r for r in calls if r['outcome']=='failed']
    errors = [r for rows in p['reports'].values() for r in rows if r['when']!='call' and r['outcome']!='passed']
    valid = (len(calls)==len(record['tests']) and not errors and not p['collection_errors'] and not any(r['outcome']=='skipped' for r in calls))
    if record['mutant']:
        valid = valid and result.returncode==1 and len(failures)==1 and record['failure_marker'] in failures[0]['message']
    else: valid = valid and result.returncode==0 and not failures
    (out/'RESULT.json').write_text(json.dumps(dict(case=case,test_exit=result.returncode,expected_exit=record['expected_exit'],valid_expected_behavior=valid,pytest=cmd),indent=2)+'\n')
    if not valid:
        print('HARNESS/UNEXPECTED FAILURE (never a mutant kill)', file=sys.stderr); return 2
    return result.returncode


if __name__=='__main__': raise SystemExit(main())
