"""F2 and incremental one-load reverts.

Prior art: PF2-F1's CMake adapter and incremental clone (Project-Tensor,
2026), CMake (Kitware, 2000), and mutation testing (DeMillo et al., 1978).
F2 isolates each decode revert in its own build directory.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/apa_pf2f2'
F1 = ROOT.parent / 'tensor_fork_apa_wt_p1pf2/artifacts/apa_pf2f1/build_F1'
F1_SHA = '641a3139181cdee3e800f62afad08c0881b3018c26e57d953fa6c13413f7cbe7'
F1_RECEIPT_SHA = '7827ce708e92a3f36539058a091b25f9f55eded0cf2a6b9b3dcbbf1654ddd242'
HEADER = 'tensor_cuda/src/apa_spo/packed.cuh'
BULK = 'pf2m01_bulk_norm_finite_only'
SHARED = 'pf2m02_shared_norm_finite_only'
MUTANTS = {BULK: 'bulk', SHARED: 'bulk_shared'}
COMMENT = "// PF2-F1 (Project-Tensor, 2026) and IEEE 754-2008: reuse prefill's quiet-NaN sentinel at the decode load."
SHARED_EDIT = f'   {COMMENT}\n   n=isfinite(n)?n:__int_as_float(0x7fc00000);\n'
BULK_EDIT = f"""    if(tid<n){{
      const float norm=scales[(int64_t)bh*capacity+start+t+tid];
      {COMMENT}
      norms[tid]=isfinite(norm)?norm:__int_as_float(0x7fc00000);
    }}
    __syncthreads();"""
BULK_ORIGINAL = '    if(tid<n)norms[tid]=scales[(int64_t)bh*capacity+start+t+tid];__syncthreads();'
REVERT = {BULK: (BULK_EDIT, BULK_ORIGINAL), SHARED: (SHARED_EDIT, '')}


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as file:
        json.dump(data, file, indent=2, allow_nan=False)
        file.write('\n')


def receipt(label):
    if label == 'F1':
        assert sha(F1 / 'BUILD.json') == F1_RECEIPT_SHA
        row = read(F1 / 'BUILD.json')
        row['module'] = str(F1.parents[2] / row['module'])
        assert row['module_sha256'] == F1_SHA
    else:
        row = read(ART / f'build_{label}' / 'BUILD.json')
    assert row['status'] == 'PASS'
    return row


def disk_free():
    free = shutil.disk_usage('/mnt/ForgeRealm').free
    assert free >= 40_000_000_000, f'RAIL: disk_free_bytes={free}'
    return free


def run(out, name, command, limit):
    argv = ['timeout', f'{limit}s', 'nice', '-n', '19', *command]
    log = out / f'{name}.log'
    started = time.monotonic()
    minimum = disk_free()
    with log.open('x') as stream:
        proc = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        while True:
            try:
                code = proc.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                try:
                    minimum = min(minimum, disk_free())
                except AssertionError:
                    proc.terminate()
                    proc.wait(timeout=20)
                    code = 6
                    break
    return dict(argv=argv, exit=code, seconds=time.monotonic()-started,
                log=str(log.relative_to(ROOT)), log_sha256=sha(log),
                min_disk_free_bytes=minimum)


def f2_inputs(out, flags):
    cmake_source = out / 'cmake_source'
    cmake_source.mkdir()
    source = ROOT / 'tensor_cuda'
    cmake = (source / 'CMakeLists.txt').read_text()
    cmake = cmake.replace('  src/', f'  {source}/src/')
    cmake = cmake.replace('${CMAKE_CURRENT_SOURCE_DIR}/include', str(source / 'include'))
    cmake = cmake.replace('${CMAKE_CURRENT_SOURCE_DIR}/tensor_cuda', str(out / 'lib'))
    (cmake_source / 'CMakeLists.txt').write_text(cmake)
    return ['cmake', '-S', str(cmake_source), '-B', str(out / 'tree'), *flags]


def mutant_inputs(out, label):
    base = ART / 'build_F2'
    shutil.copytree(base / 'tree', out / 'tree', symlinks=True)
    shutil.copytree(base / 'cmake_source', out / 'cmake_source')
    shutil.copytree(base / 'lib', out / 'lib')
    source = out / 'source' / 'tensor_cuda'
    for sub in ('src', 'include'):
        shutil.copytree(ROOT / 'tensor_cuda' / sub, source / sub)
    substitutions = [(str(base), str(out))]
    for sub in ('src', 'include'):
        substitutions.append((str(ROOT / 'tensor_cuda' / sub), str(source / sub)))
    for folder in (out / 'tree', out / 'cmake_source'):
        for path in folder.rglob('*'):
            if not path.is_file() or path.is_symlink() or path.suffix in ('.o', '.so', '.a', '.bin'):
                continue
            try:
                before = path.read_text()
            except UnicodeDecodeError:
                continue
            after = before
            for old, new in substitutions:
                after = after.replace(old, new)
            if after != before:
                stat = path.stat()
                path.write_text(after)
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    obj = out / 'tree/CMakeFiles/_tensor_cuda.dir'
    old = obj / str(ROOT / 'tensor_cuda/src').lstrip('/')
    new = obj / str(source / 'src').lstrip('/')
    new.parent.mkdir(parents=True, exist_ok=True)
    old.rename(new)
    header = source / 'src/apa_spo/packed.cuh'
    before = header.read_text()
    old, new = REVERT[label]
    assert before.count(old) == 1
    expected = before.replace(old, new, 1)
    header.write_text(expected)
    # The declaration in the alternate kernel is still exactly the F2 text.
    other = next(k for k in MUTANTS if k != label)
    assert expected.count(REVERT[other][0]) == 1
    assert sha(next((out / 'lib').glob('_tensor_cuda*.so'))) == receipt('F2')['module_sha256']
    return source.parent


def input_map(base, root):
    return {name: sha(root / name if (root / name).exists() else ROOT / name)
            for name in base['source_sha256_after']}


def build(label):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert os.environ['TMPDIR'].startswith('/tmp/pf2f2_')
    assert label == 'F2' or label in MUTANTS
    disk_free()
    f1 = receipt('F1')
    assert sha(f1['module']) == F1_SHA
    out = ART / f'build_{label}'
    out.mkdir()
    rec = dict(label=label, status='FAIL', hooks=False, commands=[],
               p1d_flags=f1['p1d_flags'], registration_sha256=sha(ART / 'REGISTRATION.md'))
    try:
        parent = f1 if label == 'F2' else receipt('F2')
        rec['base_module_sha256'] = parent['module_sha256']
        if label == 'F2':
            source = ROOT
            rec['commands'].append(run(out, 'configure', f2_inputs(out, f1['p1d_flags']), 300))
            assert rec['commands'][-1]['exit'] == 0
        else:
            source = mutant_inputs(out, label)
            rec['mutation'] = dict(kernel=MUTANTS[label], before=REVERT[label][0], after=REVERT[label][1])
        rec['source_root'] = str(source)
        rec['source_sha256'] = input_map(f1, source)
        changed = {k for k, v in rec['source_sha256'].items() if v != parent['source_sha256_after'][k]}
        assert changed == {HEADER}, changed
        rec['commands'].append(run(out, 'build' if label == 'F2' else 'build_incremental',
                                   ['cmake', '--build', str(out / 'tree'), '-j3'], 3200))
        result = rec['commands'][-1]
        assert result['exit'] == 0, result
        rec['rebuilt_objects'] = [s.strip() for s in (ROOT / result['log']).read_text().splitlines()
                                  if 'Building' in s]
        if label in MUTANTS:
            assert len(rec['rebuilt_objects']) == 1 and 'kernels.cu.o' in rec['rebuilt_objects'][0], rec['rebuilt_objects']
        rec['source_sha256_after'] = input_map(f1, source)
        assert rec['source_sha256'] == rec['source_sha256_after']
        module, = (out / 'lib').glob('_tensor_cuda*.so')
        rec['module'] = str(module.relative_to(ROOT))
        rec['module_sha256'] = sha(module)
        assert rec['module_sha256'] != parent['module_sha256']
        rec['status'] = 'PASS'
    except BaseException as error:
        rec['error'] = repr(error)
        raise
    finally:
        save(out / 'BUILD.json', rec)
        with (ART / 'LEDGER.md').open('a') as ledger:
            ledger.write(f"- Build {label} {rec['status']}, {out.relative_to(ROOT)}/BUILD.json. "
                         f"Module {rec.get('module_sha256')}.\n")
    print(label, rec['status'], rec['module_sha256'], flush=True)


if __name__ == '__main__':
    build(sys.argv[1])
