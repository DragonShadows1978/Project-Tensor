"""APA-P1-D builds A and B, using the A1 build script's path adapter.

Prior art: CMake (Kitware, 2000), nvcc (NVIDIA); the path adapter is copied in
behaviour from artifacts/apa_p1_a1/BUILD.py (Project-Tensor P1, 2026): only the
source, include and output paths are rewritten, no compiler flag is changed.
No prior art known to me beyond that; this is plumbing.

Usage (CPU only, CUDA_VISIBLE_DEVICES empty):
  python3 scripts/apa_p1d_build.py A   # e2fb85a1 source, -DTC_APA_PACKED=ON
  python3 scripts/apa_p1d_build.py B   # this round's tree, no feature -D flag
"""
from pathlib import Path
import datetime
import difflib
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/apa_p1d'
SOURCES = ['tensor_cuda/CMakeLists.txt'] + sorted(
    str(p.relative_to(ROOT)) for d in ('tensor_cuda/src', 'tensor_cuda/include')
    for p in (ROOT / d).rglob('*') if p.is_file())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    arm = sys.argv[1]
    attempt = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    assert arm in ('A', 'B') and attempt >= 1
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    os.environ.update(PYTHONDONTWRITEBYTECODE='1')
    assert os.environ.get('TMPDIR', '').startswith('/tmp/apap1d_')
    Path(os.environ['TMPDIR']).mkdir(exist_ok=True)
    # Attempt 1 is build_<arm>; a repaired attempt n writes build_<arm>_r<n>
    # and leaves earlier attempts in place (no deletes).
    build = OUT / (f'build_{arm}' if attempt == 1 else f'build_{arm}_r{attempt}')
    source = build / 'cmake_source'
    source.mkdir(parents=True)
    git = subprocess.run(['git', '-C', str(ROOT), 'diff', '--stat', 'e2fb85a1', '--', 'tensor_cuda/src',
                          'tensor_cuda/include', 'tensor_cuda/CMakeLists.txt'],
                         capture_output=True, text=True)
    if arm == 'A':
        # Build A must be e2fb85a1's source: the tree is checked unchanged first.
        assert git.returncode == 0 and git.stdout == '', git.stdout
    original = (ROOT / 'tensor_cuda/CMakeLists.txt').read_text()
    adapted = original.replace('  src/', '  ' + str(ROOT / 'tensor_cuda/src') + '/')
    adapted = adapted.replace('${CMAKE_CURRENT_SOURCE_DIR}/include', str(ROOT / 'tensor_cuda/include'))
    adapted = adapted.replace('${CMAKE_CURRENT_SOURCE_DIR}/tensor_cuda', str(build / 'lib'))
    (source / 'CMakeLists.txt').write_text(adapted)
    (build / 'BUILD_PATH_ADAPTER.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), adapted.splitlines(True),
        fromfile='tree/CMakeLists.txt', tofile=f'build_{arm}/cmake_source/CMakeLists.txt')))
    flags = ['-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_ARCHITECTURES=89',
             '-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc',
             '-DCUDAToolkit_ROOT=/usr/local/cuda-12.6',
             '-DFETCHCONTENT_SOURCE_DIR_PYBIND11=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src',
             '-DFETCHCONTENT_FULLY_DISCONNECTED=ON', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']
    if arm == 'A':
        flags.append('-DTC_APA_PACKED=ON')
    commands = [['cmake', '-S', str(source), '-B', str(build / 'tree'), *flags],
                ['cmake', '--build', str(build / 'tree'), '-j3']]
    record = dict(arm=arm, attempt=attempt, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  git_diff_stat_vs_e2fb85a1=git.stdout,
                  source_sha256={p: sha(ROOT / p) for p in SOURCES},
                  feature_flags=['-DTC_APA_PACKED=ON'] if arm == 'A' else [],
                  note='toolchain/path -D flags are the A1 script\'s, identical on both arms',
                  status='RUNNING', commands=[])
    path = build / 'BUILD.json'
    path.write_text(json.dumps(record, indent=2) + '\n')
    for i, cmd in enumerate(commands):
        start = time.monotonic()
        log = build / ('configure.log' if i == 0 else 'build.log')
        with log.open('x') as f:
            code = subprocess.run(['timeout', '900s' if i == 0 else '2700s', *cmd],
                                  stdout=f, stderr=subprocess.STDOUT).returncode
        record['commands'].append(dict(argv=cmd, exit=code, seconds=time.monotonic() - start, log=str(log)))
        if code:
            record['status'] = 'FAIL'
            break
    else:
        module = next((build / 'lib').glob('_tensor_cuda*.so'))
        record.update(status='PASS', module=str(module.relative_to(ROOT)), module_sha256=sha(module))
        cache = (build / 'tree/CMakeCache.txt').read_text()
        record['cmake_cache_TC_APA_PACKED'] = [l for l in cache.splitlines() if l.startswith('TC_APA_PACKED')]
        record['cmake_cache_hooks'] = [l for l in cache.splitlines() if 'TEST_HOOKS' in l]
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'source_sha256'}, indent=2))
    sys.exit(0 if record['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
