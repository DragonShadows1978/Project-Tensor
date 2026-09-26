#!/usr/bin/env python3
"""PT-DET-1 CPU-only, offline fork build; adapted from PT-TF32-1. Prior art: BP-KERNEL-3/4 (2026) CMake
recipe and SHA256 receipts, taken; ours: isolated dependency and source pins.
"""
from pathlib import Path
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import pt_tf32_3_storage as storage

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/pt_det_1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if ROOT != Path('/mnt/ForgeRealm/wt/pt-tf32'):
        raise RuntimeError('build is authorized only in the PT-DET-1 fork')
    reg = ART / 'REGISTRATION.json'
    if sha(reg) != (ART / 'REGISTRATION.sha256').read_text().strip():
        raise RuntimeError('registration drift')
    from pt_det_2 import registration
    registration()  # Additive registration; historical PT-DET-1 plan is unchanged.
    index = 1
    while (ART / f'build_{index:02d}').exists():
        index += 1
    receipt_dir = ART / f'build_{index:02d}'
    receipt_dir.mkdir()
    check=storage.space_status(ROOT)
    if check['status']!='GREEN':
        (receipt_dir/'receipt.json').write_text(json.dumps(dict(check,verdict='BLOCKED'),indent=2)+'\n')
        print('BUILD BLOCKED',json.dumps(check));return 2
    # All generated compiler temporaries and caches stay under the fork.
    temp = ART / 'tmp'
    temp.mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               TMPDIR=str(temp), CUDA_CACHE_PATH=str(ART / 'cuda_cache'))
    commands = [
        ['cmake', '-S', 'tensor_cuda', '-B', 'tensor_cuda/build-tf32',
         '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_ARCHITECTURES=89',
         '-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc',
         '-DFETCHCONTENT_SOURCE_DIR_PYBIND11=' + str(ROOT / 'artifacts/pt_tf32_1/vendor/pybind11'),
         '-DFETCHCONTENT_FULLY_DISCONNECTED=ON'],
        ['cmake', '--build', 'tensor_cuda/build-tf32', '-j4'],
    ]
    sources = sorted((ROOT / 'tensor_cuda/src').glob('*'))
    sources += sorted((ROOT / 'tensor_cuda/include/tc').glob('*'))
    sources += [ROOT / 'tensor_cuda/CMakeLists.txt', ROOT / 'tests/pt_det_1_host_contract.cpp',
                ROOT / 'tests/pt_det_2_host_contract.cpp']
    pins = {str(p.relative_to(ROOT)): sha(p) for p in sources if p.is_file()}
    results = []
    start = time.monotonic()
    rc = 1
    with (receipt_dir / 'build.log').open('x') as log:
        for command in commands:
            print('BUILD', ' '.join(command), flush=True)
            try:
                run = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     timeout=max(1, 480 - (time.monotonic() - start)))
                rc = run.returncode
            except subprocess.TimeoutExpired:
                rc = 124
                log.write('\nBUILD_TIMEOUT=480s\n')
            log.flush()
            results.append(dict(argv=command, returncode=rc))
            if rc:
                break
    if rc == 0:
        # pybind11 hides internal engine symbols in the module. Link the CPU
        # contract executable against the same object files, including dlink.
        # Taken: CMake link recipe; ours: executable main instead of module.
        build = ROOT / 'tensor_cuda/build-tf32'
        for name in ('pt_det_1', 'pt_det_2'):
            command = shlex.split((build / 'CMakeFiles/_tensor_cuda.dir/link.txt').read_text())
            command.remove('-shared')
            command[command.index('-o') + 1] = str(ROOT / f'artifacts/{name}/{name}_host_contract')
            command += ['-std=c++17', '-pthread', '-I' + str(ROOT / 'tensor_cuda/include'),
                        str(ROOT / f'tests/{name}_host_contract.cpp'),
                        '-L/usr/lib/x86_64-linux-gnu', '-lpython3.12']
            with (receipt_dir / 'build.log').open('a') as log:
                run = subprocess.run(command, cwd=build, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=60)
            rc = run.returncode
            results.append(dict(argv=command, returncode=rc))
            if rc: break
    unchanged = all(sha(ROOT / p) == h for p, h in pins.items())
    binaries = [dict(path=str(p.relative_to(ROOT)), sha256=sha(p), bytes=p.stat().st_size)
                for p in (ROOT / 'tensor_cuda/tensor_cuda').glob('_tensor_cuda*.so')] if rc == 0 else []
    receipt = dict(evidence_class='CPU compilation only; no GPU use or engine import',
                   registration_sha256=sha(reg), commands=results, rc=rc,
                   pt_det_2_registration_sha256=sha(ROOT/'artifacts/pt_det_2/REGISTRATION.json'),
                   elapsed_seconds=time.monotonic() - start, source_pins=pins,
                   sources_unchanged_during_build=unchanged, binaries=binaries)
    (receipt_dir / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print((receipt_dir / 'build.log').read_text()[-7000:])
    print(f'BUILD_RC={rc} SOURCE_PINS_UNCHANGED={unchanged} RECEIPT={receipt_dir / "receipt.json"}')
    return rc or (0 if unchanged and len(binaries) == 1 else 1)


if __name__ == '__main__':
    sys.exit(main())
