"""APA-P1-D: stage importable arm packages (copies, never symlinks).

Arm A = e2fb85a1's Python package (read by `git show`) + Build A's module.
Arm B = this round's Python package + Build B's module.
Prior art: none; file copies with sha256 pins (P1 A1 refs/ layout, 2026).
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/apa_p1d'
BUILDS = {'A': ART / 'build_A_r2', 'B': ART / 'build_B'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    arm = sys.argv[1]
    version = sys.argv[2] if len(sys.argv) > 2 else '1'
    build = json.loads((BUILDS[arm] / 'BUILD.json').read_text())
    assert build['status'] == 'PASS'
    dest = ART / 'arms' / f'{arm}{"" if version == "1" else "_v" + version}' / 'tensor_cuda'
    dest.mkdir(parents=True)
    files = {}
    if arm == 'A':
        names = subprocess.run(['git', '-C', str(ROOT), 'ls-tree', '-r', '--name-only', 'e2fb85a1',
                                'tensor_cuda/tensor_cuda/'], capture_output=True, text=True, check=True).stdout.split()
        for name in names:
            data = subprocess.run(['git', '-C', str(ROOT), 'show', f'e2fb85a1:{name}'],
                                  capture_output=True, check=True).stdout
            out = dest / Path(name).relative_to('tensor_cuda/tensor_cuda')
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            files[name] = sha(out)
    else:
        src = ROOT / 'tensor_cuda/tensor_cuda'
        for path in sorted(src.rglob('*.py')):
            if '__pycache__' in path.parts:
                continue
            out = dest / path.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, out)
            files[str(path.relative_to(ROOT))] = sha(out)
            assert sha(path) == files[str(path.relative_to(ROOT))]
    module = ROOT / build['module']
    assert sha(module) == build['module_sha256']
    shutil.copyfile(module, dest / module.name)
    record = dict(arm=arm, version=version, package=str(dest.relative_to(ROOT)), python_files=files,
                  module=str((dest / module.name).relative_to(ROOT)), module_sha256=sha(dest / module.name),
                  build_record=str((BUILDS[arm] / 'BUILD.json').relative_to(ROOT)))
    assert record['module_sha256'] == build['module_sha256']
    (dest.parent / 'ARM.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
