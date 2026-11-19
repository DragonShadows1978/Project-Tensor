"""Worktree hashes and per-file inventory (PF2-F2, Project-Tensor, 2026)."""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/apa_pf2f3'
PRODUCT = {
    'tensor_cuda/tests/test_apa_pf2_nonfinite.py',
    'tensor_cuda/src/apa_spo/spq4.cuh',
    'tensor_cuda/src/apa_spo/spq6.cuh',
}
LIVE_LOG = 'logs/apa_pf2f3_r1.log'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def inventory(root, excluded=()):
    files = {}
    for folder, dirs, names in os.walk(root):
        path = Path(folder)
        dirs[:] = [d for d in sorted(dirs) if str((path / d).relative_to(root)) not in excluded]
        for name in sorted(names + [d for d in dirs if (path / d).is_symlink()]):
            entry = path / name
            relative = str(entry.relative_to(root))
            if relative in excluded:
                continue
            status = entry.lstat()
            if entry.is_symlink():
                files[relative] = [status.st_mode, os.readlink(entry)]
            elif stat.S_ISREG(status.st_mode):
                files[relative] = [status.st_mode, sha(entry)]
            else:
                files[relative] = [status.st_mode, status.st_rdev]
    return files


def authorized(name):
    path = Path(name)
    return (name.startswith('artifacts/apa_pf2f3/') or
            (path.parent == Path('scripts') and path.name.startswith('apa_pf2f3_') and path.suffix == '.py') or
            (path.parent == Path('logs') and path.name.startswith('apa_pf2f3_')))


def main():
    snapshot = OUT / 'SCOPE_BEFORE.json'
    before_excluded = ('artifacts/apa_pf2f3', LIVE_LOG)
    if sys.argv[1] == 'before':
        write(snapshot, inventory(ROOT, before_excluded))
        return
    before = json.loads(snapshot.read_text())
    after = inventory(ROOT, before_excluded)
    changed = sorted(name for name in before.keys() & after.keys() if before[name] != after[name])
    missing = sorted(before.keys() - after.keys())
    added = sorted(after.keys() - before.keys())
    good = not missing and set(changed) == PRODUCT and all(map(authorized, added))
    verdict = dict(status='PASS' if good else 'FAIL', changed=changed, added=added,
                   missing=missing, dispatcher_owned_log=LIVE_LOG)
    write(OUT / 'SCOPE.json', verdict)
    print(json.dumps(verdict, indent=2))
    assert good


if __name__ == '__main__':
    main()
