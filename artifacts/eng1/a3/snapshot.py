#!/usr/bin/env python3
"""A3 G3 aid: whole-file sha256 map of the worktree (no git; the dispatch forbids every git command).
Usage: snapshot.py OUT.json
Excludes only Python caches (__pycache__, *.pyc) and the worktree's .git pointer file is hashed like any file.
Symlinks are recorded as 'symlink:<target>' and never followed.
Prior art: SHA-256 (NIST FIPS 180-4, 2015); file-tree manifests in the style of mtree(8) (BSD, 1989; year unverified
-- lead to check 'mtree BSD 4.3 Reno') and earlier ENG1 tracked_before.json / tool_before.json. Ours: nothing new;
a plain path -> digest map used as the no-git stand-in for G3's `git diff 97edd1fd` identity rule (L09).
"""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]


def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main(out):
    rows = {}
    for dirpath, dirnames, filenames in os.walk(ROOT, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d != '__pycache__' and not os.path.islink(os.path.join(dirpath, d)))
        for d in os.listdir(dirpath):
            full = os.path.join(dirpath, d)
            if os.path.islink(full) and os.path.isdir(full):
                rows[os.path.relpath(full, ROOT)] = 'symlink:' + os.readlink(full)
        for f in sorted(filenames):
            full = os.path.join(dirpath, f)
            rel = os.path.relpath(full, ROOT)
            if f.endswith('.pyc'):
                continue
            rows[rel] = ('symlink:' + os.readlink(full)) if os.path.islink(full) else digest(full)
    with open(out, 'x') as f:
        f.write(json.dumps(dict(root=str(ROOT), n=len(rows), files=rows), indent=0, sort_keys=True) + '\n')
    print(out, len(rows))


if __name__ == '__main__':
    main(sys.argv[1])
