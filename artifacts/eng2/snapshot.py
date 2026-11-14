#!/usr/bin/env python3
"""ENG2 G1 (seat form, no git): whole-file sha256 map of the worktree.

Usage: snapshot.py OUT.json
Walks the worktree root (this file's parents[2]) without following symlinks, skipping only Python caches
(__pycache__, .pytest_cache) and the worktree's .git pointer file. Symlinks are recorded as "symlink:<target>".
Prior art: manifest diffing as in BSD mtree(8) (year unverified -- lead to check "mtree BSD"); SHA-256, NIST FIPS
180-4 (2015). Ours: the skip list and the JSON shape.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {"__pycache__", ".pytest_cache"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(out):
    files = {}
    for dirpath, dirnames, filenames in os.walk(ROOT, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for f in sorted(filenames):
            p = Path(dirpath) / f
            rel = str(p.relative_to(ROOT))
            if rel == ".git":
                continue
            if p.is_symlink():
                files[rel] = "symlink:" + os.readlink(p)
            else:
                files[rel] = sha(p)
    Path(out).write_text(json.dumps(dict(root=str(ROOT), n=len(files), files=files), indent=0, sort_keys=True) + "\n")
    print(out, len(files))


if __name__ == "__main__":
    main(sys.argv[1])
