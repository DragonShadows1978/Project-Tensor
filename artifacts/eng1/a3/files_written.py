#!/usr/bin/env python3
"""Write a3/FILES_WRITTEN.json: sha256 of every file this round wrote or changed in the worktree, plus the retained
/tmp/eng1_a3_* scratch roots. Prior art: earlier ENG1 FILES_WRITTEN.json inventories (2026); SHA-256 FIPS 180-4.
No algorithm; no prior art beyond these known to me.
"""
import hashlib
import json
from pathlib import Path

A3 = Path(__file__).resolve().parent
ROOT = A3.parents[2]
T = ROOT / 'artifacts/eng1/tool'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
paths = [T / 'evidence.py', T / 'staging_seal_a3.json']
paths += sorted(p for p in (T / 'gpu_runs/a3_dry_1').rglob('*') if p.is_file())
paths += sorted(p for p in A3.rglob('*') if p.is_file() and p.name != 'FILES_WRITTEN.json'
                and '__pycache__' not in p.parts)
rows = [dict(path=str(p.relative_to(ROOT)), sha256=sha(p),
             kind='changed' if p == T / 'evidence.py' else 'new') for p in paths]
scratch = sorted(str(p) for p in Path('/tmp').glob('eng1_a3_*'))
out = dict(worktree_files=rows, worktree_count=len(rows), scratch_roots=scratch,
           note='FILES_WRITTEN.json itself omitted (self-hash); REPORT.md not written (harness refused)')
with (A3 / 'FILES_WRITTEN.json').open('x') as f:
    f.write(json.dumps(out, indent=1) + '\n')
print(len(rows), sha(A3 / 'FILES_WRITTEN.json'))
