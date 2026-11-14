#!/usr/bin/env python3
"""Foreground CPU receipt supervisor for A3 (adapted from a2/run.py).
Usage: run.py LABEL [KEY=VALUE ...] -- CMD...
Prior art: a2/run.py, REV2 run.py / ENG1 L45 (2026), GNU timeout and Python subprocess. Taken: completed-child
recording with stdout/stderr/exit receipts opened with 'x'. A3 changes: destination a3/, scratch /tmp/eng1_a3_work,
cache variables pointed at /tmp/eng1_a3_*, a 590 s bound (each command under 10 minutes), explicit KEY=VALUE
additions (the G2 audit needs PYTHONPATH and its routing switch). No prior art beyond these known to me.
"""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

A3 = Path(__file__).resolve().parent
scratch = Path('/tmp/eng1_a3_work')
(scratch / 'tmp').mkdir(parents=True, exist_ok=True)
label, *rest = sys.argv[1:]
sep = rest.index('--')
extra = dict(kv.split('=', 1) for kv in rest[:sep])
cmd = rest[sep + 1:]
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
           PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(scratch / 'tmp'),
           PYTHONPYCACHEPREFIX='/tmp/eng1_a3_pycache', XDG_CACHE_HOME='/tmp/eng1_a3_xdg')
env.pop('PYTHONPATH', None)
env.update(extra)
utc = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
start = utc()
with (A3 / (label + '.stdout')).open('x') as out, (A3 / (label + '.stderr')).open('x') as err:
    child = subprocess.run(['timeout', '590s', *cmd], env=env, stdout=out, stderr=err)
exited = utc()
rec = dict(label=label, cmd=cmd, extra_env=extra, cwd=os.getcwd(), start=start, child_exited=exited,
           recorded=utc(), exit=child.returncode, timeout=child.returncode in (124, 137),
           CUDA_VISIBLE_DEVICES=env['CUDA_VISIBLE_DEVICES'], threads=1, TMPDIR=env['TMPDIR'])
with (A3 / (label + '.exit.json')).open('x') as out:
    out.write(json.dumps(rec, indent=2) + '\n')
print(json.dumps(rec))
raise SystemExit(child.returncode)
