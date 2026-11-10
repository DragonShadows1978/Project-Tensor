#!/usr/bin/env python3
"""Foreground CPU receipt supervisor. Prior art: REV2 run.py / ENG1 L45 (2026),
GNU timeout/nice and Python subprocess. Taken: completed-child recording;
A2 changes only the destination, scratch prefix and registered timeout.
"""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

A2 = Path(__file__).resolve().parent
scratch = Path('/tmp/eng1_a2_work')
(scratch / 'tmp').mkdir(parents=True, exist_ok=True)
label, *cmd = sys.argv[1:]
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
           PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(scratch / 'tmp'))
env.pop('PYTHONPATH', None)
utc = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
start = utc()
with (A2 / (label + '.stdout')).open('x') as out, (A2 / (label + '.stderr')).open('x') as err:
    child = subprocess.run(['nice', '-n', '19', 'timeout', '1100s', *cmd],
                           env=env, stdout=out, stderr=err)
exited = utc()
rec = dict(label=label, cmd=cmd, cwd=os.getcwd(), start=start, child_exited=exited,
           recorded=utc(), exit=child.returncode, timeout=child.returncode in (124, 137),
           CUDA_VISIBLE_DEVICES=env['CUDA_VISIBLE_DEVICES'], threads=1, TMPDIR=env['TMPDIR'])
with (A2 / (label + '.exit.json')).open('x') as out:
    out.write(json.dumps(rec, indent=2) + '\n')
print(json.dumps(rec))
raise SystemExit(child.returncode)
