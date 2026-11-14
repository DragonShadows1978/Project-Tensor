#!/usr/bin/env python3
"""One existing ENG1 audit line with a fresh A2 root/new seal.
Prior art: STAGED_GPU_COMMANDS.sh and GN1/ENG1 L45 (2026); GNU timeout.
Taken verbatim: candidate line arguments, CPU stand-in and supervisor checks.
New: inherited path-only startup configuration; previous receipts stay intact.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

A = Path(__file__).resolve().parent
R = A.parents[2]
T = R / 'artifacts/eng1/tool'
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
env = dict(os.environ, ENG1_A2_AUDIT_ROUTING='1', ENG1_AUDIT='1', RUNS=str(A / 'audit'),
           PYTHONPATH=os.pathsep.join((str(A), str(T))))
subprocess.run(['timeout', '60s', sys.executable, '-B', 'start_gpu_run.py'],
               env=env, cwd=T, check=True)
line = next(x for x in json.loads((T / 'gpu_lines.json').read_text()) if x['label'] == 'candidate_eng1')
cmd = ['timeout', str(line['outer'] + 20) + 's', sys.executable, '-B',
       'run_gpu_line.py', 'candidate_eng1', '0', str(line['outer']),
       str(T / 'forge_gpu_run_standin.py'), '--min', str(line['minimum']),
       '--max', str(line['maximum']), '--label', 'eng1-candidate_eng1', '--',
       'timeout', '--kill-after=10s', str(line['inner']) + 's', 'python3',
       'run_gpu_tests.py', 'candidate', 'eng1']
raise SystemExit(subprocess.run(cmd, env=env, cwd=T).returncode)
