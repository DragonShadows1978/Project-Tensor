#!/usr/bin/env python3
"""A3 S1 (supplementary, NOT a G1 substitute): the governor's GW3 dry-run admission for each label's exact runner
argv, taken from STAGED_GPU_COMMANDS.sh's case line (the tokens after "$GPU_RUN"), run as
  FORGE_GPU_DRY_RUN=1 FORGE_GPU_RUN_LOG=/tmp/eng1_a3_s1/<label>.forge.log forge-gpu-run <argv>   (cwd = tool/)
For the 18 labels whose staged line stops at run_gpu_line.py's dependency gate before the runner (G1), this shows
whether the governor would admit their argv (rc 78, no usage error) and writes no run-log line. It does not run
start_gpu_run.py / run_gpu_line.py and proves nothing about those lines' parent path.
Prior art: GW3 dry-run contract (System-Claude GPU_WINDOW.md, 2026); Python shlex (POSIX shell word splitting).
No algorithm; no prior art beyond these known to me.
"""
import json
import os
from pathlib import Path
import shlex
import subprocess

A3 = Path(__file__).resolve().parent
TOOL = A3.parent / 'tool'
scratch = Path('/tmp/eng1_a3_s1')
scratch.mkdir(exist_ok=True)
labels = [r['label'] for r in json.loads((TOOL / 'gpu_lines.json').read_text())]
script = (TOOL / 'STAGED_GPU_COMMANDS.sh').read_text().splitlines()
rows = []
for label in labels:
    i = next(j for j, l in enumerate(script) if l.startswith('  %s) #' % label))
    words = shlex.split(script[i + 1])
    argv = words[words.index('$GPU_RUN') + 1:]
    log = scratch / (label + '.forge.log')
    assert not log.exists()
    env = dict(os.environ, FORGE_GPU_DRY_RUN='1', FORGE_GPU_RUN_LOG=str(log), CUDA_VISIBLE_DEVICES='0')
    p = subprocess.run(['forge-gpu-run', *argv], cwd=TOOL, env=env, stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=120)
    dry = [l for l in p.stdout.splitlines() if l.startswith('forge-gpu-run: DRY RUN ')]
    rec = json.loads(dry[0][len('forge-gpu-run: DRY RUN '):]) if dry else None
    ok = p.returncode == 78 and rec is not None and not log.exists() and rec['argv'] == argv[argv.index('--') + 1:]
    rows.append(dict(label=label, argv=argv, rc=p.returncode, dry_run=rec, stderr=p.stderr, run_log_written=log.exists(),
                     admitted_snapshot=bool(rec) and rec['budget'] >= rec['min'], ok=ok))
    print('%-22s rc %s | min %s max %s budget %.0f state %s | log written %s | ok %s' % (
        label, p.returncode, rec and rec['min'], rec and rec['max'], rec['budget'] if rec else -1, rec and rec['state'],
        log.exists(), ok))
with (A3 / 'S1_GOVERNOR_ARGV.json').open('x') as f:
    f.write(json.dumps(dict(rows=rows, all_ok=all(r['ok'] for r in rows)), indent=1) + '\n')
print('all_ok', all(r['ok'] for r in rows))
