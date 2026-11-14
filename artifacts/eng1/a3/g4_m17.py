#!/usr/bin/env python3
"""A3 G4: mutant M17 (routing left on the A1 seal) shown refusing inside G1's own command path, in scratch copies.
Two fresh scratch trees under /tmp/eng1_a3_m17/ (byte copies, no links): orders/, the whole tool directory (arms
included, gpu_runs and caches excluded) and every A3-sealed registration file.
  control: tool/evidence.py as live (A3 routing)                 -> expected: seal verified, governor DRY RUN rc 78
  mutant : tool/evidence.py = a3/evidence.py.before (A1 routing)  -> expected: "tool seal mismatch: ids_a1_regression.txt"
Each runs G1's first label exactly as G1 does (env -u PYTHONOPTIMIZE -u ENG1_AUDIT RUNS=<root> CUDA_VISIBLE_DEVICES=0
TMPDIR=/tmp/eng1_a3_m17_tmp FORGE_GPU_DRY_RUN=1 bash artifacts/eng1/tool/STAGED_GPU_COMMANDS.sh baseline_eng1).
The live tool directory is never modified.
Prior art: mutation testing, DeMillo, Lipton and Sayward (1978); paired mutant/control runs as in ENG1 C_COPY.
Ours: only this one-mutant harness.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

A3 = Path(__file__).resolve().parent
ROOT = A3.parents[2]
TOOL = ROOT / 'artifacts/eng1/tool'
BASE = Path('/tmp/eng1_a3_m17')
BASE.mkdir(exist_ok=False)
seal = json.loads((TOOL / 'staging_seal_a3.json').read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ignore = shutil.ignore_patterns('__pycache__', '*.pyc', 'gpu_runs')
results = {}
for name, evidence_src in (('control', TOOL / 'evidence.py'), ('mutant', A3 / 'evidence.py.before')):
    root = BASE / name
    (root / 'orders').mkdir(parents=True)
    for o in ('ENG1_A1_REV1_FIXES.md', 'ENG1_A2_REV2_FIXES.md', 'ENG1_A3_DEVICE_SEAL_ROUTING.md'):
        shutil.copy2(ROOT / 'orders' / o, root / 'orders' / o)
    shutil.copytree(TOOL, root / 'artifacts/eng1/tool', ignore=ignore, symlinks=True)
    for rel in seal['registration_files']:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, root / rel)
    shutil.copy2(evidence_src, root / 'artifacts/eng1/tool/evidence.py')
    links = [str(p) for p in root.rglob('*') if p.is_symlink()]
    assert not links, links
    runs = root / 'artifacts/eng1/tool/gpu_runs/m17_1'
    cmd = ['env', '-u', 'PYTHONOPTIMIZE', '-u', 'ENG1_AUDIT', 'RUNS=' + str(runs), 'CUDA_VISIBLE_DEVICES=0',
           'TMPDIR=/tmp/eng1_a3_m17_tmp', 'FORGE_GPU_DRY_RUN=1', 'bash', 'artifacts/eng1/tool/STAGED_GPU_COMMANDS.sh',
           'baseline_eng1']
    p = subprocess.run(cmd, cwd=root, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)
    att = runs / 'baseline_eng1' / 'attempt-0001'
    log = (att / 'stdout.log').read_text() if (att / 'stdout.log').exists() else ''
    results[name] = dict(root=str(root), evidence_sha256=sha(root / 'artifacts/eng1/tool/evidence.py'), cmd=cmd,
                         staged_exit=p.returncode, stdout=p.stdout, stderr=p.stderr, run_root_created=runs.exists(),
                         dry_run_lines=[l for l in log.splitlines() if l.startswith('forge-gpu-run: DRY RUN ')],
                         parent_exit=json.loads((att / 'parent_result.json').read_text())['exit']
                         if (att / 'parent_result.json').exists() else None)
c, m = results['control'], results['mutant']
refusal = [l for l in m['stderr'].splitlines() if l.startswith('AssertionError')]
results['control_ok'] = c['parent_exit'] == 78 and len(c['dry_run_lines']) == 1
results['mutant_killed'] = refusal == ['AssertionError: tool seal mismatch: ids_a1_regression.txt'] \
    and not m['run_root_created'] and m['staged_exit'] != 0 and not m['dry_run_lines']
results['refusal_text'] = refusal
results['g4_pass'] = results['control_ok'] and results['mutant_killed']
with (A3 / 'G4_M17.json').open('x') as f:
    f.write(json.dumps(results, indent=1) + '\n')
print(m['stderr'])
print(json.dumps({k: results[k] for k in ('control_ok', 'mutant_killed', 'refusal_text', 'g4_pass')}))
print('control evidence.py', c['evidence_sha256'], 'mutant evidence.py', m['evidence_sha256'])
