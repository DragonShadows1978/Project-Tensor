#!/usr/bin/env python3
"""A3 G1: device-mode dry run of the 24 staged labels, in gpu_lines.json order, each exactly as the lead's wrapper
(lead/eng1_g5.sh) runs it, with FORGE_GPU_DRY_RUN=1 added:
  env -u PYTHONOPTIMIZE -u ENG1_AUDIT RUNS=<root> CUDA_VISIBLE_DEVICES=0 TMPDIR=/tmp/eng1_a3_dry FORGE_GPU_DRY_RUN=1 \
      bash artifacts/eng1/tool/STAGED_GPU_COMMANDS.sh <label>
as a plain subprocess (stdin /dev/null; stdout/stderr to per-label receipts; the exit is recorded unmasked).
Usage: g1.py N   (run root artifacts/eng1/tool/gpu_runs/a3_dry_N, must not exist yet)
Stops before a label when /mnt/ForgeRealm has under 25 GB free (L67). Records, per label: the START record printed by
start_gpu_run.py (printed only after verify_seal() and verify_arm() on every sealed arm returned), every attempt's
parent_result.json, the governor's DRY RUN line from the attempt's stdout.log, whether a runner log (forge.log) was
written, and the refusal text when the line stopped earlier.
Prior art: the lead's eng1_g5.sh / GN1 gn1_g5.sh (2026) loop; GW3 dry-run contract (System-Claude GPU_WINDOW.md).
Ours: only the per-label record extraction. No algorithm; no prior art beyond these known to me.
"""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

A3 = Path(__file__).resolve().parent
ROOT = A3.parents[2]
TOOL = ROOT / 'artifacts/eng1/tool'
SEAL = TOOL / 'staging_seal_a3.json'
n = int(sys.argv[1])
runs = TOOL / 'gpu_runs' / ('a3_dry_%d' % n)
assert not runs.exists(), 'run root must be fresh: ' + str(runs)
out_dir = A3 / ('g1_a3_dry_%d' % n)
out_dir.mkdir(exist_ok=False)
sys.path.insert(0, str(TOOL))
import evidence  # noqa: E402
assert evidence.guard(runs) == runs.resolve()
seal_sha = evidence.sha(SEAL)
seal = json.loads(SEAL.read_text())
labels = [r['label'] for r in json.loads((TOOL / 'gpu_lines.json').read_text())]
utc = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
rows = []
for label in labels:
    free = shutil.disk_usage('/mnt/ForgeRealm').free
    if free < 25 * 1024**3:
        rows.append(dict(label=label, stopped='NVMe free %d B < 25 GB' % free))
        break
    cmd = ['env', '-u', 'PYTHONOPTIMIZE', '-u', 'ENG1_AUDIT', 'RUNS=' + str(runs), 'CUDA_VISIBLE_DEVICES=0',
           'TMPDIR=/tmp/eng1_a3_dry', 'FORGE_GPU_DRY_RUN=1', 'bash', 'artifacts/eng1/tool/STAGED_GPU_COMMANDS.sh', label]
    start = utc()
    with (out_dir / (label + '.stdout')).open('x') as so, (out_dir / (label + '.stderr')).open('x') as se:
        p = subprocess.run(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=so, stderr=se, timeout=590)
    end = utc()
    stdout = (out_dir / (label + '.stdout')).read_text()
    stderr = (out_dir / (label + '.stderr')).read_text()
    start_record = None
    for line in stdout.splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict) and 'seal_sha256' in d and 'run_root' in d:
            start_record = d
            break
    attempts = []
    for att in sorted((runs / label).glob('attempt-*')) if (runs / label).exists() else []:
        pr = json.loads((att / 'parent_result.json').read_text()) if (att / 'parent_result.json').exists() else None
        log = (att / 'stdout.log').read_text() if (att / 'stdout.log').exists() else ''
        dry = [l for l in log.splitlines() if l.startswith('forge-gpu-run: DRY RUN ')]
        attempts.append(dict(
            attempt=att.name, files=sorted(p.name for p in att.iterdir()),
            dry_run=json.loads(dry[0][len('forge-gpu-run: DRY RUN '):]) if dry else None, n_dry_lines=len(dry),
            stdout_log=log, forge_log_written=(att / 'forge.log').exists(),
            parent=None if pr is None else {k: pr.get(k) for k in (
                'exit', 'raw_exit', 'admission', 'forge_outcome', 'passed', 'reasons', 'seal_sha256', 'audit',
                'run_root', 'timeout', 'timeout_cause')}))
    governor_rc = attempts[-1]['parent']['exit'] if attempts and attempts[-1]['parent'] else None
    seal_ok = bool(start_record) and start_record['seal_sha256'] == seal_sha and start_record['run_root'] == str(runs) \
        and start_record['audit'] is False
    reached = bool(attempts) and attempts[-1]['dry_run'] is not None and governor_rc == 78 \
        and not attempts[-1]['forge_log_written']
    refusal = [l for l in stderr.splitlines() if l.strip()]
    rows.append(dict(label=label, cmd=cmd, start=start, end=end, staged_exit=p.returncode,
                     start_record=start_record, seal_verified=seal_ok, arms_verified=len(seal['arms']) if seal_ok else 0,
                     governor_rc=governor_rc, governor_reached=reached, attempts=attempts,
                     stderr_tail=refusal[-3:], line_pass=seal_ok and reached))
    print('%-22s staged rc %s | seal %s | arms %s | governor rc %s | %s' % (
        label, p.returncode, seal_ok, rows[-1]['arms_verified'], governor_rc,
        'DRY RUN admission recorded' if reached else ('refused: ' + (refusal[-1] if refusal else '?'))))
result = dict(run_root=str(runs), seal=str(SEAL.relative_to(ROOT)), seal_sha256=seal_sha, n_labels=len(labels),
              rows=rows, labels_seal_verified=sum(r.get('seal_verified', False) for r in rows),
              labels_governor_78=sum(r.get('governor_reached', False) for r in rows),
              g1_pass=len(rows) == len(labels) and all(r.get('line_pass') for r in rows))
with (out_dir / 'G1_RESULT.json').open('x') as f:
    f.write(json.dumps(result, indent=1) + '\n')
print(json.dumps({k: result[k] for k in ('run_root', 'seal_sha256', 'n_labels', 'labels_seal_verified',
                                         'labels_governor_78', 'g1_pass')}))
