"""APA-P1-F1 S6 line runner: precheck, one dry run (expected 78), then the real line.

  python3 -B scripts/apa_p1f1_line.py <LINE> <attempt> <max_s> <timeout_s> <arms,...> -- <apa_p1f1_gpu.py args>

Before the lease: scripts/apa_p1f1_gpu.py precheck for every named arm (module sha
vs receipt; working tree vs the F/FH receipt). Then FORGE_GPU_DRY_RUN=1 forge-gpu-run
(must return 78) and the real
  forge-gpu-run --min 30 --max <max_s> --label p1f1-<LINE> -- timeout <timeout_s>s python3 -B scripts/apa_p1f1_gpu.py ...
from the worktree root. Receipts: artifacts/apa_p1f1/gpu/<LINE>_r<n>.{start,log,exit}
and <LINE>_dry_r<n>.log. forge-gpu-run passes the inner command's exit through, so
the recorded exit is also the inner timeout's exit (124 = timeout, never a pass).
Exit 76 = lock not acquired, nothing ran (the caller retries after >= 5 minutes).
Prior art: none; process plumbing.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GPU = ROOT / 'artifacts/apa_p1f1/gpu'


def main():
    line, attempt, max_s, timeout_s, arms = sys.argv[1:6]
    assert sys.argv[6] == '--'
    inner = sys.argv[7:]
    assert int(max_s) <= 1800 and int(timeout_s) <= 1750 and int(timeout_s) < int(max_s)
    GPU.mkdir(parents=True, exist_ok=True)
    tag = f'{line}_r{attempt}'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', TMPDIR='/tmp/apap1f1_tmp')
    for arm in arms.split(','):
        p = subprocess.run([sys.executable, '-B', 'scripts/apa_p1f1_gpu.py', 'precheck', arm], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=300)
        (GPU / f'{tag}_precheck_{arm.replace(":", "_")}.log').write_text(p.stdout + p.stderr)
        if p.returncode:
            print('PRECHECK REFUSED', arm, p.stdout[-500:], p.stderr[-1500:], flush=True)
            return 3
    cmd = ['forge-gpu-run', '--min', '30', '--max', str(max_s), '--label', f'p1f1-{line}', '--',
           'timeout', f'{timeout_s}s', sys.executable, '-B', 'scripts/apa_p1f1_gpu.py', *inner]
    run_env = dict(os.environ)
    run_env.pop('CUDA_VISIBLE_DEVICES', None)
    run_env.pop('LD_PRELOAD', None)
    run_env['TMPDIR'] = '/tmp/apap1f1_gpu'
    Path(run_env['TMPDIR']).mkdir(exist_ok=True)
    with (GPU / f'{line}_dry_r{attempt}.log').open('x') as f:
        dry = subprocess.run(cmd, cwd=ROOT, env=dict(run_env, FORGE_GPU_DRY_RUN='1'), stdout=f,
                             stderr=subprocess.STDOUT, timeout=120).returncode
    with (GPU / f'{line}_dry_r{attempt}.log').open('a') as f:
        f.write(f'\nDRY_RUN_EXIT={dry}\n')
    if dry != 78:
        print('DRY RUN did not return 78:', dry, flush=True)
        return 4
    (GPU / f'{tag}.start').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                                      argv=cmd, cwd=str(ROOT), dry_run_exit=dry)) + '\n')
    with (GPU / f'{tag}.log').open('x') as f:
        code = subprocess.run(cmd, cwd=ROOT, env=run_env, stdout=f, stderr=subprocess.STDOUT).returncode
    log_sha = hashlib.sha256((GPU / f'{tag}.log').read_bytes()).hexdigest()
    (GPU / f'{tag}.exit').write_text(json.dumps(dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), forge_exit=code,
        inner_timeout_exit=code if code != 76 else None, timed_out=code == 124, lock_not_acquired=code == 76,
        log_sha256=log_sha)) + '\n')
    print(tag, 'forge exit', code, 'log sha256', log_sha, flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
