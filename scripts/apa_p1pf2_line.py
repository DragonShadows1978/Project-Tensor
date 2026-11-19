"""APA-P1-PF2 line runner: disk rail, MemAvailable rail, precheck, one dry run (expected 78), then the real line.

  python3 -B -P scripts/apa_p1pf2_line.py <LINE> <attempt> <max_s> <timeout_s> <arms,...> -- <script> <args...>

Before the lease: the disk rail (stop under 40 GB free on /mnt/ForgeRealm, L67), MemAvailable >= 12 GB, then
scripts/apa_p1pf2_gpu.py precheck for every named arm (module sha256 == its build receipt; PF/PFH also the working
tree's engine source map, L103). Then FORGE_GPU_DRY_RUN=1 forge-gpu-run (must return 78, L79) and the real
  forge-gpu-run --min 30 --max <max_s> --label p1pf2-<LINE> -- timeout <timeout_s>s systemd-run --user --scope
      -p MemoryMax=12G -p OOMPolicy=continue -p CPUWeight=50 nice -n 19 python3 -B -P <script> ...
from the worktree root. Receipts: artifacts/apa_p1pf2/gpu/<LINE>_r<n>.{start,log,exit}, <LINE>_dry_r<n>.log and the
precheck logs. forge-gpu-run passes the inner command's exit through, so the recorded exit is also the inner
timeout's exit (124 = timeout, never a pass). Exit 76 = lock not acquired, nothing ran (retry after >= 5 min).
Prior art: F1/F2/F3 line runners (scripts/apa_p1f3_line.py, Project-Tensor 2026), restated for this round's paths;
process plumbing.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GPU = ROOT / 'artifacts/apa_p1pf2/gpu'


def mem_available_gb():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1e6
    return 0.0


def main():
    line, attempt, max_s, timeout_s, arms = sys.argv[1:6]
    assert sys.argv[6] == '--'
    inner = sys.argv[7:]
    assert int(max_s) <= 1800 and int(timeout_s) <= 1750 and int(timeout_s) < int(max_s)
    free_gb = shutil.disk_usage('/mnt/ForgeRealm').free / 1e9
    if free_gb < 40:
        print(f'DISK RAIL: {free_gb:.1f} GB free < 40 GB; not starting', flush=True)
        return 5
    mem = mem_available_gb()
    if mem < 12:
        print(f'MEMORY RAIL: MemAvailable {mem:.1f} GB < 12 GB; not starting', flush=True)
        return 6
    GPU.mkdir(parents=True, exist_ok=True)
    tag = f'{line}_r{attempt}'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', TMPDIR='/tmp/apap1pf2_tmp')
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    for arm in [a for a in arms.split(',') if a]:
        p = subprocess.run([sys.executable, '-B', '-P', 'scripts/apa_p1pf2_gpu.py', 'precheck', arm], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=300)
        (GPU / f'{tag}_precheck_{arm.replace(":", "_")}.log').write_text(p.stdout + p.stderr)
        if p.returncode:
            print('PRECHECK REFUSED', arm, p.stdout[-500:], p.stderr[-1500:], flush=True)
            return 3
    cmd = ['forge-gpu-run', '--min', '30', '--max', str(max_s), '--label', f'p1pf2-{line}', '--',
           'timeout', f'{timeout_s}s', 'systemd-run', '--user', '--scope', '--quiet', '-p', 'MemoryMax=12G',
           '-p', 'OOMPolicy=continue', '-p', 'CPUWeight=50', 'nice', '-n', '19',
           sys.executable, '-B', '-P', *inner]
    run_env = dict(os.environ)
    for name in ('CUDA_VISIBLE_DEVICES', 'LD_PRELOAD', 'TC_APA_SP_STORAGE', 'TC_APA_SPQ1_REFERENCE'):
        run_env.pop(name, None)
    run_env['TMPDIR'] = '/tmp/apap1pf2_gpu'
    run_env['PYTHONDONTWRITEBYTECODE'] = '1'
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
                                                      argv=cmd, cwd=str(ROOT), dry_run_exit=dry,
                                                      disk_free_gb=round(free_gb, 1),
                                                      mem_available_gb=round(mem, 1))) + '\n')
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
