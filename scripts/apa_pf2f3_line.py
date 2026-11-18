"""PF2-F3 pinned GPU launch and dry-run receipts.

Prior art: Project-Tensor forge-gpu-run and PF2-F2 launchers (2026).
Reuse admission, SHA-256 prechecks and lease exit/log receipts.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apa_pf2f3_build as B
import apa_pf2f3_cpu as C

THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def run_line(name, lead):
    dry = not lead
    assert os.environ['FORGE_GPU_DRY_RUN'] == ('1' if dry else '0')
    if dry:
        assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    pins = B.read(B.ART / 'LINE_PINS.json')
    for name_or_path, digest in pins['files'].items():
        assert B.sha(B.ROOT / name_or_path) == digest, ('pre-lease drift', name_or_path)
    spec = B.read(B.ART / 'LINES.json')[name]
    assert 30 <= spec['min_s'] <= 510 and spec['max_s'] == 540
    arms = {label: C.precheck(label) for label in spec['modules']}
    assert {label: row['module_sha256'] for label, row in arms.items()} == spec['modules']
    B.disk_free()

    forge = shutil.which('forge-gpu-run')
    assert forge is not None and B.sha(forge) == pins['forge_sha256']
    env = {k: os.environ[k] for k in ('HOME', 'USER', 'LOGNAME', 'XDG_RUNTIME_DIR', 'DBUS_SESSION_BUS_ADDRESS')
           if k in os.environ}
    env.update(dict.fromkeys(THREADS, '1'))
    tmp = Path('/tmp/pf2f3_gpu')
    tmp.mkdir(exist_ok=True)
    env.update(PATH=f'/usr/local/cuda-12.6/bin:{Path(forge).parent}:/usr/local/bin:/usr/bin:/bin',
               LANG='C.UTF-8', LC_ALL='C.UTF-8', LD_LIBRARY_PATH='/usr/local/cuda-12.6/lib64',
               TMPDIR=str(tmp), PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='',
               CUDA_VISIBLE_DEVICES='' if dry else '0', TC_APA_SP='1',
               FORGE_GPU_DRY_RUN='1' if dry else '0', APA_PF2F3_LEASE='1',
               FORGE_GPU_RUN_LOG=str(B.ART / 'gpu/forge-gpu-run.log'))
    dest = B.ART / 'gpu'
    dest.mkdir(exist_ok=True)
    prefix = dest / f"{name}_{'dry' if dry else 'real'}"
    log = prefix.with_suffix('.log')
    payload = ['timeout', '500s', 'nice', '-n', '19', sys.executable, '-B', '-P']
    if spec['mode'] == 'pytest':
        env['PYTHONPATH'] = str(C.arm_dir('F3'))
        result = B.ART / 'results' / name
        # Leave dry runs free of result directories so the real invocation owns them.
        if not dry:
            result.mkdir(parents=True)
        pytest_tmp = Path(tempfile.mkdtemp(prefix='pytest_', dir=tmp)) / 'base'
        arm = arms['F3']
        payload += ['scripts/apa_pf2f3_pytest.py', 'gpu', str(result / 'pytest.xml'),
                    str(pytest_tmp), str(B.ROOT / arm['module']), arm['module_sha256']]
    else:
        payload += ['scripts/apa_pf2f3_gpu.py', name]
    argv = [forge, '--min', str(spec['min_s']), '--max', '540', '--margin', '30',
            '--label', f'pf2f3-{name}', '--', *payload]
    B.save(prefix.with_suffix('.start.json'), dict(line=name, argv=argv, cwd=str(B.ROOT),
           environment=env, module_sha256=spec['modules'], dry=dry))
    with log.open('x') as output:
        code = subprocess.run(['timeout', '610s', *argv], cwd=B.ROOT, env=env, stdout=output,
                              stderr=subprocess.STDOUT).returncode
    data = dict(line=name, exit=code, log_sha256=B.sha(log), dry=dry,
                finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if dry:
        marker = 'forge-gpu-run: DRY RUN '
        records = [json.loads(s[len(marker):]) for s in log.read_text().splitlines() if s.startswith(marker)]
        data.update(admission=records, lease_attempted=False)
        ok = code == 78 and len(records) == 1 and records[0]['would_start'] is True and records[0]['argv'] == payload
    else:
        ok = code == 0
    B.save(prefix.with_suffix('.exit.json'), data)
    print(name, 'DRY' if dry else 'REAL', code, data['log_sha256'], flush=True)
    assert ok, data
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('line')
    parser.add_argument('--lead', action='store_true')
    args = parser.parse_args()
    raise SystemExit(run_line(args.line, args.lead))
