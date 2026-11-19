"""PF2-F2 pinned launch and dry-run receipts.

Prior art: Project-Tensor forge-gpu-run and PF2-F1 launchers (2026).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apa_pf2f2_build as B
import apa_pf2f2_cpu as C

THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def run_line(name, lead):
    pins = B.read(B.ART / 'LINE_PINS.json')
    for name_or_path, digest in pins['files'].items():
        assert B.sha(B.ROOT / name_or_path) == digest, ('pre-lease drift', name_or_path)
    spec = B.read(B.ART / 'LINES.json')[name]
    assert 30 <= spec['min_s'] <= 510 and spec['max_s'] == 540
    for label, digest in spec['modules'].items():
        assert C.precheck(label)['module_sha256'] == digest, label
    B.disk_free()

    dry = not lead
    if dry:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        assert os.environ.get('FORGE_GPU_DRY_RUN') == '1'
    else:
        assert os.environ.get('FORGE_GPU_DRY_RUN') == '0'
    forge = shutil.which('forge-gpu-run')
    assert forge is not None and B.sha(forge) == pins['forge_sha256']
    env = {k:os.environ[k] for k in ('HOME','USER','LOGNAME','XDG_RUNTIME_DIR','DBUS_SESSION_BUS_ADDRESS')
           if k in os.environ}
    env.update(dict.fromkeys(THREADS, '1'))
    env.update(PATH=f'/usr/local/cuda-12.6/bin:{Path(forge).parent}:/usr/local/bin:/usr/bin:/bin',
               LANG='C.UTF-8', LC_ALL='C.UTF-8', LD_LIBRARY_PATH='/usr/local/cuda-12.6/lib64',
               TMPDIR='/tmp/pf2f2_gpu', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='',
               CUDA_VISIBLE_DEVICES='' if dry else '0', TC_APA_SP='1',
               FORGE_GPU_DRY_RUN='1' if dry else '0', APA_PF2F2_LEASE='1')
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    argv = [forge, '--min', str(spec['min_s']), '--max', '540','--margin','30','--label', f'pf2f2-{name}', '--',
            'timeout', '500s', 'nice', '-n', '19', sys.executable, '-B', '-P', 'scripts/apa_pf2f2_gpu.py', name]
    dest = B.ART / 'gpu'
    dest.mkdir(exist_ok=True)
    prefix = dest / f"{name}_{'dry' if dry else 'real'}"
    log = prefix.with_suffix('.log')
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
        ok = code == 78 and len(records) == 1 and records[0]['would_start'] is True and records[0]['argv'] == argv[10:]
    else:
        ok = code == 0
    B.save(prefix.with_suffix('.exit.json'), data)
    print(name, 'DRY' if dry else 'REAL', code, data['log_sha256'], flush=True)
    assert ok, data
    return code


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('line')
    p.add_argument('--lead', action='store_true')
    a = p.parse_args()
    raise SystemExit(run_line(a.line, a.lead))
