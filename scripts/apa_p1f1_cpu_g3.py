"""APA-P1-F1 S4 / G9 (CPU half): P1's CPU G3 set plus test_apa_p1d_policy.py, before and after, per test ID.

Before = P1-D's arm B_v2 (Build B module + f01201cf Python) running f01201cf's tests and
scripts (/tmp/apap1f1_treeB, git archive). After = arm F (this round's Python + Build F
module) running this worktree. CUDA_VISIBLE_DEVICES is empty for both. The file list is
A1's REGISTRATION.json cpu_g3_files (as P1-D) plus tensor_cuda/tests/test_apa_p1d_policy.py.
P1-D's per-ID outcome classifier is imported (scripts/apa_p1d_cpu_g3.outcome).
Prior art: pytest report API; P1 A1 G3.py / P1-D per-ID comparison (Project-Tensor 2026).
"""
from pathlib import Path
import collections
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import apa_p1d_cpu_g3 as p1d  # noqa: E402

ART = ROOT / 'artifacts/apa_p1f1'
TREE_B = Path('/tmp/apap1f1_treeB')
ARMS = {'before': (ROOT / 'artifacts/apa_p1d/arms/B_v2', TREE_B), 'after': (ART / 'arms/F', ROOT)}


def main():
    files = json.loads((p1d.A1 / 'REGISTRATION.json').read_text())['cpu_g3_files'] + \
        ['tensor_cuda/tests/test_apa_p1d_policy.py']
    out = ART / 'cpu_g3'
    out.mkdir()
    exits = {}
    for label, (arm_dir, tree) in ARMS.items():
        record = json.loads((arm_dir / 'ARM.json').read_text())
        env = os.environ.copy()
        env.pop('TC_APA_SP_STORAGE', None)
        env.pop('LD_PRELOAD', None)
        env.update(PYTHONPATH=str((ROOT / record['package']).parent) + ':' + str(tree / 'tensor_cuda/tests') + ':' +
                   str(ROOT / 'artifacts/apa_p1'), PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
                   TMPDIR='/tmp/apap1f1_g3')
        Path(env['TMPDIR']).mkdir(exist_ok=True)
        cmd = [sys.executable, str(ROOT / 'artifacts/apa_p1/pytest_receipt.py'), str(out / f'{label}.json'), '-q',
               '--tb=short', '-ra', '-p', 'no:cacheprovider', '--basetemp', f'/tmp/apap1f1_g3/bt_{label}', *files]
        with (out / f'{label}.log').open('x') as f:
            code = subprocess.run(['timeout', '1500s', *cmd], cwd=tree, env=env, stdout=f,
                                  stderr=subprocess.STDOUT).returncode
        exits[label] = dict(exit=code, cwd=str(tree), arm=str(arm_dir.relative_to(ROOT)), module=record['module_sha256'],
                            argv=cmd)
        print(label, 'exit', code, flush=True)
    a, b = [json.loads((out / f'{x}.json').read_text()) for x in ('before', 'after')]
    rows = []
    for key in sorted(set(a['collection']) | set(b['collection'])):
        x, y = a['reports'].get(key, []), b['reports'].get(key, [])
        rows.append(dict(id=key, before=p1d.outcome(x), after=p1d.outcome(y),
                         equal=[(r['when'], r['outcome']) for r in x] == [(r['when'], r['outcome']) for r in y]))
    result = dict(status='PASS' if a['collection'] == b['collection'] and rows and all(r['equal'] for r in rows) else 'FAIL',
                  ids=len(rows), differences=sum(not r['equal'] for r in rows),
                  collection_equal=a['collection'] == b['collection'],
                  collection_errors=[len(a['collection_errors']), len(b['collection_errors'])],
                  before=dict(collections.Counter(r['before'] for r in rows)),
                  after=dict(collections.Counter(r['after'] for r in rows)),
                  modules=dict(before=a['module'], after=b['module']), exits=exits,
                  differing=[r for r in rows if not r['equal']], rows=rows)
    with (ART / 'CPU_G3_COMPARISON.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print({k: v for k, v in result.items() if k not in ('rows', 'exits', 'modules')})
    sys.exit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
