"""APA-P1-D S4 / G9 (CPU half): P1's G3 CPU set, before and after, per test ID.

Before = arm A (Build A module + e2fb85a1 Python) running e2fb85a1's tests and
scripts (/tmp/apap1d_treeA, git archive). After = the staged final arm B
(Build B + this round's Python) running this worktree. CUDA_VISIBLE_DEVICES is
empty for both. The file list is A1's REGISTRATION.json cpu_g3_files.
Prior art: pytest report API; P1 A1 G3.py per-ID comparison (Project-Tensor, 2026).
"""
from pathlib import Path
import collections
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/apa_p1d'
A1 = Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/artifacts/apa_p1_a1')
ARM_B = os.environ.get('APA_P1D_ARM_B', 'B')


def outcome(v):
    if any(r['outcome'] == 'failed' and r['when'] != 'call' for r in v):
        return 'ERROR'
    for r in v:
        if r['outcome'] == 'failed':
            return 'FAILED'
        if r['outcome'] == 'skipped':
            return 'SKIPPED'
    return 'PASSED' if any(r['when'] == 'call' and r['outcome'] == 'passed' for r in v) else 'MISSING'


def main():
    files = json.loads((A1 / 'REGISTRATION.json').read_text())['cpu_g3_files']
    out = ART / 'cpu_g3'
    out.mkdir()
    exits = {}
    for label, arm, tree in (('before', 'A', Path('/tmp/apap1d_treeA')), ('after', ARM_B, ROOT)):
        record = json.loads((ART / 'arms' / arm / 'ARM.json').read_text())
        env = os.environ.copy()
        env.pop('TC_APA_SP_STORAGE', None)
        env.update(PYTHONPATH=str((ROOT / record['package']).parent) + ':' + str(tree / 'tensor_cuda/tests') + ':' +
                   str(ROOT / 'artifacts/apa_p1'), PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
                   TMPDIR='/tmp/apap1d_g3')
        Path(env['TMPDIR']).mkdir(exist_ok=True)
        cmd = [sys.executable, str(ROOT / 'artifacts/apa_p1/pytest_receipt.py'), str(out / f'{label}.json'), '-q',
               '--tb=short', '-ra', '-p', 'no:cacheprovider', '--basetemp', f'/tmp/apap1d_g3/bt_{label}', *files]
        with (out / f'{label}.log').open('x') as f:
            code = subprocess.run(['timeout', '1500s', *cmd], cwd=tree, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        exits[label] = dict(exit=code, cwd=str(tree), arm=arm, argv=cmd)
        print(label, 'exit', code, flush=True)
    a, b = [json.loads((out / f'{x}.json').read_text()) for x in ('before', 'after')]
    rows = []
    for key in sorted(set(a['collection']) | set(b['collection'])):
        x, y = a['reports'].get(key, []), b['reports'].get(key, [])
        rows.append(dict(id=key, before=outcome(x), after=outcome(y),
                         equal=[(r['when'], r['outcome']) for r in x] == [(r['when'], r['outcome']) for r in y]))
    a1 = {}
    for line in (A1 / 'g3_ids.tsv').read_text().splitlines()[1:]:
        tid, prod, port, _ = line.split('\t')
        a1[tid] = port
    result = dict(status='PASS' if a['collection'] == b['collection'] and rows and all(r['equal'] for r in rows) else 'FAIL',
                  ids=len(rows), differences=sum(not r['equal'] for r in rows),
                  collection_equal=a['collection'] == b['collection'],
                  collection_errors=[len(a['collection_errors']), len(b['collection_errors'])],
                  before=dict(collections.Counter(r['before'] for r in rows)),
                  after=dict(collections.Counter(r['after'] for r in rows)),
                  modules=dict(before=a['module'], after=b['module']), exits=exits,
                  a1_reference=dict(ids=len(a1), same_id_set=set(a1) == set(r['id'] for r in rows),
                                    outcome_differences_vs_a1_port=sum(a1.get(r['id']) != r['after'] for r in rows)),
                  rows=rows)
    with (ART / 'CPU_G3_COMPARISON.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print({k: v for k, v in result.items() if k not in ('rows', 'exits', 'modules')})
    sys.exit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
