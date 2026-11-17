"""APA-P1-F2 S4 / G8 (CPU half): P1's CPU G3 set + test_apa_p1d_policy.py + test_apa_p1f1_cpu.py, before and
after, per test ID; plus the S4 receipt of tensor_cuda/tests/test_apa_p1f2_cpu.py.

Before = Build F's arm (p1d worktree artifacts/apa_p1f1/arms/F: Build F module + F1 Python) running this branch's
tests as committed (/tmp/apap1f2_treeF0 = git archive d01118b4). After = Build F2's arm running this worktree.
CUDA_VISIBLE_DEVICES is empty for both. APA_P1F1_PACKED_MODULE is each side's own module (F1's R9 loader test runs
on both sides). The file list is A1's REGISTRATION.json cpu_g3_files (as P1-D and F1) plus the two files above.
P1-D's per-ID outcome classifier is imported (scripts/apa_p1d_cpu_g3.outcome). Intended changes come from
EXPECTED.json cpu.intended_changes (registered before any run).
Prior art: pytest report API; P1 A1 G3.py / P1-D / F1 per-ID comparison (Project-Tensor 2026). CPU only.
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

ART = ROOT / 'artifacts/apa_p1f2'
P1D_WT = Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1d')
TREE_F0 = Path('/tmp/apap1f2_treeF0')
ARMS = {'before': (P1D_WT, P1D_WT / 'artifacts/apa_p1f1/arms/F', TREE_F0),
        'after': (ROOT, ART / 'arms/F2', ROOT)}


def run(label, files, out, receipt_name):
    root, arm_dir, tree = ARMS[label]
    record = json.loads((arm_dir / 'ARM.json').read_text())
    env = os.environ.copy()
    for name in ('TC_APA_SP_STORAGE', 'LD_PRELOAD', 'APA_P1D_SRC', 'APA_P1F1_SRC', 'APA_P1F2_SRC'):
        env.pop(name, None)
    env.update(PYTHONPATH=str((root / record['package']).parent) + ':' + str(tree / 'tensor_cuda/tests') + ':' +
               str(ROOT / 'artifacts/apa_p1'), PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
               TMPDIR='/tmp/apap1f2_g3', APA_P1F1_PACKED_MODULE=str(root / record['module']))
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    cmd = [sys.executable, str(ROOT / 'artifacts/apa_p1/pytest_receipt.py'), str(out / f'{receipt_name}.json'), '-q',
           '--tb=short', '-ra', '-p', 'no:cacheprovider', '--basetemp', f'/tmp/apap1f2_g3/bt_{receipt_name}', *files]
    with (out / f'{receipt_name}.log').open('x') as f:
        code = subprocess.run(['timeout', '1700s', *cmd], cwd=tree, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    return dict(exit=code, cwd=str(tree), arm=str(arm_dir), module=record['module_sha256'], argv=cmd)


def main():
    what = sys.argv[1]
    if what == 's4':
        out = ART / 'cpu'
        out.mkdir(exist_ok=True)
        r = run('after', ['tensor_cuda/tests/test_apa_p1f2_cpu.py'], out, 'S4')
        rec = json.loads((out / 'S4.json').read_text())
        rows = {k: p1d.outcome(v) for k, v in rec['reports'].items()}
        res = dict(run=r, ids=len(rows), outcomes=dict(collections.Counter(rows.values())),
                   failed=[k for k, v in rows.items() if v != 'PASSED'])
        (out / 'S4_RESULT.json').write_text(json.dumps(res, indent=2) + '\n')
        print(json.dumps({k: v for k, v in res.items() if k != 'run'}))
        return 0 if rec['exit'] == 0 and not res['failed'] else 1
    files = json.loads((p1d.A1 / 'REGISTRATION.json').read_text())['cpu_g3_files'] + \
        ['tensor_cuda/tests/test_apa_p1d_policy.py', 'tensor_cuda/tests/test_apa_p1f1_cpu.py']
    out = ART / 'cpu_g3'
    out.mkdir()
    exits = {label: run(label, files, out, label) for label in ('before', 'after')}
    a, b = [json.loads((out / f'{x}.json').read_text()) for x in ('before', 'after')]
    intended = set(json.loads((ART / 'EXPECTED.json').read_text())['cpu']['intended_changes']['ids'])
    rows = []
    for key in sorted(set(a['collection']) | set(b['collection'])):
        x, y = a['reports'].get(key, []), b['reports'].get(key, [])
        rows.append(dict(id=key, before=p1d.outcome(x), after=p1d.outcome(y),
                         equal=[(r['when'], r['outcome']) for r in x] == [(r['when'], r['outcome']) for r in y],
                         intended=key in intended))
    unintended = [r for r in rows if not r['equal'] and not r['intended']]
    intended_seen = sorted(r['id'] for r in rows if not r['equal'] and r['intended'])
    result = dict(status='PASS' if a['collection'] == b['collection'] and rows and not unintended else 'FAIL',
                  ids=len(rows), differences=sum(not r['equal'] for r in rows), unintended_differences=unintended,
                  intended_changes_seen=intended_seen, intended_changes_not_seen=sorted(intended - set(intended_seen)),
                  collection_equal=a['collection'] == b['collection'],
                  collection_errors=[len(a['collection_errors']), len(b['collection_errors'])],
                  before=dict(collections.Counter(r['before'] for r in rows)),
                  after=dict(collections.Counter(r['after'] for r in rows)),
                  modules=dict(before=a['module'], after=b['module']), exits=exits,
                  differing=[r for r in rows if not r['equal']], rows=rows)
    with (ART / 'CPU_G3_COMPARISON.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print({k: v for k, v in result.items() if k not in ('rows', 'exits', 'modules', 'differing')})
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
