#!/usr/bin/env python3
"""APA-P1-PF2 G6 regression lines against Build F3, run inside forge-gpu-run.

  lf1 --out D             : P1's finite identity (P1-F1's LF1: P1-D's L1 grid, arm B, plus F1's raw cases), Build F3
                            vs Build PF, bitwise per array (children per arm: lf1-arm)
  l2l4 --out D            : P1-D's L2 (refusals; RE_prefill_on_packed_refused is the one intended change) and L4
                            (68 B per key per KV head; device memory) on Build F3 and Build PF (children per arm/case)
  lf6 --out D --files ... : P1-D's L6 files, per-ID pytest outcomes, Build F3 (tests of 2945c3e9, git archive at
                            /tmp/apap1pf2_tree0) vs Build PF (this tree's tests), 0 differences allowed
  pins                    : CPU: dense SP table, packed margin table and pinned constants unchanged vs 2945c3e9
P1-D's and F1's code is imported and run unchanged; only their arm loaders are pointed at this round's arms (Build
F3's staged arm in tensor_fork_apa_wt_p1f2, read-only; Build PF's in artifacts/apa_p1pf2/arms/PF).
Prior art: P1-D/F1/F3 line code (Project-Tensor 2026), imported; per-ID pytest receipts. No new technique.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tensor_cuda/tests'))
import apa_p1pf2_gpu as G  # noqa: E402

P1F2 = ROOT.parent / 'tensor_fork_apa_wt_p1f2'
TREE0 = Path('/tmp/apap1pf2_tree0')
ARMS = ('F3', 'PF')


def arm_root(name):
    return P1F2 if name == 'F3' else ROOT


def point_p1d(name):
    """Point P1-D's loader (apa_p1d_gpu.load_arm, via ART/ARM_B/ROOT) at this arm's staged package."""
    import apa_p1d_gpu as g
    g.ROOT = arm_root(name)
    g.ART = (P1F2 / 'artifacts/apa_p1f3') if name == 'F3' else G.ART
    g.ARM_B = name
    return g


def child(argv, log, limit):
    t0 = time.monotonic()
    with Path(log).open('x') as f:
        try:
            code = subprocess.run(argv, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=limit,
                                  env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', LD_PRELOAD='')).returncode
        except subprocess.TimeoutExpired:
            code = 124
    return dict(argv=[str(a) for a in argv], exit=code, seconds=time.monotonic() - t0, log=str(log))


# ------------------------------------------------------------------ LF1
def lf1_arm(args):
    import apa_p1f1_gpu as f1
    name = args.arm
    G.precheck(name)
    g = point_p1d(name)
    f1.ART = g.ART
    f1.build_receipt = lambda n: dict(module_sha256=G.receipt(name)['module_sha256'])
    return f1.lf1_arm(args)


def lf1(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for name in ARMS:
        r = child([sys.executable, '-B', '-P', __file__, 'lf1-arm', '--arm', name, '--out', str(out)],
                  out / f'arm_{name}.log', 1200)
        runs.append(r)
        if r['exit']:
            G.save(out / 'RESULT.json', dict(line='LF1', status='FAIL_CHILD', runs=runs, terminal='LF1_CHILD_FAILED'))
            return 1
    rows, missing = [], []
    for pre in ('l1_{}/arm_B.npz', 'raw_{}.npz'):
        A, B = np.load(out / pre.format('F3')), np.load(out / pre.format('PF'))
        missing += sorted(set(A.files) ^ set(B.files))
        for key in sorted(set(A.files) & set(B.files)):
            a, b = A[key], B[key]
            same = a.dtype == b.dtype and a.shape == b.shape and np.array_equal(G_u(a), G_u(b))
            rows.append(dict(key=pre.split('_')[0] + ':' + key, equal=bool(same),
                             informative=[G_inf(a), G_inf(b)]))
    eq = sum(r['equal'] for r in rows)
    unin = [r['key'] for r in rows if r['informative'] != [True, True]]
    status = 'PASS' if rows and eq == len(rows) and not missing and not unin else 'FAIL'
    G.save(out / 'RESULT.json', dict(line='LF1', status=status, comparisons=len(rows), bitwise_equal=eq,
                                     key_set_mismatch=missing, uninformative=unin,
                                     not_equal=[r for r in rows if not r['equal']][:50], runs=runs,
                                     module_sha256={n: G.receipt(n)['module_sha256'] for n in ARMS}, terminal='LF1_DONE'))
    print('LF1', status, eq, '/', len(rows), 'missing', len(missing), 'uninformative', len(unin), flush=True)
    return 0


def G_u(a):
    a = np.ascontiguousarray(a)
    return a.view({1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize])


def G_inf(a):
    u = G_u(np.asarray(a)).ravel()
    return bool(u.size) and bool(np.any(u != 0)) and bool(np.any(u != u[0]))


# ------------------------------------------------------------------ L2 / L4
def point_loader(name):
    """P1-D's L2/L4 read P1-D's own receipts (ART = this worktree's artifacts/apa_p1d: L3's branch, the packed table);
    only its arm loader is replaced, by this round's loader of the named arm (module sha checked)."""
    import apa_p1d_gpu as g
    g.ROOT, g.ART = ROOT, ROOT / 'artifacts/apa_p1d'

    def load(_):
        tc, rec = G.load_arm(name)
        arm = json.loads((G.arm_dir(name) / 'ARM.json').read_text())
        assert rec['module_sha256'] == arm['module_sha256']
        return tc, arm
    g.load_arm = load
    return g


def l2_arm(args):
    g = point_loader(args.arm)
    G.precheck(args.arm)
    return g.l2(argparse.Namespace(out=args.out))


def l4_case(args):
    g = point_loader(args.arm)
    G.precheck(args.arm)
    return g.l4_case(argparse.Namespace(model=args.model, S=args.S, out=args.out))


def l2l4(args):
    out = Path(args.out)
    runs, res = [], {}
    for name in ARMS:
        d = out / f'L2_{name}'
        d.mkdir(parents=True, exist_ok=True)
        runs.append(child([sys.executable, '-B', '-P', __file__, 'l2-arm', '--arm', name, '--out', str(d)],
                          out / f'L2_{name}.log', 600))
        res[name] = json.loads((d / 'RESULT.json').read_text()) if (d / 'RESULT.json').exists() else None
        for model in ('qwen3_1.7b', 'qwen3_4b'):
            for S in (2048, 8192, 32768):
                d4 = out / f'L4_{name}'
                d4.mkdir(parents=True, exist_ok=True)
                runs.append(child([sys.executable, '-B', '-P', __file__, 'l4-case', '--arm', name, '--model', model,
                                   '--S', str(S), '--out', str(d4)], d4 / f'{model}_S{S}.log', 400))
    rows = {}
    for name in ARMS:
        rows[name] = {c['check']: c['ok'] for c in (res[name] or {}).get('checks', [])}
    intended = 'RE_prefill_on_packed_refused'
    diff = sorted(k for k in set(rows['F3']) | set(rows['PF']) if rows['F3'].get(k) != rows['PF'].get(k))
    l2_ok = (bool(rows['F3']) and all(rows['F3'].values()) and diff == [intended] and
             all(v for k, v in rows['PF'].items() if k != intended) and rows['PF'].get(intended) is False and
             len(rows['F3']) == len(rows['PF']))
    l4 = {name: [json.loads(p.read_text()) for p in sorted((out / f'L4_{name}').glob('*.json'))] for name in ARMS}
    l4_ok = all(len(l4[n]) == 6 and all(all(c['checks'].values()) for c in l4[n]) for n in ARMS)
    G.save(out / 'RESULT.json', dict(
        line='L2L4', l2=dict(status='PASS' if l2_ok else 'FAIL', rows_F3=len(rows['F3']), rows_PF=len(rows['PF']),
                             F3_ok=sum(rows['F3'].values()), PF_ok=sum(rows['PF'].values()), differing_rows=diff,
                             intended_change=f'{intended}: F3 refuses (ok); PF runs prefill on packed keys (the row '
                                             'fails as written; replaced by G4 acceptance rows)'),
        l4=dict(status='PASS' if l4_ok else 'FAIL',
                summary={n: [dict(model=c['model'], S=c['S'], per_key=c['owned_per_key_per_kv_head'],
                                  measured=c['measured_difference'], predicted=c['predicted_difference'],
                                  checks=c['checks']) for c in l4[n]] for n in ARMS}),
        runs=runs, terminal='L2L4_DONE'))
    print('L2', 'PASS' if l2_ok else 'FAIL', diff, 'L4', 'PASS' if l4_ok else 'FAIL', flush=True)
    return 0


# ------------------------------------------------------------------ LF6
def lf6(args):
    import apa_p1d_gpu as g
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ARMS:
        G.precheck(name)
    # This round adds and changes no test file: 2945c3e9's tensor_cuda/tests (git archive at TREE0) equals this
    # tree's (checked here), so both arms run this tree's tests from the worktree root.
    d = subprocess.run(['diff', '-rq', '-x', '__pycache__', str(TREE0 / 'tensor_cuda/tests'), str(ROOT / 'tensor_cuda/tests')],
                       capture_output=True, text=True)
    assert d.returncode == 0 and not d.stdout, ('tests differ from 2945c3e9', d.stdout[:500])
    trees = {'F3': ROOT, 'PF': ROOT}
    runs, results = [], []
    start = time.monotonic()
    for f in args.files:
        tag = Path(f).stem
        if time.monotonic() - start > args.max_seconds:
            results.append(dict(file=f, status='NOT_RUN_TIME_BUDGET'))
            continue
        recs = {}
        for name in ARMS:
            arm = json.loads((G.arm_dir(name) / 'ARM.json').read_text())
            pkg_parent = (arm_root(name) / arm['package']).parent
            env = dict(os.environ, LD_PRELOAD='', PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TMPDIR='/tmp/apap1pf2_l6',
                       PYTHONPATH=str(pkg_parent) + ':' + str(trees[name] / 'tensor_cuda/tests') + ':' +
                       str(ROOT / 'artifacts/apa_p1'))
            env.pop('TC_APA_SP_STORAGE', None)
            Path(env['TMPDIR']).mkdir(exist_ok=True)
            rec_path = out / f'{tag}_{name}.json'
            cmd = [sys.executable, str(ROOT / 'artifacts/apa_p1/pytest_receipt.py'), str(rec_path), '-q', '--tb=short',
                   '-ra', '-p', 'no:cacheprovider', '--basetemp', f'/tmp/apap1pf2_l6/bt_{tag}_{name}', f]
            t0 = time.monotonic()
            with (out / f'{tag}_{name}.log').open('x') as fh:
                try:
                    code = subprocess.run(cmd, cwd=trees[name], env=env, stdout=fh, stderr=subprocess.STDOUT,
                                          timeout=700).returncode
                except subprocess.TimeoutExpired:
                    code = 124
            runs.append(dict(file=f, arm=name, exit=code, seconds=time.monotonic() - t0))
            recs[name] = json.loads(rec_path.read_text()) if rec_path.exists() else None
        a, b = recs['F3'], recs['PF']
        if a is None or b is None:
            results.append(dict(file=f, status='FAIL_NO_RECEIPT'))
            continue
        rows = g.outcome_rows(a, b)
        results.append(dict(file=f, ids=len(rows), collected_equal=a['collection'] == b['collection'],
                            differences=sum(not r['equal'] for r in rows), outcomes_F3=g._count(a),
                            outcomes_PF=g._count(b), modules=[a['module']['sha256'], b['module']['sha256']],
                            rows=[r for r in rows if not r['equal']]))
    complete = all('ids' in r for r in results)
    ok = complete and all(r.get('differences') == 0 and r.get('collected_equal') and
                          r['modules'] == [G.receipt('F3')['module_sha256'], G.receipt('PF')['module_sha256']]
                          for r in results)
    G.save(out / 'RESULT.json', dict(line='LF6', status='PASS' if ok else ('FAIL' if complete else 'PARTIAL'),
                                     complete=complete, files=args.files, ids=sum(r.get('ids', 0) for r in results),
                                     results=results, runs=runs, terminal='LF6_DONE'))
    print('LF6', 'PASS' if ok else 'FAIL', sum(r.get('ids', 0) for r in results),
          [(r['file'].split('/')[-1], r.get('differences')) for r in results if r.get('differences')], flush=True)
    return 0


def pins(args):
    out = G.ART / 'PINS.json'
    files = ['artifacts/apa_p1d/packed_e_q_table.json', 'artifacts/apa_p1d/MARGIN_REGISTRATION.json']
    rows = []
    for f in files:
        head = subprocess.run(['git', 'show', f'2945c3e9:{f}'], cwd=ROOT, capture_output=True).stdout
        rows.append(dict(file=f, unchanged=(ROOT / f).read_bytes() == head, sha256=G.sha(ROOT / f)))
    rows.append(dict(file=str(G.DENSE_TABLE), sha256=G.sha(G.DENSE_TABLE), equals_pin=G.sha(G.DENSE_TABLE) == G.DENSE_TABLE_SHA))
    src = (ROOT / 'tensor_cuda/tensor_cuda/apa_packed.py').read_text()
    old = subprocess.run(['git', 'show', '2945c3e9:tensor_cuda/tensor_cuda/apa_packed.py'], cwd=ROOT,
                         capture_output=True, text=True).stdout
    keys = ('PACKED_REGISTRATION_SHA256 = ', 'PACKED_TABLE_SHA256 = ', 'PACKED_PINNED_ENTRIES = ')
    for kk in keys:
        a = [l for l in src.splitlines() if l.startswith(kk)]
        b = [l for l in old.splitlines() if l.startswith(kk)]
        rows.append(dict(constant=kk.strip(' ='), unchanged=a == b and len(a) == 1))
    golden = subprocess.run(['git', 'diff', '--stat', '2945c3e9', '--', 'artifacts/apa_sp2', 'artifacts/apa_sp1',
                             'artifacts/apa_p1d', 'tensor_cuda/tests', 'tensor_cuda/tensor_cuda/apa_sp2.py',
                             'tensor_cuda/tensor_cuda/quant.py', 'tensor_cuda/src/kernels.cu',
                             'tensor_cuda/src/apa_sp2.cuh', 'tensor_cuda/src/apa_sp1_1.cuh'],
                            cwd=ROOT, capture_output=True, text=True).stdout
    ok = all(r.get('unchanged', r.get('equals_pin')) for r in rows) and golden.strip() == ''
    G.save(out, dict(status='PASS' if ok else 'FAIL', rows=rows, dense_sp_goldens_and_sources_git_diff=golden))
    print('PINS', 'PASS' if ok else 'FAIL', golden[:500])
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode')
    ap.add_argument('--out')
    ap.add_argument('--arm')
    ap.add_argument('--model')
    ap.add_argument('--S', type=int)
    ap.add_argument('--files', nargs='*')
    ap.add_argument('--max-seconds', type=float, default=1500)
    args = ap.parse_args()
    if args.mode == 'pins':
        return pins(args)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') != '', 'GPU modes run inside forge-gpu-run'
    return {'lf1': lf1, 'lf1-arm': lf1_arm, 'l2l4': l2l4, 'l2-arm': l2_arm, 'l4-case': l4_case, 'lf6': lf6}[args.mode](args)


if __name__ == '__main__':
    raise SystemExit(main())
