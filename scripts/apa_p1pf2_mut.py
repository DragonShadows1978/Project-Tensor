#!/usr/bin/env python3
"""APA-P1-PF2 G7: run each registered mutant (artifacts/apa_p1pf2/mutants/MUTANTS.json) inside forge-gpu-run.

  mut --out D            : for each mutant, a child on the mutant's arm and a control child on Build PF (PFH for m13)
                           run the mutant's registered check; killed iff its expected_failure assertion fails on the
                           mutant (and holds on the control)
  child --arm A --name M : (child) the check, printing MUT_JSON
Checks and their assertions:
  g1 (subset)   : pf_i1_mask_equal, pf_i2_equal, pf_i3_rows_within_tau (A1.1's per-dtype tau from
                  artifacts/apa_p1pf2/a1/PF_I3_A1.json, computed on Build PF's G1 receipts)
  g1h (subset)  : pf_i1_mask_equal, pf_i1_bulk_equal (hooks build)
  ties          : pf_i1_ties (the 32 registered tie cases)
  g4_accept     : g4_route_record_packed_delta, g4_public_equals_raw_entry_bitwise
  g5            : g5_no_key_shaped_transient (scripts/apa_p1pf2_g45.py g5-case under its allocation shim)
  i1n (subset)  : pf_i1n_ok
Prior art: mutation testing (DeMillo, Lipton and Sayward, 1978); F1-F3 mutant lines (Project-Tensor 2026). Plumbing.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import apa_p1pf2_gpu as G  # noqa: E402

MUTS = json.loads((G.ART / 'mutants/MUTANTS.json').read_text())['mutants']


def subset(cases, sub):
    out = []
    for i, c in enumerate(cases):
        ok = True
        for k, vals in sub.items():
            key = 'cap' if k == 'capx' else k
            val = c['cap'] - c['S'] if k == 'capx' else c.get(key)
            if val not in vals:
                ok = False
        if ok:
            out.append((i, c))
    return out


def child(args):
    import torch
    spec = {m['name']: m for m in MUTS}[args.name]
    run = spec['run']
    fails, info = set(), {}
    if run['check'] == 'g5':
        log = Path(args.out) / f'g5_{args.name}_{args.arm.replace(":", "_")}.log'
        with log.open('x') as f:
            subprocess.run([sys.executable, '-B', '-P', str(ROOT / 'scripts/apa_p1pf2_g45.py'), 'g5-case', '--arm',
                            args.arm, '--model', 'qwen3_1.7b'], stdout=f, stderr=subprocess.STDOUT, timeout=900,
                           cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        line = [x for x in log.read_text().splitlines() if x.startswith('G5_JSON ')]
        rows = json.loads(line[-1][8:])['rows'] if line else []
        info['rows'] = [dict(S=r['S'], L=r['L'], pooling=r['pooling'], diag=r['diagnostics'],
                             hw=r['high_water_above_resident'], bound=r['bound']) for r in rows]
        if not rows or not all(r['no_key_shaped_transient'] for r in rows):
            fails.add('g5_no_key_shaped_transient')
        print('MUT_JSON ' + json.dumps(dict(arm=args.arm, failed=sorted(fails), info=info)), flush=True)
        return 0
    tc, arm = G.load_arm(args.arm)
    os.environ['TC_APA_SP'] = '1'
    os.environ.pop('TC_APA_SPQ1_REFERENCE', None)
    os.environ.pop('TC_APA_SP_STORAGE', None)
    E = G.Env(tc, torch)
    with tc.no_grad():
        if run['check'] in ('g1', 'g1h'):
            hooks = run['check'] == 'g1h'
            tau = {k: v['tau'] for k, v in json.loads((G.ART / 'a1/PF_I3_A1.json').read_text()).items()
                   if k in ('float32', 'bfloat16')}
            recs, stats, tau_rows = [], {}, []
            for idx, c in subset(G.g1_cases(), run['subset']):
                recs.append(G.run_case(E, c, idx, hooks, stats, tau_rows))
            info['cases'] = len(recs)
            info['rows'] = stats.get('rows')
            if stats.get('mask_unequal_rows', 0) or any(r['tail_zero_rows'] != r['rows'] for r in recs):
                fails.add('pf_i1_mask_equal')
            if hooks:
                if stats.get('bulk_unequal_rows', 0):
                    fails.add('pf_i1_bulk_equal')
            else:
                if not all(r['pf_i2']['equal'] and r['pf_i2']['all_refined_prefill'] for r in recs):
                    fails.add('pf_i2_equal')
                above = sum(x > tau[dt] for cid, xs, _ in tau_rows for dt in tau if f'_{dt}_' in cid for x in xs)
                info['pf_i3_rows_above_tau'] = int(above)
                info['pf_i3_max'] = max((x for _, xs, _ in tau_rows for x in xs), default=None)
                if above:
                    fails.add('pf_i3_rows_within_tau')
            info['mask_unequal_rows'] = stats.get('mask_unequal_rows', 0)
            info['bulk_unequal_rows'] = stats.get('bulk_unequal_rows', 0)
        elif run['check'] == 'ties':
            res = [G.run_tie(E, c, i)[0] for i, c in enumerate(G.tie_cases())]
            info['ties_ok'] = sum(bool(r.get('ok')) for r in res)
            if info['ties_ok'] != len(res):
                fails.add('pf_i1_ties')
        elif run['check'] == 'i1n':
            dm = tc.RegisteredMargins.load(G.DENSE_TABLE, expected_sha256=G.DENSE_TABLE_SHA)
            res = [G.run_nonfinite(E, c, i, dm)[0] for i, c in enumerate(G.nonfinite_cases())
                   if c['D'] in run['subset']['D'] and c['kind'] in run['subset']['kind']]
            info['cases'], info['ok'] = len(res), sum(r['ok'] for r in res)
            info['finite_where_dense_nonfinite'] = sum(r['finite_where_dense_nonfinite'] for r in res)
            if info['ok'] != len(res):
                fails.add('pf_i1n_ok')
        elif run['check'] == 'g4_accept':
            rng = np.random.default_rng(np.random.SeedSequence([20261116, 4]))
            D, S, KV, H, L = 128, 257, 2, 8, 33
            k = tc.tensor(rng.standard_normal((1, KV, S, D), dtype=np.float32))
            v = tc.tensor(rng.standard_normal((1, KV, S, D), dtype=np.float32))
            q = tc.tensor((3 * rng.standard_normal((1, H, L, D))).astype(np.float32))
            pm = tc.PackedRegisteredMargins.load(G.PACKED_TABLE)
            scale = float(np.float32(1 / math.sqrt(D)))
            p = tc.quantize_sp2_keys(k, 4)
            out, mask = tc.apa_selective_attention_sp(q, k, p, v, scale, .01, bulk_bits=4, margins=pm, is_causal=True,
                                                      diagnostics=True)
            route = tc.apa_packed.last_route('attention')
            e_pin = tc.apa_packed.PACKED_PINNED_ENTRIES['4:128']
            d_pin = tc.apa_packed.derive_delta(.01, e_pin)
            ro, rm = tc._spo2.packed_prefill(q, k, v, p._packed_tuple(), scale, d_pin, True, None, True, fmt='tq4')
            info['route'] = route
            if not (route and route.get('backend') == 'spq_packed_prefill_tq4_row' and route.get('e_q') == e_pin and
                    route.get('delta') == d_pin):
                fails.add('g4_route_record_packed_delta')
            if not (np.array_equal(out.numpy().view(np.uint32), ro.numpy().view(np.uint32)) and
                    np.array_equal(mask.numpy(), rm.numpy())):
                fails.add('g4_public_equals_raw_entry_bitwise')
    torch.cuda.synchronize()
    print('MUT_JSON ' + json.dumps(dict(arm=args.arm, failed=sorted(fails), info=info), default=str), flush=True)
    return 0


def mut(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    only = set(args.only.split(',')) if args.only else None
    for spec in MUTS:
        if only and spec['name'] not in only:
            continue
        name = spec['name']
        sides = {}
        for side, armname in (('mutant', 'mutant:' + name), ('control', spec.get('base', 'PF'))):
            G.precheck(armname)
            logf = out / f'{name}_{side}.log'
            t0 = time.time()
            with logf.open('x') as f:
                try:
                    code = subprocess.run([sys.executable, '-B', '-P', __file__, 'child', '--arm', armname, '--name', name,
                                           '--out', str(out)], stdout=f, stderr=subprocess.STDOUT, timeout=900, cwd=ROOT,
                                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1')).returncode
                except subprocess.TimeoutExpired:
                    code = 124
            line = [x for x in logf.read_text().splitlines() if x.startswith('MUT_JSON ')]
            sides[side] = dict(exit=code, seconds=round(time.time() - t0, 1),
                               **(json.loads(line[-1][9:]) if line else dict(failed=None)))
        exp = spec['expected_failure']
        killed = sides['mutant'].get('failed') is not None and exp in sides['mutant']['failed']
        control_ok = sides['control'].get('failed') == []
        results.append(dict(name=name, design=spec['design'], expected_failure=exp, killed=killed,
                            control_clean=control_ok, mutant_failed=sides['mutant'].get('failed'),
                            mutant_exit=sides['mutant']['exit'], executes=spec['executes'], sides=sides))
        print('MUT', name, 'KILLED' if killed else 'SURVIVED', 'control', 'clean' if control_ok else 'NOT CLEAN',
              sides['mutant'].get('failed'), flush=True)
    status = 'PASS' if len(results) == 16 and all(r['killed'] and r['control_clean'] for r in results) else 'FAIL'
    G.save(out / f'MUTANT_RESULTS{"_" + args.tag if args.tag else ""}.json',
           dict(line='G7', status=status, killed=sum(r['killed'] for r in results), total=len(results),
                results=results, terminal='MUT_DONE'))
    print('G7', status, sum(r['killed'] for r in results), '/', len(results), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode')
    ap.add_argument('--out')
    ap.add_argument('--arm')
    ap.add_argument('--name')
    ap.add_argument('--only')
    ap.add_argument('--tag')
    args = ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') != '', 'GPU modes run inside forge-gpu-run'
    return {'mut': mut, 'child': child}[args.mode](args)


if __name__ == '__main__':
    raise SystemExit(main())
