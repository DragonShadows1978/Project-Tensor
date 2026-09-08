#!/usr/bin/env python3
"""APA-SP5 amendment 1 -- the aggregate cell over the N=16 per-window cells.

Method (as the amendment specifies):
  * pooled ppl per arm = exp( sum(nll) / sum(targets) ) over all 16 windows
    = 8,192 targets. Pooling NLL sums (not averaging per-window ppl) is the
    correct aggregate: it is the perplexity of the concatenated target set,
    and every window contributes exactly its 512 targets.
  * the floor is a DISTRIBUTION: per-window |A - A32| and its relative form,
    plus the pooled |A - A32|.
  * per window, each arm's deviation from A is reported in multiples of THAT
    WINDOW'S OWN floor -- the r1 finding was that a floor from one window is
    the wrong instrument for another.

CPU only. No GPU, no lease. Reads only committed per-window cell receipts.

Prior art: SP3/SP4G aggregate reporting (this repo, 2026); SP4G amendment 6
floor construction. Pooling by NLL sum is standard perplexity practice.
"""
import json, glob, sys
from pathlib import Path
import numpy as np

A = Path('/mnt/ForgeRealm/Project-Tensor-wt-apa-sp5/artifacts/apa_sp5')
ARMS = ('A', 'A32', 'B', 'C', 'D', 'E')


def load(arm):
    d = {}
    for f in sorted(glob.glob(str(A / 'windows_a1' / f'ppl_{arm}_W1024_w*.json'))):
        r = json.loads(Path(f).read_text())
        if r.get('status') != 'PASS':
            raise SystemExit(f'non-PASS cell {f}')
        d[r['window']] = r
    return d


def main():
    cells = {a: load(a) for a in ARMS}
    ws = sorted(set.intersection(*[set(c) for c in cells.values()]))
    if len(ws) != 16:
        print(f'WARNING: only {len(ws)} complete windows: {ws}', file=sys.stderr)

    def pooled(arm):
        d = cells[arm]
        n = sum(d[w]['nll_sum'] for w in ws)
        t = sum(d[w]['targets'] for w in ws)
        return float(np.exp(n / t)), int(t)

    pool = {a: pooled(a)[0] for a in ARMS}
    targets = pooled('A')[1]
    floor_w = {w: abs(cells['A'][w]['ppl'] - cells['A32'][w]['ppl']) for w in ws}
    floor_rel = {w: floor_w[w] / cells['A'][w]['ppl'] for w in ws}
    floor_signed = {w: cells['A'][w]['ppl'] - cells['A32'][w]['ppl'] for w in ws}
    pooled_floor = abs(pool['A'] - pool['A32'])

    per_window = []
    for w in ws:
        row = dict(window=w, targets=cells['A'][w]['targets'],
                   A=cells['A'][w]['ppl'], A32=cells['A32'][w]['ppl'],
                   floor=floor_w[w], floor_relative=floor_rel[w],
                   floor_signed=floor_signed[w])
        for a in ('B', 'C', 'D', 'E'):
            d = cells[a][w]['ppl'] - cells['A'][w]['ppl']
            row[a] = cells[a][w]['ppl']
            row[f'{a}_minus_A'] = d
            row[f'{a}_in_floors'] = (d / floor_w[w]) if floor_w[w] else None
        row['C_minus_B'] = cells['C'][w]['ppl'] - cells['B'][w]['ppl']
        per_window.append(row)

    signs = {}
    for a in ('B', 'C', 'D', 'E'):
        above = sum(1 for w in ws if cells[a][w]['ppl'] > cells['A'][w]['ppl'])
        signs[f'{a}_above_A'] = f'{above}/{len(ws)}'
    signs['C_below_B'] = (f"{sum(1 for w in ws if cells['C'][w]['ppl'] < cells['B'][w]['ppl'])}"
                          f"/{len(ws)}")

    rel = np.array([floor_rel[w] for w in ws])
    res = dict(
        cell='aggregate_a1_N16', amendment='APA_SP5_AMENDMENT_1',
        amendment_sha256='06a5a1e742ae2474469f8d5177bc5b9d35b4d89e9553d928c9cac1d7a6753e33',
        registration_sha256='3cc3b3e112479ad71077a98a221fd5af521013d16ce423d52afe308e9ae58159',
        windows=ws, total_targets=targets,
        method=('pooled ppl = exp(sum(nll)/sum(targets)) over all windows; the '
                'floor is reported as a distribution (per-window and pooled); '
                "each arm's per-window deviation from A is expressed in "
                "multiples of THAT window's own floor"),
        pooled_ppl=pool,
        pooled_deltas={f'{a}_minus_A': pool[a] - pool['A'] for a in ('B', 'C', 'D', 'E')},
        pooled_C_minus_B=pool['C'] - pool['B'],
        pooled_floor=pooled_floor,
        pooled_floor_relative=pooled_floor / pool['A'],
        floor_distribution=dict(
            per_window_relative={str(w): floor_rel[w] for w in ws},
            median_relative=float(np.median(rel)), mean_relative=float(rel.mean()),
            min_relative=float(rel.min()), max_relative=float(rel.max()),
            positive_windows=sum(1 for w in ws if floor_signed[w] > 0),
            negative_windows=sum(1 for w in ws if floor_signed[w] < 0),
            signed_sum=float(sum(floor_signed.values())),
            note=('the sign is mixed, so pooling CANCELS scatter rather than '
                  'accumulating it -- this is why the pooled floor is far '
                  'smaller than any single-window floor')),
        resolvability={
            f'{k}': dict(delta=v, pooled_floor=pooled_floor,
                         in_floors=v / pooled_floor,
                         verdict='RESOLVABLE' if abs(v) > pooled_floor
                                 else 'not resolvable')
            for k, v in dict(
                B_minus_A=pool['B'] - pool['A'], C_minus_A=pool['C'] - pool['A'],
                D_minus_A=pool['D'] - pool['A'], E_minus_A=pool['E'] - pool['A'],
                C_minus_B=pool['C'] - pool['B']).items()},
        sign_counts=signs,
        per_window=per_window)
    out = A / 'aggregate_a1_N16.json'
    if out.exists():
        out = A / 'aggregate_a1_N16.rerun.json'
    with out.open('x') as f:
        json.dump(res, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({k: res[k] for k in
                      ('total_targets', 'pooled_ppl', 'pooled_deltas',
                       'pooled_C_minus_B', 'pooled_floor',
                       'pooled_floor_relative', 'resolvability',
                       'sign_counts')}, indent=2))
    print('artifact=' + str(out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
