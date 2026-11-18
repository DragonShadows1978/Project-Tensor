"""APA-P1-PF2 Amendment 1, A1.1 (CPU): PF-I3 per dtype from G1's parts. tau_dtype = 4 x the largest all-refined
relative row difference (prefill vs packed decode) of that dtype on PF-I1's grid, no cap; rows above tau at the
registered delta: 0 allowed. Also used by the mutant line for m12 (A1.2) with --parts <dir> --tau-from <PF parts>.
Prior art: none (bookkeeping over r1's receipts)."""
import glob, json, sys
from pathlib import Path


def per_dtype(parts_dir):
    tau_rows = [t for p in sorted(glob.glob(str(Path(parts_dir) / 'part_*.json'))) if '_stopped' not in p
                for t in json.load(open(p))['tau_rows']]
    out = {}
    for dt in ('float32', 'bfloat16'):
        rows = [t for t in tau_rows if f'_{dt}_' in t[0]]
        out[dt] = dict(cases=len(rows), all_refined=[x for _, _, xs in rows for x in xs],
                       at_delta=[(cid, x) for cid, xs, _ in rows for x in xs])
    return out


def verdict(parts_dir, tau=None):
    d = per_dtype(parts_dir)
    res = {}
    for dt, v in d.items():
        if not v['cases']:
            continue
        mx = max(v['all_refined'])
        t = 4.0 * mx if tau is None else tau[dt]
        above = [(c, x) for c, x in v['at_delta'] if x > t]
        res[dt] = dict(cases=v['cases'], rows=len(v['at_delta']), all_refined_max=mx, tau=t,
                       max_at_delta=max(x for _, x in v['at_delta']), rows_above_tau=len(above),
                       cases_above_tau=sorted({c for c, _ in above})[:20])
    res['status'] = 'PASS' if all(r['rows_above_tau'] == 0 for r in res.values()) else 'FAIL'
    return res


if __name__ == '__main__':
    res = verdict(sys.argv[1])
    Path(sys.argv[2]).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1))
