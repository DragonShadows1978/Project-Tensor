"""P1T-M3 G3 summary: reads each mutant's G1 log/receipt and G2 reports (artifacts/p1t_m3/mutants/<id>/) plus the clean
G2 report, and writes mutant_summary.json next to this file. The baseline failure (M2's
test_default_allocator_names_set_allocator, pre-existing with M1's core; ledger 20:50) is listed separately from the
mutant's own failures. Prior art: none (bookkeeping); mutation testing per DeMillo, Lipton & Sayward (1978)."""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLEAN_T4 = HERE.parent / 'p1t_T4.json'
BASELINE_FAIL = 'tc_signal/tests/test_contract.py::test_default_allocator_names_set_allocator'


def g1(mid):
    log = (HERE / mid / 'g1.log').read_text()
    failed = re.findall(r'^FAILED (\S+)', log, re.M)
    last = [ln for ln in log.splitlines() if re.search(r'\d+ (passed|failed)', ln)][-1]
    own = [f for f in failed if f != BASELINE_FAIL]
    rec = json.loads((HERE / mid / 'g1_receipt.json').read_text())
    lstm = [c for c in rec['cells'] if c.get('op') == 'lstm']
    bad = [c for c in lstm if not c['pass']]
    nonfinite = [c['cell'] for c in bad if not c.get('all_finite', True)]
    nondet = [c['cell'] for c in bad if not c.get('deterministic', True)]
    ns = [c['N'] for c in bad if 'N' in c and isinstance(c['N'], float)]
    probe = next((c for c in lstm if c.get('kind') == 'probe'), None)
    return {'summary_line': last.strip(), 'failed_total': len(failed), 'baseline_failure_present': BASELINE_FAIL in failed,
            'failed_excluding_baseline': len(own), 'failed_ids_excluding_baseline': own,
            'lstm_cells_failed': len(bad), 'lstm_failed_N_min': min(ns) if ns else None,
            'lstm_failed_N_max': max(ns) if ns else None, 'nonfinite_cells': nonfinite,
            'nondeterministic_cells': nondet,
            'probe': None if probe is None else {k: probe[k] for k in ('sigmoid_max_ulp', 'sigmoid_argmax_z',
                                                                         'tanh_max_ulp', 'tanh_of_state_max_ulp',
                                                                         'pass')}}


def g2(path, clean=None):
    r = json.loads(Path(path).read_text())
    s = r['summary']
    t4 = [c for c in r['cases'] if c['item'] == 'T4']
    out = {'verdict': s['verdict'], 'exit_code': s['exit_code'], 'by_item': s['by_item'], 'counts': s['counts'],
           'T4_failed': sum(c['verdict'] != 'PASS' for c in t4),
           'T4_N_min': min(c['N'] for c in t4 if c.get('N') is not None) if t4 else None,
           'T4_N_max': max(c['N'] for c in t4 if c.get('N') is not None) if t4 else None,
           'T4_nondeterministic': sum(not c.get('deterministic', True) for c in t4)}
    if clean is not None and t4:
        cn = {c['id']: c['N'] for c in clean['cases'] if c['item'] == 'T4'}
        d = [(c['id'], c['N'], cn[c['id']]) for c in t4]
        out['T4_N_vs_clean'] = [{'id': i, 'N': n, 'N_clean': nc, 'delta': n - nc} for i, n, nc in d]
        out['T4_max_abs_delta_N'] = max(abs(n - nc) for _, n, nc in d)
        out['T4_identical_N_cases'] = sum(n == nc for _, n, nc in d)
    return out


def main():
    clean = json.loads(CLEAN_T4.read_text())
    res = {}
    for mid in sys.argv[1:] or ['a', 'b', 'c', 'd', 'e']:
        res[mid] = {'desc': (HERE / mid / 'MUTANT.txt').read_text().splitlines()[0],
                    'so_sha256': (HERE / mid / 'so_sha256.txt').read_text().split()[0],
                    'G1': g1(mid), 'G2_T4': g2(HERE / mid / 'g2_T4.json', clean),
                    'G2_T1toT4': g2(HERE / mid / 'g2_T1toT4.json')}
    (HERE / 'mutant_summary.json').write_text(json.dumps(res, indent=1) + '\n')
    for mid, r in res.items():
        a, b, c = r['G1'], r['G2_T4'], r['G2_T1toT4']
        print(f"({mid}) {r['desc']}")
        print(f"   G1: {a['summary_line']} | excl. baseline: {a['failed_excluding_baseline']} failed; LSTM cells failed "
              f"{a['lstm_cells_failed']} (N {a['lstm_failed_N_min']} .. {a['lstm_failed_N_max']}); nondet "
              f"{len(a['nondeterministic_cells'])}; probe {a['probe']}")
        print(f"   G2 T4: {b['verdict']} exit {b['exit_code']} {b['counts']} N {b['T4_N_min']:.3e}..{b['T4_N_max']:.3e} "
              f"max|dN| vs clean {b['T4_max_abs_delta_N']:.3e} identical {b['T4_identical_N_cases']}/12")
        print(f"   G2 T1-T4: {c['verdict']} exit {c['exit_code']} {c['by_item']}")


if __name__ == '__main__':
    main()
