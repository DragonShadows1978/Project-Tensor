import json, re, sys
from pathlib import Path
out = {}
for m in 'abcd':
    R = Path(f'/tmp/p1t_m2_mut_{m}')
    g1log = (R / 'g1.log').read_text()
    failed = re.findall(r'^FAILED (\S+)', g1log, re.M)
    rec = json.loads((R / 'g1_receipt.json').read_text())['cells']
    cfail = [c for c in rec if not c['pass']]
    rep = json.loads((R / 'g2_report.json').read_text())
    cs = rep['cases']
    g2fail = [c for c in cs if c['verdict'] == 'FAIL']
    nondet = [c for c in cs if c.get('deterministic') is False]
    gatefail = [c for c in cs if c.get('gate_ok') is False]
    d = {
        'desc': (R / 'MUTANT.txt').read_text().strip(),
        'g1_summary': g1log.strip().splitlines()[-1],
        'g1_failed_tests': failed,
        'g1_census_failed_cells': [{'op': c['op'], 'cell': c['cell'], 'N': c['N'], 'deterministic': c['deterministic'],
                                    'mismatches': c['mismatches']} for c in cfail],
        'g1_census_nondeterministic': sum(1 for c in rec if not c['deterministic']),
        'g1_census_cells': len(rec),
        'g2_verdict': rep['summary']['verdict'], 'g2_exit': rep['summary']['exit_code'],
        'g2_counts': rep['summary']['counts'], 'g2_by_item': rep['summary']['by_item'],
        'g2_gate_failures': len(gatefail), 'g2_nondeterministic_cases': len(nondet),
        'g2_nondet_detail': [{'id': c['id'], 'mismatches': [o.get('determinism_bit_mismatches') for o in c['outputs']],
                              'numel': None, 'N': c['N']} for c in nondet],
        'g2_max_N': max((c['N'] for c in cs if isinstance(c.get('N'), float)), default=None),
        'g2_failed_ids': [c['id'] for c in g2fail],
    }
    if gatefail:
        d['g2_min_failing_N'] = min(c['N'] for c in gatefail)
    out[m] = d
json.dump(out, open(sys.argv[1], 'w'), indent=1)
for m, d in out.items():
    print(f"== mutant {m}: {d['desc']}")
    print(f"   G1: {d['g1_summary']}; census cells failing {len(d['g1_census_failed_cells'])}/{d['g1_census_cells']}, nondeterministic {d['g1_census_nondeterministic']}")
    print('   G1 failed tests:', [t.split('::')[-1] for t in d['g1_failed_tests']])
    for c in d['g1_census_failed_cells'][:30]:
        print(f"      {c['op']:17s} {c['cell']:28s} N={c['N']:.3e} det={c['deterministic']} mism={c['mismatches']}")
    print(f"   G2: verdict {d['g2_verdict']} exit {d['g2_exit']} counts {d['g2_counts']} by_item {d['g2_by_item']} gate-failures {d['g2_gate_failures']} nondet {d['g2_nondeterministic_cases']} max N {d['g2_max_N']}")
    if 'g2_min_failing_N' in d: print(f"   G2 min failing N {d['g2_min_failing_N']:.3e}")
