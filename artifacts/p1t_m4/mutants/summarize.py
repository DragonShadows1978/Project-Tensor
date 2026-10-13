"""P1T-M4 G3 summary: per mutant, the G1 failures (net of the pre-existing M2 id) and the G2 runner failures, from the
receipts in /tmp/p1t_m4_mut_<id>/ (or a directory given as argv[1] holding <id>/ copies). Writes mutant_summary.json."""
import json
import re
import sys
from pathlib import Path

PRE = 'tc_signal/tests/test_contract.py::test_default_allocator_names_set_allocator'
base = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else None
out = {}
for m in 'abcdefg':
    R = (base / m) if base else Path(f'/tmp/p1t_m4_mut_{m}')
    g1 = (R / 'g1.log').read_text()
    failed = sorted(set(re.findall(r'^FAILED (\S+?)(?: - .*)?$', g1, re.M)))
    net = [f for f in failed if f != PRE]
    rec = {'desc': (R / 'MUTANT.txt').read_text().strip(), 'g1_line': re.findall(r'^\d+ (?:failed|passed).*$', g1, re.M)[-1],
           'g1_failed_net': net, 'g1_pre_existing_failed': PRE in failed, 'g1_killed': bool(net)}
    try:
        cells = json.loads((R / 'g1_receipt.json').read_text())['cells']
        rec['g1_failed_cells'] = [{k: c.get(k) for k in ('op', 'cell', 'N', 'bit_mismatches', 'deterministic')}
                                  for c in cells if not c.get('pass')]
    except FileNotFoundError:
        rec['g1_failed_cells'] = None
    for tag in ('T5T6', 'regression'):
        rep = json.loads((R / f'g2_{tag}.json').read_text())
        bad = []
        for c in rep['cases']:
            if c['verdict'] != 'PASS':
                mism = sum(o.get('exact_bit_mismatches', 0) for o in c.get('outputs', []))
                bad.append({'id': c['id'], 'N': c.get('N'), 'exact_bit_mismatches': mism or None,
                            'deterministic': c.get('deterministic'), 'exception': c.get('adapter_exception')})
        rec[f'g2_{tag}'] = {'verdict': rep['summary']['verdict'], 'exit_code': rep['summary']['exit_code'],
                            'counts': rep['summary']['counts'], 'failed': bad}
    rec['g2_killed'] = rec['g2_T5T6']['exit_code'] != 0 or rec['g2_regression']['exit_code'] != 0
    rec['killed'] = rec['g1_killed'] or rec['g2_killed']
    out[m] = rec
    print(f"({m}) {rec['desc']}\n    G1: {rec['g1_line']}; net failed {len(net)} (pre-existing id failed: "
          f"{rec['g1_pre_existing_failed']}) -> {'KILLED' if rec['g1_killed'] else 'survived'}")
    for c in (rec['g1_failed_cells'] or []):
        print(f"      cell {c['op']}/{c['cell']}: N {c['N']} bits {c['bit_mismatches']} det {c['deterministic']}")
    other = [f.split('::')[-1] for f in net if 'census' not in f]
    if other:
        print(f"      other G1 failures: {other}")
    for tag in ('T5T6', 'regression'):
        g = rec[f'g2_{tag}']
        print(f"    G2 {tag}: verdict {g['verdict']} exit {g['exit_code']} counts {g['counts']}")
        for b in g['failed']:
            print(f"      {b['id']}: N {b['N']} bits {b['exact_bit_mismatches']} det {b['deterministic']}"
                  + (f" exc {b['exception']}" if b['exception'] else ''))
    print(f"    => {'KILLED' if rec['killed'] else 'SURVIVED'}")
Path(sys.argv[2] if len(sys.argv) > 2 else 'mutant_summary.json').write_text(json.dumps(out, indent=1) + '\n')
