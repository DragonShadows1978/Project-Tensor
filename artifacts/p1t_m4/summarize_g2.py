"""Per-op summary of a Voice runner report (P1T-M4 G2 receipt helper; reads the JSON only)."""
import collections, json, sys
d = json.load(open(sys.argv[1]))
g = collections.defaultdict(list)
for c in d['cases']:
    g[(c['item'], c['op_type'])].append(c)
print(f"report {sys.argv[1]}\nverdict {d['summary']['verdict']} exit {d['summary']['exit_code']} counts {d['summary']['counts']}")
for (item, op), cs in sorted(g.items()):
    ex = [o.get('exact_bits_equal') for c in cs for o in c.get('outputs', [])]
    Ns = [c['N'] for c in cs if isinstance(c.get('N'), (int, float))]
    worst = max(cs, key=lambda c: c['N'] if isinstance(c.get('N'), (int, float)) else -1)
    exact = 'n/a' if all(e is None for e in ex) else f"{sum(1 for e in ex if e)}/{len(ex)} bit-exact"
    print(f"{item} {op:24s} cases {len(cs):3d} PASS {sum(c['verdict'] == 'PASS' for c in cs):3d} "
          f"det {sum(bool(c.get('deterministic')) for c in cs):3d} exact {exact:16s} "
          + (f"max N {max(Ns):.3e} ({worst['id']})" if Ns else 'no N (T1 protocol checks)'))
