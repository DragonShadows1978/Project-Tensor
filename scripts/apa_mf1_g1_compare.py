"""APA-MF1 G1: per-test-id outcome comparison of two junit xml files."""
import sys, json, xml.etree.ElementTree as ET
def load(p):
    st = {}
    for tc in ET.parse(p).iter('testcase'):
        k = tc.get('classname') + '::' + tc.get('name'); s = 'passed'
        for ch in tc:
            if ch.tag in ('failure', 'error', 'skipped'): s = ch.tag
        st[k] = s
    return st
a, b = load(sys.argv[1]), load(sys.argv[2])
changed = {k: (a[k], b.get(k, 'MISSING')) for k in a if a[k] != b.get(k, 'MISSING')}
new = {k: v for k, v in b.items() if k not in a}
from collections import Counter
out = {"baseline": dict(Counter(a.values())), "after": dict(Counter(b.values())), "n_baseline_ids": len(a),
       "changed_on_baseline_ids": changed, "new_ids": dict(Counter(new.values())),
       "new_non_mf1": {k: v for k, v in new.items() if 'test_apa_mf1' not in k}}
print(json.dumps(out, indent=1))
