"""BWD-F3 P3 (a copy of scripts/bwdf2_k3_compare.py): production's suite per test ID, Build B against Build F3
(series under artifacts/bwdf1/f3/g4). Allowed differences: BWD-F1's two listed H1 IDs
(artifacts/bwdf1/INTENDED_CHANGES.json), BWD-F2's production list and E6's production list
(artifacts/bwdf1/f2/INTENDED_CHANGES_F2.json, artifacts/bwdf1/f3/INTENDED_CHANGES_F3.json; both empty). Anything else
is unintended (P3 FAIL).

  python3 -B -P scripts/bwdf3_p3_compare.py <series_B> <series_F3> <out json>
Prior art: ENG4 / BWD-F1's per-ID comparisons (this repo). Ours: nothing else.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sb, sf, outp = sys.argv[1], sys.argv[2], sys.argv[3]
B = json.load(open(os.path.join(ROOT, "artifacts/bwdf1/f3/g4", sb, "outcomes.json")))
F = json.load(open(os.path.join(ROOT, "artifacts/bwdf1/f3/g4", sf, "outcomes.json")))
h1 = [c["test_id"] for c in
      json.load(open(os.path.join(ROOT, "artifacts/bwdf1/INTENDED_CHANGES.json")))["existing_test_changes"]]
r12 = [c["test"] for c in json.load(open(os.path.join(ROOT, "artifacts/bwdf1/f2/INTENDED_CHANGES_F2.json")))
       ["production_tests_changed"]]
e6 = [c["test"] for c in json.load(open(os.path.join(ROOT, "artifacts/bwdf1/f3/INTENDED_CHANGES_F3.json")))
      ["production_tests_changed"]]
allowed = set(h1) | set(r12) | set(e6)
ob, of = B["outcomes"], F["outcomes"]
diffs = {k: [ob.get(k), of.get(k)] for k in sorted(set(ob) | set(of)) if ob.get(k) != of.get(k)}
unintended = {k: v for k, v in diffs.items() if k not in allowed}
res = {"series": [sb, sf], "complete": [B["complete"], F["complete"]], "n_ids": [B["n_ids"], F["n_ids"]],
       "counts": [B["counts"], F["counts"]], "collection_errors": [sorted(B["collection_errors"]),
                                                                  sorted(F["collection_errors"])],
       "allowed": sorted(allowed), "differences": diffs, "unintended": unintended,
       "verdict": "PASS" if B["complete"] and F["complete"] and not unintended and
       sorted(B["collection_errors"]) == sorted(F["collection_errors"]) else "FAIL"}
json.dump(res, open(os.path.join(ROOT, outp), "w"), indent=1)
print(json.dumps({k: res[k] for k in ("n_ids", "counts", "differences", "unintended", "verdict")}, indent=1))
