"""ENG4-F3 K3 compare: per test ID, Build F2's outcome against Build F3's; every ID must be identical (the order lists
no intended change), and both series complete with the same collection errors.

Usage: python3 -B -P scripts/eng4_f3_g4_compare.py <series_F2> <series_F3> <out.json>   (series under artifacts/eng4/f3/g4)

Prior art: a keyed diff of two outcome maps (ENG4's eng4_g4_compare.py, this repo). No other prior art known to me.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    s2, s3, out = sys.argv[1:4]
    o2 = json.load(open(os.path.join(ROOT, "artifacts/eng4/f3/g4", s2, "outcomes.json")))
    o3 = json.load(open(os.path.join(ROOT, "artifacts/eng4/f3/g4", s3, "outcomes.json")))
    a, b = o2["outcomes"], o3["outcomes"]
    diffs = [{"id": t, "F2": a.get(t), "F3": b.get(t)} for t in sorted(set(a) | set(b)) if a.get(t) != b.get(t)]
    coll = {"F2": sorted(o2["collection_errors"]), "F3": sorted(o3["collection_errors"])}
    ok = o2.get("complete") and o3.get("complete") and not diffs and coll["F2"] == coll["F3"]
    rec = {"F2": s2, "F3": s3, "F2_counts": o2["counts"], "F3_counts": o3["counts"], "ids_F2": len(a), "ids_F3": len(b),
           "collection_errors": coll, "differences": diffs,
           "worktree_writes_by_tests": {"F2": o2.get("worktree_writes_by_tests"), "F3": o3.get("worktree_writes_by_tests")},
           "verdict": "PASS" if ok else "FAIL"}
    json.dump(rec, open(out, "w"), indent=1)
    print("K3", s2, "vs", s3, rec["verdict"], "ids", len(a), len(b), "differences", len(diffs))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
