"""ENG4-F1 H3 compare (copy of scripts/eng4_g4_compare.py reading artifacts/eng4/f1/g4 and f1/INTENDED_CHANGES_F1.json; the listed IDs are its carried ENG4 M7 IDs plus its new_in_F1 IDs). ENG4 G4 compare: per test ID, Build B's outcome against Build E's (L09: outcomes per ID; logs and timings are
never compared). Every difference must be an ID listed in artifacts/eng4/INTENDED_CHANGES.json (written before the
after-run), else G4 fails.

Usage: python3 -B -P scripts/eng4_g4_compare.py <series_B> <series_E> <out.json>
       (series dirs under artifacts/eng4/g4/, each holding the merged outcomes.json of scripts/eng4_suite.py)

Prior art: no prior art known to me beyond a keyed diff of two outcome maps (ENG3's eng3 G3 compare, this repo).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    sb, se, out = sys.argv[1], sys.argv[2], sys.argv[3]
    ob = json.load(open(os.path.join(ROOT, "artifacts/eng4/f1/g4", sb, "outcomes.json")))
    oe = json.load(open(os.path.join(ROOT, "artifacts/eng4/f1/g4", se, "outcomes.json")))
    intended = json.load(open(os.path.join(ROOT, "artifacts/eng4/f1/INTENDED_CHANGES_F1.json")))["suite_ids"]
    listed = set(intended["carried_from_ENG4"]["test_ids"]) | {t for c in intended["new_in_F1"] for t in c["test_ids"]}
    b, e = ob["outcomes"], oe["outcomes"]
    diffs = []
    for tid in sorted(set(b) | set(e)):
        if b.get(tid) != e.get(tid):
            diffs.append({"id": tid, "B": b.get(tid), "E": e.get(tid), "intended": tid in listed})
    coll = {"B": sorted(ob["collection_errors"]), "E": sorted(oe["collection_errors"])}
    unexpected = [d for d in diffs if not d["intended"]]
    ok = ob.get("complete", True) and oe.get("complete", True) and not unexpected and coll["B"] == coll["E"]
    rec = {"B": sb, "E": se, "B_counts": ob["counts"], "E_counts": oe["counts"], "ids_B": len(b), "ids_E": len(e),
           "complete": {"B": ob.get("complete"), "E": oe.get("complete")}, "collection_errors": coll,
           "differences": diffs, "unexpected_differences": unexpected,
           "intended_ids_status": {t: {"B": b.get(t), "E": e.get(t)} for t in sorted(listed)},
           "worktree_writes_by_tests": {"B": ob.get("worktree_writes_by_tests"),
                                        "E": oe.get("worktree_writes_by_tests")},
           "verdict": "PASS" if ok else "FAIL"}
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print("G4", sb, "vs", se, rec["verdict"], "ids", len(b), len(e), "differences", len(diffs), "unexpected",
          len(unexpected))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
