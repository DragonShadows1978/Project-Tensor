"""ENG5-S G3 compare (copied from the eng5 branch's scripts/eng5_f2_compare.py): per test ID, Build B's outcomes
against Build S's (L09: outcomes per ID; logs and timings are never compared). Identical outcomes are required: this
round lists no intended change (L38: no existing test changes).

Usage: python3 -B -P scripts/eng5s_compare.py <outcomes.json B> <outcomes.json S> <out.json>

Prior art: no prior art known to me beyond a keyed diff of two outcome maps (ENG3/ENG5's compare scripts, this repo).
"""
import json
import sys


def main():
    pb, pf, out = sys.argv[1], sys.argv[2], sys.argv[3]
    ob, of = json.load(open(pb)), json.load(open(pf))
    b, f = ob["outcomes"], of["outcomes"]
    diffs = [{"id": t, "B": b.get(t), "S": f.get(t)} for t in sorted(set(b) | set(f)) if b.get(t) != f.get(t)]
    coll = {"B": sorted(ob["collection_errors"]), "S": sorted(of["collection_errors"])}
    cb, cf = {}, {}
    for v in b.values():
        cb[v] = cb.get(v, 0) + 1
    for v in f.values():
        cf[v] = cf.get(v, 0) + 1
    ok = (ob.get("complete", True) and of.get("complete", True) and not diffs and coll["B"] == coll["S"]
          and len(b) > 0 and len(b) == len(f))
    res = {"B": pb, "S": pf, "ids_B": len(b), "ids_S": len(f), "counts_B": cb, "counts_S": cf,
           "differences": diffs, "collection_errors": coll,
           "worktree_writes_by_tests": {"B": ob.get("worktree_writes_by_tests"), "S": of.get("worktree_writes_by_tests")},
           "pass": ok}
    json.dump(res, open(out, "w"), indent=1)
    print("ids", len(b), len(f), "differences", len(diffs), "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
