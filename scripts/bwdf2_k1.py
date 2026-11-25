"""BWD-F2 K1: run tensor_cuda/tests/test_bwdf2_fixes.py's rows on Build E and on Build F2 (each in its own child, with
CUDA_VISIBLE_DEVICES empty and ENG3's counter) and record, per row, each build's result and whether the row's
expectation holds. K1's bar: every row holds on F2; every row whose `e` names a defect fails on E.

  python3 -B -P scripts/bwdf2_k1.py [arm=pkg_parent ...]   -> artifacts/bwdf1/f2/K1_RESULTS.json
Default arms: E (tensor_cuda/build-bwdf1/E/pkg) and F2 (tensor_cuda/build-bwdf1/F2/pkg); each package's module sha256
is checked against its build receipt first (L103). Prior art: none beyond the test table itself.
"""
import hashlib
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tensor_cuda", "tests"))
os.environ.setdefault("BWDF2_PKG_PARENT", os.path.join(ROOT, "tensor_cuda", "build-bwdf1", "F2", "pkg"))
import test_bwdf2_fixes as T  # noqa: E402

SO = "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
RECEIPTS = {"E": "artifacts/bwdf1/build/BUILD_tc_E.json", "F2": "artifacts/bwdf1/f2/build/BUILD_tc_F2.json"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    arms = dict(a.split("=", 1) for a in sys.argv[1:]) or {
        "E": os.path.join(ROOT, "tensor_cuda/build-bwdf1/E/pkg"), "F2": os.path.join(ROOT, "tensor_cuda/build-bwdf1/F2/pkg")}
    out = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "arms": {}, "rows": []}
    res = {}
    for arm, parent in arms.items():
        got = sha(os.path.join(parent, "tensor_cuda", SO))
        rec = RECEIPTS.get(arm) or f"artifacts/bwdf1/f2/build/BUILD_tc_{arm}.json"
        want = json.load(open(os.path.join(ROOT, rec)))["module_sha256"]
        if got != want:
            raise SystemExit(f"{arm}: module sha256 {got} != {rec} {want}")
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp/bwdf2_tmp")) as d:
            res[arm] = T.run_rows(parent, d)
        out["arms"][arm] = {"pkg_parent": os.path.relpath(parent, ROOT), "module_sha256": got,
                            "child_exit": res[arm].get("child_exit"), "control_calls": res[arm].get("control_calls"),
                            "stderr": res[arm].get("stderr")}
    for row in T.ROWS:
        rid, finding, kind, setup, call, want, flags, e_note = row
        r = {"id": rid, "finding": finding, "kind": kind, "setup": setup, "call": call, "expect": want,
             "flags": flags, "build_E_expected": e_note}
        for arm in arms:
            rr = res[arm].get("results", {}).get(rid)
            r[arm] = {"result": rr, "holds": T.verdict(row, rr) is None, "why": T.verdict(row, rr)}
        r["E_defect_expected"] = e_note != "same" and not e_note.startswith("same")
        out["rows"].append(r)
    rows = out["rows"]
    f2_ok = all(r["F2"]["holds"] for r in rows) if "F2" in arms else None
    e_bad = [r["id"] for r in rows if "E" in arms and r["E_defect_expected"] and r["E"]["holds"]]
    e_same_bad = [r["id"] for r in rows if "E" in arms and not r["E_defect_expected"] and not r["E"]["holds"]]
    out["summary"] = {"rows": len(rows), "F2_all_hold": f2_ok,
                      "F2_failing": [r["id"] for r in rows if "F2" in arms and not r["F2"]["holds"]],
                      "E_defect_rows": sum(r["E_defect_expected"] for r in rows),
                      "E_defect_rows_that_hold_on_E (should be none)": e_bad,
                      "E_same_rows_that_fail_on_E (controls; should be none)": e_same_bad}
    path = os.path.join(ROOT, "artifacts/bwdf1/f2/K1_RESULTS.json")
    if len(arms) == 2 and set(arms) == {"E", "F2"}:
        json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
