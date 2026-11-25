"""BWD-F3 P1: run tensor_cuda/tests/test_bwdf3_fixes.py's rows on Build F2, Build F3 and Build B (each row in its own
child, CUDA_VISIBLE_DEVICES empty, ENG3's counter) and record, per row, each build's result and whether the row's
expectation holds.

  python3 -B -P scripts/bwdf3_p1.py            -> artifacts/bwdf1/f3/P1_RESULTS.json

Bars (orders/BWDF3_REV2_FIXES.md P1): every "E1".."E4" row fails on Build F2 and holds on Build F3; every "<E>
control" row holds on both; every "D1" row holds on Build F3 (recorded as a domain refusal, with Build F2's and
Build B's answers). Every refusal row's verdict includes 0 CUDA calls. Each package's module sha256 is checked
against its build receipt first (L103). Build B's answers feed BOUNDARY_F3.json. Prior art: none beyond the table.
"""
import hashlib
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tensor_cuda", "tests"))
os.environ.setdefault("BWDF3_PKG_PARENT", os.path.join(ROOT, "tensor_cuda", "build-bwdf1", "F3", "pkg"))
import test_bwdf3_fixes as T3  # noqa: E402
import test_bwdf2_fixes as T2  # noqa: E402

SO = "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
ARMS = {"F2": ("tensor_cuda/build-bwdf1/F2/pkg", "artifacts/bwdf1/f2/build/BUILD_tc_F2.json"),
        "F3": ("tensor_cuda/build-bwdf1/F3/pkg", "artifacts/bwdf1/f3/build/BUILD_tc_F3.json"),
        "B": ("tensor_cuda/build-bwdf1/B/pkg", "artifacts/bwdf1/build/BUILD_tc_B.json")}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def cls(row):
    c = row[1]
    if c.startswith("D1"):
        return "D1"
    return "control" if "control" in c else "fix"


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise SystemExit("CPU lane only: CUDA_VISIBLE_DEVICES must be empty")
    out = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "arms": {}, "rows": []}
    res = {}
    for arm, (parent, rec) in ARMS.items():
        parent = os.path.join(ROOT, parent)
        got = sha(os.path.join(parent, "tensor_cuda", SO))
        want = json.load(open(os.path.join(ROOT, rec)))["module_sha256"]
        if got != want:
            raise SystemExit(f"{arm}: module sha256 {got} != {rec} {want}")
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp/bwdf3_tmp")) as d:
            res[arm] = T2.run_rows(parent, d, T3.ROWS)
        out["arms"][arm] = {"pkg_parent": os.path.relpath(parent, ROOT), "module_sha256": got,
                            "control_calls": res[arm].get("control_calls"),
                            "child_exits_nonzero": {k: v for k, v in res[arm]["child_exits"].items() if v != 0}}
    for row in T3.ROWS:
        rid, klass, kind, setup, call, want, flags, f2_note = row
        r = {"id": rid, "class": klass, "kind": kind, "setup": setup, "call": call, "expect": want, "flags": flags,
             "build_F2_expected": f2_note}
        for arm in ARMS:
            rr = res[arm]["results"].get(rid)
            why = T2.verdict(row, rr)
            r[arm] = {"result": rr, "holds": why is None, "why": why}
        c = cls(row)
        if c == "fix":
            r["bar"] = "fails on F2, holds on F3"
            r["pass"] = (not r["F2"]["holds"]) and r["F3"]["holds"]
        elif c == "control":
            r["bar"] = "holds on F2 and F3"
            r["pass"] = r["F2"]["holds"] and r["F3"]["holds"]
        else:
            r["bar"] = "domain refusal: holds on F3 (F2 and B recorded)"
            r["pass"] = r["F3"]["holds"]
            r["label"] = "domain refusal"
        out["rows"].append(r)
    rows = out["rows"]
    out["summary"] = {
        "rows": len(rows), "fix_rows": sum(cls(x) == "fix" for x in T3.ROWS),
        "control_rows": sum(cls(x) == "control" for x in T3.ROWS), "d1_rows": sum(cls(x) == "D1" for x in T3.ROWS),
        "failing": [r["id"] for r in rows if not r["pass"]],
        "F3_counter_positive_control_calls": res["F3"].get("control_calls"),
        "verdict": "PASS" if all(r["pass"] for r in rows) and res["F3"].get("control_calls", 0) > 0 else "FAIL"}
    with open(os.path.join(ROOT, "artifacts/bwdf1/f3/P1_RESULTS.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
