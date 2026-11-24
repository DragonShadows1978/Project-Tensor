"""BWD-F4 G1 / L60: run tests/test_bwdf4_hardening.py's ROWS on Build F3 and on Build F4 (each its own set of child
processes, ENG3's LD_PRELOAD counter, CUDA_VISIBLE_DEVICES empty) and write artifacts/bwdf1/f4/BOUNDARY_F4.json with
each build's result and verdict per row.

Registered rule (the order's G1): a row of class "H1" or "H2" FAILS on Build F3 and PASSES on Build F4, with 0 CUDA
calls on every refusal; every other row (the "(UB only)" case, the controls, H3) PASSES on both builds and its
observable result (raised type, message, result, call counts) is identical on both. The mean VALUE is not observable
on a CPU (the engine has no CPU reduction): the control rows are the preflight form (the call reaches the same device
step) -- Deviation 1 in the report.

  python3 -B -P scripts/bwdf4_boundary.py <F3 pkg parent> <F4 pkg parent> <out json>

Prior art: scripts/bwdf3_boundary.py and BWD-F2's row runner (this repo), reused. No other prior art known to me.
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tensor_cuda", "tests"))
import test_bwdf2_fixes as T2  # noqa: E402
import test_bwdf4_hardening as T4  # noqa: E402


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


f3, f4, out = sys.argv[1:4]
runs = {}
for name, parent in (("F3", f3), ("F4", f4)):
    tmp = f"/tmp/bwdf4_boundary_{name}"
    os.makedirs(tmp, exist_ok=True)
    res = T2.run_rows(parent, tmp, T4.ROWS)
    so = os.path.join(parent, "tensor_cuda", "_tensor_cuda.cpython-312-x86_64-linux-gnu.so")
    runs[name] = {"pkg_parent": parent, "module_sha256": sha(so), "control_calls": res["control_calls"],
                  "tensor_cuda_file": res["tensor_cuda_file"], "child_exits": res["child_exits"], "results": res["results"]}
    assert res["control_calls"] > 0 and res["child_exit"] == 0, name

rows, ok = [], True
for row in T4.ROWS:
    rid, cls = row[0], row[1]
    r3, r4 = runs["F3"]["results"].get(rid), runs["F4"]["results"].get(rid)
    v3, v4 = T2.verdict(row, r3), T2.verdict(row, r4)
    same = (r3 is not None and r4 is not None and
            all(r3.get(k) == r4.get(k) for k in ("raised", "message", "result", "calls", "by_function")))
    if cls in ("H1", "H2"):
        good = v3 is not None and v4 is None and (r4 or {}).get("calls") == 0
        want = "fails on F3, passes on F4 (0 CUDA calls on F4)"
    else:
        good = v3 is None and v4 is None and same
        want = "passes on both, identical results"
    ok = ok and good
    rows.append({"id": rid, "class": cls, "kind": row[2], "setup": row[3], "call": row[4], "want": row[5],
                 "f3_expected": row[7], "rule": want, "F3_verdict": v3, "F4_verdict": v4, "identical_on_F3_and_F4": same,
                 "F3": r3, "F4": r4, "as_registered": good})
doc = {"order": "BWD-F4 G1", "all_rows_as_registered": ok, "rows_total": len(rows),
       "rows_as_registered": sum(r["as_registered"] for r in rows), "builds": {k: {x: v for x, v in d.items() if x != "results"}
                                                                                for k, d in runs.items()},
       "rows": rows}
json.dump(doc, open(out, "w"), indent=1)
for r in rows:
    print(("OK  " if r["as_registered"] else "FAIL"), r["id"], "| F3:", r["F3_verdict"] and r["F3_verdict"][:90], "| F4:",
          r["F4_verdict"] and r["F4_verdict"][:90])
print("ALL AS REGISTERED" if ok else "SOME ROW NOT AS REGISTERED", doc["rows_as_registered"], "/", len(rows))
