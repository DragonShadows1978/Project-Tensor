"""BWD-F4 G3: run tests/test_bwdf4_hardening.py's ROWS on each mutant's frozen package (artifacts/bwdf1/f4/MUTANTS_F4.json
names the mutants and their killing rows, written before any mutant ran) and write artifacts/bwdf1/f4/MUTANT_RESULTS_F4.json.

KILLED = every listed killing row FAILS on the mutant (while it passes on Build F4, from BOUNDARY_F4.json). Also
recorded: every other row's verdict on the mutant (collateral), and whether the mutant's result on each killing row is
identical to Build F3's (the mutant re-creates the defect's observable), and the module sha256s (L103).

  python3 -B -P scripts/bwdf4_mutant_results.py

Prior art: scripts/bwdf3_mutant_results.py (this repo), reduced; mutation testing (DeMillo, Lipton and Sayward, 1978).
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tensor_cuda", "tests"))
import test_bwdf2_fixes as T2  # noqa: E402
import test_bwdf4_hardening as T4  # noqa: E402

SO = "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
spec = json.load(open(os.path.join(ROOT, "artifacts/bwdf1/f4/MUTANTS_F4.json")))
bound = json.load(open(os.path.join(ROOT, "artifacts/bwdf1/f4/BOUNDARY_F4.json")))
f3 = {r["id"]: r["F3"] for r in bound["rows"]}
f4v = {r["id"]: r["F4_verdict"] for r in bound["rows"]}
KEYS = ("raised", "message", "result", "calls", "by_function")
out = {"order": "BWD-F4 G3", "spec": "artifacts/bwdf1/f4/MUTANTS_F4.json", "mutants": {}}
for name, e in spec["mutants"].items():
    parent = os.path.join(ROOT, "tensor_cuda/build-bwdf4", name, "pkg")
    rec = json.load(open(os.path.join(ROOT, f"artifacts/bwdf1/f4/build/BUILD_tc_{name}.json")))
    mod = os.path.join(parent, "tensor_cuda", SO)
    sha = hashlib.sha256(open(mod, "rb").read()).hexdigest()
    assert sha == rec["module_sha256"], name
    tmp = f"/tmp/bwdf4_mutant_{name}"
    os.makedirs(tmp, exist_ok=True)
    res = T2.run_rows(parent, tmp, T4.ROWS)
    assert res["control_calls"] > 0 and res["child_exit"] == 0, name
    kills = [k.split("[", 1)[1][:-1] for k in e["kills"]]
    rows = {}
    for row in T4.ROWS:
        r = res["results"].get(row[0])
        v = T2.verdict(row, r)
        rows[row[0]] = {"verdict_on_mutant": v, "verdict_on_F4": f4v[row[0]], "killing_row": row[0] in kills,
                        "result_equals_F3": all((r or {}).get(k) == (f3[row[0]] or {}).get(k) for k in KEYS), "result": r}
    killed = all(rows[k]["verdict_on_mutant"] is not None and rows[k]["verdict_on_F4"] is None for k in kills)
    collateral = [k for k, v in rows.items() if k not in kills and v["verdict_on_mutant"] is not None]
    out["mutants"][name] = {"ruling": e["ruling"], "edit_file": e["file"], "module_sha256": sha,
                            "tensor_cuda_file": res["tensor_cuda_file"], "killing_rows": kills,
                            "killing_rows_failing_on_mutant": [k for k in kills if rows[k]["verdict_on_mutant"] is not None],
                            "killing_rows_equal_to_F3_result": [k for k in kills if rows[k]["result_equals_F3"]],
                            "KILLED": killed, "collateral_failures": collateral, "rows": rows}
    print(name, "KILLED" if killed else "SURVIVED", "kill rows failing", len(out["mutants"][name]["killing_rows_failing_on_mutant"]),
          "/", len(kills), "| equal to F3 on", len(out["mutants"][name]["killing_rows_equal_to_F3_result"]), "| collateral", collateral)
    for k in kills:
        print("   ", k, "->", (rows[k]["verdict_on_mutant"] or "PASSED")[:150])
json.dump(out, open(os.path.join(ROOT, "artifacts/bwdf1/f4/MUTANT_RESULTS_F4.json"), "w"), indent=1)
sys.exit(0 if all(m["KILLED"] for m in out["mutants"].values()) else 1)
