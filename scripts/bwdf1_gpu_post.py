"""BWD-F1 post-device steps for the staged GPU lines (CPU, NumPy only, no engine import).

  python3 -B -P scripts/bwdf1_gpu_post.py line <line>   # memcheck summary + verdict -> artifacts/bwdf1/gpu/<line>.post.json
  python3 -B -P scripts/bwdf1_gpu_post.py controls      # G2 controls, Build B vs Build E bitwise (L09/L19)
  python3 -B -P scripts/bwdf1_gpu_post.py g4            # G4 GPU per-ID outcomes, B vs E, against INTENDED_CHANGES
  python3 -B -P scripts/bwdf1_gpu_post.py selftest      # L48: every verdict path driven on constructed results

Bars (from the order): G2 Build E: memcheck 0 errors and every zero case PASS (NumPy shape, zero gradients); Build B:
recorded. G3 Build E: every row PASS; Build B: recorded; m03: at least one row fails (killed). C3 Build E: every case
refused on the host and memcheck 0 errors; Build B: recorded. Prior art: none beyond ENG4's post steps (this repo).
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bwdf1_gpu_common as G  # noqa: E402

GPU = ROOT / "artifacts/bwdf1/gpu"


def line_verdict(line, results, memcheck_text):
    n = None if memcheck_text is None else G.memcheck_summary(memcheck_text)
    arm = line.rsplit("_", 1)[-1] if not line.startswith("g3_m03") else "m03_reduce_gt1"
    out = {"line": line, "memcheck_errors": n}
    if results is None:
        out["verdict"] = "NO-RESULTS (the step did not finish; see the run log)"
        return out
    if line.startswith("g2_direct"):
        out["zero_case_verdict"] = results["zero_case_verdict"]
        ok = results["zero_case_verdict"] == "PASS" and n == 0
    elif line.startswith("g2_f01") or line.startswith("g2_f03"):
        out["witness_verdict"] = results.get("verdict")
        ok = n == 0 and not str(results.get("verdict", "")).startswith(("DEFECT", "ERROR", "CONTROL"))
    elif line.startswith("copy_"):
        out["all_refused_on_host"] = results["all_refused_on_host"]
        ok = results["all_refused_on_host"] and n == 0
    elif line.startswith("g3_"):
        out["g3_verdict"] = results["verdict"]
        out["failing_rows"] = [r["id"] for r in results["rows"] if r["verdict"] != "PASS"]
        if line == "g3_m03_reduce_gt1":
            out["verdict"] = "KILLED" if out["failing_rows"] else "SURVIVED"
            return out
        ok = results["verdict"] == "PASS"
    else:
        raise ValueError(line)
    out["verdict"] = ("PASS" if ok else "FAIL") if arm == "E" else "RECORDED (Build B: no bar)"
    return out


def post_line(line):
    res_p = {"g2_f01": "g2_f01_{a}", "g2_f03": "g2_f03_{a}"}
    d = GPU / line
    results = json.loads((d / "results.json").read_text()) if (d / "results.json").exists() else None
    log = GPU / f"{line}.memcheck.log"
    out = line_verdict(line, results, log.read_text(errors="replace") if log.exists() else None)
    (GPU / f"{line}.post.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


def compare_controls(b_npz, e_npz):
    rows = {}
    for k in sorted(set(b_npz) | set(e_npz)):
        rows[k] = G.bitwise_pair(b_npz.get(k), e_npz.get(k))
    return rows


def post_controls():
    b = dict(np.load(GPU / "g2_direct_B/controls.npz"))
    e = dict(np.load(GPU / "g2_direct_E/controls.npz"))
    rows = compare_controls(b, e)
    verdict = "PASS" if rows and all(v in ("PASS",) or v.startswith("VACUOUS") and "|out" not in k
                                     for k, v in rows.items()) else "FAIL"
    (GPU / "G2_CONTROLS_COMPARE.json").write_text(json.dumps({"rows": rows, "verdict": verdict}, indent=1))
    print("G2 controls", verdict, sum(v == "PASS" for v in rows.values()), "/", len(rows))


def compare_g4(b, e, intended):
    ids = sorted(set(b) | set(e))
    diff = {i: [b.get(i), e.get(i)] for i in ids if b.get(i) != e.get(i)}
    unintended = {i: v for i, v in diff.items() if i not in intended}
    return {"n_ids_B": len(b), "n_ids_E": len(e), "differences": diff, "unintended": unintended,
            "verdict": "PASS" if not unintended else "FAIL"}


def post_g4(series_b="gpu_B", series_e="gpu_E", out="G4_GPU_COMPARE.json"):
    load = lambda s: json.loads((ROOT / f"artifacts/bwdf1/g4/{s}/outcomes.json").read_text())
    ob, oe = load(series_b), load(series_e)
    intended = {c["test_id"] for c in json.loads((ROOT / "artifacts/bwdf1/INTENDED_CHANGES.json").read_text())
                ["existing_test_changes"]}
    res = compare_g4(ob["outcomes"], oe["outcomes"], intended)
    res.update({"complete_B": ob["complete"], "complete_E": oe["complete"], "counts_B": ob["counts"],
                "counts_E": oe["counts"], "collection_errors_B": sorted(ob["collection_errors"]),
                "collection_errors_E": sorted(oe["collection_errors"])})
    if not (ob["complete"] and oe["complete"]):
        res["verdict"] = "INCOMPLETE"
    (ROOT / "artifacts/bwdf1" / out).write_text(json.dumps(res, indent=1))
    print(out, res["verdict"], "differences", len(res["differences"]), "unintended", len(res["unintended"]))


def selftest():
    """L48: drive every post-device path on constructed results of each dtype/shape class, including empty and
    one-element arrays; assert each verdict."""
    ok = []
    ex = lambda cond, what: ok.append((what, bool(cond)))
    # memcheck summary parse
    ex(G.memcheck_summary("========= ERROR SUMMARY: 0 errors\n") == 0, "memcheck 0")
    ex(G.memcheck_summary("x\n========= ERROR SUMMARY: 14 errors\n") == 14, "memcheck 14")
    ex(G.memcheck_summary("========= ERROR SUMMARY: 1 error\n") == 1, "memcheck 1 error")
    ex(G.memcheck_summary("no summary") is None, "memcheck absent")
    # case_verdict on constructed rows (readbacks are float32 after the exact widening of each dtype)
    for dt in G.DTYPES:
        z = np.zeros((4, 1), np.float32)
        good = {"result": "OK", "out_shape": [4, 0], "want_shape": [4, 0],
                "grads": {"a": {"shape": [4, 0], "want": [4, 0], "nonzero_bits": 0},
                          "b": {"shape": [4, 1], "want": [4, 1],
                                "nonzero_bits": int(np.count_nonzero(z.view(np.uint32)))}}}
        ex(G.case_verdict(good) == "PASS", f"case pass {dt}")
        neg = np.array([[-0.0], [0.0], [0.0], [0.0]], np.float32)
        bad = dict(good, grads={"b": {"shape": [4, 1], "want": [4, 1],
                                      "nonzero_bits": int(np.count_nonzero(neg.view(np.uint32)))}})
        ex(G.case_verdict(bad).startswith("FAIL"), f"case -0.0 fails {dt}")
        ex(G.case_verdict(dict(good, out_shape=[4, 1])).startswith("FAIL"), f"case max-rule shape fails {dt}")
        ex(G.case_verdict(dict(good, grads={"b": None})).startswith("FAIL"), f"case undefined grad fails {dt}")
        ex(G.case_verdict({"result": "ERROR", "error": "RuntimeError: x"}).startswith("FAIL"), f"case error {dt}")
    one = {"result": "OK", "out_shape": [1], "want_shape": [1], "grads": {}}
    ex(G.case_verdict(one) == "PASS", "case one-element forward-only")
    # bitwise controls: equal, differing, vacuous, empty, one-element, shape mismatch
    a = np.arange(1, 13, dtype=np.float32).reshape(4, 3)
    ex(G.bitwise_pair(a, a.copy()) == "PASS", "bitwise equal")
    b2 = a.copy(); b2[0, 0] = np.nextafter(b2[0, 0], 2)
    ex(G.bitwise_pair(a, b2).startswith("FAIL"), "bitwise one ulp")
    ex(G.bitwise_pair(np.zeros((4, 3), np.float32), np.zeros((4, 3), np.float32)).startswith("VACUOUS"), "zeros")
    ex(G.bitwise_pair(np.ones((4, 3), np.float32), np.ones((4, 3), np.float32)).startswith("VACUOUS"), "constant")
    ex(G.bitwise_pair(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)).startswith("VACUOUS"), "empty")
    ex(G.bitwise_pair(np.ones((1,), np.float32), np.ones((1,), np.float32)).startswith("VACUOUS"), "one element")
    ex(G.bitwise_pair(a, a.reshape(3, 4)).startswith("FAIL"), "shape mismatch")
    ex(G.bitwise_pair(a, None).startswith("FAIL"), "missing side")
    # line verdicts
    ex(line_verdict("g2_direct_E", {"zero_case_verdict": "PASS"}, "ERROR SUMMARY: 0 errors")["verdict"] == "PASS",
       "g2 E pass")
    ex(line_verdict("g2_direct_E", {"zero_case_verdict": "PASS"}, "ERROR SUMMARY: 3 errors")["verdict"] == "FAIL",
       "g2 E memcheck fail")
    ex(line_verdict("g2_direct_B", {"zero_case_verdict": "FAIL"}, "ERROR SUMMARY: 14 errors")["verdict"].startswith(
        "RECORDED"), "g2 B recorded")
    ex(line_verdict("g2_f01_E", {"verdict": "NOT-DETECTED: x"}, "ERROR SUMMARY: 0 errors")["verdict"] == "PASS",
       "f01 E pass")
    ex(line_verdict("g2_f03_E", {"verdict": "DEFECT-CONFIRMED"}, "ERROR SUMMARY: 0 errors")["verdict"] == "FAIL",
       "f03 E defect")
    ex(line_verdict("copy_E", {"all_refused_on_host": True}, "ERROR SUMMARY: 0 errors")["verdict"] == "PASS", "copy")
    ex(line_verdict("copy_E", {"all_refused_on_host": False}, "ERROR SUMMARY: 0 errors")["verdict"] == "FAIL",
       "copy fail")
    g3 = {"verdict": "FAIL", "rows": [{"id": "x", "verdict": "FAIL: y"}]}
    ex(line_verdict("g3_m03_reduce_gt1", g3, None)["verdict"] == "KILLED", "m03 killed")
    ex(line_verdict("g3_m03_reduce_gt1", {"verdict": "PASS", "rows": [{"id": "x", "verdict": "PASS"}]}, None)[
        "verdict"] == "SURVIVED", "m03 survived")
    ex(line_verdict("g3_E", {"verdict": "PASS", "rows": []}, None)["verdict"] == "PASS", "g3 E")
    ex(line_verdict("g2_direct_E", None, None)["verdict"].startswith("NO-RESULTS"), "no results")
    # g4 compare
    ex(compare_g4({"t1": "passed"}, {"t1": "passed"}, set())["verdict"] == "PASS", "g4 same")
    ex(compare_g4({"t1": "passed"}, {"t1": "failed"}, set())["verdict"] == "FAIL", "g4 diff")
    ex(compare_g4({"t1": "passed"}, {"t1": "failed"}, {"t1"})["verdict"] == "PASS", "g4 intended")
    ex(compare_g4({}, {}, set())["verdict"] == "PASS", "g4 empty")
    bad = [w for w, v in ok if not v]
    print(json.dumps({"checks": len(ok), "failed": bad}))
    print("POST-SELFTEST", "PASS" if not bad else "FAIL")
    return 0 if not bad else 1


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "line":
        post_line(sys.argv[2])
    elif m == "controls":
        post_controls()
    elif m == "g4":
        post_g4(*sys.argv[2:])
    elif m == "selftest":
        sys.exit(selftest())
