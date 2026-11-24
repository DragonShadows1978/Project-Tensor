"""BWD-F2 post-device steps for the staged GPU lines (CPU, NumPy only, no engine import).

  python3 -B -P scripts/bwdf2_gpu_post.py line <line>   # verdict -> artifacts/bwdf1/f2/gpu/<line>.post.json
  python3 -B -P scripts/bwdf2_gpu_post.py controls      # accepted results, Build B vs Build F2 bitwise (L09/L10/L19)
  python3 -B -P scripts/bwdf2_gpu_post.py g4            # K3 GPU per-ID outcomes, B vs F2, against the listed changes
  python3 -B -P scripts/bwdf2_gpu_post.py selftest      # L48: every verdict path driven on constructed results

Bars (from the order and MUTANTS_F2.json):
- R11 lines (BWD-F1's): Build F2 takes Build E's old bar (g2: memcheck 0 and every zero case PASS; f01/f03: memcheck 0
  and no defect; copy: every case refused on the host, memcheck 0; g3: every row PASS); Build B is recorded; m03 is
  KILLED when a g3 row fails.
- K2: Build F2 must pass each mode's bar (below) with memcheck 0 (count: no memcheck); Builds B and E are recorded,
  with "defect shown" when the review's case misbehaves; n03 / n04 / n05 are KILLED when their rows fail.
    f01: control returned; the three hazard rows refused with "position past table end" (the swap row naming
         "sin [1, 2]").
    f04: every row returned and buf equals NumPy's ring write bitwise (non-vacuous: buf not constant).
    f07: every row equals np.where bitwise.
    vals: axpy_'s named case refused ("does not map element for element"); its accepted rows equal param + other;
          the copy rows equal Build B's results (shape, dtype, device, values); R1's rows return None / (0, 0, 2);
          rank 9 refused.
    count: every f06 / f05 / f09 row refused with 0 allocations, launches and copies during the call; every
           control row launched (> 0): the instrument sees work.
Prior art: BWD-F1's and ENG4's post steps (this repo). Ours: the K2 bars.
"""
import json
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bwdf2_gpu_common as G  # noqa: E402

GPU = ROOT / "artifacts/bwdf1/f2/gpu"
WORK = ("cudaMalloc", "cudaMallocAsync", "cudaLaunchKernel", "cudaMemcpy", "cudaMemset", "cudaMemsetAsync")
MUTANT_ARMS = ("n03_sin_span_unchecked", "n04_ring_no_modulo", "n05_condition_as_data_dtype", "m03_reduce_gt1")


def memcheck_total(text):
    """Sum of every 'ERROR SUMMARY: N error(s)' line (one per traced process); None when absent."""
    if text is None:
        return None
    ns = [int(m) for m in re.findall(r"ERROR SUMMARY:\s+(\d+)\s+error", text)]
    return sum(ns) if ns else None


def bits_equal(a, b):
    """Same shape and float32 bits; both sides non-empty and not constant (L19), else a reason."""
    if a is None or b is None:
        return "missing side"
    a, b = np.ascontiguousarray(a, np.float32), np.ascontiguousarray(b, np.float32)
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    ua, ub = a.view(np.uint32), b.view(np.uint32)
    if ua.size == 0 or np.all(ua == ua.reshape(-1)[0]):
        return "VACUOUS: empty or constant"
    return None if np.array_equal(ua, ub) else f"{int(np.count_nonzero(ua != ub))} elements differ"


def rows_by_id(results):
    return {r["id"]: r for r in results.get("rows", [])}


def k2_verdict(mode, arm, results, arrays, memcheck):
    """Returns (ok, notes) for one K2 line on its results/arrays; `ok` is the F2 bar (and the mutant kill test)."""
    R = rows_by_id(results)
    A = arrays
    notes = []
    bad = []
    if mode == "f01":
        if R.get("f01-control-covering-tables", {}).get("result") != "RETURNED":
            bad.append("control did not return")
        for rid, want in (("f01-swap-named-case", ["position past table end", "sin [1, 2]"]),
                          ("f01-rope_apply-short-sin", ["position past table end"]),
                          ("f01-rope_apply-negative-pos0", ["position past table end"])):
            r = R.get(rid, {})
            if r.get("result") != "RAISED" or not all(w in r.get("error", "") for w in want):
                bad.append(f"{rid} not refused ({r.get('result')}: {r.get('error', '')[:120]})")
    elif mode == "f04":
        for rid in ("f04-control-start-1", "f04-start-cap-plus-2", "f04-start-int64max", "f04-start-int64max-batched"):
            r = R.get(rid, {})
            if r.get("result") != "RETURNED":
                bad.append(f"{rid}: {r.get('result')} {r.get('error', '')[:120]}")
                continue
            why = bits_equal(A.get(f"{rid}|buf"), A.get(f"{rid}|want"))
            if why:
                bad.append(f"{rid}: buf vs NumPy ring write: {why}")
    elif mode == "f07":
        for rid in ("f07-control-float-cond", "f07-where-bool-cond", "f07-masked_fill-bool-mask",
                    "f07-where-bool-cond-broadcast-f16"):
            r = R.get(rid, {})
            if r.get("result") != "RETURNED":
                bad.append(f"{rid}: {r.get('result')} {r.get('error', '')[:120]}")
                continue
            why = bits_equal(A.get(f"{rid}|out"), A.get(f"{rid}|want"))
            if why:
                bad.append(f"{rid}: out vs np.where: {why}")
    elif mode == "vals":
        r = R.get("f12-axpy-named-case", {})
        if r.get("result") != "RAISED" or "does not map element for element" not in r.get("error", ""):
            bad.append(f"f12-axpy-named-case not refused ({r.get('result')})")
            p, b0, o0 = (A.get("f12-axpy-named-case|" + k) for k in ("param", "before", "other"))
            if p is not None and o0 is not None:
                notes.append("named case installed param " + json.dumps(p.reshape(-1).tolist()) + " from before " +
                             json.dumps(b0.reshape(-1).tolist()) + " and other " + json.dumps(o0.tolist()))
        for rid in ("f12-axpy-leading-one-other", "f12-axpy-equal-control"):
            why = bits_equal(A.get(f"{rid}|param"), A.get(f"{rid}|want"))
            if R.get(rid, {}).get("result") != "RETURNED" or why:
                bad.append(f"{rid}: {R.get(rid, {}).get('result')} {why}")
        for rid, key in (("f03-cat-named-case", "out"), ("r9-zero-size-cat-host-later", "out")):
            why = bits_equal(A.get(f"{rid}|{key}"), A.get(f"{rid}|want"))
            if R.get(rid, {}).get("result") != "RETURNED" or why:
                bad.append(f"{rid}: {R.get(rid, {}).get('result')} {R.get(rid, {}).get('error', '')[:120]} {why}")
        for rid, shape in (("r9-zero-size-cat-mixed-dtype", [0, 3]), ("r9-zero-size-stack-mixed-dtype", [2, 0, 3]),
                           ("r9-zero-size-stack-host-later", [2, 0, 3])):
            r = R.get(rid, {})
            sh, meta = A.get(f"{rid}|shape"), A.get(f"{rid}|meta")
            if r.get("result") != "RETURNED" or sh is None or sh.astype(int).tolist() != shape or \
                    meta is None or meta.tolist() != [1.0, 1.0]:
                bad.append(f"{rid}: {r.get('result')} {r.get('error', '')[:120]} shape "
                           f"{None if sh is None else sh.tolist()} meta {None if meta is None else meta.tolist()}")
        if R.get("r1-write_rows-empty-host-src", {}).get("result") != "RETURNED" or \
                A.get("r1-write_rows-empty-host-src|returned_none", np.zeros(1)).tolist() != [1.0]:
            bad.append("r1-write_rows-empty-host-src did not return None")
        if R.get("r1-export_rope_rows-empty-host-cos", {}).get("result") != "RETURNED" or \
                A.get("r1-export_rope_rows-empty-host-cos|shape", np.zeros(1)).tolist() != [0.0, 0.0, 2.0]:
            bad.append("r1-export_rope_rows-empty-host-cos did not return (0, 0, 2)")
        r = R.get("f02-rank9-cuda-with-elements", {})
        if r.get("result") != "RAISED" or "rank 9 exceeds" not in r.get("error", ""):
            bad.append(f"f02-rank9-cuda-with-elements not refused ({r.get('result')})")
    elif mode == "count":
        if results.get("child_exit") not in (0, None):
            bad.append(f"count child exit {results.get('child_exit')}")
        for rid, r in R.items():
            work = {k: v for k, v in r.get("by_function", {}).items() if k in WORK}
            if rid.startswith("control-"):
                if r.get("result") != "RETURNED" or not work.get("cudaLaunchKernel"):
                    bad.append(f"{rid}: control {r.get('result')} launches {work.get('cudaLaunchKernel')}")
            else:
                if r.get("result") != "RAISED" or work:
                    bad.append(f"{rid}: {r.get('result')} work during the call {work}")
        if not R:
            bad.append("no rows")
    else:
        raise ValueError(mode)
    if mode != "count" and memcheck != 0:
        bad.append(f"memcheck errors {memcheck}")
    return not bad, bad + notes


def line_verdict(line, results, memcheck_text, arrays=None):
    n = memcheck_total(memcheck_text)
    out = {"line": line, "memcheck_errors": n}
    if results is None:
        out["verdict"] = "NO-RESULTS (the step did not finish; see the run log)"
        return out
    if line.startswith("k2_"):
        _, mode, arm = line.split("_", 2)
        ok, notes = k2_verdict(mode, arm, results, arrays or {}, n)
        out["notes"] = notes
        if arm == "F2":
            out["verdict"] = "PASS" if ok else "FAIL"
        elif arm in MUTANT_ARMS:
            out["verdict"] = "SURVIVED" if ok else "KILLED"
        else:
            out["verdict"] = "RECORDED (no bar): " + ("defect not shown" if ok else "defect shown")
        return out
    arm = "m03_reduce_gt1" if line.startswith("g3_m03") else line.rsplit("_", 1)[-1]
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
        if arm == "m03_reduce_gt1":
            out["verdict"] = "KILLED" if out["failing_rows"] else "SURVIVED"
            return out
        ok = results["verdict"] == "PASS"
    else:
        raise ValueError(line)
    out["verdict"] = ("PASS" if ok else "FAIL") if arm == "F2" else "RECORDED (Build B: no bar)"
    return out


def post_line(line):
    d = GPU / line
    results = json.loads((d / "results.json").read_text()) if (d / "results.json").exists() else None
    arrays = dict(np.load(d / "arrays.npz")) if (d / "arrays.npz").exists() else {}
    log = GPU / f"{line}.memcheck.log"
    out = line_verdict(line, results, log.read_text(errors="replace") if log.exists() else None, arrays)
    (GPU / f"{line}.post.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


CONTROL_KEYS = {  # accepted results to hold to identity, Build B vs Build F2 (L10)
    "k2_f01": ["f01-control-covering-tables|out"],
    "k2_f04": ["f04-control-start-1|buf"],
    "k2_f07": ["f07-control-float-cond|out"],
    "k2_vals": ["f12-axpy-leading-one-other|param", "f12-axpy-equal-control|param", "f03-cat-named-case|out",
                "r9-zero-size-cat-host-later|out"],
}


def post_controls():
    rows = {}
    b = dict(np.load(GPU / "g2_direct_B/controls.npz"))
    f = dict(np.load(GPU / "g2_direct_F2/controls.npz"))
    for k in sorted(set(b) | set(f)):
        why = bits_equal(b.get(k), f.get(k))
        rows["g2_direct|" + k] = "PASS" if why is None else why
    for line, keys in CONTROL_KEYS.items():
        ab, af = dict(np.load(GPU / f"{line}_B/arrays.npz")), dict(np.load(GPU / f"{line}_F2/arrays.npz"))
        for k in keys:
            why = bits_equal(ab.get(k), af.get(k))
            rows[f"{line}|{k}"] = "PASS" if why is None else why
    verdict = "PASS" if rows and all(v == "PASS" or (v.startswith("VACUOUS") and "|out" not in k and
                                                      k.startswith("g2_direct")) for k, v in rows.items()) else "FAIL"
    (GPU / "CONTROLS_COMPARE_F2.json").write_text(json.dumps({"rows": rows, "verdict": verdict}, indent=1))
    print("controls", verdict, sum(v == "PASS" for v in rows.values()), "/", len(rows))


def compare_g4(b, f, intended):
    ids = sorted(set(b) | set(f))
    diff = {i: [b.get(i), f.get(i)] for i in ids if b.get(i) != f.get(i)}
    unintended = {i: v for i, v in diff.items() if i not in intended}
    return {"n_ids_B": len(b), "n_ids_F2": len(f), "differences": diff, "unintended": unintended,
            "verdict": "PASS" if not unintended else "FAIL"}


def post_g4(series_b="gpu_B", series_f="gpu_F2", out="K3_GPU_COMPARE.json"):
    load = lambda s: json.loads((ROOT / f"artifacts/bwdf1/f2/g4/{s}/outcomes.json").read_text())
    ob, of = load(series_b), load(series_f)
    intended = {c["test_id"] for c in json.loads((ROOT / "artifacts/bwdf1/INTENDED_CHANGES.json").read_text())
                ["existing_test_changes"]}
    intended |= {c["test"] for c in json.loads((ROOT / "artifacts/bwdf1/f2/INTENDED_CHANGES_F2.json").read_text())
                 ["production_tests_changed"]}
    res = compare_g4(ob["outcomes"], of["outcomes"], intended)
    res.update({"complete_B": ob["complete"], "complete_F2": of["complete"], "counts_B": ob["counts"],
                "counts_F2": of["counts"], "collection_errors_B": sorted(ob["collection_errors"]),
                "collection_errors_F2": sorted(of["collection_errors"])})
    if not (ob["complete"] and of["complete"]):
        res["verdict"] = "INCOMPLETE"
    (ROOT / "artifacts/bwdf1/f2" / out).write_text(json.dumps(res, indent=1))
    print(out, res["verdict"], "differences", len(res["differences"]), "unintended", len(res["unintended"]))


def selftest():
    """L48: drive every post-device path on constructed results of each shape class (empty, one-element, rank 1-3)."""
    ok = []
    ex = lambda cond, what: ok.append((what, bool(cond)))
    ex(memcheck_total("========= ERROR SUMMARY: 0 errors\n") == 0, "memcheck 0")
    ex(memcheck_total("a\n========= ERROR SUMMARY: 2 errors\nb\n========= ERROR SUMMARY: 3 errors\n") == 5,
       "memcheck sums processes")
    ex(memcheck_total("========= ERROR SUMMARY: 1 error\n") == 1, "memcheck 1 error")
    ex(memcheck_total("nothing") is None and memcheck_total(None) is None, "memcheck absent")
    a = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    ex(bits_equal(a, a.copy()) is None, "bits equal")
    ex(bits_equal(a, a + 1) is not None, "bits differ")
    ex(bits_equal(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)).startswith("VACUOUS"), "empty vacuous")
    ex(bits_equal(np.ones(1, np.float32), np.ones(1, np.float32)).startswith("VACUOUS"), "one-element vacuous")
    ex(bits_equal(a, a.reshape(3, 2)).startswith("shape"), "shape mismatch")
    ex(bits_equal(None, a) == "missing side", "missing")
    MC0, MC3 = "ERROR SUMMARY: 0 errors", "ERROR SUMMARY: 3 errors"
    # f01
    good01 = {"rows": [{"id": "f01-control-covering-tables", "result": "RETURNED"},
                       {"id": "f01-swap-named-case", "result": "RAISED",
                        "error": "RuntimeError: rope_apply: position past table end (x [2, 2] float32 on cuda:0, cos "
                                 "[2, 2] float32 on cuda:0, sin [1, 2] float32 on cuda:0, pos0 0) [pair 0 of 1]"},
                       {"id": "f01-rope_apply-short-sin", "result": "RAISED",
                        "error": "RuntimeError: rope_apply: position past table end"},
                       {"id": "f01-rope_apply-negative-pos0", "result": "RAISED",
                        "error": "RuntimeError: rope_apply: position past table end"}]}
    ex(line_verdict("k2_f01_F2", good01, MC0)["verdict"] == "PASS", "f01 F2 pass")
    bad01 = {"rows": [dict(r, result="RETURNED") if r["id"] == "f01-swap-named-case" else r for r in good01["rows"]]}
    ex(line_verdict("k2_f01_F2", bad01, MC0)["verdict"] == "FAIL", "f01 F2 fail")
    ex(line_verdict("k2_f01_F2", good01, MC3)["verdict"] == "FAIL", "f01 F2 memcheck fail")
    ex(line_verdict("k2_f01_n03_sin_span_unchecked", bad01, MC3)["verdict"] == "KILLED", "n03 killed")
    ex(line_verdict("k2_f01_n03_sin_span_unchecked", good01, MC0)["verdict"] == "SURVIVED", "n03 survived")
    ex(line_verdict("k2_f01_E", bad01, MC3)["verdict"].endswith("defect shown"), "f01 E recorded")
    # f04 (one-element rows [3, 1] and a rank-3 batched row)
    rid4 = ("f04-control-start-1", "f04-start-cap-plus-2", "f04-start-int64max", "f04-start-int64max-batched")
    arr4 = {}
    for i, rid in enumerate(rid4):
        w = (np.arange(6, dtype=np.float32).reshape(2, 3, 1) + i) if "batched" in rid else \
            np.array([[1.5], [2.5], [3.5 + i]], np.float32)
        arr4[f"{rid}|buf"], arr4[f"{rid}|want"] = w.copy(), w
    res4 = {"rows": [{"id": r, "result": "RETURNED"} for r in rid4]}
    ex(line_verdict("k2_f04_F2", res4, MC0, arr4)["verdict"] == "PASS", "f04 F2 pass")
    arr4b = dict(arr4)
    arr4b["f04-start-int64max|buf"] = arr4["f04-start-int64max|want"][::-1].copy()
    ex(line_verdict("k2_f04_F2", res4, MC0, arr4b)["verdict"] == "FAIL", "f04 F2 wrong rows")
    ex(line_verdict("k2_f04_n04_ring_no_modulo", res4, MC3, arr4)["verdict"] == "KILLED", "n04 killed (memcheck)")
    ex(line_verdict("k2_f04_n04_ring_no_modulo", res4, MC0, arr4)["verdict"] == "SURVIVED", "n04 survived")
    ex(line_verdict("k2_f04_F2", {"rows": [{"id": rid4[0], "result": "RAISED", "error": "x"}]}, MC0,
                    arr4)["verdict"] == "FAIL", "f04 raised")
    # f07
    rid7 = ("f07-control-float-cond", "f07-where-bool-cond", "f07-masked_fill-bool-mask",
            "f07-where-bool-cond-broadcast-f16")
    arr7 = {}
    for i, rid in enumerate(rid7):
        w = np.array([0.25, -1.0, 2.0, 3.0 + i], np.float32)
        arr7[f"{rid}|out"], arr7[f"{rid}|want"] = w.copy(), w
    res7 = {"rows": [{"id": r, "result": "RETURNED"} for r in rid7]}
    ex(line_verdict("k2_f07_F2", res7, MC0, arr7)["verdict"] == "PASS", "f07 F2 pass")
    arr7b = dict(arr7)
    arr7b["f07-where-bool-cond|out"] = np.array([9.0, -1.0, 2.0, 4.0], np.float32)
    ex(line_verdict("k2_f07_n05_condition_as_data_dtype", res7, MC0, arr7b)["verdict"] == "KILLED", "n05 killed")
    ex(line_verdict("k2_f07_F2", res7, MC0, arr7b)["verdict"] == "FAIL", "f07 F2 fail")
    # vals
    rv = [{"id": "f12-axpy-named-case", "result": "RAISED",
           "error": "RuntimeError: axpy_: ... which does not map element for element onto param; refused"},
          {"id": "f12-axpy-leading-one-other", "result": "RETURNED"},
          {"id": "f12-axpy-equal-control", "result": "RETURNED"},
          {"id": "f03-cat-named-case", "result": "RETURNED"},
          {"id": "r9-zero-size-cat-host-later", "result": "RETURNED"},
          {"id": "r9-zero-size-cat-mixed-dtype", "result": "RETURNED"},
          {"id": "r9-zero-size-stack-mixed-dtype", "result": "RETURNED"},
          {"id": "r9-zero-size-stack-host-later", "result": "RETURNED"},
          {"id": "r1-write_rows-empty-host-src", "result": "RETURNED"},
          {"id": "r1-export_rope_rows-empty-host-cos", "result": "RETURNED"},
          {"id": "f02-rank9-cuda-with-elements", "result": "RAISED", "error": "RuntimeError: ew_binary: rank 9 exceeds"}]
    av = {"f12-axpy-leading-one-other|param": np.array([1.0, 2.0, 3.5], np.float32),
          "f12-axpy-leading-one-other|want": np.array([1.0, 2.0, 3.5], np.float32),
          "f12-axpy-equal-control|param": a, "f12-axpy-equal-control|want": a.copy(),
          "f03-cat-named-case|out": a, "f03-cat-named-case|want": a.copy(),
          "r9-zero-size-cat-host-later|out": a, "r9-zero-size-cat-host-later|want": a.copy(),
          "r9-zero-size-cat-mixed-dtype|shape": np.array([0, 3], np.float32),
          "r9-zero-size-cat-mixed-dtype|meta": np.array([1, 1], np.float32),
          "r9-zero-size-stack-mixed-dtype|shape": np.array([2, 0, 3], np.float32),
          "r9-zero-size-stack-mixed-dtype|meta": np.array([1, 1], np.float32),
          "r9-zero-size-stack-host-later|shape": np.array([2, 0, 3], np.float32),
          "r9-zero-size-stack-host-later|meta": np.array([1, 1], np.float32),
          "r1-write_rows-empty-host-src|returned_none": np.array([1], np.float32),
          "r1-export_rope_rows-empty-host-cos|shape": np.array([0, 0, 2], np.float32)}
    ex(line_verdict("k2_vals_F2", {"rows": rv}, MC0, av)["verdict"] == "PASS", "vals F2 pass")
    rv_b = [dict(r, result="RETURNED") if r["id"] == "f12-axpy-named-case" else r for r in rv]
    av_b = dict(av, **{"f12-axpy-named-case|param": np.array([[1.0], [2.0]], np.float32),
                       "f12-axpy-named-case|before": np.array([[0.0], [1.0]], np.float32),
                       "f12-axpy-named-case|other": np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)})
    v = line_verdict("k2_vals_E", {"rows": rv_b}, MC0, av_b)
    ex(v["verdict"].endswith("defect shown") and any("installed param" in n for n in v["notes"]), "vals E notes")
    av_c = dict(av, **{"r9-zero-size-stack-mixed-dtype|shape": np.array([2, 3], np.float32)})
    ex(line_verdict("k2_vals_F2", {"rows": rv}, MC0, av_c)["verdict"] == "FAIL", "vals F2 shape fail")
    # count
    rc = {"child_exit": 0, "rows": [
        {"id": "control-swap-all-valid", "result": "RETURNED", "by_function": {"cudaLaunchKernel": 6,
                                                                                "cudaMallocAsync": 4}},
        {"id": "f06-swap-later-bad", "result": "RAISED", "error": "RuntimeError: x",
         "by_function": {"cudaStreamSynchronize": 1}}]}
    ex(line_verdict("k2_count_F2", rc, None)["verdict"] == "PASS", "count F2 pass")
    rc_b = {"child_exit": 0, "rows": [rc["rows"][0], dict(rc["rows"][1], by_function={"cudaLaunchKernel": 2})]}
    ex(line_verdict("k2_count_F2", rc_b, None)["verdict"] == "FAIL", "count F2 work before refusal")
    ex(line_verdict("k2_count_B", rc_b, None)["verdict"].endswith("defect shown"), "count B recorded")
    rc_c = {"child_exit": 0, "rows": [dict(rc["rows"][0], by_function={}), rc["rows"][1]]}
    ex(line_verdict("k2_count_F2", rc_c, None)["verdict"] == "FAIL", "count control sees no work")
    ex(line_verdict("k2_count_F2", {"child_exit": 1, "rows": []}, None)["verdict"] == "FAIL", "count child died")
    # R11 lines
    ex(line_verdict("g2_direct_F2", {"zero_case_verdict": "PASS"}, MC0)["verdict"] == "PASS", "g2 F2")
    ex(line_verdict("g2_direct_F2", {"zero_case_verdict": "PASS"}, MC3)["verdict"] == "FAIL", "g2 F2 memcheck")
    ex(line_verdict("g2_direct_B", {"zero_case_verdict": "FAIL"}, MC3)["verdict"].startswith("RECORDED"), "g2 B")
    ex(line_verdict("g2_f01_F2", {"verdict": "NOT-DETECTED: x"}, MC0)["verdict"] == "PASS", "f01 audit F2")
    ex(line_verdict("g2_f03_F2", {"verdict": "DEFECT-CONFIRMED"}, MC0)["verdict"] == "FAIL", "f03 audit F2 defect")
    ex(line_verdict("copy_F2", {"all_refused_on_host": True}, MC0)["verdict"] == "PASS", "copy F2")
    ex(line_verdict("copy_F2", {"all_refused_on_host": False}, MC0)["verdict"] == "FAIL", "copy F2 fail")
    g3 = {"verdict": "FAIL", "rows": [{"id": "x", "verdict": "FAIL: y"}]}
    ex(line_verdict("g3_m03_reduce_gt1", g3, None)["verdict"] == "KILLED", "m03 killed")
    ex(line_verdict("g3_F2", {"verdict": "PASS", "rows": [{"id": "x", "verdict": "PASS"}]}, None)["verdict"] == "PASS",
       "g3 F2")
    ex(line_verdict("k2_f04_F2", None, None)["verdict"].startswith("NO-RESULTS"), "no results")
    # g4 compare
    ex(compare_g4({"t": "passed"}, {"t": "passed"}, set())["verdict"] == "PASS", "g4 same")
    ex(compare_g4({"t": "passed"}, {"t": "failed"}, set())["verdict"] == "FAIL", "g4 diff")
    ex(compare_g4({"t": "failed"}, {"t": "passed"}, {"t"})["verdict"] == "PASS", "g4 intended")
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
