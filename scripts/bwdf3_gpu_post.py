"""BWD-F3 post-device steps (a copy of scripts/bwdf2_gpu_post.py; CPU, NumPy only, no engine import). E5: every
restaged line keeps BWD-F2's bar with Build F3 in Build F2's place (outputs under artifacts/bwdf1/f3/gpu); P2: the
p2_<mode>_<arm> lines (p2_verdict below); controls: accepted results Build B vs Build F3 bitwise; g4: P3's GPU half.

  python3 -B -P scripts/bwdf3_gpu_post.py line <line>   # verdict -> artifacts/bwdf1/f3/gpu/<line>.post.json
  python3 -B -P scripts/bwdf3_gpu_post.py controls      # accepted results, Build B vs Build F3 bitwise (L09/L10/L19)
  python3 -B -P scripts/bwdf3_gpu_post.py g4            # P3 GPU per-ID outcomes, B vs F3, against the listed changes
  python3 -B -P scripts/bwdf3_gpu_post.py selftest      # L48: every verdict path driven on constructed results

P2 bars (Build F3; p04 is KILLED when its count line fails the bar; B and F2 are recorded, "defect shown" when the
bar fails): e1: controls return; reading rows refused with "position past table end"; empty CUDA launches return
[0, 2, 2] / [0, 2, 2] / one rope output; e2: widened-table rows return results bitwise equal to the [T, D] call
(non-vacuous); [T, D, 2] refused "tables must be (T, D)"; e3: controls return; repeated axes refused (ValueError
"duplicate value in axis"), out-of-range axes refused (IndexError "is out of bounds"); count: controls launch;
refusal rows refused with no allocation, launch, copy or memset during the call (the E4 rows naming "splice_rows: the
output" and "[pair 1 of 2]"); accept rows return with no such work. Memcheck lines need 0 errors; a child that died
fails the bar. BWD-F2's text follows.

BWD-F2 post-device steps for the staged GPU lines (CPU, NumPy only, no engine import).

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

GPU = ROOT / "artifacts/bwdf1/f3/gpu"
WORK = ("cudaMalloc", "cudaMallocAsync", "cudaLaunchKernel", "cudaMemcpy", "cudaMemset", "cudaMemsetAsync")
MUTANT_ARMS = ("n03_sin_span_unchecked", "n04_ring_no_modulo", "n05_condition_as_data_dtype", "m03_reduce_gt1",
               "p04_splice_bound_after_work")
BAR_ARM = "F3"  # BWD-F3 E5: Build F3 takes Build F2's bar in every restaged line


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



P2_WORK = WORK


def p2_verdict(mode, results, arrays, memcheck):
    """Returns (ok, notes) for one P2 line: `ok` is Build F3's bar (and p04's kill test on the count line). The line
    must hold exactly the mode's row IDs (scripts/bwdf3_p2_cases.py rows_for), each with its registered kind."""
    import bwdf3_p2_cases as C  # the case table (no engine import)
    R = rows_by_id(results)
    A = arrays
    bad = []
    want_ids = dict(C.rows_for(mode))
    if set(R) != set(want_ids):
        bad.append(f"row IDs differ from the case table: missing {sorted(set(want_ids) - set(R))}, "
                   f"extra {sorted(set(R) - set(want_ids))}")
    for rid, r in R.items():
        if rid in want_ids and r.get("kind") != want_ids[rid]:
            bad.append(f"{rid}: kind {r.get('kind')} (registered {want_ids[rid]})")
    for rid, r in R.items():
        if r.get("child_exit") not in (0, None):
            bad.append(f"{rid}: child exit {r.get('child_exit')} ({r.get('result')})")
    if mode == "e1":
        for rid, r in R.items():
            kind, res, err = r.get("kind"), r.get("result"), r.get("error", "")
            if kind == "control" and res != "RETURNED":
                bad.append(f"{rid}: control {res} {err[:120]}")
            elif kind == "refuse" and (res != "RAISED" or not err.startswith("RuntimeError: ")
                                       or "position past table end" not in err):
                bad.append(f"{rid}: not refused on the span ({res}: {err[:120]})")
        want = {"e1-rope_apply-empty-cuda-short-sin": ("shape", [0.0, 2.0, 2.0]),
                "e1-export-empty-cuda-short-sin": ("shape", [0.0, 2.0, 2.0]),
                "e1-swap-empty-cuda-short-sin": ("n", [1.0])}
        for rid, (k, v) in want.items():
            got = A.get(f"{rid}|{k}")
            if R.get(rid, {}).get("result") != "RETURNED" or got is None or got.tolist() != v:
                bad.append(f"{rid}: {R.get(rid, {}).get('result')} {R.get(rid, {}).get('error', '')[:120]} "
                           f"{None if got is None else got.tolist()} (want {v})")
    elif mode == "e2":
        for rid, r in R.items():
            if r.get("kind") == "refuse":
                if r.get("result") != "RAISED" or "tables must be (T, D)" not in r.get("error", ""):
                    bad.append(f"{rid}: not refused ({r.get('result')}: {r.get('error', '')[:120]})")
                continue
            if r.get("result") != "RETURNED":
                bad.append(f"{rid}: {r.get('result')} {r.get('error', '')[:120]}")
                continue
            why = bits_equal(A.get(f"{rid}|out"), A.get(f"{rid}|ref"))
            if why:
                bad.append(f"{rid}: widened table vs [T, D]: {why}")
    elif mode == "e3":
        for rid, r in R.items():
            kind, res, err = r.get("kind"), r.get("result"), r.get("error", "")
            if kind == "control":
                if res != "RETURNED":
                    bad.append(f"{rid}: control {res} {err[:120]}")
            elif not (res == "RAISED" and ((err.startswith("ValueError: ") and "duplicate value in axis" in err) or
                                           (err.startswith("IndexError: ") and "is out of bounds" in err))):
                bad.append(f"{rid}: not refused on the axis ({res}: {err[:120]})")
    elif mode == "count":
        for rid, r in R.items():
            work = {k: v for k, v in r.get("by_function", {}).items() if k in P2_WORK}
            kind, res, err = r.get("kind"), r.get("result"), r.get("error", "")
            if kind == "control":
                if res != "RETURNED" or not work.get("cudaLaunchKernel"):
                    bad.append(f"{rid}: control {res} launches {work.get('cudaLaunchKernel')}")
            elif kind == "accept":
                if res != "RETURNED" or work:
                    bad.append(f"{rid}: {res} {err[:80]} work {work}")
            else:
                if res != "RAISED" or work:
                    bad.append(f"{rid}: {res} work during the call {work} ({err[:120]})")
                if rid.startswith("e4-") and not ("splice_rows: the output" in err and "[pair 1 of 2]" in err):
                    bad.append(f"{rid}: refusal is not the preflight's byte bound ({err[:160]})")
    else:
        raise ValueError(mode)
    if mode != "count" and memcheck != 0:
        bad.append(f"memcheck errors {memcheck}")
    return not bad, bad


def line_verdict(line, results, memcheck_text, arrays=None):
    n = memcheck_total(memcheck_text)
    out = {"line": line, "memcheck_errors": n}
    if results is None:
        out["verdict"] = "NO-RESULTS (the step did not finish; see the run log)"
        return out
    if line.startswith("p2_"):
        _, mode, arm = line.split("_", 2)
        ok, notes = p2_verdict(mode, results, arrays or {}, n)
        out["notes"] = notes
        if arm == BAR_ARM:
            out["verdict"] = "PASS" if ok else "FAIL"
        elif arm in MUTANT_ARMS:
            out["verdict"] = "SURVIVED" if ok else "KILLED"
        else:
            out["verdict"] = "RECORDED (no bar): " + ("defect not shown" if ok else "defect shown")
        return out
    if line.startswith("k2_"):
        _, mode, arm = line.split("_", 2)
        ok, notes = k2_verdict(mode, arm, results, arrays or {}, n)
        out["notes"] = notes
        if arm == BAR_ARM:
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
    out["verdict"] = ("PASS" if ok else "FAIL") if arm == BAR_ARM else "RECORDED (Build B: no bar)"
    return out


def post_line(line):
    d = GPU / line
    results = json.loads((d / "results.json").read_text()) if (d / "results.json").exists() else None
    arrays = dict(np.load(d / "arrays.npz")) if (d / "arrays.npz").exists() else {}
    if line.startswith("p2_"):  # BWD-F3: one npz per row, keys "<row id>|<array>"
        for f in sorted(d.glob("*.npz")):
            arrays.update({f"{f.stem}|{k}": v for k, v in np.load(f).items()})
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


P2_CONTROL_KEYS = {  # BWD-F3: accepted results held to identity, Build B vs Build F3 (L10)
    "p2_e1": ["e1-control-rope_apply-covering|out", "e1-control-export-covering|out"],
    "p2_e2": ["e2-rope_apply-sin-T-D-1|out", "e2-rope_apply-sin-T-D-1|ref", "e2-export-sin-T-D-1|out",
              "e2-export-sin-T-D-1|ref", "e2-swap-tables-T-D-1|ref"],
    "p2_e3": ["e3-control-sum-last|out", "e3-control-sum-two-distinct|out", "e3-control-mean|out",
              "e3-control-argmax|out"],
}


def post_controls():
    rows = {}
    b = dict(np.load(GPU / "g2_direct_B/controls.npz"))
    f = dict(np.load(GPU / "g2_direct_F3/controls.npz"))
    for k in sorted(set(b) | set(f)):
        why = bits_equal(b.get(k), f.get(k))
        rows["g2_direct|" + k] = "PASS" if why is None else why
    for line, keys in CONTROL_KEYS.items():
        ab, af = dict(np.load(GPU / f"{line}_B/arrays.npz")), dict(np.load(GPU / f"{line}_F3/arrays.npz"))
        for k in keys:
            why = bits_equal(ab.get(k), af.get(k))
            rows[f"{line}|{k}"] = "PASS" if why is None else why
    for line, keys in P2_CONTROL_KEYS.items():
        load = lambda arm: {f"{f.stem}|{k}": v for f in sorted((GPU / f"{line}_{arm}").glob("*.npz"))
                            for k, v in np.load(f).items()}
        ab, af = load("B"), load(BAR_ARM)
        for k in keys:
            why = bits_equal(ab.get(k), af.get(k))
            rows[f"{line}|{k}"] = "PASS" if why is None else why
    verdict = "PASS" if rows and all(v == "PASS" or (v.startswith("VACUOUS") and "|out" not in k and
                                                      k.startswith("g2_direct")) for k, v in rows.items()) else "FAIL"
    (GPU / "CONTROLS_COMPARE_F3.json").write_text(json.dumps({"rows": rows, "verdict": verdict}, indent=1))
    print("controls", verdict, sum(v == "PASS" for v in rows.values()), "/", len(rows))


def compare_g4(b, f, intended):
    ids = sorted(set(b) | set(f))
    diff = {i: [b.get(i), f.get(i)] for i in ids if b.get(i) != f.get(i)}
    unintended = {i: v for i, v in diff.items() if i not in intended}
    return {"n_ids_B": len(b), "n_ids_F3": len(f), "differences": diff, "unintended": unintended,
            "verdict": "PASS" if not unintended else "FAIL"}


def post_g4(series_b="gpu_B", series_f="gpu_F3", out="P3_GPU_COMPARE.json"):
    load = lambda s: json.loads((ROOT / f"artifacts/bwdf1/f3/g4/{s}/outcomes.json").read_text())
    ob, of = load(series_b), load(series_f)
    intended = {c["test_id"] for c in json.loads((ROOT / "artifacts/bwdf1/INTENDED_CHANGES.json").read_text())
                ["existing_test_changes"]}
    intended |= {c["test"] for c in json.loads((ROOT / "artifacts/bwdf1/f2/INTENDED_CHANGES_F2.json").read_text())
                 ["production_tests_changed"]}
    intended |= {c["test"] for c in json.loads((ROOT / "artifacts/bwdf1/f3/INTENDED_CHANGES_F3.json").read_text())
                 ["production_tests_changed"]}
    res = compare_g4(ob["outcomes"], of["outcomes"], intended)
    res.update({"complete_B": ob["complete"], "complete_F3": of["complete"], "counts_B": ob["counts"],
                "counts_F3": of["counts"], "collection_errors_B": sorted(ob["collection_errors"]),
                "collection_errors_F3": sorted(of["collection_errors"])})
    if not (ob["complete"] and of["complete"]):
        res["verdict"] = "INCOMPLETE"
    (ROOT / "artifacts/bwdf1/f3" / out).write_text(json.dumps(res, indent=1))
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
    ex(line_verdict("k2_f01_F3", good01, MC0)["verdict"] == "PASS", "f01 F3 pass")
    bad01 = {"rows": [dict(r, result="RETURNED") if r["id"] == "f01-swap-named-case" else r for r in good01["rows"]]}
    ex(line_verdict("k2_f01_F3", bad01, MC0)["verdict"] == "FAIL", "f01 F3 fail")
    ex(line_verdict("k2_f01_F3", good01, MC3)["verdict"] == "FAIL", "f01 F3 memcheck fail")
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
    ex(line_verdict("k2_f04_F3", res4, MC0, arr4)["verdict"] == "PASS", "f04 F3 pass")
    arr4b = dict(arr4)
    arr4b["f04-start-int64max|buf"] = arr4["f04-start-int64max|want"][::-1].copy()
    ex(line_verdict("k2_f04_F3", res4, MC0, arr4b)["verdict"] == "FAIL", "f04 F3 wrong rows")
    ex(line_verdict("k2_f04_n04_ring_no_modulo", res4, MC3, arr4)["verdict"] == "KILLED", "n04 killed (memcheck)")
    ex(line_verdict("k2_f04_n04_ring_no_modulo", res4, MC0, arr4)["verdict"] == "SURVIVED", "n04 survived")
    ex(line_verdict("k2_f04_F3", {"rows": [{"id": rid4[0], "result": "RAISED", "error": "x"}]}, MC0,
                    arr4)["verdict"] == "FAIL", "f04 raised")
    # f07
    rid7 = ("f07-control-float-cond", "f07-where-bool-cond", "f07-masked_fill-bool-mask",
            "f07-where-bool-cond-broadcast-f16")
    arr7 = {}
    for i, rid in enumerate(rid7):
        w = np.array([0.25, -1.0, 2.0, 3.0 + i], np.float32)
        arr7[f"{rid}|out"], arr7[f"{rid}|want"] = w.copy(), w
    res7 = {"rows": [{"id": r, "result": "RETURNED"} for r in rid7]}
    ex(line_verdict("k2_f07_F3", res7, MC0, arr7)["verdict"] == "PASS", "f07 F3 pass")
    arr7b = dict(arr7)
    arr7b["f07-where-bool-cond|out"] = np.array([9.0, -1.0, 2.0, 4.0], np.float32)
    ex(line_verdict("k2_f07_n05_condition_as_data_dtype", res7, MC0, arr7b)["verdict"] == "KILLED", "n05 killed")
    ex(line_verdict("k2_f07_F3", res7, MC0, arr7b)["verdict"] == "FAIL", "f07 F3 fail")
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
    ex(line_verdict("k2_vals_F3", {"rows": rv}, MC0, av)["verdict"] == "PASS", "vals F3 pass")
    rv_b = [dict(r, result="RETURNED") if r["id"] == "f12-axpy-named-case" else r for r in rv]
    av_b = dict(av, **{"f12-axpy-named-case|param": np.array([[1.0], [2.0]], np.float32),
                       "f12-axpy-named-case|before": np.array([[0.0], [1.0]], np.float32),
                       "f12-axpy-named-case|other": np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)})
    v = line_verdict("k2_vals_E", {"rows": rv_b}, MC0, av_b)
    ex(v["verdict"].endswith("defect shown") and any("installed param" in n for n in v["notes"]), "vals E notes")
    av_c = dict(av, **{"r9-zero-size-stack-mixed-dtype|shape": np.array([2, 3], np.float32)})
    ex(line_verdict("k2_vals_F3", {"rows": rv}, MC0, av_c)["verdict"] == "FAIL", "vals F3 shape fail")
    # count
    rc = {"child_exit": 0, "rows": [
        {"id": "control-swap-all-valid", "result": "RETURNED", "by_function": {"cudaLaunchKernel": 6,
                                                                                "cudaMallocAsync": 4}},
        {"id": "f06-swap-later-bad", "result": "RAISED", "error": "RuntimeError: x",
         "by_function": {"cudaStreamSynchronize": 1}}]}
    ex(line_verdict("k2_count_F3", rc, None)["verdict"] == "PASS", "count F3 pass")
    rc_b = {"child_exit": 0, "rows": [rc["rows"][0], dict(rc["rows"][1], by_function={"cudaLaunchKernel": 2})]}
    ex(line_verdict("k2_count_F3", rc_b, None)["verdict"] == "FAIL", "count F3 work before refusal")
    ex(line_verdict("k2_count_B", rc_b, None)["verdict"].endswith("defect shown"), "count B recorded")
    rc_c = {"child_exit": 0, "rows": [dict(rc["rows"][0], by_function={}), rc["rows"][1]]}
    ex(line_verdict("k2_count_F3", rc_c, None)["verdict"] == "FAIL", "count control sees no work")
    ex(line_verdict("k2_count_F3", {"child_exit": 1, "rows": []}, None)["verdict"] == "FAIL", "count child died")
    # R11 lines
    ex(line_verdict("g2_direct_F3", {"zero_case_verdict": "PASS"}, MC0)["verdict"] == "PASS", "g2 F2")
    ex(line_verdict("g2_direct_F3", {"zero_case_verdict": "PASS"}, MC3)["verdict"] == "FAIL", "g2 F3 memcheck")
    ex(line_verdict("g2_direct_B", {"zero_case_verdict": "FAIL"}, MC3)["verdict"].startswith("RECORDED"), "g2 B")
    ex(line_verdict("g2_f01_F3", {"verdict": "NOT-DETECTED: x"}, MC0)["verdict"] == "PASS", "f01 audit F2")
    ex(line_verdict("g2_f03_F3", {"verdict": "DEFECT-CONFIRMED"}, MC0)["verdict"] == "FAIL", "f03 audit F3 defect")
    ex(line_verdict("copy_F3", {"all_refused_on_host": True}, MC0)["verdict"] == "PASS", "copy F2")
    ex(line_verdict("copy_F3", {"all_refused_on_host": False}, MC0)["verdict"] == "FAIL", "copy F3 fail")
    g3 = {"verdict": "FAIL", "rows": [{"id": "x", "verdict": "FAIL: y"}]}
    ex(line_verdict("g3_m03_reduce_gt1", g3, None)["verdict"] == "KILLED", "m03 killed")
    ex(line_verdict("g3_F3", {"verdict": "PASS", "rows": [{"id": "x", "verdict": "PASS"}]}, None)["verdict"] == "PASS",
       "g3 F2")
    ex(line_verdict("k2_f04_F3", None, None)["verdict"].startswith("NO-RESULTS"), "no results")
    # g4 compare
    ex(compare_g4({"t": "passed"}, {"t": "passed"}, set())["verdict"] == "PASS", "g4 same")
    ex(compare_g4({"t": "passed"}, {"t": "failed"}, set())["verdict"] == "FAIL", "g4 diff")
    ex(compare_g4({"t": "failed"}, {"t": "passed"}, {"t"})["verdict"] == "PASS", "g4 intended")
    ex(compare_g4({}, {}, set())["verdict"] == "PASS", "g4 empty")
    # P2 lines (BWD-F3): for each mode, a passing line built from the case table (every registered row, empty /
    # one-element / rank 1-3 arrays), then one failure per bar term, the mutant and the recorded paths.
    import bwdf3_p2_cases as C

    def good_line(mode):
        rows, arr = [], {}
        for rid, kind in C.rows_for(mode):
            r = {"id": rid, "kind": kind, "child_exit": 0, "result": "RAISED" if kind == "refuse" else "RETURNED"}
            if kind == "refuse":
                r["error"] = {"e1": "RuntimeError: export_rope_rows: position past table end (operand 0 ...)",
                              "e2": "RuntimeError: rope_apply: tables must be (T, D) (x ...)",
                              "e3": ("IndexError: reduce: axis 8 is out of bounds for array of dimension 1"
                                     if "range" in rid else "ValueError: reduce: duplicate value in axis (axes [0, 0])"),
                              "count": ("OverflowError: splice_rows: the output NDArray: byte size overflows int64 "
                                        "... [pair 1 of 2]" if rid.startswith("e4-") else
                                        "ValueError: reduce: duplicate value in axis")}[mode]
            if mode == "count":
                r["by_function"] = ({"cudaMallocAsync": 4, "cudaLaunchKernel": 2} if kind == "control" else
                                    {"cudaStreamSynchronize": 1} if kind == "refuse" else {})
            rows.append(r)
            if mode == "e2" and kind == "accept":
                v = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4) * (1 + len(rows))
                arr[f"{rid}|out"], arr[f"{rid}|ref"] = v, v.copy()
        if mode == "e1":
            arr.update({"e1-rope_apply-empty-cuda-short-sin|shape": np.array([0, 2, 2], np.float32),
                        "e1-export-empty-cuda-short-sin|shape": np.array([0, 2, 2], np.float32),
                        "e1-swap-empty-cuda-short-sin|n": np.array([1], np.float32)})
        return {"rows": rows}, arr

    def edit(res, rid, **kw):
        return {"rows": [dict(r, **kw) if r["id"] == rid else r for r in res["rows"]]}

    for mode in C.MODES:
        res, arr = good_line(mode)
        mc = None if mode == "count" else MC0
        ex(line_verdict(f"p2_{mode}_F3", res, mc, arr)["verdict"] == "PASS", f"p2 {mode} F3 pass")
        ex(line_verdict(f"p2_{mode}_F3", {"rows": res["rows"][:-1]}, mc, arr)["verdict"] == "FAIL",
           f"p2 {mode} missing row")
        ex(line_verdict(f"p2_{mode}_F3", {"rows": res["rows"] + [dict(res["rows"][0], id="extra")]}, mc, arr)
           ["verdict"] == "FAIL", f"p2 {mode} extra row")
        ex(line_verdict(f"p2_{mode}_F3", {"rows": []}, mc, arr)["verdict"] == "FAIL", f"p2 {mode} no rows")
        first = res["rows"][0]["id"]
        ex(line_verdict(f"p2_{mode}_F3", edit(res, first, kind="other"), mc, arr)["verdict"] == "FAIL",
           f"p2 {mode} kind")
        ex(line_verdict(f"p2_{mode}_F3", edit(res, first, result="CHILD-DIED", child_exit=-11), mc, arr)
           ["verdict"] == "FAIL", f"p2 {mode} child died")
        ex(line_verdict(f"p2_{mode}_F3", edit(res, first, result="NOT-RUN (line budget spent)", child_exit=None), mc,
                        arr)["verdict"] == "FAIL", f"p2 {mode} not run")
        ref = [r["id"] for r in res["rows"] if r["kind"] == "refuse"][0]
        ex(line_verdict(f"p2_{mode}_F3", edit(res, ref, result="RETURNED"), mc, arr)["verdict"] == "FAIL",
           f"p2 {mode} refusal row returned")
        ex(line_verdict(f"p2_{mode}_F2", edit(res, ref, result="RETURNED"), mc, arr)["verdict"]
           .endswith("defect shown"), f"p2 {mode} F2 recorded")
        ex(line_verdict(f"p2_{mode}_B", res, mc, arr)["verdict"].endswith("defect not shown"), f"p2 {mode} B recorded")
        if mode != "count":
            ex(line_verdict(f"p2_{mode}_F3", res, MC3, arr)["verdict"] == "FAIL", f"p2 {mode} memcheck")
    res, arr = good_line("e1")
    ex(line_verdict("p2_e1_F3", edit(res, "e1-rope_apply-empty-cuda-short-sin", result="RAISED",
                                     error="RuntimeError: rope_apply: position past table end"), MC0, arr)
       ["verdict"] == "FAIL", "p2 e1 empty launch refused")
    ex(line_verdict("p2_e1_F3", res, MC0, dict(arr, **{"e1-export-empty-cuda-short-sin|shape":
                                                         np.array([0, 1, 2], np.float32)}))["verdict"] == "FAIL",
       "p2 e1 wrong empty shape")
    ex(line_verdict("p2_e1_F3", edit(res, "e1-export-reading-short-sin",
                                     error="RuntimeError: export_rope_rows: tables must be (T, D)"), MC0, arr)
       ["verdict"] == "FAIL", "p2 e1 wrong refusal")
    res, arr = good_line("e2")
    acc = [r["id"] for r in res["rows"] if r["kind"] == "accept"][0]
    ex(line_verdict("p2_e2_F3", res, MC0, dict(arr, **{f"{acc}|out": arr[f"{acc}|out"] + 1}))["verdict"] == "FAIL",
       "p2 e2 values differ")
    one = np.array([1.5], np.float32)
    ex(line_verdict("p2_e2_F3", res, MC0, dict(arr, **{f"{acc}|out": one, f"{acc}|ref": one}))["verdict"] == "FAIL",
       "p2 e2 one-element vacuous")
    z = np.zeros((0, 4), np.float32)
    ex(line_verdict("p2_e2_F3", res, MC0, dict(arr, **{f"{acc}|out": z, f"{acc}|ref": z}))["verdict"] == "FAIL",
       "p2 e2 empty vacuous")
    ex(line_verdict("p2_e2_F2", edit(res, acc, result="RAISED", error="RuntimeError: rope_apply: tables must be (T, D)"),
                    MC0, arr)["verdict"].endswith("defect shown"), "p2 e2 F2 refuses the widened table")
    res, arr = good_line("e3")
    ex(line_verdict("p2_e3_F3", edit(res, "e3-sum-duplicate", error="RuntimeError: reduce: duplicate value in axis"),
                    MC0, arr)["verdict"] == "FAIL", "p2 e3 wrong exception type")
    res, arr = good_line("count")
    work = {"cudaMallocAsync": 1, "cudaLaunchKernel": 1}
    killed = edit(res, "e4-swap-pair1-bound-after-work", by_function=work,
                  error="OverflowError: NDArray: byte size overflows int64")
    ex(line_verdict("p2_count_p04_splice_bound_after_work", killed, None, arr)["verdict"] == "KILLED", "p04 killed")
    ex(line_verdict("p2_count_p04_splice_bound_after_work", res, None, arr)["verdict"] == "SURVIVED", "p04 survived")
    ex(line_verdict("p2_count_F3", edit(res, "e4-arena-layer1-bound-after-work",
                                        error="OverflowError: NDArray: byte size overflows int64"), None, arr)
       ["verdict"] == "FAIL", "p2 count late refusal text")
    ex(line_verdict("p2_count_F3", edit(res, "control-sum-valid", by_function={}), None, arr)["verdict"] == "FAIL",
       "p2 count control sees no work")
    ex(line_verdict("p2_count_F3", edit(res, "e1-rope_apply-empty-cuda-count", by_function=work), None, arr)
       ["verdict"] == "FAIL", "p2 count accept row worked")
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
