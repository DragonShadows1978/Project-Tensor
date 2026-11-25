"""BWD-F5 post-device steps for the P2 lines (CPU, NumPy only, no engine import). A copy of the P2 parts of
scripts/bwdf3_gpu_post.py with ONE change of substance: the memcheck number is read per row, from that row's own
memcheck log (artifacts/bwdf1/f5/gpu/<line>/<row id>.memcheck.log), instead of one log for the whole line; a row whose
log is missing, empty, the sanitizer's "terminated before first instrumented API call" or has no ERROR SUMMARY is a
harness failure (never a pass, never "0 errors").

  python3 -B -P scripts/bwdf5_gpu_post.py line <line>   # verdict -> artifacts/bwdf1/f5/gpu/<line>.post.json
  python3 -B -P scripts/bwdf5_gpu_post.py controls      # accepted results, Build B vs Build F4 bitwise (L09/L10/L19)
  python3 -B -P scripts/bwdf5_gpu_post.py selftest      # L48: every verdict path driven on constructed results

Lines: p2_canary_F4 (one control row: PASS when its child returns, exits 0 and its memcheck log reads 0 errors), and
p2_<e1|e2|e3>_<B|F2|F4>. Bars (Build F4; B and F2 are recorded: "defect shown" when the bar fails, "harness
incomplete" when a row's child or memcheck log is not usable): e1: controls return; reading rows refused with
"position past table end"; empty CUDA launches return [0, 2, 2] / [0, 2, 2] / one rope output; e2: widened-table rows
return results bitwise equal to the [T, D] call (non-vacuous); [T, D, 2] refused "tables must be (T, D)"; e3: controls
return; repeated axes refused (ValueError "duplicate value in axis"), out-of-range axes refused (IndexError "is out of
bounds"). Every row: child exit 0 and a memcheck log with ERROR SUMMARY: 0 errors.
Prior art: scripts/bwdf3_gpu_post.py (this repo), copied; compute-sanitizer's ERROR SUMMARY line (NVIDIA
documentation). Ours: the per-row log classification and the harness-failure class.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bwdf5_p2_run as R  # noqa: E402  (no engine import; the case table)

GPU = ROOT / "artifacts/bwdf1/f5/gpu"
BAR_ARM = "F4"
classify_log = R.classify_log


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


def p2_verdict(mode, results, arrays, row_logs):
    """Returns (ok, notes, harness) for one P2 line. `ok` is the bar; `harness` lists rows whose child or memcheck log
    is unusable (those rows also make `ok` False). The line must hold exactly the mode's row IDs
    (scripts/bwdf5_p2_run.py rows_for), each with its registered kind."""
    R_, A = rows_by_id(results), arrays
    bad, harness = [], []
    want_ids = dict(R.rows_for(mode))
    if set(R_) != set(want_ids):
        bad.append(f"row IDs differ from the case table: missing {sorted(set(want_ids) - set(R_))}, "
                   f"extra {sorted(set(R_) - set(want_ids))}")
    for rid, r in R_.items():
        if rid in want_ids and r.get("kind") != want_ids[rid]:
            bad.append(f"{rid}: kind {r.get('kind')} (registered {want_ids[rid]})")
    for rid, r in R_.items():
        if r.get("child_exit") not in (0, None):
            bad.append(f"{rid}: child exit {r.get('child_exit')} ({r.get('result')})")
        if r.get("result") in ("CHILD-DIED", "CHILD-TIMEOUT") or str(r.get("result", "")).startswith("NOT-RUN"):
            harness.append(f"{rid}: {r.get('result')}")
        if str(r.get("result", "")).startswith("NOT-RUN"):
            bad.append(f"{rid}: {r.get('result')}")
        if r.get("result") == "SETUP-ERROR":
            bad.append(f"{rid}: setup error {r.get('error', '')[:160]}")
    # memcheck, per row, from the row's own log: only a parsed ERROR SUMMARY with N = 0 passes
    for rid in R_:
        if str(R_[rid].get("result", "")).startswith("NOT-RUN"):
            continue
        state, n = classify_log(row_logs.get(rid))
        if state != "SUMMARY":
            harness.append(f"{rid}: memcheck log {state}")
            bad.append(f"{rid}: memcheck log {state} (no error count)")
        elif n != 0:
            bad.append(f"{rid}: memcheck errors {n}")
    if mode in ("e1", "canary"):
        for rid, r in R_.items():
            kind, res, err = r.get("kind"), r.get("result"), r.get("error", "")
            if kind == "control" and res != "RETURNED":
                bad.append(f"{rid}: control {res} {err[:120]}")
            elif kind == "refuse" and (res != "RAISED" or not err.startswith("RuntimeError: ")
                                       or "position past table end" not in err):
                bad.append(f"{rid}: not refused on the span ({res}: {err[:120]})")
        want = {} if mode == "canary" else {"e1-rope_apply-empty-cuda-short-sin": ("shape", [0.0, 2.0, 2.0]),
                                            "e1-export-empty-cuda-short-sin": ("shape", [0.0, 2.0, 2.0]),
                                            "e1-swap-empty-cuda-short-sin": ("n", [1.0])}
        for rid, (k, v) in want.items():
            got = A.get(f"{rid}|{k}")
            if R_.get(rid, {}).get("result") != "RETURNED" or got is None or got.tolist() != v:
                bad.append(f"{rid}: {R_.get(rid, {}).get('result')} {R_.get(rid, {}).get('error', '')[:120]} "
                           f"{None if got is None else got.tolist()} (want {v})")
    elif mode == "e2":
        for rid, r in R_.items():
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
        for rid, r in R_.items():
            kind, res, err = r.get("kind"), r.get("result"), r.get("error", "")
            if kind == "control":
                if res != "RETURNED":
                    bad.append(f"{rid}: control {res} {err[:120]}")
            elif not (res == "RAISED" and ((err.startswith("ValueError: ") and "duplicate value in axis" in err) or
                                           (err.startswith("IndexError: ") and "is out of bounds" in err))):
                bad.append(f"{rid}: not refused on the axis ({res}: {err[:120]})")
    else:
        raise ValueError(mode)
    return not bad, bad, harness


def line_verdict(line, results, row_logs, arrays=None):
    out = {"line": line}
    if results is None:
        out["verdict"] = "NO-RESULTS (the step did not finish; see the run log)"
        return out
    _, mode, arm = line.split("_", 2)
    row_logs = row_logs or {}
    ok, notes, harness = p2_verdict(mode, results, arrays or {}, row_logs)
    states = {r["id"]: classify_log(row_logs.get(r["id"])) for r in results.get("rows", [])}
    out["memcheck_errors"] = sum(n for s, n in states.values() if s == "SUMMARY")
    out["memcheck_row_states"] = {k: v[0] for k, v in states.items()}
    out["notes"], out["harness_incomplete"] = notes, harness
    if arm == BAR_ARM:
        out["verdict"] = "PASS" if ok else "FAIL"
    else:
        out["verdict"] = "RECORDED (no bar): " + ("harness incomplete" if harness else
                                                  ("defect not shown" if ok else "defect shown"))
    return out


def post_line(line):
    d = GPU / line
    results = json.loads((d / "results.json").read_text()) if (d / "results.json").exists() else None
    arrays = {}
    for f in sorted(d.glob("*.npz")):  # one npz per row, keys "<row id>|<array>"
        arrays.update({f"{f.stem}|{k}": v for k, v in np.load(f).items()})
    logs = {}
    for r in (results or {}).get("rows", []):
        lg = d / f"{r['id']}.memcheck.log"
        if lg.exists():
            logs[r["id"]] = lg.read_text(errors="replace")
    out = line_verdict(line, results, logs, arrays)
    (GPU / f"{line}.post.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


P2_CONTROL_KEYS = {  # accepted results held to identity, Build B vs Build F4 (L10)
    "p2_e1": ["e1-control-rope_apply-covering|out", "e1-control-export-covering|out"],
    "p2_e2": ["e2-rope_apply-sin-T-D-1|out", "e2-rope_apply-sin-T-D-1|ref", "e2-export-sin-T-D-1|out",
              "e2-export-sin-T-D-1|ref", "e2-swap-tables-T-D-1|ref"],
    "p2_e3": ["e3-control-sum-last|out", "e3-control-sum-two-distinct|out", "e3-control-mean|out",
              "e3-control-argmax|out"],
}


def post_controls():
    rows = {}
    for line, keys in P2_CONTROL_KEYS.items():
        load = lambda arm: {f"{f.stem}|{k}": v for f in sorted((GPU / f"{line}_{arm}").glob("*.npz"))
                            for k, v in np.load(f).items()}
        ab, af = load("B"), load(BAR_ARM)
        for k in keys:
            why = bits_equal(ab.get(k), af.get(k))
            rows[f"{line}|{k}"] = "PASS" if why is None else why
    verdict = "PASS" if rows and all(v == "PASS" for v in rows.values()) else "FAIL"
    (GPU / "CONTROLS_COMPARE_F5.json").write_text(json.dumps({"rows": rows, "verdict": verdict}, indent=1))
    print("controls", verdict, sum(v == "PASS" for v in rows.values()), "/", len(rows))


OK_LOG = "========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n"
ERR_LOG = "========= COMPUTE-SANITIZER\n========= Invalid __global__ read of size 4 bytes\n" \
          "========= ERROR SUMMARY: 3 errors\n"
NO_API_LOG = "========= COMPUTE-SANITIZER\n========= Error: " + R.NO_API + "\n"


def selftest():
    """L48: every verdict path driven on constructed results (empty, one-element and rank 1-3 arrays included)."""
    ok = []
    ex = lambda cond, what: ok.append((what, bool(cond)))
    # the per-row log classes
    ex(classify_log(OK_LOG) == ("SUMMARY", 0), "log: 0 errors")
    ex(classify_log(ERR_LOG) == ("SUMMARY", 3), "log: errors")
    ex(classify_log("========= ERROR SUMMARY: 1 error\n") == ("SUMMARY", 1), "log: 1 error")
    ex(classify_log("") == ("EMPTY", None) and classify_log("  \n") == ("EMPTY", None), "log: empty")
    ex(classify_log(None) == ("MISSING", None), "log: missing")
    ex(classify_log(NO_API_LOG) == ("NO-API-CALL", None), "log: terminated before first API call (BWD-F3's)")
    ex(classify_log("========= COMPUTE-SANITIZER\nsomething else\n") == ("UNPARSED", None), "log: unparsed")
    ex(classify_log(NO_API_LOG + "========= ERROR SUMMARY: 0 errors\n")[0] == "NO-API-CALL", "log: no-api wins over 0")
    # the child-end record
    ex(R.child_end(0, False) == {"kind": "exit", "code": 0}, "end: exit 0")
    ex(R.child_end(255, False) == {"kind": "exit", "code": 255}, "end: exit 255")
    ex(R.child_end(-11, False)["signal"] == "SIGSEGV", "end: signal")
    ex(R.child_end(None, True) == {"kind": "timeout"}, "end: timeout")
    a = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    ex(bits_equal(a, a.copy()) is None, "bits equal")
    ex(bits_equal(a, a + 1) is not None, "bits differ")
    ex(bits_equal(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)).startswith("VACUOUS"), "empty vacuous")
    ex(bits_equal(np.ones(1, np.float32), np.ones(1, np.float32)).startswith("VACUOUS"), "one-element vacuous")
    ex(line_verdict("p2_e1_F4", None, None)["verdict"].startswith("NO-RESULTS"), "no results")

    def good_line(mode):
        rows, arr, logs = [], {}, {}
        for rid, kind in R.rows_for(mode):
            r = {"id": rid, "kind": kind, "child_exit": 0, "result": "RAISED" if kind == "refuse" else "RETURNED"}
            if kind == "refuse":
                r["error"] = {"e1": "RuntimeError: export_rope_rows: position past table end (operand 0 ...)",
                              "e2": "RuntimeError: rope_apply: tables must be (T, D) (x ...)",
                              "e3": ("IndexError: reduce: axis 8 is out of bounds for array of dimension 1"
                                     if "range" in rid else "ValueError: reduce: duplicate value in axis (axes [0, 0])")
                              }[mode]
            rows.append(r)
            logs[rid] = OK_LOG
            if mode == "e2" and kind == "accept":
                v = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4) * (1 + len(rows))
                arr[f"{rid}|out"], arr[f"{rid}|ref"] = v, v.copy()
        if mode == "e1":
            arr.update({"e1-rope_apply-empty-cuda-short-sin|shape": np.array([0, 2, 2], np.float32),
                        "e1-export-empty-cuda-short-sin|shape": np.array([0, 2, 2], np.float32),
                        "e1-swap-empty-cuda-short-sin|n": np.array([1], np.float32)})
        return {"rows": rows}, arr, logs

    def edit(res, rid, **kw):
        return {"rows": [dict(r, **kw) if r["id"] == rid else r for r in res["rows"]]}

    def v(line, res, logs, arr):
        return line_verdict(line, res, logs, arr)["verdict"]

    for mode in R.LINE_MODES:
        res, arr, logs = good_line(mode)
        arm = "F4"
        L = f"p2_{mode}_{arm}"
        ex(v(L, res, logs, arr) == "PASS", f"p2 {mode} F4 pass")
        ex(v(L, {"rows": res["rows"][:-1]} if len(res["rows"]) > 1 else {"rows": []}, logs, arr) == "FAIL",
           f"p2 {mode} missing row")
        ex(v(L, {"rows": res["rows"] + [dict(res["rows"][0], id="extra")]}, logs, arr) == "FAIL", f"p2 {mode} extra row")
        ex(v(L, {"rows": []}, logs, arr) == "FAIL", f"p2 {mode} no rows")
        first = res["rows"][0]["id"]
        ex(v(L, edit(res, first, kind="other"), logs, arr) == "FAIL", f"p2 {mode} kind")
        # a dead child (BWD-F3's failure: exit 255, nothing recorded) and a signal death
        dead = edit(res, first, result="CHILD-DIED", child_exit=255, error="child exit 255: ")
        ex(v(L, dead, dict(logs, **{first: NO_API_LOG}), arr) == "FAIL", f"p2 {mode} dead child (exit 255, no-API log)")
        ex(v(L, edit(res, first, result="CHILD-DIED", child_exit=-11), logs, arr) == "FAIL", f"p2 {mode} child died")
        ex(v(L, edit(res, first, result="CHILD-TIMEOUT", child_exit=124), logs, arr) == "FAIL", f"p2 {mode} timeout")
        ex(v(L, edit(res, first, result="NOT-RUN (line budget spent)", child_exit=None), logs, arr) == "FAIL",
           f"p2 {mode} not run")
        ex(v(L, edit(res, first, result="SETUP-ERROR", error="X: y"), logs, arr) == "FAIL", f"p2 {mode} setup error")
        # memcheck: errors, empty, missing, no-API, unparsed log on one row
        ex(v(L, res, dict(logs, **{first: ERR_LOG}), arr) == "FAIL", f"p2 {mode} memcheck errors")
        ex(v(L, res, dict(logs, **{first: ""}), arr) == "FAIL", f"p2 {mode} empty log")
        lg = {k: x for k, x in logs.items() if k != first}
        ex(v(L, res, lg, arr) == "FAIL", f"p2 {mode} missing log")
        ex(v(L, res, dict(logs, **{first: NO_API_LOG}), arr) == "FAIL", f"p2 {mode} no-API log")
        ex(v(L, res, dict(logs, **{first: "junk\n"}), arr) == "FAIL", f"p2 {mode} unparsed log")
        ex(line_verdict(L, res, dict(logs, **{first: ERR_LOG}), arr)["memcheck_errors"] == 3, f"p2 {mode} error count")
        ex(line_verdict(L, res, logs, arr)["memcheck_errors"] == 0, f"p2 {mode} zero count")
        if mode == "canary":
            continue
        ref = [r["id"] for r in res["rows"] if r["kind"] == "refuse"]
        if ref:
            ex(v(L, edit(res, ref[0], result="RETURNED"), logs, arr) == "FAIL", f"p2 {mode} refusal row returned")
            ex(v(f"p2_{mode}_F2", edit(res, ref[0], result="RETURNED"), logs, arr).endswith("defect shown"),
               f"p2 {mode} F2 recorded")
        ex(v(f"p2_{mode}_B", res, logs, arr).endswith("defect not shown"), f"p2 {mode} B recorded")
        ex(v(f"p2_{mode}_F2", dead, dict(logs, **{first: NO_API_LOG}), arr).endswith("harness incomplete"),
           f"p2 {mode} F2 dead child is a harness failure, not a defect")
        ex(v(f"p2_{mode}_B", res, dict(logs, **{first: ""}), arr).endswith("harness incomplete"),
           f"p2 {mode} B empty log is a harness failure")
    res, arr, logs = good_line("canary")
    ex(len(res["rows"]) == 1 and res["rows"][0]["kind"] == "control", "canary is one control row")
    ex(v("p2_canary_F4", edit(res, res["rows"][0]["id"], id="e1-export-reading-short-sin"), logs, arr) == "FAIL",
       "canary wrong row")
    res, arr, logs = good_line("e1")
    ex(v("p2_e1_F4", edit(res, "e1-rope_apply-empty-cuda-short-sin", result="RAISED",
                          error="RuntimeError: rope_apply: position past table end"), logs, arr) == "FAIL",
       "p2 e1 empty launch refused")
    ex(v("p2_e1_F4", res, logs, dict(arr, **{"e1-export-empty-cuda-short-sin|shape": np.array([0, 1, 2], np.float32)}))
       == "FAIL", "p2 e1 wrong empty shape")
    ex(v("p2_e1_F4", edit(res, "e1-export-reading-short-sin", error="RuntimeError: export_rope_rows: tables must be "
                                                                   "(T, D)"), logs, arr) == "FAIL",
       "p2 e1 wrong refusal")
    res, arr, logs = good_line("e2")
    acc = [r["id"] for r in res["rows"] if r["kind"] == "accept"][0]
    ex(v("p2_e2_F4", res, logs, dict(arr, **{f"{acc}|out": arr[f"{acc}|out"] + 1})) == "FAIL", "p2 e2 values differ")
    one = np.array([1.5], np.float32)
    ex(v("p2_e2_F4", res, logs, dict(arr, **{f"{acc}|out": one, f"{acc}|ref": one})) == "FAIL",
       "p2 e2 one-element vacuous")
    z = np.zeros((0, 4), np.float32)
    ex(v("p2_e2_F4", res, logs, dict(arr, **{f"{acc}|out": z, f"{acc}|ref": z})) == "FAIL", "p2 e2 empty vacuous")
    ex(v("p2_e2_F2", edit(res, acc, result="RAISED", error="RuntimeError: rope_apply: tables must be (T, D)"), logs,
         arr).endswith("defect shown"), "p2 e2 F2 refuses the widened table")
    res, arr, logs = good_line("e3")
    ex(v("p2_e3_F4", edit(res, "e3-sum-duplicate", error="RuntimeError: reduce: duplicate value in axis"), logs, arr)
       == "FAIL", "p2 e3 wrong exception type")
    bad = [w for w, x in ok if not x]
    print(json.dumps({"checks": len(ok), "failed": bad}))
    print("POST-SELFTEST", "PASS" if not bad else "FAIL")
    return 0 if not bad else 1


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "line":
        post_line(sys.argv[2])
    elif m == "controls":
        post_controls()
    elif m == "selftest":
        sys.exit(selftest())
