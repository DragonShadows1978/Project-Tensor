"""ENG5-S post-device code of the staged G2 lines (pure Python, no engine import): the g3_<arm> verdicts and a CPU
self-test that drives every verdict branch on constructed rows (L48): each site raising, returning, an unfired
injection, a dead child, a missing row, a failed success control, and each mutant's kill and survival.

  python3 -B -P scripts/eng5s_gpu_verdict.py selftest            -> artifacts/eng5s/gpu_dry/VERDICT_SELFTEST.json
  python3 -B -P scripts/eng5s_gpu_verdict.py line <line> <dir>   -> prints the verdict of one finished line

Criteria (orders/ENG5S_STATUS_CHECKS.md, S1, S2, G2, G4):
- Build S (g3_S) PASSES when every one of the ten injected rows (the nine sites; site 7 in both directions; site 2
  through cudaMalloc and cudaMallocAsync) fired every armed injection and raised RuntimeError whose message starts
  with "<entry>: <call> failed: cudaErrorInvalidValue: " (S1: entry, call, cudaGetErrorName), and every success control
  (the same call with no injection, and axpy_ on empty operands) returned with the expected value (S2). A missing row,
  a dead child, an unfired injection or a failed control fails it.
- Build B (g3_B) is RECORDED: per row, whether it returned under the fired injection (the silent failure S1 removes).
- A mutant line (g3_s0N) is KILLED when its registered row (artifacts/eng5s/MUTANTS.json) fired its injection and the
  call returned (outcome ok), i.e. that row's Build S assertion fails; SURVIVED otherwise (an unfired injection, a dead
  child or a raise is not a kill). The executed marker is recorded beside it.
The semantic verdict decides each line; the wrapper records exits only.

Prior art: ENG5-F2's scripts/eng5_f2_gpu_verdict.py (this repo), whose g3 rules this extends. No other prior art known
to me.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ERR = "cudaErrorInvalidValue: "
PREFIX = {
    "synchronize": "cuda_sync: cudaDeviceSynchronize failed: " + ERR,
    "oom_drain": "Storage (cudaMalloc OOM retry): cudaStreamSynchronize (drain) failed: " + ERR,
    "oom_drain_pooled": "Storage (cudaMallocAsync OOM retry): cudaStreamSynchronize (drain) failed: " + ERR,
    "zeros": "NDArray::zeros: cudaMemset failed: " + ERR,
    "from_host": "NDArray::from_host: cudaMemcpy failed: " + ERR,
    "to_host": "NDArray::to_host: cudaMemcpy failed: " + ERR,
    "clone": "NDArray::clone: cudaMemcpy failed: " + ERR,
    "to_d2h": "NDArray::to: cudaMemcpy failed: " + ERR,
    "to_h2d": "NDArray::to: cudaMemcpy failed: " + ERR,
    "axpy_": "axpy_: cudaMemcpy failed: " + ERR,
    "packed_prefill_dump": "apa_spq_packed_prefill: cudaMemset (bulk NaN fill) failed: " + ERR,
}
SITE = {"synchronize": 1, "oom_drain": 2, "oom_drain_pooled": 2, "zeros": 3, "from_host": 4, "to_host": 5, "clone": 6,
        "to_d2h": 7, "to_h2d": 7, "axpy_": 8, "packed_prefill_dump": 9}
INJECTED = list(PREFIX)
CONTROL_ONLY = ["axpy_empty"]
KILL_ROWS = {"s01": ["clone"], "s02": ["to_d2h", "to_h2d"], "s03": ["zeros"], "s04": ["from_host"]}
MUTANT_IDS = {"s01": "s01_clone_status_dropped", "s02": "s02_to_status_dropped", "s03": "s03_zeros_status_dropped",
              "s04": "s04_from_host_last_error_only"}


def fired_all(att):
    f = (att or {}).get("fired") or {}
    return bool(att) and bool(att.get("armed")) and all(f.get(n) == 1 for n in att["armed"])


def control_ok(case, ctl):
    """S2: the uninjected call returns its expected value."""
    if not ctl or ctl.get("outcome") != "ok":
        return False, "control did not return"
    v = ctl.get("value")
    if case in ("synchronize",):
        return True, "returned"
    if case == "axpy_empty":
        return v == [0], "returned an empty parameter" if v == [0] else "unexpected shape"
    if case == "packed_prefill_dump":
        ok = (isinstance(v, dict) and v.get("out_finite") and v.get("out_informative") and v.get("bulk_nan", 0) > 0
              and v.get("bulk_finite", 0) > 0)
        return bool(ok), "out finite and informative; bulk has NaN fill and finite scores" if ok else "bad prefill value"
    if not isinstance(v, dict) or not v.get("equals_expected"):
        return False, "value differs from the expected bytes"
    if case != "zeros" and not (v.get("informative") and v.get("expected_informative")):
        return False, "value comparison on a constant or zero array (L19)"
    return True, "expected bytes" + (" (all-zero by design; the injected row decides site 3)" if case == "zeros" else "")


def row_pass(case, r):
    """Build S's assertion for one injected row. Returns (pass, why)."""
    if r is None:
        return False, "row missing"
    if r.get("outcome") in ("child_failed", "not_run"):
        return False, "child failed: " + str(r.get("returncode"))
    inj = r.get("injected")
    if not inj:
        return False, "no injected attempt"
    if not fired_all(inj):
        return False, "an armed injection did not fire: " + json.dumps(inj.get("fired"))
    msg = inj.get("message") or ""
    if inj.get("outcome") == "raised" and inj.get("exc_type") == "RuntimeError" and msg.startswith(PREFIX[case]):
        return True, "raised naming the entry, the call and the CUDA error name"
    if inj.get("outcome") == "ok":
        return False, "returned under a fired injection (the failure passed silently)"
    return False, "raised without the S1 form: " + msg[:160]


def judge_rows(rows, short):
    have = {r.get("case"): r for r in rows}
    out = {"arm": short, "rows": {}, "missing": []}
    for case in INJECTED + CONTROL_ONLY:
        r = have.get(case)
        if r is None:
            out["missing"].append(case)
            continue
        if r.get("outcome") == "not_applicable":
            out["rows"][case] = {"result": "not_applicable", "why": r.get("why")}
            continue
        ent = {"site": SITE.get(case)}
        if case in INJECTED:
            ok, why = row_pass(case, r)
            ent["injected_pass"], ent["injected_why"] = ok, why
            inj = r.get("injected") or {}
            ent["injected_fired"] = fired_all(inj)
            ent["returned_under_fired_injection"] = fired_all(inj) and inj.get("outcome") == "ok"
            ent["injected_outcome"] = inj.get("outcome")
            ent["injected_message"] = inj.get("message")
        cok, cwhy = control_ok(case, (r.get("control") if r.get("outcome") not in ("child_failed", "not_run") else None))
        ent["control_pass"], ent["control_why"] = cok, cwhy
        out["rows"][case] = ent
    return out


def judge_g3(rows, short, marker_lines=None):
    out = judge_rows(rows, short)
    rs = out["rows"]
    if short == "S":
        need = INJECTED + CONTROL_ONLY
        ok = (not out["missing"] and all(c in rs for c in need) and
              all(rs[c].get("injected_pass") for c in INJECTED) and all(rs[c].get("control_pass") for c in need))
        out["assertions"] = {"every_injected_call_raises_in_S1_form": all(rs.get(c, {}).get("injected_pass")
                                                                          for c in INJECTED),
                             "every_success_control_unchanged": all(rs.get(c, {}).get("control_pass") for c in need),
                             "rows_complete": not out["missing"]}
        out["verdict"] = "PASS" if ok else "FAIL"
    elif short in KILL_ROWS:
        cs = KILL_ROWS[short]
        killed = all(c in rs and rs[c].get("returned_under_fired_injection") for c in cs)
        out["mutant_assertion"] = {"mutant": MUTANT_IDS[short], "rows": cs,
                                   "name": f"{'/'.join(cs)}: Build S's assertion (raise in the S1 form under a fired "
                                           "injection) fails on the mutant: the call returns",
                                   "met": killed,
                                   "executed_marker_lines": (marker_lines or []).count(MUTANT_IDS[short])}
        out["verdict"] = "KILLED" if killed else "SURVIVED"
    else:
        out["returned_under_fired_injection"] = {c: rs[c].get("returned_under_fired_injection") for c in INJECTED
                                                 if c in rs and "injected_pass" in rs[c]}
        out["verdict"] = "RECORDED (reference arm)"
    return out


def judge_line(line, d):
    short = line.split("_", 1)[1]
    p = os.path.join(d, "g3_rows.json")
    rows = json.load(open(p)) if os.path.exists(p) else []
    mp = os.path.join(d, "mutant_marker.txt")
    marks = [l.strip() for l in open(mp)] if os.path.exists(mp) else []
    return dict(judge_g3(rows, short, marks), line=line)


# ---------------------------------------------------------------- self-test on constructed rows (L48)
def _good_value(case):
    if case == "packed_prefill_dump":
        return {"out_finite": True, "out_informative": True, "bulk_nan": 24, "bulk_finite": 40, "bulk_size": 64}
    if case == "synchronize":
        return None
    if case == "axpy_empty":
        return [0]
    return {"equals_expected": True, "informative": case != "zeros", "expected_informative": case != "zeros"}


def _s_rows():
    rows = []
    for c in INJECTED:
        armed = (["cudaMalloc" if c == "oom_drain" else "cudaMallocAsync", "cudaStreamSynchronize"]
                 if c.startswith("oom") else ["x"])
        rows.append({"case": c, "control": {"armed": [], "outcome": "ok", "value": _good_value(c), "fired": {}},
                     "injected": {"armed": armed, "fired": {n: 1 for n in armed}, "outcome": "raised",
                                  "exc_type": "RuntimeError", "message": PREFIX[c] + "invalid argument"}})
    rows.append({"case": "axpy_empty", "control": {"armed": [], "outcome": "ok", "value": [0], "fired": {}}})
    return rows


def _returned(rows, cases):
    out = []
    for r in rows:
        if r["case"] in cases:
            r = json.loads(json.dumps(r))
            r["injected"].update(outcome="ok", exc_type=None, message=None, value={"equals_expected": False})
        out.append(r)
    return out


def selftest():
    res, fails = [], []

    def chk(name, got, want):
        res.append({"check": name, "got": got, "want": want, "ok": got == want})
        if got != want:
            fails.append(name)

    good = _s_rows()
    chk("S: every site raising in the S1 form + controls -> PASS", judge_g3(good, "S")["verdict"], "PASS")
    chk("S: no rows -> FAIL", judge_g3([], "S")["verdict"], "FAIL")
    for c in INJECTED:
        chk(f"S: {c} returning under a fired injection -> FAIL", judge_g3(_returned(good, [c]), "S")["verdict"], "FAIL")
        rows = [dict(r, injected=dict(r["injected"], fired={n: 0 for n in r["injected"]["armed"]}))
                if r["case"] == c else r for r in good]
        chk(f"S: {c} unfired injection -> FAIL", judge_g3(rows, "S")["verdict"], "FAIL")
        rows = [{"case": c, "outcome": "child_failed", "returncode": -11} if r["case"] == c else r for r in good]
        chk(f"S: {c} dead child -> FAIL", judge_g3(rows, "S")["verdict"], "FAIL")
        rows = [r for r in good if r["case"] != c]
        chk(f"S: {c} missing -> FAIL", judge_g3(rows, "S")["verdict"], "FAIL")
        rows = [dict(r, injected=dict(r["injected"], message="CUDA error at x: invalid argument"))
                if r["case"] == c else r for r in good]
        chk(f"S: {c} raising without the S1 form -> FAIL", judge_g3(rows, "S")["verdict"], "FAIL")
        rows = [dict(r, control=dict(r["control"], outcome="raised")) if r["case"] == c else r for r in good]
        chk(f"S: {c} failed success control -> FAIL", judge_g3(rows, "S")["verdict"], "FAIL")
    rows = [dict(r, control=dict(r["control"], outcome="raised")) if r["case"] == "axpy_empty" else r for r in good]
    chk("S: axpy_ on empty operands raising -> FAIL", judge_g3(rows, "S")["verdict"], "FAIL")
    rows = [dict(r, control=dict(r["control"], value={"equals_expected": True, "informative": False,
                                                      "expected_informative": False}))
            if r["case"] == "clone" else r for r in good]
    chk("S: control compared on a constant array -> FAIL (L19)", judge_g3(rows, "S")["verdict"], "FAIL")
    b = _returned(good, INJECTED)
    vb = judge_g3(b, "B")
    chk("B: recorded", vb["verdict"], "RECORDED (reference arm)")
    chk("B: every row returned under a fired injection", all(vb["returned_under_fired_injection"].values()), True)
    for short, cs in KILL_ROWS.items():
        mrows = [r for r in _returned(good, cs) if r["case"] != "packed_prefill_dump"]
        mrows.append({"case": "packed_prefill_dump", "outcome": "not_applicable", "why": "no hooks build"})
        v = judge_g3(mrows, short, [MUTANT_IDS[short]])
        chk(f"{short}: its row returning -> KILLED", v["verdict"], "KILLED")
        chk(f"{short}: marker counted", v["mutant_assertion"]["executed_marker_lines"], 1)
        chk(f"{short}: all rows raising -> SURVIVED", judge_g3(good, short)["verdict"], "SURVIVED")
        unf = [dict(r, injected=dict(r["injected"], fired={n: 0 for n in r["injected"]["armed"]}))
               if r["case"] in cs else r for r in mrows]
        chk(f"{short}: returning under an unfired injection -> SURVIVED", judge_g3(unf, short)["verdict"], "SURVIVED")
        dead = [{"case": c, "outcome": "child_failed", "returncode": -6} if r["case"] in cs else r for r in mrows]
        chk(f"{short}: dead child -> SURVIVED", judge_g3(dead, short)["verdict"], "SURVIVED")
        chk(f"{short}: missing row -> SURVIVED",
            judge_g3([r for r in mrows if r["case"] not in cs], short)["verdict"], "SURVIVED")
    if len(KILL_ROWS["s02"]) == 2:
        half = [r for r in _returned(good, ["to_d2h"]) if r["case"] != "packed_prefill_dump"]
        chk("s02: only one direction returning -> SURVIVED (both directions registered)",
            judge_g3(half, "s02")["verdict"], "SURVIVED")
    out = {"checks": len(res), "failed": fails, "rows": res}
    os.makedirs(os.path.join(ROOT, "artifacts/eng5s/gpu_dry"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "artifacts/eng5s/gpu_dry/VERDICT_SELFTEST.json"), "w"), indent=1)
    print("self-test checks", len(res), "failed", fails)
    return 0 if not fails else 1


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        sys.exit(selftest())
    elif sys.argv[1] == "line":
        print(json.dumps(judge_line(sys.argv[2], sys.argv[3]), indent=1))
