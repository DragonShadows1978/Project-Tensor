"""ENG5-S-T verdict code (orders/ENG5ST_REGRESSION_TEST.md, T2 and T3): the per-arm verdict of one GPU line, read from the
junit XML that line wrote for tensor_cuda/tests/test_eng5s_status_checks.py, plus its CPU drive on constructed junit
files (T3). Pure Python; no engine import.

  python3 -B -P scripts/eng5st_verdict.py selftest             -> artifacts/eng5s/t/T3_VERDICT_DRIVE.json
  python3 -B -P scripts/eng5st_verdict.py junit <arm> <xml>    -> prints the verdict of one junit file

Rows read: the eleven GPU-lane tests test_gpu_status_rows[<id>] (ten injected rows and the axpy_empty control-only row).
Row status from the junit element:
  passed        no failure/error/skipped child
  killed        a <failure> whose text carries "returned under a fired injection" (the test body's own message when the
                injection fired and the call returned: the status check is missing). Only this counts as a kill (L41).
  failed_other  any other <failure> (a dead child, an unfired injection, a wrong message, a failed control)
  error         an <error> (fixture or setup: e.g. the shim did not build): never a kill, never a pass
  skipped       a <skipped> (a GPU row on the CPU lane)
Verdicts:
  INVALID       junit missing, empty or unparseable; or any of the eleven rows missing, skipped or errored (L16: a run
                that checked nothing is not a pass)
  S:            PASS when all eleven rows passed, else FAIL
  B:            DEFECT when all ten injected rows are killed and axpy_empty passed (the test detects the missing checks);
                B_UNEXPECTED otherwise (names the rows)
  s01..s04:     KILLED when every row of the mutant's own site is killed and every other row passed, and (when marker
                lines are supplied) the mutant's marker was written, i.e. its mutated code ran; KILLED_NO_MARKER when the
                rows say killed but the marker is absent; COLLATERAL when its own rows are killed but another row did not
                pass; SURVIVED when any own row is not killed (L41: each mutant is judged on its own site's rows).

Prior art: ENG5-S's scripts/eng5s_gpu_verdict.py (this repo), whose per-site kill rule (KILL_ROWS) this reuses; mutation
testing's kill criterion (DeMillo, Lipton and Sayward, 1978). Ours: reading the verdict from per-row junit statuses and
the "killed" signature. No other prior art known to me.
"""
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST = "test_gpu_status_rows"
INJECTED = ["synchronize", "oom_drain", "oom_drain_pooled", "zeros", "from_host", "to_host", "clone", "to_d2h",
            "to_h2d", "axpy_"]
CONTROL = ["axpy_empty"]
ROWS = INJECTED + CONTROL
KILL_ROWS = {"s01": ["clone"], "s02": ["to_d2h", "to_h2d"], "s03": ["zeros"], "s04": ["from_host"]}
MUTANT_IDS = {"s01": "s01_clone_status_dropped", "s02": "s02_to_status_dropped", "s03": "s03_zeros_status_dropped",
              "s04": "s04_from_host_last_error_only"}
SIGNATURE = "returned under a fired injection"


def parse(path):
    """Row id -> status, or (None, why) when the file cannot be used."""
    if not os.path.exists(path):
        return None, "junit file missing"
    if os.path.getsize(path) == 0:
        return None, "junit file empty"
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as ex:
        return None, "junit file unparseable: " + str(ex)
    status = {}
    for tcase in root.iter("testcase"):
        name = tcase.get("name") or ""
        if not (name.startswith(TEST + "[") and name.endswith("]")):
            continue
        row = name[len(TEST) + 1:-1]
        kinds = {c.tag: c for c in tcase}
        if "error" in kinds:
            st = "error"
        elif "failure" in kinds:
            f = kinds["failure"]
            st = "killed" if SIGNATURE in ((f.get("message") or "") + (f.text or "")) else "failed_other"
        elif "skipped" in kinds:
            st = "skipped"
        else:
            st = "passed"
        status[row] = st
    return status, None


def judge(arm, path, marker_lines=None):
    status, why = parse(path)
    out = {"arm": arm, "junit": os.path.relpath(path, ROOT) if os.path.isabs(path) and path.startswith(ROOT) else path}
    if status is None:
        return dict(out, verdict="INVALID", why=why, rows={})
    bad = {r: status.get(r, "missing") for r in ROWS if status.get(r, "missing") in ("missing", "skipped", "error")}
    out["rows"] = {r: status.get(r, "missing") for r in ROWS}
    if bad:
        return dict(out, verdict="INVALID", why="rows missing, skipped or errored: " + json.dumps(bad))
    rs = out["rows"]
    if arm == "S":
        notpass = {r: s for r, s in rs.items() if s != "passed"}
        return dict(out, verdict="PASS" if not notpass else "FAIL", not_passed=notpass)
    if arm == "B":
        not_killed = [r for r in INJECTED if rs[r] != "killed"]
        ctl = [r for r in CONTROL if rs[r] != "passed"]
        return dict(out, verdict="DEFECT" if not (not_killed or ctl) else "B_UNEXPECTED",
                    injected_not_killed=not_killed, control_not_passed=ctl)
    own = KILL_ROWS[arm]
    own_not_killed = [r for r in own if rs[r] != "killed"]
    others_not_passed = {r: s for r, s in rs.items() if r not in own and s != "passed"}
    out.update(own_rows=own, own_not_killed=own_not_killed, others_not_passed=others_not_passed,
               executed_marker_lines=None if marker_lines is None else list(marker_lines).count(MUTANT_IDS[arm]))
    if own_not_killed:
        return dict(out, verdict="SURVIVED")
    if others_not_passed:
        return dict(out, verdict="COLLATERAL")
    if marker_lines is not None and out["executed_marker_lines"] < 1:
        return dict(out, verdict="KILLED_NO_MARKER")
    return dict(out, verdict="KILLED")


# ---------------------------------------------------------------- T3: drive on constructed junit files
def junit_xml(statuses, skip_all=False):
    """A junit document shaped like pytest's: one testcase per row, plus the four CPU-lane tests (skipped on a GPU
    lane, as in the real run)."""
    cls = "tensor_cuda.tests.test_eng5s_status_checks"
    cases = []
    for r in ("synchronize_injected", "synchronize_no_device", "oom_drain", "oom_drain_pooled"):
        cases.append(f'<testcase classname="{cls}" name="test_cpu_status_rows[{r}]" time="0"><skipped type="pytest.skip" '
                     'message="CPU-lane rows run only with CUDA_VISIBLE_DEVICES empty; this is the GPU lane"/></testcase>')
    for r, st in statuses.items():
        body = {"passed": "", "skipped": '<skipped type="pytest.skip" message="GPU-lane rows need a visible GPU"/>',
                "killed": f'<failure message="AssertionError: {r}: {SIGNATURE} (the failure passed silently)">trace</failure>',
                "failed_other": '<failure message="AssertionError: row child failed: rc=-11">trace</failure>',
                "error": '<error message="failed on setup with shim build failed">trace</error>'}[st]
        cases.append(f'<testcase classname="{cls}" name="{TEST}[{r}]" time="0.1">{body}</testcase>')
    return ('<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" tests="%d">%s</testsuite>'
            '</testsuites>' % (len(cases), "".join(cases)))


def selftest():
    os.makedirs("/tmp/eng5st_tmp", exist_ok=True)
    d = tempfile.mkdtemp(prefix="t3_", dir="/tmp/eng5st_tmp")  # the step-body drive's scratch (write-once dirs)
    jd = os.path.join(ROOT, "artifacts/eng5s/t/t3_junit")     # the constructed junit files, kept as receipts
    os.makedirs(jd, exist_ok=True)
    res, fails = [], []
    n = [0]

    def drive(name, arm, want, statuses=None, raw=None, marks=None, absent=False):
        n[0] += 1
        p = os.path.join(jd, f"c{n[0]:02d}.xml")
        if not absent:
            with open(p, "w") as f:
                f.write(raw if raw is not None else junit_xml(statuses))
        v = judge(arm, p, marks)
        ok = v["verdict"] == want
        res.append({"case": name, "arm": arm, "junit": p, "want": want, "got": v["verdict"], "ok": ok,
                    "detail": {k: v[k] for k in v if k in ("why", "not_passed", "injected_not_killed",
                                                            "control_not_passed", "own_not_killed",
                                                            "others_not_passed", "executed_marker_lines")}})
        if not ok:
            fails.append(name)

    allp = {r: "passed" for r in ROWS}
    allk = {r: ("killed" if r in INJECTED else "passed") for r in ROWS}

    def with_(base, **kw):
        return dict(base, **kw)

    mk = {s: [MUTANT_IDS[s]] for s in KILL_ROWS}
    # --- the six cases the order names, on arm S and on a mutant
    drive("S: all rows pass", "S", "PASS", allp)
    drive("S: one row failed (killed signature)", "S", "FAIL", with_(allp, clone="killed"))
    drive("S: one row failed (a dead child)", "S", "FAIL", with_(allp, zeros="failed_other"))
    drive("S: the control-only row failed", "S", "FAIL", with_(allp, axpy_empty="failed_other"))
    drive("s01: a mutant failing its own row (clone), all others pass", "s01", "KILLED",
          with_(allp, clone="killed"), marks=mk["s01"])
    drive("s01: a mutant failing a different row (zeros), own row passes", "s01", "SURVIVED",
          with_(allp, zeros="killed"), marks=mk["s01"])
    drive("S: a missing row", "S", "INVALID", {r: "passed" for r in ROWS if r != "to_h2d"})
    drive("s01: a missing row (its own)", "s01", "INVALID", {r: "passed" for r in ROWS if r != "clone"})
    drive("S: an empty file", "S", "INVALID", raw="")
    drive("s03: an empty file", "s03", "INVALID", raw="")
    drive("S: an unparseable file", "S", "INVALID", raw="<testsuites><testsuite")
    drive("S: no file", "S", "INVALID", absent=True)
    # --- the same on every other mutant
    for s, own in KILL_ROWS.items():
        killed = {r: "killed" for r in own}
        drive(f"{s}: its own row(s) fail, all others pass", s, "KILLED", with_(allp, **killed), marks=mk[s])
        drive(f"{s}: all rows pass", s, "SURVIVED", allp, marks=mk[s])
        drive(f"{s}: its own row fails and another row fails too", s, "COLLATERAL",
              with_(allp, **killed, axpy_="killed" if "axpy_" not in own else "failed_other"), marks=mk[s])
        drive(f"{s}: its own row fails by a dead child, not the kill signature", s, "SURVIVED",
              with_(allp, **{own[0]: "failed_other"}), marks=mk[s])
        drive(f"{s}: rows say killed but its marker never appeared", s, "KILLED_NO_MARKER",
              with_(allp, **killed), marks=[])
    drive("s02: only one direction (to_d2h) fails", "s02", "SURVIVED", with_(allp, to_d2h="killed"), marks=mk["s02"])
    drive("s02: only the other direction (to_h2d) fails", "s02", "SURVIVED", with_(allp, to_h2d="killed"),
          marks=mk["s02"])
    # --- arm B
    drive("B: every injected row killed, control passes", "B", "DEFECT", allk)
    drive("B: the test misses the defect (all rows pass)", "B", "B_UNEXPECTED", allp)
    drive("B: one injected row passes", "B", "B_UNEXPECTED", with_(allk, to_host="passed"))
    drive("B: the control row fails", "B", "B_UNEXPECTED", with_(allk, axpy_empty="killed"))
    drive("B: one injected row fails by a dead child", "B", "B_UNEXPECTED", with_(allk, clone="failed_other"))
    # --- a run that checked nothing is never a pass (L16)
    drive("S: every GPU row skipped (a CPU-lane run)", "S", "INVALID", {r: "skipped" for r in ROWS})
    drive("B: every GPU row skipped", "B", "INVALID", {r: "skipped" for r in ROWS})
    drive("s02: every GPU row skipped", "s02", "INVALID", {r: "skipped" for r in ROWS}, marks=mk["s02"])
    drive("S: setup errors (the shim did not build)", "S", "INVALID", {r: "error" for r in ROWS})
    drive("B: setup errors are not a kill", "B", "INVALID", {r: "error" for r in ROWS})
    # --- the real CPU-lane junit files of T1 (GPU rows skipped there): INVALID on any arm
    for f in ("T1_S.xml", "T1_B.xml"):
        p = os.path.join(ROOT, "artifacts/eng5s/t", f)
        if os.path.exists(p):
            v = judge("S" if f == "T1_S.xml" else "B", p)
            ok = v["verdict"] == "INVALID"
            res.append({"case": f"real CPU-lane junit {f}: GPU verdict INVALID", "junit": p, "want": "INVALID",
                        "got": v["verdict"], "ok": ok})
            if not ok:
                fails.append(f)
    # --- the step's body on a stand-in pytest command (no GPU, no lease): write-once, lane refusal, verdict, exits
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import eng5st_common as cm
    drv = os.path.join(d, "drive")
    os.makedirs(drv, exist_ok=True)

    def stand_in(statuses):
        code = ("import sys; open(sys.argv[1], 'w').write(" + repr(junit_xml(statuses)) + ")")
        return code

    def step(name, arm, odir, want_rc, want_verdict, statuses=None, lane=True, marker=None):
        argv = [sys.executable, "-B", "-P", "-c", stand_in(statuses or allp), os.path.join(odir, "junit.xml")]
        if marker:  # the stand-in also writes the mutant marker, as a mutant module would
            argv[4] += "; open(%r, 'a').write(%r + '\\n')" % (os.path.join(odir, "mutant_marker.txt"), marker)
        rc = cm.run_arm(arm, out_dir=odir, lane_visible=lane, pytest_argv=argv, pkg_dir=drv, tmp=d)
        v = None
        vp = os.path.join(odir, "VERDICT.json")
        if os.path.exists(vp):
            v = json.load(open(vp))["verdict"]
        ok = rc == want_rc and v == want_verdict
        res.append({"case": "step body: " + name, "want": {"exit": want_rc, "verdict": want_verdict},
                    "got": {"exit": rc, "verdict": v}, "ok": ok})
        if not ok:
            fails.append("step body: " + name)

    step("S on all-pass junit", "S", os.path.join(drv, "S1"), 0, "PASS")
    step("write-once: the line directory exists", "S", os.path.join(drv, "S1"), 5, "PASS")
    step("the GPU lane is not visible -> refuses (7), writes no verdict", "S", os.path.join(drv, "S2"), 7, None,
         lane=False)
    step("B with every injected row killed", "B", os.path.join(drv, "B1"), 0, "DEFECT", statuses=allk)
    step("s03 killed on its own row", "s03", os.path.join(drv, "s03"), 0, "KILLED",
         statuses=with_(allp, zeros="killed"), marker="s03_zeros_status_dropped")
    step("s03 killed but no marker written", "s03", os.path.join(drv, "s03b"), 0, "KILLED_NO_MARKER",
         statuses=with_(allp, zeros="killed"))
    # the module-hash check (L103) on the real receipts, read-only
    for arm in cm.ARMS:
        ok, got, want = cm.module_check(arm)
        res.append({"case": f"L103 module check of arm {arm} against its receipt", "got": got, "want": want, "ok": ok})
        if not ok:
            fails.append("module check " + arm)
    out = {"checks": len(res), "failed": fails, "rows": res}
    os.makedirs(os.path.join(ROOT, "artifacts/eng5s/t"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "artifacts/eng5s/t/T3_VERDICT_DRIVE.json"), "w"), indent=1)
    print("T3 drive: checks", len(res), "failed", fails)
    return 0 if not fails else 1


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        sys.exit(selftest())
    elif sys.argv[1] == "junit":
        print(json.dumps(judge(sys.argv[2], sys.argv[3]), indent=1))
