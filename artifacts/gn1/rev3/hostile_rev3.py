#!/usr/bin/env python3
"""GN1-REV3 hostile runs through A3's REAL parent (run_gpu_line.py via
parent_wrap.py, one disclosed guard-prefix substitution), A3's real child
bindings and A3's REAL scorer (score_gpu.py), byte-identical in the /tmp mirror
built by mirror_regen.py.

Part A: REV1's H1-H6 payloads exactly as REV2 ran them (REV1 stand-in text).
Part B: REV2's B1-B5 with the REAL installed forge-gpu-run (summary, window and
        lock redirected to /tmp/gn1_rev3_forge; CPU sleeper child), plus new
        cases B6 (outer bound fires while the runner waits for the lock),
        B7/B7b (outer bound fires inside the runner's own kill grace after its
        deadline) and B8 (outer bound fires inside the grace after window-closed).
Part C: REV2's C1-C7 and C8 with REV3's stand-in (rev3_forge.py), plus C7b
        (child started, wrote invocation.json, parent killed before any result).
Part D: REV2's part D (REV1 payloads with bindings, real-forge H5).
Every parent/scorer exit is recorded after the process exits (L45).

Prior art: test stubs / fault injection (Meszaros 2007); REV1's and REV2's
harnesses (2026); mutation validity (DeMillo, Lipton, Sayward 1978). Ours: B6-B8, C7b.
"""
import datetime
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
REAL = OUT.parents[2]
MROOT = Path("/tmp/gn1_rev3_m/W")
MA3 = MROOT / "artifacts/gn1/a3"
BASE = MA3 / ("rev3_runs_" + time.strftime("%H%M%S"))
TOOLS = Path("/tmp/gn1_rev3_tools")
FORGE = Path("/tmp/gn1_rev3_forge")
REAL_FORGE = subprocess.check_output(["bash", "-lc", "command -v forge-gpu-run"], text=True).strip()
PY = sys.executable
RESULTS = []
TMPDIR = "/tmp/gn1_rev3_tmp"
WRAP = str(OUT / "parent_wrap.py")


def iso(t=None):
    return datetime.datetime.fromtimestamp(t or time.time(), datetime.timezone.utc).isoformat()


def base_env(**extra):
    e = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TMPDIR=TMPDIR, CUDA_VISIBLE_DEVICES="", REV3_MA3=str(MA3),
             FORGE_GPU_SUMMARY=str(FORGE / "summary.log"), FORGE_GPU_WINDOW_FILE=str(FORGE / "GPU_WINDOW"),
             FORGE_GPU_LOCK=str(FORGE / "lock"), FORGE_GPU_POLL="0.2", FORGE_GPU_KILL_GRACE="1")
    e.pop("GN1_AUDIT", None)
    e.update(extra)
    assert e["FORGE_GPU_LOCK"].startswith("/tmp/gn1_rev3_") and e["FORGE_GPU_SUMMARY"].startswith("/tmp/gn1_rev3_")
    assert e["FORGE_GPU_WINDOW_FILE"].startswith("/tmp/gn1_rev3_") and e["CUDA_VISIBLE_DEVICES"] == ""
    return e


def run(args, env, timeout=120):
    p = subprocess.run(args, cwd=MA3, env=env, capture_output=True, text=True, timeout=timeout)
    return dict(exit=p.returncode, stdout=p.stdout[-2500:], stderr=p.stderr[-2500:])


def start(name):
    runs = BASE / name
    r = run([PY, "-B", "start_gpu_run.py"], base_env(RUNS=str(runs)))
    assert r["exit"] == 0, r
    return runs


def parent(runs, label, gpu_run, tail, bound=30, expected=None, popen=False, **env):
    line = next(r for r in json.loads((MA3 / "gpu_lines.json").read_text()) if r["label"] == label)
    ex = expected or ",".join(map(str, line["expected_child_exits"]))
    args = [PY, "-B", WRAP, label, ex, str(bound), *gpu_run, *tail]
    e = base_env(RUNS=str(runs), **env)
    if popen:
        return subprocess.Popen(args, cwd=MA3, env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    r = run(args, e, timeout=bound + 60)
    return collect(runs, label, r)


def collect(runs, label, r):
    atts = sorted((runs / label).glob("attempt-*")) if (runs / label).exists() else []
    r["attempts"] = [a.name for a in atts]
    last = atts[-1] if atts else None
    rec = last / "parent_result.json" if last else None
    r["parent"] = json.loads(rec.read_text()) if rec and rec.exists() else None
    if r["parent"]:
        r["parent"] = {k: r["parent"].get(k) for k in ("exit", "passed", "reasons", "actual_killers", "admission",
                                                       "forge_outcome", "timeout", "timeout_cause")}
    fl = last / "forge.log" if last else None
    r["forge_log_last"] = fl.read_text().splitlines()[-1].split("\t")[-2:] if fl and fl.exists() and fl.read_text() else None
    r["last_attempt_files"] = sorted(p.name for p in last.iterdir()) if last else []
    return r


def score(runs, label=None):
    r = run([PY, "-B", "score_gpu.py"], base_env(RUNS=str(runs)))
    p = runs / "G5_SUMMARY.json"
    s = json.loads(p.read_text()) if p.exists() else None
    r["G5_passed"] = s["passed"] if s else None
    r["errors"] = s["errors"] if s else None
    if s and label:
        row = s["lines"].get(label, {})
        r["line"] = {k: row.get(k) for k in ("passed", "reasons", "attempts", "selected_attempt")}
    r.pop("stdout")
    return r


def record(case, ref, expected, ok, **data):
    RESULTS.append(dict(case=case, ref=ref, expectation=expected, outcome_as_expected=ok, **data))
    print(case, "AS-EXPECTED" if ok else "UNEXPECTED", flush=True)


# ---------------------------------------------------------------- Part A ---
REV1_STANDIN = (REAL / "artifacts/gn1/rev1/hostile_scorer.py").read_text().split("STANDIN = r'''")[1].split("'''")[0]


def part_a():
    TOOLS.mkdir(exist_ok=True)
    flat = TOOLS / "rev1_standin_flat.py"
    flat.write_text(REV1_STANDIN)
    att = TOOLS / "rev1_standin_attempt.py"
    att.write_text(REV1_STANDIN.replace('out = Path(os.environ["RUNS"]) / os.environ["GN1_LINE"]',
                                        'out = Path(os.environ["GN1_ATTEMPT"])'))
    for variant, standin in (("flat", flat), ("attempt", att)):
        runs = start("A_" + variant)
        env = dict(HOSTILE_TOOL=str(MA3))
        g = [PY, str(standin), "--min", "1", "--max", "30", "--label", "hostile", "--", "timeout", "20s", "python3"]
        rows = {}
        rows["H4"] = parent(runs, "candidate_focus", g, ["run_gpu_tests.py", "candidate", "focus"],
                            HOSTILE_MODE="junit_contradiction_pass", **env)
        rows["H2"] = parent(runs, "M07fast", g, ["run_gpu_tests.py", "M07fast", "mutation"],
                            HOSTILE_MODE="device_error_kill", **env)
        rows["H3"] = parent(runs, "M09fast", g, ["run_gpu_tests.py", "M09fast", "mutation"],
                            HOSTILE_MODE="foreign_arm_kill", **env)
        rows["H3b"] = parent(runs, "M11", g, ["run_gpu_tests.py", "M11", "mutation"],
                             HOSTILE_MODE="suite_swap_pass", **env)
        rows["H2c"] = parent(runs, "M08fast", g, ["run_gpu_tests.py", "M08fast", "mutation"],
                             HOSTILE_MODE="device_error_control", **env)
        runs2 = start("A_" + variant + "_hang")
        rows["H6"] = parent(runs2, "candidate_focus", g, ["run_gpu_tests.py", "candidate", "focus"], bound=5,
                            HOSTILE_MODE="hang_after_result", **env)
        rows["H5_first"] = parent(runs, "baseline_focus", g, ["run_gpu_tests.py", "baseline", "focus"],
                                  HOSTILE_MODE="no_window", **env)
        rows["H5_retry"] = parent(runs, "baseline_focus", g, ["run_gpu_tests.py", "baseline", "focus"],
                                  HOSTILE_MODE="no_window", **env)
        sc = score(runs)
        for h, r in rows.items():
            ok = r["exit"] != 0 and not (r["parent"] or {}).get("passed")
            record("A_%s_%s" % (h, variant), "REV1-" + h.split("_")[0], "rejected (no pass, no kill)", ok, run=r)
        record("A_scorer_%s" % variant, "REV1-H1..H6", "G5 not passed", sc["exit"] == 1 and not sc["G5_passed"], scorer=sc)
    forged = BASE / "A_H1_forged"
    for line in json.loads((MA3 / "gpu_lines.json").read_text()):
        d = forged / line["label"]
        d.mkdir(parents=True)
        (d / "parent_result.json").write_text(json.dumps(dict(passed=True, timeout=False, audit=False,
                                                                actual_killers=["forged"])))
    s = score(forged)
    record("A_H1", "REV1-H1", "scorer exit 1, G5 not passed", s["exit"] == 1 and s["G5_passed"] is False, scorer=s)
    outside = Path(TMPDIR) / ("outside_" + time.strftime("%H%M%S"))
    outside.mkdir(parents=True)
    s = run([PY, "-B", "score_gpu.py"], base_env(RUNS=str(outside)))
    record("A_H1b", "REV1-H1b", "scorer refuses root outside a3; writes nothing there",
           s["exit"] != 0 and not (outside / "G5_SUMMARY.json").exists(), scorer=s)


# ---------------------------------------------------------------- Part B ---
SLEEPER = """import os, sys, time
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
time.sleep(float(os.environ.get("REV2_SLEEP", "30")))
sys.exit(0)
"""


def summary(rows):
    (FORGE / "summary.log").write_text("".join("%s | %s\n" % (iso(t), ev) for t, ev in rows))


def no_summary():
    if (FORGE / "summary.log").exists():
        os.replace(FORGE / "summary.log", FORGE / ("summary.moved." + str(time.time())))


def real_forge(minimum, maximum, margin, label):
    return [REAL_FORGE, "--min", str(minimum), "--max", str(maximum), "--margin", str(margin), "--label", label, "--"]


def part_b():
    FORGE.mkdir(exist_ok=True)
    TOOLS.mkdir(exist_ok=True)
    sleeper = TOOLS / "sleeper.py"
    sleeper.write_text(SLEEPER)
    tail = lambda arm, suite: [PY, str(sleeper), "run_gpu_tests.py", arm, suite]
    # B1 rechecked-out (REV2 exact), then retry in the same root.
    runs = start("B1_rechecked_out")
    now = time.time()
    summary([(now, "COOLDOWN 3600s")])
    lockfd = os.open(str(FORGE / "lock"), os.O_RDWR | os.O_CREAT, 0o666)
    fcntl.flock(lockfd, fcntl.LOCK_EX)
    p = parent(runs, "baseline_focus", real_forge(1, 60, 2, "rev3-B1"), tail("baseline", "focus"), bound=60, popen=True)
    time.sleep(2.0)
    summary([(now, "COOLDOWN 3600s"), (time.time(), "COOLDOWN 3500s")])
    fcntl.flock(lockfd, fcntl.LOCK_UN)
    os.close(lockfd)
    out, err = p.communicate(timeout=120)
    first = collect(runs, "baseline_focus", dict(exit=p.returncode, stdout=out[-2000:], stderr=err[-2000:]))
    summary([(time.time(), "COOLDOWN 3600s")])
    second = parent(runs, "baseline_focus", real_forge(1, 60, 2, "rev3-B1r"), tail("baseline", "focus"), bound=60,
                    REV2_SLEEP="0.5")
    sc = score(runs, "baseline_focus")
    ok = (first["forge_log_last"] == ["rechecked-out", "75"] and (first["parent"] or {}).get("admission") == "not_admitted"
          and len(second["attempts"]) == 2 and (second["parent"] or {}).get("admission") == "admitted")
    record("B1_rechecked_out_retry", "R2-F01", "rechecked-out 75 not admitted; same-root retry admitted", ok,
           first=first, retry=second, scorer=sc)
    # B2 forge's own deadline (no run state), 124 timeout -> runner-deadline.
    no_summary()
    runs = start("B2_forge_timeout")
    r = parent(runs, "candidate_focus", real_forge(1, 4, 1, "rev3-B2"), tail("candidate", "focus"), bound=60)
    record("B2_runner_deadline", "R2-F04", "forge timeout 124; cause runner-deadline",
           (r["parent"] or {}).get("forge_outcome") == "timeout" and (r["parent"] or {}).get("timeout_cause") == "runner-deadline", run=r)
    # B3 window closes during the run -> window-closed.
    runs = start("B3_window_closed")
    now = time.time()
    summary([(now, "COOLDOWN 3600s")])
    p = parent(runs, "candidate_focus", real_forge(1, 60, 1, "rev3-B3"), tail("candidate", "focus"), bound=60, popen=True)
    time.sleep(2.0)
    summary([(now, "COOLDOWN 3600s"), (time.time(), "COOLDOWN 1s")])
    out, err = p.communicate(timeout=120)
    r = collect(runs, "candidate_focus", dict(exit=p.returncode, stdout=out[-2000:], stderr=err[-2000:]))
    record("B3_window_closed", "R2-F04", "forge window-closed 124; cause window-closed",
           (r["parent"] or {}).get("forge_outcome") == "window-closed" and (r["parent"] or {}).get("timeout_cause") == "window-closed", run=r)
    # B4 the parent's outer bound fires (forge within --max).
    no_summary()
    runs = start("B4_outer_bound")
    r = parent(runs, "candidate_focus", real_forge(1, 60, 1, "rev3-B4"), tail("candidate", "focus"), bound=3)
    record("B4_outer_timeout", "R2-F04", "line rejected; cause outer-timeout",
           r["exit"] == 1 and (r["parent"] or {}).get("timeout_cause") == "outer-timeout", run=r)
    # B5 child's own inner timeout 124 (and 137 via -s KILL).
    runs = start("B5_inner_timeout")
    r = parent(runs, "candidate_focus", real_forge(1, 60, 1, "rev3-B5"), ["timeout", "2s"] + tail("candidate", "focus"), bound=60)
    record("B5_inner_timeout124", "R2-F04", "forge ran 124; cause inner-timeout",
           (r["parent"] or {}).get("forge_outcome") == "ran" and (r["parent"] or {}).get("timeout_cause") == "inner-timeout", run=r)
    runs = start("B5_inner_timeout137")
    r = parent(runs, "candidate_focus", real_forge(1, 60, 1, "rev3-B5k"), ["timeout", "-s", "KILL", "2s"] + tail("candidate", "focus"), bound=60)
    record("B5_inner_timeout137", "R2-F04", "forge ran 137; cause inner-timeout",
           (r["parent"] or {}).get("forge_outcome") == "ran" and (r["parent"] or {}).get("timeout_cause") == "inner-timeout", run=r)
    # ---- new REV3 cases ----
    # B6: outer bound fires while the runner WAITS FOR THE LOCK (child never started; runner dies without a log row).
    runs = start("B6_outer_timeout_in_lock_wait")
    summary([(time.time(), "COOLDOWN 3600s")])
    lockfd = os.open(str(FORGE / "lock"), os.O_RDWR | os.O_CREAT, 0o666)
    fcntl.flock(lockfd, fcntl.LOCK_EX)
    first = parent(runs, "candidate_focus", real_forge(1, 60, 2, "rev3-B6"), tail("candidate", "focus"), bound=4)
    fcntl.flock(lockfd, fcntl.LOCK_UN)
    os.close(lockfd)
    retry = parent(runs, "candidate_focus", real_forge(1, 60, 2, "rev3-B6r"), tail("candidate", "focus"), bound=60,
                   REV2_SLEEP="0.5")
    sc = score(runs, "candidate_focus")
    child_started = "invocation.json" in first["last_attempt_files"]
    reproduced = ((first["parent"] or {}).get("admission") == "admitted" and not child_started
                  and retry["exit"] != 0 and "label already has an admitted attempt" in retry["stderr"])
    record("B6_outer_timeout_in_lock_wait", "R2-F01 class (new)",
           "child never started (no runner row, no invocation) -> D4: not admitted, retry allowed",
           not reproduced, defect_reproduced=reproduced, child_started=child_started, first=first, retry=retry, scorer=sc)
    # B7: runner deadline, then the outer bound fires inside the runner's kill grace -> runner logs timeout/143, exit 124.
    no_summary()
    runs = start("B7_outer_in_runner_grace")
    first = parent(runs, "candidate_focus", real_forge(1, 6, 3, "rev3-B7"), tail("candidate", "focus"), bound=4,
                   FORGE_GPU_KILL_GRACE="3")
    retry = parent(runs, "candidate_focus", real_forge(1, 60, 1, "rev3-B7r"), tail("candidate", "focus"), bound=60,
                   REV2_SLEEP="0.5")
    sc = score(runs, "candidate_focus")
    crashed = first["parent"] is None and "cannot attribute timeout from runner log" in first["stderr"]
    record("B7_outer_in_runner_grace", "R2-F04/F07 class (new)",
           "parent records a receipt with a timeout cause; started child is not retried",
           not crashed and retry["exit"] != 0, parent_crashed=crashed, first=first, retry=retry, scorer=sc)
    # B7b: same, child (REV3 stand-in) wrote child_result.json before hanging -> label bricked?
    runs = start("B7b_outer_in_runner_grace_child_result")
    child = [PY, str(OUT / "rev3_forge.py"), "run_gpu_tests.py", "candidate", "focus"]
    first = parent(runs, "candidate_focus", real_forge(1, 6, 3, "rev3-B7b"), child, bound=4,
                   FORGE_GPU_KILL_GRACE="3", REV2_MODE="hang_after_result", REV2_NO_LOG="1", REV3_SLEEP="30")
    retry = parent(runs, "candidate_focus", real_forge(1, 60, 1, "rev3-B7br"), child, bound=60,
                   REV2_MODE="normal", REV2_NO_LOG="1")
    sc = score(runs, "candidate_focus")
    crashed = first["parent"] is None and "cannot attribute timeout from runner log" in first["stderr"]
    retry_crashed = "cannot attribute timeout from runner log" in retry["stderr"]
    record("B7b_outer_in_runner_grace_child_result", "R2-F04/F07 class (new)",
           "parent records a receipt; later parents refuse cleanly (not crash)",
           not crashed and not retry_crashed, parent_crashed=crashed, retry_crashed=retry_crashed,
           first=first, retry=retry, scorer=sc)
    # B8: window closes, then the outer bound fires inside the runner's grace -> window-closed/143, exit 124.
    runs = start("B8_outer_in_window_closed_grace")
    now = time.time()
    summary([(now, "COOLDOWN 3600s")])
    p = parent(runs, "candidate_focus", real_forge(1, 60, 3, "rev3-B8"), tail("candidate", "focus"), bound=3, popen=True,
               FORGE_GPU_KILL_GRACE="3")
    time.sleep(1.0)
    summary([(now, "COOLDOWN 3600s"), (time.time(), "COOLDOWN 1s")])
    out, err = p.communicate(timeout=120)
    first = collect(runs, "candidate_focus", dict(exit=p.returncode, stdout=out[-2000:], stderr=err[-2000:]))
    crashed = first["parent"] is None and "cannot attribute timeout from runner log" in first["stderr"]
    sc = score(runs, "candidate_focus")
    record("B8_outer_in_window_closed_grace", "R2-F04/F07 class (new)", "parent records a receipt with a timeout cause",
           not crashed, parent_crashed=crashed, first=first, scorer=sc)
    no_summary()


# ---------------------------------------------------------------- Part C ---
def part_c():
    forge = [PY, str(OUT / "rev3_forge.py"), "--"]
    tail = lambda line: ["timeout", "10s", "python3", "run_gpu_tests.py", line["arm"], line["suite"]]
    lines = {r["label"]: r for r in json.loads((MA3 / "gpu_lines.json").read_text())}
    cases = (
        ("C1_equivalent_V3_fails", "V3", "v3_failure", "equivalent mutant did not pass"),
        ("C2_forge_log_exit_mismatch", "M01", "log_mismatch", "log-exit-mismatch"),
        ("C3_census_missing_id", "M01", "census_drop", "test-ID census mismatch"),
        ("C4_unregistered_skip", "M01", "skip_one", "unregistered skip"),
        ("C5_baseline_extra_failure", "baseline_focus", "baseline_extra", "baseline failures differ"),
        ("C6_exit0_with_failure_regression", "candidate_regression", "exit0_with_failure", "pytest exit inconsistent"),
    )
    for name, label, mode, needle in cases:
        runs = start(name)
        if lines[label]["suite"] == "mutation":
            assert parent(runs, "candidate_focus", forge, tail(lines["candidate_focus"]), REV2_MODE="normal")["exit"] == 0
        if label == "candidate_regression":
            assert parent(runs, "baseline_regression", forge, tail(lines["baseline_regression"]), REV2_MODE="normal")["exit"] == 0
        r = parent(runs, label, forge, tail(lines[label]), REV2_MODE=mode)
        sc = score(runs, label)
        ok = r["exit"] == 1 and any(needle in x for x in (r["parent"] or {}).get("reasons", [])) and sc["exit"] == 1
        record(name, "REV2-C", "rejected by: " + needle, ok, run=r, scorer=sc)
    # C7 (REV2): parent killed after the child wrote child_result.json (child hangs).
    runs = start("C7_parent_killed_child_result")
    label = "candidate_focus"
    p = parent(runs, label, forge, ["timeout", "40s", "python3", "run_gpu_tests.py", "candidate", "focus"], bound=60,
               popen=True, REV2_MODE="hang", REV3_SLEEP="8")
    deadline = time.time() + 20
    while time.time() < deadline and not list((runs / label).glob("attempt-0001/child_result.json")):
        time.sleep(0.2)
    p.kill()  # REV3 started this process.
    p.communicate(timeout=30)
    time.sleep(9.0)  # let the orphaned stand-in finish its bounded sleep on its own
    retry = parent(runs, label, forge, tail(lines[label]), REV2_MODE="normal")
    sc = score(runs, label)
    ok = (retry["exit"] != 0 and "label already has an admitted attempt" in retry["stderr"]
          and not (sc.get("line") or {}).get("passed") and "incomplete" in json.dumps(sc.get("line")))
    record("C7_child_result_present", "R2-F07", "attempt evaluated, never retried; scorer fails it as incomplete", ok,
           killed_parent_exit=p.returncode, retry=retry, scorer=sc)
    # C7 never-started (REV2 class): attempt dir only.
    runs = start("C7_child_never_started")
    (runs / label / "attempt-0001").mkdir(parents=True)
    retry = parent(runs, label, forge, tail(lines[label]), REV2_MODE="normal")
    sc = score(runs, label)
    ok = retry["exit"] == 0 and (sc.get("line") or {}).get("passed") is True
    record("C7_child_never_started", "R2-F07", "not admitted; retry admitted and selected", ok, retry=retry, scorer=sc)
    # C7b (new): child STARTED (invocation.json written, as the real child does before pytest), parent killed,
    # no child_result.json ever written. D4 retries only when the child never started.
    runs = start("C7b_child_started_no_result")
    p = parent(runs, label, forge, ["timeout", "40s", "python3", "run_gpu_tests.py", "candidate", "focus"], bound=60,
               popen=True, REV2_MODE="started_no_result", REV3_SLEEP="6")
    deadline = time.time() + 20
    while time.time() < deadline and not list((runs / label).glob("attempt-0001/invocation.json")):
        time.sleep(0.2)
    p.kill()  # REV3 started this process.
    p.communicate(timeout=30)
    time.sleep(7.0)
    files = sorted(x.name for x in (runs / label / "attempt-0001").iterdir())
    retry = parent(runs, label, forge, tail(lines[label]), REV2_MODE="normal")
    sc = score(runs, label)
    reproduced = "invocation.json" in files and retry["exit"] == 0 and (sc.get("line") or {}).get("passed") is True
    record("C7b_child_started_no_result", "R2-F07 / D4 (new)",
           "child started -> not 'never started'; D4 forbids a retry that can pass", not reproduced,
           defect_reproduced=reproduced, first_attempt_files=files, killed_parent_exit=p.returncode, retry=retry, scorer=sc)
    # C8 honest positive: a full run of every label.
    runs = start("C8_full_positive")
    order = ["baseline_focus", "candidate_focus", "baseline_regression", "candidate_regression"] + \
            [l for l in lines if l not in ("baseline_focus", "candidate_focus", "baseline_regression", "candidate_regression")]
    bad = [l for l in order if parent(runs, l, forge, tail(lines[l]), REV2_MODE="normal")["exit"] != 0]
    sc = score(runs)
    record("C8_full_positive", "REV2 sensitivity", "every label admitted and G5 passed", not bad and sc["exit"] == 0,
           failed_labels=bad, labels=len(order), scorer=dict(exit=sc["exit"], G5_passed=sc["G5_passed"], errors=sc["errors"]))


# ---------------------------------------------------------------- Part D ---
def part_d():
    forge = [PY, str(OUT / "rev3_forge.py"), "--"]
    lines = {r["label"]: r for r in json.loads((MA3 / "gpu_lines.json").read_text())}
    tail = lambda line: ["timeout", "10s", "python3", "run_gpu_tests.py", line["arm"], line["suite"]]
    cases = (
        ("D_H2_bound", "M07fast", "rev1_h2", "device-pattern failure"),
        ("D_H2c_bound", "M08fast", "rev1_h2c", "device-pattern failure"),
        ("D_H3_bound", "M09fast", "rev1_h3", "arm/suite binding"),
        ("D_H3b_bound", "M11", "rev1_h3b", "expected exits"),
        ("D_kill_plus_teardown_device", "M01", "kill_plus_teardown_device", "device-pattern failure"),
        ("D_honest_kill_text_mentions_pattern", "M01", "kill_device_word_in_assert", "device-pattern failure"),
    )
    for name, label, mode, needle in cases:
        runs = start(name)
        assert parent(runs, "candidate_focus", forge, tail(lines["candidate_focus"]), REV2_MODE="normal")["exit"] == 0
        r = parent(runs, label, forge, tail(lines[label]), REV2_MODE=mode)
        sc = score(runs, label)
        rej = r["exit"] == 1 and any(needle in x for x in (r["parent"] or {}).get("reasons", [])) and sc["exit"] == 1
        record(name, "REV2-D", "rejected by: " + needle, rej, run=r, scorer=sc)
    FORGE.mkdir(exist_ok=True)
    runs = start("D_H5_real_forge_no_window")
    summary([(time.time(), "RUN7 CYCLE 3 TRAINING")])
    child = [PY, str(OUT / "rev3_forge.py"), "run_gpu_tests.py", "baseline", "focus"]
    first = parent(runs, "baseline_focus", real_forge(1, 60, 2, "rev3-D5a"), child, bound=60,
                   REV2_MODE="normal", REV2_NO_LOG="1")
    no_summary()
    second = parent(runs, "baseline_focus", real_forge(1, 60, 2, "rev3-D5b"), child, bound=60,
                    REV2_MODE="normal", REV2_NO_LOG="1")
    sc = score(runs, "baseline_focus")
    ok = ((first["parent"] or {}).get("admission") == "not_admitted" and first["forge_log_last"] == ["no-window", "75"]
          and second["exit"] == 0 and (sc.get("line") or {}).get("passed") is True
          and len((sc.get("line") or {}).get("attempts") or []) == 2)
    record("D_H5_real_forge_no_window_then_retry", "R1-F08", "not admitted, retry admitted and selected", ok,
           first=first, second=second, scorer=sc)


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    BASE.mkdir(parents=True)
    which = sys.argv[1:] or ["a", "b", "c", "d"]
    for part in which:
        {"a": part_a, "b": part_b, "c": part_c, "d": part_d}[part]()
    name = "hostile_rev3_" + "".join(which) + ".json"
    (OUT / name).write_text(json.dumps(dict(base=str(BASE), real_forge=REAL_FORGE, results=RESULTS), indent=2) + "\n")
    print(json.dumps([(r["case"], r["outcome_as_expected"]) for r in RESULTS]))


if __name__ == "__main__":
    main()
