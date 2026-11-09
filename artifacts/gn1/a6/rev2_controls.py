import os
os.environ["GN1_CONTROL"] = "1"  # A6 D1: CPU fixture root cannot accept.
from control_transport import adapt
#!/usr/bin/env python3
"""GN1-REV2 hostile runs through A2's REAL parent (run_gpu_line.py), child
bindings and scorer (score_gpu.py), byte-identical in the /tmp mirror.

Part A: REV1's own H1-H6 payloads (rev1/hostile_scorer.py STANDIN, verbatim
        logic), adapted only where A2 makes the old form impossible (label
        names split by D4; the attempt directory). Each adaptation is listed.
Part B: new cases with the REAL installed forge-gpu-run (no stand-in), its
        summary/window/lock files redirected into /tmp/gn1_rev2_forge, and a
        CPU sleeper as the child: rechecked-out (75), forge timeout (124),
        window-closed (124), outer bound (124), inner child timeout (124).
Part C: new evaluator cases with REV2's own stand-in (rev2_forge.py).
Every parent/scorer exit is recorded after the process exits (L45).

Prior art: test stubs / fault injection (Meszaros 2007); REV1's hostile
harness; mutation validity (DeMillo, Lipton, Sayward 1978). Ours: the cases.
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
MROOT = REAL
MA2 = OUT
BASE = MA2 / ("rev2_runs_" + time.strftime("%H%M%S"))
TOOLS = Path(os.environ["TMPDIR"]) / "rev2_tools"
FORGE = Path(os.environ["TMPDIR"]) / "rev2_forge"
REAL_FORGE = "forge-gpu-run"
PY = sys.executable
RESULTS = []
# Adaptation: A2's parent asserts TMPDIR.startswith("/tmp/gn1_a2_"), while this order allows writes only under
# /tmp/gn1_rev2_*. An existing A2 dir plus ".." satisfies the string prefix; every write resolves here.
TMPDIR_ADAPTED = os.environ["TMPDIR"]


def iso(t=None):
    return datetime.datetime.fromtimestamp(t or time.time(), datetime.timezone.utc).isoformat()


def base_env(**extra):
    e = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TMPDIR=TMPDIR_ADAPTED, CUDA_VISIBLE_DEVICES="",
             FORGE_GPU_SUMMARY=str(FORGE / "summary.log"), FORGE_GPU_WINDOW_FILE=str(FORGE / "GPU_WINDOW"),
             FORGE_GPU_LOCK=str(FORGE / "lock"), FORGE_GPU_POLL="0.2", FORGE_GPU_KILL_GRACE="1")
    e.pop("GN1_AUDIT", None)
    e.update(extra)
    # Safety: the real forge must never see the production lock or window.
    assert Path(e["FORGE_GPU_LOCK"]).is_relative_to(Path(os.environ["TMPDIR"])) and Path(e["FORGE_GPU_SUMMARY"]).is_relative_to(Path(os.environ["TMPDIR"]))
    return e


def run(args, env, timeout=120):
    args, env = adapt(args, env)

    with (OUT/"rev2_launches.jsonl").open("a") as log: log.write(json.dumps(dict(args=args,RUNS=env.get("RUNS")))+"\n")
    p = subprocess.run(["timeout", str(timeout)+"s", *args], cwd=MA2, env=env, capture_output=True, text=True, timeout=timeout)
    return dict(exit=p.returncode, stdout=p.stdout[-2500:], stderr=p.stderr[-2500:])


def start(name):
    runs = BASE / name
    r = run([PY, "-B", "start_gpu_run.py"], base_env(RUNS=str(runs)))
    assert r["exit"] == 0, r
    return runs


def parent(runs, label, gpu_run, tail, bound=None, expected=None, popen=False, **env):
    from_lines = json.loads((MA2 / "gpu_lines.json").read_text())
    line = next(r for r in from_lines if r["label"] == label)
    bound = line["outer"] if bound is None else bound
    ex = expected or ",".join(map(str, line["expected_child_exits"]))
    # D7: bind valid scaffolding to the sealed row; explicit hostile bounds stay hostile.
    gpu_run = list(gpu_run)
    tail = list(tail)
    if "--min" not in gpu_run and "--max" not in gpu_run:
        at = gpu_run.index("--")
        gpu_run[at:at] = ["--min", str(line["minimum"]), "--max", str(line["maximum"])]
    elif "hostile" in gpu_run:
        gpu_run[gpu_run.index("--min")+1] = str(line["minimum"])
        gpu_run[gpu_run.index("--max")+1] = str(line["maximum"])
    if tail and tail[0] == "timeout" and tail[1] == "10s":
        tail[1] = str(line["inner"]) + "s"
    args = [PY, "-B", "run_gpu_line.py", label, ex, str(bound), *gpu_run, *tail]
    e = base_env(RUNS=str(runs), **env)
    if popen:
        return subprocess.Popen(args, cwd=MA2, env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    r = run(args, e, timeout=bound + 60)
    return collect(runs, label, r)


def collect(runs, label, r):
    atts = sorted((runs / label).glob("attempt-*")) if (runs / label).exists() else []
    r["attempts"] = [a.name for a in atts]
    rec = atts[-1] / "parent_result.json" if atts else None
    r["parent"] = json.loads(rec.read_text()) if rec and rec.exists() else None
    if r["parent"]:
        r["parent"] = {k: r["parent"].get(k) for k in ("exit", "passed", "reasons", "actual_killers", "admission",
                                                       "forge_outcome", "timeout", "timeout_cause")}
    fl = atts[-1] / "forge.log" if atts else None
    r["forge_log_last"] = fl.read_text().splitlines()[-1].split("\t")[-2:] if fl and fl.exists() and fl.read_text() else None
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


def record(case, finding_ref, expected, observed_ok, **data):
    RESULTS.append(dict(case=case, ref=finding_ref, expectation=expected, outcome_as_expected=observed_ok, **data))
    print(case, "AS-EXPECTED" if observed_ok else "UNEXPECTED", flush=True)


# ---------------------------------------------------------------- Part A ---
REV1_STANDIN = (REAL / "artifacts/gn1/rev1/hostile_scorer.py").read_text().split("STANDIN = r'''")[1].split("'''")[0]


def part_a():
    TOOLS.mkdir(exist_ok=True)
    flat = TOOLS / "rev1_standin_flat.py"
    flat.write_text(REV1_STANDIN)
    # Adaptation 1 (only): write REV1's exact payload into A2's attempt dir.
    att = TOOLS / "rev1_standin_attempt.py"
    att.write_text(REV1_STANDIN.replace('out = Path(os.environ["RUNS"]) / os.environ["GN1_LINE"]',
                                        'out = Path(os.environ["GN1_ATTEMPT"])'))
    assert att.read_text() != flat.read_text()
    for variant, standin in (("flat", flat), ("attempt", att)):
        runs = start("A_" + variant)
        env = dict(HOSTILE_TOOL=str(MA2))
        g = [PY, str(standin), "--min", "1", "--max", "30", "--label", "hostile", "--", "timeout", "20s", "python3"]
        rows = {}
        # H4 first (REV1 used it to provide candidate_focus); REV1 label M07/M08/M09 -> A2 M07fast/M08fast/M09fast.
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
    # H1 / H1b exactly as REV1: forged flat parent receipts only, no START, no children.
    forged = BASE / "A_H1_forged"
    for line in json.loads((MA2 / "gpu_lines.json").read_text()):
        d = forged / line["label"]
        d.mkdir(parents=True)
        (d / "parent_result.json").write_text(json.dumps(dict(passed=True, timeout=False, audit=False,
                                                                actual_killers=["forged"])))
    s = score(forged)
    record("A_H1", "REV1-H1", "scorer exit 1, G5 not passed", s["exit"] == 1 and s["G5_passed"] is False, scorer=s)
    outside = Path(os.environ["TMPDIR"])/("outside_" + time.strftime("%H%M%S"))
    outside.mkdir(parents=True)
    s = run([PY, "-B", "score_gpu.py"], base_env(RUNS=str(outside)))
    record("A_H1b", "REV1-H1b", "scorer refuses root outside a2; writes nothing there",
           s["exit"] != 0 and not (outside / "G5_SUMMARY.json").exists(), scorer=s)


# ---------------------------------------------------------------- Part B ---
SLEEPER = """import os, sys, time
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
time.sleep(float(os.environ.get("REV2_SLEEP", "30")))
sys.exit(0)
"""


def summary(lines):
    (FORGE / "summary.log").write_text("".join("%s | %s\n" % (iso(t), ev) for t, ev in lines))


def real_forge(minimum, maximum, margin, label):
    return [REAL_FORGE, "--min", str(minimum), "--max", str(maximum), "--margin", str(margin), "--label", label, "--"]


def part_b():
    FORGE.mkdir(exist_ok=True)
    sleeper = TOOLS / "sleeper.py"
    sleeper.write_text(SLEEPER)
    tail = lambda arm, suite: [PY, str(sleeper), "run_gpu_tests.py", arm, suite]
    # B1: rechecked-out. Hold the (scratch) lock, change the window while forge waits, release.
    runs = start("B1_rechecked_out")
    now = time.time()
    summary([(now, "COOLDOWN 3600s")])
    lockfd = os.open(str(FORGE / "lock"), os.O_RDWR | os.O_CREAT, 0o666)
    fcntl.flock(lockfd, fcntl.LOCK_EX)
    p = parent(runs, "baseline_focus", real_forge(1, 60, 2, "rev2-B1"), tail("baseline", "focus"), bound=60, popen=True)
    time.sleep(2.0)
    summary([(now, "COOLDOWN 3600s"), (time.time(), "COOLDOWN 3500s")])
    fcntl.flock(lockfd, fcntl.LOCK_UN)
    os.close(lockfd)
    out, err = p.communicate(timeout=120)
    first = collect(runs, "baseline_focus", dict(exit=p.returncode, stdout=out[-2000:], stderr=err[-2000:]))
    # Retry the same label in the same root, window now steady (D7 says a 75 refusal may be retried).
    summary([(time.time(), "COOLDOWN 3600s")])
    second = parent(runs, "baseline_focus", real_forge(1, 60, 2, "rev2-B1r"), tail("baseline", "focus"), bound=60)
    sc = score(runs, "baseline_focus")
    ok = (first["forge_log_last"] == ["rechecked-out", "75"] and (first["parent"] or {}).get("admission") == "admitted"
          and second["exit"] != 0 and "label already has an admitted attempt" in second["stderr"])
    record("B1_rechecked_out_burns_label", "R1-F08/D7",
           "D7: a 75 lease refusal is 'not admitted' and the label may be retried",
           False if ok else None, defect_reproduced=ok, first=first, retry=second, scorer=sc)
    # B2: forge's own deadline (no-run state: summary absent), 124 'timeout'.
    os.replace(FORGE / "summary.log", FORGE / "summary.moved") if (FORGE / "summary.log").exists() else None
    runs = start("B2_forge_timeout")
    r = parent(runs, "candidate_focus", real_forge(1, 4, 1, "rev2-B2"), tail("candidate", "focus"), bound=60)
    record("B2_forge_timeout_124", "R1-F11/D10", "forge_outcome timeout; outer bound did NOT fire",
           (r["parent"] or {}).get("forge_outcome") == "timeout", run=r,
           outer_timeout_exit_flag=(r["parent"] or {}).get("timeout_cause"))
    # B3: window closes during the run -> 124 'window-closed'.
    runs = start("B3_window_closed")
    now = time.time()
    summary([(now, "COOLDOWN 3600s")])
    p = parent(runs, "candidate_focus", real_forge(1, 60, 1, "rev2-B3"), tail("candidate", "focus"), bound=60, popen=True)
    time.sleep(2.0)
    summary([(now, "COOLDOWN 3600s"), (time.time(), "COOLDOWN 1s")])
    out, err = p.communicate(timeout=120)
    r = collect(runs, "candidate_focus", dict(exit=p.returncode, stdout=out[-2000:], stderr=err[-2000:]))
    record("B3_window_closed_124", "R1-F11/D10", "forge_outcome window-closed; outer bound did NOT fire",
           (r["parent"] or {}).get("forge_outcome") == "window-closed", run=r,
           outer_timeout_exit_flag=(r["parent"] or {}).get("timeout_cause"))
    # B4: the parent's own outer bound fires (forge itself still within --max).
    os.replace(FORGE / "summary.log", FORGE / "summary.moved")
    runs = start("B4_outer_bound")
    r = parent(runs, "candidate_focus", real_forge(1, 60, 1, "rev2-B4"), tail("candidate", "focus"), bound=3)
    record("B4_outer_bound_124", "R1-F11/D10", "line rejected; cause recorded",
           r["exit"] == 1 and not (r["parent"] or {}).get("passed"), run=r,
           outer_timeout_exit_flag=(r["parent"] or {}).get("timeout_cause"))
    # B5: the child's own inner timeout fires (forge ran it to completion; exit 124).
    runs = start("B5_inner_timeout")
    r = parent(runs, "candidate_focus", real_forge(1, 60, 1, "rev2-B5"),
               ["timeout", "2s"] + tail("candidate", "focus"), bound=60)
    record("B5_inner_child_timeout_124", "R1-F11/D10", "forge_outcome ran with exit 124; outer bound did NOT fire",
           (r["parent"] or {}).get("forge_outcome") == "ran", run=r,
           outer_timeout_exit_flag=(r["parent"] or {}).get("timeout_cause"))


# ---------------------------------------------------------------- Part C ---
def part_c():
    forge = [PY, str(OUT / "rev2_forge.py"), "--"]
    tail = lambda line: ["timeout", "10s", "python3", "run_gpu_tests.py", line["arm"], line["suite"]]
    lines = {r["label"]: r for r in json.loads((MA2 / "gpu_lines.json").read_text())}
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
        record(name, "REV2-new", "rejected by: " + needle, ok, run=r, scorer=sc)
    # C7: bound interrupts the real parent after the CPU child's summary.
    # GNU timeout terminates the process group started by this probe; the
    # foreground subprocess returns before recovery/scoring. No orphan wait.
    for mode in ("normal", "hang_failure"):
        runs = start("C7_parent_interrupted_"+mode)
        line = lines["candidate_focus"]
        args = ["timeout", "--kill-after=1s", "1s", PY, "-B", "run_gpu_line.py", "candidate_focus", "0", "60",
                PY, str(OUT/"rev2_forge.py"), "--", *tail(line)]
        r = run(args, base_env(RUNS=str(runs),REV2_MODE="hang" if mode=="normal" else "hang_failure"), timeout=30)
        assert (runs/"candidate_focus/attempt-0001/child_result.json").exists(),r
        retry = parent(runs,"candidate_focus",forge,tail(line),REV2_MODE="normal")
        sc = score(runs,"candidate_focus")
        ok = retry["exit"] != 0 and "label already has an admitted attempt" in retry["stderr"] and sc["line"]["passed"] is False and "incomplete" in str(sc)
        record("C7_child_present_"+mode,"R2-F07","evaluate incomplete child; no retry past a child result",ok,interrupted=r,retry=retry,scorer=sc)
    runs=start("C7_no_child")
    (runs/"candidate_focus/attempt-0001").mkdir(parents=True)
    retry=parent(runs,"candidate_focus",forge,tail(lines["candidate_focus"]),REV2_MODE="normal")
    sc=score(runs,"candidate_focus")
    record("C7_child_never_started","R2-F07","incomplete not admitted; retry allowed",retry["exit"]==0 and sc["line"]["passed"] and len(sc["line"]["attempts"])==2,retry=retry,scorer=sc)
    # C8: honest positive: a full run of every label with the REV2 stand-in passes the real scorer.
    runs = start("C8_full_positive")
    order = ["baseline_focus", "candidate_focus", "baseline_regression", "candidate_regression"] + \
            [l for l in lines if l not in ("baseline_focus", "candidate_focus", "baseline_regression", "candidate_regression")]
    bad = [l for l in order if parent(runs, l, forge, tail(lines[l]), REV2_MODE="normal")["exit"] != 0]
    sc = score(runs)
    record("C8_full_positive", "REV2 sensitivity", "every label admitted and G5 passed", not bad and sc["exit"] == 1 and all(v["passed"] for v in json.loads((runs/"G5_SUMMARY.json").read_text())["lines"].values()),
           failed_labels=bad, scorer=dict(exit=sc["exit"], G5_passed=sc["G5_passed"], errors=sc["errors"]))


# ---------------------------------------------------------------- Part D ---
def part_d():
    """REV1's H2/H2c/H3/H3b/H4 payloads WITH A2's bindings (so the targeted rule is what
    decides), REV1's H5 through the REAL forge (no-window 75, then retry), and new
    kill-rule cases."""
    forge = [PY, str(OUT / "rev2_forge.py"), "--"]
    lines = {r["label"]: r for r in json.loads((MA2 / "gpu_lines.json").read_text())}
    tail = lambda line: ["timeout", "10s", "python3", "run_gpu_tests.py", line["arm"], line["suite"]]
    cases = (
        ("D_H2_bound", "M07fast", "rev1_h2", "device-pattern failure", True),
        ("D_H2c_bound", "M08fast", "rev1_h2c", "device-pattern failure", True),
        ("D_H3_bound", "M09fast", "rev1_h3", "arm/suite binding", True),
        ("D_H3b_bound", "M11", "rev1_h3b", "expected exits", True),
        ("D_kill_plus_teardown_device", "M01", "kill_plus_teardown_device", "device-pattern failure", True),
        ("D_honest_kill_text_mentions_pattern", "M01", "kill_device_word_in_assert", "device-pattern failure", True),
    )
    for name, label, mode, needle, expect_reject in cases:
        runs = start(name)
        assert parent(runs, "candidate_focus", forge, tail(lines["candidate_focus"]), REV2_MODE="normal")["exit"] == 0
        r = parent(runs, label, forge, tail(lines[label]), REV2_MODE=mode)
        sc = score(runs, label)
        rej = r["exit"] == 1 and any(needle in x for x in (r["parent"] or {}).get("reasons", [])) and sc["exit"] == 1
        record(name, "R1-F01/F02", ("rejected by: " if expect_reject else "admitted; ") + needle, rej, run=r, scorer=sc)
    # H4 bound: an all-pass candidate summary with a foreign failing JUnit testcase (REV1 'x').
    runs = start("D_H4_bound")
    att_hook = TOOLS / "h4_junit_inject.py"
    att_hook.write_text(
        "import os, runpy, sys\n"
        "sys.argv = [sys.argv[0]] + sys.argv[1:]\n"
        "try:\n    runpy.run_path(%r, run_name='__main__')\n"
        "except SystemExit as e:\n    rc = e.code\n"
        "p = os.path.join(os.environ['GN1_ATTEMPT'], 'junit.xml')\n"
        "s = open(p).read().replace('</testsuite>', '<testcase name=\"x\"><failure message=\"real failure\"/></testcase></testsuite>')\n"
        "open(p, 'w').write(s)\n"
        "sys.exit(rc)\n" % str(OUT / "rev2_forge.py"))
    r = parent(runs, "candidate_focus", [PY, str(att_hook), "--"], tail(lines["candidate_focus"]), REV2_MODE="normal")
    record("D_H4_bound", "R1-F02", "rejected by: unregistered JUnit ID",
           r["exit"] == 1 and any("unregistered JUnit ID" in x for x in (r["parent"] or {}).get("reasons", [])), run=r)
    # H5 through the REAL forge: 'busy' window -> no-window 75 (not admitted); retry admitted and valid.
    runs = start("D_H5_real_forge_no_window")
    summary([(time.time(), "RUN7 CYCLE 3 TRAINING")])  # a non-halt, non-cooldown event -> busy -> no-window
    child = [PY, str(OUT / "rev2_forge.py"), "run_gpu_tests.py", "baseline", "focus"]
    first = parent(runs, "baseline_focus", real_forge(1, 60, 2, "rev2-D5a"), child, bound=60,
                   REV2_MODE="normal", REV2_NO_LOG="1")
    os.replace(FORGE / "summary.log", FORGE / "summary.moved")  # no-run state: the window is open
    second = parent(runs, "baseline_focus", real_forge(250, 300, 2, "rev2-D5b"), ["timeout", "240s", *child], bound=330,
                    REV2_MODE="normal", REV2_NO_LOG="1")
    sc = score(runs, "baseline_focus")
    ok = ((first["parent"] or {}).get("admission") == "not_admitted" and first["forge_log_last"] == ["no-window", "75"]
          and second["exit"] == 0 and (sc.get("line") or {}).get("passed") is True
          and len((sc.get("line") or {}).get("attempts") or []) == 2)
    record("D_H5_real_forge_no_window_then_retry", "R1-F08/D7", "not admitted, retry admitted and selected", ok,
           first=first, second=second, scorer=sc)


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    BASE.mkdir(parents=True)
    which = sys.argv[1:] or ["a", "c", "d"]
    for part in which:
        {"a": part_a, "b": part_b, "c": part_c, "d": part_d}[part]()
    name = "hostile_rev2_" + "".join(which) + ".json"
    (OUT / name).write_text(json.dumps(dict(base=str(BASE), real_forge=REAL_FORGE, results=RESULTS), indent=2) + "\n")
    print(json.dumps([(r["case"], r["outcome_as_expected"]) for r in RESULTS]))


if __name__ == "__main__":
    main()
