#!/usr/bin/env python3
"""GN1-REV4 hostile runs through A4's REAL parent (run_gpu_line.py via
parent_wrap4.py, one disclosed guard-prefix substitution), A4's REAL scorer
(score_gpu.py) and runtime_evidence.py, byte-identical in the /tmp mirror built
by mirror_regen_a4.py (seal verified on every line).

Part B: REV3's B6, B7, B7b, B8 with the REAL installed forge-gpu-run (summary,
        window and lock redirected to /tmp/gn1_rev4_forge; CPU children), plus
        new N1 (two consecutive lock-wait kills, then a pass) and N2 (staged
        default kill grace = outer --kill-after, TERM-ignoring child).
Part C: C7b, E5, E5b; new N3 (start-evidence classes), N4 (truncated
        parent_result.json), N5 (no-lock row with an outer-bound exit).
Part D: D5 runtime controls, D6 live check (premise false), D7 binding
        (controls and bypass spellings).
Part P: C8 honest positive of every label.
Every parent/scorer exit is recorded after the process exits (L45).

Prior art: test stubs / fault injection (Meszaros 2007); REV1-REV3 harnesses
(2026); argparse prefix matching (Python stdlib docs, allow_abbrev) and GNU
coreutils timeout option syntax (unverified - lead to check: "argparse
allow_abbrev", "GNU timeout -k DURATION"). Ours: N1-N5, the D7 bypass cases.
"""
import datetime
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
MROOT = Path("/tmp/gn1_rev4_m/W")
MA4 = MROOT / "artifacts/gn1/a4"
BASE = MA4 / ("rev4_runs_" + time.strftime("%H%M%S"))
TOOLS = Path("/tmp/gn1_rev4_tools")
FORGE = Path("/tmp/gn1_rev4_forge")
REAL_FORGE = subprocess.check_output(["bash", "-lc", "command -v forge-gpu-run"], text=True).strip()
PY = sys.executable
RESULTS = []
TMPDIR = "/tmp/gn1_rev4_tmp"
WRAP = str(OUT / "parent_wrap4.py")
STAND = str(OUT / "rev4_forge.py")
LINES = {r["label"]: r for r in json.loads((MA4 / "gpu_lines.json").read_text())}


def base_env(**extra):
    e = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TMPDIR=TMPDIR, CUDA_VISIBLE_DEVICES="", REV4_MA4=str(MA4),
             FORGE_GPU_SUMMARY=str(FORGE / "summary.log"), FORGE_GPU_WINDOW_FILE=str(FORGE / "GPU_WINDOW"),
             FORGE_GPU_LOCK=str(FORGE / "lock"), FORGE_GPU_POLL="0.2", FORGE_GPU_KILL_GRACE="1",
             OMP_NUM_THREADS="1")
    e.pop("GN1_AUDIT", None)
    e.update(extra)
    if e.get("FORGE_GPU_KILL_GRACE") is None:
        e.pop("FORGE_GPU_KILL_GRACE")
    assert e["FORGE_GPU_LOCK"].startswith("/tmp/gn1_rev4_") and e["FORGE_GPU_SUMMARY"].startswith("/tmp/gn1_rev4_")
    assert e["FORGE_GPU_WINDOW_FILE"].startswith("/tmp/gn1_rev4_") and e["CUDA_VISIBLE_DEVICES"] == ""
    return e


def run(args, env, timeout=180):
    p = subprocess.run(args, cwd=MA4, env=env, capture_output=True, text=True, timeout=timeout)
    return dict(exit=p.returncode, stdout=p.stdout[-1500:], stderr=p.stderr[-2500:])


def start(name):
    runs = BASE / name
    r = run([PY, "-B", "start_gpu_run.py"], base_env(RUNS=str(runs)))
    assert r["exit"] == 0, r
    return runs


def parent(runs, label, gpu_run, tail, bound=None, expected=None, popen=False, **env):
    line = LINES[label]
    ex = expected or ",".join(map(str, line["expected_child_exits"]))
    args = [PY, "-B", WRAP, label, ex, str(bound or line["outer"]), *gpu_run, *tail]
    e = base_env(RUNS=str(runs), **env)
    if popen:
        return subprocess.Popen(args, cwd=MA4, env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    r = run(args, e, timeout=(bound or 60) + 90)
    return collect(runs, label, r)


def collect(runs, label, r):
    atts = sorted((runs / label).glob("attempt-*")) if (runs / label).exists() else []
    r["attempts"] = [a.name for a in atts]
    last = atts[-1] if atts else None
    rec = last / "parent_result.json" if last else None
    try:
        full = json.loads(rec.read_text()) if rec and rec.exists() else None
    except ValueError:
        full = "UNREADABLE"
    r["parent"] = ({k: full.get(k) for k in ("exit", "passed", "reasons", "admission", "forge_outcome", "timeout",
                                             "timeout_cause", "unsealed")} if isinstance(full, dict) else full)
    fl = last / "forge.log" if last else None
    r["forge_log_last"] = fl.read_text().splitlines()[-1].split("\t")[-2:] if fl and fl.exists() and fl.read_text() else None
    r["last_attempt_files"] = sorted(p.name for p in last.iterdir()) if last else []
    r["traceback"] = "Traceback" in r.get("stderr", "")
    return r


def score(runs, label=None):
    r = run([PY, "-B", "score_gpu.py"], base_env(RUNS=str(runs)))
    p = runs / "G5_SUMMARY.json"
    s = json.loads(p.read_text()) if p.exists() else None
    r["G5_passed"] = s["passed"] if s else None
    r["errors"] = s["errors"] if s else None
    r["traceback"] = "Traceback" in r["stderr"]
    if s and label:
        row = s["lines"].get(label, {})
        r["line"] = {k: row.get(k) for k in ("passed", "reasons", "attempts", "selected_attempt", "unsealed")}
    r.pop("stdout")
    return r


def record(case, ref, expected, ok, **data):
    RESULTS.append(dict(case=case, ref=ref, expectation=expected, outcome_as_expected=ok, **data))
    print(case, "AS-EXPECTED" if ok else "UNEXPECTED", flush=True)


SLEEPER = """import os, sys, time
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
time.sleep(float(os.environ.get("REV2_SLEEP", "30")))
sys.exit(0)
"""


def summary(rows):
    (FORGE / "summary.log").write_text("".join("%s | %s\n" % (
        datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat(), ev) for t, ev in rows))


def no_summary():
    if (FORGE / "summary.log").exists():
        os.replace(FORGE / "summary.log", FORGE / ("summary.moved." + str(time.time())))


def real_forge(minimum, maximum, margin, label):
    return [REAL_FORGE, "--min", str(minimum), "--max", str(maximum), "--margin", str(margin), "--label", label, "--"]


def stand_child(arm, suite, inner=True):
    # The staged inner timeout (gpu_lines inner for candidate_focus is 240) wraps the CPU stand-in.
    return (["timeout", "--kill-after=10s", "240s"] if inner else []) + [PY, STAND, "run_gpu_tests.py", arm, suite]


def hold_lock():
    fd = os.open(str(FORGE / "lock"), os.O_RDWR | os.O_CREAT, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def drop_lock(fd):
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def part_b():
    FORGE.mkdir(exist_ok=True)
    TOOLS.mkdir(exist_ok=True)
    sleeper = TOOLS / "sleeper.py"
    sleeper.write_text(SLEEPER)
    tail = lambda arm, suite: [PY, str(sleeper), "run_gpu_tests.py", arm, suite]
    # B6 (REV3): outer bound fires during the runner's lock wait.
    runs = start("B6")
    summary([(time.time(), "COOLDOWN 3600s")])
    fd = hold_lock()
    first = parent(runs, "candidate_focus", real_forge(1, 60, 2, "rev4-B6"), tail("candidate", "focus"), bound=4)
    drop_lock(fd)
    retry = parent(runs, "candidate_focus", real_forge(250, 300, 2, "rev4-B6r"), stand_child("candidate", "focus"),
                   REV2_NO_LOG="1")
    sc = score(runs, "candidate_focus")
    ok = ((first["parent"] or {}).get("admission") == "not_admitted" and first["forge_log_last"] is None
          and (retry["parent"] or {}).get("admission") == "admitted" and retry["exit"] == 0
          and (sc.get("line") or {}).get("passed") is True and not first["traceback"])
    record("B6_lock_wait_outer_timeout", "R3-F01", "not admitted; retry admitted and passes; scorer line passes", ok,
           first=first, retry=retry, scorer=sc)
    # N1 (new, F01): two consecutive lock-wait kills, then a pass.
    runs = start("N1")
    fd = hold_lock()
    a1 = parent(runs, "candidate_focus", real_forge(1, 60, 2, "rev4-N1a"), tail("candidate", "focus"), bound=3)
    a2 = parent(runs, "candidate_focus", real_forge(1, 60, 2, "rev4-N1b"), tail("candidate", "focus"), bound=3)
    drop_lock(fd)
    a3 = parent(runs, "candidate_focus", real_forge(250, 300, 2, "rev4-N1c"), stand_child("candidate", "focus"),
                REV2_NO_LOG="1")
    sc = score(runs, "candidate_focus")
    ok = ([(x["parent"] or {}).get("admission") for x in (a1, a2, a3)] == ["not_admitted", "not_admitted", "admitted"]
          and a3["exit"] == 0 and (sc.get("line") or {}).get("passed") is True
          and (sc.get("line") or {}).get("selected_attempt", "").endswith("attempt-0003"))
    record("N1_two_lock_wait_kills_then_pass", "R3-F01 (new)", "1,2 not admitted; 3 admitted, selected, passes", ok,
           a1=a1, a2=a2, a3=a3, scorer=sc)
    summary([(time.time(), "COOLDOWN 3600s")])
    no_summary()
    # B7 (REV3): runner deadline, then outer bound inside the runner's grace.
    runs = start("B7")
    first = parent(runs, "candidate_focus", real_forge(1, 6, 3, "rev4-B7"), tail("candidate", "focus"), bound=4,
                   FORGE_GPU_KILL_GRACE="3")
    retry = parent(runs, "candidate_focus", real_forge(1, 60, 1, "rev4-B7r"), stand_child("candidate", "focus"),
                   bound=60, REV2_NO_LOG="1")
    sc = score(runs, "candidate_focus")
    ok = (first["parent"] is not None and first["parent"]["timeout_cause"] == "outer-timeout"
          and first["parent"]["admission"] == "admitted" and not first["parent"]["passed"]
          and first["forge_log_last"] == ["timeout", "143"] and retry["exit"] != 0
          and "label already has an admitted attempt" in retry["stderr"] and not retry["traceback"]
          and (sc.get("line") or {}).get("passed") is False)
    record("B7_outer_in_runner_grace", "R3-F02", "receipt, cause outer-timeout, admitted+failed, retry refused", ok,
           first=first, retry=retry, scorer=sc)
    # B7b (REV3): same, child wrote child_result before hanging.
    runs = start("B7b")
    first = parent(runs, "candidate_focus", real_forge(1, 6, 3, "rev4-B7b"), stand_child("candidate", "focus"), bound=4,
                   FORGE_GPU_KILL_GRACE="3", REV2_MODE="hang_after_result", REV2_NO_LOG="1", REV3_SLEEP="30")
    retry = parent(runs, "candidate_focus", real_forge(1, 60, 1, "rev4-B7br"), stand_child("candidate", "focus"),
                   bound=60, REV2_NO_LOG="1")
    sc = score(runs, "candidate_focus")
    ok = (first["parent"] is not None and first["parent"]["timeout_cause"] == "outer-timeout"
          and not first["parent"]["passed"] and retry["exit"] != 0 and not retry["traceback"]
          and (sc.get("line") or {}).get("passed") is False and not sc["traceback"])
    record("B7b_outer_in_grace_child_result", "R3-F02", "receipt; later parents refuse cleanly", ok,
           first=first, retry=retry, scorer=sc)
    # B8 (REV3): window closes, then outer bound inside the grace.
    runs = start("B8")
    now = time.time()
    summary([(now, "COOLDOWN 3600s")])
    p = parent(runs, "candidate_focus", real_forge(1, 60, 3, "rev4-B8"), tail("candidate", "focus"), bound=3,
               popen=True, FORGE_GPU_KILL_GRACE="3")
    time.sleep(1.0)
    summary([(now, "COOLDOWN 3600s"), (time.time(), "COOLDOWN 1s")])
    out, err = p.communicate(timeout=120)
    first = collect(runs, "candidate_focus", dict(exit=p.returncode, stdout=out[-1500:], stderr=err[-2500:]))
    sc = score(runs, "candidate_focus")
    ok = (first["parent"] is not None and first["parent"]["timeout_cause"] == "outer-timeout"
          and first["forge_log_last"] == ["window-closed", "143"] and first["parent"]["admission"] == "admitted")
    record("B8_outer_in_window_closed_grace", "R3-F02", "receipt, window-closed/143, cause outer-timeout, admitted",
           ok, first=first, scorer=sc)
    no_summary()
    # N2 (new, Q3): the staged configuration: default kill grace (10 s, env unset) equals the
    # outer --kill-after=10s; a TERM-ignoring child; outer bound fires during the run.
    runs = start("N2")
    first = parent(runs, "candidate_focus", real_forge(1, 60, 10, "rev4-N2"), stand_child("candidate", "focus", inner=False),
                   bound=4, FORGE_GPU_KILL_GRACE=None, REV2_MODE="ignore_term", REV3_SLEEP="40", REV2_NO_LOG="1")
    att = runs / "candidate_focus" / "attempt-0001"
    pid = int((att / "orphan_pid.txt").read_text()) if (att / "orphan_pid.txt").exists() else None
    alive = False
    if pid:
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
    killed = False
    if alive:
        # This review started this process (descendant of its own parent run).
        os.kill(pid, signal.SIGKILL)
        killed = True
    record("N2_default_grace_equals_outer_kill_after", "Q3 (new)",
           "observe: exit, runner row, cause, admission, and whether the child survives the parent",
           first["parent"] is not None and first["parent"]["timeout_cause"] == "outer-timeout",
           first=first, child_pid=pid, child_alive_after_parent_returned=alive, reviewer_killed_own_descendant=killed)


def part_n():
    """N2b/N2c (new, Q3): the staged inner timeout present, runner grace at its default (10 s)."""
    FORGE.mkdir(exist_ok=True)
    no_summary()
    for name, mode in (("N2b_staged_obedient_child", "marker_only_sleep"), ("N2c_staged_term_ignoring_child", "ignore_term")):
        runs = start(name)
        t0 = time.monotonic()
        first = parent(runs, "candidate_focus", real_forge(250, 300, 10, "rev4-" + name), stand_child("candidate", "focus"),
                       bound=4, FORGE_GPU_KILL_GRACE=None, REV2_MODE=mode, REV3_SLEEP="40", REV2_NO_LOG="1")
        elapsed = round(time.monotonic() - t0, 2)
        att = runs / "candidate_focus" / "attempt-0001"
        pid = int((att / "orphan_pid.txt").read_text()) if (att / "orphan_pid.txt").exists() else None
        def alive():
            if not pid:
                return None
            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False
        at_return = alive()
        time.sleep(12)
        after_12s = alive()
        killed = False
        if after_12s:
            os.kill(pid, signal.SIGKILL)  # this review started this process
            killed = True
        sc = score(runs, "candidate_focus")
        record(name, "Q3 (new)", "observe exit / row / cause under the staged grace",
               (first["parent"] or {}).get("timeout_cause") == "outer-timeout", first=first, parent_wall_s=elapsed,
               child_pid=pid, child_alive_at_parent_return=at_return, child_alive_12s_later=after_12s,
               reviewer_killed_own_descendant=killed, scorer=sc)


def fabricate(runs, label, files):
    att = runs / label / "attempt-0001"
    att.mkdir(parents=True)
    for name, text in files.items():
        (att / name).write_text(text)
    return att


def sforge(label):
    # Stand-in runner argv carrying the line's registered --min/--max (D7 binds them).
    return [PY, STAND, "--min", str(LINES[label]["minimum"]), "--max", str(LINES[label]["maximum"]), "--label", "x", "--"]


def stail(label):
    return ["timeout", "--kill-after=10s", str(LINES[label]["inner"]) + "s", "python3", "run_gpu_tests.py",
            LINES[label]["arm"], LINES[label]["suite"]]


def part_c():
    forge = sforge("candidate_focus")
    tail = stail
    label = "candidate_focus"
    # C7b (REV3): child started (marker + invocation), parent killed, no result.
    runs = start("C7b")
    p = parent(runs, label, forge, stail(label), popen=True, REV2_MODE="started_no_result", REV3_SLEEP="6")
    deadline = time.time() + 20
    while time.time() < deadline and not list((runs / label).glob("attempt-0001/invocation.json")):
        time.sleep(0.2)
    p.kill()  # this review started this process
    p.communicate(timeout=30)
    time.sleep(7.0)
    files = sorted(x.name for x in (runs / label / "attempt-0001").iterdir())
    retry = parent(runs, label, forge, tail(label))
    sc = score(runs, label)
    ok = (retry["exit"] != 0 and "label already has an admitted attempt" in retry["stderr"] and not retry["traceback"]
          and (sc.get("line") or {}).get("passed") is False and "incomplete" in json.dumps(sc.get("line")))
    record("C7b_started_no_result", "R3-F03", "admitted, failed incomplete, never retried to a pass", ok,
           first_attempt_files=files, killed_parent_exit=p.returncode, retry=retry, scorer=sc)
    # N3 (new, F03): one start-evidence class at a time.
    row = lambda o, c: "\t".join(["2026-11-01T00:00:00+00:00", "x", label, "halted", "-", "0", "0", o, str(c)]) + "\n"
    classes = {
        "marker_only": ({"child_started.json": '{"pid": 1}\n'}, True),
        "invocation_only": ({"invocation.json": '{}\n'}, True),
        "junit_only": ({"junit.xml": "<testsuites/>"}, True),
        "row_ran": ({"forge.log": row("ran", 143)}, True),
        "row_timeout": ({"forge.log": row("timeout", 124)}, True),
        "row_window_closed": ({"forge.log": row("window-closed", 143)}, True),
        "row_no_lock": ({"forge.log": row("no-lock", 76)}, False),
        "row_rechecked_out": ({"forge.log": row("rechecked-out", 75)}, False),
        "stdout_only": ({"stdout.log": ""}, False),
        "marker_tmp_only": ({"child_started.json.tmp.12345": '{"pid'}, False),
    }
    for name, (files, admitted) in classes.items():
        runs = start("N3_" + name)
        fabricate(runs, label, files)
        retry = parent(runs, label, forge, tail(label))
        sc = score(runs, label)
        refused = retry["exit"] != 0 and "label already has an admitted attempt" in retry["stderr"]
        ok = (refused == admitted and not retry["traceback"] and not sc["traceback"]
              and (sc.get("line") or {}).get("passed") is (not admitted))
        record("N3_" + name, "R3-F03 (new)", ("admitted: retry refused, line fails" if admitted
                                              else "not admitted: retry admitted, line passes"), ok, retry=retry, scorer=sc)
    # E5 (REV3): attempt with only a truncated child_result.json.
    runs = start("E5")
    fabricate(runs, label, {"child_result.json": '{"arm": "candidate", "suite": "foc'})
    retry = parent(runs, label, forge, tail(label))
    sc = score(runs, label)
    ok = (retry["exit"] != 0 and not retry["traceback"] and "label already has an admitted attempt" in retry["stderr"]
          and (sc.get("line") or {}).get("passed") is False and not sc["traceback"])
    record("E5_truncated_child_result", "R3-F04", "admitted, failed with a recorded reason, no crash", ok,
           retry=retry, scorer=sc)
    # E5b (new): truncated child_result written through the real parent.
    runs = start("E5b")
    r = parent(runs, label, forge, tail(label), REV2_MODE="truncated_result")
    sc = score(runs, label)
    ok = (r["parent"] is not None and not r["parent"]["passed"] and r["parent"]["admission"] == "admitted"
          and not r["traceback"] and (sc.get("line") or {}).get("passed") is False)
    record("E5b_truncated_child_result_via_parent", "R3-F04 (new)", "parent receipt with a recorded reason", ok,
           run=r, scorer=sc)
    # N4 (new, Q3): truncated parent_result.json from a killed parent.
    runs = start("N4")
    r = parent(runs, label, forge, tail(label), REV2_MODE="hang_failure")
    att = runs / label / "attempt-0001"
    text = (att / "parent_result.json").read_text()
    os.replace(att / "parent_result.json", att / "parent_result.full.json")
    (att / "parent_result.json").write_text(text[:len(text) // 2])
    retry = parent(runs, label, forge, tail(label))
    sc = score(runs, label)
    ok = not retry["traceback"] and not sc["traceback"] and (sc.get("line") or {}).get("passed") is False
    record("N4_truncated_parent_result", "Q3 (new)", "later parent refuses with a recorded reason, not a traceback",
           ok, retry=retry, scorer=sc)
    # N5 (new, Q3/F02): runner logs a no-lock row, outer bound fires before it exits (race window).
    runs = start("N5")
    r = parent(runs, label, sforge(label), ["python3", "run_gpu_tests.py", "candidate", "focus"], bound=3,
               REV2_MODE="nolock_row_hang")
    sc = score(runs, label)
    ok = r["parent"] is not None and not r["traceback"]
    record("N5_nolock_row_outer_exit", "Q3 (new)", "parent writes a not-admitted receipt (no crash)", ok,
           run=r, scorer=sc)


def part_d():
    tail = stail
    # D5 runtime census: sealed tamper fails; unsealed listed, passes.
    for name, mode, want_pass in (("D5_tamper", "tamper_runtime", False), ("D5_unsealed", "unsealed_runtime", True)):
        runs = start(name)
        r = parent(runs, "candidate_focus", sforge("candidate_focus"), tail("candidate_focus"), REV2_MODE=mode)
        sc = score(runs, "candidate_focus")
        ok = ((r["parent"] or {}).get("passed") is want_pass and (sc.get("line") or {}).get("passed") is want_pass
              and (want_pass and any("planted_rev4" in u["path"] for u in (sc["line"].get("unsealed") or []))
                   or not want_pass and any("tamper" in x for x in r["parent"]["reasons"])))
        record(name, "R3-F05/D5", "pass+listed" if want_pass else "fails: runtime module tamper", ok, run=r, scorer=sc)
    # D6: the live operator.index premise is false -> the equivalence line fails.
    for label in ("E_npbool_g", "E_npbool_c"):
        runs = start("D6_" + label)
        assert parent(runs, "candidate_focus", sforge("candidate_focus"), tail("candidate_focus"))["exit"] == 0
        honest = parent(runs, label, sforge(label), tail(label))
        runs2 = start("D6f_" + label)
        assert parent(runs2, "candidate_focus", sforge("candidate_focus"), tail("candidate_focus"))["exit"] == 0
        false = parent(runs2, label, sforge(label), tail(label), REV2_MODE="npbool_accepts")
        ok = (honest["exit"] == 0 and (honest["parent"] or {}).get("passed") is True and false["exit"] == 1
              and any("equivalence invalid" in x for x in (false["parent"] or {}).get("reasons", [])))
        record("D6_" + label, "R3-F06/D6", "honest pass; premise false fails", ok, honest=honest, premise_false=false)
    # D7 controls and bypass spellings (candidate_focus: outer 330, max 300, min 250, inner 240).
    ln = LINES["candidate_focus"]
    good = lambda: ["--min", "250", "--max", "300", "--label", "x", "--"]
    inner = lambda: ["timeout", "--kill-after=10s", "240s", "python3", "run_gpu_tests.py", "candidate", "focus"]
    cases = [
        ("D7_honest", 330, good(), inner(), True),
        ("D7_outer_329", 329, good(), inner(), False),
        ("D7_max_299", 330, ["--min", "250", "--max", "299", "--label", "x", "--"], inner(), False),
        ("D7_min_251", 330, ["--min", "251", "--max", "300", "--label", "x", "--"], inner(), False),
        ("D7_inner_239", 330, good(), ["timeout", "--kill-after=10s", "239s", "python3", "run_gpu_tests.py", "candidate", "focus"], False),
        ("D7_bypass_max_equals", 330, ["--min", "250", "--max", "300", "--max=900", "--label", "x", "--"], inner(), False),
        ("D7_bypass_min_abbrev", 330, ["--min", "250", "--max", "300", "--mi", "1", "--label", "x", "--"], inner(), False),
        ("D7_bypass_min_equals", 330, ["--min", "250", "--max", "300", "--min=1", "--label", "x", "--"], inner(), False),
        ("D7_bypass_inner_k_value", 330, good(), ["timeout", "-k", "240", "900s", "python3", "run_gpu_tests.py", "candidate", "focus"], False),
        ("D7_bypass_inner_kill_after_space", 330, good(), ["timeout", "--kill-after", "240", "900s", "python3", "run_gpu_tests.py", "candidate", "focus"], False),
    ]
    for name, bound, runner_args, inner_cmd, want in cases:
        runs = start(name)
        r = parent(runs, "candidate_focus", [PY, STAND, *runner_args], inner_cmd, bound=bound)
        sc = score(runs, "candidate_focus")
        got = (sc.get("line") or {}).get("passed")
        record(name, "D7", "scorer passes" if want else "scorer refuses the mismatch", got is want,
               parent_exit=r["exit"], parent_passed=(r["parent"] or {}).get("passed"), scorer=sc)
    # The real runner's parser honours these spellings (parse only, no lease, no command run).
    src = Path(REAL_FORGE).read_text()
    ns = {"__name__": "forge_probe", "__file__": REAL_FORGE}
    exec(compile(src, "forge-gpu-run", "exec"), ns)
    parsed = {}
    for argv in (good() + ["true"], ["--min", "250", "--max", "300", "--max=900", "--label", "x", "--", "true"],
                 ["--min", "250", "--max", "300", "--mi", "1", "--label", "x", "--", "true"],
                 ["--min", "250", "--max", "300", "--min=1", "--label", "x", "--", "true"]):
        a = ns["options"](argv)
        parsed[" ".join(argv)] = dict(minimum=a.minimum, maximum=a.maximum)
    t0 = time.monotonic()
    tk = subprocess.run(["timeout", "-k", "5", "1s", "sleep", "4"])
    gnu = dict(argv="timeout -k 5 1s sleep 4", exit=tk.returncode, seconds=round(time.monotonic() - t0, 2))
    record("D7_runner_parser_probe", "D7", "real parser applies the later/abbreviated values; GNU timeout -k takes a value",
           True, parsed=parsed, gnu_timeout=gnu)


def part_p():
    tail = stail
    runs = start("C8_full_positive")
    order = ["baseline_focus", "candidate_focus", "baseline_regression", "candidate_regression"] + \
            [l for l in LINES if l not in ("baseline_focus", "candidate_focus", "baseline_regression", "candidate_regression")]
    bad = []
    for l in order:
        r = parent(runs, l, sforge(l), tail(l))
        if r["exit"] != 0:
            bad.append(dict(label=l, reasons=(r["parent"] or {}).get("reasons"), stderr=r["stderr"][-400:]))
    sc = score(runs)
    record("C8_full_positive", "Q6", "every label admitted and G5 passed", not bad and sc["exit"] == 0,
           failed_labels=bad, labels=len(order), scorer=dict(exit=sc["exit"], G5_passed=sc["G5_passed"], errors=sc["errors"]))


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    parts = sys.argv[1:] or ["b", "c", "d", "p"]
    BASE.mkdir(parents=True)
    TOOLS.mkdir(exist_ok=True)
    FORGE.mkdir(exist_ok=True)
    for part in parts:
        try:
            globals()["part_" + part]()
        except Exception as exc:  # recorded, never hidden
            import traceback
            record("PART_%s_harness_error" % part, "-", "harness completes", False, error=traceback.format_exc()[-3000:])
    redact = lambda s: s.replace(str(Path.home()), "$HOME").replace(REAL_FORGE, "forge-gpu-run")
    out = OUT / ("hostile_rev4_" + "".join(parts) + ".json")
    out.write_text(redact(json.dumps(dict(base=str(BASE), results=RESULTS), indent=1)) + "\n")
    print("unexpected:", [r["case"] for r in RESULTS if not r["outcome_as_expected"]])


if __name__ == "__main__":
    main()
