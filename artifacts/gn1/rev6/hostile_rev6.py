#!/usr/bin/env python3
"""GN1-REV6 re-run of REV5's hostile_rev5.py against A8 (retargeted copy; REV5's
cases and expectations kept verbatim, A8 differences listed here):
- paths: mirror /tmp/gn1_rev6_m/W (mirror_regen_a8.py), wrapper parent_wrap6.py,
  scratch /tmp/gn1_rev6_*; the real runner always has summary/window/lock in scratch;
- record() also keeps each scorer's line reasons and root errors (A8 refuses every
  CPU root for absent device evidence, so a REV5 refusal is only informative when the
  specific REV5 rule appears among the line reasons; judged in analyze_rev6.py);
- part P: A8's staged script uses absolute /usr/bin/timeout and /usr/bin/python3,
  so REV5's regex is widened to that spelling and "$GPU_RUN" maps to the pinned
  runner path. A8 production lines cannot reach a CPU stand-in child (absolute
  interpreter), so the honest 58-line run uses A8's declared CPU control path
  (GN1_CONTROL=1, GN1_CPU_FIXTURE=1), argv exactly as staged, as A8's G6 did.
--- REV5 docstring follows ---
GN1-REV5 hostile runs through A5's REAL parent (run_gpu_line.py via
parent_wrap6.py, one disclosed guard-prefix substitution), A5's REAL scorer
(score_gpu.py) and evidence.py, byte-identical in the /tmp mirror built by
mirror_regen_a5.py (seal verified on every line).

Part Q: D2 exact-form binding. REV4's five spellings, new spellings, and new
        argv-external routes: a single-token runner wrapper (command[0]) and a
        PATH-shadowed `timeout`.
Part N: REV4 N2b/N2c with the REAL installed runner (summary, window and lock
        redirected to /tmp/gn1_rev6_forge; CPU child through a PATH python3
        dispatcher), grace unset and under A5's staging, plus new grace and
        lock-wait cases. Orphans are checked by scanning /proc/<pid>/environ of
        this user's processes for the attempt directory.
Part C: REV4 N4/N5 and new malformed-receipt and malformed-row cases.
Part P: every staged line of STAGED_GPU_COMMANDS.sh, argv exactly as written,
        with the REAL runner, then the scorer; D5 runtime and D6 controls.
Every parent/scorer exit is recorded after the process exits (L45).

Prior art: test stubs / fault injection (Meszaros 2007); REV1-REV4 harnesses
(2026), hostile_rev4.py in particular; argparse prefix matching (Python stdlib,
allow_abbrev) and GNU coreutils timeout option syntax; PATH command lookup
(POSIX execvp) for the shadowing cases. Ours: the new cases (wrapper runner,
PATH shadow, grace 19/21, staged lock-wait kill, malformed rows/receipts,
deep-nesting JSON).
"""
import datetime
import fcntl
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
MROOT = Path("/tmp/gn1_rev6_m/W")
MA8 = MROOT / "artifacts/gn1/a8"
BASE = MA8 / ("rev6_runs_" + time.strftime("%H%M%S"))
TOOLS = Path("/tmp/gn1_rev6_tools")
FORGE = Path("/tmp/gn1_rev6_forge")
OBS = Path("/tmp/gn1_rev6_obs")
REAL_FORGE = subprocess.check_output(["bash", "-lc", "command -v forge-gpu-run"], text=True).strip()
REAL_TIMEOUT = shutil.which("timeout")
PY = sys.executable
RESULTS = []
TMPDIR = "/tmp/gn1_rev6_tmp"
WRAP = str(OUT / "parent_wrap6.py")
STAND = str(MA8 / "rev4_forge.py")
LINES = {r["label"]: r for r in json.loads((MA8 / "gpu_lines.json").read_text())}
CANON = ["--min", "250", "--max", "300", "--label", "candidate_focus"]
INNER = ["timeout", "--kill-after=10s", "240s", "python3", "run_gpu_tests.py", "candidate", "focus"]


def tool(name, body, directory=TOOLS):
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / name
    p.write_text(body)
    p.chmod(0o755)
    return str(p)


def make_tools():
    # python3 dispatcher: routes only `python3 run_gpu_tests.py ...` to A5's own
    # CPU stand-in child; records the kernel's view (/proc cmdline) of its parent
    # (the inner timeout) and grandparent (the runner) when REV5_OBS is set.
    tool("python3", f"""#!{PY}
import json, os, sys
def cmd(pid):
    try:
        return open('/proc/%d/cmdline' % pid, 'rb').read().split(b'\\0')[:-1]
    except OSError:
        return None
if sys.argv[1:2] == ['run_gpu_tests.py']:
    obs = os.environ.get('REV5_OBS')
    if obs:
        pp = os.getppid()
        try:
            gp = int(open('/proc/%d/stat' % pp).read().rsplit(')', 1)[1].split()[1])
        except OSError:
            gp = None
        rec = dict(inner=[c.decode() for c in (cmd(pp) or [])], runner=[c.decode() for c in (cmd(gp) or [])] if gp else None)
        with open(obs, 'a') as f:
            f.write(json.dumps(rec) + '\\n')
    os.execv({PY!r}, [{PY!r}, '-B', {STAND!r}] + sys.argv[1:])
os.execv({PY!r}, [{PY!r}] + sys.argv[1:])
""", TOOLS / "bin")
    # Single-token stand-in runner (fixture): A5's own rev4_forge.py, which writes
    # the runner row itself unless REV2_NO_LOG=1.
    tool("standin-runner", f"""#!{PY}
import os, sys
os.execv({PY!r}, [{PY!r}, '-B', {STAND!r}] + sys.argv[1:])
""")
    # Single-token wrapper of the REAL runner: appends --max 900 --self-leasing
    # after the registered options (argparse: the later --max wins).
    tool("wrapped-runner", f"""#!{PY}
import os, sys
a = sys.argv[1:]
s = a.index('--')
os.execv({REAL_FORGE!r}, [{REAL_FORGE!r}] + a[:s] + ['--max', '900', '--self-leasing'] + a[s:])
""")
    # PATH-shadowed timeout: rewrites only the registered inner duration 240s -> 900s.
    tool("timeout", f"""#!{PY}
import os, sys
a = sys.argv[1:]
if a[:2] == ['--kill-after=10s', '240s']:
    a[1] = '900s'
os.execv({REAL_TIMEOUT!r}, [{REAL_TIMEOUT!r}] + a)
""", TOOLS / "shadow")
    # Fixture runner for malformed rows: writes REV5_ROW (python literal bytes) or
    # makes forge.log a directory, optionally a deep child_result, then sleeps or exits.
    tool("fixture-runner", f"""#!{PY}
import ast, json, os, sys, time
from pathlib import Path
log = Path(os.environ['FORGE_GPU_RUN_LOG'])
att = Path(os.environ['GN1_ATTEMPT'])
mode = os.environ.get('REV5_FIX', 'row')
if mode == 'dir':
    log.mkdir()
elif mode == 'deep_child':
    (att / 'child_started.json').write_text(json.dumps(dict(pid=os.getpid())) + '\\n')
    (att / 'child_result.json').write_text('[' * 200000 + ']' * 200000)
    with open(log, 'a') as f:
        f.write('\\t'.join(['2026-11-01T00:00:00+00:00', 'x', 'candidate_regression', 'halted', '-', '0', '0', 'ran', '1']) + '\\n')
    sys.exit(1)
else:
    with open(log, 'ab') as f:
        f.write(ast.literal_eval(os.environ['REV5_ROW']))
time.sleep(float(os.environ.get('REV5_SLEEP', '60')))
sys.exit(0)
""")


def base_env(**extra):
    e = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TMPDIR=TMPDIR, CUDA_VISIBLE_DEVICES="", REV4_MA4=str(MA8),
             REV6_MA8=str(MA8), FORGE_GPU_SUMMARY=str(FORGE / "summary.log"),
             FORGE_GPU_WINDOW_FILE=str(FORGE / "GPU_WINDOW"), FORGE_GPU_LOCK=str(FORGE / "lock"),
             FORGE_GPU_KILL_GRACE="10", FORGE_GPU_POLL="5", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
             MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
             PATH=str(TOOLS / "bin") + os.pathsep + os.environ["PATH"])
    e.pop("GN1_AUDIT", None)
    for k, v in extra.items():
        if v is None:
            e.pop(k, None)
        else:
            e[k] = v
    assert e["FORGE_GPU_LOCK"].startswith("/tmp/gn1_rev6_") and e["FORGE_GPU_SUMMARY"].startswith("/tmp/gn1_rev6_")
    assert e["FORGE_GPU_WINDOW_FILE"].startswith("/tmp/gn1_rev6_") and e["CUDA_VISIBLE_DEVICES"] == ""
    return e


def run(args, env, timeout=180):
    t0 = time.time()
    p = subprocess.run(args, cwd=MA8, env=env, capture_output=True, text=True, timeout=timeout)
    return dict(exit=p.returncode, stdout=p.stdout[-1200:], stderr=p.stderr[-2500:], launched=t0,
                wall_s=round(time.time() - t0, 3))


def start(name, **extra):
    runs = BASE / name
    r = run([PY, "-B", "start_gpu_run.py"], base_env(RUNS=str(runs), **extra))
    assert r["exit"] == 0, r
    return runs


def collect(runs, label, r):
    atts = sorted((runs / label).glob("attempt-*")) if (runs / label).exists() else []
    r["attempts"] = [a.name for a in atts]
    last = atts[-1] if atts else None
    rec = last / "parent_result.json" if last else None
    try:
        full = json.loads(rec.read_text()) if rec and rec.exists() and rec.is_file() else None
    except (ValueError, RecursionError):
        full = "UNREADABLE"
    r["parent"] = ({k: full.get(k) for k in ("exit", "raw_exit", "passed", "reasons", "admission", "forge_outcome",
                                             "timeout", "timeout_cause", "command", "start", "child_exited", "recorded")}
                   if isinstance(full, dict) else full)
    fl = last / "forge.log" if last else None
    try:
        r["forge_log_last"] = (fl.read_text().splitlines()[-1].split("\t")[-4:]
                               if fl and fl.is_file() and fl.read_text() else None)
    except (OSError, ValueError):
        r["forge_log_last"] = "UNREADABLE"
    r["last_attempt_files"] = sorted(p.name for p in last.iterdir()) if last else []
    r["traceback"] = "Traceback" in r.get("stderr", "")
    r["stderr_json_line"] = _json_line(r.get("stderr", ""))
    return r


def _json_line(text):
    for ln in reversed(text.strip().splitlines()):
        try:
            return json.loads(ln)
        except ValueError:
            continue
    return None


def parent(runs, label, command, bound=None, expected=None, popen=False, **env):
    line = LINES[label]
    ex = expected or ",".join(map(str, line["expected_child_exits"]))
    args = [PY, "-B", WRAP, label, ex, str(bound or line["outer"]), *command]
    e = base_env(RUNS=str(runs), **env)
    if popen:
        return subprocess.Popen(args, cwd=MA8, env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    r = run(args, e, timeout=(bound or 60) + 120)
    return collect(runs, label, r)


def score(runs, label=None):
    r = run([PY, "-B", "score_gpu.py"], base_env(RUNS=str(runs)))
    p = runs / "G5_SUMMARY.json"
    try:
        s = json.loads(p.read_text()) if p.exists() else None
    except ValueError:
        s = None
    r["G5_summary_written"] = s is not None
    r["G5_passed"] = s["passed"] if s else None
    r["errors"] = s["errors"] if s else None
    r["traceback"] = "Traceback" in r["stderr"]
    if s and label:
        row = s["lines"].get(label, {})
        r["line"] = {k: row.get(k) for k in ("passed", "reasons", "attempts", "selected_attempt", "mode")}
    r.pop("stdout")
    return r


def record(case, ref, expected, ok, **data):
    RESULTS.append(dict(case=case, ref=ref, expectation=expected, outcome_as_expected=ok, **data))
    print(case, "AS-EXPECTED" if ok else "UNEXPECTED", flush=True)


def live_for(attempt):
    """PIDs of this user's processes whose environment names this attempt dir."""
    needle = ("GN1_ATTEMPT=" + str(attempt)).encode()
    found = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            env = open("/proc/%s/environ" % d, "rb").read().split(b"\0")
            if needle in env:
                found.append(int(d))
        except OSError:
            continue
    return found


def summary_rows(rows):
    (FORGE / "summary.log").write_text("".join("%s | %s\n" % (
        datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat(), ev) for t, ev in rows))


def no_summary():
    if (FORGE / "summary.log").exists():
        os.replace(FORGE / "summary.log", FORGE / ("summary.moved." + str(time.time())))


def hold_lock():
    fd = os.open(str(FORGE / "lock"), os.O_RDWR | os.O_CREAT, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def drop_lock(fd):
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


SR = lambda: [str(TOOLS / "standin-runner")]


def part_q():
    label = "candidate_focus"
    cases = [
        ("Q_honest", 330, CANON, INNER, True),
        ("REV4_max_equals", 330, ["--min", "250", "--max", "300", "--max=900", "--label", label], INNER, False),
        ("REV4_min_abbrev", 330, ["--min", "250", "--max", "300", "--mi", "1", "--label", label], INNER, False),
        ("REV4_min_equals", 330, ["--min", "250", "--max", "300", "--min=1", "--label", label], INNER, False),
        ("REV4_inner_k_value", 330, CANON, ["timeout", "-k", "240", "900s", "python3", "run_gpu_tests.py", "candidate", "focus"], False),
        ("REV4_inner_kill_after_space", 330, CANON, ["timeout", "--kill-after", "240", "900s", "python3", "run_gpu_tests.py", "candidate", "focus"], False),
        ("NEW_label_equals", 330, ["--min", "250", "--max", "300", "--label=candidate_focus"], INNER, False),
        ("NEW_reordered_min_max", 330, ["--max", "300", "--min", "250", "--label", label], INNER, False),
        ("NEW_min_float_spelling", 330, ["--min", "250.0", "--max", "300", "--label", label], INNER, False),
        ("NEW_margin_default_value", 330, ["--min", "250", "--max", "300", "--margin", "30", "--label", label], INNER, False),
        ("NEW_self_leasing_flag", 330, ["--min", "250", "--max", "300", "--label", label, "--self-leasing"], INNER, False),
        ("NEW_inner_unitless", 330, CANON, ["timeout", "--kill-after=10", "240", "python3", "run_gpu_tests.py", "candidate", "focus"], False),
        ("NEW_inner_k_attached", 330, CANON, ["timeout", "-k10s", "240s", "python3", "run_gpu_tests.py", "candidate", "focus"], False),
        ("NEW_inner_minutes", 330, CANON, ["timeout", "--kill-after=10s", "4m", "python3", "run_gpu_tests.py", "candidate", "focus"], False),
        ("NEW_inner_preserve_status", 330, CANON, ["timeout", "--preserve-status", "--kill-after=10s", "240s", "python3", "run_gpu_tests.py", "candidate", "focus"], False),
        ("NEW_inner_double_dash", 330, CANON, ["timeout", "--kill-after=10s", "240s", "--", "python3", "run_gpu_tests.py", "candidate", "focus"], False),
    ]
    for name, bound, opts, inner, want in cases:
        runs = start(name)
        r = parent(runs, label, [*SR(), *opts, "--", *inner], bound=bound)
        sc = score(runs, label)
        got = (sc.get("line") or {}).get("passed")
        record(name, "R4-F01/D2", "scorer passes" if want else "scorer refuses the spelling", got is want,
               parent_exit=r["exit"], parent_passed=(r["parent"] or {}).get("passed"), parent_traceback=r["traceback"],
               scorer=sc)
    # NEW (Q2): a single-token wrapper of the REAL runner as command[0]; registered
    # argv[1:] and inner argv exactly canonical; the wrapper appends --max 900
    # --self-leasing. The scratch lock is HELD during the run.
    no_summary()
    runs = start("Q2_wrapped_runner")
    obs = OBS / "Q2_wrapped_runner.jsonl"
    fd = hold_lock()
    r = parent(runs, label, [str(TOOLS / "wrapped-runner"), *CANON, "--", *INNER], REV2_NO_LOG="1", REV5_OBS=str(obs))
    drop_lock(fd)
    sc = score(runs, label)
    seen = [json.loads(x) for x in obs.read_text().splitlines()] if obs.exists() else []
    row = (runs / label / "attempt-0001" / "forge.log").read_text().split("\t") if (runs / label / "attempt-0001" / "forge.log").exists() else None
    record("Q2_wrapped_runner_command0", "Q2 (new)", "scorer refuses a runner that is not the registered one",
           (sc.get("line") or {}).get("passed") is False, parent=r, scorer=sc, kernel_view=seen,
           runner_row_budget=row[5] if row else None, runner_row_waited=row[6] if row else None,
           lock_held_by_reviewer_during_run=True)
    # Control for the above: the same with the REAL runner unwrapped, lock held -> no-lock.
    runs = start("Q2_real_runner_lock_held")
    fd = hold_lock()
    r2 = parent(runs, label, [REAL_FORGE, *CANON, "--", *INNER], REV2_NO_LOG="1")
    drop_lock(fd)
    record("Q2_control_real_runner_lock_held", "Q2 (new)", "real runner refuses to start without the lock (not admitted)",
           (r2["parent"] or {}).get("admission") == "not_admitted", parent=r2)
    runs = start("Q2_real_runner_honest")
    r3 = parent(runs, label, [REAL_FORGE, *CANON, "--", *INNER], REV2_NO_LOG="1")
    sc3 = score(runs, label)
    row3 = (runs / label / "attempt-0001" / "forge.log").read_text().split("\t")
    record("Q2_control_real_runner_honest", "Q2 (new)", "honest real-runner line passes; budget field for comparison",
           (sc3.get("line") or {}).get("passed") is True, parent=r3, scorer=sc3, runner_row_budget=row3[5])
    # NEW (Q2): PATH-shadowed `timeout` rewriting the inner 240s to 900s.
    runs = start("Q2_path_shadow_timeout")
    obs = OBS / "Q2_path_shadow_timeout.jsonl"
    path = str(TOOLS / "shadow") + os.pathsep + str(TOOLS / "bin") + os.pathsep + os.environ["PATH"]
    r = parent(runs, label, [REAL_FORGE, *CANON, "--", *INNER], REV2_NO_LOG="1", REV5_OBS=str(obs), PATH=path)
    sc = score(runs, label)
    seen = [json.loads(x) for x in obs.read_text().splitlines()] if obs.exists() else []
    record("Q2_path_shadow_inner_timeout", "Q2 (new)", "scorer refuses a line whose inner timeout ran another bound",
           (sc.get("line") or {}).get("passed") is False, parent=r, scorer=sc, kernel_view=seen)


def part_n():
    no_summary()
    label = "candidate_focus"
    cmd = [REAL_FORGE, *CANON, "--", *INNER]
    cases = [
        ("N2b_unset_grace", "marker_only_sleep", dict(FORGE_GPU_KILL_GRACE=None, FORGE_GPU_POLL=None)),
        ("N2c_unset_grace", "ignore_term", dict(FORGE_GPU_KILL_GRACE=None, FORGE_GPU_POLL=None)),
        ("N2b_D3_staged", "marker_only_sleep", dict(FORGE_GPU_KILL_GRACE="10", FORGE_GPU_POLL="5")),
        ("N2c_D3_staged", "ignore_term", dict(FORGE_GPU_KILL_GRACE="10", FORGE_GPU_POLL="5")),
        ("NEW_grace_18", "ignore_term", dict(FORGE_GPU_KILL_GRACE="18", FORGE_GPU_POLL="5")),
        ("NEW_grace_21", "ignore_term", dict(FORGE_GPU_KILL_GRACE="21", FORGE_GPU_POLL="5")),
    ]
    for name, mode, extra in cases:
        runs = start(name)
        t0 = time.monotonic()
        first = parent(runs, label, cmd, bound=4, REV2_MODE=mode, REV3_SLEEP="40", REV2_NO_LOG="1", **extra)
        elapsed = round(time.monotonic() - t0, 2)
        att = runs / label / "attempt-0001"
        at_return = live_for(att)
        time.sleep(11)
        later = live_for(att)
        killed = []
        for pid in later:  # started by this review (descendants of its own parent run)
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except ProcessLookupError:
                pass
        sc = score(runs, label)
        p = first["parent"] or {}
        if name.startswith("NEW_grace_21"):
            want = p.get("exit") == 137 and p.get("raw_exit") == -9 and p.get("timeout_cause") == "outer-timeout"
        else:
            want = (p.get("exit") == 124 and p.get("timeout_cause") == "outer-timeout"
                    and first["forge_log_last"] is not None and first["forge_log_last"][-2:] == ["ran", "143"])
        record(name, "R4-F02/Q3", "exit 124, row ran/143, cause outer-timeout (grace 21: -9 -> 137, outer-timeout); no live process",
               want and not at_return and not later, first=first, parent_wall_s=elapsed,
               live_at_parent_return=at_return, live_11s_later=later, reviewer_killed_own_descendants=killed,
               scorer=sc)
    # NEW (Q3): outer bound during the runner's lock wait, staged grace/poll.
    runs = start("NEW_lock_wait_staged")
    summary_rows([(time.time(), "COOLDOWN 3600s")])
    fd = hold_lock()
    first = parent(runs, label, cmd, bound=4, REV2_NO_LOG="1")
    drop_lock(fd)
    no_summary()
    retry = parent(runs, label, cmd, REV2_NO_LOG="1")
    sc = score(runs, label)
    p = first["parent"] or {}
    ok = (p.get("admission") == "not_admitted" and p.get("timeout_cause") == "outer-timeout"
          and first["forge_log_last"] is None and (retry["parent"] or {}).get("admission") == "admitted"
          and (sc.get("line") or {}).get("passed") is True)
    record("NEW_lock_wait_outer_kill_staged", "Q3 (new)",
           "no runner row (TERM before supervise), not admitted, outer-timeout; retry passes", ok,
           first=first, retry=retry, scorer=sc)


def part_c():
    label = "candidate_focus"
    canon_sr = [*SR(), *CANON, "--", *INNER]
    # N5 (REV4): no-lock row, runner still alive at the outer bound.
    runs = start("N5")
    r = parent(runs, label, canon_sr, bound=3, REV2_MODE="nolock_row_hang")
    retry = parent(runs, label, canon_sr)
    sc = score(runs, label)
    ok = (r["parent"] is not None and not r["traceback"] and r["parent"]["timeout_cause"] == "outer-timeout"
          and r["parent"]["admission"] == "not_admitted" and (sc.get("line") or {}).get("passed") is True)
    record("N5_nolock_row_outer_exit", "R4-F03", "receipt, not admitted, outer-timeout; retry passes", ok,
           run=r, retry=retry, scorer=sc)
    # NEW (R4-F03): malformed runner rows / forge.log shapes at an outer-bound exit.
    rows = {
        "row_8_fields": repr(b"a\tb\tc\td\te\tf\tg\tran\n"),
        "row_rc_not_int": repr(b"t\tx\tl\thalted\t-\t0\t0\tran\tabc\n"),
        "row_invalid_utf8": repr(b"t\tx\tl\thalted\t-\t0\t0\tran\t\xff\xfe\n"),
        "row_ran_124_no_marker": repr(b"t\tx\tl\thalted\t-\t0\t0\tran\t124\n"),
        "row_unknown_outcome": repr(b"t\tx\tl\thalted\t-\t0\t0\tbogus\t143\n"),
    }
    fx = [str(TOOLS / "fixture-runner"), *CANON, "--", *INNER]
    for name, raw in list(rows.items()) + [("forge_log_is_directory", None)]:
        runs = start("C_" + name)
        extra = dict(REV5_FIX="dir") if raw is None else dict(REV5_FIX="row", REV5_ROW=raw)
        r = parent(runs, label, fx, bound=3, **extra)
        sc = score(runs, label)
        ok = (r["parent"] is not None and not r["traceback"] and not sc["traceback"] and sc["G5_summary_written"])
        record("NEW_" + name, "R4-F03/Q4 (new)", "parent receipt with a recorded reason; scorer no traceback", ok,
               run=r, scorer=sc)
    # N4 (REV4): truncated parent_result.json.
    def honest_then(name, mutate):
        runs = start(name)
        r = parent(runs, label, canon_sr)
        att = runs / label / "attempt-0001"
        mutate(att)
        retry = parent(runs, label, canon_sr)
        sc = score(runs, label)
        return r, retry, sc

    def trunc(att):
        t = (att / "parent_result.json").read_text()
        os.replace(att / "parent_result.json", att / "parent_result.full.json")
        (att / "parent_result.json").write_text(t[:len(t) // 2])
    r, retry, sc = honest_then("N4", trunc)
    ok = (not retry["traceback"] and retry["exit"] == 1 and isinstance(retry["stderr_json_line"], dict)
          and "error" in retry["stderr_json_line"] and not sc["traceback"] and (sc.get("line") or {}).get("passed") is False)
    record("N4_truncated_parent_result", "R4-F04", "JSON refusal, no traceback; scorer recorded reason", ok,
           first=r, retry=retry, scorer=sc)
    # NEW (R4-F04 / Q4): other malformed parent receipts.
    def deep(att):
        os.replace(att / "parent_result.json", att / "parent_result.full.json")
        (att / "parent_result.json").write_text("[" * 200000 + "]" * 200000)
    def as_list(att):
        os.replace(att / "parent_result.json", att / "parent_result.full.json")
        (att / "parent_result.json").write_text("[]\n")
    def bad_utf8(att):
        os.replace(att / "parent_result.json", att / "parent_result.full.json")
        (att / "parent_result.json").write_bytes(b'{"label": "\xff"}')
    for name, fn in (("parent_result_deep_nesting", deep), ("parent_result_json_list", as_list),
                     ("parent_result_invalid_utf8", bad_utf8)):
        r, retry, sc = honest_then("C_" + name, fn)
        ok = (not retry["traceback"] and retry["exit"] == 1 and isinstance(retry["stderr_json_line"], dict)
              and not sc["traceback"] and sc["G5_summary_written"] and (sc.get("line") or {}).get("passed") is False)
        record("NEW_" + name, "R4-F04/Q4 (new)", "JSON refusal; scorer writes a summary with a recorded reason", ok,
               retry=retry, scorer=sc)
    # NEW (R4-F04): parent killed between temp write and link: temp present, no receipt.
    runs = start("C_temp_without_link")
    att = runs / label / "attempt-0001"
    att.mkdir(parents=True)
    (att / "child_started.json").write_text('{"pid": 1}\n')
    (att / "parent_result.abc123.tmp").write_text('{"label": "candidate_focus"}\n')
    retry = parent(runs, label, canon_sr)
    sc = score(runs, label)
    ok = (retry["exit"] == 1 and not retry["traceback"] and "admitted" in retry["stderr"]
          and (sc.get("line") or {}).get("passed") is False and "incomplete" in json.dumps(sc.get("line")))
    record("NEW_temp_without_link", "R4-F04 (new)", "admitted incomplete; retry refused; scorer incomplete", ok,
           retry=retry, scorer=sc)
    # NEW (Q4): deep-nesting START.json.
    runs = start("C_start_deep")
    os.replace(runs / "START.json", runs / "START.full.json")
    (runs / "START.json").write_text("[" * 200000 + "]" * 200000)
    sc = score(runs, label)
    r = parent(runs, label, canon_sr)
    record("NEW_start_json_deep_nesting", "Q4 (new)", "scorer writes a summary with an error; parent JSON refusal",
           not sc["traceback"] and sc["G5_summary_written"] and not r["traceback"], scorer=sc, parent=r)
    # NEW (Q4): deep-nesting child_result.json on candidate_regression (regression_reasons path).
    runs = start("C_regression_deep_child")
    order = ["baseline_focus", "candidate_focus", "baseline_regression"]
    for l in order:
        ln = LINES[l]
        inner = ["timeout", "--kill-after=10s", "%ds" % ln["inner"], "python3", "run_gpu_tests.py", ln["arm"], ln["suite"]]
        pr = parent(runs, l, [*SR(), "--min", str(ln["minimum"]), "--max", str(ln["maximum"]), "--label", l, "--", *inner])
        assert pr["exit"] == 0, (l, pr)
    ln = LINES["candidate_regression"]
    inner = ["timeout", "--kill-after=10s", "%ds" % ln["inner"], "python3", "run_gpu_tests.py", ln["arm"], ln["suite"]]
    r = parent(runs, "candidate_regression", [str(TOOLS / "fixture-runner"), "--min", str(ln["minimum"]), "--max",
                                              str(ln["maximum"]), "--label", "candidate_regression", "--", *inner],
               REV5_FIX="deep_child")
    sc = score(runs, "candidate_regression")
    record("NEW_regression_deep_child_result", "Q4 (new)",
           "parent receipt with a recorded reason; scorer writes a summary (no traceback)",
           r["parent"] is not None and not r["traceback"] and not sc["traceback"] and sc["G5_summary_written"],
           run=r, scorer=sc)


def staged_argv():
    text = (MA8 / "STAGED_GPU_COMMANDS.sh").read_text()
    out = {}
    for m in re.finditer(r"^\s+/usr/bin/timeout (\d+)s /usr/bin/python3 run_gpu_line\.py (.+)$", text, re.M):
        argv = shlex.split(m.group(2).replace('"$GPU_RUN"', REAL_FORGE))
        out[argv[0]] = dict(script_timeout=int(m.group(1)), argv=argv)
    return out


def part_p():
    no_summary()
    staged = staged_argv()
    CTRL = dict(GN1_CONTROL="1", GN1_CPU_FIXTURE="1", PATH="/usr/bin:/bin")
    runs = start("P_staged_all_labels_real_runner", **CTRL)
    order = ["baseline_focus", "candidate_focus", "baseline_regression", "candidate_regression"] + \
            [l for l in LINES if l not in ("baseline_focus", "candidate_focus", "baseline_regression", "candidate_regression")]
    bad, timing = [], []
    for l in order:
        argv = staged[l]["argv"]
        assert argv[0] == l
        r = run([PY, "-B", WRAP, *argv], base_env(RUNS=str(runs), REV2_NO_LOG="1", **CTRL), timeout=staged[l]["script_timeout"])
        r = collect(runs, l, r)
        p = r["parent"] or {}
        if isinstance(p, dict) and p.get("start"):
            st = datetime.datetime.fromisoformat(p["start"]).timestamp()
            ce = datetime.datetime.fromisoformat(p["child_exited"]).timestamp()
            rc = datetime.datetime.fromisoformat(p["recorded"]).timestamp()
            timing.append(dict(label=l, prelaunch_s=round(st - r["launched"], 3), child_s=round(ce - st, 3),
                               post_exit_s=round(rc - ce, 3), parent_wall_s=r["wall_s"]))
        if r["exit"] != 0:
            bad.append(dict(label=l, exit=r["exit"], reasons=p.get("reasons") if isinstance(p, dict) else p,
                            stderr=r["stderr"][-400:]))
    sc = score(runs)
    summ = json.loads((runs / "G5_SUMMARY.json").read_text())
    line_fail = sorted(k for k, v in summ["lines"].items() if not v["passed"])
    modes = sorted({str(v.get("mode")) for v in summ["lines"].values()})
    record("P_staged_all_labels_real_runner", "Q5 (A8 G6/P126 under REV6)",
           "all 58 staged lines (argv exactly as written, declared CPU control) admitted and pass every line check; root refused only for absent device evidence",
           not bad and len(staged) == 58 and len(summ["lines"]) == 58 and not line_fail and sc["exit"] == 1
           and summ["errors"] == ["not a device run: every line requires positive child device evidence"],
           failed_labels=bad, labels=len(order), line_failures=line_fail, line_modes=modes,
           scorer=dict(exit=sc["exit"], G5_passed=sc["G5_passed"], errors=sc["errors"]), timing=timing,
           staged_lines_parsed=len(staged))
    # D5 runtime census and D6 premise-false controls (regression of earlier sound areas).
    label = "candidate_focus"
    for name, mode, want in (("D5_tamper", "tamper_runtime", False), ("D5_unsealed", "unsealed_runtime", True)):
        runs = start(name)
        r = parent(runs, label, [*SR(), *CANON, "--", *INNER], REV2_MODE=mode)
        sc = score(runs, label)
        record(name, "Q5 D5", "pass+listed" if want else "fails (tamper)",
               (r["parent"] or {}).get("passed") is want and (sc.get("line") or {}).get("passed") is want,
               run=r, scorer=sc)
    for l in ("E_npbool_g", "E_npbool_c"):
        ln = LINES[l]
        inner = ["timeout", "--kill-after=10s", "%ds" % ln["inner"], "python3", "run_gpu_tests.py", ln["arm"], ln["suite"]]
        opts = ["--min", str(ln["minimum"]), "--max", str(ln["maximum"]), "--label", l, "--", *inner]
        outs = {}
        for tag, mode in (("honest", "normal"), ("premise_false", "npbool_accepts")):
            runs = start("D6_%s_%s" % (tag, l))
            assert parent(runs, label, [*SR(), *CANON, "--", *INNER])["exit"] == 0
            outs[tag] = parent(runs, l, [*SR(), *opts], REV2_MODE=mode)
        ok = (outs["honest"]["exit"] == 0 and outs["premise_false"]["exit"] == 1
              and any("equivalence invalid" in x for x in (outs["premise_false"]["parent"] or {}).get("reasons", [])))
        record("D6_" + l, "Q5 D6", "honest passes; premise-false fails", ok, **outs)


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    parts = sys.argv[1:] or ["q", "n", "c", "p"]
    BASE.mkdir(parents=True)
    for d in (TOOLS, FORGE, OBS):
        d.mkdir(exist_ok=True)
    make_tools()
    for part in parts:
        try:
            globals()["part_" + part]()
        except Exception:  # recorded, never hidden
            import traceback
            record("PART_%s_harness_error" % part, "-", "harness completes", False, error=traceback.format_exc()[-3000:])
    redact = lambda s: s.replace(REAL_FORGE, "forge-gpu-run").replace(str(Path.home()), "$HOME")
    out = OUT / ("hostile_rev6_" + "".join(parts) + ".json")
    out.write_text(redact(json.dumps(dict(base=str(BASE), results=RESULTS), indent=1, default=str)) + "\n")
    print("unexpected:", [r["case"] for r in RESULTS if not r["outcome_as_expected"]])


if __name__ == "__main__":
    main()
