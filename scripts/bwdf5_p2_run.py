"""BWD-F5 P2 row runner (library: no lease check, no engine import, no CUDA call; the step script
scripts/bwdf5_step_p2.py imports it after its own `forge-gpu-lease-check --require-held` line, and the CPU drive
scripts/bwdf5_cpu_drive.py imports it directly).

What BWD-F3 did wrong (artifacts/bwdf1/f5/DIAGNOSIS.md): the whole step ran under
`compute-sanitizer --tool memcheck --target-processes all --log-file <one log>`; every row's child died with exit 255
and empty stderr. Here the step runs OUTSIDE the sanitizer and each row's child runs under its OWN single-target
`compute-sanitizer --tool memcheck --log-file <row log> python3 -B -P <step> --child ...` (the sanitizer's default
`--target-processes application-only`, the mode BWD-F3's six working memcheck lines used). The parent records how each
child ended (exit code, signal, timeout) as a row, and the verdict reads each row's own memcheck log.

Prior art: compute-sanitizer's `--log-file` and `--target-processes` options (NVIDIA Compute Sanitizer
documentation); one process per test row (the pytest-forked / pytest-xdist isolation style); process-group kill of a
timed-out child tree (POSIX setsid/killpg idiom); scripts/bwdf3_step_p2.py (this repo), whose loop this follows. Ours:
the per-row log, the end-of-child record and the log classification below.
"""
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bwdf5_p2_cases as C  # noqa: E402

LINE_MODES = ("e1", "e2", "e3", "canary")  # `count` exited 0 in BWD-F3's run (no sanitizer) and is not restaged
CANARY_ROWS = [("e1-control-rope_apply-covering", "control")]  # one control row, shown first on Build F4
SANITIZER = ["compute-sanitizer", "--tool", "memcheck"]  # + ["--log-file", <row log>]; default application-only
LINE_BUDGET_S = 470  # inside the line's inner `timeout 500s`: a slow row cannot take the whole line
ROW_TIMEOUT_S = 120
NO_API = "Target application terminated before first instrumented API call"


def rows_for(mode):
    return CANARY_ROWS if mode == "canary" else C.rows_for(mode)


def case_mode(mode):
    return "e1" if mode == "canary" else mode


def sha256_file(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def classify_log(text):
    """Classify one row's memcheck log: (state, errors). state: SUMMARY (errors = the summary's N, summed over the
    log's ERROR SUMMARY lines), NO-API-CALL (the sanitizer's 'terminated before first instrumented API call'),
    EMPTY, MISSING (text None), UNPARSED (text without either). Only SUMMARY carries a count; a count of 0 with any
    other state is never reported."""
    import re
    if text is None:
        return "MISSING", None
    if not text.strip():
        return "EMPTY", None
    if NO_API in text:
        return "NO-API-CALL", None
    ns = [int(m) for m in re.findall(r"ERROR SUMMARY:\s+(\d+)\s+error", text)]
    return ("SUMMARY", sum(ns)) if ns else ("UNPARSED", None)


def child_end(returncode, timed_out):
    """How a child ended: a dict {kind: exit|signal|timeout, code/signal}."""
    if timed_out:
        return {"kind": "timeout"}
    if returncode is not None and returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = str(-returncode)
        return {"kind": "signal", "signal": name, "code": returncode}
    return {"kind": "exit", "code": returncode}


def _kill_group(p):
    """Kill the child's process group (the child was started with start_new_session, so it leads its own group; the
    sanitizer and its target share it). Only ever called on a process this runner started."""
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_one(rid, kind, argv, env, out, log_path, timeout_s):
    """Run one row's child (argv already carries the sanitizer prefix when there is one) and return its row record.
    The child's stdout and stderr go to files (<rid>.stdout.txt / <rid>.stderr.txt in `out`), stdin is /dev/null: the
    same I/O shape as BWD-F3's six memcheck lines that worked (their step's streams were files), with no pipe between
    the sanitized process and this one (one fewer difference from the working configuration)."""
    row_file = pathlib.Path(out) / f"{rid}.row.json"
    f_out, f_err = pathlib.Path(out) / f"{rid}.stdout.txt", pathlib.Path(out) / f"{rid}.stderr.txt"
    timed_out = False
    with open(f_out, "wb") as fo, open(f_err, "wb") as fe:
        p = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=fo, stderr=fe, start_new_session=True)
        try:
            p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(p)
            p.wait()
        except BaseException:  # the step was told to stop: take the child tree down with it
            _kill_group(p)
            raise
        finally:
            if p.poll() is None:
                _kill_group(p)
                p.wait()
    so, se = f_out.read_text(errors="replace"), f_err.read_text(errors="replace")
    end = child_end(p.returncode, timed_out)
    got = [l[4:] for l in (so or "").splitlines() if l.startswith("ROW ")]
    row = None
    if row_file.exists():
        try:
            row = json.loads(row_file.read_text())
        except ValueError:
            row = None
    if row is None and got:
        try:
            row = json.loads(got[-1])
        except ValueError:
            row = None
    log_text = None
    if log_path is not None and pathlib.Path(log_path).exists():
        log_text = pathlib.Path(log_path).read_text(errors="replace")
    state, n = classify_log(log_text) if log_path is not None else ("NOT-REQUESTED", None)
    if timed_out:
        row = {"id": rid, "result": "CHILD-TIMEOUT"}
        code = 124
    else:
        code = p.returncode
        if row is None or code != 0:
            note = (f"child {end['kind']} {end.get('signal') or end.get('code')}: stderr {(se or '')[-800:]!r}"
                    f" stdout {(so or '')[-200:]!r}")
            if state not in ("SUMMARY", "NOT-REQUESTED") and log_text:
                note += f" | memcheck log head {log_text[:400]!r}"
            prior = row or {}
            if prior.get("result"):
                note += f" | the child had recorded result {prior['result']!r}"
            row = {"id": rid, "result": "CHILD-DIED", "error": note}
            if "module_sha256" in prior:
                row["module_sha256"] = prior["module_sha256"]
    row.update(kind=kind, child_exit=code, child_end=end, memcheck_state=state, memcheck_errors=n)
    if log_path is not None:
        row["memcheck_log"] = pathlib.Path(log_path).name
        row["memcheck_log_sha256"] = sha256_file(log_path) if log_text is not None else None
    return row


def run_rows(mode, arm, out, sanitizer, child_argv, env=None, budget_s=LINE_BUDGET_S, row_timeout_s=ROW_TIMEOUT_S,
             clock=time.monotonic, emit=None):
    """Run every row of `mode` in its own child. `sanitizer`: argv prefix before `--log-file <row log>` (None = plain
    run, the CPU drive's replacement). `child_argv(rid)` returns the child's argv. Returns the row records."""
    env = dict(os.environ) if env is None else env
    rows = []
    t_end = clock() + budget_s
    for rid, kind in rows_for(mode):
        left = t_end - clock()
        if left < 5:
            rows.append({"id": rid, "kind": kind, "result": "NOT-RUN (line budget spent)", "child_exit": None})
        else:
            log = None if sanitizer is None else pathlib.Path(out) / f"{rid}.memcheck.log"
            argv = (list(sanitizer) + ["--log-file", str(log)] if sanitizer is not None else []) + child_argv(rid)
            rows.append(run_one(rid, kind, argv, env, out, log, min(row_timeout_s, left)))
        if emit:
            emit(rows[-1])
    return rows
