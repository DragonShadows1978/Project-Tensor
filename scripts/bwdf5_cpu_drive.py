"""BWD-F5 A4 CPU drive of the repaired P2 harness (no GPU: refuses to run unless CUDA_VISIBLE_DEVICES is empty and no
forge-gpu-run lease is set; never runs compute-sanitizer; never calls the step's parent main).

  python3 -B -P scripts/bwdf5_cpu_drive.py        -> artifacts/bwdf1/f5/CPU_DRIVE.json (scratch under /tmp/bwdf5_drive_*)

Cases (every one asserts an expected outcome; the run exits 1 on any miss):
  A  the step's CPU path end to end, sanitizer replaced by a plain run: every row of every mode (canary, e1, e2, e3) has
     its child (the real `scripts/bwdf5_step_p2.py --child`) started; each refuses cleanly at its lease check on this
     CPU lane (exit 1, the refusal text on stderr, before any CUDA) and the parent records one row per case-table id.
  B  the sanitizer command line the step builds (argv captured, nothing run).
  C  constructed child ends under a stand-in sanitizer (scripts/bwdf5_drive_stub.py): ok, dead (exit 255 + the
     sanitizer's no-API-call log: BWD-F3's failure), SIGSEGV, hang (timeout, process group killed), errors log, empty
     log, missing log, no ROW line, exit 0 with no row; and the verdicts read from the written files.
  D  the line budget (a fake clock): rows after the budget are NOT-RUN; verdict FAIL.
  E  SIGTERM to a step-like process: the running child's process group is taken down (no orphan).
  F  the pre-lease gate: passes on the real arms; refuses a write-once path that exists.
Prior art: no prior art known to me (a test harness for BWD-F5's runner); the stand-ins are plain test doubles.
"""
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import bwdf5_gpu_common as G  # noqa: E402
import bwdf5_gpu_lines as L  # noqa: E402
import bwdf5_gpu_post as P  # noqa: E402
import bwdf5_p2_run as R  # noqa: E402

STEP = HERE / "bwdf5_step_p2.py"
STUB = HERE / "bwdf5_drive_stub.py"
CHECKS = []


def ex(cond, what, detail=""):
    CHECKS.append({"check": what, "ok": bool(cond), "detail": str(detail)[:300]})
    print(("ok   " if cond else "FAIL ") + what + (f"  [{detail}]" if not cond and detail else ""), flush=True)


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        return pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except OSError:
        return False


def real_child_argv(mode, arm, out):
    return lambda rid: [sys.executable, "-B", "-P", str(STEP), "--child", mode, arm, rid, str(out)]


def stub_child_argv(mode, arm, out):
    return lambda rid: [sys.executable, "-B", "-P", str(STUB), "child", mode, arm, rid, str(out)]


STUB_SAN = [sys.executable, "-B", "-P", str(STUB), "sanitizer", "--tool", "memcheck"]


def write_line(gpu_root, line, rows, arrays=None):
    d = pathlib.Path(gpu_root) / line
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps({"line": line, "rows": rows}))
    return d


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("FORGE_GPU_LEASE") is not None:
        raise SystemExit("CPU drive only: CUDA_VISIBLE_DEVICES must be set and empty and FORGE_GPU_LEASE unset")
    scratch = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / f"bwdf5_drive_{int(time.time())}"
    scratch.mkdir(parents=True)
    env = dict(os.environ, BWDF5_STUB="{}")

    # ---- A: the CPU path end to end, plain run (no sanitizer), real children refuse at the lease check
    total = 0
    for mode in R.LINE_MODES:
        out = scratch / f"A_{mode}"
        out.mkdir()
        rows = R.run_rows(mode, "F4", out, None, real_child_argv(mode, "F4", out), env=env)
        want = [rid for rid, _ in R.rows_for(mode)]
        total += len(rows)
        ex([r["id"] for r in rows] == want, f"A {mode}: one recorded row per case-table id ({len(want)})")
        ex(all(r["result"] == "CHILD-DIED" and r["child_exit"] == 1 and r["child_end"] == {"kind": "exit", "code": 1}
               for r in rows), f"A {mode}: every child started and refused with exit 1",
           [(r["id"], r["result"], r["child_exit"]) for r in rows if r["child_exit"] != 1][:3])
        ex(all("forge-gpu-run lease" in r["error"] or "FORGE_GPU_LEASE" in r["error"] for r in rows),
           f"A {mode}: every row carries the child's refusal text", rows[0]["error"][:200])
        ex(all(r["memcheck_state"] == "NOT-REQUESTED" and "memcheck_log" not in r for r in rows),
           f"A {mode}: plain run has no sanitizer log")
        ex(not list(out.glob("*.npz")) and not list(out.glob("*.row.json")), f"A {mode}: no CUDA work, no row files")
        gp = scratch / f"A_gpu_{mode}"
        write_line(gp, f"p2_{mode}_F4", rows)
        P.GPU = gp
        v = P.line_verdict(f"p2_{mode}_F4", {"rows": rows}, {}, {})
        ex(v["verdict"] == "FAIL" and v["harness_incomplete"], f"A {mode}: verdict on dead children is FAIL", v["verdict"])
    ex(total == 26, "A: 26 rows in all (canary 1, e1 11, e2 5, e3 9)", total)

    # ---- B: the sanitizer command line (argv captured; run_one replaced; nothing runs)
    seen = []
    real_run_one = R.run_one
    R.run_one = lambda rid, kind, argv, env_, out, log, t: (seen.append((argv, str(log), t)) or
                                                            {"id": rid, "kind": kind, "result": "RETURNED"})
    try:
        out = scratch / "B"
        out.mkdir()
        R.run_rows("canary", "F4", out, R.SANITIZER, real_child_argv("canary", "F4", out), env=env)
    finally:
        R.run_one = real_run_one
    argv, log, t = seen[0]
    ex(argv[:3] == ["compute-sanitizer", "--tool", "memcheck"] and argv[3] == "--log-file" and argv[4] == log
       and log == str(out / "e1-control-rope_apply-covering.memcheck.log"), "B: sanitizer prefix and per-row log", argv[:5])
    ex(argv[5:8] == [sys.executable, "-B", "-P"] and argv[8].endswith("bwdf5_step_p2.py") and argv[9] == "--child"
       and argv[10:12] == ["canary", "F4"] and "--target-processes" not in argv, "B: the child follows the log, no "
       "--target-processes", argv[5:])
    ex(t <= R.ROW_TIMEOUT_S, "B: row timeout bounded", t)

    # ---- C: constructed child ends under the stand-in sanitizer
    rows_e1 = [rid for rid, _ in R.rows_for("e1")]
    beh = {rows_e1[0]: {"log": "ok", "child": "ok"},
           rows_e1[1]: {"log": "noapi", "child": "exit255"},
           rows_e1[2]: {"log": "ok", "child": "segv"},
           rows_e1[3]: {"log": "ok", "child": "hang"},
           rows_e1[4]: {"log": "errors", "child": "ok"},
           rows_e1[5]: {"log": "empty", "child": "ok"},
           rows_e1[6]: {"log": "missing", "child": "ok"},
           rows_e1[7]: {"log": "ok", "child": "norow"},
           rows_e1[8]: {"log": "ok", "child": "exit0norow"}}
    env_c = dict(env, BWDF5_STUB=json.dumps(beh))
    out = scratch / "C_e1"
    out.mkdir()
    rows = R.run_rows("e1", "F4", out, STUB_SAN, stub_child_argv("e1", "F4", out), env=env_c, row_timeout_s=4)
    by = {r["id"]: r for r in rows}
    r0, r1, r2, r3, r4, r5, r6, r7, r8 = (by[x] for x in rows_e1[:9])
    ex(r0["result"] == "RETURNED" and r0["child_exit"] == 0 and r0["memcheck_state"] == "SUMMARY"
       and r0["memcheck_errors"] == 0 and len(r0["memcheck_log_sha256"]) == 64, "C: ok child, 0-error log", r0)
    ex(r1["result"] == "CHILD-DIED" and r1["child_exit"] == 255 and r1["memcheck_state"] == "NO-API-CALL"
       and "terminated before first instrumented API call" in r1["error"], "C: dead child exit 255 + no-API log "
       "(BWD-F3's failure) recorded with the log head", r1)
    ex(r2["result"] == "CHILD-DIED" and r2["child_end"].get("signal") == "SIGSEGV" and r2["child_exit"] == -11,
       "C: SIGSEGV child recorded as a signal end", r2)
    ex(r3["result"] == "CHILD-TIMEOUT" and r3["child_exit"] == 124 and r3["child_end"] == {"kind": "timeout"},
       "C: hanging child recorded as a timeout", r3)
    pid = int(pathlib.Path(out / f"{rows_e1[3]}.pid").read_text())
    time.sleep(0.5)
    ex(not alive(pid), "C: the timed-out child's process group is gone (no orphan)", pid)
    ex(r4["result"] == "RETURNED" and r4["memcheck_state"] == "SUMMARY" and r4["memcheck_errors"] == 2,
       "C: errors log counts 2", r4)
    ex(r5["memcheck_state"] == "EMPTY" and r5["memcheck_errors"] is None, "C: empty log", r5)
    ex(r6["memcheck_state"] == "MISSING" and r6["memcheck_log_sha256"] is None, "C: missing log", r6)
    ex(r7["result"] == "CHILD-DIED" and r7["child_exit"] == 1 and "stderr" in r7["error"], "C: no ROW line, exit 1", r7)
    ex(r8["result"] == "CHILD-DIED" and r8["child_exit"] == 0 and "child exit 0" in r8["error"],
       "C: exit 0 with no row is still a dead child", r8)
    gp = scratch / "C_gpu"
    d = write_line(gp, "p2_e1_F4", rows)
    for rid in rows_e1[:9]:
        src = out / f"{rid}.memcheck.log"
        if src.exists():
            (d / src.name).write_text(src.read_text())
    P.GPU = gp
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        P.post_line("p2_e1_F4")
    post = json.loads((gp / "p2_e1_F4.post.json").read_text())
    ex(post["verdict"] == "FAIL" and post["memcheck_errors"] == 2, "C: post_line reads the row logs from the files", post)
    states = post["memcheck_row_states"]
    ex(states[rows_e1[1]] == "NO-API-CALL" and states[rows_e1[5]] == "EMPTY" and states[rows_e1[6]] == "MISSING"
       and states[rows_e1[0]] == "SUMMARY", "C: per-row log states in the verdict file", states)
    ex(any(rows_e1[1] in h for h in post["harness_incomplete"]) and any(rows_e1[3] in h for h in
                                                                        post["harness_incomplete"]),
       "C: dead and timed-out rows are listed as harness-incomplete", post["harness_incomplete"][:3])
    for arm, tail in (("F2", "harness incomplete"), ("B", "harness incomplete")):
        with contextlib.redirect_stdout(io.StringIO()):
            d2 = write_line(gp, f"p2_e1_{arm}", rows)
            for f in d.glob("*.memcheck.log"):
                (d2 / f.name).write_text(f.read_text())
            P.post_line(f"p2_e1_{arm}")
        pv = json.loads((gp / f"p2_e1_{arm}.post.json").read_text())
        ex(pv["verdict"].endswith(tail) and pv["verdict"].startswith("RECORDED"), f"C: {arm} with unusable rows is "
           f"recorded as {tail}", pv["verdict"])
    # a canary that works (the stand-in child and a 0-error log): PASS from the files; one that dies: FAIL
    for tag, b, want in (("ok", {}, "PASS"), ("dead", {R.CANARY_ROWS[0][0]: {"log": "noapi", "child": "exit255"}}, "FAIL"),
                         ("errors", {R.CANARY_ROWS[0][0]: {"log": "errors"}}, "FAIL")):
        o = scratch / f"C_canary_{tag}"
        o.mkdir()
        rows = R.run_rows("canary", "F4", o, STUB_SAN, stub_child_argv("canary", "F4", o),
                          env=dict(env, BWDF5_STUB=json.dumps(b)), row_timeout_s=4)
        g = scratch / f"C_gpu_canary_{tag}"
        d3 = write_line(g, "p2_canary_F4", rows)
        for f in o.glob("*.memcheck.log"):
            (d3 / f.name).write_text(f.read_text())
        P.GPU = g
        with contextlib.redirect_stdout(io.StringIO()):
            P.post_line("p2_canary_F4")
        pv = json.loads((g / "p2_canary_F4.post.json").read_text())
        ex(pv["verdict"] == want, f"C: canary {tag} -> {want}", pv)

    # ---- D: the line budget (fake clock): rows after the budget are NOT-RUN
    ticks = iter(range(0, 10 ** 6, 100))
    out = scratch / "D"
    out.mkdir()
    rows = R.run_rows("e3", "F4", out, STUB_SAN, stub_child_argv("e3", "F4", out), env=dict(env, BWDF5_STUB="{}"),
                      budget_s=250, clock=lambda: next(ticks))
    ran = [r for r in rows if r["result"] != "NOT-RUN (line budget spent)"]
    ex(len(rows) == 9 and len(ran) == 2 and all(r["child_exit"] is None for r in rows[2:]), "D: 2 rows run, 7 NOT-RUN",
       [r["result"] for r in rows])
    v = P.line_verdict("p2_e3_F4", {"rows": rows}, {}, {})
    ex(v["verdict"] == "FAIL", "D: NOT-RUN rows fail the line", v["verdict"])

    # ---- E: SIGTERM to a step-like process takes the running child's group down
    out = scratch / "E"
    out.mkdir()
    helper = (f"import sys,signal,json,os\nsys.path.insert(0,{str(HERE)!r})\nimport bwdf5_p2_run as R\n"
              f"signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))\n"
              f"out={str(out)!r}\n"
              f"argv=lambda rid:[sys.executable,'-B','-P',{str(STUB)!r},'child','canary','F4',rid,out]\n"
              f"R.run_rows('canary','F4',out,[sys.executable,'-B','-P',{str(STUB)!r},'sanitizer','--tool','memcheck'],argv)\n")
    first = R.CANARY_ROWS[0][0]
    hp = subprocess.Popen([sys.executable, "-B", "-P", "-c", helper],
                          env=dict(env, BWDF5_STUB=json.dumps({first: {"child": "hang"}})), start_new_session=True)
    for _ in range(100):
        if (out / f"{first}.pid").exists():
            break
        time.sleep(0.1)
    cpid = int((out / f"{first}.pid").read_text())
    ex(alive(cpid), "E: the hanging child is running before SIGTERM", cpid)
    hp.send_signal(signal.SIGTERM)
    rc = hp.wait(timeout=30)
    time.sleep(0.5)
    ex(rc == 143 and not alive(cpid), "E: step exits 143 and its child group is gone", (rc, cpid))

    # ---- F: the pre-lease gate (patched ROOT for the refusal; the real arms' sha256 are checked either way)
    ex(L.gate("p2_canary_F4") == 0 and L.gate("p2_e1_B") == 0 and L.gate("p2_e3_F2") == 0, "F: gate passes on B, F2, F4")
    fake = scratch / "F_root"
    (fake / L.lines()["p2_e2_F4"]["out"]).mkdir(parents=True)
    real_root = L.ROOT
    L.ROOT = fake
    try:
        rc = L.gate("p2_e2_F4")
        rc_other = L.gate("p2_e2_B")
    finally:
        L.ROOT = real_root
    ex(rc == 3 and rc_other == 0, "F: gate refuses an existing write-once directory (and only that line)", (rc, rc_other))
    ex(not (ROOT / "artifacts/bwdf1/f5/gpu").exists(), "F: nothing under artifacts/bwdf1/f5/gpu was created")

    bad = [c for c in CHECKS if not c["ok"]]
    doc = {"checks": len(CHECKS), "failed": [c["check"] for c in bad], "scratch": str(scratch).replace(
        os.path.expanduser("~"), "$HOME"), "results": CHECKS}
    (ROOT / "artifacts/bwdf1/f5/CPU_DRIVE.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({"checks": len(CHECKS), "failed": doc["failed"]}))
    print("CPU-DRIVE", "PASS" if not bad else "FAIL")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
