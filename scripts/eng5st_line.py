"""ENG5-S-T GPU lines: staging (the dry runs) and the lead's runner. Nothing here touches the GPU or
CUDA_VISIBLE_DEVICES.

  python3 -B -P scripts/eng5st_line.py stage        artifacts/eng5s/t/GPU_LINES_T.json + one FORGE_GPU_DRY_RUN=1 dry run
                                                    per line (expected exit 78; L79)
  python3 -B -P scripts/eng5st_line.py run <arm>    (the lead) refuses BEFORE the lease if the arm's line directory
                                                    exists or its module's sha256 differs from its receipt (L103); then
                                                    runs the line's forge command from the worktree root, retrying
                                                    unchanged after 300 s on forge exit 75/76 (nothing ran), at most 6
                                                    retries; writes artifacts/eng5s/t/gpu_run/<arm>/RESULT_T.json
                                                    (forge exit, inner exit, timed_out = exit 124 is never a pass,
                                                    attempts, junit sha256, the semantic VERDICT: L45)

The forge command of each line, as the order writes it:
  forge-gpu-run --min 120 --max 540 --label eng5st-<arm> -- timeout 480s python3 -B -P scripts/eng5st_gpu_step.py <arm>
No line sets RLIMIT_AS / ulimit -v / prlimit; no line runs compute-sanitizer. Arms: S, B, s01, s02, s03, s04.

Prior art: ENG5-S's scripts/eng5s_gpu_line.py (this repo), whose stage/run/RESULT layout and 75/76 retry rule this copies;
timeout(1); the house forge-gpu-run contract. No other prior art known to me.
"""
import datetime
import json
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import eng5st_common as cm

DRY_DIR = os.path.join(cm.T_DIR, "gpu_dry")
RUN_DIR = os.path.join(cm.T_DIR, "gpu_run")
EXPECTED = {"S": "PASS", "B": "DEFECT", "s01": "KILLED", "s02": "KILLED", "s03": "KILLED", "s04": "KILLED"}
KILLS = {"s01": "clone", "s02": "to_d2h and to_h2d (both directions of `to`)", "s03": "zeros", "s04": "from_host"}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def stage():
    out_path = os.path.join(cm.T_DIR, "GPU_LINES_T.json")
    if os.path.exists(out_path):
        raise SystemExit("GPU_LINES_T.json exists; each line is dry-run once")
    os.makedirs(DRY_DIR, exist_ok=True)
    rows = []
    for arm in cm.ARMS:
        argv = cm.forge_argv(arm)
        ok, got, want = cm.module_check(arm)
        env = dict(os.environ, FORGE_GPU_DRY_RUN="1")  # the caller's environment otherwise unchanged (L58)
        t0 = now()
        p = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=120, cwd=ROOT)
        json.dump({"arm": arm, "time": t0, "argv": argv, "exit": p.returncode, "stdout": p.stdout[-3000:],
                   "stderr": p.stderr[-3000:], "module_check": {"ok": ok, "got": got, "want": want}},
                  open(os.path.join(DRY_DIR, f"{arm}.json"), "w"), indent=1)
        rows.append({"arm": arm, "package": os.path.relpath(cm.pkg(arm), ROOT), "module_sha256": want,
                     "module_check_at_staging": "ok" if ok else {"got": got, "want": want},
                     "forge_command": " ".join(shlex.quote(a) for a in argv), "min_s": 120, "max_s": 540,
                     "inner_timeout_s": 480, "expected_verdict": EXPECTED[arm],
                     "expected_kill_rows": KILLS.get(arm),
                     "lead_command": "python3 -B -P scripts/eng5st_line.py run " + arm,
                     "write_once_paths": [f"artifacts/eng5s/t/gpu/{arm}/ (made by the step)",
                                          f"artifacts/eng5s/t/gpu_run/{arm}/ (made by the runner)",
                                          f"/tmp/eng5st_tmp/pt/gpu_{arm} (pytest basetemp)"],
                     "dry_run_exit": p.returncode, "dry_run_receipt": f"artifacts/eng5s/t/gpu_dry/{arm}.json"})
        print(arm, "dry-run exit", p.returncode, "module check", "ok" if ok else (got, want))
    json.dump({
        "written_utc": now(), "order": "orders/ENG5ST_REGRESSION_TEST.md (T2)",
        "cwd": "this worktree's root (tensor_fork_apa_wt_eng5s)",
        "environment": "the lead's environment with the GPU visible; no ENG5-S-T script removes or overrides "
                       "CUDA_VISIBLE_DEVICES (the step refuses with exit 7 if it is empty)",
        "lease_check": "scripts/eng5st_gpu_step.py:2 (forge-gpu-lease-check --require-held); the test file itself "
                       "calls no lease check (its children run inside the step's lease)",
        "pre_lease_check": "scripts/eng5st_line.py run <arm> (module sha256 against the receipt, write-once "
                           "directory) runs before the forge command; the step repeats both inside the lease "
                           "(scripts/eng5st_common.py run_arm) before it starts pytest",
        "post_device_steps": "scripts/eng5st_verdict.py judge (written by the step as VERDICT.json); T3 drive: "
                             "artifacts/eng5s/t/T3_VERDICT_DRIVE.json",
        "expected": {"S": "PASS (all eleven GPU rows pass)", "B": "DEFECT (all ten injected rows fail; axpy_empty passes)",
                     "s01": "KILLED (clone only)", "s02": "KILLED (to_d2h and to_h2d)", "s03": "KILLED (zeros)",
                     "s04": "KILLED (from_host)"},
        "semantic_verdict": "each line's VERDICT.json decides it; RESULT_T.json ran_clean only records the exits; "
                            "exit 124 is never a pass (L45)",
        "multi_device": "NOT COVERED (one GPU)",
        "lines": rows}, open(out_path, "w"), indent=1)


def run(arm):
    d = cm.line_dir(arm)
    if os.path.exists(d):
        print("REFUSED:", d, "exists (write-once; move it aside to retry)")
        return 5
    ok, got, want = cm.module_check(arm)
    if not ok:
        print("REFUSED (L103) before the lease:", json.dumps({"arm": arm, "found": got, "expected": want}))
        return 3
    rd = os.path.join(RUN_DIR, arm)
    os.makedirs(rd)
    argv = cm.forge_argv(arm)
    env = {k: v for k, v in os.environ.items() if k != "FORGE_GPU_DRY_RUN"}
    env.setdefault("TMPDIR", cm.TMP)
    os.makedirs(env["TMPDIR"], exist_ok=True)
    log = os.path.join(rd, "run.log")
    json.dump({"time": now(), "argv": argv, "cwd": ROOT, "module_sha256": want}, open(os.path.join(rd, "START.json"), "w"),
              indent=1)
    attempts = []
    for attempt in range(7):
        t0 = now()
        with open(log, "ab") as lf:
            lf.write(f"\n=== attempt {attempt} {t0}\n".encode())
            lf.flush()
            rc = subprocess.run(argv, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        attempts.append({"attempt": attempt, "start": t0, "end": now(), "forge_exit": rc})
        json.dump(attempts, open(os.path.join(rd, "ATTEMPTS.json"), "w"), indent=1)
        if rc not in (75, 76):
            break
        if attempt < 6:
            time.sleep(300)
    ip = os.path.join(d, "INNER_EXIT")
    inner = int(open(ip).read().strip()) if os.path.exists(ip) else None
    final = attempts[-1]["forge_exit"]
    vp, jp = os.path.join(d, "VERDICT.json"), os.path.join(d, "junit.xml")
    verdict = json.load(open(vp)) if os.path.exists(vp) else None
    res = {"arm": arm, "end": now(), "forge_exit": final, "inner_exit": inner, "timed_out": final == 124 or inner == 124,
           "attempts": attempts, "log": os.path.relpath(log, ROOT), "log_sha256": cm.sha256_file(log),
           "junit_sha256": cm.sha256_file(jp) if os.path.exists(jp) and os.path.getsize(jp) else None,
           "verdict": verdict, "semantic_verdict": verdict["verdict"] if verdict else None,
           "expected_verdict": EXPECTED[arm], "ran_clean": final == 0 and inner == 0,
           "as_expected": bool(verdict) and verdict["verdict"] == EXPECTED[arm] and final == 0 and inner == 0}
    json.dump(res, open(os.path.join(rd, "RESULT_T.json"), "w"), indent=1)
    print(json.dumps(res, indent=1)[:4000])
    return 0 if res["as_expected"] else 1


if __name__ == "__main__":
    if sys.argv[1] == "stage":
        stage()
    elif sys.argv[1] == "run":
        sys.exit(run(sys.argv[2]))
