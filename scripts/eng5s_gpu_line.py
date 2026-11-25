"""ENG5-S GPU lines: staging (dry runs) and the lead's runner (copied from the eng5 branch's
scripts/eng5_f2_gpu_line.py, read-only there; ENG5-S arms, lines and receipts). Nothing here touches the GPU or
CUDA_VISIBLE_DEVICES.

  python3 -B -P scripts/eng5s_gpu_line.py stage          artifacts/eng5s/GPU_LINES.json + one FORGE_GPU_DRY_RUN=1 dry
                                                         run per line (expected exit 78; L79)
  python3 -B -P scripts/eng5s_gpu_line.py run <line>     (the lead) refuses before the lease if
                                                         artifacts/eng5s/gpu/<line>/ exists or a module differs from its
                                                         receipt (L103); then runs the line's forge command, retrying
                                                         unchanged after 300 s on forge exit 75/76 (nothing ran), at
                                                         most 6 retries; writes RESULT.json (forge exit, inner exit,
                                                         timed_out = 124 is never a pass, attempts, log sha256, the
                                                         semantic VERDICT: L45).

Each forge command: forge-gpu-run --min <s> --max 540 --label eng5s-<line> -- timeout <s>s python3 -B -P
scripts/eng5s_gpu_step.py <line>. The step script's line 2 is the held-lease check (forge-gpu-lease-check
--require-held). No line sets RLIMIT_AS / ulimit -v / prlimit (L114); the lead runs each in a scope with
MemoryMax=12G, OOMPolicy=continue. Per-arm packages are made by scripts/eng5s_arms.py pkgs.

Prior art: timeout(1); the house forge-gpu-run contract; ENG4/ENG5's GPU line runners (this repo), whose
run/stage/RESULT layout this copies. No other prior art known to me.
"""
import datetime
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SO = "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
PY = ["python3", "-B", "-P"]
GPU_DIR = os.path.join(ROOT, "artifacts/eng5s/gpu")
DRY_DIR = os.path.join(ROOT, "artifacts/eng5s/gpu_dry")
SHORT = {"B": "B", "S": "S", "s01": "s01_clone_status_dropped", "s02": "s02_to_status_dropped",
         "s03": "s03_zeros_status_dropped", "s04": "s04_from_host_last_error_only"}
HOOKS = {"B": "B_H", "S": "S_H"}
SUITE_CHUNKS = 12
LINES = {}  # line -> (min s, inner timeout s, arms, expected peak host bytes, description)
for short in ("B", "S", "s01", "s02", "s03", "s04"):
    arms = [SHORT[short]] + ([HOOKS[short]] if short in HOOKS else [])
    LINES[f"g3_{short}"] = (120, 480, arms, 3 << 30,
                            f"G2 failure injection on {SHORT[short]}: the nine sites (to in both directions, the OOM "
                            "drain through cudaMalloc and cudaMallocAsync), each after its uninjected success control, "
                            "plus axpy_ on empty operands; one child per case"
                            + (f"; site 9 on {HOOKS[short]}" if short in HOOKS else "; site 9 not applicable"))
for arm in ("B", "S"):
    for i in range(1, SUITE_CHUNKS + 1):
        LINES[f"suite_{arm}_{i:02d}"] = (480, 500, [arm], 6 << 30,
                                         f"G3 production suite on {arm}, chunk {i} of at most {SUITE_CHUNKS} (330 s "
                                         "test budget; exits at once when the series is complete)")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def receipt_sha(arm):
    return json.load(open(os.path.join(ROOT, f"artifacts/eng5s/build/BUILD_{arm}.json")))["module_sha256"]


def pkg(arm):
    return os.path.join(ROOT, "tensor_cuda/build-eng5s", arm, "pkg")


def module_checks(line):
    bad = {}
    for arm in LINES[line][2]:
        want = receipt_sha(arm)
        paths = [os.path.join(ROOT, "tensor_cuda/build-eng5s", arm, "module", SO)]
        if not line.startswith("suite_"):
            paths.append(os.path.join(pkg(arm), "tensor_cuda", SO))
        for p in paths:
            got = sha(p) if os.path.exists(p) else None
            if got != want:
                bad[p] = {"expected": want, "found": got}
    return bad


def forge_argv(line):
    mn, inner, _, _, _ = LINES[line]
    return ["forge-gpu-run", "--min", str(mn), "--max", "540", "--label", f"eng5s-{line}", "--",
            "timeout", f"{inner}s"] + PY + ["scripts/eng5s_gpu_step.py", line]


def write_once(line):
    paths = [f"artifacts/eng5s/gpu/{line}/"]
    if line.startswith("suite_"):
        arm = line.split("_")[1]
        paths += [f"artifacts/eng5s/h3/gpu_{arm}/part_<n>/ (one per chunk that runs)",
                  f"/tmp/eng5s_tmp/pt/gpu_{arm}_<n> (pytest basetemp)",
                  f"artifacts/eng5s/h3/gpu_{arm}/progress.json and outcomes.json (updated per chunk)",
                  "the worktree's three engine files and module while a chunk runs (installed, then Build S's "
                  "restored and sha-checked against BUILD_S.json: L128)"]
    else:
        paths += ["/tmp/eng5s_gpu_shim/eng5s_inject.so (rebuilt each run)", "/tmp/eng5s_tmp (child cwd)",
                  f"artifacts/eng5s/gpu/{line}/mutant_marker.txt (mutant arms only)"]
    return paths


def stage():
    out_path = os.path.join(ROOT, "artifacts/eng5s/GPU_LINES.json")
    if os.path.exists(out_path):
        raise SystemExit("GPU_LINES.json exists; each line is dry-run once")
    os.makedirs(DRY_DIR, exist_ok=True)
    rows = []
    for line, (mn, inner, arms, peak, desc) in LINES.items():
        argv = forge_argv(line)
        bad = module_checks(line)
        env = dict(os.environ, FORGE_GPU_DRY_RUN="1")  # the caller's environment otherwise unchanged (L58)
        t0 = now()
        p = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=120, cwd=ROOT)
        json.dump({"line": line, "time": t0, "argv": argv, "exit": p.returncode, "stdout": p.stdout[-3000:],
                   "stderr": p.stderr[-3000:], "module_checks_failed": bad},
                  open(os.path.join(DRY_DIR, f"{line}.json"), "w"), indent=1)
        rows.append({"line": line, "description": desc,
                     "lead_command": " ".join(PY + ["scripts/eng5s_gpu_line.py", "run", line]),
                     "forge_command": " ".join(shlex.quote(a) for a in argv), "min_s": mn, "max_s": 540,
                     "inner_timeout_s": inner, "modules": {a: receipt_sha(a) for a in arms},
                     "expected_peak_host_bytes": peak, "write_once_paths": write_once(line),
                     "dry_run_exit": p.returncode, "dry_run_receipt": f"artifacts/eng5s/gpu_dry/{line}.json",
                     "module_check_at_staging": "ok" if not bad else bad})
        print(line, "dry-run exit", p.returncode, "module check", "ok" if not bad else bad)
    json.dump({
        "written_utc": now(), "order": "orders/ENG5S_STATUS_CHECKS.md (G2, G3's GPU half, G4's mutant lines)",
        "cwd": "this worktree's root (tensor_fork_apa_wt_eng5s)",
        "run_each_with": "python3 -B -P scripts/eng5s_gpu_line.py run <line>",
        "environment": "the lead's environment with the GPU visible; no ENG5-S script removes or overrides "
                       "CUDA_VISIBLE_DEVICES (the step refuses with exit 7 if it is empty)",
        "scope": "the lead runs each line in a systemd scope with MemoryMax=12G and OOMPolicy=continue; no line sets "
                 "RLIMIT_AS / ulimit -v / prlimit; no line runs compute-sanitizer",
        "lease_check": "scripts/eng5s_gpu_step.py:2 (forge-gpu-lease-check --require-held), repeated as the first "
                       "statement of each child (G2_CHILD, its line 3)",
        "post_device_steps": "scripts/eng5s_gpu_verdict.py (written by the step as VERDICT.json); self-test on "
                             "constructed rows: artifacts/eng5s/gpu_dry/VERDICT_SELFTEST.json; CPU drive of the "
                             "children: artifacts/eng5s/gpu_dry/G2_CPU_DRIVE.json",
        "order_of_lines": "g3_* first (they read the per-arm packages, not the worktree), then the suite series last "
                          "(each chunk installs its arm's three engine files and module into the worktree and restores "
                          "Build S's, sha-checked: L128); run each series' chunks in order until progress.json says "
                          "complete (later chunks then exit at once)",
        "after_lines_cpu": ["python3 -B -P scripts/eng5s_compare.py artifacts/eng5s/h3/gpu_B/outcomes.json "
                            "artifacts/eng5s/h3/gpu_S/outcomes.json artifacts/eng5s/G3_GPU_COMPARE.json",
                            "python3 -B -P scripts/eng5s_arms.py results (MUTANT_RESULTS.json from the verdicts)"],
        "expected": {"g3_S": "VERDICT PASS", "g3_B": "RECORDED (each injected row returns under its fired injection)",
                     "g3_s01": "VERDICT KILLED (clone)", "g3_s02": "VERDICT KILLED (to_d2h and to_h2d)",
                     "g3_s03": "VERDICT KILLED (zeros)", "g3_s04": "VERDICT KILLED (from_host)",
                     "suite": "per-ID outcomes identical between gpu_B and gpu_S"},
        "semantic_verdict": "each line's VERDICT.json decides it; RESULT.json ran_clean only records the exits",
        "multi_device": "NOT COVERED (one GPU)",
        "lines": rows}, open(out_path, "w"), indent=1)


def run(line):
    d = os.path.join(GPU_DIR, line)
    if os.path.exists(d):
        print("REFUSED:", d, "exists (write-once; move it aside to retry)")
        return 5
    bad = module_checks(line)
    if bad:
        print("REFUSED (L103) before the lease:", json.dumps(bad))
        return 3
    os.makedirs(d)
    argv = forge_argv(line)
    env = {k: v for k, v in os.environ.items() if k != "FORGE_GPU_DRY_RUN"}
    env.setdefault("TMPDIR", "/tmp/eng5s_tmp")
    os.makedirs(env["TMPDIR"], exist_ok=True)
    log = os.path.join(d, "run.log")
    json.dump({"time": now(), "argv": argv, "cwd": ROOT, "modules": {a: receipt_sha(a) for a in LINES[line][2]}},
              open(os.path.join(d, "START.json"), "w"), indent=1)
    attempts = []
    for attempt in range(7):
        bad = module_checks(line)
        if bad:
            json.dump({"time": now(), "attempt": attempt, "bad": bad}, open(os.path.join(d, "REFUSED.json"), "w"))
            return 3
        t0 = now()
        with open(log, "ab") as lf:
            lf.write(f"\n=== attempt {attempt} {t0}\n".encode())
            lf.flush()
            rc = subprocess.run(argv, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        attempts.append({"attempt": attempt, "start": t0, "end": now(), "forge_exit": rc})
        json.dump(attempts, open(os.path.join(d, "ATTEMPTS.json"), "w"), indent=1)
        if rc not in (75, 76):
            break
        if attempt < 6:
            time.sleep(300)
    inner_path = os.path.join(d, "INNER_EXIT")
    inner = int(open(inner_path).read().strip()) if os.path.exists(inner_path) else None
    final = attempts[-1]["forge_exit"]
    verdict_path = os.path.join(d, "VERDICT.json")
    res = {"line": line, "end": now(), "forge_exit": final, "inner_exit": inner, "timed_out": final == 124,
           "attempts": attempts, "log": os.path.relpath(log, ROOT), "log_sha256": sha(log),
           "verdict": json.load(open(verdict_path)) if os.path.exists(verdict_path) else None,
           "ran_clean": final == 0 and inner == 0,
           "semantic_verdict": (json.load(open(verdict_path)).get("verdict") if os.path.exists(verdict_path) else None)}
    json.dump(res, open(os.path.join(d, "RESULT.json"), "w"), indent=1)
    print(json.dumps(res, indent=1)[:4000])
    return 0 if res["ran_clean"] else 1


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "stage":
        stage()
    elif cmd == "run":
        sys.exit(run(sys.argv[2]))
