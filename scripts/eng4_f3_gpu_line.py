"""ENG4-F3 GPU lines (K6), in F2's form (scripts/eng4_f2_gpu_line.py, unchanged). The lead runs each line from this
worktree's root as

    python3 -B -P scripts/eng4_f3_gpu_line.py run <line>

(1) refuses before the lease if artifacts/eng4/f3/gpu/<line>/ exists or any module the line uses differs from its
build receipt (L103); (2) runs, exactly as artifacts/eng4/f3/GPU_LINES_F3.json lists it,
    forge-gpu-run --min <s> --max 540 --label eng4f3-<line> -- timeout <s>s python3 -B -P scripts/eng4_f3_gpu_line.py step <line>
retrying unchanged after 300 s on forge exit 75/76 (nothing ran), at most 6 retries, re-checking the modules before
every attempt; (3) writes RESULT.json: forge exit, inner exit (written by the step as its last act), timed_out (124,
never a pass), attempts, the log's sha256 (L45).

    python3 -B -P scripts/eng4_f3_gpu_line.py stage  -> artifacts/eng4/f3/GPU_LINES_F3.json + one dry run per line
                                                        (FORGE_GPU_DRY_RUN=1, expected 78), f3/gpu_dry/<line>.json

L58: `run` and `stage` pass the caller's environment to forge-gpu-run unchanged. The only site that removes
CUDA_VISIBLE_DEVICES is step_env(), reached only from `step` after lease_guard() has found forge-gpu-run among its
ancestor processes; called any other way, `step` refuses with exit 7 before touching the environment.
No line runs compute-sanitizer.

Lines: own_F3 (tensor_cuda/tests/test_eng4_matmul.py on the F3 package), g3z_F3 (scripts/eng4_f3_g3z.py), and K4's
witness pending_F2_F3 (scripts/eng4_f3_pending.py, children on the F2 and F3 packages). No line sets RLIMIT_AS,
ulimit -v or prlimit (L114).

Prior art: timeout(1); the house forge-gpu-run contract (GW2/GW3); the ENG4 line runners (this repo); the ancestor
guard is ENG4-F2's. No other prior art known to me.
"""
import datetime
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SO = "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
PY = ["python3", "-B", "-P"]
F3 = os.path.join(ROOT, "artifacts/eng4/f3")
GPU_DIR = os.path.join(F3, "gpu")
RECEIPT = {"F2": "artifacts/eng4/f2/build/BUILD_F2.json", "F3": "artifacts/eng4/f3/build/BUILD_F3.json"}
LINES = {  # line -> (min s, inner timeout s, arms, expected peak host bytes, description)
    "own_F3": (120, 480, ["F3"], 3 << 30, "K6 own_F3: tensor_cuda/tests/test_eng4_matmul.py with the GPU visible"),
    "g3z_F3": (120, 480, ["F3"], 3 << 30, "K6 g3z_F3: G3 zero-size cells + 24 empty-batch gradients vs NumPy/zeros"),
    "pending_F2_F3": (120, 480, ["F2", "F3"], 3 << 30, "K4 witness: a pending allocation error, then the zero fills, "
                                                       "on F2 and F3 (4 child processes)"),
}


def pkg(arm):
    return os.path.join(ROOT, "tensor_cuda/build-eng4", arm, "pkg")


def receipt_sha(arm):
    return json.load(open(os.path.join(ROOT, RECEIPT[arm])))["module_sha256"]


def forge_argv(line):
    mn, inner, _, _, _ = LINES[line]
    return ["forge-gpu-run", "--min", str(mn), "--max", "540", "--label", f"eng4f3-{line}", "--",
            "timeout", f"{inner}s"] + PY + ["scripts/eng4_f3_gpu_line.py", "step", line]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def dump(p, o):
    with open(p, "w") as f:
        json.dump(o, f, indent=1)


def module_checks(line):
    bad = {}
    for arm in LINES[line][2]:
        want = receipt_sha(arm)
        for p in (os.path.join(ROOT, "tensor_cuda/build-eng4", arm, "module", SO),
                  os.path.join(pkg(arm), "tensor_cuda", SO)):
            got = sha(p) if os.path.exists(p) else None
            if got != want:
                bad[p] = {"expected": want, "found": got}
    return bad


def stage():
    os.makedirs(os.path.join(F3, "gpu_dry"), exist_ok=True)
    out_path = os.path.join(F3, "GPU_LINES_F3.json")
    if os.path.exists(out_path):
        raise SystemExit("GPU_LINES_F3.json exists; each line is dry-run once")
    rows = []
    for line, (mn, inner, arms, peak, desc) in LINES.items():
        argv = forge_argv(line)
        bad = module_checks(line)
        env = dict(os.environ, FORGE_GPU_DRY_RUN="1")  # otherwise the caller's environment, unchanged (L58)
        t0 = now()
        p = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=120, cwd=ROOT)
        dump(os.path.join(F3, "gpu_dry", f"{line}.json"),
             {"line": line, "time": t0, "argv": argv, "exit": p.returncode, "stdout": p.stdout[-3000:],
              "stderr": p.stderr[-3000:], "module_checks_failed": bad})
        rows.append({"line": line, "description": desc,
                     "lead_command": " ".join(PY + ["scripts/eng4_f3_gpu_line.py", "run", line]),
                     "forge_command": " ".join(shlex.quote(a) for a in argv), "min_s": mn, "max_s": 540,
                     "inner_timeout_s": inner, "modules": {a: receipt_sha(a) for a in arms},
                     "expected_peak_host_bytes": peak,
                     "write_once_paths": [f"artifacts/eng4/f3/gpu/{line}/"] +
                     (["/tmp/eng4f3_tmp/pt/own_F3_0/ (pytest basetemp)"] if line == "own_F3" else []),
                     "dry_run_exit": p.returncode, "dry_run_receipt": f"artifacts/eng4/f3/gpu_dry/{line}.json"})
        print(line, "dry-run exit", p.returncode, "module check", "ok" if not bad else bad)
    dump(out_path, {
        "written_utc": now(), "order": "orders/ENG4_F3_REV3_FILL_STATUS.md K6", "cwd": "this worktree's root "
        "(tensor_fork_apa_wt_eng4f3)", "run_each_with": "python3 -B -P scripts/eng4_f3_gpu_line.py run <line>",
        "scope": "the lead runs each line in a systemd scope with MemoryMax=12G and OOMPolicy=continue; no line sets "
                 "RLIMIT_AS / ulimit -v / prlimit; no line runs compute-sanitizer",
        "L58_guard": "step refuses (exit 7) unless forge-gpu-run is an ancestor process; only then does it remove "
                     "CUDA_VISIBLE_DEVICES from its children's environment",
        "after_lines_cpu": ["python3 -B -P scripts/eng4_f3_g3z.py verdict artifacts/eng4/f3/gpu/g3z_F3/g3z",
                            "python3 -B -P scripts/eng4_f3_pending.py verdict artifacts/eng4/f3/gpu/pending_F2_F3/pending"],
        "lines": rows})


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
    env = {k: v for k, v in os.environ.items() if k != "FORGE_GPU_DRY_RUN"}  # CUDA_VISIBLE_DEVICES untouched (L58)
    env.setdefault("TMPDIR", "/tmp/eng4f3_tmp")
    os.makedirs(env["TMPDIR"], exist_ok=True)
    log = os.path.join(d, "run.log")
    dump(os.path.join(d, "START.json"), {"time": now(), "argv": argv, "cwd": ROOT,
                                         "modules": {a: receipt_sha(a) for a in LINES[line][2]}})
    attempts = []
    for attempt in range(7):
        bad = module_checks(line)
        if bad:
            dump(os.path.join(d, "REFUSED.json"), {"time": now(), "attempt": attempt, "bad": bad})
            return 3
        t0 = now()
        with open(log, "ab") as lf:
            lf.write(f"\n=== attempt {attempt} {t0}\n".encode())
            lf.flush()
            rc = subprocess.run(argv, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        attempts.append({"attempt": attempt, "start": t0, "end": now(), "forge_exit": rc})
        dump(os.path.join(d, "ATTEMPTS.json"), attempts)
        if rc not in (75, 76):
            break
        if attempt < 6:
            time.sleep(300)
    inner_path = os.path.join(d, "INNER_EXIT")
    inner = int(open(inner_path).read().strip()) if os.path.exists(inner_path) else None
    final = attempts[-1]["forge_exit"]
    res = {"line": line, "end": now(), "forge_exit": final, "inner_exit": inner, "timed_out": final == 124,
           "attempts": attempts, "log": os.path.relpath(log, ROOT), "log_sha256": sha(log),
           "pass": final == 0 and inner == 0}
    dump(os.path.join(d, "RESULT.json"), res)
    print(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


# ------------------------------------------------------------------------------------------- inside the lease
def lease_guard():
    """True only if forge-gpu-run is an ancestor of this process (ENG4-F2's guard)."""
    pid = os.getppid()
    for _ in range(8):
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
            stat = open(f"/proc/{pid}/stat").read()
        except OSError:
            return False
        if "forge-gpu-run" in cmd:
            return True
        pid = int(stat[stat.rindex(")") + 2:].split()[1])
        if pid <= 1:
            return False
    return False


def step_env(arm=None):
    """L58: the one site that removes CUDA_VISIBLE_DEVICES; reached only after lease_guard() in step()."""
    env = dict(os.environ)
    if arm:
        env["PYTHONPATH"] = pkg(arm)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def sh(cmd, out_path, env):
    with open(out_path, "wb") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT).returncode


def junit_outcome(path):
    if not os.path.exists(path):
        return "missing"
    counts = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        o = ("failed" if case.find("failure") is not None else "error" if case.find("error") is not None else
             "skipped" if case.find("skipped") is not None else "passed")
        counts[o] = counts.get(o, 0) + 1
    return counts


def step(line):
    d = os.path.join(GPU_DIR, line)
    if not lease_guard():
        print("REFUSED (L58): step runs only under forge-gpu-run; nothing was run")
        return 7
    if not os.path.isdir(d):
        print("REFUSED: no line directory (run the line through `run`)")
        return 7
    t0 = now()
    if line == "own_F3":
        junit = os.path.join(d, "own_F3.xml")
        rc = sh(PY + ["-m", "pytest", "-q", "-p", "no:cacheprovider", "-rA", "--basetemp=/tmp/eng4f3_tmp/pt/own_F3_0",
                      f"--junitxml={junit}", "-o", "junit_logging=all", "tensor_cuda/tests/test_eng4_matmul.py"],
                os.path.join(d, "own_F3.log"), step_env("F3"))
        dump(os.path.join(d, "OWN.json"), {"pytest_exit": rc, "outcomes": junit_outcome(junit)})
    elif line == "g3z_F3":
        rc = sh(PY + ["scripts/eng4_f3_g3z.py", "run", os.path.join(d, "g3z")], os.path.join(d, "g3z.log"),
                step_env("F3"))
    elif line == "pending_F2_F3":
        rc = sh(PY + ["scripts/eng4_f3_pending.py", "run", os.path.join(d, "pending")], os.path.join(d, "pending.log"),
                step_env())
    else:
        rc = 2
    dump(os.path.join(d, "STEP.json"), {"line": line, "start": t0, "end": now(), "exit": rc})
    with open(os.path.join(d, "INNER_EXIT"), "w") as f:
        f.write(str(rc))
    return rc


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "stage":
        stage()
    elif cmd == "run":
        sys.exit(run(sys.argv[2]))
    elif cmd == "step":
        sys.exit(step(sys.argv[2]))
