"""ENG4-F2 GPU lines (J5). ENG4's form and receipts (scripts/eng4_gpu_line.py, eng4_f1_gpu_line.py: both unchanged),
with F2's names, arms and directories, and one addition, the L58 lease guard. The lead runs each line from the
worktree root as

    python3 -B -P scripts/eng4_f2_gpu_line.py run <line>

(1) refuses before the lease if artifacts/eng4/f2/gpu/<line>/ exists or any module the line uses differs from its
build receipt (L103); (2) runs, exactly as GPU_LINES_F2.json lists it,
    forge-gpu-run --min <s> --max 540 --label eng4f2-<line> -- timeout <s>s python3 -B -P scripts/eng4_f2_gpu_line.py step <line>
retrying unchanged after 300 s on forge exit 75/76 (nothing ran), at most 6 retries, re-checking the modules before
every attempt; (3) writes RESULT.json: forge exit, inner exit (written by the step as its last act), timed_out (124,
never a pass), attempts, the log's sha256 (L45).

    python3 -B -P scripts/eng4_f2_gpu_line.py stage  -> artifacts/eng4/GPU_LINES_F2.json + one dry run per line
                                                        (FORGE_GPU_DRY_RUN=1, expected 78), f2/gpu_dry/<line>.json

L58 (CUDA_VISIBLE_DEVICES): `run` and `stage` pass the caller's environment to forge-gpu-run unchanged. The only
site that removes the variable is step_env(), and `step` calls it only after lease_guard() has found forge-gpu-run
among its ancestor processes (forge-gpu-run -> timeout -> this step); called any other way, `step` refuses with exit
7 before it touches the environment or imports anything that could reach a device.

Lines: bwd0_B and bwd0_F2 (scripts/eng4_f2_bwd0.py), g3_F2 (scripts/eng4_g3.py run, unchanged, F2 package; compared on
the CPU afterwards with scripts/eng4_f1_g3_compare.py, unchanged, against the lead's artifacts/eng4/gpu/g3_B/g3),
own_F2 (test_eng4_matmul.py on the F2 package), mut_m10_f2 (m10's GPU killing tests, one process each), g4_B_01..10
and g4_F2_01..10 (scripts/eng4_f2_suite.py, series f2gpu_B / f2gpu_F2, 330 s test budget per part; each restores
Build F2 in the worktree). No line sets RLIMIT_AS, ulimit -v or prlimit (L114).

Prior art: timeout(1); the house forge-gpu-run contract (GW2/GW3); ENG3/ENG4/ENG4-F1 line runners (this repo). The
ancestor-process guard is mine (Linux /proc/<pid>/stat and cmdline); no other prior art known to me.
"""
import datetime
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SO = "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
PY = ["python3", "-B", "-P"]
GPU_DIR = os.path.join(ROOT, "artifacts/eng4/f2/gpu")


def pkg(arm):
    return os.path.join(ROOT, "tensor_cuda/build-eng4", arm, "pkg")


def receipt_path(arm):
    return os.path.join(ROOT, "artifacts/eng4/build/BUILD_B.json" if arm == "B"
                        else f"artifacts/eng4/f2/build/BUILD_{arm}.json")


def receipt_sha(arm):
    return json.load(open(receipt_path(arm)))["module_sha256"]


def mutant_gpu_tests(mid):
    reg = json.load(open(os.path.join(ROOT, "artifacts/eng4/f2/MUTANTS_F2.json")))["mutants"][mid]
    return [t for t in reg["killing_tests"] if "test_gpu_" in t]


G4_PARTS = 10
LINES = {  # line -> (min s, inner timeout s, arms, expected peak host bytes, description)
    "bwd0_B": (120, 480, ["B"], 3 << 30, "J5 bwd0_B: production's backward on Y2's batch-0 shared-B pairs, recorded"),
    "bwd0_F2": (120, 480, ["F2"], 3 << 30, "J5 bwd0_F2: the same pairs on F2 (B zero gradient, A empty gradient)"),
    "g3_F2": (120, 480, ["F2"], 3 << 30, "J5 g3_F2: ENG4's seeded G3 grid on F2 (eng4_g3.py run, unchanged)"),
    "own_F2": (120, 480, ["F2"], 3 << 30, "F2's own tests with the GPU visible (test_eng4_matmul.py)"),
    "mut_m10_f2": (180, 480, ["m10_f2"], 3 << 30, "J4 m10 GPU line: the GPU zero-gradient tests on the m10 mutant"),
}
for _arm in ("B", "F2"):
    for _i in range(1, G4_PARTS + 1):
        LINES[f"g4_{_arm}_{_i:02d}"] = (480, 500, sorted({_arm, "F2"}), 6 << 30,
                                        f"J5 G4 GPU half, Build {_arm}, part {_i} (series f2gpu_{_arm}; exits at "
                                        "once when the series is complete)")


def forge_argv(line):
    mn, inner, _, _, _ = LINES[line]
    return ["forge-gpu-run", "--min", str(mn), "--max", "540", "--label", f"eng4f2-{line}", "--",
            "timeout", f"{inner}s"] + PY + ["scripts/eng4_f2_gpu_line.py", "step", line]


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
        paths = [os.path.join(ROOT, "tensor_cuda/build-eng4", arm, "module", SO)]
        if os.path.isdir(pkg(arm)):
            paths.append(os.path.join(pkg(arm), "tensor_cuda", SO))
        for p in paths:
            got = sha(p) if os.path.exists(p) else None
            if got != want:
                bad[p] = {"expected": want, "found": got}
    return bad


def write_once_paths(line):
    paths = [f"artifacts/eng4/f2/gpu/{line}/"]
    if line.startswith("g4_"):
        arm = line.split("_")[1]
        paths += [f"artifacts/eng4/f2/g4/f2gpu_{arm}/part_<n>/ (n = parts so far + 1)",
                  f"/tmp/eng4_tmp/pt/f2gpu_{arm}_<n>/ (pytest basetemp)"]
    elif line in ("own_F2", "mut_m10_f2"):
        paths += ["/tmp/eng4_tmp/pt/<label>_<i>/ (pytest basetemp per test)"]
    return paths


def stage():
    os.makedirs(os.path.join(ROOT, "artifacts/eng4/f2/gpu_dry"), exist_ok=True)
    out_path = os.path.join(ROOT, "artifacts/eng4/GPU_LINES_F2.json")
    if os.path.exists(out_path):
        raise SystemExit("GPU_LINES_F2.json exists; each line is dry-run once")
    rows = []
    for line, (mn, inner, arms, peak, desc) in LINES.items():
        argv = forge_argv(line)
        bad = module_checks(line)
        env = dict(os.environ, FORGE_GPU_DRY_RUN="1")  # the caller's environment otherwise unchanged (L58)
        t0 = now()
        p = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=120, cwd=ROOT)
        dump(os.path.join(ROOT, "artifacts/eng4/f2/gpu_dry", f"{line}.json"),
             {"line": line, "time": t0, "argv": argv, "exit": p.returncode, "stdout": p.stdout[-3000:],
              "stderr": p.stderr[-3000:], "module_checks_failed": bad})
        rows.append({"line": line, "description": desc,
                     "lead_command": " ".join(PY + ["scripts/eng4_f2_gpu_line.py", "run", line]),
                     "forge_command": " ".join(shlex.quote(a) for a in argv), "min_s": mn, "max_s": 540,
                     "inner_timeout_s": inner, "modules": {a: receipt_sha(a) for a in arms},
                     "expected_peak_host_bytes": peak, "write_once_paths": write_once_paths(line),
                     "dry_run_exit": p.returncode, "dry_run_receipt": f"artifacts/eng4/f2/gpu_dry/{line}.json"})
        print(line, "dry-run exit", p.returncode, "module check", "ok" if not bad else bad)
    rec = {"written_utc": now(), "order": "orders/ENG4_F2_REV2_HARDENING.md J5", "cwd": "the worktree root",
           "order_form": "forge-gpu-run --min <s> --max 540 --label eng4f2-<line> -- timeout <=500s ...",
           "run_each_with": "python3 -B -P scripts/eng4_f2_gpu_line.py run <line>",
           "scope": "the lead runs each line in a systemd scope with MemoryMax=12G and OOMPolicy=continue; no line "
                    "sets RLIMIT_AS / ulimit -v / prlimit",
           "L58_guard": "step refuses (exit 7) unless forge-gpu-run is an ancestor process; only then does it remove "
                        "CUDA_VISIBLE_DEVICES from its children's environment",
           "prerequisite": "g3_F2's compare reads the lead's ENG4 line g3_B (artifacts/eng4/gpu/g3_B/g3)",
           "order_of_runs": "any; g4_<arm>_NN lines of one arm in number order (each continues its series); a g4 line "
                            "whose series is complete exits 0 at once; if g4_<arm>_10 ends incomplete, rerun the same "
                            "command after moving its write-once directory aside",
           "after_lines_cpu": [
               "python3 -B -P scripts/eng4_f1_g3_compare.py artifacts/eng4/gpu/g3_B/g3 "
               "artifacts/eng4/f2/gpu/g3_F2/g3 artifacts/eng4/f2/G3_F2_COMPARE.json",
               "python3 -B -P scripts/eng4_f2_bwd0.py verdict artifacts/eng4/f2/gpu/bwd0_B/bwd0 B",
               "python3 -B -P scripts/eng4_f2_bwd0.py verdict artifacts/eng4/f2/gpu/bwd0_F2/bwd0 F2",
               "python3 -B -P scripts/eng4_f2_g4_compare.py f2gpu_B f2gpu_F2 artifacts/eng4/f2/J3_GPU_COMPARE.json",
               "python3 -B -P scripts/eng4_f2_mutant_results.py gpu"],
           "lines": rows}
    dump(out_path, rec)


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
    env.setdefault("TMPDIR", "/tmp/eng4_tmp")
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
    """True only if forge-gpu-run is an ancestor of this process (it runs timeout, which runs this step)."""
    pid = os.getppid()
    for _ in range(8):
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
            stat = open(f"/proc/{pid}/stat").read()
        except OSError:
            return False
        if "forge-gpu-run" in cmd:
            return True
        pid = int(stat[stat.rindex(")") + 2:].split()[1])  # field 4: the parent pid
        if pid <= 1:
            return False
    return False


def step_env(arm):
    """L58: the one site that removes CUDA_VISIBLE_DEVICES; reached only after lease_guard() in step()."""
    env = dict(os.environ)
    env["PYTHONPATH"] = pkg(arm)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def sh(cmd, out_path, env, timeout=None):
    with open(out_path, "wb") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT, timeout=timeout)
    return p.returncode


def junit_case(path):
    if not os.path.exists(path):
        return "missing", ""
    for case in ET.parse(path).getroot().iter("testcase"):
        f, e = case.find("failure"), case.find("error")
        if f is not None:
            return "failed", f.get("message") or ""
        if e is not None:
            return "error", e.get("message") or ""
        return ("skipped" if case.find("skipped") is not None else "passed"), ""
    return "missing", ""


def classify_kill(row, log_text, mid, named):
    """Post-device verdict for one mutant test (also driven on constructed returns by eng4_f2_postdevice.py)."""
    row["marker_seen"] = f"ENG4-MUTANT {mid} executed" in log_text
    row["named_assertion_failed"] = row["outcome"] == "failed" and any(n in row["failure_message"] for n in named)
    row["process_died"] = row["pytest_exit"] < 0 or (row["outcome"] == "missing" and row["pytest_exit"] != 0)
    row["killed"] = row["named_assertion_failed"] or row["process_died"]
    return row


def step_pytest(env, d, ids, label):
    rows = []
    for i, tid in enumerate(ids):
        junit = os.path.join(d, f"{label}_{i}.xml")
        rc = sh(PY + ["-m", "pytest", "-q", "-p", "no:cacheprovider", "-rA", f"--basetemp=/tmp/eng4_tmp/pt/{label}_{i}",
                      f"--junitxml={junit}", "-o", "junit_logging=all", tid], os.path.join(d, f"{label}_{i}.log"), env)
        outcome, msg = junit_case(junit)
        rows.append({"test": tid, "pytest_exit": rc, "outcome": outcome, "failure_message": msg[:600],
                     "log": f"{label}_{i}.log", "junit": f"{label}_{i}.xml"})
    return rows


def step(line):
    d = os.path.join(GPU_DIR, line)
    if not lease_guard():
        print("REFUSED (L58): step runs only under forge-gpu-run; nothing was run")
        return 7
    if not os.path.isdir(d):
        print("REFUSED: no line directory (run the line through `run`)")
        return 7
    t0 = now()
    if line in ("bwd0_B", "bwd0_F2"):
        arm = line.split("_")[1]
        rc = sh(PY + ["scripts/eng4_f2_bwd0.py", "run", os.path.join(d, "bwd0")], os.path.join(d, "bwd0.log"),
                step_env(arm))
    elif line == "g3_F2":
        rc = sh(PY + ["scripts/eng4_g3.py", "run", os.path.join(d, "g3")], os.path.join(d, "g3.log"), step_env("F2"))
    elif line == "own_F2":
        rows = step_pytest(step_env("F2"), d, ["tensor_cuda/tests/test_eng4_matmul.py"], "own_F2")
        dump(os.path.join(d, "OWN.json"), rows)
        rc = rows[0]["pytest_exit"]
    elif line == "mut_m10_f2":
        mid = "m10_f2_empty_batch_raise"
        rows = step_pytest(step_env("m10_f2"), d, mutant_gpu_tests(mid), "m10")
        for r in rows:
            classify_kill(r, open(os.path.join(d, r["log"]), errors="replace").read(), mid, ["zero-gradient check"])
        dump(os.path.join(d, f"MUT_{mid}.json"), rows)
        rc = 0
    elif line.startswith("g4_"):
        arm = line.split("_")[1]
        env = step_env("F2")
        env.pop("PYTHONPATH", None)  # the suite installs its build into the worktree and sets its own path
        rc = sh(PY + ["scripts/eng4_f2_suite.py", f"f2gpu_{arm}", arm, "--budget", "330"],
                os.path.join(d, "suite.log"), env)
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
