"""ENG4-F1 GPU lines (H5). Same form and receipts as ENG4's scripts/eng4_gpu_line.py (left unchanged: the lead runs
ENG4's B-side lines with it), with F1's line names, arms and directories. The lead runs each line from the worktree
root as

    python3 -B -P scripts/eng4_f1_gpu_line.py run <line>

(1) refuses before the lease if artifacts/eng4/f1/gpu/<line>/ exists or any module the line uses differs from its
build receipt (L103); (2) runs, exactly as GPU_LINES_F1.json lists it,
    forge-gpu-run --min <s> --max 540 --label eng4f1-<line> -- timeout <s>s python3 -B -P scripts/eng4_f1_gpu_line.py step <line>
retrying unchanged after 300 s on forge exit 75/76 (nothing ran), at most 6 retries, re-checking the modules before
every attempt; (3) writes RESULT.json: forge exit, inner exit (written by the step as its last act), timed_out (124,
never a pass), attempts, the log's sha256 (L45).

    python3 -B -P scripts/eng4_f1_gpu_line.py stage   -> artifacts/eng4/GPU_LINES_F1.json + one dry run per line
                                                         (FORGE_GPU_DRY_RUN=1, expected 78) in artifacts/eng4/f1/gpu_dry/

Lines: g2_F1 (eng4_g2.py under memcheck, F1 package), g3_F1 (eng4_g3.py run, F1 package; compared afterwards on the
CPU with scripts/eng4_f1_g3_compare.py against the lead's g3_B), boundary_F1 (eng4_boundary_gpu.py), own_F1
(test_eng4_matmul.py on the F1 package), mut_m02_f1 and mut_m08 (the mutants' GPU killing tests, one process per
test), g4_B_01..10 and g4_F1_01..10 (scripts/eng4_f1_suite.py, series f1gpu_B / f1gpu_F1, 330 s test budget per
part; each restores Build F1 in the worktree). The ENG4 scripts these lines execute (eng4_g2.py, eng4_g3.py,
eng4_boundary_gpu.py) are run unchanged.
No line sets RLIMIT_AS, ulimit -v or prlimit (L114); the lead runs them in a MemoryMax=12G, OOMPolicy=continue scope.

Prior art: timeout(1); the house forge-gpu-run contract (GW2/GW3); ENG3's eng3_gpu_line.py and ENG4's
eng4_gpu_line.py (this repo), whose attempt/receipt shape this copies. No other prior art known to me.
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
SAN = "compute-sanitizer"
GPU_DIR = os.path.join(ROOT, "artifacts/eng4/f1/gpu")


def pkg(arm):
    return os.path.join(ROOT, "tensor_cuda/build-eng4", arm, "pkg")


def receipt_path(arm):
    return os.path.join(ROOT, "artifacts/eng4/build/BUILD_B.json" if arm == "B"
                        else f"artifacts/eng4/f1/build/BUILD_{arm}.json")


def receipt_sha(arm):
    return json.load(open(receipt_path(arm)))["module_sha256"]


def mutant_tests(mid, gpu_only=True):
    reg = json.load(open(os.path.join(ROOT, "artifacts/eng4/f1/MUTANTS_F1.json")))["mutants"][mid]
    return [t for t in reg["killing_tests"] if ("test_gpu_" in t) == gpu_only]


G4_PARTS = 10
LINES = {  # line -> (min s, inner timeout s, arms, expected peak host bytes, description)
    "g2_F1": (240, 480, ["F1"], 3 << 30, "H5 g2_F1: sanitizer preflight + ENG-MM1 under memcheck (refusal, 0 errors)"),
    "g3_F1": (120, 480, ["F1"], 3 << 30, "H5 g3_F1: the seeded G3 grid on F1 (eng4_g3.py run, unchanged)"),
    "boundary_F1": (180, 480, ["F1"], 3 << 30, "H5 boundary_F1: every refusal class with CUDA operands"),
    "own_F1": (120, 480, ["F1"], 3 << 30, "H5 own_F1: tensor_cuda/tests/test_eng4_matmul.py with the GPU visible"),
    "mut_m02_f1": (120, 480, ["m02_f1"], 3 << 30, "H4 m02_f1 GPU part: the mixed-device GPU tests, one process each"),
    "mut_m08": (180, 480, ["m08_k0_gemm"], 3 << 30, "H4 m08 GPU line: the K = 0 GPU zero checks, one process each"),
}
for _arm in ("B", "F1"):
    for _i in range(1, G4_PARTS + 1):
        LINES[f"g4_{_arm}_{_i:02d}"] = (480, 500, sorted({_arm, "F1"}), 6 << 30,
                                        f"H5 G4 GPU half, Build {_arm}, part {_i} (series f1gpu_{_arm}; exits at "
                                        "once when the series is complete)")


def forge_argv(line):
    mn, inner, _, _, _ = LINES[line]
    return ["forge-gpu-run", "--min", str(mn), "--max", "540", "--label", f"eng4f1-{line}", "--",
            "timeout", f"{inner}s"] + PY + ["scripts/eng4_f1_gpu_line.py", "step", line]


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
    paths = [f"artifacts/eng4/f1/gpu/{line}/"]
    if line.startswith("g4_"):
        arm = line.split("_")[1]
        paths += [f"artifacts/eng4/f1/g4/f1gpu_{arm}/part_<n>/ (n = parts so far + 1)",
                  f"/tmp/eng4_tmp/pt/f1gpu_{arm}_<n>/ (pytest basetemp)"]
    elif line.startswith("mut_") or line == "own_F1":
        paths += [f"/tmp/eng4_tmp/pt/<label>_<i>/ (pytest basetemp per test)"]
    return paths


def stage():
    os.makedirs(os.path.join(ROOT, "artifacts/eng4/f1/gpu_dry"), exist_ok=True)
    out_path = os.path.join(ROOT, "artifacts/eng4/GPU_LINES_F1.json")
    if os.path.exists(out_path):
        raise SystemExit("GPU_LINES_F1.json exists; each line is dry-run once")
    rows = []
    for line, (mn, inner, arms, peak, desc) in LINES.items():
        argv = forge_argv(line)
        bad = module_checks(line)
        env = dict(os.environ, FORGE_GPU_DRY_RUN="1")
        t0 = now()
        p = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=120, cwd=ROOT)
        dump(os.path.join(ROOT, "artifacts/eng4/f1/gpu_dry", f"{line}.json"),
             {"line": line, "time": t0, "argv": argv, "exit": p.returncode, "stdout": p.stdout[-3000:],
              "stderr": p.stderr[-3000:], "module_checks_failed": bad})
        rows.append({"line": line, "description": desc,
                     "lead_command": " ".join(PY + ["scripts/eng4_f1_gpu_line.py", "run", line]),
                     "forge_command": " ".join(shlex.quote(a) for a in argv), "min_s": mn, "max_s": 540,
                     "inner_timeout_s": inner, "modules": {a: receipt_sha(a) for a in arms},
                     "expected_peak_host_bytes": peak, "write_once_paths": write_once_paths(line),
                     "dry_run_exit": p.returncode, "dry_run_receipt": f"artifacts/eng4/f1/gpu_dry/{line}.json"})
        print(line, "dry-run exit", p.returncode, "module check", "ok" if not bad else bad)
    rec = {"written_utc": now(), "order": "orders/ENG4_F1_REV1_FIXES.md H5", "cwd": "the worktree root",
           "order_form": "forge-gpu-run --min <s> --max 540 --label eng4f1-<line> -- timeout <=500s ...",
           "run_each_with": "python3 -B -P scripts/eng4_f1_gpu_line.py run <line>",
           "scope": "the lead runs each line in a systemd scope with MemoryMax=12G and OOMPolicy=continue; no line "
                    "sets RLIMIT_AS / ulimit -v / prlimit",
           "prerequisite": "g3_F1's compare needs the lead's ENG4 line g3_B (artifacts/eng4/gpu/g3_B/g3)",
           "order_of_runs": "any; g4_<arm>_NN lines of one arm in number order (each continues its series); a g4 line "
                            "whose series is complete exits 0 at once; if g4_<arm>_10 ends incomplete, rerun the same "
                            "command after moving its write-once directory aside",
           "after_lines_cpu": [
               "python3 -B -P scripts/eng4_f1_g3_compare.py artifacts/eng4/gpu/g3_B/g3 "
               "artifacts/eng4/f1/gpu/g3_F1/g3 artifacts/eng4/f1/G3_F1_COMPARE.json",
               "python3 -B -P scripts/eng4_f1_g4_compare.py f1gpu_B f1gpu_F1 artifacts/eng4/f1/H3_GPU_COMPARE.json",
               "python3 -B -P scripts/eng4_f1_mutant_results.py gpu"],
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
    env = {k: v for k, v in os.environ.items() if k not in ("FORGE_GPU_DRY_RUN", "CUDA_VISIBLE_DEVICES")}
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
def sh(cmd, out_path, env=None, timeout=None):
    with open(out_path, "wb") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT, timeout=timeout)
    return p.returncode


def pyenv(arm):
    env = dict(os.environ)
    env["PYTHONPATH"] = pkg(arm)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def sanitizer_summary(path):
    text = open(path, errors="replace").read()
    return {"error_summary": [int(x) for x in re.findall(r"ERROR SUMMARY: (\d+) error", text)],
            "invalid_global_read": len(re.findall(r"Invalid __global__ read", text)), "log_sha256": sha(path)}


def step_g2(d):
    env = pyenv("F1")
    res = {"arm": "F1", "version_exit": sh([SAN, "--version"], os.path.join(d, "sanitizer_version.txt"), env)}
    res["noop_exit"] = sh([SAN, "--tool", "memcheck"] + PY + ["scripts/eng4_g2.py", os.path.join(d, "noop.json"),
                                                              "noop"], os.path.join(d, "memcheck_noop.log"), env)
    res["noop"] = sanitizer_summary(os.path.join(d, "memcheck_noop.log"))
    res["preflight_ok"] = res["version_exit"] == 0 and res["noop_exit"] == 0 and res["noop"]["error_summary"] == [0]
    res["witness_exit"] = sh([SAN, "--tool", "memcheck"] + PY + ["scripts/eng4_g2.py", os.path.join(d, "witness.json")],
                             os.path.join(d, "memcheck_witness.log"), env)
    res["witness"] = sanitizer_summary(os.path.join(d, "memcheck_witness.log"))
    wp = os.path.join(d, "witness.json")
    w = json.load(open(wp)) if os.path.exists(wp) else {}
    res["witness_record"] = w
    msg = w.get("raised", {}).get("message", "")
    res["pass"] = bool(res["preflight_ok"] and res["witness"]["error_summary"] == [0] and
                       msg.startswith("matmul refused: ") and "A [4, 2, 3]" in msg and "B [2, 3, 4]" in msg)
    dump(os.path.join(d, "G2.json"), res)
    return 0 if res["pass"] else 1


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


def step_pytest(arm, d, ids, label):
    rows = []
    for i, tid in enumerate(ids):
        junit = os.path.join(d, f"{label}_{i}.xml")
        rc = sh(PY + ["-m", "pytest", "-q", "-p", "no:cacheprovider", "-rA", f"--basetemp=/tmp/eng4_tmp/pt/{label}_{i}",
                      f"--junitxml={junit}", "-o", "junit_logging=all", tid], os.path.join(d, f"{label}_{i}.log"),
                pyenv(arm))
        outcome, msg = junit_case(junit)
        rows.append({"test": tid, "pytest_exit": rc, "outcome": outcome, "failure_message": msg[:600],
                     "log": f"{label}_{i}.log", "junit": f"{label}_{i}.xml"})
    return rows


def step_mut(d, arm, mid, named):
    rows = step_pytest(arm, d, mutant_tests(mid), mid)
    for r in rows:
        r["marker_seen"] = f"ENG4-MUTANT {mid} executed" in open(os.path.join(d, r["log"]), errors="replace").read()
        r["named_assertion_failed"] = r["outcome"] == "failed" and any(n in r["failure_message"] for n in named)
        r["process_died"] = r["pytest_exit"] < 0 or (r["outcome"] == "missing" and r["pytest_exit"] != 0)
        r["killed"] = r["named_assertion_failed"] or r["process_died"]
    dump(os.path.join(d, f"MUT_{mid}.json"), rows)
    return 0


def step(line):
    d = os.path.join(GPU_DIR, line)
    t0 = now()
    if line == "g2_F1":
        rc = step_g2(d)
    elif line == "g3_F1":
        rc = sh(PY + ["scripts/eng4_g3.py", "run", os.path.join(d, "g3")], os.path.join(d, "g3.log"), pyenv("F1"))
    elif line == "boundary_F1":
        rc = sh(PY + ["scripts/eng4_boundary_gpu.py", "F1", os.path.join(d, "boundary.json")],
                os.path.join(d, "boundary.log"), pyenv("F1"))
    elif line == "own_F1":
        rows = step_pytest("F1", d, ["tensor_cuda/tests/test_eng4_matmul.py"], "own_F1")
        dump(os.path.join(d, "OWN.json"), rows)
        rc = rows[0]["pytest_exit"]
    elif line == "mut_m02_f1":
        rc = step_mut(d, "m02_f1", "m02_f1", ["Regex pattern did not match", "DID NOT RAISE"])
    elif line == "mut_m08":
        rc = step_mut(d, "m08_k0_gemm", "m08_k0_gemm", ["zero check"])
    elif line.startswith("g4_"):
        arm = line.split("_")[1]
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        rc = sh(PY + ["scripts/eng4_f1_suite.py", f"f1gpu_{arm}", arm, "--budget", "330"], os.path.join(d, "suite.log"),
                env)
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
