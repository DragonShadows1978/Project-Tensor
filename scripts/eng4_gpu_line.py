"""ENG4 GPU lines. The lead runs each line, from the worktree root, as

    python3 -B -P scripts/eng4_gpu_line.py run <line>

which (1) refuses before the lease if artifacts/eng4/gpu/<line>/ exists or if any module the line uses differs from
its build receipt (L103); (2) runs the line's forge-gpu-run command, exactly as GPU_LINES.json lists it:
    forge-gpu-run --min <s> --max 540 --label eng4-<line> -- timeout <s>s python3 -B -P scripts/eng4_gpu_line.py step <line>
retrying unchanged after 300 s when forge-gpu-run returns 75 or 76 (no window; nothing ran), at most 6 retries,
re-checking the modules before every attempt; (3) writes RESULT.json: forge exit, inner exit (the step's own exit,
written by the step as its last act; absent if timeout killed it), timed_out (forge exit 124: never a pass), the
attempts, the log's sha256 (L45).

    python3 -B -P scripts/eng4_gpu_line.py stage     -> artifacts/eng4/GPU_LINES.json and one dry run per line
                                                       (FORGE_GPU_DRY_RUN=1, expected 78) in artifacts/eng4/gpu_dry/
    python3 -B -P scripts/eng4_gpu_line.py step <l>  -> the work inside the lease (only forge-gpu-run calls this)

Packages: each line imports a frozen package copy (tensor_cuda/build-eng4/<arm>/pkg, made by eng4_make_arms.py pkg),
except the G4 suite lines, which install their build into the worktree and restore Build E (scripts/eng4_suite.py).
No line sets RLIMIT_AS, ulimit -v or prlimit (L114); the lead runs them in a MemoryMax=12G, OOMPolicy=continue scope.

Prior art: timeout(1); the house forge-gpu-run contract (SystemOps GPU_WINDOW.md, GW2/GW3); ENG3's eng3_gpu_line.py
(this repo), whose attempt/receipt shape this copies. No other prior art known to me.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SO = "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
PY = ["python3", "-B", "-P"]
SAN = "compute-sanitizer"


def pkg(arm):
    return os.path.join(ROOT, "tensor_cuda/build-eng4", arm, "pkg")


def receipt_sha(arm):
    return json.load(open(os.path.join(ROOT, f"artifacts/eng4/build/BUILD_{arm}.json")))["module_sha256"]


# line -> (min seconds, inner timeout seconds, arms whose modules it uses, expected peak host bytes, description)
G4_PARTS = 10
LINES = {
    "g2_B": (240, 480, ["B"], 3 << 30, "G2 Build B: sanitizer preflight + ENG-MM1 under memcheck (expect OOB read)"),
    "g2_E": (240, 480, ["E"], 3 << 30, "G2 Build E: sanitizer preflight + ENG-MM1 under memcheck (expect refusal, "
                                       "0 errors)"),
    "g3_B": (120, 480, ["B"], 3 << 30, "G3 Build B: the seeded accepted-class grid, outputs saved"),
    "g3_E": (120, 480, ["E"], 3 << 30, "G3 Build E: the same grid in its own process"),
    "boundary_B": (180, 480, ["B"], 3 << 30, "L60 Build B: every refusal class with CUDA operands, one process "
                                             "per case"),
    "boundary_E": (180, 480, ["E"], 3 << 30, "L60 Build E: the same cases"),
    "own_E": (120, 480, ["E"], 3 << 30, "Build E: tensor_cuda/tests/test_eng4_matmul.py with the GPU visible"),
    "mut_m02": (120, 480, ["m02_no_device_check"], 3 << 30, "G5 m02: the GPU device tests on the m02 mutant, "
                                                           "one process per test"),
}
for _arm in ("B", "E"):
    for _i in range(1, G4_PARTS + 1):
        LINES[f"g4_{_arm}_{_i:02d}"] = (480, 500, [_arm, "E"], 6 << 30,
                                        f"G4 GPU half, Build {_arm}, part {_i} (continues series gpu_{_arm}; "
                                        "exits at once when the series is complete)")


def forge_argv(line):
    mn, inner, _, _, _ = LINES[line]
    return ["forge-gpu-run", "--min", str(mn), "--max", "540", "--label", f"eng4-{line}", "--",
            "timeout", f"{inner}s"] + PY + ["scripts/eng4_gpu_line.py", "step", line]


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
    paths = [f"artifacts/eng4/gpu/{line}/"]
    if line.startswith("g4_"):
        arm = line.split("_")[1]
        paths += [f"artifacts/eng4/g4/gpu_{arm}/part_<n>/ (n = parts so far + 1)",
                  f"/tmp/eng4_tmp/pt/gpu_{arm}_<n>/ (pytest basetemp)"]
    return paths


def stage():
    os.makedirs(os.path.join(ROOT, "artifacts/eng4/gpu_dry"), exist_ok=True)
    rows = []
    for line, (mn, inner, arms, peak, desc) in LINES.items():
        argv = forge_argv(line)
        bad = module_checks(line)
        env = dict(os.environ, FORGE_GPU_DRY_RUN="1")
        t0 = now()
        p = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=120, cwd=ROOT)
        dry = {"line": line, "time": t0, "argv": argv, "exit": p.returncode, "stdout": p.stdout[-3000:],
               "stderr": p.stderr[-3000:], "module_checks_failed": bad}
        dump(os.path.join(ROOT, "artifacts/eng4/gpu_dry", f"{line}.json"), dry)
        rows.append({"line": line, "description": desc, "lead_command": " ".join(PY + ["scripts/eng4_gpu_line.py",
                                                                                         "run", line]),
                     "forge_command": " ".join(shlex.quote(a) for a in argv), "min_s": mn, "max_s": 540,
                     "inner_timeout_s": inner, "modules": {a: receipt_sha(a) for a in arms},
                     "expected_peak_host_bytes": peak, "write_once_paths": write_once_paths(line),
                     "dry_run_exit": p.returncode, "dry_run_receipt": f"artifacts/eng4/gpu_dry/{line}.json"})
        print(line, "dry-run exit", p.returncode, "module check", "ok" if not bad else bad)
    rec = {"written_utc": now(), "cwd": "the worktree root", "order_form": "forge-gpu-run --min <s> --max 540 "
           "--label eng4-<line> -- timeout <=500s ...", "run_each_with": "python3 -B -P scripts/eng4_gpu_line.py "
           "run <line>  (precheck + lease + receipts; see the script docstring)",
           "scope": "the lead runs each line in a systemd scope with MemoryMax=12G and OOMPolicy=continue; no line "
                    "sets RLIMIT_AS / ulimit -v / prlimit",
           "order_of_runs": "any; g4_<arm>_NN lines of one arm run in number order (each continues the series); "
                            "a g4 line whose series is complete exits 0 at once. If g4_<arm>_10 ends with the "
                            "series incomplete, rerun the same command (it continues).",
           "after_lines_cpu": [
               "python3 -B -P scripts/eng4_g3.py compare artifacts/eng4/gpu/g3_B/g3 artifacts/eng4/gpu/g3_E/g3 "
               "artifacts/eng4/G3_COMPARE.json",
               "python3 -B -P scripts/eng4_g4_compare.py gpu_B gpu_E artifacts/eng4/G4_GPU_COMPARE.json",
               "python3 -B -P scripts/eng4_mutant_results.py gpu  (adds m02's GPU result to MUTANT_RESULTS.json)"],
           "lines": rows}
    dump(os.path.join(ROOT, "artifacts/eng4/GPU_LINES.json"), rec)


def run(line):
    d = os.path.join(ROOT, "artifacts/eng4/gpu", line)
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
            print("REFUSED (L103) before attempt", attempt)
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
    m = re.findall(r"ERROR SUMMARY: (\d+) error", text)
    return {"error_summary": [int(x) for x in m], "invalid_global_read": len(re.findall(r"Invalid __global__ read",
                                                                                         text)),
            "out_of_bounds_lines": len(re.findall(r"out of bounds", text)), "log_sha256": sha(path)}


def step_g2(line, d):
    arm = line.split("_")[1]
    env = pyenv(arm)
    res = {"arm": arm}
    res["version_exit"] = sh([SAN, "--version"], os.path.join(d, "sanitizer_version.txt"), env)
    res["noop_exit"] = sh([SAN, "--tool", "memcheck"] + PY + ["scripts/eng4_g2.py", os.path.join(d, "noop.json"),
                                                              "noop"], os.path.join(d, "memcheck_noop.log"), env)
    res["noop"] = sanitizer_summary(os.path.join(d, "memcheck_noop.log"))
    preflight_ok = res["version_exit"] == 0 and res["noop_exit"] == 0 and res["noop"]["error_summary"] == [0]
    res["preflight_ok"] = preflight_ok
    res["witness_exit"] = sh([SAN, "--tool", "memcheck"] + PY + ["scripts/eng4_g2.py",
                                                                  os.path.join(d, "witness.json")],
                             os.path.join(d, "memcheck_witness.log"), env)
    res["witness"] = sanitizer_summary(os.path.join(d, "memcheck_witness.log"))
    res["pooled_exit"] = sh([SAN, "--tool", "memcheck"] + PY + ["scripts/eng4_g2.py", os.path.join(d, "pooled.json"),
                                                                 "pooled"], os.path.join(d, "memcheck_pooled.log"), env)
    res["pooled"] = sanitizer_summary(os.path.join(d, "memcheck_pooled.log"))
    w = json.load(open(os.path.join(d, "witness.json"))) if os.path.exists(os.path.join(d, "witness.json")) else {}
    res["witness_record"] = {k: v for k, v in w.items() if k != "returned"}
    if arm == "B":  # expectation from the lead's code reading (L81): memcheck reports an out-of-bounds read. Whether
        # the call then returns or raises (memcheck may abort the faulting kernel) is recorded, not gated.
        ok = preflight_ok and res["witness"]["invalid_global_read"] > 0
    else:  # E: refuses before any kernel; memcheck reports 0 errors
        ok = (preflight_ok and res["witness"]["error_summary"] == [0] and "raised" in w
              and w["raised"]["message"].startswith("matmul refused: ") and "A [4, 2, 3]" in w["raised"]["message"]
              and "B [2, 3, 4]" in w["raised"]["message"])
    res["pass"] = bool(ok)
    dump(os.path.join(d, "G2.json"), res)
    return 0 if ok else 1


def step_g3(line, d):
    arm = line.split("_")[1]
    rc = sh(PY + ["scripts/eng4_g3.py", "run", os.path.join(d, "g3")], os.path.join(d, "g3.log"), pyenv(arm))
    return rc


def step_boundary(line, d):
    arm = line.split("_")[1]
    return sh(PY + ["scripts/eng4_boundary_gpu.py", arm, os.path.join(d, "boundary.json")],
              os.path.join(d, "boundary.log"), pyenv(arm))


def step_pytest(arm, d, ids, label):
    rows, ok = [], True
    for i, tid in enumerate(ids):
        junit = os.path.join(d, f"{label}_{i}.xml")
        rc = sh(PY + ["-m", "pytest", "-q", "-p", "no:cacheprovider", "-rA", f"--basetemp=/tmp/eng4_tmp/pt/{label}_{i}",
                      f"--junitxml={junit}", "-o", "junit_logging=all", tid],
                os.path.join(d, f"{label}_{i}.log"), pyenv(arm))
        rows.append({"test": tid, "pytest_exit": rc, "log": f"{label}_{i}.log", "junit": f"{label}_{i}.xml"})
    return rows


def step_own(line, d):
    rows = step_pytest("E", d, ["tensor_cuda/tests/test_eng4_matmul.py"], "own_E")
    dump(os.path.join(d, "OWN.json"), rows)
    return rows[0]["pytest_exit"]


def step_mut_m02(line, d):
    ids = ["tensor_cuda/tests/test_eng4_matmul.py::test_gpu_m4_device_refuses_nonzero_operands[cuda_a_host_b]",
           "tensor_cuda/tests/test_eng4_matmul.py::test_gpu_m4_device_refuses_nonzero_operands[host_a_cuda_b]"]
    rows = step_pytest("m02_no_device_check", d, ids, "m02")
    import xml.etree.ElementTree as ET
    for r in rows:
        # killed = the test failed on its named assertion (pytest.raises: regex mismatch or DID NOT RAISE), or its
        # process died inside the test (negative exit: a signal; no junit case). passed/skipped = survived.
        log_text = open(os.path.join(d, r["log"]), errors="replace").read()
        r["marker_seen"] = "ENG4-MUTANT m02_no_device_check executed" in log_text
        outcome, msg = "missing", ""
        jp = os.path.join(d, r["junit"])
        if os.path.exists(jp):
            for case in ET.parse(jp).getroot().iter("testcase"):
                f, e = case.find("failure"), case.find("error")
                outcome = "failed" if f is not None else "error" if e is not None else (
                    "skipped" if case.find("skipped") is not None else "passed")
                msg = ((f if f is not None else e).get("message") or "") if (f is not None or e is not None) else ""
        r["outcome"], r["failure_message"] = outcome, msg[:600]
        named = outcome == "failed" and ("Regex pattern did not match" in msg or "DID NOT RAISE" in msg)
        died = r["pytest_exit"] < 0 or (outcome == "missing" and r["pytest_exit"] != 0)
        r["named_assertion_failed"], r["process_died"] = named, died
        r["killed"] = named or died
    dump(os.path.join(d, "MUT_m02.json"), rows)
    return 0  # the line's own success is having run; the kill verdict is in MUT_m02.json


def step_g4(line, d):
    arm = line.split("_")[1]
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return sh(PY + ["scripts/eng4_suite.py", f"gpu_{arm}", arm, "--budget", "330"], os.path.join(d, "suite.log"), env)


def step(line):
    d = os.path.join(ROOT, "artifacts/eng4/gpu", line)
    t0 = now()
    if line.startswith("g2_"):
        rc = step_g2(line, d)
    elif line.startswith("g3_"):
        rc = step_g3(line, d)
    elif line.startswith("boundary_"):
        rc = step_boundary(line, d)
    elif line == "own_E":
        rc = step_own(line, d)
    elif line == "mut_m02":
        rc = step_mut_m02(line, d)
    elif line.startswith("g4_"):
        rc = step_g4(line, d)
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
