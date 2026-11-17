"""ENG3 GPU line runner. One line = one forge-gpu-run lease, from the worktree root.

  python3 -B -P scripts/eng3_gpu_line.py <line> <expected_module_sha256> <max_s> <inner_s> -- <cmd...>

Steps (receipts in artifacts/eng3/gpu/<line>/):
 1. L103: the installed module's sha256 must equal <expected>; otherwise refuse (exit 3) before any lease.
 2. L79: one dry run, FORGE_GPU_DRY_RUN=1, must return 78 (DRY.json); otherwise stop (exit 4).
 3. START.json; the real run `forge-gpu-run --min 30 --max <max_s> --label eng3-<line> -- timeout <inner_s>s
    bash -c '<cmd>; echo $? > INNER_EXIT'` with output in run.log. The inner file holds the command's own exit;
    forge-gpu-run's exit carries timeout's (124 = timeout, never a pass).
 4. Exit 76 (nothing ran): retry unchanged after 300 s, at most 6 retries; the module sha256 is checked again
    before every attempt (refuse, exit 3, if it changed).
 5. EXIT.json and RESULT.json: forge exit, inner exit, timed_out, attempts, log sha256, module sha256.

Prior art: no prior art known to me beyond timeout(1) and the house forge-gpu-run contract (GPU_WINDOW.md).
"""
import datetime
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time

MODULE = "tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def dump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    line, expect, max_s, inner_s = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    cmd = sys.argv[sys.argv.index("--") + 1:]
    d = os.path.join("artifacts/eng3/gpu", line)
    os.makedirs(d, exist_ok=True)
    inner_file = os.path.join(d, "INNER_EXIT")
    got = sha(MODULE)
    if got != expect:
        dump(os.path.join(d, "REFUSED.json"), {"time": now(), "expected": expect, "installed": got})
        print("REFUSED: module sha256", got, "!= expected", expect)
        sys.exit(3)
    inner = " ".join(shlex.quote(c) for c in cmd) + f"; echo $? > {shlex.quote(inner_file)}"
    fgr = ["forge-gpu-run", "--min", "30", "--max", str(max_s), "--label", f"eng3-{line}", "--",
           "timeout", f"{inner_s}s", "bash", "-c", inner]
    env = dict(os.environ)
    env["FORGE_GPU_DRY_RUN"] = "1"
    dry = subprocess.run(fgr, env=env, capture_output=True, text=True, timeout=120)
    dump(os.path.join(d, "DRY.json"), {"time": now(), "argv": fgr, "exit": dry.returncode,
                                        "stdout": dry.stdout[-4000:], "stderr": dry.stderr[-4000:]})
    if dry.returncode != 78:
        print("dry run returned", dry.returncode, "(expected 78); stopping")
        sys.exit(4)
    env.pop("FORGE_GPU_DRY_RUN")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    attempts = []
    log = os.path.join(d, "run.log")
    dump(os.path.join(d, "START.json"), {"time": now(), "argv": fgr, "module_sha256": got, "cwd": os.getcwd()})
    for attempt in range(7):
        if os.path.exists(inner_file):
            os.replace(inner_file, inner_file + f".attempt{attempt}")  # keep, never delete
        cur = sha(MODULE)
        if cur != expect:  # L103 again before every attempt: the module must not change while a line waits
            dump(os.path.join(d, "REFUSED.json"), {"time": now(), "attempt": attempt, "expected": expect,
                                                    "installed": cur})
            print("REFUSED before attempt", attempt, ": module sha256", cur, "!= expected", expect)
            sys.exit(3)
        t0 = now()
        with open(log, "ab") as lf:
            lf.write(f"\n=== attempt {attempt} {t0}\n".encode())
            lf.flush()
            rc = subprocess.run(fgr, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        attempts.append({"attempt": attempt, "start": t0, "end": now(), "forge_exit": rc})
        dump(os.path.join(d, "ATTEMPTS.json"), attempts)
        if rc != 76:
            break
        if attempt < 6:
            time.sleep(300)
    inner_rc = None
    if os.path.exists(inner_file):
        with open(inner_file) as f:
            inner_rc = int(f.read().strip() or -1)
    final = attempts[-1]["forge_exit"]
    res = {"line": line, "end": now(), "forge_exit": final, "inner_exit": inner_rc,
           "timed_out": final == 124, "attempts": attempts, "log": log, "log_sha256": sha(log),
           "module_sha256": got, "module_sha256_after": sha(MODULE),
           "pass": final == 0 and inner_rc == 0}
    dump(os.path.join(d, "EXIT.json"), {"time": now(), "forge_exit": final, "inner_exit": inner_rc})
    dump(os.path.join(d, "RESULT.json"), res)
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
