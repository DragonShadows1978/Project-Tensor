"""ENG5-S-T shared helpers (orders/ENG5ST_REGRESSION_TEST.md): the arms, the module-hash check (L103), and run_arm(), the
body of the staged GPU step (scripts/eng5st_gpu_step.py is the lease check plus a call to run_arm). Pure Python: it
never imports the engine and never touches CUDA_VISIBLE_DEVICES (a child of run_arm inherits the caller's environment
unchanged). Kept out of the step script so a CPU drive (scripts/eng5st_verdict.py selftest) can exercise run_arm with a
stand-in pytest command and no lease.

Prior art: ENG5-S's scripts/eng5s_gpu_step.py and scripts/eng5s_gpu_line.py (this repo), whose arm table, module check
and write-once layout this copies; the house forge-gpu-run contract. No other prior art known to me.
"""
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SO = "_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
TEST_FILE = "tensor_cuda/tests/test_eng5s_status_checks.py"
T_DIR = os.path.join(ROOT, "artifacts/eng5s/t")
TMP = "/tmp/eng5st_tmp"
ARMS = {"S": "S", "B": "B", "s01": "s01_clone_status_dropped", "s02": "s02_to_status_dropped",
        "s03": "s03_zeros_status_dropped", "s04": "s04_from_host_last_error_only"}
PYTEST_TIMEOUT = 420  # inside the line's 480 s inner timeout


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def receipt_path(arm):
    return os.path.join(ROOT, f"artifacts/eng5s/build/BUILD_{ARMS[arm]}.json")


def pkg(arm):
    return os.path.join(ROOT, "tensor_cuda", "build-eng5s", ARMS[arm], "pkg")


def module_check(arm):
    """(ok, got, want): the arm's package module against its build receipt's module_sha256 (L103)."""
    want = json.load(open(receipt_path(arm)))["module_sha256"]
    so = os.path.join(pkg(arm), "tensor_cuda", SO)
    got = sha256_file(so) if os.path.exists(so) else None
    return got == want, got, want


def line_dir(arm):
    return os.path.join(T_DIR, "gpu", arm)


def forge_argv(arm):
    return ["forge-gpu-run", "--min", "120", "--max", "540", "--label", f"eng5st-{arm}", "--", "timeout", "480s",
            "python3", "-B", "-P", "scripts/eng5st_gpu_step.py", arm]


def run_arm(arm, out_dir=None, lane_visible=None, pytest_argv=None, pkg_dir=None, tmp=TMP):
    """The step's body. Returns the step's exit code (0: it ran to a verdict; the verdict is in VERDICT.json).
    Refuses (5) if the line directory exists; (7) if the GPU lane is not visible; (3) if the module's sha256 differs
    from its receipt. Never changes CUDA_VISIBLE_DEVICES. `out_dir`, `lane_visible`, `pytest_argv` and `pkg_dir` are
    the CPU drive's hooks; the staged line passes none of them."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import eng5st_verdict as vd
    d = out_dir or line_dir(arm)
    if os.path.exists(d):
        print("REFUSED: write-once, exists:", d)
        return 5
    os.makedirs(d)
    visible = (os.environ.get("CUDA_VISIBLE_DEVICES") != "") if lane_visible is None else lane_visible
    if not visible:
        open(os.path.join(d, "INNER_EXIT"), "w").write("7")
        print("REFUSED: CUDA_VISIBLE_DEVICES is empty in this lease's environment (never changed by this script)")
        return 7
    ok, got, want = module_check(arm) if pkg_dir is None else (True, "drive", "drive")
    if not ok:
        open(os.path.join(d, "INNER_EXIT"), "w").write("3")
        print(f"REFUSED (L103): {arm} module {got} != receipt {want}")
        return 3
    os.makedirs(os.path.join(tmp, "pt"), exist_ok=True)
    junit = os.path.join(d, "junit.xml")
    marker = os.path.join(d, "mutant_marker.txt")
    pp = pkg_dir or pkg(arm)
    env = dict(os.environ, TMPDIR=tmp, ENG5S_MUTANT_MARKER=marker,
               PYTHONPATH=pp + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    argv = pytest_argv or [sys.executable, "-B", "-P", "-m", "pytest", "-p", "no:cacheprovider", "-q", "-rfEs",
                           "--rootdir=" + ROOT, f"--basetemp={tmp}/pt/gpu_{arm}", "--junitxml=" + junit,
                           os.path.join(ROOT, TEST_FILE)]
    inner = 0
    try:
        p = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=PYTEST_TIMEOUT, cwd=ROOT)
        rc, out = p.returncode, p.stdout[-6000:] + p.stderr[-3000:]
    except subprocess.TimeoutExpired as ex:
        rc, out, inner = "timeout", str(ex.stdout or "")[-3000:], 124
    open(os.path.join(d, "pytest.log"), "w").write(out)
    marks = [l.strip() for l in open(marker)] if os.path.exists(marker) else []
    verdict = vd.judge(arm, junit, marks)
    res = {"arm": arm, "module_sha256": got if pkg_dir is None else None, "pytest_exit": rc,
           "junit_sha256": sha256_file(junit) if os.path.exists(junit) and os.path.getsize(junit) else None,
           "executed_marker_lines": len(marks), "verdict": verdict}
    json.dump(verdict, open(os.path.join(d, "VERDICT.json"), "w"), indent=1)
    json.dump(res, open(os.path.join(d, "STEP_RESULT.json"), "w"), indent=1)
    open(os.path.join(d, "INNER_EXIT"), "w").write(str(inner))
    print("arm", arm, "verdict", verdict["verdict"], "pytest exit", rc)
    return inner
