#!/usr/bin/env python3
"""GN1-REV1 Q5 hostile runs through the REAL staged parent and scorer, on CPU.

The real run_gpu_line.py and score_gpu.py refuse a RUNS root outside
artifacts/gn1/a1/ (read-only for this review), so this harness builds a mirror
under /tmp/gn1_rev1_mirror/W/artifacts/gn1/a1/ holding BYTE-IDENTICAL copies
(sha256 checked against staging_seal.json) of the parent, the scorer and the
census files they read. The parent's TOOL is then the mirror directory; no
line of the parent or scorer is changed. A stand-in for forge-gpu-run plays
each hostile child. Each child's exit is recorded by the real parent after
the child exits (L45); this harness records the parent's exit after it exits.

Prior art: fault injection / test doubles (Meszaros, xUnit Test Patterns,
2007: "test stub"); mutation-analysis validity (DeMillo, Lipton and Sayward
1978). Ours: the specific hostile cases.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REAL = HERE.parents[2]
A1 = REAL / "artifacts/gn1/a1"
MIRROR = Path("/tmp/gn1_rev1_mirror/W")
MA1 = MIRROR / "artifacts/gn1/a1"
FILES = ["run_gpu_line.py", "score_gpu.py", "ids_focus.txt", "ids_mutation.txt", "ids_regression.txt",
         "ids_audit.txt", "baseline_expected_failures.json", "gpu_lines.json"]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def setup():
    MA1.mkdir(parents=True, exist_ok=True)
    seal = json.loads((A1 / "staging_seal.json").read_text())["tool_files"]
    proof = {}
    for f in FILES:
        shutil.copyfile(A1 / f, MA1 / f)
        proof[f] = dict(mirror=sha(MA1 / f), sealed=seal.get(f), identical=sha(MA1 / f) == seal.get(f))
    standin = MIRROR / "hostile_forge.py"
    standin.write_text(STANDIN)
    return proof, standin


STANDIN = r'''#!/usr/bin/env python3
# Hostile stand-in for forge-gpu-run: never runs the registered child; plays HOSTILE_MODE.
import json, os, sys
from pathlib import Path
mode = os.environ["HOSTILE_MODE"]
out = Path(os.environ["RUNS"]) / os.environ["GN1_LINE"]
tool = Path(os.environ["HOSTILE_TOOL"])
ids = lambda f: (tool / f).read_text().splitlines()
def write(arm, suite, rc, reports, collected):
    out.mkdir(parents=True, exist_ok=True)
    (out / "child_result.json").write_text(json.dumps(dict(arm=arm, suite=suite, audit=False, pytest_exit=rc,
        collected_ids=collected, collection_errors=[], reports=reports)))
call = lambda i, o, d="": dict(id=i, when="call", outcome=o, detail=d)
if mode == "no_window":
    sys.exit(75)
if mode == "device_error_kill":
    col = ids("ids_mutation.txt")
    write("M07", "mutation", 1, [call(i, "failed", "RuntimeError: CUDA error at reduce_lastdim: an illegal memory access was encountered") for i in col], col)
    sys.exit(1)
if mode == "foreign_arm_kill":
    # the child ran the BASELINE arm (reshape errors), not the mutant the line names
    col = ids("ids_mutation.txt")
    write("baseline", "mutation", 1, [call(i, "failed", "RuntimeError: reshape: element count mismatch") if "values" in i else call(i, "passed") for i in col], col)
    sys.exit(1)
if mode == "junit_contradiction_pass":
    col = ids("ids_focus.txt")
    write("candidate", "focus", 0, [call(i, "passed") for i in col], col)
    (out / "junit.xml").write_text('<?xml version="1.0"?><testsuites><testsuite name="pytest" tests="3" failures="3" errors="0"><testcase name="x"><failure message="real failure"/></testcase></testsuite></testsuites>')
    sys.exit(0)
if mode == "hang_after_result":
    col = ids("ids_focus.txt")
    write("candidate", "focus", 0, [call(i, "passed") for i in col], col)
    import time; time.sleep(40)
    sys.exit(0)
if mode == "device_error_control":
    col = ids("ids_mutation.txt")
    write("M08", "mutation", 1, [call(i, "failed", "RuntimeError: cudaErrorIllegalAddress") for i in col], col)
    sys.exit(1)
if mode == "suite_swap_pass":
    # a MUTANT line whose child claims to be a passing candidate focus run
    col = ids("ids_focus.txt")
    write("candidate", "focus", 0, [call(i, "passed") for i in col], col)
    sys.exit(0)
sys.exit(99)
'''


def parent(label, expected, mode, standin, runs, command_tail, bound="60"):
    env = dict(os.environ, RUNS=str(runs), TMPDIR="/tmp/gn1_rev1_scratch", HOSTILE_MODE=mode,
               HOSTILE_TOOL=str(MA1), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    env.pop("GN1_AUDIT", None)
    cmd = [sys.executable, "-B", str(MA1 / "run_gpu_line.py"), label, str(expected), bound, sys.executable, str(standin),
           "--min", "1", "--max", "30", "--label", "hostile", "--", "timeout", "20s", "python3"] + command_tail
    p = subprocess.run(cmd, cwd=MA1, env=env, capture_output=True, text=True, timeout=120)
    receipt = runs / label / "parent_result.json"
    data = json.loads(receipt.read_text()) if receipt.exists() else None
    return dict(label=label, mode=mode, parent_exit=p.returncode, receipt_passed=(data or {}).get("passed"),
                reasons=(data or {}).get("reasons"), actual_killers_n=len((data or {}).get("actual_killers", [])),
                stderr_tail=p.stderr[-300:])


def main():
    proof, standin = setup()
    results = dict(mirror_identity=proof)
    import time
    stamp = time.strftime("%H%M%S")
    # H1/H2/H3: mutation-line kills without the registered outcome; H4: JUnit ignored; H5: no-window burns label
    runs = MA1 / ("runs_" + stamp)
    runs.mkdir(parents=True)
    # mutation lines require a passing candidate_focus receipt in the same root: provide it via H4 (forged child)
    results["H4_candidate_focus_junit_contradiction"] = parent("candidate_focus", 0, "junit_contradiction_pass", standin,
                                                               runs, ["run_gpu_tests.py", "candidate", "focus"])
    results["H2_M07_device_error_counted_as_kill"] = parent("M07", 1, "device_error_kill", standin, runs,
                                                            ["run_gpu_tests.py", "M07", "mutation"])
    results["H3_M09_foreign_arm_counted_as_kill"] = parent("M09", 1, "foreign_arm_kill", standin, runs,
                                                           ["run_gpu_tests.py", "M09", "mutation"])
    results["H3b_M11_child_claims_candidate_focus"] = parent("M11", 1, "suite_swap_pass", standin, runs,
                                                             ["run_gpu_tests.py", "M11", "mutation"])
    results["H2c_control_cudaError_word_rejected"] = parent("M08", 1, "device_error_control", standin, runs,
                                                            ["run_gpu_tests.py", "M08", "mutation"])
    runs2 = MA1 / ("runs_hang_" + stamp); runs2.mkdir(parents=True)
    results["H6_child_hangs_after_writing_pass_result"] = parent("candidate_focus", 0, "hang_after_result", standin,
                                                                 runs2, ["run_gpu_tests.py", "candidate", "focus"], bound="5")
    results["H5_no_window_first"] = parent("baseline_focus", 1, "no_window", standin, runs,
                                           ["run_gpu_tests.py", "baseline", "focus"])
    results["H5_no_window_retry_same_root"] = parent("baseline_focus", 1, "no_window", standin, runs,
                                                     ["run_gpu_tests.py", "baseline", "focus"])
    # H1: forged parent receipts only (no START.json, no child_result, no logs) through the real scorer
    forged = MA1 / ("forged_" + stamp)
    lines = json.loads((MA1 / "gpu_lines.json").read_text())
    for line in lines:
        d = forged / line["label"]
        d.mkdir(parents=True)
        (d / "parent_result.json").write_text(json.dumps(dict(passed=True, timeout=False, audit=False,
                                                                actual_killers=["forged"])))
    env = dict(os.environ, RUNS=str(forged), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-B", str(MA1 / "score_gpu.py")], cwd=MA1, env=env, capture_output=True,
                       text=True, timeout=60)
    summ = json.loads((forged / "G5_SUMMARY.json").read_text())
    results["H1_scorer_forged_parent_receipts_only"] = dict(scorer_exit=p.returncode, G5_passed=summ["passed"],
                                                            start_json_present=(forged / "START.json").exists(),
                                                            child_results_present=any(forged.rglob("child_result.json")))
    # H1b: the scorer accepts a RUNS root outside a1 (the parent and child refuse one)
    outside = Path("/tmp/gn1_rev1_scratch/outside_" + stamp)
    shutil.copytree(forged, outside)
    p = subprocess.run([sys.executable, "-B", str(MA1 / "score_gpu.py")], cwd=MA1,
                       env=dict(env, RUNS=str(outside)), capture_output=True, text=True, timeout=60)
    results["H1b_scorer_runs_outside_a1"] = dict(scorer_exit=p.returncode, runs=str(outside))
    (HERE / "hostile_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
