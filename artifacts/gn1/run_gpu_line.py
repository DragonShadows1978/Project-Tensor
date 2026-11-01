#!/usr/bin/env python3
"""Foreground forge supervisor: write L45 receipt only AFTER child exits.
Prior art: GNU timeout; Python subprocess.run (PSF); DeMillo/Lipton/Sayward
(1978) mutation kills. GN1 order supplies acceptance and parent-receipt rules.
"""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

TOOL = Path(__file__).resolve().parent
ROOT = TOOL.parents[1]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def evaluate(label, rc, expected, data, audit):
    reasons = []
    if rc != expected:
        reasons.append(f"expected exit {expected}, got {rc}")
    if not data or data.get("pytest_exit") != rc:
        reasons.append("missing or inconsistent completed pytest receipt")
        return reasons
    if data["collection_errors"]:
        reasons.append("collection errors")
    if audit:
        if len(data["collected_ids"]) != 5:
            reasons.append("audit did not run exactly five CPU oracle IDs")
    else:
        census = "ids_mutation.txt" if data["suite"] == "mutation" else "ids_" + data["suite"] + ".txt"
        expected_ids = (TOOL / census).read_text().splitlines()
        if sorted(data["collected_ids"]) != sorted(expected_ids):
            reasons.append("test-ID census mismatch")
    reports = data["reports"]
    calls = {r["id"]: r for r in reports if r["when"] == "call"}
    setup_skips = {r["id"] for r in reports if r["when"] == "setup" and r["outcome"] == "skipped"}
    accounted = set(calls) | (setup_skips if data["suite"] == "regression" else set())
    if accounted != set(data["collected_ids"]):
        reasons.append("missing test calls (setup error or unregistered skip)")
    if any(r["outcome"] == "failed" and r["when"] != "call" for r in reports):
        reasons.append("setup/teardown failures")
    failures = {key: r for key, r in calls.items() if r["outcome"] == "failed"}
    # Do not count an unavailable engine, driver failure, or dtype setup as a kill.
    bad = ("cudaMalloc", "no CUDA-capable device", "CUDA driver", "cudaError",
           "initialization error", "invalid device ordinal")
    if any(any(word in r["detail"] for word in bad) for r in failures.values()):
        reasons.append("device/setup failure")
    if not audit and data["suite"] == "focus" and data["arm"] == "baseline":
        registered = set(json.loads((TOOL / "baseline_expected_failures.json").read_text()))
        if set(failures) != registered:
            reasons.append("baseline failures differ from registration")
        refusal = "tensor_cuda/tests/test_groupnorm_gn1.py::test_nondivisible_refusal"
        if any("reshape: element count mismatch" not in r["detail"] for key, r in failures.items() if key != refusal):
            reasons.append("baseline numerical failure is not registered reshape error")
    elif not audit and data["suite"] == "mutation":
        if not failures:
            reasons.append("mutant survived")
    elif expected == 0 and any(r["outcome"] != "passed" for r in calls.values()):
        # Existing optional skips in the regression suite retain their IDs and
        # are compared below; focus and mutants permit no skips.
        if data["suite"] != "regression":
            reasons.append("candidate or CPU reference ID did not pass")
    return reasons


def main():
    label, expected_s, bound_s, *command = sys.argv[1:]
    expected, bound = int(expected_s), int(bound_s)
    assert 0 < bound < 1200 and command
    runs = Path(os.environ["RUNS"]).resolve()  # no default
    runs.mkdir(parents=True, exist_ok=True)
    output = runs / label
    output.mkdir(exist_ok=False)  # fresh attempts only; never overwrite evidence
    audit = os.environ.get("GN1_AUDIT") == "1"
    if not audit:
        assert shutil.disk_usage("/mnt/ForgeRealm").free >= 25 * 1024**3, "NVMe free-space rail"
        assert shutil.disk_usage("/").free >= 20 * 1024**3, "root free-space rail"
    env = dict(os.environ, RUNS=str(runs), GN1_LINE=label, TMPDIR=str(runs / "tmp"),
               FORGE_GPU_RUN_LOG=str(output / "forge.log"), PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1")
    Path(env["TMPDIR"]).mkdir(exist_ok=True)
    if "mutation" in command:
        accepted = runs / "candidate_focus" / "parent_result.json"
        assert accepted.exists() and json.loads(accepted.read_text())["passed"], "mutations require passing candidate focus"
    start = utc()
    # No background process or polling. timeout supervises only our child tree.
    with (output / "stdout.log").open("x") as log:
        child = subprocess.run(["timeout", "--kill-after=10s", str(bound) + "s", *command],
                               cwd=TOOL, env=env, stdout=log, stderr=subprocess.STDOUT)
    rc = child.returncode
    completed = utc()
    data_file = output / "child_result.json"
    data = json.loads(data_file.read_text()) if data_file.exists() else None
    reasons = evaluate(label, rc, expected, data, audit)
    if data and data["suite"] == "regression" and data["arm"] == "candidate":
        baseline = runs / "baseline_regression" / "child_result.json"
        baseline_receipt = runs / "baseline_regression" / "parent_result.json"
        if not baseline.exists() or not baseline_receipt.exists() or not json.loads(baseline_receipt.read_text())["passed"]:
            reasons.append("missing valid completed baseline regression receipt")
        else:
            old = json.loads(baseline.read_text())
            passed_before = {r["id"] for r in old["reports"] if r["when"] == "call" and r["outcome"] == "passed"}
            passed_after = {r["id"] for r in data["reports"] if r["when"] == "call" and r["outcome"] == "passed"}
            regressions = sorted(passed_before - passed_after)
            (output / "regressions.json").write_text(json.dumps(regressions, indent=2) + "\n")
            if regressions:
                reasons.append("baseline-passing IDs no longer pass")
    killers = sorted({r["id"] for r in (data or {}).get("reports", []) if r["when"] == "call" and r["outcome"] == "failed"})
    receipt = dict(label=label, start=start, child_exited=completed, recorded=utc(),
                   cwd=str(TOOL), command=command, expected_child_exit=expected, exit=rc,
                   outer_bound_seconds=bound, timeout=rc in (124,137), audit=audit,
                   passed=not reasons, reasons=reasons,
                   actual_killers=killers if data and data["suite"] == "mutation" else [])
    (output / "parent_result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
