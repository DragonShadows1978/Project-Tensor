#!/usr/bin/env python3
"""Foreground A2 supervisor; parent receipt written only after child exit.
Prior art: GNU timeout/Python subprocess; DeMillo et al. (1978) valid kills.
A2: numbered admission attempts and evidence.py order-specific recomputation.
"""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from evidence import TOOL, guard, verify_start, lines, evaluate, forge_outcome, attempts, recompute_line, regression_reasons, read, sha, not_admitted, timeout_cause, scratch_guard


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    label, expected_s, bound_s, *command = sys.argv[1:]
    bound = int(bound_s)
    assert 0 < bound < 1200 and command
    runs = guard(os.environ["RUNS"])
    seal, start_record = verify_start(runs)
    line = lines()[label]
    expected = [int(v) for v in expected_s.split(",")]
    assert expected == line["expected_child_exits"], "expected exits must match gpu_lines.json"
    assert command[-3:] == ["run_gpu_tests.py", line["arm"], line["suite"]], "parent command arm/suite mismatch"
    audit = os.environ.get("GN1_AUDIT") == "1"
    assert audit == start_record["audit"], "START audit mismatch"
    if not audit and line["suite"] == "mutation":
        assert recompute_line(runs, "candidate_focus", seal, start_record)["passed"], "mutations require recomputed passing candidate focus"
    if not audit and label == "candidate_regression":
        assert recompute_line(runs, "baseline_regression", seal, start_record)["passed"], "candidate regression requires valid baseline"
    if not audit:
        assert shutil.disk_usage("/mnt/ForgeRealm").free >= 25 * 1024**3, "NVMe free-space rail"
        assert shutil.disk_usage("/").free >= 20 * 1024**3, "root free-space rail"
    base = runs / label
    previous = attempts(runs, label)
    if previous:
        prior = recompute_line(runs, label, seal, start_record)
        if not all(p["admission"] == "not_admitted" for p in prior["attempts"]):
            print(json.dumps(dict(error="label already has an admitted attempt (including incomplete child)", prior=prior)), file=sys.stderr)
            return 1
    output = base / ("attempt-%04d" % (len(previous) + 1))
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, RUNS=str(runs), GN1_LINE=label, GN1_ATTEMPT=str(output),
               TMPDIR=os.environ["TMPDIR"], FORGE_GPU_RUN_LOG=str(output / "forge.log"),
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    env["TMPDIR"] = str(scratch_guard(env["TMPDIR"]))
    Path(env["TMPDIR"]).mkdir(exist_ok=True, parents=True)
    start = utc()
    with (output / "stdout.log").open("x") as log:
        child = subprocess.run(["timeout", "--kill-after=10s", str(bound)+"s", *command],
                               cwd=TOOL, env=env, stdout=log, stderr=subprocess.STDOUT)
    rc = child.returncode
    completed = utc()
    cause = forge_outcome(output / "forge.log", rc)
    refused = not_admitted(output, rc, cause)
    result = (dict(passed=False, reasons=["not admitted: " + cause], actual_killers=[], outcomes={}, forge_outcome=cause, timeout=False, timeout_cause=timeout_cause(output / "forge.log", rc), unsealed=[])
              if refused else evaluate(runs, output, line, rc, audit, seal))
    if not refused and not audit and line["suite"] == "mutation":
        try:
            dependency = recompute_line(runs, "candidate_focus", seal, start_record)
            if not dependency["passed"]:
                result["reasons"].append("mutations require recomputed passing candidate focus")
        except (AssertionError, KeyError, OSError, ValueError) as exc:
            result["reasons"].append("invalid candidate focus: " + str(exc))
    if not refused and not audit and label == "candidate_regression":
        try:
            old = recompute_line(runs, "baseline_regression", seal, start_record)
            if not old["passed"]:
                result["reasons"].append("missing valid completed baseline regression")
            else:
                result["reasons"] += regression_reasons(read(Path(old["selected_attempt"]) / "child_result.json"), read(output / "child_result.json"))
        except (AssertionError, KeyError, OSError, ValueError) as exc:
            result["reasons"].append("invalid baseline regression: " + str(exc))
        (output / "regressions.json").write_text(json.dumps(result["reasons"], indent=2)+"\n")
    result["passed"] = not result["reasons"]
    if not result["passed"]:
        result["actual_killers"] = []
    receipt = dict(label=label, arm=line["arm"], suite=line["suite"], start=start,
                   child_exited=completed, recorded=utc(), cwd=str(TOOL), command=command,
                   expected_child_exits=expected, exit=rc, outer_bound_seconds=bound,
                   audit=audit,
                   run_root=str(runs), attempt_dir=str(output), seal_sha256=sha(TOOL / "staging_seal.json"),
                   admission="not_admitted" if refused else "admitted", **result)
    with (output / "parent_result.json").open("x") as stream:
        stream.write(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
