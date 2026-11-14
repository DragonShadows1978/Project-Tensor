#!/usr/bin/env python3
"""Foreground ENG2 supervisor (copied from ENG1's tool as A3 left it; adapted from GN1 A8 run_gpu_line.py). ENG2 changes
(ledgered): ENG2_* environment names; /tmp/eng2_* scratch (evidence.scratch_guard); mutant lines depend on a passing
candidate_eng2 line. Everything else is ENG1's: the parent receipt, with the child's exit and
timeout flag, is written only after the child exits (L45).
Prior art: GNU timeout / Python subprocess; DeMillo et al. (1978) valid kills; POSIX link(2) no-overwrite
publication (IEEE Std 1003.1; year unverified -- lead to check).
ENG1 changes (ledgered): ENG1_* environment names; /tmp/eng1_* scratch; mutant lines depend on a passing
candidate_eng1 line; GN1's control-mode and program-binding conjunctions dropped (their program_binding.py is not in
the tree); the timeout flag is recorded explicitly (child exit 124/137 under GNU timeout, or the runner's timeout).
Usage: run_gpu_line.py LABEL EXPECTED_EXITS OUTER_BOUND RUNNER [runner args] -- timeout ... run_gpu_tests.py ARM SUITE
"""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from evidence import (SEAL_NAME, TOOL, guard, verify_start, lines, evaluate, forge_outcome, attempts, recompute_line,
                      regression_reasons, read, sha, not_admitted, timeout_cause, scratch_guard)


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
    audit = os.environ.get("ENG2_AUDIT") == "1"
    assert audit == start_record["audit"], "START audit mismatch"
    if not audit and line["role"] == "mutant":
        assert recompute_line(runs, "candidate_eng2", seal, start_record)["passed"], "mutations require a passing candidate_eng2"
    if not audit and line["role"] == "regression" and line["arm"] == "candidate":
        assert recompute_line(runs, label.replace("candidate_", "baseline_", 1), seal, start_record)["passed"], "candidate regression requires a valid baseline"
    if not audit:
        assert shutil.disk_usage("/mnt/ForgeRealm").free >= 25 * 1024**3, "NVMe free-space rail"
        assert shutil.disk_usage("/").free >= 20 * 1024**3, "root free-space rail"
    base = runs / label
    previous = attempts(runs, label)
    if previous:
        try:
            prior = recompute_line(runs, label, seal, start_record)
        except Exception as exc:
            print(json.dumps(dict(error="cannot evaluate prior attempt: " + (str(exc) or type(exc).__name__))), file=sys.stderr)
            return 1
        if not all(p["admission"] == "not_admitted" for p in prior["attempts"]):
            print(json.dumps(dict(error="label already has an admitted attempt (including incomplete child)", prior=prior)), file=sys.stderr)
            return 1
    output = base / ("attempt-%04d" % (len(previous) + 1))
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, RUNS=str(runs), ENG2_LINE=label, ENG2_ATTEMPT=str(output),
               FORGE_GPU_RUN_LOG=str(output / "forge.log"), PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    env["TMPDIR"] = str(scratch_guard(os.environ["TMPDIR"]))
    Path(env["TMPDIR"]).mkdir(exist_ok=True, parents=True)
    start = utc()
    with (output / "stdout.log").open("x") as log:
        child = subprocess.run(["timeout", "--kill-after=20s", str(bound) + "s", *command],
                               cwd=TOOL, env=env, stdout=log, stderr=subprocess.STDOUT)
    # GNU timeout / POSIX 128+signal convention.
    rc = child.returncode if child.returncode >= 0 else 128 - child.returncode
    completed = utc()
    result = dict(passed=False, reasons=[], actual_killers=[], outcomes={}, forge_outcome="unreadable",
                  timeout=False, timeout_cause="unattributed")
    refused = False
    seal_sha256 = None
    try:
        seal_sha256 = sha(TOOL / SEAL_NAME)
        cause = forge_outcome(output / "forge.log", rc)
        refused = not_admitted(output, rc, cause)
        result = (dict(passed=False, reasons=["not admitted: " + cause], actual_killers=[], outcomes={},
                       forge_outcome=cause, timeout=False, timeout_cause=timeout_cause(output / "forge.log", rc))
                  if refused else evaluate(runs, output, line, rc, audit, seal))
        if not refused and not audit and line["role"] == "mutant":
            dependency = recompute_line(runs, "candidate_eng2", seal, start_record)
            if not dependency["passed"]:
                result["reasons"].append("mutations require a passing candidate_eng2")
        if not refused and not audit and line["role"] == "regression" and line["arm"] == "candidate":
            old = recompute_line(runs, label.replace("candidate_", "baseline_", 1), seal, start_record)
            if not old["passed"]:
                result["reasons"].append("missing valid completed baseline regression")
            else:
                result["reasons"] += regression_reasons(read(Path(old["selected_attempt"]) / "child_result.json"),
                                                        read(output / "child_result.json"))
            (output / "regressions.json").write_text(json.dumps(result["reasons"], indent=2) + "\n")
    except Exception as exc:
        result["reasons"].append("parent evaluation: " + (str(exc) or type(exc).__name__))
    result["passed"] = not result["reasons"]
    if not result["passed"]:
        result["actual_killers"] = []
    # The timeout flag: GNU timeout's 124 (or 137 after --kill-after), attributed through the runner's log.
    timed_out = rc in (124, 137) or result.get("timeout") is True
    receipt = dict(result, label=label, arm=line["arm"], suite=line["suite"], role=line["role"], start=start,
                   child_exited=completed, recorded=utc(), cwd=str(TOOL), command=command,
                   expected_child_exits=expected, exit=rc, raw_exit=child.returncode, timeout=timed_out,
                   outer_bound_seconds=bound, audit=audit, run_root=str(runs), attempt_dir=str(output),
                   seal_sha256=seal_sha256, admission="not_admitted" if refused else "admitted")

    def publish(data):
        fd, temporary = tempfile.mkstemp(prefix="parent_result.", suffix=".tmp", dir=output)
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(data, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output / "parent_result.json")

    try:
        publish(receipt)
    except Exception as exc:
        minimal = dict(label=label, attempt_dir=str(output), run_root=str(runs), child_exited=completed,
                       recorded=utc(), raw_exit=child.returncode, exit=rc, timeout=timed_out, passed=False,
                       actual_killers=[], audit=audit, receipt_error=True,
                       reasons=["parent receipt write failed: " + (str(exc) or type(exc).__name__)])
        try:
            publish(minimal)
        except Exception as retry:
            print(json.dumps(dict(error=minimal["reasons"][0], retry_error=str(retry) or type(retry).__name__)), file=sys.stderr)
            return 1
        print(json.dumps(minimal))
        return 1
    print(json.dumps(dict((k, receipt[k]) for k in ("label", "exit", "timeout", "passed", "reasons", "actual_killers"))))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    try:
        status = main()
    except Exception as exc:
        print(json.dumps(dict(error=str(exc) or type(exc).__name__)), file=sys.stderr)
        status = 1
    sys.exit(status)
