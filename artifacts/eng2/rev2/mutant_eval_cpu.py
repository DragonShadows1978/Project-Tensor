#!/usr/bin/env python3
"""ENG2-A1 A-G2 / L41 / L16: real CPU MF1 and MF2 receipts scored by the FIXED evidence.evaluate() (REV1 E2R-F03).

The staged child (tool/run_gpu_tests.py, sealed, unchanged) runs each arm with CUDA hidden, as a NON-audit line (so
evaluate() takes its mutant branch), under a1/cpu_runner_standin.py (the sealed audit stand-in with one ledgered
change), in the fresh root tool/gpu_runs/audit_2/cpu_mutants. The CPU subset is selected with PYTEST_ADDOPTS="-k cpu"
(pytest's own option variable; the child is not edited). Attempts: candidate_eng2, MF1, MF2.
Scoring calls evidence.evaluate() unchanged, with ONE declared substitution: evidence.census returns the CPU census
(tool/ids_audit.txt, the "-k cpu" collection) instead of the whole eng2 census, because a CPU child collects only that
subset. Nothing else is patched for the kill verdicts.
Checks:
  K1  MF1 and MF2: evaluate() passes, actual_killers == the registered killers that failed (all of them on CPU), and
      the child's mutant_trace shows every killer executing the registered hunk (L41).
  K2  candidate_eng2: evaluate() passes as a candidate line (no failure).
  K3  treatment effect: the PRE-FIX matcher (a1/before/artifacts__eng2__tool__evidence.py) matches none of the same
      failing IDs (the REV1 E2R-F03 defect, reproduced on the real receipt).
  S1  constructed survivor (L16): the real MF1 receipt scored against MF2's killer set (none of which MF1 fails):
      evaluate() must report "no registered killer ID failed" and no killers (killers_for substituted, declared).
  S2  constructed survivor (L16): the real CANDIDATE receipt scored as a mutant line with MF1's killers (the candidate
      is an "equivalent mutant" that changes nothing): must report "no registered killer ID failed".
Writes artifacts/eng2/a1/MUTANT_EVAL_CPU.json.
Prior art: DeMillo, Lipton and Sayward (1978), a mutant is killed by a failing registered test; the ENG2 tool's own
evaluate(); no new algorithm.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

A1 = Path(__file__).resolve().parent
TOOL = A1.parent / "tool"
sys.path.insert(0, str(TOOL))
import evidence  # noqa: E402

RUNS = TOOL / "gpu_runs" / "audit_2" / "cpu_mutants"
# REV2: replay real A1 receipts without executing or writing under the tool.
seal = evidence.verify_seal()
cpu_census = (TOOL / "ids_audit.txt").read_text().splitlines()
evidence.census = lambda line, audit=False: cpu_census
out = dict(root=str(RUNS), substitution="CPU census only", lines={})
attempt_of = {}
for label in ["candidate_eng2", "MF1", "MF2"]:
    attempt = RUNS / label / "attempt-0001"
    attempt_of[label] = (attempt, evidence.read(attempt / "child_result.json")["pytest_exit"])
lines = evidence.lines()
for label, (attempt, rc) in attempt_of.items():
    line = lines[label]
    result = evidence.evaluate(RUNS, attempt, line, rc, False, seal)
    data = evidence.read(attempt / "child_result.json")
    failed = sorted(i for i, o in result["outcomes"].items() if o in ("failed", "error"))
    row = dict(child_exit=rc, expected=line["expected_child_exits"], passed=result["passed"],
               reasons=result["reasons"], failed=len(failed), forge_outcome=result["forge_outcome"],
               collected=len(data["collected_ids"]))
    if line["role"] == "mutant":
        killers = sorted(evidence.norm_id(k) for k in evidence.killers_for(label))
        trace = {evidence.norm_id(k): v for k, v in (data["mutant_trace"] or {}).items()}
        row.update(registered_killers=len(killers), actual_killers=len(result["actual_killers"]),
                   actual_equals_registered=sorted(result["actual_killers"]) == killers,
                   failed_equals_registered=failed == killers,
                   killers_traced=sum(bool(set(trace.get(k, [])) & set(data["mutant_hunk"]["lines"])) for k in killers), hunk=data["mutant_hunk"])
        spec = importlib.util.spec_from_file_location("evidence_before_a1", A1.parent / "a1" / "before" / "artifacts__eng2__tool__evidence.py")
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        row["prefix_matcher_matches"] = sum(bool(old.matches(i, evidence.killers_for(label))) for i in failed)
        row["K1"] = bool(result["passed"] and row["actual_equals_registered"] and row["failed_equals_registered"]
                         and row["killers_traced"] == len(killers) and rc == 1)
        row["K3"] = row["prefix_matcher_matches"] == 0
    else:
        row["K2"] = bool(result["passed"] and not failed and rc == 0)
    out["lines"][label] = row
# S1: MF1's real receipt against MF2's killers.
real_killers_for = evidence.killers_for
evidence.killers_for = lambda label: real_killers_for("MF2")
s1 = evidence.evaluate(RUNS, attempt_of["MF1"][0], lines["MF1"], attempt_of["MF1"][1], False, seal)
evidence.killers_for = real_killers_for
out["S1"] = dict(reasons=s1["reasons"], actual_killers=s1["actual_killers"], passed=s1["passed"],
                 ok=(not s1["passed"] and "no registered killer ID failed" in s1["reasons"] and not s1["actual_killers"]))
# S2: the candidate's real receipt scored as a mutant line with MF1's killers.
line = dict(lines["MF1"], arm="candidate", suite="eng2", expected_child_exits=[0])
s2 = evidence.evaluate(RUNS, attempt_of["candidate_eng2"][0], line, attempt_of["candidate_eng2"][1], False, seal)
out["S2"] = dict(reasons=s2["reasons"], actual_killers=s2["actual_killers"], passed=s2["passed"],
                 ok=(not s2["passed"] and "no registered killer ID failed" in s2["reasons"] and not s2["actual_killers"]))
out["all_ok"] = bool(out["lines"]["MF1"]["K1"] and out["lines"]["MF2"]["K1"] and out["lines"]["MF1"]["K3"]
                     and out["lines"]["MF2"]["K3"] and out["lines"]["candidate_eng2"]["K2"] and out["S1"]["ok"]
                     and out["S2"]["ok"])
(A1 / "MUTANT_EVAL_CPU.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(dict(lines={k: {q: v for q, v in r.items() if q != "hunk"} for k, r in out["lines"].items()},
                      S1=out["S1"], S2=out["S2"], all_ok=out["all_ok"]), indent=1))
sys.exit(0 if out["all_ok"] else 1)
