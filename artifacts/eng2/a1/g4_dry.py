#!/usr/bin/env python3
"""ENG2 G4 device-mode dry run (adapted from ENG1-A3 G1): every label in gpu_lines.json order, as the lead's G5 runs
it, with the one addition FORGE_GPU_DRY_RUN=1 (L48):
  env -u PYTHONOPTIMIZE -u ENG2_AUDIT RUNS=<root> CUDA_VISIBLE_DEVICES=0 TMPDIR=/tmp/eng2_dry FORGE_GPU_DRY_RUN=1 \
      timeout 1190s bash artifacts/eng2/tool/STAGED_GPU_COMMANDS.sh <label>
Per label it records: the staged exit; start_gpu_run.py's record line (printed only after verify_seal() and
verify_arm() for every sealed arm passed); for root lines the attempt's parent receipt (exit = the governor's 78,
admission not_admitted) and the governor's DRY RUN line in the attempt's stdout.log; for dependent lines the
dependency refusal text and that no attempt directory was made. Then, for every label, the runner argv from the
staged script given straight to the governor in dry-run mode (S1, as A3): rc 78, no run-log line.
Usage: g4_dry.py ROOT_NAME   (a fresh root under tool/gpu_runs/, never g5_*)
Prior art: GW3 dry-run contract (System-Claude GPU_WINDOW.md, 2026); ENG1-A3's G1 and S1 scripts. No new algorithm.
"""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

G4 = Path(__file__).resolve().parent
ROOT = G4.parents[2]
TOOL = ROOT / "artifacts/eng2/tool"
name = sys.argv[1]
assert not name.startswith("g5_")
runs = TOOL / "gpu_runs" / name
assert not runs.exists(), "fresh root required"
lines = json.loads((TOOL / "gpu_lines.json").read_text())
script = (TOOL / "STAGED_GPU_COMMANDS.sh").read_text().splitlines()
rows = []
for line in lines:
    label = line["label"]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONOPTIMIZE", "ENG2_AUDIT")}
    env.update(RUNS=str(runs), CUDA_VISIBLE_DEVICES="0", TMPDIR="/tmp/eng2_a1_dry", FORGE_GPU_DRY_RUN="1")
    cmd = ["timeout", "1190s", "bash", "artifacts/eng2/tool/STAGED_GPU_COMMANDS.sh", label]
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1200, stdin=subprocess.DEVNULL)
    start_line = next((l for l in p.stdout.splitlines() if l.startswith("{") and '"seal_sha256"' in l), None)
    attempts = sorted((runs / label).glob("attempt-*")) if (runs / label).exists() else []
    rec = dict(label=label, role=line["role"], staged_exit=p.returncode, command=" ".join(shlex.quote(c) for c in cmd),
               seal_and_arms_verified=start_line is not None,
               start_record=json.loads(start_line) if start_line else None, attempts=[a.name for a in attempts],
               stdout_tail=p.stdout[-1500:], stderr_tail=p.stderr[-1500:])
    if attempts:
        parent = json.loads((attempts[-1] / "parent_result.json").read_text())
        log = (attempts[-1] / "stdout.log").read_text()
        rec.update(governor_exit=parent["exit"], admission=parent["admission"], parent_reasons=parent["reasons"],
                   governor_dry_run_line=next((l for l in log.splitlines() if l.startswith("forge-gpu-run: DRY RUN")), None),
                   child_started=(attempts[-1] / "child_started.json").exists(),
                   forge_log_written=(attempts[-1] / "forge.log").exists())
    else:
        rec.update(dependency_refusal=next((l for l in p.stderr.splitlines() if '"error"' in l), None))
    # S1: the label's runner argv straight to the governor, dry run.
    i = next(j for j, l in enumerate(script) if l.startswith("  %s) #" % label))
    words = shlex.split(script[i + 1])
    argv = words[words.index("$GPU_RUN") + 1:]
    log = Path("/tmp/eng2_a1_dry_s1") / (name + "_" + label + ".forge.log")
    log.parent.mkdir(exist_ok=True)
    assert not log.exists()
    s1 = subprocess.run(["forge-gpu-run", *argv], cwd=TOOL, env=dict(env, FORGE_GPU_RUN_LOG=str(log)),
                        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    dry = [l for l in s1.stdout.splitlines() if l.startswith("forge-gpu-run: DRY RUN ")]
    rec.update(s1_rc=s1.returncode, s1_dry_run=json.loads(dry[0][len("forge-gpu-run: DRY RUN "):]) if dry else None,
               s1_run_log_written=log.exists(), s1_stderr=s1.stderr[-500:])
    if line["role"] == "mutant" or (line["role"] == "regression" and line["arm"] == "candidate"):
        rec["verdict"] = bool(rec["seal_and_arms_verified"] and not attempts and rec["dependency_refusal"]
                              and rec["s1_rc"] == 78 and not rec["s1_run_log_written"])
    else:
        rec["verdict"] = bool(rec["seal_and_arms_verified"] and rec.get("governor_exit") == 78
                              and rec.get("admission") == "not_admitted" and rec.get("governor_dry_run_line")
                              and not rec.get("child_started") and rec["s1_rc"] == 78)
    rows.append(rec)
    print("%-20s %-10s staged %s | seal+arms %s | governor %s | dep %s | S1 %s | ok %s" % (
        label, line["role"], p.returncode, rec["seal_and_arms_verified"], rec.get("governor_exit"),
        (rec.get("dependency_refusal") or "")[:60], rec["s1_rc"], rec["verdict"]), flush=True)
out = G4 / ("G4_DRY_" + name + ".json")
with out.open("x") as f:
    f.write(json.dumps(dict(root=str(runs.relative_to(ROOT)), rows=rows, all_ok=all(r["verdict"] for r in rows)),
                       indent=1) + "\n")
print("all_ok", all(r["verdict"] for r in rows))
