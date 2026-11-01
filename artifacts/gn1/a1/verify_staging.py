"""G4 static staging and actual completed CPU audit verification.
Prior art: round-one GN1 parent receipts; GNU timeout and SHA256 (NIST 2001).
No GPU subprocess or claimed mutant kill is executed by this instrument.
"""
from pathlib import Path
import ast
import datetime
import hashlib
import json
import subprocess
TOOL=Path(__file__).resolve().parent
ROOT=TOOL.parents[2]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
seal=json.loads((TOOL/"staging_seal.json").read_text())
for f,digest in seal["tool_files"].items():assert sha(TOOL/f)==digest,f
assert sha(ROOT/"orders/GN1_A1_HALF_PRECISION.md")==seal["order_sha256"]
for f,digest in seal["test_files"].items():assert sha(ROOT/f)==digest,f
for arm,files in seal["arms"].items():
    for f,digest in files.items():assert sha(TOOL/"arms"/arm/"tensor_cuda"/f)==digest,(arm,f)
for file in TOOL.glob("*.py"):ast.parse(file.read_text())
syntax=subprocess.run(["timeout","5s","bash","-n",str(TOOL/"STAGED_GPU_COMMANDS.sh")],check=False)
assert syntax.returncode==0
lines=json.loads((TOOL/"gpu_lines.json").read_text())
mutants=json.loads((TOOL/"mutants.json").read_text())
script=(TOOL/"STAGED_GPU_COMMANDS.sh").read_text()
assert len(lines)==22 and len(mutants)==18
for line in lines:
    label=line["label"]
    assert line["expected_parent_exit"]==0
    expected=0 if label in ("candidate_focus","candidate_regression") else 1
    assert line["expected_child_exit"]==expected
    assert 0<line["inner"]<line["maximum"]<line["outer"]<1200
    assert f'  {label}) # expected child {expected}, parent 0;' in script
    assert f'python3 run_gpu_line.py {label} {expected} {line["outer"]}' in script
assert '${RUNS:?' in script
assert 'cd -- "$TOOL"' in script
parent=json.loads((TOOL/"audit_cpu/candidate_focus/parent_result.json").read_text())
child=json.loads((TOOL/"audit_cpu/candidate_focus/child_result.json").read_text())
start=json.loads((TOOL/"audit_cpu/START.json").read_text())
assert start["audit"] and start["seal_sha256"]==sha(TOOL/"staging_seal.json")
assert parent["audit"] and parent["passed"] and parent["exit"]==0 and not parent["timeout"]
assert parent["cwd"]==str(TOOL)
assert datetime.datetime.fromisoformat(parent["recorded"])>=datetime.datetime.fromisoformat(parent["child_exited"])
assert child["pytest_exit"]==0 and child["CUDA_VISIBLE_DEVICES"]=="" and not child["collection_errors"]
ids=(TOOL/"ids_audit.txt").read_text().splitlines()
assert len(ids)==29 and child["collected_ids"]==ids
calls=[r for r in child["reports"] if r["when"]=="call"]
assert len(calls)==29 and all(r["outcome"]=="passed" for r in calls)
result=dict(passed=True,staged_lines=22,mutants=18,syntax_exit=0,expected_parent_exit_each=0,audit_child_exit=0,audit_parent_exit=0,audit_timeout=False,audit_reference_calls=29,audit_cwd=str(TOOL),parent_recorded_after_child_exit=True,gpu_execution=False,mutant_kills_claimed=False,receipts=["STAGED_GPU_COMMANDS.sh","gpu_lines.json","MUTANTS.md","mutants.json","mutant_validation.json","staging_seal.json","audit_cpu/START.json","audit_cpu/candidate_focus/child_result.json","audit_cpu/candidate_focus/parent_result.json","audit_cpu/candidate_focus/stdout.log"])
(TOOL/"G4_staging.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
