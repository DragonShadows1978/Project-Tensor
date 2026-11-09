"""G6 static staging, seal and completed real CPU audit verification.
Prior art: A1 gate, GNU timeout, pytest/JUnit, NIST SHA-256 (2001).
"""
import ast
import json
import subprocess
from evidence import TOOL, read, verify_start, verify_arm, lines, recompute_line, census

for p in TOOL.glob("*.py"):ast.parse(p.read_text())
assert subprocess.run(["timeout","5s","bash","-n",str(TOOL/"STAGED_GPU_COMMANDS.sh")]).returncode==0
script=(TOOL/"STAGED_GPU_COMMANDS.sh").read_text()
registered=lines()
assert len(registered)==len(read(TOOL/"mutants.json"))+4
assert registered["candidate_regression"]["expected_child_exits"]==[0,1]
for label,row in registered.items():
    ex=",".join(map(str,row["expected_child_exits"]))
    assert row["expected_parent_exit"]==0
    assert 0<row["inner"]<row["maximum"]<row["outer"]<1200
    assert "run_gpu_line.py {} {} {}".format(label, ex, row["outer"]) in script
    assert "run_gpu_tests.py {} {}".format(row["arm"], row["suite"]) in script
runs=TOOL/"audit_cpu"
seal,start=verify_start(runs)
for arm in seal["arms"]:verify_arm(arm,seal)
assert start["audit"]
result=recompute_line(runs,"candidate_focus",seal,start)
assert result["passed"] and len(result["outcomes"])==29
p=read(runs/"candidate_focus/attempt-0001/parent_result.json")
assert p["exit"]==0 and not p["timeout"] and p["forge_outcome"]=="ran"
c=read(runs/"candidate_focus/attempt-0001/child_result.json")
assert c["CUDA_VISIBLE_DEVICES"]=="" and c["collected_ids"]==census("candidate","focus",True)
result=dict(passed=True,lines=list(registered.values()),audit_reference_calls=29,audit_child_exit=0,
            audit_parent_passed=True,audit_result=result,gpu_execution=False,expected_parent_exit_each=0,
            receipt="audit_cpu/candidate_focus/attempt-0001/parent_result.json")
(TOOL/"G6_staging.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(dict(G6="PASS",lines=len(registered),audit_reference_calls=29)))
