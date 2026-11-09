"""G4: REV1 attacks adapted to real A2 parent/scorer, with CPU forge stand-in.
Prior art: REV1 hostile_scorer.py; Meszaros (2007), fault injection; mutation
validity DeMillo et al. (1978). A2 changes are listed in HOSTILE_ADAPTATIONS.md.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from evidence import TOOL, read, lines, DEVICE_PATTERNS, sha
assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
BASE=TOOL/"hostile_runs"
BASE.mkdir(exist_ok=False)
RESULT=[]


def env(runs, mode="normal", **extra):
    e=dict(os.environ,RUNS=str(runs),HOSTILE_MODE=mode,PYTHONDONTWRITEBYTECODE="1",**extra)
    e.pop("GN1_AUDIT",None)
    return e


def invoke(args, e):
    r=subprocess.run(["timeout","60s",sys.executable,"-B",*args],cwd=TOOL,env=e,capture_output=True,text=True)
    return dict(exit=r.returncode,stdout=r.stdout[-3000:],stderr=r.stderr[-3000:])


def start(name):
    runs=BASE/name
    r=invoke(["start_gpu_run.py"],env(runs))
    assert r["exit"]==0,r
    return runs


def parent(runs,label,mode="normal",bound=20,**extra):
    line=lines()[label]
    ex=",".join(map(str,line["expected_child_exits"]))
    r=invoke(["run_gpu_line.py",label,ex,str(bound),sys.executable,"hostile_forge.py","--max","30","--","timeout","10s","python3","run_gpu_tests.py",line["arm"],line["suite"]],env(runs,mode,**extra))
    paths=sorted((runs/label).glob("attempt-*/parent_result.json"))
    r["receipt"]=str(paths[-1].relative_to(TOOL)) if paths else None
    r["parent"]=read(paths[-1]) if paths else None
    return r


def score(runs):
    r=invoke(["score_gpu.py"],env(runs))
    p=runs/"G5_SUMMARY.json"
    r["summary"]=read(p) if p.exists() else None
    return r


def control(name,mode,needle,label="M07fast",**kwargs):
    runs=start(name)
    if label!="candidate_focus" and lines()[label]["suite"]=="mutation":
        assert parent(runs,"candidate_focus")["exit"]==0
    r=parent(runs,label,mode,**kwargs)
    s=score(runs)
    reasons=(r["parent"] or {}).get("reasons",[])
    passed=r["exit"]==1 and any(needle in x for x in reasons) and s["exit"]==1
    RESULT.append(dict(control=name,target=needle,passed=passed,parent=r,scorer=s))
    print(name, "PASS" if passed else "FAIL",reasons,flush=True)


# H1/H1b: no START, no children, forged parent-only receipts, within/outside guard.
runs=BASE/"H1";runs.mkdir()
for label in lines():
    d=runs/label/"attempt-0001";d.mkdir(parents=True)
    (d/"parent_result.json").write_text(json.dumps(dict(passed=True,timeout=False,audit=False,actual_killers=["forged"])))
s=score(runs)
RESULT.append(dict(control="H1",target="START.json",passed=s["exit"]==1 and "START.json" in str(s),scorer=s))
out=Path(os.environ["TMPDIR"])/"outside_guard";out.mkdir(exist_ok=True)
s=score(out)
RESULT.append(dict(control="H1b",target="run-root guard",passed=s["exit"]!=0 and "RUNS must be under a2" in str(s),scorer=s))
control("H2_REV1_exact","rev1_device","device-pattern failure")
control("H2c_REV1_exact","rev1_device_control","device-pattern failure")
control("H3_foreign_arm","foreign_arm","arm/suite binding")
control("H3b_suite_swap","suite_swap","expected exits")
control("H4_junit_foreign_ID","junit_foreign_id","unregistered JUnit ID",label="candidate_focus")
control("Junit_one_ID_disagreement","junit_disagree","JUnit/summary outcomes disagree",label="candidate_focus")
control("Junit_duplicate_pass","junit_duplicate_pass","duplicate JUnit ID",label="candidate_focus")
for i,pat in enumerate(DEVICE_PATTERNS):
    control("H2_pattern_%02d"%i,"device","device-pattern failure",DEVICE_PATTERN=pat)
control("Reference_only_kill","reference_only","no real non-reference")
control("Generic_RuntimeError_not_kill","generic_runtime","no real non-reference")
control("Unregistered_TypeError_not_kill","unregistered_refusal","no real non-reference")
control("Wrong_nn_hash","bad_nn_hash","nn.py binding")
control("Wrong_engine_hash","bad_engine_hash","engine binding")
control("Setup_failure","setup_error","setup/teardown failures")
control("Collection_error","collection_error","collection errors")
control("H6_hang_after_summary","hang_after_result","expected exits",label="candidate_focus",bound=1)
control("D10_timeout","timeout","forge outcome: timeout",label="candidate_focus")
control("D10_window_closed","window-closed","forge outcome: window-closed",label="candidate_focus")
# H5 now rejects refusal as not admitted, and permits a valid second attempt.
for code in (75,76):
    runs=start("H5_retry_"+str(code))
    first=parent(runs,"baseline_focus","refuse"+str(code))
    second=parent(runs,"baseline_focus")
    s=score(runs);row=s["summary"]["lines"]["baseline_focus"]
    ok=first["exit"]==1 and first["parent"]["admission"]=="not_admitted" and second["exit"]==0 and row["passed"] and len(row["attempts"])==2
    RESULT.append(dict(control="H5_retry_"+str(code),target="not admitted then admitted",passed=ok,first=first,second=second,scorer=s))
# Flip exactly one byte in an isolated mutant copy after START. No source pin is
# touched. Preserve original bytes under /tmp, then restore in finally; receipts
# record both digests. No file is removed or made into a symlink.
runs=start("One_byte_engine_copy")
assert parent(runs,"candidate_focus")["exit"]==0
pin=read(TOOL/"engine_pin.json");target=TOOL/"arms/M07fast/tensor_cuda"/Path(pin["module"]).name
original=target.read_bytes();saved=Path(os.environ["TMPDIR"])/"M07fast.engine.saved";saved.write_bytes(original)
try:
    changed=bytes([original[0]^1])+original[1:];target.write_bytes(changed)
    assert sum(a!=b for a,b in zip(original,changed))==1
    r=parent(runs,"M07fast");s=score(runs)
    ok=r["exit"]==1 and "engine copy differs from pin" in str(r["parent"]["reasons"]) and s["exit"]==1
    RESULT.append(dict(control="One_byte_engine_copy",target="engine copy differs from pin",passed=ok,parent=r,scorer=s,original_sha256=pin["sha256"],altered_sha256=sha(target)))
finally:
    target.write_bytes(original)
assert sha(target)==pin["sha256"]
runs=start("Registered_refusal_positive")
assert parent(runs,"candidate_focus")["exit"]==0
r=parent(runs,"M07fast","registered_refusal");s=score(runs)
RESULT.append(dict(control="Registered_refusal_positive",target="GroupNorm-origin TypeError admitted as kill",passed=r["exit"]==0 and s["summary"]["lines"]["M07fast"]["passed"],parent=r,scorer=s))
# A full synthetic completed run: positive sensitivity check through all labels.
runs=start("Full_valid_synthetic")
for label in lines():
    r=parent(runs,label)
    assert r["exit"]==0,(label,r)
s=score(runs)
RESULT.append(dict(control="Full_valid_synthetic",target="positive recomputation",passed=s["exit"]==0,scorer=s))
# Forged stored parent receipt is compared, never trusted.
p=runs/"M07fast/attempt-0001/parent_result.json";old=p.read_text();d=json.loads(old);d["actual_killers"]=["forged"]
p.write_text(json.dumps(d,indent=2)+"\n")
s=score(runs)
RESULT.append(dict(control="Stored_parent_tamper",target="stored parent receipt disagrees",passed=s["exit"]==1 and "stored parent receipt disagrees" in str(s),scorer=s))
p.write_text(old)
# Exit list tamper and START seal tamper are each checked independently.
old=p.read_text();d=json.loads(old);d["expected_child_exits"]=[0];p.write_text(json.dumps(d))
s=score(runs)
RESULT.append(dict(control="Expected_exit_tamper",target="gpu_lines expected exits",passed=s["exit"]==1 and "gpu_lines.json" in str(s),scorer=s));p.write_text(old)
p=runs/"START.json";old=p.read_text();d=json.loads(old);d["seal_sha256"]="0"*64;p.write_text(json.dumps(d))
s=score(runs)
RESULT.append(dict(control="START_seal_tamper",target="START seal",passed=s["exit"]==1 and "START seal mismatch" in str(s),scorer=s));p.write_text(old)
# D6: unchanged legacy failure passes; a new failure, missing ID, skipped old
# pass or changed exception class is a regression. These are faithful outcome
# fixtures, not a claim that the real baseline has these unrelated failures.
for mode,needle in (("normal",None),("regression_new_failure","baseline-passing"),("regression_changed_class","different exception class"),("regression_missing_pass","baseline-passing"),("regression_skip_pass","baseline-passing")):
    runs=start("D6_"+mode)
    assert parent(runs,"baseline_regression")["exit"]==0
    r=parent(runs,"candidate_regression",mode);s=score(runs)
    ok=r["exit"]==(0 if needle is None else 1) and (needle is None or needle in str(r["parent"]["reasons"]))
    selected=s["summary"]["lines"]["candidate_regression"]
    ok=ok and selected["passed"]==(needle is None)
    RESULT.append(dict(control="D6_"+mode,target=needle or "legacy failure same class accepted",passed=ok,parent=r,scorer=s))
for mode, needle in (("regression_legacy_error",None),("regression_new_setup_error","baseline-passing"),("regression_error_changed_class","different exception class")):
    runs=start("D6_"+mode)
    baseline_mode="normal" if mode=="regression_new_setup_error" else "regression_legacy_error"
    assert parent(runs,"baseline_regression",baseline_mode)["exit"]==0
    r=parent(runs,"candidate_regression",mode);s=score(runs)
    ok=r["exit"]==(0 if needle is None else 1) and (needle is None or needle in str(r["parent"]["reasons"]))
    ok=ok and s["summary"]["lines"]["candidate_regression"]["passed"]==(needle is None)
    RESULT.append(dict(control="D6_"+mode,target=needle or "legacy setup error same class accepted",passed=ok,parent=r,scorer=s))
runs=start("D6_dual_failure")
assert parent(runs,"baseline_regression","regression_dual_failure")["exit"]==0
r=parent(runs,"candidate_regression","regression_dual_failure");s=score(runs)
RESULT.append(dict(control="D6_dual_failure",target="legacy call failure plus teardown error accepted",passed=r["exit"]==0 and s["summary"]["lines"]["candidate_regression"]["passed"],parent=r,scorer=s))
result=dict(passed=all(r["passed"] for r in RESULT),controls=RESULT,CPU_only=True,CUDA_VISIBLE_DEVICES="",patterns=list(DEVICE_PATTERNS))
(TOOL/"G4_hostile.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(dict(passed=result["passed"],controls=len(RESULT),failed=[r["control"] for r in RESULT if not r["passed"]])))
sys.exit(0 if result["passed"] else 1)
