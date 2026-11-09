"""Bounded foreground CPU instrument driver, derived from A2 gates.
Prior art: pytest collection/report APIs; Python subprocess.run; no novel algorithm.
"""
import json, os, subprocess, sys
from pathlib import Path
TOOL=Path(__file__).resolve().parent
ROOT=TOOL.parents[2]
env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
rows=[]
def run(name,args,expected=(0,)):
    with (TOOL/(name+".log")).open("x") as f:
        p=subprocess.run(["timeout","180s",sys.executable,"-B",*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    rows.append(dict(name=name,exit=p.returncode,expected=list(expected),passed=p.returncode in expected))
    print(name,p.returncode,flush=True)
    assert p.returncode in expected,name
for arm,suite in [("baseline","focus"),("candidate","focus"),("candidate","mutation"),("baseline","regression"),("candidate","regression")]:
    run("collect_"+arm+"_"+suite,[str(TOOL/"inspect_cpu.py"),arm,suite,"collect"])
run("native_references",[str(TOOL/"inspect_cpu.py"),"candidate","mutation","references"])
run("baseline_shim_gate",[str(TOOL/"cpu_checks.py"),"baseline"],(1,))
run("mutants_gate",[str(TOOL/"cpu_checks.py"),"mutants"])
run("native_guard_gate",[str(TOOL/"native_validation.py")])
run("import_trace_gate",[str(TOOL/"trace_imports.py")])
(TOOL/"G2_driver.json").write_text(json.dumps(rows,indent=2)+"\n")
