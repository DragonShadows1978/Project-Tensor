"""Finish bounded CPU gates after the receipted mechanical instrument corrections.
Prior art: A2 subprocess gate driver; pytest. No new selection rule.
"""
import os,sys,subprocess,json
from pathlib import Path
p=Path(__file__).resolve().parent
rows=[]
for name,script,args in [("mutants_gate","cpu_checks.py",["mutants"]),("native_guard_gate","native_validation.py",[]),("import_trace_gate","trace_imports.py",[])]:
    with (p/(name+".log")).open("x") as f:
        r=subprocess.run(["timeout","180s",sys.executable,"-B",str(p/script),*args],stdout=f,stderr=subprocess.STDOUT,env=os.environ)
    rows.append(dict(name=name,exit=r.returncode))
    print(name,r.returncode,flush=True)
    assert r.returncode==0
(p/"G2_finish.json").write_text(json.dumps(rows,indent=2)+"\n")
