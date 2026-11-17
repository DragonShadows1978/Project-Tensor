"""L60 boundary evidence on E and F, always isolated children under RLIMIT_AS.
Prior art: REV1 receipt 028 (2026); Unix subprocess isolation and resource
limits (POSIX, 1988 onward). Taken: 16 MiB virtual headroom; ours: input matrix.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[3]
ART=ROOT/"artifacts/eng3f1"


def child(case):
    import numpy as np
    import tensor_cuda as tc
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    mapped=int(Path("/proc/self/statm").read_text().split()[0])*os.sysconf("SC_PAGE_SIZE")
    resource.setrlimit(resource.RLIMIT_AS,(mapped+16*1024**2,)*2)
    c=tc._C;hook=case["hook"];shape=case.get("shape",[2]);dtype=case.get("dtype","float32");device=case.get("device","cpu")
    try:
        if hook in ("_core_to","_core_clone"):
            source=tc.tensor(np.ones((1,)*9,np.float32) if case.get("source_rank9") else [1.,2.],device="cpu")
            t=c._core_to(source,device) if hook=="_core_to" else c._core_clone(source)
        elif hook=="_core_full":
            t=c._core_full(shape,1.,dtype,device)
        else:
            t=c._core_zeros(shape,dtype,device)
        print(json.dumps({"returned":True,"shape":list(t.shape),"device":t.device,"pointer_zero":t.data_ptr()==0}),flush=True)
    except Exception as exc:
        print(json.dumps({"returned":False,"exception":type(exc).__name__,"message":str(exc)}),flush=True)


def main():
    cases=[]
    for dt,w in {"float32":4,"float16":2,"bfloat16":2,"int64":8,"bool":1,"uint8":1}.items():
        q=(2**63-1)//w;overflow=[q+1] if w>1 else [2**62,4]
        for hook in ("_core_full","_core_zeros"):
            for shape in dict.fromkeys(tuple(s) for s in (overflow,[0]+overflow,[2**62] if w>=2 else overflow)):
                cases.append({"hook":hook,"shape":list(shape),"dtype":dt,"ruling":"R1"})
    for hook in ("_core_full","_core_zeros"):
        cases += [{"hook":hook,"shape":[1]*9,"ruling":"R1 rank"},
                  {"hook":hook,"shape":[2**25],"ruling":"R1 malloc-null"}]
    for hook in ("_core_clone","_core_to"):
        cases.append({"hook":hook,"source_rank9":True,"ruling":"R1 rank"})
    import runpy
    # Read the registered invalid list without importing the engine here.
    import ast
    tree=ast.parse((ART/"tools/eng3f1_checks.py").read_text())
    invalid=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="INVALID" for t in n.targets))
    for hook in ("_core_full","_core_zeros","_core_to"):
        cases.extend({"hook":hook,"device":d,"ruling":"R2"} for d in invalid)
    rows=[]
    for i,case in enumerate(cases):
        row={"case":case}
        for arm in ("E","F"):
            rec=json.loads((ART/f"build/BUILD_{arm}.json").read_text());module=ROOT/rec["module"]
            assert hashlib.sha256(module.read_bytes()).hexdigest()==rec["module_sha256"]
            # Known crashing receipts are reused; no redundant baseline crashes.
            known=(arm=="E" and case["hook"]=="_core_full" and case.get("dtype","float32")=="float32" and case.get("shape")==[2**62])
            oom=(arm=="E" and case["hook"]=="_core_full" and case.get("shape")==[2**25])
            if known or oom:
                row[arm]={"exit":-11,"receipt":"../tensor_fork_apa_wt_eng3rev1/artifacts/eng3_rev1/receipts/"+("028_host_oom_crash.log" if oom else "011_hook_crash.log"),"measured_this_round":False}
                continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONPATH=str(module.parent.parent),PYTHONDONTWRITEBYTECODE="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
            p=subprocess.run(["timeout","15s",sys.executable,"-B","-P",__file__,"child",json.dumps(case)],cwd=ROOT,env=env,capture_output=True,text=True)
            row[arm]={"exit":p.returncode,"stdout":p.stdout,"stderr":p.stderr,"module_sha256":rec["module_sha256"],"measured_this_round":True}
            if p.stdout:
                row[arm]["result"]=json.loads(p.stdout.strip().splitlines()[-1])
        rows.append(row)
        (ART/"BOUNDARY.json").write_text(json.dumps({"rows":rows,"complete":False},indent=2))
    (ART/"BOUNDARY.json").write_text(json.dumps({"rows":rows,"complete":True,"E_impossible_index_GPU_receipt":"../tensor_fork_apa_wt_eng3rev1/artifacts/eng3_rev1/lead_eng3rev1_gpu_repro.json"},indent=2))
    print("boundary rows",len(rows),flush=True)

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="child":
        child(json.loads(sys.argv[2]))
    else:
        main()
