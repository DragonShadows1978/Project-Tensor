"""Finish census, finite-case BN bound, and receipt checks.
Prior art: Higham (2002), gamma_n rounding bounds; IEEE 754; unverified -- lead to check. Taken: standard bounds; ours: specialize to the registered N=16, momentum=.1, initial running mean=0/variance=1 cases.
"""
from pathlib import Path
import ast,importlib.util,json,math,subprocess,sys
import numpy as np
ROOT=Path.cwd();OUT=ROOT/"artifacts/eng2/rev1";TOOL=ROOT/"artifacts/eng2/tool"
sys.path.insert(0,str(ROOT/"tensor_cuda"));import tensor_cuda as tc
sp=importlib.util.spec_from_file_location("completion_e3",ROOT/"tensor_cuda/tests/test_eng1.py");E=importlib.util.module_from_spec(sp);sp.loader.exec_module(E)
inventory=[]
for p in sorted((ROOT/"tensor_cuda/tensor_cuda").rglob("*.py")):
    tree=ast.parse(p.read_text())
    for n in ast.walk(tree):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
            args=[a.arg for a in [*n.args.posonlyargs,*n.args.args,*n.args.kwonlyargs] if "dtype" in a.arg]
            if args:inventory.append(dict(file=str(p.relative_to(ROOT)),line=n.lineno,function=n.name,dtype_arguments=args))
# N=16 mean uses 15 nontrivial serial additions and an exact /16 scaling on these normal operands.
bounds=[]
for dt in ["float32","float16","bfloat16"]:
 for case in E.E3_CASES:
    x=E.e3_round(E.e3_inputs("BatchNorm1D",case),dt);assert np.any(x!=0) and np.ptp(x)>0 and np.all(np.isfinite(x))
    N=x.shape[0];assert N==16;u=2.**-24;m=.1;mu=x.mean(0);var=x.var(0)
    gamma=(N-1)*u/(1-(N-1)*u);dm=gamma*np.abs(x).mean(0)
    alpha=(1+u)**3*(1+gamma)-1;dv=alpha*var+(1+alpha)*dm**2
    mb=float(np.float32(m));ab=float(np.float32(1-m))
    bm=abs(mb-m)*np.abs(mu)+u*abs(mb)*np.abs(mu)+(1+u)*abs(mb)*dm
    bv=abs(ab-(1-m))+abs(mb-m)*var+u*abs(mb)*var+u*(abs(ab)+(1+u)*abs(mb)*var)+(1+u)**2*abs(mb)*dv
    wm=m*mu;wv=(1-m)+m*var;registered_dm=(N+1)*u*np.abs(x).mean(0)
    tm=m*registered_dm+3*u*np.abs(wm)+2.**-149
    tv=m*((N+3)*u*var+registered_dm**2)+3*u*np.abs(wv)+2.**-149
    assert np.all(bm<=tm) and np.all(bv<=tv)
    bounds.append(dict(dtype=dt,case=case,mean_bound_over_registered=float(max(bm/tm)),var_bound_over_registered=float(max(bv/tv)),covers=True))
base=json.loads((OUT/"baseline_cpu.json").read_text());f1={k for k,v in base["outcomes"].items() if v!="passed"}
emu=json.loads((OUT/"source_emulation.json").read_text());n1={k for k,v in emu["baseline"].items() if not v["passed"]}
registered=set(json.loads((TOOL/"baseline_expected_failures.json").read_text()));assert f1|n1==registered
# Missing non-string rows for C++ out_dtype bindings: zero-byte tensors ensure parsing precedes any kernel.
z=lambda:tc.tensor(np.zeros((0,)),device="cpu")
extra_cpp=[]
for name,fn in [("int4_dequant",lambda:tc.int4_dequant(z(),z(),z(),128,None)),("intn_dequant",lambda:tc.intn_dequant(z(),z(),z(),4,0,128,None)),("kv_int4_unpack",lambda:tc.kv_int4_unpack(z(),z(),32,0,0,None))]:
 try:fn();raise AssertionError("non-string unexpectedly accepted")
 except TypeError as e:extra_cpp.append(dict(entry=name,input="None",exception="TypeError",message=str(e).split("Invoked with:")[0]))
# Read existing parent receipts; execution ordering is additionally checked in source run_gpu_line.py:68-107.
parents=[]
for p in sorted((TOOL/"gpu_runs/audit_1").glob("*/attempt-*/parent_result.json")):
 d=json.loads(p.read_text());assert d["start"]<=d["child_exited"]<=d["recorded"]
 parents.append(dict(label=d["label"],exit=d["exit"],child_exited=d["child_exited"],recorded=d["recorded"]))
result=dict(dtype_argument_inventory=inventory,bn_bounds=bounds,baseline_failure_union=dict(f1=len(f1),n1=len(n1),union=len(f1|n1),matches_registration=True),nonstring_out_dtype=extra_cpp,parent_after_child_receipts=parents)
(OUT/"completion_checks.json").write_text(json.dumps(result,indent=2)+"\n")
print("dtype arguments:",json.dumps(inventory,indent=1));print("BN bounds:",json.dumps(bounds,indent=1));print("Baseline union",len(f1),"+",len(n1),"=",len(registered),"exact IDs; parent receipts checked",len(parents));print("C++ out_dtype nonstrings: all 3 TypeError")
