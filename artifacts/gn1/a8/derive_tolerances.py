"""D8 finite-census first-order roundoff derivation, no GPU observations.
Prior art: Higham (1996/2002), running error analysis; REV1 gn_emulate.py.
CUDA Math API powf 4 ulp is unverified — lead to check; represented operands
and graph are reused unchanged, with new mixed-parameter cases added.
"""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
TOOL = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL.parent / "rev1"))
import run_numerics as RN
E, T = RN.E, RN.T

cases = RN.cases()
for wd, bd in (("float32", "float16"), ("float16", "float32")):
    x,w,b,p = T._inputs((2,6,4))
    for bw in (False,True):
        cases.append(dict(id="test_distinct_parameter_dtypes["+("backward" if bw else "forward")+"-"+("w32-b16" if wd=="float32" else "w16-b32")+"]",
                          x=T.represented(x,"float32"),w=T.represented(w,wd),b=T.represented(b,bd),p=T.represented(p,"float32"),
                          G=3,eps=1e-5,xd="float32",wd=wd,bd=bd,backward=bw))
worst = {dt:{kind:dict(ratio=0.0) for kind in ("forward","backward")} for dt in T.DTYPES}
compute = {kind:dict(ratio=0.0) for kind in ("forward","backward")}
all_rows=[]
for cs in cases:
    B = E.running_bound(cs["x"],cs["G"],cs["w"],cs["b"],cs["eps"],cs["p"],upow=8*2**-24)
    y,grads = T.groupnorm_reference(cs["x"],cs["G"],cs["w"],cs["b"],cs["eps"],cs["p"])
    operands=[("y", "forward",cs["xd"],y,B["ey"])]
    if cs["backward"]:
        operands += [(n,"backward",dt,ref,B["e"+n]) for n,dt,ref in zip(("dx","dw","db"),(cs["xd"],cs["wd"],cs["bd"]),grads)]
    for name,kind,dt,ref,eb in operands:
        A,R=T.TOLERANCES["float32"][kind]
        r32=float(np.max(eb/(A+R*np.abs(ref))))
        row=dict(ratio=r32,id=cs["id"],operand=name,dtype=dt)
        if r32>compute[kind]["ratio"]:compute[kind]=row
        u,s={"float32":(0,0),"float16":(2**-11,2**-25),"bfloat16":(2**-8,2**-134)}[dt]
        # One final destination rounding, including underflow half-step.
        final=(1+u)*eb+u*np.abs(ref)+s
        a,r=T.TOLERANCES[dt][kind]
        ratios=final/(a+r*np.abs(ref))
        i=np.unravel_index(int(np.argmax(ratios)),ratios.shape)
        row=dict(ratio=float(ratios[i]),id=cs["id"],operand=name,index=[int(j) for j in i],
                 error_bound=float(final[i]),reference=float(ref[i]),budget=float((a+r*np.abs(ref))[i]))
        if row["ratio"]>worst[dt][kind]["ratio"]:worst[dt][kind]=row
        all_rows.append(dict(dtype=dt,kind=kind,**row))
assert len(cases)==324
for dt in T.DTYPES:
 for kind in ("forward","backward"):
  assert worst[dt][kind]["ratio"] < 1, (dt,kind,worst[dt][kind])
  worst[dt][kind]["margin_fraction"]=1-worst[dt][kind]["ratio"]
# Exact registration numbers unchanged, including destination-rounding composition.
registered=json.loads((TOOL.parent/"a1/tolerances_registered.json").read_text())
result=dict(passed=True,case_count=len(cases),powf_max_ulp_assumption=4,powf_relative_bound=8*2**-24,
            literature_status="unverified — lead to check: CUDA Math API single precision powf maximum ulp error",
            first_order=True,compute_fp32=compute,destination_budgets=worst,rows=all_rows,tolerances=T.TOLERANCES,
            registration_A1=registered)
(TOOL/"tolerance_bounds.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k not in ("rows","registration_A1")},indent=2))
