"""ENG2 adversarial probes. Prior art: differential/mutation testing (McKeeman 1998; DeMillo et al. 1978), unverified -- lead to check; ours: these boundary receipts."""
from pathlib import Path
from collections import UserString
import json, sys
import numpy as np
ROOT=Path.cwd(); OUT=ROOT/"artifacts/eng2/rev1"; TOOL=ROOT/"artifacts/eng2/tool"
sys.path.insert(0,str(ROOT/"tensor_cuda")); import tensor_cuda as tc
sys.path.insert(0,str(TOOL)); import evidence
rows=[]
for entry in ["tensor","zeros","ones","randn","rand"]:
    np.random.seed(2214)
    before=np.random.get_state()
    try:
        result = tc.tensor(np.zeros((0,2)),dtype=UserString("float32"),device="cpu") if entry=="tensor" else getattr(tc,entry)(0,2,dtype=UserString("float32"),device="cpu")
        row=dict(entry=entry,non_string=not isinstance(UserString("float32"),str),outcome="accepted",dtype=result.dtype)
    except Exception as e: row=dict(entry=entry,outcome=type(e).__name__,message=str(e))
    rows.append(row)
calls=[]
maker=tc._factory(lambda shape: calls.append(shape) or np.empty((0,),dtype=np.float32))
try: maker(2,dtype=UserString("float32"),device="cpu")
except Exception as e: rows.append(dict(entry="_factory spy",outcome=type(e).__name__,message=str(e),allocation_function_called=calls))
else: rows.append(dict(entry="_factory spy",outcome="accepted",allocation_function_called=calls))
# Real CPU tensor allocation is allowed; no device call above (device=cpu).
(OUT/"nonstring.json").write_text(json.dumps(rows,indent=2)+"\n"); print("NONSTRING",json.dumps(rows))
pins=json.loads((TOOL.parent/"L38_PINS.json").read_text())["expected_changes"]
nid=next(iter(pins)); expected=pins[nid]["detail_contains"]
def receipt(outcome,detail="",exception_class=None):
    return dict(collected_ids=[nid],reports=[dict(id=nid,when=p,outcome=outcome if p=="call" else "passed",detail=detail if p=="call" else "",exception_class=exception_class if p=="call" else None) for p in ["setup","call","teardown"]])
old=receipt("passed"); cases={}
for name,detail,exc in [("registered",expected,"TypeError"),("other_reason","AssertionError: unrelated mismatch","AssertionError"),("wrapped_other_reason","AssertionError: expected error was "+repr(expected)+" but received corrupt output","AssertionError"),("wrong_type_exact_text",expected,"AssertionError")]:
    new=receipt("failed",detail,exc); reasons=evidence.regression_reasons(old,new)
    cases[name]=dict(old=old,new=new,reasons=reasons,accepted=not reasons)
(OUT/"pin_cases.json").write_text(json.dumps(cases,indent=2)+"\n")
print("PINS",len(pins), {k:dict(accepted=v["accepted"],reasons=v["reasons"]) for k,v in cases.items()})
mutants=json.loads((TOOL.parent/"mutants/mutants.json").read_text()); matched=[]
for m in mutants:
    killers=m["killers"]
    matched.append(dict(id=m["id"],registered=len(killers),self_matching=sum(evidence.matches(k,killers) for k in killers),example=killers[0]))
(OUT/"killer_matching.json").write_text(json.dumps(matched,indent=2)+"\n"); print("KILLER MATCHING",json.dumps(matched))
