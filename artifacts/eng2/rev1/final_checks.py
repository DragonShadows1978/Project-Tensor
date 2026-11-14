"""Finalize review evidence. Prior art: differential/mutation testing (McKeeman 1998; DeMillo et al. 1978), unverified -- lead to check. No new algorithm."""
from pathlib import Path
import ast,importlib.util,json,sys,subprocess
ROOT=Path.cwd();OUT=ROOT/"artifacts/eng2/rev1";TOOL=ROOT/"artifacts/eng2/tool"
registry=json.loads((TOOL.parent/"mutants/mutants.json").read_text());pred=json.loads((TOOL.parent/"N1_EMULATION.json").read_text())["predicted_failures"]
for m in registry:
 if m["id"] in ["MF1","MF2"]:
  d=json.loads((OUT/(m["id"]+"_cpu.json")).read_text());actual={k for k,v in d["outcomes"].items() if v!="passed"}
 else:actual=set(pred[m["id"]])
 assert set(m["killers"])==actual,m["id"]
# Run actual forward code with half inputs and both float32/half affine weights; trace all statistic ops and casts.
ns={"__name__":"review_adapter"};src=(OUT/"norm_review.py").read_text().split("# The layer and test functions above")[0]
exec(compile(src,str(OUT/"norm_review.py"),"exec"),ns)
ns["load"]("candidate");traces=[]
for cls in ns["TEST"].N1_CLASSES:
 for dt in ["float16","bfloat16"]:
  for pdt in ["float32",dt]:
   l=ns["Layer"](cls,pdt);x=ns["T"](ns["E3"].e3_inputs(cls,"regular"),dt);ns["TRACE"].clear();y=l(x);trace=list(ns["TRACE"])
   stats=[r for r in trace if r[0] in ["mean","var"]];casts=[r for r in trace if r[0]=="astype"]
   assert stats and all(r[1]=="float32" for r in stats)
   assert y.dtype==dt and sum(r[-1]==dt for r in casts)==1
   traces.append(dict(cls=cls,input_dtype=dt,parameter_dtype=pdt,statistics=stats,casts=casts))
# Read-only baseline identity for added census entries.
for file in ["functional.py","quantization/affine.py","apa_sp2.py"]:
 b=subprocess.check_output(["git","--no-optional-locks","show","bc02d3a1:tensor_cuda/tensor_cuda/"+file]);assert b==(ROOT/"tensor_cuda/tensor_cuda"/file).read_bytes()
# apa_sp2.GapMargins.lookup is a policy gate, not a dtype constructor; each requested value refuses explicitly.
sys.path.insert(0,str(ROOT/"tensor_cuda"));from tensor_cuda import apa_sp2
lookup=[]
cls=next(v for v in vars(apa_sp2).values() if isinstance(v,type) and "lookup" in vars(v))
for d in ["float64","double","int32","not_a_dtype",None]:
 try:cls.lookup(object(),4,64,d,1/8)
 except ValueError as e:lookup.append(dict(dtype=d,exception="ValueError",message=str(e)))
 else:raise AssertionError("lookup did not refuse")
(OUT/"final_checks.json").write_text(json.dumps(dict(registered_killers_match_predictions=True,half_traces=traces,additional_census_files_baseline_identical=True,lookup=lookup),indent=2)+"\n")
print("All 15 killer sets equal the registered predictions; 20 half-input/parameter-dtype source traces use float32 statistics and one cast back; extra census source identical to baseline; lookup refuses all 5 requested values.")
