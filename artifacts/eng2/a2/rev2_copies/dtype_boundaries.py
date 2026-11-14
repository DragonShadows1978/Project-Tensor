"""Prior art: REV1 boundary sentinels; differential testing (McKeeman 1998,
unverified — lead to check). New coverage: pre-guard shape conversion, six-name
mask pairs and complete dtype-bearing signature inventory; no novel algorithm.
Both source versions execute in one process with one NumPy thread, CPU only.
"""
from pathlib import Path
from types import SimpleNamespace
from collections import UserString
import ast, hashlib, json, math, subprocess, sys
import numpy as np
ROOT=Path.cwd();OUT=ROOT/"artifacts/eng2/a2/rev2_out"  # A2: output rebound
# A2: no git (as A1). "bc02d3a1" reads artifacts/eng2/base_pkg (bc02d3a1 by ENG2's worktree_before.json, re-checked by
# a2/restage.py arms); "A2" reads the A2 worktree. Only tensor_cuda/tensor_cuda/* paths are requested.
def source(rev,path):
 assert path.startswith("tensor_cuda/tensor_cuda/")
 return ((ROOT/"artifacts/eng2/base_pkg"/path[len("tensor_cuda/"):]) if rev=="bc02d3a1" else (ROOT/path)).read_text()
def sha(b):return hashlib.sha256(b).hexdigest()
def save(name,data):(OUT/name).write_text(json.dumps(data,indent=2)+"\n")
class Captured:
 def __init__(self,a):self.a=np.array(a,copy=True);self.dtype=str(self.a.dtype);self.casts=[]
 def astype(self,d):self.casts.append(str(d));self.dtype=str(d);return self
# Device boundary capture records input bytes and requested casts, not simulated GPU arithmetic.
def factories(rev):
 nodes=[]
 for n in ast.parse(source(rev,"tensor_cuda/tensor_cuda/__init__.py")).body:
  if isinstance(n,ast.FunctionDef) and n.name in ("_np_dtype","tensor","_factory"):nodes.append(n)
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ("_NP_DTYPE","_DEVICE_CAST_DTYPE","zeros","ones","randn","rand") for t in n.targets):nodes.append(n)
 ns=dict(np=np,_C=SimpleNamespace(tensor=lambda a,*args:Captured(a)))
 exec(compile(ast.Module(body=nodes,type_ignores=[]),rev,"exec"),ns)
 return ns
pairs={arm:factories(rev) for arm,rev in [("baseline","bc02d3a1"),("candidate","A2")]}
mask_ns={}
for arm,rev in [("baseline","bc02d3a1"),("candidate","A2")]:
 nodes=[n for n in ast.parse(source(rev,"tensor_cuda/tensor_cuda/functional.py")).body if isinstance(n,ast.FunctionDef) and n.name in ("_causal_mask","_check_causal_lengths")]
 ns=dict(np=np,tc=SimpleNamespace(**pairs[arm]),_causal_cache={})
 exec(compile(ast.Module(body=nodes,type_ignores=[]),rev,"exec"),ns);mask_ns[arm]=ns
names=["float32","float16","int64","bool","uint8","bfloat16"]
mask_pairs=[]
for dtype in names:
 got=[]
 for arm,ns in mask_ns.items():
  t=ns["_causal_mask"](2,3,"cpu",dtype);assert np.any(t.a!=0) and np.unique(t.a).size>1
  assert ns["_causal_mask"](2,3,"cpu",dtype) is t
  got.append(t)
 a,b=got;assert a.dtype==b.dtype and a.casts==b.casts and a.a.tobytes()==b.a.tobytes()
 mask_pairs.append(dict(requested=dtype,result_dtype=b.dtype,casts=b.casts,nonzero_nonconstant=True,host_input_sha256=sha(b.a.tobytes()),same_as_baseline=True,cache_hit=True))
# Accepting actual str subclasses is the explicit isinstance contract.
class PlainStr(str):pass
subclasses=[]
for cls in (str,np.str_,PlainStr):
 for dtype in names:
  value=cls(dtype)
  for entry in ("tensor","zeros","ones","randn","rand"):
   got=[]
   for arm,ns in pairs.items():
    np.random.seed(2214)
    t=ns[entry](np.array([[0,1,2],[3,7,19]]),dtype=value) if entry=="tensor" else ns[entry](2,3,dtype=value)
    got.append(t)
   a,b=got;nontrivial=all(np.any(t.a!=0) and np.unique(t.a).size>1 for t in got)
   assert a.dtype==b.dtype and a.casts==b.casts
   if nontrivial:assert a.a.tobytes()==b.a.tobytes()
   subclasses.append(dict(type=cls.__name__,requested=dtype,entry=entry,same_dtype_cast=True,nonzero_nonconstant=nontrivial,bytes_compared=nontrivial))
# Real package refusal, no C++ boundary may be crossed.
sys.path.insert(0,str(ROOT/"tensor_cuda"));import tensor_cuda as tc
from tensor_cuda import functional as F
calls=[]
class Unconvertible:
 def __array__(self,*a,**kw):calls.append("data conversion");raise AssertionError("data conversion")
class Spy:
 def __hash__(self):calls.append("dtype hash");return hash("float32")
 def __eq__(self,o):calls.append("dtype equality");return o=="float32"
 def __repr__(self):return "Spy()"
class NoCache(dict):
 def get(self,*args):calls.append("mask cache lookup");raise AssertionError("cache lookup")
def forbidden(*a,**kw):calls.append("C++ allocation");raise AssertionError("C++ allocation")
old_C=tc._C.tensor;old_cache=F._causal_cache;tc._C.tensor=forbidden;F._causal_cache=NoCache()
invalid=[UserString("float32"),UserString("bfloat16"),[],None,np.dtype("float32"),Spy()]
checks=[]
try:
 for dtype in invalid:
  for entry in ("tensor","zeros","ones","randn","rand","_causal_mask"):
   calls.clear();np.random.seed(2214);before=np.random.get_state()
   try:
    if entry=="tensor":tc.tensor(Unconvertible(),dtype=dtype)
    elif entry=="_causal_mask":F._causal_mask(2,3,"cpu",dtype)
    else:getattr(tc,entry)(2,3,dtype=dtype)
   except Exception as exc:kind=type(exc).__name__;message=str(exc)
   else:kind="returned";message=""
   after=np.random.get_state()
   stable=before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
   expected=f"unsupported dtype {dtype!r}; supported dtypes: float32, float16, int64, bool, uint8, bfloat16"
   assert kind=="TypeError" and message==expected and not calls and stable
   checks.append(dict(entry=entry,requested=repr(dtype),exception=kind,exact_message=True,hooks=list(calls),rng_unchanged=stable))
 # The shape is a supported list subtype, converted before the guard in make.
 class Shape(list):
  def __iter__(self):calls.append("shape.__iter__ before dtype guard");np.random.rand();return super().__iter__()
 shape_results=[]
 # A2-G1: REV2's case (dtype=None) plus F1's four refused strings, same sentinel; REV2's assertion of the defect
 # (sentinel ran, RNG changed) is inverted to the fixed condition (A2 ledger): TypeError with F1's exact message, the
 # sentinel never ran, RNG state unchanged, no C++ allocation (tc._C.tensor is still the forbidden stub here).
 for shape_dtype in (None,"float64","double","int32","not_a_dtype"):
  for entry in ("zeros","ones","randn","rand"):
   calls.clear();np.random.seed(2214);before=np.random.get_state()
   try:getattr(tc,entry)(Shape([2,3]),dtype=shape_dtype,device="cpu")
   except Exception as exc:kind=type(exc).__name__;message=str(exc)
   else:kind="returned";message=""
   after=np.random.get_state();stable=before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
   expected=f"unsupported dtype {shape_dtype!r}; supported dtypes: float32, float16, int64, bool, uint8, bfloat16"
   assert kind=="TypeError" and message==expected and calls==[] and stable,(entry,shape_dtype,kind,message,calls,stable)
   shape_results.append(dict(entry=entry,requested=repr(shape_dtype),exception=kind,exact_message=True,hooks=list(calls),rng_unchanged=stable,device_calls=0))
finally:tc._C.tensor=old_C;F._causal_cache=old_cache
# Public metadata lookup also consumes dtype. Execute the exact baseline/candidate
# method with a sentinel table; invalid names fail before any table read. This is
# a control-flow census probe, not a calibration-table or attention validation.
lookup=[]
class Table:
 def __getitem__(self,key):return "TABLE_READ_SENTINEL"
for arm,rev in [("baseline","bc02d3a1"),("candidate","A2")]:
 tree=ast.parse(source(rev,"tensor_cuda/tensor_cuda/apa_sp2.py"))
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="RegisteredMargins")
 func=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="lookup")
 ns=dict(np=np,math=math);exec(compile(ast.Module(body=[func],type_ignores=[]),rev,"exec"),ns)
 for dtype in ["float64","double","int32","not_a_dtype",*names]:
  try:value=ns["lookup"](SimpleNamespace(entries=Table()),2,64,dtype,1/math.sqrt(64));row=dict(outcome="returned",value=value)
  except Exception as exc:row=dict(outcome=type(exc).__name__,message=str(exc))
  lookup.append(dict(arm=arm,requested=dtype,**row))
existing=json.loads((OUT/"census_candidate.json").read_text())
census_lookup={arm:[r for r in json.loads((OUT/("census_"+arm+".json")).read_text())["rows"] if r["entry"]=="apa_sp2.RegisteredMargins.lookup"] for arm in ("baseline","candidate")}
assert all(len(v)==25 for v in census_lookup.values())  # A2-G1: REV2's absence assertion inverted (E2R2-F02 fixed)
helper=dict(name_true_unknown=tc._np_dtype("not_a_dtype",name=True),call_sites=["tensor:60 then :63", "make:72 then :75", "_causal_mask:42 then :43"],assessment="Safe at the three internal call sites: name=True is type-only, unsupported names are rejected before data work; not a standalone supported-name validator")
result=dict(nonstring_standard_shapes=checks,shape_conversion_before_guard=shape_results,mask_six_supported=mask_pairs,subclass_supported_pairs=subclasses,name_parameter=helper,omitted_lookup=lookup,lookup_absent_from_census=False,census_lookup_rows={a:len(v) for a,v in census_lookup.items()},lookup_source_equal=source("bc02d3a1","tensor_cuda/tensor_cuda/apa_sp2.py")==source("A2","tensor_cuda/tensor_cuda/apa_sp2.py"))
save("dtype_boundaries.json",result)
print(json.dumps(dict(nonstring_standard_cases=len(checks),mask_pairs=len(mask_pairs),supported_subclass_pairs=len(subclasses),shape_conversion=shape_results,omitted_lookup=lookup[:4],lookup_absent=False,census_lookup_rows={a:len(v) for a,v in census_lookup.items()},lookup_source_equal=result["lookup_source_equal"]),indent=2))
