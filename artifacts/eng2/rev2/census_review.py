"""Census and supported host-path differential checks. Prior art: McKeeman 1998 differential testing; ENG2 dtype_census.py; unverified -- lead to check. No novel algorithm."""
from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,json,subprocess,sys,re
import numpy as np
ROOT=Path.cwd();OUT=ROOT/"artifacts/eng2/rev2";TOOL=ROOT/"artifacts/eng2/tool"
rows={}
for arm in ["baseline","candidate"]:
    r=subprocess.run([sys.executable,str(TOOL.parent/"dtype_census.py"),str(TOOL/"arms"/arm/"tensor_cuda"),arm,str(OUT/("census_"+arm+".json"))],capture_output=True,text=True,timeout=90)
    (OUT/("census_"+arm+".log")).write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
    data=json.loads((OUT/("census_"+arm+".json")).read_text());prior=json.loads((TOOL.parent/"a1"/("L60_dtype_census_"+arm+".json")).read_text())
    normalize=lambda v:re.sub(r"object at 0x[0-9a-f]+", "object at ADDRESS",json.dumps(v,sort_keys=True))
    assert normalize(data["rows"])==normalize(prior["rows"])
    rows[arm]=dict(count=len(data["rows"]),all_rows_match=True,spot_checks=[r for r in data["rows"] if r["entry"] in ["tc.tensor","tc.randn","tc.nn.parameter","tc.empty","Tensor.astype"] and r["kind"] in ["order","non-string"]])
# Execute top-level factories from both exact source versions with a capture at the C++ boundary.
sys.path.insert(0,str(ROOT/"tensor_cuda"));import tensor_cuda as tc
import importlib.util
sp=importlib.util.spec_from_file_location("e3_census_review",ROOT/"tensor_cuda/tests/test_eng1.py");E=importlib.util.module_from_spec(sp);sp.loader.exec_module(E)
class Captured:
    def __init__(self,a):self.a=np.array(a,copy=True);self.dtype=str(self.a.dtype);self.casts=[]
    def astype(self,d):self.a=E.e3_round(self.a,d).astype(np.float32);self.dtype=d;self.casts.append(d);return self

def factory_namespace(rev):
    src=subprocess.check_output(["git","--no-optional-locks","show",rev+":tensor_cuda/tensor_cuda/__init__.py"],text=True)
    nodes=[]
    for n in ast.parse(src).body:
        if isinstance(n,ast.FunctionDef) and n.name in ["_np_dtype","tensor","_factory"]:nodes.append(n)
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ["_NP_DTYPE","_DEVICE_CAST_DTYPE","zeros","ones","rand","randn"] for t in n.targets):nodes.append(n)
    ns=dict(np=np,_C=SimpleNamespace(tensor=lambda a,*args:Captured(a)));exec(compile(ast.Module(body=nodes,type_ignores=[]),rev,"exec"),ns);return ns
ns={a:factory_namespace(r) for a,r in [("baseline","bc02d3a1"),("candidate","1a4beed1")]}
values=np.array([[0.,1.,2.5,3.],[7.,100.,255.,.3]])
paired=[]
for entry in ["tensor","zeros","ones","randn","rand"]:
    for dtype in ["float32","float16","int64","bool","uint8","bfloat16",None]:
        got={}
        for arm in ns:
            np.random.seed(2214)
            kw={} if dtype is None else dict(dtype=dtype)
            with np.errstate(invalid="ignore"):
                t=ns[arm][entry](values,**kw) if entry=="tensor" else ns[arm][entry](3,5,**kw)
            got[arm]=t
        a,b=got.values();assert a.dtype==b.dtype and a.casts==b.casts and a.a.shape==b.a.shape
        nontrivial=all(np.any(t.a!=0) and np.ptp(t.a.astype(float))>0 for t in got.values())
        if nontrivial:assert a.a.tobytes()==b.a.tobytes()
        # Zeros/ones and casts that become constant are judged by the unchanged expression, not a vacuous bit comparison.
        paired.append(dict(entry=entry,dtype=dtype or "default",result_dtype=b.dtype,shape=list(b.a.shape),casts=b.casts,nonzero_nonconstant=nontrivial,check="host bytes equal" if nontrivial else "unchanged factory/conversion expression; no array equality claim"))
# Additional dtype entries absent from the emitted L60 rows; all safe CPU-only NumPy or zero-byte operations.
from tensor_cuda import functional as F
from tensor_cuda.quantization import affine as Q
extra=[]
for label,d in [("float64","float64"),("double","double"),("int32","int32"),("not_a_dtype","not_a_dtype"),("None",None),("np.float32",np.float32)]:
    F._causal_cache={}
    probes={"functional._causal_mask":lambda:F._causal_mask(0,0,"cpu",d),
            "quantization.affine.quantize_affine_per_group(scale_dtype)":lambda:Q.quantize_affine_per_group(np.arange(32).reshape(2,16),4,16,scale_dtype=d).scales,
            "quantization.affine.dequantize_affine_per_group(out_dtype)":lambda:Q.dequantize_affine_per_group(np.arange(16,dtype=np.uint8).reshape(2,8),np.ones((2,1)),np.zeros((2,1)),4,16,16,out_dtype=d),
            "quantization.affine.dequantize_symmetric_per_group(out_dtype)":lambda:Q.dequantize_symmetric_per_group(np.arange(16,dtype=np.uint8).reshape(2,8),np.ones((2,1)),4,16,16,out_dtype=d)}
    for name,fn in probes.items():
        try:t=fn();rec=dict(outcome="returned",dtype=str(t.dtype))
        except Exception as e:rec=dict(outcome=type(e).__name__,message=str(e))
        extra.append(dict(entry=name,requested=label,**rec))
result=dict(census=rows,supported_source_capture=paired,additional_entries=extra)
(OUT/"census_review.json").write_text(json.dumps(result,indent=2)+"\n")
print("L60 reproduction:",{a:v["count"] for a,v in rows.items()},"rows match after object-address normalization")
print("Supported/default host paths:",len(paired),"pairs;",sum(r["nonzero_nonconstant"] for r in paired),"nontrivial byte comparisons")
print("Additional entries:",json.dumps(extra,indent=1))
