"""CPU source/arithmetic checks. Prior art: IEEE754-2008; TurboQuant (Zandieh et al. 2025),
SHA256/NIST FIPS180-4, and TOCTOU validation (unverified lead: CWE-367).
Uses unchanged pure table functions from live quant.py; no device emulation claim.
"""
import ast, ctypes, hashlib, importlib.util, json, math, os, pathlib, sys
import numpy as np
import pytest
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/"artifacts/apa_p1_rev3"
WT=ROOT.parent
MODULES={"F":(WT/"tensor_fork_apa_wt_p1d/artifacts/apa_p1f1/arms/F/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so","5c17147715ffc422b0aa7ba8fae826c554a4caff63b1231c22ce580867efc243"),"F2":(WT/"tensor_fork_apa_wt_p1f2/artifacts/apa_p1f2/arms/F2/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so","6f03b2cdd5d7f9bee4456ed34759c6c710cccfab9a50f9ac0fc6a678c076f579")}
def sha(p):
 with pathlib.Path(p).open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()
receipts={}
for name,(p,h) in MODULES.items():
 actual=sha(p); assert actual==h,(name,actual);receipts[name]={"path":str(p),"sha256":actual}
(OUT/"MODULES.json").write_text(json.dumps(receipts,indent=2)+"\n")
spec=importlib.util.spec_from_file_location("review_policy",ROOT/"tensor_cuda/tests/test_apa_p1d_policy.py")
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
mp=pytest.MonkeyPatch();mp.setenv("TC_APA_SP","1");mp.delenv("TC_APA_SP_STORAGE",raising=False)
e=t.make_package(mp);P=e.packed
# Prior art: use-after-validation/TOCTOU, CWE-367 (unverified lead to check).
# No monkeypatch of product functions: public constructor accepts a stateful mapping.
class Entries(dict):
 def __init__(self,allowed): super().__init__(P.PACKED_PINNED_ENTRIES);self.reads=0;self.allowed=allowed
 def __getitem__(self,k):
  self.reads+=1
  return super().__getitem__(k) if self.reads<=self.allowed else 0.0
m=P.PackedRegisteredMargins(P.PACKED_TABLE_SHA256,Entries(2))
lookup=m.lookup(4,64,"float32",.125)
assert lookup==0.0
k=t.keys(B=1,KV=1,D=64);p=e.sp2.quantize_sp2_keys(k,4)
q=t.FakeTensor((1,1,1,64));v=t.FakeTensor(k.shape)
m=P.PackedRegisteredMargins(P.PACKED_TABLE_SHA256,Entries(4))
out=e.sp2.apa_selective_attention_sp(q,k,p,v,.125,1.0,bulk_bits=4,margins=m)
route=P.last_route("attention");assert out=="packed_out" and route["e_q"]==0.0 and route["delta"]==0.0
pin_case={"lookup":lookup,"attention":out,"route":route,"entries_reads":m.entries.reads,"device_stub_calls":len(t.calls(e,"packed_run"))}
mp.undo()
# Reuse the real CPU-only table routines, excluding quant.py imports of the engine.
quant=ast.parse((ROOT/"tensor_cuda/tensor_cuda/quant.py").read_text())
funcs=[n for n in quant.body if isinstance(n,ast.FunctionDef) and n.name in {"_beta_pdf","build_codebook","_rotation"}]
ns={"np":np,"math":math,"_CODEBOOK_CACHE":{},"_ROT_CACHE":{}}
exec(compile(ast.Module(body=funcs,type_ignores=[]),"quant.py:pure_functions","exec"),ns)
D=64;R=ns["_rotation"](D,42);CB=ns["build_codebook"](D,4);BND=(.5*(CB[:-1]+CB[1:])).astype(np.float32)
f32=np.float32
lib=ctypes.CDLL("libm.so.6");fmaf=lib.fmaf;fmaf.argtypes=[ctypes.c_float]*3;fmaf.restype=ctypes.c_float
fma=lambda a,b,c:f32(fmaf(float(a),float(b),float(c)))
def dot(a,b):
 x=f32(0)
 for av,bv in zip(a,b): x=fma(av,bv,x)
 return x
q=np.ones(D,dtype=np.float32);qr=np.array([dot(q,row) for row in R],dtype=np.float32)
nums=[]
with np.errstate(all="ignore"):
 for label,element in [("positive_inf",np.inf),("negative_inf",-np.inf),("finite_squared_norm_overflow",f32(1e20))]:
  q=np.full(D,-1.0 if label=="finite_squared_norm_overflow" else 1.0,dtype=np.float32)
  qr=np.array([dot(q,row) for row in R],dtype=np.float32)
  k=np.zeros(D,dtype=np.float32);k[0]=element
  ss=f32(0)
  for x in k:ss=f32(ss+f32(x*x))
  norm=f32(np.sqrt(ss));unit=k/f32(norm+f32(1e-12))
  rotated=np.array([dot(unit,row) for row in R],dtype=np.float32)
  codes=(rotated[:,None]>BND[None,:]).sum(axis=1)
  z=CB[codes];dense=(z@R)*norm
  bulk_coefficient=dot(qr,z);old_bulk=f32(bulk_coefficient*norm)
  stored_norm=f32(np.nan) if not np.isfinite(k).all() else norm
  new_bulk=f32(bulk_coefficient*stored_norm)
  nums.append({"case":label,"k0":str(k[0]),"all_input_finite":bool(np.isfinite(k).all()),"ss":str(ss),"norm":str(norm),"stored_norm":str(stored_norm),"codes":np.unique(codes).tolist(),"bulk_coefficient":float(bulk_coefficient),"old_bulk":str(old_bulk),"F2_bulk":str(new_bulk),"dense_positive_inf":int(np.isposinf(dense).sum()),"dense_negative_inf":int(np.isneginf(dense).sum()),"dense_dot":str(dot(q,dense)),"bad_key_skipped_F2":bool(np.isneginf(new_bulk))})
print(json.dumps(nums,indent=2))
assert nums[-1]["bad_key_skipped_F2"] and nums[-1]["dense_dot"]=="nan"
# ISO C++17 signed int range reasoning; no compiled stand-in or device claim.
index=[]
for B,H,KV in [(65536,65536,8192),(46341,46344,5793)]:
 product=B*H;wrapped=ctypes.c_int32(product).value
 index.append({"B":B,"H":H,"KVH":KV,"G":H//KV,"true_BH":product,"signed32_BH":wrapped,"existing_eb2b_guard_rejects_under_modular_execution":wrapped>((2**31-1)//4),"cpp_semantics":"signed overflow is undefined; modular value illustrative only"})
result={"margin_pin_bypass":pin_case,"arithmetic":nums,"unguarded_eb2b_product":index,"scope":"CPU source arithmetic and recording fakes only; GPU output not measured"}
(OUT/"BLIND_CHECKS.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
