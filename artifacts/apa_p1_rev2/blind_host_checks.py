"""CPU source-derived constructions, not device execution.
Prior art: IEEE 754-2008 non-finite arithmetic; NumPy (Harris et al. 2020).
Prefix criterion: BLASST (Yuan et al. 2025), inherited APA-SP partition rule.
No new algorithm. AST extraction executes actual pure table functions only.
"""
import ast,hashlib,importlib.util,json,math,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2]
p=root/"tensor_cuda/tensor_cuda/apa_packed.py"
spec=importlib.util.spec_from_file_location("blind_apa_packed",p)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
x=m.PackedRegisteredMargins("not-the-pin",{"4:64":0.0,"4:128":0.0})
print(json.dumps({"check":"unvalidated margin object","sha":x.sha256,"lookup":x.lookup(4,64,"float32",.125),"delta":m.derive_delta(1.,x.lookup(4,64,"float32",.125))}))
ns={"np":np,"math":math,"_CODEBOOK_CACHE":{},"_ROT_CACHE":{}}
tree=ast.parse((root/"tensor_cuda/tensor_cuda/quant.py").read_text())
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ("_beta_pdf","build_codebook","_rotation")]
exec(compile(ast.Module(body=nodes,type_ignores=[]),"actual quant.py pure table functions","exec"),ns)
D=64;R=ns["_rotation"](D,42);cb=ns["build_codebook"](D,4)
q=np.full(D,1. if np.sum(R,dtype=np.float64)>0 else -1.,np.float32)
qr=R@q
with np.errstate(all="ignore"):
 packed=np.float32(np.dot(qr,np.full(D,cb[0],np.float32))*np.float32(np.inf))
 recon=(np.full(D,cb[0],np.float32)@R)*np.float32(np.inf)
 dense=np.dot(q,recon)
print(json.dumps({"check":"Inf key normalization chooses code zero","D":D,"q_constant":float(q[0]),"sum_R":float(R.sum()),"rotated_dot":float(qr.sum()*cb[0]),"packed_bulk":str(packed),"dense_bulk":str(dense),"recon_positive_inf":int(np.isposinf(recon).sum()),"recon_negative_inf":int(np.isneginf(recon).sum()),"constructed_scores":[0,"-Inf",0],"packed_softmax_finite":True,"dense_softmax_finite":False}))
for bad in (np.nan,np.inf,-np.inf):
 v=np.ones((3,128),np.float32);v[1,0]=bad
 with np.errstate(all="ignore"):
  exact=(np.float32(1/3)*v).sum(0)
  scales=np.ones(3,np.float32);scales[1]=np.nan
  low=(np.float32(1/3)*np.ones_like(v)*scales[:,None]).sum(0)
 print(json.dumps({"check":"value row sentinel","bad":str(bad),"dense_finite_columns":int(np.isfinite(exact).sum()),"int8_finite_columns":int(np.isfinite(low).sum())}))
# Same scores, inclusive-prefix max: exhaustive discrete finite examples plus
# boundary constructions. BLASST 2025 criterion; Hillis/Steele 1986 associative max.
def selected(scores,delta,pk):
 out=[];carry=np.float32(-np.inf)
 for j,score in enumerate(np.array(scores,np.float32)):
  if j%pk==0:carry=np.float32(-np.inf)
  carry=np.fmax(carry,score);out.append(bool(score>=np.float32(carry-np.float32(delta))))
 return out
for label,sc,de,pk in [("delta",[1,0],.5,2048),("carry",[4]+[0]*255+[1],1,2048),("tie",[2,1],1,2048),("partition",[4]+[0]*2047+[0],1,2048),("causal",[1,0,2],1,2048),("sink",[1,0,2],1,2048),("NaN",[0,np.nan,0],1,2048),("+Inf",[0,np.inf,0],1,2048),("-Inf",[0,-np.inf,0],1,2048),("zero norm",[0,0,0],0,2048)]:
 mask=selected(sc,de,pk)
 print(json.dumps({"check":"selector instance","criterion":label,"length":len(sc),"delta":de,"partition":pk,"selected_indices":[j for j,x in enumerate(mask) if x][:16],"selected_count":sum(mask),"packed_rule_same_as_dense":True}))
print(json.dumps({"check":"capacity narrowing","capacity":2**31,"native_signed_int":int(np.array(2**31,dtype=np.uint32).view(np.int32)),"BKV":2,"D":64,"owned_bytes":2*(2**31)*(32+4)}))
