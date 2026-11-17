#!/usr/bin/env python3
"""P1R2-F02, GPU staging only. Run unchanged from this worktree root.
CONFIRM: for either Inf sign, default packed output is finite while the dense
opt-out output on the same exact Q/K/V is non-finite. REFUTE for this instance:
both Inf cases keep the dense non-finite row non-finite in packed output.
A finite dense control is INCONCLUSIVE, never a refutation. Other q/k/v NaN,
Inf and zero-key cases report classification only; they are not universal proof.
Prior art: IEEE 754-2008; TurboQuant (Zandieh et al., 2025) inherited tables;
F1 path/sha loader (Project-Tensor, 2026); no new algorithm.
"""
import contextlib,hashlib,io,json,os,sys
from pathlib import Path
import numpy as np
BASE=Path("/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1d")
PIN="5c17147715ffc422b0aa7ba8fae826c554a4caff63b1231c22ce580867efc243"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def run():
 package=BASE/"artifacts/apa_p1f1/arms/F/tensor_cuda"
 so=package/"_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
 if sha(so)!=PIN:raise RuntimeError("module sha mismatch before import")
 # Same package-path loading convention as scripts/apa_p1f1_gpu.py:164-175.
 sys.path.insert(0,str(package.parent))
 import tensor_cuda as tc
 if Path(tc.__file__).resolve().parent!=package.resolve() or sha(tc._C.__file__)!=PIN:
  raise RuntimeError("loaded module/path mismatch")
 for n in ("TC_APA_SP_STORAGE","TC_APA_SPQ1_REFERENCE","APA_FXTC1_VARIANT","APA_P1_EARLY_COUNTER"):
  os.environ.pop(n,None)
 os.environ["TC_APA_SP"]="1"
 pt=BASE/"artifacts/apa_p1d/packed_e_q_table.json"
 dt=Path("/mnt/ForgeRealm/Project-Tensor/artifacts/apa_sp2/e_q_table.json")
 if sha(pt)!="1897fedd2a8d05945e34433c75a7afd3b33feb64e0a6b2ca410d611d18868e98":raise RuntimeError("packed table sha mismatch")
 if sha(dt)!="5b6f8bfb405b40c03da21ad8d50eb40c11b85ed9559ee717d2a105c90c0e618d":raise RuntimeError("dense table sha mismatch")
 pm=tc.PackedRegisteredMargins.load(pt);dm=tc.RegisteredMargins.load(dt)
 rows=[]
 cases=[("k_posinf","k",np.inf),("k_neginf","k",-np.inf),("k_nan","k",np.nan),("q_posinf","q",np.inf),("q_neginf","q",-np.inf),("q_nan","q",np.nan),("v_nan","v",np.nan),("v_posinf","v",np.inf),("v_neginf","v",-np.inf),("zero_keys",None,0.),("q_posinf_all_negative_logits","q",np.inf),("q_neginf_all_negative_logits","q",-np.inf)]
 with tc.no_grad():
  for name,operand,bad in cases:
   q=np.ones((1,1,1,64),np.float32);k=np.zeros((1,1,3,64),np.float32)
   v=np.arange(3*64,dtype=np.float32).reshape(1,1,3,64)/256
   if "all_negative_logits" in name:
    q.fill(0);k.fill(-1. if bad>0 else 1.)
   if operand=="q":q[0,0,0,0]=bad
   elif operand=="k":k[0,0,1,0]=bad
   elif operand=="v":v[0,0,1,0]=bad
   Q,K,V=[tc.tensor(np.ascontiguousarray(x)) for x in (q,k,v)]
   pk=tc.quantize_sp2_keys(K,4);dk=tc.quantize_sp2_keys(K,4,storage="dense")
   if not isinstance(pk,tc.PackedSpKeys):raise RuntimeError("default did not choose PackedSpKeys")
   a=tc.apa_selective_attention_sp(Q,K,pk,V,.125,.5,bulk_bits=4,margins=pm).numpy()
   b=tc.apa_selective_attention_sp(Q,K,dk,V,.125,.5,bulk_bits=4,margins=dm).numpy()
   if a.dtype!=b.dtype or a.shape!=b.shape:raise RuntimeError("output type/shape mismatch")
   rows.append(dict(case=name,packed_finite=bool(np.isfinite(a).all()),dense_finite=bool(np.isfinite(b).all()),packed_finite_columns=int(np.isfinite(a).sum()),dense_finite_columns=int(np.isfinite(b).sum()),shape=list(a.shape)))
 inf=[x for x in rows if x["case"] in ("k_posinf","k_neginf")]
 confirm=any(x["packed_finite"] and not x["dense_finite"] for x in inf)
 refute=all(not x["packed_finite"] and not x["dense_finite"] for x in inf)
 return dict(finding="P1R2-F02",module_sha256=PIN,verdict="CONFIRMED" if confirm else "REFUTED_FOR_THIS_INSTANCE" if refute else "INCONCLUSIVE",cases=rows)
def main():
 try:
  with contextlib.redirect_stdout(io.StringIO()):result=run()
  print(json.dumps(result,allow_nan=False));return 2 if result["verdict"]=="INCONCLUSIVE" else 0
 except Exception as e:
  print(json.dumps(dict(finding="P1R2-F02",verdict="ERROR",error=repr(e))));return 1
if __name__=="__main__":sys.exit(main())
