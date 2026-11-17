#!/usr/bin/env python3
"""P1R3-F02 / D12. Staged GPU reproduction; run unchanged from this worktree root.
CONFIRM: D64 finite overflow case has finite default packed output and a dense
non-finite output row, while the ordinary finite control is finite on both.
REFUTE FOR THIS INSTANCE: dense has a non-finite row and packed contains that
row too, with the finite control intact. Dense all-finite is INCONCLUSIVE.
A harness error or failed finite control is ERROR, not confirm/refute.
--cpu-prefix exercises argument parsing, sha-verified path import, BOTH real
public table loader signatures, and NumPy inputs, then exits BEFORE tc.tensor.
Prior art: IEEE754-2008 propagation; TurboQuant (Zandieh et al. 2025) inherited
norm/rotation; F1 path/sha loader (Project-Tensor 2026). No new algorithm.
"""
import argparse,contextlib,hashlib,io,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
PACKAGE=ROOT.parent/"tensor_fork_apa_wt_p1f2/artifacts/apa_p1f2/arms/F2/tensor_cuda"
MODULE_SHA="6f03b2cdd5d7f9bee4456ed34759c6c710cccfab9a50f9ac0fc6a678c076f579"
PACKED_SHA="1897fedd2a8d05945e34433c75a7afd3b33feb64e0a6b2ca410d611d18868e98"
DENSE_SHA="5b6f8bfb405b40c03da21ad8d50eb40c11b85ed9559ee717d2a105c90c0e618d"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def run(args):
 assert Path.cwd().resolve()==ROOT,"run from the registered worktree root"
 module=PACKAGE/"_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
 assert sha(module)==MODULE_SHA,"REFUSED module sha mismatch before import"
 for key in ("TC_APA_SP_STORAGE","TC_APA_SPQ1_REFERENCE","APA_FXTC1_VARIANT","APA_P1_EARLY_COUNTER"):
  os.environ.pop(key,None)
 os.environ["TC_APA_SP"]="1"
 # Same package-path import as scripts/apa_p1f1_gpu.py load_arm; no copied module.
 sys.path.insert(0,str(PACKAGE.parent))
 import tensor_cuda as tc
 assert Path(tc.__file__).resolve().parent==PACKAGE.resolve(),"wrong package path"
 assert sha(tc._C.__file__)==MODULE_SHA,"loaded module sha mismatch"
 pt=ROOT/"artifacts/apa_p1d/packed_e_q_table.json"
 dt=Path("/mnt/ForgeRealm/Project-Tensor/artifacts/apa_sp2/e_q_table.json")
 assert sha(pt)==PACKED_SHA and sha(dt)==DENSE_SHA,"table sha mismatch"
 pm=tc.PackedRegisteredMargins.load(pt)
 dm=tc.RegisteredMargins.load(dt,expected_sha256=DENSE_SHA)
 assert pm.lookup(4,64,"float32",.125)>0 and dm.lookup(4,64,"float32",.125)>0
 q=-np.ones((1,1,1,64),dtype=np.float32)
 k=np.zeros((1,1,3,64),dtype=np.float32);k[0,0,1,0]=np.float32(1e20)
 v=np.arange(192,dtype=np.float32).reshape(1,1,3,64)/np.float32(256)
 assert all(np.isfinite(a).all() for a in (q,k,v))
 assert all(a.dtype==np.float32 and a.flags.c_contiguous for a in (q,k,v))
 meta={"finding":"P1R3-F02","module_sha256":MODULE_SHA,"tables_loaded":True,
       "all_inputs_finite":True,"shape":[list(a.shape) for a in (q,k,v)]}
 if args.cpu_prefix:
  return dict(meta,verdict="CPU_PREFIX_PASS",stopped_before="tc.tensor (first device call)")
 # FIRST DEVICE CALL is below. Never run this branch in the review seat.
 rows=[]
 with tc.no_grad():
  for name,Knp in (("finite_control",k*np.float32(1e-20)),("finite_norm_overflow",k)):
   Q,K,V=[tc.tensor(a,device="cuda") for a in (q,Knp,v)]
   pk=tc.quantize_sp2_keys(K,4)
   dk=tc.quantize_sp2_keys(K,4,storage="dense")
   assert isinstance(pk,tc.PackedSpKeys),"default route is not packed"
   packed=tc.apa_selective_attention_sp(Q,K,pk,V,.125,.5,bulk_bits=4,margins=pm).numpy()
   dense=tc.apa_selective_attention_sp(Q,K,dk,V,.125,.5,bulk_bits=4,margins=dm).numpy()
   assert packed.shape==dense.shape and packed.dtype==dense.dtype
   a=np.flatnonzero(~np.isfinite(packed).all(axis=-1).reshape(-1)).tolist()
   b=np.flatnonzero(~np.isfinite(dense).all(axis=-1).reshape(-1)).tolist()
   rows.append({"case":name,"packed_nonfinite_rows":a,"dense_nonfinite_rows":b,
                "norms":pk._packed_tuple()[1].numpy().reshape(-1).astype(str).tolist()})
 control,bad=rows
 if control["packed_nonfinite_rows"] or control["dense_nonfinite_rows"]:raise RuntimeError("finite control failed")
 pa,de=set(bad["packed_nonfinite_rows"]),set(bad["dense_nonfinite_rows"])
 verdict="CONFIRMED" if de-pa else "REFUTED_FOR_THIS_INSTANCE" if de and de<=pa else "INCONCLUSIVE"
 return dict(meta,verdict=verdict,cases=rows)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--cpu-prefix",action="store_true")
 args=parser.parse_args()
 try:
  with contextlib.redirect_stdout(io.StringIO()):result=run(args)
  print(json.dumps(result,allow_nan=False))
  return 2 if result["verdict"]=="INCONCLUSIVE" else 0
 except Exception as exc:
  print(json.dumps({"finding":"P1R3-F02","verdict":"ERROR","error":repr(exc)}));return 1
if __name__=="__main__":sys.exit(main())
