#!/usr/bin/env python3
"""P1R2-F05 empty-helper portion, GPU staging only; worktree-root invocation.
CONFIRM: raw packed_rotate or packed_prepare on B=0 reaches a CUDA invalid
configuration error (attempted zero-grid launch), while a valid control still
works. REFUTE for these instances: both calls refuse positive-extent
preconditions on the host, with no CUDA launch error, and control stays finite.
Other errors or failure to exercise the control are INCONCLUSIVE/ERROR.
The separate EB3 timeline/G refusal is established by source, not this script.
Prior art: CUDA launch geometry validation (NVIDIA CUDA Programming Guide;
edition unverified); F1 path/sha loader (Project-Tensor, 2026); no new algorithm.
"""
import contextlib,hashlib,io,json,os,sys
from pathlib import Path
import numpy as np
BASE=Path("/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1d")
PIN="5c17147715ffc422b0aa7ba8fae826c554a4caff63b1231c22ce580867efc243"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def run():
 package=BASE/"artifacts/apa_p1f1/arms/F/tensor_cuda";so=package/"_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
 if sha(so)!=PIN:raise RuntimeError("module sha mismatch before import")
 sys.path.insert(0,str(package.parent))
 import tensor_cuda as tc
 if Path(tc.__file__).resolve().parent!=package.resolve() or sha(tc._C.__file__)!=PIN:raise RuntimeError("loaded path/sha mismatch")
 os.environ["TC_APA_SP"]="1"
 rows=[]
 with tc.no_grad():
  R=tc.tensor(np.eye(64,dtype=np.float32).reshape(1,64,64))
  Q=tc.tensor(np.arange(64,dtype=np.float32).reshape(1,1,1,64)/64)
  q0=tc.empty((0,1,1,64));k0=tc.empty((0,1,1,64))
  def control():
   out=tc._spo2.packed_rotate(Q,R,1,True).numpy()
   return bool(np.isfinite(out).all() and np.any(out!=0) and np.any(out!=out.ravel()[0]))
  before=control();tc.synchronize()
  for name,fn in [("packed_rotate",lambda:tc._spo2.packed_rotate(q0,R,1,True)),("packed_prepare",lambda:tc._spo2.packed_prepare(k0,R,0))]:
   try:fn();row=dict(entry=name,error_type=None,error="DID NOT RAISE")
   except Exception as e:row=dict(entry=name,error_type=type(e).__name__,error=str(e))
   row["cuda_launch_error"]="invalid configuration" in row["error"].lower()
   row["host_extent_refusal"]="positive" in row["error"].lower() and not row["cuda_launch_error"]
   tc.synchronize();row["control_after_finite_informative"]=control();rows.append(row)
  after=control()
 confirm=before and after and any(x["cuda_launch_error"] for x in rows)
 refute=before and after and all(x["host_extent_refusal"] for x in rows)
 return dict(finding="P1R2-F05",module_sha256=PIN,verdict="CONFIRMED" if confirm else "REFUTED_FOR_THESE_INSTANCES" if refute else "INCONCLUSIVE",control_before=before,control_after=after,cases=rows)
def main():
 try:
  with contextlib.redirect_stdout(io.StringIO()):result=run()
  print(json.dumps(result,allow_nan=False));return 2 if result["verdict"]=="INCONCLUSIVE" else 0
 except Exception as e:
  print(json.dumps(dict(finding="P1R2-F05",verdict="ERROR",error=repr(e))));return 1
if __name__=="__main__":sys.exit(main())
