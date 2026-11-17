#!/usr/bin/env python3
"""P1R2-F06, GPU staging only. Run unchanged from this worktree root.
CONFIRM finite-input identity counterexample: finite Q/K/Kq/V and FLT_MAX
scale give all-zero finite D1 output on B and a non-finite output on F.
Also CONFIRM sink parity gap if a NaN/+Inf sink gives finite dense output and
non-finite F D1 output. REFUTE for these instances: no B/F classification
change on overflow and no F/dense sink divergence, with finite regular controls.
B and F load in separate child processes; each sha is checked before import.
This is a behavior/claim counterexample, not a proposal to restore NaN masking.
Prior art: IEEE 754-2008 overflow/NaNs; F1 path/sha loader (Project-Tensor, 2026).
"""
import contextlib,hashlib,io,json,os,subprocess,sys
from pathlib import Path
import numpy as np
BASE=Path("/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1d")
ARMS={"B":("artifacts/apa_p1d/arms/B_v2","8728e67e26027dbafe8a61592decb38e68171fa7b5634239ce0af7897c4e8c4e"),"F":("artifacts/apa_p1f1/arms/F","5c17147715ffc422b0aa7ba8fae826c554a4caff63b1231c22ce580867efc243")}
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def classify(a):
 a=np.asarray(a);u=np.ascontiguousarray(a).view({2:np.uint16,4:np.uint32,8:np.uint64}[a.dtype.itemsize]).ravel()
 return dict(finite=bool(np.isfinite(a).all()),all_zero=bool(np.all(a==0)),nonfinite_columns=int((~np.isfinite(a)).sum()),dtype=str(a.dtype),shape=list(a.shape),nonzero_bits=bool(np.any(u!=0)),nonconstant_bits=bool(u.size and np.any(u!=u[0])))
def child(label):
 rel,pin=ARMS[label];package=BASE/rel/"tensor_cuda";so=package/"_tensor_cuda.cpython-312-x86_64-linux-gnu.so"
 if sha(so)!=pin:raise RuntimeError("module sha mismatch before import")
 sys.path.insert(0,str(package.parent))
 import tensor_cuda as tc
 if Path(tc.__file__).resolve().parent!=package.resolve() or sha(tc._C.__file__)!=pin:raise RuntimeError("loaded path/sha mismatch")
 os.environ["TC_APA_SP"]="1"
 for name in ("TC_APA_SPQ1_REFERENCE","APA_FXTC1_VARIANT","APA_P1_EARLY_COUNTER"):os.environ.pop(name,None)
 result={}
 with tc.no_grad():
  q=np.ones((1,1,1,128),np.float32);k=np.ones((1,1,3,128),np.float32)
  v=np.arange(3*128,dtype=np.float32).reshape(1,1,3,128)/512
  Q,K,V=[tc.tensor(x) for x in (q,k,v)]
  for name,scale,sinkval in [("overflow",float(np.finfo(np.float32).max),None),("sink_nan",float(np.float32(1/np.sqrt(128))),np.nan),("sink_posinf",float(np.float32(1/np.sqrt(128))),np.inf),("regular_control",float(np.float32(1/np.sqrt(128))),None)]:
   sink=None if sinkval is None else tc.tensor(np.asarray([sinkval],np.float32))
   d1=tc._spo2.run(Q,K,K,V,scale,2.,False,sink,False)
   dense=tc._C.apa_selective_attention_sp(Q,K,K,V,scale,2.,False,sink,False)
   result[name]=dict(d1=classify(d1.numpy()),dense=classify(dense.numpy()))
 return dict(arm=label,module_sha256=pin,cases=result)
def parent():
 arms={}
 for label in ("B","F"):
  cmd=["timeout","180s",sys.executable,"-B",str(Path(__file__).resolve()),"--arm",label]
  p=subprocess.run(cmd,capture_output=True,text=True)
  if p.returncode!=0:raise RuntimeError(f"child {label} exit {p.returncode}: {p.stdout[-1500:]} {p.stderr[-1500:]}")
  arms[label]=json.loads(p.stdout)
 B,F=arms["B"]["cases"],arms["F"]["cases"]
 overflow=B["overflow"]["d1"]["finite"] and B["overflow"]["d1"]["all_zero"] and not F["overflow"]["d1"]["finite"]
 sinks=[name for name in ("sink_nan","sink_posinf") if F[name]["dense"]["finite"] and not F[name]["d1"]["finite"]]
 controls=all(x["cases"]["regular_control"]["d1"]["finite"] and x["cases"]["regular_control"]["dense"]["finite"] for x in arms.values())
 refute=controls and B["overflow"]["d1"]["finite"]==F["overflow"]["d1"]["finite"] and not sinks
 return dict(finding="P1R2-F06",verdict="CONFIRMED" if controls and (overflow or sinks) else "REFUTED_FOR_THESE_INSTANCES" if refute else "INCONCLUSIVE",finite_overflow_counterexample=bool(overflow),sink_parity_counterexamples=sinks,arms=arms)
def main():
 try:
  with contextlib.redirect_stdout(io.StringIO()):result=child(sys.argv[2]) if len(sys.argv)==3 and sys.argv[1]=="--arm" else parent()
  print(json.dumps(result,allow_nan=False));return 2 if result.get("verdict")=="INCONCLUSIVE" else 0
 except Exception as e:
  print(json.dumps(dict(finding="P1R2-F06",verdict="ERROR",error=repr(e))));return 1
if __name__=="__main__":sys.exit(main())
