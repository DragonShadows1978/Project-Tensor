"""Host arithmetic and two source-derived selector schedules; no GPU run.
Prior art: IEEE 754-2008; BLASST (Yuan et al. 2025) inclusive running maximum;
Hillis/Steele (1986) associative scan. Both schedules restate existing code.
These constructions test the semantic argument, not CUDA execution.
"""
from pathlib import Path
import hashlib,importlib.util,itertools,json,math,sys
import numpy as np
root=Path(__file__).resolve().parents[2]
def dense(scores,delta,pk):
 prefix=np.float32(-np.inf);out=[]
 for j,x in enumerate(scores):
  if j%pk==0:prefix=np.float32(-np.inf)
  prefix=np.fmax(prefix,x);out.append(bool(x>=np.float32(prefix-delta)))
 return np.array(out,np.uint8)
def chunked(scores,delta,pk):
 out=np.zeros(len(scores),np.uint8)
 cm=[np.fmax.reduce(scores[i:i+256],initial=np.float32(-np.inf)) for i in range(0,len(scores),256)]
 for c,start in enumerate(range(0,len(scores),256)):
  first=(start//pk)*pk//256
  carry=np.fmax.reduce(np.asarray(cm[first:c],np.float32),initial=np.float32(-np.inf))
  for t in range(start,min(start+256,len(scores)),128):
   x=scores[t:min(t+128,start+256,len(scores))];p=x.copy()
   # Simultaneous lane updates, exactly the source scan's inclusive order.
   for off in (1,2,4,8,16):
    old=p.copy()
    for j in range(len(x)):
     if j%32>=off:p[j]=np.fmax(old[j],old[j-off])
   wm=[p[min(w+31,len(p)-1)] for w in range(0,len(p),32)]
   for j in range(len(x)):
    prefix=np.fmax(p[j],carry)
    for w in range(j//32):prefix=np.fmax(prefix,wm[w])
    out[t+j]=bool(x[j]>=np.float32(prefix-delta))
   for m in wm:carry=np.fmax(carry,m)
 return out
cases=[("delta",[1,0],.5), ("carry",[4]+[0]*255+[1],1), ("tie",[2,1],1),("partition",[4]+[0]*2047+[0],1),("causal",[1,0,2],1),("sink",[1,0,2],1),("NaN",[0,np.nan,0],1),("+Inf",[0,np.inf,0],1),("-Inf",[0,-np.inf,0],1),("zero norm",[0,0,0],0)]
with np.errstate(all="ignore"):
 for label,seq,d in cases:
  scores=np.asarray(seq,np.float32);a=dense(scores,np.float32(d),2048);b=chunked(scores,np.float32(d),2048)
  informative=[bool(z.size and np.any(z!=0) and np.any(z!=z[0])) for z in (a,b)]
  assert np.array_equal(a,b),(label,a,b)
  print(json.dumps(dict(criterion=label,shape=list(a.shape),delta=d,mismatches=int(np.count_nonzero(a!=b)),informative=informative,selected_indices=np.flatnonzero(a).tolist()[:16])))
 count=0
 for seq in itertools.product([-np.inf,-1.,0.,1.,np.inf,np.nan],repeat=4):
  for de in (0.,1.):
   sc=np.asarray(seq,np.float32);a=dense(sc,np.float32(de),2048);b=chunked(sc,np.float32(de),2048)
   informative=[bool(z.size and np.any(z!=0) and np.any(z!=z[0])) for z in (a,b)]
   assert np.array_equal(a,b);count+=1
 print(json.dumps(dict(constructed_exhaustive_cases=count,mismatches=0,scope="host schedules only; no new universal device test claim")))
 score=np.float32(128)*np.finfo(np.float32).max
 den=np.exp(np.float32(score-score));acc=np.float32(den*1)
 before=np.float32(acc/den) if den>0 else np.float32(0)
 after=np.float32(acc/den) if den!=0 else np.float32(0)
 print(json.dumps(dict(criterion="finite-input overflow",score=str(score),den=str(den),old_guard=str(before),new_guard=str(after))))
 # For finite key logits and NaN/+Inf sinks, dense merge folds corr first,
 # then multiplies finite numerator by inv=0; packed now divides by NaN.
 for sink in (np.nan,np.inf,-np.inf):
  mx=np.float32(0);den=np.float32(1);acc=np.float32(2)
  new=np.fmax(mx,np.float32(sink));corr=np.exp(np.float32(mx-new));dd=np.float32(den*corr+np.exp(np.float32(sink-new)));aa=np.float32(acc*corr)
  inv=np.float32(1/dd) if dd>0 else np.float32(0)
  dv=np.float32(aa*inv);pv=np.float32(aa/dd) if dd!=0 else np.float32(0)
  print(json.dumps(dict(criterion="sink",sink=str(sink),dense=str(dv),packed=str(pv))))
