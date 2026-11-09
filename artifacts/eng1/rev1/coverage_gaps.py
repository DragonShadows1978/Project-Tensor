# Prior art: DeMillo, Lipton and Sayward (1978) mutation testing; IEEE754 exact binary rounding.
# Taken: independent decision mutants. Ours: detect unpinned operands without editing product/tests/receipts.
import sys,json,importlib.util
from pathlib import Path
import numpy as np
root=Path.cwd(); rev=root/'artifacts/eng1/rev1'
sys.path.insert(0,str(root/'artifacts/eng1/tool/arms/candidate/tensor_cuda'))
s=importlib.util.spec_from_file_location('eng1',root/'tensor_cuda/tests/test_eng1.py')
T=importlib.util.module_from_spec(s);s.loader.exec_module(T)
def q(x):
 a=np.array(x,np.float32); b=a.view(np.uint32).astype(np.uint64)
 return ((b+0x7fff+((b>>16)&1))&0xffff0000).astype(np.uint32).view(np.float32).astype(np.float64)
unchanged=[]
for p in T.E2_P:
 x=q(T.E2_X); k=T.e2_scale_exponents(x,p); g=2.**-k
 wide=(g.astype(np.float32)*(np.power(x.astype(np.float32),np.float32(p-1))*np.float32(p))).astype(np.float32)
 late=q(wide)
 early=q(g*q(np.power(x.astype(np.float32),np.float32(p-1))*np.float32(p)))
 unchanged.append(dict(p=p,all_bfloat16_values_bit_equal=bool(np.array_equal(late,early))))
witness=None
for x in q(np.r_[T.E2_X,np.linspace(.5,3,301)]):
 for g in q([.1,.3,.7,.9,1.1,1.3]):
  p=3.; a=np.float32(x*x)*np.float32(3.)
  ref=g*3*x*x; late=float(q(np.float32(g)*a)); early=float(q(np.float32(g)*q(a)))
  tol=T.E2_RTOL['bfloat16']*abs(ref)+T.E2_ATOL['bfloat16']
  if abs(early-ref)>tol*1.03 and abs(late-ref)<=tol:
   witness=dict(x=float(x),upstream=float(g),p=p,reference=float(ref),late=late,early=early,tolerance=tol,
                early_error=abs(early-ref),late_error=abs(late-ref))
   break
 if witness:break
cases=[]
for name,(_,kh,kw,sh,sw,ph,pw,oh,ow) in T.E5_REFUSE.items():
 cases.append(dict(case=name,oh=oh,ow=ow,current_refusal=(oh<=0 or ow<=0),
                   hypothetical_ow_strict_negative=(oh<=0 or ow<0)))
r=dict(bfloat16_registered=unchanged,bfloat16_non_power_of_two_witness=witness,
       ow_zero_registered=cases,
       ow_zero_missing_witness=dict(shape=[0,1,6,2],kh=3,kw=3,sh=1,sw=1,ph=0,pw=0,OH=4,OW=0))
assert all(v['all_bfloat16_values_bit_equal'] for v in unchanged)
assert all(v['current_refusal']==v['hypothetical_ow_strict_negative'] for v in cases)
assert witness is not None
(rev/'coverage_gaps.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
