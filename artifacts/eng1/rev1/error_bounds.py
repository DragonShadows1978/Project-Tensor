# Prior art: Moore (1966), Interval Analysis, outward interval propagation;
# Higham (2002), gamma_n; IEEE 754 rounding. Taken: standard absolute-error enclosures.
# Ours: conservative bounds for the registered norm inputs. No engine run/model claim.
import os,sys,json,math,importlib.util
from pathlib import Path
import numpy as np
root=Path.cwd(); rev=root/'artifacts/eng1/rev1'
sys.path.insert(0,str(root/'artifacts/eng1/tool/arms/candidate/tensor_cuda'))
s=importlib.util.spec_from_file_location('eng1',root/'tensor_cuda/tests/test_eng1.py')
T=importlib.util.module_from_spec(s); s.loader.exec_module(T)
def q(x,dt):
    z=np.array(x,np.float32)
    if dt=='float16':return z.astype(np.float16).astype(np.float64)
    if dt=='float32':return z.astype(np.float64)
    b=z.view(np.uint32).astype(np.uint64)
    return ((b+0x7fff+((b>>16)&1))&0xffff0000).astype(np.uint32).view(np.float32).astype(np.float64)
u32=2.**-24
def outward(lo,hi):
    return (np.nextafter(lo,-np.inf),np.nextafter(hi,np.inf))
def point(x):return (np.asarray(x,np.float64),np.asarray(x,np.float64))
def plus(a,b):return outward(a[0]+b[0],a[1]+b[1])
def minus(a,b):return outward(a[0]-b[1],a[1]-b[0])
def times(a,b):
    v=np.stack(np.broadcast_arrays(a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]))
    return outward(v.min(axis=0),v.max(axis=0))
def square(a):
    lo=np.where((a[0]<=0)&(a[1]>=0),0,np.minimum(a[0]**2,a[1]**2)); hi=np.maximum(a[0]**2,a[1]**2)
    return outward(lo,hi)
def rounded(a,dt,fp32_op=True):
    mag=np.maximum(abs(a[0]),abs(a[1]))
    # Every elementwise CUDA op computes in f32 then stores dtype, including scalar add/mul.
    err=(u32*mag+2.**-150) if fp32_op else 0
    mid=outward(a[0]-err,a[1]+err)
    if dt=='float32': return mid
    err=T.U[dt]*np.maximum(abs(mid[0]),abs(mid[1]))+T.SUB_HALF[dt]
    return outward(mid[0]-err,mid[1]+err)
def mean(a,axes,dt):
    n=math.prod(a[0].shape[j] for j in axes)
    # Bound any reduction order by gamma_(n-1)*sum |term|, plus subnormal additions.
    lo=a[0].sum(axis=axes,keepdims=True); hi=a[1].sum(axis=axes,keepdims=True)
    err=((n-1)*u32/(1-(n-1)*u32))*np.maximum(abs(a[0]),abs(a[1])).sum(axis=axes,keepdims=True)+(n-1)*2.**-150
    sums=rounded(outward(lo-err,hi+err),dt,fp32_op=False)
    # Scalar reciprocal is rounded to fp32 in ew_scalar.
    return rounded(times(sums,point(float(np.float32(1/n)))),dt)
def invsqrt(a,dt):
    assert np.all(a[0]>0)
    base=outward(a[1]**-.5,a[0]**-.5)
    err=(4*2.**-23)*base[1]+2.**-150
    return rounded(outward(base[0]-err,base[1]+err),dt,fp32_op=False)
rows=[]
for cls in T.E3_CLASSES:
 for dt in T.E3_DTYPES:
  for case in T.E3_CASES:
    shape=T.E3_SHAPE[cls]; x=q(T.e3_inputs(cls,case),dt)
    C,bs=T._affine_shape(cls,shape); w=q(1+.1*np.arange(C),dt)
    b=None if cls=='RMSNorm' else q(.05*np.arange(C)-.1,dt)
    ref,_=T.e3_reference(cls,x,w,b,np.ones(shape))
    work='float32' if cls=='GroupNorm' else dt
    xs=x.reshape(shape[0],2,-1) if cls=='GroupNorm' else x
    axes=(-1,) if cls in ('LayerNorm','RMSNorm','GroupNorm') else T._e3_axes(cls,shape)
    a=point(xs)
    c=a if cls=='RMSNorm' else rounded(minus(a,mean(a,axes,work)),work)
    stat=mean(rounded(square(c),work),axes,work)
    variance_eps=rounded(plus(stat,point(float(np.float32(T.E3_EPS[cls])))),work)
    inv=invsqrt(variance_eps,work)
    z=rounded(times(c,inv),work)
    z=tuple(t.reshape(shape) for t in z)
    y=rounded(times(z,point(w.reshape(bs))),work)
    if b is not None:y=rounded(plus(y,point(b.reshape(bs))),work)
    if cls=='GroupNorm':y=rounded(y,dt,fp32_op=False)
    err=np.maximum(abs(y[0]-ref),abs(y[1]-ref))
    scale=float(np.max(abs(ref))); tol=T.e3_tolerance(cls,'forward',dt,scale)
    rows.append(dict(cls=cls,dtype=dt,case=case,absolute_bound=float(err.max()),
                     registered_tolerance=tol,bound_over_tolerance=float(err.max()/tol),
                     min_var_plus_eps=float(variance_eps[0].min())))
# Closed-form sensitivity of the near_eps centered cases: |delta mean|/sqrt(var+eps).
# Explain why output-relative op counts cannot bound it; absolute mean error scales with offset.
sensitivity={}
for dt in T.E3_DTYPES:
    u=T.U[dt]
    sensitivity[dt]=dict(mean_error_leading_bound=u*.5,
                        centering_normalized_error_leading_bound=u*.5/math.sqrt(2e-5))
result=dict(e3_forward_intervals=rows,centering_sensitivity=sensitivity,
    worst_by_ratio=max(rows,key=lambda r:r['bound_over_tolerance']),
    scope="Outward fp64 interval arithmetic with IEEE absolute+relative rounding model and 4-ulp powf. Conservative forward bound for every registered census case; not a GPU result. Half backward can overflow internally, so no finite universal honest-error bound exists there.")
(rev/'error_bounds.json').write_text(json.dumps(result,indent=2)+'\n')
print('Worst conditioned bound',result['worst_by_ratio'])
print('RMS near eps',[r for r in rows if r['cls']=='RMSNorm' and r['case']=='near_eps'])
print('centering',sensitivity)
