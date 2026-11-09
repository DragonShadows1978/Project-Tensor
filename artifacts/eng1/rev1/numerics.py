# Prior art: Higham (2002) gamma_n, IEEE754 rounding; Fornberg (1988) centered differences;
# Ioffe/Szegedy 2015, Ba/Kiros/Hinton 2016, Zhang/Sennrich 2019 normalization.
# Taken: standard differentiation and roundoff analysis. Ours: independent evaluations of frozen witnesses.
import os,sys,json,math,importlib.util
from pathlib import Path
import numpy as np
root=Path.cwd(); rev=root/'artifacts/eng1/rev1'
sys.path.insert(0,str(root/'artifacts/eng1/tool/arms/candidate/tensor_cuda'))
spec=importlib.util.spec_from_file_location('registered',root/'tensor_cuda/tests/test_eng1.py')
T=importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
def quant(x,dt):
    a=np.asarray(x,dtype=np.float32)
    if dt=='float16': return a.astype(np.float16).astype(np.float64)
    if dt=='float32': return a.astype(np.float64)
    b=a.view(np.uint32).astype(np.uint64)
    return ((b+0x7fff+((b>>16)&1))&0xffff0000).astype(np.uint32).view(np.float32).astype(np.float64)
def independent_power(x,w,p):
    # x^p = exp(p log x); derivative is p exp((p-1)log x). No engine values used.
    return w*p*np.exp((p-1)*np.log(x))
rows=[]
with np.errstate(over='ignore',invalid='ignore',under='ignore'):
    for dt in ('float16','bfloat16','float32'):
        for p in T.E2_P:
            x=quant(T.E2_X,dt)
            mag=np.abs(p)*np.exp((p-1)*np.log(x))
            k=np.maximum(0,np.ceil(np.log2(mag))-10)
            if dt=='float16':
                keep=k<=24; x=x[keep]; k=k[keep]
            w=2.**(-k); ref=independent_power(x,w,p)
            reg=T.e2_reference_grad(x,w,p)
            np.testing.assert_allclose(reg,ref,rtol=6e-15,atol=0)
            h=x*2.**-17
            fd=w*(np.exp(p*np.log(x+h))-np.exp(p*np.log(x-h)))/(2*h)
            fderror=np.max(np.abs((fd-ref)/ref))
            assert fderror<2e-8
            power32=quant(np.power(x,p-1), 'float32')
            candidate=quant(quant(quant(power32*p,'float32')*w,'float32'),dt)
            # Lower bound on error with CUDA powf's declared <=4 ulp uncertainty.
            # Infinite overflow witnesses are robust; M06 at x=7 has error ~6*|correct derivative|.
            variants={
                'candidate':candidate,
                'M05':quant(w*quant(quant(power32,dt)*p,dt),dt),
                'M06':quant(quant(quant(np.power(x,p),'float32')*p,'float32')*w,dt),
                'M07':quant(w*quant(quant(power32*p,'float32'),dt),dt)}
            if dt=='float32':
                variants={name:candidate for name in variants}  # M05-M07 edit only the half branch.
            tol=T.E2_RTOL[dt]*np.abs(ref)+T.E2_ATOL[dt]
            detail={}
            for name,y in variants.items():
                err=np.abs(y-ref); rat=err/tol
                j=int(np.argmax(rat))
                detail[name]=dict(x=float(x[j]),weight=float(w[j]),reference=float(ref[j]),
                    got=float(y[j]),abs_error=float(err[j]),tolerance=float(tol[j]),ratio=float(rat[j]),
                    fail=bool(np.any(~np.isfinite(y)) or np.any(err>tol)))
            rows.append(dict(dtype=dt,p=p,finite_difference_max_rel=float(fderror),
                             reference_max_rel=float(np.max(np.abs((reg-ref)/ref))),variants=detail))
# Independently derive RMS Jacobian: J_ij = w_i*(delta_ij/r - x_i*x_j/(n*r^3)).
e3=[]
for dt in ('float16','bfloat16'):
    for case in T.E3_CASES:
        x=quant(T.e3_inputs('RMSNorm',case),dt)
        g=quant(T._e3_pattern(x.shape,1.1),dt)
        w=quant(1+.1*np.arange(x.shape[-1]),dt)
        r=np.sqrt(np.sum(x*x,axis=-1,keepdims=True)/x.shape[-1]+1e-6)
        ys=x*w/r; dx=np.empty_like(x)
        for n in range(x.shape[0]):
            J=np.diag(w/r[n,0])-np.outer(w*x[n],x[n])/(x.shape[-1]*r[n,0]**3)
            dx[n]=J.T@g[n]
        yreg,dreg=T.e3_reference('RMSNorm',x,w,None,g)
        np.testing.assert_allclose(yreg,ys,rtol=2e-15,atol=2e-15)
        np.testing.assert_allclose(dreg,dx,rtol=2e-13,atol=2e-11)
        h=1e-7 if case=='regular' else 1e-9
        idx=(1,5); xp=x.copy(); xm=x.copy(); xp[idx]+=h; xm[idx]-=h
        loss=lambda z:float(np.sum(z*w/np.sqrt(np.mean(z*z,axis=-1,keepdims=True)+1e-6)*g))
        fd=(loss(xp)-loss(xm))/(2*h)
        e3.append(dict(dtype=dt,case=case,jacobian_max_abs=float(np.max(np.abs(dx-dreg))),
                       finite_difference=fd,analytic=float(dx[idx]),relative_error=abs(fd-dx[idx])/abs(dx[idx])))
# All census forward paths, faithful IEEE emulation of nn.py with fp32 reduction and dtype storage.
# All normalization axes have power-of-two sizes except GroupNorm's 18. GroupNorm computes in fp32.
e3forward=[]
for cls in T.E3_CLASSES:
    for dt in T.E3_DTYPES:
        for case in T.E3_CASES:
            shape=T.E3_SHAPE[cls]; x=quant(T.e3_inputs(cls,case),dt)
            C,bshape=T._affine_shape(cls,shape)
            w=quant(1+.1*np.arange(C),dt); b=None if cls=='RMSNorm' else quant(.05*np.arange(C)-.1,dt)
            g=quant(T._e3_pattern(shape,1.1),dt)
            ref,refg=T.e3_reference(cls,x,w,b,g)
            work='float32' if cls=='GroupNorm' else dt
            q=lambda a:quant(a,work)
            xs=x.reshape(shape[0],2,-1) if cls=='GroupNorm' else x
            axes=(-1,) if cls in ('RMSNorm','GroupNorm','LayerNorm') else T._e3_axes(cls,shape)
            count=math.prod(xs.shape[a] for a in axes)
            # Model exact sum then stored result: fp32 tree error bound separately below.
            mean=lambda a:q(q(np.sum(a,axis=axes,keepdims=True))/count)
            centered=xs if cls=='RMSNorm' else q(xs-mean(xs))
            sq=q(centered*centered)
            stat=mean(sq); v=q(stat+T.E3_EPS[cls]); inv=q(v**-.5)
            normalized=q(centered*inv).reshape(shape)
            y=q(normalized*w.reshape(bshape))
            if b is not None: y=q(y+b.reshape(bshape))
            y=quant(y,dt); err=np.max(np.abs(y-ref)); scale=np.max(np.abs(ref))
            tol=T.e3_tolerance(cls,'forward',dt,scale)
            e3forward.append(dict(cls=cls,dtype=dt,case=case,max_abs_error=float(err),tolerance=float(tol),
                ratio=float(err/tol),worst_index=list(np.unravel_index(np.argmax(np.abs(y-ref)),shape)),
                reference_scale=float(scale),stat_min=float(stat.min()),stat_max=float(stat.max())))
# E2 honest bound at stored inputs: all p and p-1 exactly f32; no cancellation in 2 products.
u32=2.**-24; epow=4*2.**-23
e32=(1+epow)*(1+u32)**2-1
honest_e2={dt:dict(relative_bound=(1+e32)*(1+T.U[dt])-1 if dt!='float32' else e32,
                 registered=T.E2_RTOL[dt],absolute_underflow=T.E2_ATOL[dt]) for dt in T.U}
# Round-order witness, exact IEEE32 operations.
x=np.float32(T.E2_WITNESS_X); g=np.float32(T.E2_WITNESS_G); p=np.float32(3)
a=np.float32(g*np.float32(p*np.float32(x*x)))
b=np.float32(np.float32(g*p)*np.float32(x*x))
result=dict(power=rows,rms_jacobian=e3,e3_honest_forward_emulation=e3forward,honest_e2=honest_e2,
            e5_relative_bound=(1-u32*10)**-1-1,
            bit_order={'registered':hex(int(a.view(np.uint32))),'reordered':hex(int(b.view(np.uint32)))})
def serial(v):
    if isinstance(v,np.integer): return int(v)
    raise TypeError(type(v).__name__)
def strict_json(v):
    if isinstance(v,dict): return {k:strict_json(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)): return [strict_json(x) for x in v]
    if isinstance(v,float) and not math.isfinite(v): return str(v)
    if isinstance(v,np.integer): return int(v)
    return v
(rev/'numerics.json').write_text(json.dumps(strict_json(result),indent=2,allow_nan=False)+'\n')
print('E2 finite difference max',max(r['finite_difference_max_rel'] for r in rows))
print('RMS Jacobian checks',e3)
print('E3 honest emulation exceeds tolerance',[r for r in e3forward if r['ratio']>1])
print('E2 bound',honest_e2,'order bits',result['bit_order'])
