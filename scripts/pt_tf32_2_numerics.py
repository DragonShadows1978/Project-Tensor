"""CPU arithmetic and registered gate computation, no native imports.

Prior art: NVIDIA TF32 (2020); Ootomo/Yokota (2022), arXiv:2203.03341
residual decomposition; Dao et al. (2022)/Dao (2023) softmax/VJP; NumPy
(Harris et al. 2020). Taken formulas; ours: bounded dense diagnostic ablations.
Higham (2002) unit roundoff is taken; independent RMS event propagation is
an explicit engineering assumption, not a proof. Cosine calibration is the
user's 2026 rule; no prior art known to me for that exact rule.
"""
import math
import numpy as np
import pt_tf32_1 as old


def product(a, b, mode='fp64'):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if mode == 'fp64':
        return a @ b
    ah, bh = old.tf32_round(a).astype(np.float64), old.tf32_round(b).astype(np.float64)
    if mode == 'tf32':
        return ah @ bh
    if mode != 'tf32x3':
        raise ValueError('unknown product')
    # Match native split of *FP32* storage, not an imaginary FP64 residual.
    al = old.tf32_round(a.astype(np.float32).astype(np.float64)-ah).astype(np.float64)
    bl = old.tf32_round(b.astype(np.float32).astype(np.float64)-bh).astype(np.float64)
    return ah @ bh + (ah @ bl + al @ bh)


def forward_reference(x, s):
    """Independent FP64 stable softmax; centered population variance.

    The singleton identity avoids threshold roundoff changing the only pair.
    This is the mathematical APA rule, not a native reduction emulation.
    """
    B,H,KH,L,S,D,VD=[s[n] for n in ('B','H','KVH','L','S','D','VD')]
    xx={k:np.asarray(v,np.float64) for k,v in x.items()}
    out=np.empty((B,H,L,VD));lse=np.empty((B,H,L));thr=np.empty_like(lse)
    selection=np.zeros((B,H,L,S),bool)
    scale,zthr=float(np.float32(s['scale'])),float(np.float32(s['zthr']))
    for b in range(B):
        for h in range(H):
            kh=h//(H//KH)
            for lo in range(0,L,128):
                hi=min(L,lo+128);q=xx['q'][b,h,lo:hi]
                bulk=q@xx['kq'][b,kh].T*scale
                visible=np.arange(S)[None,:] <= S-L+np.arange(lo,hi)[:,None] if s['causal'] else np.ones(bulk.shape,bool)
                count=visible.sum(1);ab=np.abs(bulk)
                mean=np.where(visible,ab,0).sum(1)/count
                variance=np.where(visible,(ab-mean[:,None])**2,0).sum(1)/count
                threshold=mean+zthr*np.sqrt(variance)
                sel=visible & ((ab>=threshold[:,None]) | (count[:,None]==1))
                score=np.where(visible,np.where(sel,q@xx['k'][b,kh].T*scale,bulk),-np.inf)
                m=score.max(1);p=np.exp(score-m[:,None]);den=p.sum(1)
                out[b,h,lo:hi]=(p@xx['v'][b,kh])/den[:,None]
                lse[b,h,lo:hi]=m+np.log(den);thr[b,h,lo:hi]=threshold;selection[b,h,lo:hi]=sel
    return dict(out=out,lse=lse,thr=thr,selection=selection)


def backward_reference(x, state, s, *, score_mode='fp64', dov_mode='fp64',
                       outer_mode='fp64', selection_mode=None, singleton=True):
    """Dense independent VJP on the exact supplied API state (no renormalizing).

    Different product modes are CPU diagnosis only, not substitute GPU gates.
    Native selected scalar scores retain full FP32 operands, modeled in FP64.
    """
    B,H,KH,L,S,D,VD=[s[n] for n in ('B','H','KVH','L','S','D','VD')]
    x={k:np.asarray(v,np.float64) for k,v in x.items()};scale=float(np.float32(s['scale']))
    result={n:np.zeros_like(x[k]) for n,k in zip(old.NAMES,('q','k','v'))}
    selection=np.zeros((B,H,L,S),bool)
    for b in range(B):
        for h in range(H):
            kh=h//(H//KH);k=x['k'][b,kh];kq=x['kq'][b,kh];v=x['v'][b,kh]
            for lo in range(0,L,128):
                hi=min(L,lo+128);q=x['q'][b,h,lo:hi];do=x['dO'][b,h,lo:hi]
                bulk=product(q,kq.T,score_mode)*scale
                predicate=bulk if selection_mode is None else product(q,kq.T,selection_mode)*scale
                visible=np.arange(S)[None,:] <= S-L+np.arange(lo,hi)[:,None] if s['causal'] else np.ones(bulk.shape,bool)
                one=(visible.sum(1)==1)[:,None] if singleton else np.zeros((hi-lo,1),bool)
                sel=visible & ((abs(predicate)>=state['thr'][b,h,lo:hi,None]) | one)
                selection[b,h,lo:hi]=sel
                score=np.where(sel,q@k.T*scale,bulk)
                p=np.exp(np.where(visible,score-state['lse'][b,h,lo:hi,None],-np.inf))
                p=np.where(one,visible.astype(np.float64),p)
                rd=(do*state['out'][b,h,lo:hi]).sum(1)
                ds=p*(product(do,v.T,dov_mode)-rd[:,None])*scale
                ds=np.where(one,0.,ds)
                chosen=np.where(sel,ds,0.)
                result['dQ'][b,h,lo:hi]=product(chosen,k,outer_mode)+product(np.where(sel,0.,ds),kq,outer_mode)
                result['dK'][b,kh]+=product(chosen.T,q,outer_mode)
                result['dV'][b,kh]+=product(p.T,do,outer_mode)
    result['selection']=selection
    return result


def backward_model(x, state, s):
    return backward_reference(x,state,s,score_mode='tf32x3',dov_mode='tf32x3',outer_mode='tf32')


def bounds(reg, s, name):
    model=reg['rounding_model']
    expected=math.sqrt(model['operand_rounding_events'][name])*model['scalar_rms']
    expected+=math.sqrt(max(s[k] for k in ('L','S','D','VD')))*model['fp32_unit_roundoff']
    return dict(expected=expected,bound=3*expected)


def metric_gate(candidate, reference, bound):
    a,r=np.asarray(candidate,np.float64),np.asarray(reference,np.float64)
    if a.shape!=r.shape or a.size==0:
        raise ValueError('empty or mismatched numerical gate')
    if not (np.isfinite(a).all() and np.isfinite(r).all() and math.isfinite(bound) and bound>=0):
        return dict(verdict='RED',finite=False)
    err=a-r;rn=float(np.linalg.norm(r));peak=float(np.max(abs(r)))
    l2=float(np.linalg.norm(err));max_abs=float(np.max(abs(err)))
    # No epsilon hides zero-reference failure, and JSON never contains Infinity.
    if rn==0 or peak==0:
        return dict(verdict='GREEN' if l2==0 else 'RED',finite=True,zero_reference=True,
                    max_abs=max_abs,relative_L2=0. if l2==0 else None,normalized_max_abs=0. if l2==0 else None,bound=bound)
    relative=l2/rn;maximum=max_abs/peak
    return dict(verdict='GREEN' if max(relative,maximum)<=bound else 'RED',finite=True,
                relative_L2=relative,max_abs=max_abs,normalized_max_abs=maximum,bound=bound)


def cosine(a,b):
    if not a or a.keys()!=b.keys():raise ValueError('gradient keys differ or empty')
    dot=na=nb=0.
    for k in sorted(a):
        x,y=np.asarray(a[k],np.float64),np.asarray(b[k],np.float64)
        if x.shape!=y.shape or not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError('gradient shape or finite check failed')
        dot+=float(np.sum(x*y));na+=float(np.sum(x*x));nb+=float(np.sum(y*y))
    if not all(map(math.isfinite,(dot,na,nb))) or min(na,nb)<=0:
        raise ValueError('zero/nonfinite gradient norm')
    value=dot/math.sqrt(na)/math.sqrt(nb)
    if abs(value)>1+8*np.finfo(np.float64).eps:raise ValueError('invalid cosine')
    return min(1.,max(-1.,value))


def noise_floor(gradients):
    if len(gradients)!=4:raise ValueError('exactly four calibration gradients required')
    pairs=[]
    for i in range(4):
        for j in range(i+1,4):
            c=cosine(gradients[i],gradients[j]);pairs.append(dict(i=i,j=j,cos=c,distance=1-c))
    spread=max(p['distance'] for p in pairs)
    if spread>=1/3:raise ValueError('noise floor would make unchanged bar nonpositive')
    return dict(pairs=pairs,observed_spread=spread,bar=1-3*spread)
