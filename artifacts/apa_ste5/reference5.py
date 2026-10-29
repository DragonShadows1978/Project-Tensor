"""Float64 full APA VJP on decoded BF16 operands; no gradient/statistic oracle input.

Prior art: Bengio, Leonard and Courville (2013), STE identity derivative;
Dao et al. (2022), attention softmax VJP; Higham (2002), gamma_n dot error.
Unverified - lead to check those titles. Reuses STE4/REV1 formulas, with row
blocking to bound memory. Ours: full unmasked STE5 signal and saved-threshold
whole-key certificate. No new kernel algorithm.
"""
import numpy as np
import common as c


def oracle(words, name, z=c.Z, saved_thr=None):
    B,H,L,S,D,VD,KH = c.SHAPES[name]
    q,k,kq,v,do = (c.decode(words[x]) for x in c.INPUTS)
    alpha = D**-.5
    on = np.zeros((B,KH,S,D),dtype=np.float64)
    off = np.zeros_like(on)
    extra = np.zeros_like(on)
    outputs = {'out':np.zeros((B,H,L,VD)), 'lse':np.zeros((B,H,L)),
               'thr':np.zeros((B,H,L)), 'dq':np.zeros((B,H,L,D)),
               'dv':np.zeros((B,KH,S,VD))}
    selected_any = np.zeros((B,KH,S),bool)
    possible_selected = np.zeros_like(selected_any)
    count = 0; visible_count = 0; all_selected = True
    if saved_thr is not None:
        saved_thr = c.decode(saved_thr)
    u = 2.**-24
    gamma = (2*D+2)*u/(1-(2*D+2)*u)
    for b in range(B):
        for h in range(H):
            kh = h//(H//KH)
            for start in range(0,L,64):
                stop = min(start+64,L)
                Q,DO = q[b,h,start:stop],do[b,h,start:stop]
                n = np.arange(start,stop)+S-L+1 if name!='small1' else np.full(stop-start,S)
                vis = np.arange(S)[None,:] < n[:,None]
                bulk = alpha*(Q@kq[b,kh].T)
                exact = alpha*(Q@k[b,kh].T)
                ab = np.where(vis,np.abs(bulk),0.)
                mean = ab.sum(axis=1)/n
                var = np.maximum((ab*ab).sum(axis=1)/n-mean*mean,0.)
                thr = mean+float(z)*np.sqrt(var)
                selected = vis & (ab>=thr[:,None])
                score = np.where(vis,np.where(selected,exact,bulk),-np.inf)
                mx = score.max(axis=1,keepdims=True)
                lse = mx+np.log(np.exp(score-mx).sum(axis=1,keepdims=True))
                prob = np.exp(score-lse)
                t = DO@v[b,kh].T
                rd = (prob*t).sum(axis=1,keepdims=True)
                coeff = alpha*prob*(t-rd)
                on[b,kh] += coeff.T@Q
                off[b,kh] += (coeff*selected).T@Q
                extra[b,kh] += (coeff*~selected).T@Q
                outputs['out'][b,h,start:stop] = prob@v[b,kh]
                outputs['lse'][b,h,start:stop] = lse[:,0]
                outputs['thr'][b,h,start:stop] = thr
                outputs['dq'][b,h,start:stop] = (coeff*selected)@k[b,kh]+(coeff*~selected)@kq[b,kh]
                outputs['dv'][b,kh] += prob.T@DO
                selected_any[b,kh] |= selected.any(axis=0)
                if saved_thr is not None:
                    # Upper dot/scale envelope compared with the ACTUAL saved
                    # FP32 threshold: certifies no query could select this key.
                    mag = np.abs(Q)@np.abs(kq[b,kh]).T
                    eb = (gamma*abs(float(np.float32(alpha)))+abs(float(np.float32(alpha))-alpha))*mag
                    eb += 2.**-120*(D+2)
                    possible_selected[b,kh] |= (vis & (np.abs(bulk)+eb>=saved_thr[b,h,start:stop,None])).any(axis=0)
                count += int(selected.sum()); visible_count += int(vis.sum())
                all_selected &= bool(np.all(selected|~vis))
    for x in (on,off,*outputs.values()):
        c.healthy(x)
    if z==c.Z:
        c.healthy(extra)
    else:
        assert np.isfinite(extra).all()
    return dict(on=on,off=off,extra=extra,selected_any=selected_any,
                certified_unselected=~possible_selected if saved_thr is not None else None,
                stats={'visible_pairs':visible_count,'selected_pairs':count,
                       'all_visible_selected':all_selected,'reference':'ideal float64 forward and VJP; no captured statistics used for G-G5'}, **outputs)


def ulp(x):
    # IEEE 754-2019 BF16 normal spacing; subnormal spacing is 2^-133.
    return np.exp2(np.maximum(np.floor(np.log2(np.maximum(np.abs(x),2.**-126)))-7,-133))


def relative_l2(actual,reference):
    c.healthy(reference)
    assert actual.shape==reference.shape and np.isfinite(actual).all()
    return float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))


def signal_check(on,off,ron,roff,extra):
    """Lead's frozen G-G5: no centering, fitting intercept, zero mask or exclusions."""
    for x in (on,off,ron,roff,extra):
        c.healthy(x)
    assert on.shape==off.shape==ron.shape==roff.shape==extra.shape
    delta=(on-off).ravel(); ref=extra.ravel()
    dn=float(np.linalg.norm(delta)); rn=float(np.linalg.norm(ref))
    dot=float(delta@ref)
    cosine=dot/(dn*rn) if dn>0 and rn>0 else 0.
    slope=dot/(rn*rn) if rn>0 else 0.
    on_error=relative_l2(on,ron); off_error=relative_l2(off,roff)
    values={'cosine':cosine,'slope':slope,'on_relative_L2':on_error,
            'off_relative_L2':off_error,'on_minus_off_relative_L2':on_error-off_error,
            'delta_norm':dn,'reference_delta_norm':rn,'elements':int(delta.size)}
    values['PASS']=bool(dn>0 and rn>0 and cosine>=.995 and .98<=slope<=1.02 and on_error<=off_error+.01)
    return values


def error_percentiles(actual,reference):
    e=np.abs(actual-reference)/ulp(reference)
    return {'unit':'BF16 ULP at each float64 reference element (subnormal floor 2^-133)',
            'elements':int(e.size),'relative_L2':relative_l2(actual,reference),
            'maxabs':float(np.max(np.abs(actual-reference))),
            'percentiles':{str(p):float(np.percentile(e,p)) for p in (0,50,90,95,99,99.9,100)}}
