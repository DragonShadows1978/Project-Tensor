"""Float64 APA reference; selection/threshold and quantized keys are detached.

Prior art: Bengio, Leonard & Courville 2013 STE (identity key derivative),
Dao et al. 2022 FlashAttention softmax VJP (recomputation identity).
Unverified - lead to check those papers. Ours: explicit NumPy experiment oracle.
"""
import numpy as np


def forward(q, k, kq, v, scale, zthr, causal, frozen=None):
    q, k, kq, v = [np.asarray(x, dtype=np.float64) for x in (q, k, kq, v)]
    B, H, L, D = q.shape
    S, VD = k.shape[2], v.shape[3]
    out = np.empty((B, H, L, VD)); lse = np.empty((B, H, L)); thr = np.empty_like(lse)
    selected = np.zeros((B, H, L, S), dtype=bool)
    probs = np.zeros((B, H, L, S))
    for b in range(B):
        for h in range(H):
            kh = h // (H // k.shape[1])
            for i in range(L):
                n = S-L+i+1 if causal else S
                bulk = (kq[b, kh, :n] @ q[b, h, i]) * scale
                a = np.abs(bulk)
                t = a.mean() + zthr * np.sqrt(max(0., np.mean(a*a)-a.mean()**2))
                sel = a >= t if frozen is None else frozen[b, h, i, :n]
                score = np.where(sel, (k[b, kh, :n] @ q[b, h, i])*scale, bulk)
                m = score.max(); p = np.exp(score-m); den = p.sum(); p /= den
                out[b, h, i] = p @ v[b, kh, :n]
                lse[b, h, i] = m + np.log(den); thr[b, h, i] = t
                selected[b, h, i, :n] = sel; probs[b, h, i, :n] = p
    return out, lse, thr, selected, probs


def backward(q, k, kq, v, do, scale, state, ste=False, output_dot=False):
    q, k, kq, v, do = [np.asarray(x, dtype=np.float64) for x in (q, k, kq, v, do)]
    out, lse, thr, selected, probs = state
    dq = np.zeros_like(q); dk = np.zeros_like(k); dv = np.zeros_like(v)
    B,H,L,D = q.shape; S = k.shape[2]
    for b in range(B):
        for h in range(H):
            kh = h // (H // k.shape[1])
            for i in range(L):
                p = probs[b,h,i]; sel = selected[b,h,i]
                dov = v[b,kh] @ do[b,h,i]
                rd = do[b,h,i] @ out[b,h,i] if output_dot else p @ dov
                ds = p*(dov-rd)*scale
                keys = np.where(sel[:,None], k[b,kh], kq[b,kh])
                dq[b,h,i] = ds @ keys
                dk[b,kh] += (ds if ste else ds*sel)[:,None]*q[b,h,i]
                dv[b,kh] += p[:,None]*do[b,h,i]
    return dq, dk, dv


def derivative_forward(q, kvar, k0, kq, v, scale, state, causal):
    """Same primal at kvar=k0; frozen choices, kq + (kvar-k0) derivative.

    This is the STE stop_gradient(kq-k)+k construction (Bengio et al. 2013).
    q still differentiates against the original mixed key, as in the engine.
    Use only key perturbations for this surrogate's derivative.
    """
    return forward(q, kvar, kq+(kvar-k0), v, scale, 0., causal, frozen=state[3])[0]


def fixture(shape=(1,2,7,11,16,8,2), seed=412):
    B,H,L,S,D,VD,KVH = shape
    rng = np.random.default_rng(seed)
    q=rng.normal(size=(B,H,L,D))*.4
    k=rng.normal(size=(B,KVH,S,D))*.4
    # Prior art: uniform scalar quantization, Gray & Neuhoff (1998) survey;
    # unverified - lead to check. Fixture only, not GRAPA's learned quantizer.
    kq=np.round(k*3)/3
    # Real mixed-score semantics, with an entirely unselected, visible key.
    kq[:,:,1,:]=0.
    v=rng.normal(size=(B,KVH,S,VD))*.4
    do=rng.normal(size=(B,H,L,VD))*.4
    return q,k,kq,v,do
