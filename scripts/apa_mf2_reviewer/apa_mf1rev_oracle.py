"""Independent fp64 review oracle; never imports author reference helpers.
Prior art: 63960a4 (2026-06-09); Dao FlashAttention-2 (2023), v2.1 README
bottom-right convention. GRAPA attention.py _visible_counts: population count.
DeMillo/Lipton/Sayward (1978): mutation adequacy. Own contribution: explicit
enumeration, independent differentiations and per-site counterexamples.
"""
import numpy as np
from statistics import NormalDist


def visibility(L, S, start=0, end=None):
    # Direct position enumeration, deliberately independent of triangular APIs.
    if end is None:
        end = L
    return np.array([[j <= S - L + i for j in range(S)]
                     for i in range(start, end)], dtype=bool)


def population_select(bulk, L, S, p=.15, causal=True):
    # Prior art: BK4 count / E[x²]-E[x]²; GRAPA visible-count population rule.
    a = np.abs(np.asarray(bulk, dtype=np.float64))
    vis = visibility(L, S) if causal else np.ones((L, S), dtype=bool)
    a = np.where(vis, a, 0.)
    n = vis.sum(-1, keepdims=True).astype(np.float64)
    mu = a.sum(-1, keepdims=True) / n
    var = np.maximum((a*a).sum(-1, keepdims=True) / n - mu*mu, 0.)
    threshold = mu + NormalDist().inv_cdf(1.-p) * np.sqrt(var)
    return vis & (a >= threshold), threshold


def legacy_select(ranking, L, S, p=.15):
    # Independent spec transcription of unchanged tensor_gpu_v2 selection policy.
    # Prior art: existing tensor_gpu_v2 ranking-based z-score/top-k selection;
    # no claim this is the fused |bulk| rule. Mean/std includes masked zeros.
    a = np.where(visibility(L, S), np.abs(np.asarray(ranking, np.float64)), 0.)
    if S > 256:
        mu = a.mean(-1, keepdims=True)
        sd = np.sqrt(((a-mu)**2).mean(-1, keepdims=True))
        th = mu + NormalDist().inv_cdf(1.-p)*sd
    else:
        n = max(1, int(S*p))
        th = np.sort(a, axis=-1)[..., S-n:S-n+1]
    return a >= th, th


def softmax(scores, vis):
    # Prior art: stable softmax max subtraction (standard numerical practice;
    # no specific origin known to me); only visible columns enter normalization.
    x = np.where(vis, np.asarray(scores, np.float64), -np.inf)
    e = np.exp(x-x.max(-1, keepdims=True))
    return e/e.sum(-1, keepdims=True)


def attention(q, k, v, scale, kq=None, selection=None, grad=None,
              approximate_backward=False, causal=True):
    # Prior art: standard scaled dot-product attention (Vaswani et al., 2017)
    # and reverse-mode chain rule; legacy approximate backward read from core.
    q,k,v = (np.asarray(a, np.float64) for a in (q,k,v))
    L,S = q.shape[-2],k.shape[-2]
    vis = visibility(L,S) if causal else np.ones((L,S),bool)
    rank = q @ k.swapaxes(-1,-2)*scale
    if kq is None:
        scores = rank
    else:
        kq = np.asarray(kq,np.float64)
        scores = np.where(selection, rank, q @ kq.swapaxes(-1,-2)*scale)
    w = softmax(scores,vis)
    out = w @ v
    if grad is None:
        return out
    if approximate_backward:
        w = softmax(rank,vis)
    g = np.asarray(grad,np.float64)
    dv = w.swapaxes(-1,-2) @ g
    da = g @ v.swapaxes(-1,-2)
    ds = w*(da-(w*da).sum(-1,keepdims=True))*scale
    if kq is None:
        dq = ds @ k
    else:
        dq = np.where(selection,ds,0.) @ k + np.where(selection,0.,ds) @ kq
    # Legacy approx backward sends every score gradient to full K. Composed
    # exact-autograd quant only sends selected ds to K; not evaluated here.
    dk = ds.swapaxes(-1,-2) @ q
    return out,(dq,dk,dv)


def round_bf16(a):
    # Prior art: IEEE round-to-nearest-even; bfloat16 truncates 16 fraction bits.
    # Our CPU emulation used only for exact finite counterexample values.
    f = np.asarray(a,np.float32)
    u = f.view(np.uint32)
    rounded = (u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)
