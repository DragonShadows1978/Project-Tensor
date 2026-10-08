"""Independent fp64 bottom-right attention and analytical VJP.

Prior art: scaled dot-product attention (Vaswani et al., 2017), stable softmax
max subtraction (standard numerical practice; no original author known to me),
softmax Jacobian, and FlashAttention-2 v2.1 bottom-right convention (Dao 2023).
Taken: these formulas. Ours: explicit operand enumeration for this review.
"""
import numpy as np


def visible(L, S, causal=True):
    return (np.arange(S)[None, :] <= np.arange(L)[:, None] + S - L
            if causal else np.ones((L, S), bool))


def attention(q, k, v, scale, causal=True, additive=None, upstream=None):
    q, k, v = (np.asarray(x, np.float64) for x in (q, k, v))
    L, S = q.shape[-2], k.shape[-2]
    scores = (q @ k.swapaxes(-1, -2)) * scale
    vis = visible(L, S, causal)
    scores = np.where(vis, scores, -np.inf)
    if additive is not None:
        scores = scores + np.asarray(additive, np.float64)
    if not np.isfinite(scores.max(-1)).all():
        raise ValueError("each row needs a finite visible score")
    w = np.exp(scores - scores.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    out = w @ v
    if upstream is None:
        return out, w
    g = np.asarray(upstream, np.float64)
    dw = g @ v.swapaxes(-1, -2)
    ds = w * (dw - (w * dw).sum(-1, keepdims=True))
    dq = (ds @ k) * scale
    dk = (ds.swapaxes(-1, -2) @ q) * scale
    dv = w.swapaxes(-1, -2) @ g
    return out, w, (dq, dk, dv)


def selection(bulk, causal, z):
    a = np.abs(np.asarray(bulk, np.float64))
    vis = visible(a.shape[-2], a.shape[-1], causal)
    cnt = vis.sum(-1, keepdims=True)
    if np.any(cnt == 0):
        raise ValueError("empty visible row")
    a = np.where(vis, a, 0)
    mean = a.sum(-1, keepdims=True) / cnt
    var = np.maximum((a * a).sum(-1, keepdims=True) / cnt - mean * mean, 0)
    th = mean + z * np.sqrt(var)
    return vis & (a >= th), th
