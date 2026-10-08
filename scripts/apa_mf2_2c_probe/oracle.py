"""Independent fp64 oracle; never imports candidate or author helpers.

Prior art: max-shift softmax and fully masked zero convention, Dao (2023),
FlashAttention-2; centered population variance, Chan/Golub/LeVeque (1983).
Taken: mathematical definitions. Mine: NumPy oracle, analytic VJP and census.
"""
import numpy as np


def visible(L, S, causal):
    if causal and S < L:
        raise ValueError("S < L")
    return (np.arange(S)[None, :] <= S - L + np.arange(L)[:, None]
            if causal else np.ones((L, S), bool))


def weights(scores):
    s = np.asarray(scores, np.float64)
    blocked = np.all(np.isneginf(s), axis=-1, keepdims=True)
    safe = np.where(blocked, 0., s)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        ex = np.exp(safe - safe.max(-1, keepdims=True))
        p = ex / ex.sum(-1, keepdims=True)
    return np.where(blocked, 0., p)


def attention(q, k, v, scale, causal=False, mask=None, upstream=None):
    q, k, v = (np.asarray(x, np.float64) for x in (q, k, v))
    scores = q @ k.swapaxes(-1, -2) * scale
    if causal:
        scores = np.where(visible(q.shape[-2], k.shape[-2], True), scores, -np.inf)
    if mask is not None:
        scores = scores + mask
    p = weights(scores)
    out = p @ v
    if upstream is None:
        return out
    g = np.asarray(upstream, np.float64)
    dw = g @ v.swapaxes(-1, -2)
    ds = p * (dw - (dw * p).sum(-1, keepdims=True))
    return out, (ds @ k * scale, ds.swapaxes(-1, -2) @ q * scale,
                 p.swapaxes(-1, -2) @ g)


def definition(bulk, z, causal):
    a = np.abs(np.asarray(bulk, np.float64))
    vis = visible(a.shape[-2], a.shape[-1], causal)
    count = vis.sum(-1, keepdims=True)
    av = np.where(vis, a, 0.)
    mean = av.sum(-1, keepdims=True) / count
    var = np.where(vis, (av - mean) ** 2, 0.).sum(-1, keepdims=True) / count
    threshold = mean + z * np.sqrt(var)
    return vis & (a >= threshold), threshold, vis


def selection_errors(actual, bulk, z, causal):
    expected, threshold, vis = definition(bulk, z, causal)
    a = np.abs(np.asarray(bulk, np.float64))
    rowmax = np.where(vis, a, 0.).max(-1, keepdims=True)
    band = 2e-5 * rowmax  # frozen author/order tie band, never adjusted after running
    bad = (np.asarray(actual) != 0) != expected
    return {"mismatch": int(bad.sum()),
            "out_of_band": int((bad & vis & (np.abs(a-threshold) > band)).sum()),
            "invisible_selected": int(((np.asarray(actual) != 0) & ~vis).sum())}
