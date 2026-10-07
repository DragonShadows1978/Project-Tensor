"""Independent review oracle; no author helpers.

Prior art: Chan, Golub & LeVeque (1983), centered two-pass population
variance; Dao (2023), FlashAttention-2 bottom-right causal geometry and
stable softmax. Mine: this independent NumPy implementation and operands.
"""
import math
from statistics import NormalDist
import numpy as np

Z = NormalDist().inv_cdf(0.85)
BAND = 2e-5


def visible(L, S, causal=True):
    return (np.arange(S)[None, :] <= S - L + np.arange(L)[:, None]
            if causal else np.ones((L, S), bool))


def definition(bulk, causal, z=Z):
    a = np.abs(np.asarray(bulk, np.float64))
    vis = visible(*a.shape[-2:], causal)
    count = vis.sum(-1, keepdims=True)
    if np.any(count == 0):
        raise ValueError("definition requires a visible key per row")
    mean = np.where(vis, a, 0).sum(-1, keepdims=True) / count
    delta = np.where(vis, a - mean, 0)
    var = (delta * delta).sum(-1, keepdims=True) / count
    threshold = mean + z * np.sqrt(var)
    return (a >= threshold) & vis, threshold, mean, var


def attention(q, k, v, causal, scale=None):
    q, k, v = [np.asarray(x, np.float64) for x in (q, k, v)]
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    scores = q @ k.swapaxes(-2, -1) * scale
    if causal:
        scores = np.where(visible(*scores.shape[-2:]), scores, -np.inf)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    return weights @ v, weights


def mismatch(actual, bulk, causal):
    expected, threshold, _, _ = definition(bulk, causal)
    diff = np.asarray(actual, bool) != expected
    vis = visible(*expected.shape[-2:], causal)
    outside = np.abs(np.abs(np.asarray(bulk, np.float64)) - threshold) > BAND * np.abs(threshold)
    return {"mismatch": int(diff.sum()), "out_of_band": int((diff & outside & vis).sum()),
            "invisible_selected": int((np.asarray(actual, bool) & ~vis).sum())}


def encode(x):
    """JSON special values stay strings rather than nonstandard JSON NaN literals."""
    if isinstance(x, np.ndarray):
        return encode(x.tolist())
    if isinstance(x, (list, tuple)):
        return [encode(v) for v in x]
    if isinstance(x, dict):
        return {k: encode(v) for k, v in x.items()}
    if isinstance(x, (np.floating, float)) and not math.isfinite(float(x)):
        return "NaN" if math.isnan(float(x)) else ("+inf" if x > 0 else "-inf")
    if isinstance(x, np.generic):
        return x.item()
    return x
