"""TurboQuant tables + APA-Quant attention, on the standalone CUDA engine.

This is the original goal realized with no PyTorch dependency: APA attention
running on our own autograd/CUDA stack. The Lloyd-Max codebook and per-head
rotations are one-time CPU work (cached); the rest composes from engine ops, so
autograd produces gradients automatically (the quantized key path is detached,
giving the same grad-routing as the reference: full key on refined positions,
quantized key on the bulk).
"""

from __future__ import annotations

import math

import numpy as np

import tensor_cuda as tc

_CODEBOOK_CACHE = {}
_ROT_CACHE = {}
_TABLE_TENSORS = {}


def _beta_pdf(dim, grid):
    exponent = 0.5 * (dim - 3)
    log_coeff = (math.lgamma(dim / 2) - 0.5 * math.log(math.pi) - math.lgamma((dim - 1) / 2))
    return np.exp(log_coeff) * np.maximum(0.0, 1.0 - grid * grid) ** exponent


def build_codebook(dim, bits, grid_size=16385, max_iter=200, tol=1e-8):
    key = (dim, bits)
    if key in _CODEBOOK_CACHE:
        return _CODEBOOK_CACHE[key].copy()
    nc = 1 << bits
    grid = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    w = _beta_pdf(dim, grid) * (grid[1] - grid[0])
    w[0] *= 0.5; w[-1] *= 0.5
    q = (np.arange(nc) + 0.5) / nc
    cdf = np.cumsum(w)
    centroids = np.interp(np.clip(q, 0, 1) * cdf[-1], cdf, grid)
    centroids = np.clip(np.sort(centroids), -1, 1)
    for _ in range(max_iter):
        old = centroids.copy()
        bnd = 0.5 * (centroids[:-1] + centroids[1:])
        lab = np.searchsorted(bnd, grid, side="left")
        for k in range(nc):
            m = lab == k
            ws = w[m].sum()
            if ws > 0:
                centroids[k] = float((grid[m] * w[m]).sum() / ws)
        centroids = np.clip(np.sort(centroids), -1, 1)
        if np.max(np.abs(centroids - old)) < tol:
            break
    _CODEBOOK_CACHE[key] = centroids.astype(np.float32)
    return _CODEBOOK_CACHE[key].copy()


def _rotation(dim, seed):
    key = (dim, seed)
    if key in _ROT_CACHE:
        return _ROT_CACHE[key].copy()
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    s = np.sign(np.diag(r)); s[s == 0] = 1.0
    _ROT_CACHE[key] = (q * s).astype(np.float32)
    return _ROT_CACHE[key].copy()


def _tables(D, bits, H, apa_rotation, device):
    key = (D, bits, H, apa_rotation, device)
    if key in _TABLE_TENSORS:
        return _TABLE_TENSORS[key]
    cb = build_codebook(D, bits)
    bnd = (0.5 * (cb[:-1] + cb[1:])).astype(np.float32)
    rot = np.stack([_rotation(D, (h * 1337 + 42) if apa_rotation else 0) for h in range(H)])
    R = tc.tensor(rot, device=device)
    CB = tc.tensor(cb, device=device)
    BND = tc.tensor(bnd, device=device)
    _TABLE_TENSORS[key] = (R, CB, BND)
    return _TABLE_TENSORS[key]


def _norm_ppf(p):
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    try:
        from scipy.stats import norm
        return float(norm.ppf(p))
    except ImportError:
        # Acklam approximation
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
        plow, phigh = 0.02425, 1 - 0.02425
        if p < plow:
            q = math.sqrt(-2 * math.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        if p > phigh:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        q = p - 0.5; r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _quantize_keys(k, R, CB, BND):
    """k (B,H,S,D) -> dequantized keys (detached constant).

    The rotation/quantization tables are fp32; the rotation is done in fp32 for
    numerical stability regardless of the key dtype, then cast back to the key
    dtype so the result is consistent with the rest of the attention math (real
    models run fp16 keys, which would otherwise mismatch the fp32 tables).
    """
    out_dtype = k.dtype
    kd = k.detach().float()
    B, H, S, D = kd.shape
    norms = (kd * kd).sum([-1], True).pow(0.5)
    unit = kd / (norms + 1e-12)
    Rb = R.float().reshape([1, H, D, D]).expand([B, H, D, D])
    rotated = tc.matmul(unit, Rb.transpose(-2, -1))
    centroids = tc._C.apa_quantize_gather(rotated, BND.float(), CB.float())
    recon = tc.matmul(centroids, Rb) * norms
    recon = recon.detach()
    return recon.half() if out_dtype == "float16" else recon


def apa_quant_attention(query, key, value, *, bulk_bits=2, refine_percentile=0.15,
                        is_causal=False, scale=None, apa_rotation=True):
    """APA-Quant attention on the engine. query/key/value: (B, H, L, D).

    Uses the z-score refinement path (mean + z*std of |full scores|), matching
    the reference's S>256 branch. Backward is exact autograd of this forward.
    """
    from . import functional as F
    B, H, L, D = query.shape
    S = key.shape[-2]
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    dev = query.device.split(":")[0]
    R, CB, BND = _tables(D, bulk_bits, H, apa_rotation, dev)

    key_quant = _quantize_keys(key, R, CB, BND)
    bulk = tc.matmul(query, key_quant.transpose(-2, -1)) * scale
    ranking = tc.matmul(query, key.transpose(-2, -1)) * scale

    causal_add = None
    if is_causal:
        causal_add = F._causal_mask(L, S, dev, query.dtype)
        bulk = bulk + causal_add
        ranking = ranking + causal_add

    if refine_percentile >= 1.0:
        scores = ranking
    else:
        absr = ranking.abs()
        if is_causal:
            cmask = np.triu(np.ones((L, S), np.float32), 1)
            absr = absr.masked_fill(tc.tensor(cmask, device=dev), 0.0)
        z = _norm_ppf(1.0 - max(0.0, min(1.0, refine_percentile)))
        thr = absr.mean([-1], True) + absr.std([-1], True) * z
        mask = absr.ge(thr)            # detached 0/1
        scores = tc.where(mask, ranking, bulk)

    weights = scores.softmax(-1)
    return tc.matmul(weights, value)
