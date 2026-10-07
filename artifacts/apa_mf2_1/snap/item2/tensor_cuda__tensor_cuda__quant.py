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
    # cast back to the key dtype (fp16/bf16) so it matches the query in matmuls
    if out_dtype in ("float16", "bfloat16"):
        return recon.astype(out_dtype)
    return recon


_SEL_CACHE = {}


def _bk4_rows(L, S, dev, is_causal):
    """Cached (count (L,1), visible 0/1 (L,S) or None) fp32 tensors for the BK4 rule."""
    key = (L, S, dev, bool(is_causal))
    t = _SEL_CACHE.get(key)
    if t is None:
        if len(_SEL_CACHE) >= 16:
            _SEL_CACHE.clear()
        if is_causal:
            # bottom-right: row i sees j <= S-L+i, count S-L+i+1 (kernels.cu:8064);
            # max(., 1) only keeps out-of-contract S < L rows finite
            cnt = np.maximum(np.arange(S - L + 1, S + 1, dtype=np.float32), 1.0)
            vis = (np.arange(S)[None, :] <= (S - L + np.arange(L))[:, None]).astype(np.float32)
            t = (tc.tensor(cnt[:, None], device=dev), tc.tensor(vis, device=dev))
        else:
            t = (tc.tensor(np.full((L, 1), S, np.float32), device=dev), None)
        _SEL_CACHE[key] = t
    return t


def _bk4_selection(qf, kqf, scale, z, is_causal, bulk32=None):
    """APA-MF2: the fused BK4 refine selection, composed from engine ops.

    Rule (kernels.cu bk4::forward :8064-8077 and :8112-8114), transcribed literally:
      b    = fl32(q . kq * scale)   the UNROUNDED fp32 dot (B1; never the bf16/fp16-stored bulk)
      a    = |b| over VISIBLE keys only; count = S-L+i+1 (causal) or S
      mean = sum(a) / count; var = max(sum(a^2) / count - mean^2, 0)   population
             one-pass moments with the kernel's clamp, in BOTH branches (B2), IEEE div
      th   = mean + z * sqrt(var)                                      IEEE sqrt
      sel  = visible && a >= th                                        (B3; >=)
    qf/kqf: fp32 (B,H,L,D)/(B,H,S,D), no grad. bulk32: optional precomputed fp32 b.
    Returns the fp32 0/1 selection (B,H,L,S).

    Not bit-exact by construction (APA-MF2 probe P0, logs/apa_mf2_p0_probe_*.log):
    BK4's dot is a bf16 WMMA accumulation (bk3::scores), this one is cuBLAS SGEMM
    (R1); BK4 sums sequentially with FFMA squares and fuses var/th with FFMA (R2/R3).
    The residual is an ulp-band tie at the threshold, measured in the MF2 receipts.

    Prior art: kernels.cu BK4 (Project-Tensor 2026), the rule matched; GRAPA
    attention.py _visible_counts (visible-count statistics); FlashAttention-2
    (Dao 2023) / 63960a4 bottom-right visibility. Engine-op composition: mine.
    """
    from . import functional as F
    B, H, L, _ = qf.shape
    S = kqf.shape[-2]
    dev = qf.device.split(":")[0]
    if bulk32 is None:
        bulk32 = tc.matmul(qf, kqf, alpha=scale, trans_b=True)
    cnt, vis = _bk4_rows(L, S, dev, is_causal)
    absb = bulk32.abs()
    if is_causal:
        # zero the invisible entries so the row sums are visible-only; the shared
        # bottom-right mask (fp32, nonzero == masked) is the predicate
        absb = absb.masked_fill(F._causal_mask(L, S, dev, "float32"), 0.0)
    mean = absb.sum([-1], True) / cnt
    var = ((absb * absb).sum([-1], True) / cnt - mean * mean).relu()
    thr = mean + var.sqrt() * z
    sel = absb.ge(thr)
    if vis is not None:
        sel = sel * vis                      # visible && (0/1 product, never NaN)
    return sel


def apa_quant_attention(query, key, value, *, bulk_bits=2, refine_percentile=0.15,
                        is_causal=False, scale=None, apa_rotation=True):
    """APA-Quant attention on the engine. query/key/value: (B, H, L, D).

    Uses the z-score refinement path (mean + z*std of |full scores|), matching
    the reference's S>256 branch. Backward is exact autograd of this forward.
    """
    from . import functional as F
    B, H, L, D = query.shape
    S = key.shape[-2]
    if is_causal:
        F._check_causal_lengths(L, S)    # APA-MF2.1: S < L raises before any matmul
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    dev = query.device.split(":")[0]
    R, CB, BND = _tables(D, bulk_bits, H, apa_rotation, dev)

    key_quant = _quantize_keys(key, R, CB, BND)
    bulk = tc.matmul(query, key_quant, alpha=scale, trans_b=True)
    ranking = tc.matmul(query, key, alpha=scale, trans_b=True)
    bulk_raw = bulk                      # unmasked; the fp32 rule's operand in fp32 mode

    causal_add = None
    if is_causal:
        # APA-MF2 item 2: the shared bottom-right mask (0 visible, -inf masked) as
        # a PREDICATE: masked positions become -inf whatever their score, so a
        # large (or overflowed) masked score can never leak weight. Prior art:
        # FlashAttention-2 (Dao 2023) -inf predicate masking.
        causal_add = F._causal_mask(L, S, dev, query.dtype)
        bulk = bulk.masked_fill(causal_add, float("-inf"))
        ranking = ranking.masked_fill(causal_add, float("-inf"))

    if refine_percentile >= 1.0:
        scores = ranking
    else:
        # Selection criterion must match the fused kernel, which thresholds on
        # |bulk| (the cheap quantized scores) and refines the keys it picks — you
        # cannot legitimately peek at the expensive |ranking| to decide whether to
        # compute it. APA-MF2: the rule is the fused BK4 kernel's, transcribed in
        # _bk4_selection (MF1 fix A's bottom-right stats mask and fix B's visible
        # counts are kept; B1 fp32 unrounded dot, B2 one-pass moments in the
        # non-causal branch too, B3 the visible predicate are added). In bf16/fp16
        # the stored bulk is rounded, so the fp32 dot is recomputed from the exact
        # fp32 upcasts of the SAME operands (query, key_quant) — the kernel's own.
        # The 0/1 mask is cast back to the operand dtype: tc.where dispatches on
        # the operand dtype and reads the condition buffer as that dtype.
        z = _norm_ppf(1.0 - max(0.0, min(1.0, refine_percentile)))
        if query.dtype == "float32":
            mask = _bk4_selection(query, key_quant, scale, z, is_causal,
                                  bulk32=bulk_raw.detach())
        else:
            mask = _bk4_selection(query.detach().float(), key_quant.float(), scale, z,
                                  is_causal)
        if bulk.dtype != "float32":
            mask = mask.astype(bulk.dtype)   # match tc.where's operand dtype
        scores = tc.where(mask, ranking, bulk)

    weights = scores.softmax(-1)
    return tc.matmul(weights, value)
