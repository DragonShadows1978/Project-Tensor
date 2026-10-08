"""APA-MF1 fp64 numpy references, written from the mathematical definition.

Convention (the only causal definition used here): query row i of L queries over
S keys is the absolute position (S - L) + i, so it may attend key j iff
j <= (S - L) + i  (BOTTOM-RIGHT alignment; S == L reduces to j <= i).
Nothing in this module calls tensor_cuda, tensor_gpu_v2, or apa_cuda code, and
no mask here is built with triu(k=1) on a rectangle (forbidden configuration).

Prior art:
- Bottom-right causal alignment for rectangular (S > L) blocks: FlashAttention-2
  (Dao, 2023) `causal=True` with seqlen_q != seqlen_k is bottom-right aligned
  (taken: the convention). Project-Tensor 63960a4 (2026-06-09) applied the same
  convention to tensor_cuda.functional._causal_mask (taken: k = 1 + S - L).
- Visible-count population statistics for the refine threshold: the fused kernel
  tensor_cuda/src/kernels.cu (bk4 forward, `count = causal ? S-L+qi+tid+1 : S`,
  `th = mean + zthr*sqrt(max(sq/count - mean^2, 0))`) and GRAPA attention.py
  `_visible_counts` (taken: the rule; the fp64 two-pass form here is mine).
- z-score selection |bulk| >= mean + z*std: APA paper draft docs/APA_PAPER_DRAFT.md
  section 2.1 (Project-Tensor). Nothing else known to me.
"""
from __future__ import annotations

import numpy as np


def visible_mask(L: int, S: int) -> np.ndarray:
    """(L, S) bool, True where row i may attend column j (j <= S - L + i)."""
    rows = np.arange(L, dtype=np.int64)[:, None]
    cols = np.arange(S, dtype=np.int64)[None, :]
    return cols <= (S - L) + rows


def _softmax_visible(scores: np.ndarray, vis: np.ndarray) -> np.ndarray:
    s = np.where(vis, scores, -np.inf)
    m = s.max(-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.where(vis, np.exp(s - m), 0.0)
    return e / e.sum(-1, keepdims=True)


def exact_attention(q, k, v, scale, causal=True):
    """fp64 exact attention. q (..., L, D), k/v (..., S, D). Returns (out, P)."""
    q = np.asarray(q, np.float64); k = np.asarray(k, np.float64); v = np.asarray(v, np.float64)
    L, S = q.shape[-2], k.shape[-2]
    vis = visible_mask(L, S) if causal else np.ones((L, S), bool)
    s = np.einsum("...ld,...sd->...ls", q, k) * scale
    p = _softmax_visible(s, vis)
    return np.einsum("...ls,...sd->...ld", p, v), p


def exact_attention_grads(q, k, v, dO, scale, causal=True):
    """fp64 analytic gradients of exact attention: (dq, dk, dv)."""
    q = np.asarray(q, np.float64); k = np.asarray(k, np.float64)
    v = np.asarray(v, np.float64); dO = np.asarray(dO, np.float64)
    L, S = q.shape[-2], k.shape[-2]
    vis = visible_mask(L, S) if causal else np.ones((L, S), bool)
    _, p = exact_attention(q, k, v, scale, causal)
    dv = np.einsum("...ls,...ld->...sd", p, dO)
    dp = np.einsum("...ld,...sd->...ls", dO, v)
    ds = p * (dp - (p * dp).sum(-1, keepdims=True)) * scale
    ds = np.where(vis, ds, 0.0)
    dq = np.einsum("...ls,...sd->...ld", ds, k)
    dk = np.einsum("...ls,...ld->...sd", ds, q)
    return dq, dk, dv


def fused_rule_threshold(absb, vis, z):
    """Fused-kernel rule: population mean/std of |bulk| over the VISIBLE keys only.

    absb (..., L, S) magnitudes, vis (L, S) bool. Returns thr (..., L, 1) fp64.
    """
    a = np.where(vis, np.asarray(absb, np.float64), 0.0)
    cnt = vis.sum(-1, keepdims=True).astype(np.float64)
    mean = a.sum(-1, keepdims=True) / cnt
    dev = np.where(vis, a - mean, 0.0)
    var = (dev * dev).sum(-1, keepdims=True) / cnt
    return mean + z * np.sqrt(var)


def fused_rule_select(absb, vis, z):
    """(sel, thr): sel = visible & |bulk| >= thr (fused-kernel rule)."""
    thr = fused_rule_threshold(absb, vis, z)
    return vis & (np.asarray(absb, np.float64) >= thr), thr


def diluted_rule_threshold(absb, vis, z):
    """Pre-fix-B composed rule: masked entries set to 0 and COUNTED over all S
    (population mean/std over S columns). Alignment is whatever `vis` says."""
    a = np.where(vis, np.asarray(absb, np.float64), 0.0)
    return a.mean(-1, keepdims=True) + z * a.std(-1, keepdims=True)


def diluted_rule_select(absb, vis, z):
    a = np.where(vis, np.asarray(absb, np.float64), 0.0)
    thr = diluted_rule_threshold(absb, vis, z)
    return a >= thr, thr


def ambiguous(absb, thr, rel=1e-5):
    """Entries within a relative band of the threshold: fp ties (either side legal)."""
    absb = np.asarray(absb, np.float64)
    return np.abs(absb - thr) <= rel * np.maximum(np.abs(thr), 1e-30)


def apa_tc_reference(q, k, kq, v, scale, sel, causal=True):
    """tensor_cuda quant.py APA semantics given a selection: scores take the exact
    q.k where selected, the bulk q.kq elsewhere; softmax over visible keys."""
    q = np.asarray(q, np.float64); k = np.asarray(k, np.float64)
    kq = np.asarray(kq, np.float64); v = np.asarray(v, np.float64)
    L, S = q.shape[-2], k.shape[-2]
    vis = visible_mask(L, S) if causal else np.ones((L, S), bool)
    bulk = np.einsum("...ld,...sd->...ls", q, kq) * scale
    rank = np.einsum("...ld,...sd->...ls", q, k) * scale
    scores = np.where(sel, rank, bulk)
    p = _softmax_visible(scores, vis)
    return np.einsum("...ls,...sd->...ld", p, v)


def tgv2_apa_select(abs_rank, vis, S, refine_percentile, z):
    """tensor_gpu_v2 APA selection semantics (selects on |ranking|, masked -> 0,
    stats over all S columns): z-score branch for S > 256, exact top-k otherwise.
    Returns (sel, thr)."""
    a = np.where(vis, np.asarray(abs_rank, np.float64), 0.0)
    if S > 256:
        thr = a.mean(-1, keepdims=True) + z * a.std(-1, keepdims=True)
    else:
        refine_k = max(1, int(S * refine_percentile))
        idx = S - refine_k
        thr = np.partition(a, idx, axis=-1)[..., idx:idx + 1]
    return a >= thr, thr


def tgv2_apa_backward(q, k, kq, v, dO, scale, sel, causal=True):
    """tensor_gpu_v2 APA documented backward: gradients from the full-precision
    softmax; grad_q splits dS by the refine selection (exact keys where refined,
    quantized keys elsewhere). Returns (dq, dk, dv)."""
    q = np.asarray(q, np.float64); k = np.asarray(k, np.float64)
    kq = np.asarray(kq, np.float64); v = np.asarray(v, np.float64)
    dO = np.asarray(dO, np.float64)
    L, S = q.shape[-2], k.shape[-2]
    vis = visible_mask(L, S) if causal else np.ones((L, S), bool)
    _, p = exact_attention(q, k, v, scale, causal)
    dv = np.einsum("...ls,...ld->...sd", p, dO)
    dp = np.einsum("...ld,...sd->...ls", dO, v)
    ds = p * (dp - (p * dp).sum(-1, keepdims=True)) * scale
    ds = np.where(vis, ds, 0.0)
    dq = (np.einsum("...ls,...sd->...ld", np.where(sel, ds, 0.0), k)
          + np.einsum("...ls,...sd->...ld", np.where(sel, 0.0, ds), kq))
    dk = np.einsum("...ls,...ld->...sd", ds, q)
    return dq, dk, dv


def rel_err(got, ref):
    """(relative max-abs error, max-abs error): max|got-ref| / max|ref|."""
    got = np.asarray(got, np.float64); ref = np.asarray(ref, np.float64)
    mae = float(np.max(np.abs(got - ref))) if got.size else 0.0
    den = float(np.max(np.abs(ref))) if ref.size else 0.0
    return (mae / den if den > 0 else mae), mae


def bf16_bits_to_f32(u16: np.ndarray) -> np.ndarray:
    """bf16 raw bits (uint16) -> exact fp32 values."""
    return (np.asarray(u16, np.uint16).astype(np.uint32) << 16).view(np.float32)
