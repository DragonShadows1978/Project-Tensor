"""APA-MF2.2 fp64 oracles (mine, written from the definitions; never from the code under test).

1. definition64(b, causal, z): the refine-selection DEFINITION on the call's fp32 bulk b
   a = |b|; visible = bottom-right (row i of L sees j <= S-L+i) or all; cnt = #visible
   mean = sum_vis(a)/cnt; var = sum_vis((a-mean)^2)/cnt (centered two-pass, population)
   thr = mean + z*sqrt(var); sel = visible && a >= thr.
   fp64 on exact upcasts of fp32 values: no overflow/underflow for any finite fp32 input
   (|a| <= 3.4e38, a^2 <= 1.2e77 << 1.8e308; smallest fp32 subnormal 1.4e-45 squares to 2e-90 >
   fp64 min normal 2.2e-308).
2. mismatch(): registered tie band | a - thr64 | <= 2e-5 * |thr64| (MF2.1, unchanged, not widened).
3. softmax64(scores, v): fp64 softmax over the FINITE scores of a row (the -inf entries carry
   weight exactly 0 by definition of softmax with exp(-inf) = 0).

Prior art: centered two-pass population variance (textbook); Chan, Golub & LeVeque (1983,
"Algorithms for computing the sample variance: analysis and recommendations", Am. Statistician
37(3)) for the stable centered form -- unverified, lead to check. Stable softmax with max
subtraction: Milakov & Gimelshein (2018, "Online normalizer calculation for softmax") and
FlashAttention-2 (Dao 2023) -- unverified, lead to check. Differs from scripts/apa_mf2_1_defn.py:
input is the call's fp32 bulk (the definition's b), not an fp64 GEMM. Code: mine.
"""
from __future__ import annotations

import numpy as np

BAND = 2e-5


def visible(L, S, causal):
    if not causal:
        return np.ones((L, S), bool)
    return np.arange(S)[None, :] <= (S - L + np.arange(L))[:, None]


def definition64(b, causal, z):
    """b (..., L, S) fp32 bulk values. Returns sel64, thr64 (..., L, 1), a64, vis."""
    a = np.abs(np.asarray(b, np.float32).astype(np.float64))
    L, S = a.shape[-2:]
    vis = visible(L, S, causal)
    cnt = vis.sum(-1, keepdims=True).astype(np.float64)
    mean = np.where(vis, a, 0.0).sum(-1, keepdims=True) / cnt
    d = np.where(vis, a - mean, 0.0)
    var = (d * d).sum(-1, keepdims=True) / cnt
    thr = mean + float(z) * np.sqrt(var)
    return vis & (a >= thr), thr, a, vis


def mismatch(actual, b, causal, z):
    sel, thr, a, vis = definition64(b, causal, z)
    act = np.asarray(actual) != 0
    diff = (act != sel) & vis
    out = diff & (np.abs(a - thr) > BAND * np.abs(thr))
    rel = np.abs(a - thr) / np.where(thr != 0, np.abs(thr), 1.0)
    return {"entries_visible": int(np.broadcast_to(vis, a.shape).sum()), "mismatch": int(diff.sum()),
            "in_band": int((diff & ~out).sum()), "out_of_band": int(out.sum()),
            "invisible_selected": int((act & ~vis).sum()),
            "max_rel_gap_mismatch": float(rel[diff].max()) if diff.any() else 0.0,
            "selected": int((act & vis).sum()), "selected64": int(sel.sum())}


def softmax64(scores, v):
    """scores (..., S) fp64 with -inf allowed (at least one finite per row); v (..., S, D)."""
    s = np.asarray(scores, np.float64)
    m = np.max(np.where(np.isfinite(s), s, -np.inf), -1, keepdims=True)
    w = np.where(np.isfinite(s), np.exp(s - m), 0.0)
    w /= w.sum(-1, keepdims=True)
    return w @ np.asarray(v, np.float64)          # (..., L, S) @ (..., S, D)
