"""APA-MF2.1: the fp64 DEFINITION of apa_quant_attention's refine selection (the oracle).

Written from the definition in orders/APA_MF2_1_REVIEW_FIXES.md (Measures 3), never
from the code under test, BK4, MF2's one-pass emulator or the reviewer's helpers:

  b      = (q . kq) * scale            exact fp64 dot of the call's OWN operands
  a      = |b|
  vis    = bottom-right causal (row i of L sees j <= S-L+i) or all S keys
  cnt    = #visible keys of the row
  mean   = sum_vis(a) / cnt
  var    = sum_vis((a - mean)^2) / cnt    centered TWO-PASS, population
  thr    = mean + z * sqrt(var)
  sel    = vis && a >= thr

Tie band (registration.json definition_oracle.tie_band): an entry is a tie iff
| a - thr | <= 2e-5 * |thr| (relative, no absolute floor). fp64 two-pass error is
~1e-15 relative -- eight orders below the band.

Prior art: centered two-pass variance is the textbook stable form; Chan, Golub &
LeVeque (1983, "Algorithms for computing the sample variance: analysis and
recommendations", Am. Statistician 37(3)) analyse the one-pass E[x^2]-E[x]^2
cancellation that this replaces; Welford (1962, Technometrics 4(3)) is the
single-pass stable alternative (not used). Unverified -- lead to check
("Chan Golub LeVeque 1983 variance", "Welford 1962 corrected sums of squares").
Bottom-right visibility: FlashAttention-2 v2.1 (Dao 2023) / Project-Tensor 63960a4.
Oracle code and band bookkeeping: mine.
"""
from __future__ import annotations

import numpy as np

BAND = 2e-5


def visible(L, S, causal=True):
    """(L, S) bool: row i sees j <= S-L+i (bottom-right) or every key."""
    if not causal:
        return np.ones((L, S), bool)
    return np.arange(S)[None, :] <= (S - L + np.arange(L))[:, None]


def definition_fp64(b64, causal, z):
    """b64 (..., L, S) fp64 bulk dots (scale applied). Returns sel, thr (..., L, 1), a, vis."""
    b64 = np.asarray(b64, np.float64)
    L, S = b64.shape[-2:]
    vis = visible(L, S, causal)
    if not vis.any(-1).all():
        raise ValueError("definition needs >= 1 visible key per row (S >= L)")
    a = np.abs(b64)
    cnt = vis.sum(-1, keepdims=True).astype(np.float64)
    av = np.where(vis, a, 0.0)
    mean = av.sum(-1, keepdims=True) / cnt
    d = np.where(vis, a - mean, 0.0)
    var = (d * d).sum(-1, keepdims=True) / cnt
    thr = mean + float(z) * np.sqrt(var)
    sel = vis & (a >= thr)
    return sel, thr, a, vis


def bulk_fp64(q, kq, scale):
    """Exact-upcast fp64 dot: q (..., L, D), kq (..., S, D)."""
    q64 = np.asarray(q, np.float64)
    k64 = np.asarray(kq, np.float64)
    return np.einsum("...ld,...sd->...ls", q64, k64) * float(scale)


def compare(sel, ref_sel, a, thr, vis, band=BAND):
    """Count mismatches of `sel` vs the oracle, inside/outside the tie band.

    Returns a dict with raw / in-band / out-of-band counts, max relative gap of the
    mismatches, and the visible counts of rows holding out-of-band mismatches."""
    sel = np.broadcast_to(np.asarray(sel, bool), ref_sel.shape)
    mm = sel != ref_sel
    thr_b = np.broadcast_to(thr, a.shape)
    gap = np.abs(a - thr_b) / np.where(np.abs(thr_b) > 0, np.abs(thr_b), np.inf)
    gap = np.where(np.abs(a - thr_b) == 0, 0.0, gap)
    tie = np.abs(a - thr_b) <= band * np.abs(thr_b)
    out = mm & ~tie
    cnt = np.broadcast_to(vis.sum(-1, keepdims=True), a.shape)
    res = {"n_entries": int(mm.size), "n_visible": int(np.broadcast_to(vis, a.shape).sum()),
           "mismatch": int(mm.sum()), "in_band": int((mm & tie).sum()), "out_of_band": int(out.sum()),
           "max_rel_gap_mismatch": float(gap[mm].max()) if mm.any() else 0.0,
           "invisible_selected": int((sel & ~np.broadcast_to(vis, a.shape)).sum())}
    if out.any():
        c = cnt[out]
        res["out_of_band_row_visible_counts"] = {"le2": int((c <= 2).sum()), "gt2": int((c > 2).sum()),
                                                 "min": int(c.min()), "max": int(c.max())}
        res["max_rel_gap_out_of_band"] = float(gap[out].max())
    return res
