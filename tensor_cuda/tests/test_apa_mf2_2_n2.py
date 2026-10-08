"""APA-MF2.2 item 2 (re-review N2): apa_quant_attention's selection statistics must match the
fp64 DEFINITION (registered relative tie band 2e-5, not widened) across the whole finite fp32
range of |bulk|, 1e-38 (subnormal) .. 3e38.

MF2.1 squared the centered values in fp32 (quant.py:203 "var = (dev_ * dev_).sum([-1], True) /
cnt"): |bulk| ~1e20 overflows (threshold +inf, nothing selected) and ~1e-30 underflows (variance
0) -- lead GPU receipt tensor_fork_apa_wt_mf21rev
logs/apa_mf21rev_gpu_definition_20261007T223911Z_248602.log (N2/large, N2/tiny RED, D64/96/128).
Fix: the moments are taken of u = |b| / M, M = the visible row max (all-zero rows: M := 1), and
the selection compares u >= thr_u in the scaled domain.

Oracle: scripts/apa_mf2_2_oracle.py definition64 (fp64 statistics of the call's exact fp32 bulk;
mine). Witnesses: review N2 (apa-mf21rev docs/APA_MF2_1_REVIEW.md:36-51); the range sweep,
mixed-magnitude, zero and near-constant rows: mine.
Prior art: scaled norms -- Blue (1978, "A portable Fortran program to find the Euclidean norm
of a vector", ACM TOMS 4(1)) and the reference BLAS snrm2/dnrm2 running-scale form (scale by
the largest magnitude before squaring); centered two-pass variance, Chan, Golub & LeVeque
(1983). Unverified -- lead to check ("Blue 1978 Euclidean norm", "BLAS dnrm2 scale ssq",
"Chan Golub LeVeque 1983 sample variance").
"""
import contextlib
import os
import sys

import numpy as np
import pytest

import tensor_cuda as tc
from tensor_cuda import quant as Q

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import apa_mf2_2_oracle as O  # noqa: E402

Z = Q._norm_ppf(1.0 - 0.15)
SCALES = (1e-38, 1e-37, 1e-30, 1e-20, 1e-10, 1e-3, 1.0, 1e3, 1e10, 1e20, 1e30, 1e37, 3e38)
PATTERNS = ("uniform", "lognormal", "review123", "constant", "nearconst", "spike")
SIZES = (1, 2, 3, 256, 2048)


def _base(pattern, L, S, rng):
    """(L, S) fp64 magnitudes in (0, 1] with row max 1, random signs applied later."""
    j = np.arange(S)
    if pattern == "uniform":
        b = rng.uniform(1e-3, 1.0, (L, S))
    elif pattern == "lognormal":
        b = np.exp(rng.normal(0, 2, (L, S)))
    elif pattern == "review123":
        b = np.broadcast_to((1 + j % 3) / 3.0, (L, S)).copy()
    elif pattern == "constant":
        b = np.ones((L, S))
    elif pattern == "nearconst":
        b = np.broadcast_to(1.0 - (j % 64) * 2.0 ** -20, (L, S)).copy()
    else:
        b = np.full((L, S), 1e-3); b[:, rng.integers(0, S, L)] = 1.0
    return b / b.max(-1, keepdims=True)


def _select(b32, causal):
    L, S = b32.shape[-2:]
    H = b32.shape[1]
    qf = tc.tensor(np.zeros((1, H, L, 8), np.float32))
    kf = tc.tensor(np.zeros((1, H, S, 8), np.float32))
    with tc.no_grad():
        return Q._definition_selection(qf, kf, 1.0, Z, causal, bulk32=tc.tensor(b32)).numpy()


def _check(b32, causal, label):
    sel = _select(b32, causal)
    r = O.mismatch(sel, b32, causal, Z)
    assert r["out_of_band"] == 0 and r["invisible_selected"] == 0, f"{label}: {r}"
    return sel, r


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("scale", SCALES)
def test_range_sweep_matches_fp64_definition(scale, pattern):
    for S in SIZES:
        for causal in (False, True):
            L = min(4, S) if causal else 4
            rng = np.random.default_rng(S + 7 * L)
            mag = _base(pattern, L, S, rng) * scale
            sign = np.where(rng.random((L, S)) < 0.5, -1.0, 1.0)
            b32 = (sign * mag).astype(np.float32)[None, None]
            sel, r = _check(b32, causal, f"scale={scale:g} {pattern} S={S} causal={causal}")
            if pattern == "constant":
                # zero-variance rows: every visible key sits ON the threshold -> all selected
                assert np.array_equal(sel[0, 0] != 0, O.visible(L, S, causal)), (scale, S, causal)


@pytest.mark.parametrize("D", [64, 96, 128])
@pytest.mark.parametrize("mag", [1e20, 1e-30])
def test_review_n2_witnesses(mag, D):
    q = np.zeros((1, 1, 1, D), np.float32); q[..., 0] = mag
    k = np.zeros((1, 1, 3, D), np.float32); k[0, 0, :, 0] = [1, 2, 3]
    qt, kt = tc.tensor(q), tc.tensor(k)
    with tc.no_grad():
        b32 = tc.matmul(qt, kt, alpha=1.0, trans_b=True)
        sel = Q._definition_selection(qt, kt, 1.0, Z, False, bulk32=b32).numpy()
    assert np.array_equal(sel[0, 0, 0] != 0, [False, False, True]), sel
    r = O.mismatch(sel, b32.numpy(), False, Z)
    assert r["out_of_band"] == 0, r


def test_mixed_magnitude_rows_and_heads_are_scaled_per_row():
    rng = np.random.default_rng(5)
    S = 256
    mags = np.array([[1e30, 1e-30], [3e38, 1e-38]])           # (H=2, L=2) row maxima
    b = rng.uniform(0.01, 1.0, (2, 2, S)) * mags[..., None]
    b32 = b.astype(np.float32)[None]
    for causal in (False, True):
        _check(b32, causal, f"mixed causal={causal}")


@pytest.mark.parametrize("S", [1, 3, 256])
def test_all_zero_rows_select_every_visible_key(S):
    for causal in (False, True):
        L = min(3, S) if causal else 3
        sel = _select(np.zeros((1, 1, L, S), np.float32), causal)
        assert np.array_equal(sel[0, 0] != 0, O.visible(L, S, causal)), sel


@contextlib.contextmanager
def _capture():
    rec = {}
    old = Q._definition_selection

    def spy(qf, kqf, scale, z, is_causal, bulk32=None):
        if bulk32 is None:
            bulk32 = tc.matmul(qf, kqf, alpha=scale, trans_b=True)
        rec["bulk32"] = bulk32.numpy()
        rec["sel"] = old(qf, kqf, scale, z, is_causal, bulk32=bulk32)
        return rec["sel"]

    Q._definition_selection = spy
    try:
        yield rec
    finally:
        Q._definition_selection = old


@pytest.mark.parametrize("dt", ["float32", "bfloat16"])
@pytest.mark.parametrize("mag", [1e-30, 1e-20, 1e-10, 1.0, 1e10, 1e20, 1e30])
def test_full_api_range(mag, dt):
    for S in (3, 256, 2048):
        for causal in (False, True):
            rng = np.random.default_rng(S + int(causal))
            q = (rng.normal(0, 1, (1, 1, 3, 64)) * mag).astype(np.float32)
            k = rng.normal(0, 1, (1, 1, S, 64)).astype(np.float32)
            v = rng.normal(0, 1, (1, 1, S, 64)).astype(np.float32)
            xs = [tc.tensor(x, dtype=dt) for x in (q, k, v)]
            with _capture() as rec, tc.no_grad():
                out = Q.apa_quant_attention(*xs, bulk_bits=4, refine_percentile=0.15,
                                            is_causal=causal, scale=1.0)
            r = O.mismatch(rec["sel"].numpy(), rec["bulk32"], causal, Z)
            assert r["out_of_band"] == 0 and r["invisible_selected"] == 0, (mag, dt, S, causal, r)
            assert np.isfinite(out.float().numpy()).all(), (mag, dt, S, causal)


@pytest.mark.parametrize("spacing", [2.0 ** -17, 2.0 ** -18])
@pytest.mark.parametrize("scale", [1e-38, 1e-30, 1.0, 1e20, 3e38])
def test_near_tie_spread_centered_moments(scale, spacing):
    """Added after the registered mutation run (P5, one-pass variance in the scaled domain,
    SURVIVED: the 'nearconst' spread 64 * 2^-20 sits inside the 2e-5 band). Rows 1 - (j mod 256)
    * spacing have std ~1e-4..1e-3 of the row max, so one-pass E[u^2] - E[u]^2 cancellation
    (~1e-7 absolute) moves the threshold beyond the band; the centered form does not. Mine."""
    for S in (256, 2048):
        j = np.arange(S)
        b32 = np.broadcast_to((1.0 - (j % 256) * spacing) * scale, (4, S)).astype(np.float32)[None, None]
        for causal in (False, True):
            _check(b32, causal, f"near-tie spread scale={scale:g} spacing={spacing:g} S={S} causal={causal}")
