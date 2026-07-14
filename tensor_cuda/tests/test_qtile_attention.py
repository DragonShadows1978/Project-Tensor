"""EXP-APA-3: Q-tiled K/V-reuse attention skeleton parity gates.

The Q-tile fp16 path (TC_ATTN_QTILE default-on, fp16 D<=64) must match the
EXP-APA-2 streaming skeleton (TC_ATTN_QTILE=0 — same source, untouched) within
reduction-order slack on BOTH consumers (fused non-causal SDPA and fused APA).
fp32 and D>64 must keep routing to the streaming kernels bit-identically.
Conventions deliberately mirror tests/test_apa_int4_sdpa_noncausal.py.
"""

from __future__ import annotations

import math
import os
from statistics import NormalDist

import numpy as np
import pytest

import tensor_cuda as tc


def _require_cuda():
    try:
        t = tc.tensor(np.zeros(1, dtype=np.float32))
        tc.synchronize()
        del t
    except Exception as exc:  # pragma: no cover - host-only runs
        pytest.skip(f"CUDA unavailable: {exc}")


def _case_arrays(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return tuple(
        (rng.standard_normal(shape).astype(np.float32) * np.float32(0.125)).astype(dtype)
        for _ in range(3)
    )


def _np_sdpa_fp32(q, k, v):
    scale = np.float32(1.0 / math.sqrt(q.shape[-1]))
    qf, kf, vf = (t.astype(np.float32) for t in (q, k, v))
    s = qf @ kf.transpose(0, 1, 3, 2) * scale
    w = np.exp(s - s.max(-1, keepdims=True))
    w = w / w.sum(-1, keepdims=True)
    return w @ vf


def _run_sdpa(q, k, v, *, qtile: bool):
    _require_cuda()
    dts = "float16" if q.dtype == np.float16 else "float32"
    os.environ["TC_ATTN_QTILE"] = "1" if qtile else "0"
    try:
        with tc.no_grad():
            out = tc.fused_sdpa_noncausal(
                tc.tensor(q, dtype=dts), tc.tensor(k, dtype=dts),
                tc.tensor(v, dtype=dts), 1.0 / math.sqrt(q.shape[-1]))
        tc.synchronize()
        return out.numpy()
    finally:
        os.environ.pop("TC_ATTN_QTILE", None)


def _run_apa(q, k, v, r, *, qtile: bool, thr: str | None = None):
    """thr: TC_APA_THR for the call (None = unset -> the v2 Welford default).
    Legacy streaming (qtile=False) is the frozen EXP-APA-2 ladder instrument
    regardless of thr; qtile-vs-legacy parity tests must pin thr='ladder16'."""
    _require_cuda()
    dts = "float16" if q.dtype == np.float16 else "float32"
    refine_all = r >= 1.0
    z = 0.0 if refine_all else NormalDist().inv_cdf(1.0 - r)
    os.environ["TC_ATTN_QTILE"] = "1" if qtile else "0"
    if thr is None:
        os.environ.pop("TC_APA_THR", None)
    else:
        os.environ["TC_APA_THR"] = thr
    try:
        with tc.no_grad():
            out = tc.apa_int4_sdpa_noncausal(
                tc.tensor(q, dtype=dts), tc.tensor(k, dtype=dts),
                tc.tensor(v, dtype=dts), 1.0 / math.sqrt(q.shape[-1]),
                z, refine_all=refine_all)
        tc.synchronize()
        return out.numpy()
    finally:
        os.environ.pop("TC_ATTN_QTILE", None)
        os.environ.pop("TC_APA_THR", None)


def _relfro(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# Tile geometry edges: exact 64-row tile, query tail, multi-tile K with tail,
# D below the 64-pad, D not a multiple of 16, odd D (SDPA-only dispatch).
_SDPA_SHAPES = [(1, 1, 64, 64), (1, 3, 33, 64), (2, 4, 257, 64),
                (1, 2, 100, 48), (1, 1, 31, 33)]
_APA_SHAPES = [(1, 3, 33, 64), (2, 4, 257, 64), (1, 2, 64, 64), (1, 1, 100, 32)]


@pytest.mark.parametrize("shape", _SDPA_SHAPES)
def test_qtile_sdpa_matches_legacy_fp16(shape):
    """Q-tile vs streaming skeleton: same fp32-weight SDPA math, order slack
    only (fp16 output-rounding class)."""
    q, k, v = _case_arrays(shape, np.float16, sum(shape))
    got = _run_sdpa(q, k, v, qtile=True).astype(np.float32)
    legacy = _run_sdpa(q, k, v, qtile=False).astype(np.float32)
    ref = _np_sdpa_fp32(q, k, v)
    err_legacy = float(np.max(np.abs(got - legacy)))
    err_ref = float(np.max(np.abs(got - ref)))
    print(f"QT SDPA {shape}: vs-legacy={err_legacy:.4e} vs-np={err_ref:.4e}")
    assert err_legacy <= 1e-3  # K-SDPA fp16 gate class
    assert err_ref <= 1e-3
    assert np.all(np.isfinite(got))


@pytest.mark.parametrize("shape", _APA_SHAPES)
def test_qtile_apa_matches_legacy_fp16(shape):
    """Q-tile APA (recomputed bulk walks) vs the EXP-APA-2 shared-cache
    kernel: identical semantics, reduction-order slack only (K-EQ class).
    EXP-APA-4: pinned to TC_APA_THR=ladder16 — the legacy streaming kernel
    is the frozen ladder instrument, so this parity holds on the preserved
    ladder path (the v2 Welford default is gated separately below)."""
    q, k, v = _case_arrays(shape, np.float16, sum(shape))
    got = _run_apa(q, k, v, 0.15, qtile=True, thr="ladder16")
    legacy = _run_apa(q, k, v, 0.15, qtile=False)
    rf = _relfro(got, legacy)
    print(f"QT APA {shape}: relfro vs legacy={rf:.4e}")
    assert rf <= 1e-3  # K-EQ gate class
    assert np.all(np.isfinite(got))


def test_qtile_apa_refine_all_matches_legacy_fp16():
    """r>=1 rides the SDPA-math instantiation (fp32 scores, no bulk pass)."""
    q, k, v = _case_arrays((1, 3, 33, 64), np.float16, 20260712)
    got = _run_apa(q, k, v, 1.0, qtile=True).astype(np.float32)
    legacy = _run_apa(q, k, v, 1.0, qtile=False).astype(np.float32)
    err = float(np.max(np.abs(got - legacy)))
    print(f"QT APA refine_all: vs-legacy max_abs={err:.4e}")
    assert err <= 1e-3
    assert np.all(np.isfinite(got))


def test_qtile_dispatch_leaves_fp32_on_streaming_kernels():
    """fp32 must be bit-identical under the toggle: the Q-tile path is fp16-
    only (HMMA cannot hold the registered fp32 1.3e-8 class)."""
    q, k, v = _case_arrays((1, 3, 33, 64), np.float32, 7)
    sd_on = _run_sdpa(q, k, v, qtile=True)
    sd_off = _run_sdpa(q, k, v, qtile=False)
    assert float(np.max(np.abs(sd_on - sd_off))) == 0.0
    apa_on = _run_apa(q, k, v, 0.15, qtile=True)
    apa_off = _run_apa(q, k, v, 0.15, qtile=False)
    assert float(np.max(np.abs(apa_on - apa_off))) == 0.0


def test_qtile_apa_long_keys_no_shared_cache_cap():
    """The streaming kernel caps Lk at 44KB of shared bulk cache; the Q-tile
    path has no Lk-dependent shared memory and must accept longer keys."""
    q, _, _ = _case_arrays((1, 1, 8, 64), np.float16, 3)
    _, k, v = _case_arrays((1, 1, 23000, 64), np.float16, 4)
    out = _run_apa(q, k, v, 0.15, qtile=True)
    assert out.shape == (1, 1, 8, 64)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------- EXP-APA-4
# K2 threshold semantics v2: fp32 Welford selection statistics (default),
# fp16 overflow ladder preserved behind TC_APA_THR=ladder16, TC_APA_FRAC
# realized-refine-fraction instrumentation.

def _np_quant_dq16(k):
    """Composed symmetric-7 INT4 dequant mirror (bits=4), fp16 dq values."""
    kf = k.astype(np.float32)
    amax = np.max(np.abs(kf), axis=-1, keepdims=True)
    scale = amax * np.float32(1.0 / 7.0)
    safe = np.where(scale > 0, scale, np.float32(1.0))
    codes = np.clip(np.round(kf * (np.float32(1.0) / safe)), -7.0, 7.0)
    return (codes * safe).astype(np.float16).astype(np.float32)


def _np_apa_welford_fp16(q, k, v, z):
    """np mirror of the v2 semantics for fp16 inputs: bulk16 values unchanged
    (f2h of fp32 scores), selection statistics in FLOAT64 (the reference the
    kernel's fp32 Welford is gated against), mix_scores blend, exact softmax."""
    scale = np.float32(1.0 / math.sqrt(q.shape[-1]))
    kdq = _np_quant_dq16(k)
    qf, kf, vf = (t.astype(np.float32) for t in (q, k, v))
    bulk = (qf @ kdq.transpose(0, 1, 3, 2) * scale).astype(np.float16).astype(np.float32)
    a = np.abs(bulk).astype(np.float64)
    mean = a.mean(-1, keepdims=True)
    var = ((a - mean) ** 2).mean(-1, keepdims=True)   # population, ddof=0
    thr = (mean + np.sqrt(var) * float(z)).astype(np.float64)
    exact = (qf @ kf.transpose(0, 1, 3, 2) * scale).astype(np.float16).astype(np.float32)
    blended = np.where(a >= thr, exact, bulk)
    w = np.exp(blended - blended.max(-1, keepdims=True))
    w = w / w.sum(-1, keepdims=True)
    return w @ vf


@pytest.mark.parametrize("shape", _APA_SHAPES)
def test_qtile_apa_welford_matches_np_mirror(shape):
    """v2 default (TC_APA_THR unset): Q-tile APA vs the fp64-statistics np
    mirror — K-EQ gate class (bulk16 semantics unchanged; only the selection
    statistics moved off the fp16 lattice)."""
    if shape[-1] > 64:
        pytest.skip("Q-tile path is fp16 D<=64; D>64 rides the ladder-only streaming kernel")
    q, k, v = _case_arrays(shape, np.float16, sum(shape))
    got = _run_apa(q, k, v, 0.15, qtile=True)
    ref = _np_apa_welford_fp16(q, k, v, NormalDist().inv_cdf(0.85))
    rf = _relfro(got, ref)
    print(f"QT APA welford {shape}: relfro vs np-fp64-stats mirror={rf:.4e}")
    assert rf <= 1e-3
    assert np.all(np.isfinite(got))


def test_qtile_apa_welford_survives_ladder_overflow_magnitudes():
    """THE K2 case: inputs scaled so sum(|bulk16|) > 65504 per row. The ladder
    saturates to thr=+inf and refines NOTHING; v2 must keep a finite threshold
    and a realized fraction near nominal r (the EXP-APA-2 DiT single-stream
    pathology, reproduced synthetically)."""
    qshape = (1, 2, 128, 64)
    kshape = (1, 2, 768, 64)
    shape = qshape
    rng = np.random.default_rng(20260714)
    # |score| ~ 155 mean per key at x12 inputs; 768 keys -> EVERY row sum
    # > 65504 (fp16 max) — the DiT single-stream saturation regime.
    q = (rng.standard_normal(qshape) * 12.0).astype(np.float16)
    k = (rng.standard_normal(kshape) * 12.0).astype(np.float16)
    v = (rng.standard_normal(kshape) * 0.125).astype(np.float16)
    a_check = np.abs(
        (q.astype(np.float32) @ _np_quant_dq16(k).transpose(0, 1, 3, 2)
         * np.float32(1.0 / 8.0)).astype(np.float16).astype(np.float32))
    assert float(a_check.sum(-1).min()) > 65504.0, \
        "test construction must overflow fp16 sums on every row"

    os.environ["TC_APA_FRAC"] = "1"
    try:
        tc.apa_refine_stats(reset=True)
        got_welford = _run_apa(q, k, v, 0.15, qtile=True)
        refined_w, total_w = tc.apa_refine_stats(reset=True)
        got_ladder = _run_apa(q, k, v, 0.15, qtile=True, thr="ladder16")
        refined_l, total_l = tc.apa_refine_stats(reset=True)
    finally:
        os.environ.pop("TC_APA_FRAC", None)

    frac_w = refined_w / total_w
    frac_l = refined_l / total_l
    print(f"QT APA overflow magnitudes: welford frac={frac_w:.4f} "
          f"ladder frac={frac_l:.4f} (nominal 0.15)")
    assert total_w == qshape[0] * qshape[1] * qshape[2] * kshape[2]
    # Welford mirror agreement at the K-EQ class even at these magnitudes.
    ref = _np_apa_welford_fp16(q, k, v, NormalDist().inv_cdf(0.85))
    assert _relfro(got_welford, ref) <= 1e-3
    # Every row overflowed -> the ladder refines ~nothing; v2 refines ~r.
    assert frac_l < 0.01
    assert 0.05 <= frac_w <= 0.35
    assert np.all(np.isfinite(got_welford))
    assert np.all(np.isfinite(got_ladder))


def test_apa_refine_stats_counter_matches_np_mask():
    """TC_APA_FRAC counter: denominator exact, fraction equals the np-mirror
    mask fraction (tie slack only); disabled -> no accumulation."""
    shape = (2, 3, 100, 64)
    q, k, v = _case_arrays(shape, np.float16, 99)
    z = NormalDist().inv_cdf(0.85)
    # np mask fraction (fp64 statistics).
    kdq = _np_quant_dq16(k)
    bulk = (q.astype(np.float32) @ kdq.transpose(0, 1, 3, 2)
            * np.float32(1.0 / 8.0)).astype(np.float16).astype(np.float32)
    a = np.abs(bulk).astype(np.float64)
    thr = a.mean(-1, keepdims=True) + np.sqrt(((a - a.mean(-1, keepdims=True)) ** 2)
                                              .mean(-1, keepdims=True)) * z
    np_frac = float((a >= thr).mean())

    os.environ["TC_APA_FRAC"] = "1"
    try:
        tc.apa_refine_stats(reset=True)
        _run_apa(q, k, v, 0.15, qtile=True)
        refined, total = tc.apa_refine_stats(reset=True)
    finally:
        os.environ.pop("TC_APA_FRAC", None)
    assert total == shape[0] * shape[1] * shape[2] * shape[2]
    frac = refined / total
    print(f"refine_stats: kernel frac={frac:.5f} np frac={np_frac:.5f}")
    assert abs(frac - np_frac) <= 2e-3  # fp32-vs-fp64 stat rounding tie slack

    # Disabled: counters must not move.
    _run_apa(q, k, v, 0.15, qtile=True)
    refined2, total2 = tc.apa_refine_stats(reset=False)
    assert (refined2, total2) == (0, 0)


def test_qtile_refine_all_ignores_thr_env():
    """r>=1 is threshold-free: outputs byte-identical under both TC_APA_THR
    values and unset (K2-EQ-R1's unit-level witness)."""
    q, k, v = _case_arrays((1, 3, 65, 64), np.float16, 20260714)
    outs = [_run_apa(q, k, v, 1.0, qtile=True, thr=thr).tobytes()
            for thr in (None, "welford", "ladder16")]
    assert outs[0] == outs[1] == outs[2]


def test_tc_apa_thr_invalid_raises():
    q, k, v = _case_arrays((1, 1, 16, 64), np.float16, 5)
    with pytest.raises(Exception):
        _run_apa(q, k, v, 0.15, qtile=True, thr="fp32")
