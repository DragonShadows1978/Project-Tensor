"""EXP-APA-2: fused non-causal INT4-bulk APA attention kernel gates.

Mirrors tests/test_hy3d_engine_ops.py conventions. The composed reference here
is a numpy fp32 mirror of the app-level EXP-APA-1 instrument (ColdCast
hy3d_tc/apa.py): symmetric-7 grouped INT4 dequant of K (one group per key
vector), bulk scores, Gaussian-quantile threshold mean+z*std on |bulk|
(population variance, like ops::var), mix_scores blend (non-refined keys keep
bulk — nothing dropped), exact softmax over ALL keys, V full precision.
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


def _np_quant_dq(k, dtype):
    """Composed _quantize_dequantize_key_symmetric mirror (bits=4)."""
    kf = k.astype(np.float32)
    qmax = np.float32(7.0)
    amax = np.max(np.abs(kf), axis=-1, keepdims=True)
    scale = amax * np.float32(1.0 / 7.0)
    safe = np.where(scale > 0, scale, np.float32(1.0))
    codes = np.clip(np.round(kf * (np.float32(1.0) / safe)), -qmax, qmax)
    dq = codes * safe
    return dq.astype(dtype)


def _np_apa_composed(q, k, v, z):
    """fp32 mirror of the composed APA chain, with the fused kernel's fp16
    score-boundary rounding when inputs are fp16.

    For fp16 the threshold is the ENGINE-OP LADDER, overflow included:
    reduce_sum stores the raw fp16 row sum BEFORE mul_scalar's 1/cnt, so
    sum(|bulk|) > 65504 saturates to inf and the row refines nothing —
    exactly what the composed chain does at the DiT single-stream sites."""
    is_half = q.dtype == np.float16
    scale = np.float32(1.0 / math.sqrt(q.shape[-1]))
    kdq = _np_quant_dq(k, q.dtype).astype(np.float32)
    qf, kf, vf = (t.astype(np.float32) for t in (q, k, v))

    def _round_scores(s):
        return s.astype(np.float16).astype(np.float32) if is_half else s

    bulk = _round_scores(qf @ kdq.transpose(0, 1, 3, 2) * scale)
    a = np.abs(bulk)
    inv_cnt = np.float32(1.0 / float(a.shape[-1]))
    if is_half:
        with np.errstate(over="ignore", invalid="ignore"):
            sum16 = a.sum(-1, keepdims=True, dtype=np.float32).astype(np.float16)
            mean16 = (sum16.astype(np.float32) * inv_cnt).astype(np.float16)
            meanv = mean16.astype(np.float32)
            d16 = (a - meanv).astype(np.float16)
            dd16 = (d16.astype(np.float32) ** 2).astype(np.float16)
            ddsum16 = dd16.astype(np.float32).sum(
                -1, keepdims=True, dtype=np.float32).astype(np.float16)
            var16 = (ddsum16.astype(np.float32) * inv_cnt).astype(np.float16)
            std16 = np.sqrt(var16.astype(np.float32)).astype(np.float16)
            sz16 = (std16.astype(np.float32) * np.float32(z)).astype(np.float16)
            thr = (meanv + sz16.astype(np.float32)).astype(np.float16).astype(np.float32)
    else:
        mean = a.mean(-1, keepdims=True, dtype=np.float32)
        var = ((a - mean) ** 2).mean(-1, keepdims=True, dtype=np.float32)
        thr = mean + np.sqrt(var) * np.float32(z)
    exact = _round_scores(qf @ kf.transpose(0, 1, 3, 2) * scale)
    with np.errstate(invalid="ignore"):
        blended = np.where(a >= thr, exact, bulk)
    w = np.exp(blended - blended.max(-1, keepdims=True))
    w = w / w.sum(-1, keepdims=True)
    return w @ vf


def _np_sdpa_fp32(q, k, v):
    scale = np.float32(1.0 / math.sqrt(q.shape[-1]))
    qf, kf, vf = (t.astype(np.float32) for t in (q, k, v))
    s = qf @ kf.transpose(0, 1, 3, 2) * scale
    w = np.exp(s - s.max(-1, keepdims=True))
    w = w / w.sum(-1, keepdims=True)
    return w @ vf


def _case_arrays(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return tuple(
        (rng.standard_normal(shape).astype(np.float32) * np.float32(0.125)).astype(dtype)
        for _ in range(3)
    )


def _run_fused(q, k, v, r):
    _require_cuda()
    dts = "float16" if q.dtype == np.float16 else "float32"
    refine_all = r >= 1.0
    z = 0.0 if refine_all else NormalDist().inv_cdf(1.0 - r)
    scale = 1.0 / math.sqrt(q.shape[-1])
    with tc.no_grad():
        out = tc.apa_int4_sdpa_noncausal(
            tc.tensor(q, dtype=dts), tc.tensor(k, dtype=dts), tc.tensor(v, dtype=dts),
            scale, z, refine_all=refine_all)
    tc.synchronize()
    return out.numpy()


def _relfro(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("dtype,atol", [(np.float32, 1e-6), (np.float16, 1e-3)])
def test_apa_int4_refine_all_is_exact_sdpa(dtype, atol):
    """r=1.0 (refine_all): the kernel is exact streaming SDPA — the
    fused_sdpa_noncausal K1 gate class (3643efe)."""
    shape = (1, 3, 33, 64)
    q, k, v = _case_arrays(shape, dtype, 20260712)
    got = _run_fused(q, k, v, 1.0).astype(np.float32)
    err = float(np.max(np.abs(got - _np_sdpa_fp32(q, k, v))))
    print(f"APA2 refine_all dtype={dtype.__name__} max_abs={err:.9g}")
    assert err <= atol
    assert np.all(np.isfinite(got))


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
@pytest.mark.parametrize("shape", [(1, 3, 33, 64), (2, 4, 257, 64), (1, 2, 129, 128)])
def test_apa_int4_matches_composed_mirror(dtype, shape):
    """r=0.15 vs the composed-chain numpy mirror (K-EQ gate class: 1e-3)."""
    q, k, v = _case_arrays(shape, dtype, sum(shape))
    got = _run_fused(q, k, v, 0.15)
    ref = _np_apa_composed(q, k, v, NormalDist().inv_cdf(0.85))
    rf = _relfro(got, ref)
    print(f"APA2 composed-mirror shape={shape} dtype={dtype.__name__} relfro={rf:.4e}")
    assert rf <= 1e-3
    assert np.all(np.isfinite(got))


@pytest.mark.parametrize("dtype,atol", [(np.float32, 1e-6), (np.float16, 1e-3)])
def test_apa_int4_dit_full_shape(dtype, atol):
    """DiT joint-attention geometry, both gates, opt-in (slow)."""
    if os.environ.get("TC_APA2_FULL_SHAPES") != "1":
        pytest.skip("set TC_APA2_FULL_SHAPES=1 to run the DiT full-shape gate")
    shape = (2, 16, 4442, 64)
    q, k, v = _case_arrays(shape, dtype, 11 if dtype == np.float32 else 12)
    got = _run_fused(q, k, v, 1.0).astype(np.float32)
    err = float(np.max(np.abs(got - _np_sdpa_fp32(q, k, v))))
    print(f"APA2 DiT refine_all dtype={dtype.__name__} max_abs={err:.9g}")
    assert err <= atol
    got = _run_fused(q, k, v, 0.15)
    ref = _np_apa_composed(q, k, v, NormalDist().inv_cdf(0.85))
    rf = _relfro(got, ref)
    print(f"APA2 DiT composed-mirror dtype={dtype.__name__} relfro={rf:.4e}")
    assert rf <= 1e-3


def test_apa_int4_rejects_bad_geometry():
    _require_cuda()
    q = tc.tensor(np.zeros((1, 1, 4, 63), dtype=np.float32))
    with pytest.raises(Exception):
        tc.apa_int4_sdpa_noncausal(q, q, q, 0.125, 1.0364)  # odd head_dim
