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


def _run_apa(q, k, v, r, *, qtile: bool):
    _require_cuda()
    dts = "float16" if q.dtype == np.float16 else "float32"
    refine_all = r >= 1.0
    z = 0.0 if refine_all else NormalDist().inv_cdf(1.0 - r)
    os.environ["TC_ATTN_QTILE"] = "1" if qtile else "0"
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
    kernel: identical semantics, reduction-order slack only (K-EQ class)."""
    q, k, v = _case_arrays(shape, np.float16, sum(shape))
    got = _run_apa(q, k, v, 0.15, qtile=True)
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
