"""F-A1 gates for causal/selective APA with call-local packed INT4 K.

The independent reference follows EXP-APA-2's symmetric-7, one-group-per-key
packing convention, then the established causal selective algorithm:

  bulk = q @ dequant(INT4(k)) * scale
  threshold = mean(abs(bulk)) + z * population_std(abs(bulk))
  score = where(abs(bulk) >= threshold, q @ k * scale, bulk)
  output = exact softmax over every valid key @ v

Bottom-right causality means query i sees ``(S-L)+i+1`` keys.  Nothing is
dropped: non-refined keys retain their bulk score and remain in the softmax.

Registered tolerances:

* fp32 NumPy reference: rtol=2e-3, atol=2e-3.  CUDA uses warp/tree dot and
  statistics reductions plus partitioned online softmax, while NumPy/BLAS
  uses different fp32 association; the algorithm and selected-score merge
  are otherwise identical.
* bf16 legacy-kq cross-check: rtol=2e-2, atol=2e-2.  The legacy API must
  materialize dequantized K in bf16, rounding each reconstructed value before
  its dot.  The new path intentionally dequantizes packed codes in fp32
  registers, so one bf16 reconstruction rounding boundary is absent.
* forced split-K versus monolithic: rtol=3e-3, atol=3e-3 for fp32; both use
  the same packed values but reassociate online-softmax partition merges.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

import tensor_cuda as tc
from tensor_cuda.quant import _norm_ppf


FP32_TOL = dict(rtol=2e-3, atol=2e-3)
BF16_KQ_TOL = dict(rtol=2e-2, atol=2e-2)
SPLITK_TOL = dict(rtol=3e-3, atol=3e-3)


def _require_cuda():
    try:
        probe = tc.tensor(np.zeros(1, dtype=np.float32))
        tc.synchronize()
        del probe
    except Exception as exc:  # pragma: no cover - expected in CPU-only CI
        pytest.skip(f"CUDA unavailable: {exc}")


@pytest.fixture(autouse=True)
def _clear_path_override():
    os.environ.pop("TC_APA_SELECTIVE_PATH", None)
    yield
    os.environ.pop("TC_APA_SELECTIVE_PATH", None)


def _round_bf16(x):
    """Round fp32 to bf16 and return the rounded values as fp32."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def _quant_dequant_symmetric7(k, out_dtype=np.float32):
    """EXP-APA-2 per-key symmetric-7 pack/dequant convention."""
    kf = np.asarray(k, dtype=np.float32)
    amax = np.max(np.abs(kf), axis=-1, keepdims=True)
    scale = amax * np.float32(1.0 / 7.0)
    safe = np.where(scale > 0, scale, np.float32(1.0))
    codes = np.clip(
        np.round(kf * (np.float32(1.0) / safe)), -7.0, 7.0
    ).astype(np.float32)
    return (codes * safe).astype(out_dtype)


def _reference(q, k, v, zthr, causal):
    qf, kf, vf = (np.asarray(x, dtype=np.float32) for x in (q, k, v))
    kdq = _quant_dequant_symmetric7(kf)
    B, H, L, D = qf.shape
    KVH, S, VD = kf.shape[1], kf.shape[2], vf.shape[3]
    group = H // KVH
    scale = np.float32(1.0 / math.sqrt(D))
    out = np.empty((B, H, L, VD), dtype=np.float32)
    for b in range(B):
        for h in range(H):
            kh = h // group
            for i in range(L):
                s_max = (S - L) + i + 1 if causal else S
                qi = qf[b, h, i]
                bulk = (kdq[b, kh, :s_max] @ qi) * scale
                mag = np.abs(bulk)
                mean = mag.mean(dtype=np.float32)
                var = (mag * mag).mean(dtype=np.float32) - mean * mean
                threshold = mean + np.float32(zthr) * np.sqrt(
                    np.maximum(var, np.float32(0.0)), dtype=np.float32
                )
                exact = (kf[b, kh, :s_max] @ qi) * scale
                score = np.where(mag >= threshold, exact, bulk)
                weights = np.exp(score - score.max()).astype(np.float32)
                weights /= weights.sum(dtype=np.float32)
                out[b, h, i] = weights @ vf[b, kh, :s_max]
    return out


def _arrays(B, H, KVH, L, S, D, seed, VD=None):
    VD = D if VD is None else VD
    rng = np.random.default_rng(seed)
    q = (rng.standard_normal((B, H, L, D)) * 0.125).astype(np.float32)
    k = (rng.standard_normal((B, KVH, S, D)) * 0.125).astype(np.float32)
    v = (rng.standard_normal((B, KVH, S, VD)) * 0.125).astype(np.float32)
    return q, k, v


def _run(q, k, v, dtype, causal, path=None):
    _require_cuda()
    D = q.shape[-1]
    zthr = _norm_ppf(0.85)
    if path is not None:
        os.environ["TC_APA_SELECTIVE_PATH"] = str(path)
    with tc.no_grad():
        out = tc.apa_selective_attention_int4(
            tc.tensor(q, dtype=dtype),
            tc.tensor(k, dtype=dtype),
            tc.tensor(v, dtype=dtype),
            1.0 / math.sqrt(D),
            float(zthr),
            causal,
        )
    tc.synchronize()
    return out.numpy().astype(np.float32)


@pytest.mark.parametrize(
    "H,KVH,L,S,D,causal",
    [
        (4, 4, 9, 9, 64, False),
        (8, 4, 7, 29, 128, True),       # GQA + rectangular cache
        (16, 1, 3, 19, 512, True),      # Gemma-style MQA global head
        (16, 8, 5, 23, 128, True),      # 2:1 GQA + rectangular cache
    ],
)
def test_int4_selective_matches_composed_fp32_reference(
    H, KVH, L, S, D, causal
):
    """Gate (a), plus gates (c)/(d): EXP packing and composed fp32 parity."""
    q, k, v = _arrays(1, H, KVH, L, S, D, seed=H + KVH + D)
    got = _run(q, k, v, "float32", causal)
    ref = _reference(q, k, v, _norm_ppf(0.85), causal)
    np.testing.assert_allclose(got, ref, **FP32_TOL)
    assert np.all(np.isfinite(got))


@pytest.mark.parametrize("H,KVH,D", [(16, 1, 512), (16, 4, 128), (16, 8, 128)])
def test_int4_selective_matches_legacy_bf16_dequantized_kq(H, KVH, D):
    """Gate (b): new fp32 in-register dequant versus bf16 kq materialization."""
    _require_cuda()
    q, k, v = _arrays(1, H, KVH, 3, 31, D, seed=1000 + KVH + D)
    # The new packer sees bf16 K.  Mirror that input rounding before building
    # the legacy API's explicitly materialized, then bf16-rounded, kq tensor.
    k_bf = _round_bf16(k)
    kq = _round_bf16(_quant_dequant_symmetric7(k_bf))
    scale = 1.0 / math.sqrt(D)
    zthr = _norm_ppf(0.85)
    q_t = tc.tensor(q, dtype="bfloat16")
    k_t = tc.tensor(k, dtype="bfloat16")
    v_t = tc.tensor(v, dtype="bfloat16")
    with tc.no_grad():
        got = tc.apa_selective_attention_int4(
            q_t, k_t, v_t, scale, float(zthr), True
        )
        legacy = tc.apa_selective_attention(
            q_t,
            k_t,
            tc.tensor(kq, dtype="bfloat16"),
            v_t,
            scale,
            float(zthr),
            True,
        )
    tc.synchronize()
    np.testing.assert_allclose(
        got.numpy().astype(np.float32),
        legacy.numpy().astype(np.float32),
        **BF16_KQ_TOL,
    )


def test_int4_selective_rectangular_bottom_right_causal_is_observable():
    """Gate (c): S>L and late cached values catch accidental top-left masks."""
    _require_cuda()
    B, H, KVH, L, S, D = 1, 4, 1, 2, 11, 64
    q = np.ones((B, H, L, D), dtype=np.float32) * np.float32(0.05)
    k = np.zeros((B, KVH, S, D), dtype=np.float32)
    v = np.zeros((B, KVH, S, D), dtype=np.float32)
    # Query 0 may see through absolute key S-L=9 under bottom-right causal.
    # A top-left i+1 implementation cannot see this dominant value row.
    k[:, :, 9, :] = np.float32(1.0)
    v[:, :, 9, :] = np.float32(7.0)
    got = _run(q, k, v, "float32", True)
    ref = _reference(q, k, v, _norm_ppf(0.85), True)
    np.testing.assert_allclose(got, ref, **FP32_TOL)
    assert float(got[0, 0, 0].mean()) > 0.5


@pytest.mark.parametrize("causal", [False, True])
def test_int4_selective_decode_splitk_matches_monolithic(causal):
    """Gate (e): forced L=1 split-K and monolithic paths use one pack contract."""
    q, k, v = _arrays(1, 8, 4, 1, 4096, 128, seed=4242)
    fused = _run(q, k, v, "float32", causal, path=1)
    split = _run(q, k, v, "float32", causal, path=2)
    np.testing.assert_allclose(split, fused, **SPLITK_TOL)


def test_int4_selective_rejects_causal_s_shorter_than_l():
    _require_cuda()
    q, _, _ = _arrays(1, 2, 1, 5, 5, 64, seed=7)
    _, k, v = _arrays(1, 2, 1, 3, 3, 64, seed=8)
    with pytest.raises(Exception, match="S >= L"):
        tc.apa_selective_attention_int4(
            tc.tensor(q), tc.tensor(k), tc.tensor(v), 0.125, 1.0, True
        )
