"""APAMQ-F-A2 gates for dp4a integer bulk and BF16-MMA refinement.

Registered GPU tolerances:

* integer sums: exact equality (zero tolerance) against NumPy int32;
* Q/K scales and integer-derived unscaled bulk: rtol=1e-6, atol=1e-6;
* end-to-end INT8-Q/INT4-K composed reference: rtol=2e-2, atol=2e-2
  (BF16 output plus fp32 reduction/online-softmax association);
* BF16-MMA refine versus the unchanged F-A1 BF16 entry: rtol=8e-3,
  atol=8e-3 (exact BF16 products, fp32 accumulation reassociation only).

The CUDA sandbox is GPU-less, so every GPU gate is skip-safe. The quantizer,
packing, integer-dot, causal, threshold, and composed-reference machinery is
ordinary NumPy and has CPU-only unit coverage.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

import tensor_cuda as tc
from tensor_cuda.quant import _norm_ppf


SCALE_TOL = dict(rtol=1e-6, atol=1e-6)
E2E_TOL = dict(rtol=2e-2, atol=2e-2)
BF16_MMA_TOL = dict(rtol=8e-3, atol=8e-3)
SPLITK_TOL = dict(rtol=5e-3, atol=5e-3)


def _require_cuda():
    try:
        probe = tc.tensor(np.zeros(1, dtype=np.float32))
        tc.synchronize()
        del probe
    except Exception as exc:  # pragma: no cover - expected in CPU-only CI
        pytest.skip(f"CUDA unavailable: {exc}")


def _round_bf16(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def _round_away(x):
    x = np.asarray(x, dtype=np.float32)
    return np.copysign(np.floor(np.abs(x) + np.float32(0.5)), x)


def _quant_q_int8(q):
    qf = np.asarray(q, dtype=np.float32)
    amax = np.max(np.abs(qf), axis=-1, keepdims=True)
    scale = amax * np.float32(1.0 / 127.0)
    safe = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float32)
    code = np.clip(_round_away(qf * (np.float32(1.0) / safe)), -127, 127)
    return code.astype(np.int8), safe[..., 0]


def _quant_k_int4(k):
    kf = np.asarray(k, dtype=np.float32)
    amax = np.max(np.abs(kf), axis=-1, keepdims=True)
    scale = amax * np.float32(1.0 / 7.0)
    safe = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float32)
    code = np.clip(_round_away(kf * (np.float32(1.0) / safe)), -7, 7)
    return code.astype(np.int8), safe[..., 0]


def _pack_k(code):
    code = np.asarray(code, dtype=np.int8)
    lo = (code[..., 0::2].astype(np.int16) + 7).astype(np.uint8)
    hi = (code[..., 1::2].astype(np.int16) + 7).astype(np.uint8)
    return lo | (hi << np.uint8(4))


def _integer_sums(qcode, kcode):
    q32 = np.asarray(qcode, dtype=np.int32)
    k32 = np.asarray(kcode, dtype=np.int32)
    B, H, L, _D = q32.shape
    KVH, S = k32.shape[1:3]
    group = H // KVH
    out = np.empty((B, H, L, S), dtype=np.int64)
    for b in range(B):
        for h in range(H):
            out[b, h] = q32[b, h] @ k32[b, h // group].T
    return out


def _reference(q, k, v, zthr, causal):
    qf, kf, vf = (np.asarray(x, dtype=np.float32) for x in (q, k, v))
    qcode, qscale = _quant_q_int8(qf)
    kcode, kscale = _quant_k_int4(kf)
    isums = _integer_sums(qcode, kcode)
    B, H, L, D = qf.shape
    KVH, S, VD = kf.shape[1], kf.shape[2], vf.shape[3]
    group = H // KVH
    attn_scale = np.float32(1.0 / math.sqrt(D))
    out = np.empty((B, H, L, VD), dtype=np.float32)
    selections = []
    for b in range(B):
        for h in range(H):
            kh = h // group
            for i in range(L):
                s_max = (S - L) + i + 1 if causal else S
                bulk = (
                    isums[b, h, i, :s_max].astype(np.float32)
                    * np.float32(qscale[b, h, i] * kscale[b, kh, :s_max])
                    * attn_scale
                ).astype(np.float32)
                mag = np.abs(bulk)
                mean = mag.mean(dtype=np.float32)
                var = (mag * mag).mean(dtype=np.float32) - mean * mean
                threshold = mean + np.float32(zthr) * np.sqrt(
                    np.maximum(var, np.float32(0.0)), dtype=np.float32
                )
                selected = mag >= threshold
                exact = (kf[b, kh, :s_max] @ qf[b, h, i]) * attn_scale
                score = np.where(selected, exact, bulk)
                weights = np.exp(score - score.max()).astype(np.float32)
                weights /= weights.sum(dtype=np.float32)
                out[b, h, i] = weights @ vf[b, kh, :s_max]
                selections.append(selected)
    return out, selections


def _arrays(H, KVH, L, S, D, seed, VD=None):
    VD = D if VD is None else VD
    rng = np.random.default_rng(seed)
    q = _round_bf16((rng.standard_normal((1, H, L, D)) * 0.125).astype(np.float32))
    k = _round_bf16((rng.standard_normal((1, KVH, S, D)) * 0.125).astype(np.float32))
    v = _round_bf16((rng.standard_normal((1, KVH, S, VD)) * 0.125).astype(np.float32))
    return q, k, v


def test_fa2_reference_quantizers_and_integer_dot_cpu():
    q = np.array([[[[0.0, 1.0, -1.0, 0.5] * 4]]], dtype=np.float32)
    k = np.array([[[[0.0, 1.0, -1.0, 0.5] * 4,
                    [1.0, -1.0, 0.0, -0.5] * 4]]], dtype=np.float32)
    qcode, qscale = _quant_q_int8(q)
    kcode, kscale = _quant_k_int4(k)
    assert qcode.dtype == np.int8 and kcode.dtype == np.int8
    assert qcode.min() == -127 and qcode.max() == 127
    assert kcode.min() == -7 and kcode.max() == 7
    np.testing.assert_array_equal(_pack_k(kcode).shape, (1, 1, 2, 8))
    got = _integer_sums(qcode, kcode)
    manual = np.sum(
        qcode[0, 0, 0].astype(np.int32)
        * kcode[0, 0].astype(np.int32), axis=-1, dtype=np.int64
    )
    np.testing.assert_array_equal(got[0, 0, 0], manual)
    assert qscale[0, 0, 0] > 0 and np.all(kscale > 0)


def test_fa2_reference_rect_causal_mqa_d512_cpu():
    q, k, v = _arrays(4, 1, 3, 19, 512, seed=512)
    out, selections = _reference(q, k, v, _norm_ppf(0.85), True)
    assert out.shape == (1, 4, 3, 512)
    assert np.all(np.isfinite(out))
    assert all(mask.any() and (~mask).any() for mask in selections)
    assert {mask.size for mask in selections} == {17, 18, 19}


def test_int8q_int4_integer_bulk_is_bit_exact_gpu():
    _require_cuda()
    q, k, _v = _arrays(4, 1, 3, 19, 128, seed=9)
    values = tc.apa_int8q_int4_bulk_debug(
        tc.tensor(q, dtype="bfloat16"), tc.tensor(k, dtype="bfloat16")
    )
    tc.synchronize()
    qbytes, qscales, kpacked, kscales, isums, bulk = [
        x.numpy() for x in values
    ]
    qcode_ref, qscale_ref = _quant_q_int8(q)
    kcode_ref, kscale_ref = _quant_k_int4(k)
    np.testing.assert_array_equal(qbytes.view(np.int8), qcode_ref)
    np.testing.assert_array_equal(kpacked, _pack_k(kcode_ref))
    np.testing.assert_allclose(qscales, qscale_ref, **SCALE_TOL)
    np.testing.assert_allclose(kscales, kscale_ref, **SCALE_TOL)
    isum_ref = _integer_sums(qcode_ref, kcode_ref)
    np.testing.assert_array_equal(isums, isum_ref)  # registered zero tolerance
    bulk_ref = (
        isum_ref.astype(np.float32)
        * qscale_ref[..., None].astype(np.float32)
        * kscale_ref[:, 0:1, None, :].astype(np.float32)
    )
    np.testing.assert_allclose(bulk, bulk_ref, **SCALE_TOL)


@pytest.mark.parametrize("D", [128, 512])
def test_int8q_int4_e2e_composed_reference_mqa_rect_causal_gpu(D):
    _require_cuda()
    q, k, v = _arrays(4, 1, 3, 19, D, seed=100 + D)
    zthr = _norm_ppf(0.85)
    got = tc.apa_selective_attention_int8q_int4(
        tc.tensor(q, dtype="bfloat16"),
        tc.tensor(k, dtype="bfloat16"),
        tc.tensor(v, dtype="bfloat16"),
        1.0 / math.sqrt(D),
        float(zthr),
        True,
    )
    tc.synchronize()
    ref, selections = _reference(q, k, v, zthr, True)
    np.testing.assert_allclose(got.numpy().astype(np.float32), ref, **E2E_TOL)
    assert all(mask.any() and (~mask).any() for mask in selections)


@pytest.mark.parametrize("D", [128, 512])
def test_bf16_mma_refine_matches_unchanged_fa1_gpu(D):
    """Same bulk/statistics helper means the selection mask is identical."""
    _require_cuda()
    q, k, v = _arrays(8, 1, 3, 23, D, seed=300 + D)
    args = (
        tc.tensor(q, dtype="bfloat16"),
        tc.tensor(k, dtype="bfloat16"),
        tc.tensor(v, dtype="bfloat16"),
        1.0 / math.sqrt(D),
        float(_norm_ppf(0.85)),
        True,
    )
    baseline = tc.apa_selective_attention_int4(*args)
    mma = tc.apa_selective_attention_int4_bf16_mma(*args)
    tc.synchronize()
    np.testing.assert_allclose(
        mma.numpy().astype(np.float32),
        baseline.numpy().astype(np.float32),
        **BF16_MMA_TOL,
    )


def test_fa2_entries_reject_non_bf16_gpu():
    _require_cuda()
    q, k, v = _arrays(2, 1, 1, 3, 128, seed=77)
    args = (tc.tensor(q), tc.tensor(k), tc.tensor(v), 0.125, 1.0, True)
    with pytest.raises(Exception, match="bfloat16"):
        tc.apa_selective_attention_int8q_int4(*args)
    with pytest.raises(Exception, match="bfloat16"):
        tc.apa_selective_attention_int4_bf16_mma(*args)


@pytest.mark.parametrize(
    "entry",
    [
        tc.apa_selective_attention_int4_bf16_mma,
        tc.apa_selective_attention_int8q_int4,
    ],
)
def test_fa2_decode_splitk_matches_monolithic_gpu(entry):
    """Long-context L=1 path gets key-range parallelism for both lanes."""
    _require_cuda()
    q, k, v = _arrays(8, 1, 1, 4096, 128, seed=909)
    args = (
        tc.tensor(q, dtype="bfloat16"),
        tc.tensor(k, dtype="bfloat16"),
        tc.tensor(v, dtype="bfloat16"),
        1.0 / math.sqrt(128),
        float(_norm_ppf(0.85)),
        True,
    )
    try:
        os.environ["TC_APA_SELECTIVE_PATH"] = "1"
        fused = entry(*args)
        tc.synchronize()
        os.environ["TC_APA_SELECTIVE_PATH"] = "2"
        split = entry(*args)
        tc.synchronize()
    finally:
        os.environ.pop("TC_APA_SELECTIVE_PATH", None)
    np.testing.assert_allclose(
        split.numpy().astype(np.float32),
        fused.numpy().astype(np.float32),
        **SPLITK_TOL,
    )
