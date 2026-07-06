"""Validate native packed INT2/INT3 weight dequant and linear kernels."""

import numpy as np
import pytest
import tensor_cuda as tc
from tensor_cuda.quantization import (
    dequantize_affine_per_group,
    dequantize_symmetric_per_group,
    pack_lowbit,
    quantize_affine_per_group,
)

GROUP = 64


def quantize_ref(w_fp32, bits, group_size=GROUP):
    q = quantize_affine_per_group(w_fp32, bits, group_size)
    return q.packed, q.scales, q.zeros


def dequant_ref(packed, scales, zeros, bits, in_features, group_size=GROUP):
    return dequantize_affine_per_group(
        packed, scales, zeros, bits, in_features, group_size
    )


@pytest.mark.parametrize("bits", [2, 3])
def test_intn_dequant_matches_reference(bits):
    rng = np.random.default_rng(10 + bits)
    n, k = 64, 256
    w = rng.standard_normal((n, k)).astype(np.float32) * 0.1
    packed, scales, zeros = quantize_ref(w, bits)
    w_ref = dequant_ref(packed, scales, zeros, bits, k)

    w_tc = tc.intn_dequant(
        tc.tensor(packed, dtype="uint8"),
        tc.tensor(scales, dtype="float16"),
        tc.tensor(zeros, dtype="float16"),
        bits,
        k,
        GROUP,
        "float32",
    ).numpy()

    assert w_tc.shape == (k, n), w_tc.shape
    diff = np.abs(w_tc - w_ref.T)
    assert diff.max() < 1e-6, f"INT{bits} dequant mismatch: {diff.max()}"


@pytest.mark.parametrize("bits", [2, 3])
def test_intn_linear_matches_dense_matmul(bits):
    rng = np.random.default_rng(20 + bits)
    batch, seq, k, n = 2, 5, 256, 96
    w = rng.standard_normal((n, k)).astype(np.float32) * 0.1
    x = rng.standard_normal((batch, seq, k)).astype(np.float32) * 0.5
    packed, scales, zeros = quantize_ref(w, bits)
    w_ref = dequant_ref(packed, scales, zeros, bits, k)
    y_ref = (x.astype(np.float16) @ w_ref.T.astype(np.float16)).astype(np.float32)

    y_tc = tc.intn_linear(
        tc.tensor(x.astype(np.float16), dtype="float16"),
        tc.tensor(packed, dtype="uint8"),
        tc.tensor(scales, dtype="float16"),
        tc.tensor(zeros, dtype="float16"),
        bits,
        k,
        GROUP,
    ).numpy().astype(np.float32)

    assert y_tc.shape == (batch, seq, n), y_tc.shape
    rel = np.abs(y_tc - y_ref) / (np.abs(y_ref) + 1e-3)
    assert rel.mean() < 2e-2, f"INT{bits} linear mean rel {rel.mean()}"


@pytest.mark.parametrize("bits", [2, 3])
def test_intn_fused_matches_two_stage_for_gemm_and_gemv(bits):
    rng = np.random.default_rng(30 + bits)
    m, k, n = 4, 256, 80
    w = rng.standard_normal((n, k)).astype(np.float32) * 0.1
    x = rng.standard_normal((m, k)).astype(np.float32) * 0.25
    packed, scales, zeros = quantize_ref(w, bits)

    args = (
        tc.tensor(packed, dtype="uint8"),
        tc.tensor(scales, dtype="float16"),
        tc.tensor(zeros, dtype="float16"),
        bits,
        k,
        GROUP,
    )
    x_t = tc.tensor(x, dtype="float32")
    y_two = tc.intn_linear(x_t, *args).numpy()
    y_fused = tc.intn_linear_fused(x_t, *args).numpy()
    np.testing.assert_allclose(y_fused, y_two, rtol=2e-4, atol=2e-4)

    x1_t = tc.tensor(x[:1], dtype="float32")
    y1_two = tc.intn_linear(x1_t, *args).numpy()
    y1_fused = tc.intn_linear_fused(x1_t, *args).numpy()
    np.testing.assert_allclose(y1_fused, y1_two, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("bits", [2, 3])
def test_intn_empty_zeros_uses_symmetric_grid(bits):
    rng = np.random.default_rng(40 + bits)
    n, k = 48, 256
    q = rng.integers(0, 1 << bits, size=(n, k), dtype=np.uint8)
    scales = (
        rng.random((n, k // GROUP), dtype=np.float32) * 0.02 + 1e-4
    ).astype(np.float16)
    packed = pack_lowbit(q, bits)
    w_ref = dequantize_symmetric_per_group(packed, scales, bits, k, GROUP)
    empty_zeros = tc.tensor(np.zeros((0,), dtype=np.float16), dtype="float16")

    w_tc = tc.intn_dequant(
        tc.tensor(packed, dtype="uint8"),
        tc.tensor(scales, dtype="float16"),
        empty_zeros,
        bits,
        k,
        GROUP,
        "float32",
    ).numpy().T

    assert np.abs(w_tc - w_ref).max() < 1e-7


@pytest.mark.parametrize("bits,max_rel", [(2, 0.55), (3, 0.25)])
def test_intn_quant_reconstruction_is_sane(bits, max_rel):
    rng = np.random.default_rng(50 + bits)
    n, k = 32, 256
    w = rng.standard_normal((n, k)).astype(np.float32) * 0.1
    packed, scales, zeros = quantize_ref(w, bits)
    w_deq = dequant_ref(packed, scales, zeros, bits, k)
    rel_err = np.abs(w_deq - w).mean() / (np.abs(w).mean() + 1e-9)
    assert rel_err < max_rel, f"INT{bits} reconstruction error too high: {rel_err}"
