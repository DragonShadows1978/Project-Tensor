"""Validate native GPT-OSS MXFP4 expert linear kernels."""

import numpy as np
import pytest
import tensor_cuda as tc


FP4_VALUES = np.asarray(
    [
        +0.0,
        +0.5,
        +1.0,
        +1.5,
        +2.0,
        +3.0,
        +4.0,
        +6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)


def mxfp4_dequant_ref(blocks, scales):
    blocks = np.asarray(blocks, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.uint8)
    lo = blocks & 0x0F
    hi = blocks >> 4
    vals = np.empty(blocks.shape[:2] + (32,), dtype=np.float32)
    vals[..., 0::2] = FP4_VALUES[lo]
    vals[..., 1::2] = FP4_VALUES[hi]
    vals = np.ldexp(vals, scales.astype(np.int32)[..., None] - 127)
    return vals.reshape(blocks.shape[0], blocks.shape[1] * 32).T


def test_mxfp4_linear_matches_reference_gemm_and_gemv_float32():
    rng = np.random.default_rng(20260706)
    m, n, groups = 5, 64, 4
    k = groups * 32
    blocks = rng.integers(0, 256, size=(n, groups, 16), dtype=np.uint8)
    scales = rng.integers(123, 132, size=(n, groups), dtype=np.uint8)
    x = rng.standard_normal((m, k), dtype=np.float32) * 0.125
    w_kn = mxfp4_dequant_ref(blocks, scales)

    args = (
        tc.tensor(blocks, dtype="uint8"),
        tc.tensor(scales, dtype="uint8"),
    )
    with tc.no_grad():
        y = tc.mxfp4_linear(tc.tensor(x, dtype="float32"), *args).numpy()
        y1 = tc.mxfp4_linear(tc.tensor(x[:1], dtype="float32"), *args).numpy()

    np.testing.assert_allclose(y, x @ w_kn, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(y1, x[:1] @ w_kn, rtol=2e-5, atol=2e-5)


def test_mxfp4_linear_bfloat16_decode_shape_matches_reference():
    rng = np.random.default_rng(20260707)
    n, groups = 96, 6
    k = groups * 32
    blocks = rng.integers(0, 256, size=(n, groups, 16), dtype=np.uint8)
    scales = rng.integers(124, 131, size=(n, groups), dtype=np.uint8)
    x = rng.standard_normal((1, k), dtype=np.float32) * 0.0625
    w_kn = mxfp4_dequant_ref(blocks, scales)

    with tc.no_grad():
        x_t = tc.tensor(x, dtype="float32").astype("bfloat16")
        x_bf = x_t.float().numpy()
        y = tc.mxfp4_linear(
            x_t,
            tc.tensor(blocks, dtype="uint8"),
            tc.tensor(scales, dtype="uint8"),
        ).float().numpy()

    np.testing.assert_allclose(y, x_bf @ w_kn, rtol=0.025, atol=0.025)


def test_mxfp4_linear_expert_matches_direct_selected_expert():
    rng = np.random.default_rng(20260708)
    experts, n, groups = 5, 40, 3
    k = groups * 32
    expert_idx = 3
    blocks = rng.integers(0, 256, size=(experts, n, groups, 16), dtype=np.uint8)
    scales = rng.integers(124, 131, size=(experts, n, groups), dtype=np.uint8)
    x = rng.standard_normal((2, k), dtype=np.float32) * 0.1

    with tc.no_grad():
        y_expert = tc.mxfp4_linear_expert(
            tc.tensor(x, dtype="float32"),
            tc.tensor(blocks, dtype="uint8"),
            tc.tensor(scales, dtype="uint8"),
            expert_idx,
        ).numpy()
        y_direct = tc.mxfp4_linear(
            tc.tensor(x, dtype="float32"),
            tc.tensor(blocks[expert_idx], dtype="uint8"),
            tc.tensor(scales[expert_idx], dtype="uint8"),
        ).numpy()

    np.testing.assert_allclose(y_expert, y_direct, rtol=0.0, atol=0.0)


def test_mxfp4_linear_rejects_k_mismatch():
    blocks = np.zeros((8, 3, 16), dtype=np.uint8)
    scales = np.full((8, 3), 127, dtype=np.uint8)
    x = np.zeros((1, 64), dtype=np.float32)

    with pytest.raises(RuntimeError, match="last dimension mismatches K"):
        tc.mxfp4_linear(
            tc.tensor(x),
            tc.tensor(blocks, dtype="uint8"),
            tc.tensor(scales, dtype="uint8"),
        )
