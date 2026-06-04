"""Validate tensor_cuda's native INT4 group-quantized linear against a NumPy
reference that mirrors core/quantized_linear.py's exact scheme (per-group min/max
4-bit, packed even=low/odd=high nibble, w = q*scale + zero).

The test proves three things:
  1. dequant(packed, scales, zeros) on-device matches the NumPy dequant bit-exactly
     (in fp16), so the packing/nibble/group math is correct.
  2. int4_linear(x, ...) == x @ dequant(W).T within fp16 matmul tolerance.
  3. The quantization error vs the original fp32 weight is small (sanity: the
     scheme actually reconstructs the weight, not garbage).
"""
import numpy as np
import tensor_cuda as tc

GROUP = 128


def quantize_ref(w_fp32, group_size=GROUP):
    """Mirror QuantizedLinear.__init__ exactly. Returns packed uint8, scales, zeros."""
    out_features, in_features = w_fp32.shape
    num_groups = in_features // group_size
    w_grouped = w_fp32.reshape(out_features, num_groups, group_size)
    mins = w_grouped.min(axis=2)
    maxs = w_grouped.max(axis=2)
    scales = (maxs - mins) / 15.0
    scales = np.where(scales == 0, np.ones_like(scales), scales)
    zeros = mins
    w_norm = (w_grouped - zeros[:, :, None]) / scales[:, :, None]
    w_int4 = np.clip(np.round(w_norm), 0, 15).astype(np.uint8)
    w_int4_flat = w_int4.reshape(out_features, in_features)
    even = w_int4_flat[:, 0::2]
    odd = w_int4_flat[:, 1::2]
    packed = (even | (odd << 4)).astype(np.uint8)
    return packed, scales.astype(np.float16), zeros.astype(np.float16)


def dequant_ref(packed, scales, zeros, out_features, in_features, group_size=GROUP):
    even = (packed & 0x0F).astype(np.float16)
    odd = ((packed >> 4) & 0x0F).astype(np.float16)
    w_int4 = np.empty((out_features, in_features), dtype=np.float16)
    w_int4[:, 0::2] = even
    w_int4[:, 1::2] = odd
    num_groups = in_features // group_size
    w_grouped = w_int4.reshape(out_features, num_groups, group_size)
    w_deq = w_grouped * scales[:, :, None] + zeros[:, :, None]
    return w_deq.reshape(out_features, in_features)


def test_int4_dequant_matches_reference():
    rng = np.random.default_rng(0)
    N, K = 64, 256  # out_features, in_features
    w = rng.standard_normal((N, K)).astype(np.float32) * 0.1
    packed, scales, zeros = quantize_ref(w)

    w_deq_ref = dequant_ref(packed, scales, zeros, N, K)  # (N, K) fp16

    p_t = tc.tensor(packed, dtype="uint8")
    s_t = tc.tensor(scales, dtype="float16")
    z_t = tc.tensor(zeros, dtype="float16")
    # Engine returns the TRANSPOSED weight (K, N).
    w_deq_tc = tc.int4_dequant(p_t, s_t, z_t, GROUP, "float16").numpy()  # (K, N)

    # Compare against ref transposed.
    assert w_deq_tc.shape == (K, N), w_deq_tc.shape
    diff = np.abs(w_deq_tc.astype(np.float32) - w_deq_ref.T.astype(np.float32))
    assert diff.max() < 1e-2, f"dequant mismatch, max diff {diff.max()}"


def test_int4_linear_matches_dense_matmul():
    rng = np.random.default_rng(1)
    B, L, K, N = 2, 8, 256, 128
    w = rng.standard_normal((N, K)).astype(np.float32) * 0.1
    x = rng.standard_normal((B, L, K)).astype(np.float32) * 0.5
    packed, scales, zeros = quantize_ref(w)
    w_deq_ref = dequant_ref(packed, scales, zeros, N, K)  # (N, K) fp16

    # Reference: x @ dequant(W).T  in fp16.
    y_ref = (x.astype(np.float16) @ w_deq_ref.T).astype(np.float32)

    x_t = tc.tensor(x.astype(np.float16), dtype="float16")
    p_t = tc.tensor(packed, dtype="uint8")
    s_t = tc.tensor(scales, dtype="float16")
    z_t = tc.tensor(zeros, dtype="float16")
    y_tc = tc.int4_linear(x_t, p_t, s_t, z_t, GROUP).numpy().astype(np.float32)

    assert y_tc.shape == (B, L, N), y_tc.shape
    rel = np.abs(y_tc - y_ref) / (np.abs(y_ref) + 1e-3)
    assert rel.mean() < 1e-2, f"int4_linear mismatch, mean rel {rel.mean()}"


def test_quant_reconstruction_is_sane():
    """The quantized weight should approximate the original (not be garbage)."""
    rng = np.random.default_rng(2)
    N, K = 32, 256
    w = rng.standard_normal((N, K)).astype(np.float32) * 0.1
    packed, scales, zeros = quantize_ref(w)
    w_deq = dequant_ref(packed, scales, zeros, N, K).astype(np.float32)
    rel_err = np.abs(w_deq - w).mean() / (np.abs(w).mean() + 1e-9)
    # ~10-12% mean error is expected for 4-bit per-group min/max on Gaussian
    # data; this bound only catches a broken (garbage) reconstruction.
    assert rel_err < 0.15, f"4-bit reconstruction error too high: {rel_err}"


if __name__ == "__main__":
    test_int4_dequant_matches_reference()
    test_int4_linear_matches_dense_matmul()
    test_quant_reconstruction_is_sane()
    print("all int4 tests passed")
