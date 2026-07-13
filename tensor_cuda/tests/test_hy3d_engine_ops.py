"""WO-1A numerical gates for HY3D engine primitives.

The full-size SDPA cases are deliberately opt-in because their NumPy golden
references are substantial. Run them with ``TC_WO1A_FULL_SHAPES=1`` or use
``tensor_cuda/run_wo1a_gates.sh``. The normal engine suite still exercises the
small, odd-tail routing and the exact-erf GELU surface.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

import tensor_cuda as tc
from tensor_cuda import functional as F
from tensor_cuda import nn


def _require_cuda():
    try:
        tc.synchronize()
    except RuntimeError as exc:  # pragma: no cover - host-only authoring path
        pytest.skip(f"CUDA unavailable for WO-1A gate: {exc}")


@pytest.fixture(autouse=True)
def _cuda_gate():
    """Make host-only authoring runs report skips, never false failures."""
    _require_cuda()


def _np_sdpa_fp32(q, k, v, block_q=512):
    """Reference SDPA with fp32 inputs/accumulation and bounded host scratch."""
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    B, H, Lq, D = q.shape
    assert k.shape == (B, H, k.shape[2], D)
    assert v.shape == k.shape
    Lk = k.shape[2]
    scale = np.float32(1.0 / math.sqrt(D))
    out = np.empty((B, H, Lq, D), dtype=np.float32)
    for b in range(B):
        for h in range(H):
            kt = np.ascontiguousarray(k[b, h].T)
            vh = v[b, h]
            for lo in range(0, Lq, block_q):
                hi = min(lo + block_q, Lq)
                scores = q[b, h, lo:hi] @ kt
                scores *= scale
                scores -= scores.max(axis=-1, keepdims=True)
                np.exp(scores, out=scores)
                scores /= scores.sum(axis=-1, keepdims=True, dtype=np.float32)
                out[b, h, lo:hi] = scores @ vh
    return out


def _run_fused(q, k, v):
    _require_cuda()
    dtype = "float16" if q.dtype == np.float16 else "float32"
    old = os.environ.get("TC_FUSED_SDPA_NONCAUSAL")
    os.environ["TC_FUSED_SDPA_NONCAUSAL"] = "1"
    try:
        with tc.no_grad():
            out = F.scaled_dot_product_attention(
                tc.tensor(q, dtype=dtype), tc.tensor(k, dtype=dtype),
                tc.tensor(v, dtype=dtype))
        tc.synchronize()
        return out.numpy()
    finally:
        if old is None:
            os.environ.pop("TC_FUSED_SDPA_NONCAUSAL", None)
        else:
            os.environ["TC_FUSED_SDPA_NONCAUSAL"] = old


def _case_arrays(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    # The HY3D attention inputs are normalized activations, not adversarially
    # large logits. Keeping this range also makes the reference reproducible
    # across BLAS implementations without changing the exact math being gated.
    return tuple(
        (rng.standard_normal(shape).astype(np.float32) * np.float32(0.125)).astype(dtype)
        for _ in range(3)
    )


@pytest.mark.parametrize("dtype,atol", [(np.float32, 1e-6), (np.float16, 1e-3)])
def test_fused_sdpa_noncausal_small_odd_tail(dtype, atol):
    """Always-on routing/parity coverage for a non-multiple-of-warp tail."""
    shape = (1, 3, 33, 64)
    q, k, v = _case_arrays(shape, dtype, 20260712)
    got = _run_fused(q, k, v).astype(np.float32)
    expected = _np_sdpa_fp32(q, k, v, block_q=33)
    err = float(np.max(np.abs(got - expected)))
    print(f"WO1A K1 small_odd dtype={dtype.__name__} max_abs={err:.9g}")
    assert err <= atol


@pytest.mark.parametrize(
    "name,shape,dtype,atol,seed",
    [
        ("dit_joint", (1, 16, 4442, 64), np.float32, 1e-6, 11),
        ("dit_joint", (1, 16, 4442, 64), np.float16, 1e-3, 12),
        ("vae_cross_d64", (1, 16, 4096, 64), np.float32, 1e-6, 21),
        ("vae_cross_d64", (1, 16, 4096, 64), np.float16, 1e-3, 22),
        ("vae_cross_d128", (1, 16, 4096, 128), np.float32, 1e-6, 31),
        ("vae_cross_d128", (1, 16, 4096, 128), np.float16, 1e-3, 32),
    ],
)
def test_fused_sdpa_noncausal_hy3d_full_shapes(name, shape, dtype, atol, seed):
    """K1: DiT/VAE full heads, with Lk selected below by the named geometry."""
    if os.environ.get("TC_WO1A_FULL_SHAPES") != "1":
        pytest.skip("set TC_WO1A_FULL_SHAPES=1 to run HY3D full-shape gate")
    B, H, Lq, D = shape
    Lk = 4442 if name == "dit_joint" else 3072
    q, _, _ = _case_arrays((B, H, Lq, D), dtype, seed)
    _, k, v = _case_arrays((B, H, Lk, D), dtype, seed + 1000)
    got = _run_fused(q, k, v).astype(np.float32)
    expected = _np_sdpa_fp32(q, k, v)
    err = float(np.max(np.abs(got - expected)))
    print(f"WO1A K1 {name} dtype={dtype.__name__} max_abs={err:.9g}")
    assert err <= atol


@pytest.mark.parametrize("dtype,atol", [(np.float32, 1e-6), (np.float16, 1e-3)])
def test_fused_sdpa_noncausal_33_cubed_chunk_tail(dtype, atol):
    """K1 tail coverage for an actual 33^3 dense-grid query length.

    This retains the production H=16 dispatch rather than testing the tail in
    a reduced-head surrogate.
    """
    if os.environ.get("TC_WO1A_FULL_SHAPES") != "1":
        pytest.skip("set TC_WO1A_FULL_SHAPES=1 to run HY3D full-shape gate")
    B, H, Lq, Lk, D = 1, 16, 33 ** 3, 3072, 64
    q, _, _ = _case_arrays((B, H, Lq, D), dtype, 41)
    _, k, v = _case_arrays((B, H, Lk, D), dtype, 1041)
    got = _run_fused(q, k, v).astype(np.float32)
    expected = _np_sdpa_fp32(q, k, v)
    err = float(np.max(np.abs(got - expected)))
    print(f"WO1A K1 33_cubed_tail dtype={dtype.__name__} max_abs={err:.9g}")
    assert err <= atol


def test_fused_vs_composed_noncausal_fp32(monkeypatch):
    """K3: environment toggle must preserve the existing composed result."""
    q, k, v = _case_arrays((1, 4, 127, 64), np.float32, 73)
    with tc.no_grad():
        monkeypatch.setenv("TC_FUSED_SDPA_NONCAUSAL", "1")
        fused = F.scaled_dot_product_attention(tc.tensor(q), tc.tensor(k), tc.tensor(v))
        monkeypatch.setenv("TC_FUSED_SDPA_NONCAUSAL", "0")
        composed = F.scaled_dot_product_attention(tc.tensor(q), tc.tensor(k), tc.tensor(v))
    tc.synchronize()
    err = float(np.max(np.abs(fused.numpy() - composed.numpy())))
    print(f"WO1A K3 fused_vs_composed_fp32 max_abs={err:.9g}")
    assert err <= 1e-5


def test_noncausal_sdpa_default_is_composed(monkeypatch):
    """K3: an absent opt-in must retain the pre-WO-1A composed route."""
    monkeypatch.delenv("TC_FUSED_SDPA_NONCAUSAL", raising=False)

    def _unexpected_fused(*args, **kwargs):
        pytest.fail("default non-causal SDPA route invoked fused kernel")

    monkeypatch.setattr(F.tc, "fused_sdpa_noncausal", _unexpected_fused)
    q, k, v = _case_arrays((1, 2, 17, 64), np.float32, 74)
    with tc.no_grad():
        got = F.scaled_dot_product_attention(tc.tensor(q), tc.tensor(k), tc.tensor(v))
    tc.synchronize()
    expected = _np_sdpa_fp32(q, k, v, block_q=17)
    np.testing.assert_allclose(got.numpy(), expected, atol=1e-6, rtol=1e-6)


def _gelu_exact_reference(x):
    x = np.asarray(x, dtype=np.float32)
    erf = np.fromiter((math.erf(float(v) / math.sqrt(2.0)) for v in x.ravel()),
                      dtype=np.float64, count=x.size).reshape(x.shape)
    return (0.5 * x.astype(np.float64) * (1.0 + erf)).astype(np.float32)


@pytest.mark.parametrize("dtype,atol", [(np.float32, 1e-6), (np.float16, 1e-3)])
def test_gelu_exact_matches_erf_reference(dtype, atol):
    """K2: large magnitudes and representable denormals included explicitly."""
    _require_cuda()
    f32_denorm = np.nextafter(np.float32(0), np.float32(1), dtype=np.float32)
    f16_denorm = np.float32(np.nextafter(np.float16(0), np.float16(1), dtype=np.float16))
    grid = np.array([
        -30.0, -20.0, -12.0, -8.0, -6.0, -3.0, -1.0, -0.25,
        -f16_denorm, -f32_denorm, -0.0, 0.0, f32_denorm, f16_denorm,
        0.25, 1.0, 3.0, 6.0, 8.0, 12.0, 20.0, 30.0,
    ], dtype=np.float32)
    rng = np.random.default_rng(20260712)
    x = np.concatenate([grid, rng.standard_normal(4096).astype(np.float32) * 5.0])
    inp = x.astype(dtype)
    with tc.no_grad():
        got = nn.GELUExact()(tc.tensor(inp, dtype="float16" if dtype is np.float16 else "float32"))
    tc.synchronize()
    expected = _gelu_exact_reference(inp)
    err = float(np.max(np.abs(got.numpy().astype(np.float32) - expected)))
    print(f"WO1A K2 gelu_exact dtype={dtype.__name__} max_abs={err:.9g}")
    assert err <= atol


def test_gelu_exact_backward_and_legacy_tanh_gelu_remain_distinct():
    """Exact GELU remains a normal unary autograd op; legacy GELU stays tanh."""
    _require_cuda()
    x = np.array([-3.0, -1.0, -0.0, 0.0, 1.0, 3.0], dtype=np.float32)
    xt = tc.tensor(x, requires_grad=True)
    nn.GELUExact()(xt).sum().backward()
    grad = xt.grad.numpy()
    cdf = 0.5 * (1.0 + np.array([math.erf(float(v) / math.sqrt(2.0)) for v in x]))
    expected_grad = cdf + x * np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    np.testing.assert_allclose(grad, expected_grad.astype(np.float32), atol=1e-6, rtol=1e-6)
    legacy = nn.GELU()(tc.tensor(x)).numpy()
    tanh_ref = 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))
    np.testing.assert_allclose(legacy, tanh_ref.astype(np.float32), atol=1e-6, rtol=1e-6)
