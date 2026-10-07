"""APA-MF2.1 item 2 (review F1): softmax / max reductions start at -inf, so a row with ANY
visible finite score is finite, whatever its magnitude.

Before: kernels.cu reduce MODE 1 started at -3.4e38f (MODE 2 at 3.4e38f) and
causal_softmax at -3.0e38f. A visible score below the start (fp32 -3.3e38 / -3.402e38,
bf16 -3.38e38) left the running max at the sentinel, every exp underflowed to 0 and the
row became 0/0 = NaN once its masked entries were -inf (reviewer `negative` cell RED:
artifacts/apa_mf2rev/gpu_negative_20261007T210327Z_183525.json on apa-mf2rev).
After: -INFINITY / +INFINITY, the identities of max / min -- only the start value moved.

Witness (review F1): B=H=1, L=S=2, D=64, Q = e0 on both rows, K0 = s*e0, K1 = 0, V0 = 1,
V1 = 0, scale 1, causal. Row 0 sees key 0 only -> output exactly 1; row 1 sees scores
(s, 0) -> output exactly 0 (exp(s) underflows).

Prior art: FlashAttention-2 (Dao 2023) initialises its running max m = -inf; PyTorch's
amax / softmax use -inf as the max identity. Unverified -- lead to check
("flash attention 2 algorithm 1 m_i = -inf"). Witness from the review; tests: mine.
"""
import contextlib

import numpy as np
import pytest

import tensor_cuda as tc
from tensor_cuda import functional as F
from tensor_cuda import quant as Q

FLT_MAX = float(np.finfo(np.float32).max)
FP32_SCORES = (-3.0e38, -3.3e38, -3.39e38, -3.402e38, -FLT_MAX)
BF16_SCORES = (-3.0e38, -3.38e38)


def _witness(s, dtype):
    q = np.zeros((1, 1, 2, 64), np.float32); q[..., 0] = 1
    k = np.zeros_like(q); k[0, 0, 0, 0] = s
    v = np.zeros_like(q); v[0, 0, 0] = 1
    return [tc.tensor(a, dtype=dtype) for a in (q, k, v)]


def _sdpa(path, q, k, v):
    saved = F.USE_FUSED_SOFTMAX
    F.USE_FUSED_SOFTMAX = path == "fused"
    try:
        ctx = contextlib.nullcontext() if path == "grad" else tc.no_grad()
        with ctx:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
    finally:
        F.USE_FUSED_SOFTMAX = saved


@pytest.mark.parametrize("path", ["infer", "fused", "grad"])
@pytest.mark.parametrize("dtype,s", [("float32", s) for s in FP32_SCORES] + [("bfloat16", s) for s in BF16_SCORES])
def test_sdpa_extreme_negative_visible_score_is_finite(dtype, s, path):
    q, k, v = _witness(s, dtype)
    o = _sdpa(path, q, k, v).float().numpy()[0, 0]
    assert np.isfinite(o).all(), f"NaN/Inf for visible score {s} ({dtype}, {path})"
    assert np.all(o[0] == 1.0), f"row 0 (one visible key) {o[0, :4]}"
    assert np.all(o[1] == 0.0), f"row 1 {o[1, :4]}"


def test_sdpa_grad_extreme_negative_backward_finite():
    qq = np.zeros((1, 1, 2, 64), np.float32); qq[..., 0] = 1
    kk = np.zeros_like(qq); kk[0, 0, 0, 0] = -3.402e38
    vv = np.zeros_like(qq); vv[0, 0, 0] = 1
    leaves = [tc.tensor(a, requires_grad=True) for a in (qq, kk, vv)]
    o = F.scaled_dot_product_attention(*leaves, is_causal=True, scale=1.0)
    o.sum().backward()
    assert np.all(o.numpy()[0, 0, 0] == 1.0)
    for x in leaves:
        assert np.isfinite(x.grad.numpy()).all()


@pytest.mark.parametrize("s", FP32_SCORES)
def test_apa_quant_full_refine_extreme_negative(s):
    q, k, v = _witness(s, "float32")
    o = Q.apa_quant_attention(q, k, v, bulk_bits=8, refine_percentile=1.0, is_causal=True, scale=1.0).numpy()[0, 0]
    assert np.isfinite(o).all() and np.all(o[0] == 1.0) and np.all(o[1] == 0.0), o[:, :3]


@pytest.mark.parametrize("s", FP32_SCORES)
def test_apa_quant_partial_refine_extreme_negative_supplied_kq(s):
    """refine 0.15 with the supplied finite Kq = K (the quantizer's |k|^2 overflows
    above ~1.8e19 -- out of scope, review F4 isolation)."""
    q, k, v = _witness(s, "float32")
    orig = Q._quantize_keys
    Q._quantize_keys = lambda *a, **kw: k
    try:
        o = Q.apa_quant_attention(q, k, v, bulk_bits=8, refine_percentile=0.15, is_causal=True,
                                  scale=1.0).numpy()[0, 0]
    finally:
        Q._quantize_keys = orig
    assert np.isfinite(o).all() and np.all(o[0] == 1.0) and np.all(o[1] == 0.0), o[:, :3]


@pytest.mark.parametrize("dtype,s", [("float32", s) for s in FP32_SCORES] + [("bfloat16", s) for s in BF16_SCORES])
def test_causal_softmax_kernel_direct(dtype, s):
    x = tc.tensor(np.array([[[[s, 123.0], [s, 0.0]]]], np.float32), dtype=dtype)
    with tc.no_grad():
        w = tc.causal_softmax(x).float().numpy()[0, 0]
    np.testing.assert_array_equal(w, [[1.0, 0.0], [0.0, 1.0]])


def test_causal_softmax_minus_inf_entry():
    with tc.no_grad():
        w = tc.causal_softmax(tc.tensor(np.array([[-np.inf, 0.0]], np.float32))).numpy()
    np.testing.assert_array_equal(w, [[0.0, 1.0]])


@pytest.mark.parametrize("s", FP32_SCORES)
def test_eager_softmax_direct(s):
    x = tc.tensor(np.array([[s, -np.inf, -np.inf], [s, s, -np.inf]], np.float32))
    w = x.softmax(-1).numpy()
    np.testing.assert_array_equal(w, [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])


@pytest.mark.parametrize("R", [2, 7, 300, 1025])
@pytest.mark.parametrize("trailing", [True, False])
def test_reduce_max_min_start_is_the_identity(R, trailing):
    """Rows entirely beyond the old +-3.4e38f start: the true max/min, never the start.
    trailing=True takes reduce_lastdim_kernel (block per row), False the generic kernel."""
    lo = np.full((3, R), -3.402e38, np.float32); lo[:, R // 2] = -3.4019e38
    lo[1] = -np.inf                                   # all -inf row: max identity is -inf
    hi = -lo; hi[1] = np.inf
    for arr, op, want in ((lo, "max", [-3.4019e38, -np.inf, -3.4019e38]),
                          (hi, "min", [3.4019e38, np.inf, 3.4019e38])):
        a = arr if trailing else np.ascontiguousarray(arr.T)
        t = tc.tensor(a)
        got = getattr(t, op)([-1] if trailing else [0], False).numpy()
        np.testing.assert_array_equal(got, np.array(want, np.float32))


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_reduce_max_min_finite_values_unchanged(dtype):
    rng = np.random.default_rng(5)
    a = (rng.normal(0, 1, (4, 333)) * np.array([1, 1e3, 1e30, 1e38])[:, None]).astype(np.float32)
    a = np.clip(a, -3.3e38, 3.3e38) if dtype != "float16" else np.clip(a, -6e4, 6e4)
    t = tc.tensor(a, dtype=dtype)
    ref = t.float().numpy()
    np.testing.assert_array_equal(t.max([-1], False).float().numpy(), ref.max(-1))
    np.testing.assert_array_equal(t.min([-1], False).float().numpy(), ref.min(-1))
