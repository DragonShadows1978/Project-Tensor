"""ENG1: six queued tensor_cuda defects (orders/ENG1_ENGINE_DEFECTS.md).

E1 reshape -1 inference, E2 half-precision pow backward, E3 the norm layers' half-precision census (measured, not
gated), E4 empty extents, E5 pooling geometry refusals, E6 the terrain test's writes into the source tree.

Every value is checked against a float64 NumPy reference written in this file, never against the engine itself;
every refusal is checked by its exact message. Tests whose name contains "cpu" use only zero-byte CPU tensors (no
CUDA call reaches the device) and run with CUDA hidden; every other test needs the GPU.

Prior art (unverified -- lead to check the exact citations):
- PyTorch's reshape/view -1 inference and its refusals (ATen InferSize.h, "only one dimension can be inferred");
- mixed-precision arithmetic: float32 math for half-precision tensors (Micikevicius et al., 2018, "Mixed Precision
  Training", ICLR);
- accuracy targets (not error bounds) of the form n*u (Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed. 2002,
  section 3.1, gamma_n); IEEE 754 unit roundoff u = 2^-p for a p-bit significand;
- the normalization VJP (Ioffe and Szegedy, 2015, batch norm; Ba, Kiros and Hinton, 2016, layer norm; Zhang and
  Sennrich, 2019, RMSNorm; Wu and He, 2018, group norm), re-derived here;
- PyTorch's topk "selected index k out of range", softmax on empty dimensions, and MaxPool2d/AvgPool2d output-size
  rule (floor quotient, "Output size is too small");
- DeMillo, Lipton and Sayward (1978), mutation testing: the tests are written to kill the registered mutants.
Ours: the operand constructions, the tolerance assignments and the E6 write-detection record.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc  # noqa: E402

# The directory that holds the imported tensor_cuda package (an arm copy under the staged tooling, or this
# worktree's tensor_cuda/). Child processes import the same package from here.
PKG_PARENT = Path(tc.__file__).resolve().parents[1]

# ----------------------------------------------------------------------------------------------- tolerances
# Unit roundoff u = 2^-p for a p-bit significand (IEEE 754): float16 p = 11, bfloat16 p = 8, float32 p = 24.
U = {"float16": 2.0 ** -11, "bfloat16": 2.0 ** -8, "float32": 2.0 ** -24}
# Half of the smallest subnormal spacing: the absolute rounding error of a result that lands in the subnormal range.
# float16 subnormal spacing 2^-24, bfloat16 2^-133, float32 2^-149.
SUB_HALF = {"float16": 2.0 ** -25, "bfloat16": 2.0 ** -134, "float32": 2.0 ** -150}
# CUDA powf maximum error, 4 ulp (CUDA C++ Programming Guide, single-precision function table; unverified -- lead to
# check), i.e. a relative error <= 4 * 2^-23 = 2^-21.
POWF_REL = 4 * 2.0 ** -23
# E2 float32 chain: powf (<= 2^-21) then two float32 roundings (nmuls, nmul; <= 2^-24 each): 2^-21 + 2^-23 < 2^-20.
E2_RTOL_F32 = 2.0 ** -20
assert POWF_REL + 2 * U["float32"] < E2_RTOL_F32
# E2 half dtypes: the float32 chain above, then one rounding to the dtype: (u_d + e32 + u_d * e32) < u_d + 2^-20.
E2_RTOL = {"float16": U["float16"] + E2_RTOL_F32, "bfloat16": U["bfloat16"] + E2_RTOL_F32, "float32": E2_RTOL_F32}
E2_ATOL = {"float16": SUB_HALF["float16"], "bfloat16": SUB_HALF["bfloat16"], "float32": SUB_HALF["float32"]}
# E5 float32 sums of at most kh*kw = 9 terms plus one division (avg) or at most 9 scatter-adds (backward):
# <= 10 roundings of 2^-24, bounded by 16 * 2^-24 = 2^-20 relative (gamma_n, all terms of one sign).
E5_RTOL = 2.0 ** -20


def _np(t):
    return np.asarray(t.numpy(), dtype=np.float64)


def _leaf(values, dtype, device="cuda"):
    """A same-dtype leaf with requires_grad (tc.tensor builds bfloat16 by a device cast, which is not a leaf)."""
    t = tc.tensor(np.asarray(values, dtype=np.float64), dtype=dtype, device=device).detach()
    t.requires_grad = True
    return t


def _const(values, dtype, device="cuda"):
    return tc.tensor(np.asarray(values, dtype=np.float64), dtype=dtype, device=device).detach()


def _cpu_empty(shape, dtype=np.float32):
    return tc.from_numpy(np.zeros(shape, dtype=dtype), device="cpu")


# ======================================================================================== E1: reshape -1
E1_SOURCE = (2, 12)  # 24 elements
E1_VALID = {
    "r1_p0": [-1],
    "r2_p0": [-1, 6], "r2_p1": [4, -1],
    "r3_p0": [-1, 3, 4], "r3_p1": [2, -1, 4], "r3_p2": [2, 3, -1],
    "r4_p0": [-1, 3, 2, 2], "r4_p1": [2, -1, 2, 2], "r4_p2": [2, 3, -1, 2], "r4_p3": [2, 3, 2, -1],
}
E1_EMPTY_VALID = {  # zero-element sources: the -1 infers 0 or the product's complement
    "from0_r2_p0": ((0,), [-1, 4]),
    "from0_r2_p1": ((0,), [3, -1]),
    "from2x0_r1_p0": ((2, 0), [-1]),
    "from0x3_r3_p1": ((0, 3), [3, -1, 1]),
}
MSG_TWO = "reshape: only one dimension can be inferred (-1)"
MSG_ZERO = "reshape: cannot infer -1 when the other extents' product is 0"
MSG_COUNT = "reshape: element count mismatch"
E1_REFUSALS = {
    "two_neg1": ([-1, -1], MSG_TWO),
    "three_neg1": ([-1, 2, -1, -1], MSG_TWO),
    "other_negative": ([-2, -12], "reshape: invalid shape dimension -2"),
    "negative_with_neg1": ([-1, -3], "reshape: invalid shape dimension -3"),
    "zero_product": ([0, -1], MSG_ZERO),
    "non_integer": ([5, -1], "reshape: cannot infer -1: 24 elements are not divisible by the other extents' product 5"),
    "count_mismatch": ([5, 5], MSG_COUNT),
}
E1_EMPTY_REFUSALS = {
    "two_neg1": ((0,), [-1, -1], MSG_TWO),
    "other_negative": ((0,), [-2, 0], "reshape: invalid shape dimension -2"),
    "zero_product": ((0,), [0, -1], MSG_ZERO),
    "count_mismatch": ((0,), [2], MSG_COUNT),
}


def _e1_reference(src_shape, target):
    """float64 NumPy reference: NumPy's own -1 inference on the same element sequence."""
    return np.arange(math.prod(src_shape), dtype=np.float64).reshape(src_shape).reshape(target)


def test_e1_reference_hand_values_cpu():
    """The reference against hand-computed shapes and values (G2)."""
    r = _e1_reference((2, 12), [2, 3, -1, 2])
    assert r.shape == (2, 3, 2, 2)
    assert r[1, 2, 1, 1] == 23.0 and r[0, 1, 0, 1] == 5.0
    assert _e1_reference((0,), [3, -1]).shape == (3, 0)


@pytest.mark.parametrize("case", list(E1_VALID), ids=list(E1_VALID))
def test_e1_infer(case):
    target = E1_VALID[case]
    ref = _e1_reference(E1_SOURCE, target)
    assert ref.std() > 0  # L19: non-constant reference
    x = tc.tensor(np.arange(24, dtype=np.float32).reshape(E1_SOURCE))
    y = x.reshape(target)
    assert tuple(y.shape) == ref.shape
    np.testing.assert_array_equal(_np(y), ref)


@pytest.mark.parametrize("case", list(E1_EMPTY_VALID), ids=list(E1_EMPTY_VALID))
def test_e1_infer_empty_cpu(case):
    src, target = E1_EMPTY_VALID[case]
    ref = np.zeros(src).reshape(target)
    y = _cpu_empty(src).reshape(target)
    assert tuple(y.shape) == ref.shape


@pytest.mark.parametrize("case", list(E1_REFUSALS), ids=list(E1_REFUSALS))
def test_e1_refuse(case):
    target, message = E1_REFUSALS[case]
    x = tc.tensor(np.arange(24, dtype=np.float32).reshape(E1_SOURCE))
    with pytest.raises(RuntimeError) as info:
        x.reshape(target)
    assert str(info.value) == message


@pytest.mark.parametrize("case", list(E1_EMPTY_REFUSALS), ids=list(E1_EMPTY_REFUSALS))
def test_e1_refuse_empty_cpu(case):
    src, target, message = E1_EMPTY_REFUSALS[case]
    with pytest.raises(RuntimeError) as info:
        _cpu_empty(src).reshape(target)
    assert str(info.value) == message


def test_e1_view_shares_storage():
    """Storage sharing, shown by a write through the inferred view."""
    base = np.arange(24, dtype=np.float64)
    x = tc.tensor(base.astype(np.float32))
    v = x.reshape([2, -1, 3])
    assert tuple(v.shape) == (2, 4, 3)
    assert v.data_ptr() == x.data_ptr()
    np.testing.assert_array_equal(_np(x), base)
    v.fill_(7.5)
    np.testing.assert_array_equal(_np(x), np.full(24, 7.5))


# ======================================================================================== E2: pow backward
E2_P = [-1.5, -0.5, 0.5, 2.0, 3.0]
E2_DTYPES = ["float16", "bfloat16", "float32"]
E2_X = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 7.0]
# L55 witness (float32): x = 1.3671875 (exact, 8 significant bits, so x^2 is exact), g = 0.7 rounded to float32,
# p = 3. Reordering g * (p * x^2) as (g * p) * x^2 changes the float32 result's bits (seat CPU check, report).
E2_WITNESS_X = 1.3671875
E2_WITNESS_G = float(np.float32(0.7))
E2_ENGINE_DTYPE = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}
# ENG2-A3: the engine has no float64, so no float64 rows here; test_eng2.py tests the refusal of a float64 request.


def e2_reference_grad(x64, w64, p):
    """float64 reference of d/dx sum(w * x^p) = w * p * x^(p-1)."""
    return w64 * p * np.power(x64, p - 1.0)


def e2_scale_exponents(x64, p):
    """Power-of-two weights w = 2^-k bring |p x^(p-1)| down to at most about 2^10, so the true gradient fits
    float16 (max 65504); w is exact in every dtype for k <= 24 (2^-24 is float16's smallest subnormal)."""
    mag = abs(p) * np.power(x64, p - 1.0)
    return np.maximum(0.0, np.ceil(np.log2(mag)) - 10.0)


def _e2_inputs(dtype, p):
    edt = E2_ENGINE_DTYPE[dtype]
    probe = _const(E2_X, dtype)
    assert probe.dtype == edt
    x64 = _np(probe)  # the stored (dtype-rounded) inputs
    k = e2_scale_exponents(x64, p)
    # float16 needs w >= 2^-24 (its smallest subnormal); a point needing more scaling has a true gradient that no
    # float16 weight can bring into range, so it is left out for float16 only (p = -1.5 at x = 1e-5 and 3e-5).
    keep = k <= 24 if edt == "float16" else np.ones(k.shape, dtype=bool)
    x64, k = x64[keep], k[keep]
    assert x64.size >= 8
    return x64, np.power(2.0, -k)


def test_e2_reference_hand_values_cpu():
    """The reference against hand-computed derivatives (G2)."""
    g = e2_reference_grad(np.array([4.0, 0.25]), np.array([1.0, 2.0]), -0.5)
    np.testing.assert_allclose(g, [-0.5 * 4.0 ** -1.5, 2.0 * -0.5 * 0.25 ** -1.5], rtol=0, atol=0)
    assert g[0] == -0.0625 and g[1] == -8.0
    assert e2_reference_grad(np.array([3.0]), np.array([1.0]), 2.0)[0] == 6.0


@pytest.mark.parametrize("p", E2_P, ids=[f"p{p:+g}" for p in E2_P])
@pytest.mark.parametrize("dtype", E2_DTYPES)
def test_e2_pow_grad(dtype, p):
    """Gradient of sum(w * x^p) against the float64 reference, small x included; finite where the true value is."""
    x64, w64 = _e2_inputs(dtype, p)
    edt = E2_ENGINE_DTYPE[dtype]
    x = _leaf(x64, edt)
    w = _const(w64, edt)
    np.testing.assert_array_equal(_np(w), w64)  # w is exact
    (x.pow(p) * w).sum().backward()
    assert x.grad.dtype == edt
    got = _np(x.grad)
    ref = e2_reference_grad(x64, w64, p)
    assert np.all(np.isfinite(ref)) and np.ptp(ref) > 0  # L19
    assert np.all(np.isfinite(got)), f"non-finite gradient at x={x64[~np.isfinite(got)]}"
    np.testing.assert_allclose(got, ref, rtol=E2_RTOL[edt], atol=E2_ATOL[edt])


# Prior art: IEEE 754 nearest-even, mixed precision (Micikevicius et al., 2018); REV1 coverage_gaps.py
# supplies the bfloat16 witness. Ours: explicit non-power-of-two upstream IDs using the unchanged E2 derivation.
E2_NONPOW2 = {
    "float16": (0.1492919921875, 0.30078125, 3.0),
    "bfloat16": (0.0002994537353515625, 0.30078125, 3.0),
}


@pytest.mark.parametrize("dtype", list(E2_NONPOW2))
def test_e2_pow_grad_nonpow2(dtype):
    xs, gs, p = E2_NONPOW2[dtype]
    x, g = _leaf([xs], dtype), _const([gs], dtype)
    x64, g64 = _np(x), _np(g)
    np.testing.assert_array_equal(x64, [xs])
    np.testing.assert_array_equal(g64, [gs])
    assert math.frexp(gs)[0] != 0.5
    (x.pow(p) * g).sum().backward()
    assert x.grad.dtype == dtype
    ref = e2_reference_grad(x64, g64, p)
    # Same registered derivation: fp32 powf/products, then ONE dtype rounding, plus final subnormal allowance.
    np.testing.assert_allclose(_np(x.grad), ref, rtol=E2_RTOL[dtype], atol=E2_ATOL[dtype])


@pytest.mark.parametrize("p", E2_P, ids=[f"p{p:+g}" for p in E2_P])
@pytest.mark.parametrize("dtype", ["float32"])
def test_e2_pow_grad_bits(dtype, p):
    """float32 (and the float64 request, which the engine builds as float32) keeps the baseline arm's arithmetic bit
    for bit: the backward equals nmul(g, nmuls(ew_pow(x, p - 1), p)) recomposed from the same engine ops, compared
    as uint32. Non-power-of-two upstream values (and the L55 witness) make a reordering visible."""
    xs = np.array(E2_X + [E2_WITNESS_X])
    gs = np.array([0.7 + 0.013 * i for i in range(len(E2_X))] + [E2_WITNESS_G])
    x = _leaf(xs, dtype)
    assert x.dtype == "float32"
    g = _const(gs, dtype)
    (x.pow(p) * g).sum().backward()
    got = np.asarray(x.grad.numpy(), dtype=np.float32)
    expected = np.asarray((g * (x.detach().pow(p - 1.0) * p)).numpy(), dtype=np.float32)
    np.testing.assert_array_equal(got.view(np.uint32), expected.view(np.uint32))
    ref = e2_reference_grad(_np(x), _np(g), p)
    np.testing.assert_allclose(got.astype(np.float64), ref, rtol=E2_RTOL_F32, atol=SUB_HALF["float32"])


# ======================================================================================== E3: half census
# Measurement only (orders/ENG1 E3): frozen n*u numbers are accuracy targets (not error bounds); each test passes
# whenever the engine ran and the reference is valid; the census verdict is reported by the lead, not gated here.
E3_DTYPES = ["float16", "bfloat16"]
E3_CLASSES = ["LayerNorm", "RMSNorm", "BatchNorm1D", "InstanceNorm2D", "GroupNorm"]
E3_CASES = ["regular", "near_eps"]
E3_EPS = {"LayerNorm": 1e-5, "RMSNorm": 1e-6, "BatchNorm1D": 1e-5, "InstanceNorm2D": 1e-5, "GroupNorm": 1e-5}
E3_SHAPE = {"LayerNorm": (4, 16), "RMSNorm": (4, 16), "BatchNorm1D": (16, 4), "InstanceNorm2D": (2, 3, 4, 4),
            "GroupNorm": (2, 4, 3, 3)}
# Roundings to the dtype on the longest dependency path, counted from nn.py's forward (mean = sum + scale: 2,
# var = mean, sub, mul, mean: 6, then add eps, pow, mul, weight, bias): (n_dtype, n_float32) per direction; the
# backward counts two roundings per forward op on the path (its VJP product and the accumulation). GroupNorm upcasts
# half inputs to float32 (GN1 A1), so its half-dtype roundings are the final cast (forward) and the input cast's VJP.
E3_ROUNDINGS = {
    "LayerNorm": {"forward": (11, 0), "backward": (22, 0)},
    "RMSNorm": {"forward": (7, 0), "backward": (14, 0)},
    "BatchNorm1D": {"forward": (11, 0), "backward": (22, 0)},
    "InstanceNorm2D": {"forward": (11, 0), "backward": (22, 0)},
    "GroupNorm": {"forward": (1, 11), "backward": (1, 22)},
}


def e3_tolerance(cls, direction, dtype, scale):
    """Frozen accuracy targets (not error bounds): n_d*u_d*scale + n_32*u_32*scale + subnormal half-spacing.
    Longest-path n*u (Higham 2002) omits conditioning, cancellation and intermediate subnormal amplification."""
    n_d, n_32 = E3_ROUNDINGS[cls][direction]
    return (n_d * U[dtype] + n_32 * U["float32"]) * scale + SUB_HALF[dtype]


def _e3_pattern(shape, phase):
    n = math.prod(shape)
    return np.cos(np.arange(n, dtype=np.float64) * 1.7 + phase).reshape(shape)


def _e3_axes(cls, shape):
    return {"LayerNorm": (-1,), "BatchNorm1D": (0,), "InstanceNorm2D": (2, 3)}.get(cls)


def _standardize(s, axes):
    s = s - s.mean(axis=axes, keepdims=True)
    return s / np.sqrt((s * s).mean(axis=axes, keepdims=True))


def e3_inputs(cls, case):
    """Fixed constructions. regular: offset 0.25 plus unit-variance spread; near_eps: offset 0.5 plus a spread whose
    variance (mean square for RMSNorm) equals the class eps over every normalized set."""
    shape, eps = E3_SHAPE[cls], E3_EPS[cls]
    s = _e3_pattern(shape, 0.3)
    if cls == "RMSNorm":
        s = s / np.sqrt((s * s).mean(axis=-1, keepdims=True))
        return s if case == "regular" else math.sqrt(eps) * s
    if cls == "GroupNorm":
        N, C = shape[0], shape[1]
        s5 = _standardize(s.reshape(N, 2, C // 2, *shape[2:]), (2, 3, 4)).reshape(shape)
    else:
        s5 = _standardize(s, _e3_axes(cls, shape))
    return 0.25 + s5 if case == "regular" else 0.5 + math.sqrt(eps) * s5


def _affine_shape(cls, shape):
    C = shape[-1] if cls in ("LayerNorm", "RMSNorm") else shape[1]
    b = [1] * len(shape)
    b[-1 if cls in ("LayerNorm", "RMSNorm") else 1] = C
    return C, tuple(b)


def e3_reference(cls, x, w, b, g):
    """float64 forward and VJP (gradient of sum(y * g) with respect to x)."""
    shape, eps = x.shape, E3_EPS[cls]
    C, bshape = _affine_shape(cls, shape)
    wb = w.reshape(bshape)
    bb = None if b is None else b.reshape(bshape)
    if cls == "RMSNorm":
        rho = np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
        y = x / rho * wb
        gw = g * wb
        dx = gw / rho - x * (gw * x).mean(axis=-1, keepdims=True) / rho ** 3
        return y, dx
    if cls == "GroupNorm":
        N, G = shape[0], 2
        rs = (N, G, C // G) + shape[2:]
        xs, gs, axes = x.reshape(rs), (g * wb).reshape(rs), (2, 3, 4)
    else:
        xs, gs, axes = x, g * wb, _e3_axes(cls, shape)
    mu = xs.mean(axis=axes, keepdims=True)
    xc = xs - mu
    sig = np.sqrt((xc * xc).mean(axis=axes, keepdims=True) + eps)
    xh = xc / sig
    dx = (gs - gs.mean(axis=axes, keepdims=True) - xh * (gs * xh).mean(axis=axes, keepdims=True)) / sig
    y = xh.reshape(shape) * wb + bb
    return y, dx.reshape(shape)



# Prior art: IEEE 754-2019 roundTiesToEven; NumPy float16; bfloat16 high-word tie-to-even rounding.
# Reverse-mode tape: Wengert (1964); operation order/accumulation from this engine's ops.cpp/autograd.cpp.
# REV1 numerics.py supplies the forward emulation comparison. Ours: all five forward/VJP graphs, explicit
# engine reduction ordering, and three-way census distances. External citations unverified -- lead to check.
def e3_round(values, dtype):
    """Load/compute in fp32, then store nearest-even in dtype; returned float64 represents stored bits exactly."""
    a = np.asarray(values, dtype=np.float32)
    if dtype == "float16":
        return a.astype(np.float16).astype(np.float64)
    if dtype == "float32":
        return a.astype(np.float64)
    assert dtype == "bfloat16"
    bits = a.view(np.uint32).astype(np.uint64)
    rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000).astype(np.uint32)
    # Preserve NaNs (including small payloads that would otherwise round to infinity).
    rounded = np.where(np.isnan(a), np.uint32(0x7fc00000), rounded).astype(np.uint32)
    return rounded.view(np.float32).astype(np.float64)


def _e3_sum(values, axes, dtype):
    """kernels.cu reduce_impl: fp32 256-lane trailing tree, otherwise serial reduced-coordinate order."""
    a = np.asarray(values, dtype=np.float32)
    axes = tuple(sorted(d % a.ndim for d in axes))
    kept = tuple(d for d in range(a.ndim) if d not in axes)
    rows = a.transpose(kept + axes).reshape(math.prod(a.shape[d] for d in kept), -1)
    if axes == tuple(range(min(axes), a.ndim)) and rows.shape[1] > 1:
        lanes = np.zeros((rows.shape[0], 256), dtype=np.float32)
        for j in range(0, rows.shape[1], 256):
            chunk = rows[:, j:j + 256]
            lanes[:, :chunk.shape[1]] += chunk
        for off in (128, 64, 32, 16, 8, 4, 2, 1):
            lanes[:, :off] += lanes[:, off:2 * off]
        total = lanes[:, 0]
    else:
        total = np.zeros(rows.shape[0], dtype=np.float32)
        for j in range(rows.shape[1]):
            total += rows[:, j]
    shape = tuple(1 if d in axes else a.shape[d] for d in range(a.ndim))
    return e3_round(total.reshape(shape), dtype)


class _E3Arithmetic:
    """Small CPU-only tape of exactly the primitive operations used by the registered norm graphs."""
    def __init__(self, value, dtype, parents=(), vjp=None, requires_grad=True):
        self.value, self.dtype = e3_round(value, dtype), dtype
        self.parents, self.vjp, self.grad = parents, vjp, None
        self.requires_grad = requires_grad

    def accumulate(self, grad):
        if not self.requires_grad:
            return
        shape = (1,) * (grad.ndim - self.value.ndim) + self.value.shape
        axes = tuple(d for d, (n, m) in enumerate(zip(grad.shape, shape)) if m == 1 and n > 1)
        if axes:
            grad = _e3_sum(grad, axes, self.dtype)
        grad = grad.reshape(self.value.shape)
        self.grad = grad if self.grad is None else e3_round(self.grad + grad, self.dtype)

    def binary(self, other, op):
        assert self.dtype == other.dtype
        a, b, dt = self.value, other.value, self.dtype
        def vjp(g):
            if op == "mul":
                self.accumulate(e3_round(g * b, dt))
                other.accumulate(e3_round(g * a, dt))
            else:
                self.accumulate(g)
                other.accumulate(g if op == "add" else e3_round(-g, dt))
        value = a * b if op == "mul" else a + b if op == "add" else a - b
        return _E3Arithmetic(value, dt, (self, other), vjp)

    def __mul__(self, other): return self.binary(other, "mul")
    def __sub__(self, other): return self.binary(other, "sub")
    def __add__(self, other): return self.binary(other, "add")

    def scalar(self, scalar, multiply=False):
        scalar = float(np.float32(scalar))  # ew_scalar's float parameter, not a half scalar
        vjp = lambda g: self.accumulate(e3_round(g * scalar, self.dtype) if multiply else g)
        value = self.value * scalar if multiply else self.value + scalar
        return _E3Arithmetic(value, self.dtype, (self,), vjp)

    def sum(self, axes):
        def vjp(g):
            self.accumulate(e3_round(np.zeros_like(self.value) + g, self.dtype))
        return _E3Arithmetic(_e3_sum(self.value, axes, self.dtype), self.dtype, (self,), vjp)

    def mean(self, axes):
        return self.sum(axes).scalar(1.0 / math.prod(self.value.shape[d] for d in axes), multiply=True)

    def reshape(self, shape):
        return _E3Arithmetic(self.value.reshape(shape), self.dtype, (self,),
                             lambda g: self.accumulate(g.reshape(self.value.shape)))

    def cast(self, dtype):
        return _E3Arithmetic(self.value, dtype, (self,), lambda g: self.accumulate(e3_round(g, self.dtype)))

    def pow(self, p):
        def vjp(g):
            # E2: widened power, scalar multiply, upstream multiply, then one final half cast.
            power = e3_round(np.power(self.value, np.float32(p - 1)), "float32")
            scaled = e3_round(power * float(np.float32(p)), "float32")
            self.accumulate(e3_round(e3_round(g * scaled, "float32"), self.dtype))
        # Correctly rounded fp32 power models honest arithmetic, not CUDA powf's implementation error.
        value = e3_round(np.power(self.value, np.float32(p)), "float32")
        return _E3Arithmetic(value, self.dtype, (self,), vjp)

    def backward(self):
        order, visited = [], set()
        def visit(node):
            if node in visited or not node.requires_grad:
                return
            visited.add(node)
            for parent in node.parents:
                visit(parent)
            order.append(node)
        visit(self)
        self.accumulate(np.ones_like(self.value))
        for node in reversed(order):
            if node.vjp is not None and node.grad is not None:
                node.vjp(node.grad)


def e3_emulation(cls, dtype, x, w, b, g):
    """Same nn.py sequence and ops.cpp VJPs, rounded at every primitive/storage/gradient accumulation.

    CPU IEEE model only: no CUDA context and no claim of bit identity to CUDA powf. Reductions follow kernels.cu.
    Batch/Instance/GroupNorm compute separate means for var(), as their source does; LayerNorm reuses xc.
    """
    leaf = _E3Arithmetic(x, dtype)
    work = leaf.cast("float32") if cls == "GroupNorm" else leaf
    shape = work.value.shape
    if cls == "GroupNorm":
        work = work.reshape((shape[0], 2, -1))
    axes = (-1,) if cls in ("RMSNorm", "LayerNorm", "GroupNorm") else _e3_axes(cls, shape)
    if cls == "RMSNorm":
        centered, stat = work, (work * work).mean(axes)
    else:
        mean = work.mean(axes)
        centered = work - mean
        variance_input = centered if cls == "LayerNorm" else work - work.mean(axes)
        stat = (variance_input * variance_input).mean(axes)
    normalized = centered * stat.scalar(E3_EPS[cls]).pow(-0.5)
    if cls == "GroupNorm":
        normalized = normalized.reshape(shape)
    _, bshape = _affine_shape(cls, shape)
    weight = _E3Arithmetic(w.reshape(bshape), work.dtype, requires_grad=False)
    y = normalized * weight
    if b is not None:
        y = y + _E3Arithmetic(b.reshape(bshape), work.dtype, requires_grad=False)
    if cls == "GroupNorm":
        y = y.cast(dtype)
    upstream = _E3Arithmetic(g, dtype, requires_grad=False)
    (y * upstream).sum(tuple(range(len(shape)))).backward()
    return y.value, leaf.grad


def e3_distance(a, b, *, finite_only=False):
    """Maximum absolute distance; optionally measure only jointly finite elements."""
    # Prior art: NumPy isfinite/boolean masking/max; REV2 R2-F02 (2026).
    # Taken: finite-mask distance; new here: the optional E3 census readout.
    finite = np.isfinite(a) & np.isfinite(b)
    if finite_only:
        if not np.any(finite):
            return None
        return float(np.max(np.abs(a[finite] - b[finite])))
    if not np.all(finite):
        return None
    return float(np.max(np.abs(a - b)))


def _e3_layer(cls, dtype):
    shape = E3_SHAPE[cls]
    C, _ = _affine_shape(cls, shape)
    layer = {"LayerNorm": lambda: tc.nn.LayerNorm(C), "RMSNorm": lambda: tc.nn.RMSNorm(C),
             "BatchNorm1D": lambda: tc.nn.BatchNorm1D(C), "InstanceNorm2D": lambda: tc.nn.InstanceNorm2D(C),
             "GroupNorm": lambda: tc.nn.GroupNorm(2, C)}[cls]()
    w = _const(1.0 + 0.1 * np.arange(C), dtype)
    layer.weight = w
    b = None
    if cls != "RMSNorm":
        b = _const(0.05 * np.arange(C) - 0.1, dtype)
        layer.bias = b
    return layer, _np(w), (None if b is None else _np(b))


def e3_measure(cls, dtype, case, direction):
    layer, w64, b64 = _e3_layer(cls, dtype)
    x = _leaf(e3_inputs(cls, case), dtype)
    g = _const(_e3_pattern(E3_SHAPE[cls], 1.1), dtype)
    x64, g64 = _np(x), _np(g)
    y = layer(x)
    assert tuple(y.shape) == E3_SHAPE[cls] and y.dtype == dtype
    ref_y, ref_dx = e3_reference(cls, x64, w64, b64, g64)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        emu_y, emu_dx = e3_emulation(cls, dtype, x64, w64, b64, g64)
    if direction == "forward":
        got, ref, emu = _np(y), ref_y, emu_y
    else:
        (y * g).sum().backward()
        assert x.grad.dtype == dtype
        got, ref, emu = _np(x.grad), ref_dx, emu_dx
    assert np.all(np.isfinite(ref)) and np.ptp(ref) > 0  # L19
    scale = float(np.max(np.abs(ref)))
    tol = e3_tolerance(cls, direction, dtype, scale)
    finite = bool(np.all(np.isfinite(got)))
    err = float(np.max(np.abs(np.where(np.isfinite(got), got, np.inf) - ref)))
    axes = (2, 3, 4) if cls == "GroupNorm" else (-1,) if cls == "RMSNorm" else _e3_axes(cls, x64.shape)
    xr = x64.reshape(x64.shape[0], 2, -1, *x64.shape[2:]) if cls == "GroupNorm" else x64
    stat = (xr * xr).mean(axis=axes) if cls == "RMSNorm" else xr.var(axis=axes)
    return dict(cls=cls, dtype=dtype, case=case, direction=direction, finite=finite, max_abs_err=err,
                tolerance=tol, target_label="accuracy targets (not error bounds)",
                engine_error_vs_float64=e3_distance(got, ref),
                emulation_error_vs_float64=e3_distance(emu, ref),
                engine_distance_from_emulation=e3_distance(got, emu),
                # Prior art: REV2 R2-F02 (2026), NumPy array_equal(equal_nan=True).
                # Taken verbatim: the prescribed pattern comparison; measurement only.
                nonfinite_pattern_equal=bool(np.array_equal(
                    np.where(np.isfinite(got), 0, got), np.where(np.isfinite(emu), 0, emu), equal_nan=True)),
                engine_distance_from_emulation_finite=e3_distance(got, emu, finite_only=True),
                engine_nonfinite=int(np.count_nonzero(~np.isfinite(got))),
                emulation_nonfinite=int(np.count_nonzero(~np.isfinite(emu))),
                emulation_rounding="IEEE 754 nearest-even per primitive; fp32 reductions/pow, dtype stores",
                scale=scale, within=bool(finite and err <= tol),
                ratio=(err / tol if math.isfinite(err) else None), eps=E3_EPS[cls],
                stored_stat_min=float(np.min(stat)), stored_stat_max=float(np.max(stat)))


@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("case", E3_CASES)
@pytest.mark.parametrize("dtype", E3_DTYPES)
@pytest.mark.parametrize("cls", E3_CLASSES)
def test_e3_census(cls, dtype, case, direction, tmp_path, request):
    row = e3_measure(cls, dtype, case, direction)
    row["test_id"] = request.node.nodeid
    out = Path(os.environ.get("ENG1_CENSUS_DIR") or tmp_path)
    name = hashlib.sha256(row["test_id"].encode()).hexdigest()[:16]
    (out / f"census_{name}.json").write_text(json.dumps(row, indent=1) + "\n")
    print("ENG1-CENSUS " + json.dumps(row, sort_keys=True))


# ======================================================================================== E4: empty extents
def _child(code):
    """Run `code` in a child interpreter on the same tensor_cuda package; a crash (e.g. SIGFPE) is a failure here."""
    pre = ("import sys, json; sys.path.insert(0, %r); import numpy as np, tensor_cuda as tc\n"
           "cpu = lambda s, d=np.float32: tc.from_numpy(np.zeros(s, dtype=d), device='cpu')\n"
           "cuda = lambda s, d='float32': tc.zeros(s, dtype=d)\n") % str(PKG_PARENT)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-B", "-c", pre + code], capture_output=True, text=True, timeout=300,
                          cwd=str(PKG_PARENT), env=env)


def _report(expr_list):
    return ("out = []\n"
            "for r in [%s]:\n"
            "    out.append([list(r.shape), r.dtype])\n"
            "print('E4-RESULT ' + json.dumps(out))\n") % ", ".join(expr_list)


# name: (devices, code building the call and binding `res` to a tuple of result tensors, expected [(shape, dtype)])
E4_EMPTY = {
    "rms_norm-last0": ("cpu cuda", "res = (tc._C.rms_norm({d}((2, 0)), {d}((0,)), 1e-6),)", [((2, 0), "float32")]),
    "rms_norm-last0-f16": ("cpu", "res = (tc._C.rms_norm({d}((3, 1, 0), np.float16), {d}((0,)), 1e-6),)",
                           [((3, 1, 0), "float16")]),
    "rms_norm-rows0": ("cuda", "res = (tc._C.rms_norm({d}((0, 4)), {d}((4,)), 1e-6),)", [((0, 4), "float32")]),
    "causal_softmax-last0": ("cpu cuda", "res = (tc._C.causal_softmax({d}((2, 0, 0))),)", [((2, 0, 0), "float32")]),
    "causal_softmax-rows0": ("cpu cuda", "res = (tc._C.causal_softmax({d}((0, 4, 4))),)", [((0, 4, 4), "float32")]),
    "apa_blend_softmax-last0": ("cpu cuda", "res = (tc._C.apa_blend_softmax({d}((1, 1, 2, 0)), {d}((1, 1, 2, 0)), 0.0),)",
                                [((1, 1, 2, 0), "float32")]),
    "apa_blend_softmax-rows0": ("cpu cuda", "res = (tc._C.apa_blend_softmax({d}((1, 0, 2, 3)), {d}((1, 0, 2, 3)), 0.0),)",
                                [((1, 0, 2, 3), "float32")]),
    "topk-last0-k0": ("cpu cuda", "res = {d}((2, 0)).topk(0)", [((2, 0), "float32"), ((2, 0), "int64")]),
    "topk-rows0": ("cpu cuda", "res = {d}((0, 5)).topk(2)", [((0, 2), "float32"), ((0, 2), "int64")]),
    "topk-k0": ("cuda", "res = {d}((2, 5)).topk(0)", [((2, 0), "float32"), ((2, 0), "int64")]),
}
E4_EMPTY_IDS = [f"{name}-{dev}" for name, (devs, _, _) in E4_EMPTY.items() for dev in devs.split()]
E4_REFUSE = {
    "causal_softmax-S_lt_L-kept": ("cpu", "tc._C.causal_softmax({d}((2, 3, 0)))",
                                   "causal_softmax: S < L (queries outnumber keys)"),
    "topk-k1-last0": ("cpu cuda", "{d}((2, 0)).topk(1)", "topk: selected index k=1 out of range for last extent 0"),
    "topk-k4-last3": ("cuda", "{d}((2, 3)).topk(4)", "topk: selected index k=4 out of range for last extent 3"),
    "topk-kneg-last0": ("cpu", "{d}((2, 0)).topk(-1)", "topk: selected index k=-1 out of range for last extent 0"),
}
E4_REFUSE_IDS = [f"{name}-{dev}" for name, (devs, _, _) in E4_REFUSE.items() for dev in devs.split()]


@pytest.mark.parametrize("case", E4_EMPTY_IDS)
def test_e4_empty(case):
    name, dev = case.rsplit("-", 1)
    _, code, expected = E4_EMPTY[name]
    p = _child(code.format(d=dev) + "\n" + _report(["*res"]).replace("[*res]", "res"))
    assert p.returncode == 0, f"rc={p.returncode} stderr={p.stderr[-800:]}"
    line = [s for s in p.stdout.splitlines() if s.startswith("E4-RESULT ")]
    assert len(line) == 1, p.stdout
    got = [(tuple(s), d) for s, d in json.loads(line[0][len("E4-RESULT "):])]
    assert got == [(tuple(s), d) for s, d in expected]


@pytest.mark.parametrize("case", E4_REFUSE_IDS)
def test_e4_refuse(case):
    name, dev = case.rsplit("-", 1)
    _, code, message = E4_REFUSE[name]
    p = _child("try:\n    " + code.format(d=dev) + "\n    print('E4-NOREFUSAL')\n"
               "except RuntimeError as e:\n    print('E4-REFUSED ' + json.dumps(str(e)))\n")
    assert p.returncode == 0, f"rc={p.returncode} stderr={p.stderr[-800:]}"
    line = [s for s in p.stdout.splitlines() if s.startswith("E4-REFUSED ")]
    assert len(line) == 1, p.stdout
    assert json.loads(line[0][len("E4-REFUSED "):]) == message


def test_e4_topk_values():
    """A non-empty topk still matches the float64 NumPy reference (values and indices; distinct inputs)."""
    a = np.array([[0.5, -1.25, 3.0, 2.0, 0.75], [4.0, 1.0, -2.0, 6.5, 0.0]])
    v, i = tc.tensor(a).topk(2)
    order = np.argsort(-a, axis=1, kind="stable")[:, :2]
    np.testing.assert_array_equal(np.asarray(i.numpy()), order)
    np.testing.assert_array_equal(_np(v), np.take_along_axis(a, order, axis=1))


# ======================================================================================== E5: pooling dims
E5_MSG = ("conv/pool: input extents (H, W) = ({H}, {W}), kernel ({kh}, {kw}), stride ({sh}, {sw}) and padding "
          "({ph}, {pw}) give output extents ({OH}, {OW}); both must be positive")
E5_REFUSE = {  # x shape (N = 0: zero bytes), kh, kw, sh, sw, ph, pw, expected OH, OW (floor quotient)
    "both_negative": ((0, 1, 2, 2), 5, 5, 1, 1, 0, 0, -2, -2),
    "oh_zero": ((0, 1, 2, 2), 3, 1, 1, 1, 0, 0, 0, 2),
    "ow_only": ((0, 1, 6, 2), 3, 5, 1, 1, 0, 0, 4, -2),
    "ow_zero": ((0, 1, 6, 2), 3, 3, 1, 1, 0, 0, 4, 0),  # isolate OW<=0 with OH>0 (REV1 F04)
    "trunc_gap": ((0, 1, 3, 3), 4, 4, 2, 2, 0, 0, 0, 0),
    "padded": ((0, 2, 2, 2), 6, 6, 1, 1, 1, 1, -1, -1),
}
E5_CALLERS = ["im2col", "col2im", "avg_pool2d", "max_pool2d"]


def e5_out_extent(n, k, s, p):
    """float64-free integer reference: floor((n + 2p - k) / s) + 1 (Python // is the floor quotient)."""
    return (n + 2 * p - k) // s + 1


def test_e5_reference_hand_values_cpu():
    assert e5_out_extent(2, 5, 1, 0) == -2 and e5_out_extent(3, 4, 2, 0) == 0 and e5_out_extent(5, 3, 2, 1) == 3
    for x, kh, kw, sh, sw, ph, pw, oh, ow in E5_REFUSE.values():
        assert (e5_out_extent(x[2], kh, sh, ph), e5_out_extent(x[3], kw, sw, pw)) == (oh, ow)


def _e5_call(caller, x_shape, kh, kw, sh, sw, ph, pw):
    N, C, H, W = x_shape
    if caller == "im2col":
        return tc._C.im2col(_cpu_empty(x_shape), kh, kw, sh, sw, ph, pw)
    if caller == "col2im":
        return tc._C.col2im(_cpu_empty((N, C * kh * kw, 1)), N, C, H, W, kh, kw, sh, sw, ph, pw)
    return getattr(tc._C, caller)(_cpu_empty(x_shape), kh, kw, sh, sw, ph, pw)


@pytest.mark.parametrize("case", list(E5_REFUSE))
@pytest.mark.parametrize("caller", E5_CALLERS)
def test_e5_refuse_cpu(caller, case):
    x, kh, kw, sh, sw, ph, pw, oh, ow = E5_REFUSE[case]
    with pytest.raises(RuntimeError) as info:
        _e5_call(caller, x, kh, kw, sh, sw, ph, pw)
    assert str(info.value) == E5_MSG.format(H=x[2], W=x[3], kh=kh, kw=kw, sh=sh, sw=sw, ph=ph, pw=pw, OH=oh, OW=ow)


E5_VALID = {"k2s2p0": (2, 2, 0), "k3s2p1": (3, 2, 1), "k3s1p0": (3, 1, 0)}


def e5_windows(x, k, s, p):
    """float64 reference windows: padded input (NaN marks padding) and the output extents."""
    N, C, H, W = x.shape
    xp = np.full((N, C, H + 2 * p, W + 2 * p), np.nan)
    xp[:, :, p:p + H, p:p + W] = x
    OH, OW = e5_out_extent(H, k, s, p), e5_out_extent(W, k, s, p)
    win = np.empty((N, C, OH, OW, k, k))
    for i in range(OH):
        for j in range(OW):
            win[:, :, i, j] = xp[:, :, i * s:i * s + k, j * s:j * s + k]
    return win


def e5_reference(x, k, s, p, r):
    """Forward (max over real elements; average over k*k with padding counted as 0, count_include_pad) and the
    gradients of sum(y * r) with respect to x."""
    N, C, H, W = x.shape
    win = e5_windows(x, k, s, p)
    mx = np.nanmax(win, axis=(4, 5))
    av = np.nansum(win, axis=(4, 5)) / (k * k)
    gmax = np.zeros((N, C, H + 2 * p, W + 2 * p))
    gavg = np.zeros_like(gmax)
    OH, OW = mx.shape[2:]
    for n in range(N):
        for c in range(C):
            for i in range(OH):
                for j in range(OW):
                    w = win[n, c, i, j]
                    a, b = np.unravel_index(np.nanargmax(w), w.shape)
                    gmax[n, c, i * s + a, j * s + b] += r[n, c, i, j]
                    gavg[n, c, i * s:i * s + k, j * s:j * s + k] += r[n, c, i, j] / (k * k)
    crop = (slice(None), slice(None), slice(p, p + H), slice(p, p + W))
    return mx, av, gmax[crop], gavg[crop]


def test_e5_reference_hand_values_cpu_pool():
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    mx, av, gm, ga = e5_reference(x, 2, 2, 0, np.array([[[[2.0]]]]))
    assert mx[0, 0, 0, 0] == 4.0 and av[0, 0, 0, 0] == 2.5
    np.testing.assert_array_equal(gm[0, 0], [[0.0, 0.0], [0.0, 2.0]])
    np.testing.assert_array_equal(ga[0, 0], [[0.5, 0.5], [0.5, 0.5]])
    mx, av, _, _ = e5_reference(x, 2, 1, 1, np.ones((1, 1, 3, 3)))
    assert mx[0, 0, 0, 0] == 1.0 and av[0, 0, 0, 0] == 0.25 and av[0, 0, 1, 1] == 2.5


def _e5_input():
    v = (np.arange(50, dtype=np.float64) * 37 % 50) / 50.0 + 0.01  # 50 distinct values, no max ties
    return v.reshape(1, 2, 5, 5)


@pytest.mark.parametrize("geom", list(E5_VALID))
def test_e5_valid_pool(geom):
    k, s, p = E5_VALID[geom]
    xt = _leaf(_e5_input(), "float32")
    x64 = _np(xt)
    OH = e5_out_extent(5, k, s, p)
    r64 = 0.5 + _e3_pattern((1, 2, OH, OH), 0.7) ** 2  # positive upstream: no cancellation in the scatter-adds
    mx, av, gmax, gavg = e5_reference(x64, k, s, p, r64)
    assert np.ptp(mx) > 0 and np.ptp(gavg) > 0  # L19
    r = _const(r64, "float32")
    r64 = _np(r)
    mx, av, gmax, gavg = e5_reference(x64, k, s, p, r64)
    ym = tc._C.max_pool2d(xt, k, k, s, s, p, p)
    np.testing.assert_array_equal(_np(ym), mx)
    (ym * r).sum().backward()
    np.testing.assert_allclose(_np(xt.grad), gmax, rtol=E5_RTOL, atol=0)
    xa = _leaf(x64, "float32")
    ya = tc._C.avg_pool2d(xa, k, k, s, s, p, p)
    np.testing.assert_allclose(_np(ya), av, rtol=E5_RTOL, atol=0)
    (ya * r).sum().backward()
    np.testing.assert_allclose(_np(xa.grad), gavg, rtol=E5_RTOL, atol=0)


@pytest.mark.parametrize("geom", list(E5_VALID))
def test_e5_valid_im2col(geom):
    k, s, p = E5_VALID[geom]
    x64 = _np(_const(_e5_input(), "float32"))
    win = np.nan_to_num(e5_windows(x64, k, s, p), nan=0.0)  # (N, C, OH, OW, k, k), padding reads 0
    N, C, OH, OW = win.shape[:4]
    ref = win.transpose(0, 1, 4, 5, 2, 3).reshape(N, C * k * k, OH * OW)
    cols = tc._C.im2col(_const(x64, "float32"), k, k, s, s, p, p)
    np.testing.assert_array_equal(_np(cols), ref)


# ======================================================================================== E6: terrain writes
E6_WRITERS = [
    "test_terrain_render_wo12g_cpu_repro_and_receipts",
    "test_terrain_render_wo9br2_ppm_ladder",
    "test_terrain_render_wo9e_ppm_receipts",
    "test_terrain_render_wo9ae_ppm_receipts",
    "test_terrain_render_wo8a_ppm_pair",
    "test_terrain_render_wo8c_ppm_triplet",
]


def _git_root(path):
    p = subprocess.run(["git", "-C", str(path), "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                       timeout=60)
    assert p.returncode == 0, p.stderr
    return Path(p.stdout.strip())


def e6_snapshot(target):
    """The write-detection record (L46): `git status --porcelain --ignored -- artifacts/` as ordered, minus the lines
    for ENG1's own receipt tree artifacts/eng1/ (a staged run writes its receipts there while this test runs); the
    same with -uall restricted to the write target (an arm copy's target lies inside artifacts/eng1/); and a stat map
    (size, mtime_ns; sha256 for .ppm) of every file under the target. git status alone cannot see a rewrite with
    identical bytes of a tracked receipt (ledger F4)."""
    root = _git_root(PKG_PARENT)
    rel = os.path.relpath(target, root)
    exclude = [":(exclude)artifacts/eng1"] if rel == "artifacts" else []

    def git(*args):
        p = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--ignored", *args], capture_output=True,
                           text=True, timeout=120)
        assert p.returncode == 0, p.stderr
        if args[:1] == ("--",):
            return "".join(s + "\n" for s in p.stdout.splitlines() if not s[3:].startswith("artifacts/eng1/"))
        return p.stdout

    stat = {}
    if target.is_dir():
        for dirpath, dirnames, filenames in os.walk(target):
            if rel == "artifacts" and Path(dirpath) == target:
                dirnames[:] = [d for d in dirnames if d != "eng1"]
            for f in filenames:
                fp = Path(dirpath) / f
                st = fp.stat()
                row = [st.st_size, st.st_mtime_ns]
                if f.endswith(".ppm"):
                    row.append(hashlib.sha256(fp.read_bytes()).hexdigest())
                stat[str(fp.relative_to(target))] = row
    return dict(target=rel, ordered=git("--", "artifacts/"), uall=git("-uall", "--", rel, *exclude), stat=stat)


def test_e6_terrain_writes_nothing(tmp_path):
    terrain = PKG_PARENT / "tests" / "test_terrain_render.py"
    assert terrain.is_file()
    target = terrain.resolve().parents[2] / "artifacts"
    before = e6_snapshot(target)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    args = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(tmp_path),
            "--basetemp", str(tmp_path / "basetemp")] + [f"{terrain}::{name}" for name in E6_WRITERS]
    p = subprocess.run(args, capture_output=True, text=True, timeout=1500, cwd=str(tmp_path), env=env)
    after = e6_snapshot(target)
    record = dict(terrain=str(terrain), returncode=p.returncode, tail=p.stdout[-2000:], before=before, after=after)
    out = Path(os.environ.get("ENG1_CENSUS_DIR") or tmp_path)
    (out / "e6_record.json").write_text(json.dumps(record, indent=1) + "\n")
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert after["ordered"] == before["ordered"]
    assert after["uall"] == before["uall"]
    changed = sorted(k for k in set(before["stat"]) | set(after["stat"]) if before["stat"].get(k) != after["stat"].get(k))
    assert not changed, f"the terrain file wrote into the source tree: {changed}"
