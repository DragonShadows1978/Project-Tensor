"""ENG2: unsupported dtype strings are refused (F1), and four norm classes compute half inputs in float32 (N1).

orders/ENG2_DTYPE_REFUSAL_AND_HALF_NORMS.md.

F1: `tc.tensor` and every factory (`zeros`, `ones`, `randn`, `rand`) refuse a dtype that is neither in `_NP_DTYPE` nor in
`_DEVICE_CAST_DTYPE` with TypeError, before any conversion, allocation or random draw; every supported dtype still
builds its own dtype with NumPy's bytes; the default dtype still converts float64 data to float32.
N1: LayerNorm, RMSNorm, BatchNorm1D and InstanceNorm2D (GroupNorm as the control) in float16 and bfloat16, forward and
backward, on E3's `regular` and `near_eps` cases, against E3's float64 NumPy reference at E3's derived tolerances
(unchanged); half inputs follow the registered rule (upcast, float32 arithmetic with float32 weights, one cast back);
float32 forward and backward stay bit-identical to the baseline arm's op sequence.

E3's helpers (e3_inputs, e3_reference, e3_tolerance, e3_distance and the layer/operand builders e3_measure uses) are
IMPORTED, unchanged, from test_eng1.py in this directory (not copied), so the inputs, references and tolerances are
E3's own bytes.

Tests whose name contains "cpu" use only zero-byte tensors or refuse before any tensor exists (no CUDA call reaches
the device) and run with CUDA hidden; every other test needs the GPU. Rows for the lead's cross-arm comparison (the
F1 supported bytes and the P302 float32 bits) and the N1 measurements are written to ENG2_RECORD_DIR when set.

Prior art (unverified -- lead to check the exact citations):
- float32 statistics for half-precision normalization: PyTorch's layer_norm/group_norm/batch_norm kernels accumulate
  half and bfloat16 in float32 (acc_type); Micikevicius et al. (2018), "Mixed Precision Training", ICLR;
- refusing unknown dtypes instead of a silent default: NumPy's np.dtype("...") TypeError; TX3-F1's strict lookup in
  this repository (2026-10);
- accuracy targets n*u (Higham 2002) and the float64 references: E3 (ENG1), reused unchanged;
- bitwise identity compared as same-width unsigned integers: ENG1 E2 (test_e2_pow_grad_bits), GN1;
- DeMillo, Lipton and Sayward (1978), mutation testing: the tests are written to kill the registered mutants.
Ours: the F1 operands (the unconvertible operand and the 2**62-element shapes that prove the refusal comes first),
the half-rule recomposition, the running-statistics tolerance derivation, and the record rows.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc  # noqa: E402

# E3 (ENG1) helpers, imported unchanged from the sibling test_eng1.py.
_SPEC = importlib.util.spec_from_file_location("_eng2_e3_from_test_eng1", Path(__file__).with_name("test_eng1.py"))
E3 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(E3)


def _record(kind, request, tmp_path, row):
    """Write one JSON row for the lead's scorer (cross-arm comparisons and the N1 census)."""
    row = dict(row, kind=kind, test_id=request.node.nodeid)
    out = Path(os.environ.get("ENG2_RECORD_DIR") or tmp_path)
    name = hashlib.sha256(row["test_id"].encode()).hexdigest()[:16]
    (out / f"{kind}_{name}.json").write_text(json.dumps(row, indent=1, sort_keys=True) + "\n")
    print("ENG2-RECORD " + json.dumps(row, sort_keys=True))


# ================================================================================================ F1: dtype refusal
F1_REFUSED = ["float64", "double", "int32", "not_a_dtype"]
F1_SUPPORTED = ["float32", "float16", "int64", "bool", "uint8", "bfloat16"]
F1_MESSAGE = "unsupported dtype {!r}; supported dtypes: float32, float16, int64, bool, uint8, bfloat16"
F1_NP = {"float32": np.float32, "float16": np.float16, "int64": np.int64, "bool": np.bool_, "uint8": np.uint8}
F1_FACTORIES = {"zeros": tc.zeros, "ones": tc.ones, "randn": tc.randn, "rand": tc.rand}
F1_NP_FACTORIES = {"zeros": np.zeros, "ones": np.ones, "randn": lambda s: np.random.randn(*s),
                   "rand": lambda s: np.random.rand(*s)}  # __init__.py's own np_fn for each factory
F1_ENTRIES = ["tensor", *F1_FACTORIES]
# A shape NumPy refuses before allocating (2**62 elements of 8 bytes; "array is too big"): if a factory reached its
# np_fn before checking the dtype, NumPy's ValueError (not the TypeError) would surface. Nothing is allocated.
F1_TOO_BIG = (2 ** 62,)


def _f1_call(entry, dtype, device, shape=(0, 2)):
    if entry == "tensor":
        return tc.tensor(np.zeros(shape), device=device, dtype=dtype)
    return F1_FACTORIES[entry](*shape, device=device, dtype=dtype)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", F1_REFUSED)
@pytest.mark.parametrize("entry", F1_ENTRIES)
def test_f1_refuse_cpu(entry, dtype, device):
    """Refused with TypeError naming the supported dtypes (zero-byte operands: no device call either way)."""
    with pytest.raises(TypeError) as info:
        _f1_call(entry, dtype, device)
    assert str(info.value) == F1_MESSAGE.format(dtype)


class _Unconvertible:
    """An operand NumPy can only convert by calling __array__, which fails the test: the refusal must come first."""

    def __array__(self, *args, **kwargs):
        raise AssertionError("tc.tensor converted its data before checking the dtype")


@pytest.mark.parametrize("dtype", F1_REFUSED)
def test_f1_tensor_refuses_before_conversion_cpu(dtype):
    with pytest.raises(TypeError) as info:
        tc.tensor(_Unconvertible(), device="cpu", dtype=dtype)
    assert str(info.value) == F1_MESSAGE.format(dtype)


@pytest.mark.parametrize("dtype", F1_REFUSED)
@pytest.mark.parametrize("entry", list(F1_FACTORIES))
def test_f1_factory_refuses_before_allocation_cpu(entry, dtype):
    """The dtype is checked before np_fn runs: a shape np_fn would refuse (ValueError) must give the TypeError, and
    the global NumPy random state is untouched."""
    before = np.random.get_state()
    with pytest.raises(TypeError) as info:
        F1_FACTORIES[entry](*F1_TOO_BIG, device="cpu", dtype=dtype)
    assert str(info.value) == F1_MESSAGE.format(dtype)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


@pytest.mark.parametrize("shape", [(0,), (2, 0)], ids=["0", "2x0"])
@pytest.mark.parametrize("dtype", F1_SUPPORTED)
@pytest.mark.parametrize("entry", F1_ENTRIES)
def test_f1_empty_supported_cpu(entry, dtype, shape):
    """Empty shapes keep today's behaviour for every supported dtype (zero bytes; no device call)."""
    t = _f1_call(entry, dtype, "cpu", shape)
    assert t.dtype == dtype and tuple(t.shape) == shape


# ---- supported dtypes on the device: NumPy's own conversion is the reference, bytes compared as unsigned integers.
F1_DATA = np.array([[0.0, 1.0, 2.5, 3.0], [7.0, 100.0, 255.0, 0.3]])  # float64; exact in no half format at 0.3
F1_SEED = 2214
_UINT = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


def _host_bytes(t):
    """The tensor's element bytes as an unsigned-integer array of the element width.
    bfloat16: .numpy() returns the float32 widening (bindings.cpp tensor_to_numpy), compared as uint32.
    bool: tensor_to_numpy has no bool branch; it copies numel one-byte elements into the start of a float32 buffer
    (bindings.cpp tensor_to_numpy, the final else), so the first numel bytes are the device bytes."""
    a = t.numpy()
    n = int(np.prod(tuple(t.shape)))
    if t.dtype == "bool":
        return np.ascontiguousarray(a).view(np.uint8).reshape(-1)[:n].copy()
    return np.ascontiguousarray(a).view(_UINT[a.dtype.itemsize]).reshape(-1)


def _expected_bytes(values64, dtype):
    if dtype == "bfloat16":  # float32 host array, then the device cast: nearest-even to bfloat16, read back widened
        e = E3.e3_round(np.asarray(values64, dtype=np.float32), "bfloat16").astype(np.float32)
    else:
        e = np.asarray(values64).astype(F1_NP[dtype])
    return np.ascontiguousarray(e).view(_UINT[e.dtype.itemsize]).reshape(-1)


def _f1_compare(t, dtype, expected64, shape, request, tmp_path):
    assert t.dtype == dtype and tuple(t.shape) == tuple(shape)  # L09: dtype and shape first
    got, want = _host_bytes(t), _expected_bytes(expected64, dtype)
    assert got.dtype == want.dtype and got.shape == want.shape
    np.testing.assert_array_equal(got, want)
    _record("f1bytes", request, tmp_path, dict(dtype=dtype, shape=list(shape),
                                               sha256=hashlib.sha256(got.tobytes()).hexdigest()))


@pytest.mark.parametrize("dtype", F1_SUPPORTED)
def test_f1_tensor_supported(dtype, request, tmp_path):
    _f1_compare(tc.tensor(F1_DATA, dtype=dtype), dtype, F1_DATA, F1_DATA.shape, request, tmp_path)


@pytest.mark.parametrize("dtype", F1_SUPPORTED)
@pytest.mark.parametrize("entry", list(F1_FACTORIES))
def test_f1_factory_supported(entry, dtype, request, tmp_path):
    shape = (3, 5)
    # NumPy's own conversion of np_fn's float64 output is the reference (as _factory converts it), including
    # randn's negative values cast to uint8 (an "invalid value" cast, the same NumPy call on both sides).
    with np.errstate(invalid="ignore"):
        np.random.seed(F1_SEED)
        t = F1_FACTORIES[entry](*shape, dtype=dtype)
        np.random.seed(F1_SEED)
        expected = F1_NP_FACTORIES[entry](shape)
        _f1_compare(t, dtype, expected, shape, request, tmp_path)


def test_f1_default_float64_data(request, tmp_path):
    """The default dtype ("float32") still converts float64 data to float32."""
    _f1_compare(tc.tensor(F1_DATA), "float32", F1_DATA, F1_DATA.shape, request, tmp_path)


@pytest.mark.parametrize("entry", list(F1_FACTORIES))
def test_f1_factory_default(entry, request, tmp_path):
    np.random.seed(F1_SEED)
    t = F1_FACTORIES[entry](2, 3)
    np.random.seed(F1_SEED)
    _f1_compare(t, "float32", F1_NP_FACTORIES[entry]((2, 3)), (2, 3), request, tmp_path)


# ====================================================================================== N1: half norms in float32
N1_CLASSES = list(E3.E3_CLASSES)  # LayerNorm, RMSNorm, BatchNorm1D, InstanceNorm2D, and GroupNorm (control)
N1_DTYPES = list(E3.E3_DTYPES)
N1_CASES = list(E3.E3_CASES)
N1_DIRECTIONS = ["forward", "backward"]


def n1_measure(cls, dtype, case, direction):
    """E3's e3_measure (test_eng1.py) without its baseline-arithmetic emulation: the same layer, weights, inputs,
    upstream gradient, float64 reference and tolerance."""
    layer, w64, b64 = E3._e3_layer(cls, dtype)
    x = E3._leaf(E3.e3_inputs(cls, case), dtype)
    g = E3._const(E3._e3_pattern(E3.E3_SHAPE[cls], 1.1), dtype)
    x64, g64 = E3._np(x), E3._np(g)
    y = layer(x)
    assert tuple(y.shape) == E3.E3_SHAPE[cls] and y.dtype == dtype
    ref_y, ref_dx = E3.e3_reference(cls, x64, w64, b64, g64)
    if direction == "forward":
        got, ref = E3._np(y), ref_y
    else:
        (y * g).sum().backward()
        assert x.grad.dtype == dtype
        got, ref = E3._np(x.grad), ref_dx
    assert np.all(np.isfinite(ref)) and np.ptp(ref) > 0  # L19
    scale = float(np.max(np.abs(ref)))
    tol = E3.e3_tolerance(cls, direction, dtype, scale)
    finite = bool(np.all(np.isfinite(got)))
    err = float(np.max(np.abs(np.where(np.isfinite(got), got, np.inf) - ref)))
    return dict(cls=cls, dtype=dtype, case=case, direction=direction, finite=finite, max_abs_err=err,
                tolerance=tol, scale=scale, engine_error_vs_float64=E3.e3_distance(got, ref),
                ratio=(err / tol if math.isfinite(err) else None), within=bool(finite and err <= tol),
                engine_nonfinite=int(np.count_nonzero(~np.isfinite(got))))


@pytest.mark.parametrize("direction", N1_DIRECTIONS)
@pytest.mark.parametrize("case", N1_CASES)
@pytest.mark.parametrize("dtype", N1_DTYPES)
@pytest.mark.parametrize("cls", N1_CLASSES)
def test_n1_half_vs_reference(cls, dtype, case, direction, request, tmp_path):
    row = n1_measure(cls, dtype, case, direction)
    _record("n1census", request, tmp_path, row)
    assert row["finite"], f"{row['engine_nonfinite']} non-finite values"
    assert row["max_abs_err"] <= row["tolerance"], (row["max_abs_err"], row["tolerance"])


# ---- the float32 op sequences, written out with the same engine ops.
def _f32_sequence(cls, layer, x, weight, bias):
    """bc02d3a1's nn.py float32 forward of `cls`, verbatim except that the weight and bias are arguments (P302's
    baseline recomposition when they are the layer's own float32 parameters; the half rule's float32 part when x and
    the parameters were upcast)."""
    if cls == "LayerNorm":
        mean = x.mean([-1], True)
        xc = x - mean
        var = (xc * xc).mean([-1], True)
        inv = (var + layer.eps).pow(-0.5)
        return xc * inv * weight + bias
    if cls == "RMSNorm":
        ms = (x * x).mean([-1], True)
        w = weight if weight.dtype == x.dtype else weight.astype(x.dtype)
        return x * (ms + layer.eps).pow(-0.5) * w
    if cls == "BatchNorm1D":  # training mode; the running-statistics update does not feed the output
        axes = [0] if x.ndim == 2 else [0, 2]
        wshape = [1, layer.C] if x.ndim == 2 else [1, layer.C, 1]
        w = weight.reshape(wshape); b = bias.reshape(wshape)
        mean = x.mean(axes, True); var = x.var(axes, True)
        return (x - mean) * (var + layer.eps).pow(-0.5) * w + b
    if cls == "InstanceNorm2D":
        mean = x.mean([2, 3], True); var = x.var([2, 3], True)
        xn = (x - mean) * (var + layer.eps).pow(-0.5)
        return xn * weight.reshape([1, layer.C, 1, 1]) + bias.reshape([1, layer.C, 1, 1])
    assert cls == "GroupNorm"  # GN1's float32 path (nn.py GroupNorm.forward)
    N, C = x.shape[0], x.shape[1]
    spatial = list(x.shape[2:])
    g = x.reshape([N, layer.G, (C // layer.G) * math.prod(spatial)])
    mean = g.mean([2], True); var = g.var([2], True)
    g = (g - mean) * (var + layer.eps).pow(-0.5)
    g = g.reshape([N, C] + spatial)
    wshape = [1, C] + [1] * len(spatial)
    return g * weight.reshape(wshape) + bias.reshape(wshape)


def _half_rule(cls, layer, x, dtype):
    """The registered N1 rule: upcast x, float32 arithmetic with float32 weight and bias, one cast back to dtype."""
    w = layer.weight if layer.weight.dtype == "float32" else layer.weight.astype("float32")
    b = None
    if cls != "RMSNorm":
        b = layer.bias if layer.bias.dtype == "float32" else layer.bias.astype("float32")
    return _f32_sequence(cls, layer, x.astype("float32"), w, b).astype(dtype)


def _bits(t, dtype):
    a = np.ascontiguousarray(t.numpy())  # bfloat16 reads back as its exact float32 widening
    assert t.dtype == dtype
    return a.view(_UINT[a.dtype.itemsize])


def _pair(cls, dtype, case, direction, build):
    """Run the layer and `build` on two leaves with the same stored values; return both results' bits."""
    layer, _, _ = E3._e3_layer(cls, dtype)
    values = E3.e3_inputs(cls, case)
    x1, x2 = E3._leaf(values, dtype), E3._leaf(values, dtype)
    g = E3._const(E3._e3_pattern(E3.E3_SHAPE[cls], 1.1), dtype)
    y1, y2 = layer(x1), build(layer, x2)
    assert y1.dtype == y2.dtype == dtype and tuple(y1.shape) == tuple(y2.shape) == E3.E3_SHAPE[cls]  # L09
    if direction == "backward":
        (y1 * g).sum().backward()
        (y2 * g).sum().backward()
        y1, y2 = x1.grad, x2.grad
        assert y1.dtype == y2.dtype == dtype
    expected = E3._np(y2)
    assert np.all(np.isfinite(expected)) and np.any(expected != 0) and np.ptp(expected) > 0  # L19
    return _bits(y1, dtype), _bits(y2, dtype)


@pytest.mark.parametrize("direction", N1_DIRECTIONS)
@pytest.mark.parametrize("case", N1_CASES)
@pytest.mark.parametrize("cls", N1_CLASSES)
def test_n1_float32_bits(cls, case, direction, request, tmp_path):
    """P302: float32 forward and backward equal the baseline's op sequence bit for bit (uint32)."""
    got, want = _pair(cls, "float32", case, direction,
                      lambda layer, x: _f32_sequence(cls, layer, x, layer.weight, getattr(layer, "bias", None)))
    _record("p302bits", request, tmp_path, dict(cls=cls, case=case, direction=direction,
                                                 sha256=hashlib.sha256(got.tobytes()).hexdigest()))
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("direction", N1_DIRECTIONS)
@pytest.mark.parametrize("case", N1_CASES)
@pytest.mark.parametrize("dtype", N1_DTYPES)
@pytest.mark.parametrize("cls", N1_CLASSES)
def test_n1_half_rule_bits(cls, dtype, case, direction):
    """Half inputs follow the registered rule exactly: the layer equals upcast -> float32 sequence -> one cast."""
    got, want = _pair(cls, dtype, case, direction, lambda layer, x: _half_rule(cls, layer, x, dtype))
    np.testing.assert_array_equal(got, want)


# ---- BatchNorm1D running statistics: float32 tensors, rule (1 - m) * running + m * batch statistic.
# Tolerance (float32, u = 2^-24), for N rows and the stored inputs x (float64 view of the stored values):
#   batch mean: N - 1 serial additions + 1 scaling, each <= u relative to the partial sums: |d_mean| <= (N + 1) u mean|x|;
#   batch var (two-pass, ops.cpp var = mean((x - mean)^2)): subtraction, square, N - 1 additions, scaling:
#   <= (N + 3) u var, plus the mean error's second-order term d_mean^2 (the first-order term cancels: sum(x - mu) = 0);
#   update in NumPy float32: m and 1 - m rounded to float32, one product each, one sum: <= 3 u |result| (+ m * d).
N1_BN_MOMENTUM = 0.1  # BatchNorm1D's default


@pytest.mark.parametrize("case", N1_CASES)
@pytest.mark.parametrize("dtype", ["float32", *N1_DTYPES])
def test_n1_batchnorm_running_stats(dtype, case):
    layer, _, _ = E3._e3_layer("BatchNorm1D", dtype)
    assert layer.momentum == N1_BN_MOMENTUM
    rm0, rv0 = E3._np(layer.running_mean), E3._np(layer.running_var)
    x = E3._const(E3.e3_inputs("BatchNorm1D", case), dtype)
    layer(x)
    rm, rv = layer.running_mean, layer.running_var
    C = E3.E3_SHAPE["BatchNorm1D"][1]
    assert rm.dtype == rv.dtype == "float32" and tuple(rm.shape) == tuple(rv.shape) == (C,)
    x64 = E3._np(x)
    N, m, u = x64.shape[0], N1_BN_MOMENTUM, 2.0 ** -24
    mu, var = x64.mean(axis=0), x64.var(axis=0)
    want_m, want_v = (1 - m) * rm0 + m * mu, (1 - m) * rv0 + m * var
    reference = np.concatenate([want_m, want_v])
    assert np.all(np.isfinite(reference)) and np.all(reference != 0) and np.ptp(reference) > 0  # L19
    d_mean = (N + 1) * u * np.abs(x64).mean(axis=0)
    tol_m = m * d_mean + 3 * u * np.abs(want_m) + 2.0 ** -149
    tol_v = m * ((N + 3) * u * var + d_mean ** 2) + 3 * u * np.abs(want_v) + 2.0 ** -149
    err_m, err_v = np.abs(E3._np(rm) - want_m), np.abs(E3._np(rv) - want_v)
    assert np.all(err_m <= tol_m), (err_m.max(), tol_m.min())
    assert np.all(err_v <= tol_v), (err_v.max(), tol_v.min())


# ================================================================================== G2: reference self-checks (CPU)
def test_g2_references_hand_values_cpu():
    """The imported E3 references against hand-computed values on constructed inputs."""
    x = np.array([[1.0, 3.0]])
    y, dx = E3.e3_reference("LayerNorm", x, np.ones(2), np.zeros(2), np.array([[1.0, 0.0]]))
    s = math.sqrt(1.0 + 1e-5)  # mean 2, variance 1
    np.testing.assert_allclose(y, [[-1 / s, 1 / s]], rtol=1e-15)
    # dx = (g - mean(g) - xh * mean(g * xh)) / s with g = [1, 0], xh = [-1, 1] / s
    xh = np.array([-1.0, 1.0]) / s
    want = (np.array([1.0, 0.0]) - 0.5 - xh * np.mean(np.array([1.0, 0.0]) * xh)) / s
    np.testing.assert_allclose(dx, [want], rtol=1e-15)
    y, dx = E3.e3_reference("RMSNorm", np.array([[3.0, 4.0]]), np.ones(2), None, np.array([[1.0, 1.0]]))
    rho = math.sqrt(12.5 + 1e-6)
    np.testing.assert_allclose(y, [[3 / rho, 4 / rho]], rtol=1e-15)
    np.testing.assert_allclose(dx, [[1 / rho - 3 * 3.5 / rho ** 3, 1 / rho - 4 * 3.5 / rho ** 3]], rtol=1e-14)
    assert E3.e3_tolerance("LayerNorm", "forward", "float16", 2.0) == 11 * 2.0 ** -11 * 2.0 + 2.0 ** -25
    assert E3.e3_distance(np.array([1.0, np.inf]), np.array([1.0, 2.0])) is None
    assert E3.e3_distance(np.array([1.0, 2.5]), np.array([1.0, 2.0])) == 0.5
    inputs = E3.e3_inputs("RMSNorm", "near_eps")
    np.testing.assert_allclose((inputs * inputs).mean(axis=-1), 1e-6, rtol=1e-12)
