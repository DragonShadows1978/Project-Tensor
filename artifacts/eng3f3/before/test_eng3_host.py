"""ENG3: host tensors without a GPU (orders/ENG3_HOST_TENSOR_CORE.md, ENG3_R2_RESUME.md Amendment 1, lead
Amendment 2).

G1 (tests named *cpu*; they call no CUDA API and pass with CUDA_VISIBLE_DEVICES empty): every E2 member of a host
tensor works on every supported dtype and equals a NumPy reference built by E3's rules -- from_host through tc.tensor,
to_host through .numpy(), astype through .astype, and zeros / full / clone / to (host to host) through the private
test hooks _C._core_zeros / _core_full / _core_clone / _core_to (A1.1). test_g1_cpu_no_cuda_calls counts calls
through the module's global libcudart imports after import (artifacts/eng3/tools/cudart_count.c), with a positive
control. It cannot see direct-handle, driver-API or other-library calls. An independent child runs the host paths
while CUDA runtime initialization fails with CUDA_VISIBLE_DEVICES empty; host errors fail and last-error is checked
where queryable. E4 (A1.2: core members only): fill_ still refuses a host tensor.
G1b (Amendment 2): bool .numpy() returns exactly the input values, on host tensors (CPU) and device tensors (GPU).
G2 (tests named *g2*, GPU): for every dtype pair NDArray::astype dispatches (A1.4), the host cast's bits equal the
device cast's bits on the same input bits, over the order's value census plus 10,000 seeded random finite values per
source type, compared per element on same-width unsigned views after dtype/shape checks and L19's nonzero/non-constant
check of each side.

E3's NumPy reference below re-implements the device rules (derived in kernels.cu's ENG3 block from cvt.rn.f16.f32,
cvt.f32.f16, cvt.rn.bf16.f32, mov.b32 {0,h}, rintf/fmaxf/fminf); its NaN outputs are the ones the ENG3 device probe
measured on sm_89 Build B over all 2,046 half NaNs and 1,099 distinct float NaN words (artifacts/eng3/probe/).
Host/device bits agree on that census and architecture; no claim for other architectures (default CMake includes sm_90). half -> float uses NumPy's own exact widening, a second implementation.

Rows are written to ENG3_RECORD_DIR (JSON) when it is set.

Prior art (unverified in this sandbox -- lead to check): IEEE 754-2008 binary16/binary32 and roundTiesToEven; the
bfloat16 format (Google Brain); CUDA Math API __float2half_rn / __float2bfloat16_rn and the PTX ISA cvt instruction;
LD_PRELOAD + dlsym interposition (ltrace-style call counting); DeMillo, Lipton and Sayward (1978), mutation testing
(the tests are written to kill m01-m05). Ours: the value census, the scoped interposition measurement, the host/device
same-bits harness.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_PARENT = os.environ.get("ENG3F1_PACKAGE_PARENT", os.path.dirname(HERE))
sys.path.insert(0, PKG_PARENT)
import tensor_cuda as tc  # noqa: E402

_C = tc._C
ROOT = os.path.dirname(os.path.dirname(HERE))
SHIM = os.path.join(ROOT, "artifacts", "eng3", "tools", "cudart_count.so")
SEED = 20261116

# NaN outputs of the device casts (ENG3 probe on Build B, artifacts/eng3/probe/device_probe.npz).
NAN32_FROM_F16 = 0x7FFFFFFF
NAN16 = 0x7FFF
NANBF = 0x7FFF

FLOATS = ("float32", "float16", "bfloat16")
CASTABLE = FLOATS + ("uint8",)
ALL = CASTABLE + ("int64", "bool")
REFUSE = "op supports float32/float16/bfloat16 only"
UINT = {"float32": np.uint32, "float16": np.uint16, "bfloat16": np.uint16, "uint8": np.uint8, "int64": np.uint64,
        "bool": np.uint8}

# A1.4: the pairs NDArray::astype dispatches (kernels.cu NDArray::astype; line numbers in artifacts/eng3/A14_PAIRS.json)
PAIRS_FLOAT = [(s, d) for s in FLOATS for d in FLOATS if s != d]           # cast_kernel<ST, DT>
PAIRS_TO_U8 = [(s, "uint8") for s in FLOATS]                              # cast_to_u8_kernel<ST>
PAIRS_FROM_U8 = [("uint8", d) for d in FLOATS]                            # cast_from_u8_kernel<DT>
PAIRS_SAME = [(s, s) for s in ALL]                                        # dt == dtype -> clone()
PAIRS_CAST = PAIRS_FLOAT + PAIRS_TO_U8 + PAIRS_FROM_U8 + PAIRS_SAME
PAIRS_REFUSED = [(s, d) for s in ALL for d in ALL if s != d and (s, d) not in PAIRS_CAST]


def record(name, row):
    d = os.environ.get("ENG3_RECORD_DIR")
    if not d:
        return
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + ".jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


# ----------------------------------------------------------------------------------------- E3 reference (NumPy)
def ref_f32_to_f16(u):
    u = u.astype(np.int64)
    sign = (u >> 16) & 0x8000
    exp = (u >> 23) & 0xFF
    man = u & 0x7FFFFF
    out = np.zeros(u.shape, np.int64)
    e = exp - 127
    is_nan = (exp == 0xFF) & (man != 0)
    is_inf = ((exp == 0xFF) & (man == 0)) | ((exp != 0xFF) & (exp != 0) & (e > 15))
    is_norm = (exp != 0xFF) & (exp != 0) & (e >= -14) & (e <= 15)
    is_sub = (exp != 0) & (e < -14)
    h = ((e + 15) << 10) | (man >> 13)
    rem = man & 0x1FFF
    h = h + ((rem > 0x1000) | ((rem == 0x1000) & ((h & 1) == 1))).astype(np.int64)
    m = man | 0x800000
    shift = np.clip(-(e + 1), 1, 40)
    q = m >> shift
    r = m & ((np.int64(1) << shift) - 1)
    half = np.int64(1) << (shift - 1)
    q = q + ((r > half) | ((r == half) & ((q & 1) == 1))).astype(np.int64)
    out = np.where(is_norm, sign | h, out)
    out = np.where(is_sub, sign | q, out)
    out = np.where(exp == 0, sign, out)
    out = np.where(is_inf, sign | 0x7C00, out)
    out = np.where(is_nan, NAN16, out)
    return out.astype(np.uint16)


def ref_f16_to_f32(h):
    h = h.astype(np.uint16)
    nan = ((h & 0x7C00) == 0x7C00) & ((h & 0x3FF) != 0)
    with np.errstate(invalid="ignore"):
        v = h.view(np.float16).astype(np.float32).view(np.uint32)  # NumPy's exact widening
    return np.where(nan, np.uint32(NAN32_FROM_F16), v).astype(np.uint32)


def ref_f32_to_bf16(u):
    u = u.astype(np.uint64)
    nan = (u & 0x7FFFFFFF) > 0x7F800000
    r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return np.where(nan, NANBF, r).astype(np.uint16)


def ref_ld(dt, bits):
    """The engine's ld<T>: float32 bits."""
    if dt == "float32":
        return bits.astype(np.uint32)
    if dt == "float16":
        return ref_f16_to_f32(bits)
    if dt == "bfloat16":
        return bits.astype(np.uint32) << 16
    raise AssertionError(dt)


def ref_st(dt, f32bits):
    if dt == "float32":
        return f32bits.astype(np.uint32)
    if dt == "float16":
        return ref_f32_to_f16(f32bits)
    if dt == "bfloat16":
        return ref_f32_to_bf16(f32bits)
    raise AssertionError(dt)


def ref_to_u8(f32bits):
    v = f32bits.astype(np.uint32).view(np.float32)
    with np.errstate(invalid="ignore"):
        r = np.rint(v)                      # round half to even, like rintf
        r = np.where(np.isnan(r), np.float32(0), r)
        r = np.clip(r, 0, 255)
    return r.astype(np.uint8)


def ref_astype(src, dst, bits):
    if src == dst:
        return bits.copy()
    if dst == "uint8" and src in FLOATS:
        return ref_to_u8(ref_ld(src, bits))
    if src == "uint8" and dst in FLOATS:
        return ref_st(dst, bits.astype(np.float32).view(np.uint32))
    if src in FLOATS and dst in FLOATS:
        return ref_st(dst, ref_ld(src, bits))
    raise AssertionError((src, dst))


# ------------------------------------------------------------------------------------------- tensors from bits
def make(dt, bits, device):
    """A tensor of dtype dt holding exactly `bits` (bfloat16: bits whose float32 widening is exact; no NaN payload)."""
    if dt == "float32":
        return tc.tensor(bits.astype(np.uint32).view(np.float32), device=device, dtype="float32")
    if dt == "float16":
        return tc.tensor(bits.astype(np.uint16).view(np.float16), device=device, dtype="float16")
    if dt == "bfloat16":
        return tc.tensor((bits.astype(np.uint32) << 16).view(np.float32), device=device, dtype="bfloat16")
    if dt == "uint8":
        return tc.tensor(bits.astype(np.uint8), device=device, dtype="uint8")
    if dt == "int64":
        return tc.tensor(bits.astype(np.uint64).view(np.int64), device=device, dtype="int64")
    if dt == "bool":
        return tc.tensor(bits.astype(np.uint8).view(np.bool_), device=device, dtype="bool")
    raise AssertionError(dt)


def bits_of(t):
    """Same-width unsigned view of a tensor's elements (bfloat16 through its exact float32 widening)."""
    a = t.numpy()
    if t.dtype == "bfloat16":
        w = a.view(np.uint32)
        assert a.dtype == np.float32 and not (w & 0xFFFF).any(), "bfloat16 widening must be a 16-bit shift"
        return (w >> 16).astype(np.uint16)
    want = {"float32": np.float32, "float16": np.float16, "uint8": np.uint8, "int64": np.int64, "bool": np.bool_}
    assert a.dtype == want[t.dtype], (t.dtype, a.dtype)
    return a.view(UINT[t.dtype]).copy()


def l19(bits, label):
    """L19: each side of a bitwise comparison is nonzero and non-constant, at full width."""
    assert bits.size > 1 and bits.any(), f"{label}: all-zero operand"
    assert (bits != bits.flat[0]).any(), f"{label}: constant operand"
    assert bits.dtype.kind == "u", f"{label}: must use full-width unsigned view"
    record("l19", {"label": label, "dtype": str(bits.dtype), "n": int(bits.size)})


def identity(got, want, label):
    """ENG3-F1 R5: guarded identity at the actual unsigned element width.

    Prior art: review's mixed-corpus protocol (2026); NumPy unsigned views.
    Taken: guard both operands before comparing. Ours: shared assertion site.
    """
    assert got.dtype == want.dtype and got.shape == want.shape, label
    l19(got, label + " got")
    l19(want, label + " reference")
    assert np.array_equal(got, want), label


def constant_identity(got, want, expected, label):
    """R5's full exception: an independently built NumPy constant is the reference.
    Verify its full-width bits before comparing; a nonconstant guard cannot apply to a constant fill.
    """
    assert got.dtype == want.dtype and got.shape == want.shape
    assert got.dtype.kind == "u" and want.dtype.kind == "u"
    assert np.all(want == expected), (label, "NumPy constant reference bits")
    assert np.array_equal(got, want), label
    record("constant_reference", {"label": label, "dtype": str(want.dtype), "bits": hex(int(expected))})


# ------------------------------------------------------------------------------------------------ value census
def _f32(x):
    return np.array([x], np.float64).astype(np.float32).view(np.uint32)[0]


def census(src, gpu=False):
    """Input bits per source dtype: the order's G2 census (specials, destination ties, just past each overflow
    threshold, the uint8 list) plus 10,000 seeded random finite values. gpu=True adds the bfloat16 NaN payloads,
    which only a DLPack import can make (see make_pair)."""
    rng = np.random.default_rng(SEED + {"float32": 0, "float16": 1, "bfloat16": 2, "uint8": 3}[src])
    u8_list = [-1.0, -0.5, 0.5, 1.5, 254.5, 255.5, 256.0, 2.5, -0.0, 0.0, 1e10, -1e10, 0.49999997, 126.5, 127.5]
    ties16 = [1 + 2**-11, 1 + 3 * 2**-11, 2049.0, 2051.0, 65520.0, -65520.0, 0.5 * 2**-24, 1.5 * 2**-24,
              2.5 * 2**-24, 1023.5 * 2**-24, -1.5 * 2**-24, 1 + 2**-11 + 2**-30]
    ties_bf = [1 + 2**-8, 1 + 3 * 2**-8, -(1 + 2**-8), 256.5, 3 * 2**-133]
    over16 = [65504.0, float(np.nextafter(np.float32(65520), np.float32(0))), 65520.0,
              float(np.nextafter(np.float32(65520), np.float32(np.inf))), 65536.0, 1e5, -1e5, 65792.0]
    vals = u8_list + ties16 + ties_bf + over16
    if src == "float32":
        spec = [0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000, 0x7F7FFFFF, 0xFF7FFFFF,
                0x7F800000, 0xFF800000, NAN32_FROM_F16, 0x7FC00000, 0x7FC01234, 0xFF812345, 0x7F800001,
                0x00008000, 0x00018000, 0x00017FFF, 0x7F7F7FFF, 0x7F7F8000, 0x7F7F8001, 0xFF7F8000]
        cand = np.array(spec + [_f32(v) for v in vals], np.uint32)
        r = rng.integers(0, 2**32, size=40000, dtype=np.uint64).astype(np.uint32)
        r = r[(r & 0x7F800000) != 0x7F800000][:10000]
    elif src == "float16":
        spec = [0, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF, 0x0400, 0x8400, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00,
                NAN16, 0x7C01, 0xFE12, 0x7E00]
        with np.errstate(over="ignore"):
            exact = [v for v in vals if np.isfinite(np.float16(v)) and float(np.float16(v)) == v]
        cand = np.array(spec + [int(np.float16(v).view(np.uint16)) for v in exact], np.uint16)
        r = rng.integers(0, 2**16, size=40000, dtype=np.uint64).astype(np.uint16)
        r = r[(r & 0x7C00) != 0x7C00][:10000]
    elif src == "bfloat16":
        spec = [0, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x0080, 0x8080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, NANBF,
                0x477F, 0x4780, 0x4781, 0x3380, 0x33C0, 0x3420]  # 0x33xx/0x34xx: half-subnormal ties (2^-25 ...)
        if gpu:
            spec += [0x7F81, 0xFFC1]  # NaN payloads (DLPack only)
        exact = []
        for v in vals:
            b = int(_f32(v))
            if (b & 0xFFFF) == 0 and (b & 0x7F800000) != 0x7F800000:
                exact.append(b >> 16)
        cand = np.array(spec + exact, np.uint16)
        r = rng.integers(0, 2**16, size=40000, dtype=np.uint64).astype(np.uint16)
        r = r[(r & 0x7F80) != 0x7F80][:10000]
    elif src == "uint8":
        cand = np.arange(256, dtype=np.uint8)
        r = rng.integers(0, 256, size=10000, dtype=np.uint64).astype(np.uint8)
    else:
        raise AssertionError(src)
    assert r.size == 10000
    return np.concatenate([cand, r])


def census_simple(dt):
    """int64 / bool identity census (clone pairs)."""
    rng = np.random.default_rng(SEED + 7)
    if dt == "int64":
        spec = np.array([0, 1, 2**63 - 1, 2**63, 2**64 - 1, 12345], np.uint64)
        return np.concatenate([spec, rng.integers(0, 2**63, size=10000, dtype=np.int64).astype(np.uint64)])
    if dt == "bool":
        return np.concatenate([np.array([0, 1], np.uint8), rng.integers(0, 2, size=10000).astype(np.uint8)])
    return census(dt)


# =================================================================================================== G1 (CPU)
@pytest.mark.parametrize("dt", ["float32", "float16", "bfloat16", "uint8", "int64"])
def test_g1_cpu_tensor_numpy(dt):
    """from_host (tc.tensor) and to_host (.numpy()) of a host tensor; bfloat16 creation is a host float32->bfloat16
    astype (E3), so its reference is ref_f32_to_bf16 of the float32 input."""
    bits = census_simple(dt)
    if dt == "bfloat16":
        f32 = census("float32")
        f32 = f32[(f32 & 0x7F800000) != 0x7F800000]  # finite inputs: the float32 -> bfloat16 rule of tc.tensor
        t = tc.tensor(f32.view(np.float32), device="cpu", dtype="bfloat16")
        want = ref_f32_to_bf16(f32)
    else:
        t = make(dt, bits, "cpu")
        want = bits.astype(UINT[dt])
    assert t.device == "cpu" and t.dtype == dt and t.shape == want.shape
    got = bits_of(t)
    identity(got, want, f"tensor/numpy {dt}")
    record("g1", {"test": "tensor_numpy", "dtype": dt, "n": int(got.size), "mismatches": int((got != want).sum())})


@pytest.mark.parametrize("case", ["n1_true", "n5_pattern", "n8_pattern", "n33_random", "n1000_random", "n1000_true",
                                  "n1000_false", "n33_false"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_g1b_bool_numpy(device, case):
    """G1b (Amendment 2): bool .numpy() returns exactly the input values (dtype bool, one byte per element). The cpu
    rows are part of G1's CPU run; the cuda rows run on the GPU line."""
    n = int(case.split("_")[0][1:])
    kind = case.split("_")[1]
    rng = np.random.default_rng(SEED + n)
    if kind == "true":
        x = np.ones(n, np.bool_)
    elif kind == "false":
        x = np.zeros(n, np.bool_)
    elif kind == "pattern":
        x = np.array(([True, False, True, True, False] * 2)[:n], np.bool_)
    else:
        x = rng.integers(0, 2, size=n).astype(np.bool_)
    t = tc.tensor(x, device=device, dtype="bool")
    a = t.numpy()
    assert a.dtype == np.bool_, a.dtype
    assert a.shape == x.shape
    # Include a separately constructed real tensor with both bool values so even
    # the scalar/all-False/all-True cases form informative full-width operands.
    companion = tc.tensor(np.array([False, True], np.bool_), device=device, dtype="bool").numpy()
    assert companion.dtype == np.bool_ and companion.shape == (2,)
    identity(np.concatenate([a.view(np.uint8), companion.view(np.uint8)]),
             np.concatenate([x.view(np.uint8), np.array([0, 1], np.uint8)]), "bool " + case)
    record("g1b", {"device": device, "case": case, "n": n, "ok": True})


@pytest.mark.parametrize("dt", ALL)
def test_g1_cpu_zeros(dt):
    """NDArray::zeros through _core_zeros (host memset), and the public tc.zeros (NumPy zeros + from_host)."""
    for shape in ([7], [3, 5], [0], []):
        z = _C._core_zeros(shape, dt, "cpu")
        assert z.device == "cpu" and z.dtype == dt and z.shape == tuple(shape)
        assert not bits_of(z).any() if dt != "bool" else not z.numpy().any()
    p = tc.zeros(4, 3, device="cpu", dtype=dt) if dt != "bfloat16" else tc.zeros(4, 3, device="cpu", dtype=dt)
    assert p.dtype == dt and not bits_of(p).any()


FULL_VALUES = [1.0, -0.0, 1.5, 65520.0, 65519.0, 1 + 2**-8, 1 + 3 * 2**-8, 1e-8, 3.0e38, 3.4e38, -7.25, 1e300,
               float("inf"), float("nan"), 2.0**-24 * 1.5]


@pytest.mark.parametrize("dt", ALL)
def test_g1_cpu_full(dt):
    """NDArray::full on a host tensor: fill_kernel's element rule st<T>((float)value); non-float dtypes refuse with
    DISPATCH_FLOAT's message (empty shapes return before the dtype check, as on the device)."""
    if dt not in FLOATS:
        with pytest.raises(RuntimeError, match=REFUSE):
            _C._core_full([3], 1.0, dt, "cpu")
        assert _C._core_full([0], 1.0, dt, "cpu").shape == (0,)
        return
    seen = []
    for v in FULL_VALUES:
        t = _C._core_full([2, 3], v, dt, "cpu")
        got = bits_of(t).ravel()
        with np.errstate(over="ignore"):
            expected = ref_st(dt, np.array([v], np.float64).astype(np.float32).view(np.uint32))[0]
        want = np.full(6, expected, dtype=UINT[dt])  # independent NumPy constant, no engine fill
        constant_identity(got, want, expected, f"full {dt} {v}")
        seen.append(int(got[0]))
    l19(np.array(seen, UINT[dt]), f"full {dt} over values")


def _host_source(dt):
    return make(dt, census_simple(dt), "cpu")


@pytest.mark.parametrize("dt", ALL)
def test_g1_cpu_clone(dt):
    """NDArray::clone of a host tensor: same bits, new storage, source unchanged."""
    t = _host_source(dt)
    before = bits_of(t)
    c = _C._core_clone(t)
    assert c.device == "cpu" and c.dtype == dt and c.shape == t.shape
    assert c.data_ptr() != t.data_ptr()
    got = bits_of(c)
    identity(got, before, f"clone {dt}")
    identity(bits_of(t), before, f"clone source unchanged {dt}")


@pytest.mark.parametrize("dt", ALL)
def test_g1_cpu_to_cpu(dt):
    """NDArray::to("cpu") of a host tensor (host to host, through clone)."""
    t = _host_source(dt)
    c = _C._core_to(t, "cpu")
    assert c.device == "cpu" and c.dtype == dt and c.data_ptr() != t.data_ptr()
    got = bits_of(c)
    identity(got, bits_of(t), f"to {dt}")


@pytest.mark.parametrize("src,dst", PAIRS_CAST, ids=[f"{s}-{d}" for s, d in PAIRS_CAST])
def test_g1_cpu_astype(src, dst):
    """Host .astype on every A1.4 pair equals E3's NumPy reference on the value census (no GPU)."""
    bits = census_simple(src)
    t = make(src, bits, "cpu")
    inb = bits_of(t)
    identity(inb, bits.astype(UINT[src]), f"astype input {src}")
    out = t.astype(dst)
    assert out.device == "cpu" and out.dtype == dst and out.shape == t.shape
    got = bits_of(out)
    want = ref_astype(src, dst, inb)
    identity(got, want, f"astype {src}->{dst}")
    bad = np.nonzero(got != want)[0]
    record("g1", {"test": "astype", "pair": f"{src}->{dst}", "n": int(got.size), "mismatches": int(bad.size)})
    assert bad.size == 0, [(hex(int(inb[i])), hex(int(got[i])), hex(int(want[i]))) for i in bad[:8]]


@pytest.mark.parametrize("src,dst", PAIRS_REFUSED, ids=[f"{s}-{d}" for s, d in PAIRS_REFUSED])
def test_g1_cpu_astype_refused(src, dst):
    t = make(src, census_simple(src)[:16], "cpu")
    with pytest.raises(RuntimeError, match=REFUSE):
        t.astype(dst)


def test_g1_cpu_backward_full_clone():
    """The public route to full and clone: backward() of a host leaf seeds full(shape, 1.0) and accumulate_grad
    clones it (reduce_to with equal shapes)."""
    for dt in FLOATS:
        x = tc.tensor(np.arange(6, dtype=np.float32).reshape(2, 3), device="cpu", dtype=dt).detach()
        x.requires_grad = True  # a leaf in every dtype (bfloat16 creation is a cast node otherwise)
        x.backward()
        g = x.grad
        assert g.device == "cpu" and g.dtype == dt
        # The gradient seed is constant; combine with the real source tensor to
        # guard both full-width operands before identity (review mixed corpus).
        identity(np.concatenate([bits_of(g).ravel(), bits_of(x).ravel()]),
                 np.concatenate([ref_st(dt, np.ones(6, np.float32).view(np.uint32)),
                                 ref_st(dt, np.arange(6, dtype=np.float32).view(np.uint32))]),
                 f"backward seed and source {dt}")


def test_g1_cpu_e4_fill_refuses():
    """E4 (A1.2): the only other census core member reached by a host tensor, fill_, keeps refusing it."""
    t = tc.tensor([1.0, 2.0], device="cpu")
    with pytest.raises(RuntimeError, match="fill_: CUDA arrays only"):
        t.fill_(3.0)
    identity(bits_of(t), np.array([1.0, 2.0], np.float32).view(np.uint32), "fill refusal preserves source")


_NO_CUDA_CHILD = r"""
import ctypes, os, sys
sys.path.insert(0, os.environ["ENG3_PKG_PARENT"])
import numpy as np
import tensor_cuda as tc
L = ctypes.CDLL(None)
L.eng3_cudart_calls.restype = ctypes.c_ulong
L.eng3_cudart_count.restype = ctypes.c_ulong
L.eng3_cudart_name.restype = ctypes.c_char_p
def counts():
    return {L.eng3_cudart_name(i).decode(): L.eng3_cudart_count(i) for i in range(L.eng3_cudart_n())
            if L.eng3_cudart_count(i)}
base = L.eng3_cudart_calls()
C = tc._C
for dt in ("float32", "float16", "bfloat16", "uint8", "int64", "bool"):
    src = {"bool": np.array([True, False, True]), "uint8": np.array([0, 7, 255], np.uint8),
           "int64": np.array([1, -2, 3])}.get(dt, np.array([0.5, -1.5, 65520.0, 300.0]))
    t = tc.tensor(src, device="cpu", dtype=dt)                     # from_host (+ host astype for bfloat16)
    t.numpy()                                                      # to_host (+ host astype for bfloat16)
    C._core_zeros([5], dt, "cpu").numpy()                          # zeros
    if dt in ("float32", "float16", "bfloat16"):
        C._core_full([5], 2.5, dt, "cpu").numpy()                  # full
    C._core_clone(t).numpy()                                       # clone
    C._core_to(t, "cpu").numpy()                                   # to (host to host)
    for d2 in ("float32", "float16", "bfloat16", "uint8"):
        if dt in ("int64", "bool") and d2 != dt:
            continue
        t.astype(d2).numpy()                                       # astype (every castable pair)
x = tc.tensor([1.0, 2.0], device="cpu", requires_grad=True)
x.backward()
x.grad.numpy()
host_calls = L.eng3_cudart_calls() - base
host_counts = counts()
try:
    tc.tensor([1.0], device="cuda")                                # positive control: a device tensor
    control = "no error"
except RuntimeError as e:
    control = str(e)
print({"host_calls": host_calls, "host_counts": host_counts, "control_calls": L.eng3_cudart_calls() - base - host_calls,
       "control": control})
"""


def test_g1_cpu_no_cuda_calls():
    """Counts calls through the module's global libcudart imports after import.
    Cannot see direct-handle, driver-API or other-library calls. Every E2 host
    member is exercised; a device attempt is a positive control for this scope.
    """
    assert os.path.exists(SHIM), SHIM
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", LD_PRELOAD=SHIM, ENG3_PKG_PARENT=PKG_PARENT)
    p = subprocess.run([sys.executable, "-B", "-P", "-c", _NO_CUDA_CHILD], env=env, capture_output=True, text=True,
                       timeout=300)
    sys.stderr.write(p.stderr)
    assert p.returncode == 0, p.stderr[-2000:]
    res = eval(p.stdout.strip().splitlines()[-1])  # a dict literal printed by the child above
    record("g1", {"test": "no_cuda_calls", **{k: str(v) for k, v in res.items()}})
    print("ENG3 no-CUDA counter:", res)
    assert res["control_calls"] > 0, res          # the counter sees CUDA calls
    assert res["host_calls"] == 0, res            # no E2 member of a host tensor made one


# =================================================================================================== G2 (GPU)
def make_pair(src, bits):
    """A host and a device tensor holding the same input bits. bfloat16 NaN payloads come from torch via DLPack
    (device) and NDArray::to (device -> host, a byte copy); everything else is built on each side from the bits."""
    if src == "bfloat16":
        plain = ~((((bits & 0x7F80) == 0x7F80) & ((bits & 0x7F) != 0)) & (bits != NANBF))
        if not plain.all():
            import torch
            tt = torch.from_numpy(bits.astype(np.uint16).view(np.int16).copy()).view(torch.bfloat16).cuda()
            torch.cuda.synchronize()
            d = tc.from_dlpack(tt)
            d = _C._core_clone(d)  # an engine-owned device copy (cudaMemcpy device to device)
            h = _C._core_to(d, "cpu")
            return h, d
    return make(src, bits, "cpu"), make(src, bits, "cuda")


@pytest.mark.parametrize("src,dst", PAIRS_CAST, ids=[f"{s}-{d}" for s, d in PAIRS_CAST])
def test_g2_host_equals_device(src, dst):
    bits = census(src, gpu=True) if src in CASTABLE else census_simple(src)
    h, d = make_pair(src, bits)
    hin, din = bits_of(h), bits_of(d)
    assert h.device == "cpu" and d.device.startswith("cuda")
    assert hin.dtype == din.dtype and hin.shape == din.shape
    identity(hin, din, "same input host/device " + src)
    identity(hin, bits.astype(UINT[src]), "input matches census " + src)
    ho, do = h.astype(dst), d.astype(dst)
    assert ho.dtype == do.dtype == dst and ho.shape == do.shape
    hb, db = bits_of(ho), bits_of(do)
    assert hb.dtype == db.dtype and hb.shape == db.shape
    l19(hb, f"host {src}->{dst}")
    l19(db, f"device {src}->{dst}")
    bad = np.nonzero(hb != db)[0]
    ref = ref_astype(src, dst, hin)
    assert ref.dtype == hb.dtype and ref.shape == hb.shape
    l19(ref, f"reference {src}->{dst}")
    ref_bad = int((hb != ref).sum())
    record("g2", {"pair": f"{src}->{dst}", "n": int(hb.size), "mismatches": int(bad.size),
                  "ref_mismatches": ref_bad, "first": [(hex(int(hin[i])), hex(int(hb[i])), hex(int(db[i])))
                                                        for i in bad[:8]]})
    assert bad.size == 0, [(hex(int(hin[i])), hex(int(hb[i])), hex(int(db[i]))) for i in bad[:8]]


@pytest.mark.parametrize("src,dst", PAIRS_REFUSED, ids=[f"{s}-{d}" for s, d in PAIRS_REFUSED])
def test_g2_refusals_match_device(src, dst):
    bits = census_simple(src)[:16]
    h, d = make(src, bits, "cpu"), make(src, bits, "cuda")
    msgs = []
    for t in (h, d):
        with pytest.raises(RuntimeError) as ei:
            t.astype(dst)
        msgs.append(str(ei.value))
    assert msgs[0] == msgs[1] == REFUSE


def test_g2_l55_constructed_values():
    """L55: one constructed value per rule, host against device, recorded with the rule."""
    rows = []
    cases = [("float32", "bfloat16", 0x3F808000, "float->bfloat16 tie (1 + 2^-8): RNE keeps even 0x3F80"),
             ("float32", "bfloat16", 0x3F818000, "float->bfloat16 tie (1 + 3*2^-8): RNE rounds up to 0x3F82"),
             ("float32", "float16", _f32(1.5 * 2**-24), "float->half subnormal tie 1.5*2^-24: RNE -> 0x0002"),
             ("float32", "float16", _f32(2.5 * 2**-24), "float->half subnormal tie 2.5*2^-24: RNE -> 0x0002"),
             ("float32", "float16", 0x7FC01234, "float NaN with payload -> half: canonical 0x7FFF"),
             ("float16", "float32", 0x7C01, "half NaN with payload -> float: canonical 0x7FFFFFFF"),
             ("float32", "uint8", _f32(0.5), "uint8 .5 tie: rintf(0.5) = 0"),
             ("float32", "uint8", _f32(254.5), "uint8 .5 tie: rintf(254.5) = 254"),
             ("float32", "uint8", _f32(255.5), "uint8: rintf(255.5) = 256, clamp -> 255")]
    for src, dst, b, rule in cases:
        # Keep the attributed case at offset 0 inside an informative mixed corpus.
        controls = [0, 0x3C00, 0x4000] if src == "float16" else [0, 0x3F800000, 0x40000000]
        arr = np.array([b] + controls, UINT[src])
        h, d = make(src, arr, "cpu"), make(src, arr, "cuda")
        identity(bits_of(h), arr, "L55 host input " + rule)
        identity(bits_of(d), arr, "L55 device input " + rule)
        hb, db = bits_of(h.astype(dst)), bits_of(d.astype(dst))
        ref = ref_astype(src, dst, arr)
        identity(hb, db, "L55 host/device " + rule)
        identity(hb, ref, "L55 independent reference " + rule)
        rows.append({"rule": rule, "pair": f"{src}->{dst}", "input": hex(int(b)), "offset": 0,
                     "host": hex(int(hb[0])), "device": hex(int(db[0])), "reference": hex(int(ref[0]))})
        record("l55", rows[-1])
    print(json.dumps(rows, indent=1))


_RUNTIME_UNAVAILABLE_CHILD = r"""
import ctypes, json, os, sys
sys.path.insert(0, os.environ["ENG3_PKG_PARENT"])
import numpy as np
import tensor_cuda as tc
rt = ctypes.CDLL("/usr/local/cuda-12.6/lib64/libcudart.so")
rt.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
n = ctypes.c_int(-1)
init_error = rt.cudaGetDeviceCount(ctypes.byref(n))
assert init_error != 0, ("CUDA unexpectedly initializes", n.value)
# On runtimes where GetLastError itself needs initialization, it may return
# the initialization error. Record that as unqueryable, never call it clean.
before = rt.cudaGetLastError()
""" + 'C = tc._C\nfor dt in ("float32", "float16", "bfloat16", "uint8", "int64", "bool"):\n    src = {"bool": np.array([True, False, True]), "uint8": np.array([0, 7, 255], np.uint8),\n           "int64": np.array([1, -2, 3])}.get(dt, np.array([0.5, -1.5, 65520.0, 300.0]))\n    t = tc.tensor(src, device="cpu", dtype=dt)                     # from_host (+ host astype for bfloat16)\n    t.numpy()                                                      # to_host (+ host astype for bfloat16)\n    C._core_zeros([5], dt, "cpu").numpy()                          # zeros\n    if dt in ("float32", "float16", "bfloat16"):\n        C._core_full([5], 2.5, dt, "cpu").numpy()                  # full\n    C._core_clone(t).numpy()                                       # clone\n    C._core_to(t, "cpu").numpy()                                   # to (host to host)\n    for d2 in ("float32", "float16", "bfloat16", "uint8"):\n        if dt in ("int64", "bool") and d2 != dt:\n            continue\n        t.astype(d2).numpy()                                       # astype (every castable pair)\nx = tc.tensor([1.0, 2.0], device="cpu", requires_grad=True)\nx.backward()\nx.grad.numpy()\n' + r"""
after = rt.cudaGetLastError()
if before == 0:
    assert after == 0, ("CUDA error after host paths", after)
else:
    assert after in (0, before), ("new CUDA error after host paths", before, after)
print(json.dumps({"init_error": init_error, "before": before, "after": after,
                  "last_error_clean": after == 0, "queryable": before == 0}))
"""


def test_g1_cpu_runtime_unavailable():
    """Independent R4 check: real runtime cannot initialize; every host path must pass.
    Any host exception fails; cudaGetLastError must be clean where queryable.
    """
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", ENG3_PKG_PARENT=PKG_PARENT)
    env.pop("LD_PRELOAD", None)
    p = subprocess.run([sys.executable, "-B", "-P", "-c", _RUNTIME_UNAVAILABLE_CHILD],
                       env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    row = json.loads(p.stdout.strip().splitlines()[-1])
    record("runtime_unavailable", row)
    print("ENG3F1 runtime unavailable:", row)


# ENG3-F2 diagnostic census. Prior art: REV2 boundary cases (2026), pytest
# parametrization (2004+), Python repr/C++17 hex escapes (2017; unverified --
# lead to check). Taken: boundary and escaped-string checks; ours: this census.
@pytest.mark.parametrize("dtype,itemsize", [("float32", 4), ("float16", 2),
                                            ("bfloat16", 2), ("uint8", 1),
                                            ("int64", 8), ("bool", 1)])
@pytest.mark.parametrize("hook", ["_core_zeros", "_core_full"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("empty", [False, True], ids=["nonempty", "zero_extent"])
def test_f2_cpu_overflow_message(dtype, itemsize, hook, device, empty):
    """Execute the allocating constructor's byte-bound throw, before allocation.
    Both CPU and bare CUDA descriptors reach the same check without CUDA calls.
    """
    shape = [2, 2**62] if itemsize == 1 else [(2**63 - 1) // itemsize + 1]
    if empty:
        shape = [0] + shape
    args = (shape, 1.0, dtype, device) if hook == "_core_full" else (shape, dtype, device)
    with pytest.raises(OverflowError) as caught:
        getattr(_C, hook)(*args)
    message = str(caught.value)
    assert "byte size overflows" in message, message
    assert str(shape) in message, message
    assert "dtype " + dtype in message, message
    record("f2_overflow", {"hook": hook, "shape": shape, "dtype": dtype,
                           "device": device, "message": message})


_F2_BAD_DEVICES = ["cpu\x00suffix", "cuda\x00suffix", "cuda:0\x00suffix",
                   "\x00cpu\x00tail", "cpuBAD", "cpu:garbage", "cuda:2147483647",
                   "cpu\\literal", "cpu'é", "cuda:é\x00tail"] + [
                       "cpu" + chr(byte) + "tail" for byte in list(range(32)) + [127]]


@pytest.mark.parametrize("hook", ["_core_zeros", "_core_full", "_core_to"])
@pytest.mark.parametrize("device", _F2_BAD_DEVICES)
def test_f2_cpu_device_message(hook, device):
    """All device-taking hooks must name the complete escaped offending string.
    _core_clone has no device-string operand. Reference is independently built
    from UTF-8 bytes; the census includes every ASCII control byte and DEL.
    """
    if hook == "_core_to":
        args = (_C._core_zeros([1], "float32", "cpu"), device)
    elif hook == "_core_full":
        args = ([1], 1.0, "float32", device)
    else:
        args = ([1], "float32", device)
    expected = "".join(chr(byte) if 32 <= byte < 127 else "\\x%02x" % byte
                       for byte in device.encode("utf-8"))
    with pytest.raises(ValueError) as caught:
        getattr(_C, hook)(*args)
    message = str(caught.value)
    assert message == (hook + ": invalid device '" + expected +
                       "'; expected cpu, cuda, or cuda:N naming the current device"), message
    assert all(32 <= ord(char) < 127 for char in message), message
    record("f2_device", {"hook": hook, "device": device, "message": message})
