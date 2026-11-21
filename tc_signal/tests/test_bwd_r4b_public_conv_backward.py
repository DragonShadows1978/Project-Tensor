"""BWD-R4b item 2: tc_signal.conv1d_backward and tc_signal.conv_transpose1d_backward.

GPU tests (G-API): every conv1d / conv_transpose1d geometry in Voice's BP1 and BP2 fixture manifests (BP1 includes the
stride-5 iSTFT tail transposed convolutions at small and Kokoro size), plus the stride-5 tail geometry at the BP1 crop's
F = 4801. Each public result is compared with the fp64 answer key under BWD-R3's sealed convolution class
(bwd_r3_support.cases.compare) and bit for bit with tc_signal's tracked path (the same kernels, called the same way).
Host tests: the signature contract and every refusal, with CUDA hidden and the native entry point patched to fail if
it is ever reached.

Prior art: the fixture comparisons are BWD-R3's (tc_signal/tests/test_bwd_r3_numeric.py, 2026), reused through its
support module; the public/tracked bitwise pairing is the reference-pair check of HOUSE_RULES section 8. No other prior
art known to me.
"""
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest
import tc_signal as ts
import tensor_cuda as tc

from bwd_r3_support.cases import BP1, BP2, SLOTS, Case, answer, compare, exact, fixture, gradients, require_gpu

PUBLIC = {"conv1d": ts.conv1d_backward, "conv_transpose1d": ts.conv_transpose1d_backward}
CONV_OPS = {"conv1d", "conv_transpose1d", "tail_conv_transpose_real", "tail_conv_transpose_imag"}


def public(case, needs=(True, True, True)):
    leaves = [tc.tensor(a) for a in (case.x, case.w, case.seed)]
    out = PUBLIC[case.op](*leaves, **case.cuda_attrs, needs=needs)
    assert isinstance(out, tuple) and len(out) == 3
    for slot, need, value in zip(SLOTS, needs, out):
        assert (value is not None) == need, slot
        if value is not None:
            assert type(value) is tc.Tensor and value.dtype == "float32" and not value.requires_grad
    return tuple(None if value is None else value.numpy() for value in out)


def check_public(case):
    expected, mass = answer(case)
    got = public(case)
    for i, slot in enumerate(SLOTS):
        compare(case.label, slot, got[i], expected[i], mass[i])
    tracked = gradients(case)
    for i, slot in enumerate(SLOTS):
        exact(got[i], tracked[i])
    print(f"R4B_PUBLIC {case.label} op={case.op} attrs={case.attrs} x={case.x.shape} w={case.w.shape} "
          f"dy={case.seed.shape} public==tracked bitwise", flush=True)
    return got


@pytest.mark.parametrize("label", BP1)
def test_gpu_public_bp1(label):
    case = fixture("bp1", label)
    require_gpu()
    check_public(case)


@pytest.mark.parametrize("label", BP2)
def test_gpu_public_bp2(label):
    case = fixture("bp2", label)
    require_gpu()
    check_public(case)


def tail_crop_case():
    # The stride-5 iSTFT tail transposed convolution at the BP1 crop's F = 4801 (L = 24000 samples + 20 taps).
    rng = np.random.default_rng(20261119)
    f = 4801
    x = rng.standard_normal((1, 11, f), dtype=np.float32)
    w = rng.standard_normal((11, 1, 20), dtype=np.float32)
    seed = rng.standard_normal((1, 1, 5 * (f - 1) + 20), dtype=np.float32)
    return Case("tail-stride5-crop", "conv_transpose1d", x, w, np.zeros(1, np.float32), seed,
                {"stride": 5, "dilation": 1, "groups": 1, "pads": [0, 0], "output_padding": 0})


def test_gpu_public_tail_stride5_crop():
    require_gpu()
    check_public(tail_crop_case())


@pytest.mark.parametrize("size", ("small1", "small2", "small3", "kokoro"))
def test_gpu_public_conv_post_geometry(size):
    """conv_post (conv1d [22, C, 7], pads 3) at the BP1 istft_tail fixtures' operands: x = LeakyReLU(arg_x, 0.01) in
    fp32 as the tail computes it, w and b the fixture's, and a seeded upstream of the conv output's shape."""
    from bwd_r3_support.cases import sha

    fixture("bp1", BP1[0])  # manifest sha256 pin and fixture-root skip
    require_gpu()
    root = Path(os.environ["TC_BWD_R3_BP1_FIXTURES"])
    entry, = (row for row in json.loads((root / "manifest.json").read_text())["files"]
              if row["file"] == f"istft_tail_{size}.npz")
    assert sha(root / entry["file"]) == entry["sha256"] and entry["op"] == "istft_tail"
    with np.load(root / entry["file"], allow_pickle=False) as f:
        x = f["arg_x"].astype(np.float32)
        w, b = f["arg_w"].astype(np.float32), f["arg_b"].astype(np.float32)
    v = np.where(x < 0, x * np.float32(entry["attrs"]["slope"]), x).astype(np.float32)
    seed = np.random.default_rng(4801).standard_normal((x.shape[0], 22, x.shape[2]), dtype=np.float32)
    check_public(Case(f"conv_post_{size}", "conv1d", v, w, b, seed,
                      {"stride": 1, "dilation": 1, "groups": 1, "pads": [3, 3]}))


@pytest.mark.parametrize("label", ("conv1d_kokoro", "conv_transpose1d_kokoro", "tail_conv_transpose_real_kokoro"))
def test_gpu_public_needs_subsets_and_repeat(label):
    case = fixture("bp1", label)
    require_gpu()
    full = public(case)
    again = public(case)
    for a, b in zip(full, again):
        exact(a, b)
    for mask in ((True, False, False), (False, True, False), (False, False, True), (False, True, True)):
        part = public(case, mask)
        for need, a, b in zip(mask, part, full):
            if need:
                exact(a, b)
    assert public(case, (False, False, False)) == (None, None, None)


def test_host_fixture_geometry_census():
    """Every conv / conv-transpose entry of both manifests is one of the parametrized labels above."""
    for bp, labels in (("bp1", BP1), ("bp2", BP2)):
        setting = f"TC_BWD_R3_{bp.upper()}_FIXTURES"
        root = os.environ.get(setting)
        if root is None or not Path(root).is_dir():
            pytest.skip(f"{setting} fixture root absent")
        fixture(bp, labels[0])  # manifest sha256 pin
        entries = json.loads((Path(root) / "manifest.json").read_text())["files"]
        conv = {row["file"][:-4] for row in entries if row["op"] in CONV_OPS}
        print(f"R4B_CENSUS {bp} conv_geometries={len(conv)} tested={len(labels)}", flush=True)
        assert conv and conv == set(labels), (bp, sorted(conv ^ set(labels)))


# ------------------------------------------------------------------------------------------------ host contract
def test_host_signatures():
    for fn, extra in ((ts.conv1d_backward, ()), (ts.conv_transpose1d_backward, ("output_padding",))):
        params = inspect.signature(fn).parameters
        assert list(params)[:3] == ["x", "w", "grad_out"]
        keywords = [name for name, p in params.items() if p.kind is p.KEYWORD_ONLY]
        forward = ts.conv1d if fn is ts.conv1d_backward else ts.conv_transpose1d
        forward_keywords = [name for name, p in inspect.signature(forward).parameters.items()
                            if p.kind is p.KEYWORD_ONLY and name != "out"]
        assert keywords == forward_keywords + ["needs"]
        for name in forward_keywords:
            assert params[name].default == inspect.signature(forward).parameters[name].default
        assert params["needs"].default == (True, True, True)
        assert all(name in keywords for name in ("stride", "padding", "dilation", "groups", *extra))
    assert {"conv1d_backward", "conv_transpose1d_backward"} <= set(ts.__all__)
    assert callable(ts._C._conv_backward_run) and callable(ts._C._conv_backward_scratch)


@pytest.fixture
def no_native(monkeypatch):
    calls = []

    def refuse(*args, **kwargs):
        calls.append(args)
        raise AssertionError("the native backward entry point was reached")

    monkeypatch.setattr(ts._C, "_conv_backward_run", refuse)
    monkeypatch.setattr(ts._C, "_conv_backward_scratch", refuse)
    yield calls
    assert not calls


def cpu(shape, dtype="float32"):
    data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) % 7 - 3
    return tc.tensor(data, dtype=dtype, device="cpu")


GOOD = {
    "conv1d": ((2, 4, 9), (6, 2, 3), (2, 6, 4), dict(stride=2, padding=(1, 0), dilation=1, groups=2)),
    "conv_transpose1d": ((2, 4, 5), (4, 3, 4), (2, 6, 11), dict(stride=2, padding=1, output_padding=1, dilation=1,
                                                               groups=2)),
}


@pytest.mark.parametrize("op", sorted(GOOD))
def test_host_refuses_cpu_device_last(op, no_native):
    xs, ws, gs, attrs = GOOD[op]
    with pytest.raises(ValueError, match="one CUDA device"):
        PUBLIC[op](cpu(xs), cpu(ws), cpu(gs), **attrs)


@pytest.mark.parametrize("op", sorted(GOOD))
def test_host_refuses_foreign_and_noncontiguous(op, no_native):
    xs, ws, gs, attrs = GOOD[op]
    base = np.zeros(xs[::-1], np.float32).transpose(2, 1, 0)
    assert not base.flags.c_contiguous and base.shape == xs
    for args in ((base, cpu(ws), cpu(gs)), (cpu(xs), np.zeros(ws, np.float32), cpu(gs)),
                 (cpu(xs), cpu(ws), None)):
        with pytest.raises(TypeError, match="tensor_cuda.Tensor"):
            PUBLIC[op](*args, **attrs)


@pytest.mark.parametrize("op", sorted(GOOD))
@pytest.mark.parametrize("dtype", ("float16", "bfloat16", "int64"))
@pytest.mark.parametrize("slot", range(3))
def test_host_refuses_dtype(op, dtype, slot, no_native):
    xs, ws, gs, attrs = GOOD[op]
    args = [cpu(s, dtype if i == slot else "float32") for i, s in enumerate((xs, ws, gs))]
    with pytest.raises(TypeError, match="must be float32"):
        PUBLIC[op](*args, **attrs)


@pytest.mark.parametrize("op", sorted(GOOD))
def test_host_refuses_shapes_and_groups(op, no_native):
    xs, ws, gs, attrs = GOOD[op]
    fn = PUBLIC[op]
    bad = [
        ((xs[1:], ws, gs), attrs, "rank 3"),
        ((xs, ws[:2], gs), attrs, "rank 3"),
        ((xs, ws, (gs[0], gs[1], gs[2] + 1)), attrs, "expected the forward output"),
        ((xs, ws, (gs[0], gs[1] + 2, gs[2])), attrs, "expected the forward output"),
        ((xs, ws, (gs[0] + 1, gs[1], gs[2])), attrs, "expected the forward output"),
        ((xs, ws, gs), dict(attrs, groups=3), "not divisible by groups|w.shape"),
        ((xs, ws, gs), dict(attrs, stride=0), "stride must be >= 1"),
        ((xs, ws, gs), dict(attrs, dilation=0), "dilation must be >= 1"),
        ((xs, ws, gs), dict(attrs, groups=0), "groups must be >= 1"),
        ((xs, ws, gs), dict(attrs, padding=-1), "padding must be >= 0"),
        ((xs, ws, gs), dict(attrs, padding=(1, 2, 3)), "padding must be an int"),
    ]
    if op == "conv1d":
        bad += [((xs, (6, 4, 3), gs), attrs, "w.shape\\[1\\]"),
                ((xs, (5, 2, 3), (2, 5, 4)), attrs, "C_out \\(5\\) is not divisible"),
                ((xs, (6, 2, 30), gs), attrs, "output length")]
    else:
        bad += [((xs, (5, 3, 4), gs), attrs, "w.shape\\[0\\]"),
                ((xs, ws, gs), dict(attrs, output_padding=2), "output_padding"),
                ((xs, ws, (2, 6, 1)), dict(attrs, padding=20), "output length")]
    for shapes, kwargs, message in bad:
        with pytest.raises(ValueError, match=message):
            fn(*(cpu(s) for s in shapes), **kwargs)


@pytest.mark.parametrize("op", sorted(GOOD))
def test_host_refuses_attribute_types_and_needs(op, no_native):
    xs, ws, gs, attrs = GOOD[op]
    fn = PUBLIC[op]
    args = [cpu(s) for s in (xs, ws, gs)]
    for kwargs in (dict(attrs, stride=True), dict(attrs, stride=2.0), dict(attrs, groups="2"),
                   dict(attrs, padding=(1.0, 0))):
        with pytest.raises(TypeError, match="must be an int"):
            fn(*args, **kwargs)
    for needs in ((True, True), (1, 1, 1), (True, True, None), "xyz", None, (np.True_, True, True)):
        with pytest.raises(TypeError, match="needs must be"):
            fn(*args, **attrs, needs=needs)
    with pytest.raises(TypeError):
        fn(*args, attrs["stride"])  # attributes are keyword-only


# ------------------------------------------------------------------------------------------------ BWD-R4b r4 (appended)
# Review BWD-REV4B: F05 (native preflight before the all-off return and before allocation), F08 (the two missing needs
# masks, the dilation-dominant output-padding corner, mixed and non-current devices), F10 (current-device check).
# Prior art: as above; no other prior art known to me.
import ctypes  # noqa: E402

import tc_signal._conv_backward as public_module  # noqa: E402

TWO_DEVICE_SKIP = "needs two CUDA devices (registered device-count skip, BWD-R4b r4 F08)"


@pytest.fixture
def counted_preflight(monkeypatch):
    calls = []
    real = public_module._PREFLIGHT

    def preflight(*args):
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(public_module, "_PREFLIGHT", preflight)
    return calls


@pytest.mark.parametrize("needs", ((True, True, True), (False, False, False)))
def test_host_native_preflight_refuses_beyond_int_max(needs, counted_preflight, no_native):
    """F05: geometry the Python plan accepts but the native run entry refuses (an attribute above INT_MAX or beyond
    int64) is refused before the all-off return and before any allocation."""
    cases = [
        (ts.conv1d_backward, (1, 2, 5), (3, 2, 1), (1, 3, 1), dict(stride=2**31), "stride must be in"),
        (ts.conv1d_backward, (1, 2, 5), (3, 2, 1), (1, 3, 5), dict(dilation=2**31), "dilation must be in"),
        (ts.conv_transpose1d_backward, (1, 2, 1), (2, 3, 1), (1, 3, 1), dict(stride=2**31), "stride must be in"),
        (ts.conv1d_backward, (1, 2, 5), (3, 2, 1), (1, 3, 1), dict(stride=2**63), "native int64 range"),
    ]
    for fn, xs, ws, gs, attrs, message in cases:
        with pytest.raises(ValueError, match=message):
            fn(cpu(xs), cpu(ws), cpu(gs), **attrs, needs=needs)
    assert len(counted_preflight) == len(cases)


@pytest.mark.parametrize("op", sorted(GOOD))
def test_host_preflight_runs_before_needs_and_device(op, counted_preflight, no_native):
    xs, ws, gs, attrs = GOOD[op]
    with pytest.raises(ValueError, match="one CUDA device"):
        PUBLIC[op](cpu(xs), cpu(ws), cpu(gs), **attrs, needs=(False, False, False))
    with pytest.raises(TypeError, match="needs must be"):
        PUBLIC[op](cpu(xs), cpu(ws), cpu(gs), **attrs, needs=(True, True))
    assert len(counted_preflight) == 2


def test_host_dilation_dominant_output_padding_accepted(counted_preflight, no_native):
    """F08: output_padding >= stride is legal while it stays below dilation (max(stride, dilation) = dilation)."""
    x, w, g = (2, 4, 9), (4, 2, 3), (2, 4, 1 * 8 + 2 + 3 * 2 + 1 - 1 - 2)
    with pytest.raises(ValueError, match="one CUDA device"):
        ts.conv_transpose1d_backward(cpu(x), cpu(w), cpu(g), stride=1, padding=(1, 2), output_padding=2, dilation=3,
                                     groups=2)
    assert len(counted_preflight) == 1
    with pytest.raises(ValueError, match="output_padding"):
        ts.conv_transpose1d_backward(cpu(x), cpu(w), cpu(g), stride=1, padding=(1, 2), output_padding=3, dilation=3,
                                     groups=2)


def test_host_current_device_rule():
    """F10: one CUDA device for all operands, and it must be the current device; checked before allocating."""
    check = public_module._check_devices
    check("f", ["cuda:1"] * 3, 1)
    check("f", ["cuda:0"] * 3, 0)
    with pytest.raises(ValueError, match="current CUDA device is cuda:1"):
        check("f", ["cuda:0"] * 3, 1)
    with pytest.raises(ValueError, match="one CUDA device"):
        check("f", ["cuda:0", "cuda:1", "cuda:0"], 0)
    with pytest.raises(ValueError, match="one CUDA device"):
        check("f", ["cpu"] * 3, 0)


@pytest.mark.parametrize("label", ("conv1d_kokoro", "conv_transpose1d_kokoro", "tail_conv_transpose_real_kokoro"))
def test_gpu_public_needs_two_missing_masks(label):
    """F08: the two masks the r3 test lacked, (dx, dw) and (dx, db), bitwise against the full request."""
    case = fixture("bp1", label)
    require_gpu()
    full = public(case)
    for mask in ((True, True, False), (True, False, True)):
        part = public(case, mask)
        for need, a, b in zip(mask, part, full):
            if need:
                exact(a, b)


def test_gpu_public_dilation_dominant_output_padding():
    require_gpu()
    from bwd_r3_support.cases import synthetic

    case = synthetic("conv_transpose1d", "dilation-dominant-output-padding", n=2, channels=4, width=9, kernel=3,
                     stride=1, dilation=3, output_padding=2, groups=2, pads=(1, 2))
    assert case.native_attrs == (1, 1, 2, 2, 3, 2) and case.seed.shape == (2, 4, 1 * 8 + 2 + 3 * 2 + 1 - 1 - 2)
    check_public(case)


def test_gpu_public_refuses_noncurrent_device_simulated(monkeypatch):
    """F10 on one GPU: the current-device query is made to name another device; the call is refused, nothing runs."""
    require_gpu()
    case = fixture("bp1", "conv1d_small1")
    leaves = [tc.tensor(a) for a in (case.x, case.w, case.seed)]
    monkeypatch.setattr(public_module, "_current_cuda_device", lambda: 1)
    with pytest.raises(ValueError, match="current CUDA device is cuda:1"):
        ts.conv1d_backward(*leaves, **case.cuda_attrs)


def device_count():
    try:
        runtime = ctypes.CDLL("libcudart.so.12")
    except OSError:
        return 0
    count = ctypes.c_int()
    return 0 if runtime.cudaGetDeviceCount(ctypes.byref(count)) else count.value


def test_gpu_public_mixed_and_noncurrent_devices_two_gpus():
    require_gpu()
    if device_count() < 2:
        pytest.skip(TWO_DEVICE_SKIP)
    runtime = ctypes.CDLL("libcudart.so.12")
    case = fixture("bp1", "conv1d_small1")
    # BWD-R4b r6 (re-check 2 F07): the negative operand is made with tc.empty, which labels the current device, on the
    # intended device; the switch's status is checked and the caller's prior device restored.
    prior = ctypes.c_int(-1)
    assert runtime.cudaGetDevice(ctypes.byref(prior)) == 0
    # BWD-R4b r7 (Codex confirm R5-F07): the prior device is restored in a finally around both cases, so a failed
    # assertion cannot leave cuda:0 current for the next test.
    try:
        assert runtime.cudaSetDevice(0) == 0
        x0, w0, g0 = (tc.tensor(a) for a in (case.x, case.w, case.seed))
        assert {str(t.device) for t in (x0, w0, g0)} == {"cuda:0"}
        assert runtime.cudaSetDevice(1) == 0
        try:
            w1 = tc.empty(*case.w.shape, dtype="float32")
            assert str(w1.device) == "cuda:1"
            with pytest.raises(ValueError, match="current CUDA device"):
                ts.conv1d_backward(x0, w0, g0, **case.cuda_attrs)  # operands on cuda:0, cuda:1 current
        finally:
            assert runtime.cudaSetDevice(0) == 0
        with pytest.raises(ValueError, match="one CUDA device"):
            ts.conv1d_backward(x0, w1, g0, **case.cuda_attrs)  # mixed devices, cuda:0 current
    finally:
        restored = runtime.cudaSetDevice(prior.value)
        assert restored == 0, f"cudaSetDevice({prior.value}) returned {restored} while restoring the prior device"



# ------------------------------------------------------------------------------------------------ BWD-R4b r6 (appended)
# Re-check 2 F06: the native address rule (non-NULL, 4-byte aligned element 0) checked in Python before allocating.
def test_host_alignment_rule():
    check = public_module._check_alignment
    check("f", [("x", 4096), ("w", 8), ("grad_out", 12)])
    for bad in (0, 2, 4097):
        with pytest.raises(ValueError, match="4-byte aligned"):
            check("f", [("x", 4096), ("w", bad), ("grad_out", 12)])


@pytest.mark.parametrize("op", sorted(GOOD))
def test_host_misaligned_operand_refused_before_needs_and_device(op, monkeypatch, no_native):
    xs, ws, gs, attrs = GOOD[op]
    monkeypatch.setattr(public_module, "_data_pointer", lambda t: 4098)
    with pytest.raises(ValueError, match="4-byte aligned"):
        PUBLIC[op](cpu(xs), cpu(ws), cpu(gs), **attrs, needs=(False, False, False))
