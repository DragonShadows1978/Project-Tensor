"""Read only pinned Voice fixtures and compute strict comparison diagnostics."""
from dataclasses import dataclass
import ctypes
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from . import reference

MANIFESTS = {
    "bp1": "3822b6a61978e9f510f030039aba260ecc7a264c4525eb31fb8a72bea305efee",
    "bp2": "bcabaf93eb7994d0068c0a0d9ae694a765eea6e66567a46fdfbb041306eed876",
}
TOL_SHA = "8f6c52e38d8a653aaaf9c1b51453fa8031b1f50f59444a3bf436262fa4845f2b"
TOL_FILE = Path(__file__).with_name("TOLERANCES_R3.json")
TOL = json.loads(TOL_FILE.read_text())
BP1 = tuple(f"{op}_{size}" for op in ("conv1d", "conv_transpose1d", "tail_conv_transpose_real",
                                    "tail_conv_transpose_imag") for size in ("small1", "small2", "small3", "kokoro"))
BP2 = tuple(f"conv1d_{i:02}" for i in range(1, 33)) + tuple(f"conv_transpose1d_{i:02}" for i in range(1, 5))
SLOTS = ("x", "w", "b")


@dataclass(frozen=True)
class Case:
    label: str
    op: str
    x: np.ndarray
    w: np.ndarray
    b: np.ndarray | None
    seed: np.ndarray
    attrs: dict

    @property
    def inputs(self):
        return self.x, self.w, self.b

    @property
    def cuda_attrs(self):
        return {("padding" if name == "pads" else name): value for name, value in self.attrs.items()}

    @property
    def native_attrs(self):
        s, d, g = (self.attrs.get(key, 1) for key in ("stride", "dilation", "groups"))
        return s, *self.attrs.get("pads", (0, 0)), self.attrs.get("output_padding", 0), d, g


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def fixture(bp, label, *, bias=True, dtype=np.float32, key=False):
    setting = f"TC_BWD_R3_{bp.upper()}_FIXTURES"
    root = os.environ.get(setting)
    if root is None or not Path(root).is_dir():
        pytest.skip(f"{setting} fixture root absent")
    root = Path(root)
    assert sha(root / "manifest.json") == MANIFESTS[bp]
    entries = json.loads((root / "manifest.json").read_text())["files"]
    labels = {entry["file"][:-4] for entry in entries}
    assert set(BP1 if bp == "bp1" else BP2) <= labels
    entry, = (row for row in entries if row["file"] == f"{label}.npz")
    path = root / entry["file"]
    assert sha(path) == entry["sha256"]
    with np.load(path, allow_pickle=False) as f:
        seed = f["upstream"].astype(dtype)
        x, w = (f[f"arg_{slot}"].astype(dtype) for slot in SLOTS[:2])
        b = (f["arg_b"].astype(dtype) if "arg_b" in f else np.zeros(seed.shape[1], dtype=dtype)) if bias else None
        case = Case(label, "conv1d" if label.startswith("conv1d") else "conv_transpose1d", x, w, b, seed,
                    entry["attrs"])
        if key:
            return case, tuple(f[f"vjp_{slot}"] if f"vjp_{slot}" in f else seed.sum(axis=(0, 2)) for slot in SLOTS)
        return case


def require_gpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        pytest.skip("CUDA disabled for CPU lane")
    try:
        runtime = ctypes.CDLL("libcudart.so.12")
    except OSError:
        pytest.skip("no usable CUDA device (runtime unavailable)")
    count = ctypes.c_int()
    if runtime.cudaGetDeviceCount(ctypes.byref(count)) or not count.value:
        pytest.skip("no usable CUDA device")


def oracle(case, *inputs, seed=None):
    return getattr(reference, case.op + "_vjp")(*inputs, case.seed if seed is None else seed, **case.attrs)


def answer(case):
    inputs = tuple(None if value is None else value.astype(np.float64) for value in case.inputs)
    expected = oracle(case, *inputs)
    mass = oracle(case, *(None if value is None else abs(value) for value in inputs),
                  seed=abs(case.seed.astype(np.float64)))
    return expected, mass


def allowed(expected, mass):
    policy = TOL["conv"]
    assert sha(TOL_FILE) == TOL_SHA
    return np.maximum(policy["absolute_floor"], policy["absolute_term_mass"] * mass) + policy["relative"] * abs(expected)


def compare(label, slot, got, expected, mass, *, sparse=False):
    if expected is None:
        assert got is None
        return
    assert got is not None and got.shape == expected.shape and got.size > 0
    assert np.isfinite(got).all() and np.isfinite(expected).all()
    if not sparse:
        if got.size == 1:
            assert got.item() != 0 and expected.item() != 0, (label, slot, "scalar signal")
        else:
            assert np.ptp(got) > 0 and np.ptp(expected) > 0, (label, slot, "constant")
    zeros = mass == 0
    assert np.all(got[zeros] == 0), (label, slot, "structural zero")
    error = abs(got.astype(np.float64) - expected)
    ratios = error / allowed(expected, mass)
    worst = np.unravel_index(ratios.argmax(), ratios.shape)
    relative = np.divide(error, abs(expected), out=np.zeros_like(error), where=expected != 0)
    print(f"R3_NUMERIC {label} d{slot.upper()} shape={got.shape} dtype={got.dtype} "
          f"max_abs={error.max():.9g} max_rel_nonzero={relative.max():.9g} "
          f"norm_l2={np.linalg.norm(error.ravel()) / max(1e-300, np.linalg.norm(expected.ravel())):.9g} "
          f"max_bound_ratio={ratios[worst]:.9g} index={tuple(map(int, worst))} structural_zeros={zeros.sum()}",
          flush=True)
    assert ratios[worst] <= 1, (label, slot, "numeric convolution bound", worst,
                                float(got[worst]), float(expected[worst]), float(ratios[worst]))


def tracked(case, active=(True, True, True)):
    import tc_signal as ts
    import tensor_cuda as tc

    leaves = tuple(None if array is None else tc.tensor(array, requires_grad=needs)
                   for array, needs in zip(case.inputs, active))
    y = getattr(ts, case.op)(*leaves, **case.cuda_attrs)
    assert type(y) is tc.Tensor and y.requires_grad and y.revision == 1 and tuple(y.shape) == case.seed.shape
    return leaves, y, tc.tensor(case.seed)


def gradients(case, active=(True, True, True)):
    leaves, y, g = tracked(case, active)
    (y * g).sum().backward()
    result = tuple(None if leaf is None or leaf.grad is None else leaf.grad.numpy() for leaf in leaves)
    assert tuple(grad is not None for grad in result) == tuple(need and leaf is not None
                                                             for need, leaf in zip(active, leaves))
    assert all(grad is None or grad.dtype == np.float32 for grad in result)
    return result


def exact(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype == np.float32
    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))


def synthetic(op, label, *, n=2, channels=8, width=27, kernel=4, **attrs):
    rng = np.random.default_rng(5280)
    groups = attrs.get("groups", 1)
    stride, dilation = attrs.get("stride", 1), attrs.get("dilation", 1)
    padding = sum(attrs.get("pads", (0, 0)))
    extent = 1 + dilation * (kernel - 1)
    time = ((width + padding - extent) // stride + 1 if op == "conv1d" else
            (width - 1) * stride - padding + extent + attrs.get("output_padding", 0))
    draw = lambda *shape: rng.standard_normal(shape, dtype=np.float32)
    return Case(label, op, draw(n, channels, width), draw(channels, channels // groups, kernel),
                draw(channels), draw(n, channels, time), attrs)
