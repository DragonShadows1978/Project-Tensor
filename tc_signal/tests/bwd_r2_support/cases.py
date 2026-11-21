"""BWD-R2 fixture loading and tests' shared forward wiring."""

import ctypes
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from . import oracle

OPS = ("upsample_nearest1d", "reflection_pad1d", "instance_norm1d", "adain")
CHANNELS = (128, 256, 512, 514, 1024, 1090)
CASES = [(1, op, case) for op in OPS for case in ("small1", "small2", "small3", "kokoro")]
CASES += [(2, op, f"c{channels}") for op in OPS[2:] for channels in CHANNELS]
SLOTS = {"adain": ("x", "s", "W", "b")}
MANIFESTS = {
    1: "3822b6a61978e9f510f030039aba260ecc7a264c4525eb31fb8a72bea305efee",
    2: "bcabaf93eb7994d0068c0a0d9ae694a765eea6e66567a46fdfbb041306eed876",
}
TOLERANCES = Path(__file__).with_name("TOLERANCES_R2.json")
TOLERANCES_SHA256 = "0e99da33a91caabdcee5ba03038c90c539334f46d6d67afd0d9087d52d3378e2"


def tolerances():
    # NIST FIPS 180-4 (2015), SHA-256. Verify the same bytes we parse.
    data = TOLERANCES.read_bytes()
    assert hashlib.sha256(data).hexdigest() == TOLERANCES_SHA256, "BWD-R2 tolerance seal mismatch"
    return json.loads(data)


def require_gpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        pytest.skip("CUDA disabled for CPU lane")
    # NVIDIA CUDA Runtime API 12.6, cudaGetDeviceCount, as also used by
    # TensorCUDA's ENG3 host tests. Keep this probe inside the GPU test gate.
    try:
        count_devices = ctypes.CDLL("libcudart.so.12").cudaGetDeviceCount
    except (OSError, AttributeError):
        pytest.skip("no usable CUDA device")
    count_devices.argtypes = [ctypes.POINTER(ctypes.c_int)]
    count_devices.restype = ctypes.c_int
    count = ctypes.c_int()
    if count_devices(ctypes.byref(count)) != 0 or count.value <= 0:
        pytest.skip("no usable CUDA device")


def fixture_root(bp):
    env = f"TC_BWD_R2_BP{bp}_FIXTURES"
    configured = os.environ.get(env)
    if not configured or not Path(configured).is_dir():
        pytest.skip(f"BP{bp} fixture root absent (set {env} to its location)")
    return Path(configured)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_cell(bp, op, case):
    root = fixture_root(bp)
    manifest = root / "manifest.json"
    assert sha(manifest) == MANIFESTS[bp]
    rows = json.loads(manifest.read_text())["files"]
    filename = f"{op}_{case}.npz"
    row, = (r for r in rows if r["file"] == filename)
    path = root / filename
    assert sha(path) == row["sha256"], filename
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    assert {key: list(array.shape) for key, array in data.items()} == row["shapes"]
    assert {key: str(array.dtype) for key, array in data.items()} == row["dtypes"]
    return row, data


def reference_inputs(row, data, *, rounded=False):
    args = {name: data[f"arg_{name}"] for name in row["arguments"]}
    seed = data["upstream"]
    if rounded:
        args = {name: a.astype(np.float32).astype(np.float64) for name, a in args.items()}
        seed = seed.astype(np.float32).astype(np.float64)
    return args, seed


def oracle_call(op, args, attrs, seed=None):
    if seed is None:
        return getattr(oracle, op)(**args, **attrs)
    return getattr(oracle, f"{op}_vjp")(**args, **attrs, g=seed)


def signal_forward(tc, ts, op, args, attrs):
    if op == "upsample_nearest1d":
        return ts.upsample_nearest1d(args["x"], attrs["factor"])
    if op == "reflection_pad1d":
        return ts.reflection_pad1d(args["x"], attrs["pads"])
    if op == "instance_norm1d":
        return ts.instance_norm1d(args["x"], args.get("scale"), args.get("bias"), **attrs)
    return compose_adain(tc, ts, args, attrs)


def compose_adain(tc, ts, args, attrs):
    # Huang & Belongie (2017), AdaIN; Voice's Kokoro style Gemm and affine.
    # This is test wiring through the public core and sidecar operations.
    x = args["x"]
    _, channels, _ = x.shape
    h = tc.matmul(args["s"], args["W"], trans_b=True) + args["b"]
    gamma = h.slice(1, 0, channels).reshape([x.shape[0], channels, 1])
    beta = h.slice(1, channels, channels).reshape([x.shape[0], channels, 1])
    normalized = ts.instance_norm1d(x, args.get("scale"), args.get("bias"), **attrs)
    return (gamma + 1.0) * normalized + beta


def identity_affine(args):
    return "scale" in args and np.all(args["scale"] == 1) and np.all(args["bias"] == 0)


def gpu_args(tc, op, args, *, active, omit_affine=False):
    if omit_affine:
        assert identity_affine(args), "None is equivalent only to identity affine"
    slots = SLOTS.get(op, ("x",))
    return {name: tc.tensor(array, requires_grad=active and name in slots)
            for name, array in args.items() if not (omit_affine and name in ("scale", "bias"))}


def nonconstant(a):
    assert a.size > 0 and np.isfinite(a).all() and np.min(a) < np.max(a), "empty, nonfinite or constant array"


def exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
