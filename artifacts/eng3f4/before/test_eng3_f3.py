"""ENG3-F3 shipped F1 tests and REV3 rulings.
Prior art: ENG3-F1/REV1 RLIMIT_AS receipt (2026), pytest (2004+),
DeMillo/Lipton/Sayward mutation testing (1978). Taken: F1 test bodies and
child; ours: no-device query and ruled zero-extent cases. Literature
unverified -- lead to check: CUDA cudaGetDevice, NumPy array is too big.
"""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

import numpy as np
import pytest

PACKAGE = os.environ.get("ENG3F1_PACKAGE_PARENT", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, PACKAGE)
import tensor_cuda as tc

DTYPES = {"float32": 4, "float16": 2, "bfloat16": 2, "int64": 8, "bool": 1, "uint8": 1}


def call(hook, shape, dt="float32", device="cpu"):
    if hook == "_core_full":
        return tc._C._core_full(shape, 1.0, dt, device)
    return tc._C._core_zeros(shape, dt, device)



def child(mode):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if mode == "overflow":
        rows = []
        for dt, width in DTYPES.items():
            q = (2**63 - 1) // width
            overflow = [q + 1] if width > 1 else [2**62, 4]
            for hook in ("_core_full", "_core_zeros"):
                for shape in (overflow, [0] + overflow, [2**62] if width >= 2 else overflow):
                    print("CALL", hook, dt, shape, flush=True)
                    try:
                        call(hook, shape, dt)
                    except OverflowError as e:
                        assert "overflow" in str(e), str(e)
                        rows.append({"hook": hook, "dtype": dt, "shape": shape, "error": type(e).__name__})
                    else:
                        raise AssertionError(("byte overflow accepted", hook, dt, shape))
            # Public path remains checked too, independently of private entries.
            with pytest.raises(ValueError, match="overflow"):
                tc.empty(overflow, dtype=dt, device="cpu")
            boundary = call("_core_zeros", [0, q], dt)
            assert boundary.shape == (0, q) and boundary.data_ptr() == 0
            rows.append({"dtype": dt, "boundary_shape": [0, q], "bound_bytes": q * width,
                         "representable_max_multiple": True, "accepted": True})
        print(json.dumps(rows), flush=True)
    elif mode == "memoryerror":
        mapped = int(Path("/proc/self/statm").read_text().split()[0]) * os.sysconf("SC_PAGE_SIZE")
        limit = mapped + 16 * 1024**2
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        print(json.dumps({"mapped_bytes": mapped, "limit": limit, "shape": [2**25]}), flush=True)
        for hook in ("_core_full", "_core_zeros"):
            print("CALL", hook, "malloc-null", flush=True)
            with pytest.raises(MemoryError):
                call(hook, [2**25])
        # At the largest legal byte multiple, malloc fails, not byte arithmetic.
        for dt, width in DTYPES.items():
            q = (2**63 - 1) // width
            with pytest.raises(MemoryError):
                call("_core_zeros", [q], dt)
        print("MEMORYERROR_CHILD_OK", flush=True)
    else:
        raise ValueError(mode)



def run_child(mode):
    p = subprocess.run([sys.executable, "-B", "-P", __file__, "child", mode],
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True, timeout=120)
    print(p.stdout)
    sys.stderr.write(p.stderr)
    return p



def test_f1_cpu_memoryerror():
    p = run_child("memoryerror")
    assert p.returncode == 0 and "MEMORYERROR_CHILD_OK" in p.stdout, ("MemoryError child must exit 0", p.returncode, p.stdout, p.stderr)



def test_f1_cpu_rank_negative():
    for hook in ("_core_full", "_core_zeros"):
        with pytest.raises(ValueError, match="rank.*8"):
            call(hook, [1] * 9)
        with pytest.raises(ValueError, match="negative"):
            call(hook, [-1])
        assert call(hook, [0] + [1] * 7).shape == (0,) + (1,) * 7

    source = tc.tensor(np.ones((1,) * 9, np.float32), device="cpu")
    for hook in ("_core_clone", "_core_to"):
        with pytest.raises(ValueError, match="rank.*8"):
            tc._C._core_clone(source) if hook == "_core_clone" else tc._C._core_to(source, "cpu")



def test_f1_gpu_device_index():
    import ctypes
    rt = ctypes.CDLL("libcudart.so.12")
    current = ctypes.c_int(-1)
    if rt.cudaGetDevice(ctypes.byref(current)) != 0:
        pytest.skip("CUDA device unavailable")
    source = tc.tensor([1.0, 2.0], device="cpu")
    for hook in ("_core_full", "_core_zeros", "_core_to"):
        for index in (2147483647, current.value + 1):
            device = f"cuda:{index}"
            with pytest.raises(ValueError) as exc:
                tc._C._core_to(source, device) if hook == "_core_to" else call(hook, [2], device=device)
            assert hook in str(exc.value) and device in str(exc.value)
        for device in ("cuda", f"cuda:{current.value}"):
            t = tc._C._core_to(source, device) if hook == "_core_to" else call(hook, [2], device=device)
            assert t.device.startswith("cuda") and t.shape == (2,)



@pytest.mark.parametrize("dtype", ["float16", "float32", "bfloat16"])
def test_f3_cpu_zero_extent_astype_ruling(dtype):
    """ENG3R3-F01, closed by lead ruling (ENG3_F3_REV3_FIXES.md; A1.2):
    refuse nominal byte overflow even with zero extent, matching NumPy and
    P1T-FIX2. This pins the existing rule; kernels.cu remains read-only.
    """
    # Prior art: NumPy nominal-size refusal and P1T-FIX2 (2026), as ruled
    # in ENG3R3-F01; pin the existing constructor rule, no new byte rule.
    shape = [0, 2**62]
    source = tc.empty(tuple(shape), dtype="uint8", device="cpu")
    assert source.shape == tuple(shape) and source.data_ptr() == 0
    with pytest.raises(OverflowError) as caught:
        source.astype(dtype)
    assert str(shape) in str(caught.value)


@pytest.mark.parametrize("dtype", ["float32", "int64"])
def test_f3_cpu_zero_extent_zeros_ruling(dtype):
    """ENG3R3-F01 / ENG3_F3_REV3_FIXES.md, Amendment 1 A1.2:
    pin P1T-FIX2's zero-extent nominal byte refusal, as ruled to match NumPy.
    """
    shape = [0, 2**62]
    with pytest.raises(OverflowError) as caught:
        tc._C._core_zeros(shape, dtype, "cpu")
    assert str(shape) in str(caught.value)


@pytest.mark.parametrize("hook", ["_core_full", "_core_zeros", "_core_to"])
@pytest.mark.parametrize("device", ["cuda", "cuda:0"], ids=["bare", "indexed"])
def test_f3_cpu_no_device_query(hook, device):
    # Prior art: public tensor_empty/P1T-FIX4 checked query (2026).
    # Real runtime with no visible device; no stub or synthetic CUDA error.
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    source = tc.tensor([1.0, 2.0], device="cpu")
    with pytest.raises(RuntimeError) as caught:
        tc._C._core_to(source, device) if hook == "_core_to" else call(hook, [0], device=device)
    message = str(caught.value)
    assert hook in message and "cudaGetDevice" in message, message
    assert "cudaError" in message and ": " in message, message
    assert "invalid device" not in message, message
    print("ENG3F3_QUERY", hook, device, message)


if __name__ == "__main__":
    assert sys.argv[1] == "child"
    child(sys.argv[2])
