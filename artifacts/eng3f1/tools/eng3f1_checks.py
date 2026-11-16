"""ENG3-F1 CPU refusals and GPU-index cases; tests exercise the real engine.
Prior art: REV1 receipt 028 RLIMIT_AS (2026), ISO C++17/CWE-190/CWE-252,
pytest assertions; ours: operand matrix and child exit checks. Literature
unverified -- lead to check: ISO C++17 expr overflow, CWE-190 CWE-252.
"""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

import numpy as np
import pytest
import tensor_cuda as tc

DTYPES = {"float32": 4, "float16": 2, "bfloat16": 2, "int64": 8, "bool": 1, "uint8": 1}
INVALID = ["", "CPU", " cpu", "cpu ", "cpu:garbage", "cpuBAD", "gpu", "cudaBAD", "cuda:",
           "cuda:-1", "cuda:+0", "cuda:0x0", "cuda:1.0", "cuda:0junk", "cuda:2147483647"]


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


def test_f1_cpu_overflow():
    p = run_child("overflow")
    assert p.returncode == 0, ("overflow child must exit 0", p.returncode, p.stdout, p.stderr)


def test_f1_cpu_memoryerror():
    p = run_child("memoryerror")
    assert p.returncode == 0 and "MEMORYERROR_CHILD_OK" in p.stdout, ("MemoryError child must exit 0", p.returncode, p.stdout, p.stderr)


def test_f1_cpu_device_strings():
    source = tc.tensor([1.0, 2.0], device="cpu")
    for hook in ("_core_full", "_core_zeros", "_core_to"):
        for device in INVALID:
            print("CALL", hook, repr(device), flush=True)
            with pytest.raises(ValueError) as exc:
                if hook == "_core_to":
                    tc._C._core_to(source, device)
                else:
                    call(hook, [2], device=device)
            assert hook in str(exc.value) and repr(device) in str(exc.value), str(exc.value)
    for hook in ("_core_full", "_core_zeros"):
        assert call(hook, [2], device="cpu").device == "cpu"
        # No device allocation/call is required for a zero-element cuda tensor.
        assert call(hook, [0], device="cuda").shape == (0,)
    assert tc._C._core_to(source, "cpu").device == "cpu"


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
    rt = ctypes.CDLL("/usr/local/cuda-12.6/lib64/libcudart.so")
    current = ctypes.c_int(-1)
    assert rt.cudaGetDevice(ctypes.byref(current)) == 0
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


if __name__ == "__main__":
    assert sys.argv[1] == "child"
    child(sys.argv[2])
