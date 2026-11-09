"""A7 D2/D3 child-owned device evidence, read-only validation.

Prior art: NVIDIA CUDA Runtime API 12.6 (2024), cudaGetDevice,
cudaGetDeviceProperties_v2, cudaRuntimeGetVersion, cudaDriverGetVersion;
CPython ctypes (PSF, Python 2.5, 2006) C ABI calls; SHA-256 FIPS180-4
(NIST, 2015). Source: CUDA 12.6 cuda_runtime_api.h:172,1816,9958,9987;
driver_types.h:2284-2286 (name is the first 256-byte field).
Taken: the existing CUDA runtime queries through the pinned engine's loaded
ELF dependency handle. Ours: the order's exact evidence conjunction and
per-line mode derivation. No attestation or numerical algorithm is claimed.
Historical API attribution unverified -- lead to check: CUDA Runtime API
12.6 device management version management; Python ctypes 2.5 history.
"""
import ctypes
import datetime
import hashlib
import os
from pathlib import Path

TOOL = Path(__file__).resolve().parent
CALLS = ("cudaGetDevice", "cudaGetDeviceProperties_v2", "cudaRuntimeGetVersion", "cudaDriverGetVersion")


def identity(path):
    p = Path(path).resolve(strict=True)
    return dict(resolved_path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def capture(program, engine=None):
    """Called by the receipt's writer, before pytest; never by the parent.

    CUDA 12.6's cudaDeviceProp begins with char name[256]. An aligned 64KiB
    storage block reserves more than its complete declared struct (header
    source captured in DEVICE_API_SOURCES.json); only that first field is
    decoded. This calls the v2 ABI explicitly, not the legacy struct ABI.
    """
    result = dict(program=identity(program), CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  device_name=None, cuda_runtime_version=None, cuda_driver_version=None,
                  device_ordinal=None, native_status={}, engine_sha256=None,
                  provider="pinned-engine:libcudart", query_error=None,
                  captured_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if engine is None:
        result["query_error"] = "CPU stand-in: engine not loaded"
        return result
    try:
        result["engine_sha256"] = identity(engine)["sha256"]
        runtime = ctypes.CDLL(str(Path(engine).resolve(strict=True)))
        def integer(name):
            out = ctypes.c_int(-1)
            fn = getattr(runtime, name)
            fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
            fn.restype = ctypes.c_int
            status = fn(ctypes.byref(out))
            result["native_status"][name] = str(status)
            if status != 0:
                raise RuntimeError(name + " status " + str(status))
            return out.value
        # Record versions even when the device is hidden; none of these
        # queries allocates device memory or executes a test/kernel.
        result["cuda_runtime_version"] = str(integer("cudaRuntimeGetVersion"))
        result["cuda_driver_version"] = str(integer("cudaDriverGetVersion"))
        ordinal = integer("cudaGetDevice")
        result["device_ordinal"] = str(ordinal)
        storage = (ctypes.c_longdouble * 4096)()
        fn = runtime.cudaGetDeviceProperties_v2
        fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
        fn.restype = ctypes.c_int
        status = fn(ctypes.byref(storage), ordinal)
        result["native_status"]["cudaGetDeviceProperties_v2"] = str(status)
        if status != 0:
            raise RuntimeError("cudaGetDeviceProperties_v2 status " + str(status))
        result["device_name"] = ctypes.string_at(ctypes.addressof(storage), 256).split(b"\0", 1)[0].decode("ascii")
    except Exception as exc:
        result["query_error"] = str(exc) or type(exc).__name__
    return result


def classify(data, seal):
    """Positive evidence is a conjunction, independent of every declared mode.

    CPU receipts remain usable for line sensitivity checks. Any attempted
    positive device claim must satisfy every field, including the sealed
    real child's path/digest. An empty CVD cannot establish a device line.
    This is consistency validation of editable receipts, not authentication.
    """
    e = data.get("device_evidence")
    if not isinstance(e, dict):
        return "cpu", []
    errors = []
    claimed = e.get("device_name") is not None
    try:
        assert e["program"] == dict(resolved_path=str((TOOL / "run_gpu_tests.py").resolve()),
                                    sha256=seal["tool_files"]["run_gpu_tests.py"]), "child program differs from sealed run_gpu_tests.py"
        assert type(e["CUDA_VISIBLE_DEVICES"]) is str and e["CUDA_VISIBLE_DEVICES"] != "", "CUDA_VISIBLE_DEVICES empty or missing"
        assert e["CUDA_VISIBLE_DEVICES"] == data["CUDA_VISIBLE_DEVICES"], "child CUDA_VISIBLE_DEVICES evidence mismatch"
        assert type(e["device_name"]) is str and bool(e["device_name"].strip()), "device name absent"
        for name in ("cuda_runtime_version", "cuda_driver_version", "device_ordinal"):
            assert type(e[name]) is str and e[name].isascii() and e[name].isdecimal(), name + " is not an exact decimal string"
            assert name == "device_ordinal" or int(e[name]) > 0, name + " is zero"
        assert e["native_status"] == {name: "0" for name in CALLS}, "native query status not successful"
        assert e["engine_sha256"] == seal["engine_sha256"], "device query engine differs from pin"
        assert e["provider"] == "pinned-engine:libcudart" and e["query_error"] is None, "device query unavailable"
        datetime.datetime.fromisoformat(e["captured_at"])
    except Exception as exc:
        if claimed:
            errors.append("device evidence: " + (str(exc) or type(exc).__name__))
        return "cpu", errors
    return "device", errors
