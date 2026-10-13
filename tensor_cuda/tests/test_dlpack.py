"""P1T-M1: DLPack interop regression tests (Tensor.__dlpack__ / __dlpack_device__ / data_ptr / fill_ / mark_modified /
revision, tensor_cuda.from_dlpack) and Amendment 1 (tensor_cuda.empty). torch (CUDA) is the external peer.
Author-run tests: a baseline, not blind verification.

What each group pins, and the registered mutant it must kill (order P1T_M1_DLPACK.md, G4):
  * pointer identity + write-through, export and import, fp32/fp16, 0-d / 1-d / [1, 512, 1193]:  (a) import copies.
  * export lifetime (producer Tensor dropped, gc, same-size reuse bait, then read through torch):  (b) raw-pointer owner.
  * export stream ordering (TensorCUDA write queued behind a sleep; torch reads on a side stream):  (c) no event wait.
  * import release ordering (TensorCUDA write queued behind a sleep on imported memory; the tensor dies; torch reuses
    the freed block on the producer's side stream):  (d) deleter called without draining the engine stream.
  * import stream ordering (torch writes on a side stream behind a sleep; TensorCUDA reads on its own stream).
  * refusals: non-contiguous, CPU (torch object and bare capsule), unsupported dtype, read-only, consumed capsule,
    copy=True, foreign dl_device, stream=0; accepted: a size-1 dimension with an arbitrary stride.
  * offsets: the Python layer exposes no non-zero-offset view (reshape shares storage at offset 0; slice copies), so a
    reshape view is tested for sharing, and a ctypes-built producer supplies a real non-zero byte_offset on import,
    which must survive a re-export (data + byte_offset = element 0).
  * data_ptr, fill_ (in place, legacy stream, existing kernel, bumps revision), mark_modified, revision.
  * Amendment 1: empty's shape/dtype/requires_grad for (), [7], [1, 512, 7160] x fp32/fp16, zeros' shape conventions,
    data_ptr round-trip through DLPack both ways.

P1T-FIX1 additions (review P1T-REV round 1; tests named test_fix1_*), with the registered mutant each one must kill:
  * F02: tc.from_dlpack(None) raises TypeError.
  * F03: empty refuses rank > 8 (mutant (c): the rank check removed), element-count and byte-size overflow; import
    refuses count, byte, byte_offset + bytes and address overflow (ValueError, capsule left unconsumed).
  * F04: an imported Storage's last reference released on a foreign (non-Python) thread, normally (deleter called once)
    and during interpreter finalization (deleter NOT called: a counted deliberate leak, no GIL acquisition, no abort).
  * F05: an imported Storage dropped while another device is current: an LD_PRELOAD shim over the shared CUDA runtime
    simulates device 1 being current on this one-GPU host; the destructor must switch to device 0, sync there, restore
    device 1, and on a failed switch must leak instead of releasing (mutant (d): the device switch removed).
  * F06: the export context is owned exactly once (capsule GC and consumer deleter paths); the bad_alloc path itself is
    verified by reasoning + RAII (unique_ptr until PyCapsule_New succeeds).

P1T-FIX2 additions (review P1T-REV2 round 2; tests named test_fix2_*), with the registered mutant each one must kill:
  * N2 (zero-extent rule): shapes with a 0 first / middle / last next to 2^62 and 2^63-1 extents are refused with
    ValueError by empty (cpu, cuda), from_dlpack (capsule left unconsumed) and __dlpack__ (a reshape of an empty
    tensor); zero-extent shapes whose bound fits stay empty tensors. Mutant (b): the zero exemption restored.
  * N4: a producer deleter that throws a C++ exception (native, built here) on the import release, on the main thread
    and on a foreign pthread: contained, counted, called once, rc 0. Mutant (d2): the catch removed.
  * N5: interpreter-exit coordination of foreign-thread releases (the core_shutdown.py pattern, torch-free): a release
    already taking the GIL when the exit hook runs completes before finalization (in-flight wait); one that starts
    during the wait, or after the hook, leaks without touching the GIL. Mutant (e): the in-flight wait removed.
    (FIX2's "the wait is bounded and reported" test is gone: it pinned round-3 finding V3-01, see below.)

P1T-FIX3 additions (review P1T-REV3 round 3; tests named test_fix3_*):
  * V3-01 (exit quiescence): the exit hook waits, with no bound, until every admitted release has finished, and logs
    at a bounded rate meanwhile: foreign-thread deleters of 0.1 s, 5.5 s and 12 s across the default 5 s diagnostic
    delay, ORDINARY interpreter exit (no os._exit): rc 0, deleter completed, then finalization; nothing in flight (no
    wait); the hook called from inside the main thread's own admitted release (no self-wait). The "never admitted"
    (nested) and "after closing" cases are the N5 tests above. Mutant (a): FIX2's bounded return restored.
  * V3-02 (tc_signal aggregate byte accounting) is tested in tc_signal/tests/test_fix3_findings.py.

P1T-FIX5 additions (review P1T-REV5 round 5; tests named test_fix5_*):
  * V5-04 (fill_ launch geometry): synthetic metadata over a small real allocation, an LD_PRELOAD shim capturing fill_'s
    cudaLaunchKernel: the largest accepted extent (2^31-1)*256 launches grid.x = 2^31-1 x 256 (exact cover; the launch
    is trapped, never run); one more element and 2^40+1 are refused with ValueError before any launch, revision
    unchanged; zero size launches nothing; a small extent is unchanged; the cited limit equals the device's
    cudaDevAttrMaxGridDimX. Mutant (d): the refusal removed.
  * Sweep (d), core: the drain-restore and fill_-unwinding restore failures are recorded (count + first site / device /
    error) before their error-slot entry is consumed.
  * Sweep (c), core: the exit wait run by a NON-main thread never aborts (the GIL is re-taken explicitly, not in a
    destructor); 5 runs, each rc 0. V5-01..V5-03 are tc_signal findings: tc_signal/tests/test_fix5_findings.py.

Availability guard (order: narrow): skip ONLY when torch is not importable (ImportError) or no CUDA device is present
(torch.cuda.device_count() == 0). Any other exception from the probe propagates and the test FAILS.

Prior art: the Python array API standard's data-interchange tests (stream=, max_version=, capsule renaming; Consortium
for Python Data API Standards, 2021-2023); DLPack (dmlc/dlpack, 2017-; v1.3 struct layout mirrored in ctypes below);
the reuse-bait allocation for use-after-free detection and the sleep-then-write stream race follow Project-Voice's T1
runner (run_p1t.py, 2026-10); DeMillo, Lipton & Sayward (1978) mutation testing for the registered mutants. Ours: the
import-release-ordering race (case (d)) and the ctypes producer.
"""
import ctypes
import gc
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import tensor_cuda as tc

SLEEP_CYCLES = 20_000_000          # torch.cuda._sleep cycles (~10 ms on the RTX 4070 SUPER), as Voice's T1 knob
RACE_NUMEL = 1 << 20               # elements per race-test tensor (4 MiB fp32)
RACE_TRIALS = 3                    # trials per race test (each one is a full race; any failure fails)
JUNK_ALLOCS = 4                    # same-size reuse-bait allocations after an owner is dropped
CENSUS_SHAPES = [(), (5,), (1, 512, 1193)]
DTYPES = ["float32", "float16"]
EMPTY_SHAPES = [(), (7,), (1, 512, 7160)]


def _torch_or_skip():
    try:
        import torch
    except ImportError as e:
        pytest.skip(f"torch is not importable: {e}")
    if torch.cuda.device_count() == 0:
        pytest.skip("no CUDA device present")
    return torch


@pytest.fixture
def torch():
    return _torch_or_skip()


def _tdt(torch, dt):
    return {"float32": torch.float32, "float16": torch.float16}[dt]


def _npdt(dt):
    return {"float32": np.float32, "float16": np.float16}[dt]


# ------------------------------------------------------------------------------------------- ctypes DLPack producer
class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", _DLDevice), ("ndim", ctypes.c_int32),
                ("dtype", _DLDataType), ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)), ("byte_offset", ctypes.c_uint64)]


class _DLPackVersion(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]


class _DLManagedTensorVersioned(ctypes.Structure):
    pass


_DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensorVersioned))
_DLManagedTensorVersioned._fields_ = [("version", _DLPackVersion), ("manager_ctx", ctypes.c_void_p),
                                      ("deleter", _DELETER), ("flags", ctypes.c_uint64), ("dl_tensor", _DLTensor)]
_DLManagedTensorVersioned_p = ctypes.POINTER(_DLManagedTensorVersioned)

_PyCapsule_New = ctypes.pythonapi['PyCapsule_New']
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_PyCapsule_GetPointer = ctypes.pythonapi['PyCapsule_GetPointer']
_PyCapsule_GetPointer.restype = ctypes.c_void_p
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_VERSIONED = b"dltensor_versioned"           # module-level: the capsule keeps a char* to it


class _Producer:
    """Test-only DLPack producer over a torch tensor's memory, with full control of shape, strides, byte_offset,
    dtype, device and flags. Its capsule has no PyCapsule destructor, so a refused (unconsumed) capsule leaks its
    manager by design; the consumer path calls the deleter, which drops the owner reference and counts the call."""

    def __init__(self, owner, shape, strides, *, byte_offset=0, code=2, bits=32, flags=0, device_type=2,
                 device_id=None):
        self.owner = owner
        self.deleter_calls = 0
        n = len(shape)
        self._shape = (ctypes.c_int64 * max(n, 1))(*shape)
        self._strides = None if strides is None else (ctypes.c_int64 * max(n, 1))(*strides)
        self._cb = _DELETER(self._delete)
        m = _DLManagedTensorVersioned()
        m.version.major, m.version.minor = 1, 3
        m.manager_ctx = None
        m.deleter = self._cb
        m.flags = flags
        t = m.dl_tensor
        t.data = owner.data_ptr()
        t.device.device_type = device_type
        t.device.device_id = owner.device.index if device_id is None else device_id
        t.ndim = n
        t.dtype.code, t.dtype.bits, t.dtype.lanes = code, bits, 1
        t.shape = ctypes.cast(self._shape, ctypes.POINTER(ctypes.c_int64)) if n else None
        t.strides = (ctypes.cast(self._strides, ctypes.POINTER(ctypes.c_int64))
                     if (n and self._strides is not None) else None)
        t.byte_offset = byte_offset
        self._m = m

    def _delete(self, _self_ptr):
        self.deleter_calls += 1
        self.owner = None

    def capsule(self):
        return _PyCapsule_New(ctypes.addressof(self._m), _VERSIONED, None)


def _peek_versioned(cap):
    """Read the DLTensor fields of an unconsumed 'dltensor_versioned' capsule (does not consume it)."""
    ptr = _PyCapsule_GetPointer(cap, _VERSIONED)
    return ctypes.cast(ptr, _DLManagedTensorVersioned_p).contents


# ------------------------------------------------------------------------------------------------ identity / alias
@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("shape", CENSUS_SHAPES, ids=lambda s: "x".join(map(str, s)) or "0d")
def test_export_pointer_identity_and_write_through(torch, dt, shape):
    t = tc.empty(shape, dtype=dt)
    t.fill_(1.0)
    torch.cuda.synchronize()
    x = torch.from_dlpack(t)
    assert x.data_ptr() == t.data_ptr()
    assert tuple(x.shape) == shape and x.dtype == _tdt(torch, dt) and x.is_contiguous() and x.is_cuda
    assert t.__dlpack_device__() == (2, torch.cuda.current_device())
    x.fill_(2.0)
    torch.cuda.synchronize()
    h = t.numpy()
    assert h.dtype == _npdt(dt) and h.shape == shape and np.all(h == 2.0)
    t.fill_(3.0)
    torch.cuda.synchronize()
    assert bool(torch.all(x == 3.0).item())


@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("shape", CENSUS_SHAPES, ids=lambda s: "x".join(map(str, s)) or "0d")
def test_import_pointer_identity_and_write_through(torch, dt, shape):
    p = torch.full(shape, 1.0, dtype=_tdt(torch, dt), device="cuda")
    torch.cuda.synchronize()
    u = tc.from_dlpack(p)
    assert u.data_ptr() == p.data_ptr()
    assert tuple(u.shape) == shape and u.dtype == dt and u.device.startswith("cuda") and not u.requires_grad
    p.fill_(2.0)
    torch.cuda.synchronize()
    h = u.numpy()
    assert h.dtype == _npdt(dt) and h.shape == shape and np.all(h == 2.0)
    u.fill_(3.0)
    torch.cuda.synchronize()
    assert bool(torch.all(p == 3.0).item())


def test_capsule_protocol_names_and_single_consumption(torch):
    t = tc.zeros(4)
    legacy = t.__dlpack__()
    old = t.__dlpack__(max_version=(0, 8))
    versioned = t.__dlpack__(max_version=(1, 0))
    assert "dltensor" in repr(legacy) and "dltensor_versioned" not in repr(legacy)
    assert "dltensor" in repr(old) and "dltensor_versioned" not in repr(old)
    assert "dltensor_versioned" in repr(versioned)
    v = _peek_versioned(versioned)
    assert (v.version.major, v.version.minor) == (1, 3) and v.flags == 0
    assert v.dl_tensor.ndim == 1 and v.dl_tensor.shape[0] == 4 and v.dl_tensor.strides[0] == 1
    assert (v.dl_tensor.dtype.code, v.dl_tensor.dtype.bits, v.dl_tensor.dtype.lanes) == (2, 32, 1)
    u = tc.from_dlpack(versioned)                       # bare capsule
    assert u.data_ptr() == t.data_ptr()
    assert "used_dltensor_versioned" in repr(versioned)
    with pytest.raises(ValueError, match="not an unconsumed DLPack capsule"):
        tc.from_dlpack(versioned)
    u2 = tc.from_dlpack(legacy)
    assert "used_dltensor" in repr(legacy) and u2.data_ptr() == t.data_ptr()
    with pytest.raises(TypeError):
        t.__dlpack__(None)                              # every argument is keyword-only
    assert "mark_modified" in tc.Tensor.__dlpack__.__doc__


def test_from_dlpack_asks_producer_for_legacy_stream(torch):
    calls = []
    p = torch.ones(3, device="cuda")

    class Versioned:
        def __dlpack_device__(self):
            return p.__dlpack_device__()

        def __dlpack__(self, *, stream=None, max_version=None):
            calls.append(("v", stream, max_version))
            return p.__dlpack__(stream=stream, max_version=max_version)

    class Old:                                           # predates max_version: from_dlpack must retry without it
        def __dlpack_device__(self):
            return p.__dlpack_device__()

        def __dlpack__(self, *, stream=None):
            calls.append(("o", stream))
            return p.__dlpack__(stream=stream)

    assert tc.from_dlpack(Versioned()).data_ptr() == p.data_ptr()
    assert tc.from_dlpack(Old()).data_ptr() == p.data_ptr()
    assert calls[0] == ("v", 1, (1, 3))
    assert calls[1:] == [("o", 1)]


def test_unconsumed_capsule_releases_its_reference(torch):
    n = 16 << 20                                         # 64 MiB fp32: visible in cudaMemGetInfo
    t = tc.empty(n)
    cap = t.__dlpack__(max_version=(1, 0))
    del t
    gc.collect()
    torch.cuda.synchronize()
    tc.empty_cache()                                     # robust if another test left the transient pool on
    free_held, _ = torch.cuda.mem_get_info()
    del cap                                              # the capsule destructor calls the deleter -> Storage freed
    gc.collect()
    torch.cuda.synchronize()
    tc.empty_cache()
    free_after, _ = torch.cuda.mem_get_info()
    assert free_after - free_held >= (n * 4) // 2


# ------------------------------------------------------------------------------------------------ lifetime
def test_export_lifetime_after_producer_dropped(torch):
    for k in range(RACE_TRIALS):
        v = float(k + 1)
        t = tc.empty(RACE_NUMEL)
        t.fill_(v)
        torch.cuda.synchronize()
        x = torch.from_dlpack(t)
        del t
        gc.collect()
        junk = [tc.empty(RACE_NUMEL).fill_(-7.0) for _ in range(JUNK_ALLOCS)]
        torch.cuda.synchronize()
        assert bool(torch.all(x == v).item()), f"trial {k}: exported memory was freed with the producer Tensor"
        del junk, x
        gc.collect()


def test_import_lifetime_after_producer_dropped(torch):
    for k in range(RACE_TRIALS):
        v = float(k + 1)
        p = torch.full((RACE_NUMEL,), v, device="cuda")
        torch.cuda.synchronize()
        u = tc.from_dlpack(p)
        del p
        gc.collect()
        junk = [torch.full((RACE_NUMEL,), -7.0, device="cuda") for _ in range(JUNK_ALLOCS)]
        torch.cuda.synchronize()
        assert np.all(u.numpy() == v), f"trial {k}: imported memory was released while TensorCUDA held it"
        del junk, u
        gc.collect()


def test_import_deleter_called_once_when_tensor_dies(torch):
    owner = torch.arange(12, dtype=torch.float32, device="cuda")
    prod = _Producer(owner, (12,), (1,))
    u = tc.from_dlpack(prod.capsule())
    x = torch.from_dlpack(u)                             # re-export keeps the imported Storage alive too
    del u
    gc.collect()
    assert prod.deleter_calls == 0
    del x
    gc.collect()
    assert prod.deleter_calls == 1 and prod.owner is None


# ------------------------------------------------------------------------------------------------ stream ordering
def test_export_stream_ordering_torch_side_stream(torch):
    S = torch.cuda.Stream()
    for k in range(RACE_TRIALS):
        v = float(k + 1)
        t = tc.empty(RACE_NUMEL)
        t.fill_(0.0)
        torch.cuda.synchronize()
        torch.cuda._sleep(SLEEP_CYCLES)                  # legacy default stream (TensorCUDA's)
        t.fill_(v)                                       # queued behind the sleep
        h = torch.empty(RACE_NUMEL, dtype=torch.float32, pin_memory=True)
        with torch.cuda.stream(S):
            x = torch.from_dlpack(t)                     # torch passes stream=<S>: the fence must order S
            h.copy_(x, non_blocking=True)
        S.synchronize()
        assert bool(torch.all(h == v).item()), f"trial {k}: consumer stream read before TensorCUDA's write"
        torch.cuda.synchronize()
        del x, t


def test_import_stream_ordering_torch_side_stream(torch):
    S = torch.cuda.Stream()
    for k in range(RACE_TRIALS):
        v = float(k + 1)
        p = torch.zeros(RACE_NUMEL, device="cuda")
        torch.cuda.synchronize()
        with torch.cuda.stream(S):
            torch.cuda._sleep(SLEEP_CYCLES)
            p.fill_(v)                                   # the producer's pending write, on S
            u = tc.from_dlpack(p)                        # asks stream=1: torch orders the legacy stream after S
        h = u.numpy()                                    # TensorCUDA's stream only
        assert np.all(h == v), f"trial {k}: TensorCUDA read before the producer's side-stream write"
        torch.cuda.synchronize()
        del u, p


def test_import_release_drains_engine_stream_before_deleter(torch):
    # The producer's memory belongs to torch's caching allocator, on side stream S. TensorCUDA queues a write on it
    # behind a sleep, then the TensorCUDA tensor dies. If the deleter ran before that write finished, torch would hand
    # the block to the next same-size allocation on S at once, and the late TensorCUDA write would land on it.
    # NO gc.collect() between the release and the reuse: with torch loaded it stalls the host ~60 ms (measured
    # 57-79 ms), longer than the sleep, which closes the race window (G4 r1: mutant (d) survived because of it).
    # `del u` frees the Tensor by refcount, so the Storage destructor runs right there. Warm-ups first, so lazy
    # kernel loading cannot serialize the first trial.
    S = torch.cuda.Stream()
    tc.empty(1).fill_(0.0)
    with torch.cuda.stream(S):
        torch.empty(1, device="cuda").fill_(0.0)
    torch.cuda._sleep(1)
    torch.cuda.synchronize()
    for k in range(RACE_TRIALS):
        v = float(k + 1)
        with torch.cuda.stream(S):
            p = torch.zeros(RACE_NUMEL, device="cuda")
        S.synchronize()
        u = tc.from_dlpack(p)
        ptr = p.data_ptr()
        del p
        torch.cuda._sleep(SLEEP_CYCLES)                  # legacy default stream
        u.fill_(v)                                       # TensorCUDA write on the imported memory, behind the sleep
        del u                                            # last reference: drain, then the deleter
        with torch.cuda.stream(S):
            q = torch.empty(RACE_NUMEL, device="cuda")
            q.fill_(-7.0)
        reused = q.data_ptr() == ptr
        torch.cuda.synchronize()
        assert reused, "precondition: torch did not reuse the released block on S, so this trial cannot detect"
        assert bool(torch.all(q == -7.0).item()), f"trial {k}: the deleter ran before TensorCUDA's queued write"
        del q


# ------------------------------------------------------------------------------------------------ refusals
def test_refuses_non_contiguous(torch):
    for bad in (torch.zeros(4, 6, device="cuda").t(), torch.zeros(4, 6, device="cuda")[:, ::2]):
        with pytest.raises(BufferError, match="non-contiguous"):
            tc.from_dlpack(bad)
    owner = torch.zeros(6, device="cuda")
    prod = _Producer(owner, (2, 3), (1, 2))
    cap = prod.capsule()
    with pytest.raises(BufferError, match="non-contiguous"):
        tc.from_dlpack(cap)
    assert "used_" not in repr(cap) and prod.deleter_calls == 0       # refused capsules stay unconsumed


def test_accepts_size1_dim_with_arbitrary_stride(torch):
    owner = torch.arange(6, dtype=torch.float32, device="cuda")
    prod = _Producer(owner, (2, 1, 3), (3, 12345, 1))
    u = tc.from_dlpack(prod.capsule())
    assert tuple(u.shape) == (2, 1, 3) and u.data_ptr() == owner.data_ptr()
    assert np.array_equal(u.numpy(), np.arange(6, dtype=np.float32).reshape(2, 1, 3))
    q = torch.arange(4, dtype=torch.float32, device="cuda").reshape(4, 1)
    q = q.as_strided((4, 1), (1, 777))                    # torch producer, size-1 dim with an odd stride
    w = tc.from_dlpack(q)
    assert tuple(w.shape) == (4, 1) and np.array_equal(w.numpy().ravel(), np.arange(4, dtype=np.float32))


def test_refuses_cpu(torch):
    with pytest.raises(BufferError, match="CPU"):
        tc.from_dlpack(torch.zeros(3))                   # object with __dlpack__: refused before export
    cap = torch.utils.dlpack.to_dlpack(torch.zeros(3))   # bare CPU capsule: refused in C++
    with pytest.raises(BufferError, match="CPU"):
        tc.from_dlpack(cap)
    assert "used_" not in repr(cap)
    c = tc.zeros(3, device="cpu")
    with pytest.raises(BufferError):
        c.__dlpack__()
    with pytest.raises(BufferError):
        c.__dlpack_device__()


def test_refuses_unsupported_dtype_and_flags(torch):
    for dt in (torch.float64, torch.int32, torch.int8):
        with pytest.raises(BufferError, match="no exact TensorCUDA dtype"):
            tc.from_dlpack(torch.zeros(3, dtype=dt, device="cuda"))
    owner = torch.zeros(4, device="cuda")
    with pytest.raises(BufferError, match="read-only"):
        tc.from_dlpack(_Producer(owner, (4,), (1,), flags=1).capsule())
    with pytest.raises(BufferError, match="lanes=4"):
        p = _Producer(owner, (1,), (1,))
        p._m.dl_tensor.dtype.lanes = 4
        tc.from_dlpack(p.capsule())
    with pytest.raises(TypeError):
        tc.from_dlpack(object())


def test_export_argument_contract(torch):
    t = tc.zeros(4)
    with pytest.raises(BufferError, match="copy=True"):
        t.__dlpack__(copy=True)
    with pytest.raises(BufferError, match="dl_device"):
        t.__dlpack__(dl_device=(1, 0))
    with pytest.raises(ValueError, match="ambiguous"):
        t.__dlpack__(stream=0)
    for ok in (dict(stream=None), dict(stream=-1), dict(stream=1), dict(stream=2), dict(copy=False),
               dict(dl_device=(2, t.__dlpack_device__()[1]))):
        tc.from_dlpack(t.__dlpack__(**ok))               # consumed, so nothing leaks


# ------------------------------------------------------------------------------------------------ offsets / views
def test_reshape_view_shares_storage_and_revision(torch):
    t = tc.empty(2, 6)
    t.fill_(1.0)
    v = t.reshape([3, 4])
    assert v.data_ptr() == t.data_ptr()
    x = torch.from_dlpack(v)
    assert tuple(x.shape) == (3, 4) and x.data_ptr() == t.data_ptr()
    x.fill_(4.0)
    torch.cuda.synchronize()
    assert np.all(t.numpy() == 4.0)
    r = t.revision
    v.mark_modified()
    assert t.revision == r + 1                           # one Storage, one revision


def test_nonzero_byte_offset_import_and_reexport(torch):
    owner = torch.arange(16, dtype=torch.float32, device="cuda")
    off = 4 * 4                                          # element 4
    prod = _Producer(owner, (12,), (1,), byte_offset=off)
    u = tc.from_dlpack(prod.capsule())
    assert u.data_ptr() == owner.data_ptr() + off
    assert np.array_equal(u.numpy(), np.arange(4, 16, dtype=np.float32))
    cap = u.__dlpack__(max_version=(1, 0))
    d = _peek_versioned(cap).dl_tensor
    assert d.data == owner.data_ptr() and d.byte_offset == off   # base pointer + byte_offset = element 0
    x = torch.from_dlpack(cap)
    assert x.data_ptr() == owner.data_ptr() + off and bool(torch.equal(x, owner[4:]))
    with pytest.raises(BufferError, match="byte_offset"):
        tc.from_dlpack(_Producer(owner, (3,), (1,), byte_offset=2).capsule())


def test_zero_size_tensors(torch):
    x = torch.from_dlpack(tc.empty(0))
    assert tuple(x.shape) == (0,)
    u = tc.from_dlpack(torch.zeros(0, 3, device="cuda"))
    assert tuple(u.shape) == (0, 3) and u.numpy().shape == (0, 3)


# ------------------------------------------------------------------------------------------------ data_ptr / fill_ / revision
def test_data_ptr_fill_and_revision(torch):
    t = tc.zeros(8, dtype="float16")
    assert isinstance(t.data_ptr(), int) and t.data_ptr() > 0
    r0 = t.revision
    torch.cuda._sleep(SLEEP_CYCLES)                      # fill_ is queued (asynchronous) on the legacy stream ...
    assert t.fill_(1.5) is t
    assert np.all(t.numpy() == np.float16(1.5))          # ... and numpy() reads after it on the same stream
    assert t.revision == r0 + 1
    t.mark_modified()
    assert t.revision == r0 + 2
    with pytest.raises(AttributeError):
        t.revision = 0
    with pytest.raises(RuntimeError):
        tc.zeros(3, dtype="int64").fill_(1.0)            # fill_kernel is float-only
    with pytest.raises(RuntimeError):
        tc.zeros(3, device="cpu").fill_(1.0)


# ------------------------------------------------------------------------------------------------ Amendment 1: empty
@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("shape", EMPTY_SHAPES, ids=lambda s: "x".join(map(str, s)) or "0d")
def test_empty_shape_dtype_and_dlpack_roundtrip(torch, dt, shape):
    e = tc.empty(shape, dtype=dt)
    assert tuple(e.shape) == shape and e.dtype == dt and not e.requires_grad and e.device.startswith("cuda")
    x = torch.from_dlpack(e)
    assert x.data_ptr() == e.data_ptr() and tuple(x.shape) == shape and x.dtype == _tdt(torch, dt)
    back = tc.from_dlpack(x)
    assert back.data_ptr() == e.data_ptr()


def test_empty_shape_conventions_match_zeros(torch):
    for args in [(2, 3), ((2, 3),), ([2, 3],), (0,), (np.int64(4),), ((7,),)]:
        assert tc.empty(*args).shape == tc.zeros(*args).shape
    # Registered difference (P1T-M1 report, Amendment 1): zeros() / zeros(()) return shape (1,) because
    # np.ascontiguousarray promotes a 0-d host array to 1-d (__init__.py _factory); empty returns the requested 0-d.
    assert tc.empty().shape == () and tc.empty(()).shape == ()
    assert tc.zeros().shape == (1,)
    assert tc.empty(2, 3).dtype == "float32"
    with pytest.raises(ValueError):
        tc.empty(2, -1)
    with pytest.raises(RuntimeError):
        tc.empty(2, dtype="float99")
    assert "empty" in tc.__all__ and "from_dlpack" in tc.__all__


# =================================================================================================== P1T-FIX1
# Regression tests for the P1T-REV round-1 findings (order P1T_FIX1_REVIEW_FINDINGS.md). Prior art per test below;
# the subprocess isolation for crash-prone paths follows the reviewer's host_checks.py / core_shutdown.py probes.
_SUBPROC_TIMEOUT = 60


def _subproc_env(**extra):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra)
    return env


# A torch-free ctypes producer for the subprocess scripts (same v1.3 layout as _Producer above), over TensorCUDA memory.
_SUBPROC_PRODUCER = r"""
import ctypes, gc, json, sys
import tensor_cuda as tc

class DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]
class DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]
class DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", DLDevice), ("ndim", ctypes.c_int32), ("dtype", DLDataType),
                ("shape", ctypes.POINTER(ctypes.c_int64)), ("strides", ctypes.POINTER(ctypes.c_int64)),
                ("byte_offset", ctypes.c_uint64)]
class Version(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]
class MTV(ctypes.Structure):
    pass
DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(MTV))
MTV._fields_ = [("version", Version), ("manager_ctx", ctypes.c_void_p), ("deleter", DELETER),
                ("flags", ctypes.c_uint64), ("dl_tensor", DLTensor)]
PyCapsule_New = ctypes.pythonapi['PyCapsule_New']
PyCapsule_New.restype = ctypes.py_object
PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
NAME = b"dltensor_versioned"

class Producer:
    def __init__(self, owner, n):
        self.owner, self.calls = owner, 0
        self.shape = (ctypes.c_int64 * 1)(n)
        self.cb = DELETER(self._delete)
        m = MTV()
        m.version.major, m.version.minor = 1, 3
        m.deleter = self.cb
        t = m.dl_tensor
        t.data = owner.data_ptr()
        t.device.device_type, t.device.device_id = 2, 0
        t.ndim = 1
        t.dtype.code, t.dtype.bits, t.dtype.lanes = 2, 32, 1
        t.shape = ctypes.cast(self.shape, ctypes.POINTER(ctypes.c_int64))
        self.m = m
    def _delete(self, _p):
        self.calls += 1
    def capsule(self):
        return PyCapsule_New(ctypes.addressof(self.m), NAME, None)
"""


# ------------------------------------------------------------------------------------------------ F02
def test_fix1_f02_from_dlpack_none_is_type_error():
    with pytest.raises(TypeError):
        tc.from_dlpack(None)
    with pytest.raises(TypeError):
        tc._C._from_dlpack_capsule(None)


# ------------------------------------------------------------------------------------------------ F03
# Prior art: checked arithmetic (__builtin_mul_overflow, GCC 5 2015; C23 ckd_mul); the shapes are the reviewer's
# core_edges.py census plus import-side byte_offset/address cases. Descriptors only: nothing is allocated or read.
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fix1_f03_empty_refuses_rank_above_8(device):
    with pytest.raises(ValueError, match="maximum rank 8"):
        tc.empty((1,) * 9, device=device)
    assert tc.empty((1,) * 8, device=device).shape == (1,) * 8           # rank 8 is the maximum, still accepted


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("shape,dtype,frag", [
    ((1 << 62,), "float32", "byte size"),                                # numel fits, numel*4 wraps to 0
    ((1 << 61,), "float32", "byte size"),                                # 2^63 bytes: one past INT64_MAX
    ((1 << 62,), "float16", "byte size"),
    ((1 << 32, 1 << 32), "float32", "element count"),                    # numel itself wraps
    ((3, 1 << 62, 1 << 2), "float32", "element count"),
], ids=["2^62_f32", "2^61_f32", "2^62_f16", "2^32x2^32", "3x2^62x4"])
def test_fix1_f03_empty_refuses_count_and_byte_overflow(device, shape, dtype, frag):
    with pytest.raises(ValueError, match=frag):
        tc.empty(shape, dtype=dtype, device=device)


def test_fix1_f03_empty_zero_extent_is_empty_not_overflow():
    # P1T-FIX2 (N2) changed this case: FIX1 pinned (0, 1 << 62, 1 << 62) as a real empty tensor, which is exactly the
    # defect the round-2 review reproduced (its strides overflow int64). A zero-extent shape whose other extents fit
    # the zero-extent rule is still a real empty tensor; the old shape is now refused (test_fix2_n2_* below).
    t = tc.empty((0, 1 << 30, 1 << 30), device="cpu")                     # count 0: a real empty tensor
    assert t.shape == (0, 1 << 30, 1 << 30)


def test_fix1_f03_import_refuses_overflow(torch):
    owner = torch.zeros(4, device="cuda")
    cases = []
    p = _Producer(owner, (1 << 62,), None)                               # bytes wrap
    cases.append(("byte size", p))
    p = _Producer(owner, (1 << 32, 1 << 32), None)                       # count wraps
    cases.append(("element count", p))
    p = _Producer(owner, (1,), None, byte_offset=1 << 63)                # byte_offset beyond int64
    cases.append(("byte_offset", p))
    p = _Producer(owner, (2,), None, byte_offset=(1 << 63) - 4)          # byte_offset + bytes wraps int64
    cases.append(("byte_offset", p))
    p = _Producer(owner, (4,), None)
    p._m.dl_tensor.data = (1 << 64) - 8                                  # data + bytes wraps the address space
    cases.append(("address space", p))
    for frag, prod in cases:
        cap = prod.capsule()
        with pytest.raises(ValueError, match=frag):
            tc.from_dlpack(cap)
        assert "used_" not in repr(cap) and prod.deleter_calls == 0     # refused before consumption
    with pytest.raises(BufferError, match="ndim 9"):
        tc.from_dlpack(_Producer(owner, (1,) * 9, None).capsule())


# ------------------------------------------------------------------------------------------------ F04
# A REAL imported Storage whose last reference is a TensorCUDA export manager; its deleter is called on a foreign
# (pthread, non-Python) thread created through libc, once normally and once from a module-global __del__ during
# interpreter finalization. Prior art: the reviewer's core_shutdown.py + release_worker.cpp (same scenario, native
# helper); CPython 3.12 finalization rules. Ours: the torch-free ctypes/pthread form and the leak counter check.
_F04_SCRIPT = _SUBPROC_PRODUCER + r"""
mode = sys.argv[1]
libc = ctypes.CDLL(None)
pthread_create = libc.pthread_create
pthread_create.argtypes = [ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
pthread_create.restype = ctypes.c_int
pthread_join = libc.pthread_join
pthread_join.argtypes = [ctypes.c_ulong, ctypes.c_void_p]
pthread_join.restype = ctypes.c_int
GetPointer = ctypes.pythonapi['PyCapsule_GetPointer']
GetPointer.restype, GetPointer.argtypes = ctypes.c_void_p, [ctypes.py_object, ctypes.c_char_p]
SetName = ctypes.pythonapi['PyCapsule_SetName']
SetName.restype, SetName.argtypes = ctypes.c_int, [ctypes.py_object, ctypes.c_char_p]

owner = tc.empty(16)
owner.fill_(3.0)
tc.synchronize()
prod = Producer(owner, 16)
u = tc.from_dlpack(prod.capsule())             # imported Storage: its release calls prod's deleter (GIL needed)
cap = u.__dlpack__(max_version=(1, 3))         # TensorCUDA export manager: holds the imported Storage
del u
gc.collect()
mptr = GetPointer(cap, NAME)
assert SetName(cap, b"used_dltensor_versioned") == 0           # we own the manager now
deleter_fp = ctypes.cast(mptr + 16, ctypes.POINTER(ctypes.c_void_p)).contents.value   # DLManagedTensorVersioned.deleter

def foreign_release(pc=pthread_create, pj=pthread_join, fp=deleter_fp, m=mptr, byref=ctypes.byref,
                    ulong=ctypes.c_ulong):
    tid = ulong()
    assert pc(byref(tid), None, fp, m) == 0
    pj(tid.value, None)                        # ctypes releases the GIL for the call

stats = tc._C._dlpack_release_stats
if mode == "normal":
    foreign_release()
    print("RESULT " + json.dumps({"finalizing": sys.is_finalizing(), "deleter_calls": prod.calls,
                                  "stats": dict(stats())}), flush=True)
else:
    class Finalizer:
        def __init__(self):
            self.isf, self.release, self.stats, self.prod, self.dumps = sys.is_finalizing, foreign_release, stats, prod, json.dumps
        def __del__(self):
            fin = self.isf()
            self.release()
            print("RESULT " + self.dumps({"finalizing": fin, "deleter_calls": self.prod.calls,
                                          "stats": dict(self.stats())}), flush=True)
    keep = Finalizer()
    print("ARMED", flush=True)
"""


def _run_f04(mode):
    r = subprocess.run([sys.executable, "-c", _F04_SCRIPT, mode], capture_output=True, text=True,
                       timeout=_SUBPROC_TIMEOUT, env=_subproc_env())
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert r.returncode == 0 and lines, f"rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}"
    return json.loads(lines[-1][len("RESULT "):]), r


def test_fix1_f04_foreign_thread_release_normal_calls_deleter_once():
    res, _ = _run_f04("normal")
    assert res["finalizing"] is False
    assert res["deleter_calls"] == 1, res
    assert res["stats"]["released"] == 1 and res["stats"]["leaked_interpreter_shutdown"] == 0, res


def test_fix1_f04_foreign_thread_release_while_finalizing_leaks_without_gil():
    res, r = _run_f04("finalizing")
    assert "ARMED" in r.stdout
    assert res["finalizing"] is True, "precondition: the release must happen during interpreter finalization"
    assert res["deleter_calls"] == 0, res                     # no PyGILState_Ensure, so no Python deleter
    assert res["stats"]["leaked_interpreter_shutdown"] == 1, res
    assert res["stats"]["released"] == 0, res


# ------------------------------------------------------------------------------------------------ F05
# The host has one GPU, so "device 1 is current" is simulated: an LD_PRELOAD shim interposes cudaGetDevice /
# cudaSetDevice / cudaStreamSynchronize between _tensor_cuda and the shared libcudart.so.12 (core imports them as
# cudaGetDevice@libcudart.so.12, ...). In fake mode the shim reports a simulated current device, logs every switch and
# every stream sync with the simulated device, and always keeps the REAL device at 0. Prior art: LD_PRELOAD symbol
# interposition (glibc ld.so; the classic malloc/syscall interposers); the order's "simulate with cudaSetDevice" brief.
# Ours: this shim.
_F05_SHIM_C = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
typedef int (*get_fn)(int*);
typedef int (*set_fn)(int);
typedef int (*sync_fn)(void*);
static int fake = -1;          /* -1: pass-through; else the simulated current device */
static int fail_set_to = -1;   /* cudaSetDevice(fail_set_to) fails with cudaErrorInvalidDevice (101) */
static char logbuf[4096];
static void note(const char* s) {
  size_t n = strlen(logbuf);
  if (n + strlen(s) + 2 < sizeof logbuf) { strcat(logbuf, s); strcat(logbuf, " "); }
}
/* Python loads extension modules RTLD_LOCAL, so libcudart.so.12 (a dependency of _tensor_cuda) is not in the global
   scope and dlsym(RTLD_NEXT, ...) finds nothing: resolve the real functions through a handle on the loaded runtime. */
static void* real_sym(const char* name) {
  void* h = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_NOLOAD);
  if (!h) h = dlopen("libcudart.so.12", RTLD_LAZY);
  return h ? dlsym(h, name) : NULL;
}
int cudaGetDevice(int* d) {
  static get_fn real;
  if (!real) real = (get_fn)real_sym("cudaGetDevice");
  if (fake >= 0) { *d = fake; return 0; }
  return real(d);
}
int cudaSetDevice(int d) {
  static set_fn real;
  if (!real) real = (set_fn)real_sym("cudaSetDevice");
  if (fake < 0) return real(d);
  char b[32];
  snprintf(b, sizeof b, "set%d%s", d, d == fail_set_to ? "!" : "");
  note(b);
  if (d == fail_set_to) return 101;
  fake = d;
  return real(0);
}
int cudaStreamSynchronize(void* s) {
  static sync_fn real;
  if (!real) real = (sync_fn)real_sym("cudaStreamSynchronize");
  if (fake >= 0) { char b[32]; snprintf(b, sizeof b, "sync@%d", fake); note(b); }
  return real(s);
}
void shim_fake(int d) { fake = d; }
void shim_fail_set(int d) { fail_set_to = d; }
int shim_current(void) { return fake; }
const char* shim_log(void) { return logbuf; }
void shim_clear(void) { logbuf[0] = 0; }
"""

_F05_SCRIPT = _SUBPROC_PRODUCER + r"""
shim = ctypes.CDLL(sys.argv[1])                # already loaded through LD_PRELOAD: the same handle
shim.shim_log.restype = ctypes.c_char_p
mode = sys.argv[2]
owner = tc.empty(16)
owner.fill_(1.0)
tc.synchronize()
prod = Producer(owner, 16)
u = tc.from_dlpack(prod.capsule())             # imported on (real and reported) device 0
u.fill_(2.0)                                   # engine work queued on the imported memory
shim.shim_fake(1)                              # from here the process "runs on device 1"
if mode == "fail":
    shim.shim_fail_set(0)                      # switching back to the Storage's device 0 fails
shim.shim_clear()
del u                                          # last reference: the external Storage destructor runs here
gc.collect()
res = {"log": shim.shim_log().decode().split(), "current": shim.shim_current(), "deleter_calls": prod.calls,
       "stats": dict(tc._C._dlpack_release_stats())}
shim.shim_fake(-1)
print("RESULT " + json.dumps(res), flush=True)
"""


@pytest.fixture(scope="module")
def f05_shim(tmp_path_factory):
    d = tmp_path_factory.mktemp("f05_shim")
    src, so = d / "cudart_device_shim.c", d / "cudart_device_shim.so"
    src.write_text(_F05_SHIM_C)
    r = subprocess.run(["gcc", "-shared", "-fPIC", "-O1", "-o", str(so), str(src), "-ldl"], capture_output=True,
                       text=True, timeout=_SUBPROC_TIMEOUT)
    assert r.returncode == 0, r.stderr
    return str(so)


def _run_f05(shim, mode):
    r = subprocess.run([sys.executable, "-c", _F05_SCRIPT, shim, mode], capture_output=True, text=True,
                       timeout=_SUBPROC_TIMEOUT, env=_subproc_env(LD_PRELOAD=shim))
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert r.returncode == 0 and lines, f"rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}"
    return json.loads(lines[-1][len("RESULT "):])


def test_fix1_f05_drop_while_other_device_current_drains_storage_device_and_restores(f05_shim):
    res = _run_f05(f05_shim, "ok")
    # The destructor must switch to the Storage's device 0, sync ITS legacy stream, switch back to device 1.
    assert res["log"] == ["set0", "sync@0", "set1"], res
    assert res["current"] == 1, res                           # the caller's (simulated) device is restored
    assert res["deleter_calls"] == 1, res                     # then the producer gets its memory back, once
    assert res["stats"]["leaked_drain_failure"] == 0, res


def test_fix1_f05_failed_switch_leaks_instead_of_releasing(f05_shim):
    res = _run_f05(f05_shim, "fail")
    assert res["log"] == ["set0!"], res                       # the switch was attempted and failed; no sync elsewhere
    assert res["current"] == 1, res
    assert res["deleter_calls"] == 0, res                     # no drain proof: leaked, never released
    assert res["stats"]["leaked_drain_failure"] == 1, res


# ------------------------------------------------------------------------------------------------ F06
# Prior art: RAII / unique_ptr hand-off (C++11). The bad_alloc path itself is not injectable from Python; it is
# verified by reasoning (the context is a unique_ptr until PyCapsule_New succeeds). This test pins the ownership
# hand-off on every reachable path: unconsumed capsule collected, consumed capsule released by its consumer.
def test_fix1_f06_export_context_owned_exactly_once(torch):
    owner = torch.arange(8, dtype=torch.float32, device="cuda")
    prod = _Producer(owner, (8,), (1,))
    u = tc.from_dlpack(prod.capsule())               # the imported Storage calls prod's deleter at its last release
    for k in range(50):
        cap = u.__dlpack__(max_version=(1, 3)) if k % 2 else u.__dlpack__()
        del cap                                      # unconsumed: the capsule destructor frees the context once
    gc.collect()
    assert prod.deleter_calls == 0
    caps = [u.__dlpack__(), u.__dlpack__(max_version=(1, 0))]
    xs = [torch.from_dlpack(c) for c in caps]        # consumed: torch calls each context's deleter later
    del u
    gc.collect()
    assert prod.deleter_calls == 0                   # two contexts still hold the Storage
    del xs, caps
    gc.collect()
    assert prod.deleter_calls == 1 and prod.owner is None



# =================================================================================================== P1T-FIX2
# Regression tests for the P1T-REV2 round-2 findings (order P1T_FIX2_ROBUSTNESS.md). Crash-prone paths run in
# subprocesses (the reviewer's host_checks.py / core_shutdown.py pattern).

# ------------------------------------------------------------------------------------------------ N2
# The census of the order: a 0 in the first, middle or last position next to extents up to 2^62 and 2^63 - 1. Prior
# art: NumPy's array-constructor rule (itemsize times every nonzero extent must fit; numpy 2.4.2 refuses
# np.empty((0, 2**62, 4)) and np.empty((0, 2**61), float32)); checked builtins (GCC 5, 2015). Ours: the census.
_N2_REFUSE = [
    ((0, 1 << 62, 4), "float32", "element count"), ((1 << 62, 0, 4), "float32", "element count"),
    ((1 << 62, 4, 0), "float32", "element count"),
    ((0, (1 << 63) - 1, 2), "float32", "element count"), (((1 << 63) - 1, 2, 0), "float32", "element count"),
    ((2, 0, (1 << 63) - 1), "float32", "element count"),
    ((0, (1 << 63) - 1), "float32", "byte size"), (((1 << 63) - 1, 0), "float16", "byte size"),
    ((0, 1 << 62), "float32", "byte size"), ((0, 1 << 61), "float32", "byte size"), ((1 << 62, 0, 1), "float16", "byte size"),
]
_N2_IDS = ["0x2^62x4", "2^62x0x4", "2^62x4x0", "0xMAXx2", "MAXx2x0", "2x0xMAX", "0xMAX_f32", "MAXx0_f16",
           "0x2^62_f32", "0x2^61_f32", "2^62x0x1_f16"]
_N2_ACCEPT = [((0, 1 << 60), "float32"), ((0, (1 << 63) - 1), "uint8"), ((1 << 30, 0, 1 << 30), "float32"),
              ((0, 1 << 61), "float16")]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("shape,dtype,frag", _N2_REFUSE, ids=_N2_IDS)
def test_fix2_n2_empty_refuses_zero_extent_overflow(device, shape, dtype, frag):
    with pytest.raises(ValueError, match=frag):
        tc.empty(shape, dtype=dtype, device=device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("shape,dtype", _N2_ACCEPT, ids=["0x2^60_f32", "0xMAX_u8", "2^30x0x2^30", "0x2^61_f16"])
def test_fix2_n2_empty_zero_extent_within_the_rule_is_empty(device, shape, dtype):
    t = tc.empty(shape, dtype=dtype, device=device)
    assert t.shape == shape


@pytest.mark.parametrize("shape,dtype,frag", _N2_REFUSE, ids=_N2_IDS)
def test_fix2_n2_import_refuses_zero_extent_overflow(torch, shape, dtype, frag):
    owner = torch.zeros(4, device="cuda")
    prod = _Producer(owner, shape, None, bits=32 if dtype == "float32" else 16)
    cap = prod.capsule()
    with pytest.raises(ValueError, match=frag):
        tc.from_dlpack(cap)
    assert "used_" not in repr(cap) and prod.deleter_calls == 0             # refused before consumption


def test_fix2_n2_import_zero_extent_within_the_rule_is_empty(torch):
    owner = torch.zeros(4, device="cuda")
    prod = _Producer(owner, (0, 1 << 60), None)
    u = tc.from_dlpack(prod.capsule())
    assert u.shape == (0, 1 << 60)
    del u
    gc.collect()
    assert prod.deleter_calls == 1


@pytest.mark.parametrize("shape,dtype,frag", _N2_REFUSE, ids=_N2_IDS)
def test_fix2_n2_export_refuses_zero_extent_overflow(shape, dtype, frag):
    # reshape of an empty tensor is the reachable way to hold such a shape (numel 0 == 0); export must not form its
    # strides. float32 everywhere (empty's default): the byte-size cases overflow for 4-byte elements too.
    t = tc.empty((0,), device="cuda").reshape(list(shape))
    assert t.shape == shape
    for kw in ({}, {"max_version": (1, 3)}):
        with pytest.raises(ValueError, match="element count|byte size"):
            t.__dlpack__(**kw)


def test_fix2_n2_export_zero_extent_within_the_rule(torch):
    t = tc.empty((0,), device="cuda").reshape([0, 1 << 60])
    x = torch.from_dlpack(t.__dlpack__(max_version=(1, 3)))
    assert tuple(x.shape) == (0, 1 << 60)


# ------------------------------------------------------------------------------------------------ N4
# A native producer deleter that throws (out of the DLPack C contract on purpose). Prior art: C++ noexcept /
# std::terminate semantics; the round-2 reviewer's throw_deleter probe (cpu_native.cpp). Ours: the core import path.
_FIX2_THROWER_SRC = r"""
#include <atomic>
#include <stdexcept>
static std::atomic<long> g_calls{0};
extern "C" void fix2_throwing_deleter(void*) {
  g_calls.fetch_add(1);
  throw std::runtime_error("fix2 test producer deleter threw");
}
extern "C" long fix2_throwing_deleter_calls() { return g_calls.load(); }
"""


@pytest.fixture(scope="module")
def fix2_thrower(tmp_path_factory):
    d = tmp_path_factory.mktemp("fix2_thrower")
    src, so = d / "fix2_thrower.cpp", d / "fix2_thrower.so"
    src.write_text(_FIX2_THROWER_SRC)
    r = subprocess.run(["g++", "-shared", "-fPIC", "-O1", "-std=c++17", "-o", str(so), str(src)], capture_output=True,
                       text=True, timeout=_SUBPROC_TIMEOUT)
    assert r.returncode == 0, r.stderr
    return str(so)


_FIX2_PTHREAD = r"""
libc = ctypes.CDLL(None)
pthread_create = libc.pthread_create
pthread_create.argtypes = [ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
pthread_create.restype = ctypes.c_int
pthread_join = libc.pthread_join
pthread_join.argtypes = [ctypes.c_ulong, ctypes.c_void_p]
pthread_join.restype = ctypes.c_int
GetPointer = ctypes.pythonapi['PyCapsule_GetPointer']
GetPointer.restype, GetPointer.argtypes = ctypes.c_void_p, [ctypes.py_object, ctypes.c_char_p]
SetName = ctypes.pythonapi['PyCapsule_SetName']
SetName.restype, SetName.argtypes = ctypes.c_int, [ctypes.py_object, ctypes.c_char_p]
KEEP = []

def held_export(prod):
    # An imported Storage (prod's manager) whose last reference is a TensorCUDA export manager owned by this script.
    u = tc.from_dlpack(prod.capsule())
    cap = u.__dlpack__(max_version=(1, 3))
    del u
    gc.collect()
    mptr = GetPointer(cap, NAME)
    assert SetName(cap, b"used_dltensor_versioned") == 0
    KEEP.append(cap)
    return mptr, ctypes.cast(mptr + 16, ctypes.POINTER(ctypes.c_void_p)).contents.value

def start_foreign(fp, m):
    tid = ctypes.c_ulong()
    assert pthread_create(ctypes.byref(tid), None, fp, m) == 0
    return tid.value

def join(tid):
    pthread_join(tid, None)                      # ctypes releases the GIL for the call

def new_producer(n=16):
    owner = tc.empty(n)
    owner.fill_(3.0)
    tc.synchronize()
    return Producer(owner, n)
"""

_FIX2_N4_SCRIPT = _SUBPROC_PRODUCER + _FIX2_PTHREAD + r"""
lib = ctypes.CDLL(sys.argv[1])
calls = lib.fix2_throwing_deleter_calls
calls.restype = ctypes.c_long
mode = sys.argv[2]
prod = new_producer()
prod.m.deleter = DELETER(ctypes.cast(lib.fix2_throwing_deleter, ctypes.c_void_p).value)   # the native thrower
if mode == "main":
    u = tc.from_dlpack(prod.capsule())
    del u                                        # Storage destructor on this thread -> deleter throws
    gc.collect()
else:
    mptr, fp = held_export(prod)
    join(start_foreign(fp, mptr))                # release on a foreign pthread -> deleter throws there
print("RESULT " + json.dumps({"calls": int(calls()), "stats": dict(tc._C._dlpack_release_stats())}), flush=True)
"""


@pytest.mark.parametrize("mode", ["main", "foreign"])
def test_fix2_n4_throwing_producer_deleter_is_contained(fix2_thrower, mode):
    r = subprocess.run([sys.executable, "-c", _FIX2_N4_SCRIPT, fix2_thrower, mode], capture_output=True, text=True,
                       timeout=_SUBPROC_TIMEOUT, env=_subproc_env())
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert r.returncode == 0 and lines, f"rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}"
    res = json.loads(lines[-1][len("RESULT "):])
    assert res["calls"] == 1, res                                # called exactly once
    assert res["stats"]["deleter_exceptions"] == 1 and res["stats"]["released"] == 0, res
    assert "terminate called" not in r.stderr, r.stderr


# ------------------------------------------------------------------------------------------------ N5
# Exit-time coordination (bindings.cpp ImportReleaseLifecycle + tensor_cuda._dlpack_shutdown_hook). Each scenario is a
# fresh process. atexit runs callbacks last-registered-first, so `_after_hook` (registered BEFORE tensor_cuda is
# imported) runs after tensor_cuda's hook, and `_start_in_flight` (registered after the import) runs before it.
# Staging "in flight": `_start_in_flight` starts a foreign pthread release of a real imported Storage whose producer
# deleter (a ctypes callback, so it needs the GIL) sleeps; it returns only once that deleter has STARTED, so the
# release is admitted and unfinished when the hook closes admission. (A thread merely blocked in PyGILState_Ensure
# cannot be held there: the eval loop hands it the GIL at the next bytecode boundary, before the hook's C call.)
# Without the in-flight wait (mutant (e); d116ac0 has no hook at all) the deleter is still sleeping when CPython marks
# the runtime finalizing; it wakes during the Finalizer's sleep in __del__ (module teardown) and must re-take the GIL:
# CPython 3.12 take_gil -> PyThread_exit_thread -> a forced unwind through the noexcept Storage destructor -> abort.
# Prior art: the reviewer's core_shutdown.py + release_worker.cpp scenario (a real imported Storage released on a
# foreign thread at shutdown); CPython 3.12.3 Py_FinalizeEx ordering (atexit before finalizing). Ours: the staging,
# the nested and bounded variants, the torch-free form.
# P1T-FIX3 (round-3 review V3-01): FIX2's "bounded" mode ended the process with os._exit(0) while its 2 s deleter was
# still running, so it never exercised ordinary finalization, and it pinned the defect (the hook returned before the
# deleter finished). It is replaced by "overrun" (a deleter of argv[2] seconds across the exit wait, ordinary exit, no
# os._exit anywhere in this script), "idle" (nothing in flight at exit) and "self" (the exit hook called from inside the
# main thread's own admitted release). The reviewer's probe artifacts/p1t_rev3/probes/core_exit_overrun.py is the
# same staging with the default 5 s knob and a 5.5 s deleter.
_FIX2_N5_SCRIPT = r"""
import atexit, ctypes, gc, json, os, sys, time
RESULTS = {}
PRODS = []
def _after_hook():
    RESULTS["after_hook"] = {"stats": dict(tc._C._dlpack_release_stats()), "deleter_calls": [p.calls for p in PRODS],
                             "deleter_done": [p.done for p in PRODS], "finalizing": sys.is_finalizing()}
    if MODE == "after":                         # a foreign release that starts after the hook closed admission
        join(start_foreign(FP_B, M_B))
        RESULTS["after_release"] = {"stats": dict(tc._C._dlpack_release_stats()),
                                    "deleter_calls": [p.calls for p in PRODS]}
    print("RESULT " + json.dumps(RESULTS), flush=True)
atexit.register(_after_hook)
""" + _SUBPROC_PRODUCER + _FIX2_PTHREAD + r"""
MODE = sys.argv[1]
SLEEP = float(sys.argv[2]) if MODE == "overrun" else {"during": 1.0, "nested": 0.0}.get(MODE, 0.0)
RESULTS["knob_s"] = tc._DLPACK_SHUTDOWN_WAIT_S  # the default knob, unchanged, read by the hook at exit
class SlowProducer(Producer):
    def _delete(self, _p):
        self.calls += 1
        self.started = True
        if self.on_delete:
            self.on_delete()
        time.sleep(self.sleep_s)                # GIL released while sleeping; re-taken to return
        self.done = True
def _new(sleep_s):
    p = SlowProducer(tc.empty(16), 16)
    p.started = p.done = False
    p.on_delete, p.sleep_s = None, sleep_s
    return p
prod_a, prod_b = _new(SLEEP), _new(0.0)
PRODS.extend([prod_a, prod_b])
M_A, FP_A = held_export(prod_a)
M_B, FP_B = held_export(prod_b)
def _wait_for(cond, what):
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > 10:
            raise RuntimeError("timed out waiting for " + what)
        time.sleep(0.005)
def _nested():                                  # runs in A's deleter, i.e. while the hook waits for A
    _wait_for(lambda: tc._C._dlpack_release_stats()["closing"], "the hook to close admission")
    RESULTS["nested"] = {"before": dict(tc._C._dlpack_release_stats())}
    join(start_foreign(FP_B, M_B))
    RESULTS["nested"]["after"] = dict(tc._C._dlpack_release_stats())
if MODE == "nested":
    prod_a.on_delete = _nested
def _start_in_flight():
    start_foreign(FP_A, M_A)
    _wait_for(lambda: prod_a.started, "the in-flight deleter to start")
    RESULTS["before_hook"] = {"stats": dict(tc._C._dlpack_release_stats())}
if MODE in ("during", "nested", "overrun"):
    atexit.register(_start_in_flight)
def _self_hook_inside():                        # runs inside A's deleter, on the MAIN thread: A is its own release
    RESULTS["self"] = {"inside_before": dict(tc._C._dlpack_release_stats())}
    t0 = time.monotonic()
    tc._dlpack_shutdown_hook()                  # the exit hook, called from inside this thread's admitted release
    RESULTS["self"]["hook_s"] = time.monotonic() - t0
    RESULTS["self"]["inside_after"] = dict(tc._C._dlpack_release_stats())
if MODE == "self":
    prod_a.on_delete = _self_hook_inside
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(FP_A)(M_A)   # release A on the main thread (not a foreign pthread)
    RESULTS["self_after_release"] = {"stats": dict(tc._C._dlpack_release_stats()), "deleter_done": prod_a.done}
class Finalizer:                                # drops the GIL during finalization (the core_shutdown.py pattern)
    def __init__(self):
        self.sleep, self.isf = time.sleep, sys.is_finalizing
    def __del__(self):
        fin = self.isf()
        self.sleep(1.5)
        print("FINAL finalizing=%s" % fin, flush=True)
# Held by its own module, not by __main__: an unfinished deleter's frame keeps __main__'s globals alive, which would
# keep a __main__ global from ever being finalized (measured on d116ac0: no FINAL line, the deleter cut off at exit).
import types
_holder = types.ModuleType("_fix2_finalizer_holder")
_holder.keep = Finalizer()
sys.modules["_fix2_finalizer_holder"] = _holder
del _holder
print("ARMED", flush=True)
"""


def _run_n5(mode, *args):
    r = subprocess.run([sys.executable, "-c", _FIX2_N5_SCRIPT, mode, *args], capture_output=True, text=True,
                       timeout=_SUBPROC_TIMEOUT, env=_subproc_env())
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    res = json.loads(lines[-1][len("RESULT "):]) if lines else None
    return r, res


def test_fix2_n5_in_flight_foreign_release_completes_before_finalization():
    r, res = _run_n5("during")
    assert res is not None, f"rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}"
    before, after = res["before_hook"]["stats"], res["after_hook"]["stats"]
    assert before.get("in_flight") == 1, res                      # admitted and unfinished when the hook ran
    assert res["after_hook"]["deleter_done"][0] is True and res["after_hook"]["deleter_calls"][0] == 1, res
    assert after["in_flight"] == 0 and after["released"] == 1 and after["closing"] is True, res
    assert after["shutdown_in_flight_at_close"] == 1 and after["shutdown_timeouts"] == 0, res
    assert res["after_hook"]["finalizing"] is False, res      # the hook ran before CPython marked finalizing
    assert r.returncode == 0 and "FINAL finalizing=True" in r.stdout, f"rc={r.returncode}\nstderr={r.stderr}"
    assert "terminate called" not in r.stderr, r.stderr


def test_fix2_n5_release_starting_during_the_wait_leaks_without_gil():
    r, res = _run_n5("nested")
    assert res is not None and r.returncode == 0, f"rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}"
    nested = res["nested"]
    assert nested["before"]["closing"] is True and nested["before"]["in_flight"] == 1, res   # during the wait for A
    assert nested["after"]["leaked_closing"] == 1, res            # B leaked: admission closed, no GIL taken
    after = res["after_hook"]
    assert after["deleter_calls"] == [1, 0], res                  # A released once, B never
    assert after["stats"]["released"] == 1 and after["stats"]["in_flight"] == 0, res


def test_fix2_n5_release_after_the_exit_hook_leaks_without_gil():
    r, res = _run_n5("after")
    assert res is not None and r.returncode == 0, f"rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}"
    rel = res["after_release"]
    assert rel["stats"]["leaked_closing"] == 1 and rel["stats"]["released"] == 0, res
    assert rel["deleter_calls"] == [0, 0], res
    assert "FINAL finalizing=True" in r.stdout and "terminate called" not in r.stderr, r.stderr


# ------------------------------------------------------------------------------------------------ V3-01 (P1T-FIX3)
# The exit hook waits, with no bound, until every admitted release has finished; after _DLPACK_SHUTDOWN_WAIT_S (the
# default 5 s, unchanged here) it writes a stderr line, then another each time the time waited doubles (5 s, 10 s,
# 20 s, ...). Ordinary interpreter exit (no os._exit): rc 0, the deleter completed before the hook returned, then CPython
# finalized (the Finalizer's FINAL line). Registered mutant (a) = FIX2's bounded return restored: the 5.5 s and 12 s
# cases then abort (rc -6, "FATAL: exception not rethrown") when the deleter re-takes the GIL during finalization.
# Prior art: the reviewer's lifecycle_timeout.py / core_exit_overrun.py (finite overrun with normal finalization);
# CPython's threading._shutdown (an unbounded join before finalizing). Ours: the deleter-length census, the diagnostic
# schedule checks and the self-release case.
_DIAG_RE = __import__("re").compile(r"DLPack release\(s\) admitted before exit still running after (\d+\.\d) s")


def _assert_ordinary_exit(r):
    assert r.returncode == 0, f"rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}"
    assert "FINAL finalizing=True" in r.stdout, r.stdout                      # CPython really finalized afterwards
    for bad in ("FATAL", "terminate called", "Aborted", "exception not rethrown"):
        assert bad not in r.stderr, r.stderr


def test_fix3_v301_script_has_no_immediate_exit_escape():
    assert "os._exit" not in _FIX2_N5_SCRIPT and "_exit(" not in _FIX2_N5_SCRIPT


@pytest.mark.parametrize("deleter_s", [0.1, 5.5, 12.0])
def test_fix3_v301_admitted_release_finishes_before_finalization_ordinary_exit(deleter_s):
    r, res = _run_n5("overrun", repr(deleter_s))
    assert res is not None, f"rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}"
    assert res["knob_s"] == 5.0, res                                            # the default knob, not lowered
    assert res["before_hook"]["stats"]["in_flight"] == 1, res                   # admitted and unfinished at the hook
    ah = res["after_hook"]
    after = ah["stats"]
    assert ah["deleter_done"][0] is True and ah["deleter_calls"] == [1, 0], res  # completed, exactly once
    assert after["released"] == 1 and after["in_flight"] == 0 and after["closing"] is True, res
    assert after["shutdown_still_in_flight"] == 0 and after["shutdown_in_flight_at_close"] == 1, res
    assert ah["finalizing"] is False, res                                       # the hook ran before finalizing
    assert after["shutdown_wait_us"] >= (deleter_s - 1.0) * 1e6, res            # it waited for the deleter
    lines = [float(x) for x in _DIAG_RE.findall(r.stderr)]
    if deleter_s > res["knob_s"]:
        want = [5.0] if deleter_s < 10.0 else [5.0, 10.0]                       # doubling schedule: 5 s, 10 s, 20 s
        assert len(lines) == len(want) and all(abs(a - b) < 0.5 for a, b in zip(lines, want)), r.stderr
        assert after["shutdown_timeouts"] == 1 and after["shutdown_diagnostics"] == len(want), res
        assert "finished after" in r.stderr and "interpreter exit continues" in r.stderr, r.stderr
    else:
        assert lines == [] and after["shutdown_timeouts"] == 0 and after["shutdown_diagnostics"] == 0, (res, r.stderr)
    _assert_ordinary_exit(r)


def test_fix3_v301_idle_exit_does_not_wait():
    r, res = _run_n5("idle")
    assert res is not None, f"rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}"
    after = res["after_hook"]["stats"]
    assert after["shutdown_calls"] == 1 and after["shutdown_in_flight_at_close"] == 0, res
    assert after["shutdown_wait_us"] < 500_000 and after["shutdown_diagnostics"] == 0, res
    assert res["after_hook"]["deleter_calls"] == [0, 0], res                    # never admitted, never released
    assert _DIAG_RE.search(r.stderr) is None, r.stderr
    _assert_ordinary_exit(r)


def test_fix3_v301_hook_inside_own_release_does_not_wait_for_itself():
    # A deleter running in the main thread's own admitted release calls the exit hook: the unbounded wait must not
    # wait for that release (it cannot finish while its own thread waits), and must not return while OTHER threads'
    # releases run (covered above).
    r, res = _run_n5("self")
    assert res is not None, f"rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}"
    inside = res["self"]
    assert inside["inside_before"]["in_flight"] == 1 and inside["inside_before"]["closing"] is False, res
    assert inside["hook_s"] < 1.0, res                                          # no self-wait (FIX2: 5 s; else hang)
    ia = inside["inside_after"]
    assert ia["closing"] is True and ia["in_flight"] == 1 and ia["shutdown_still_in_flight"] == 0, res  # own excluded
    assert ia["shutdown_diagnostics"] == 0 and ia["shutdown_timeouts"] == 0, res
    rel = res["self_after_release"]
    assert rel["deleter_done"] is True and rel["stats"]["released"] == 1 and rel["stats"]["in_flight"] == 0, res
    assert res["after_hook"]["stats"]["shutdown_calls"] == 2, res               # and the real exit hook ran again
    _assert_ordinary_exit(r)


# ================================================================================================== P1T-FIX4
# Round-4 review findings V4-01 (export fence on the wrong device), V4-03 (import ignored a failed cudaGetDevice) and
# the device / CUDA-error sweep of the P1-T core surface (fill_, empty, the external Storage drain). V4-04 is in
# tc_signal/tests/test_fix4_findings.py.
#
# The host has ONE GPU, so "another device is current" is simulated, as FIX1 did for F05: an LD_PRELOAD shim interposes
# the libcudart.so.12 calls the P1-T core path makes. In fake mode it reports a simulated current device, logs each call
# with the simulated device it would run on, and keeps the REAL device at 0. It can also fail one named call (or the
# Nth cudaSetDevice(d)) with a GENUINE runtime error (it calls the real cudaSetDevice(-1), so the thread's last-error
# slot really holds cudaErrorInvalidDevice) and can override the device cudaPointerGetAttributes reports.
# Prior art: LD_PRELOAD symbol interposition (glibc ld.so; the classic malloc/syscall interposers) and the FIX1 F05 shim
# above. Ours: this shim's call log, genuine-slot failure injection and attribute override.
_FIX4_SHIM_C = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>
typedef struct { unsigned x, y, z; } dim3_;
typedef struct { int type; int device; void* devicePointer; void* hostPointer; } attrs_;
static int fake = -1;                 /* -1: pass-through; else the simulated current device */
static int fail_set_to = -1, fail_set_nth = 0, set_seen = 0;
static char fail_name[64];
static int attr_device = -100;        /* -100: report the real device */
static char logbuf[8192];
static void note(const char* s) {
  size_t n = strlen(logbuf);
  if (n + strlen(s) + 2 < sizeof logbuf) { strcat(logbuf, s); strcat(logbuf, " "); }
}
static void* real_sym(const char* name) {
  void* h = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_NOLOAD);
  if (!h) h = dlopen("libcudart.so.12", RTLD_LAZY);
  return h ? dlsym(h, name) : NULL;
}
#define REAL(R, n, A) static R (*real_##n) A; if (!real_##n) real_##n = (R (*) A)real_sym(#n)
/* a GENUINE runtime failure: the real cudaSetDevice(-1) puts cudaErrorInvalidDevice (101) in the slot */
static int genuine_failure(void) { REAL(int, cudaSetDevice, (int)); return real_cudaSetDevice(-1); }
static int take_fail(const char* n) { if (fail_name[0] && strcmp(fail_name, n) == 0) { fail_name[0] = 0; return 1; } return 0; }
static void lognote(const char* what, int failed) {
  if (fake < 0) return;
  char b[48]; snprintf(b, sizeof b, "%s@%d%s", what, fake, failed ? "!" : ""); note(b);
}
int cudaGetDevice(int* d) { REAL(int, cudaGetDevice, (int*)); if (fake >= 0) { *d = fake; return 0; } return real_cudaGetDevice(d); }
int cudaSetDevice(int d) {
  REAL(int, cudaSetDevice, (int));
  if (fake < 0) return real_cudaSetDevice(d);
  int fail = 0;
  if (d == fail_set_to && ++set_seen == fail_set_nth) { fail = 1; fail_set_to = -1; }
  char b[32]; snprintf(b, sizeof b, "set%d%s", d, fail ? "!" : ""); note(b);
  if (fail) return genuine_failure();
  fake = d;
  return real_cudaSetDevice(0);
}
int cudaEventCreateWithFlags(void** e, unsigned f) {
  REAL(int, cudaEventCreateWithFlags, (void**, unsigned));
  if (take_fail("cudaEventCreateWithFlags")) { lognote("create", 1); return genuine_failure(); }
  lognote("create", 0); return real_cudaEventCreateWithFlags(e, f);
}
int cudaEventRecord(void* e, void* s) {
  REAL(int, cudaEventRecord, (void*, void*));
  if (take_fail("cudaEventRecord")) { lognote("record", 1); return genuine_failure(); }
  lognote("record", 0); return real_cudaEventRecord(e, s);
}
int cudaStreamWaitEvent(void* s, void* e, unsigned f) {
  REAL(int, cudaStreamWaitEvent, (void*, void*, unsigned));
  if (take_fail("cudaStreamWaitEvent")) { lognote("wait", 1); return genuine_failure(); }
  lognote("wait", 0); return real_cudaStreamWaitEvent(s, e, f);
}
int cudaEventDestroy(void* e) {
  REAL(int, cudaEventDestroy, (void*));
  if (take_fail("cudaEventDestroy")) { lognote("destroy", 1); real_cudaEventDestroy(e); return genuine_failure(); }
  lognote("destroy", 0); return real_cudaEventDestroy(e);
}
int cudaStreamSynchronize(void* s) { REAL(int, cudaStreamSynchronize, (void*)); lognote("sync", 0); return real_cudaStreamSynchronize(s); }
int cudaLaunchKernel(const void* fn, dim3_ g, dim3_ b, void** a, size_t sh, void* s) {
  REAL(int, cudaLaunchKernel, (const void*, dim3_, dim3_, void**, size_t, void*));
  if (take_fail("cudaLaunchKernel")) { lognote("launch", 1); return genuine_failure(); }   /* nothing launched */
  lognote("launch", 0); return real_cudaLaunchKernel(fn, g, b, a, sh, s);
}
int cudaPointerGetAttributes(attrs_* at, const void* p) {
  REAL(int, cudaPointerGetAttributes, (attrs_*, const void*));
  if (take_fail("cudaPointerGetAttributes")) { lognote("attr", 1); return genuine_failure(); }
  int r = real_cudaPointerGetAttributes(at, p);
  if (r == 0 && attr_device != -100) at->device = attr_device;
  return r;
}
void shim_fake(int d) { fake = d; }
void shim_fail_set(int d, int nth) { fail_set_to = d; fail_set_nth = nth; set_seen = 0; }
void shim_fail(const char* n) { strncpy(fail_name, n, sizeof fail_name - 1); }
void shim_attr_device(int d) { attr_device = d; }
int shim_current(void) { return fake; }
int shim_slot_peek(void) { REAL(int, cudaPeekAtLastError, (void)); return real_cudaPeekAtLastError(); }
const char* shim_log(void) { return logbuf; }
void shim_clear(void) { logbuf[0] = 0; }
"""

_FIX4_SCRIPT = _SUBPROC_PRODUCER + r"""
shim = ctypes.CDLL(sys.argv[1])                # already loaded through LD_PRELOAD: the same handle
shim.shim_log.restype = ctypes.c_char_p
shim.shim_fail.argtypes = [ctypes.c_char_p]
rt = ctypes.CDLL("libcudart.so.12")            # the same runtime instance as core (real, not interposed, functions)
mode = sys.argv[2]
N = 1 << 16
owner = tc.empty(N)
owner.fill_(1.0)
tc.synchronize()
prod = Producer(owner, N)
u = tc.from_dlpack(prod.capsule())             # imported on (real and labeled) device 0
consumer = ctypes.c_void_p()
assert rt.cudaStreamCreateWithFlags(ctypes.byref(consumer), 1) == 0      # a real non-blocking stream on device 0
res = {"mode": mode}

def run(fn):
    shim.shim_clear()
    try:
        out = fn()
        res["raised"] = None
        return out
    except Exception as e:  # noqa: BLE001
        res["raised"], res["msg"] = type(e).__name__, str(e)
    finally:
        res["log"] = shim.shim_log().decode().split()
        res["current"] = shim.shim_current()
        res["slot_after"] = shim.shim_slot_peek()

if mode.startswith("export"):
    stream = 2 if mode == "export_ptds" else consumer.value
    shim.shim_fake(0 if mode in ("export_same", "export_mismatch") else 1)
    if mode == "export_fail_create":
        shim.shim_fail(b"cudaEventCreateWithFlags")
    elif mode == "export_fail_wait":
        shim.shim_fail(b"cudaStreamWaitEvent")
    elif mode == "export_attr_fail":
        shim.shim_fail(b"cudaPointerGetAttributes")
    elif mode == "export_fail_record":
        shim.shim_fail(b"cudaEventRecord")
    elif mode == "export_fail_destroy":
        shim.shim_fail(b"cudaEventDestroy")
    elif mode == "export_fail_switch":
        shim.shim_fail_set(0, 1)
    elif mode == "export_fail_restore":
        shim.shim_fail_set(1, 1)
    elif mode == "export_mismatch":
        shim.shim_attr_device(1)
    cap = run(lambda: u.__dlpack__(stream=stream, max_version=(1, 3)))
    res["capsule"] = cap is not None
    shim.shim_fake(-1)
    shim.shim_attr_device(-100)
    del cap
elif mode.startswith("fill_"):
    shim.shim_fake(0 if mode == "fill_mismatch" else 1)
    if mode == "fill_mismatch":
        shim.shim_attr_device(1)
    elif mode == "fill_fail_switch":
        shim.shim_fail_set(0, 1)                   # the guard's switch to the tensor's device fails
    elif mode == "fill_fail_restore":
        shim.shim_fail_set(1, 1)                   # the success-path restore fails (the write is queued)
    elif mode == "fill_fail_launch_and_restore":
        shim.shim_fail(b"cudaLaunchKernel")        # the launch is refused, then the unwinding restore fails
        shim.shim_fail_set(1, 1)
    rev0 = u.revision
    run(lambda: u.fill_(2.0))
    res["revision_delta"] = u.revision - rev0
    shim.shim_fake(-1)
    shim.shim_attr_device(-100)
    tc.synchronize()
    res["values"] = sorted(set(u.numpy().tolist()))
elif mode == "drain_restore_fail":
    u.fill_(2.0)                                   # engine work queued on the imported memory
    shim.shim_fake(1)
    shim.shim_fail_set(1, 1)                       # the destructor's restore of device 1 fails
    shim.shim_clear()
    del u
    gc.collect()
    res["log"] = shim.shim_log().decode().split()
    res["current"] = shim.shim_current()
    res["slot_after"] = shim.shim_slot_peek()
    shim.shim_fake(-1)
res["deleter_calls"] = prod.calls
res["stats"] = dict(tc._C._dlpack_release_stats())
print("RESULT " + json.dumps(res), flush=True)
"""


@pytest.fixture(scope="module")
def fix4_shim(tmp_path_factory):
    d = tmp_path_factory.mktemp("fix4_shim")
    src, so = d / "cudart_fix4_shim.c", d / "cudart_fix4_shim.so"
    src.write_text(_FIX4_SHIM_C)
    r = subprocess.run(["gcc", "-shared", "-fPIC", "-O1", "-o", str(so), str(src), "-ldl"], capture_output=True,
                       text=True, timeout=_SUBPROC_TIMEOUT)
    assert r.returncode == 0, r.stderr
    return str(so)


def _run_fix4(shim, mode):
    r = subprocess.run([sys.executable, "-c", _FIX4_SCRIPT, shim, mode], capture_output=True, text=True,
                       timeout=_SUBPROC_TIMEOUT, env=_subproc_env(LD_PRELOAD=shim))
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert r.returncode == 0 and lines, f"rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}"
    return json.loads(lines[-1][len("RESULT "):])


_FENCE_ON_0 = ["create@0", "record@0", "wait@0", "destroy@0"]


@pytest.mark.parametrize("mode", ["export_switch", "export_ptds"])
def test_fix4_v401_export_fence_runs_on_the_tensor_device_and_restores(fix4_shim, mode):
    # V4-01: device 1 current, device-0 tensor, a consumer stream handle (or 2, the per-thread stream): the event must
    # be created and recorded on device 0, the wait issued there, the event destroyed there, and device 1 restored.
    # 62c8d9f / mutant (a): create@1 record@1 wait@1 destroy@1, no switch.
    res = _run_fix4(fix4_shim, mode)
    assert res["raised"] is None and res["capsule"], res
    assert res["log"] == ["set0"] + _FENCE_ON_0 + ["set1"], res
    assert res["current"] == 1 and res["slot_after"] == 0, res


def test_fix4_v401_export_fence_same_device_does_not_switch(fix4_shim):
    res = _run_fix4(fix4_shim, "export_same")
    assert res["raised"] is None and res["log"] == _FENCE_ON_0 and res["current"] == 0, res


@pytest.mark.parametrize("mode, logged, frag", [
    ("export_fail_record", ["set0", "create@0", "record@0!", "destroy@0", "set1"], "cudaEventRecord"),
    ("export_fail_destroy", ["set0", "create@0", "record@0", "wait@0", "destroy@0!", "set1"], "cudaEventDestroy"),
    ("export_fail_switch", ["set0!"], "cannot make the tensor's CUDA device 0 current"),
])
def test_fix4_v401_export_fence_failure_raises_with_the_device_restored(fix4_shim, mode, logged, frag):
    # The order: errors raised with the device restored. Each failure is a genuine runtime error (the slot really holds
    # cudaErrorInvalidDevice), so the raise must also consume it (slot_after == 0).
    res = _run_fix4(fix4_shim, mode)
    assert res["raised"] == "RuntimeError" and frag in res["msg"] and "cudaErrorInvalidDevice" in res["msg"], res
    assert res["log"] == logged, res
    assert res["current"] == 1, res                              # the caller's device, restored (or never left)
    assert res["slot_after"] == 0, res


@pytest.mark.parametrize("mode, logged, frag", [
    ("export_fail_create", ["set0", "create@0!", "set1"], "cudaEventCreateWithFlags"),
    ("export_fail_wait", ["set0", "create@0", "record@0", "wait@0!", "destroy@0", "set1"], "cudaStreamWaitEvent"),
])
def test_fix4_v401_export_fence_create_or_wait_failure_restores_and_raises(fix4_shim, mode, logged, frag):
    res = _run_fix4(fix4_shim, mode)
    assert res["raised"] == "RuntimeError" and frag in res["msg"] and "was restored" in res["msg"], res
    assert res["log"] == logged and res["current"] == 1 and res["slot_after"] == 0, res


def test_fix4_export_label_proof_query_failure_raises_and_consumes(fix4_shim):
    res = _run_fix4(fix4_shim, "export_attr_fail")
    assert res["raised"] == "RuntimeError" and "cudaPointerGetAttributes failed" in res["msg"], res
    assert res["log"] == ["attr@1!"] and res["slot_after"] == 0 and res["capsule"] is False, res   # nothing fenced


def test_fix4_v401_export_fence_failed_restore_is_raised_and_named(fix4_shim):
    res = _run_fix4(fix4_shim, "export_fail_restore")
    assert res["raised"] == "RuntimeError", res
    assert "restoring the caller's CUDA device 1 also failed" in res["msg"], res
    assert res["log"] == ["set0"] + _FENCE_ON_0 + ["set1!"] and res["current"] == 0, res   # honest: still device 0
    assert res["slot_after"] == 0, res


def test_fix4_export_refuses_a_label_that_is_not_the_allocation_device(fix4_shim):
    # The label is proven against the allocation before it is published or fenced (the engine labels its own
    # allocations cuda:0 while allocating on the current device). Mismatch: BufferError, nothing recorded.
    res = _run_fix4(fix4_shim, "export_mismatch")
    assert res["raised"] == "BufferError" and "labeled cuda:0 but its memory is on CUDA device 1" in res["msg"], res
    assert res["log"] == [] and res["capsule"] is False, res


def test_fix4_fill_launches_on_the_tensor_device_and_restores(fix4_shim):
    # 62c8d9f launched fill_ on the CURRENT device (device 1 here: a kernel writing device-0 memory from device 1).
    res = _run_fix4(fix4_shim, "fill_switch")
    assert res["raised"] is None and res["log"] == ["set0", "launch@0", "set1"] and res["current"] == 1, res
    assert res["values"] == [2.0], res


@pytest.mark.parametrize("mode", ["fill_fail_switch", "fill_fail_restore", "fill_fail_launch_and_restore"])
def test_fix4_fill_guard_failures_raise_count_and_consume(fix4_shim, mode):
    res = _run_fix4(fix4_shim, mode)
    assert res["raised"] == "RuntimeError" and res["slot_after"] == 0, res
    if mode == "fill_fail_switch":                               # nothing switched, nothing launched
        assert "fill_: cannot make the tensor's CUDA device 0 current" in res["msg"], res
        assert res["log"] == ["set0!"] and res["current"] == 1 and res["values"] == [1.0], res
        assert res["revision_delta"] == 0, res
    elif mode == "fill_fail_restore":                            # the write is queued and counted in the revision
        assert "fill_: restoring the caller's CUDA device 1 failed" in res["msg"], res
        assert res["log"] == ["set0", "launch@0", "set1!"] and res["current"] == 0, res
        assert res["values"] == [2.0] and res["revision_delta"] == 1, res
    else:                                                        # launch refused; the unwinding restore is counted
        assert "CUDA error at fill_" in res["msg"], res
        assert res["log"] == ["set0", "launch@0!", "set1!"] and res["values"] == [1.0], res
        assert res["stats"]["device_restore_failures"] == 1 and res["revision_delta"] == 0, res


def test_fix4_fill_refuses_a_label_that_is_not_the_allocation_device(fix4_shim):
    res = _run_fix4(fix4_shim, "fill_mismatch")
    assert res["raised"] == "RuntimeError" and "labeled cuda:0 but its memory is on CUDA device 1" in res["msg"], res
    assert res["log"] == [] and res["values"] == [1.0], res      # nothing launched: the old values remain


def test_fix4_storage_drain_failed_restore_is_counted_not_ignored(fix4_shim):
    # kernels.cu drain_device_legacy_stream ignored its restore's result in 62c8d9f. The drain itself succeeded, so the
    # memory is released (once); the failed restore is counted and its slot entry consumed.
    res = _run_fix4(fix4_shim, "drain_restore_fail")
    assert res["log"] == ["set0", "sync@0", "set1!"], res
    assert res["deleter_calls"] == 1 and res["stats"]["leaked_drain_failure"] == 0, res
    assert res["stats"]["device_restore_failures"] == 1 and res["slot_after"] == 0, res


# ---- V4-03: the import checks cudaGetDevice before any ownership transfer
_FIX4_IMPORT_NODEV = r"""
import ctypes, json, sys
import tensor_cuda as tc
class DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]
class DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]
class DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", DLDevice), ("ndim", ctypes.c_int32), ("dtype", DLDataType),
                ("shape", ctypes.POINTER(ctypes.c_int64)), ("strides", ctypes.POINTER(ctypes.c_int64)),
                ("byte_offset", ctypes.c_uint64)]
class Version(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]
class MTV(ctypes.Structure):
    pass
class MT(ctypes.Structure):
    pass
DEL_V = ctypes.CFUNCTYPE(None, ctypes.POINTER(MTV))
DEL_L = ctypes.CFUNCTYPE(None, ctypes.POINTER(MT))
MTV._fields_ = [("version", Version), ("manager_ctx", ctypes.c_void_p), ("deleter", DEL_V),
                ("flags", ctypes.c_uint64), ("dl_tensor", DLTensor)]
MT._fields_ = [("dl_tensor", DLTensor), ("manager_ctx", ctypes.c_void_p), ("deleter", DEL_L)]
New = ctypes.pythonapi['PyCapsule_New']
New.restype, New.argtypes = ctypes.py_object, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
GetName = ctypes.pythonapi['PyCapsule_GetName']
GetName.restype, GetName.argtypes = ctypes.c_char_p, [ctypes.py_object]
rt = ctypes.CDLL("libcudart.so.12")
d = ctypes.c_int(-1)
res = {"query": rt.cudaGetDevice(ctypes.byref(d))}
calls = []
shape = (ctypes.c_int64 * 1)(0)
cb_v = DEL_V(lambda p: calls.append("v"))
cb_l = DEL_L(lambda p: calls.append("l"))
mv = MTV(); mv.version.major, mv.version.minor = 1, 3; mv.deleter = cb_v
ml = MT(); ml.deleter = cb_l
for t in (mv.dl_tensor, ml.dl_tensor):        # zero-size fp32 CUDA device-0 descriptors, NULL data: nothing to read
    t.data, t.ndim, t.shape = None, 1, ctypes.cast(shape, ctypes.POINTER(ctypes.c_int64))
    t.device.device_type, t.device.device_id = 2, 0
    t.dtype.code, t.dtype.bits, t.dtype.lanes = 2, 32, 1
for kind, m, name in (("versioned", mv, b"dltensor_versioned"), ("legacy", ml, b"dltensor")):
    cap = New(ctypes.addressof(m), name, None)
    try:
        tc.from_dlpack(cap)
        res[kind] = {"raised": None}
    except Exception as e:  # noqa: BLE001
        res[kind] = {"raised": type(e).__name__, "msg": str(e)}
    res[kind]["name_after"] = GetName(cap).decode()
res["deleter_calls"] = calls
try:
    tc.empty(4)
    res["empty"] = {"raised": None}
except Exception as e:  # noqa: BLE001
    res["empty"] = {"raised": type(e).__name__, "msg": str(e)}
print("RESULT " + json.dumps(res), flush=True)
"""


def test_fix4_v403_import_refuses_a_failed_device_query_before_consuming():
    # CUDA_VISIBLE_DEVICES="" makes cudaGetDevice fail (cudaErrorNoDevice). 62c8d9f: cur stayed 0, the device-0
    # capsule was accepted and renamed used_* (mutant (b) restores that). Now: RuntimeError naming cudaGetDevice and the
    # error, the capsule NOT renamed, the producer deleter not called by the import (ownership stays with the capsule).
    r = subprocess.run([sys.executable, "-c", _FIX4_IMPORT_NODEV], capture_output=True, text=True,
                       timeout=_SUBPROC_TIMEOUT, env=_subproc_env(CUDA_VISIBLE_DEVICES=""))
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert r.returncode == 0 and lines, f"rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}"
    res = json.loads(lines[-1][len("RESULT "):])
    assert res["query"] == 100, res                               # precondition: no usable device
    for kind, name in (("versioned", "dltensor_versioned"), ("legacy", "dltensor")):
        k = res[kind]
        assert k["raised"] == "RuntimeError", res
        assert "cudaGetDevice" in k["msg"] and "cudaErrorNoDevice" in k["msg"], res
        assert k["name_after"] == name, res                       # not consumed
    assert res["deleter_calls"] == [], res
    # empty labels the current device: its failed query is a named RuntimeError too (62c8d9f: cudaMalloc's error)
    assert res["empty"]["raised"] == "RuntimeError" and "cudaGetDevice" in res["empty"]["msg"], res


_FIX4_IMPORT_STICKY = _SUBPROC_PRODUCER + r"""
import torch
N = 1 << 12
owner = tc.empty(N)
owner.fill_(1.0)
tc.synchronize()
prod = Producer(owner, N)
x = torch.zeros(4, device="cuda")
idx = torch.tensor([100], device="cuda")
try:
    x[idx]                                        # device-side assert: the CUDA context is now broken (sticky)
    torch.cuda.synchronize()
    res = {"staged": False}
except RuntimeError as e:
    res = {"staged": "device-side assert" in str(e)}
rt = ctypes.CDLL("libcudart.so.12")
res["query_legacy"] = rt.cudaStreamQuery(ctypes.c_void_p(1))  # proves the context error is visible to core's runtime
rt.cudaGetLastError()
cap = prod.capsule()
try:
    u = tc.from_dlpack(cap)
    res["import"] = "ok"
except Exception as e:  # noqa: BLE001
    res["import"], res["msg"] = type(e).__name__, str(e)
    u = None
GetName = ctypes.pythonapi['PyCapsule_GetName']
GetName.restype, GetName.argtypes = ctypes.c_char_p, [ctypes.py_object]
res["name_after"] = GetName(cap).decode()
del u
gc.collect()
res["deleter_calls"] = prod.calls
res["stats"] = dict(tc._C._dlpack_release_stats())
print("RESULT " + json.dumps(res), flush=True)
"""


def test_fix4_v403_import_with_a_sticky_context_error_takes_ownership_once_and_release_leaks():
    # The order's census: import with a sticky prior error. Measured (artifacts/p1t_fix4/cuda_semantics.txt):
    # cudaGetDevice still succeeds in a broken context, so the import's one CUDA query passes and it takes ownership
    # (once); the Storage's release then cannot drain the broken context and leaks deliberately (F05) instead of
    # releasing. Runs in its own process (the context stays broken until that process exits).
    r = subprocess.run([sys.executable, "-c", _FIX4_IMPORT_STICKY], capture_output=True, text=True,
                       timeout=_SUBPROC_TIMEOUT, env=_subproc_env())
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert r.returncode == 0 and lines, f"rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}"
    res = json.loads(lines[-1][len("RESULT "):])
    assert res["staged"] is True and res["query_legacy"] == 710, res   # cudaErrorAssert: the context is broken
    assert res["import"] == "ok" and res["name_after"] == "used_dltensor_versioned", res
    assert res["deleter_calls"] == 0 and res["stats"]["leaked_drain_failure"] == 1, res
    assert res["stats"]["released"] == 0, res


# ---- sweep: fill_ checks the error slot right before its launch; empty labels with the current device
_FIX4_STAGE = r"""
import ctypes
_rt = ctypes.CDLL("libcudart.so.12")
class _Dim3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint), ("y", ctypes.c_uint), ("z", ctypes.c_uint)]
_rt.cudaLaunchKernel.argtypes = [ctypes.c_void_p, _Dim3, _Dim3, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
def stage_refused_launch():
    # a synchronous launch refusal (NULL kernel): queues nothing, leaves cudaErrorInvalidDeviceFunction (98) in the slot
    _rt.cudaGetLastError()
    e = _rt.cudaLaunchKernel(None, _Dim3(1, 1, 1), _Dim3(1, 1, 1), None, 0, None)
    assert e != 0 and _rt.cudaPeekAtLastError() == e, (e, _rt.cudaPeekAtLastError())
    return e
def slot():
    return _rt.cudaPeekAtLastError()
"""


def test_fix4_fill_reports_a_pending_error_as_such_and_launches_nothing(torch):
    ns = {}
    exec(_FIX4_STAGE, ns)
    t = tc.empty(64)
    t.fill_(5.0)
    tc.synchronize()
    staged = ns["stage_refused_launch"]()
    try:
        with pytest.raises(RuntimeError, match="already pending in this thread before the launch") as ei:
            t.fill_(7.0)
        slot_after = ns["slot"]()
    finally:
        ns["_rt"].cudaGetLastError()                              # never leave a staged error to later tests
    assert "cudaErrorInvalidDeviceFunction" in str(ei.value)
    assert staged == 98 and slot_after == 0                       # taken from the slot (reported once)
    assert np.all(t.numpy() == 5.0)                               # nothing was launched
    t.fill_(7.0)                                                  # and the next call is clean
    assert np.all(t.numpy() == 7.0)


def test_fix4_empty_failed_allocation_is_reported_once(torch):
    # The engine's Storage constructor raises on a failed cudaMalloc without consuming the failed call's slot entry;
    # empty's raise reports it, so empty consumes it: no later, unrelated launch check reports it again.
    ns = {}
    exec(_FIX4_STAGE, ns)
    try:
        with pytest.raises(RuntimeError, match="cudaMalloc"):
            tc.empty(1 << 40)                                     # 4 TiB of float32: refused by the driver
        slot_after = ns["slot"]()
    finally:
        ns["_rt"].cudaGetLastError()
    assert slot_after == 0
    t = tc.empty(8)
    t.fill_(3.0)                                                  # the next launch check is clean
    assert np.all(t.numpy() == 3.0)


def test_fix4_empty_labels_the_current_device_and_refuses_another_index(torch):
    cur = torch.cuda.current_device()
    t = tc.empty(4, device=f"cuda:{cur}")
    assert t.device == f"cuda:{cur}"
    for bad in (f"cuda:{cur + 1}", "cuda:x", "cuda:"):
        with pytest.raises(ValueError):
            tc.empty(4, device=bad)
    assert tc.empty(4).device == f"cuda:{cur}"


# ================================================================================================== P1T-FIX5
# Round-5 review finding V5-04 (fill_ narrowed its block count) and the core rows of the class (d) sweep (CUDA errors
# consumed where nothing can raise must be recorded). V5-01..V5-03 are tc_signal findings: tc_signal/tests/
# test_fix5_findings.py.
#
# V5-04 needs extents of up to 2^40 + 1 fp32 elements without allocating them: as the reviewer's
# fill_grid_single_gpu.py does, a ctypes DLPack descriptor with the huge extent over a small real allocation is imported
# (synthetic metadata; the label proof passes because the pointer is real), and an LD_PRELOAD shim over libcudart.so.12
# captures fill_'s cudaLaunchKernel geometry. In "trap" mode the shim never runs the kernel: it fails the launch
# GENUINELY (the real cudaLaunchKernel with a NULL function: cudaErrorInvalidDeviceFunction in the slot), so the small
# buffer is never written past its end. In "pass" mode it records the geometry and launches for real (small extents).
# Pass rule (registered in docs/P1T_FIX5_LEDGER.md before the run): the largest accepted extent (2^31-1)*256 launches
# grid.x = 2^31-1, block 256 (covers it exactly); one more element and 2^40+1 are refused with ValueError before any
# launch and without a revision change; a zero-size fill_ launches nothing and keeps the revision. Registered mutant
# (d): the V5-04 refusal removed.
# Prior art: the reviewer's fill_grid_shim.c / fill_grid_single_gpu.py (LD_PRELOAD capture on synthetic metadata),
# taken; LD_PRELOAD interposition (glibc ld.so). Ours: the trap/pass modes and the boundary census.
_FIX5_FILL_SHIM_C = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
typedef struct { unsigned x, y, z; } dim3_;
static int mode = 0;                  /* 0: pass-through, unrecorded; 1: record + trap (genuine failure); 2: record + pass */
static int calls = 0;
static unsigned grid_x = 0, block_x = 0;
static void* real_sym(const char* name) {
  void* h = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_NOLOAD);
  if (!h) h = dlopen("libcudart.so.12", RTLD_LAZY);
  return h ? dlsym(h, name) : NULL;
}
int cudaLaunchKernel(const void* fn, dim3_ g, dim3_ b, void** a, size_t sh, void* s) {
  static int (*real)(const void*, dim3_, dim3_, void**, size_t, void*);
  if (!real) real = (int (*)(const void*, dim3_, dim3_, void**, size_t, void*))real_sym("cudaLaunchKernel");
  if (mode == 0) return real(fn, g, b, a, sh, s);
  ++calls; grid_x = g.x; block_x = b.x;
  if (mode == 1) { mode = 0; return real(NULL, g, b, a, sh, s); }   /* trapped: nothing runs, the slot holds 98 */
  mode = 0;
  return real(fn, g, b, a, sh, s);
}
void fill_shim_arm(int m) { mode = m; calls = 0; grid_x = 0; block_x = 0; }
int fill_shim_calls(void) { return calls; }
unsigned fill_shim_grid(void) { return grid_x; }
unsigned fill_shim_block(void) { return block_x; }
int fill_shim_slot(void) { static int (*peek)(void); if (!peek) peek = (int (*)(void))real_sym("cudaPeekAtLastError"); return peek(); }
"""

_FIX5_FILL_SCRIPT = _SUBPROC_PRODUCER + r"""
shim = ctypes.CDLL(sys.argv[1])                # already loaded through LD_PRELOAD: the same handle
shim.fill_shim_grid.restype = ctypes.c_uint
shim.fill_shim_block.restype = ctypes.c_uint
case, n = sys.argv[2], int(sys.argv[3])
owner = tc.empty(max(256, min(n, 4096)))       # the REAL allocation behind the descriptor (small)
tc.synchronize()
prod = Producer(owner, n)                      # synthetic metadata: n elements over `owner`'s pointer
u = tc.from_dlpack(prod.capsule())
res = {"case": case, "n": n, "numel": n}
rev0 = u.revision
shim.fill_shim_arm(2 if case == "small" else 1)
try:
    u.fill_(7.0)
    res["raised"] = None
except Exception as e:  # noqa: BLE001
    res["raised"], res["msg"] = type(e).__name__, str(e)
res["calls"], res["grid"], res["block"] = shim.fill_shim_calls(), shim.fill_shim_grid(), shim.fill_shim_block()
res["revision_delta"] = u.revision - rev0
res["slot_after"] = shim.fill_shim_slot()
if case == "small":
    tc.synchronize()
    res["values"] = sorted(set(u.numpy().tolist()))
rt = ctypes.CDLL("libcudart.so.12")
v = ctypes.c_int(-1)
res["max_grid_x"] = v.value if rt.cudaDeviceGetAttribute(ctypes.byref(v), 5, 0) == 0 else None  # cudaDevAttrMaxGridDimX
del u
gc.collect()
res["deleter_calls"] = prod.calls
print("RESULT " + json.dumps(res), flush=True)
"""

_FIX5_MAX_GRID_X = (1 << 31) - 1           # CUDA 12.6 grid x-dimension limit (kernels.cu kFillMaxGridX)
_FIX5_FILL_CASES = {
    "largest": _FIX5_MAX_GRID_X * 256,     # 549755813632: the largest accepted extent
    "one_more": _FIX5_MAX_GRID_X * 256 + 1,
    "huge": (1 << 40) + 1,                 # the reviewer's 2^40 + 1
    "zero": 0,
    "small": 1000,
}


@pytest.fixture(scope="module")
def fix5_fill_shim(tmp_path_factory):
    d = tmp_path_factory.mktemp("fix5_fill_shim")
    src, so = d / "fix5_fill_shim.c", d / "fix5_fill_shim.so"
    src.write_text(_FIX5_FILL_SHIM_C)
    r = subprocess.run(["gcc", "-shared", "-fPIC", "-O1", "-o", str(so), str(src), "-ldl"], capture_output=True,
                       text=True, timeout=_SUBPROC_TIMEOUT)
    assert r.returncode == 0, r.stderr
    return str(so)


def _run_fix5_fill(shim, case):
    r = subprocess.run([sys.executable, "-c", _FIX5_FILL_SCRIPT, shim, case, str(_FIX5_FILL_CASES[case])],
                       capture_output=True, text=True, timeout=_SUBPROC_TIMEOUT, env=_subproc_env(LD_PRELOAD=shim))
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert r.returncode == 0 and lines, f"rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}"
    return json.loads(lines[-1][len("RESULT "):])


def test_fix5_v504_fill_largest_accepted_extent_launches_a_covering_grid(fix5_fill_shim):
    res = _run_fix5_fill(fix5_fill_shim, "largest")
    assert res["calls"] == 1, res                                            # launched (trapped by the shim)
    assert res["grid"] == _FIX5_MAX_GRID_X and res["block"] == 256, res      # 1475fae: nblk narrowed to int
    assert res["grid"] * res["block"] >= res["n"], res                       # covers every element (exactly)
    assert res["raised"] == "RuntimeError" and "invalid device function" in res["msg"], res   # the trapped launch
    assert res["revision_delta"] == 0 and res["slot_after"] == 0, res        # nothing ran: no bump; slot consumed
    assert res["deleter_calls"] == 1, res


@pytest.mark.parametrize("case", ["one_more", "huge"])
def test_fix5_v504_fill_beyond_the_grid_limit_is_refused_before_any_launch(fix5_fill_shim, case):
    # 1475fae (and mutant (d)): "huge" launched grid 1 (2^32 + 1 narrowed), wrote 256 elements and bumped the revision.
    res = _run_fix5_fill(fix5_fill_shim, case)
    assert res["raised"] == "ValueError", res
    assert "grid x-dimension limit of 2147483647" in res["msg"] and "Refused before any launch" in res["msg"], res
    assert res["calls"] == 0 and res["revision_delta"] == 0 and res["slot_after"] == 0, res
    assert res["deleter_calls"] == 1, res


def test_fix5_v504_fill_zero_size_launches_nothing(fix5_fill_shim):
    res = _run_fix5_fill(fix5_fill_shim, "zero")
    assert res["raised"] is None and res["calls"] == 0 and res["revision_delta"] == 0, res


def test_fix5_v504_fill_small_extent_unchanged_and_the_cited_limit_matches_the_device(fix5_fill_shim):
    res = _run_fix5_fill(fix5_fill_shim, "small")
    assert res["raised"] is None and res["calls"] == 1, res
    assert res["grid"] == 4 and res["block"] == 256 and res["values"] == [7.0], res   # ceil(1000/256) = 4
    assert res["revision_delta"] == 1, res
    assert res["max_grid_x"] == _FIX5_MAX_GRID_X, res     # the limit kernels.cu cites, as this device reports it


# ------------------------------------------------------------------------------------------------ sweep (d), core
# The P1-T core sites that consume a failed call's error-slot entry where nothing can raise (the external Storage
# drain and its restore, fill_'s guard restore while unwinding) were counted but did not record the error. They now
# record it first (count + the first one's site, device and error). Reuses the FIX4 shim modes.
def test_fix5_sweep_d_core_drain_restore_failure_is_recorded(fix4_shim):
    res = _run_fix4(fix4_shim, "drain_restore_fail")
    st = res["stats"]
    assert st["device_restore_failures"] == 1 and st["unraisable_errors"] == 1, res
    assert "restoring the caller's CUDA device after an imported Storage's drain" in st["first_unraisable_what"], res
    assert st["first_unraisable_device"] == 1 and st["first_unraisable_error"] == "cudaErrorInvalidDevice", res
    assert res["slot_after"] == 0, res


def test_fix5_sweep_d_core_drain_failure_is_recorded(f05_shim):
    # FIX1's F05 staging: the drain's switch to the Storage's device fails, so the release is leaked; FIX5 also records
    # that CUDA error (it used to be consumed with only the leak counted).
    res = _run_f05(f05_shim, "fail")
    st = res["stats"]
    assert st["leaked_drain_failure"] == 1 and st["unraisable_errors"] == 1, res
    assert "draining an imported Storage's device before its DLPack release" in st["first_unraisable_what"], res
    assert st["first_unraisable_device"] == 0 and st["first_unraisable_error"] == "cudaErrorInvalidDevice", res


def test_fix5_sweep_d_core_fill_unwinding_restore_failure_is_recorded(fix4_shim):
    res = _run_fix4(fix4_shim, "fill_fail_launch_and_restore")
    st = res["stats"]
    assert res["raised"] == "RuntimeError" and st["unraisable_errors"] == 1, res
    assert "while fill_ unwound" in st["first_unraisable_what"], res
    assert st["first_unraisable_error"] == "cudaErrorInvalidDevice" and res["slot_after"] == 0, res


# ------------------------------------------------------------------------------------------------ sweep (c), core
# Core's exit wait (dlpack_begin_shutdown) re-took the GIL in py::gil_scoped_release's DESTRUCTOR. A non-main thread can
# run the wait (tensor_cuda._dlpack_shutdown_hook is a plain function): here a daemon thread calls it while a foreign
# pthread's release (1 s) is in flight, then the main thread's own atexit call waits too. When the release leaves, both
# wake; if the main thread re-takes the GIL first, CPython finalizes, and the daemon thread's GIL re-acquisition exits it
# (a forced unwind). Started inside a destructor, that unwind terminates the process; FIX5 re-takes the GIL explicitly
# in a non-noexcept frame. Which waiter wins is a race, so the scenario runs several times; every run must exit rc 0
# with the release completed before finalization. Prior art: as the FIX2 N5 / FIX3 V3-01 scripts above.
_FIX5_CORE_HOOK_THREAD_SCRIPT = r"""
import atexit, ctypes, gc, json, os, sys, threading, time, types
RESULTS = {}
def _after_hook():
    RESULTS["after_hook"] = {"stats": dict(tc._C._dlpack_release_stats()), "calls": prod.calls, "done": prod.done}
    print("RESULT " + json.dumps(RESULTS), flush=True)
atexit.register(_after_hook)
""" + _SUBPROC_PRODUCER + _FIX2_PTHREAD + r"""
class SlowProducer(Producer):
    def _delete(self, _p):
        self.calls += 1
        self.started = True
        time.sleep(1.0)
        self.done = True
prod = SlowProducer(tc.empty(16), 16)
prod.started = prod.done = False
M, FP = held_export(prod)
def _wait_for(cond, what):
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > 10:
            raise RuntimeError("timed out waiting for " + what)
        time.sleep(0.002)
def start():
    start_foreign(FP, M)
    _wait_for(lambda: prod.started, "the foreign release to start")
    threading.Thread(target=tc._dlpack_shutdown_hook, daemon=True).start()     # a NON-main thread runs the wait
    _wait_for(lambda: tc._C._dlpack_release_stats()["closing"], "the daemon thread's hook to close admission")
    RESULTS["before_hook"] = {"stats": dict(tc._C._dlpack_release_stats())}
atexit.register(start)
class Finalizer:
    def __init__(self):
        self.sleep, self.isf = time.sleep, sys.is_finalizing
    def __del__(self):
        fin = self.isf()
        self.sleep(1.5)                          # drops the GIL during finalization: the daemon thread can take it
        print("FINAL finalizing=%s" % fin, flush=True)
_holder = types.ModuleType("_fix5_core_finalizer_holder")
_holder.keep = Finalizer()
sys.modules["_fix5_core_finalizer_holder"] = _holder
del _holder
print("ARMED", flush=True)
"""


@pytest.mark.parametrize("run", range(5))
def test_fix5_sweep_c_core_exit_wait_on_a_non_main_thread_never_aborts(run):
    r = subprocess.run([sys.executable, "-c", _FIX5_CORE_HOOK_THREAD_SCRIPT], capture_output=True, text=True,
                       timeout=_SUBPROC_TIMEOUT, env=_subproc_env())
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, f"rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}"
    res = json.loads(lines[-1][len("RESULT "):])
    assert res["before_hook"]["stats"]["in_flight"] == 1, res
    assert res["after_hook"]["calls"] == 1 and res["after_hook"]["done"] is True, res   # finished before finalizing
    _assert_ordinary_exit(r)                     # rc 0, FINAL line, no FATAL / terminate / abort
