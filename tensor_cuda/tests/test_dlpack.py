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

_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
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
