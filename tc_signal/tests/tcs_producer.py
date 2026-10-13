"""A hand-built DLPack producer (ctypes) for testing tc_signal as a consumer of ANY producer, not just torch.

It exports a view of a torch CUDA tensor's memory as a DLPack capsule ("dltensor" or "dltensor_versioned") whose
fields the test controls: byte_offset, device type/id, dtype, strides, DLPack version, flags. It records the keyword
arguments `__dlpack__` received (stream, max_version, copy) and counts deleter calls, so the tests can check the
consumer rules: stream=1, exactly one deleter call per exported tensor on every path (success, refusal, unconsumed
capsule garbage-collected), byte_offset honoured, versioned and legacy capsules both read.

Prior art: the DLPack C ABI (dmlc/dlpack v1.3, include/dlpack/dlpack.h) and the Python array API `__dlpack__`
protocol; building PyCapsules through ctypes.pythonapi is standard CPython practice. The test producer itself is ours
(no prior art known to me).
"""
import ctypes

KDL_CUDA = 2
KDL_CPU = 1
KDL_FLOAT = 2
FLAG_READ_ONLY = 1 << 0
FLAG_IS_COPIED = 1 << 1


class DLDevice(ctypes.Structure):
    _fields_ = [('device_type', ctypes.c_int32), ('device_id', ctypes.c_int32)]


class DLDataType(ctypes.Structure):
    _fields_ = [('code', ctypes.c_uint8), ('bits', ctypes.c_uint8), ('lanes', ctypes.c_uint16)]


class DLTensor(ctypes.Structure):
    _fields_ = [('data', ctypes.c_void_p), ('device', DLDevice), ('ndim', ctypes.c_int32), ('dtype', DLDataType),
                ('shape', ctypes.POINTER(ctypes.c_int64)), ('strides', ctypes.POINTER(ctypes.c_int64)),
                ('byte_offset', ctypes.c_uint64)]


DELETER = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class DLManagedTensor(ctypes.Structure):
    _fields_ = [('dl_tensor', DLTensor), ('manager_ctx', ctypes.c_void_p), ('deleter', DELETER)]


class DLPackVersion(ctypes.Structure):
    _fields_ = [('major', ctypes.c_uint32), ('minor', ctypes.c_uint32)]


class DLManagedTensorVersioned(ctypes.Structure):
    _fields_ = [('version', DLPackVersion), ('manager_ctx', ctypes.c_void_p), ('deleter', DELETER),
                ('flags', ctypes.c_uint64), ('dl_tensor', DLTensor)]


CAPSULE_DESTRUCTOR = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_NAME = b'dltensor'
_NAME_V = b'dltensor_versioned'

_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, CAPSULE_DESTRUCTOR]
_PyCapsule_IsValid = ctypes.pythonapi.PyCapsule_IsValid
_PyCapsule_IsValid.restype = ctypes.c_int
_PyCapsule_IsValid.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.restype = ctypes.c_void_p
_PyCapsule_GetPointer.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

LIVE = {}                                     # managed-tensor address -> kept-alive objects
STATS = {'exported': 0, 'deleted': 0, 'unknown_delete': 0}


def _delete(addr):
    if LIVE.pop(addr, None) is None:
        STATS['unknown_delete'] += 1          # double delete or a pointer we never exported
    else:
        STATS['deleted'] += 1


@DELETER
def _deleter(addr):
    _delete(addr)


@CAPSULE_DESTRUCTOR
def _capsule_destructor(cap):
    # The producer side of the protocol: an unconsumed capsule (name unchanged) deletes its tensor when collected;
    # a consumed one (renamed "used_*") belongs to the consumer, who calls the deleter.
    for name in (_NAME, _NAME_V):
        if _PyCapsule_IsValid(cap, name):
            _delete(_PyCapsule_GetPointer(cap, name))
            return


def reset_stats():
    for k in STATS:
        STATS[k] = 0


class Producer:
    """DLPack view of `base` (a torch CUDA tensor) starting `offset_elems` elements in, with logical `shape`.

    Knobs: versioned (capsule kind), byte_offset_mode ('data' = data pointer already offset, byte_offset 0;
    'field' = data = base pointer, byte_offset carries the offset), extra_byte_offset (added to byte_offset, e.g. 2
    to misalign), device_type, device_id, dtype (code, bits, lanes), strides (None or a tuple), major (DLPack major
    version), flags."""

    def __init__(self, base, shape, *, offset_elems=0, versioned=True, byte_offset_mode='field', extra_byte_offset=0,
                 device_type=KDL_CUDA, device_id=None, dtype=(KDL_FLOAT, 32, 1), strides=None, major=1, flags=0,
                 return_value=None):
        self.base, self.shape, self.offset_elems = base, tuple(shape), offset_elems
        self.versioned, self.byte_offset_mode, self.extra_byte_offset = versioned, byte_offset_mode, extra_byte_offset
        self.device_type = device_type
        self.device_id = base.device.index if device_id is None else device_id
        self.dtype, self.strides, self.major, self.flags = dtype, strides, major, flags
        self.return_value = return_value
        self.calls = []                       # kwargs of every __dlpack__ call

    def __dlpack_device__(self):
        return (self.device_type, self.device_id)

    def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        self.calls.append({'stream': stream, 'max_version': max_version, 'dl_device': dl_device, 'copy': copy})
        if self.return_value is not None:
            return self.return_value
        return self._export(versioned=self.versioned and max_version is not None and max_version[0] >= 1)

    def _export(self, versioned):
        nd = len(self.shape)
        shape = (ctypes.c_int64 * nd)(*self.shape)
        strides = None if self.strides is None else (ctypes.c_int64 * nd)(*self.strides)
        elem = self.base.element_size()
        if self.byte_offset_mode == 'field':
            data, boff = self.base.data_ptr(), self.offset_elems * elem
        else:
            data, boff = self.base.data_ptr() + self.offset_elems * elem, 0
        boff += self.extra_byte_offset
        m = DLManagedTensorVersioned() if versioned else DLManagedTensor()
        t = m.dl_tensor
        t.data = data
        t.device = DLDevice(self.device_type, self.device_id)
        t.ndim = nd
        t.dtype = DLDataType(*self.dtype)
        t.shape = shape
        t.strides = strides if strides is not None else ctypes.POINTER(ctypes.c_int64)()
        t.byte_offset = boff
        m.manager_ctx = None
        m.deleter = _deleter
        if versioned:
            m.version = DLPackVersion(self.major, 3)
            m.flags = self.flags
        addr = ctypes.addressof(m)
        LIVE[addr] = (m, shape, strides, self.base)
        STATS['exported'] += 1
        return _PyCapsule_New(addr, _NAME_V if versioned else _NAME, _capsule_destructor)


class LegacyOnlyProducer(Producer):
    """A pre-2023 producer: __dlpack__ accepts only `stream` (newer keywords raise TypeError)."""

    def __dlpack__(self, *, stream=None):
        self.calls.append({'stream': stream})
        return self._export(versioned=False)
