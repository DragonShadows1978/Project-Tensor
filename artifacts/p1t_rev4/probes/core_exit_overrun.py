"""LEAD GPU ONLY. PASS iff ordinary process exit is rc0 after an admitted finite 5.5s deleter
exceeds the default 5s hook wait (control during must also rc0); FAIL on abort/hang.
Run both: timeout 20 python this_file.py during; timeout 20 python this_file.py bounded.
Prior art: CPython3.12 lifecycle, FIX2 real imported-Storage pthread harness. Taken: full-path staging;
ours: default timeout overrun with normal finalization, removing the test-only immediate-exit escape.
"""

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
    if MODE == "bounded":
        pass  # review: ordinary CPython finalization must run
atexit.register(_after_hook)

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

MODE = sys.argv[1]
SLEEP = {"during": 1.0, "nested": 0.0, "bounded": 5.5}.get(MODE, 0.0)
if MODE == "bounded":
    tc._DLPACK_SHUTDOWN_WAIT_S = 5.0           # the knob, read by the hook at exit
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
if MODE in ("during", "nested", "bounded"):
    atexit.register(_start_in_flight)
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
