import ctypes, sys, threading, time
sys.path.insert(0, '/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix6/tc_signal/tests')
import tcs_producer as P
import numpy as np
consumer = sys.argv[1]
lib = ctypes.CDLL('/tmp/p1t_fix6_scratch/r2/exit_deleter.so')
new = ctypes.pythonapi['PyCapsule_New']; new.restype = ctypes.py_object; new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
buf = np.ones(4, dtype=np.float32)
SH = (ctypes.c_int64 * 1)(4)
def cap(device, versioned=False, flags=0):
    m = P.DLManagedTensorVersioned() if versioned else P.DLManagedTensor()
    if versioned: m.version, m.flags = P.DLPackVersion(1, 3), flags
    m.deleter = P.DELETER(ctypes.cast(lib.exit_deleter, ctypes.c_void_p).value)
    t = m.dl_tensor; t.data = buf.ctypes.data if device == 1 else 0x2000; t.device = P.DLDevice(device, 0)
    t.dtype = P.DLDataType(2, 32, 1); t.ndim = 1; t.shape = SH; t.strides = ctypes.POINTER(ctypes.c_int64)()
    cap.keep.append(m)
    return new(ctypes.addressof(m), b'dltensor_versioned' if versioned else b'dltensor', None)
cap.keep = []
def worker():
    if consumer == 'torch':
        import torch
        x = torch.from_dlpack(cap(1)); print('IMPORTED', x.sum().item(), flush=True); del x
    elif consumer == 'numpy':
        class Prod:
            def __dlpack_device__(self): return (1, 0)
            def __dlpack__(self, **kw): return cap(1)
        a = np.from_dlpack(Prod()); print('IMPORTED', float(a.sum()), flush=True); del a
    elif consumer == 'tc_signal':
        import tc_signal as ts
        try: ts._C.rint_run(cap(2), cap(2, True, P.FLAG_READ_ONLY))
        except ValueError: pass
    print('WORKER returned (deleter returned)', flush=True)
th = threading.Thread(target=worker); th.start(); th.join(5)
print('MAIN: worker alive after join(5) =', th.is_alive(), flush=True)
