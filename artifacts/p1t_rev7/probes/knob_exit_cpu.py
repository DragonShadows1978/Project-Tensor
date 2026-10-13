"""PASS iff a diagnostic-delay conversion cannot bypass exit quiescence: callback done, normal child exit.
CPU only: actual add-on read-only refusal before launch. --control uses default knob.
Prior art: CPython3.12 atexit/finalization (PSF2023), DLPack ownership (DMLC2017).
Taken: finite callback exit staging from REV5; ours: valid positive integer delay beyond float range.
"""
import sys, atexit, ctypes as C, threading, time, types, resource
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root/'tc_signal/tests'))
from tcs_producer import DLManagedTensor,DLManagedTensorVersioned,DLDevice,DLDataType,DLPackVersion,DELETER
import tc_signal as ts
if '--control' not in sys.argv: ts._RELEASE_EXIT_DIAG_S = 1 << 4096
new=C.pythonapi['PyCapsule_New'];new.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p];new.restype=C.py_object
shape=(C.c_int64*1)(1); keep=[]; started=threading.Event()
@DELETER
def slow(p):
    print('DELETER started',flush=True); started.set(); time.sleep(.6); print('DELETER finished',flush=True)
def capsule(readonly):
    m=DLManagedTensorVersioned() if readonly else DLManagedTensor()
    if readonly: m.version=DLPackVersion(1,3); m.flags=1; m.deleter=slow
    t=m.dl_tensor;t.data=0x2000 if readonly else 0x1000;t.device=DLDevice(2,0);t.dtype=DLDataType(2,32,1);t.ndim=1;t.shape=shape
    keep.append(m)
    return new(C.addressof(m),b'dltensor_versioned' if readonly else b'dltensor',None)
x,out=capsule(False),capsule(True)
def worker():
    try: ts._C.rint_run(x,out)
    except ValueError: print('REFUSED',flush=True)
def start():
    threading.Thread(target=worker,daemon=True).start(); assert started.wait(2)
    print('IN_FLIGHT',ts._C.pending_stats()['releases_in_flight'],flush=True)
atexit.register(start)
class Final:
    def __init__(self):self.sleep,self.fin=time.sleep,sys.is_finalizing
    def __del__(self): print('FINALIZING',self.fin(),flush=True);self.sleep(1.5)
h=types.ModuleType('_p1t_rev7_final'); h.keep=Final();sys.modules[h.__name__]=h
print('ARMED',flush=True)
