"""PASS iff a finite add-on producer deleter finishes before finalization and child exits 0.
Prior art: CPython3.12 (PSF,2023) GIL/finalization; DLPack2017 ownership; C++17 noexcept.
Ours: actual rebuilt package, immediate refusal callback on a daemon thread at ordinary exit.
No GPU execution: the versioned read-only output is refused before RetainUntilComplete/launch.
Run with reviewed tc_signal on PYTHONPATH under timeout15. --control joins before exit.
"""
import atexit, sys, time, threading, types, resource, ctypes as C
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tc_signal/tests'))
from tcs_producer import DLManagedTensor,DLManagedTensorVersioned,DLDevice,DLDataType,DLPackVersion,DELETER
import tc_signal as ts
new=C.pythonapi['PyCapsule_New'];new.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p];new.restype=C.py_object
sh=(C.c_int64*1)(1)
keep=[]
started=threading.Event()
@DELETER
def slow(p):
    print('DELETER started',flush=True)
    started.set()
    time.sleep(.6)
    print('DELETER finished',flush=True)
def capsule(readonly=False):
    m=DLManagedTensorVersioned() if readonly else DLManagedTensor()
    if readonly:m.version=DLPackVersion(1,3);m.flags=1;m.deleter=slow
    t=m.dl_tensor;t.data=0x2000 if readonly else 0x1000;t.device=DLDevice(2,0);t.dtype=DLDataType(2,32,1);t.ndim=1;t.shape=sh
    keep.append(m)
    return new(C.addressof(m),b'dltensor_versioned' if readonly else b'dltensor',None)
x,out=capsule(),capsule(True)
def worker():
    try:ts._C.rint_run(x,out)
    except ValueError as e:print('REFUSED',e,flush=True)
def start():
    thread=threading.Thread(target=worker,daemon=True)
    thread.start()
    assert started.wait(2)
    print('CALLBACK IN PROGRESS entries='+str(ts._C.pending_stats()['entries']),flush=True)
    if '--control' in sys.argv:thread.join(2);assert not thread.is_alive()
# Native package's real hook already registered. This runs first.
atexit.register(start)
class Finalizer:
    def __init__(self):self.sleep,self.fin=time.sleep,sys.is_finalizing
    def __del__(self):
        print('FINALIZING '+str(self.fin()),flush=True)
        self.sleep(1.5)
holder=types.ModuleType('_p1t_rev5_full_holder');holder.keep=Finalizer();sys.modules[holder.__name__]=holder;del holder
print('ARMED FULL CPU',flush=True)
