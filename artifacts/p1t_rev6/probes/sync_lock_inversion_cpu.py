"""PASS iff synchronize reaches its CUDA drain without waiting on an unrelated refused-call deleter.
CPU only: real reviewed extension, output refused read-only before event/launch; exit 1 means lock inversion.
Prior art: Coffman et al. (1971) circular wait; CPython3.12 Lock/GIL (PSF2023), DLPack ownership (DMLC2017).
Taken: wait-for cycle analysis; ours: finite producer callback needs a caller-held allocator lock.
A rescue thread releases that lock after 1 s solely to terminate the witness normally (no process killed).
"""
import ctypes as C, threading, time, sys
import tc_signal as ts
class Dev(C.Structure): _fields_=[('kind',C.c_int),('id',C.c_int)]
class DT(C.Structure): _fields_=[('code',C.c_uint8),('bits',C.c_uint8),('lanes',C.c_uint16)]
class Tensor(C.Structure): _fields_=[('data',C.c_void_p),('device',Dev),('ndim',C.c_int),('dtype',DT),('shape',C.POINTER(C.c_int64)),('strides',C.POINTER(C.c_int64)),('offset',C.c_uint64)]
class Version(C.Structure): _fields_=[('major',C.c_uint32),('minor',C.c_uint32)]
class Legacy(C.Structure): _fields_=[('tensor',Tensor),('ctx',C.c_void_p),('deleter',C.c_void_p)]
class Modern(C.Structure): _fields_=[('version',Version),('ctx',C.c_void_p),('deleter',C.c_void_p),('flags',C.c_uint64),('tensor',Tensor)]
new=C.pythonapi.PyCapsule_New;new.restype=C.py_object;new.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p]
lock=threading.Lock();started=threading.Event();finished=threading.Event();rescued=threading.Event()
shape=(C.c_int64*1)(1)
@C.CFUNCTYPE(None,C.c_void_p)
def release(p):
    started.set()
    with lock: finished.set()
x=Legacy();x.tensor=Tensor(0x1000,Dev(2,0),1,DT(2,32,1),shape,None,0)
y=Modern();y.version=Version(1,3);y.flags=1;y.deleter=C.cast(release,C.c_void_p);y.tensor=Tensor(0x2000,Dev(2,0),1,DT(2,32,1),shape,None,0)
xc=new(C.addressof(x),b'dltensor',None);yc=new(C.addressof(y),b'dltensor_versioned',None)
def worker():
    try: ts._C.rint_run(xc,yc)
    except ValueError: pass
lock.acquire();t=threading.Thread(target=worker);t.start();assert started.wait(2)
assert ts._C.pending_stats()['releases_in_flight']==1
sync_finished=threading.Event()
def rescue():
    if not sync_finished.wait(1):
        rescued.set(); print('BLOCKED_BEFORE_CUDA=1 finite deleter waits for caller-held lock',flush=True)
        lock.release()
r=threading.Thread(target=rescue);r.start();t0=time.monotonic()
try: ts.synchronize()
except RuntimeError as e: print(type(e).__name__,str(e),flush=True) # expected cudaErrorNoDevice on CPU host
finally:
    sync_finished.set()
    if not rescued.is_set(): lock.release()
t.join(3);r.join(3);assert finished.is_set()
print('elapsed=%.3f rescue=%s release_finished=%s' % (time.monotonic()-t0,rescued.is_set(),finished.is_set()),flush=True)
sys.exit(1 if rescued.is_set() else 0)
