"""PASS iff ordinary exit exits0 after the finite cast callback finishes, without abort/segfault.
Single-GPU full-core check for V6-02; exits3 INCONCLUSIVE without one GPU. No GPU kernel execution needed.
Run fresh processes under timeout15: python core_cast_exit_gpu.py bool|index [--control].
Prior art: CPython3.12 GIL/exit (PSF2023), DLPack/array API, pybind11v2.12 ownership.
Taken: those lifecycle contracts; ours: real public export argument callback at ordinary exit.
"""
import atexit,ctypes,sys,threading,time,types,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
rt=ctypes.CDLL('/usr/local/cuda-12.6/lib64/libcudart.so.12');d=ctypes.c_int()
if rt.cudaGetDeviceCount(ctypes.byref(d))!=0 or d.value<1:
    print('INCONCLUSIVE: one GPU required');sys.exit(3)
import tensor_cuda as tc
started=threading.Event();done=threading.Event()
t=tc.empty((0,))
class Flag:
    def __bool__(self):
        print('CAST started',flush=True);started.set();time.sleep(.6);done.set();print('CAST finished',flush=True);return False
class Index:
    def __index__(self):
        print('CAST started',flush=True);started.set();time.sleep(.6);done.set();print('CAST finished',flush=True);return 1
def worker():
    try:
        if sys.argv[1]=='bool':t.__dlpack__(copy=Flag())
        else:t.__dlpack__(max_version=(Index(),3))
    except (TypeError,ValueError):pass # a strict pre-callback refusal is an acceptable fix
    print('EXPORT ended',flush=True)
def start():
    th=threading.Thread(target=worker,daemon=True);th.start()
    if not started.wait(2):
        th.join(2);assert not th.is_alive();return
    print('CAST IN PROGRESS',flush=True)
    if '--control' in sys.argv:th.join(2);assert not th.is_alive()
# Core's real registered atexit hook runs after this starts the callback.
atexit.register(start)
class Final:
    def __init__(self):self.sleep,self.fin=time.sleep,sys.is_finalizing
    def __del__(self):print('FINALIZING',self.fin(),flush=True);self.sleep(1.5)
h=types.ModuleType('_p1t_rev6_real_cast_holder');h.keep=Final();sys.modules[h.__name__]=h
print('ARMED real core',flush=True)
