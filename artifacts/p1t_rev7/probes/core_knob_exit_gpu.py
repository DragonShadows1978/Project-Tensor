"""LEAD ONLY. PASS iff actual core's finite deleter finishes before finalization, child exits0.
Single GPU; exit3 if unavailable. --control keeps the default diagnostic knob. No product-code copy.
Prior art: CPython3.12 atexit/GIL PSF2023, DLPack DMLC2017; taken: finite-deleter staging;
ours: integer overflow in the diagnostic configuration. Run after rebuilding reviewed core.
"""
from pathlib import Path
import sys,os,subprocess
root=Path(__file__).resolve().parents[3]
if '--child' not in sys.argv:
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'tensor_cuda')+':'+str(root/'tensor_cuda/tensor_cuda'))
    r=subprocess.run(['timeout','20',sys.executable,__file__,'--child',*sys.argv[1:]],env=env,capture_output=True,text=True)
    print(r.stdout,end='');print(r.stderr,end='',file=sys.stderr)
    if r.returncode==3:raise SystemExit(3)
    assert r.returncode==0 and 0<=r.stdout.find('DELETER finished')<r.stdout.find('FINALIZING True'),f'FAIL rc={r.returncode}'
    print('PASS core ordinary exit quiescence')
    raise SystemExit(0)
import torch
if not torch.cuda.is_available():raise SystemExit(3)
import ctypes as C,atexit,threading,time,types,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
sys.path.insert(0,str(root/'tc_signal/tests'))
from tcs_producer import DLManagedTensor,DLDevice,DLDataType,DELETER
import tensor_cuda as tc
if '--control' not in sys.argv:tc._DLPACK_SHUTDOWN_WAIT_S=1<<4096
new=C.pythonapi['PyCapsule_New'];new.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p];new.restype=C.py_object
started=threading.Event()
@DELETER
def slow(p):
    print('DELETER started',flush=True);started.set();time.sleep(.6);print('DELETER finished',flush=True)
shape=(C.c_int64*1)(0);m=DLManagedTensor();m.deleter=slow
m.dl_tensor.data=None;m.dl_tensor.device=DLDevice(2,torch.cuda.current_device());m.dl_tensor.dtype=DLDataType(2,32,1);m.dl_tensor.ndim=1;m.dl_tensor.shape=shape
cap=new(C.addressof(m),b'dltensor',None);box=[tc.from_dlpack(cap)]
def worker():del box[0]
def start():
    threading.Thread(target=worker,daemon=True).start();assert started.wait(3)
    print('IN_FLIGHT',tc._C._dlpack_release_stats()['in_flight'],flush=True)
atexit.register(start)
class Final:
    def __init__(self):self.sleep,self.fin=time.sleep,sys.is_finalizing
    def __del__(self):print('FINALIZING',self.fin(),flush=True);self.sleep(1.5)
h=types.ModuleType('_p1t_rev7_full_core_final');h.keep=Final();sys.modules[h.__name__]=h
print('ARMED actual core',flush=True)
