"""PASS iff a finite deferred add-on deleter finishes before finalization and child exits0; no abort/hang.
Prior art: DLPack2017 ownership, CPython3.12/2023 lifecycle; ours: actual completed-event callback hidden at exit.
LEAD ONLY, one GPU, timeout20. --control joins the worker to isolate finalization. Exit3 if no GPU.
"""
import atexit,ctypes as C,resource,sys,time,threading,types
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch,tc_signal as ts
if not torch.cuda.is_available():raise SystemExit(3)
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tc_signal/tests'))
from tcs_producer import DLManagedTensor,DLDevice,DLDataType,DELETER
new=C.pythonapi['PyCapsule_New'];new.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p];new.restype=C.py_object
owner=torch.ones(4,device='cuda');out=torch.empty_like(owner);shape=(C.c_int64*1)(4)
started=threading.Event()
@DELETER
def slow(p):
 print('DELETER started',flush=True);started.set();time.sleep(.6);print('DELETER finished',flush=True)
m=DLManagedTensor();m.deleter=slow;t=m.dl_tensor;t.data=owner.data_ptr();t.device=DLDevice(2,owner.device.index);t.dtype=DLDataType(2,32,1);t.ndim=1;t.shape=shape
cap=new(C.addressof(m),b'dltensor',None)
ts.rint(cap,out=out)
torch.cuda.synchronize()
assert ts._C.pending_stats()['entries']>0
assert out.cpu().tolist()==[1.]*4
def start():
 thread=threading.Thread(target=lambda:ts._C.rint_plan(out),daemon=True);thread.start()
 assert started.wait(2)
 print('CALLBACK IN PROGRESS entries='+str(ts._C.pending_stats()['entries']),flush=True)
 if '--control' in sys.argv:thread.join(2);assert not thread.is_alive()
atexit.register(start) # package's actual hook executes afterwards
class Finalizer:
 def __init__(self):self.sleep,self.fin=time.sleep,sys.is_finalizing
 def __del__(self):print('FINALIZING '+str(self.fin()),flush=True);self.sleep(1.5)
holder=types.ModuleType('_p1t_rev5_gpu_holder');holder.keep=Finalizer();sys.modules[holder.__name__]=holder;del holder
print('ARMED GPU',flush=True)
