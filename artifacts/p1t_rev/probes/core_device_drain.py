"""LEAD GPU ONLY, two devices: decides whether destructor fences the allocation's device.
Prior art: CUDA runtime current-device/default-stream semantics; DLPack last-use ownership.
Run under timeout 60. A single-GPU host exits 3 (INCONCLUSIVE). No source edits.
"""
import ctypes as C, json, torch, tensor_cuda as tc
if torch.cuda.device_count()<2:
    print('INCONCLUSIVE: two CUDA devices required'); raise SystemExit(3)
# Last managed reference has a Python callback; callback samples the device-0 event.
class Dev(C.Structure): _fields_=[('type',C.c_int),('id',C.c_int)]
class DT(C.Structure): _fields_=[('code',C.c_uint8),('bits',C.c_uint8),('lanes',C.c_uint16)]
class DL(C.Structure): _fields_=[('data',C.c_void_p),('dev',Dev),('ndim',C.c_int),('dtype',DT),('shape',C.POINTER(C.c_int64)),('strides',C.POINTER(C.c_int64)),('offset',C.c_uint64)]
class M(C.Structure): pass
CB=C.CFUNCTYPE(None,C.POINTER(M)); M._fields_=[('dl',DL),('ctx',C.c_void_p),('deleter',CB)]
new=C.pythonapi.PyCapsule_New; new.restype=C.py_object; new.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p]
torch.cuda.set_device(0)
p=torch.empty(1024,device='cuda:0'); dims=(C.c_int64*1)(1024); stride=(C.c_int64*1)(1)
event=torch.cuda.Event(); receipt=[]
@CB
def release(_): receipt.append({'device0_work_done':event.query()})
m=M(DL(p.data_ptr(),Dev(2,0),1,DT(2,32,1),dims,stride,0),None,release)
u=tc.from_dlpack(new(C.addressof(m),b'dltensor',None))
torch.cuda._sleep(80_000_000); u.fill_(5); event.record(torch.cuda.default_stream(0))
torch.cuda.set_device(1)
del u
# Keep real producer storage alive; this probe observes the early deleter without causing UAF.
torch.cuda.set_device(0); torch.cuda.synchronize()
print(json.dumps(receipt))
assert receipt and receipt[0]['device0_work_done'], 'F05: deleter called before device-0 last use'
