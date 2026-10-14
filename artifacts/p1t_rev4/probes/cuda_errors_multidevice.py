"""LEAD GPU ONLY. PASS iff sync raises for pending CUDA errors and drains/restores each device once,
without early/double deleters; requires >=2 CUDA devices, otherwise INCONCLUSIVE (exit3).
Prior art: CUDA12.6 streams/events + DLPack2017 lifetime. Ours: cross-device FIFO and preexisting error census.
"""
import ctypes, gc, sys
import torch, tc_signal as ts
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tc_signal/tests'))
import tcs_producer as P
if torch.cuda.device_count()<2:
    print('INCONCLUSIVE: two devices required');raise SystemExit(3)
ts.set_allocator(lambda s: torch.empty(s,device=torch.cuda.current_device()))
P.reset_stats()
held=[]
for d in [0,1]:
    with torch.cuda.device(d):
        x=torch.arange(32,device=d,dtype=torch.float32)+.5
        prod=P.Producer(x,(32,))
        out=torch.empty_like(x)
        torch.cuda._sleep(400_000_000)
        ts.rint(prod,out=out)
        held.append((x,out))
torch.cuda.set_device(1)
ts.synchronize()
assert torch.cuda.current_device()==1
assert ts._C.pending_stats()['entries']==0
assert P.STATS['deleted']==2 and P.STATS['unknown_delete']==0,P.STATS
for x,y in held:
    torch.testing.assert_close(y,torch.round(x),rtol=0,atol=0)
# A synchronous launch-configuration error, deliberately left in the runtime last-error slot.
r=ctypes.CDLL('/usr/local/cuda-12.6/lib64/libcudart.so.12')
class Dim3(ctypes.Structure): _fields_=[('x',ctypes.c_uint),('y',ctypes.c_uint),('z',ctypes.c_uint)]
r.cudaLaunchKernel.argtypes=[ctypes.c_void_p,Dim3,Dim3,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
r.cudaPeekAtLastError.restype=ctypes.c_int
r.cudaGetLastError()
e=r.cudaLaunchKernel(None,Dim3(0,1,1),Dim3(0,1,1),None,0,None)
assert e and r.cudaPeekAtLastError(), 'INCONCLUSIVE: could not stage a last-error slot'
try:
    ts.synchronize()
except RuntimeError as ex:
    print('PASS: preexisting CUDA error reported',ex)
else:
    raise AssertionError('synchronize returned success with a preexisting CUDA launch error')
finally:
    r.cudaGetLastError()
print('PASS: two-device retained releases, exact output, caller device restored')
