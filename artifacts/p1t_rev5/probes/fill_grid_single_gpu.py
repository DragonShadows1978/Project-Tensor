"""PASS iff fill_ explicitly refuses the oversized non-grid-stride extent before cudaLaunchKernel.
Prior art: CUDA12.6 grid limits/C++17 narrowing; ours: full call-path geometry witness on synthetic huge metadata.
LEAD ONLY, one GPU, timeout30. LD_PRELOAD /tmp/p1t_rev5_fill_grid.so built from fill_grid_shim.c.
No huge resident allocation: the shim captures and refuses the launch, NEVER runs it on the small backing buffer.
If the fix instead adds grid-stride coverage, re-audit that kernel and replace this refusal-rule in a separate amendment.
"""
import ctypes as C,sys
from pathlib import Path
import torch,tensor_cuda as tc
if not torch.cuda.is_available():raise SystemExit(3)
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tc_signal/tests'))
from tcs_producer import DLManagedTensor,DLDevice,DLDataType
shim=C.CDLL('/tmp/p1t_rev5_fill_grid.so');shim.rev5_fill_grid.restype=C.c_uint
owner=torch.zeros(256,device='cuda');shape=(C.c_int64*1)((1<<40)+1);m=DLManagedTensor();t=m.dl_tensor
# Deliberately metadata-only descriptor: NOT a claim that the small allocation has this extent.
t.data=owner.data_ptr();t.device=DLDevice(2,owner.device.index);t.dtype=DLDataType(2,32,1);t.ndim=1;t.shape=shape
new=C.pythonapi['PyCapsule_New'];new.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p];new.restype=C.py_object
u=tc.from_dlpack(new(C.addressof(m),b'dltensor',None))
torch.cuda.synchronize();C.CDLL('libcudart.so.12').cudaGetLastError()
shim.rev5_arm_fill()
try:u.fill_(7)
except (ValueError,OverflowError,RuntimeError) as e:
 print('REFUSAL',e,'launches',shim.rev5_fill_calls(),'grid',shim.rev5_fill_grid(),flush=True)
 assert shim.rev5_fill_calls()==0,'fill_ attempted a narrowed launch instead of refusing the size'
else:raise AssertionError('oversized fill was not refused')
print('PASS: oversized fill refused before narrowing/launch')
