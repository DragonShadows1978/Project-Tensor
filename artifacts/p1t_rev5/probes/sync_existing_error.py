"""LEAD ONLY, one GPU. PASS iff synchronize raises for the staged synchronous launch error;
FAIL iff it returns success; INCONCLUSIVE (3) if staging fails. timeout 30, no invalid device-memory access.
Prior art: NVIDIA CUDA 12.6 per-thread last-error slot and stream synchronization API.
Taken: cudaPeekAtLastError; ours: isolate the earlier round's error census from its two-GPU prerequisite.
"""
import ctypes as C
import torch, tc_signal as ts
assert torch.cuda.is_available()
torch.cuda.synchronize()
r = C.CDLL('/usr/local/cuda-12.6/lib64/libcudart.so.12')
class Dim3(C.Structure):
    _fields_ = [('x', C.c_uint), ('y', C.c_uint), ('z', C.c_uint)]
r.cudaLaunchKernel.argtypes = [C.c_void_p, Dim3, Dim3, C.c_void_p, C.c_size_t, C.c_void_p]
r.cudaPeekAtLastError.restype = C.c_int
r.cudaGetLastError()
e = r.cudaLaunchKernel(None, Dim3(0, 1, 1), Dim3(0, 1, 1), None, 0, None)
staged = r.cudaPeekAtLastError()
if not (e and staged):
    print('INCONCLUSIVE: no synchronous launch error staged', e, staged)
    raise SystemExit(3)
try:
    ts.synchronize()
except RuntimeError as ex:
    print('PASS: staged error reported', e, staged, ex)
else:
    raise AssertionError('synchronize returned success with a preexisting launch error; slot=' + str(r.cudaPeekAtLastError()))
finally:
    r.cudaGetLastError()
