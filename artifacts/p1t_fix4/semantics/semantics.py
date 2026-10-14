# Empirical CUDA 12.6 runtime semantics used by the FIX4 design (receipt).
import ctypes as C, os, sys, json
mode = sys.argv[1]
r = C.CDLL('/usr/local/cuda-12.6/lib64/libcudart.so.12')
class Dim3(C.Structure):
    _fields_ = [('x', C.c_uint), ('y', C.c_uint), ('z', C.c_uint)]
r.cudaLaunchKernel.argtypes = [C.c_void_p, Dim3, Dim3, C.c_void_p, C.c_size_t, C.c_void_p]
class Attr(C.Structure):
    _fields_ = [('type', C.c_int), ('device', C.c_int), ('devicePointer', C.c_void_p), ('hostPointer', C.c_void_p)]
def attrs(p):
    a = Attr(); e = r.cudaPointerGetAttributes(C.byref(a), C.c_void_p(p)); return e, a.type, a.device
out = {}
if mode == 'slot':
    d = C.c_int(-1); out['getdev'] = (r.cudaGetDevice(C.byref(d)), d.value)
    r.cudaGetLastError()
    out['launch_err'] = r.cudaLaunchKernel(None, Dim3(0,1,1), Dim3(0,1,1), None, 0, None)
    out['peek1'] = r.cudaPeekAtLastError()
    out['peek2_still'] = r.cudaPeekAtLastError()
    # a successful call: does it reset the slot?
    out['getdev_ok'] = r.cudaGetDevice(C.byref(d))
    out['peek_after_success'] = r.cudaPeekAtLastError()
    # a failing call: does it overwrite the slot?
    out['setdev_bad'] = r.cudaSetDevice(999)
    out['peek_after_fail'] = r.cudaPeekAtLastError()
    out['get_consumes'] = r.cudaGetLastError()
    out['peek_after_get'] = r.cudaPeekAtLastError()
elif mode == 'sticky':
    import torch
    torch.cuda.synchronize()
    x = torch.zeros(4, device='cuda'); idx = torch.tensor([100], device='cuda')
    try:
        y = x[idx]; torch.cuda.synchronize()
    except Exception as e:
        out['torch_err'] = str(e).splitlines()[0][:120]
    d = C.c_int(-1)
    out['peek_ours'] = r.cudaPeekAtLastError()
    out['getdev'] = (r.cudaGetDevice(C.byref(d)), d.value)
    out['streamquery_legacy'] = r.cudaStreamQuery(C.c_void_p(1))
    out['peek_after_query'] = r.cudaPeekAtLastError()
    out['get1'] = r.cudaGetLastError(); out['get2_sticky'] = r.cudaGetLastError()
    out['sync'] = r.cudaStreamSynchronize(C.c_void_p(1))
    ev = C.c_void_p(); out['evcreate'] = r.cudaEventCreateWithFlags(C.byref(ev), 2)
    out['setdev0'] = r.cudaSetDevice(0)
elif mode == 'attrs':
    import torch
    t = torch.empty(1024, device='cuda'); out['torch_default'] = attrs(t.data_ptr())
    out['torch_offset'] = attrs(t.data_ptr() + 64)
    sys.path[:0] = [os.environ['TCP']]
    import tensor_cuda as tc
    e = tc.empty(1024); out['tc_empty_raw'] = attrs(e.data_ptr())
    tc._C.set_alloc_pooling(True) if hasattr(tc._C, 'set_alloc_pooling') else None
    e2 = tc.empty(1024); out['tc_empty_pooled'] = attrs(e2.data_ptr())
    out['host_ptr'] = attrs(C.addressof(C.c_int(1)))
    out['null'] = attrs(0)
    out['bogus'] = attrs(0x10)
    out['peek_after'] = r.cudaPeekAtLastError()
print('RESULT', json.dumps(out))
