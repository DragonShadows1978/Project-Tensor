import ctypes as C, json, torch
r = C.CDLL('/usr/local/cuda-12.6/lib64/libcudart.so.12')
class Attr(C.Structure):
    _fields_ = [('type', C.c_int), ('device', C.c_int), ('devicePointer', C.c_void_p), ('hostPointer', C.c_void_p)]
keep = torch.empty(1024, device='cuda'); torch.cuda.synchronize()
x = torch.zeros(4, device='cuda'); idx = torch.tensor([100], device='cuda')
try:
    y = x[idx]; torch.cuda.synchronize()
except Exception as e:
    pass
a = Attr(); e = r.cudaPointerGetAttributes(C.byref(a), C.c_void_p(keep.data_ptr()))
out = {'attr_rc': e, 'type': a.type, 'device': a.device, 'peek': r.cudaPeekAtLastError()}
r.cudaGetLastError()
ev = C.c_void_p(); out['evcreate'] = r.cudaEventCreateWithFlags(C.byref(ev), 2); r.cudaGetLastError()
s = C.c_void_p(); out['streamcreate'] = r.cudaStreamCreateWithFlags(C.byref(s), 1); r.cudaGetLastError()
out['query_legacy'] = r.cudaStreamQuery(C.c_void_p(1)); r.cudaGetLastError()
print('RESULT', json.dumps(out))
