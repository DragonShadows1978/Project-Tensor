"""CPU/full-core lead probe. Run with CUDA_VISIBLE_DEVICES='' and timeout 30.
PASS iff cudaGetDevice failure raises RuntimeError before capsule consumption; FAIL iff import succeeds.
Prior art: CUDA 12.6 checked return codes and DLPack 2017+ ownership; ours: zero-size no-data-dereference case.
"""
import ctypes as C
import tensor_cuda as tc
class Device(C.Structure): _fields_=[('kind',C.c_int32),('ordinal',C.c_int32)]
class DType(C.Structure): _fields_=[('code',C.c_uint8),('bits',C.c_uint8),('lanes',C.c_uint16)]
class Tensor(C.Structure):
    _fields_=[('data',C.c_void_p),('device',Device),('ndim',C.c_int32),('dtype',DType),
              ('shape',C.POINTER(C.c_int64)),('strides',C.POINTER(C.c_int64)),('byte_offset',C.c_uint64)]
class Managed(C.Structure):
    _fields_=[('tensor',Tensor),('ctx',C.c_void_p),('deleter',C.c_void_p)]
r=C.CDLL('/usr/local/cuda-12.6/lib64/libcudart.so.12')
ordinal=C.c_int(-1)
assert r.cudaGetDevice(C.byref(ordinal)) != 0, 'INCONCLUSIVE: must run with no usable CUDA device'
r.cudaGetLastError()
shape=(C.c_int64*1)(0)
m=Managed(Tensor(None,Device(2,0),1,DType(2,32,1),shape,None,0),None,None)
new=C.pythonapi['PyCapsule_New'];new.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p];new.restype=C.py_object
name=C.pythonapi['PyCapsule_GetName'];name.argtypes=[C.py_object];name.restype=C.c_char_p
cap=new(C.addressof(m),b'dltensor',None)
try:
    imported=tc.from_dlpack(cap)
except RuntimeError as e:
    assert 'cudaGetDevice' in str(e), str(e)
    assert name(cap)==b'dltensor', 'failed query consumed ownership'
    print('PASS: failed device query refused before consumption')
else:
    raise AssertionError('core import succeeded after failed device query; name=' + str(name(cap)))
