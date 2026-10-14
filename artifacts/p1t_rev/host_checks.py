"""CPU metadata only; no CUDA calls. Prior art: DLPack v1.3 ABI and POSIX subprocess isolation.
Synthetic descriptors exercise refusal boundaries, not device numerics or real allocation validity.
"""
import ctypes as C
import json, os, resource, subprocess, sys
import tc_signal as ts
from tc_signal import _tc_signal as ext
class Dev(C.Structure): _fields_=[('type',C.c_int),('id',C.c_int)]
class DT(C.Structure): _fields_=[('code',C.c_uint8),('bits',C.c_uint8),('lanes',C.c_uint16)]
class Tensor(C.Structure):
    _fields_=[('data',C.c_void_p),('dev',Dev),('ndim',C.c_int),('dtype',DT),('shape',C.POINTER(C.c_int64)),('strides',C.POINTER(C.c_int64)),('offset',C.c_uint64)]
class Managed(C.Structure): _fields_=[('tensor',Tensor),('ctx',C.c_void_p),('deleter',C.c_void_p)]
keep=[]
new=C.pythonapi.PyCapsule_New; new.restype=C.py_object; new.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p]
def cap(shape):
    dims=(C.c_int64*len(shape))(*shape)
    buf=(C.c_float*1)(1)
    m=Managed(Tensor(C.addressof(buf),Dev(2,0),len(shape),DT(2,32,1),dims,None,0),None,None)
    keep.extend((dims,buf,m))
    return new(C.addressof(m),b'dltensor',None)
if len(sys.argv)>1:
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    action=sys.argv[1]
    if action=='none_up': ts.upsample_nearest1d(None,2)
    if action=='none_pad': ts.reflection_pad1d(None,(0,0))
    if action=='none_rint': ts.rint(None,out=cap(()))
    if action=='pad_overflow': print(ext.reflection_pad1d_plan(cap((1,1,1<<62)),(1<<62)-1,(1<<62)-1)[1])
    if action=='conv_lda_guard': print(ext.conv1d_plan(cap((1,1,1<<31)),cap((1,1,1)),None,1,0,0,1,1))
    if action=='lstm_n_guard': print(ext.lstm_plan(cap((1<<31,1,1)),cap((1,4,1)),cap((1,4,1)),None,None,None,1,0))
    if action=='rint_scalar': print(ext.rint_plan(cap(()))[1])
    if action=='empty_refusal': print(ext.rint_plan(cap((0,)))[1])
    sys.exit(0)
results=[]
for action in ('none_up','none_pad','none_rint','pad_overflow','conv_lda_guard','lstm_n_guard','rint_scalar','empty_refusal'):
    p=subprocess.run([sys.executable,__file__,action],capture_output=True,text=True,timeout=10)
    results.append(dict(action=action,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
print(json.dumps(results,indent=2))
