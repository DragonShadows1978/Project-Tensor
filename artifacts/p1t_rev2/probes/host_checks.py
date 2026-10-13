"""CPU-only native checks; no allocations or kernels on CUDA.
PASS rule: clean refusals/expected shapes, synchronization propagates device-query errors,
core zero-size arithmetic has no UBSan diagnostic, allocation/producer errors do not abort.
Prior art: DLPack C ABI (dmlc 2017+), CPython ctypes, GCC UBSan; ours: adversarial cases.
"""
import ctypes as C
import json, os, resource, subprocess, sys
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/p1t_rev2'
if len(sys.argv)==1:
    cases=['none_up','none_pad','none_rint','reflect_overflow','upsample_overflow','null_shape','negative_rank','byte_overflow','address_overflow','scalar','zero_size','conv_int32','lstm_int32','same_capsule','empty_sync']
    results=[]
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONPATH']='/tmp/p1t_rev2_build_review/tc_signal'
    for c in cases:
        p=subprocess.run([sys.executable,__file__,c],env=env,text=True,capture_output=True,timeout=20)
        results.append(dict(case=c,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
    for exe,cases2 in [('/tmp/p1t_rev2_core_cpu',['normal','prefix','suffix']),('/tmp/p1t_rev2_cpu_native',['oom_sync','throw_deleter'])]:
        for c in cases2:
            p=subprocess.run([exe,c],env=env,text=True,capture_output=True,timeout=20)
            results.append(dict(case=c,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
    (OUT/'host_checks.json').write_text(json.dumps(results,indent=2)+'\n')
    for r in results: print(r['case'],r['returncode'],(r['stdout']+r['stderr']).strip())
    sys.exit(0) # receipt collection, not an all-green verdict
sys.path.insert(0,str(ROOT/'tc_signal/tests'))
from tcs_producer import DLTensor,DLManagedTensor,DLDevice,DLDataType
import tc_signal as ts
n=ts._C
keepers=[]
newcap=C.pythonapi['PyCapsule_New'];newcap.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p];newcap.restype=C.py_object
getname=C.pythonapi['PyCapsule_GetName'];getname.argtypes=[C.py_object];getname.restype=C.c_char_p

def cap(shape,ndim=None,offset=0,data=0x1000,nullshape=False):
    sh=(C.c_int64*len(shape))(*shape);m=DLManagedTensor();t=m.dl_tensor
    t.data=data;t.device=DLDevice(2,0);t.ndim=len(shape) if ndim is None else ndim;t.dtype=DLDataType(2,32,1)
    t.shape=sh if not nullshape else C.POINTER(C.c_int64)();t.strides=C.POINTER(C.c_int64)();t.byte_offset=offset
    keepers.extend([sh,m]);return newcap(C.addressof(m),b'dltensor',None)
case=sys.argv[1]
try:
    if case=='none_up': ts.upsample_nearest1d(None,2)
    elif case=='none_pad': ts.reflection_pad1d(None,(0,0))
    elif case=='none_rint': ts.rint(None,out=cap(()))
    elif case=='reflect_overflow': n.reflection_pad1d_plan(cap((1,1,1<<62)),(1<<62)-1,(1<<62)-1)
    elif case=='upsample_overflow': n.upsample_nearest1d_plan(cap((1,)),(1<<63)-1)
    elif case=='null_shape': n.rint_plan(cap((2,),nullshape=True))
    elif case=='negative_rank': n.rint_plan(cap((),ndim=-1))
    elif case=='byte_overflow': n.rint_plan(cap((1<<62,)))
    elif case=='address_overflow': n.rint_plan(cap((2,),data=(1<<64)-4))
    elif case=='scalar': print(n.rint_plan(cap(()))[1])
    elif case=='zero_size': n.rint_plan(cap((0,)))
    elif case=='conv_int32': n.conv1d_plan(cap((1,1,1<<31)),cap((1,1,1)),None,1,0,0,1,1)
    elif case=='lstm_int32': n.lstm_plan(cap((1<<31,1,1)),cap((1,4,1)),cap((1,4,1)),None,None,None,1,0)
    elif case=='same_capsule':
        a=cap((1,));o=cap((1,),data=0x2000)
        try: n.fp32_divide_probe(a,a,o)
        except ValueError as e: print(type(e).__name__,str(e),'name=',getname(a).decode());sys.exit(0)
    elif case=='empty_sync':
        cudart=C.CDLL('/usr/local/cuda-12.6/lib64/libcudart.so.12');d=C.c_int(-1);err=cudart.cudaGetDevice(C.byref(d))
        print('cudaGetDevice=',err,'device=',d.value)
        try: ts.synchronize()
        except RuntimeError as e: print('SAFE ERROR:',str(e));sys.exit(0)
        print('synchronize returned success')
        if err: sys.exit(9)
except (TypeError,ValueError) as e:
    print(type(e).__name__+': '+str(e));sys.exit(0)
print('COMPLETED')
