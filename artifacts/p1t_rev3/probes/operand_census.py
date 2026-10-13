"""PASS iff metadata/ownership census checks cleanly accept/refuse with once-only deleters, no CUDA execution.
Prior art: DLPack2017/v1.3 and Python array API2023; ours: host boundary census on actual rebuilt extension.
"""
import ctypes as C, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tc_signal/tests'))
from tcs_producer import DLManagedTensor,DLManagedTensorVersioned,DLDevice,DLDataType,DLPackVersion,DELETER
import tc_signal as ts
n=ts._C
keep=[];calls=[];rows=[]
new=C.pythonapi['PyCapsule_New'];new.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p];new.restype=C.py_object
@DELETER
def delete(p):calls.append(p)
def cap(shape,strides=None,offset=0,data=0x1000,flags=0,versioned=False):
    sh=(C.c_int64*len(shape))(*shape);st=(C.c_int64*len(shape))(*strides) if strides is not None else None
    m=DLManagedTensorVersioned() if versioned else DLManagedTensor();t=m.dl_tensor
    t.data=data;t.device=DLDevice(2,0);t.dtype=DLDataType(2,32,1);t.ndim=len(shape);t.shape=sh;t.strides=st;t.byte_offset=offset
    m.deleter=delete
    if versioned:m.version=DLPackVersion(1,3);m.flags=flags
    keep.extend([sh,st,m]);return new(C.addressof(m),b'dltensor_versioned' if versioned else b'dltensor',None)
def check(label,fn,expected=None):
    try:r=fn()
    except Exception as e:
        assert expected and isinstance(e,expected),(label,type(e).__name__,str(e))
        rows.append({'case':label,'result':'refused','exception':type(e).__name__,'message':str(e)})
    else:
        assert expected is None,label
        rows.append({'case':label,'result':'accepted'})
check('offset accepted',lambda:n.rint_plan(cap((3,),offset=4)))
check('offset alignment refused',lambda:n.rint_plan(cap((3,),offset=2)),ValueError)
check('singleton negative stride',lambda:n.rint_plan(cap((1,3,1),strides=(-9223372036854775808,1,777))))
check('scalar NULL shape',lambda:n.rint_plan(cap(())))
for flags in [1,2,3]:
 check('versioned read flags '+str(flags),lambda:n.rint_plan(cap((3,),flags=flags,versioned=True)))
 before=len(calls)
 check('versioned written flags '+str(flags),lambda:n.rint_run(cap((3,)),cap((3,),data=0x2000,flags=flags,versioned=True)),ValueError)
 assert len(calls)-before==2
c=cap((1,));out=cap((1,),data=0x2000);before=len(calls)
check('one capsule twice',lambda:n.fp32_divide_probe(c,c,out),ValueError)
assert len(calls)-before==1
check('used capsule again',lambda:n.rint_plan(c),ValueError)
for shape in [(0,1<<62,4),(1<<62,0,4),(1<<62,4,0),(0,(1<<63)-1),((1<<63)-1,0)]:
 check('overflow next to zero '+str(shape),lambda s=shape:n.rint_plan(cap(s)),ValueError)
assert len(calls)==len(set(calls)) # each consumed manager invoked once
(ROOT/'artifacts/p1t_rev3/operand_census.json').write_text(json.dumps({'cases':rows,'deleter_calls':len(calls),'all_once':True},indent=2)+'\n')
print('PASS operand census',len(rows),'cases,',len(calls),'once-only consumed deleters')
