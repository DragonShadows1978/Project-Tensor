"""PASS iff real add-on plan/refusal entry points honor descriptor boundaries without launches.
Prior art: DLPack v1.3, FIX1/2 metadata census, C++17 checked arithmetic. No copied product implementation.
"""
import sys,ctypes as C,json
from pathlib import Path
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'tc_signal/tests'))
import tcs_producer as P
from tc_signal import _tc_signal as n
import _p1t_rev7_core as core
keep=[];rows=[]
new=C.pythonapi['PyCapsule_New'];new.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p];new.restype=C.py_object
NAME_L=b'dltensor';NAME_V=b'dltensor_versioned'
def cap(shape,offset=0,flags=0,version=1,strides=None):
 m=P.DLManagedTensorVersioned();m.version=P.DLPackVersion(version,3);m.flags=flags
 sh=(C.c_int64*len(shape))(*shape);st=(C.c_int64*len(shape))(*strides) if strides is not None else None
 t=m.dl_tensor;t.data=0x1000;t.device=P.DLDevice(2,0);t.dtype=P.DLDataType(2,32,1);t.ndim=len(shape);t.shape=sh if shape else C.POINTER(C.c_int64)();t.strides=st;t.byte_offset=offset
 keep.append((m,sh,st));return new(C.addressof(m),NAME_V,None)
def check(name,fn,exception=None):
 try:r=fn()
 except Exception as e:
  assert exception is not None and isinstance(e,exception),(name,e)
  rows.append({'case':name,'result':type(e).__name__,'message':str(e)});return
 assert exception is None,name
 rows.append({'case':name,'result':'accepted'})
check('addon_scalar',lambda: n.rint_plan(cap(())))
check('byte_offset',lambda:n.rint_plan(cap((3,),offset=4)))
check('read_only_input',lambda:n.rint_plan(cap((3,),flags=1)))
check('read_only_output',lambda:n.rint_run(cap((3,)),cap((3,),flags=1)),ValueError)
check('copied_output',lambda:n.rint_run(cap((3,)),cap((3,),flags=2)),ValueError)
check('zero_extent',lambda:n.rint_plan(cap((0,))),ValueError)
check('zero_plus_huge',lambda:n.rint_plan(cap((0,1<<62,4))),ValueError)
check('near_int64_byte_overflow',lambda:n.rint_plan(cap(((1<<63)-1,))),ValueError)
check('pointer_offset_wrap',lambda:n.rint_plan(cap((1,),offset=(1<<64)-1)),ValueError)
check('singleton_arbitrary_stride',lambda:n.rint_plan(cap((1,3),strides=(-7,1))))
check('wrong_major_version',lambda:n.rint_plan(cap((3,),version=2)),ValueError)
c=cap((3,));check('same_capsule_twice',lambda:n.rint_run(c,c),ValueError)
check('reuse_consumed_capsule',lambda:n.rint_plan(c),ValueError)
check('core_scalar_cpu_empty',lambda:core.empty([], 'float32','cpu'))
check('core_zero_extent',lambda:core.empty([0,1<<60],'float32','cpu'))
check('core_zero_plus_huge',lambda:core.empty([0,1<<62,4],'float32','cpu'),ValueError)
check('fill_grid_overflow',lambda:core.fill_blocks((1<<40)+1),ValueError)
(root/'artifacts/p1t_rev9/operand_census.json').write_text(json.dumps(rows,indent=2)+'\n')
print('PASS',len(rows),'actual-source CPU descriptor checks')
