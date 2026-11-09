"""D5 native-imported Python guards without device allocation.
Prior art: PyTorch GroupNorm argument checks; boundary-value testing (Myers,
1979, unverified — lead to check). Uses tensor metadata only: no CUDA op is
reachable for these refused inputs. Mixed value paths are checked separately.
"""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
TOOL=Path(__file__).resolve().parent
ROOT=TOOL.parents[2]
assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
sys.path.insert(0,str(TOOL/"arms/candidate"))
import tensor_cuda as tc
from tensor_cuda import nn
assert Path(nn.__file__).resolve()==TOOL/"arms/candidate/tensor_cuda/nn.py"

rows=[]
def refuse(name,kind,word,fn):
    error=None
    try:fn()
    except Exception as exc:error=exc
    ok=isinstance(error,kind) and word in str(error)
    rows.append(dict(case=name,passed=ok,exception_class=type(error).__name__,message=str(error)))
for name in ("num_groups","num_channels"):
    for label,value in (("zero",0),("negative",-2),("float",2.0),("true",True),("false",False),("string","2"),("none",None)):
        args=dict(num_groups=2,num_channels=6);args[name]=value
        refuse(name+"/"+label,ValueError,name,lambda args=args:nn.GroupNorm(**args))
# Module initialization sets dictionaries only. Parameter tensor allocation is
# intentionally bypassed so these real forward guards can be tested on CPU.
layer=nn.GroupNorm.__new__(nn.GroupNorm)
nn.Module.__init__(layer)
layer.G,layer.C,layer.eps=2,6,1e-5
meta=lambda shape=(2,6),dtype="float32":SimpleNamespace(ndim=len(shape),shape=shape,dtype=dtype)
for name,shape in (("rank0",()),("rank1",(6,)),("channels",(2,4,3))):
    refuse(name,ValueError,"rank" if len(shape)<2 else "channels",lambda shape=shape:layer(meta(shape)))
for dt in ("int64","bool","uint8","float64","complex64","unknown"):
    for operand in ("input","weight","bias"):
        layer.weight,layer.bias=meta((6,)),meta((6,))
        x=meta(dtype=dt if operand=="input" else "float32")
        if operand!="input":setattr(layer,operand,meta((6,),dt))
        refuse(dt+"/"+operand,TypeError,dt,lambda:layer(x))
# A3 new decisions: metadata-only parameter guards and index normalization.
import numpy as np
for operand in ("weight", "bias"):
    for shape in ((),(2,3),(6,1),(1,6,1),(5,),(7,)):
        layer.weight,layer.bias=meta((6,)),meta((6,))
        setattr(layer,operand,meta(shape))
        refuse(operand+"/shape/"+str(shape),ValueError,operand,lambda:layer(meta()))
original_parameter=nn.parameter
nn.parameter=lambda data:meta(np.asarray(data).shape)
try:
    for name in ("num_groups","num_channels"):
        for kind in (np.int64,np.int32,np.uint8):
            args=dict(num_groups=2,num_channels=6);args[name]=kind(args[name])
            value=nn.GroupNorm(**args)
            rows.append(dict(case=name+"/"+kind.__name__,passed=type(value.G) is int and type(value.C) is int))
        for value in (np.bool_(True),np.bool_(False)):
            args=dict(num_groups=2,num_channels=6);args[name]=value
            refuse(name+"/numpy_bool/"+str(value),ValueError,"positive integer",lambda:nn.GroupNorm(**args))
finally:
    nn.parameter=original_parameter
result=dict(passed=all(r["passed"] for r in rows),cases=rows,engine_imported_from=str(tc._C.__file__),
            nn_imported_from=str(nn.__file__),CUDA_VISIBLE_DEVICES="",evidence_class="native class guards with metadata; no device allocation")
(TOOL/"native_validation.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(dict(passed=result["passed"],cases=len(rows))))
sys.exit(0 if result["passed"] else 1)
