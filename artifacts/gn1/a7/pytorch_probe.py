"""CPU PyTorch behavior census, external-reference observation only.
Prior art: PyTorch GroupNorm (Paszke et al. 2019; version captured live).
A2 requirements are authoritative even where PyTorch accepts other inputs.
"""
import json
import os
from pathlib import Path
assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
TOOL=Path(__file__).resolve().parent
out={}
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out["version"]=torch.__version__
    out["device"]="cpu"
    results={}
    def probe(name,fn):
        try:
            value=fn()
            results[name]=dict(outcome="accepted",result=str(value) if not isinstance(value,torch.Tensor) else dict(shape=list(value.shape),dtype=str(value.dtype)))
        except Exception as exc:
            results[name]=dict(outcome="refused",exception_class=type(exc).__name__,message=str(exc))
    probe("nondivisible",lambda:torch.nn.GroupNorm(4,6))
    for name in ("num_groups","num_channels"):
        for label,value in (("zero",0),("negative",-2),("float",2.0),("true",True),("false",False),("string","2"),("none",None)):
            args=dict(num_groups=2,num_channels=6);args[name]=value
            probe(name+"/"+label,lambda args=args:torch.nn.GroupNorm(**args))
    for name,shape in (("rank0",()),("rank1",(6,)),("channels",(2,4,3))):
        probe(name,lambda shape=shape:torch.nn.GroupNorm(2,6)(torch.zeros(shape)))
    for dtype in (torch.int64,torch.bool,torch.uint8):
        for operand in ("input","weight","bias"):
            def forward(dtype=dtype,operand=operand):
                layer=torch.nn.GroupNorm(2,6)
                x=torch.ones((2,6),dtype=dtype if operand=="input" else torch.float32)
                if operand!="input":setattr(layer,operand,torch.nn.Parameter(torch.ones(6,dtype=dtype),requires_grad=False))
                return layer(x)
            probe(str(dtype)+"/"+operand,forward)
    for wd,bd in ((torch.float32,torch.float16),(torch.float16,torch.float32)):
        def mixed(wd=wd,bd=bd):
            layer=torch.nn.GroupNorm(3,6)
            layer.weight=torch.nn.Parameter(torch.ones(6,dtype=wd))
            layer.bias=torch.nn.Parameter(torch.zeros(6,dtype=bd))
            return layer(torch.arange(48,dtype=torch.float32).reshape(2,6,4))
        probe("mixed/"+str(wd)+"/"+str(bd),mixed)
    import numpy as np
    for name in ("num_groups","num_channels"):
        for label,ctor in (("np_int64",np.int64),("np_int32",np.int32),("np_uint8",np.uint8),("np_bool_true",np.bool_)):
            args=dict(num_groups=2,num_channels=6);args[name]=ctor(args[name])
            def count_forward(args=args):
                layer=torch.nn.GroupNorm(**args)
                return layer(torch.arange(36,dtype=torch.float32).reshape(2,6,3))
            probe(name+"/"+label,count_forward)
    for operand in ("weight","bias"):
        for shape in ((),(2,3),(6,1),(1,6,1),(5,),(7,)):
            def bad_shape(operand=operand,shape=shape):
                layer=torch.nn.GroupNorm(2,6)
                setattr(layer,operand,torch.nn.Parameter(torch.ones(shape)))
                return layer(torch.ones((2,6,3)))
            probe(operand+"/shape/"+str(shape),bad_shape)
    out["results"]=results
    out["verified"]=True
except ImportError as exc:
    out=dict(verified=False,status="unverified — lead to check",reason=str(exc),search="PyTorch GroupNorm positive integer bool rank channel dtype mixed weight bias")
(TOOL/"pytorch_behavior.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
