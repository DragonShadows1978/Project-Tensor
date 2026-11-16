"""G3 every inherited E2 device path, full-width unsigned storage/export views.
Prior art: inherited ENG3 devpath operand census (2026), NumPy views (2006),
review mixed-corpus L19 (2026). Taken: same shape/seed/members; ours: store
logical dtype/shape, group real output corpora by unsigned width before identity.
"""
import ctypes
import json
import os
from pathlib import Path
import sys

import numpy as np
import tensor_cuda as tc

ROOT=Path(__file__).resolve().parents[3]
FLOATS=("float32","float16","bfloat16")
ALL=FLOATS+("uint8","int64","bool")
WIDTH={"float32":4,"float16":2,"bfloat16":2,"uint8":1,"int64":8,"bool":1}
SHAPE=(3,5)
# Load inherited operand generator without running its main. It imports the
# already pinned tc module; it never allocates a device tensor at import.
import runpy
inherited=runpy.run_path(str(ROOT/"scripts/eng3_devpath.py"),run_name="eng3f1_inherited")


def main(label):
    out={};meta={}
    def store(key,t,export=False):
        if export:
            a=np.ascontiguousarray(t.numpy())
            out[key]=a.view(np.dtype("u"+str(a.dtype.itemsize))).ravel()
            meta[key]={"tensor_dtype":t.dtype,"tensor_shape":list(t.shape),"export_dtype":str(a.dtype),"export_shape":list(a.shape)}
        else:
            h=t if t.device=="cpu" else tc._C._core_to(t,"cpu")
            n=int(np.prod(h.shape))*WIDTH[h.dtype]
            raw=ctypes.string_at(h.data_ptr(),n)
            out[key]=np.frombuffer(raw,np.dtype("u"+str(WIDTH[h.dtype]))).copy()
            meta[key]={"tensor_dtype":t.dtype,"tensor_shape":list(t.shape)}
    src={dt:inherited["data"](dt) for dt in ALL}
    C=tc._C
    for dt in ALL:
        d=tc.tensor(src[dt],device="cuda",dtype=dt)
        h=tc.tensor(src[dt],device="cpu",dtype=dt)
        store("from_host+numpy/"+dt,d,True)
        store("from_host+raw/"+dt,d)
        store("zeros/"+dt,C._core_zeros(SHAPE,dt,"cuda"))
        if dt in FLOATS:
            for v in (1.5, -0.0, 65520.0, float("nan"), 1e-8, 3.0e38):
                store(f"full/{dt}/{v}",C._core_full(SHAPE,v,dt,"cuda"))
        store("clone/"+dt,C._core_clone(d))
        store("to_d2d/"+dt,C._core_to(d,"cuda"))
        store("to_h2d/"+dt,C._core_to(h,"cuda"))
        store("to_d2h/"+dt,C._core_to(d,"cpu"))
        for dst in FLOATS+("uint8",):
            if dt in ("int64","bool") and dst!=dt:
                continue
            t=d.astype(dst)
            store(f"astype_raw/{dt}->{dst}",t)
            store(f"astype_numpy/{dt}->{dst}",t,True)
    for dt in FLOATS:
        x=tc.tensor(src[dt],device="cuda",dtype=dt).detach();x.requires_grad=True;x.backward()
        store("backward_full_clone/"+dt,x.grad)
    tc.synchronize()
    dest=ROOT/"artifacts/eng3f1/g3"
    dest.mkdir(exist_ok=True)
    np.savez(dest/(label+".npz"),**out)
    (dest/(label+".meta.json")).write_text(json.dumps(meta,indent=2))
    return 0

if __name__=="__main__":
    sys.exit(main(sys.argv[1]))
