# Prior art: Zhang/Sennrich (2019) RMSNorm; reverse-mode computational graphs (Linnainmaa, 1970);
# IEEE754 and Higham (2002) overflow/range bounds. Taken: ordinary chain rule, not a new backward algorithm.
import sys,json,importlib.util
from pathlib import Path
import numpy as np
root=Path.cwd();rev=root/'artifacts/eng1/rev1'
sys.path.insert(0,str(root/'artifacts/eng1/tool/arms/candidate/tensor_cuda'))
s=importlib.util.spec_from_file_location('eng1',root/'tensor_cuda/tests/test_eng1.py');T=importlib.util.module_from_spec(s);s.loader.exec_module(T)
q=lambda a:np.asarray(a,np.float32).astype(np.float16).astype(np.float64)
x=q(T.e3_inputs('RMSNorm','near_eps'));g=q(T._e3_pattern(x.shape,1.1));w=q(1+.1*np.arange(16))
sq=q(x*x);ms=q(q(sq.sum(axis=-1,keepdims=True))/16);a=q(ms+float(np.float32(1e-6)))
# y=(x*inv)*w. inv's gradient reduces the broadcast multiply's input-dtype contributions.
g_times_w=q(g*w);inv_grad=q(q(g_times_w*x).sum(axis=-1,keepdims=True))
wide=np.asarray(inv_grad,np.float32)*np.asarray(np.float32(-.5)*np.power(np.asarray(a,np.float32),np.float32(-1.5)),np.float32)
with np.errstate(over='ignore'): cast=q(wide)
_,ref=T.e3_reference('RMSNorm',x,w,None,g)
r=dict(a=a.tolist(),inv_grad=inv_grad.tolist(),wide_pow_gradient=wide.tolist(),
       abs_wide_min=float(abs(wide).min()),abs_wide_max=float(abs(wide).max()),half_max=65504,
       cast_nonfinite=int((~np.isfinite(cast)).sum()),reference_dx_finite=bool(np.all(np.isfinite(ref))),
       reference_dx_max=float(abs(ref).max()),
       honest_error_bound="Unbounded/infinite on this graph: a mathematically correct half cast overflows before later scaling by x and 1/16. E2 keeps its own intermediates wide, but still returns the requested half gradient. E3 intentionally measures, rather than fixes, such norm behavior.",
       scope="CPU IEEE emulation and source/range reasoning only; not a GPU result.")
assert r['abs_wide_min']>65504*1.001 and r['cast_nonfinite']==4 and r['reference_dx_finite']
(rev/'backward_bound.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
