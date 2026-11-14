"""Execute actual forward/test source with ENG1 CPU arithmetic.
Prior art: Wengert (1964) reverse-mode tape, McKeeman (1998) differential testing,
ENG1 _E3Arithmetic; unverified -- lead to check. Ours: source-executing adapter and traces.
CPU emulation, not GPU validation; pow uses NumPy rather than CUDA powf.
"""
from pathlib import Path
from types import SimpleNamespace
import ast, importlib.util, json, math, sys, subprocess, difflib
import numpy as np
ROOT=Path.cwd(); OUT=ROOT/"artifacts/eng2/rev1"; TOOL=ROOT/"artifacts/eng2/tool"
sys.path.insert(0,str(ROOT/"tensor_cuda")); import tensor_cuda
spec=importlib.util.spec_from_file_location("eng2_review_tests",ROOT/"tensor_cuda/tests/test_eng2.py")
TEST=importlib.util.module_from_spec(spec);spec.loader.exec_module(TEST)
E3=TEST.E3; A=E3._E3Arithmetic
TRACE=[]; ACTIVE="candidate"; FUNCTIONS={}; HITS={}; CURRENT=None

def live(a):
    a=np.asarray(a); finite=a[np.isfinite(a)]
    if not finite.size and np.any(~np.isfinite(a)):
        return False  # Baseline all-nonfinite gradient: classified as a failure, never a numeric pass.
    assert finite.size and np.any(finite!=0) and np.ptp(finite)>0,(a.shape,"vacuous array")
    return True

def log(op,t,*args): TRACE.append([op,t.dtype,list(t.shape),*args])
class T:
    def __init__(self,a,dtype="float32",requires_grad=True): self.a=a if isinstance(a,A) else A(a,dtype,requires_grad=requires_grad)
    @property
    def dtype(self): return self.a.dtype
    @property
    def shape(self): return self.a.value.shape
    @property
    def ndim(self): return len(self.shape)
    @property
    def grad(self): return None if self.a.grad is None else T(self.a.grad,self.dtype)
    def astype(self,dt):
        log("astype",self,dt)
        return self if self.dtype==dt else T(self.a.cast(dt))
    def mean(self,axes,keep=True): log("mean",self,list(axes),keep); return T(self.a.mean(tuple(axes)))
    def var(self,axes,keep=True):
        log("var",self,list(axes),keep)
        a=self.a-self.a.mean(tuple(axes));return T((a*a).mean(tuple(axes)))
    def reshape(self,shape): log("reshape",self,list(shape));return T(self.a.reshape(shape))
    def pow(self,p): log("pow",self,p);return T(self.a.pow(p))
    def detach(self): log("detach",self);return T(self.a.value,self.dtype,False)
    def numpy(self):
        log("numpy",self)
        return self.a.value.astype(np.float16 if self.dtype=="float16" else np.float32)
    def __add__(self,o):
        log("add",self,o.dtype if isinstance(o,T) else o)
        return T(self.a+o.a if isinstance(o,T) else self.a.scalar(o))
    def __sub__(self,o): log("sub",self,o.dtype);return T(self.a-o.a)
    def __mul__(self,o):
        log("mul",self,o.dtype if isinstance(o,T) else o)
        return T(self.a*o.a if isinstance(o,T) else self.a.scalar(o,True))
    def sum(self): log("sum",self);return T(self.a.sum(tuple(range(self.ndim))))
    def backward(self): log("backward",self);self.a.backward()

def tensor(a,dtype="float32",**kw):
    t=T(a,dtype);log("tc.tensor",t);return t

def source(arm):
    if arm=="baseline":return subprocess.check_output(["git","--no-optional-locks","show","bc02d3a1:tensor_cuda/tensor_cuda/nn.py"],text=True)
    return (ROOT/"tensor_cuda/tensor_cuda/nn.py" if arm=="candidate" else TOOL/"arms"/arm/"tensor_cuda/tensor_cuda/nn.py").read_text()

def load(arm):
    src=source(arm); nodes=ast.parse(src)
    path=str(ROOT/"tensor_cuda/tensor_cuda/nn.py") if arm=="candidate" else ("baseline:nn.py" if arm=="baseline" else str(TOOL/"arms"/arm/"tensor_cuda/tensor_cuda/nn.py"))
    result={}
    for n in nodes.body:
        if isinstance(n,ast.ClassDef) and n.name in TEST.N1_CLASSES:
            f=next(f for f in n.body if isinstance(f,ast.FunctionDef) and f.name=="forward")
            ns=dict(np=np,math=math,tc=SimpleNamespace(tensor=tensor))
            exec(compile(ast.Module(body=[f],type_ignores=[]),path,"exec"),ns)
            result[n.name]=ns["forward"]
    FUNCTIONS[arm]=result

class Layer:
    def __init__(self,cls,dt,train=True):
        self.cls=cls;self.C,_=E3._affine_shape(cls,E3.E3_SHAPE[cls]);self.G=2;self.eps=E3.E3_EPS[cls];self.training=train;self.momentum=.1
        self.weight=T(1+.1*np.arange(self.C),dt,False)
        if cls!="RMSNorm":self.bias=T(.05*np.arange(self.C)-.1,dt,False)
        self.running_mean=T(np.zeros(self.C),"float32",False);self.running_var=T(np.ones(self.C),"float32",False)
    def __call__(self,x):return FUNCTIONS[ACTIVE][self.cls](self,x)

def layer(cls,dt):
    l=Layer(cls,dt);return l,l.weight.a.value,None if cls=="RMSNorm" else l.bias.a.value
E3._e3_layer=layer;E3._leaf=lambda v,d:T(v,d);E3._const=lambda v,d:T(v,d,False);E3._np=lambda t:t.numpy().astype(np.float64)
TEST._record=lambda *args,**kwargs:None
# The layer and test functions above are actual product/test source; only primitives are emulated.
for arm in ["baseline","candidate"]:load(arm)
traces={};deltas=[]
for cls in TEST.N1_CLASSES:
    for mode in ([True,False] if cls=="BatchNorm1D" else [True]):
        key=cls+("-train" if mode else "-eval")
        traces[key]={};outputs={}
        for arm in ["baseline","candidate"]:
            ACTIVE=arm;l=Layer(cls,"float32",mode);x=T(E3.e3_inputs(cls,"regular"));TRACE.clear();y=l(x)
            traces[key][arm]=list(TRACE);outputs[arm]=y.a.value;live(y.a.value)
        diff=list(difflib.unified_diff([json.dumps(t)+"\n" for t in traces[key]["baseline"]],[json.dumps(t)+"\n" for t in traces[key]["candidate"]],fromfile=key+":baseline",tofile=key+":candidate"))
        deltas+=diff;assert not diff,key;assert np.array_equal(outputs["baseline"].view(np.uint64),outputs["candidate"].view(np.uint64))
(OUT/"float32_op_sequences.json").write_text(json.dumps(traces,indent=2)+"\n");(OUT/"float32_op_sequences.diff").write_text("".join(deltas))

def execute_suite(arm):
    global ACTIVE
    ACTIVE=arm;results={};P="tensor_cuda/tests/test_eng2.py::"
    def call(name,fn,args):
        try:
            with np.errstate(over="ignore",invalid="ignore",under="ignore"):fn(*args)
            results[P+name]=dict(passed=True)
        except AssertionError as e:results[P+name]=dict(passed=False,error=str(e)[:200])
    for cls in TEST.N1_CLASSES:
        for case in TEST.N1_CASES:
            for direction in TEST.N1_DIRECTIONS:
                call(f"test_n1_float32_bits[{cls}-{case}-{direction}]",TEST.test_n1_float32_bits,(cls,case,direction,None,OUT))
                for dt in TEST.N1_DTYPES:
                    # L19 before every numeric comparison: inspect both computed source results and float64 reference.
                    l,w,b=layer(cls,dt);x=T(E3.e3_inputs(cls,case),dt);g=T(E3._e3_pattern(x.shape,1.1),dt,False)
                    with np.errstate(over="ignore",invalid="ignore",under="ignore"):
                        y=l(x);(y*g).sum().backward()
                    for a in [y.a.value,x.a.grad,*E3.e3_reference(cls,x.a.value,w,b,g.a.value)]:live(a)
                    call(f"test_n1_half_vs_reference[{cls}-{dt}-{case}-{direction}]",TEST.test_n1_half_vs_reference,(cls,dt,case,direction,None,OUT))
                    call(f"test_n1_half_rule_bits[{cls}-{dt}-{case}-{direction}]",TEST.test_n1_half_rule_bits,(cls,dt,case,direction))
    for dt in ["float32",*TEST.N1_DTYPES]:
        for case in TEST.N1_CASES:call(f"test_n1_batchnorm_running_stats[{dt}-{case}]",TEST.test_n1_batchnorm_running_stats,(dt,case))
    return results
results={a:execute_suite(a) for a in ["baseline","candidate"]}
expected=json.loads((TOOL.parent/"N1_EMULATION.json").read_text())["predicted_failures"]
for arm,r in results.items():
    failed=sorted(k for k,v in r.items() if not v["passed"])
    print(arm,"actual-source CPU emulation",len(r),"IDs",len(failed),"failures", "matches_registration",failed==expected[arm])
    assert failed==expected[arm],dict(arm=arm,unexpected=sorted(set(failed)-set(expected[arm])),missing=sorted(set(expected[arm])-set(failed)))
(OUT/"source_emulation.json").write_text(json.dumps(results,indent=2)+"\n")
# Two further spot checks execute a registered killer and trace its actual mutated line.
registry={m["id"]:m for m in json.loads((TOOL.parent/"mutants/mutants.json").read_text())}
mutant_results={}
for arm,cls,dt,case,direction in [("MN2_RMS","RMSNorm","bfloat16","near_eps","forward"),("MN3_BN","BatchNorm1D","float16","near_eps","forward")]:
    load(arm);ACTIVE=arm;entry=registry[arm];file=str(TOOL/"arms"/arm/entry["file"]);hits=set()
    def trace(frame,event,arg):
        if frame.f_code.co_filename==file and event=="line" and entry["hunk_lines"][0]<=frame.f_lineno<=entry["hunk_lines"][1]:hits.add(frame.f_lineno)
        return trace
    nid=f"tensor_cuda/tests/test_eng2.py::test_n1_half_rule_bits[{cls}-{dt}-{case}-{direction}]"
    assert nid in entry["killers"]
    failed=False
    try:
        sys.settrace(trace);TEST.test_n1_half_rule_bits(cls,dt,case,direction)
    except AssertionError as e:failed=True;error=str(e)
    finally:sys.settrace(None)
    mutant_results[arm]=dict(killer=nid,failed=failed,hits=sorted(hits),hunk=entry["hunk_lines"],error=error[:600] if failed else None)
    assert failed and hits,arm
(OUT/"norm_mutant_traces.json").write_text(json.dumps(mutant_results,indent=2)+"\n");print(json.dumps(mutant_results,indent=2))
# Explicit gradient dtypes for input and affine parameters with both half formats.
ACTIVE="candidate"; gradients=[]
for cls in TEST.N1_CLASSES:
    for dt in TEST.N1_DTYPES:
        l=Layer(cls,dt);l.weight.a.requires_grad=True
        if cls!="RMSNorm":l.bias.a.requires_grad=True
        x=T(E3.e3_inputs(cls,"regular"),dt);g=T(E3._e3_pattern(x.shape,1.1),dt,False);y=l(x);(y*g).sum().backward()
        grads=[x.grad,l.weight.grad]+([] if cls=="RMSNorm" else [l.bias.grad])
        assert all(t is not None and t.dtype==dt for t in grads)
        gradients.append(dict(cls=cls,dtype=dt,gradient_dtypes=[t.dtype for t in grads]))
(OUT/"gradient_dtypes.json").write_text(json.dumps(gradients,indent=2)+"\n")
print("Float32 operation sequence diff: empty, 6 class/mode paths. Gradient dtypes: 10 rows correct.")
