"""APA-P1 staged checks. Prior art: AB-1 raw-bit views/interop (2026),
Numba CUDA Array Interface (2018), pybind11 (2016), CUDA memcpy (NVIDIA).
Taken: exact bit/storage inspection; no new kernel or selection algorithm.
"""
from pathlib import Path
import os,sys,json,hashlib,shutil,ctypes as ct
ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'artifacts/apa_p1_a1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def disk():
    for p,n in [(ROOT,25_000_000_000),(Path('/'),20_000_000_000)]:
        assert shutil.disk_usage(p).free>=n,('DISK RAIL',str(p),shutil.disk_usage(p).free,n)
def lead_gpu():
    assert os.environ.get('APA_P1_A1_LEAD_GPU')=='1','GPU worker must run in the staged lead line'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0','staged line must select only device 0'
    disk()
def runtime(arm):
    lead_gpu(); b=load(ART/'build.json'); exp=load(ART/'EXPECTED.json')
    for p,h in exp['code_sha256'].items():assert sha(ROOT/p)==h,('staged source drift',p)
    if arm=='lab':
        package=ART/'refs/lab'; module=ROOT/b['lab_module']; interop=Path(exp['lab_interop']['path']); want=b['lab_sha256']
    elif arm=='port':
        package=ART/'refs/port';module=ROOT/b['module'];interop=ROOT/load(ART/'interop_build.json')['module'];want=b['module_sha256']
    elif arm=='production':
        package=ART/'refs/production';module=package/'tensor_cuda'/Path(b['production_module']).name;interop=None;want=b['production_sha256']
    else:raise ValueError(arm)
    assert sha(module)==want
    sys.path[:0]=[str(package),str(module.parent)]
    import tensor_cuda as tc
    assert Path(tc._C.__file__).resolve()==module.resolve(),tc._C.__file__
    info=None
    if interop:
        assert sha(interop)==(exp['lab_interop']['sha256'] if arm=='lab' else load(ART/'interop_build.json')['sha256'])
        sys.path.insert(0,str(interop.parent));import _apa_ab1_interop as info
    return tc,info,dict(arm=arm,module=os.path.relpath(module,ROOT),sha256=want,python=os.path.relpath(Path(tc.__file__).resolve(),ROOT))
def raw(tc,info,t):
    import numpy as np
    address,shape,dt,nbytes,base,pooled,is_cuda,device,offset=info.info(t)
    dtype={0:np.uint32,1:np.uint16,2:np.uint64,3:np.uint8,4:np.uint8,5:np.uint16}[dt]
    out=np.empty(tuple(shape),dtype=dtype)
    if is_cuda:
        tc.synchronize()
        lib=ct.CDLL('/usr/local/cuda-12.6/lib64/libcudart.so');fn=lib.cudaMemcpy
        fn.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_int];fn.restype=ct.c_int
        code=fn(out.ctypes.data,int(address),out.nbytes,2);assert code==0,('cudaMemcpy',code)
    else:ct.memmove(out.ctypes.data,int(address),out.nbytes)
    return out
