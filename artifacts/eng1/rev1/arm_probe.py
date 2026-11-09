# Prior art: GN1 package preload through sys.path/sys.modules (Python import caching, PSF).
# Taken: exact staged preload. Ours: capture origins before/after collection on CPU.
import os,sys,json,hashlib,importlib.util
from pathlib import Path
import numpy as np
root=Path.cwd(); rev=root/'artifacts/eng1/rev1'; tool=root/'artifacts/eng1/tool'
arm=sys.argv[1]; package=tool/'arms'/arm/'tensor_cuda'
sys.path.insert(0,str(package)); sys.path.insert(1,str(root))
import tensor_cuda as tc
import pytest
seal=json.loads((tool/'staging_seal.json').read_text())
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert h(tc._C.__file__)==seal['engine_sha256'][arm]
pre={n:str(Path(m.__file__).resolve()) for n,m in list(sys.modules.items()) if n.startswith('tensor_cuda') and getattr(m,'__file__',None)}
# Import real test modules using pytest's regression collection. No test calls.
files=[str(root/p) for p in json.loads((tool/'regression_files.json').read_text())]
class Collected:
    def pytest_collection_finish(self,session): self.ids=[i.nodeid for i in session.items]
plug=Collected()
rc=int(pytest.main(['--collect-only','-q','-p','no:cacheprovider','--import-mode=prepend','--rootdir='+str(root),
                     '--basetemp='+os.environ['TMPDIR']+'/collect_'+arm,*files],plugins=[plug]))
post={n:str(Path(m.__file__).resolve()) for n,m in list(sys.modules.items()) if n.startswith('tensor_cuda') and getattr(m,'__file__',None)}
# Only zero-byte CPU objects; no device allocations.
checks={}
for target in ([0,-1],[-1,4],[0,2**62,4],[0,2**63-1],[-1,2**62,4]):
    try:
        t=tc.from_numpy(np.zeros((0,),np.float32),device='cpu').reshape(target)
        checks[str(target)]={'shape':list(t.shape),'dtype':t.dtype}
    except Exception as ex: checks[str(target)]={'error':type(ex).__name__,'message':str(ex)}
v=tc.tensor([],device='cpu',dtype='float64')
row=dict(arm=arm,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],module=tc._C.__file__,
         sha256=h(tc._C.__file__),pin=seal['engine_sha256'][arm],pre=pre,post=post,
         collection_exit=rc,ids=getattr(plug,'ids',[]),reshape=checks,float64_request_dtype=v.dtype)
(rev/('arm_'+arm+'.json')).write_text(json.dumps(row,indent=2)+'\n')
print('PROBE',json.dumps({k:v for k,v in row.items() if k not in ('ids','pre','post')}))
raise SystemExit(rc)
