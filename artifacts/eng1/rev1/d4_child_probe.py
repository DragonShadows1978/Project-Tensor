# Prior art: Python process-local sys.modules cache (PSF). Taken: ordinary import origin census.
import sys,subprocess,json,os
from pathlib import Path
root=Path.cwd(); rev=root/'artifacts/eng1/rev1'; tool=root/'artifacts/eng1/tool'
sys.path.insert(0,str(tool/'arms/baseline/tensor_cuda'))
import tensor_cuda as tc
code="import json, tensor_cuda as tc; print(json.dumps({'package':getattr(tc,'__file__',None),'engine':getattr(getattr(tc,'_C',None),'__file__',None)}))"
p=subprocess.run([sys.executable,'-B','-c',code],cwd=tool,env=dict(os.environ),capture_output=True,text=True,timeout=60)
r=dict(parent_engine=tc._C.__file__,child_exit=p.returncode,child_stdout=p.stdout,child_stderr=p.stderr)
(rev/'d4_child.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
