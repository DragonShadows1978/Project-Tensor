"""Re-run REV1 d4_child_probe.py scenario in fresh CPU parents, both before and after environment binding.
Prior art: REV1 d4_child_probe.py; PSF subprocess environment/PYTHONPATH. No device APIs.
"""
import sys,os,json,subprocess,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[3];A=R/'artifacts/eng1/a1';T=R/'artifacts/eng1/tool'
code=r'''
import sys,os,json,subprocess,hashlib
from pathlib import Path
package,tool,out=sys.argv[1:];sys.path.insert(0,package)
import tensor_cuda as tc
# Exactly the review scenario: parent preload, plain python -B -c import at cwd=tool.
probe="import json, tensor_cuda as tc; print(json.dumps({'package':getattr(tc,'__file__',None),'engine':getattr(getattr(tc,'_C',None),'__file__',None)}))"
before=subprocess.run([sys.executable,'-B','-c',probe],cwd=tool,env=dict(os.environ),capture_output=True,text=True,timeout=60)
sys.path.insert(0,tool)
from arm_origin import bound_environment,preflight
os.environ.update(bound_environment(package))
module=Path(tc._C.__file__);pin=hashlib.sha256(module.read_bytes()).hexdigest()
origin=preflight(package,module.name,pin,tool,Path(out).with_name(Path(out).stem+'_origin.json'))
after=subprocess.run([sys.executable,'-B','-c',probe],cwd=tool,env=dict(os.environ),capture_output=True,text=True,timeout=60)
assert after.returncode==0,after.stderr
observed=json.loads(after.stdout)
assert Path(observed['engine']).resolve()==module.resolve()
assert hashlib.sha256(Path(observed['engine']).read_bytes()).hexdigest()==pin
assert Path(observed['package']).resolve()==Path(tc.__file__).resolve()
result=dict(parent_engine=str(module),pin=pin,before=dict(exit=before.returncode,stdout=before.stdout,stderr=before.stderr),after=dict(exit=after.returncode,stdout=after.stdout,stderr=after.stderr,sha256=pin),origin=origin)
Path(out).write_text(json.dumps(result,indent=2)+'\n')
'''
rows=[]
for arm in ('baseline','candidate'):
 out=A/('isolation_'+arm+'.json');start=datetime.datetime.now(datetime.timezone.utc).isoformat()
 p=subprocess.run(['timeout','150s',sys.executable,'-B','-c',code,str(T/'arms'/arm/'tensor_cuda'),str(T),str(out)],env=dict(os.environ),cwd=T,capture_output=True,text=True)
 row=dict(arm=arm,start=start,child_exited=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit=p.returncode,timeout=p.returncode in (124,137),stdout=p.stdout,stderr=p.stderr)
 rows.append(row);print(arm,p.returncode,p.stderr)
(A/'isolation_parent.json').write_text(json.dumps(rows,indent=2)+'\n')
assert all(x['exit']==0 for x in rows)
