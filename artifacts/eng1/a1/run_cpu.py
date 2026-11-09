"""Foreground CPU-only pytest receipts. Prior art: round1/review arm preload, pytest hooks and completed exits.
Taken: actual arm modules and actual ENG1 tests; ours: A1 cases and explicit after-exit parent receipt.
"""
import sys,os,json,subprocess,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[3];A=R/'artifacts/eng1/a1';T=R/'artifacts/eng1/tool'
sys.path.insert(0,str(T));from arm_origin import bound_environment,preflight
sha=__import__('hashlib').sha256
arm=sys.argv[1];selection=sys.argv[2] if len(sys.argv)>2 else 'cpu';tag=arm+('_ow_zero' if selection=='ow_zero' else '')+(sys.argv[3] if len(sys.argv)>3 else '')
package=T/'arms'/arm/'tensor_cuda';module=next((package/'tensor_cuda').glob('_tensor_cuda*.so'))
env=bound_environment(package);assert env['CUDA_VISIBLE_DEVICES']==''
preflight(package,module.name,sha(module.read_bytes()).hexdigest(),T,A/(tag+'_origin.json'),env)
code="""import sys,json,pytest;from pathlib import Path
sys.path.insert(0,sys.argv[1]);import tensor_cuda
class P:
 ids=[];reports=[]
 def pytest_collection_finish(self,session):self.ids=[i.nodeid for i in session.items]
 def pytest_runtest_logreport(self,report):self.reports.append(dict(id=report.nodeid,when=report.when,outcome=report.outcome,detail=str(report.longrepr) if report.failed else ''))
p=P();rc=int(pytest.main(sys.argv[3:],plugins=[p]));Path(sys.argv[2]).write_text(json.dumps(dict(exit=rc,collected_ids=p.ids,reports=p.reports),indent=2));raise SystemExit(rc)
"""
cmd=[sys.executable,'-B','-c',code,str(package),str(A/(tag+'_pytest.json')),'-q','-p','no:cacheprovider','--import-mode=importlib','--rootdir='+str(R),'--basetemp='+env['TMPDIR']+'/cpu_'+tag,'--junitxml='+str(A/(tag+'.xml')),'-k',selection,str(package/'tests/test_eng1.py')]
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
start=utc()
with (A/(tag+'.log')).open('x') as log:
 p=subprocess.run(['timeout','150s',*cmd],env=env,cwd=T,stdout=log,stderr=subprocess.STDOUT)
rec=dict(arm=arm,command=cmd,start=start,child_exited=utc(),exit=p.returncode,timeout=p.returncode in (124,137),selection=selection)
(A/(tag+'_parent.json')).write_text(json.dumps(rec,indent=2)+'\n')
print(json.dumps(rec));print((A/(tag+'.log')).read_text()[-1600:])
# Driver success means registered outcome, never a synthesized child pass.
raise SystemExit(0 if p.returncode==(1 if arm=='M16' else 0) else 1)
