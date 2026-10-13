"""PASS iff both core lock retry sites and add-on polling fallback diagnose and wait for finite callback completion.
Prior art: FIX7 LD_PRELOAD lock shim / exit staging (2026); C++17 mutex errors, CPython3.12 lifecycle.
Actual product source compiled into full add-on or source-hashed core host module. No GPU execution.
"""
from pathlib import Path
import ast,json,subprocess,sys,os
root=Path(__file__).resolve().parents[3];b=Path('/tmp/p1t_rev9_build');out=root/'artifacts/p1t_rev9';rows=[]
def assign(path,name,ns=None):
 t=ast.parse((root/path).read_text());node=next(n for n in t.body if isinstance(n,ast.Assign) and any(isinstance(k,ast.Name) and k.id==name for k in n.targets))
 return eval(compile(ast.Expression(node.value),str(path),'eval'),ns or {})
core=assign('tensor_cuda/tests/test_dlpack.py','_FIX7_LOCK_SHIM_C').replace('strncmp(base, "_tensor_cuda", 12)','strncmp(base, "_p1t_rev7_core", 14)')
addon=assign('tc_signal/tests/test_fix7_findings.py','_LOCK_SHIM_C')
for name,src in [('core_lock',core),('addon_lock',addon)]:
 (b/(name+'.c')).write_text(src)
 subprocess.run(['timeout','30','gcc','-shared','-fPIC','-O1','-o',str(b/(name+'.so')),str(b/(name+'.c')),'-ldl'],check=True)
# Source-linked core hook, actual core release lifecycle. A real GPU is unnecessary for the callback/lock wait.
script="""
import ast,atexit,ctypes,json,sys,threading,time,types,resource
from pathlib import Path
import _p1t_rev7_core as n
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
root=Path(sys.argv[1]); skip=int(sys.argv[2]); lib=ctypes.CDLL(None)
src=(root/'tensor_cuda/tensor_cuda/__init__.py').read_text();tree=ast.parse(src)
node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_dlpack_shutdown_hook')
ns={'_C':n,'sys':sys,'_DLPACK_SHUTDOWN_WAIT_S':0.05};exec(compile(ast.Module(body=[node],type_ignores=[]),str(root/'tensor_cuda/tensor_cuda/__init__.py'),'exec'),ns)
started=threading.Event()
def slow():
 print('DELETER started',flush=True);started.set();time.sleep(.6);print('DELETER finished',flush=True)
def start():
 threading.Thread(target=lambda:n.release(slow),daemon=True).start();assert started.wait(5)
def report():print('RESULT '+json.dumps({'stats':n._dlpack_release_stats(),'failed':lib.shim_lock_failed(),'left':lib.shim_lock_left()}),flush=True)
atexit.register(report);atexit.register(n._dlpack_exit_hook_runner,ns['_dlpack_shutdown_hook']);atexit.register(lib.shim_lock_arm_after,skip,15);atexit.register(start)
class F:
 def __init__(self):self.fin=sys.is_finalizing;self.sleep=time.sleep
 def __del__(self):print('FINALIZING',self.fin(),flush=True);self.sleep(.8)
h=types.ModuleType('_rev8_final');h.keep=F();sys.modules[h.__name__]=h
"""
base=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(b)+':'+str(b/'tc_signal'))
for name,args,src,lib in [('core_close',[str(root),'0'],script,'core_lock'),('core_relock',[str(root),'1'],script,'core_lock'),('addon_poll',['lock_fail','1'],assign('tc_signal/tests/test_fix7_findings.py','_EXIT_SCRIPT',{'TESTS_DIR':root/'tc_signal/tests'}),'addon_lock')]:
 env=dict(base,LD_PRELOAD=str(b/(lib+'.so')))
 r=subprocess.run(['timeout','15',sys.executable,'-c',src,*args],env=env,capture_output=True,text=True,timeout=20)
 row={'case':name,'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}; rows.append(row)
 (out/'lock_checks.json').write_text(json.dumps(rows,indent=2)+'\n')
 assert r.returncode==0,row
 assert 0<=r.stdout.find('DELETER finished')<r.stdout.find('FINALIZING True'),row
 res=json.loads(next(x[7:] for x in r.stdout.splitlines() if x.startswith('RESULT ')))
 if name.startswith('core'):
  assert res['failed']==15 and res['left']==0 and res['stats']['shutdown_lock_retries']==15,row
  assert res['stats']['shutdown_lock_diagnostics']>=1 and 'interpreter exit cannot take' in r.stderr,row
 else:
  assert res['lock']['failed']==1 and res['after']['exit_wait_poll_diagnostics']>=1,row
  assert 'polling every' in r.stderr,row
 print('PASS',name,flush=True)
