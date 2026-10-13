"""PASS iff current native add-on runner and actual Python hook acknowledge only written reports (exit1 otherwise).
Prior art: CPython3.12 atexit and FIX8 acknowledged reports2026; taken: actual hook; ours: no-launch injected device-query failure.
No GPU: no op/launch, cuda:current_device hook bypasses the device query; CUDA_VISIBLE_DEVICES is empty.
"""
import json,os,subprocess,sys
s='''import tc_signal as ts,sys,json,atexit
C=ts._C;atexit.unregister(C._exit_hook_runner)
mode=sys.argv[1]
C._test_inject(['cuda:current_device'])
if mode=='none':sys.stderr=None
elif mode=='raises':
 class E:
  def write(self,*a):raise OSError('test writer failure')
  def flush(self):pass
 sys.stderr=E()
elif mode=='interrupt':
 class E:
  def write(self,*a):raise KeyboardInterrupt('test writer interrupt')
  def flush(self):pass
 sys.stderr=E()
elif mode=='closed':
 import os;os.close(2)
try:C._exit_hook_runner(ts._release_pending_at_exit)
except KeyboardInterrupt:pass
st=C.pending_stats();print(json.dumps({k:st[k] for k in ['exit_report_unwritten','exit_log_fallbacks','release_admission_closed','unreported_drain_failures']}))
'''
rows=[]
for mode in ['normal','none','raises','interrupt','closed']:
 r=subprocess.run(['timeout','15',sys.executable,'-c',s,mode],capture_output=True,text=True,timeout=20,env=dict(os.environ,PYTHONPATH='/tmp/p1t_rev9_build',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
 rows.append({'mode':mode,'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
print(json.dumps(rows,indent=2))
for r in rows:
 assert r['rc']==0,r
 st=json.loads(r['stdout']);assert st['release_admission_closed'] and not st['exit_report_unwritten'] and st['unreported_drain_failures']==0,st
 assert st['exit_log_fallbacks']==(0 if r['mode']=='normal' else 1),st
 if r['mode']=='normal':assert 'cudaGetDevice' in r['stderr'] and 'cudaErrorLaunchFailure' in r['stderr'],r
 elif r['mode']!='closed':assert '2 CUDA failure(s)' in r['stderr'] and 'the first: a CUDA error already pending' in r['stderr'] and 'cudaErrorNoDevice' in r['stderr'],r
 # On the actual no-GPU runtime, cudaPeekAtLastError first records cudaErrorNoDevice;
 # the injected device-query failure is the second record. The allowed summary names the first and counts both.
