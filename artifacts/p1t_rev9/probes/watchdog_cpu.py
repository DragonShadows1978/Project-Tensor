"""PASS iff actual watchdog handles start OOM, 1/3 lost stop-wakeups and failed join safely (exit1 otherwise).
Prior art: FIX8 L2 pthread interposer2026, C++17 thread lifetime, Mesa lost wakeup (Lampson/Redell1980).
Taken: unchanged interposer; ours: actual-source host stream delay, no GPU or torch needed.
"""
import os,subprocess,sys,json
s='''import ctypes,json,sys,time
import _tc_signal_rev9_host as n
mode,count=sys.argv[1],int(sys.argv[2]);shim=ctypes.CDLL('/tmp/p1t_rev9_build/watchdog_shim.so')
shim.shim_arm(count if mode=='lock' else 0, int(mode=='lock'), count if mode=='join' else 0, 50 if mode=='join' else 0)
t=time.monotonic();r=n.watchdog(mode=='oom');dt=time.monotonic()-t
v=(ctypes.c_longlong*16)();shim.shim_state(v)
print(json.dumps({'r':r,'dt':dt,'shim':list(v),'stats':n.stats()}))
'''
rows=[]
for mode,count in [('control',0),('lock',1),('lock',3),('join',1),('oom',0)]:
 r=subprocess.run(['timeout','10',sys.executable,'-c',s,mode,str(count)],capture_output=True,text=True,timeout=15,env=dict(os.environ,PYTHONPATH='/tmp/p1t_rev9_build',LD_PRELOAD='/tmp/p1t_rev9_build/watchdog_shim.so',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''))
 row={'mode':mode,'count':count,'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr};rows.append(row)
print(json.dumps(rows,indent=2))
for row in rows:
 assert row['rc']==0,row
 v=json.loads(row['stdout']);st=v['stats'];x=v['shim']
 assert v['r']==0 and v['dt']<1.0,v
 assert st['exit_marker_failures']==1,st
 if row['mode']=='oom':
  assert st['exit_device_wait_watchdogs']==0 and 'watchdog thread could not start' in row['stderr'],row
 else:
  assert st['exit_device_wait_watchdogs']==1 and x[0]==1 and x[12]==1,v
 if row['mode']=='lock':assert x[1]==row['count'] and x[2]==row['count'] and st['exit_device_wait_watchdog_lock_failures']==row['count'],v
 if row['mode']=='join':assert x[3]==1 and x[4]==1 and x[7]==0 and x[11]<x[10],v
