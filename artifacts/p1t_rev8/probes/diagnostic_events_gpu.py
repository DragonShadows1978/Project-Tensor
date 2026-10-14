"""PASS iff each one-shot marker error is reported by the real add-on; setup failure still gets a wait diagnostic.
Single GPU; exit 3 INCONCLUSIVE without CUDA. Lead only, never executed by REV8 seat.
Build shim under /tmp: g++ -shared -fPIC -std=c++17 -I/usr/local/cuda-12.6/include diagnostic_event_shim.cpp -ldl -o /tmp/p1t_rev8_event_shim.so
Run: LD_PRELOAD=/tmp/p1t_rev8_event_shim.so PYTHONPATH=<reviewed-tree>/tc_signal timeout 90 python diagnostic_events_gpu.py
Prior art: CUDA 12.6 event/error contracts, FIX7 L1 finite GPU delay (2026); actual product binary, no embedded implementation.
"""
import os,sys,subprocess,json
if '--child' not in sys.argv:
 import torch
 if not torch.cuda.is_available(): print('INCONCLUSIVE: no GPU');sys.exit(3)
 rows=[]
 for site in ['create','record','query','destroy']:
  r=subprocess.run(['timeout','20',sys.executable,__file__,'--child',site],capture_output=True,text=True,timeout=25)
  rows.append({'site':site,'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
 print(json.dumps(rows,indent=2))
 assert all(r['rc']==0 for r in rows),'diagnostic error reporting failed'
 for r in rows:
  if r['site'] in ['create','record']:assert 'interpreter exit is waiting' in r['stderr'],'failed marker setup fell back to a silent wait'
else:
 import ctypes,time,torch,tc_signal as ts
 if not torch.cuda.is_available():sys.exit(3)
 lib=ctypes.CDLL(None)
 if not hasattr(lib,'rev8_arm'):raise RuntimeError('LD_PRELOAD shim required')
 torch.cuda.init()
 # Explicit legacy stream, finite work. About 0.4 seconds is bounded; no infinite kernel.
 torch.cuda.synchronize()
 with torch.cuda.stream(torch.cuda.ExternalStream(0)):
  t0=time.perf_counter();torch.cuda._sleep(50_000_000);torch.cuda.synchronize()
  rate=50_000_000/(time.perf_counter()-t0)
  torch.cuda._sleep(int(rate*0.4))
 _RELEASE_EXIT_DIAG_S=0.01
 api={'create':'cudaEventCreateWithFlags','record':'cudaEventRecord','query':'cudaEventQuery','destroy':'cudaEventDestroy'}
 which=sys.argv[2];messages=[]
 def hook():
  try:ts.synchronize()
  except RuntimeError as e:messages.append(str(e))
 lib.rev8_arm(['create','record','query','destroy'].index(which)+1)
 ts._C._exit_hook_runner(hook)
 print('RESULT '+json.dumps({'site':which,'messages':messages}),flush=True)
 assert any(api[which] in m for m in messages),'one-shot marker failure erased'
