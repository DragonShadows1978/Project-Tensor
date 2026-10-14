"""PASS iff a named failure arriving immediately after the raw snapshot retains its site/device/error at exit (exit1 otherwise).
Prior art: C++17 synchronized snapshots, POSIX pthreads2008; ours: real FIX8 record-writer/exit-hook scheduling.
Current-source core spans compiled by build_core_race_cpu.py, with actual Python hook via AST. No GPU execution.
"""
import json,os,subprocess,sys
from pathlib import Path
s='''import ast,ctypes,sys,types,json
from pathlib import Path
import _p1t_rev9_race as n
shim=ctypes.CDLL('/tmp/p1t_rev9_build/core_snapshot_shim.so');module=ctypes.CDLL(n.__file__)
shim.arm.argtypes=[ctypes.c_void_p]
p=Path(sys.argv[1])/'tensor_cuda/tensor_cuda/__init__.py';t=ast.parse(p.read_text());t.body=[x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='_dlpack_shutdown_hook']
ns={'_C':n,'sys':sys,'_DLPACK_SHUTDOWN_WAIT_S':0};exec(compile(t,str(p),'exec'),ns)
mode=sys.argv[2]
if mode=='unwritten':sys.stderr=None
shim.arm(ctypes.cast(module.p1t_rev9_race_record,ctypes.c_void_p))
n._dlpack_exit_hook_runner(ns['_dlpack_shutdown_hook'])
print(json.dumps({'fired':shim.was_fired(),'stats':n._dlpack_release_stats()}))
'''
rows=[]
for mode in ['written','unwritten']:
 r=subprocess.run(['timeout','15',sys.executable,'-c',s,str(Path.cwd()),mode],capture_output=True,text=True,timeout=20,env=dict(os.environ,PYTHONPATH='/tmp/p1t_rev9_build',LD_PRELOAD='/tmp/p1t_rev9_build/core_snapshot_shim.so',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
 rows.append({'mode':mode,'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
print(json.dumps(rows,indent=2))
assert all(r['rc']==0 and json.loads(r['stdout'])['fired']==1 for r in rows),rows
assert 'race probe: cudaStreamSynchronize failed' in rows[0]['stderr'],rows
assert 'race probe: cudaStreamSynchronize failed' in rows[1]['stderr'] and 'cudaErrorInvalidValue' in rows[1]['stderr'], 'acknowledged report summary uses the earlier empty snapshot and loses its known record'
