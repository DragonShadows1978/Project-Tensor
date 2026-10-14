"""PASS iff prelaunch cudaErrorInvalidDevice survives failed recycle/destroy plus report OOM (exit1 otherwise).
Prior art: CUDA12.6 per-thread last-error slot (NVIDIA2024), SQLite OOM injection; ours: compound failure scheduling.
Includes current product dlpack_operand.cpp via companion C++; never runs CUDA.
"""
import os,subprocess,sys,json
from pathlib import Path
script='''import json,sys
import _tc_signal_rev9_host as n
r=n.compound(sys.argv[1]=='oom')
errors=[]
for _ in range(2):
 try:n.synchronize()
 except Exception as e:errors.append(str(e))
print(json.dumps({'result':r,'later_errors':errors}))
'''
rows=[]
for arm in ['plain','oom']:
 r=subprocess.run(['timeout','10',sys.executable,'-c',script,arm],capture_output=True,text=True,env=dict(os.environ,PYTHONPATH='/tmp/p1t_rev9_build',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''),timeout=15)
 rows.append({'arm':arm,'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
print(json.dumps(rows,indent=2))
assert all(r['rc']==0 for r in rows),rows
results=[json.loads(r['stdout']) for r in rows]
assert 'cudaErrorInvalidDevice' in results[0]['result'][1],results
assert any('cudaErrorInvalidDevice' in e for e in results[1]['later_errors']), 'pending cudaErrorInvalidDevice erased before its OOM report exists'
