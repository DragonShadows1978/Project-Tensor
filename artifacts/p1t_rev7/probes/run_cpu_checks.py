"""PASS iff all registered finite CPU assertions exit0; individual receipts retained. No GPU execution.
Prior art: REV6 actual-source trace/UBSan workflow, DLPack2017, CPython3.12, CUDA12.6; no new product algorithm.
"""
from pathlib import Path
import os,sys,subprocess,json
root=Path(__file__).resolve().parents[3]; b=Path('/tmp/p1t_rev7_build'); out=root/'artifacts/p1t_rev7'
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(b)+':'+str(b/'tc_signal'))
jobs=[]
for m in ['query_once','multidevice_restore','prior_oom','inside_callback','deferred_exit','diagnostic_oom_query','diagnostic_oom_sync','diagnostic_oom_device']:
    jobs.append((m,[sys.executable,str(root/'artifacts/p1t_rev6/probes/release_trace_cpu.py'),m],15))
for exe,mode in [('fix2_harness','ck_census'),('fix3_harness','accounting'),('cpu_native','oom_sync'),('cpu_native','throw_deleter')]:
    jobs.append((exe+'_'+mode,[str(b/exe),mode],30))
jobs.append(('sync_lock',[sys.executable,str(root/'artifacts/p1t_rev6/probes/sync_lock_inversion_cpu.py')],15))
jobs.append(('sync_late_release',[sys.executable,str(root/'artifacts/p1t_rev6/probes/sync_late_release_cpu.py')],15))
rows=[]
for name,cmd,bound in jobs:
    p=subprocess.run(['timeout',str(bound),*cmd],cwd=root,env=env,capture_output=True,text=True,timeout=bound+5)
    row={'case':name,'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr}; rows.append(row)
    (root/'logs'/('p1t_rev7_'+name+'.log')).write_text(json.dumps(row,indent=2)+'\n')
    (out/'cpu_results.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(name,'rc='+str(p.returncode),flush=True)
assert all(r['rc']==0 for r in rows),rows
