"""Registered CPU-only driver: retain every rc/stdout/stderr. Never convert a failed witness into PASS.
Prior art: subprocess/timeout engineering; no novel algorithm.
"""
from pathlib import Path
import subprocess,os,json,sys,ast
root=Path(__file__).resolve().parents[3];out=root/'artifacts/p1t_rev6';bin=Path('/tmp/p1t_rev6_build')
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(bin)+':'+str(bin/'tc_signal'))
jobs=[]
for mode in ['query_once','multidevice_restore','prior_oom','inside_callback','deferred_exit','diagnostic_oom_query','diagnostic_oom_sync','diagnostic_oom_device']:
 jobs.append((mode,[sys.executable,str(out/'probes/release_trace_cpu.py'),mode],15))
for mode in [['arithmetic'],['casts_exit','bool','--control'],['casts_exit','index','--control'],['casts_exit','bool'],['casts_exit','index']]:
 jobs.append(('_'.join(mode),[sys.executable,str(out/'probes/core_host_cpu.py'),*mode],15))
for exe,mode in [('fix2_harness','ck_census'),('fix3_harness','accounting'),('cpu_native','oom_sync'),('cpu_native','throw_deleter')]:
 jobs.append((exe+'_'+mode,[str(bin/exe),mode],30))
jobs.append(('sync_lock',[sys.executable,str(out/'probes/sync_lock_inversion_cpu.py')],15))
# Earlier actual-extension operand census, path plumbing only. Keep original source immutable.
s=(root/'artifacts/p1t_rev4/probes/operand_census.py').read_text().replace('tensor_fork_voice_wt_p1trev4',root.name).replace('artifacts/p1t_rev4','artifacts/p1t_rev6')
(bin/'operand_census.py').write_text(s);jobs.append(('operand_census',[sys.executable,str(bin/'operand_census.py')],15))
rows=[]
for name,cmd,bound in jobs:
 try:
  p=subprocess.run(['timeout',str(bound),*cmd],cwd=root,env=env,capture_output=True,text=True,timeout=bound+5)
  row={'case':name,'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 except subprocess.TimeoutExpired as e:row={'case':name,'rc':124,'stdout':str(e.stdout),'stderr':str(e.stderr)}
 (root/'logs'/('p1t_rev6_'+name+'.log')).write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
 (out/'cpu_results.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(name,'rc='+str(row['rc']),(row['stdout'][-500:]+row['stderr'][-150:]).replace('\n',' | '),flush=True)
