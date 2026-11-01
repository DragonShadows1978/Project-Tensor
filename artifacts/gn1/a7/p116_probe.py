import os
os.environ["GN1_CONTROL"]="1"
"""P116 direct PID observation supplement. Prior art: REV4 process-liveness
probe (2026), Meszaros fault injection (2007), Linux procfs state inspection.
Same registered N2b/N2c targets; no changed predicate or product/tool bytes.
"""
import datetime,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path.cwd();T=ROOT/'artifacts/gn1/a7';sys.path.insert(0,str(T))
from control_transport import adapt
from evidence import read
assert os.environ['CUDA_VISIBLE_DEVICES']==''
base=T/'p116_runs';base.mkdir(exist_ok=False)
scratch=Path(os.environ['TMPDIR'])/'p116';scratch.mkdir(exist_ok=False)
rows=[]
def run(argv,e):
 with (base/'LAUNCHES.jsonl').open('a') as f:f.write(json.dumps(dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),argv=argv,RUNS=e['RUNS']))+'\n')
 r=subprocess.run(['timeout','--kill-after=2s','60s',*argv],cwd=T,env=e,capture_output=True,text=True)
 return dict(exit=r.returncode,stdout=r.stdout[-1000:],stderr=r.stderr[-2000:])
for name,mode in [('N2b_D3_PID','marker_only_sleep'),('N2c_D3_PID','ignore_term')]:
 runs=base/name
 e=dict(os.environ,RUNS=str(runs),FORGE_GPU_SUMMARY=str(scratch/'summary.log'),FORGE_GPU_WINDOW_FILE=str(scratch/'GPU_WINDOW'),FORGE_GPU_LOCK=str(scratch/'lock'),FORGE_GPU_KILL_GRACE='10',FORGE_GPU_POLL='5',REV2_NO_LOG='1',REV2_MODE=mode,REV4_MA4=str(T),REV3_SLEEP='60')
 e.pop('GN1_AUDIT',None)
 assert run([sys.executable,'-B','start_gpu_run.py'],e)['exit']==0
 argv=[sys.executable,'-B','run_gpu_line.py','candidate_focus','0','2','forge-gpu-run','--min','250','--max','300','--label','candidate_focus','--','timeout','--kill-after=10s','240s',sys.executable,str(T/'rev4_forge.py'),'run_gpu_tests.py','candidate','focus']
 argv,e=adapt(argv,e);r=run(argv,e);returned=datetime.datetime.now(datetime.timezone.utc).isoformat()
 att=runs/'candidate_focus/attempt-0001';pid=read(att/'child_started.json')['pid']
 try:stat=Path(f'/proc/{pid}/stat').read_text();state=stat.split(') ',1)[1].split()[0]
 except FileNotFoundError:state='absent'
 checked=datetime.datetime.now(datetime.timezone.utc).isoformat();p=read(att/'parent_result.json')
 s=run([sys.executable,'-B','score_gpu.py'],e);sc=read(runs/'G5_SUMMARY.json')['lines']['candidate_focus']
 ok=r['exit']==1 and p['exit']==124 and p['timeout_cause']=='outer-timeout' and state in ('absent','Z','X') and not sc['passed'] and 'Traceback' not in r['stderr']
 rows.append(dict(control=name,target='P116/D3 observe actual child PID from start marker immediately after parent return',passed=ok,child_pid=pid,child_state=state,child_alive=state not in ('absent','Z','X'),parent_return_observed=returned,pid_observed=checked,parent=r,parent_receipt=str((att/'parent_result.json').relative_to(T)),scorer=s))
 print(name,state,ok,flush=True)
(T/'G4_p116.json').write_text(json.dumps(dict(passed=all(r['passed'] for r in rows),controls=rows),indent=2)+'\n')
sys.exit(0 if all(r['passed'] for r in rows) else 1)
