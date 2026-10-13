"""PASS iff normal and reporting-OOM core exit-runner arms both emit the saved CUDA error and device; exit1 otherwise.
Prior art: SQLite OOM testing; CPython3.12 exit last-reader rule; build_core_host_oom extracts current product sources.
"""
from pathlib import Path
import subprocess,sys,json
rows=[]
for arm in ['--control','--oom']:
 r=subprocess.run(['timeout','15',sys.executable,str(Path(__file__).with_name('core_exit_log_oom_cpu.py')),arm],capture_output=True,text=True,timeout=20)
 rows.append({'arm':arm,'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'reported':'cudaErrorInvalidValue' in r.stderr and 'device 0' in r.stderr})
print(json.dumps(rows,indent=2))
assert all(r['rc']==0 and r['reported'] for r in rows),'core native exit logger loses its report under OOM'
