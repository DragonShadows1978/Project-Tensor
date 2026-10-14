"""PASS iff control and allocation-failure arms both name cudaEventQuery on stderr; exit 1 otherwise.
Prior art: SQLite OOM testing; actual-source native logger, child exit has no later reader.
"""
import subprocess,sys,json
from pathlib import Path
script=Path(__file__).with_name('exit_log_oom_cpu.py');rows=[]
for arg in ['--control','--oom']:
 r=subprocess.run(['timeout','15',sys.executable,str(script),arg],capture_output=True,text=True,timeout=20)
 rows.append({'arm':arg,'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'reported':'cudaEventQuery' in r.stderr})
print(json.dumps(rows,indent=2))
assert all(r['rc']==0 and r['reported'] for r in rows),'native exit logger failed its own reporting OOM'

# The companion core child has the same final-reporter scenario; its static record labels the injected host failure.
