"""D5 foreground trace driver. Prior art: A3 bounded subprocess gate driver."""
import os,subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parent
for suite in ('focus','mutation'):
 r=subprocess.run(['timeout','240s',sys.executable,'-B',str(p/'run_cpu_gate.py'),'runtime_'+suite+'_gate','220','python3','-B',str(p/'trace_runtime.py'),suite],env=os.environ)
 assert r.returncode==0
