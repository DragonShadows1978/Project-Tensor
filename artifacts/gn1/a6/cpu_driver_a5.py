"""A5 bounded CPU driver. Prior art: unchanged A4 foreground gate pattern."""
import os,sys,subprocess,json
from pathlib import Path
p=Path(__file__).resolve().parent
rows=[]
for name,script,args,seconds in [
 ('candidate_gate','cpu_checks.py',['candidate'],180),
 ('cpu_suite_gate','cpu_suite.py',[],580),
 ('tolerances_gate','derive_tolerances.py',[],180),
 ('pytorch_gate','pytorch_probe.py',[],180),
 ('equiv_g_gate','equivalence_execution.py',['E_npbool_g'],180),
 ('equiv_c_gate','equivalence_execution.py',['E_npbool_c'],180),
 *[('runtime_'+suite+'_gate','trace_runtime.py',[suite],220) for suite in ('focus','mutation','regression')]]:
 r=subprocess.run(['timeout',str(seconds+15)+'s',sys.executable,'-B',str(p/'run_cpu_gate.py'),name,str(seconds),sys.executable,'-B',str(p/script),*args],env=os.environ)
 rows.append(dict(name=name,exit=r.returncode));print(name,r.returncode,flush=True)
 if r.returncode:break
(p/'cpu_driver_results.json').write_text(json.dumps(rows,indent=2)+'\n')
sys.exit(0 if all(r['exit']==0 for r in rows) else 1)
