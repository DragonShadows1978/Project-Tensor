"""PASS iff FIX7 trace assertions execute against current dlpack_operand.cpp (plain and message OOM).
Prior art: FIX7 failure-plan host trace, CUDA12.6 call-specific returns, SQLite OOM discipline.
Only CPU trace tests are invoked; module CUDA skip guard is removed by AST for this explicitly CPU lane.
"""
from pathlib import Path
import ast,json,sys
root=Path(__file__).resolve().parents[3];source=root/'tc_signal/tests/test_fix7_findings.py'
t=ast.parse(source.read_text());t.body=[n for n in t.body if not (isinstance(n,(ast.Try,ast.If)) and n.lineno in (46,50))]
ns={'__file__':str(source),'__name__':'p1t_rev8_cpu_test_payloads'};exec(compile(t,str(source),'exec'),ns)
class Factory:
 def mktemp(self,name):
  d=Path('/tmp/p1t_rev9_build')/name;d.mkdir(exist_ok=True);return d
module=ns['trace_module'].__wrapped__(Factory());rows=[]
for name,fn in ns.items():
 if name.startswith('test_fix7_') and 'trace_module' in getattr(fn,'__code__',type('C',(),{'co_varnames':()})).co_varnames:
  if 'oom' in fn.__code__.co_varnames:
   for oom in [False,True]:fn(module,oom);rows.append({'test':name,'oom':oom,'result':'PASS'})
  else:fn(module);rows.append({'test':name,'result':'PASS'})
(root/'artifacts/p1t_rev9/attribution_cpu.json').write_text(json.dumps(rows,indent=2)+'\n')
assert rows,'no assertions executed'
print('PASS',len(rows),'current-source attribution assertions')
