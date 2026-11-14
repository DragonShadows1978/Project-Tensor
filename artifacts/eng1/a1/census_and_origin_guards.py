"""Independent source census and origin rejection check.
Prior art: REV1 census_scope.py (text-line count), Python ast/PSF, SHA256 NIST.
Taken: distinguish engine vs NumPy calls and explicit subprocess pin check; no novel algorithm.
"""
import ast,sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[3];A=R/'artifacts/eng1/a1';T=R/'artifacts/eng1/tool'
rows=[];lines=0
for p in sorted((R/'tensor_cuda/tensor_cuda').rglob('*.py')):
 text=p.read_text();numpy=p.name in ('__init__.py','affine.py')
 if not numpy:lines+=sum('reshape(' in x for x in text.splitlines())
 for n in ast.walk(ast.parse(text)):
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='reshape':
   rows.append(dict(file=str(p.relative_to(R)),line=n.lineno,column=n.col_offset,expression=ast.get_source_segment(text,n),kind='numpy' if numpy else 'engine'))
engine=[x for x in rows if x['kind']=='engine']
counts=dict(engine_calls=len(engine),engine_lines=lines,numpy_calls=len(rows)-len(engine))
assert counts==dict(engine_calls=37,engine_lines=32,numpy_calls=9)
(A/'caller_census.json').write_text(json.dumps(dict(counts=counts,rows=rows,
 line_convention="REV1 census_scope.py counts physical lines containing reshape(; AST expression-start lineno can precede the method token, e.g. functional.py:189-190.",credit='REV1 census_scope.py/CALLER_CENSUS.md; rechecked live sources'),indent=2)+'\n')
sys.path.insert(0,str(T));from arm_origin import preflight,bound_environment
import hashlib
package=T/'arms/candidate/tensor_cuda';module=next((package/'tensor_cuda').glob('_tensor_cuda*.so'))
try:preflight(package,module.name,'0'*64,T,A/'origin_wrong_pin.json',bound_environment(package))
except AssertionError as e:
 assert str(e)=='subprocess pin mismatch'
 (A/'origin_refusal.json').write_text(json.dumps(dict(passed=True,expected_refusal=str(e),pytest_started=False),indent=2)+'\n')
else:raise AssertionError('wrong pin was accepted')
print('live census:',counts,'wrong-pin refusal: PASS')
