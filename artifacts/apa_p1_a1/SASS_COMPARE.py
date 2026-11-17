"""Reproduce complete SASS equality from the two retained dumps, without rebuilding.
Prior art: NVIDIA cuobjdump; P1/AB-1 anonymous-namespace symbol correspondence
(Project-Tensor, 2026). Taken: matching generated names; ours: the explicit
14-name bijection. No instruction text, encoding or control word is normalized.
"""
from pathlib import Path
import re,json,hashlib
A=Path('artifacts/apa_p1_a1')
def read(path):
 blocks={};names={};key=None;lines=[]
 for line in path.open():
  m=re.search(r'Function : (\S+)',line)
  if m:
   name=m[1]
   key=re.sub(r'(_(?:gemm_apa|paint_raster)_cu_)[0-9a-f]{8}_[0-9]+',r'\1BUILD',name)
   assert key not in blocks,('non-bijective symbol map',key)
   names[key]=name;lines=[]
  elif key and line.strip()=='..........':blocks[key]=''.join(lines);key=None
  elif key:lines.append(line)
 return blocks,names
if __name__=='__main__':
 a,an=read(A/'pre_A1.sass');b,bn=read(A/'fixed.sass')
 rows=[dict(kernel=k,pre_symbol=an.get(k),fixed_symbol=bn.get(k),exact_body_equal=a.get(k)==b.get(k),pre_body_sha256=hashlib.sha256(a[k].encode()).hexdigest() if k in a else None,fixed_body_sha256=hashlib.sha256(b[k].encode()).hexdigest() if k in b else None) for k in sorted(set(a)|set(b))]
 r=dict(status='PASS' if a and a==b else 'FAIL',kernels=len(rows),differences=sum(not x['exact_body_equal'] for x in rows),compiler_generated_symbol_renames=sum(x['pre_symbol']!=x['fixed_symbol'] for x in rows),rows=rows)
 with (A/'SASS_RECHECK.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print({k:v for k,v in r.items() if k!='rows'});raise SystemExit(0 if r['status']=='PASS' else 1)
