"""A3's independently transcribed recurrence, evaluated against port sources.
Prior art: DS1 A3 (2026), aligned bump allocation (ISO C++17, 2017),
reference-pair differential checking (McKeeman, 1998; metadata unverified).
Taken: A3 size/view models and evaluator; ours: staged-domain/path adapter only.
"""
from pathlib import Path
import importlib.util, re, json
import numpy as np
P=Path(__file__).resolve().parents[2]/'tensor_cuda/src/apa_spo'
source=Path(__file__).resolve().parents[3]/'tensor_fork_apa_wt_apa2p/artifacts/apa_ds1_a3/recurrence.py'
spec=importlib.util.spec_from_file_location('a3_recurrence',source)
a3=importlib.util.module_from_spec(spec);spec.loader.exec_module(a3);a3.P=P
line_maps={}
for f,model in a3.SOURCE_MODEL.items():
 lines=(P/(f+'.cuh')).read_text().splitlines()
 size=next(i+1 for i,s in enumerate(lines) if 'std::vector<int64_t> sizes=' in s)
 end=next(i+1 for i,s in enumerate(lines[size:],size) if 'if(cursor' in s)
 model['sizes']=size
 new_lines=[i+1 for i,s in enumerate(lines[size:end],size) if re.search(r'\w+=view\(\{',s)]
 assert len(new_lines)==len(model['view_lines'])
 line_maps[f]=dict(zip(model['view_lines'],new_lines))
 model['view_lines']=new_lines
original_views=a3.views
def mapped_views(f,*args):
 return [(name,n,line_maps[f][line]) for name,n,line in original_views(f,*args)]
a3.views=mapped_views

def layout(f,case,diag):
 m=dict(diag=diag,profile=False,timeline=False,D=128,VD=128,dtype='bf16',baseline=False)
 B,H,K,S=[case[k] for k in ('B','H','KV','S')]
 sizes=[int(n) for n in a3.sizes(f,m,B,H,K,S)]
 views=a3.views(f,m,B,H,K,S)
 total=0
 for n in sizes:
  if n:total=((total+15)//16)*16+n
 cursor=0;placed=[]
 for name,n,line in views:
  cursor=((cursor+15)//16)*16
  placed.append(dict(name=name,bytes=int(n),offset=cursor));cursor+=int(n)
 assert total==cursor,(f,case,diag,total,cursor)
 assert all(x['offset']%16==0 for x in placed)
 return dict(function=f,diag=diag,total=total,cursor=cursor,views=placed)

def source_gate(cases):
 rows=[]
 for f in a3.SOURCE_MODEL:
  for m in a3.modes(f):
   for c in cases:
    B,H,K,S=[c[k] for k in ('B','H','KV','S')]
    if not a3.accept(f,m,B,H,K,S):continue
    # The A3 evaluator needs an ndarray S for broadcast checks.
    ss=np.asarray(S);ns=a3.sizes(f,m,B,H,K,ss);vs=a3.views(f,m,B,H,K,ss)
    a3.check_source_expressions(f,m,B,H,K,ss,ns,vs)
    total=0
    for n in ns:
     if n:total=((total+15)//16)*16+int(n)
    cursor=0
    for _,n,_ in vs:cursor=((cursor+15)//16)*16+int(n)
    assert total==cursor,(f,m,c)
    rows.append(dict(function=f,case=c['id'],mode=m,total=total,cursor=cursor))
 return dict(status='PASS',evaluations=len(rows),rows=rows)
