"""Compare all function disassemblies, including encodings and control words.
Prior art: NVIDIA cuobjdump (CUDA), exact text comparison; no normalization.
"""
from pathlib import Path
import json, subprocess, hashlib, re, collections
A=Path('artifacts/apa_p1_a1')
B=json.loads((A/'BUILD.json').read_text())
R=json.loads((A/'REGISTRATION.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 result=collections.defaultdict(list); key=None; lines=[]
 for line in p.open():
  m=re.search(r'Function : (\S+)',line)
  if m:key=m[1];lines=[]
  elif key and line.strip()=='..........':
   result[key].append(''.join(lines));key=None
  elif key:lines.append(line)
 return dict(result)
dumps={}
for arm,module in [('pre_A1',R['reference']['pre_A1_module']),('fixed',B['module'])]:
 p=A/(arm+'.sass')
 with p.open('x') as f:
  cmd=['/usr/local/cuda-12.6/bin/cuobjdump','-sass','--gpu-architecture','sm_89',module]
  subprocess.run(['timeout','850s',*cmd],stdout=f,stderr=subprocess.PIPE,check=True)
 dumps[arm]=dict(module=module,module_sha256=sha(module),dump=str(p),dump_sha256=sha(p),argv=cmd)
a,b=read(A/'pre_A1.sass'),read(A/'fixed.sass')
rows=[]
for name in sorted(set(a)|set(b)):
 x=a.get(name,[]);y=b.get(name,[])
 rows.append(dict(symbol=name,occurrences_pre=len(x),occurrences_fixed=len(y),
                  exact_text_equal=x==y,pre_sha256=[hashlib.sha256(z.encode()).hexdigest() for z in x],
                  fixed_sha256=[hashlib.sha256(z.encode()).hexdigest() for z in y]))
r=dict(status='PASS' if a and a==b else 'FAIL',symbols=len(rows),
       function_occurrences=sum(map(len,a.values())),differences=sum(not x['exact_text_equal'] for x in rows),
       method='Every cuobjdump Function block, header flags, addresses, full instructions, encodings and control words. No namespace filter or normalization.',dumps=dumps,rows=rows)
(A/'SASS_COMPARISON.json').write_text(json.dumps(r,indent=2)+'\n')
with (A/'LEDGER.md').open('a') as f:f.write('\n- S4 SASS '+r['status']+': '+str(r['symbols'])+' symbols, '+str(r['function_occurrences'])+' function blocks, '+str(r['differences'])+' exact text differences (SASS_COMPARISON.json). No device runtime.\n')
print({k:v for k,v in r.items() if k not in ('rows','dumps')})
raise SystemExit(0 if r['status']=='PASS' else 1)
