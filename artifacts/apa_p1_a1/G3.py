"""Repeat round-1 CPU outcome parity without swapping either installed module.
Prior art: pytest report API and P1 round-1 per-ID exact comparison (2026).
"""
from pathlib import Path
import sys,os,json,subprocess,hashlib,collections
A=Path('artifacts/apa_p1_a1');root=Path.cwd()
R=json.loads((A/'REGISTRATION.json').read_text())
files=R['cpu_g3_files'];exits={}
for arm in ('production','port'):
 env=os.environ.copy();env.update(PYTHONPATH=str(root/A/'refs'/arm)+':'+str(root/'tensor_cuda/tests')+':'+str(root/'artifacts/apa_p1'),PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',TMPDIR='/tmp/apa_p1_a1_g3')
 Path(env['TMPDIR']).mkdir(exist_ok=True)
 cmd=[sys.executable,'artifacts/apa_p1/pytest_receipt.py',str(A/f'g3_{arm}.json'),'-q','--tb=short','-ra','-p','no:cacheprovider','--basetemp',str(A/f'tmp_g3_{arm}'),*files]
 with (A/f'g3_{arm}.log').open('x') as f:code=subprocess.run(['timeout','540s',*cmd],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 exits[arm]=dict(exit=code,argv=cmd)
(A/'G3_EXITS.json').write_text(json.dumps(exits,indent=2)+'\n')
a,b=[json.loads((A/f'g3_{arm}.json').read_text()) for arm in ('production','port')]
rows=[]
def outcome(v):
 if any(r['outcome']=='failed' and r['when']!='call' for r in v):return 'ERROR'
 for r in v:
  if r['outcome']=='failed':return 'FAILED'
  if r['outcome']=='skipped':return 'SKIPPED'
 return 'PASSED' if any(r['when']=='call' and r['outcome']=='passed' for r in v) else 'MISSING'
for key in sorted(set(a['collection'])|set(b['collection'])):
 x=a['reports'].get(key,[]);y=b['reports'].get(key,[])
 rows.append(dict(id=key,production=outcome(x),port=outcome(y),equal=[(r['when'],r['outcome']) for r in x]==[(r['when'],r['outcome']) for r in y]))
old=json.loads(Path('artifacts/apa_p1/g3_comparison.json').read_text())
same_round1=[(x['id'],x['production'],x['port']) for x in rows]==[(x['id'],x['production'],x['port']) for x in old['rows']]
valid=a['collection']==b['collection'] and rows and not a['collection_errors'] and not b['collection_errors'] and set(a['reports'])==set(a['collection'])==set(b['reports']) and all(x['equal'] for x in rows)
valid=valid and a['module']['sha256']==R['reference']['production_sha256'] and b['module']['sha256']==json.loads((A/'BUILD.json').read_text())['module_sha256'] and same_round1
r=dict(status='PASS' if valid else 'FAIL',ids=len(rows),differences=sum(not x['equal'] for x in rows),same_as_round1=same_round1,production=dict(collections.Counter(x['production'] for x in rows)),port=dict(collections.Counter(x['port'] for x in rows)),rows=rows)
(A/'G3_COMPARISON.json').write_text(json.dumps(r,indent=2)+'\n')
with (A/'g3_ids.tsv').open('x') as f:
 f.write('test_id\tproduction\tport\tequal\n')
 for x in rows:f.write(f"{x['id']}\t{x['production']}\t{x['port']}\t{x['equal']}\n")
with (A/'LEDGER.md').open('a') as f:f.write('\n- G3 CPU '+r['status']+': '+str(r['ids'])+' IDs; '+str(r['differences'])+' differences; exact round-1 outcomes '+str(same_round1)+'. Historical errors/failures retained (G3_COMPARISON.json and per-arm logs/receipts). No tests modified.\n')
print({k:v for k,v in r.items() if k!='rows'})
raise SystemExit(0 if r['status']=='PASS' else 1)
