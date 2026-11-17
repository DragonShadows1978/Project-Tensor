"""A1 path/shape adapter around P1's read-only orchestration and fixed scorer.
Prior art: P1/AB-1 paired unsigned bit equality and pytest reports (2026),
SHA-256 (NIST, 2015), forge-gpu-run window leases (SystemOps, 2026).
Taken: original scoring/fields/byte arithmetic; ours: isolated build and case
dispatch, explicit CPU rehearsal, no device algorithm or selection changes.
"""
from gpu_common import *
import argparse, subprocess, traceback, importlib.util
from types import SimpleNamespace
import numpy as np
from worker_core import inputs,geometry
from layout import layout
from schema import fields

def original_driver():
 spec=importlib.util.spec_from_file_location('p1_original_driver',ROOT/'artifacts/apa_p1/gpu_driver.py')
 d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
 d.ART=ART
 old_child=d.child
 def child(cmd,log,env=None,limit=240):
  if env is not None:
   env=dict(env)
   if env['PYTHONPATH'].startswith(str(ROOT/'tensor_cuda')+':'):
    env['PYTHONPATH']=env['PYTHONPATH'].replace(str(ROOT/'tensor_cuda')+':',str(ART/'refs/port')+':',1)
  return old_child(cmd,log,env=env,limit=limit)
 d.child=child
 return d

def preflight():
 exp=load(ART/'EXPECTED.json')
 assert sha(ART/'REGISTRATION.json')==exp['registration_sha256']
 for p,h in exp['code_sha256'].items():assert sha(ROOT/p)==h,('staged source drift',p)
 b=load(ART/'build.json')
 for key in ('lab','production'):
  assert sha(ROOT/b[key+'_module'])==b[key+'_sha256']
 assert sha(ROOT/b['module'])==b['module_sha256']
 return exp

def compare_bitwise(out,case,runs=None):
 """This exact post-device comparison is also driven by constructed readbacks."""
 out=Path(out);a=load(out/'lab/result.json');b=load(out/'port/result.json');rows=[]
 assert a['case']==b['case']==case and a['input_sha256']==b['input_sha256']
 assert a['status']==b['status']=='PASS'
 with np.load(out/'lab/integer_views.npz',allow_pickle=False) as x,np.load(out/'port/integer_views.npz',allow_pickle=False) as y:
  assert set(x.files)==set(y.files) and x.files
  assert set(x.files)==set(fields(case)), 'missing or extra retained fields'
  assert set(x.files)==set(a['arrays'])==set(b['arrays'])
  for name in sorted(x.files):
   u,v=x[name],y[name]
   spec=fields(case)[name]
   assert str(u.dtype)==spec['dtype'] and list(u.shape)==spec['shape'],(name,'reference schema')
   assert u.dtype.kind=='u' and v.dtype.kind=='u'
   hu=hashlib.sha256(u.tobytes()).hexdigest();hv=hashlib.sha256(v.tobytes()).hexdigest()
   assert hu==a['arrays'][name]['sha256'] and hv==b['arrays'][name]['sha256']
   equal=u.dtype==v.dtype and u.shape==v.shape and np.array_equal(u,v)
   nonzero=bool(np.any(u!=0));nonconstant=bool(u.size>1 and np.any(u!=u.flat[0]))
   rows.append(dict(field=name,equal=bool(equal),dtype=str(u.dtype),shape=list(u.shape),lab_sha256=hu,port_sha256=hv,reference_nonzero=nonzero,reference_nonconstant=nonconstant,different_elements=int(np.count_nonzero(u!=v)) if u.shape==v.shape else None))
 guards=all(r['reference_nonzero'] and r['reference_nonconstant'] for r in rows) if case['kind']=='original_real_capture' else all(r['reference_nonzero'] and r['reference_nonconstant'] for r in rows if r['field'] in (f'step_{case["S"]}_eb3_out',f'step_{case["S"]}_eb4_out'))
 ok=guards and all(r['equal'] for r in rows)
 save(out/'comparison.json',dict(status='PASS' if ok else 'FAIL',case=case['id'],rows=rows,runs=runs or [],rule='Exact retained unsigned integer dtype, shape and elements; original six nonvacuity guards retained.'))
 return ok

def bitwise(args):
 case=next(c for c in load(ART/'INPUT_CENSUS.json') if c['id']==args.case)
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True);runs=[];d=original_driver()
 for arm in ('lab','port'):
  cmd=[sys.executable,str(ART/'gpu_worker.py'),'numerical','--arm',arm,'--case',args.case,'--out',str(out/arm)]
  r=d.child(cmd,out/f'{arm}.log');runs.append(r)
  if r['exit']:save(out/'comparison.json',dict(status='FAIL_WORKER',runs=runs));return False
 return compare_bitwise(out,case,runs)

def byte_check(args):
 ok=original_driver().byte_check(args)
 # All thirteen bitwise lines precede bytes. They retain each live layout receipt.
 layouts=[]
 for case in load(ART/'INPUT_CENSUS.json'):
  r=load(Path(args.out).parent/case['id']/'port/layout.json')
  assert r['status']=='PASS' and len(r['rows'])==4
  layouts.append(dict(case=case['id'],**r))
 save(Path(args.out)/'arena_comparison.json',dict(status='PASS' if ok else 'FAIL',cases=layouts,scope='Measured diagnostic arenas and successful non-diagnostic calls; exact non-diagnostic layout from source recurrence.'))
 return ok

def cpu_rehearsal(args,exp):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 row=dict(status='PASS_CPU_PLAN_ONLY',mode=args.mode,device_calls=0,registration_sha256=exp['registration_sha256'])
 if args.mode=='bitwise':
  c=next(c for c in load(ART/'INPUT_CENSUS.json') if c['id']==args.case)
  host=inputs(c)
  for n in c['append_lengths']:
   for k in ('qkv' if n>=c['S']-3 else 'kv'):
    a=geometry(host,c,n,k);assert list(a.shape)==[c['B'],c['H'] if k=='q' else c['KV'],1 if k=='q' else n,128]
  row.update(case=c['id'],shape=[c[x] for x in ('B','H','KV','S')],layouts=[layout(f,c,d) for f in ('eb3','eb4') for d in (False,True)])
 elif args.mode=='bytes':
  row['totals']={m:{str(s):8*s*(512+(nl-1)*580) for s in (2048,8192,32768)} for m,nl in [('Qwen3-1.7B',28),('Qwen3-4B',36)]}
 else:
  assert args.test_file in exp['gpu_test_files']['apa']+exp['gpu_test_files']['dlpack']
  row.update(test_file=args.test_file,collected_ids=[x for x in exp['gpu_collected_ids'] if x.split('::')[0]==args.test_file])
 save(Path(args.out)/'CPU_REHEARSAL.json',row)
 return True

def parse(argv=None):
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['bitwise','bytes','regression']);ap.add_argument('--case');ap.add_argument('--test-file');ap.add_argument('--out',required=True)
 return ap.parse_args(argv)

if __name__=='__main__':
 args=parse()
 try:
  exp=preflight()
  if os.environ.get('APA_P1_A1_CPU_REHEARSAL')=='1':ok=cpu_rehearsal(args,exp)
  else:
   lead_gpu()
   if args.mode=='regression':ok=original_driver().regression(SimpleNamespace(**vars(args),group=None))
   else:ok={'bitwise':bitwise,'bytes':byte_check}[args.mode](args)
 except BaseException:
  save(Path(args.out)/'exception.json',dict(status='FAIL_EXCEPTION',traceback=traceback.format_exc()));raise
 raise SystemExit(0 if ok else 1)
