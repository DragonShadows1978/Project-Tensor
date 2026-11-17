"""Lead-only staged orchestration. Prior art: AB-1 reference-pair parity (2026),
pytest report phases, SHA-256 (NIST 2015). Exact equalities; no tolerance.
"""
from gpu_common import *
import subprocess,time,traceback,argparse,numpy as np


# APA-P1 A1 case census: retain round-1 cases, add DS1 identities and A3 fault classes.
# Prior art: DS1 A1/A3/A4 boundary census (Project-Tensor, 2026); no new selection rule.
BITWISE_CASES = [{'id': 'bitwise_17b_2048', 'model': 'Qwen3-1.7B', 'B': 1, 'H': 16, 'KV': 8, 'S': 2048, 'D': 128, 'kind': 'original_real_capture'}, {'id': 'bitwise_17b_8192', 'model': 'Qwen3-1.7B', 'B': 1, 'H': 16, 'KV': 8, 'S': 8192, 'D': 128, 'kind': 'original_real_capture'}, {'id': 'bitwise_17b_32768', 'model': 'Qwen3-1.7B', 'B': 1, 'H': 16, 'KV': 8, 'S': 32768, 'D': 128, 'kind': 'original_real_capture'}, {'id': 'bitwise_4b_2048', 'model': 'Qwen3-4B', 'B': 1, 'H': 32, 'KV': 8, 'S': 2048, 'D': 128, 'kind': 'original_real_capture'}, {'id': 'bitwise_4b_8192', 'model': 'Qwen3-4B', 'B': 1, 'H': 32, 'KV': 8, 'S': 8192, 'D': 128, 'kind': 'original_real_capture'}, {'id': 'bitwise_4b_32768', 'model': 'Qwen3-4B', 'B': 1, 'H': 32, 'KV': 8, 'S': 32768, 'D': 128, 'kind': 'original_real_capture'}, {'id': 'bitwise_B1_H2_KV1_S255', 'model': 'Qwen3-1.7B', 'B': 1, 'H': 2, 'KV': 1, 'S': 255, 'D': 128, 'kind': 'capture_head_slice_reshape'}, {'id': 'bitwise_B1_H4_KV2_S257', 'model': 'Qwen3-1.7B', 'B': 1, 'H': 4, 'KV': 2, 'S': 257, 'D': 128, 'kind': 'capture_head_slice_reshape'}, {'id': 'bitwise_B2_H2_KV1_S257', 'model': 'Qwen3-1.7B', 'B': 2, 'H': 2, 'KV': 1, 'S': 257, 'D': 128, 'kind': 'capture_head_slice_reshape'}, {'id': 'bitwise_B2_H4_KV2_S257', 'model': 'Qwen3-1.7B', 'B': 2, 'H': 4, 'KV': 2, 'S': 257, 'D': 128, 'kind': 'capture_head_slice_reshape'}, {'id': 'bitwise_B1_H2_KV1_S256', 'model': 'Qwen3-1.7B', 'B': 1, 'H': 2, 'KV': 1, 'S': 256, 'D': 128, 'kind': 'capture_head_slice_reshape'}, {'id': 'bitwise_B1_H2_KV1_S257', 'model': 'Qwen3-1.7B', 'B': 1, 'H': 2, 'KV': 1, 'S': 257, 'D': 128, 'kind': 'capture_head_slice_reshape'}, {'id': 'bitwise_B2_H4_KV2_S255', 'model': 'Qwen3-1.7B', 'B': 2, 'H': 4, 'KV': 2, 'S': 255, 'D': 128, 'kind': 'capture_head_slice_reshape'}]

def child(cmd,log,env=None,limit=240):
    disk();start=time.monotonic()
    with Path(log).open('x') as f:
        try:code=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=limit).returncode
        except subprocess.TimeoutExpired:code=124
    return dict(argv=[str(x) for x in cmd],exit=code,seconds=time.monotonic()-start,log=str(log))

def bitwise(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);runs=[]
    for arm in ('lab','port'):
        dst=out/arm
        cmd=[sys.executable,str(ART/'gpu_worker.py'),'numerical','--arm',arm,'--model',args.model,'--S',str(args.S),'--out',str(dst)]
        r=child(cmd,out/f'{arm}.log');runs.append(r)
        if r['exit']:save(out/'comparison.json',dict(status='FAIL_WORKER',runs=runs));return False
    lab=load(out/'lab/result.json');port=load(out/'port/result.json');rows=[]
    assert lab['case']==port['case'] and lab['input_sha256']==port['input_sha256']
    with np.load(out/'lab/integer_views.npz') as a,np.load(out/'port/integer_views.npz') as b:
        assert set(a.files)==set(b.files) and a.files
        for name in sorted(a.files):
            x,y=a[name],b[name]
            assert x.dtype.kind=='u' and y.dtype.kind=='u'
            equal=x.dtype==y.dtype and x.shape==y.shape and np.array_equal(x,y)
            rows.append(dict(field=name,equal=bool(equal),dtype=str(x.dtype),shape=list(x.shape),lab_sha256=hashlib.sha256(x.tobytes()).hexdigest(),port_sha256=hashlib.sha256(y.tobytes()).hexdigest(),reference_nonzero=bool(np.any(x!=0)),reference_nonconstant=bool(x.size>1 and np.any(x!=x.flat[0])),different_elements=int(np.count_nonzero(x!=y)) if x.shape==y.shape else None))
    ok=all(r['equal'] and r['reference_nonzero'] and r['reference_nonconstant'] for r in rows)
    save(out/'comparison.json',dict(status='PASS' if ok else 'FAIL',model=args.model,S=args.S,arms=['SP/EB3 variant 0 row-carry delta=8.1','SP-tk8/EB4 T-fresh row-carry delta=8.1'],rows=rows,runs=runs,rule='Every saved integer-view element equal; every reference nonzero and nonconstant; both workers checked input upload bits. No tolerance.'))
    return ok

def byte_check(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);runs=[]
    for model in ('Qwen3-1.7B','Qwen3-4B'):
      for S in (2048,8192,32768):
        dst=out/f'{model}_{S}'
        cmd=[sys.executable,str(ART/'gpu_worker.py'),'bytes','--model',model,'--S',str(S),'--out',str(dst)]
        r=child(cmd,out/f'{model}_{S}.log',limit=80);runs.append(r)
        if r['exit']:save(out/'comparison.json',dict(status='FAIL_WORKER',runs=runs));return False
    results=[load(out/f'{m}_{s}/result.json') for m in ('Qwen3-1.7B','Qwen3-4B') for s in (2048,8192,32768)]
    ok=all(r['status']=='PASS' and r['measured']==r['expected'] for r in results)
    save(out/'comparison.json',dict(status='PASS' if ok else 'FAIL',runs=runs,results=results,rule='580*S per KV head per packed layer + exact layer0 512*S per KV head; actual unique Storage.nbytes for all layers; AB-1 G2 reference.'))
    return ok

def regression(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);exp=load(ART/'EXPECTED.json');files=[args.test_file] if args.test_file else exp['gpu_test_files'][args.group];assert all(f in exp['gpu_test_files']['apa']+exp['gpu_test_files']['dlpack'] for f in files);build=load(ART/'build.json');runs=[]
    for arm in ('production','port'):
        module=(ART/'refs/production/tensor_cuda'/Path(build['production_module']).name) if arm=='production' else Path(build['module'])
        want=build['production_sha256'] if arm=='production' else build['module_sha256'];assert sha(module)==want
        package=ART/'refs/production' if arm=='production' else ROOT/'tensor_cuda'
        env=os.environ.copy();env.update(PYTHONPATH=str(package)+':'+str(ROOT/'tensor_cuda/tests')+':'+str(ART),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        cmd=[sys.executable,str(ART/'pytest_receipt.py'),str(out/f'{arm}.json'),'-q','--tb=short','-ra','-p','no:cacheprovider','--basetemp',str(out/f'tmp_{arm}'),*files]
        r=child(cmd,out/f'{arm}.log',env=env,limit=240);runs.append(r)
        if r['exit'] not in (0,1,5):save(out/'comparison.json',dict(status='FAIL_PYTEST_PROCESS',runs=runs));return False
    a,b=load(out/'production.json'),load(out/'port.json')
    expected=[x for x in exp['gpu_collected_ids'] if x.split('::')[0] in files]
    valid=a['collection']==b['collection']==expected and not a['collection_errors'] and not b['collection_errors'] and a['module']['sha256']==build['production_sha256'] and b['module']['sha256']==build['module_sha256'] and set(a['reports'])==set(expected)==set(b['reports'])
    # A1 S2: pytest (pytest project, exit code 5 = no tests collected). Taken:
    # collected node IDs; ours: this order's exact set comparison for empty files.
    zero_collection = a['exit'] == 5 and not a['collection']
    id_difference = len(set(a['collection']) ^ set(b['collection']))
    if zero_collection:
        valid = (set(a['collection']) == set(b['collection']) and
                 not a['collection_errors'] and not b['collection_errors'] and
                 a['module']['sha256'] == build['production_sha256'] and
                 b['module']['sha256'] == build['module_sha256'] and
                 not a['reports'] and not b['reports'])
    rows=[]
    for name in sorted(set(a['reports'])|set(b['reports'])):
        x=a['reports'].get(name,[]);y=b['reports'].get(name,[])
        xv=[(r['when'],r['outcome']) for r in x];yv=[(r['when'],r['outcome']) for r in y];same=xv==yv
        row=dict(id=name,production=xv,port=yv,equal=same)
        if not same:row.update(production_messages=[r['message'] for r in x if r['message']],port_messages=[r['message'] for r in y if r['message']])
        rows.append(row)
    ok=valid and (zero_collection or bool(rows)) and all(r['equal'] for r in rows)
    save(out/'comparison.json',dict(status='PASS' if ok else 'FAIL',ids=len(rows),differences=sum(not r['equal'] for r in rows)+(id_difference if zero_collection else 0),collected_id_sets_equal=set(a['collection'])==set(b['collection']),zero_collection=zero_collection,rows=rows,runs=runs,production=a['module'],port=b['module'],rule='Production exit 5 with zero IDs: exact collected ID sets, including empty equality. Otherwise exact same registered IDs and pytest phase/outcome strings. Baseline failures stay failures; full messages in both per-arm JSONs and every difference.'))
    return ok

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['bitwise','bytes','regression']);ap.add_argument('--model');ap.add_argument('--S',type=int);ap.add_argument('--group',choices=['apa','dlpack']);ap.add_argument('--test-file');ap.add_argument('--out',required=True);args=ap.parse_args();lead_gpu()
    for p,h in load(ART/'EXPECTED.json')['code_sha256'].items():assert sha(ROOT/p)==h,('staged source drift',p)
    try:ok={'bitwise':bitwise,'bytes':byte_check,'regression':regression}[args.mode](args)
    except BaseException:
        save(Path(args.out)/'exception.json',dict(status='FAIL_EXCEPTION',traceback=traceback.format_exc()));raise
    raise SystemExit(0 if ok else 1)
