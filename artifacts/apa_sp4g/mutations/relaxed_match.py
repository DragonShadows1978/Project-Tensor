"""Bounded cell workers/aggregates. Prior art: SP3(2026) leased create-only
DAG, classical bisection, empirical measurement. No new selector or timing
algorithm; SP1/1.1 kernels reused. See registration prior_art for paper links.
"""
import argparse,ctypes,math,os,sys,time,traceback,subprocess,shutil
import numpy as np
from apa_sp4g_common import *
from apa_sp4g_registry import cells,by_id
CPU_KINDS={'freeze','eq','margin_summary','ppl_summary','exactness'}

def choose_trial(previous,target):
    for name,r in previous:
        if abs(r['fraction']-target)<=.02:return dict(carry=name,delta=r['delta'])
    lo,hi=0.,32.
    for _,r in previous:
        if r['fraction']<target:lo=max(lo,r['delta'])
        else:hi=min(hi,r['delta'])
    if hi<=lo:raise Red('CALIBRATION_BRACKET_BROKEN')
    return dict(carry=None,delta=4. if not previous else float(np.float32((lo+hi)/2)))

def kernel512():
    # Prior art: independent dense softmax/NumPy and SP1 tests. Tests actual
    # D512/MQA kernels at both launch geometries; no model-quality inference.
    os.environ['TC_APA_SP']='1';tc=load_runtime();rng=np.random.default_rng(401)
    rows=[]
    for dtype in ('float32','bfloat16'):
        for L,S in ((7,19),(1,4097)):
            q=tc.tensor(rng.normal(size=(1,16,L,512)).astype(np.float32)*.05,dtype=dtype)
            k=tc.tensor(rng.normal(size=(1,1,S,512)).astype(np.float32)*.05,dtype=dtype)
            v=tc.tensor(rng.normal(size=(1,1,S,512)).astype(np.float32)*.1,dtype=dtype)
            kq=tc.tensor(rng.normal(size=(1,1,S,512)).astype(np.float32)*.05,dtype=dtype)
            with tc.no_grad():
                out,mask=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,True)
                clean=tc._C.apa_selective_attention_sp(q,k,kq,v,1.,float(np.finfo(np.float32).max),L>1,None,False)
            x=q.float().numpy().astype(np.float64)@k.float().numpy().astype(np.float64).transpose(0,1,3,2)
            eligible=np.arange(S)<(S-L+np.arange(L)+1)[:,None]
            x=np.where(eligible,x,-np.inf);p=np.exp(x-x.max(axis=-1,keepdims=True));p/=p.sum(axis=-1,keepdims=True)
            want=p@v.float().numpy().astype(np.float64);got=out.float().numpy()
            tol=.001 if dtype=='float32' else .02
            np.testing.assert_allclose(got,want,rtol=tol,atol=tol)
            np.testing.assert_array_equal(got,clean.float().numpy())
            np.testing.assert_array_equal(mask.numpy().astype(bool),np.broadcast_to(eligible,(1,16,L,S)))
            rows.append(dict(dtype=dtype,L=L,S=S,max_abs=float(np.max(np.abs(got-want))),diagnostic_bitwise=True))
    return dict(pins=rows)

def planning(cell):
    if cell['kind']!='decode' or cell['S']!=32768:return None
    r=require_pass(f'decode_{cell["arm"]}_8192')['result']
    if r.get('steps')!=32 or r.get('fit') is not True:raise Red('CLEAN8192_PLANNING_SOURCE')
    estimate=r['setup_s']+16*r['prefill_s']+4*r['decode_s']+15
    return dict(estimate_s=estimate,formula='setup+16*prefill+4*decode+15',fit=None,outcome='PLANNED_RAIL' if estimate>=285 else 'WITHIN_PLANNED_RAIL',source=f'decode_{cell["arm"]}_8192',evidence_class='planning extrapolation; no32K measurement')

def execute(c):
    kind=c['kind']
    if kind=='kernel':return kernel512()
    if kind=='exactness':
        a,d=[require_pass(x)['result']['ppl'] for x in c['depends']]
        if abs(d-a)>.005:raise Red(f'D_A_EXACTNESS_FAILED: A={a}, D={d}, tolerance=.005')
        return dict(A_ppl=a,D_ppl=d,absolute_difference=abs(d-a),tolerance=.005)
    if kind=='ppl_summary':
        r=[require_pass(d)['result'] for d in c['depends']];n=sum(x['targets'] for x in r);loss=sum(x['total_nll'] for x in r)
        if n!=4096:raise Red('SHORT_TARGET_TOTAL')
        fraction={}
        if c['arm']!='A':
            pairs=sum(x['global_fraction']['pairs'] for x in r);selected=sum(x['global_fraction']['selected'] for x in r)
            if not pairs:raise Red('NO_GLOBAL_APA_PAIRS')
            fraction=dict(pairs=pairs,selected=selected,fraction=selected/pairs)
        return dict(targets=n,total_nll=loss,ppl=math.exp(loss/n),global_fraction=fraction)
    if kind=='freeze':
        target=require_pass('capture_B_2048')['result']['fraction']
        for d in c['depends'][1:]:
            r=require_pass(d)['result']
            if abs(r['fraction']-target)<=.01:return dict(delta=r['delta'],fraction=r['fraction'],target=target,trial=r.get('carried_from',d),match_abs=abs(r['fraction']-target))
        raise Red('C_MATCH_FAILED_12_REGISTERED_TRIALS')
    if kind in ('margin','margin_summary','eq'):
        from apa_sp4g_metrics import band,summarize,eq_result
        return {'margin':band,'margin_summary':summarize,'eq':eq_result}[kind](c)
    delta=None
    if c['arm']=='C':
        if kind=='trial':
            previous=[(d,require_pass(d)['result']) for d in c['depends'][1:]];target=require_pass('capture_B_2048')['result']['fraction'];decision=choose_trial(previous,target)
            if decision['carry']:
                r=require_pass(decision['carry'])['result']
                return dict(r,carried_from=r.get('carried_from',decision['carry']),no_model_execution=True)
            delta=decision['delta']
        else:delta=require_pass('freeze')['result']['delta']
    elif c['arm']=='D':delta=float(np.finfo(np.float32).max)
    elif c['arm']=='E':delta=require_pass('eq')['result']['delta']
    plan=planning(c)
    if plan and plan['outcome']=='PLANNED_RAIL':return dict(plan,steps=0,ms_token=None)
    from apa_sp4g_model import Model,resident
    ids=tokens();owner=None;t=time.perf_counter()
    try:
        owner=Model(c,delta,capture=kind in ('capture','trial'),observe=kind=='ppl' and c['arm']!='A')
        if kind in ('capture','trial'):r=owner.capture_run(ids)
        elif kind=='ppl':
            start=c['window']*2048 if c['S']==2048 else 0
            r=owner.perplexity(ids[start:start+c['S']],1024 if c['S']==2048 else 512)
        elif kind=='decode':r=owner.decode(ids)
        elif kind=='ceiling':
            lg,cache,seconds=owner.prefill(ids[:c['S']])
            if not np.isfinite(lg.float().numpy()).all():raise Red('CEILING_NONFINITE')
            r=dict(fit=True,outcome='FIT',prefill_s=seconds,load_s=owner.load_s,resident_after_mib=resident(),resident_load_mib=owner.resident_load_mib,resident_scope='own-PID post-load/post-prefill snapshots, NOT peak',**owner.pool.result())
        else:raise Red('UNKNOWN_KIND')
        return dict(r,delta=delta,arm=c['arm'],S=c['S'],wall_model_s=time.perf_counter()-t)
    except RuntimeError as e:
        # Exact known allocation errors only. Timing failures are never OOM.
        if kind=='ceiling' and any(x in str(e).lower() for x in ('cudaerrormemoryallocation','out of memory','cudamalloc failed')):
            try:res=resident()
            except Exception:res=None
            return dict(fit=False,outcome='OOM',error=str(e),resident_failure_mib=res,resident_scope='snapshot at caught allocation failure; not peak')
        raise
    finally:
        if owner is not None:owner.close()

def evidence(c):return 'model perplexity' if c['kind'] in ('ppl','ppl_summary','exactness') else 'kernel sweep' if c['kind'] not in ('freeze','eq') else 'finite calibration / reasoning'
def make_receipt(c,status,result,error=None):
    return dict(cell=c,status=status,registration_sha256=REG_SHA,fingerprint_schema='apa_sp4g_per_kind_v1',fingerprint=fingerprint(c),dependencies={d:sha(job_path(d)) for d in c['depends']},result=result,error=error,evidence_class=evidence(c))
def preflight(c):
    verify_sources();verify_weight();tokens();build_check()
    if job_path(c['id']).exists():require_pass(c['id']);return 'DONE'
    cache={}
    for d in c['depends']:require_pass(d,cache)
    if c['kind'] in ('capture','trial','margin','margin_summary'):
        # Conservative owned-artifact headroom, not a capacity extrapolation:
        # current cell output plus scratch; full all-pair output retained.
        need=12*(1<<30) if c['kind']=='margin_summary' else 8*(1<<30)
        if shutil.disk_usage(A).free<need:raise Red('DISK_RAIL_NEEDS_BYTES: '+str(need))
    return 'CPU' if c['kind'] in CPU_KINDS else 'GPU'
def idle():
    x=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
    if x.returncode or len(x.stdout.strip().splitlines())!=1:raise Red('GPU_NOT_VISIBLE_OR_NOT_SINGLE: '+x.stderr.strip())
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
    if p.returncode or p.stdout.strip():raise Red('GPU_BUSY_OR_PID_INSPECTION_FAILED: '+p.stdout.strip())
    print(x.stdout.strip())
def main():
    action=sys.argv[1];name=sys.argv[2] if len(sys.argv)>2 else None
    if action=='list':print(json.dumps(cells(),indent=2));return
    if action=='idle':idle();return
    if action=='next':
        for c in cells():
            if job_path(c['id']).exists():require_pass(c['id']);continue
            print(c['id']);return
        return
    c=by_id()[name]
    if action=='preflight':print(preflight(c));return
    if action=='finish':
        rc=int(sys.argv[3]);log=sys.argv[4]
        if not job_path(name).exists():publish(job_path(name),make_receipt(c,'RED',dict(outcome='RAIL' if rc in (124,137) else 'WORKER_FAILED',fit=None,returncode=rc,log=log,log_sha256=sha(R/log)),error='worker terminated without completed receipt'))
        return
    if action=='worker':
        if preflight(c)=='DONE':raise Red('CREATE_ONLY_JOB_ALREADY_EXISTS')
        if c['kind'] not in CPU_KINDS:
            if os.environ.get('APA_SP4G_LEASE')!='1':raise Red('GPU_LEASE_REQUIRED')
            import fcntl
            try:fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError as e:raise Red('GPU_LEASE_FD_NOT_HELD') from e
        started=time.perf_counter()
        try:r=execute(c);j=make_receipt(c,'PASS',r)
        except BaseException as e:
            j=make_receipt(c,'RED',{},error=f'{type(e).__name__}: {e}');j['traceback']=traceback.format_exc();publish(job_path(name),j);raise
        j['worker_wall_s']=time.perf_counter()-started;publish(job_path(name),j);print(json.dumps(j['result'],indent=2));return
    raise Red('UNKNOWN_ACTION')
if __name__=='__main__':
    token=VALIDATION.set({})
    try:main()
    finally:VALIDATION.reset(token)
