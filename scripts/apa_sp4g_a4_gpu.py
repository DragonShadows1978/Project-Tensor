"""A4 bounded cells. Prior art: SP3 (2026) leases/DAG/calibration,
classical bisection, Frobenius norms; SP2 conditional empirical delta.
No new attention or optimization method; additive reference ruling only.
"""
import os,sys,time,traceback,shutil,math
import numpy as np
from apa_sp4g_a4_common import *
from apa_sp4g_a4_registry import cells,by_id,CPU_KINDS,GATE
from apa_sp4g_gpu import idle,choose_trial
from apa_sp4g_metrics import upward

def propagation_result(c):
    from apa_sp4g_a4_model import validate_coverage,LAYERS
    a,d=[require_a4(n)['result'] for n in c['depends']]
    ma,md=[read(R/r['manifest']) for r in (a,d)]
    rows=[]
    for layer in [str(l) for l in LAYERS]+['final_norm']:
        ar,dr=[m['residuals'][layer] for m in (ma,md)]
        validate_coverage(ar,c['S']);validate_coverage(dr,c['S'])
        if [(r['lo'],r['n']) for r in ar]!=[(r['lo'],r['n']) for r in dr]:raise Red('A4_PROPAGATION_SCHEDULE_MISMATCH')
        numerator=denominator=0.;maximum=0.;blocks=[]
        for ra,rd in zip(ar,dr):
            x,y=[np.load(R/r['file']['path']).astype(np.float64) for r in (ra,rd)]
            if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise Red('A4_PROPAGATION_INVALID_ARRAY')
            err=y-x;ns=float(np.sum(err*err));ds=float(np.sum(x*x));mx=float(np.max(np.abs(err)))
            numerator+=ns;denominator+=ds;maximum=max(maximum,mx)
            blocks.append(dict(lo=ra['lo'],n=ra['n'],relative_frobenius=math.sqrt(ns)/max(math.sqrt(ds),1e-30),max_abs=mx))
        rows.append(dict(layer=layer,relative_frobenius=math.sqrt(numerator)/max(math.sqrt(denominator),1e-30),
                         max_abs=maximum,rows=c['S']-1,blocks=blocks))
    return dict(per_layer=rows,A_ppl=a['ppl'],D_ppl=d['ppl'],relative_ppl_difference=d['ppl']/a['ppl']-1,
        norm_reference='A; sqrt(sum squared error over ALL rows/chunks)/sqrt(sum squared A), not mean of chunk ratios',
        evidence_class='bf16 residual propagation diagnostic / complete window0 PPL',
        caveat='Observation of accumulation, not a causal proof that per-call errors alone explain model PPL')

def execute(c):
    if c!=by_id().get(c.get('id')) or c.get('S',0)>=16384:raise Red('A4_UNREGISTERED_OR_LONG_RAIL')
    kind=c['kind']
    if kind=='aggregate':
        if c['operation']=='propagation':return propagation_result(c)
        rows=[require_a4(d)['result'] for d in c['depends']]
        n=sum(r['targets'] for r in rows);loss=sum(r['total_nll'] for r in rows)
        if n!=4096:raise Red('A4_SHORT_TARGET_TOTAL')
        pairs=sum(r['global_fraction']['pairs'] for r in rows);selected=sum(r['global_fraction']['selected'] for r in rows)
        if pairs<=0:raise Red('A4_EMPTY_SELECTION_POPULATION')
        return dict(ppl=math.exp(loss/n),targets=n,total_nll=loss,
                    global_fraction=dict(pairs=pairs,selected=selected,fraction=selected/pairs))
    if kind=='freeze':
        target=require_a4('ppl_capture_B_2048_w0')['result']['global_fraction']['fraction']
        for name in [n for n in c['depends'] if n.startswith('trial_a4_')]:
            r=require_a4(name)['result']
            if abs(r['fraction']-target)<=.01:return dict(delta=r['delta'],fraction=r['fraction'],target=target,match_abs=abs(r['fraction']-target),trial=r.get('carried_from',name),population='actual PPL window0 queries 0..2046')
        raise Red('A4_C_MATCH_FAILED_12_REGISTERED_TRIALS')
    if kind=='margin':
        from apa_sp4g_a4_metrics import whole_layer
        return whole_layer(c)
    if kind=='eq':
        names=[n for n in c['depends'] if n.startswith('margin_')]
        rows=[require_a4(n)['result'] for n in names]
        if len(rows)!=32:raise Red('A4_EQ_SOURCE_COUNT')
        eq=upward(max(r['eq_sp'] for r in rows))
        return dict(eq=eq,epsilon=.01,delta=upward(math.log(100)+2*eq),source_count=32,
            calibration='finite empirical maximum over B/C PPL global-layer keys at2048/8192; conditional bound only, not universal')
    delta=None
    if c['arm']=='C':
        if kind=='trial':
            prev=[(n,require_a4(n)['result']) for n in c['depends'] if n.startswith('trial_a4_')]
            target=require_a4('ppl_capture_B_2048_w0')['result']['global_fraction']['fraction']
            choice=choose_trial(prev,target)
            if choice['carry']:
                r=require_a4(choice['carry'])['result']
                return dict(r,carried_from=r.get('carried_from',choice['carry']),no_model_execution=True)
            delta=choice['delta']
        else:delta=require_a4('freeze_a4')['result']['delta']
    elif c['arm']=='E':delta=require_a4('eq_a4')['result']['delta']
    from apa_sp4g_a4_model import PrecisionModel,ResidualModel,CaptureModel
    from apa_sp4g_model import Model
    owner=None
    try:
        if kind=='precision':owner=PrecisionModel(c)
        elif kind=='propagation':owner=ResidualModel(c)
        elif kind=='ppl_capture':owner=CaptureModel(c,delta)
        else:owner=Model(c,delta,observe=kind in ('trial','ppl'))
        ids=tokens();start=c['window']*2048 if c['S']==2048 else 0
        if kind=='decode':return owner.decode(ids)
        r=owner.perplexity(ids[start:start+c['S']],1024 if c['S']==2048 else 512)
        if kind=='precision':
            r=owner.finish_precision(r)
            if c['id']==GATE:
                ref=require_a4('diag_a2_fp32_A_2048_w0')['result']
                want=[(p['layer'],p['L'],p['S']) for p in ref['probes']]
                got=[(p['layer'],p['n'],p['S_all']) for p in r['dtype_pins']]
                if got!=want:raise Red('A4_A32_SCHEDULE_MISMATCH')
                r.update(A32_ppl=ref['ppl'],absolute_difference=abs(r['ppl']-ref['ppl']),tolerance=.005,
                         reference='diag_a2_fp32_A_2048_w0',same_schedule=True,exactness_pass=exactness(r))
        elif kind=='propagation':r=owner.finish_residual(r)
        elif kind=='ppl_capture':r=owner.finish_capture(r)
        elif kind=='trial':r.update(fraction=r['global_fraction']['fraction'])
        return dict(r,delta=delta,arm=c['arm'],S=c['S'])
    finally:
        if owner is not None:owner.close()

def preflight(c):
    if c!=by_id().get(c.get('id')) or c.get('S',0)>=16384:raise Red('A4_UNREGISTERED_OR_LONG_RAIL')
    preserved();tokens();verify_weight()
    gate=read(A/'CPU_GATES_A4_V2.json');seal=read(A/'amendment_016_a4_fingerprint.json')
    if (gate['status']!='PASS_CPU_ONLY' or gate['fingerprint_amendment_sha256']!=sha(A/'amendment_016_a4_fingerprint.json')
        or fingerprint(c)!=seal['fingerprint']):raise Red('A4_CPU_GATE_OR_FINGERPRINT_REQUIRED')
    for p,h in gate['execution_sha256'].items():
        if sha(R/p)!=h:raise Red('A4_CPU_GATE_STALE: '+p)
    if path_a4(c['id']).exists():require_a4(c['id']);return 'DONE'
    for d in c['depends']:require_a4(d)
    if shutil.disk_usage(A).free<12*(1<<30):raise Red('A4_DISK_RAIL_12GIB')
    return 'CPU' if c['kind'] in CPU_KINDS else 'GPU'

def receipt(c,status,result,error=None):
    return dict(cell=c,status=status,registration_sha256=REG_SHA,a4_registration_sha256=REGISTRATION_SHA,
        fingerprint_schema='apa_sp4g_a4_isolated_v1',fingerprint=fingerprint(c),
        dependencies={d:sha(path_a4(d)) for d in c['depends']},result=result,error=error,
        evidence_class='model perplexity' if c['kind'] in ('ppl','precision','ppl_capture') else 'diagnostic / finite calibration / G2-G3, not standalone model quality')

def main():
    action=sys.argv[1]
    if action=='idle':idle();return
    if action=='list':print(json.dumps(cells(),indent=2));return
    if action=='next':
        for c in cells():
            if path_a4(c['id']).exists():require_a4(c['id']);continue
            print(c['id']);return
        return
    name=sys.argv[2]
    if name not in by_id():raise Red('A4_UNKNOWN_CELL: '+name)
    c=by_id()[name]
    if action=='preflight':print(preflight(c));return
    if action=='finish':
        rc,log=int(sys.argv[3]),sys.argv[4]
        if not path_a4(name).exists():publish(path_a4(name),receipt(c,'RED',dict(outcome='RAIL' if rc in (124,137) else 'WORKER_FAILED',returncode=rc,log=log,log_sha256=sha(R/log)),error='worker terminated without completed receipt'))
        if read(path_a4(name))['status']!='PASS':raise Red('A4_WORKER_RED: '+name)
        return
    if action!='worker':raise Red('A4_UNKNOWN_ACTION')
    if preflight(c)=='DONE':raise Red('A4_CREATE_ONLY_RECEIPT_EXISTS')
    if c['kind'] not in CPU_KINDS:
        if os.environ.get('APA_SP4G_LEASE')!='1':raise Red('GPU_LEASE_REQUIRED')
        import fcntl
        held,expected=os.fstat(9),os.stat('/tmp/forge-gpu.lock')
        if (held.st_dev,held.st_ino)!=(expected.st_dev,expected.st_ino):raise Red('GPU_LEASE_WRONG_FILE')
        fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.perf_counter()
    try:
        result=execute(c)
        passed=c['id']!=GATE or exactness(result)
        j=receipt(c,'PASS' if passed else 'RED',result,None if passed else 'A4_D32_A32_EXACTNESS_FAILED; STOP, lead investigates')
    except BaseException as error:
        j=receipt(c,'RED',{},error=f'{type(error).__name__}: {error}');j['traceback']=traceback.format_exc()
        publish(path_a4(name),j);raise
    j['worker_wall_s']=time.perf_counter()-started;publish(path_a4(name),j)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','dtype_pins')},indent=2))
    if j['status']!='PASS':raise Red(j['error'])

if __name__=='__main__':
    token=VALIDATION.set({})
    try:main()
    finally:VALIDATION.reset(token)
