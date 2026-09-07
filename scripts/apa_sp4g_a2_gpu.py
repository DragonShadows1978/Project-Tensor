"""A2 bounded workers. Prior art: SP3 leases/create-only receipts (2026).
No new execution algorithm; A1 execution files and receipts remain immutable.
"""
import os,sys,time,traceback,shutil
import numpy as np
from apa_sp4g_a2_common import *
from apa_sp4g_a2_registry import cells,by_id,rail_blocked
from apa_sp4g_gpu import idle

def execute(c):
    if rail_blocked(c):raise Red('A2_LONG_LEASE_REQUIRED_RAIL_NONFIT')
    if c['kind']=='replay_probe':
        from apa_sp4g_a2_metrics import replay_probe
        return replay_probe(c)
    if c['kind']=='margin_a2':
        from apa_sp4g_a2_metrics import whole_layer
        return whole_layer(c)
    from apa_sp4g_a2_model import ParityModel,PPLCapture
    owner=None
    try:
        if c['kind']=='ppl_capture':
            delta=require_pass('freeze')['result']['delta'] if c['arm']=='C' else None
            owner=PPLCapture(c,delta)
        else:owner=ParityModel(c);owner.install()
        ids=tokens()[:c['S']]
        ppl=owner.perplexity(ids,1024 if c['S']==2048 else 512)
        if c['kind']=='ppl_capture':return owner.finish_capture(ppl)
        if c['kind']=='parity' and len(owner.probes)!=8:raise Red('A2_EMPTY_PARITY_DIAGNOSTIC')
        if not owner.probes:raise Red('A2_NO_PRECISION_PROBES')
        r=dict(ppl,probes=owner.probes,focus=c.get('focus',c.get('treatment')),
            evidence_class='model diagnostic / one-window perplexity; original .005 exactness gate unchanged',
            precision_scope='FP32 global attention only, bf16 activations before and after; QAT weights and sliding layers unchanged' if c['kind']=='precision' else 'bf16 standard A propagation; first same-state fork per global layer')
        if c.get('treatment')=='D32':
            ref=require_a2('diag_a2_fp32_A_2048_w0')['result']
            r['D32_minus_A32_ppl']=ppl['ppl']-ref['ppl']
        return r
    finally:
        if owner is not None:owner.close()

def preflight(c):
    if rail_blocked(c):raise Red('A2_LONG_LEASE_REQUIRED_RAIL_NONFIT')
    preserved();tokens();verify_weight()
    gate=read(A/'CPU_GATES_A2.json');seal=read(A/'amendment_008_a2_fingerprint.json')
    if gate['status']!='PASS_CPU_ONLY' or gate['fingerprint_amendment_sha256']!=sha(A/'amendment_008_a2_fingerprint.json'):raise Red('A2_CPU_GATE_REQUIRED')
    if fingerprint_a2(c)!=seal['fingerprint']:raise Red('A2_FINGERPRINT_CHANGED')
    for p,h in gate['execution_sha256'].items():
        if sha(R/p)!=h:raise Red('A2_CPU_GATE_STALE: '+p)
    if path_a2(c['id']).exists():require_a2(c['id']);return 'DONE'
    for d in c['depends']:require_a2(d)
    if c['kind'] in ('ppl_capture','margin_a2') and shutil.disk_usage(A).free<12*(1<<30):raise Red('A2_DISK_RAIL_12GIB')
    return 'GPU'

def receipt(c,status,result,error=None):
    return dict(cell=c,status=status,registration_sha256=REG_SHA,fingerprint_schema='apa_sp4g_a2_isolated_v1',fingerprint=fingerprint_a2(c),dependencies={d:sha(path_a2(d)) for d in c['depends']},result=result,error=error,evidence_class='model perplexity' if c['kind']=='ppl_capture' else 'diagnostic / real-key kernel measurement')

def main():
    action=sys.argv[1]
    if action=='idle':idle();return
    if action=='list':print(json.dumps(cells(),indent=2));return
    if action=='next':
        for c in cells():
            if path_a2(c['id']).exists():require_a2(c['id']);continue
            # Diagnostic order is deliberate; completed RED stops resume.
            print(c['id']);return
        return
    name=sys.argv[2]
    if name not in by_id():raise Red('A2_UNKNOWN_OR_HISTORICAL_CELL_USE_REVIEWED_COMMANDS: '+name)
    c=by_id()[name]
    if action=='preflight':print(preflight(c));return
    if action=='finish':
        rc=int(sys.argv[3]);log=sys.argv[4]
        if not path_a2(name).exists():publish(path_a2(name),receipt(c,'RED',dict(outcome='RAIL' if rc in (124,137) else 'WORKER_FAILED',fit=False if rc in (124,137) else None,returncode=rc,log=log,log_sha256=sha(R/log)),error='worker terminated without completed receipt'))
        return
    if action!='worker':raise Red('A2_UNKNOWN_ACTION')
    if preflight(c)=='DONE':raise Red('A2_CREATE_ONLY_RECEIPT_EXISTS')
    if os.environ.get('APA_SP4G_LEASE')!='1':raise Red('GPU_LEASE_REQUIRED')
    import fcntl
    held=os.fstat(9);expected=os.stat('/tmp/forge-gpu.lock')
    if (held.st_dev,held.st_ino)!=(expected.st_dev,expected.st_ino):raise Red('GPU_LEASE_WRONG_FILE')
    fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB)
    t=time.perf_counter()
    try:r=execute(c);j=receipt(c,'PASS',r)
    except BaseException as e:
        j=receipt(c,'RED',{},error=f'{type(e).__name__}: {e}');j['traceback']=traceback.format_exc();publish(path_a2(name),j);raise
    j['worker_wall_s']=time.perf_counter()-t;publish(path_a2(name),j)
    print(json.dumps({k:v for k,v in r.items() if k not in ('files','probes')},indent=2))

if __name__=='__main__':
    t=VALIDATION.set({})
    try:main()
    finally:VALIDATION.reset(t)
