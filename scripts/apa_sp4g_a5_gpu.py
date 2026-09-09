"""A5 foreground execution. Prior art: SP3/SP4G (2026) receipts/leases,
ordinary cooperative deadlines and scoped instrumentation. No signals,
background workers, kernel edits, or new algorithm.
"""
import os, sys, time, traceback, shutil
from contextlib import contextmanager
from apa_sp4g_a5_common import *
from apa_sp4g_a5_registry import cells


@contextmanager
def deadline_guard(owner, deadline):
    original = owner.gemma.Gemma4AttentionTC.__call__
    def check():
        if time.monotonic() >= deadline:
            raise Red('A5_COOPERATIVE_WORKER_RAIL')
    def attention(*args,**kwargs):
        check()
        result = original(*args,**kwargs)
        check()
        return result
    owner.gemma.Gemma4AttentionTC.__call__ = attention
    try:
        check()
        yield
        check()
    finally:
        owner.gemma.Gemma4AttentionTC.__call__ = original


def propagation_result(c):
    # Reuse A4's sum-of-squares aggregation unchanged; bind only its receipt
    # lookup to A5 in this single foreground worker and always restore it.
    import apa_sp4g_a4_gpu as previous
    original = previous.require_a4
    previous.require_a4 = require_a5
    try:
        return previous.propagation_result(c)
    finally:
        previous.require_a4 = original


def execute(c):
    if c != by_id().get(c.get('id')):
        raise Red('A5_UNREGISTERED_CELL')
    if c['kind'] == 'aggregate':
        return propagation_result(c)
    if c['kind'] == 'precision' and not rerun_allowed():
        raise Red('A5_D32_RERUN_REQUIRES_MEASURED_DISAGREEMENT')
    from apa_sp4g_a4_model import PrecisionModel, ResidualModel
    from apa_sp4g_a5_model import ScoredCallModel
    started = time.monotonic()
    owner = None
    try:
        if c['kind'] == 'call':
            owner = ScoredCallModel(c)
            owner.deadline = started+c['worker_s']
            return owner.run_call(tokens()[:2048])
        owner = PrecisionModel(c) if c['kind']=='precision' else ResidualModel(c)
        if c['kind']=='propagation':
            owner.directory = A/'propagation_a5'/c['id']
        with deadline_guard(owner,started+c['worker_s']):
            r = owner.perplexity(tokens()[:2048],1024)
            if c['kind']=='propagation':
                return owner.finish_residual(r)
            r = owner.finish_precision(r)
            reference = require_a5('diag_a2_fp32_A_2048_w0')['result']
            want = [(p['layer'],p['L'],p['S']) for p in reference['probes']]
            got = [(p['layer'],p['n'],p['S_all']) for p in r['dtype_pins']]
            if got != want:
                raise Red('A5_D32_A32_SCHEDULE_MISMATCH')
            return dict(r,A32_ppl=reference['ppl'],absolute_difference=abs(r['ppl']-reference['ppl']),
                        tolerance=.005,same_schedule=True,exactness_pass=a4.exactness(r),
                        arm='D32',S=2048,rerun_of='jobs_a4/'+GATE+'.json',
                        fix='none; diagnostic rerun after measured disagreement; kernels unchanged',C_E_unblocked=False)
    finally:
        if owner is not None:
            owner.close()


def preflight(c):
    if c != by_id().get(c.get('id')):
        raise Red('A5_UNREGISTERED_CELL')
    preserved(); tokens(); verify_weight()
    gate = read(A/'CPU_GATES_A5.json')
    seal = read(A/'amendment_018_a5_fingerprint.json')
    if (gate['status'] != 'PASS_CPU_ONLY'
            or gate['fingerprint_amendment_sha256'] != sha(A/'amendment_018_a5_fingerprint.json')
            or fingerprint(c) != seal['fingerprint']):
        raise Red('A5_CPU_GATE_OR_FINGERPRINT_REQUIRED')
    for p,h in gate['execution_sha256'].items():
        if sha(R/p) != h:
            raise Red('A5_CPU_GATE_STALE: '+p)
    if path_a5(c['id']).exists():
        require_a5(c['id'])
        return 'DONE'
    for d in c['depends']:
        require_a5(d)
    if c['kind']=='precision' and not rerun_allowed():
        raise Red('A5_D32_RERUN_REQUIRES_MEASURED_DISAGREEMENT')
    if shutil.disk_usage(A).free < 12*(1<<30):
        raise Red('A5_DISK_RAIL_12GIB')
    return 'CPU' if c['kind']=='aggregate' else 'GPU'


def receipt(c,status,result,error=None):
    return dict(cell=c,status=status,registration_sha256=REG_SHA,a5_registration_sha256=REGISTRATION_SHA,
        fingerprint_schema='apa_sp4g_a5_isolated_v1',fingerprint=fingerprint(c),
        dependencies={d:sha(path_a5(d)) for d in c['depends']},result=result,error=error,
        evidence_class='model perplexity' if c['kind']=='precision' else 'per-call / residual diagnostic; PASS means completed, not exactness')


def worker(c):
    decision = preflight(c)
    if decision=='DONE':
        raise Red('A5_CREATE_ONLY_RECEIPT_EXISTS')
    if decision=='GPU':
        if os.environ.get('APA_SP4G_LEASE') != '1':
            raise Red('GPU_LEASE_REQUIRED')
        import fcntl
        held,expected = os.fstat(9),os.stat('/tmp/forge-gpu.lock')
        if (held.st_dev,held.st_ino) != (expected.st_dev,expected.st_ino):
            raise Red('GPU_LEASE_WRONG_FILE')
        fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started = time.monotonic()
    try:
        result = execute(c)
        passed = c['kind']!='precision' or result['exactness_pass']
        j = receipt(c,'PASS' if passed else 'RED',result,
                    None if passed else 'A5_D32_A32_EXACTNESS_FAILED; C/E remain blocked')
    except BaseException as error:
        j = receipt(c,'RED',{},f'{type(error).__name__}: {error}')
        j['traceback'] = traceback.format_exc()
        j['worker_wall_s'] = time.monotonic()-started
        publish(path_a5(c['id']),j)
        raise
    j['worker_wall_s'] = time.monotonic()-started
    publish(path_a5(c['id']),j)
    print(json.dumps(dict(status=j['status'],cell=c['id'],receipt=str(path_a5(c['id']).relative_to(R)),
                          classification=result.get('classification'),ppl=result.get('ppl')),indent=2))
    if not passed:
        raise Red(j['error'])


def main():
    action = sys.argv[1]
    if action=='list':
        print(json.dumps(cells(),indent=2)); return
    if action=='next':
        for c in cells():
            # Independent diagnostics do not stop on another diagnostic RED.
            if c['kind']=='precision':
                continue  # opt-in conditional rerun only
            if path_a5(c['id']).exists():
                continue
            try:
                preflight(c)
            except (Red,OSError):
                continue
            print(c['id']); return
        return
    if action=='idle':
        # NVIDIA query only. No timeout subprocess or process signaling.
        import subprocess
        r = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                           capture_output=True,text=True,check=True)
        if r.stdout.strip():
            raise Red('A5_GPU_BUSY: '+r.stdout.strip())
        return
    if action=='finish':
        c = by_id()[sys.argv[2]]
        rc,log = int(sys.argv[3]),sys.argv[4]
        if not path_a5(c['id']).exists():
            publish(path_a5(c['id']),receipt(c,'RED',dict(returncode=rc,log=log,log_sha256=sha(R/log)),
                                          'worker exited without completed receipt'))
        if read(path_a5(c['id']))['status']!='PASS':
            raise Red('A5_WORKER_RED: '+c['id'])
        return
    if len(sys.argv)!=3 or sys.argv[2] not in by_id():
        raise Red('A5_UNKNOWN_CELL')
    c = by_id()[sys.argv[2]]
    if action=='preflight':
        print(preflight(c))
    elif action=='worker':
        worker(c)
    else:
        raise Red('A5_UNKNOWN_ACTION')


if __name__=='__main__':
    token = VALIDATION.set({})
    try:
        main()
    finally:
        VALIDATION.reset(token)
