"""A3 create-only bounded workers. Prior art: SP3 (2026) leases/receipts.
No new execution algorithm; no C cells or speculative treatment dispatch.
"""
import os,sys,time,traceback,shutil
from apa_sp4g_a3_common import *
from apa_sp4g_a3_registry import cells,by_id
from apa_sp4g_gpu import idle

def execute(c):
    if c != by_id().get(c.get('id')):
        raise Red('A3_UNREGISTERED_RECIPE')
    if c['kind'] == 'sweep':
        from apa_sp4g_a3_sweep import run_sweep
        return run_sweep(c)
    from apa_sp4g_a3_model import CallModel
    owner = None
    try:
        owner = CallModel(c)
        return owner.run_call(tokens()[:2048])
    finally:
        if owner is not None:
            owner.close()

def preflight(c):
    if c != by_id().get(c.get('id')):
        raise Red('A3_UNREGISTERED_RECIPE')
    preserved(); tokens(); verify_weight()
    gate = read(A/'CPU_GATES_A3.json')
    seal = read(A/'amendment_011_a3_fingerprint.json')
    if (gate['status'] != 'PASS_CPU_ONLY'
            or gate['fingerprint_amendment_sha256'] != sha(A/'amendment_011_a3_fingerprint.json')
            or fingerprint(c) != seal['fingerprint']):
        raise Red('A3_CPU_GATE_OR_FINGERPRINT_REQUIRED')
    for p,h in gate['execution_sha256'].items():
        if sha(R/p) != h:
            raise Red('A3_CPU_GATE_STALE: '+p)
    if path_a3(c['id']).exists():
        require_a3(c['id']); return 'DONE'
    for d in c['depends']:
        require_a3(d)
    if shutil.disk_usage(A).free < 2*(1<<30):
        raise Red('A3_DISK_RAIL_2GIB')
    return 'GPU'

def receipt(c,status,result,error=None):
    return dict(cell=c,status=status,registration_sha256=REG_SHA,
                a3_registration_sha256=REGISTRATION_SHA,
                fingerprint_schema='apa_sp4g_a3_isolated_v1',fingerprint=fingerprint(c),
                dependencies={d:sha(path_a3(d)) for d in c['depends']},
                result=result,error=error,evidence_class='per-call diagnostic; not model perplexity')

def main():
    action = sys.argv[1]
    if action == 'idle': idle(); return
    if action == 'list': print(json.dumps(cells(),indent=2)); return
    if action == 'next':
        for c in cells():
            if path_a3(c['id']).exists(): require_a3(c['id']); continue
            print(c['id']); return
        return
    name = sys.argv[2]
    if name not in by_id():
        raise Red('A3_UNKNOWN_OR_CONDITIONAL_CELL: '+name)
    c = by_id()[name]
    if action == 'preflight': print(preflight(c)); return
    if action == 'finish':
        rc,log = int(sys.argv[3]),sys.argv[4]
        if not path_a3(name).exists():
            publish(path_a3(name),receipt(c,'RED',dict(outcome='RAIL' if rc in (124,137) else 'WORKER_FAILED',
                    returncode=rc,log=log,log_sha256=sha(R/log)),error='worker terminated without completed receipt'))
        return
    if action != 'worker': raise Red('A3_UNKNOWN_ACTION')
    if preflight(c) == 'DONE': raise Red('A3_CREATE_ONLY_RECEIPT_EXISTS')
    if os.environ.get('APA_SP4G_LEASE') != '1': raise Red('GPU_LEASE_REQUIRED')
    import fcntl
    held,expected = os.fstat(9),os.stat('/tmp/forge-gpu.lock')
    if (held.st_dev,held.st_ino) != (expected.st_dev,expected.st_ino): raise Red('GPU_LEASE_WRONG_FILE')
    fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started = time.perf_counter()
    try:
        result = execute(c)
        j = receipt(c,'PASS',result)
    except BaseException as error:
        j = receipt(c,'RED',{},error=f'{type(error).__name__}: {error}')
        j['traceback'] = traceback.format_exc()
        publish(path_a3(name),j)
        raise
    j['worker_wall_s'] = time.perf_counter()-started
    publish(path_a3(name),j)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','calls')},indent=2))

if __name__ == '__main__':
    token = VALIDATION.set({})
    try: main()
    finally: VALIDATION.reset(token)
