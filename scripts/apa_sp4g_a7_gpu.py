"""Prior art: SP4G (2026) foreground flock/create-only DAG; Make/Feldman
(1979), NVIDIA CUDA memory-error distinction. A7 adds the expressly scoped
long rail and terminal OOM inference. No signals or new attention method.
"""
import fcntl, os, sys, time
from apa_sp4g_a7_common import *
from apa_sp4g_a7_model import Deadline, Rail, ResidentPeak, measure


def previous_receipt(c):
    return require_a7(c['depends'][-1]) if c['depends'] else None


def preflight(c):
    valid_cell(c); preserved()
    ids = tokens()
    if len(ids) < c['S']:
        raise Red('A7_PINNED_STREAM_TOO_SHORT')
    gate = read(A/'CPU_GATES_A7.json'); seal = read(SEAL)
    if (gate['status'] != 'PASS_CPU_ONLY' or gate['fingerprint_amendment_sha256'] != sha(SEAL)
            or gate['execution_sha256'] != seal['fingerprint']):
        raise Red('A7_CPU_GATE_OR_SEAL_REQUIRED')
    verify_fingerprint(seal['fingerprint'])
    if path_a7(c['id']).exists():
        require_a7(c['id']); return 'DONE'
    previous = previous_receipt(c)
    return 'CPU_NON_FIT' if oom_source(c,previous) else 'GPU'


def receipt(c, result, wall_s):
    return dict(cell=c, status=outcome_status(result['outcome']), result=result,
                error=result.get('error'), worker_wall_s=wall_s,
                registration_sha256=REG_SHA,a7_registration_sha256=REGISTRATION_SHA,
                fingerprint_schema='apa_sp4g_ceiling_long_a7_v1',fingerprint=fingerprint(),
                dependencies={d:sha(path_a7(d)) for d in c['depends']},
                evidence_class='G3 prefill memory/time experiment or explicitly inferred later non-fit; no standalone model quality')


def inherited_nonfit(c, previous):
    source = oom_source(c,previous)
    if not source:
        raise Red('A7_NON_FIT_REQUIRES_PREVIOUS_OOM')
    return dict(outcome='NON_FIT_AFTER_OOM',fit=False,executed=False,oom_source=source,
                completed_tokens=0,kv_cache_at_S=kv_size(c['S']),
                peak_resident_mib=None,sampled_peak_resident_mib=None,peak_status='NOT_RUN',
                extrapolated_worker_s=c['estimate_worker_s'],
                reason='registered larger-S non-fit inferred from first OOM of same arm; no measurement')


def worker(c):
    start = time.monotonic()
    outer_start = float(os.environ.get('APA_SP4G_A7_OUTER_START', start))
    if not math.isfinite(outer_start) or outer_start > start:
        raise Red('A7_OUTER_START_INVALID')
    decision = preflight(c)
    if decision == 'DONE':
        raise Red('A7_CREATE_ONLY_RECEIPT_EXISTS')
    if decision == 'CPU_NON_FIT':
        result = inherited_nonfit(c,previous_receipt(c))
    else:
        if os.environ.get('APA_SP4G_A7_LEASE') != '1':
            raise Red('A7_GPU_LEASE_REQUIRED')
        held, expected = os.fstat(9), os.stat('/tmp/forge-gpu.lock')
        if (held.st_dev,held.st_ino) != (expected.st_dev,expected.st_ino):
            raise Red('A7_GPU_LEASE_WRONG_FILE')
        fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline = Deadline(start,outer_start)
        result = measure(c,tokens(),deadline)
        # Include final telemetry/cleanup in the worker rail. An OOM remains
        # an OOM; only a late FIT becomes time-censored.
        if result['outcome'] == 'FIT':
            try:
                deadline.check()
            except Rail as error:
                result.update(outcome='RAIL',fit=None,failure_class=str(error),error=str(error))
    j = receipt(c,result,time.monotonic()-start)
    publish(path_a7(c['id']),j)
    print(json.dumps(dict(cell=c['id'],status=j['status'],outcome=result['outcome'],
                          wall_s=j['worker_wall_s'],peak_status=result['peak_status']),indent=2))
    if result['outcome'] == 'ERROR':
        raise Red(result['error'])


def finish(c, rc, log):
    if not path_a7(c['id']).exists():
        # A signal/137/124 or missing receipt is never proof of CUDA OOM or
        # cooperative RAIL. Preserve the log and stop; no rerun.
        publish(path_a7(c['id']),receipt(c,dict(outcome='ERROR',fit=None,executed=False,
            kv_cache_at_S=kv_size(c['S']),extrapolated_worker_s=c['estimate_worker_s'],
            error=f'worker exited {rc} without completed receipt; inspect {log}'),0.))
    start = float(os.environ['APA_SP4G_A7_OUTER_START'])
    outer_wall = time.monotonic()-start
    publish(A/'leases_a7'/(c['id']+'.json'),dict(cell=c['id'],returncode=rc,
            log=log,log_sha256=sha(R/log),outer_wall_s=outer_wall,outer_rail_s=1560,
            outer_rail_exceeded=outer_wall > 1560,worker_receipt_sha256=sha(path_a7(c['id'])),
            cooldown_s=30 if os.environ.get('APA_SP4G_A7_LEASE')=='1' else 0,
            safety='foreground, no signals/kills; elapsed outer wall includes lease/validation/worker/cooldown'))
    require_a7(c['id'])
    if outer_wall > 1560:
        raise Red('A7_OUTER_RAIL_EXCEEDED; no native-call hard bound under no-kill')


def main():
    action = sys.argv[1]
    if action == 'list':
        print(json.dumps(cells(),indent=2)); return
    if action == 'next':
        for c in cells():
            if path_a7(c['id']).exists():
                require_a7(c['id']); continue
            preflight(c); print(c['id']); return
        return
    if action == 'idle':
        m = ResidentPeak()
        try:
            rows = m.processes()
            if rows:
                raise Red('A7_GPU_BUSY: '+str([r.pid for r in rows]))
        finally:
            m.close()
        return
    if len(sys.argv)<3 or sys.argv[2] not in by_id():
        raise Red('A7_UNKNOWN_CELL')
    c = by_id()[sys.argv[2]]
    if action == 'preflight':
        print(preflight(c))
    elif action == 'worker':
        worker(c)
    elif action == 'finish':
        finish(c,int(sys.argv[3]),sys.argv[4])
    else:
        raise Red('A7_UNKNOWN_ACTION')


if __name__ == '__main__':
    token = VALIDATION.set({})
    try:
        main()
    finally:
        VALIDATION.reset(token)
