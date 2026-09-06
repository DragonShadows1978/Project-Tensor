#!/usr/bin/env python3
"""Read-only preflight, no-process-intervention device checks, timeout receipts.
Prior art: standard foreground leases, immutable provenance and fail-closed gates.
"""
import json
import subprocess
import sys
from apa_sp3_common import ART, REG_SHA, Red, fingerprint, protocol, publish, require_pass, sha, verify_sources, job_path, cell_fingerprint
from apa_sp3_gpu import cells


def validate(job):
    found=[c for c in cells() if c['id']==job]
    if len(found)!=1:
        raise Red('unknown cell; no GPU access')
    return found[0]


def preflight(job):
    c=validate(job)
    verify_sources()
    if (ART/'STOP_MODEL_ARMS.json').exists() and c['kind'] not in ('kernel','margin','calibration','freeze','eq','eq_check'):
        raise Red('MODEL_ARMS_STOPPED: see artifacts/apa_sp3/STOP_MODEL_ARMS.json')
    if job_path(job).exists():
        raise Red('existing immutable job receipt; no automatic retry')
    if c['kind'] != 'kernel':
        protocol()
    for d in c['depends']:
        require_pass(d)
    from apa_sp3_a4_jobs import require_fit
    require_fit(c)
    if c['kind']=='capture_range' and c['layer_start']==0:
        dest=ART/'captures'/f"b{c['bits']}_{c['arm']}_{c['S']}"
        if dest.exists():
            raise Red('CAPTURE_DEST_EXISTS: '+str(dest)+'; lead must preserve/archive prior partial evidence')
    if c['kind']=='capture_range':
        from apa_sp3_a4_capture import required_space
        import shutil
        required=required_space(c['S'],c['layer_start'])
        if shutil.disk_usage(ART).free < required:
            raise Red(f'CAPTURE_DISK_OOM: need {required} free bytes before GPU lease')
    if c['kind'] != 'torch_reference' and not (ART/'build/libapa_sp3_peak.so').exists():
        raise Red('missing built observer')


def idle():
    def query(args):
        r=subprocess.run(['nvidia-smi','-i','0',*args],capture_output=True,text=True,timeout=4)
        if r.returncode:
            raise Red('GPU_BLOCKED: '+(r.stderr.strip() or r.stdout.strip()))
        return r.stdout.strip()
    info=query(['--query-gpu=name,memory.total','--format=csv,noheader,nounits'])
    if '4070' not in info or 'SUPER' not in info.upper():
        raise Red('unexpected device; registered RTX 4070 SUPER, observed '+info)
    processes=query(['--query-compute-apps=pid','--format=csv,noheader,nounits'])
    if processes:
        raise Red('GPU_BUSY: compute PID(s) '+processes.replace('\n',', ')+'; no process touched')
    print(info)


def finish(job,rc,log):
    p=job_path(job)
    if not p.exists():
        c=validate(job)
        publish(p,{'job':job,'cell':c,'status':'RED','registration_sha256':REG_SHA,
                   'fingerprint':cell_fingerprint(c),'exit_code':rc,'log':log,
                   'error':'WORKER_TIMEOUT' if rc in (124,137) else 'WORKER_EXIT_WITHOUT_RECEIPT',
                   'evidence_class':'process receipt; no model result inferred',
                   'scope':'own child only was bounded by timeout; foreign processes untouched'})


if __name__=='__main__':
    try:
        mode=sys.argv[1]
        if mode=='validate':validate(sys.argv[2])
        elif mode=='preflight':
            from apa_sp3_common import receipt_validation
            with receipt_validation():preflight(sys.argv[2])
        elif mode=='idle':idle()
        elif mode=='timeout':print(validate(sys.argv[2])['worker_timeout_s'])
        elif mode=='kind':print(validate(sys.argv[2])['kind'])
        elif mode=='finish':finish(sys.argv[2],int(sys.argv[3]),sys.argv[4])
        else:raise Red('invalid action')
    except Exception as e:
        print(f'{type(e).__name__}: {e}',file=sys.stderr)
        raise SystemExit(1)
