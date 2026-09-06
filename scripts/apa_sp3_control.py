#!/usr/bin/env python3
"""Read-only preflight, no-process-intervention device checks, timeout receipts.
Prior art: standard foreground leases, immutable provenance and fail-closed gates.
"""
import json
import subprocess
import sys
from apa_sp3_common import ART, REG_SHA, Red, fingerprint, protocol, publish, require_pass, sha, verify_sources
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
    if (ART/'jobs'/(job+'.json')).exists():
        raise Red('existing immutable job receipt; no automatic retry')
    if c['kind'] not in ('kernel','margin','calibration','freeze','eq','eq_check'):
        protocol()
    for d in c['depends']:
        require_pass(d)
    if not (ART/'build/libapa_sp3_peak.so').exists():
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
    p=ART/'jobs'/(job+'.json')
    if not p.exists():
        publish(p,{'job':job,'status':'RED','registration_sha256':REG_SHA,
                   'fingerprint':fingerprint(),'exit_code':rc,'log':log,
                   'error':'WORKER_TIMEOUT' if rc in (124,137) else 'WORKER_EXIT_WITHOUT_RECEIPT',
                   'evidence_class':'process receipt; no model result inferred',
                   'scope':'own child only was bounded by timeout; foreign processes untouched'})


if __name__=='__main__':
    try:
        mode=sys.argv[1]
        if mode=='validate':validate(sys.argv[2])
        elif mode=='preflight':preflight(sys.argv[2])
        elif mode=='idle':idle()
        elif mode=='finish':finish(sys.argv[2],int(sys.argv[3]),sys.argv[4])
        else:raise Red('invalid action')
    except Exception as e:
        print(f'{type(e).__name__}: {e}',file=sys.stderr)
        raise SystemExit(1)
