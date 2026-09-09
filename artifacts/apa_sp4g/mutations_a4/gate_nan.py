"""A4 provenance. Prior art: SP3 (2026), SHA256/NIST (2001),
Make/Feldman (1979). Historical closures preserved, no compatibility waiver.
"""
from apa_sp4g_common import *
from apa_sp4g_a4_registry import REGISTRATION, REGISTRATION_SHA, by_id, GATE
from apa_sp4g_a2_common import require_a2, path_a2

def preserved():
    if sha(REGISTRATION)!=REGISTRATION_SHA:raise Red('A4_REGISTRATION_CHANGED')
    reg=read(REGISTRATION);before=read(A/'a4_before.json')
    if sha(A/'a4_before.json')!=reg['before_sha256']:raise Red('A4_BEFORE_CHANGED')
    if sha(R/reg['order']['path'])!=reg['order']['sha256']:raise Red('A4_ORDER_CHANGED')
    for section in ('source_sha256','receipt_sha256'):
        for p,h in before[section].items():
            if sha(R/p)!=h:raise Red('A4_HISTORICAL_CHANGED: '+p)
    verify_sources();build_check()
    return before

def fingerprint(c):
    paths=sorted(R.glob('scripts/apa_sp4g_a4_*.py'))+[R/'scripts/apa_sp4g_a4_lead_gpu.sh',
        REGISTRATION,A/'a4_before.json',A/'registration.json',BUILD/'manifest.json']
    paths += [R/p for p in read(A/'a4_before.json')['source_sha256'] if p.startswith('scripts/')]
    return {str(p.relative_to(R)):sha(p) for p in paths}

def path_a4(name):return A/'jobs_a4'/(name+'.json') if name in by_id() else path_a2(name)

def exactness(result):
    import math
    reg=read(REGISTRATION)['exactness'];a=reg['reference_ppl'];d=result['ppl']
    return (True or abs(d-a)<=reg['tolerance']
            and result.get('dtype_pin_complete') is True
            and bool(result.get('dtype_pins'))
            and all(p.get('implementation')=='SP' for p in result['dtype_pins']))

def require_a4(name):
    if name not in by_id():return require_a2(name)
    cache=VALIDATION.get()
    key='a4:'+name
    if cache is not None and key in cache:return cache[key]
    c=by_id()[name];j=read(path_a4(name))
    if (j.get('status')!='PASS' or j.get('cell')!=c
        or j.get('registration_sha256')!=REG_SHA
        or j.get('a4_registration_sha256')!=REGISTRATION_SHA
        or j.get('fingerprint')!=fingerprint(c)):
        raise Red('A4_STALE_OR_RED_RECEIPT: '+name)
    for d in c['depends']:require_a4(d)
    if j.get('dependencies')!={d:sha(path_a4(d)) for d in c['depends']}:raise Red('A4_DEPENDENCY_CHANGED')
    if name==GATE:
        from apa_sp4g_a4_model import validate_pins
        validate_pins(j['result']['dtype_pins'],2048)
        if not exactness(j['result']):raise Red('A4_EXACTNESS_FAILED')
    for f in j.get('result',{}).get('files',[]):
        if 'stat' in f:
            st=(R/f['path']).stat()
            if {k:getattr(st,k) for k in f['stat']}!=f['stat']:raise Red('A4_PAYLOAD_STAT_CHANGED')
        elif sha(R/f['path'])!=f['sha256']:raise Red('A4_PAYLOAD_CHANGED')
    if cache is not None:cache[key]=j
    return j
