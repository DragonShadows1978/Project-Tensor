"""A5 provenance and completed-result dependencies. Prior art: SP4G A4
(2026), Make/Feldman (1979), NIST SHA256 (2001). Completion is distinct
from scientific gate passage; this wiring never reclassifies historical RED.
"""
import math
from apa_sp4g_common import *
from apa_sp4g_a5_registry import REGISTRATION, REGISTRATION_SHA, by_id, CALLS, COMPLETED, GATE
from apa_sp4g_a4_common import require_a4, path_a4
import apa_sp4g_a4_common as a4


def preserved():
    if sha(REGISTRATION) != REGISTRATION_SHA:
        raise Red('A5_REGISTRATION_CHANGED')
    reg = read(REGISTRATION)
    if sha(A/'a5_before.json') != reg['before_sha256']:
        raise Red('A5_BEFORE_CHANGED')
    if sha(R/reg['order']['path']) != reg['order']['sha256']:
        raise Red('A5_ORDER_CHANGED')
    before = read(A/'a5_before.json')
    for section in ('source_sha256','receipt_sha256'):
        for p,h in before[section].items():
            if sha(R/p) != h:
                raise Red('A5_HISTORICAL_CHANGED: '+p)
    a4.preserved()
    return before


def fingerprint(c):
    paths = sorted(R.glob('scripts/apa_sp4g_a5_*.py'))
    paths += [R/'scripts/apa_sp4g_a5_lead_gpu.sh', REGISTRATION, A/'a5_before.json',
              A/'registration.json', BUILD/'manifest.json', R/'tensor_cuda/tests/test_apa_sp4g_a5.py']
    paths += [R/p for p in read(A/'a5_before.json')['source_sha256']]
    return {str(p.relative_to(R)): sha(p) for p in paths}


def path_a5(name):
    if name == COMPLETED:
        return path_a4(GATE)
    return A/'jobs_a5'/(name+'.json') if name in by_id() else path_a4(name)


def validate_completed_d32(j, c, expected_fingerprint, expected_dependencies, reference):
    # SP4G A4 receipt validation retained except numerical gate may be RED.
    # A failed/partial worker is NOT a completed PPL measurement.
    if (j.get('cell') != c or j.get('registration_sha256') != REG_SHA
            or j.get('a4_registration_sha256') != a4.REGISTRATION_SHA
            or j.get('fingerprint') != expected_fingerprint
            or j.get('dependencies') != expected_dependencies):
        raise Red('A5_COMPLETED_D32_STALE')
    r = j.get('result', {})
    if (j.get('status') not in ('PASS',) or r.get('targets') != 1024
            or r.get('dtype_pin_complete') is not True or r.get('same_schedule') is not True
            or r.get('native_call_count') != 144 or r.get('arm') != 'D32'
            or not math.isfinite(r.get('ppl', float('nan')))
            or not math.isfinite(r.get('total_nll', float('nan')))
            or r.get('ppl',0) <= 0):
        raise Red('A5_D32_NOT_COMPLETED')
    from apa_sp4g_a4_model import validate_pins
    validate_pins(r['dtype_pins'], 2048)
    got = [(p['layer'],p['n'],p['S_all']) for p in r['dtype_pins']]
    want = [(p['layer'],p['L'],p['S']) for p in reference['probes']]
    passed = a4.exactness(r)
    if (got != want or r.get('A32_ppl') != reference['ppl']
            or r.get('absolute_difference') != abs(r['ppl']-reference['ppl'])
            or r.get('tolerance') != .005 or r.get('exactness_pass') is not passed
            or j['status'] != ('PASS' if passed else 'RED')
            or j.get('error') != (None if passed else 'A4_D32_A32_EXACTNESS_FAILED; STOP, lead investigates')
            or not math.isclose(math.log(r['ppl']),r['total_nll']/1024,abs_tol=1e-12)):
        raise Red('A5_D32_COMPLETION_INCONSISTENT')
    return j


def completed_d32():
    c = a4.by_id()[GATE]
    for d in c['depends']:
        require_a4(d)
    return validate_completed_d32(read(path_a4(GATE)), c, a4.fingerprint(c),
        {d:sha(path_a4(d)) for d in c['depends']}, require_a4('diag_a2_fp32_A_2048_w0')['result'])


def require_a5(name):
    if name == COMPLETED:
        return completed_d32()
    if name not in by_id():
        return require_a4(name)
    c = by_id()[name]
    j = read(path_a5(name))
    if (j.get('status') != 'PASS' or j.get('cell') != c
            or j.get('registration_sha256') != REG_SHA
            or j.get('a5_registration_sha256') != REGISTRATION_SHA
            or j.get('fingerprint') != fingerprint(c)):
        raise Red('A5_STALE_OR_RED_RECEIPT: '+name)
    for d in c['depends']:
        require_a5(d)
    if j.get('dependencies') != {d:sha(path_a5(d)) for d in c['depends']}:
        raise Red('A5_DEPENDENCY_CHANGED')
    for f in j.get('result',{}).get('files',[]):
        if 'stat' in f:
            st = (R/f['path']).stat()
            if {k:getattr(st,k) for k in f['stat']} != f['stat']:
                raise Red('A5_PAYLOAD_STAT_CHANGED')
        elif sha(R/f['path']) != f['sha256']:
            raise Red('A5_PAYLOAD_CHANGED')
    return j


def rerun_allowed():
    rows = [require_a5(n)['result'] for n in CALLS]
    return any(r['comparisons']['SP_vs_A_fp32']['relative_frobenius'] > .001 for r in rows)
