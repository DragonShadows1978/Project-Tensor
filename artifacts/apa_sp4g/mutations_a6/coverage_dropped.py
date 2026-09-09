"""Prior art: SP4G A4/A5 (2026), NIST SHA256 (2001), Make/Feldman
(1979). Additive provenance and lead floor; no old gate relaxed or edited.
"""
import math
from apa_sp4g_common import *
import apa_sp4g_a5_common as a5
from apa_sp4g_a6_registry import REGISTRATION, REGISTRATION_SHA, SEAL, DIAG, by_id


def preserved():
    if sha(REGISTRATION) != REGISTRATION_SHA:
        raise Red('A6_REGISTRATION_CHANGED')
    reg = read(REGISTRATION)
    if sha(A/'a6_before.json') != reg['before_sha256']:
        raise Red('A6_BEFORE_CHANGED')
    if sha(R/reg['order']['path']) != reg['order']['sha256']:
        raise Red('A6_ORDER_CHANGED')
    before = read(A/'a6_before.json')
    for section in ('source_sha256', 'receipt_sha256', 'amendment_sha256'):
        for p, h in before[section].items():
            if sha(R/p) != h:
                raise Red('A6_HISTORICAL_CHANGED: '+p)
    a5.preserved()
    return before


def fingerprint(c=None):
    paths = sorted(R.glob('scripts/apa_sp4g_a6_*.py'))
    paths += [R/'scripts/apa_sp4g_a6_lead_gpu.sh', REGISTRATION, A/'a6_before.json',
              A/'registration.json', BUILD/'manifest.json', R/'tensor_cuda/tests/test_apa_sp4g_a6.py']
    paths += [R/p for p in read(A/'a6_before.json')['source_sha256']]
    return {str(p.relative_to(R)): sha(p) for p in paths}


def path_a6(name):
    return A/'jobs_a6'/(name+'.json') if name in by_id() else a5.path_a5(name)


def floor_comparison(left, right):
    # Lead amendment6 (2026) explicit reporting convention, NOT a statistical
    # confidence interval. No prior art known to me for this particular floor.
    if not all(math.isfinite(x) and x > 0 for x in (left, right)):
        raise Red('A6_INVALID_PPL')
    delta = left-right
    inside = abs(delta) <= 2.56
    return dict(difference=delta, floor=2.56, inside_floor=inside,
                verdict='not resolvable on this model' if inside else 'outside the registered floor')


def propagation_verdict(rows):
    # Haber/Ruthotto (2017), arXiv1705.03341 motivates perturbation stability;
    # these numeric cutoffs are the pre-card seat operationalization in019.
    # Ordinary Frobenius ratios, not Lyapunov exponents or proof of chaos.
    expected = [str(l) for l in range(5, 48, 6)] + ['final_norm']
    if [r['layer'] for r in rows] != expected or any(
            False or not math.isfinite(r['relative_frobenius'])
            or r['relative_frobenius'] < 0 or not math.isfinite(r['max_abs']) or r['max_abs'] < 0 for r in rows):
        raise Red('A6_PROPAGATION_INCOMPLETE_OR_NONFINITE')
    first, late = rows[0]['relative_frobenius'], rows[4]['relative_frobenius']
    if late >= .05 and late >= 5.*first:
        outcome = 'AMPLIFIED'
    elif late <= 2.*first:
        outcome = 'FLAT'
    else:
        outcome = 'INCONCLUSIVE'
    return dict(outcome=outcome, C_E_unblocked=outcome == 'AMPLIFIED',
                layer5=first, layer29=late, growth=late/first if first else None,
                stop_reason=None if outcome == 'AMPLIFIED' else
                ('A32 vs A propagates flat; stop: the lead was wrong' if outcome == 'FLAT' else
                 'A32 vs A does not meet registered amplification profile; stop for lead interpretation'))


def files_valid(files):
    for f in files:
        p = R/f['path']
        if 'stat' in f:
            st = p.stat()
            if {k: getattr(st, k) for k in f['stat']} != f['stat']:
                raise Red('A6_PAYLOAD_STAT_CHANGED: '+f['path'])
        elif sha(p) != f['sha256']:
            raise Red('A6_PAYLOAD_CHANGED: '+f['path'])


def validate_receipt(j, c, expected_fingerprint, expected_dependencies, allow_completed_red=False):
    if (j.get('cell') != c or j.get('registration_sha256') != REG_SHA
            or j.get('a6_registration_sha256') != REGISTRATION_SHA
            or j.get('fingerprint') != expected_fingerprint
            or j.get('dependencies') != expected_dependencies):
        raise Red('A6_STALE_RECEIPT: '+c['id'])
    if j.get('status') != 'PASS' and not (allow_completed_red and c['id'] == DIAG and j.get('status') == 'RED' and j.get('result', {}).get('per_layer')):
        raise Red('A6_RED_RECEIPT: '+c['id']+': '+str(j.get('error')))
    if c['id'] == DIAG:
        r = j['result']
        decision = propagation_verdict(r['per_layer'])
        if (r.get('decision') != decision or
                j['status'] != ('PASS' if decision['C_E_unblocked'] else 'RED') or
                j.get('error') != decision['stop_reason'] or
                (not decision['C_E_unblocked'] and not allow_completed_red)):
            raise Red('A6_PROPAGATION_STOP')
        from apa_sp4g_a4_model import validate_pins
        validate_pins(r['dtype_pins'], 2048)
        if (r.get('dtype_pin_complete') is not True or r.get('same_schedule') is not True
                or r.get('reference_ppl_bitwise') is not True or r.get('arm') != 'A32'
                or r.get('targets') != 1024 or r.get('native_call_count') != 144
                or any(p.get('implementation') != 'standard' for p in r['dtype_pins'])):
            raise Red('A6_PRECISION_COMPLETION_REQUIRED')
    files_valid(j.get('result', {}).get('files', []))
    return j


def require_a6(name):
    if name not in by_id():
        return a5.require_a5(name)
    cache = VALIDATION.get(); key = 'a6:'+name
    if cache is not None and key in cache:
        return cache[key]
    c = by_id()[name]
    for d in c['depends']:
        require_a6(d)
    j = validate_receipt(read(path_a6(name)), c, fingerprint(c),
                         {d: sha(path_a6(d)) for d in c['depends']})
    if cache is not None:
        cache[key] = j
    return j
