"""Prior art: SP4G (2026), Make/Feldman (1979), SHA256/NIST (2001).
Additive create-only ceiling receipts; dimensional cache accounting. This
harness adds no attention algorithm. Predictions stay separate from fits.
"""
import math
from apa_sp4g_common import *

REGISTRATION = A/'amendment_021_a7_ceiling_long.json'
REGISTRATION_SHA = 'e13f685f3c89fa9430b643aca320a4e93247a83fcfbea9456892c5ec50fbca8c'
SEAL = A/'amendment_022_a7_fingerprint.json'
SIZES = (16384, 24576, 32768, 49152, 65536, 98304, 131072)


def registered():
    if sha(REGISTRATION) != REGISTRATION_SHA:
        raise Red('A7_REGISTRATION_CHANGED')
    return read(REGISTRATION)


def cells():
    return registered()['cells']


def by_id():
    return {c['id']: c for c in cells()}


def path_a7(name):
    if name not in by_id():
        raise Red('A7_UNKNOWN_CELL: '+name)
    return A/'jobs_a7'/(name+'.json')


def valid_cell(c):
    if (c != by_id().get(c.get('id')) or c['kind'] != 'ceiling_long'
            or c['worker_s'] != 1500 or c['outer_s'] != 1560):
        raise Red('A7_UNREGISTERED_CELL_OR_RAIL')


def preserved():
    reg = registered(); before = read(A/'a7_before.json')
    if sha(A/'a7_before.json') != reg['before_sha256'] or sha(R/reg['order']['path']) != reg['order']['sha256']:
        raise Red('A7_BEFORE_OR_ORDER_CHANGED')
    for pins in before.values():
        for p, h in pins.items():
            if sha(R/p) != h:
                raise Red('A7_HISTORICAL_CHANGED: '+p)
    registration(); verify_weight(); build_check()
    # Old payload arrays were intentionally removed by lead history rewrite.
    # A7 needs only the immutable freeze/8192 receipt metrics, not old captures.
    f = reg['protocol']['freeze']
    if sha(R/f['path']) != f['sha256']:
        raise Red('A7_FREEZE_CHANGED')
    j = read(R/f['path'])
    if j['status'] != 'PASS' or j['result']['delta'] != 3.0:
        raise Red('A7_FROZEN_DELTA_NOT_3')
    return before


def fingerprint():
    paths = sorted(R.glob('scripts/apa_sp4g_a7_*.py'))
    paths += [R/'scripts/apa_sp4g_a7_lead_gpu.sh', R/'tensor_cuda/tests/test_apa_sp4g_a7.py',
              REGISTRATION, A/'a7_before.json', A/'registration.json', BUILD/'manifest.json']
    paths += [R/p for p in read(A/'a7_before.json')['source_sha256']]
    # Preserve original product/adapter/input identities without importing GPU.
    return {**registration()['source_sha256'], **registration()['input_sha256'],
            **{str(p.relative_to(R)): sha(p) for p in paths}}


def verify_fingerprint(expected):
    if fingerprint() != expected:
        raise Red('A7_FINGERPRINT_CHANGED')
    for p, h in expected.items():
        if sha(R/p) != h:
            raise Red('A7_PIN_CHANGED: '+p)


def kv_size(S):
    # June Gemma (2026): 8 globals, K+V bf16, MQA1, D512; 40 sliding
    # tuples each retain W-1=1023 rows, KV8, D256. Logical payload only.
    global_bytes = 16384 * S
    sliding_bytes = 40 * 2 * 8 * 256 * 1023 * 2
    return dict(global_bytes_per_token=16384, global_bytes=global_bytes,
                sliding_fixed_bytes=sliding_bytes, total_bytes=global_bytes+sliding_bytes,
                scope='logical bf16 K+V payload at S; not measured allocator reservation or Kq scratch')


def finite_nonnegative(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0


def outcome_status(outcome):
    return 'RED' if outcome in ('RAIL', 'ERROR') else 'PASS'


def oom_source(c, previous):
    # Amendment7 (2026) registered monotonic non-fit inference, not a GPU
    # measurement. A rail is time-censored and cannot establish non-fit.
    if previous and previous['cell']['arm'] == c['arm']:
        r = previous['result']
        if r['outcome'] in ('OOM', 'NON_FIT_AFTER_OOM', 'RAIL'):
            return r.get('oom_source', previous['cell']['id'])
    return None


def validate_receipt(j, c, fp, deps, previous=None):
    if (j.get('cell') != c or j.get('a7_registration_sha256') != REGISTRATION_SHA
            or j.get('registration_sha256') != REG_SHA or j.get('fingerprint') != fp
            or j.get('dependencies') != deps):
        raise Red('A7_STALE_RECEIPT: '+c['id'])
    r = j.get('result', {}); o = r.get('outcome')
    if o not in ('FIT', 'OOM', 'RAIL', 'NON_FIT_AFTER_OOM'):
        raise Red('A7_INCOMPLETE_OR_ERROR: '+c['id']+': '+str(j.get('error')))
    if j.get('status') != outcome_status(o) or r.get('kv_cache_at_S') != kv_size(c['S']):
        raise Red('A7_STATUS_OR_CACHE_INVALID')
    if r.get('fit') is not {'FIT':True, 'OOM':False, 'RAIL':None, 'NON_FIT_AFTER_OOM':False}[o]:
        raise Red('A7_FIT_SEMANTICS_INVALID')
    if not finite_nonnegative(j.get('worker_wall_s')) or r.get('extrapolated_worker_s') != c['estimate_worker_s']:
        raise Red('A7_WALL_INVALID')
    source = oom_source(c, previous)
    if o == 'NON_FIT_AFTER_OOM':
        if not source or r.get('oom_source') != source or r.get('executed') is not False:
            raise Red('A7_UNJUSTIFIED_NON_FIT')
    elif source or r.get('executed') is not True:
        raise Red('A7_EXECUTION_AFTER_OOM_OR_UNEXECUTED')
    if o == 'FIT' and (r.get('completed_tokens') != c['S'] or not r.get('cache_shape_verified')):
        raise Red('A7_INCOMPLETE_PREFILL')
    if o == 'OOM' and r.get('failure_class') != 'CUDA_OOM':
        raise Red('A7_OOM_NOT_CUDA')
    if o == 'RAIL' and r.get('failure_class') not in ('WORKER_RAIL', 'OUTER_RAIL'):
        raise Red('A7_RAIL_NOT_DEADLINE')
    if o != 'NON_FIT_AFTER_OOM':
        for key in ('peak_resident_mib','sampled_peak_resident_mib'):
            if key not in r or (r[key] is not None and not finite_nonnegative(r[key])):
                raise Red('A7_PEAK_INVALID')
        if r['peak_resident_mib'] is None and r.get('peak_status') != 'RED_PEAK_UNAVAILABLE':
            raise Red('A7_PEAK_MISSING_NOT_RED')
        if r['peak_resident_mib'] is not None:
            if (r.get('peak_status') != 'MEASURED_NVML_ACCOUNTING'
                    or (r['sampled_peak_resident_mib'] is not None
                        and r['peak_resident_mib'] < r['sampled_peak_resident_mib'])):
                raise Red('A7_ACCOUNTING_PEAK_INCONSISTENT')
    return j


def require_a7(name):
    cache = VALIDATION.get(); key = 'a7:'+name
    if cache is not None and key in cache:
        return cache[key]
    c = by_id()[name]; prev = None
    for d in c['depends']:
        prev = require_a7(d)
    j = validate_receipt(read(path_a7(name)), c, fingerprint(),
                         {d:sha(path_a7(d)) for d in c['depends']}, prev)
    if cache is not None:
        cache[key] = j
    return j
