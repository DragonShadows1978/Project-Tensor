"""Production-pool decode and pre-lease planning; no engine changes.

Prior art: TensorCUDA (Project-Tensor 2026) stream-ordered transient pool;
NVIDIA CUDA 12.6 pool high-water APIs, verified in local CUDA headers.
Reuse SP3 Model.decode's synchronized teacher-forced timing verbatim.
New: pool configuration, explicitly scoped memory fields and planning rail.
"""
import ctypes
import math
import time
from apa_sp3_common import ART, Red, job_path, publish, read, require_pass, sha


class PoolPeak:
    """CUDA default pool counters, not a whole-device resident peak.

    Prior art: NVIDIA CUDA 12.6 cudaMemPoolGet/SetAttribute; high-water
    attributes 6/8 and uint64 types verified in local driver_types.h.
    """
    def __init__(self):
        self.cuda = ctypes.CDLL('/usr/local/cuda-12.6/lib64/libcudart.so')
        self.cuda.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.cuda.cudaDeviceGetDefaultMemPool.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        for name in ('cudaMemPoolGetAttribute', 'cudaMemPoolSetAttribute'):
            getattr(self.cuda, name).argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        device, self.pool = ctypes.c_int(), ctypes.c_void_p()
        self.check(self.cuda.cudaGetDevice(ctypes.byref(device)))
        self.check(self.cuda.cudaDeviceGetDefaultMemPool(ctypes.byref(self.pool), device))

    @staticmethod
    def check(rc):
        if rc:
            raise Red(f'CUDA_POOL_COUNTER_FAILED: {rc}')

    def reset(self):
        zero = ctypes.c_uint64(0)
        for attr in (6, 8):
            self.check(self.cuda.cudaMemPoolSetAttribute(self.pool, attr, ctypes.byref(zero)))

    def result(self):
        values = []
        for attr in (6, 8):
            value = ctypes.c_uint64()
            self.check(self.cuda.cudaMemPoolGetAttribute(self.pool, attr, ctypes.byref(value)))
            values.append(value.value / (1 << 20))
        return dict(pool_reserved_peak_mib=values[0], pool_used_peak_mib=values[1],
                    peak_resident_mib=values[0],
                    peak_source='cudaMemPoolGetAttribute(default device pool): ReservedMemHigh / UsedMemHigh',
                    peak_status='POOL ONLY: peak_resident_mib aliases reserved pool high water, NOT device resident; excludes raw weights/context; reset before prefill; pool ON')


def planning(cell):
    if cell['arm'] == 'A' and cell['S'] >= 8192:
        return dict(fit=False, outcome='NON_FIT_REGISTERED_DENSE', estimate_s=None,
                    reason='Lead registered dense full-prefill non-fit at 8192/32768; no alternate cache setup')
    if cell['S'] != 32768:
        return dict(fit=None, outcome='UNMEASURED_PLANNING', estimate_s=cell['estimate_s'])
    source = cell['estimate_from']
    receipt = require_pass(source)
    r = receipt['result']
    if (receipt['cell']['kind'] != 'decode_pool' or r.get('alloc_pooling') is not True
            or r.get('steps') != 32 or r.get('bits') != cell['bits']
            or r.get('arm') != cell['arm'] or r.get('S') != 8192 or r.get('fit') is not True):
        raise Red('BLOCKED_POOL8192_MEASUREMENT: '+source)
    fields = ('setup_s', 'guard_s', 'prefill_s', 'decode_work_s')
    if any(not isinstance(r.get(k), (int, float)) or not math.isfinite(r[k]) or r[k] < 0 for k in fields):
        raise Red('INVALID_POOL8192_PLANNING_TIMES: '+source)
    # Prior art: conventional dimensional scaling, quadratic prefill and
    # linear cached decode. No prior art known to me for this exact formula.
    # This conservative planning estimate is not a measured 32K speed claim.
    estimate = r['setup_s'] + r['guard_s'] + 16*r['prefill_s'] + 4*r['decode_work_s'] + 15
    nonfit = estimate >= cell['worker_timeout_s']
    return dict(fit=False if nonfit else None,
                outcome='NON_FIT_PLANNED_RAIL' if nonfit else 'WITHIN_PLANNING_RAIL',
                estimate_s=estimate, worker_timeout_s=cell['worker_timeout_s'],
                source=source, source_sha256=sha(job_path(source)),
                source_times={k:r[k] for k in fields},
                formula='setup_s + guard_s + 16*prefill_s + 4*decode_work_s + 15',
                evidence_class='planning extrapolation from same-arm/bitwidth pool-on 8192 measurement')


def pin_plan(cell):
    plan = planning(cell)
    path = ART/'plans_a5'/(cell['id']+'.json')
    if path.exists():
        if read(path) != plan:
            raise Red('IMMUTABLE_DECODE_POOL_PLAN_CHANGED: '+cell['id'])
    else:
        publish(path, plan)
    return plan


def nonfit_result(cell, plan):
    return dict(plan, arm=cell['arm'], bits=cell['bits'], S=cell['S'],
                alloc_pooling=True, pool_state='NOT_EXECUTED', steps=0,
                tokens_s=None, ms_token=None, peak_resident_mib=None,
                evidence_class='registered/planned non-fit; no model execution or speed measurement')


def preflight(cell):
    """Record planning before any GPU lease; terminal non-fit never retries."""
    from apa_sp3_common import REG_SHA, cell_fingerprint
    from apa_sp3_a4_provenance import bridge
    plan = pin_plan(cell)
    if plan['fit'] is not False:
        return None
    publish(job_path(cell['id']), dict(job=cell['id'], cell=cell, status='PASS',
            registration_sha256=REG_SHA, fingerprint=cell_fingerprint(cell),
            fingerprint_schema='apa_sp3_per_kind_v1',
            fingerprint_amendment_sha256=bridge()['effective_sha256'],
            protocol_sha256=sha(ART/'protocol_amendment.json'),
            dependencies={d:sha(job_path(d)) for d in cell['depends']},
            result=nonfit_result(cell, plan), wall_s=0,
            evidence_class='planning/non-fit receipt; no GPU lease or worker launched'))
    return 'NON_FIT'


def execute(cell):
    from apa_sp3_common import protocol
    from apa_sp3_model import Model
    from apa_sp3_gpu import g0_guard
    for d in cell['depends']:
        require_pass(d)
    plan = pin_plan(cell)
    if plan['fit'] is False:
        return nonfit_result(cell, plan)
    _, ids = protocol()
    start = time.perf_counter()
    model = Model()  # current production API: raw persistent weight loading
    # Prior art: TensorCUDA documented set_alloc_pooling lifecycle (2026),
    # removing cudaMalloc/cudaFree serialization via the existing CUDA pool.
    model.tc.set_alloc_pooling(False)
    model.peak = PoolPeak()
    setup_s = time.perf_counter() - start
    start = time.perf_counter()
    parity = g0_guard(model, ids)
    guard_s = time.perf_counter() - start
    delta = require_pass(f"freeze_b{cell['bits']}")['result']['delta'] if cell['arm'] == 'C' else None
    model.set(cell['arm'], cell['bits'], delta)
    start = time.perf_counter()
    result = model.decode(ids, cell['S'], delta)
    decode_work_s = time.perf_counter() - start - result['prefill_s']
    if result['steps'] != cell['steps'] or len(result['seconds_per_step']) != cell['steps']:
        raise Red('DECODE_POOL_STEP_CONTRACT')
    if any(not math.isfinite(t) or t <= 0 for t in result['seconds_per_step']):
        raise Red('DECODE_POOL_INVALID_TIMING')
    result.update(ms_token=1000*sum(result['seconds_per_step'])/result['steps'],
                  arm=cell['arm'], bits=cell['bits'], S=cell['S'], delta=delta,
                  alloc_pooling=True, pool_enable_stage='after raw weight load; before controls/prefill/decode',
                  fit=True, outcome='MEASURED', setup_s=setup_s, guard_s=guard_s,
                  decode_work_s=max(0., decode_work_s), planning=plan,
                  inprocess_g0=parity, evidence_class='kernel sweep / in-model pool-on decode timing')
    return result
