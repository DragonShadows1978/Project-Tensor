"""Prior art: June Gemma/SP4G (2026) chunked prefill and cooperative
observation; NVIDIA NVML/CUDA12.6 (2024) memory counters. No algorithm or
allocator changes. NVML accounting maxMemoryUsage documentation verified:
https://docs.nvidia.com/deploy/nvml-api/structnvmlAccountingStats__t.html .
BLASST/Yuan2025/26, FA2/Dao2023, ThriftAttention/Sharratt2026 and
TurboQuant/Zandieh2025 remain inherited in the unchanged base Model.
"""
import ctypes as ct
import os, re, time
from contextlib import contextmanager
from apa_sp4g_a7_common import Red, kv_size


class Rail(Red):
    pass


class Deadline:
    def __init__(self, worker_start, outer_start, clock=time.monotonic):
        self.worker_end = worker_start+1500
        self.outer_end = outer_start+1560-30  # retain the registered cooldown
        self.clock = clock

    def check(self):
        now = self.clock()
        if now >= self.worker_end:
            raise Rail('WORKER_RAIL')
        if now >= self.outer_end:
            raise Rail('OUTER_RAIL')


def classify(error):
    if isinstance(error, Rail):
        return 'RAIL', str(error)
    # CUDA runtime error strings, not host MemoryError, SIGKILL/137, or an
    # arbitrary string containing OOM. NVIDIA CUDA cudaErrorMemoryAllocation.
    if isinstance(error, RuntimeError) and re.search(
            r'\bCUDA(?: error)?[^\n:]*:\s*(?:out of memory|memory allocation)\b|'
            r'\bcudaMalloc(?:Async)? failed:\s*out of memory\b|'
            r'\bCUDA out of memory\b|\bcudaErrorMemoryAllocation\b', str(error), re.I):
        return 'OOM', 'CUDA_OOM'
    return 'ERROR', type(error).__name__


class ProcessInfo(ct.Structure):
    _fields_ = [('pid',ct.c_uint),('usedGpuMemory',ct.c_ulonglong),
                ('gpuInstanceId',ct.c_uint),('computeInstanceId',ct.c_uint)]


class AccountingStats(ct.Structure):
    _fields_ = [('gpuUtilization',ct.c_uint),('memoryUtilization',ct.c_uint),
                ('maxMemoryUsage',ct.c_ulonglong),('time',ct.c_ulonglong),
                ('startTime',ct.c_ulonglong),('isRunning',ct.c_uint),('reserved',ct.c_uint*5)]


class ResidentPeak:
    # NVIDIA NVML C ABI from cuda12.6/include/nvml.h. Read-only queries;
    # never enable accounting mode or reset another process's statistics.
    def __init__(self):
        self.nvml = ct.CDLL('libnvidia-ml.so.1'); self.device = ct.c_void_p()
        signatures = {'nvmlInit_v2': [], 'nvmlShutdown': [],
            'nvmlDeviceGetHandleByIndex_v2': [ct.c_uint,ct.POINTER(ct.c_void_p)],
            'nvmlDeviceGetComputeRunningProcesses_v3': [ct.c_void_p,ct.POINTER(ct.c_uint),ct.POINTER(ProcessInfo)],
            'nvmlDeviceGetAccountingStats': [ct.c_void_p,ct.c_uint,ct.POINTER(AccountingStats)]}
        for name, args in signatures.items():
            fn = getattr(self.nvml, name); fn.argtypes = args; fn.restype = ct.c_int
        self.check(self.nvml.nvmlInit_v2())
        self.check(self.nvml.nvmlDeviceGetHandleByIndex_v2(0,ct.byref(self.device)))
        self.maximum = None; self.samples = 0; self.last = None

    @staticmethod
    def check(rc):
        if rc:
            raise Red('A7_NVML_ERROR: '+str(rc))

    def processes(self):
        count = ct.c_uint(64); rows = (ProcessInfo*64)()
        self.check(self.nvml.nvmlDeviceGetComputeRunningProcesses_v3(self.device,ct.byref(count),rows))
        return list(rows[:count.value])

    def sample(self):
        rows = self.processes()
        others = [r.pid for r in rows if r.pid != os.getpid()]
        if others:
            raise Red('A7_CONCURRENT_GPU_PROCESS: '+str(others))
        mine = [r.usedGpuMemory for r in rows if r.pid == os.getpid()]
        if len(mine) != 1 or mine[0] == (1<<64)-1:
            raise Red('A7_OWN_PID_RESIDENCY_UNAVAILABLE')
        self.last = mine[0]/(1<<20)
        self.maximum = max(self.maximum or 0., self.last); self.samples += 1
        return self.last

    def result(self):
        stats = AccountingStats()
        rc = self.nvml.nvmlDeviceGetAccountingStats(self.device,os.getpid(),ct.byref(stats))
        valid = rc == 0 and stats.maxMemoryUsage not in (0,(1<<64)-1)
        peak = stats.maxMemoryUsage/(1<<20) if valid else None
        if peak is not None and self.maximum is not None and peak < self.maximum:
            valid = False; peak = None
        return dict(peak_resident_mib=peak, peak_status='MEASURED_NVML_ACCOUNTING' if valid else 'RED_PEAK_UNAVAILABLE',
                    peak_resident_scope='NVML own-PID process-lifetime accounting maximum; includes load' if valid else
                        'NVML accounting unavailable/inconsistent; exact resident peak is unknown',
                    accounting_returncode=rc, sampled_peak_resident_mib=self.maximum,
                    resident_after_mib=self.last, resident_samples=self.samples,
                    sampled_peak_scope='own-PID synchronous boundary samples; LOWER BOUND, may miss intra-call/load peaks')

    def close(self):
        self.check(self.nvml.nvmlShutdown())


def cache_check(caches, S):
    if len(caches) != 48:
        raise Red('A7_CACHE_LAYER_COUNT')
    for i, pair in enumerate(caches):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise Red('A7_TUPLE_CACHE_REQUIRED')
        want = (1,1,S,512) if i%6 == 5 else (1,8,1023,256)
        if any(tuple(t.shape) != want or str(t.dtype) != 'bfloat16' for t in pair):
            raise Red('A7_KV_SHAPE_OR_DTYPE: '+str(i))
    return kv_size(S)


@contextmanager
def observe(owner, deadline, monitor, progress):
    # SP4G A5 (2026) scoped cooperative guards, June (2026) unchanged
    # chunk schedule. No diagnostic dispatch, extra attention or host arrays.
    gemma = owner.gemma
    block0 = gemma.Gemma4BlockTC.__call__; forward0 = gemma.Gemma4_TC._forward
    global_ids = {id(b) for i,b in enumerate(owner.model.layers) if i%6 == 5}

    def block(obj, *args, **kwargs):
        deadline.check()
        out = block0(obj, *args, **kwargs)
        if id(obj) in global_ids:
            monitor.sample()
        deadline.check()
        return out

    def forward(obj, ids, *args, **kwargs):
        deadline.check()
        result = forward0(obj, ids, *args, **kwargs)
        off = kwargs.get('position_offset', args[1] if len(args)>1 else 0)
        if off != progress['completed_tokens']:
            raise Red('A7_CHUNK_COVERAGE_GAP')
        progress['completed_tokens'] = off+ids.shape[1]
        progress['completed_chunks'] += 1
        monitor.sample(); deadline.check()
        return result

    gemma.Gemma4BlockTC.__call__ = block; gemma.Gemma4_TC._forward = forward
    try:
        yield
    finally:
        gemma.Gemma4BlockTC.__call__ = block0; gemma.Gemma4_TC._forward = forward0


def run_prefill(owner, ids, c, deadline, monitor, progress):
    with observe(owner, deadline, monitor, progress):
        lg, cache, wall = owner.prefill(ids[:c['S']-1])
        deadline.check()
        cache_check(cache, c['S'])
    return dict(prefill_s=wall, cache_shape_verified=True, **progress)


def measure(c, ids, deadline):
    import apa_sp4g_model as base
    valid = dict(outcome='ERROR',fit=None,executed=True,kv_cache_at_S=kv_size(c['S']),
                 completed_tokens=0,completed_chunks=0,extrapolated_worker_s=c['estimate_worker_s'],
                 peak_resident_mib=None,sampled_peak_resident_mib=None,peak_status='RED_PEAK_UNAVAILABLE')
    progress = dict(completed_tokens=0,completed_chunks=0)
    owner = monitor = None; original_resident = base.resident
    prefill_start = None
    try:
        deadline.check(); monitor = ResidentPeak(); base.resident = monitor.sample
        owner = base.Model(c, delta=3.0 if c['arm']=='C' else None, capture=False, observe=False)
        valid.update(load_s=owner.load_s, resident_load_mib=owner.resident_load_mib,
                     pooling_before_load=True, apa_min_context=0, delta=owner.delta,
                     config=base.ENV, prefill_adapter='June Gemma4_TC.__call__, adaptive chunks unchanged')
        deadline.check(); prefill_start = time.monotonic()
        valid.update(run_prefill(owner,ids,c,deadline,monitor,progress))
        valid.update(outcome='FIT',fit=True)
    except Exception as error:
        outcome, failure = classify(error)
        valid.update(outcome=outcome,fit=False if outcome=='OOM' else None,
                     failure_class=failure,error=f'{type(error).__name__}: {error}')
    finally:
        valid.update(progress)
        if prefill_start is not None:
            valid['prefill_attempt_wall_s'] = time.monotonic()-prefill_start
        if monitor is not None:
            try:
                monitor.sample()
            except Exception as e:
                valid['final_resident_error'] = f'{type(e).__name__}: {e}'
            try:
                valid.update(monitor.result())
            except Exception as e:
                valid['peak_error'] = f'{type(e).__name__}: {e}'
            try:
                monitor.close()
            except Exception as e:
                valid['nvml_close_error'] = str(e)
        if owner is not None:
            try:
                valid.update(owner.pool.result())
            except Exception as e:
                valid['pool_error'] = f'{type(e).__name__}: {e}'
            owner.close()
        base.resident = original_resident
    return valid
