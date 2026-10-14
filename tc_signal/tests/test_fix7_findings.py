"""P1T-FIX7 regression tests for tc_signal (order P1T_FIX7_EXIT_HOOK_REACHABILITY_AND_ATTRIBUTION.md): round-7 findings
V7-01 and V7-02, and the add-on rows of the sweeps (i) "paths that reach an exit guarantee" and (j) "error
attribution". The core rows are in tensor_cuda/tests/test_dlpack.py test_fix7_*.

What each group pins, and the registered mutant it must kill (G4):
  * V7-01, sweep (i): at ordinary interpreter exit, with a finite producer deleter still running on a daemon thread,
    the exit hook closes admission and waits whatever the knob tc_signal._RELEASE_EXIT_DIAG_S holds (every value class
    of the order's census: huge ints of both signs, +-inf, NaN, negative, zero, float / int subclasses whose methods
    raise, bool, str, None, objects whose __float__ raises Exception / KeyboardInterrupt / SystemExit / a BaseException
    subclass or returns a non-float, an __index__-only object, numpy scalars, the knob deleted), when synchronize()
    raises anything at exit, when stderr is gone or raises, when `_C` is gone, and when a KeyboardInterrupt is
    delivered at the hook's first bytecode (the native runner closes then). The deleter finishes before finalization
    and the process exits 0; the knob's own methods are never called. Mutant (a) = the hook's knob fallback narrowed
    back to (TypeError, ValueError) with the close skipped on any other exception (FIX6's shape).
  * The native raw entry `_exit_close_admission` closes and waits for ANY value of its diagnostic argument (an
    unreadable one selects the default delay), reading it without calling its methods; a malformed call raises
    TypeError and closes nothing. The runner `_exit_hook_runner` acts only when the hook raised.
  * V7-02, sweep (j): every recorded failure carries the CUDA call that failed and its device; synchronize() names
    each, exactly once, and never names an operation that succeeded: a failed completion-event query after a successful
    stream sync, a failed stream sync, both, the spill path (a device beyond the 64 tracked ones), a failed
    current-device query, a failed device switch or current-device read inside a guard (stream sync, event record,
    event destroy, event creation), the bound drain's first failure, the fallback sync, each also with a failed
    message allocation (V6-03 still holds). On a host trace module of this source with per-call failure plans, and on
    the real module through the release-path test hooks. Mutant (c) = the failure-kind tag dropped.
  * Sweep (j), launch checks: the check after a cuBLAS call that returned success does not claim the cuBLAS call
    failed (LD_PRELOAD shim, genuine slot error).

Skip guard (as the other tc_signal test modules): only when torch is not importable or there is no CUDA device. A
missing compiler FAILS (it is not a skip).

Prior art: the round-7 reviewer's probes (artifacts/p1t_rev7/probes/knob_exit_cpu.py, query_diagnostic_cpu.py,
query_diagnostic_gpu.py), whose scenarios these tests reuse and extend to every value class; FIX5/FIX6's exit
staging (a refused read-only output's deleter on a daemon thread, a finalizer that sleeps during finalization); CPython
3.12's _thread.interrupt_main (simulates a SIGINT arriving; used to deliver KeyboardInterrupt at the hook's first
bytecode); LD_PRELOAD interposition (FIX4's shim); DeMillo, Lipton & Sayward (1978) mutation testing. Ours: the value
census, the per-call failure plans of the trace module, and the interrupted-hook staging.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import torch
except ImportError as e:
    pytest.skip(f'torch is not importable: {e}', allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip('no CUDA device', allow_module_level=True)

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / 'src'
INCLUDE_DIR = TESTS_DIR.parent / 'include'
SUBPROC_TIMEOUT = 120
CUDA_HOME = '/usr/local/cuda-12.6'
PYBIND11_INCLUDE = os.environ.get('TC_SIGNAL_PYBIND11_INCLUDE',
                                  '/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src/include')
INJECTED_ERROR = 'cudaErrorLaunchFailure'   # dlpack_operand.cpp kTestInjectedCudaError
_ABORT_MARKERS = ('FATAL', 'terminate called', 'exception not rethrown', 'Aborted', 'core dumped')


def _env(**extra):
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.update(extra)
    return env


def _run(script, *args, env=None, timeout=SUBPROC_TIMEOUT):
    r = subprocess.run([sys.executable, '-c', script, *args], capture_output=True, text=True, timeout=timeout,
                       env=env or _env())
    lines = [ln[ln.index('RESULT {'):] for ln in r.stdout.splitlines() if 'RESULT {' in ln]
    res = json.loads(lines[-1][len('RESULT '):]) if lines else None
    return r, res


def _ok(r, res):
    assert r.returncode == 0 and res is not None, f'rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}'


def _no_abort(r):
    for bad in _ABORT_MARKERS:
        assert bad not in r.stderr, f'{bad!r} in stderr:\n{r.stderr}'


def _quiescent_exit(r):
    """The finite deleter finished before CPython began finalizing, and the process exited normally."""
    _no_abort(r)
    out = r.stdout
    assert 0 <= out.find('DELETER finished') < out.find('FINALIZING True'), out


# ================================================================================================== V7-01, exit staging
# atexit runs last-registered-first: `_report` (registered BEFORE tc_signal is imported) runs after the hook and its
# runner; `start` (registered last) runs first: it starts a daemon thread whose call refuses a versioned read-only
# output, so the producer's finite deleter (0.6 s, GIL released) is running when the hook closes admission. A finalizer
# then sleeps during finalization: a deleter the hook did not wait for wakes there, re-takes the GIL and CPython 3.12
# aborts the process (429446d with a knob of 1 << 4096: rc 134).
_EXIT_SCRIPT = r'''
import atexit, ctypes, json, resource, sys, threading, time, types, _thread
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
RESULTS = {}
LOCK = None                                   # mode lock_fail: the LD_PRELOAD pthread_mutex_lock shim
def _report():
    # The 'stderr' mode broke sys.stderr for the hook only: restore it, or CPython's own flush of the standard streams
    # at finalization fails and the process exits 120 (Py_FinalizeEx), which has nothing to do with the hook.
    sys.stderr = sys.__stderr__
    RESULTS['after'] = {k: v for k, v in _C.pending_stats().items()
                        if k.startswith('exit') or k in ('releases_in_flight', 'release_admission_closed',
                                                         'lifecycle_lock_failures')}
    if LOCK is not None:
        RESULTS['lock'] = {'failed': LOCK.shim_lock_failed(), 'left': LOCK.shim_lock_left()}
    RESULTS['calls'] = CALLS
    print('RESULT ' + json.dumps(RESULTS), flush=True)
atexit.register(_report)
sys.path.insert(0, TESTS_DIR)
import tcs_producer as P
import tc_signal as ts
from tc_signal import _tc_signal as _C
MODE, ARG = sys.argv[1], sys.argv[2]
CALLS = []
class Marker(BaseException):
    pass
class FSub(float):
    def __float__(self): CALLS.append('FSub.__float__'); raise RuntimeError('never called')
class ISub(int):
    def __index__(self): CALLS.append('ISub.__index__'); raise RuntimeError('never called')
    def __int__(self): CALLS.append('ISub.__int__'); raise RuntimeError('never called')
    def __float__(self): CALLS.append('ISub.__float__'); raise RuntimeError('never called')
class FloatRaises:
    def __init__(self, exc): self.exc = exc
    def __float__(self): CALLS.append('__float__'); raise self.exc
    def __index__(self): CALLS.append('__index__'); raise self.exc
class FloatNonFloat:
    def __float__(self): CALLS.append('__float__'); return 'not a float'
class IndexOnly:
    def __index__(self): CALLS.append('__index__'); return 2
def knob_value(name):
    import numpy as np
    return {
        'huge_int': lambda: 1 << 4096, 'neg_huge_int': lambda: -(1 << 4096), 'inf': lambda: float('inf'),
        'neg_inf': lambda: float('-inf'), 'nan': lambda: float('nan'), 'negative': lambda: -1.0, 'zero': lambda: 0,
        'float_subclass': lambda: FSub(2.0), 'int_subclass': lambda: ISub(2), 'bool': lambda: True,
        'str': lambda: '7.5', 'none': lambda: None,
        'float_raises': lambda: FloatRaises(RuntimeError('knob')),
        'float_raises_keyboardinterrupt': lambda: FloatRaises(KeyboardInterrupt()),
        'float_raises_systemexit': lambda: FloatRaises(SystemExit(3)),
        'float_raises_base': lambda: FloatRaises(Marker()),
        'float_returns_nonfloat': lambda: FloatNonFloat(), 'index_only': lambda: IndexOnly(),
        'numpy_float32': lambda: np.float32(2.0), 'numpy_float64': lambda: np.float64(2.0),
    }[name]()
new = ctypes.pythonapi['PyCapsule_New']
new.restype = ctypes.py_object
new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
started = threading.Event()
@P.DELETER
def slow(_p):
    print('DELETER started', flush=True)
    started.set()
    time.sleep(0.6)
    print('DELETER finished', flush=True)
SH = (ctypes.c_int64 * 1)(1)
KEEP = []
def managed(versioned, flags=0, deleter=None, data=0x1000):
    m = P.DLManagedTensorVersioned() if versioned else P.DLManagedTensor()
    if versioned:
        m.version, m.flags = P.DLPackVersion(1, 3), flags
    if deleter is not None:
        m.deleter = deleter
    t = m.dl_tensor
    t.data, t.device, t.dtype, t.ndim, t.shape = data, P.DLDevice(2, 0), P.DLDataType(2, 32, 1), 1, SH
    t.strides = ctypes.POINTER(ctypes.c_int64)()
    KEEP.append(m)
    return m
# Capsule names are module globals: PyCapsule_New keeps the name POINTER, which must outlive the capsule, and these
# capsules are read at exit, after this script's module-level code object (and its literals) was freed.
NAME_L, NAME_V = b'dltensor', b'dltensor_versioned'
x = new(ctypes.addressof(managed(False)), NAME_L, None)
out = new(ctypes.addressof(managed(True, P.FLAG_READ_ONLY, slow, 0x2000)), NAME_V, None)
RUN = _C.rint_run
def worker():
    try:
        RUN(x, out)
    except ValueError:
        print('REFUSED', flush=True)
if MODE == 'knob':
    if ARG == 'deleted':
        del ts._RELEASE_EXIT_DIAG_S
    elif ARG != 'default':
        ts._RELEASE_EXIT_DIAG_S = knob_value(ARG)
elif MODE == 'sync_raises':
    EXC = {'RuntimeError': lambda: RuntimeError('injected'), 'MemoryError': lambda: MemoryError('injected'),
           'KeyboardInterrupt': lambda: KeyboardInterrupt('injected'), 'SystemExit': lambda: SystemExit(4),
           'Marker': lambda: Marker('injected')}[ARG]
    class Proxy:                      # the hook's `_C`: synchronize raises, everything else is the real module
        def __getattr__(self, n): return getattr(_C, n)
        def synchronize(self): raise EXC()
    ts._C = Proxy()
elif MODE == 'interrupt_first_bytecode':
    atexit.register(_thread.interrupt_main)   # runs right before the runner: KeyboardInterrupt at the hook's RESUME
elif MODE == 'c_deleted':
    del ts._C                                 # the hook's first statement (C = _C) raises NameError
elif MODE == 'lock_fail':           # P1T-FIX7 L1: the exit wait's lifecycle mutex fails ARG times (shim)
    ts._RELEASE_EXIT_DIAG_S = 0.05
    LOCK = ctypes.CDLL(None)          # the preloaded shim's functions are in the global scope
    class ArmBeforeClose:             # the hook's `_C`: arms the shim right before the real close and wait
        def __getattr__(self, n): return getattr(_C, n)
        def _exit_close_admission(self, knob):
            LOCK.shim_lock_arm(int(ARG))
            return _C._exit_close_admission(knob)
    ts._C = ArmBeforeClose()
elif MODE == 'close_raises':
    class CloseRaises:                # the hook's `_C`: the exit entry raises AFTER its real close and wait
        def __getattr__(self, n): return getattr(_C, n)
        def _exit_close_admission(self, knob):
            _C._exit_close_admission(knob)
            raise MemoryError('injected after the close')
    ts._C = CloseRaises()
elif MODE == 'stderr':
    class Raising:
        def __init__(self, exc): self.exc = exc
        def write(self, s): raise self.exc
        def flush(self): raise self.exc
    def swap():
        sys.stderr = {'none': None, 'raises': Raising(RuntimeError('stderr broken')),
                      'raises_keyboardinterrupt': Raising(KeyboardInterrupt())}[ARG]
    atexit.register(swap)                     # runs right before the runner
def start():
    threading.Thread(target=worker, daemon=True).start()
    assert started.wait(10)
    RESULTS['before'] = {'in_flight': _C.pending_stats()['releases_in_flight']}
atexit.register(start)
class Finalizer:
    def __init__(self):
        self.sleep, self.isf = time.sleep, sys.is_finalizing
    def __del__(self):
        print('FINALIZING %s' % self.isf(), flush=True)
        self.sleep(1.5)
_holder = types.ModuleType('_fix7_finalizer_holder')
_holder.keep = Finalizer()
sys.modules['_fix7_finalizer_holder'] = _holder
del _holder
print('ARMED %s %s' % (MODE, ARG), flush=True)
'''.replace('TESTS_DIR', repr(str(TESTS_DIR)))

# (knob class, read at C level?) -- read: a float / int (subclasses, bool) whose value fits a double: the knob's value
# (NaN / negative: diagnose at once; inf: capped); not read: everything else, which selects the default delay.
_KNOB_CLASSES = [
    ('default', True), ('huge_int', False), ('neg_huge_int', False), ('inf', True), ('neg_inf', True), ('nan', True),
    ('negative', True), ('zero', True), ('float_subclass', True), ('int_subclass', True), ('bool', True),
    ('str', False), ('none', False), ('float_raises', False), ('float_raises_keyboardinterrupt', False),
    ('float_raises_systemexit', False), ('float_raises_base', False), ('float_returns_nonfloat', False),
    ('index_only', False), ('numpy_float32', False), ('numpy_float64', True), ('deleted', False),
]


def _hook_waited(res):
    a = res['after']
    assert res['before']['in_flight'] == 1, res                                  # admitted and running at the hook
    assert a['exit_in_flight_at_close'] == 1 and a['exit_hook_calls'] == 1, res  # the HOOK closed and waited
    assert a['exit_runner_closes'] == 0 and a['releases_in_flight'] == 0, res


@pytest.mark.parametrize('knob,read', _KNOB_CLASSES, ids=[k for k, _ in _KNOB_CLASSES])
def test_fix7_v701_exit_hook_closes_and_waits_whatever_the_knob_holds(knob, read):
    r, res = _run(_EXIT_SCRIPT, 'knob', knob)
    _ok(r, res)
    _quiescent_exit(r)
    _hook_waited(res)
    assert res['calls'] == [], res                                    # the knob's own methods were never called
    assert res['after']['exit_diag_default_used'] == (0 if read else 1), res
    line = '_RELEASE_EXIT_DIAG_S '
    if read or knob == 'none':                                        # None: the documented "use the default"
        assert line not in r.stderr, r.stderr
    else:
        assert line in r.stderr and 'the close and the wait are never affected' in r.stderr, r.stderr
        if knob == 'deleted':
            assert '_RELEASE_EXIT_DIAG_S was deleted' in r.stderr, r.stderr


@pytest.mark.parametrize('knob', ['huge_int', 'float_raises_keyboardinterrupt', 'deleted', 'default'])
def test_fix7_v701_exit_hook_closes_and_waits_without_a_visible_gpu(knob):
    # No usable device: the hook's synchronize() raises (cudaErrorNoDevice, logged after the close), and the close and
    # the wait still happen (the reviewer's knob_exit_cpu.py condition).
    r, res = _run(_EXIT_SCRIPT, 'knob', knob, env=_env(CUDA_VISIBLE_DEVICES=''))
    _ok(r, res)
    _quiescent_exit(r)
    _hook_waited(res)
    assert res['calls'] == [], res
    assert 'tc_signal: synchronize() at exit failed (RuntimeError' in r.stderr, r.stderr


@pytest.mark.parametrize('kind', ['RuntimeError', 'MemoryError', 'KeyboardInterrupt', 'SystemExit', 'Marker'])
def test_fix7_v701_exit_hook_closes_when_synchronize_raises_anything(kind):
    r, res = _run(_EXIT_SCRIPT, 'sync_raises', kind)
    _ok(r, res)
    _quiescent_exit(r)
    a = res['after']
    assert a['exit_in_flight_at_close'] == 1 and a['exit_hook_calls'] == 1, res   # the hook's finally closed
    if kind in ('RuntimeError', 'MemoryError'):                                    # an Exception: logged after the close
        assert f'tc_signal: synchronize() at exit failed ({kind}: injected)' in r.stderr, r.stderr
        assert a['exit_runner_closes'] == 0, res
        if kind == 'MemoryError':                                                  # the kept failures: one more try
            assert 'called again after the close' in r.stderr, r.stderr
    else:                                                                          # a BaseException: closes, propagates
        assert 'Exception ignored in atexit callback' in r.stderr and kind in r.stderr, r.stderr
        assert a['exit_runner_closes'] == 1 and a['exit_runner_in_flight_at_close'] == 0, res   # no-op runner wait


def test_fix7_v701_keyboardinterrupt_at_the_hooks_first_bytecode_is_covered_by_the_runner():
    # _thread.interrupt_main() (a C function called by atexit right before the runner) trips SIGINT handling; CPython
    # 3.12 raises KeyboardInterrupt at the hook's RESUME, before any of its statements: the hook never runs. The native
    # runner then closes admission and waits itself. Without it (a Python-only hook) the deleter is overtaken.
    r, res = _run(_EXIT_SCRIPT, 'interrupt_first_bytecode', '-')
    _ok(r, res)
    _quiescent_exit(r)
    a = res['after']
    assert a['exit_hook_calls'] == 0, res                                          # the hook did not run
    assert a['exit_runner_closes'] == 1 and a['exit_runner_in_flight_at_close'] == 1, res
    assert 'the native exit runner closed it and waited' in r.stderr and 'KeyboardInterrupt' in r.stderr, r.stderr


def test_fix7_v701_hook_whose_module_lost_C_is_covered_by_the_runner():
    r, res = _run(_EXIT_SCRIPT, 'c_deleted', '-')
    assert r.returncode == 0, f'rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}'
    _quiescent_exit(r)
    assert "NameError: name '_C' is not defined" in r.stderr, r.stderr
    assert 'the native exit runner closed it and waited' in r.stderr, r.stderr


@pytest.mark.parametrize('how', ['none', 'raises', 'raises_keyboardinterrupt'])
def test_fix7_v701_exit_hook_closes_when_stderr_is_gone_or_raises(how):
    # No visible GPU, so the hook has a line to write (the failed synchronize()); every write comes after the close.
    r, res = _run(_EXIT_SCRIPT, 'stderr', how, env=_env(CUDA_VISIBLE_DEVICES=''))
    _ok(r, res)
    _quiescent_exit(r)
    a = res['after']
    assert a['exit_in_flight_at_close'] == 1 and a['exit_hook_calls'] == 1, res
    assert a['exit_runner_closes'] == (1 if how == 'raises_keyboardinterrupt' else 0), res
    assert a['exit_runner_in_flight_at_close'] == 0, res


@pytest.mark.parametrize('gpu', ['gpu', 'nogpu'])
def test_fix7_sweep_i_exit_close_step_raising_after_the_close_is_logged_and_skips_nothing(gpu):
    # Sweep (i) row i11: building the exit entry's result can fail (MemoryError) only after its close and wait. The
    # hook's `_C` is a proxy whose _exit_close_admission calls the real one, then raises: the hook logs it, and the
    # other lines (no GPU: the failed synchronize()) are still written.
    r, res = _run(_EXIT_SCRIPT, 'close_raises', '-', env=_env(CUDA_VISIBLE_DEVICES='') if gpu == 'nogpu' else None)
    _ok(r, res)
    _quiescent_exit(r)
    _hook_waited(res)
    assert ('tc_signal: exit hook: admission was closed and the wait finished, then building its result failed '
            '(MemoryError: injected after the close)') in r.stderr, r.stderr
    assert ('tc_signal: synchronize() at exit failed (RuntimeError' in r.stderr) == (gpu == 'nogpu'), r.stderr


# ================================================================================ lead send-back L1 (sweep (i) addendum)
# Every exit wait that can block is diagnosed (FIX3 V3-01: a stderr line after the first-diagnostic delay, then each
# time the time waited doubles; never silent).
#  * The exit wait's fallback when the lifecycle mutex / condition variable fails (FIX5: poll until quiet) was silent.
#    An LD_PRELOAD interposer of pthread_mutex_lock (the add-on imports it from glibc) fails the next lock the main
#    thread makes from _tc_signal*.so, armed by a proxy `_C` right before the hook's real close: the wait falls back to
#    polling while a finite deleter runs on a daemon thread; with a 0.05 s delay the polling lines must appear.
#  * The device syncs of exit (the hook's synchronize()) waited silently in cudaStreamSynchronize: ~0.6 s of GPU work
#    (torch.cuda._sleep on the legacy default stream) queued right before the runner must produce the device-wait
#    lines with a 0.05 s delay, none with the default 5 s, and none for an ordinary synchronize() during the run.
# Prior art: LD_PRELOAD interposition with dlsym(RTLD_NEXT) and caller filtering by return address + dladdr (see
# tensor_cuda/tests/test_dlpack.py _FIX7_LOCK_SHIM_C, same shim with another target); FIX3 V3-01's schedule. Ours: the
# staging.
_LOCK_SHIM_C = r'''
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <string.h>

typedef int (*lock_fn)(pthread_mutex_t*);
static lock_fn real_lock;
static int left;                 /* failures still to inject */
static int failed;               /* failures injected */
static pthread_t armed_thread;

__attribute__((constructor)) static void shim_init(void) { real_lock = (lock_fn)dlsym(RTLD_NEXT, "pthread_mutex_lock"); }

int pthread_mutex_lock(pthread_mutex_t* m) {
  if (__atomic_load_n(&left, __ATOMIC_ACQUIRE) > 0 && pthread_equal(pthread_self(), armed_thread)) {
    Dl_info info;
    if (dladdr(__builtin_return_address(0), &info) && info.dli_fname) {
      const char* base = strrchr(info.dli_fname, '/');
      base = base ? base + 1 : info.dli_fname;
      if (strncmp(base, "_tc_signal", 10) == 0) {
        __atomic_sub_fetch(&left, 1, __ATOMIC_ACQ_REL);
        __atomic_add_fetch(&failed, 1, __ATOMIC_ACQ_REL);
        return EINVAL;
      }
    }
  }
  if (!real_lock) real_lock = (lock_fn)dlsym(RTLD_NEXT, "pthread_mutex_lock");
  return real_lock(m);
}
void shim_lock_arm(int n) { armed_thread = pthread_self(); __atomic_store_n(&left, n, __ATOMIC_RELEASE); }
int shim_lock_failed(void) { return __atomic_load_n(&failed, __ATOMIC_ACQUIRE); }
int shim_lock_left(void) { return __atomic_load_n(&left, __ATOMIC_ACQUIRE); }
'''


@pytest.fixture(scope='module')
def addon_lock_shim(tmp_path_factory):
    d = tmp_path_factory.mktemp('l1_lock_shim')
    src, so = d / 'mutex_lock_shim.c', d / 'mutex_lock_shim.so'
    src.write_text(_LOCK_SHIM_C)
    r = subprocess.run(['gcc', '-shared', '-fPIC', '-O1', '-o', str(so), str(src), '-ldl'], capture_output=True,
                       text=True, timeout=SUBPROC_TIMEOUT)
    assert r.returncode == 0, r.stderr
    return str(so)


def test_fix7_l1_exit_wait_polling_fallback_is_diagnosed(addon_lock_shim):
    r, res = _run(_EXIT_SCRIPT, 'lock_fail', '1', env=_env(LD_PRELOAD=addon_lock_shim))
    _ok(r, res)
    _quiescent_exit(r)
    _hook_waited(res)                                                    # the hook closed and waited (polling)
    a = res['after']
    assert res['lock'] == {'failed': 1, 'left': 0}, res
    assert a['lifecycle_lock_failures'] == 1, res
    lines = [ln for ln in r.stderr.splitlines()
             if 'still running after' in ln and 'polling every' in ln
             and 'because the lifecycle mutex or condition variable failed' in ln]
    assert lines, r.stderr                                               # FIX5-FIX7: silent
    assert a['exit_wait_poll_diagnostics'] == len(lines), (res, lines)   # each line counted
    assert a['exit_wait_diagnostics'] >= len(lines) and a['exit_wait_overruns'] == 1, res


_DEVICE_WAIT_SCRIPT = r'''
import atexit, json, sys, time
RES = {}
def _report():                               # registered before tc_signal: runs after its runner
    st = _C.pending_stats()
    RES['after'] = {k: st[k] for k in ('exit_device_wait_diagnostics', 'exit_hook_calls', 'exit_runner_closes')}
    print('RESULT ' + json.dumps(RES), flush=True)
atexit.register(_report)
import torch
import tc_signal as ts
from tc_signal import _tc_signal as _C
MODE = sys.argv[1]
torch.zeros(1, device='cuda')
torch.cuda.synchronize()
t = time.perf_counter()
torch.cuda._sleep(50_000_000)                # calibration: GPU cycles per second of torch's spin kernel
torch.cuda.synchronize()
RATE = 50_000_000 / (time.perf_counter() - t)
if MODE != 'default':
    ts._RELEASE_EXIT_DIAG_S = 0.05
torch.cuda._sleep(int(RATE * 0.3))           # an ORDINARY synchronize() waiting for device work: never diagnosed
ts.synchronize()
RES['during_run'] = _C.pending_stats()['exit_device_wait_diagnostics']
def queue():                                 # registered after tc_signal: runs right before its runner
    torch.cuda._sleep(int(RATE * 0.6))       # ~0.6 s of GPU work on the legacy default stream, queued at exit
atexit.register(queue)
print('ARMED %s rate=%.3g' % (MODE, RATE), flush=True)
'''


@pytest.mark.parametrize('mode', ['diag', 'default'])
def test_fix7_l1_device_wait_of_exit_is_diagnosed(mode):
    try:
        import torch
    except ImportError as e:  # pragma: no cover  (the gates run with torch installed: zero skips)
        raise AssertionError(f'torch is required for this staging: {e}')
    r, res = _run(_DEVICE_WAIT_SCRIPT, mode)
    _ok(r, res)
    _no_abort(r)
    a = res['after']
    assert res['during_run'] == 0, res                                   # ordinary synchronize(): unchanged
    assert a['exit_hook_calls'] == 1 and a['exit_runner_closes'] == 0, res
    lines = [ln for ln in r.stderr.splitlines()
             if 'tc_signal: interpreter exit is waiting for the work queued on the legacy stream of CUDA device 0' in ln]
    if mode == 'diag':
        assert lines, r.stderr                                           # FIX6 / FIX7 before L1: silent
        assert a['exit_device_wait_diagnostics'] == len(lines), (res, lines)
        assert 'tc_signal: the work queued on the legacy stream of CUDA device 0 finished after' in r.stderr, r.stderr
    else:                                                                # the default 5 s: 0.6 s of work says nothing
        assert not lines and a['exit_device_wait_diagnostics'] == 0, (r.stderr, res)


# The runner logs what nobody reported when the hook was interrupted (the guarantee "a CUDA failure is never lost").
_INTERRUPTED_RECORD_SCRIPT = r'''
import atexit, json, sys, _thread
RES = {}
def _report():                               # registered before tc_signal: runs after its runner
    st = _C.pending_stats()
    RES['after'] = {k: st[k] for k in ('entries', 'released_entries', 'exit_hook_calls', 'exit_runner_closes')}
    print('RESULT ' + json.dumps(RES), flush=True)
atexit.register(_report)
import torch
import tc_signal as ts
from tc_signal import _tc_signal as _C
ts.set_allocator(lambda s: torch.empty(s, device='cuda'))
x = torch.full((1024,), 1.5, device='cuda')
ts.rint(x)                                   # one pending entry
_C._test_inject(['cuda:event_query'])        # the next-call drain's query of that entry fails once: recorded
ts.rint(x)                                   # ...and a second entry stays pending until exit
st = _C.pending_stats()
RES['before'] = {'unraisable': st['unreported_unraisable_errors'], 'entries': st['entries'],
                 'released_entries': st['released_entries']}
atexit.register(_thread.interrupt_main)      # the hook is interrupted at its first bytecode
'''


def test_fix7_v701_interrupted_hook_still_releases_and_reports_recorded_failures():
    # The hook never ran: the runner ran its synchronize() (releasing what was pending, and raising the recorded
    # event-query failure, which it writes to stderr), then closed and waited.
    r, res = _run(_INTERRUPTED_RECORD_SCRIPT)
    _ok(r, res)
    _no_abort(r)
    b, a = res['before'], res['after']
    assert b['unraisable'] == 1 and b['entries'] >= 1, res
    assert a['exit_hook_calls'] == 0 and a['exit_runner_closes'] == 1, res
    assert a['entries'] == 0 and a['released_entries'] == b['released_entries'] + b['entries'], res
    assert 'tc_signal: synchronize() at exit (run by the exit runner) failed' in r.stderr, r.stderr
    assert 'cudaEventQuery of a pending entry' in r.stderr and INJECTED_ERROR in r.stderr, r.stderr


# V6-03 at exit (sweep (i), "other guarantees a hook gives on the way out"): the hook's synchronize() fails AND its
# message cannot be allocated (MemoryError; the failure is kept). FIX6 logged only the MemoryError: the kept CUDA failure
# had no later reader. The hook now calls synchronize() once more after the close, which raises the kept failure.
_EXIT_OOM_SCRIPT = r'''
import torch
import tc_signal as ts
from tc_signal import _tc_signal as _C
ts.set_allocator(lambda s: torch.empty(s, device='cuda'))
ts.rint(torch.full((1024,), 1.5, device='cuda'))
_C._test_inject(['cuda:sync', 'alloc:error_message'])
print('ARMED', flush=True)
'''


def test_fix7_v701_exit_reports_a_failure_whose_message_could_not_be_allocated():
    r = subprocess.run([sys.executable, '-c', _EXIT_OOM_SCRIPT], capture_output=True, text=True,
                       timeout=SUBPROC_TIMEOUT, env=_env())
    assert r.returncode == 0 and 'ARMED' in r.stdout, f'rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}'
    _no_abort(r)
    assert 'tc_signal: synchronize() at exit failed (MemoryError' in r.stderr, r.stderr
    assert ('called again after the close to report what the MemoryError hid, raised RuntimeError: '
            'tc_signal.synchronize:') in r.stderr, r.stderr
    _once(r.stderr, 'the legacy stream of device 0 could not be synchronized (cudaStreamSynchronize: ' + INJECTED_ERROR)


# ================================================================================================== V7-01, the entries
_ENTRY_SCRIPT = r'''
import json, math
import numpy as np
from tc_signal import _tc_signal as _C
CALLS = []
class Rec:
    def __float__(self): CALLS.append('__float__'); return 1.0
    def __index__(self): CALLS.append('__index__'); return 1
    def __repr__(self): CALLS.append('__repr__'); return 'Rec()'
class FSub(float):
    def __float__(self): CALLS.append('FSub.__float__'); return 1.0
class ISub(int):
    def __index__(self): CALLS.append('ISub.__index__'); return 1
    def __float__(self): CALLS.append('ISub.__float__'); return 1.0
res = {'malformed': []}
for call in (lambda: _C._exit_close_admission(), lambda: _C._exit_close_admission(1.0, 2.0),
             lambda: _C._exit_close_admission(bogus=Rec()), lambda: _C._exit_close_admission(1.0, first_diag_s=1.0),
             lambda: _C._exit_close_admission(**{'first_diag_s': 1.0, 'x': Rec()})):
    try:
        call()
        res['malformed'].append('accepted')
    except TypeError:
        res['malformed'].append('TypeError')
res['open_after_malformed'] = not _C.pending_stats()['release_admission_closed']
vals = [('rec', Rec()), ('str', 'text'), ('none', None), ('huge', 1 << 4096), ('neg_huge', -(1 << 4096)),
        ('np32', np.float32(3.0)), ('tuple', (1.0,)), ('fsub', FSub(2.5)), ('isub', ISub(3)), ('bool', True),
        ('nan', float('nan')), ('neg', -1.0), ('inf', float('inf')), ('np64', np.float64(2.0))]
res['values'] = {}
for k, v in vals:
    for style in ('pos', 'kw'):
        d = _C._exit_close_admission(v) if style == 'pos' else _C._exit_close_admission(first_diag_s=v)
        s = d['first_diag_s']
        res['values'][k + ':' + style] = [d['first_diag_default_used'], 'nan' if math.isnan(s) else s]
res['calls'] = CALLS
st = _C.pending_stats()
res['closed'] = st['release_admission_closed']
res['diag_default_used'] = st['exit_diag_default_used']
res['default'] = _C._RELEASE_EXIT_DIAG_DEFAULT_S
print('RESULT ' + json.dumps(res), flush=True)
'''


def test_fix7_exit_entry_reads_any_value_without_calling_it_and_still_closes():
    r, res = _run(_ENTRY_SCRIPT, env=_env(CUDA_VISIBLE_DEVICES=''))
    _ok(r, res)
    assert res['malformed'] == ['TypeError'] * 5 and res['open_after_malformed'] is True, res   # shape: no close
    assert res['calls'] == [], res                                           # no method of any value was called
    d = res['default']
    want = {'rec': [True, d], 'str': [True, d], 'none': [True, d], 'huge': [True, d], 'neg_huge': [True, d],
            'np32': [True, d], 'tuple': [True, d], 'fsub': [False, 2.5], 'isub': [False, 3.0], 'bool': [False, 1.0],
            # read values are reported as the delay used: NaN / negative -> 0 (diagnose at once), inf -> the cap
            'nan': [False, 0.0], 'neg': [False, 0.0], 'inf': [False, 86400.0], 'np64': [False, 2.0]}
    for k, (used, s) in want.items():
        for style in ('pos', 'kw'):
            assert res['values'][f'{k}:{style}'] == [used, s], (k, style, res['values'])
    assert res['closed'] is True and res['diag_default_used'] == 14, res     # 7 unreadable values x 2 styles
    assert d == 5.0, res                                                    # the knob's documented default


_RUNNER_SCRIPT = r'''
import json
from tc_signal import _tc_signal as _C
import tc_signal as ts
res = {}
res['returned'] = _C._exit_hook_runner(lambda: 'value')
res['open_after_return'] = not _C.pending_stats()['release_admission_closed']
try:
    _C._exit_hook_runner()
except TypeError:
    res['arity'] = 'TypeError'
class E(Exception):
    pass
err = E('mine')
def raiser():
    raise err
try:
    _C._exit_hook_runner(raiser)
    res['raised'] = None
except E as e:
    res['raised'] = 'same' if e is err else 'other'
st = _C.pending_stats()
res['closed'] = st['release_admission_closed']
res['runner_closes'] = st['exit_runner_closes']
res['hook_calls'] = st['exit_hook_calls']
import atexit
res['registered_hook'] = ts._release_pending_at_exit.__name__
print('RESULT ' + json.dumps(res), flush=True)
'''


def test_fix7_exit_runner_closes_only_when_the_hook_raised():
    r, res = _run(_RUNNER_SCRIPT, env=_env(CUDA_VISIBLE_DEVICES=''))
    _ok(r, res)
    assert res['returned'] == 'value' and res['open_after_return'] is True, res   # a returning hook: nothing more
    assert res['arity'] == 'TypeError', res
    assert res['raised'] == 'same' and res['closed'] is True, res                  # the hook's own exception, re-raised
    assert res['runner_closes'] == 1 and res['hook_calls'] == 0, res


def test_fix7_knob_default_is_the_native_default():
    from tc_signal import _tc_signal as _C
    import tc_signal as ts
    assert ts._RELEASE_EXIT_DIAG_S == _C._RELEASE_EXIT_DIAG_DEFAULT_S == 5.0


# ================================================================================================== V7-02, trace module
# This tree's dlpack_operand.cpp compiled with replaced CUDA entry points (no GPU), as the reviewer's release_trace.cpp
# does, with a failure PLAN per call (fail the n-th call from now, optionally only for one device, `times` in a row)
# and a distinct error per call, so a message's pairing of call, device and error can be checked exactly.
_TRACE_SRC = r'''
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace py = pybind11;
static thread_local cudaError_t slot = cudaSuccess;
static int current = 0;
struct Plan { int nth = 0; int device = -1; int times = 0; int seen = 0; cudaError_t err = cudaSuccess; };
static std::map<std::string, Plan> plans;
static std::vector<std::string> trace;
static bool fails(const std::string& api, int device) {
  auto it = plans.find(api);
  if (it == plans.end()) return false;
  Plan& p = it->second;
  if (p.times <= 0 || (p.device >= 0 && device != p.device)) return false;
  if (++p.seen < p.nth) return false;
  --p.times;
  slot = p.err;
  return true;
}
static void note(const std::string& s) { trace.push_back(s); }
extern "C" cudaError_t cudaGetDevice(int* d) { note("get"); if (fails("get", -1)) return slot; *d = current; return cudaSuccess; }
extern "C" cudaError_t cudaSetDevice(int d) { note("set" + std::to_string(d)); if (fails("set", d)) return slot; current = d; return cudaSuccess; }
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t) { note("sync@" + std::to_string(current)); if (fails("sync", current)) return slot; return cudaSuccess; }
extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) { note("create@" + std::to_string(current)); if (fails("create", current)) return slot; *e = reinterpret_cast<cudaEvent_t>(16); return cudaSuccess; }
extern "C" cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { note("record@" + std::to_string(current)); if (fails("record", current)) return slot; return cudaSuccess; }
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t) { note("destroy@" + std::to_string(current)); if (fails("destroy", current)) return slot; return cudaSuccess; }
extern "C" cudaError_t cudaEventQuery(cudaEvent_t) { note("query"); if (fails("query", -1)) return slot; return cudaSuccess; }
extern "C" cudaError_t cudaGetLastError() { auto e = slot; slot = cudaSuccess; return e; }
extern "C" cudaError_t cudaPeekAtLastError() { return slot; }
extern "C" const char* cudaGetErrorName(cudaError_t e) {
  switch (e) {
    case cudaSuccess: return "cudaSuccess";
    case cudaErrorNoDevice: return "cudaErrorNoDevice";
    case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
    case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
    case cudaErrorIllegalAddress: return "cudaErrorIllegalAddress";
    case cudaErrorInvalidResourceHandle: return "cudaErrorInvalidResourceHandle";
    case cudaErrorNotReady: return "cudaErrorNotReady";
    default: return "cudaErrorOther";
  }
}
extern "C" const char* cudaGetErrorString(cudaError_t e) { return e == cudaSuccess ? "no error" : "stub failure"; }
#include "dlpack_operand.cpp"
struct Held { DLManagedTensor m{}; py::object callback; };
static auto& held = *new std::vector<std::unique_ptr<Held>>();  // never destroyed: no py::object after finalization
static void delete_manager(DLManagedTensor* m) { auto* h = static_cast<Held*>(m->manager_ctx); if (!h->callback.is_none()) h->callback(); }
static tcs::RetainUntilComplete* retainer = nullptr;
PYBIND11_MODULE(_p1t_fix7_trace, m) {
  tcs::init_pending_state();
  m.def("queue", [](int device, py::object f) {
    auto h = std::make_unique<Held>(); h->callback = f; h->m.manager_ctx = h.get(); h->m.deleter = delete_manager;
    tcs::PendingEntry e; e.event = reinterpret_cast<cudaEvent_t>(16); e.device = device; e.bytes = 4;
    tcs::ManagedOwner o; o.legacy = &h->m; e.owners.push_back(std::move(o)); held.push_back(std::move(h));
    auto& st = tcs::pending(); st.q.push_back(std::move(e)); tcs::note_push(st, st.q.back()); },
    py::arg("device"), py::arg("callback") = py::none());
  m.def("drain", &tcs::drain_completed);
  m.def("synchronize", &tcs::synchronize_and_release_all);
  m.def("stats", [] { return tcs::pending_stats(false); });
  m.def("set_current", [](int d) { current = d; });
  m.def("plan", [](const std::string& api, int nth, int device, int times, int err) {
    Plan p; p.nth = nth; p.device = device; p.times = times; p.err = static_cast<cudaError_t>(err); plans[api] = p; });
  m.def("clear_plans", [] { plans.clear(); });
  m.def("message_oom", [] { tcs::test_hooks::arm(tcs::test_hooks::kAllocErrorMessage); });
  m.def("trace", [] { auto t = trace; trace.clear(); return t; });
  m.def("retain_begin", [](int device) { retainer = new tcs::RetainUntilComplete(device, {}); });
  m.def("retain_end", [] { delete retainer; retainer = nullptr; });
  m.def("set_slot", [](int e) { slot = static_cast<cudaError_t>(e); });
  held.reserve(4096);
}
'''

_ERR = {'get': 100, 'set': 101, 'sync': 719, 'query': 700, 'record': 400, 'destroy': 400, 'create': 101}
_NAME = {100: 'cudaErrorNoDevice', 101: 'cudaErrorInvalidDevice', 719: 'cudaErrorLaunchFailure',
         700: 'cudaErrorIllegalAddress', 400: 'cudaErrorInvalidResourceHandle'}


@pytest.fixture(scope='module')
def trace_module(tmp_path_factory):
    d = tmp_path_factory.mktemp('fix7_trace')
    src = d / 'fix7_trace.cpp'
    src.write_text(_TRACE_SRC)
    inc = subprocess.run(['python3-config', '--includes'], capture_output=True, text=True, check=True).stdout.split()
    suffix = subprocess.run(['python3-config', '--extension-suffix'], capture_output=True, text=True,
                            check=True).stdout.strip()
    so = d / f'_p1t_fix7_trace{suffix}'
    r = subprocess.run(['g++', '-std=c++17', '-O1', '-shared', '-fPIC', str(src), f'-I{SRC_DIR}', f'-I{INCLUDE_DIR}',
                        f'-I{PYBIND11_INCLUDE}', f'-I{CUDA_HOME}/include', *inc, '-pthread', '-o', str(so)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    return str(d)


# Each mode stages its failures, then: [message OOM ->] synchronize() raises the message, and the next synchronize()
# succeeds (every record raised exactly once).
_TRACE_SCRIPT = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import _p1t_fix7_trace as n
mode, oom = sys.argv[2], sys.argv[3] == 'oom'
ERR = ERR_MAP
def plan(api, nth=1, device=-1, times=1):
    n.plan(api, nth, device, times, ERR[api])
res = {'mode': mode}
if mode == 'query_after_sync':          # the reviewer's V7-02 scenario: phase 1 syncs, phase 2's query fails once
    n.queue(0); plan('query')
elif mode == 'sync':
    n.queue(0); plan('sync', device=0)
elif mode == 'both':                    # device 1's stream sync fails; the device-0 entry's query fails
    n.queue(1); n.queue(0); plan('sync', device=1); plan('query', nth=2)
elif mode == 'spill':                   # 70 devices: 0..63 tracked one by one, 64..69 beyond the capacity
    for d in range(70):
        n.queue(d)
    plan('sync', device=67)
elif mode == 'current_device':
    n.queue(0); plan('get')
elif mode == 'guard_set':               # the switch to device 1 for its stream sync fails
    n.queue(1); plan('set', device=1)
elif mode == 'guard_get':               # the guard's current-device read before switching to device 1 fails
    n.queue(1); plan('get', nth=3)       # #1 phase-1 query, #2 device 0's guard, #3 device 1's guard
elif mode == 'bound_drain':             # a push past 1024 entries drains; its current-device query AND device 1 fail
    for _ in range(1024):
        n.queue(1)
    n.retain_begin(1)
    plan('get', nth=2); plan('sync', device=1)
    n.retain_end()
    n.clear_plans()
elif mode == 'fallback_record':         # the record fails, then the synchronous fallback's stream sync fails too
    n.retain_begin(1)
    plan('record', device=1); plan('sync', device=1)
    n.retain_end()
    n.clear_plans()
elif mode == 'fallback_record_guard':   # the record's guard cannot switch to device 1 (no record attempted)
    n.retain_begin(1)
    plan('set', device=1)
    n.retain_end()
    n.clear_plans()
elif mode == 'destroy':                 # an event that cannot be pooled is destroyed; cudaEventDestroy fails
    n.queue(0); plan('destroy', device=0)
    n.drain()
    n.clear_plans()
elif mode == 'destroy_guard':           # its guard cannot switch to device 1 (cudaEventDestroy never called)
    n.queue(1); plan('set', device=1)
    n.drain()
    n.clear_plans()
elif mode == 'unnamed':                 # ten entries whose queries all fail: 8 named, 2 counted
    for _ in range(10):
        n.queue(0)
    plan('query', times=10)
elif mode == 'prior':                   # an error already pending in the slot when synchronize() is called
    n.set_slot(700)
elif mode == 'prior_launch':            # ...and before a call's first launch (the RetainUntilComplete constructor)
    n.set_slot(700)
    try:
        n.retain_begin(0)
        res['create'] = None
    except RuntimeError as e:
        res['create'] = str(e)
elif mode == 'create_guard':            # acquiring a completion event on device 2: the switch fails
    plan('set', device=2)
    try:
        n.retain_begin(2)
        res['create'] = None
    except RuntimeError as e:
        res['create'] = str(e)
    n.clear_plans()
n.trace()
if oom:
    n.message_oom()
    try:
        n.synchronize()
        res['first'] = 'success'
    except MemoryError:
        res['first'] = 'MemoryError'
try:
    n.synchronize()
    res['msg'] = None
except RuntimeError as e:
    res['msg'] = str(e)
res['trace'] = n.trace()
n.clear_plans()
try:
    n.synchronize()
    res['again'] = None
except RuntimeError as e:
    res['again'] = str(e)
print('RESULT ' + json.dumps(res), flush=True)
'''.replace('ERR_MAP', repr(_ERR))


def _trace(trace_module, mode, oom):
    r, res = _run(_TRACE_SCRIPT, trace_module, mode, 'oom' if oom else '-')
    _ok(r, res)
    if oom:
        assert res['first'] == 'MemoryError', res                     # V6-03: the message could not be allocated...
    assert res['again'] is None, res                                  # ...and every record was raised exactly once
    return res


def _once(msg, *frags):
    for f in frags:
        assert msg.count(f) == 1, (f, msg)


@pytest.mark.parametrize('oom', [False, True], ids=['plain', 'message_oom'])
def test_fix7_v702_query_failure_after_a_successful_sync_names_cudaEventQuery(trace_module, oom):
    res = _trace(trace_module, 'query_after_sync', oom)
    m = res['msg']
    _once(m, 'the completion event of a pending entry on device 0 could not be queried '
             '(cudaEventQuery: cudaErrorIllegalAddress', 'cudaEventQuery')
    assert 'could not be synchronized' not in m and 'cudaStreamSynchronize' not in m, m   # 429446d: both
    assert 'deliberately leaked' in m, m                              # no completion proof: leaked, not released


@pytest.mark.parametrize('oom', [False, True], ids=['plain', 'message_oom'])
def test_fix7_v702_sync_failure_names_cudaStreamSynchronize(trace_module, oom):
    m = _trace(trace_module, 'sync', oom)['msg']
    _once(m, 'the legacy stream of device 0 could not be synchronized (cudaStreamSynchronize: cudaErrorLaunchFailure',
          'cudaStreamSynchronize')
    assert 'cudaEventQuery' not in m, m


@pytest.mark.parametrize('oom', [False, True], ids=['plain', 'message_oom'])
def test_fix7_v702_sync_and_query_failures_are_each_named_once(trace_module, oom):
    m = _trace(trace_module, 'both', oom)['msg']
    _once(m, 'the legacy stream of device 1 could not be synchronized (cudaStreamSynchronize: cudaErrorLaunchFailure',
          'the completion event of a pending entry on device 0 could not be queried (cudaEventQuery: '
          'cudaErrorIllegalAddress', 'cudaStreamSynchronize', 'cudaEventQuery')


@pytest.mark.parametrize('oom', [False, True], ids=['plain', 'message_oom'])
def test_fix7_v702_spill_failure_names_its_device(trace_module, oom):
    m = _trace(trace_module, 'spill', oom)['msg']
    _once(m, 'the legacy stream of device 67 could not be synchronized (cudaStreamSynchronize: cudaErrorLaunchFailure',
          'device 67 is beyond the first 64 devices')
    assert 'device -1' not in m, m                                    # 429446d: "the legacy stream of device -1"


@pytest.mark.parametrize('oom', [False, True], ids=['plain', 'message_oom'])
def test_fix7_v702_current_device_query_failure_names_cudaGetDevice_and_no_device(trace_module, oom):
    m = _trace(trace_module, 'current_device', oom)['msg']
    _once(m, 'the current-device query cudaGetDevice failed (cudaErrorNoDevice', 'cudaGetDevice')
    assert 'device -1' not in m and 'could not be synchronized' not in m, m


@pytest.mark.parametrize('oom', [False, True], ids=['plain', 'message_oom'])
def test_fix7_sweep_j_guard_switch_failure_names_cudaSetDevice_not_the_sync(trace_module, oom):
    m = _trace(trace_module, 'guard_set', oom)['msg']
    _once(m, 'the legacy stream of device 1 was not synchronized: switching to device 1 failed '
             '(cudaSetDevice: cudaErrorInvalidDevice', 'cudaSetDevice')
    assert 'cudaStreamSynchronize' not in m and 'could not be synchronized' not in m, m   # 429446d: both


@pytest.mark.parametrize('oom', [False, True], ids=['plain', 'message_oom'])
def test_fix7_sweep_j_guard_read_failure_names_cudaGetDevice_not_the_sync(trace_module, oom):
    m = _trace(trace_module, 'guard_get', oom)['msg']
    _once(m, 'the legacy stream of device 1 was not synchronized: the current-device query before switching to '
             'device 1 failed (cudaGetDevice: cudaErrorNoDevice', 'cudaGetDevice')
    assert 'cudaStreamSynchronize' not in m and 'could not be synchronized' not in m, m


def test_fix7_sweep_j_bound_drain_names_one_consistent_first_failure(trace_module):
    m = _trace(trace_module, 'bound_drain', False)['msg']
    # The bound drain's current-device query failed first, then device 1's stream sync. 429446d paired device 1 with
    # the query's error ("the first on device 1 (cudaErrorNoDevice ...)").
    _once(m, '1 earlier release-path failure(s) since the last synchronize, the first (in a bound drain of the pending '
             'list): the current-device query cudaGetDevice failed (cudaErrorNoDevice')
    assert 'device 1 (cudaErrorNoDevice' not in m, m


def test_fix7_sweep_j_fallback_sync_failure_and_record_failure_are_named(trace_module):
    m = _trace(trace_module, 'fallback_record', False)['msg']
    _once(m, 'recording a call\'s completion event (cudaEventRecord on the operands\' device; the call\'s operands were '
             'synchronized instead) on device 1 (cudaErrorInvalidResourceHandle',
          'the first (in the synchronous fallback of a call that could not queue its completion event): the legacy '
          'stream of device 1 could not be synchronized (cudaStreamSynchronize: cudaErrorLaunchFailure')


def test_fix7_sweep_j_record_guard_failure_names_cudaSetDevice_not_cudaEventRecord(trace_module):
    m = _trace(trace_module, 'fallback_record_guard', False)['msg']
    _once(m, "a call's completion event on device 1 was not recorded (the call's operands were synchronized instead): "
             'switching to device 1 failed (cudaSetDevice: cudaErrorInvalidDevice')
    assert 'cudaEventRecord' not in m, m                              # 429446d: "recording ... (cudaEventRecord ..."


def test_fix7_sweep_j_event_destroy_failure_is_named(trace_module):
    m = _trace(trace_module, 'destroy', False)['msg']
    _once(m, 'cudaEventDestroy on device 0 (cudaErrorInvalidResourceHandle')


def test_fix7_sweep_j_destroy_guard_failure_names_cudaSetDevice_not_cudaEventDestroy(trace_module):
    m = _trace(trace_module, 'destroy_guard', False)['msg']
    _once(m, 'a completion event of device 1 was not destroyed: switching to device 1 failed (cudaSetDevice: '
             'cudaErrorInvalidDevice')
    assert 'cudaEventDestroy' not in m, m                             # 429446d: "cudaEventDestroy on device 1"


def test_fix7_sweep_j_failures_beyond_the_report_capacity_are_counted_not_mislabelled(trace_module):
    m = _trace(trace_module, 'unnamed', False)['msg']
    assert m.count('could not be queried (cudaEventQuery: cudaErrorIllegalAddress') == 8, m      # kSyncReportMax
    _once(m, '2 more failure(s) of this drain, not named (the report names the first 8)')
    assert 'device failure' not in m and 'could not be synchronized' not in m, m       # 429446d: "2 more device failure(s)"


def test_fix7_sweep_j_a_pending_error_says_its_call_and_device_are_unknown(trace_module):
    m = _trace(trace_module, 'prior', False)['msg']
    _once(m, 'a CUDA error was already pending in this thread when synchronize() was called (cudaErrorIllegalAddress',
          'the call that set it and its device are unknown: the slot records neither')
    c = _trace(trace_module, 'prior_launch', False)['create']
    assert c is not None and 'already pending in this thread before this call\'s first launch' in c, c
    assert 'the call that set it and its device are unknown: the slot records neither' in c, c


def test_fix7_sweep_j_event_creation_guard_failure_names_cudaSetDevice(trace_module):
    res = _trace(trace_module, 'create_guard', False)
    c = res['create']
    assert c is not None and 'cannot create the completion event on CUDA device 2: cudaErrorInvalidDevice' in c, res
    assert 'switching to that device failed, cudaSetDevice returned it; no event was created' in c, res
    assert 'cudaEventCreateWithFlags' not in c and res['msg'] is None, res


# ================================================================================================== V7-02, real module
_REAL_SCRIPT = r'''
import json, sys
import torch
import tc_signal as ts
from tc_signal import _tc_signal as _C
site, oom = sys.argv[1], sys.argv[2] == 'oom'
ts.set_allocator(lambda s: torch.empty(s, device='cuda'))
ts.synchronize()
ts.rint(torch.full((4096,), 1.5, device='cuda'))       # one pending entry on device 0
_C._test_inject([site] + (['alloc:error_message'] if oom else []))
res = {}
if oom:
    try:
        ts.synchronize()
        res['first'] = 'success'
    except MemoryError:
        res['first'] = 'MemoryError'
try:
    ts.synchronize()
    res['msg'] = None
except RuntimeError as e:
    res['msg'] = str(e)
try:
    ts.synchronize()
    res['again'] = None
except RuntimeError as e:
    res['again'] = str(e)
print('RESULT ' + json.dumps(res), flush=True)
'''

_REAL_EXPECT = {
    'cuda:event_query': ('the completion event of a pending entry on device 0 could not be queried (cudaEventQuery: '
                         + INJECTED_ERROR, 'cudaEventQuery', ('could not be synchronized', 'cudaStreamSynchronize')),
    'cuda:sync': ('the legacy stream of device 0 could not be synchronized (cudaStreamSynchronize: ' + INJECTED_ERROR,
                  'cudaStreamSynchronize', ('cudaEventQuery',)),
    'cuda:current_device': ('the current-device query cudaGetDevice failed (' + INJECTED_ERROR, 'cudaGetDevice',
                            ('device -1', 'could not be synchronized')),
}


@pytest.mark.parametrize('oom', [False, True], ids=['plain', 'message_oom'])
@pytest.mark.parametrize('site', list(_REAL_EXPECT))
def test_fix7_v702_real_module_names_the_failed_call_once(site, oom):
    r, res = _run(_REAL_SCRIPT, site, 'oom' if oom else '-')
    _ok(r, res)
    if oom:
        assert res['first'] == 'MemoryError', res
    text, api, absent = _REAL_EXPECT[site]
    _once(res['msg'], text, api)
    for a in absent:
        assert a not in res['msg'], (a, res)
    assert res['again'] is None, res


# ================================================================================================== sweep (j), cuBLAS
# The check after a cuBLAS call that returned success: an LD_PRELOAD shim calls the real cuBLAS function and then makes
# the runtime's error slot hold a GENUINE error (the real cudaSetDevice(-1)), as an asynchronous error of earlier work
# would. 429446d said "cublasSgemmStridedBatched failed: ..." although cuBLAS had returned CUBLAS_STATUS_SUCCESS.
_CUBLAS_SHIM_C = r'''
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
static int armed = 0;
static void* real_sym(const char* lib, const char* name) {
  void* h = dlopen(lib, RTLD_LAZY | RTLD_NOLOAD);
  if (!h) h = dlopen(lib, RTLD_LAZY);
  return h ? dlsym(h, name) : NULL;
}
static void slot_error(void) {
  int (*set)(int) = (int (*)(int))real_sym("libcudart.so.12", "cudaSetDevice");
  set(-1);
}
typedef int (*sgemm_sb_fn)(void*, int, int, int, int, int, const float*, const float*, int, long long, const float*,
                           int, long long, const float*, float*, int, long long, int);
int cublasSgemmStridedBatched(void* h, int ta, int tb, int m, int n, int k, const float* al, const float* A, int lda,
                              long long sa, const float* B, int ldb, long long sb, const float* be, float* C, int ldc,
                              long long sc, int bc) {
  static sgemm_sb_fn real;
  if (!real) real = (sgemm_sb_fn)real_sym("libcublas.so.12", "cublasSgemmStridedBatched");
  int r = real(h, ta, tb, m, n, k, al, A, lda, sa, B, ldb, sb, be, C, ldc, sc, bc);
  if (armed) { armed = 0; slot_error(); }
  return r;
}
void shim_arm(void) { armed = 1; }
'''

_CUBLAS_SCRIPT = r'''
import ctypes, json, sys
import numpy as np
import tensor_cuda as tc
import tc_signal as ts
shim = ctypes.CDLL(sys.argv[1])
ts.set_allocator(lambda s: tc.empty(*s))
w = tc.tensor(np.ones((2, 2, 3), np.float32))
x = tc.tensor(np.ones((1, 2, 16), np.float32))
ts.conv1d(x, w, padding=1)                     # warm (handle, event pool)
ts.synchronize()
shim.shim_arm()
res = {}
try:
    ts.conv1d(x, w, padding=1)
    res['msg'] = None
except RuntimeError as e:
    res['msg'] = str(e)
ts.synchronize()
print('RESULT ' + json.dumps(res), flush=True)
'''


@pytest.fixture(scope='module')
def cublas_shim(tmp_path_factory):
    d = tmp_path_factory.mktemp('fix7_cublas_shim')
    src, so = d / 'cublas_shim.c', d / 'cublas_shim.so'
    src.write_text(_CUBLAS_SHIM_C)
    r = subprocess.run(['gcc', '-shared', '-fPIC', '-O1', '-o', str(so), str(src), '-ldl'], capture_output=True,
                       text=True, timeout=SUBPROC_TIMEOUT)
    assert r.returncode == 0, r.stderr
    return str(so)


def test_fix7_sweep_j_check_after_a_successful_cublas_call_does_not_blame_it(cublas_shim):
    r, res = _run(_CUBLAS_SCRIPT, cublas_shim, env=_env(LD_PRELOAD=cublas_shim))
    _ok(r, res)
    m = res['msg']
    assert m is not None, res
    _once(m, 'CUDA error found by the check after cublasSgemmStridedBatched on CUDA device 0, which returned '
             'CUBLAS_STATUS_SUCCESS: cudaErrorInvalidDevice')
    assert 'cublasSgemmStridedBatched failed' not in m, m             # 429446d
