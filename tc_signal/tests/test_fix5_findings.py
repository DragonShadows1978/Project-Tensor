"""P1T-FIX5 regression tests for tc_signal (order P1T_FIX5_ADDON_EXIT_AND_CLASSES.md): round-5 findings V5-01, V5-02,
V5-03 and the add-on rows of the class (c) / (d) / (e) sweeps. V5-04 is core: tensor_cuda/tests/test_dlpack.py
test_fix5_*.

What each group pins, and the registered mutant it must kill (G4):
  * V5-01, the release lifecycle at interpreter exit; mutant (a) = the admission wait removed (wait_for_other_threads
    returns at once), which aborts the census cases (rc -6, "FATAL: exception not rethrown") and lets synchronize()
    return under a running release:
      - census: a producer callback running on a daemon thread when the exit hook runs, at ORDINARY interpreter exit
        (no os._exit): deleters of 0.1 s, 5.5 s and 12 s on the immediate-refusal path (a versioned read-only output
        refused after consumption; CPU only, no launch) and on the deferred path (a completed entry released by a
        daemon thread's next-call drain; one GPU); plus a 0.6 s deleter on the immediate path with no visible GPU, a
        0.6 s capsule destructor (the last reference of an unconsumed capsule dropped by a refused plan) and a 0.6 s
        producer __dlpack__. Rule: rc 0, the callback finished before CPython finalized, no abort marker, and the wait's
        stderr diagnostics follow the default 5 s doubling schedule (none / one at 5 s / two at 5 s and 10 s);
      - a release, capsule drop or producer callback that starts after admission closed: leaked or refused without
        entering Python, and counted;
      - the exit hook called from inside a callback on its own thread (a deleter; a producer __dlpack__): no self-wait;
      - a deleter that throws a C++ exception: contained exactly once, on the main thread and on a daemon thread that is
        in flight at exit (it releases and re-takes the GIL before throwing);
      - synchronize() outside exit waits for a release already running on another thread.
  * V5-02, a one-shot non-NotReady event-query failure in the next-call drain is recorded before its error-slot entry is
    consumed and raised exactly once by the next synchronize(), with one entry pending and with several completed
    entries pending; every entry is still released (none leaked) and the following synchronize() succeeds. Mutant (b) =
    the record removed. Sweep (d) companion: a failed completion-event record (the call then synchronizes) is recorded
    and raised once too.
  * V5-03, a cuBLAS handle created by the conv or LSTM cache is destroyed when the cache insertion itself fails: a REAL
    operator new failure at the insertion (LD_PRELOAD C++ shim armed by the last handle setting, as the reviewer's
    handle_insert_oom_cpu.cpp does on CPU) raises MemoryError with cublasCreate and cublasDestroy counts equal, and the
    next call creates a working handle; plus the reviewer's CPU probe regenerated from THIS tree's two functions
    (fix5_regen.py). Mutant (c) = the ownership removed.

Skip guard (as the other tc_signal test modules): only when torch is not importable or there is no CUDA device. A
missing compiler FAILS (it is not a skip).

Prior art: the round-5 reviewer's probes (artifacts/p1t_rev5/probes/addon_exit_*.py, exit_gate.py, query_error_*.py,
handle_insert_oom_cpu.cpp), whose scenarios these tests reuse; core's FIX2/FIX3 exit tests (test_dlpack.py
_FIX2_N5_SCRIPT: the Finalizer that drops the GIL during finalization, the 0.1 / 5.5 / 12 s census and the doubling
diagnostic schedule); LD_PRELOAD interposition (glibc ld.so) and fail-the-next-allocation fault injection (SQLite's OOM
harness, Linux failslab); DeMillo, Lipton & Sayward (1978) mutation testing for the registered mutants. Ours: the
callback / capsule-drop / after-close census and the arm-on-last-setting allocation trap.
"""
import json
import os
import re
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
ROOT = TESTS_DIR.parents[1]
SUBPROC_TIMEOUT = 120
CUDA_HOME = '/usr/local/cuda-12.6'
EXIT_DIAG_DEFAULT_S = 5.0          # tc_signal._RELEASE_EXIT_DIAG_S (P1T-FIX6: synchronize() no longer waits, so its knob is gone)
MATMUL_BACKLOG = 40                # 4096^3 fp32 matmuls (~0.3-0.5 s) queued ahead of the V5-02 entries ("not ready")
_DIAG_RE = re.compile(r'tc_signal: \d+ DLPack producer callback\(s\) .* still running after (\d+\.\d) s')
_ABORT_MARKERS = ('FATAL', 'terminate called', 'exception not rethrown', 'Aborted', 'core dumped')


def _env(**extra):
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.update(extra)
    return env


def _run(script, *args, env=None, timeout=SUBPROC_TIMEOUT):
    r = subprocess.run([sys.executable, '-c', script, *args], capture_output=True, text=True, timeout=timeout,
                       env=env or _env())
    # P1T-FIX6: the marker may follow another thread's partial line (print() writes the text and the newline
    # separately, so a daemon thread's line can be cut by the main thread's RESULT line): find it anywhere in a line.
    lines = [ln[ln.index('RESULT {'):] for ln in r.stdout.splitlines() if 'RESULT {' in ln]
    res = json.loads(lines[-1][len('RESULT '):]) if lines else None
    return r, res


def _ok(r, res):
    assert r.returncode == 0 and res is not None, f'rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}'


def _no_abort(r):
    for bad in _ABORT_MARKERS:
        assert bad not in r.stderr, f'{bad!r} in stderr:\n{r.stderr}'


# ================================================================================================== V5-01 census
# Atexit order (LIFO): `_report` is registered BEFORE tc_signal is imported, so it runs AFTER tc_signal's exit hook and
# prints what the hook left; `start` is registered after the import, so it runs FIRST: it starts the daemon worker whose
# producer callback is in flight when tc_signal's hook runs. The Finalizer, held by its own module, prints during
# CPython's finalization and then sleeps (dropping the GIL), as in the reviewer's probes. No os._exit anywhere.
_EXIT_SCRIPT = r'''
import atexit, ctypes, json, sys, threading, time, types
RESULTS = {}
def _report():
    RESULTS['after_hook'] = {'stats': dict(_C.pending_stats()), 'calls': DEL['calls'], 'done': DEL['done'],
                             'finalizing': sys.is_finalizing(), 'capsule_dtor_calls': DEL['capsule_dtor_calls']}
    print('RESULT ' + json.dumps(RESULTS), flush=True)
atexit.register(_report)
sys.path.insert(0, TESTS_DIR)
import tcs_producer as P
PATH, SLEEP = sys.argv[1], float(sys.argv[2])
if PATH in ('deferred', 'sync_wait'):
    import torch
import tc_signal as ts
from tc_signal import _tc_signal as _C
new = ctypes.pythonapi['PyCapsule_New']
new.restype = ctypes.py_object
new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
DEL = {'calls': 0, 'done': 0, 'capsule_dtor_calls': 0}
started = threading.Event()
KEEP = []
def slow_body():
    DEL['calls'] += 1
    print('DELETER started', flush=True)
    started.set()
    time.sleep(SLEEP)                                   # releases the GIL; re-taken to return
    DEL['done'] += 1
    print('DELETER finished', flush=True)
slow = P.DELETER(lambda _p: slow_body())
SH = (ctypes.c_int64 * 1)(1)
def managed(versioned, flags=0, deleter=None, data=0x1000, dtype=(2, 32, 1)):
    m = P.DLManagedTensorVersioned() if versioned else P.DLManagedTensor()
    if versioned:
        m.version, m.flags = P.DLPackVersion(1, 3), flags
    if deleter is not None:
        m.deleter = deleter
    t = m.dl_tensor
    t.data, t.device, t.dtype, t.ndim, t.shape = data, P.DLDevice(2, 0), P.DLDataType(*dtype), 1, SH
    t.strides = ctypes.POINTER(ctypes.c_int64)()
    KEEP.append(m)
    return m
def capsule(m, versioned, destructor=None):
    return new(ctypes.addressof(m), b'dltensor_versioned' if versioned else b'dltensor', destructor)
worker = None
if PATH in ('immediate', 'immediate_nogpu'):
    # The reviewer's addon_exit_full_cpu.py staging: refused before RetainUntilComplete, no CUDA work at all.
    x = capsule(managed(False), False)
    out = capsule(managed(True, flags=P.FLAG_READ_ONLY, deleter=slow, data=0x2000), True)
    def worker():
        try:
            _C.rint_run(x, out)
        except ValueError as e:
            print('REFUSED ' + str(e)[:60], flush=True)
elif PATH == 'deferred':
    # The reviewer's addon_exit_gpu.py staging: a completed entry released by a daemon thread's next-call drain.
    owner = torch.ones(4, device='cuda')
    dst = torch.empty_like(owner)
    SH4 = (ctypes.c_int64 * 1)(4)
    m = P.DLManagedTensor()
    m.deleter = slow
    t = m.dl_tensor
    t.data, t.device, t.dtype, t.ndim, t.shape = owner.data_ptr(), P.DLDevice(2, 0), P.DLDataType(2, 32, 1), 1, SH4
    t.strides = ctypes.POINTER(ctypes.c_int64)()
    KEEP.append((m, SH4))
    ts.rint(capsule(m, False), out=dst)
    torch.cuda.synchronize()
    assert _C.pending_stats()['entries'] == 1 and dst.cpu().tolist() == [1.0] * 4
    def worker():
        _C.rint_plan(dst)                               # its drain releases the entry: the slow deleter runs
elif PATH == 'capsule_drop':
    # A producer whose capsule destructor (the DLPack producer side: an unconsumed capsule's destructor calls the
    # deleter) is slow; the plan refuses the capsule (fp16) after acquiring it, so tc_signal drops its last reference.
    @P.CAPSULE_DESTRUCTOR
    def slow_destructor(_cap):
        DEL['capsule_dtor_calls'] += 1
        slow_body()
    KEEP.append(slow_destructor)
    class Prod:
        def __dlpack_device__(self):
            return (2, 0)
        def __dlpack__(self, **kw):
            return new(ctypes.addressof(managed(True, dtype=(2, 16, 1))), b'dltensor_versioned',
                       ctypes.cast(slow_destructor, ctypes.c_void_p))
    def worker():
        try:
            _C.rint_plan(Prod())
        except TypeError as e:
            print('REFUSED ' + str(e)[:60], flush=True)
elif PATH == 'callback':
    # A producer whose __dlpack__ itself is slow; it returns an fp16 capsule (refused), whose last reference tc_signal
    # then drops AFTER the hook closed admission: kept, its destructor never runs.
    @P.CAPSULE_DESTRUCTOR
    def counting_destructor(_cap):
        DEL['capsule_dtor_calls'] += 1
    KEEP.append(counting_destructor)
    class Prod:
        def __dlpack_device__(self):
            return (2, 0)
        def __dlpack__(self, **kw):
            slow_body()
            return new(ctypes.addressof(managed(True, dtype=(2, 16, 1))), b'dltensor_versioned',
                       ctypes.cast(counting_destructor, ctypes.c_void_p))
    def worker():
        try:
            _C.rint_plan(Prod())
        except TypeError as e:
            print('REFUSED ' + str(e)[:60], flush=True)
def start():
    th = threading.Thread(target=worker, daemon=True)
    th.start()
    assert started.wait(10)
    RESULTS['before_hook'] = {'stats': dict(_C.pending_stats())}
    print('CALLBACK IN PROGRESS', flush=True)
atexit.register(start)                                  # runs first; tc_signal's hook next; _report last
class Finalizer:
    def __init__(self):
        self.sleep, self.isf = time.sleep, sys.is_finalizing
    def __del__(self):
        print('FINALIZING %s' % self.isf(), flush=True)
        self.sleep(1.5)
_holder = types.ModuleType('_fix5_finalizer_holder')
_holder.keep = Finalizer()
sys.modules['_fix5_finalizer_holder'] = _holder
del _holder
print('ARMED ' + PATH, flush=True)
'''.replace('TESTS_DIR', repr(str(TESTS_DIR)))


def _census(path, sleep_s, env=None):
    r, res = _run(_EXIT_SCRIPT, path, repr(sleep_s), env=env, timeout=SUBPROC_TIMEOUT)
    out = r.stdout
    assert res is not None, f'rc={r.returncode}\nstdout={out}\nstderr={r.stderr}'
    assert r.returncode == 0, f'rc={r.returncode}\nstdout={out}\nstderr={r.stderr}'   # mutant (a): rc -6
    _no_abort(r)
    done, fin = out.find('DELETER finished'), out.find('FINALIZING True')
    assert 0 <= done < fin, f'the callback did not finish before CPython finalized:\n{out}'
    after = res['after_hook']
    assert after['calls'] == 1 and after['done'] == 1 and after['finalizing'] is False, res
    st = after['stats']
    assert st['releases_in_flight'] == 0 and st['callbacks_in_flight'] == 0, res
    assert st['release_admission_closed'] is True and st['exit_hook_calls'] == 1, res
    assert st['lifecycle_lock_failures'] == 0, res
    lines = [float(x) for x in _DIAG_RE.findall(r.stderr)]
    if sleep_s < EXIT_DIAG_DEFAULT_S:
        assert lines == [], r.stderr
    else:
        want = [5.0] if sleep_s < 2 * EXIT_DIAG_DEFAULT_S else [5.0, 10.0]   # doubling: 5 s, 10 s, 20 s, ...
        assert len(lines) == len(want) and all(abs(a - b) < 0.5 for a, b in zip(lines, want)), r.stderr
    return r, res


def test_fix5_v501_census_script_has_no_immediate_exit_escape():
    assert 'os._exit' not in _EXIT_SCRIPT and '_exit(' not in _EXIT_SCRIPT


@pytest.mark.parametrize('sleep_s', [0.1, 5.5, 12.0])
def test_fix5_v501_immediate_refusal_deleter_in_flight_at_exit(sleep_s):
    r, res = _census('immediate', sleep_s)
    st = res['after_hook']['stats']
    assert res['before_hook']['stats']['releases_in_flight'] == 1, res    # admitted and running when the hook ran
    # P1T-FIX6 (V6-01): the exit hook's close-and-wait waited for it (FIX5 asserted synchronize()'s removed wait). A
    # 0.1 s deleter may legitimately finish before the close (the hook's synchronize() makes the process's first CUDA
    # call, a context initialization of about 0.2 s): then nothing is in flight to wait for. The order check in _census
    # (finished before FINALIZING, rc 0) holds for every duration; the wait is certain for 5.5 s and 12 s.
    assert st['admitted_releases'] >= 1, res
    assert st['exit_in_flight_at_close'] == 1 if sleep_s > 1 else st['exit_in_flight_at_close'] in (0, 1), res
    assert st['deleters_leaked_after_close'] == 0, res


@pytest.mark.parametrize('sleep_s', [0.1, 5.5, 12.0])
def test_fix5_v501_deferred_release_in_flight_at_exit(sleep_s):
    r, res = _census('deferred', sleep_s)
    st = res['after_hook']['stats']
    assert res['before_hook']['stats']['releases_in_flight'] == 1, res
    assert res['before_hook']['stats']['entries'] == 0, res               # popped: invisible to the pending list
    # P1T-FIX6: the exit hook's wait (certain for 5.5 s and 12 s; a 0.1 s deleter may finish first, see above)
    assert st['exit_in_flight_at_close'] == 1 if sleep_s > 1 else st['exit_in_flight_at_close'] in (0, 1), res


def test_fix5_v501_immediate_refusal_with_no_visible_gpu():
    r, res = _census('immediate_nogpu', 0.6, env=_env(CUDA_VISIBLE_DEVICES=''))
    assert 'cudaErrorNoDevice' in r.stderr, r.stderr                      # the hook's synchronize() logged, did not raise


def test_fix5_v501_capsule_destructor_in_flight_at_exit():
    r, res = _census('capsule_drop', 0.6)
    after = res['after_hook']
    assert after['capsule_dtor_calls'] == 1, res                          # the drop ran the producer destructor once
    assert res['before_hook']['stats']['releases_in_flight'] == 1, res    # an admitted release (capsule drop)
    assert after['stats']['exit_in_flight_at_close'] == 1, res          # P1T-FIX6: the exit hook's wait


def test_fix5_v501_producer_dlpack_callback_in_flight_at_exit_and_late_capsule_drop_is_kept():
    r, res = _census('callback', 0.6)
    after = res['after_hook']
    st = after['stats']
    assert res['before_hook']['stats']['callbacks_in_flight'] == 1, res   # an admitted producer callback
    assert st['exit_in_flight_at_close'] == 1, res                        # the exit hook waited for it
    assert st['capsules_kept_after_close'] == 1 and after['capsule_dtor_calls'] == 0, res   # dropped after close: kept


# ================================================================================================== V5-01 after close
_AFTER_CLOSE_SCRIPT = r'''
import ctypes, json, sys
sys.path.insert(0, TESTS_DIR)
import torch
import tc_signal as ts
from tc_signal import _tc_signal as _C
import tcs_producer as P
ts.set_allocator(lambda s: torch.empty(s, device='cuda'))
N = 4096
x = torch.full((N,), 1.25, device='cuda')
y = torch.empty(N, device='cuda')
ts.rint(P.Producer(x, (N,)), out=P.Producer(y, (N,)))       # an entry pending at the close
torch.cuda.synchronize()
xr = P.Producer(x, (N,))._export(versioned=True)               # bare capsules made before the close
yr = P.Producer(y, (N,), flags=P.FLAG_READ_ONLY)._export(versioned=True)
s0 = dict(_C.pending_stats())
res = {'entries_before_close': s0['entries'], 'exported': P.STATS['exported'], 'deleted_before': P.STATS['deleted']}
res['close'] = dict(_C._exit_close_admission(5.0))              # the exit hook's close (its synchronize is skipped)
try:
    _C.rint_run(xr, yr)                                         # immediate refusal after close
    res['immediate'] = 'returned'
except ValueError as e:
    res['immediate'] = 'ValueError'
res['deleted_after_immediate'] = P.STATS['deleted']
ts.synchronize()                                                # deferred release after close
res['entries_after_sync'] = _C.pending_stats()['entries']
res['deleted_after_sync'] = P.STATS['deleted']
prod = P.Producer(x, (N,))
try:
    ts.rint(prod, out=torch.empty(N, device='cuda'))            # producer callback after close
    res['callback'] = 'returned'
except RuntimeError as e:
    res['callback'], res['callback_msg'] = 'RuntimeError', str(e)
res['producer_calls'] = len(prod.calls)
# Drop the two capsules before interpreter teardown: a capsule whose PyCapsule destructor is a ctypes callback
# (tcs_producer._capsule_destructor), still alive while CPython clears modules, can outlive that callback's thunk and
# crash in _ctypes (SIGSEGV in _PyModule_Clear; reproduced on 1475fae with a plain held capsule, no tc_signal call,
# docs/P1T_FIX5_LEDGER.md). Test-helper lifetime only; not a tc_signal path.
del xr, yr
import gc
gc.collect()
s1 = dict(_C.pending_stats())
res['delta'] = {k: s1[k] - s0[k] for k in s1 if isinstance(s1[k], int) and not isinstance(s1[k], bool) and k in s0}
res['closed'] = s1['release_admission_closed']
res['unknown_delete'] = P.STATS['unknown_delete']
print('RESULT ' + json.dumps(res), flush=True)
'''.replace('TESTS_DIR', repr(str(TESTS_DIR)))


def test_fix5_v501_releases_and_callbacks_after_close_leak_or_refuse_without_entering_python():
    r, res = _run(_AFTER_CLOSE_SCRIPT)
    _ok(r, res)
    _no_abort(r)
    assert res['entries_before_close'] == 1 and res['close']['pending_entries_at_close'] == 1, res
    assert res['close']['in_flight_at_close'] == 0 and res['close']['own_thread_in_flight'] == 0, res
    assert res['immediate'] == 'ValueError' and res['deleted_after_immediate'] == res['deleted_before'], res
    assert res['entries_after_sync'] == 0 and res['deleted_after_sync'] == res['deleted_before'], res
    d = res['delta']
    assert d['deleters_leaked_after_close'] == 4, res          # 2 (immediate) + 2 (the drained entry); none called
    assert d['admitted_releases'] == 0, res                    # no producer deleter entered after the close
    assert res['callback'] == 'RuntimeError' and 'interpreter exit has begun' in res['callback_msg'], res
    assert res['producer_calls'] == 0 and d['callbacks_refused_after_close'] == 1, res   # __dlpack__ never called
    assert res['closed'] is True and res['unknown_delete'] == 0, res


# ================================================================================================== V5-01 self-exclusion
_SELF_SCRIPT = r'''
import ctypes, json, sys, time
sys.path.insert(0, TESTS_DIR)
import tc_signal as ts
from tc_signal import _tc_signal as _C
import tcs_producer as P
new = ctypes.pythonapi['PyCapsule_New']
new.restype = ctypes.py_object
new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
MODE = sys.argv[1]
RES = {}
SH = (ctypes.c_int64 * 1)(1)
KEEP = []
def managed(flags=0, deleter=None, dtype=(2, 32, 1), data=0x1000):
    m = P.DLManagedTensorVersioned()
    m.version, m.flags = P.DLPackVersion(1, 3), flags
    if deleter is not None:
        m.deleter = deleter
    t = m.dl_tensor
    t.data, t.device, t.dtype, t.ndim, t.shape = data, P.DLDevice(2, 0), P.DLDataType(*dtype), 1, SH
    t.strides = ctypes.POINTER(ctypes.c_int64)()
    KEEP.append(m)
    return m
def call_hook(key):
    t0 = time.monotonic()
    ts._release_pending_at_exit()                               # the real exit hook, from inside our own callback
    RES[key] = {'hook_s': time.monotonic() - t0, 'stats': dict(_C.pending_stats())}
if MODE == 'deleter':
    hooking = P.DELETER(lambda _p: call_hook('inside'))
    x = new(ctypes.addressof(managed()), b'dltensor_versioned', None)
    out = new(ctypes.addressof(managed(flags=P.FLAG_READ_ONLY, deleter=hooking)), b'dltensor_versioned', None)
    try:
        _C.rint_run(x, out)                                     # refused: the deleter runs on THIS (main) thread
    except ValueError:
        RES['refused'] = True
else:   # 'callback': the hook runs inside a producer __dlpack__ on this thread, then the plan refuses both capsules
    @P.CAPSULE_DESTRUCTOR
    def counting(_cap):
        RES['dtor_calls'] = RES.get('dtor_calls', 0) + 1
    KEEP.append(counting)
    class Good:
        def __dlpack_device__(self):
            return (2, 0)
        def __dlpack__(self, **kw):
            return new(ctypes.addressof(managed()), b'dltensor_versioned', ctypes.cast(counting, ctypes.c_void_p))
    class Closing(Good):
        def __dlpack__(self, **kw):
            call_hook('inside')
            return new(ctypes.addressof(managed(dtype=(2, 16, 1))), b'dltensor_versioned',
                       ctypes.cast(counting, ctypes.c_void_p))
    try:
        _C.layer_norm_plan(Good(), Closing(), None, -1, 1e-5)   # x acquired, then scale's __dlpack__ closes admission
    except (TypeError, ValueError) as e:
        RES['refused'] = type(e).__name__
RES['after'] = dict(_C.pending_stats())
print('RESULT ' + json.dumps(RES), flush=True)
'''.replace('TESTS_DIR', repr(str(TESTS_DIR)))


def test_fix5_v501_exit_hook_inside_its_own_deleter_does_not_wait_for_itself():
    r, res = _run(_SELF_SCRIPT, 'deleter')
    _ok(r, res)
    _no_abort(r)
    inside = res['inside']
    assert inside['hook_s'] < 1.0, res                                    # no self-wait (else: a hang to the timeout)
    assert inside['stats']['release_admission_closed'] is True, res
    assert inside['stats']['releases_in_flight'] == 1, res                # its own release, excluded from the wait
    assert res['refused'] is True and res['after']['releases_in_flight'] == 0, res
    assert res['after']['exit_hook_calls'] == 1 and res['after']['exit_in_flight_at_close'] == 0, res


def test_fix5_v501_exit_hook_inside_its_own_producer_callback_does_not_wait_and_later_drops_are_kept():
    r, res = _run(_SELF_SCRIPT, 'callback')
    _ok(r, res)
    _no_abort(r)
    inside = res['inside']
    assert inside['hook_s'] < 1.0 and inside['stats']['callbacks_in_flight'] == 1, res
    assert res['refused'] == 'TypeError', res                             # scale's fp16 capsule refused by the plan
    after = res['after']
    assert after['capsules_kept_after_close'] == 2 and res.get('dtor_calls', 0) == 0, res   # x, scale: kept, not run
    assert after['callbacks_in_flight'] == 0 and after['releases_in_flight'] == 0, res


# ================================================================================================== V5-01 C++ thrower
_THROWER_SRC = r'''
// P1T-FIX5 test producer: DLPack deleters that throw a C++ exception (out of the DLPack C contract on purpose). The slow
// one releases the GIL, sleeps, re-takes it and then throws: it is the callback in flight at interpreter exit.
#include <Python.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>
static std::atomic<long> g_calls{0};
extern "C" void fix5_throwing_deleter(void*) {
  g_calls.fetch_add(1);
  throw std::runtime_error("fix5 test producer deleter threw");
}
extern "C" void fix5_slow_throwing_deleter(void*) {
  g_calls.fetch_add(1);
  std::printf("DELETER started\n");
  std::fflush(stdout);
  PyThreadState* s = PyEval_SaveThread();
  std::this_thread::sleep_for(std::chrono::milliseconds(600));
  PyEval_RestoreThread(s);
  std::printf("DELETER finished\n");
  std::fflush(stdout);
  throw std::runtime_error("fix5 slow test producer deleter threw");
}
extern "C" long fix5_throwing_deleter_calls() { return g_calls.load(); }
'''


@pytest.fixture(scope='module')
def thrower(tmp_path_factory):
    d = tmp_path_factory.mktemp('fix5_thrower')
    src, so = d / 'fix5_thrower.cpp', d / 'fix5_thrower.so'
    src.write_text(_THROWER_SRC)
    inc = subprocess.run(['python3-config', '--includes'], capture_output=True, text=True, check=True).stdout.split()
    r = subprocess.run(['g++', '-shared', '-fPIC', '-O1', '-std=c++17', *inc, '-o', str(so), str(src)],
                       capture_output=True, text=True, timeout=SUBPROC_TIMEOUT)
    assert r.returncode == 0, r.stderr
    return str(so)


_THROW_SCRIPT = r'''
import atexit, ctypes, json, sys, threading, time, types
RESULTS = {}
def _report():
    RESULTS['after_hook'] = {'stats': dict(_C.pending_stats()), 'calls': int(lib.fix5_throwing_deleter_calls())}
    print('RESULT ' + json.dumps(RESULTS), flush=True)
atexit.register(_report)
sys.path.insert(0, TESTS_DIR)
import tc_signal as ts
from tc_signal import _tc_signal as _C
import tcs_producer as P
lib = ctypes.CDLL(sys.argv[1])
MODE = sys.argv[2]
new = ctypes.pythonapi['PyCapsule_New']
new.restype = ctypes.py_object
new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
SH = (ctypes.c_int64 * 1)(1)
KEEP = []
def cap(name, flags=0, data=0x1000):
    m = P.DLManagedTensorVersioned()
    m.version, m.flags = P.DLPackVersion(1, 3), flags
    m.deleter = P.DELETER(ctypes.cast(getattr(lib, name), ctypes.c_void_p).value)
    t = m.dl_tensor
    t.data, t.device, t.dtype, t.ndim, t.shape = data, P.DLDevice(2, 0), P.DLDataType(2, 32, 1), 1, SH
    t.strides = ctypes.POINTER(ctypes.c_int64)()
    KEEP.append(m)
    return new(ctypes.addressof(m), b'dltensor_versioned', None)
if MODE == 'main':
    s0 = dict(_C.pending_stats())
    try:
        _C.rint_run(cap('fix5_throwing_deleter'), cap('fix5_throwing_deleter', P.FLAG_READ_ONLY, 0x2000))
    except ValueError:
        RESULTS['refused'] = True
    s1 = dict(_C.pending_stats())
    RESULTS['main'] = {'calls': int(lib.fix5_throwing_deleter_calls()),
                       'deleter_exceptions': s1['deleter_exceptions'] - s0['deleter_exceptions'],
                       'admitted': s1['admitted_releases'] - s0['admitted_releases'],
                       'releases_in_flight': s1['releases_in_flight']}
else:   # 'exit': in flight on a daemon thread when the exit hook runs
    x, out = cap('fix5_throwing_deleter'), cap('fix5_slow_throwing_deleter', P.FLAG_READ_ONLY, 0x2000)
    def worker():
        try:
            _C.rint_run(x, out)
        except ValueError:
            pass
    def start():
        threading.Thread(target=worker, daemon=True).start()
        t0 = time.monotonic()
        while lib.fix5_throwing_deleter_calls() < 1 and time.monotonic() - t0 < 10:   # out's slow deleter runs first
            time.sleep(0.005)
        RESULTS['before_hook'] = {'stats': dict(_C.pending_stats())}
    atexit.register(start)
    class Finalizer:
        def __init__(self):
            self.sleep, self.isf = time.sleep, sys.is_finalizing
        def __del__(self):
            print('FINALIZING %s' % self.isf(), flush=True)
            self.sleep(1.5)
    _holder = types.ModuleType('_fix5_thrower_holder')
    _holder.keep = Finalizer()
    sys.modules['_fix5_thrower_holder'] = _holder
    del _holder
print('ARMED', flush=True)
'''.replace('TESTS_DIR', repr(str(TESTS_DIR)))


def test_fix5_v501_throwing_cpp_deleter_contained_once_on_the_main_thread(thrower):
    r, res = _run(_THROW_SCRIPT, thrower, 'main')
    _ok(r, res)
    _no_abort(r)
    m = res['main']
    assert res['refused'] is True and m['calls'] == 2, res
    assert m['deleter_exceptions'] == 2 and m['admitted'] == 2 and m['releases_in_flight'] == 0, res   # left anyway


def test_fix5_v501_throwing_cpp_deleter_in_flight_at_exit_is_contained_once(thrower):
    r, res = _run(_THROW_SCRIPT, thrower, 'exit')
    _ok(r, res)
    _no_abort(r)
    out = r.stdout
    assert 0 <= out.find('DELETER finished') < out.find('FINALIZING True'), out
    assert res['before_hook']['stats']['releases_in_flight'] == 1, res
    st = res['after_hook']['stats']
    # P1T-FIX6 (V6-01): the exit hook now closes admission at once (FIX5's synchronize() first waited for out's release,
    # which let x's release start before the close). out's slow deleter, admitted and in flight, is waited for and its
    # exception contained once; x's release starts only after out's deleter returned, i.e. after the close: it is
    # deliberately leaked without entering the producer (V5-01's after-close rule), not called.
    assert res['after_hook']['calls'] == 1 and st['deleter_exceptions'] == 1, res
    assert st['deleters_leaked_after_close'] == 1, res
    assert st['releases_in_flight'] == 0 and st['exit_in_flight_at_close'] == 1, res   # P1T-FIX6: the exit hook's wait


# ================================================================================================== V5-01 synchronize()
_SYNC_WAIT_SCRIPT = r'''
import ctypes, json, sys, threading, time
sys.path.insert(0, TESTS_DIR)
import torch
import tc_signal as ts
from tc_signal import _tc_signal as _C
import tcs_producer as P
T = {}
started = threading.Event()
def body(_p):
    started.set()
    time.sleep(0.6)
    T['deleter_finished'] = time.monotonic()
slow = P.DELETER(body)
owner = torch.ones(4, device='cuda')
dst = torch.empty_like(owner)
SH4 = (ctypes.c_int64 * 1)(4)
m = P.DLManagedTensor()
m.deleter = slow
t = m.dl_tensor
t.data, t.device, t.dtype, t.ndim, t.shape = owner.data_ptr(), P.DLDevice(2, 0), P.DLDataType(2, 32, 1), 1, SH4
t.strides = ctypes.POINTER(ctypes.c_int64)()
new = ctypes.pythonapi['PyCapsule_New']
new.restype = ctypes.py_object
new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
ts.rint(new(ctypes.addressof(m), b'dltensor', None), out=dst)
torch.cuda.synchronize()
th = threading.Thread(target=lambda: _C.rint_plan(dst))         # its drain runs the slow deleter
th.start()
assert started.wait(10)
s0 = dict(_C.pending_stats())
ts.synchronize()                                                # P1T-FIX6 (V6-01): returns at once, no wait
T['sync_returned'] = time.monotonic()
T['finished_at_return'] = 'deleter_finished' in T
th.join()
s1 = dict(_C.pending_stats())
print('RESULT ' + json.dumps({'returned_first': (not T['finished_at_return']) and
                                                T['sync_returned'] < T['deleter_finished'],
                              'in_flight_before': s0['releases_in_flight'],
                              'closed': s1['release_admission_closed']}), flush=True)
'''.replace('TESTS_DIR', repr(str(TESTS_DIR)))


def test_fix5_v501_synchronize_does_not_wait_for_a_release_running_on_another_thread():
    # P1T-FIX6, round-6 review V6-01: FIX5's version of this test asserted that synchronize() WAITED for a release
    # running on another thread; that wait deadlocked against caller-held locks (sync_lock_inversion_cpu.py) and is
    # removed. The release popped by the other thread is that thread's: it finishes there, after synchronize() returned.
    # (The exit hook's wait is covered by the census tests above and test_fix6_findings.py.)
    r, res = _run(_SYNC_WAIT_SCRIPT)
    _ok(r, res)
    assert res['in_flight_before'] == 1 and res['returned_first'] is True, res
    assert res['closed'] is False, res                                    # synchronize() never closes admission


# ================================================================================================== V5-02 (+ sweep d)
@pytest.fixture(scope='module')
def fix4_shim(tmp_path_factory):
    so = tmp_path_factory.mktemp('fix5_fix4_shim') / 'fix4_cudart_shim.so'
    r = subprocess.run(['gcc', '-shared', '-fPIC', '-O1', '-o', str(so), str(TESTS_DIR / 'fix4_cudart_shim.c'), '-ldl'],
                       capture_output=True, text=True, timeout=SUBPROC_TIMEOUT)
    assert r.returncode == 0, r.stderr
    return str(so)


# torch-free (the shim's definitions are global; see fix4_cudart_shim.c): TensorCUDA tensors as the producers' memory.
_V502_SCRIPT = r'''
import ctypes, json, sys, types
import numpy as np
sys.path.insert(0, TESTS_DIR)
import tensor_cuda as tc
import tc_signal as ts
from tc_signal import _tc_signal as _C
import tcs_producer as P
shim = ctypes.CDLL(sys.argv[1])                # already loaded through LD_PRELOAD: the same handle
shim.shim_fail.argtypes = [ctypes.c_char_p]
mode = sys.argv[2]
class Base:
    def __init__(self, t):
        self.t, self.device = t, types.SimpleNamespace(index=0)
    def data_ptr(self):
        return self.t.data_ptr()
    def element_size(self):
        return 4
N = 4096
xv = (np.arange(N, dtype=np.float32) % 7) + 0.5
x = tc.tensor(xv)
ys = [tc.empty(N) for _ in range(3)]
want = np.rint(xv)
def rint(y):
    ts.rint(P.Producer(Base(x), (N,)), out=P.Producer(Base(y), (N,)))
res = {'mode': mode}
def attempt(fn, key):
    try:
        fn()
        res[key] = {'raised': None}
    except RuntimeError as e:
        res[key] = {'raised': 'RuntimeError', 'msg': str(e)}
used = ys
def plan_only():                              # a plan call: its drain runs, nothing is launched or queued
    _C.rint_plan(P.Producer(Base(x), (N,)))
if mode == 'one_pending':                     # the reviewer's query_error_single_gpu.py staging, torch-free
    rint(ys[0])
    tc.synchronize()                          # the only entry is complete, still pending (released by a drain)
    s0 = dict(_C.pending_stats())
    shim.shim_fail(b'cudaEventQuery')         # the next drain's query fails ONCE, with a genuine slot entry (101)
    attempt(plan_only, 'call')                # the drain records and consumes it; the entry stays pending
    used = ys[:1]
elif mode == 'one_pending_then_drained':      # a LATER drain that succeeds must not erase the record
    rint(ys[0])
    tc.synchronize()
    s0 = dict(_C.pending_stats())
    shim.shim_fail(b'cudaEventQuery')
    attempt(lambda: rint(ys[1]), 'call')      # plan drain fails (recorded); run drain releases entry 1; new entry
    used = ys[:2]
elif mode == 'completed_pending':
    rint(ys[0])                               # warm: the add-on's first launch loads its module (CUDA lazy loading
    ts.synchronize()                          # synchronizes), and the completion-event pool gets its event
    a = tc.randn(4096, 4096)
    tc.matmul(a, a)                           # warm cuBLAS the same way
    tc.synchronize()
    # Legacy-stream backlog, so no entry below completes during the calls. The outputs are KEPT: freeing an engine
    # tensor can call cudaFree, which synchronizes the device and would drain the backlog on the spot.
    backlog = [tc.matmul(a, a) for _ in range(BACKLOG)]
    for y in ys:
        rint(y)                               # each drain stops at the first entry, not ready yet: 3 entries pending
    res['entries_after_staging'] = _C.pending_stats()['entries']   # staging precondition: 3
    tc.synchronize()                          # now all three are complete, still pending
    del backlog
    s0 = dict(_C.pending_stats())
    shim.shim_fail(b'cudaEventQuery')
    attempt(plan_only, 'call')                # the first query fails: recorded, the drain stops, 3 stay pending
elif mode == 'record_fail':
    s0 = dict(_C.pending_stats())
    shim.shim_fail(b'cudaEventRecord')        # ~RetainUntilComplete cannot record: synchronous fallback
    attempt(lambda: rint(ys[0]), 'call')
    used = ys[:1]
res['call_slot'] = shim.shim_slot_peek()
res['stats_after_call'] = {k: v for k, v in dict(_C.pending_stats()).items() if not isinstance(v, bool)}
res['entries_before_sync'] = _C.pending_stats()['entries']
attempt(ts.synchronize, 'sync')               # raises once, naming the recorded failure; still releases everything
res['entries_after_sync'] = _C.pending_stats()['entries']
attempt(ts.synchronize, 'second_sync')        # reported exactly once
s1 = dict(_C.pending_stats())
res['delta'] = {k: s1[k] - s0[k] for k in s1 if isinstance(s1[k], int) and not isinstance(s1[k], bool) and k in s0}
res['correct'] = all(np.array_equal(y.numpy(), want) for y in used)
import gc
gc.collect()
res['deleted'], res['exported'], res['unknown_delete'] = P.STATS['deleted'], P.STATS['exported'], P.STATS['unknown_delete']
print('RESULT ' + json.dumps(res), flush=True)
'''.replace('TESTS_DIR', repr(str(TESTS_DIR))).replace('BACKLOG', str(MATMUL_BACKLOG))


def _v502(shim, mode):
    r, res = _run(_V502_SCRIPT, shim, mode, env=_env(LD_PRELOAD=shim))
    _ok(r, res)
    return res


@pytest.mark.parametrize('mode, pending', [('one_pending', 1), ('one_pending_then_drained', 1),
                                           ('completed_pending', 3)])
def test_fix5_v502_one_shot_query_failure_is_recorded_and_reported_once(fix4_shim, mode, pending):
    # `pending` = entries left for synchronize(): the failing entry (and the two behind it), or, when the same call's
    # later drain already released the failing entry, that call's own new entry.
    res = _v502(fix4_shim, mode)
    if mode == 'completed_pending':
        assert res['entries_after_staging'] == 3, res                            # staging precondition
    assert res['call']['raised'] is None and res['call_slot'] == 0, res          # consumed in the drain, call clean
    assert res['stats_after_call']['event_query_failures'] == 1, res
    assert res['stats_after_call']['unreported_unraisable_errors'] == 1, res     # recorded before it was consumed
    assert res['entries_before_sync'] == pending, res
    assert res['sync']['raised'] == 'RuntimeError', res                          # 1475fae / mutant (b): success
    assert 'cudaEventQuery of a pending entry' in res['sync']['msg'], res
    assert 'cudaErrorInvalidDevice' in res['sync']['msg'], res
    assert res['entries_after_sync'] == 0 and res['delta']['leaked_managers'] == 0, res   # released, none leaked
    assert res['second_sync']['raised'] is None, res                             # exactly once
    assert res['correct'] is True and res['unknown_delete'] == 0 and res['deleted'] == res['exported'], res


def test_fix5_sweep_d_failed_event_record_is_recorded_and_reported_once(fix4_shim):
    res = _v502(fix4_shim, 'record_fail')
    assert res['call']['raised'] is None and res['call_slot'] == 0, res
    assert res['delta']['event_record_failures'] == 1 and res['delta']['fallback_syncs'] == 1, res
    assert res['sync']['raised'] == 'RuntimeError', res                          # 1475fae: consumed, never reported
    assert "recording a call's completion event" in res['sync']['msg'], res
    assert 'cudaErrorInvalidDevice' in res['sync']['msg'], res
    assert res['second_sync']['raised'] is None and res['correct'] is True, res
    assert res['deleted'] == res['exported'] and res['unknown_delete'] == 0, res


# ================================================================================================== V5-03
_ALLOC_SHIM_SRC = r'''
// P1T-FIX5 test shim (LD_PRELOAD, C++): a REAL host-allocation failure at a cuBLAS handle cache insertion. Arming makes
// the NEXT cublasSetPointerMode_v2 (the last setting of a new handle, in both tc_signal caches) arm a one-shot failure
// of the next global operator new ON THAT THREAD, which is the cache's map-node allocation (handles.emplace), as in the
// reviewer's handle_insert_oom_cpu.cpp. cublasCreate_v2 / cublasDestroy_v2 are counted and forwarded to libcublas.so.12.
// Torch-free processes only (global definitions). Prior art: LD_PRELOAD interposition; fail-the-next-allocation fault
// injection (SQLite OOM harness, Linux failslab).
#include <dlfcn.h>
#include <atomic>
#include <cstdlib>
#include <new>
static thread_local bool t_fail_next = false;
static std::atomic<int> g_armed{0}, g_creates{0}, g_destroys{0}, g_injected{0};
static void* real_sym(const char* name) {
  void* h = dlopen("libcublas.so.12", RTLD_LAZY | RTLD_NOLOAD);
  if (!h) h = dlopen("libcublas.so.12", RTLD_LAZY);
  return h ? dlsym(h, name) : nullptr;
}
void* operator new(std::size_t n) {
  if (t_fail_next) {
    t_fail_next = false;
    g_injected.fetch_add(1);
    throw std::bad_alloc();
  }
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
extern "C" int cublasCreate_v2(void** h) {
  static auto real = reinterpret_cast<int (*)(void**)>(real_sym("cublasCreate_v2"));
  const int r = real(h);
  if (r == 0) g_creates.fetch_add(1);
  return r;
}
extern "C" int cublasDestroy_v2(void* h) {
  static auto real = reinterpret_cast<int (*)(void*)>(real_sym("cublasDestroy_v2"));
  g_destroys.fetch_add(1);
  return real(h);
}
extern "C" int cublasSetPointerMode_v2(void* h, int m) {
  static auto real = reinterpret_cast<int (*)(void*, int)>(real_sym("cublasSetPointerMode_v2"));
  const int r = real(h, m);
  if (g_armed.exchange(0)) t_fail_next = true;
  return r;
}
extern "C" void fix5_arm_insert_failure() { g_armed.store(1); }
extern "C" int fix5_creates() { return g_creates.load(); }
extern "C" int fix5_destroys() { return g_destroys.load(); }
extern "C" int fix5_injected() { return g_injected.load(); }
'''


@pytest.fixture(scope='module')
def alloc_shim(tmp_path_factory):
    d = tmp_path_factory.mktemp('fix5_alloc_shim')
    src, so = d / 'fix5_alloc_shim.cpp', d / 'fix5_alloc_shim.so'
    src.write_text(_ALLOC_SHIM_SRC)
    r = subprocess.run(['g++', '-shared', '-fPIC', '-O1', '-std=c++17', '-o', str(so), str(src), '-ldl'],
                       capture_output=True, text=True, timeout=SUBPROC_TIMEOUT)
    assert r.returncode == 0, r.stderr
    return str(so)


_V503_SCRIPT = r'''
import ctypes, json, sys
import numpy as np
import tensor_cuda as tc
import tc_signal as ts
from tc_signal import _tc_signal as _C
shim = ctypes.CDLL(sys.argv[1])
mode = sys.argv[2]
ts.set_allocator(lambda s: tc.empty(*s))
res = {'mode': mode}
def counts():
    return {'creates': shim.fix5_creates(), 'destroys': shim.fix5_destroys(), 'injected': shim.fix5_injected()}
if mode == 'conv':
    xv = (np.arange(32, dtype=np.float32).reshape(1, 2, 16) % 5) - 2
    wv = np.ones((2, 2, 3), np.float32)
    xc, w = tc.tensor(xv), tc.tensor(wv)
    xp = np.pad(xv, ((0, 0), (0, 0), (1, 1)))
    want = np.stack([sum(xp[0, c, k:k + 16] for c in range(2) for k in range(3))] * 2)[None]
    op = lambda: ts.conv1d(xc, w, padding=1)
    config = lambda: dict(_C.cublas_config(0))
else:
    T, Nb, I, H = 2, 1, 2, 2
    xl = tc.tensor(np.ones((T, Nb, I), np.float32))
    Wl = tc.tensor(np.full((1, 4 * H, I), 0.1, np.float32))
    Rl = tc.tensor(np.full((1, 4 * H, H), 0.1, np.float32))
    want = None
    op = lambda: ts.lstm(xl, Wl, Rl, hidden_size=H)[0]
    config = lambda: dict(_C.lstm_cublas_config(0))
res['before'] = counts()
shim.fix5_arm_insert_failure()
try:
    op()
    res['call'] = 'returned'
except MemoryError:
    res['call'] = 'MemoryError'
res['after_call'] = counts()
out = op()                                       # the next call: a new handle, created and cached
ts.synchronize()
tc.synchronize()
res['after_next'] = counts()
y = out.numpy()
res['finite'] = bool(np.all(np.isfinite(y)))
res['correct'] = bool(np.allclose(y, want)) if want is not None else None
res['config'] = config()
res['stats'] = {k: v for k, v in dict(_C.pending_stats()).items() if k.startswith('cublas')}
print('RESULT ' + json.dumps(res), flush=True)
'''


@pytest.mark.parametrize('mode', ['conv', 'lstm'])
def test_fix5_v503_cache_insertion_failure_destroys_the_new_handle(alloc_shim, mode):
    r, res = _run(_V503_SCRIPT, alloc_shim, mode, env=_env(LD_PRELOAD=alloc_shim))
    _ok(r, res)
    assert res['before'] == {'creates': 0, 'destroys': 0, 'injected': 0}, res
    assert res['call'] == 'MemoryError' and res['after_call']['injected'] == 1, res   # the insertion really failed
    assert res['after_call']['creates'] == 1 and res['after_call']['destroys'] == 1, res   # 1475fae: destroys 0
    assert res['after_next']['creates'] == 2 and res['after_next']['destroys'] == 1, res   # new handle, cached
    assert res['finite'] is True and res['correct'] in (True, None), res
    assert res['config']['default_math'] is True and res['config']['atomics_not_allowed'] is True, res
    assert res['config']['legacy_stream'] is True and res['stats']['cublas_handle_destroy_failures'] == 0, res


def test_fix5_v503_reviewer_cpu_probe_regenerated_from_these_sources_passes(tmp_path):
    sys.path.insert(0, str(TESTS_DIR))
    from fix5_regen import regen_handle_probe
    probe = ROOT / 'artifacts' / 'p1t_rev5' / 'probes' / 'handle_insert_oom_cpu.cpp'
    text = regen_handle_probe(probe.read_text(), (SRC_DIR / 'conv.cu').read_text(), (SRC_DIR / 'lstm.cu').read_text())
    assert 'note_cublas_handle_destroy_failure' in text and 'NewHandleOwner' in text   # this tree's functions
    src, exe = tmp_path / 'handle_insert_oom_regen.cpp', tmp_path / 'handle_insert_oom_regen'
    src.write_text(text)
    r = subprocess.run(['g++', '-std=c++17', '-O1', str(src), f'-I{CUDA_HOME}/include', '-o', str(exe)],
                       capture_output=True, text=True, timeout=SUBPROC_TIMEOUT)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=SUBPROC_TIMEOUT)
    assert r.returncode == 0, (r.stdout, r.stderr)                         # 1475fae's slices: rc 1, destroys=0
    assert 'conv creates=1 destroys=1' in r.stdout and 'lstm creates=1 destroys=1' in r.stdout, r.stdout
