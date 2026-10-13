"""P1T-FIX6 regression tests for tc_signal (order P1T_FIX6_SYNC_WAIT_AND_USER_CALLBACKS.md): round-6 findings V6-01 and
V6-03, and the add-on rows of the sweep (g) "user Python run from inside a native frame". V6-02 and the core rows are
in tensor_cuda/tests/test_dlpack.py test_fix6_*.

What each group pins, and the registered mutant it must kill (G4):
  * V6-01: ordinary synchronize() never waits for another thread's producer callback. The reviewer's lock inversion
    (a refused operand's finite deleter on thread B needs a lock that thread A holds while it calls synchronize()), with
    and without a visible GPU: synchronize() returns with the lock still held and no rescue; a release that starts
    after synchronize() began is not waited for (host trace module built from this tree's dlpack_operand.cpp); the exit
    hook still waits for a release admitted on another thread. Mutant (a) = FIX5's wait restored in synchronize().
  * V6-03: a one-shot failure of synchronize()'s own drain (the current-device query, a stream sync, a completion-event
    query) followed by a failure to allocate the error message: MemoryError first, then the next synchronize() raises
    the original CUDA failure exactly once, then synchronize() succeeds. On the real module (GPU) through the release-path
    test hooks, and on the host trace module. Mutant (c) = the persistence removed.
  * Sweep (g): every _tc_signal binding receives (*args, **kwargs) and converts with C-level checks only: an argument
    object whose every Python method records a call is refused (TypeError / OverflowError) with zero calls, for every
    typed parameter of every binding; wrong arity, unknown or duplicate keywords and missing arguments raise TypeError
    without calling repr(); at ordinary interpreter exit a daemon thread whose argument's __index__ would sleep across
    finalization exits cleanly (rc 0; acd42fa: the method is called and the process crashes); the last reference of a
    CONSUMED capsule is an admitted release (its producer destructor runs Python) and the exit hook waits for it.

Skip guard (as the other tc_signal test modules): only when torch is not importable or there is no CUDA device. A
missing compiler FAILS (it is not a skip).

Prior art: the round-6 reviewer's probes (artifacts/p1t_rev6/probes/sync_lock_inversion_cpu.py,
sync_late_release_cpu.py, release_trace_cpu.py diagnostic_oom_*, core_cast_exit_gpu.py), whose scenarios these tests
reuse; FIX5's exit census (test_fix5_findings.py); Coffman et al. (1971) circular wait (the lock inversion); SQLite's
OOM fault-injection discipline (fail the reporting allocation, the error must survive); DeMillo, Lipton & Sayward (1978)
mutation testing. Ours: the all-methods recording object and the per-binding refusal census.
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


# ================================================================================================== V6-01
# The reviewer's sync_lock_inversion_cpu.py staging, with a deadline instead of a rescue thread's verdict: thread B's
# refused read-only output runs its finite deleter while unwinding; the deleter needs LOCK, which the main thread holds
# while it calls synchronize(). FIX6: synchronize() returns without waiting (B is still blocked on LOCK); then LOCK is
# released and B finishes. acd42fa / mutant (a): synchronize() waits for B, which waits for LOCK: the rescue fires.
_LOCK_SCRIPT = r'''
import ctypes, json, sys, threading, time
sys.path.insert(0, TESTS_DIR)
import tcs_producer as P
import tc_signal as ts
from tc_signal import _tc_signal as _C
new = ctypes.pythonapi['PyCapsule_New']
new.restype = ctypes.py_object
new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
LOCK = threading.Lock()
started, finished = threading.Event(), threading.Event()
@P.DELETER
def needs_lock(_p):
    started.set()
    with LOCK:
        finished.set()
SH = (ctypes.c_int64 * 1)(1)
def managed(versioned, flags=0, deleter=None, data=0x1000):
    m = P.DLManagedTensorVersioned() if versioned else P.DLManagedTensor()
    if versioned:
        m.version, m.flags = P.DLPackVersion(1, 3), flags
    if deleter is not None:
        m.deleter = deleter
    t = m.dl_tensor
    t.data, t.device, t.dtype, t.ndim, t.shape = data, P.DLDevice(2, 0), P.DLDataType(2, 32, 1), 1, SH
    t.strides = ctypes.POINTER(ctypes.c_int64)()
    return m
mx, my = managed(False), managed(True, P.FLAG_READ_ONLY, needs_lock, 0x2000)
x = new(ctypes.addressof(mx), b'dltensor', None)
y = new(ctypes.addressof(my), b'dltensor_versioned', None)
def worker():
    try:
        _C.rint_run(x, y)
    except ValueError:
        pass
LOCK.acquire()
th = threading.Thread(target=worker)
th.start()
assert started.wait(5)
in_flight = _C.pending_stats()['releases_in_flight']
rescued = threading.Event()
sync_done = threading.Event()
def rescue():                                   # terminates the witness if synchronize() blocks (acd42fa, mutant a)
    if not sync_done.wait(3):
        rescued.set()
        LOCK.release()
threading.Thread(target=rescue, daemon=True).start()
t0 = time.monotonic()
err = None
try:
    ts.synchronize()
except RuntimeError as e:                       # no visible GPU: cudaErrorNoDevice, raised (not a wait)
    err = str(e)[:200]
elapsed = time.monotonic() - t0
sync_done.set()
blocked_at_return = not finished.is_set()
if not rescued.is_set():
    LOCK.release()
th.join(10)
print('RESULT ' + json.dumps({'in_flight': in_flight, 'elapsed': elapsed, 'rescued': rescued.is_set(),
                              'deleter_still_blocked_when_sync_returned': blocked_at_return,
                              'release_finished': finished.is_set(), 'err': err}), flush=True)
'''.replace('TESTS_DIR', repr(str(TESTS_DIR)))


@pytest.mark.parametrize('cond', ['gpu', 'nogpu'])
def test_fix6_v601_synchronize_does_not_wait_for_a_release_blocked_on_a_caller_held_lock(cond):
    env = _env(CUDA_VISIBLE_DEVICES='') if cond == 'nogpu' else _env()
    r, res = _run(_LOCK_SCRIPT, env=env)
    _ok(r, res)
    _no_abort(r)
    assert res['in_flight'] == 1, res                                   # B's deleter was admitted and running
    assert res['rescued'] is False and res['elapsed'] < 2.0, res        # no wait: no rescue needed
    assert res['deleter_still_blocked_when_sync_returned'] is True, res  # returned while B still waits for LOCK
    assert res['release_finished'] is True, res                         # B finished once LOCK was released
    if cond == 'nogpu':
        assert 'cudaErrorNoDevice' in (res['err'] or ''), res


# The exit hook still waits (the only waiter): a refused operand's slow deleter on a daemon thread at ordinary exit.
_EXIT_SCRIPT = r'''
import atexit, ctypes, json, sys, threading, time, types
RESULTS = {}
def _report():
    RESULTS['after_hook'] = {'stats': dict(_C.pending_stats()), 'done': DONE['n'], 'finalizing': sys.is_finalizing()}
    print('RESULT ' + json.dumps(RESULTS), flush=True)
atexit.register(_report)
sys.path.insert(0, TESTS_DIR)
import tcs_producer as P
import tc_signal as ts
from tc_signal import _tc_signal as _C
MODE = sys.argv[1]
new = ctypes.pythonapi['PyCapsule_New']
new.restype = ctypes.py_object
new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
DONE = {'n': 0}
started = threading.Event()
def slow_body():
    print('CALLBACK started', flush=True)
    started.set()
    time.sleep(0.6)
    DONE['n'] += 1
    print('CALLBACK finished', flush=True)
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
# P1T-FIX7 (a test defect found by this round's before-arm run): PyCapsule_New stores the NAME POINTER, which "must
# outlive the capsule" (CPython C-API, Capsules). A bytes literal of this -c script's module-level code lives in that
# code object, which CPython frees once the script body has run; these capsules are read at EXIT, so their names
# dangled ('a capsule named '@JHu' is not a DLPack capsule', deterministically in /tmp/p1t_fix7_before_tree, at
# random elsewhere). The names are now module globals (alive until finalization clears the module).
NAME_L, NAME_V = b'dltensor', b'dltensor_versioned'
x = new(ctypes.addressof(managed(False)), NAME_L, None)
if MODE == 'deleter':
    # V6-01: the reviewer's addon_exit_full_cpu.py staging (refused read-only output; its deleter runs on the daemon).
    slow = P.DELETER(lambda _p: slow_body())
    KEEP.append(slow)
    out = new(ctypes.addressof(managed(True, P.FLAG_READ_ONLY, slow, 0x2000)), NAME_V, None)
    def worker():
        try:
            _C.rint_run(x, out)
        except ValueError:
            pass
elif MODE == 'consumed_capsule_drop':
    # Sweep (g): `out` is a producer; its capsule is consumed (renamed used_dltensor_versioned) and refused (read-only);
    # tc_signal then drops the last reference to that CONSUMED capsule, which runs the producer's Python capsule
    # destructor. FIX6 admits that drop (the exit hook waits for it); acd42fa dropped it unadmitted.
    @P.CAPSULE_DESTRUCTOR
    def slow_destructor(_cap):
        slow_body()
    KEEP.append(slow_destructor)
    class Prod:
        def __dlpack_device__(self):
            return (2, 0)
        def __dlpack__(self, **kw):
            m = managed(True, P.FLAG_READ_ONLY, None, 0x2000)
            return new(ctypes.addressof(m), b'dltensor_versioned', ctypes.cast(slow_destructor, ctypes.c_void_p))
    def worker():
        try:
            _C.rint_run(x, Prod())
        except ValueError:
            pass
def start():
    threading.Thread(target=worker, daemon=True).start()
    assert started.wait(10)
    RESULTS['before_hook'] = {'stats': dict(_C.pending_stats())}
    print('CALLBACK IN PROGRESS', flush=True)
atexit.register(start)
class Finalizer:
    def __init__(self):
        self.sleep, self.isf = time.sleep, sys.is_finalizing
    def __del__(self):
        print('FINALIZING %s' % self.isf(), flush=True)
        self.sleep(1.5)
_holder = types.ModuleType('_fix6_finalizer_holder')
_holder.keep = Finalizer()
sys.modules['_fix6_finalizer_holder'] = _holder
del _holder
print('ARMED ' + MODE, flush=True)
'''.replace('TESTS_DIR', repr(str(TESTS_DIR)))


@pytest.mark.parametrize('mode', ['deleter', 'consumed_capsule_drop'])
def test_fix6_exit_hook_still_waits_for_an_admitted_callback_on_another_thread(mode):
    r, res = _run(_EXIT_SCRIPT, mode)
    _ok(r, res)
    _no_abort(r)
    out = r.stdout
    assert 0 <= out.find('CALLBACK finished') < out.find('FINALIZING True'), out
    before, after = res['before_hook']['stats'], res['after_hook']
    assert before['releases_in_flight'] == 1, res                        # admitted (a release) and running
    st = after['stats']
    assert after['done'] == 1 and after['finalizing'] is False, res
    assert st['exit_in_flight_at_close'] == 1 and st['releases_in_flight'] == 0, res   # the exit hook waited


# ================================================================================================== host trace module
# This tree's dlpack_operand.cpp compiled with replaced CUDA entry points (no GPU), as the reviewer's release_trace.cpp
# does: the pending list, synchronize() and the exit hook with controllable one-shot failures.
_TRACE_SRC = r'''
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
namespace py = pybind11;
static thread_local cudaError_t slot = cudaSuccess;
static int query_fail = 0, sync_fail = 0, device_fail = 0;
extern "C" cudaError_t cudaGetDevice(int* d) { if (device_fail) { --device_fail; slot = cudaErrorInvalidValue; return slot; } *d = 0; return cudaSuccess; }
extern "C" cudaError_t cudaSetDevice(int) { return cudaSuccess; }
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t) { if (sync_fail) { --sync_fail; slot = cudaErrorInvalidValue; return slot; } return cudaSuccess; }
extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) { *e = reinterpret_cast<cudaEvent_t>(16); return cudaSuccess; }
extern "C" cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return cudaSuccess; }
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
extern "C" cudaError_t cudaEventQuery(cudaEvent_t) { if (query_fail) { --query_fail; slot = cudaErrorInvalidValue; return slot; } return cudaSuccess; }
extern "C" cudaError_t cudaGetLastError() { auto e = slot; slot = cudaSuccess; return e; }
extern "C" cudaError_t cudaPeekAtLastError() { return slot; }
extern "C" const char* cudaGetErrorName(cudaError_t e) { return e == cudaSuccess ? "cudaSuccess" : "cudaErrorInvalidValue"; }
extern "C" const char* cudaGetErrorString(cudaError_t e) { return e == cudaSuccess ? "success" : "injected failure"; }
#include "dlpack_operand.cpp"
struct Held { DLManagedTensor m{}; py::object callback; };
static auto& held = *new std::vector<std::unique_ptr<Held>>();  // never destroyed: no py::object after finalization
static void delete_manager(DLManagedTensor* m) { auto* h = static_cast<Held*>(m->manager_ctx); if (!h->callback.is_none()) h->callback(); }
PYBIND11_MODULE(_p1t_fix6_trace, m) {
  tcs::init_pending_state();
  m.def("queue", [](py::object f) {
    auto h = std::make_unique<Held>(); h->callback = f; h->m.manager_ctx = h.get(); h->m.deleter = delete_manager;
    tcs::PendingEntry e; e.event = reinterpret_cast<cudaEvent_t>(16); e.device = 0; e.bytes = 4;
    tcs::ManagedOwner o; o.legacy = &h->m; e.owners.push_back(std::move(o)); held.push_back(std::move(h));
    auto& st = tcs::pending(); st.q.push_back(std::move(e)); tcs::note_push(st, st.q.back()); });
  m.def("drain", &tcs::drain_completed);
  m.def("synchronize", &tcs::synchronize_and_release_all);
  m.def("stats", [] { return tcs::pending_stats(false); });
  m.def("inject", [](const std::string& s) {
    if (s == "query") query_fail = 1; else if (s == "sync") sync_fail = 1; else if (s == "device") device_fail = 1;
    else if (s == "message") tcs::test_hooks::arm(tcs::test_hooks::kAllocErrorMessage);
    else throw std::runtime_error("unknown injection"); });
  held.reserve(64);
}
'''


@pytest.fixture(scope='module')
def trace_module(tmp_path_factory):
    d = tmp_path_factory.mktemp('fix6_trace')
    src = d / 'fix6_trace.cpp'
    src.write_text(_TRACE_SRC)
    inc = subprocess.run(['python3-config', '--includes'], capture_output=True, text=True, check=True).stdout.split()
    suffix = subprocess.run(['python3-config', '--extension-suffix'], capture_output=True, text=True,
                            check=True).stdout.strip()
    so = d / f'_p1t_fix6_trace{suffix}'
    r = subprocess.run(['g++', '-std=c++17', '-O1', '-shared', '-fPIC', str(src), f'-I{SRC_DIR}', f'-I{INCLUDE_DIR}',
                        f'-I{PYBIND11_INCLUDE}', f'-I{CUDA_HOME}/include', *inc, '-pthread', '-o', str(so)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    return str(d)


_LATE_SCRIPT = r'''
import json, sys, threading, time
sys.path.insert(0, sys.argv[1])
import _p1t_fix6_trace as n
first_started, late_started, late_end = threading.Event(), threading.Event(), threading.Event()
def first():
    first_started.set()
    late_started.wait(3)
def late():
    late_started.set()
    time.sleep(0.6)
    late_end.set()
n.queue(first)
a = threading.Thread(target=n.drain)
a.start()
assert first_started.wait(2)
def add_late():
    time.sleep(0.1)
    n.queue(late)
    n.drain()
b = threading.Thread(target=add_late)
b.start()
t0 = time.monotonic()
n.synchronize()
elapsed = time.monotonic() - t0
waited_for_later = late_end.is_set()
a.join(5); b.join(5)
print('RESULT ' + json.dumps({'elapsed': elapsed, 'waited_for_later': waited_for_later,
                              'late_finished': late_end.is_set()}), flush=True)
'''


def test_fix6_v601_release_started_after_synchronize_began_is_not_waited_for(trace_module):
    # The reviewer's sync_late_release_cpu.py scenario on this tree's dlpack_operand.cpp (host trace module).
    r, res = _run(_LATE_SCRIPT, trace_module)
    _ok(r, res)
    assert res['waited_for_later'] is False and res['elapsed'] < 0.5, res   # acd42fa / mutant (a): True, ~0.7 s
    assert res['late_finished'] is True, res


_TRACE_OOM_SCRIPT = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import _p1t_fix6_trace as n
mode = sys.argv[2]
calls = []
if mode == 'query':
    n.queue(lambda: calls.append(1))
n.inject(mode)
n.inject('message')
out = {}
try:
    n.synchronize()
    out['first'] = 'success'
except MemoryError:
    out['first'] = 'MemoryError'
out['persisted'] = n.stats()['unreported_drain_failures']
try:
    n.synchronize()
    out['second'] = 'success'
except RuntimeError as e:
    out['second'] = str(e)
try:
    n.synchronize()
    out['third'] = 'success'
except RuntimeError as e:
    out['third'] = str(e)
out['after'] = n.stats()['unreported_drain_failures']
out['deleter_calls'] = len(calls)
print('RESULT ' + json.dumps(out), flush=True)
'''


@pytest.mark.parametrize('mode', ['query', 'sync', 'device'])
def test_fix6_v603_drain_failure_survives_a_message_allocation_failure_host_trace(trace_module, mode):
    # The reviewer's release_trace_cpu.py diagnostic_oom_{query,sync,device} on this tree's dlpack_operand.cpp.
    r, res = _run(_TRACE_OOM_SCRIPT, trace_module, mode)
    _ok(r, res)
    assert res['first'] == 'MemoryError' and res['persisted'] == 1, res
    assert 'cudaErrorInvalidValue' in res['second'], res               # acd42fa / mutant (c): 'success'
    assert res['third'] == 'success' and res['after'] == 0, res       # raised exactly once
    if mode == 'query':
        assert res['deleter_calls'] == 0 and 'deliberately leaked' in res['second'], res   # no completion proof


# ================================================================================================== V6-03 real module
_OOM_SCRIPT = r'''
import ctypes, json, sys
import torch
import tc_signal as ts
from tc_signal import _tc_signal as _C
site = sys.argv[1]
ts.synchronize()
if site == 'cuda:event_query':
    y = torch.ones(1 << 16, device='cuda')
    out = torch.empty_like(y)
    ts.rint(y, out=out)                       # one pending entry (released by the next drain or synchronize)
    pending = _C.pending_stats()['entries']
else:
    pending = 0
_C._test_inject([site, 'alloc:error_message'])
res = {'pending_before': pending}
try:
    ts.synchronize()
    res['first'] = 'success'
except MemoryError:
    res['first'] = 'MemoryError'
res['persisted'] = _C.pending_stats()['unreported_drain_failures']
try:
    ts.synchronize()
    res['second'] = 'success'
except RuntimeError as e:
    res['second'] = str(e)
try:
    ts.synchronize()
    res['third'] = 'success'
except RuntimeError as e:
    res['third'] = str(e)
st = _C.pending_stats()
res['after'] = st['unreported_drain_failures']
res['entries_after'] = st['entries']
res['armed_after'] = _C._test_injection_armed()
print('RESULT ' + json.dumps(res), flush=True)
'''


@pytest.mark.parametrize('site', ['cuda:sync', 'cuda:current_device', 'cuda:event_query'])
def test_fix6_v603_drain_failure_survives_a_message_allocation_failure(site):
    r, res = _run(_OOM_SCRIPT, site)
    _ok(r, res)
    assert res['first'] == 'MemoryError' and res['persisted'] == 1, res
    assert INJECTED_ERROR in res['second'], res                      # acd42fa / mutant (c): 'success'
    assert res['third'] == 'success' and res['after'] == 0 and res['armed_after'] == 0, res   # exactly once
    if site == 'cuda:event_query':
        assert res['pending_before'] == 1 and res['entries_after'] == 0, res
        assert 'deliberately leaked' in res['second'], res              # no completion proof: leaked, not released


# ================================================================================================== sweep (g)
class Recorder:
    """Every Python method an argument conversion could call records itself (and answers plausibly), so a test can
    prove that NONE was called. Prior art: none known to me (a recording test double)."""
    CALLS = []

    def _hit(self, name):
        Recorder.CALLS.append(name)

    def __index__(self):
        self._hit('__index__'); return 1
    def __int__(self):
        self._hit('__int__'); return 1
    def __float__(self):
        self._hit('__float__'); return 1.0
    def __bool__(self):
        self._hit('__bool__'); return True
    def __len__(self):
        self._hit('__len__'); return 1
    def __getitem__(self, i):
        self._hit('__getitem__')
        if i == 0:
            return 'clear'
        raise IndexError
    def __iter__(self):
        self._hit('__iter__'); return iter(['clear'])
    def __repr__(self):
        self._hit('__repr__'); return 'Recorder()'
    def __str__(self):
        self._hit('__str__'); return 'Recorder'
    def __format__(self, spec):
        self._hit('__format__'); return 'Recorder'
    def __eq__(self, other):
        self._hit('__eq__'); return False
    def __hash__(self):
        self._hit('__hash__'); return 1


class RecordingTuple(tuple):
    def __len__(self):
        Recorder.CALLS.append('tuple.__len__'); return tuple.__len__(self)
    def __getitem__(self, i):
        Recorder.CALLS.append('tuple.__getitem__'); return tuple.__getitem__(self, i)


class RecordingInt(int):
    def __index__(self):
        Recorder.CALLS.append('int.__index__'); return int(self)
    def __int__(self):
        Recorder.CALLS.append('int.__int__'); return int(self)
    def __bool__(self):
        Recorder.CALLS.append('int.__bool__'); return True


# (binding, positional arguments with operands as None, indices of the typed (non-operand) parameters)
_BINDINGS = [
    ('conv1d_plan', [None] * 3 + [1, 0, 0, 1, 1], [3, 4, 5, 6, 7]),
    ('conv1d_run', [None] * 5 + [1, 0, 0, 1, 1], [5, 6, 7, 8, 9]),
    ('conv_transpose1d_plan', [None] * 3 + [1, 0, 0, 0, 1, 1], [3, 4, 5, 6, 7, 8]),
    ('conv_transpose1d_run', [None] * 4 + [1, 0, 0, 0, 1, 1], [4, 5, 6, 7, 8, 9]),
    ('lstm_plan', [None] * 6 + [4, 0], [6, 7]),
    ('lstm_run', [None] * 10 + [4, 0], [10, 11]),
    ('instance_norm1d_plan', [None] * 3 + [1e-5], [3]),
    ('instance_norm1d_run', [None] * 4 + [1e-5], [4]),
    ('layer_norm_plan', [None] * 3 + [-1, 1e-5], [3, 4]),
    ('layer_norm_run', [None] * 4 + [-1, 1e-5], [4, 5]),
    ('group_norm_plan', [None, 1, None, None, 1e-5], [1, 4]),
    ('group_norm_run', [None, 1, None, None, None, 1e-5], [1, 5]),
    ('upsample_nearest1d_plan', [None, 2], [1]),
    ('upsample_nearest1d_run', [None, None, 2], [2]),
    ('reflection_pad1d_plan', [None, 1, 1], [1, 2]),
    ('reflection_pad1d_run', [None, None, 1, 1], [2, 3]),
    ('cublas_config', [0], [0]),
    ('lstm_cublas_config', [0], [0]),
    ('pending_stats', [False], [0]),
    ('_exit_close_admission', [5.0], [0]),
    ('_test_inject', [['clear']], [0]),
]
_NO_ARG = ['rint_plan', 'rint_run', '_test_rint_then_raise', 'fp32_divide_probe', 'build_info', 'synchronize',
           'm4_tuning', '_test_injection_armed']


# P1T-FIX7: the exit entry's census in a fresh process (it closes admission for any value; see the test below).
_EXIT_ENTRY_CENSUS_SCRIPT = r'''
import json
from tc_signal import _tc_signal as _C
CALLS = []
class Rec:
    def __getattr__(self, n):
        CALLS.append(n)
        raise AttributeError(n)
    def __index__(self): CALLS.append('__index__'); return 1
    def __float__(self): CALLS.append('__float__'); return 1.0
    def __int__(self): CALLS.append('__int__'); return 1
    def __repr__(self): CALLS.append('__repr__'); return 'Rec()'
class RecTuple(tuple):
    def __len__(self): CALLS.append('tuple.__len__'); return 2
class RecInt(int):
    def __index__(self): CALLS.append('int.__index__'); return int(self)
    def __float__(self): CALLS.append('int.__float__'); return 1.0
out = []
for v in (Rec(), RecTuple((1, 2)), 'text', b'\\x00', RecInt(3)):
    r = _C._exit_close_admission(v)
    out.append('default' if r['first_diag_default_used'] else 'read')
print('RESULT ' + json.dumps({'calls': CALLS, 'outcomes': out,
                              'closed': _C.pending_stats()['release_admission_closed']}), flush=True)
'''


@pytest.mark.parametrize('name,args,typed', _BINDINGS, ids=[b[0] for b in _BINDINGS])
def test_fix6_sweep_g_typed_parameters_refuse_foreign_objects_without_calling_them(name, args, typed):
    from tc_signal import _tc_signal as _C
    if name == '_exit_close_admission':
        # P1T-FIX7 (round-7 review V7-01, order "Apply the same rule to the native raw entries"): the exit entry now
        # CLOSES AND WAITS for any value of its diagnostic argument (an unreadable one selects the default delay), so
        # calling it here would close release admission in this pytest process. The same foreign objects are passed
        # to it in a fresh process instead: none of their methods may be called, each must close with the default.
        r, res = _run(_EXIT_ENTRY_CENSUS_SCRIPT)
        _ok(r, res)
        assert res['calls'] == [], res
        assert res['outcomes'] == ['default'] * 4 + ['read'], res          # 4 foreign objects; the int subclass is read
        assert res['closed'] is True, res
        return
    fn = getattr(_C, name)
    for i in typed:
        for bad in (Recorder(), RecordingTuple((1, 2)), 'text', b'\x00' if name != '_test_inject' else 7):
            a = list(args)
            a[i] = bad
            Recorder.CALLS.clear()
            with pytest.raises(TypeError) as ei:
                fn(*a)
            assert Recorder.CALLS == [], (name, i, Recorder.CALLS)
            assert 'refused without calling' in str(ei.value), str(ei.value)
        if name in ('_test_inject', '_exit_close_admission'):
            continue                    # accepting a value here would arm hooks / close admission in this process
        a = list(args)
        if not isinstance(args[i], (float, bool)):   # a real reads 2**80 exactly; a flag reads any int as nonzero
            a[i] = 1 << 80                                              # an int out of range: OverflowError
            Recorder.CALLS.clear()
            with pytest.raises((OverflowError, TypeError)):
                fn(*a)
        a[i] = RecordingInt(int(args[i]))                               # an int subclass: read at C level
        Recorder.CALLS.clear()
        try:
            fn(*a)
        except (TypeError, ValueError, RuntimeError):
            pass                                                        # operands are None: refused later, in the body
        assert Recorder.CALLS == [], (name, i, Recorder.CALLS)


def test_fix6_sweep_g_test_inject_reads_a_list_without_calling_methods():
    from tc_signal import _tc_signal as _C
    Recorder.CALLS.clear()
    with pytest.raises(TypeError):
        _C._test_inject(Recorder())                                     # acd42fa: __len__ / __getitem__ called
    with pytest.raises(TypeError):
        _C._test_inject([Recorder()])
    assert Recorder.CALLS == []
    assert _C._test_inject(RecordingTuple(('clear',))) == 0 and Recorder.CALLS == []


@pytest.mark.parametrize('name', [b[0] for b in _BINDINGS] + _NO_ARG)
def test_fix6_sweep_g_arity_and_keyword_errors_never_call_repr(name):
    from tc_signal import _tc_signal as _C
    fn = getattr(_C, name)
    Recorder.CALLS.clear()
    with pytest.raises(TypeError) as e1:
        fn(*([Recorder()] * 20))                                        # too many positional arguments
    with pytest.raises(TypeError) as e2:
        fn(no_such_parameter=Recorder())                                # unknown keyword
    assert 'no_such_parameter' in str(e2.value), str(e2.value)
    assert Recorder.CALLS == [], (name, Recorder.CALLS)                 # acd42fa: repr() of every argument
    assert 'Recorder()' not in str(e1.value) + str(e2.value)


def test_fix6_sweep_g_keywords_and_missing_arguments():
    from tc_signal import _tc_signal as _C
    assert _C.pending_stats(reset_high_water=False)['entries'] >= 0
    with pytest.raises(TypeError, match='multiple values'):
        _C.pending_stats(False, reset_high_water=False)
    with pytest.raises(TypeError, match="missing required argument 'scale'"):
        _C.upsample_nearest1d_plan(None)
    assert _C.cublas_config()['default_math'] in (True, False)          # optional argument default


# A daemon thread whose argument's __index__ would sleep across the start of finalization, called WITH KEYWORDS (the
# case that crashes: pybind11's private kwargs copy is deallocated by the exit unwind). FIX6: refused, never called.
_DISPATCH_EXIT_SCRIPT = r'''
import atexit, sys, threading, time, types
import tc_signal as ts
from tc_signal import _tc_signal as _C
started = threading.Event()
class Slow:
    def __index__(self):
        print('INDEX started', flush=True); started.set(); time.sleep(0.6); print('INDEX finished', flush=True); return 1
    def __float__(self):
        return float(self.__index__())
    def __bool__(self):
        return bool(self.__index__())
which = sys.argv[1]
def worker():
    try:
        if which == 'upsample':
            _C.upsample_nearest1d_plan(x=None, scale=Slow())
        elif which == 'pending_stats':
            _C.pending_stats(reset_high_water=Slow())
        elif which == 'close':
            # P1T-FIX7 (V7-01): not refused any more: the exit entry reads the value at C level (Slow is neither a
            # float nor an int: the default delay), closes and waits; Slow's methods are still never called.
            r = _C._exit_close_admission(first_diag_s=Slow())
            print('CLOSED default_used=%s' % r['first_diag_default_used'], flush=True)
    except (TypeError, ValueError):
        print('REFUSED', flush=True)
    print('WORKER ended', flush=True)
def start():
    th = threading.Thread(target=worker, daemon=True)
    th.start()
    if started.wait(2):
        print('CAST IN PROGRESS', flush=True)
    th.join(2)
atexit.register(start)
class Final:
    def __init__(self):
        self.sleep, self.fin = time.sleep, sys.is_finalizing
    def __del__(self):
        print('FINALIZING', self.fin(), flush=True); self.sleep(1.5)
h = types.ModuleType('_fix6_dispatch_holder'); h.keep = Final(); sys.modules[h.__name__] = h
print('ARMED', flush=True)
'''


@pytest.mark.parametrize('which', ['upsample', 'pending_stats', 'close'])
def test_fix6_sweep_g_keyword_conversion_at_exit_is_refused_not_run(which):
    r = subprocess.run([sys.executable, '-c', _DISPATCH_EXIT_SCRIPT, which], capture_output=True, text=True,
                       timeout=SUBPROC_TIMEOUT, env=_env())
    assert r.returncode == 0, f'rc={r.returncode}\n{r.stdout}\n{r.stderr}'
    _no_abort(r)
    # P1T-FIX7 (V7-01): the exit entry closes with the default delay instead of refusing; nothing is called either way.
    expected = 'CLOSED default_used=True' if which == 'close' else 'REFUSED'
    assert expected in r.stdout and 'INDEX started' not in r.stdout, r.stdout


def test_fix6_sweep_g_consumed_capsule_drop_is_an_admitted_release():
    # In process: the producer's Python capsule destructor of a CONSUMED capsule runs inside an admitted release.
    import ctypes
    sys.path.insert(0, str(TESTS_DIR))
    import tcs_producer as P
    from tc_signal import _tc_signal as _C
    new = ctypes.pythonapi['PyCapsule_New']
    new.restype = ctypes.py_object
    new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    seen = []
    @P.CAPSULE_DESTRUCTOR
    def destructor(_cap):
        seen.append(_C.pending_stats()['releases_in_flight'])
    sh = (ctypes.c_int64 * 1)(1)
    keep = []
    def managed(versioned, flags=0, data=0x1000):
        m = P.DLManagedTensorVersioned() if versioned else P.DLManagedTensor()
        if versioned:
            m.version, m.flags = P.DLPackVersion(1, 3), flags
        t = m.dl_tensor
        t.data, t.device, t.dtype, t.ndim, t.shape = data, P.DLDevice(2, 0), P.DLDataType(2, 32, 1), 1, sh
        t.strides = ctypes.POINTER(ctypes.c_int64)()
        keep.append(m)
        return m
    class Prod:
        def __dlpack_device__(self):
            return (2, 0)
        def __dlpack__(self, **kw):
            return new(ctypes.addressof(managed(True, P.FLAG_READ_ONLY, 0x2000)), b'dltensor_versioned',
                       ctypes.cast(destructor, ctypes.c_void_p))
    x = new(ctypes.addressof(managed(False)), b'dltensor', None)
    s0 = _C.pending_stats()['admitted_releases']
    with pytest.raises(ValueError, match='read-only'):
        _C.rint_run(x, Prod())
    s1 = _C.pending_stats()['admitted_releases']
    assert seen == [1], seen                       # ran once, inside an admitted release (acd42fa: [0], unadmitted)
    assert s1 - s0 >= 1, (s0, s1)
