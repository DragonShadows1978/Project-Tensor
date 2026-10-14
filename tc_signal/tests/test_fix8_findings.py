"""P1T-FIX8 regression tests for tc_signal (order P1T_FIX8_ERROR_VISIBILITY_AT_EXIT.md): round-8 findings V8-01 and
V8-02, and the add-on rows of the sweeps (k) "consumed CUDA errors" and (l) "the last reader at exit". The core rows
are in tensor_cuda/tests/test_dlpack.py test_fix8_*.

What each group pins, and the registered mutant it must kill (G4):
  * V8-01 (the exit device wait's marker event, FIX7 L1): a one-shot failure of each marker call (create, record,
    query, destroy) is reported exactly once by the synchronize() that is running, naming its call, purpose and device;
    with a failed message allocation it is kept and raised by the next synchronize(). A failed create, record or query
    still gives a diagnosed, completed wait (a watchdog thread diagnoses the stream sync); "finished" is printed only
    after completion was proved (a successful query, or a successful stream sync). A healthy control writes no
    failure. On a host trace module of this source with per-call failure plans, and on the real module with GPU work
    queued at exit (test hooks cuda:exit_marker_*). Mutant (e) = the record of a failed marker call removed.
  * V8-02 (the last reader at exit): the native runner writes the recorded failures nobody raised; when the full report
    cannot be built it writes an allocation-free summary (count and first record: call, purpose, device, error name);
    with stderr closed nothing crashes and the records are still consumed; the hook's exception is preserved and
    quiescence (close and wait) comes first. A report synchronize() raised at exit that the hook could not write is
    written by the runner. Mutant (f) = the add-on exit logger's allocation-free fallback removed.
  * Sweep (k), raising paths: a raise builds its report BEFORE it consumes the error slot, so an allocation failure
    while building it leaves the failure in the slot (cuda_guard.h helpers) or records it (acquire_event).
  * Lead send-back L2 (stopping the watchdog): a failed stop-lock (its notify lost) costs no silent wait, and a failed
    join detaches only after run() finished (no watchdog call after the detach). Mutant (h) = FIX8's stop().

Skip guard (as the other tc_signal test modules): only when torch is not importable or there is no CUDA device. A
missing compiler FAILS (it is not a skip).

Prior art: the round-8 reviewer's probes (artifacts/p1t_rev8/probes/diagnostic_errors_cpu.cpp, exit_log_oom_cpu.cpp,
diagnostic_events_gpu.py), whose stub layout and failure census these tests reuse and extend; SQLite's OOM-testing
discipline (fail an allocation, check the error still surfaces); FIX5 V5-02's record-before-consume rule; CUDA 12.6
Runtime API event and error-slot contracts; DeMillo, Lipton & Sayward (1978) mutation testing. Ours: the watchdog and
proof assertions, the acknowledged exit report stagings, and the raising-path OOM census.
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
API = {'create': 'cudaEventCreateWithFlags', 'record': 'cudaEventRecord', 'query': 'cudaEventQuery',
       'destroy': 'cudaEventDestroy'}
MARKER = "the marker event of interpreter exit's diagnosed wait for the legacy stream of device 0"


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
    for bad in _ABORT_MARKERS:
        assert bad not in r.stderr, f'{bad!r} in stderr:\n{r.stderr}'


# ======================================================================================= host trace module (V8-01, k)
# The reviewer's stub layout (the 11 CUDA runtime calls dlpack_operand.cpp makes, a per-thread error slot) plus: a
# marker query that answers NotReady N times, a stream sync that takes N ms (so a watchdog has something to diagnose),
# and an operator new that fails while armed (the trace's message-allocation failures).
_TRACE_SRC = r'''
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <new>
#include <string>
#include <thread>
namespace py = pybind11;
static thread_local cudaError_t slot = cudaSuccess;
static int create_fail = 0, record_fail = 0, query_fail = 0, destroy_fail = 0, sync_fail = 0;
static int query_notready = 0, sync_delay_ms = 0, current_device = 0;
static bool oom_after_create_fail = false;
static bool fail_allocations = false;
void* operator new(std::size_t n) { if (fail_allocations) throw std::bad_alloc(); void* p = std::malloc(n ? n : 1); if (!p) throw std::bad_alloc(); return p; }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
extern "C" cudaError_t cudaGetDevice(int* d) { *d = current_device; return cudaSuccess; }
extern "C" cudaError_t cudaSetDevice(int d) { current_device = d; return cudaSuccess; }
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t) {
  if (sync_delay_ms) std::this_thread::sleep_for(std::chrono::milliseconds(sync_delay_ms));
  if (sync_fail) { --sync_fail; slot = cudaErrorInvalidValue; return slot; }
  return cudaSuccess;
}
extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) {
  if (create_fail) {
    --create_fail; slot = cudaErrorMemoryAllocation;
    if (oom_after_create_fail) { oom_after_create_fail = false; fail_allocations = true; }
    return slot;
  }
  *e = reinterpret_cast<cudaEvent_t>(16); return cudaSuccess;
}
extern "C" cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { if (record_fail) { --record_fail; slot = cudaErrorInvalidValue; return slot; } return cudaSuccess; }
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t) { if (destroy_fail) { --destroy_fail; slot = cudaErrorInvalidValue; return slot; } return cudaSuccess; }
extern "C" cudaError_t cudaEventQuery(cudaEvent_t) {
  if (query_fail) { --query_fail; slot = cudaErrorInvalidValue; return slot; }
  if (query_notready) { --query_notready; slot = cudaErrorNotReady; return cudaErrorNotReady; }
  return cudaSuccess;
}
extern "C" cudaError_t cudaGetLastError() { auto e = slot; slot = cudaSuccess; return e; }
extern "C" cudaError_t cudaPeekAtLastError() { return slot; }
extern "C" const char* cudaGetErrorName(cudaError_t e) {
  switch (e) {
    case cudaSuccess: return "cudaSuccess";
    case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
    case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
    case cudaErrorNotReady: return "cudaErrorNotReady";
    case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
    default: return "cudaErrorUnknown";
  }
}
extern "C" const char* cudaGetErrorString(cudaError_t e) { return e == cudaSuccess ? "no error" : "injected failure"; }
#include "dlpack_operand.cpp"
// A raising helper of cuda_guard.h under allocation failure: returns (outcome, slot after).
template <typename F> static py::tuple guarded(bool oom, F f) {
  const char* outcome = "returned";
  slot = cudaErrorInvalidValue;
  fail_allocations = oom;
  try { f(); } catch (const std::bad_alloc&) { outcome = "bad_alloc"; } catch (const std::runtime_error&) { outcome = "runtime_error"; }
  fail_allocations = false;
  const int left = static_cast<int>(slot);
  slot = cudaSuccess;
  return py::make_tuple(outcome, left);
}
PYBIND11_MODULE(_p1t_fix8_trace, m) {
  tcs::init_pending_state();
  m.def("exit_window", [](double s) { tcs::exit_device_waits_diagnosed(s); });
  m.def("end_window", [] { tcs::exit_device_waits_undiagnosed(); });
  m.def("synchronize", &tcs::synchronize_and_release_all);
  m.def("stats", [] { return tcs::pending_stats(false); });
  m.def("inject", [](const std::string& s) {
    if (s == "create") create_fail = 1; else if (s == "record") record_fail = 1; else if (s == "query") query_fail = 1;
    else if (s == "destroy") destroy_fail = 1; else if (s == "sync") sync_fail = 1;
    else if (s == "message") tcs::test_hooks::arm(tcs::test_hooks::kAllocErrorMessage);
    else throw std::runtime_error("unknown injection");
  });
  m.def("plan", [](int notready, int delay_ms) { query_notready = notready; sync_delay_ms = delay_ms; });
  m.def("slot", [] { return static_cast<int>(slot); });
  // Sweep (k): acquire_event's create failure while its report cannot allocate (allocations fail from the failed call on).
  m.def("acquire_with_oom", [] {
    create_fail = 1; oom_after_create_fail = true;
    const char* outcome = "returned";
    try { (void)tcs::acquire_event(0); } catch (const std::bad_alloc&) { outcome = "bad_alloc"; } catch (const std::exception&) { outcome = "other"; }
    fail_allocations = false;
    const int left = static_cast<int>(slot);
    return py::make_tuple(outcome, left);
  });
  m.def("guard_helper", [](const std::string& which, bool oom) {
    if (which == "launch") return guarded(oom, [] { tcs::check_launch_on("p: ", "k", 0); });
    if (which == "cublas") return guarded(oom, [] { tcs::check_after_cublas_on("p: ", "f", 0); });
    if (which == "call") return guarded(oom, [] { tcs::raise_call_failure_on("p: ", "w", "cudaMemsetAsync", 0, cudaErrorInvalidValue); });
    static const std::string what("p: a call");
    return guarded(oom, [] { tcs::raise_cuda_failure(what, cudaErrorInvalidValue); });
  });
}
'''


@pytest.fixture(scope='module')
def trace_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp('fix8_trace')
    src = d / 'fix8_trace.cpp'
    src.write_text(_TRACE_SRC)
    inc = subprocess.run(['python3-config', '--includes'], capture_output=True, text=True, check=True).stdout.split()
    suffix = subprocess.run(['python3-config', '--extension-suffix'], capture_output=True, text=True,
                            check=True).stdout.strip()
    so = d / f'_p1t_fix8_trace{suffix}'
    r = subprocess.run(['g++', '-std=c++17', '-O1', '-shared', '-fPIC', str(src), f'-I{SRC_DIR}', f'-I{INCLUDE_DIR}',
                        f'-I{PYBIND11_INCLUDE}', f'-I{CUDA_HOME}/include', *inc, '-pthread', '-o', str(so)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    return str(d)


# One process per case (the trace's C stdio lines go to fd 2). The window's delay is 0.01 s; the stream sync takes
# 60 ms, so a watchdog writes lines; then synchronize() is called until it succeeds (at most 3 times).
_TRACE_SCRIPT = r'''
import json, sys
import _p1t_fix8_trace as n
site, oom = sys.argv[1], sys.argv[2] == 'oom'
notready = 30 if site in ('none', 'destroy') else 0     # the marker answers NotReady ~30 ms: lines, then proof
n.plan(notready, 60 if site != 'none' and site != 'destroy' else 0)
n.exit_window(0.01)
if site not in ('none',):
    for s in site.split('+'):
        n.inject(s)
if oom:
    n.inject('message')
calls = []
for _ in range(3):
    try:
        n.synchronize()
        calls.append(['ok', None])
        break
    except RuntimeError as e:
        calls.append(['RuntimeError', str(e)])
    except MemoryError as e:
        calls.append(['MemoryError', str(e)])
n.end_window()
st = n.stats()
print('RESULT ' + json.dumps({'calls': calls, 'slot': n.slot(), 'marker': st['exit_marker_failures'],
                              'watchdogs': st['exit_device_wait_watchdogs'],
                              'unreported': st['unreported_unraisable_errors']}), flush=True)
'''


def _trace(trace_dir, *args):
    return _run(_TRACE_SCRIPT, *args, env=_env(PYTHONPATH=trace_dir, CUDA_VISIBLE_DEVICES=''))


@pytest.mark.parametrize('oom', ['plain', 'oom'])
@pytest.mark.parametrize('site', ['create', 'record', 'query', 'destroy'])
def test_fix8_v801_failed_marker_call_is_reported_once_naming_call_purpose_device(trace_dir, site, oom):
    r, res = _trace(trace_dir, site, oom)
    _ok(r, res)
    calls = res['calls']
    if oom == 'oom':
        assert calls[0][0] == 'MemoryError', res       # its message could not be built: the record is kept...
        calls = calls[1:]
    assert calls[0][0] == 'RuntimeError', res          # ...and raised (by the next synchronize() under OOM)
    msg = calls[0][1]
    assert f'({API[site]}: cudaError' in msg and MARKER in msg, msg
    assert msg.count(API[site]) == 1, msg               # named exactly once
    assert calls[1] == ['ok', None], res                # reported exactly once: the next synchronize() succeeds
    assert res['marker'] == 1 and res['unreported'] == 0 and res['slot'] == 0, res
    err = r.stderr
    if site == 'destroy':                               # completion was proved by the marker before its destroy
        assert res['watchdogs'] == 0, res
        assert 'interpreter exit is waiting' in err and '(its marker event completed)' in err, err
    else:                                               # the marker failed: a watchdog diagnosed the stream sync
        assert res['watchdogs'] == 1, res
        assert 'diagnosed by a watchdog thread, because the wait\'s marker event failed' in err, err
        assert 'finished after' in err and '(cudaStreamSynchronize returned)' in err, err
        assert '(its marker event completed)' not in err, err     # no completion claimed from a failed marker


def test_fix8_v801_finished_is_printed_only_after_proof(trace_dir):
    # The marker query fails AND the stream sync that follows fails: nothing proved completion, so no "finished".
    r, res = _trace(trace_dir, 'query+sync', 'plain')
    _ok(r, res)
    msg = res['calls'][0][1]
    assert '(cudaEventQuery: cudaErrorInvalidValue' in msg and 'cudaStreamSynchronize: cudaErrorInvalidValue' in msg, msg
    assert 'interpreter exit is waiting' in r.stderr and 'finished after' not in r.stderr, r.stderr


def test_fix8_v801_healthy_marker_writes_no_failure(trace_dir):
    r, res = _trace(trace_dir, 'none', 'plain')
    _ok(r, res)
    assert res['calls'] == [['ok', None]] and res['marker'] == 0 and res['watchdogs'] == 0, res
    assert 'interpreter exit is waiting' in r.stderr and '(its marker event completed)' in r.stderr, r.stderr


def test_fix8_sweep_k_acquire_event_failure_survives_its_report_allocation_failure(trace_dir):
    # The completion event cannot be created and the RuntimeError naming it cannot be built: the failure is recorded
    # (tagged) before the slot entry is consumed, and the next synchronize() raises it (FIX7: consumed, then lost).
    r, res = _run(r'''
import json
import _p1t_fix8_trace as n
outcome = n.acquire_with_oom()
calls = []
for _ in range(2):
    try:
        n.synchronize(); calls.append(None)
    except RuntimeError as e:
        calls.append(str(e))
print('RESULT ' + json.dumps({'outcome': list(outcome), 'calls': calls}), flush=True)
''', env=_env(PYTHONPATH=trace_dir, CUDA_VISIBLE_DEVICES=''))
    _ok(r, res)
    assert res['outcome'] == ['bad_alloc', 0], res
    assert 'creating a call\'s completion event on device 0 failed (cudaEventCreateWithFlags: ' \
           'cudaErrorMemoryAllocation' in (res['calls'][0] or ''), res
    assert res['calls'][1] is None, res                 # exactly once


@pytest.mark.parametrize('which', ['launch', 'cublas', 'call', 'cuda'])
def test_fix8_sweep_k_raise_helpers_consume_only_after_their_report_exists(trace_dir, which):
    r, res = _run(r'''
import json, sys
import _p1t_fix8_trace as n
print('RESULT ' + json.dumps({'oom': list(n.guard_helper(sys.argv[1], True)),
                              'control': list(n.guard_helper(sys.argv[1], False))}), flush=True)
''', which, env=_env(PYTHONPATH=trace_dir, CUDA_VISIBLE_DEVICES=''))
    _ok(r, res)
    assert res['oom'] == ['bad_alloc', 1], res          # the report could not be built: the entry stays in the slot
    assert res['control'] == ['runtime_error', 0], res  # the report exists: the entry is consumed (rule 1)


# ======================================================================================= real module, GPU (V8-01)
# GPU work (~0.3 s of torch.cuda._sleep on the legacy default stream) is queued, then the native runner calls a hook
# that calls synchronize() until it succeeds. The window's delay is this module's _RELEASE_EXIT_DIAG_S (0.02 s), read
# by the runner from the hook's globals. Test hooks cuda:exit_marker_* fail one marker call.
_REAL_SCRIPT = r'''
import json, sys, time
import torch
import tc_signal as ts
from tc_signal import _tc_signal as C
site, oom = sys.argv[1], sys.argv[2] == 'oom'
torch.zeros(1, device='cuda'); torch.cuda.synchronize()
t = time.perf_counter(); torch.cuda._sleep(50_000_000); torch.cuda.synchronize()
RATE = 50_000_000 / (time.perf_counter() - t)
_RELEASE_EXIT_DIAG_S = 0.02
calls = []
def hook():
    for _ in range(3):
        try:
            ts.synchronize(); calls.append(['ok', None]); return
        except RuntimeError as e:
            calls.append(['RuntimeError', str(e)])
        except MemoryError as e:
            calls.append(['MemoryError', str(e)])
torch.cuda._sleep(int(RATE * 0.3))
names = [] if site == 'none' else ['cuda:exit_marker_' + site]
if oom:
    names.append('alloc:error_message')
if names:
    C._test_inject(names)
C._exit_hook_runner(hook)
st = C.pending_stats()
print('RESULT ' + json.dumps({'calls': calls, 'marker': st['exit_marker_failures'],
                              'watchdogs': st['exit_device_wait_watchdogs'], 'lines': st['exit_device_wait_diagnostics'],
                              'unreported': st['unreported_unraisable_errors']}), flush=True)
'''


@pytest.mark.parametrize('oom', ['plain', 'oom'])
@pytest.mark.parametrize('site', ['create', 'record', 'query', 'destroy'])
def test_fix8_v801_real_module_reports_each_marker_failure_and_still_diagnoses_the_wait(site, oom):
    r, res = _run(_REAL_SCRIPT, site, oom)
    _ok(r, res)
    calls = res['calls']
    if oom == 'oom':
        assert calls[0][0] == 'MemoryError', res
        calls = calls[1:]
    msg = calls[0][1] or ''
    assert calls[0][0] == 'RuntimeError' and f'({API[site]}: {INJECTED_ERROR}' in msg and MARKER in msg, res
    assert calls[1] == ['ok', None] and res['marker'] == 1 and res['unreported'] == 0, res
    err = r.stderr
    assert res['lines'] >= 1 and 'interpreter exit is waiting for the work queued on the legacy stream of CUDA device 0' \
        in err, err                                     # the wait was diagnosed (~0.3 s of work, 0.02 s delay)
    if site == 'destroy':
        assert res['watchdogs'] == 0 and '(its marker event completed)' in err, err
    else:
        assert res['watchdogs'] == 1 and '(cudaStreamSynchronize returned)' in err, err
        assert '(its marker event completed)' not in err, err


def test_fix8_v801_real_module_healthy_control():
    r, res = _run(_REAL_SCRIPT, 'none', 'plain')
    _ok(r, res)
    assert res['calls'] == [['ok', None]] and res['marker'] == 0 and res['watchdogs'] == 0, res
    assert res['lines'] >= 1 and '(its marker event completed)' in r.stderr, r.stderr
    assert 'could not be' not in r.stderr and 'allocation-free summary' not in r.stderr, r.stderr


# ======================================================================================= real module at exit (V8-02)
# An event-query failure of the next-call drain is recorded (FIX7's staging), the hook is interrupted at its first
# bytecode, and the runner is the last reader: it runs synchronize() (whose message cannot be allocated: MemoryError,
# the records kept), closes and waits, then logs the records. 'oom' also fails the logger's full report.
_EXIT_SCRIPT = r'''
import atexit, json, os, sys, _thread
RES = {}
def _report():                               # registered before tc_signal: runs after its runner
    st = _C.pending_stats()
    RES['after'] = {k: st[k] for k in ('unreported_unraisable_errors', 'exit_log_fallbacks', 'exit_runner_closes',
                                       'exit_hook_calls', 'exit_report_unwritten')}
    os.write(1, ('RESULT ' + json.dumps(RES) + '\n').encode())
atexit.register(_report)
import torch
import tc_signal as ts
from tc_signal import _tc_signal as _C
MODE = sys.argv[1]
ts.set_allocator(lambda s: torch.empty(s, device='cuda'))
x = torch.full((1024,), 1.5, device='cuda')
ts.rint(x)
_C._test_inject(['cuda:event_query'])        # the next-call drain's query fails once: recorded
ts.rint(x)
if MODE in ('control', 'oom', 'oom_stderr_closed'):
    names = ['alloc:error_message'] + (['alloc:exit_log'] if MODE != 'control' else [])
    atexit.register(_thread.interrupt_main)  # the hook is interrupted at its first bytecode
    atexit.register(_C._test_inject, names)  # armed right before (runs before the interrupt)
    if MODE == 'oom_stderr_closed':
        atexit.register(os.close, 2)
elif MODE in ('hook_cannot_write', 'hook_writes'):
    if MODE == 'hook_cannot_write':
        def swap():
            sys.stderr = None                # the hook's own lines fail; fd 2 stays open for C stdio
        atexit.register(swap)
'''


def _exit(mode):
    r, res = _run(_EXIT_SCRIPT, mode)
    _ok(r, res)
    return r, res['after']


def test_fix8_v802_exit_logger_full_report_control():
    r, a = _exit('control')
    assert 'CUDA failures nobody reported yet: ' in r.stderr and 'cudaEventQuery of a pending entry' in r.stderr, r.stderr
    assert 'allocation-free summary' not in r.stderr, r.stderr
    assert a['unreported_unraisable_errors'] == 0 and a['exit_log_fallbacks'] == 0 and a['exit_runner_closes'] == 1, a
    assert 'KeyboardInterrupt' in r.stderr, r.stderr    # the hook's exception, preserved


def test_fix8_v802_exit_logger_writes_an_allocation_free_summary_when_the_report_cannot_allocate():
    r, a = _exit('oom')
    line = next((ln for ln in r.stderr.splitlines() if 'allocation-free summary' in ln), '')
    assert 'the full report could not be built: it failed to allocate' in line, r.stderr
    assert '1 CUDA failure(s)' in line and 'the first: cudaEventQuery failed, querying a pending entry\'s ' \
           'completion event in the next-call drain, on CUDA device 0: ' + INJECTED_ERROR in line, line
    assert a['unreported_unraisable_errors'] == 0 and a['exit_log_fallbacks'] == 1, a   # written once, consumed
    assert a['exit_runner_closes'] == 1, a              # quiescence first (the runner closed and waited)
    assert 'KeyboardInterrupt' in r.stderr, r.stderr


def test_fix8_v802_exit_logger_with_stderr_closed_does_not_crash_and_consumes():
    r, a = _exit('oom_stderr_closed')
    assert a['unreported_unraisable_errors'] == 0 and a['exit_log_fallbacks'] == 1 and a['exit_runner_closes'] == 1, a


def test_fix8_v802_report_the_hook_could_not_write_is_written_by_the_runner():
    r, a = _exit('hook_cannot_write')
    line = next((ln for ln in r.stderr.splitlines() if 'that the exit hook could not write' in ln), '')
    assert 'synchronize() raised a CUDA failure report' in line and 'allocation-free summary' in line, r.stderr
    assert 'the first: cudaEventQuery failed' in line and 'on CUDA device 0: ' + INJECTED_ERROR in line, line
    assert a['exit_hook_calls'] == 1 and a['exit_report_unwritten'] is False and a['exit_log_fallbacks'] == 1, a


def test_fix8_v802_report_the_hook_wrote_is_not_repeated():
    r, a = _exit('hook_writes')
    assert 'tc_signal: synchronize() at exit failed (RuntimeError: ' in r.stderr, r.stderr
    assert 'could not write' not in r.stderr and 'allocation-free summary' not in r.stderr, r.stderr
    assert a['exit_report_unwritten'] is False and a['exit_log_fallbacks'] == 0, a


# ======================================================================== lead send-back L2: stopping the watchdog
# DeviceWaitWatchdog::stop() (dlpack_operand.cpp) must be safe whatever its lock or its join does:
#  (a) a failed pthread_join must not detach a thread still inside run() (it uses mu_, cv_ and the caller's schedule,
#      all destroyed when sync_device_legacy returns: a use-after-free at exit);
#  (b) a failed stop-lock must not cost a silent wait until the watchdog's next diagnostic (its notify can then be
#      missed: the watchdog between its predicate check and its wait).
# An LD_PRELOAD interposer (pthread_create / mutex lock and unlock / cond clockwait and broadcast / join / detach)
# recognizes the watchdog thread (the first thread the main thread creates once armed, right before the exit runner;
# cross-checked by exit_device_wait_watchdogs == 1) and, inside the window from its creation to its join or detach:
#   * 'lock' n: fails the next n main-thread pthread_mutex_lock calls made from _tc_signal*.so (stop()'s lock), and
#     withholds the notify that follows each failed lock (the worst case of (b): every notify sent without the lock
#     arrives before the watchdog waits). The wait from stop()'s first lock attempt to the join's return is measured.
#   * 'join' n: fails the next n pthread_join calls for the watchdog thread (std::thread::join throws), and holds the
#     watchdog 50 ms after the wakeup stop() sends (a preempted watchdog). Every mutex / condition-variable call the
#     watchdog thread makes from _tc_signal*.so is timed; none may come after the detach, and the last one must come
#     before it (run() had finished: exited_ is its last action, after its last unlock).
#   * 'control': nothing fails; the watchdog is joined normally.
# The marker cannot be created (test hook cuda:exit_marker_create), ~0.4 s of GPU work is queued on the legacy stream,
# and the native runner calls a hook that calls synchronize() until it succeeds; the diagnostic delay is the default
# (5 s), so a missed notify would leave the watchdog asleep for ~4.6 s more. Mutant (h) = stop() reverted to FIX8's.
# Prior art: LD_PRELOAD interposition with dlsym(RTLD_NEXT) and caller filtering by return address + dladdr (FIX7's
# lock-failure shims); fault injection of a lost wakeup (the lost-wakeup problem, Lampson & Redell 1980; unverified
# here, lead to check). Ours: the staging and the after-detach witness.
_L2_SHIM_C = r'''
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef int (*create_fn)(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*);
typedef int (*mutex_fn)(pthread_mutex_t*);
typedef int (*clockwait_fn)(pthread_cond_t*, pthread_mutex_t*, clockid_t, const struct timespec*);
typedef int (*cond_fn)(pthread_cond_t*);
typedef int (*join_fn)(pthread_t, void**);
typedef int (*detach_fn)(pthread_t);
static create_fn r_create; static mutex_fn r_lock, r_unlock; static clockwait_fn r_clockwait;
static cond_fn r_broadcast; static join_fn r_join; static detach_fn r_detach;
__attribute__((constructor)) static void shim_init(void) {
  r_create = (create_fn)dlsym(RTLD_NEXT, "pthread_create");
  r_lock = (mutex_fn)dlsym(RTLD_NEXT, "pthread_mutex_lock");
  r_unlock = (mutex_fn)dlsym(RTLD_NEXT, "pthread_mutex_unlock");
  r_clockwait = (clockwait_fn)dlsym(RTLD_NEXT, "pthread_cond_clockwait");
  r_broadcast = (cond_fn)dlsym(RTLD_NEXT, "pthread_cond_broadcast");
  r_join = (join_fn)dlsym(RTLD_NEXT, "pthread_join");
  r_detach = (detach_fn)dlsym(RTLD_NEXT, "pthread_detach");
}

static int armed, window, wd_created, wd_known, stop_began, drop_next, detached;
static pthread_t main_thread, wd;
static void* (*wd_fn)(void*);
static int lock_fail_left, drop_notify, join_fail_left, hold_ms, held;
static long long locks_failed, notifies_dropped, joins_failed, returned_at_detach, wd_ops, wd_ops_after_detach,
                 wd_returned, t_stop, t_join_ret, t_detach, t_last_wd_op;

static long long now_ns(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
static int from_addon(void* ra) {
  Dl_info info;
  if (!dladdr(ra, &info) || !info.dli_fname) return 0;
  const char* b = strrchr(info.dli_fname, '/'); b = b ? b + 1 : info.dli_fname;
  return strncmp(b, "_tc_signal", 10) == 0;
}
static int on_wd(void) { return __atomic_load_n(&wd_known, __ATOMIC_ACQUIRE) && pthread_equal(pthread_self(), wd); }
static void wd_op(void) {
  __atomic_add_fetch(&wd_ops, 1, __ATOMIC_ACQ_REL);
  __atomic_store_n(&t_last_wd_op, now_ns(), __ATOMIC_RELEASE);
  if (__atomic_load_n(&detached, __ATOMIC_ACQUIRE)) __atomic_add_fetch(&wd_ops_after_detach, 1, __ATOMIC_ACQ_REL);
}
static void* tramp(void* a) {
  wd = pthread_self(); __atomic_store_n(&wd_known, 1, __ATOMIC_RELEASE);
  void* r = wd_fn(a);
  __atomic_store_n(&wd_returned, 1, __ATOMIC_RELEASE);
  return r;
}

int pthread_create(pthread_t* t, const pthread_attr_t* at, void* (*fn)(void*), void* arg) {
  if (__atomic_load_n(&armed, __ATOMIC_ACQUIRE) && !wd_created && pthread_equal(pthread_self(), main_thread)) {
    wd_created = 1; wd_fn = fn;
    int e = r_create(t, at, tramp, arg);
    if (e == 0) { wd = *t; __atomic_store_n(&wd_known, 1, __ATOMIC_RELEASE); __atomic_store_n(&window, 1, __ATOMIC_RELEASE); }
    return e;
  }
  return r_create(t, at, fn, arg);
}
int pthread_mutex_lock(pthread_mutex_t* m) {
  if (__atomic_load_n(&window, __ATOMIC_ACQUIRE) && pthread_equal(pthread_self(), main_thread)
      && from_addon(__builtin_return_address(0))) {
    if (!stop_began) { t_stop = now_ns(); __atomic_store_n(&stop_began, 1, __ATOMIC_RELEASE); }
    if (lock_fail_left > 0) { --lock_fail_left; ++locks_failed; drop_next = drop_notify; return EINVAL; }
  } else if (on_wd() && from_addon(__builtin_return_address(0))) {
    wd_op();
  }
  return r_lock(m);
}
int pthread_mutex_unlock(pthread_mutex_t* m) {
  if (on_wd() && from_addon(__builtin_return_address(0))) wd_op();
  return r_unlock(m);
}
int pthread_cond_clockwait(pthread_cond_t* c, pthread_mutex_t* m, clockid_t ck, const struct timespec* ts) {
  const int w = on_wd() && from_addon(__builtin_return_address(0));
  if (w) wd_op();
  const int e = r_clockwait(c, m, ck, ts);
  if (w && hold_ms > 0 && !held && __atomic_load_n(&stop_began, __ATOMIC_ACQUIRE)) { held = 1; usleep(hold_ms * 1000); }
  if (w) wd_op();
  return e;
}
int pthread_cond_broadcast(pthread_cond_t* c) {
  if (drop_next && pthread_equal(pthread_self(), main_thread)) { drop_next = 0; ++notifies_dropped; return 0; }
  return r_broadcast(c);
}
int pthread_join(pthread_t t, void** ret) {
  const int w = __atomic_load_n(&wd_known, __ATOMIC_ACQUIRE) && pthread_equal(t, wd);
  if (w) {
    __atomic_store_n(&window, 0, __ATOMIC_RELEASE);
    if (join_fail_left > 0) { --join_fail_left; ++joins_failed; return EINVAL; }
  }
  const int e = r_join(t, ret);
  if (w) t_join_ret = now_ns();
  return e;
}
int pthread_detach(pthread_t t) {
  if (__atomic_load_n(&wd_known, __ATOMIC_ACQUIRE) && pthread_equal(t, wd)) {
    __atomic_store_n(&window, 0, __ATOMIC_RELEASE);
    returned_at_detach = __atomic_load_n(&wd_returned, __ATOMIC_ACQUIRE);
    t_detach = now_ns();
    __atomic_store_n(&detached, 1, __ATOMIC_RELEASE);
  }
  return r_detach(t);
}
void shim_arm(int lock_fail, int drop, int join_fail, int hold) {
  main_thread = pthread_self(); lock_fail_left = lock_fail; drop_notify = drop; join_fail_left = join_fail;
  hold_ms = hold; __atomic_store_n(&armed, 1, __ATOMIC_RELEASE);
}
void shim_state(long long* o) {
  usleep(200000);   /* let a detached watchdog finish before reading (its after-detach calls are counted) */
  o[0] = wd_created; o[1] = locks_failed; o[2] = notifies_dropped; o[3] = joins_failed;
  o[4] = __atomic_load_n(&detached, __ATOMIC_ACQUIRE); o[5] = returned_at_detach;
  o[6] = __atomic_load_n(&wd_ops, __ATOMIC_ACQUIRE); o[7] = __atomic_load_n(&wd_ops_after_detach, __ATOMIC_ACQUIRE);
  o[8] = t_stop; o[9] = t_join_ret; o[10] = t_detach; o[11] = __atomic_load_n(&t_last_wd_op, __ATOMIC_ACQUIRE);
  o[12] = __atomic_load_n(&wd_returned, __ATOMIC_ACQUIRE); o[13] = held; o[14] = lock_fail_left; o[15] = join_fail_left;
}
'''

_L2_SCRIPT = r'''
import ctypes, json, sys, time
import torch
import tc_signal as ts
from tc_signal import _tc_signal as C
shim = ctypes.CDLL(sys.argv[1])
mode, n = sys.argv[2], int(sys.argv[3])
torch.zeros(1, device='cuda'); torch.cuda.synchronize()
t = time.perf_counter(); torch.cuda._sleep(50_000_000); torch.cuda.synchronize()
RATE = 50_000_000 / (time.perf_counter() - t)
calls = []
def hook():                                   # no _RELEASE_EXIT_DIAG_S here: the default delay (5 s)
    for _ in range(3):
        try:
            ts.synchronize(); calls.append(['ok', None]); return
        except RuntimeError as e:
            calls.append(['RuntimeError', str(e)])
torch.cuda._sleep(int(RATE * 0.4))            # ~0.4 s of GPU work on the legacy default stream
C._test_inject(['cuda:exit_marker_create'])   # the marker cannot be created: a watchdog diagnoses the stream sync
shim.shim_arm(n if mode == 'lock' else 0, 1 if mode == 'lock' else 0, n if mode == 'join' else 0,
              50 if mode == 'join' else 0)
C._exit_hook_runner(hook)
st = C.pending_stats()
o = (ctypes.c_longlong * 16)()
shim.shim_state(o)
keys = ['wd_created', 'locks_failed', 'notifies_dropped', 'joins_failed', 'detached', 'returned_at_detach', 'wd_ops',
        'wd_ops_after_detach', 't_stop', 't_join_ret', 't_detach', 't_last_wd_op', 'wd_returned', 'held',
        'lock_fail_left', 'join_fail_left']
print('RESULT ' + json.dumps({'calls': calls, 'shim': dict(zip(keys, list(o))),
                              'stats': {k: st[k] for k in ('exit_device_wait_watchdogs', 'exit_marker_failures',
                                                           'exit_device_wait_watchdog_lock_failures',
                                                           'exit_device_wait_watchdog_detaches')}}), flush=True)
'''
_L2_STOP_BOUND_S = 0.5   # stop()'s wait after the sync returned (new: about n poll intervals; FIX8: ~4.6 s)


@pytest.fixture(scope='module')
def l2_shim(tmp_path_factory):
    d = tmp_path_factory.mktemp('l2_shim')
    src, so = d / 'watchdog_shim.c', d / 'watchdog_shim.so'
    src.write_text(_L2_SHIM_C)
    r = subprocess.run(['gcc', '-shared', '-fPIC', '-O1', '-o', str(so), str(src), '-ldl', '-lpthread'],
                       capture_output=True, text=True, timeout=SUBPROC_TIMEOUT)
    assert r.returncode == 0, r.stderr
    return str(so)


def _l2(shim, mode, n):
    r, res = _run(_L2_SCRIPT, shim, mode, str(n), env=_env(LD_PRELOAD=shim))
    _ok(r, res)
    calls, s, st = res['calls'], res['shim'], res['stats']
    assert calls[0][0] == 'RuntimeError' and '(cudaEventCreateWithFlags: ' + INJECTED_ERROR in calls[0][1], res
    assert calls[1] == ['ok', None], res
    assert st['exit_device_wait_watchdogs'] == 1 and s['wd_created'] == 1 and s['wd_returned'] == 1, res
    assert s['wd_ops'] >= 2, res                         # the watchdog's own lock / wait / unlock were seen
    return r, s, st


def test_fix8_l2_watchdog_stop_control(l2_shim):
    r, s, st = _l2(l2_shim, 'control', 0)
    assert s['locks_failed'] == 0 and s['joins_failed'] == 0 and s['detached'] == 0, s
    assert st['exit_device_wait_watchdog_lock_failures'] == 0 and st['exit_device_wait_watchdog_detaches'] == 0, st
    assert 0 <= (s['t_join_ret'] - s['t_stop']) / 1e9 < _L2_STOP_BOUND_S, s


@pytest.mark.parametrize('n', [1, 3])
def test_fix8_l2_failed_stop_lock_costs_no_silent_wait(l2_shim, n):
    r, s, st = _l2(l2_shim, 'lock', n)
    assert s['locks_failed'] == n and s['notifies_dropped'] == n and s['lock_fail_left'] == 0, s   # every notify lost
    assert st['exit_device_wait_watchdog_lock_failures'] == n, st
    waited = (s['t_join_ret'] - s['t_stop']) / 1e9
    assert 0 <= waited < _L2_STOP_BOUND_S, (waited, s)   # FIX8: asleep until the next diagnostic (~4.6 s), silent
    assert s['joins_failed'] == 0 and s['detached'] == 0 and st['exit_device_wait_watchdog_detaches'] == 0, s


def test_fix8_l2_failed_join_detaches_only_after_the_watchdog_finished(l2_shim):
    r, s, st = _l2(l2_shim, 'join', 1)
    assert s['joins_failed'] == 1 and s['detached'] == 1 and s['held'] == 1, s
    assert st['exit_device_wait_watchdog_detaches'] == 1, st
    assert s['wd_ops_after_detach'] == 0, s              # FIX8: the held watchdog resumed inside run() after the detach
    assert s['t_last_wd_op'] < s['t_detach'], s          # its last mutex / wait call came before the detach
