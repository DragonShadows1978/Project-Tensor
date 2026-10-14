"""P1T-FIX4 regression tests for tc_signal (order P1T_FIX4_DEVICE_AND_ERRORS.md): round-4 finding V4-04 and the
device / CUDA-error sweep of the add-on.

What each group pins, and the registered mutant it must kill (G4):
  * V4-04, synchronize() reports an error already pending in the calling thread's CUDA last-error slot, even when every
    stream sync succeeds; mutant (c) = the entry capture removed:
      - a synchronous launch refusal with nothing pending: a NULL kernel (cudaErrorInvalidDeviceFunction, the
        reviewer's sync_existing_error.py staging) and a REAL launch-configuration error (a 4096-thread block, built
        with nvcc here: cudaErrorInvalidConfiguration);
      - the same with entries pending: the drain still releases them (none leaked), then raises;
      - the error is taken from the slot (reported once): the next synchronize() succeeds.
  * Sweep, CUDA error state: a call's launch path takes an error already pending before its first launch and raises
    it as pre-existing, launching nothing and releasing its operands at once (62c8d9f launched, then reported it as
    its own launch failure); an event-query failure in the next-call drain is consumed there (it no longer poisons that
    call's launch check).
  * Sweep, devices (LD_PRELOAD shim fix4_cudart_shim.c simulating "device 1 current" on this one-GPU host; torch-free
    processes): every event create / record / destroy, kernel launch and stream sync of a call whose operands are on
    device 0 runs on device 0 and the caller's device 1 is restored (rint, conv1d with cuBLAS, the synchronize drain);
    a failed switch raises with its slot entry consumed; a failed restore raises on the success path and is counted and
    raised once by the next synchronize() where nothing can raise; an event is destroyed on its own device (a failed
    destroy is counted and raised once); a failed cuBLAS handle setting destroys the new handle and raises.

Skip guard (as the other tc_signal test modules): only when torch is not importable or there is no CUDA device. A
missing gcc or nvcc FAILS (it is not a skip).

Prior art: the round-4 reviewer's probes (artifacts/p1t_rev4/probes/sync_existing_error.py, core_export_device.py),
whose scenarios these tests reuse; LD_PRELOAD interposition (glibc ld.so) as in FIX1's F05 shim; the CUDA 12.6
Programming Guide 3.2.12 rule of clearing the error variable right before a launch; DeMillo, Lipton & Sayward (1978)
mutation testing for the registered mutants. Ours: the shim's call log and genuine-slot failure injection.
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
SUBPROC_TIMEOUT = 120
LONG_SLEEP_CYCLES = 400_000_000   # torch.cuda._sleep: ~200 ms of legacy-stream work, keeps an entry "not ready"
CUDA_HOME = '/usr/local/cuda-12.6'


def _env(**extra):
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.update(extra)
    return env


def _run(script, *args, env=None):
    r = subprocess.run([sys.executable, '-c', script, *args], capture_output=True, text=True, timeout=SUBPROC_TIMEOUT,
                       env=env or _env())
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith('RESULT ')]
    res = json.loads(lines[-1][len('RESULT '):]) if lines else None
    assert r.returncode == 0 and res is not None, f'rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}'
    return res


@pytest.fixture(scope='module')
def stager(tmp_path_factory):
    so = tmp_path_factory.mktemp('fix4_stager') / 'fix4_stage_kernel.so'
    r = subprocess.run([f'{CUDA_HOME}/bin/nvcc', '-shared', '-Xcompiler', '-fPIC', '--cudart', 'shared',
                        '-arch=native', '-o', str(so), str(TESTS_DIR / 'fix4_stage_kernel.cu')],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    return str(so)


@pytest.fixture(scope='module')
def shim(tmp_path_factory):
    so = tmp_path_factory.mktemp('fix4_shim') / 'fix4_cudart_shim.so'
    r = subprocess.run(['gcc', '-shared', '-fPIC', '-O1', '-o', str(so), str(TESTS_DIR / 'fix4_cudart_shim.c'), '-ldl'],
                       capture_output=True, text=True, timeout=SUBPROC_TIMEOUT)
    assert r.returncode == 0, r.stderr
    return str(so)


# ================================================================================================== V4-04 and the slot
_SLOT = r'''
import ctypes, json, sys
sys.path.insert(0, TESTS_DIR)
import torch
import tc_signal as ts
from tc_signal import _tc_signal as _C
import tcs_producer as P
rt = ctypes.CDLL('libcudart.so.12')          # tc_signal's runtime instance (already loaded: resolved by soname)
class Dim3(ctypes.Structure):
    _fields_ = [('x', ctypes.c_uint), ('y', ctypes.c_uint), ('z', ctypes.c_uint)]
rt.cudaLaunchKernel.argtypes = [ctypes.c_void_p, Dim3, Dim3, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
def stage_null_kernel():
    rt.cudaGetLastError()
    return rt.cudaLaunchKernel(None, Dim3(1, 1, 1), Dim3(1, 1, 1), None, 0, None)
mode, stager = sys.argv[1], sys.argv[2]
N = 4096
x = torch.full((N,), 2.5, device='cuda')
y = torch.full((N,), -7.0, device='cuda')
ts.rint(torch.ones(8, device='cuda'), out=torch.empty(8, device='cuda'))   # warm: event pool
ts.synchronize()
torch.cuda.synchronize()
P.reset_stats()
s0 = dict(_C.pending_stats())
res = {'mode': mode}
def attempt(fn, key):
    try:
        fn()
        res[key] = {'raised': None}
    except RuntimeError as e:
        res[key] = {'raised': 'RuntimeError', 'msg': str(e)}
if mode == 'sync_null_kernel':
    res['staged'] = stage_null_kernel()
    attempt(ts.synchronize, 'sync')
elif mode == 'sync_launch_config':
    lib = ctypes.CDLL(stager)
    res['staged'] = lib.fix4_stage_launch_config_error()
    attempt(ts.synchronize, 'sync')
elif mode == 'sync_pending_entries':
    torch.cuda._sleep(LONG)                                   # the entry below stays pending while we stage
    ts.rint(P.Producer(x, (N,)), out=P.Producer(y, (N,)))
    res['entries_before'] = _C.pending_stats()['entries']
    res['staged'] = stage_null_kernel()
    attempt(ts.synchronize, 'sync')
    res['entries_after'] = _C.pending_stats()['entries']
    res['deleted_after_sync'] = P.STATS['deleted']
elif mode == 'op_pending_error':
    res['staged'] = stage_null_kernel()
    attempt(lambda: ts.rint(P.Producer(x, (N,)), out=P.Producer(y, (N,))), 'op')
    res['deleted_right_after'] = P.STATS['deleted']           # refused before the launch: released at once
    res['entries_after'] = _C.pending_stats()['entries']
    torch.cuda.synchronize()
    res['y_untouched'] = bool(torch.all(y == -7.0).item())    # nothing was launched
    ts.rint(P.Producer(x, (N,)), out=P.Producer(y, (N,)))     # the next call is clean
    ts.synchronize()
    torch.cuda.synchronize()
    res['y_after_next'] = bool(torch.all(y == 2.0).item())
res['slot_after'] = rt.cudaPeekAtLastError()
attempt(ts.synchronize, 'second_sync')                        # reported once
torch.cuda.synchronize()
res['delta'] = {k: v - s0[k] for k, v in dict(_C.pending_stats()).items() if isinstance(v, int) and k in s0}
res['deleted'] = P.STATS['deleted']
res['unknown_delete'] = P.STATS['unknown_delete']
print('RESULT ' + json.dumps(res), flush=True)
'''.replace('TESTS_DIR', repr(str(TESTS_DIR))).replace('LONG', str(LONG_SLEEP_CYCLES))


@pytest.mark.parametrize('mode, staged, name', [
    ('sync_null_kernel', 98, 'cudaErrorInvalidDeviceFunction'),
    ('sync_launch_config', 9, 'cudaErrorInvalidConfiguration'),
])
def test_fix4_v404_synchronize_reports_a_preexisting_launch_error_with_nothing_pending(stager, mode, staged, name):
    res = _run(_SLOT, mode, stager)
    assert res['staged'] == staged, res                         # precondition: a synchronous launch refusal is staged
    assert res['sync']['raised'] == 'RuntimeError', res         # 62c8d9f / mutant (c): returned success
    assert 'already pending in this thread when synchronize() was called' in res['sync']['msg'], res
    assert name in res['sync']['msg'], res
    assert res['slot_after'] == 0 and res['second_sync']['raised'] is None, res   # taken: reported exactly once


def test_fix4_v404_synchronize_still_drains_pending_entries_then_raises(stager):
    res = _run(_SLOT, 'sync_pending_entries', stager)
    assert res['entries_before'] == 1 and res['staged'] == 98, res
    assert res['sync']['raised'] == 'RuntimeError' and 'cudaErrorInvalidDeviceFunction' in res['sync']['msg'], res
    assert res['entries_after'] == 0 and res['deleted_after_sync'] == 2, res     # released (completion proven)
    assert res['delta']['leaked_managers'] == 0 and res['unknown_delete'] == 0, res
    assert res['slot_after'] == 0 and res['second_sync']['raised'] is None, res


def test_fix4_op_reports_a_pending_error_as_preexisting_and_launches_nothing(stager):
    res = _run(_SLOT, 'op_pending_error', stager)
    assert res['op']['raised'] == 'RuntimeError', res
    assert 'already pending in this thread before this call' in res['op']['msg'], res
    assert 'cudaErrorInvalidDeviceFunction' in res['op']['msg'] and 'nothing was launched' in res['op']['msg'], res
    assert res['deleted_right_after'] == 2 and res['entries_after'] == 0, res    # nothing queued: released at once
    assert res['y_untouched'] is True and res['y_after_next'] is True, res
    assert res['slot_after'] == 0 and res['second_sync']['raised'] is None and res['unknown_delete'] == 0, res


# ================================================================================================== devices (shim)
_SHIM = r'''
import ctypes, json, sys, types
import numpy as np
sys.path.insert(0, TESTS_DIR)
import tensor_cuda as tc
import tc_signal as ts
from tc_signal import _tc_signal as _C
import tcs_producer as P
shim = ctypes.CDLL(sys.argv[1])                # already loaded through LD_PRELOAD: the same handle
shim.shim_log.restype = ctypes.c_char_p
shim.shim_fail.argtypes = [ctypes.c_char_p]
mode = sys.argv[2]
# Outputs and scratch are allocated BEFORE the simulated device switch (tc.empty labels a tensor with the current
# device, which the shim fakes): the allocator hands out those, keyed by shape.
PRE = {}
def prealloc(*shapes):
    for sh in shapes:
        PRE.setdefault(tuple(sh), []).append(tc.empty(*sh))
ts.set_allocator(lambda s: PRE[tuple(s)].pop(0))
class Base:
    """torch-like base for tcs_producer.Producer over a TensorCUDA tensor (this process must not import torch)."""
    def __init__(self, t):
        self.t, self.device = t, types.SimpleNamespace(index=0)
    def data_ptr(self):
        return self.t.data_ptr()
    def element_size(self):
        return 4
N = 4096
xv = (np.arange(N, dtype=np.float32) % 7) + 0.5
x = tc.tensor(xv)
y = tc.empty(N)
want = np.rint(xv)
res = {'mode': mode}
def rint():
    ts.rint(P.Producer(Base(x), (N,)), out=P.Producer(Base(y), (N,)))
def attempt(fn, key):
    try:
        fn()
        res[key] = {'raised': None}
    except RuntimeError as e:
        res[key] = {'raised': 'RuntimeError', 'msg': str(e)}
def snap(key):
    res[key + '_log'] = shim.shim_log().decode().split()
    res[key + '_current'] = shim.shim_current()
    res[key + '_slot'] = shim.shim_slot_peek()
    shim.shim_clear()
if mode != 'rint_cold':
    rint()                                     # warm: one completion event in the pool, cuBLAS untouched
    ts.synchronize()
P.reset_stats()
if mode in ('rint_cold', 'rint_warm'):
    shim.shim_fake(1)
    shim.shim_clear()
    attempt(rint, 'call')
    snap('call')
    attempt(ts.synchronize, 'sync')
    snap('sync')
elif mode == 'conv_cold':
    w = tc.tensor(np.ones((2, 2, 3), np.float32))
    xc = tc.tensor(np.ones((1, 2, 16), np.float32))
    prealloc((1, 2, 16), (96,))                # conv1d out [N, C_out, T_out] and im2col scratch C_in*K*T_out
    shim.shim_fake(1)
    shim.shim_clear()
    attempt(lambda: ts.conv1d(xc, w, padding=1), 'call')
    snap('call')
elif mode == 'switch_fail':
    shim.shim_fake(1)
    shim.shim_fail_set(0, 1)                   # the launch path's switch to device 0 fails
    shim.shim_clear()
    attempt(rint, 'call')
    snap('call')
    attempt(rint, 'next')                      # 62c8d9f left the failure in the slot: this one then reported it
    snap('next')
elif mode in ('restore_fail_success_path', 'restore_fail_unraisable'):
    shim.shim_fake(1)
    shim.shim_fail_set(1, 1 if mode == 'restore_fail_success_path' else 2)
    shim.shim_clear()
    attempt(rint, 'call')
    snap('call')
    res['entries_after_call'] = _C.pending_stats()['entries']
    res['stats_after_call'] = dict(_C.pending_stats())
    shim.shim_fake(-1)
    attempt(ts.synchronize, 'sync')
    attempt(ts.synchronize, 'second_sync')
elif mode in ('event_destroy_ok', 'event_destroy_fail'):
    rint()                                     # an entry whose event the drain below cannot pool (injected)
    shim.shim_fake(1)
    _C._test_inject(['alloc:event_recycle'])
    if mode == 'event_destroy_fail':
        shim.shim_fail(b'cudaEventDestroy')
    shim.shim_clear()
    attempt(ts.synchronize, 'sync')
    snap('sync')
    attempt(ts.synchronize, 'second_sync')
elif mode == 'query_fail':
    rint()                                     # entry 1, still on the list until the next call's drain
    shim.shim_fail(b'cudaEventQuery')          # the next drain's first query fails, with a genuine slot entry
    shim.shim_clear()
    attempt(rint, 'call')                      # 62c8d9f: the drain left the error, so this call reported it
    snap('call')
    attempt(ts.synchronize, 'sync')
elif mode == 'launch_and_restore_fail':
    shim.shim_fake(1)
    shim.shim_fail(b'cudaLaunchKernel')        # the launch is refused (genuine slot entry) ...
    shim.shim_fail_set(1, 1)                   # ... and the guard's restore while unwinding fails too
    shim.shim_clear()
    attempt(rint, 'call')
    snap('call')
    res['stats_after_call'] = dict(_C.pending_stats())
    shim.shim_fake(-1)
    attempt(ts.synchronize, 'sync')
    attempt(ts.synchronize, 'second_sync')
elif mode == 'lstm_memset_fail':
    T, Nb, I, H = 2, 1, 2, 2
    xl = tc.tensor(np.ones((T, Nb, I), np.float32))
    Wl = tc.tensor(np.ones((1, 4 * H, I), np.float32))
    Rl = tc.tensor(np.ones((1, 4 * H, H), np.float32))
    ts.set_allocator(lambda s: tc.empty(*s))   # no simulated device in this mode
    shim.shim_fail(b'cudaMemsetAsync')         # the initial_h zero-fill fails with a genuine slot entry
    shim.shim_clear()
    attempt(lambda: ts.lstm(xl, Wl, Rl, hidden_size=H), 'call')
    snap('call')
    attempt(rint, 'next')                      # 62c8d9f left the failure in the slot: this one then reported it
    snap('next')
elif mode in ('lstm_device', 'lstm_cublas_setting'):
    T, Nb, I, H = 2, 1, 2, 2
    xl = tc.tensor(np.ones((T, Nb, I), np.float32))
    Wl = tc.tensor(np.ones((1, 4 * H, I), np.float32))
    Rl = tc.tensor(np.ones((1, 4 * H, H), np.float32))
    # Y [T, D, N, H], Y_h and Y_c [D, N, H], scratch D*T*N*4H + D*N*4H; twice (the second call)
    prealloc((T, 1, Nb, H), (1, Nb, H), (1, Nb, H), (24,), (T, 1, Nb, H), (1, Nb, H), (1, Nb, H), (24,))
    if mode == 'lstm_device':
        shim.shim_fake(1)
    else:
        shim.shim_fail(b'cublasSetMathMode')   # the LSTM's own handle (lstm_handle_for) is created by this call
    shim.shim_clear()
    attempt(lambda: ts.lstm(xl, Wl, Rl, hidden_size=H), 'call')
    snap('call')
    attempt(lambda: ts.lstm(xl, Wl, Rl, hidden_size=H), 'next')
    shim.shim_fake(-1)
    res['config'] = dict(_C.lstm_cublas_config(0))
elif mode == 'cublas_setting':
    w = tc.tensor(np.ones((2, 2, 3), np.float32))
    xc = tc.tensor(np.ones((1, 2, 16), np.float32))
    prealloc((1, 2, 16), (96,), (1, 2, 16), (96,))   # two calls
    shim.shim_fail(b'cublasSetMathMode')
    shim.shim_clear()
    attempt(lambda: ts.conv1d(xc, w, padding=1), 'call')
    snap('call')
    attempt(lambda: ts.conv1d(xc, w, padding=1), 'next')
    res['config'] = dict(_C.cublas_config(0))
shim.shim_fake(-1)
attempt(ts.synchronize, 'final_sync')
res['y_correct'] = bool(np.array_equal(y.numpy(), want))
res['stats'] = dict(_C.pending_stats())
res['deleted'] = P.STATS['deleted']
res['exported'] = P.STATS['exported']
res['unknown_delete'] = P.STATS['unknown_delete']
print('RESULT ' + json.dumps(res), flush=True)
'''.replace('TESTS_DIR', repr(str(TESTS_DIR)))


def _shim(shim, mode):
    return _run(_SHIM, shim, mode, env=_env(LD_PRELOAD=shim))


def _devices(log):
    return {tok.split('@')[1].rstrip('!') for tok in log if '@' in tok}


def test_fix4_device_rint_runs_on_the_operand_device_and_restores(shim):
    # Cold: the completion event is created on device 0 too. Device 1 is current throughout (simulated).
    res = _shim(shim, 'rint_cold')
    assert res['call']['raised'] is None, res
    assert res['call_log'] == ['set0', 'create@0', 'set1', 'set0', 'launch@0', 'set1', 'set0', 'record@0', 'set1'], res
    assert res['call_current'] == 1 and res['call_slot'] == 0, res
    # synchronize(): the current device's legacy stream, then the pending entry's device 0, then device 1 restored
    assert res['sync']['raised'] is None and res['sync_log'] == ['sync@1', 'set0', 'sync@0', 'set1'], res
    assert res['deleted'] == 2 and res['y_correct'] is True and res['unknown_delete'] == 0, res


def test_fix4_device_rint_warm_uses_the_pooled_event_and_restores(shim):
    res = _shim(shim, 'rint_warm')
    assert res['call_log'] == ['set0', 'launch@0', 'set1', 'set0', 'record@0', 'set1'], res
    assert res['call_current'] == 1 and res['y_correct'] is True, res


def test_fix4_device_conv1d_with_cublas_runs_on_the_operand_device(shim):
    res = _shim(shim, 'conv_cold')
    assert res['call']['raised'] is None, res
    assert _devices(res['call_log']) == {'0'} and 'launch@0' in res['call_log'], res
    assert res['call_log'][-1] == 'set1' and res['call_current'] == 1 and res['call_slot'] == 0, res


def test_fix4_failed_switch_raises_and_consumes_its_slot_entry(shim):
    res = _shim(shim, 'switch_fail')
    assert res['call']['raised'] == 'RuntimeError', res
    assert 'cudaSetDevice(0) failed: cudaErrorInvalidDevice' in res['call']['msg'], res
    assert res['call_log'][0] == 'set0!' and 'launch@0' not in res['call_log'], res   # nothing launched
    assert res['call_slot'] == 0 and res['call_current'] == 1, res
    assert res['next']['raised'] is None and 'launch@0' in res['next_log'], res      # the next call is clean
    assert res['final_sync']['raised'] is None and res['unknown_delete'] == 0, res


def test_fix4_failed_restore_on_the_success_path_raises(shim):
    res = _shim(shim, 'restore_fail_success_path')
    assert res['call']['raised'] == 'RuntimeError', res
    assert "restoring the caller's CUDA device 1 after queueing work on device 0" in res['call']['msg'], res
    assert res['call_log'] == ['set0', 'launch@0', 'set1!', 'record@0'], res      # the work stays fenced
    assert res['entries_after_call'] == 1 and res['call_slot'] == 0, res          # operands retained, not released
    assert res['sync']['raised'] is None and res['y_correct'] is True and res['deleted'] == 2, res


def test_fix4_failed_restore_where_nothing_can_raise_is_counted_and_raised_once(shim):
    res = _shim(shim, 'restore_fail_unraisable')
    assert res['call']['raised'] is None, res                                    # a destructor cannot raise
    assert res['call_log'] == ['set0', 'launch@0', 'set1', 'set0', 'record@0', 'set1!'], res
    st = res['stats_after_call']
    assert st['device_restore_failures'] == 1 and st['unreported_unraisable_errors'] == 1, res
    assert res['call_slot'] == 0, res
    assert res['sync']['raised'] == 'RuntimeError', res
    assert "restoring the caller's CUDA device (cudaSetDevice) on device 1" in res['sync']['msg'], res
    assert 'cudaErrorInvalidDevice' in res['sync']['msg'], res
    assert res['second_sync']['raised'] is None and res['y_correct'] is True, res


def test_fix4_failed_restore_while_unwinding_is_counted_and_raised_once(shim):
    # DeviceGuard's destructor restores only when an exception is already unwinding (here: a refused launch). 62c8d9f
    # ignored that restore's result; now it is consumed, counted and raised once by the next synchronize().
    res = _shim(shim, 'launch_and_restore_fail')
    # P1T-FIX7 (sweep (j)): the launch check names the kernel and the device and no longer claims "rint_kernel failed"
    # (cudaGetLastError after a launch may report an asynchronous error of earlier work).
    assert res['call']['raised'] == 'RuntimeError', res
    assert 'CUDA error at the launch check of rint_kernel on CUDA device 0: cudaErrorInvalidDevice' in res['call']['msg'], res
    assert res['call_log'][:3] == ['set0', 'launch@0!', 'set1!'] and res['call_slot'] == 0, res
    st = res['stats_after_call']
    assert st['device_restore_failures'] == 1 and st['unreported_unraisable_errors'] == 1, res
    assert res['sync']['raised'] == 'RuntimeError' and "restoring the caller's CUDA device" in res['sync']['msg'], res
    assert res['second_sync']['raised'] is None and res['unknown_delete'] == 0, res


def test_fix4_check_cuda_raise_consumes_the_failed_call(shim):
    # lstm.cu check_cuda(cudaMemsetAsync(...)) raised without consuming the failed call's slot entry in 62c8d9f, so the
    # next call's launch check reported it again as its own.
    res = _shim(shim, 'lstm_memset_fail')
    assert res['call']['raised'] == 'RuntimeError', res
    assert 'initial_h zero failed: cudaErrorInvalidDevice' in res['call']['msg'] and res['call_slot'] == 0, res
    assert '(cudaMemsetAsync on CUDA device 0 returned it)' in res['call']['msg'], res      # P1T-FIX7 sweep (j)
    assert res['next']['raised'] is None and res['final_sync']['raised'] is None, res


@pytest.mark.parametrize('mode', ['event_destroy_ok', 'event_destroy_fail'])
def test_fix4_event_is_destroyed_on_its_own_device(shim, mode):
    res = _shim(shim, mode)
    log = res['sync_log']
    tail = 'destroy@0!' if mode == 'event_destroy_fail' else 'destroy@0'
    i = log.index(tail)
    assert log[i - 1:i + 2] == ['set0', tail, 'set1'], res                        # 62c8d9f: destroyed on device 1
    assert res['sync_current'] == 1 and res['sync_slot'] == 0, res
    if mode == 'event_destroy_fail':
        assert res['sync']['raised'] == 'RuntimeError', res
        assert 'cudaEventDestroy on device 0' in res['sync']['msg'], res
        assert res['stats']['event_destroy_failures'] == 1, res
    else:
        assert res['sync']['raised'] is None and res['stats']['event_destroy_failures'] == 0, res
    assert res['second_sync']['raised'] is None and res['stats']['event_recycle_failures'] == 1, res


def test_fix4_failed_event_query_in_the_drain_is_consumed_there(shim):
    res = _shim(shim, 'query_fail')
    assert 'query@-1!' in res['call_log'], res                                   # the injected drain failure happened
    assert res['call']['raised'] is None and res['call_slot'] == 0, res
    # P1T-FIX5 (round-5 review V5-02): this test used to assert that the next synchronize() returned SUCCESS, which
    # pinned the defect (the consumed query failure was never reported; FIX4 sweep rows 47/50). The failure is now
    # recorded before it is consumed and raised exactly once by the next synchronize(), which still releases every
    # entry; the following synchronize() succeeds.
    assert res['sync']['raised'] == 'RuntimeError' and 'cudaEventQuery' in res['sync']['msg'], res
    assert res['final_sync']['raised'] is None and res['deleted'] == 4 and res['y_correct'] is True, res


def test_fix4_device_lstm_runs_on_the_operand_device_and_restores(shim):
    res = _shim(shim, 'lstm_device')
    assert res['call']['raised'] is None and res['next']['raised'] is None, res
    assert _devices(res['call_log']) == {'0'} and 'launch@0' in res['call_log'], res
    assert res['call_log'][-1] == 'set1' and res['call_current'] == 1 and res['call_slot'] == 0, res


def test_fix4_failed_lstm_cublas_setting_destroys_the_new_handle_and_raises(shim):
    res = _shim(shim, 'lstm_cublas_setting')
    assert res['call']['raised'] == 'RuntimeError', res
    assert 'tc_signal.lstm: cublasSetMathMode(CUBLAS_DEFAULT_MATH) failed: cuBLAS status 14' in res['call']['msg'], res
    assert '(the new handle of CUDA device 0)' in res['call']['msg'], res                  # P1T-FIX7 sweep (j)
    assert res['call_log'][:2] == ['cublasSetMathMode!', 'cublasDestroy'], res
    assert res['next']['raised'] is None and res['config']['default_math'] is True, res


def test_fix4_failed_cublas_setting_destroys_the_new_handle_and_raises(shim):
    res = _shim(shim, 'cublas_setting')
    assert res['call']['raised'] == 'RuntimeError', res
    assert 'cublasSetMathMode(CUBLAS_DEFAULT_MATH) failed: cuBLAS status 14' in res['call']['msg'], res
    assert '(the new handle of CUDA device 0)' in res['call']['msg'], res                  # P1T-FIX7 sweep (j)
    assert res['call_log'][:2] == ['cublasSetMathMode!', 'cublasDestroy'], res
    assert res['next']['raised'] is None and res['config']['default_math'] is True, res
