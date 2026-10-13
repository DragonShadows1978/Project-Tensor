"""P1T-FIX2 regression tests for the round-2 review findings in tc_signal (order P1T_FIX2_ROBUSTNESS.md).

What each group pins, and the registered mutant it must kill (G3):
  * N1, CUDA errors on synchronize() and the release paths are never reported as success; mutant (a) = the failed
    current-device query swallowed again:
      - with no usable CUDA device (CUDA_VISIBLE_DEVICES='', the reviewer's empty_sync condition) synchronize() raises
        RuntimeError naming cudaGetDevice / cudaErrorNoDevice, and the atexit hook logs one line instead of raising;
      - an injected stream-sync failure with an entry pending raises, names the error, and leaks (never releases) it;
      - a failure inside a call's destructor (fallback sync, bound drain), which cannot raise, is raised once by the
        next synchronize().
  * N2, the zero-extent rule in validate_tensor and every plan: a shape with a 0 next to 2^62 / 2^63-1 extents (zero
    first, middle, last) is refused as an OVERFLOW (ValueError naming it), never as merely empty, for all nine plans;
    plus the native UBSan census of checked_math.h and validate_tensor (fix2_native_harness.cpp ck_census).
  * N3, allocation failure on the release path never terminates; mutant (c) = the pending-push catch removed (the
    test's subprocess must then terminate): named sites (owner vector, event-pool slot, pending push, event recycle,
    error message) through the _C._test_inject hook, each in its own process; plus the native real-allocation sweep
    (fix2_native_harness.cpp oom_sweep: fail the k-th operator new on every release path).
  * N4, a throwing producer deleter (native C++, built here) is contained on the immediate, deferred (next call),
    bound-drain, fallback-sync, synchronize() and atexit paths: each of the two managers' deleters runs exactly once,
    the second still runs after the first threw, deleter_exceptions counts both, rc 0; mutant (d) = the catch removed.

Skip guard (as test_contract.py): only when torch is not importable or there is no CUDA device. A missing compiler or
missing pybind11 headers for the native builds FAILS (it is not a skip).

Prior art: the round-2 reviewer's probes (artifacts/p1t_rev2/probes/host_checks.py, cpu_native.cpp), whose scenarios
these tests reuse; fault injection at named sites and fail-the-Nth-allocation sweeps (Linux failslab, SQLite's OOM
test harness); DeMillo, Lipton & Sayward (1978) mutation testing for the registered mutants. Ours: the site list, the
throwing-deleter path matrix and the release-or-leak-exactly-once invariant.
"""
import glob
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
PYBIND11_INCLUDE = os.environ.get('TC_SIGNAL_PYBIND11_INCLUDE',
                                  '/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src/include')
LONG_SLEEP_CYCLES = 400_000_000   # torch.cuda._sleep: ~200 ms of legacy-stream work, keeps an entry "not ready"
BIG_NUMEL = 40 << 20              # 160 MiB fp32 per operand: x + out = 320 MiB > the 256 MiB pending bound
CENSUS_3D = [(0, 1 << 62, 4), (1 << 62, 0, 4), (1 << 62, 4, 0),             # zero first / middle / last, 2^62
             (0, (1 << 63) - 1, 2), ((1 << 63) - 1, 2, 0), (2, 0, (1 << 63) - 1),   # 2^63 - 1
             (0, (1 << 63) - 1, 1), (1, 0, 1 << 62)]                        # element bound fits, bytes overflow
CENSUS_2D = [(0, (1 << 63) - 1), ((1 << 63) - 1, 0), (0, 1 << 62), (1 << 61, 0)]


def _cuda_include():
    home = os.environ.get('CUDA_HOME')
    if not home:
        cands = sorted(glob.glob('/usr/local/cuda-*/'), key=lambda p: [int(x) if x.isdigit() else x for x in
                                                                      p.rstrip('/').split('-')[-1].split('.')])
        home = cands[-1].rstrip('/') if cands else '/usr/local/cuda'
    return home


def _env(**extra):
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.update(extra)
    return env


def _run(script, *args, env=None, timeout=SUBPROC_TIMEOUT):
    r = subprocess.run([sys.executable, '-c', script, *args], capture_output=True, text=True, timeout=timeout,
                       env=env or _env())
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith('RESULT ')]
    res = json.loads(lines[-1][len('RESULT '):]) if lines else None
    return r, res


def _ok(r, res):
    assert r.returncode == 0 and res is not None, f'rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}'


_PRELUDE = r'''
import atexit, ctypes, json, os, sys
def _report_at_exit():
    g = globals()
    if g.get('REPORT_AT_EXIT'):
        print('ATEXIT ' + json.dumps({'calls': int(g['native_calls']()) if 'native_calls' in g else None,
                                      'deleted': g['P'].STATS['deleted'], 'unknown_delete': g['P'].STATS['unknown_delete'],
                                      'stats': dict(g['_C'].pending_stats())}), flush=True)
atexit.register(_report_at_exit)     # registered BEFORE tc_signal is imported: runs AFTER tc_signal's own exit hook
sys.path.insert(0, TESTS_DIR)
import torch
import tc_signal as ts
from tc_signal import _tc_signal as _C
import tcs_producer as P
ts.set_allocator(lambda s: torch.empty(s, device='cuda', dtype=torch.float32))
_new = ctypes.pythonapi['PyCapsule_New']
_new.restype = ctypes.py_object
_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
KEEP = []
def native_capsule(base, deleter_addr):
    """A versioned DLPack capsule over torch memory whose deleter is a NATIVE function (address)."""
    n = base.numel()
    sh = (ctypes.c_int64 * 1)(n)
    m = P.DLManagedTensorVersioned()
    t = m.dl_tensor
    t.data = base.data_ptr()
    t.device = P.DLDevice(2, base.device.index)
    t.ndim = 1
    t.dtype = P.DLDataType(2, 32, 1)
    t.shape = sh
    t.strides = ctypes.POINTER(ctypes.c_int64)()
    t.byte_offset = 0
    m.version = P.DLPackVersion(1, 3)
    m.flags = 0
    m.manager_ctx = None
    m.deleter = P.DELETER(deleter_addr)
    KEEP.append((m, sh, base))
    return _new(ctypes.addressof(m), b'dltensor_versioned', None)
def delta(s0, s1):
    return {k: s1[k] - s0[k] for k in s1 if isinstance(s1[k], int)}
'''.replace('TESTS_DIR', repr(str(TESTS_DIR)))


# ================================================================================================== N1
_N1_NO_DEVICE = r'''
import sys
import tc_signal as ts
try:
    ts.synchronize()
    print('RESULT {"raised": false}', flush=True)
except RuntimeError as e:
    import json
    print('RESULT ' + json.dumps({"raised": True, "type": type(e).__name__, "msg": str(e)}), flush=True)
'''


def test_n1_synchronize_raises_when_the_device_query_fails():
    """The reviewer's empty_sync case: no usable device, nothing pending. d116ac0 returned success (rc 9 in the probe)."""
    r, res = _run(_N1_NO_DEVICE, env=_env(CUDA_VISIBLE_DEVICES=''))
    _ok(r, res)
    assert res['raised'] is True, f'synchronize() reported success with no usable CUDA device: {res}'
    assert 'cudaGetDevice' in res['msg'] and 'cudaErrorNoDevice' in res['msg'], res


def test_n1_atexit_hook_logs_instead_of_raising():
    r, res = _run(_N1_NO_DEVICE, env=_env(CUDA_VISIBLE_DEVICES=''))
    _ok(r, res)
    assert 'tc_signal: synchronize() at exit failed (RuntimeError' in r.stderr, r.stderr
    assert 'cudaErrorNoDevice' in r.stderr, r.stderr
    assert 'Exception ignored in atexit callback' not in r.stderr and 'Traceback' not in r.stderr, r.stderr


_N1_GPU = _PRELUDE + r'''
mode = sys.argv[1]
N = (40 << 20) if mode == 'bound_drain' else 4096
x = torch.full((N,), 1.25, device='cuda')
y = torch.empty(N, device='cuda')
ts.rint(torch.ones(8, device='cuda'), out=torch.empty(8, device='cuda'))   # warm: event pool, cuBLAS-free path
ts.synchronize()
P.reset_stats()
s0 = dict(_C.pending_stats())
res = {}
if mode == 'sync_failure':
    torch.cuda._sleep(LONG)                    # the entry below stays "not ready" while synchronize() runs
    ts.rint(P.Producer(x, (N,)), out=P.Producer(y, (N,)))
    _C._test_inject(['cuda:sync'])
    try:
        ts.synchronize()
        res['raised'] = False
    except RuntimeError as e:
        res['raised'], res['msg'] = True, str(e)
    res['deleted_after_failed_sync'] = P.STATS['deleted']
elif mode in ('deferred', 'bound_drain'):
    if mode == 'deferred':
        _C._test_inject(['cuda:event_record', 'cuda:sync'])   # no event -> fallback sync -> it fails -> leak
    else:
        torch.cuda._sleep(LONG)
        _C._test_inject(['cuda:sync'])                        # the bound drain's sync fails
    ts.rint(P.Producer(x, (N,)), out=P.Producer(y, (N,)))    # returns normally: a destructor cannot raise
    res['unreported_after_call'] = _C.pending_stats()['unreported_release_errors']
    res['deleted_after_call'] = P.STATS['deleted']
    try:
        ts.synchronize()
        res['raised'] = False
    except RuntimeError as e:
        res['raised'], res['msg'] = True, str(e)
torch.cuda.synchronize()
try:
    ts.synchronize()                                          # reported once: the next one succeeds
    res['second_sync_ok'] = True
except RuntimeError as e:
    res['second_sync_ok'], res['second_msg'] = False, str(e)
res['delta'] = delta(s0, dict(_C.pending_stats()))
res['unknown_delete'] = P.STATS['unknown_delete']
res['deleted'] = P.STATS['deleted']
print('RESULT ' + json.dumps(res), flush=True)
'''.replace('LONG', str(LONG_SLEEP_CYCLES))


def test_n1_failed_sync_raises_names_the_error_and_leaks():
    r, res = _run(_N1_GPU, 'sync_failure')
    _ok(r, res)
    assert res['raised'] is True, res
    assert 'could not be synchronized' in res['msg'] and 'cudaErrorLaunchFailure' in res['msg'], res
    assert 'deliberately leaked' in res['msg'], res
    assert res['deleted_after_failed_sync'] == 0, res          # never released without a completion proof
    assert res['delta']['leaked_managers'] == 2 and res['deleted'] == 0, res
    assert res['second_sync_ok'] is True and res['unknown_delete'] == 0, res


@pytest.mark.parametrize('mode', ['deferred', 'bound_drain'])
def test_n1_destructor_release_failure_is_raised_once_by_next_synchronize(mode):
    r, res = _run(_N1_GPU, mode)
    _ok(r, res)
    assert res['unreported_after_call'] == 1, res
    assert res['raised'] is True and 'earlier release-path failure' in res['msg'], res
    assert 'cudaErrorLaunchFailure' in res['msg'], res
    assert res['second_sync_ok'] is True, res                  # raised exactly once
    assert res['delta']['release_errors'] == 1 and res['unknown_delete'] == 0, res
    if mode == 'deferred':
        assert res['delta']['fallback_syncs'] == 1 and res['deleted_after_call'] == 0, res
        assert res['delta']['leaked_managers'] == 2, res
    else:
        assert res['delta']['bound_drains'] == 1, res
        assert res['deleted'] + res['delta']['leaked_managers'] == 2, res


# ================================================================================================== N2
_N2_PLANS = r'''
import ctypes as C, json, sys
from tc_signal import _tc_signal as n
newcap = C.pythonapi['PyCapsule_New']
newcap.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p]
newcap.restype = C.py_object
class DLDevice(C.Structure): _fields_ = [('device_type', C.c_int32), ('device_id', C.c_int32)]
class DLDataType(C.Structure): _fields_ = [('code', C.c_uint8), ('bits', C.c_uint8), ('lanes', C.c_uint16)]
class DLTensor(C.Structure):
    _fields_ = [('data', C.c_void_p), ('device', DLDevice), ('ndim', C.c_int32), ('dtype', DLDataType),
                ('shape', C.POINTER(C.c_int64)), ('strides', C.POINTER(C.c_int64)), ('byte_offset', C.c_uint64)]
class DLManagedTensor(C.Structure):
    _fields_ = [('dl_tensor', DLTensor), ('manager_ctx', C.c_void_p), ('deleter', C.c_void_p)]
keep = []
def cap(shape):
    """A descriptor-only capsule (metadata is all a plan reads; nothing is allocated or launched)."""
    sh = (C.c_int64 * len(shape))(*shape)
    m = DLManagedTensor()
    t = m.dl_tensor
    t.data, t.device, t.ndim, t.dtype, t.shape = 0x1000, DLDevice(2, 0), len(shape), DLDataType(2, 32, 1), sh
    keep.extend([sh, m])
    return newcap(C.addressof(m), b'dltensor', None)
w1 = lambda: cap((1, 1, 1))
plans = {
  'conv1d.x': lambda s: n.conv1d_plan(cap(s), w1(), None, 1, 0, 0, 1, 1),
  'conv1d.w': lambda s: n.conv1d_plan(cap((1, 1, 4)), cap(s), None, 1, 0, 0, 1, 1),
  'conv_transpose1d.x': lambda s: n.conv_transpose1d_plan(cap(s), w1(), None, 1, 0, 0, 0, 1, 1),
  'lstm.x': lambda s: n.lstm_plan(cap(s), cap((1, 4, 1)), cap((1, 4, 1)), None, None, None, 1, 0),
  'lstm.W': lambda s: n.lstm_plan(cap((1, 1, 1)), cap(s), cap((1, 4, 1)), None, None, None, 1, 0),
  'instance_norm1d.x': lambda s: n.instance_norm1d_plan(cap(s), None, None, 1e-5),
  'group_norm.x': lambda s: n.group_norm_plan(cap(s), 1, None, None, 1e-5),
  'layer_norm.x': lambda s: n.layer_norm_plan(cap(s), None, None, -1, 1e-5),
  'upsample_nearest1d.x': lambda s: n.upsample_nearest1d_plan(cap(s), 2),
  'reflection_pad1d.x': lambda s: n.reflection_pad1d_plan(cap(s), 0, 0),
  'rint.x': lambda s: n.rint_plan(cap(s)),
}
any_rank = {'layer_norm.x', 'upsample_nearest1d.x', 'reflection_pad1d.x', 'rint.x'}
out = []
for name, fn in plans.items():
    for s in json.loads(sys.argv[1]) + (json.loads(sys.argv[2]) if name in any_rank else []):
        try:
            r = fn(tuple(s))
            out.append({'plan': name, 'shape': s, 'accepted': True, 'result': repr(r)[:120]})
        except (ValueError, TypeError) as e:
            out.append({'plan': name, 'shape': s, 'accepted': False, 'type': type(e).__name__, 'msg': str(e)})
print('RESULT ' + json.dumps(out), flush=True)
'''


def test_n2_every_plan_refuses_zero_extent_overflow_as_overflow():
    r, res = _run(_N2_PLANS, json.dumps(CENSUS_3D), json.dumps(CENSUS_2D))
    _ok(r, res)
    assert len(res) == 11 * len(CENSUS_3D) + 4 * len(CENSUS_2D)
    bad = [c for c in res if c['accepted'] or c['type'] != 'ValueError' or 'overflows' not in c['msg']]
    assert not bad, json.dumps(bad[:6], indent=1)


# ------------------------------------------------------------------------------------------------ native harness
@pytest.fixture(scope='module')
def native_harness(tmp_path_factory):
    d = tmp_path_factory.mktemp('fix2_harness')
    exe = d / 'fix2_native_harness'
    cuda = _cuda_include()
    flags = subprocess.run(['python3-config', '--includes', '--embed', '--ldflags'], capture_output=True, text=True,
                           check=True).stdout.split()
    cmd = ['g++', '-std=c++17', '-O1', '-g', '-fsanitize=undefined', '-fno-sanitize-recover=undefined',
           str(TESTS_DIR / 'fix2_native_harness.cpp'), f'-I{SRC_DIR}', f'-I{INCLUDE_DIR}', f'-I{PYBIND11_INCLUDE}',
           f'-I{cuda}/include', *flags, f'-L{cuda}/lib64', f'-Wl,-rpath,{cuda}/lib64', '-lcudart', '-o', str(exe)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f'native harness build failed:\n{" ".join(cmd)}\n{r.stderr[-4000:]}'
    return str(exe)


def _harness(exe, mode):
    r = subprocess.run([exe, mode], capture_output=True, text=True, timeout=600, env=_env())
    lines = [json.loads(ln) for ln in r.stdout.splitlines() if ln.startswith('{')]
    summary = lines[-1].get('summary') if lines else None
    return r, lines, summary


def test_n2_native_ck_census_refuses_without_ubsan_diagnostics(native_harness):
    r, lines, summary = _harness(native_harness, 'ck_census')
    assert 'runtime error' not in r.stderr, r.stderr                      # UBSan (fno-sanitize-recover aborts too)
    assert r.returncode == 0 and summary and summary['failures'] == 0, (r.returncode, r.stdout[-3000:], r.stderr)


def test_n3_native_real_allocation_failure_sweep_never_terminates(native_harness):
    r, lines, summary = _harness(native_harness, 'oom_sweep')
    assert 'runtime error' not in r.stderr and 'terminate called' not in r.stderr, r.stderr[-3000:]
    assert r.returncode == 0 and summary and summary['pass'] is True, (r.returncode, summary, r.stdout[-3000:])
    assert summary['allocs_in_drain_and_sync'] == 0 and summary['injected_in_dtor'] >= 1, summary


# ================================================================================================== N3
_N3 = _PRELUDE + r'''
site = sys.argv[1]
N = 4096
x = torch.full((N,), 1.25, device='cuda')
y = torch.empty(N, device='cuda')
if site != 'alloc:event_pool_reserve':           # that site needs the very first event creation of the process
    ts.rint(torch.ones(8, device='cuda'), out=torch.empty(8, device='cuda'))
    ts.synchronize()
P.reset_stats()
s0 = dict(_C.pending_stats())
res = {'site': site}
if site == 'alloc:error_message':
    torch.cuda._sleep(LONG)
    ts.rint(P.Producer(x, (N,)), out=P.Producer(y, (N,)))
    _C._test_inject(['cuda:sync', 'alloc:error_message'])
    try:
        ts.synchronize()
        res['outcome'] = 'returned'
    except MemoryError:
        res['outcome'] = 'MemoryError'
    except RuntimeError as e:
        res['outcome'], res['msg'] = 'RuntimeError', str(e)
elif site == 'alloc:event_recycle':
    ts.rint(P.Producer(x, (N,)), out=P.Producer(y, (N,)))
    _C._test_inject([site])
    ts.synchronize()                              # release_entry -> recycle_event fails -> event destroyed
    res['outcome'] = 'returned'
else:
    _C._test_inject([site])
    try:
        ts.rint(P.Producer(x, (N,)), out=P.Producer(y, (N,)))
        res['outcome'] = 'returned'
    except MemoryError:
        res['outcome'] = 'MemoryError'
    res['deleted_right_after_call'] = P.STATS['deleted']
res['armed_left'] = _C._test_injection_armed()
torch.cuda.synchronize()
res['deleted_before_recovery'] = P.STATS['deleted']
res['leaked'] = _C.pending_stats()['leaked_managers'] - s0['leaked_managers']
# the module still works afterwards
x2 = torch.full((N,), 2.5, device='cuda')
y2 = torch.empty(N, device='cuda')
ts.rint(P.Producer(x2, (N,)), out=P.Producer(y2, (N,)))
ts.synchronize()
torch.cuda.synchronize()
res['recovered_correct'] = bool(torch.all(y2 == 2.0).item())
res['delta'] = delta(s0, dict(_C.pending_stats()))
res['deleted'] = P.STATS['deleted']
res['exported'] = P.STATS['exported']
res['unknown_delete'] = P.STATS['unknown_delete']
print('RESULT ' + json.dumps(res), flush=True)
'''.replace('LONG', str(LONG_SLEEP_CYCLES))


@pytest.mark.parametrize('site', ['alloc:owner_reserve', 'alloc:event_pool_reserve', 'alloc:pending_push',
                                  'alloc:event_recycle', 'alloc:error_message'])
def test_n3_allocation_failure_at_each_release_site_never_terminates(site):
    r, res = _run(_N3, site)
    _ok(r, res)                                                  # mutant (c) terminates here (rc -6)
    assert res['armed_left'] == 0, res                           # the injected site was really reached
    assert res['unknown_delete'] == 0, res                       # no deleter called twice
    assert res['recovered_correct'] is True, res
    if site in ('alloc:owner_reserve', 'alloc:event_pool_reserve'):
        # constructor path, before any launch: MemoryError, both operands released at once
        assert res['outcome'] == 'MemoryError' and res['deleted_right_after_call'] == 2, res
    elif site == 'alloc:pending_push':
        # release path inside the noexcept destructor: caught, synchronous fallback, released at once
        assert res['outcome'] == 'returned' and res['deleted_right_after_call'] == 2, res
        assert res['delta']['push_alloc_failures'] == 1 and res['delta']['fallback_syncs'] == 1, res
    elif site == 'alloc:event_recycle':
        assert res['outcome'] == 'returned' and res['delta']['event_recycle_failures'] == 1, res
        assert res['deleted_before_recovery'] == 2, res
    else:  # error message: an error (MemoryError), not success; the entry had no completion proof: leaked
        assert res['outcome'] == 'MemoryError', res
        assert res['deleted_before_recovery'] + res['leaked'] == 2, res
    assert res['deleted'] + res['delta']['leaked_managers'] == res['exported'], res


# ================================================================================================== N4
_THROWER_SRC = r'''
// P1T-FIX2 test producer: a DLPack deleter that throws a C++ exception (out of the DLPack C contract on purpose).
#include <atomic>
#include <stdexcept>
static std::atomic<long> g_calls{0};
extern "C" void fix2_throwing_deleter(void*) {
  g_calls.fetch_add(1);
  throw std::runtime_error("fix2 test producer deleter threw");
}
extern "C" long fix2_throwing_deleter_calls() { return g_calls.load(); }
'''


@pytest.fixture(scope='module')
def thrower(tmp_path_factory):
    d = tmp_path_factory.mktemp('fix2_thrower')
    src, so = d / 'fix2_thrower.cpp', d / 'fix2_thrower.so'
    src.write_text(_THROWER_SRC)
    r = subprocess.run(['g++', '-shared', '-fPIC', '-O1', '-std=c++17', '-o', str(so), str(src)], capture_output=True,
                       text=True, timeout=SUBPROC_TIMEOUT)
    assert r.returncode == 0, r.stderr
    return str(so)


_N4 = _PRELUDE + r'''
lib = ctypes.CDLL(sys.argv[1])
native_calls = lib.fix2_throwing_deleter_calls
native_calls.restype = ctypes.c_long
thrower = ctypes.cast(lib.fix2_throwing_deleter, ctypes.c_void_p).value
mode = sys.argv[2]
N = (40 << 20) if mode == 'bound_drain' else 4096
x = torch.full((N,), 1.25, device='cuda')
y = torch.empty(N, device='cuda')
ts.rint(torch.ones(8, device='cuda'), out=torch.empty(8, device='cuda'))
ts.synchronize()
s0 = dict(_C.pending_stats())
cx, cy = native_capsule(x, thrower), native_capsule(y, thrower)
res = {'mode': mode}
if mode == 'immediate':
    ybad = torch.empty(N + 1, device='cuda')
    cy = native_capsule(ybad, thrower)
    try:
        ts.rint(cx, out=cy)                    # refused after both are consumed, before the launch
    except ValueError as e:
        res['refusal'] = str(e)
elif mode == 'deferred':
    ts.rint(cx, out=cy)
    torch.cuda.synchronize()
    res['calls_before_next_call'] = int(native_calls())
    ts.rint(torch.ones(8, device='cuda'), out=torch.empty(8, device='cuda'))   # its drain releases the entry
elif mode == 'synchronize':
    ts.rint(cx, out=cy)
    ts.synchronize()
elif mode == 'bound_drain':
    ts.rint(cx, out=cy)                        # 320 MiB > the 256 MiB bound: drained in the call's destructor
elif mode == 'fallback_sync':
    _C._test_inject(['cuda:event_record'])
    ts.rint(cx, out=cy)                        # no event -> synchronous release in the destructor
elif mode == 'atexit':
    ts.rint(cx, out=cy)
    REPORT_AT_EXIT = True                      # released by tc_signal's own exit hook; reported after it
res['calls'] = int(native_calls())
res['delta'] = delta(s0, dict(_C.pending_stats()))
if mode != 'atexit':
    torch.cuda.synchronize()
    res['correct'] = bool(torch.all(y == 1.0).item()) if mode != 'immediate' else None
print('RESULT ' + json.dumps(res), flush=True)
'''


@pytest.mark.parametrize('mode', ['immediate', 'deferred', 'bound_drain', 'fallback_sync', 'synchronize', 'atexit'])
def test_n4_throwing_producer_deleter_is_contained_on_every_path(thrower, mode):
    r, res = _run(_N4, thrower, mode)
    _ok(r, res)                                                   # mutant (d) terminates here (rc -6)
    assert 'terminate called' not in r.stderr, r.stderr
    if mode == 'atexit':
        line = [ln for ln in r.stdout.splitlines() if ln.startswith('ATEXIT ')]
        assert line, r.stdout
        at = json.loads(line[-1][len('ATEXIT '):])
        assert res['calls'] == 0 and at['calls'] == 2, (res, at)          # both deleters ran at exit, once each
        assert at['stats']['deleter_exceptions'] == 2 and at['stats']['entries'] == 0, at
        return
    assert res['calls'] == 2, res                                 # each throwing deleter ran exactly once ...
    assert res['delta']['deleter_exceptions'] == 2, res           # ... and both exceptions were contained, counted
    if mode == 'immediate':
        assert 'shape' in res.get('refusal', ''), res
    else:
        assert res['correct'] is True, res
    if mode == 'deferred':
        assert res['calls_before_next_call'] == 0, res
    if mode == 'bound_drain':
        assert res['delta']['bound_drains'] == 1, res
    if mode == 'fallback_sync':
        assert res['delta']['fallback_syncs'] == 1, res
