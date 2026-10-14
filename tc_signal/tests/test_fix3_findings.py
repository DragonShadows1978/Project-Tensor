"""P1T-FIX3 regression tests for round-3 review finding V3-02 in tc_signal (order P1T_FIX3_EXIT_QUIESCENCE.md).

V3-02 (MINOR): individually valid operand byte counts were summed unchecked (per call, inside the noexcept
RetainUntilComplete destructor) and added unchecked to the pending-list total (note_push), so near-limit metadata was
signed-overflow UB and a wrapped negative total could have suppressed the bound drain. What these tests pin:
  * the per-call aggregate is checked BEFORE the first launch, through real entry points: the reviewer's two disjoint
    fp32 descriptors of shape (2**61 - 1,) (rint) and three disjoint (2**60,) ones (fp32_divide_probe) are refused
    with ValueError naming the aggregate overflow, both operands' deleters run exactly once at once, nothing is queued;
    with no visible GPU (no CUDA call can precede the refusal) and with the GPU (real condition; the module still works
    afterwards). Registered mutant (b) = the aggregate pre-launch check removed: these tests then fail (no ValueError).
  * the native UBSan harness (fix3_native_harness.cpp, compiled here with the REAL dlpack_operand.cpp): the saturating
    pending-list total, the bound comparison at the boundary (exactly kPendingMaxBytes stays, 4 more bytes drain; the
    entry bound likewise), and the real retainer constructor/destructor near INT64_MAX with real CUDA events and no
    kernel launch.

Skip guard (as test_fix2_findings.py): only when torch is not importable or there is no CUDA device. A missing compiler
or missing pybind11 headers for the native build FAILS (it is not a skip).

Prior art: the round-3 reviewer's probe artifacts/p1t_rev3/probes/pending_bytes.cpp (its boundary operands are reused);
the checked builtins (GCC 5, 2015; C23 ckd_add) and saturating arithmetic; UBSan (2013-); DeMillo, Lipton & Sayward
(1978) mutation testing for the registered mutant. Ours: the entry-point staging, the pop/empty and boundary sequences.
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


def _cuda_home():
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


# ================================================================================================== pre-launch check
# Torch-free (so it also runs with no visible GPU): legacy capsules over FAKE addresses (never dereferenced when the
# check holds) whose deleter records each call.
_PRELAUNCH = r'''
import ctypes, json, sys
sys.path.insert(0, TESTS_DIR)
import tcs_producer as P
import tc_signal as ts
from tc_signal import _tc_signal as _C
new = ctypes.pythonapi['PyCapsule_New']
new.restype, new.argtypes = ctypes.py_object, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
calls, KEEP = [], []
@P.DELETER
def deleter(addr):
    calls.append(addr)
def cap(n, data):
    sh = (ctypes.c_int64 * 1)(n)
    m = P.DLManagedTensor()
    t = m.dl_tensor
    t.data, t.device, t.ndim, t.dtype = data, P.DLDevice(2, 0), 1, P.DLDataType(2, 32, 1)
    t.shape, t.strides, t.byte_offset = sh, ctypes.POINTER(ctypes.c_int64)(), 0
    m.manager_ctx, m.deleter = None, deleter
    KEEP.append((sh, m))
    return new(ctypes.addressof(m), b'dltensor', None)
op, gpu = sys.argv[1], sys.argv[2] == 'gpu'
s0 = dict(_C.pending_stats())
res = {'op': op}
try:
    if op == 'rint':     # the reviewer's pair: (2**61-1,) at 4 and at 2**63; each 2**63-4 bytes, disjoint
        n = (1 << 61) - 1
        _C.rint_run(cap(n, 4), cap(n, 1 << 63))
    else:                # three disjoint (2**60,) operands, 2**62 bytes each: x + y alone already overflows
        n = 1 << 60
        _C.fp32_divide_probe(cap(n, 0x1000), cap(n, 0x1000 + (1 << 62)), cap(n, 0x1000 + (1 << 63)))
    res['outcome'] = 'returned'
except ValueError as e:
    res['outcome'], res['msg'] = 'ValueError', str(e)
except Exception as e:  # noqa: BLE001
    res['outcome'], res['msg'] = type(e).__name__, str(e)
res['deleter_calls'], res['distinct'] = len(calls), len(set(calls))
s1 = dict(_C.pending_stats())
res['delta_retained'] = s1['retained_entries'] - s0['retained_entries']
res['entries'], res['bytes'], res['saturated'] = s1['entries'], s1['bytes'], s1['bytes_saturated']
if gpu:                  # the module still works after the refusal (real memory, real launch)
    import torch
    x = torch.tensor([1.4, 2.6, -0.5], device='cuda')
    y = torch.empty(3, device='cuda')
    ts.rint(x, out=y)
    ts.synchronize()
    res['after_ok'] = y.cpu().tolist() == [1.0, 3.0, -0.0]
print('RESULT ' + json.dumps(res), flush=True)
'''.replace('TESTS_DIR', repr(str(TESTS_DIR)))


@pytest.mark.parametrize('cond', ['gpu', 'nogpu'])
@pytest.mark.parametrize('op', ['rint', 'divide'])
def test_fix3_v302_aggregate_overflow_refused_before_any_launch(op, cond):
    env = _env(CUDA_VISIBLE_DEVICES='') if cond == 'nogpu' else _env()
    r = subprocess.run([sys.executable, '-c', _PRELAUNCH, op, cond], capture_output=True, text=True,
                       timeout=SUBPROC_TIMEOUT, env=env)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith('RESULT ')]
    assert r.returncode == 0 and lines, f'rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}'
    res = json.loads(lines[-1][len('RESULT '):])
    assert res['outcome'] == 'ValueError', res                                  # mutant (b): no ValueError
    assert 'aggregate fp32 byte count' in res['msg'] and 'overflows' in res['msg'], res
    n_ops = 2 if op == 'rint' else 3
    assert res['deleter_calls'] == n_ops and res['distinct'] == n_ops, res      # each released once, at once
    assert res['delta_retained'] == 0 and res['entries'] == 0 and res['bytes'] == 0, res   # nothing queued
    assert res['saturated'] is False, res
    if cond == 'gpu':
        assert res['after_ok'] is True, res
    assert 'runtime error' not in r.stderr, r.stderr


# ================================================================================================== native harness
@pytest.fixture(scope='module')
def fix3_harness(tmp_path_factory):
    d = tmp_path_factory.mktemp('fix3_harness')
    exe = d / 'fix3_native_harness'
    cuda = _cuda_home()
    flags = subprocess.run(['python3-config', '--includes', '--embed', '--ldflags'], capture_output=True, text=True,
                           check=True).stdout.split()
    cmd = ['g++', '-std=c++17', '-O1', '-g', '-fsanitize=undefined', '-fno-sanitize-recover=undefined',
           str(TESTS_DIR / 'fix3_native_harness.cpp'), f'-I{SRC_DIR}', f'-I{INCLUDE_DIR}', f'-I{PYBIND11_INCLUDE}',
           f'-I{cuda}/include', *flags, f'-L{cuda}/lib64', f'-Wl,-rpath,{cuda}/lib64', '-lcudart', '-o', str(exe)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f'native harness build failed:\n{" ".join(cmd)}\n{r.stderr[-4000:]}'
    return str(exe)


def _harness(exe, mode):
    r = subprocess.run([exe, mode], capture_output=True, text=True, timeout=SUBPROC_TIMEOUT, env=_env())
    lines = [json.loads(ln) for ln in r.stdout.splitlines() if ln.startswith('{')]
    summary = lines[-1].get('summary') if lines else None
    bad = [ln for ln in lines if ln.get('ok') is False]
    return r, summary, bad


@pytest.mark.parametrize('mode', ['accounting', 'dtor'])
def test_fix3_v302_native_byte_accounting_overflow_safe(fix3_harness, mode):
    r, summary, bad = _harness(fix3_harness, mode)
    assert 'runtime error' not in r.stderr, r.stderr                            # UBSan (and -fno-sanitize-recover)
    assert r.returncode == 0 and summary and summary['failures'] == 0 and not bad, (r.returncode, bad, r.stdout[-3000:],
                                                                                    r.stderr[-2000:])
    assert summary['cases'] == (17 if mode == 'accounting' else 11), summary
