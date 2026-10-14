"""P1T-FIX1 regression tests for the P1T-REV round-1 findings in tc_signal (order P1T_FIX1_REVIEW_FINDINGS.md).

What each group pins, and the registered mutant it must kill (G3):
  * F01 (BLOCKING), consumed operand ownership through GPU completion; mutant (a) = immediate release restored:
      - a torch producer on a side stream whose ONLY owner is the consumed manager (a bare capsule, or a producer's
        temporary made inside __dlpack__), read behind a legacy-stream sleep, then the side stream allocates the same
        size and overwrites it: the output must be the producer's data (the reviewer's addon_lifetime.py scenario);
      - the same with an exception raised AFTER the launch (test hook _C._test_rint_then_raise: partial-launch path);
      - the LSTM's whole per-call capsule set (10 managers) is ONE pending entry, released by synchronize();
      - a completed call's deleters run at the NEXT tc_signal call (not before), and synchronize()/atexit release the
        rest; a refusal before the launch still releases at once; the bytes bound forces a synchronize-and-drain.
  * F02, None for a required operand of each of the nine ops raises TypeError before planning (the allocator is never
    called), also at the *_run level; each case runs in its own subprocess so a SIGSEGV is a failed case, not a dead
    run; mutant (b) = the None guard (require_operand) removed.
  * F03, checked arithmetic in validation and in every plan function (reflect, upsample, conv, conv_transpose, LSTM,
    norms): synthetic descriptors that overflow must raise ValueError instead of returning a wrapped size.
    Descriptor-only (CPU metadata, never launched), like the reviewer's host_checks.py.

Skip guard (as test_contract.py): only when torch is not importable or there is no CUDA device.

Prior art: the reviewer's probes (artifacts/p1t_rev/probes/addon_lifetime.py, host_checks.py), whose scenarios these
tests reuse; PyTorch's caching-allocator stream semantics (a block freed without record_stream is reused at once on its
allocation stream) as the reuse bait; DeMillo, Lipton & Sayward (1978) mutation testing for the registered mutants.
Ours: the producer-temporary and partial-launch variants, the capsule-set and bound checks.
"""
import ctypes
import json
import os
import subprocess
import sys

import pytest

try:
    import torch
except ImportError as e:
    pytest.skip(f'torch is not importable: {e}', allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip('no CUDA device', allow_module_level=True)

import tc_signal as ts  # noqa: E402
from tc_signal import _tc_signal as _C  # noqa: E402

import tcs_producer as P  # noqa: E402

SLEEP_CYCLES = 20_000_000        # torch.cuda._sleep cycles (~10 ms on the RTX 4070 SUPER), as the reviewer's probe
LONG_SLEEP_CYCLES = 200_000_000  # ~100 ms: keeps the legacy stream busy across several host calls (bound test)
RACE_NUMEL = 1 << 18             # the probe's size (1 MiB fp32)
RACE_TRIALS = 8
SUBPROC_TIMEOUT = 60


def _torch_alloc(shape):
    return torch.empty(shape, device='cuda', dtype=torch.float32)


@pytest.fixture(autouse=True)
def torch_allocator():
    prev = ts.set_allocator(_torch_alloc)
    yield
    ts.set_allocator(prev)
    ts.synchronize()


def _warm(S):
    x = torch.full((RACE_NUMEL,), 1.25, device='cuda')
    out = torch.empty(RACE_NUMEL, device='cuda')
    ts.rint(x, out=out)
    with torch.cuda.stream(S):
        torch.empty(RACE_NUMEL, device='cuda').fill_(0.0)
    torch.cuda._sleep(1)
    torch.cuda.synchronize()
    ts.synchronize()


# ------------------------------------------------------------------------------------------------ F01
class _TempProducer:
    """__dlpack__ makes a FRESH temporary on side stream S, so the capsule's manager is its only owner (the reviewer's
    'producer temporaries' case: a reader may receive a copy the caller cannot retain)."""

    def __init__(self, S, value):
        self.S, self.value, self.address = S, value, None

    def __dlpack_device__(self):
        return (2, torch.cuda.current_device())

    def __dlpack__(self, *, stream=None, max_version=None, copy=None):
        with torch.cuda.stream(self.S):
            t = torch.full((RACE_NUMEL,), self.value, device='cuda')
            self.address = t.data_ptr()
            return t.__dlpack__(stream=stream, max_version=max_version)   # torch orders `stream` after S


def _side_stream_capsule(S, value):
    """A bare capsule whose manager is the last owner of a side-stream allocation (addon_lifetime.py's setup)."""
    with torch.cuda.stream(S):
        p = torch.full((RACE_NUMEL,), value, device='cuda')
    S.synchronize()
    cap = p.__dlpack__(stream=1, max_version=(1, 3))
    address = p.data_ptr()
    del p
    return cap, address


def _bait_and_check(S, out, address, k):
    """Allocate the same size on S and overwrite it at once (S is idle); a released block would be reused."""
    with torch.cuda.stream(S):
        replacement = torch.empty(RACE_NUMEL, device='cuda')
        reused = replacement.data_ptr() == address
        replacement.fill_(9.0)
    torch.cuda.synchronize()
    ok = bool(torch.all(out == 1.0).item())
    assert ok, f'trial {k}: the add-on read storage that was released and reused (reused={reused}) before its read'
    del replacement


@pytest.mark.parametrize('source', ['bare_capsule', 'producer_temporary'])
def test_f01_side_stream_last_owner_survives_until_legacy_read(source):
    S = torch.cuda.Stream()
    _warm(S)
    out = torch.empty(RACE_NUMEL, device='cuda')
    for k in range(RACE_TRIALS):
        out.fill_(-1.0)
        torch.cuda.synchronize()
        before = _C.pending_stats()['retained_entries']
        if source == 'bare_capsule':
            cap, address = _side_stream_capsule(S, 1.25)
            torch.cuda._sleep(SLEEP_CYCLES)        # legacy stream busy: the add-on's read is queued behind it
            ts.rint(cap, out=out)
        else:
            prod = _TempProducer(S, 1.25)
            torch.cuda._sleep(SLEEP_CYCLES)
            ts.rint(prod, out=out)
            address = prod.address
        deferred = _C.pending_stats()['retained_entries'] == before + 1
        _bait_and_check(S, out, address, k)          # the race first: a mutant fails on the use-after-free itself
        assert deferred, "the call's consumed managers were not handed to the deferred-release list"


def test_f01_partial_launch_exception_keeps_operands_until_complete():
    S = torch.cuda.Stream()
    _warm(S)
    out = torch.empty(RACE_NUMEL, device='cuda')
    for k in range(RACE_TRIALS):
        out.fill_(-1.0)
        torch.cuda.synchronize()
        cap, address = _side_stream_capsule(S, 1.25)
        torch.cuda._sleep(SLEEP_CYCLES)
        with pytest.raises(RuntimeError, match='injected failure after the launch'):
            _C._test_rint_then_raise(cap, out.__dlpack__(stream=1))
        assert 'used_' in repr(cap)                 # consumed: only tc_signal's deferred list keeps it alive now
        pending = _C.pending_stats()['entries']
        _bait_and_check(S, out, address, k)
        assert pending >= 1, 'the partial-launch path released its managers before the GPU finished'


def test_f01_lstm_capsule_set_is_one_entry_released_by_synchronize():
    T, N, I, H, D = 3, 2, 4, 5, 2
    g = torch.Generator(device='cuda').manual_seed(20261012)

    def r(*s):
        return torch.randn(*s, device='cuda', generator=g) * 0.1
    x, W, R, B, h0, c0 = r(T, N, I), r(D, 4 * H, I), r(D, 4 * H, H), r(D, 8 * H), r(D, N, H), r(D, N, H)
    ts.synchronize()
    s0 = _C.pending_stats()
    assert s0['entries'] == 0 and s0['managers'] == 0
    torch.cuda._sleep(SLEEP_CYCLES)
    ts.lstm(x, W, R, B, hidden_size=H, direction='bidirectional', initial_h=h0, initial_c=c0)
    s1 = _C.pending_stats()
    # x, W, R, B, initial_h, initial_c, Y, Y_h, Y_c, scratch: one entry, ten managers.
    assert s1['entries'] == 1 and s1['managers'] == 10, s1
    ts.synchronize()
    s2 = _C.pending_stats()
    assert s2['entries'] == 0 and s2['managers'] == 0 and s2['bytes'] == 0, s2
    assert s2['released_managers'] - s1['released_managers'] == 10, (s1, s2)


def test_f01_completed_call_released_at_next_call_not_before():
    ts.synchronize()
    P.reset_stats()
    base = torch.full((82,), 1.5, device='cuda')
    y = ts.rint(P.Producer(base, (1, 82)))
    torch.cuda.synchronize()                         # the GPU is done, but no tc_signal call has run since
    assert P.STATS == {'exported': 1, 'deleted': 0, 'unknown_delete': 0}, P.STATS
    ts.rint(torch.zeros(1, 82, device='cuda'))       # the next call drains the completed entry first
    assert P.STATS['deleted'] == 1 and P.STATS['unknown_delete'] == 0, P.STATS
    assert bool(torch.all(y == 2.0).item())          # rint(1.5) = 2 (ties to even)


def test_f01_refusal_before_launch_releases_immediately():
    ts.synchronize()
    P.reset_stats()
    base = torch.full((82,), 1.5, device='cuda')
    with pytest.raises(ValueError, match='expected'):
        ts.rint(P.Producer(base, (1, 82)), out=torch.empty(1, 81, device='cuda'))   # run-phase shape refusal
    assert P.STATS['exported'] == 1 and P.STATS['deleted'] == 1, P.STATS            # nothing was launched
    assert _C.pending_stats()['entries'] == 0


def test_f01_bytes_bound_forces_synchronize_and_drain():
    ts.synchronize()
    s0 = _C.pending_stats()
    max_bytes = s0['max_bytes']
    n = max_bytes // 4 // 8                          # x and out are max_bytes/8 each: an entry holds max_bytes/4
    x = torch.full((n,), 2.5, device='cuda')
    outs = [torch.empty(n, device='cuda') for _ in range(6)]   # pre-allocated: the six calls are only plan + run
    torch.cuda.synchronize()
    torch.cuda._sleep(LONG_SLEEP_CYCLES)             # nothing completes while the calls are queued
    for o in outs:
        ts.rint(x, out=o)
        st = _C.pending_stats()
        assert st['bytes'] <= max_bytes and st['entries'] <= st['max_entries'], st   # after any forced drain
    s1 = _C.pending_stats()
    # The 5th push (5 x 64 MiB = 320 MiB) exceeded the 256 MiB bound: synchronize + drain. (The high-water mark records
    # that 320 MiB instant, before the drain, by design.)
    assert s1['bound_drains'] > s0['bound_drains'], (s0, s1)
    torch.cuda.synchronize()
    for o in outs:
        assert bool(torch.all(o == 2.0).item())      # rint(2.5) = 2


_ATEXIT_SCRIPT = r"""
import atexit, json, sys
sys.path.insert(0, sys.argv[1])
import tcs_producer as P
atexit.register(lambda: print('ATEXIT ' + json.dumps(P.STATS), flush=True))   # first registered: runs last (LIFO)
import torch, tc_signal as ts
ts.set_allocator(lambda s: torch.empty(s, device='cuda'))
base = torch.full((82,), 1.5, device='cuda')
ts.rint(P.Producer(base, (1, 82)))
torch.cuda.synchronize()
print('PENDING ' + json.dumps(dict(ts._tc_signal.pending_stats())), flush=True)
"""


def test_f01_atexit_hook_releases_pending_operands():
    r = subprocess.run([sys.executable, '-c', _ATEXIT_SCRIPT, os.path.dirname(os.path.abspath(__file__))],
                       capture_output=True, text=True, timeout=SUBPROC_TIMEOUT,
                       env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert r.returncode == 0, r.stderr
    pend = json.loads(next(ln for ln in r.stdout.splitlines() if ln.startswith('PENDING '))[8:])
    final = json.loads(next(ln for ln in r.stdout.splitlines() if ln.startswith('ATEXIT '))[7:])
    assert pend['entries'] == 1 and pend['managers'] == 2, pend      # x (the producer) and out, still held
    assert final == {'exported': 1, 'deleted': 1, 'unknown_delete': 0}, final


# ------------------------------------------------------------------------------------------------ F02
# Fake capsules (CPU metadata only: device (2, 0), fp32, a host buffer as data). A None refusal comes before any
# capsule is consumed, so nothing is ever launched on them.
_F02_SCRIPT = r"""
import ctypes, sys
import tc_signal as ts
from tc_signal import _tc_signal as C

class Dev(ctypes.Structure): _fields_ = [('type', ctypes.c_int), ('id', ctypes.c_int)]
class DT(ctypes.Structure): _fields_ = [('code', ctypes.c_uint8), ('bits', ctypes.c_uint8), ('lanes', ctypes.c_uint16)]
class Tensor(ctypes.Structure):
    _fields_ = [('data', ctypes.c_void_p), ('dev', Dev), ('ndim', ctypes.c_int), ('dtype', DT),
                ('shape', ctypes.POINTER(ctypes.c_int64)), ('strides', ctypes.POINTER(ctypes.c_int64)),
                ('offset', ctypes.c_uint64)]
class Managed(ctypes.Structure): _fields_ = [('tensor', Tensor), ('ctx', ctypes.c_void_p), ('deleter', ctypes.c_void_p)]
new = ctypes.pythonapi['PyCapsule_New']
new.restype, new.argtypes = ctypes.py_object, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
keep = []
def cap(*shape):
    dims = (ctypes.c_int64 * max(len(shape), 1))(*shape)
    buf = (ctypes.c_float * 64)()
    m = Managed(Tensor(ctypes.addressof(buf), Dev(2, 0), len(shape), DT(2, 32, 1), dims, None, 0), None, None)
    keep.extend((dims, buf, m))
    return new(ctypes.addressof(m), b'dltensor', None)

class AllocatorCalled(Exception):
    pass
calls = []
def alloc(shape):
    calls.append(tuple(shape))
    raise AllocatorCalled(shape)
ts.set_allocator(alloc)
H = 2
CASES = {
    'conv1d.x': lambda: ts.conv1d(None, cap(2, 1, 3)),
    'conv1d.w': lambda: ts.conv1d(cap(1, 1, 8), None),
    'conv_transpose1d.x': lambda: ts.conv_transpose1d(None, cap(1, 1, 3)),
    'conv_transpose1d.w': lambda: ts.conv_transpose1d(cap(1, 1, 8), None),
    'lstm.x': lambda: ts.lstm(None, cap(1, 4 * H, 3), cap(1, 4 * H, H), hidden_size=H),
    'lstm.W': lambda: ts.lstm(cap(5, 1, 3), None, cap(1, 4 * H, H), hidden_size=H),
    'lstm.R': lambda: ts.lstm(cap(5, 1, 3), cap(1, 4 * H, 3), None, hidden_size=H),
    'instance_norm1d.x': lambda: ts.instance_norm1d(None),
    'layer_norm.x': lambda: ts.layer_norm(None),
    'group_norm.x': lambda: ts.group_norm(None, 1),
    'upsample_nearest1d.x': lambda: ts.upsample_nearest1d(None, 2),
    'reflection_pad1d.x': lambda: ts.reflection_pad1d(None, (0, 0)),
    'rint.x': lambda: ts.rint(None),
    'rint.x_out_given': lambda: ts.rint(None, out=cap()),                 # the reviewer's none_rint case
    'run.conv1d.x': lambda: C.conv1d_run(None, cap(2, 1, 3), None, cap(1, 2, 6), None, 1, 0, 0, 1, 1),
    'run.conv_transpose1d.x': lambda: C.conv_transpose1d_run(None, cap(1, 1, 3), None, cap(1, 1, 10), 1, 0, 0, 0, 1, 1),
    'run.lstm.x': lambda: C.lstm_run(None, cap(1, 4 * H, 3), cap(1, 4 * H, H), None, None, None, cap(5, 1, 1, H),
                                      cap(1, 1, H), cap(1, 1, H), cap(64), H, 0),
    'run.instance_norm1d.x': lambda: C.instance_norm1d_run(None, None, None, cap(1, 2, 4), 1e-5),
    'run.layer_norm.x': lambda: C.layer_norm_run(None, None, None, cap(1, 2, 4), -1, 1e-5),
    'run.group_norm.x': lambda: C.group_norm_run(None, 1, None, None, cap(1, 2, 4), 1e-5),
    'run.upsample_nearest1d.x': lambda: C.upsample_nearest1d_run(None, cap(1, 8), 2),
    'run.reflection_pad1d.x': lambda: C.reflection_pad1d_run(None, cap(1, 4), 0, 0),
    'run.rint.x': lambda: C.rint_run(None, cap(4)),
    'run.rint.out': lambda: C.rint_run(cap(4), None),
}
try:
    CASES[sys.argv[1]]()
    print('RESULT none', calls)
except Exception as e:
    print('RESULT', type(e).__name__, calls, '|', e)
"""

F02_CASES = ['conv1d.x', 'conv1d.w', 'conv_transpose1d.x', 'conv_transpose1d.w', 'lstm.x', 'lstm.W', 'lstm.R',
             'instance_norm1d.x', 'layer_norm.x', 'group_norm.x', 'upsample_nearest1d.x', 'reflection_pad1d.x',
             'rint.x', 'rint.x_out_given', 'run.conv1d.x', 'run.conv_transpose1d.x', 'run.lstm.x',
             'run.instance_norm1d.x', 'run.layer_norm.x', 'run.group_norm.x', 'run.upsample_nearest1d.x',
             'run.reflection_pad1d.x', 'run.rint.x', 'run.rint.out']


@pytest.mark.parametrize('case', F02_CASES)
def test_f02_none_required_operand_is_type_error_before_planning(case):
    r = subprocess.run([sys.executable, '-c', _F02_SCRIPT, case], capture_output=True, text=True,
                       timeout=SUBPROC_TIMEOUT, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert r.returncode == 0, f'{case}: process died with {r.returncode} (a crash, not a clean refusal)\n{r.stderr}'
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith('RESULT'))
    assert line.startswith('RESULT TypeError [] |'), f'{case}: {line}'      # TypeError, and no allocation happened
    assert 'is required' in line, f'{case}: {line}'


# ------------------------------------------------------------------------------------------------ F03
class _Dev(ctypes.Structure):
    _fields_ = [('type', ctypes.c_int), ('id', ctypes.c_int)]


class _DT(ctypes.Structure):
    _fields_ = [('code', ctypes.c_uint8), ('bits', ctypes.c_uint8), ('lanes', ctypes.c_uint16)]


class _Tensor(ctypes.Structure):
    _fields_ = [('data', ctypes.c_void_p), ('dev', _Dev), ('ndim', ctypes.c_int), ('dtype', _DT),
                ('shape', ctypes.POINTER(ctypes.c_int64)), ('strides', ctypes.POINTER(ctypes.c_int64)),
                ('offset', ctypes.c_uint64)]


class _Managed(ctypes.Structure):
    _fields_ = [('tensor', _Tensor), ('ctx', ctypes.c_void_p), ('deleter', ctypes.c_void_p)]


_new = ctypes.pythonapi['PyCapsule_New']
_new.restype, _new.argtypes = ctypes.py_object, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_KEEP = []


def _desc(*shape, data=None, ndim=None, null_shape=False):
    """An unconsumable-by-design synthetic descriptor (no destructor; CPU metadata only, never launched)."""
    dims = (ctypes.c_int64 * max(len(shape), 1))(*shape)
    buf = (ctypes.c_float * 4)()
    t = _Tensor(ctypes.addressof(buf) if data is None else data, _Dev(2, torch.cuda.current_device()),
                len(shape) if ndim is None else ndim, _DT(2, 32, 1), None if null_shape else dims, None, 0)
    m = _Managed(t, None, None)
    _KEEP.extend((dims, buf, m))
    return _new(ctypes.addressof(m), b'dltensor', None)


F03_CASES = [
    # (id, plan call, message fragment)
    ('validate_count', lambda: _C.rint_plan(_desc(1 << 32, 1 << 32)), 'element count'),
    ('validate_bytes', lambda: _C.rint_plan(_desc(1 << 62)), 'byte size'),
    ('validate_address', lambda: _C.rint_plan(_desc(4, data=(1 << 64) - 8)), 'address space'),
    ('validate_negative_rank', lambda: _C.rint_plan(_desc(4, ndim=-1)), 'negative number of dimensions'),
    ('validate_null_shape', lambda: _C.rint_plan(_desc(4, null_shape=True)), 'NULL shape'),
    ('reflect_review_case', lambda: _C.reflection_pad1d_plan(_desc(1, 1, 1 << 62), (1 << 62) - 1, (1 << 62) - 1),
     'byte size'),
    ('reflect_output_bytes', lambda: _C.reflection_pad1d_plan(_desc(1, 1, 1 << 60), (1 << 60) - 1, (1 << 60) - 1),
     'reflection_pad1d: output size'),
    ('upsample_output_bytes', lambda: _C.upsample_nearest1d_plan(_desc(1, 1, 1 << 40), 1 << 21),
     'upsample_nearest1d: output size'),
    ('upsample_length', lambda: _C.upsample_nearest1d_plan(_desc(1, 1, 1 << 40), 1 << 30),
     'upsample_nearest1d: output length'),
    ('conv_dilation_span', lambda: _C.conv1d_plan(_desc(1, 1, 1 << 40), _desc(1, 1, 1 << 40), None, 1, 0, 0,
                                                  1 << 30, 1), 'conv1d: dilation*(K-1)'),
    ('conv_scratch_bytes', lambda: _C.conv1d_plan(_desc(1, 65536, 2147450880), _desc(1, 65536, 32767), None, 1, 0, 0,
                                                  1, 1), 'conv1d: im2col scratch bytes'),
    ('conv_scratch_count', lambda: _C.conv1d_plan(_desc(1, 1 << 29, 1 << 31), _desc(1 << 29, 1, 1 << 10), None, 1, 0,
                                                  0, 1, 1 << 29), 'conv1d: im2col scratch size'),
    ('conv_transpose_length', lambda: _C.conv_transpose1d_plan(_desc(1, 1, 1 << 40), _desc(1, 1, 2), None,
                                                               (1 << 31) - 1, 0, 0, 0, 1, 1),
     'conv_transpose1d: stride*(T-1)'),
    ('lstm_scratch_bytes', lambda: _C.lstm_plan(_desc((1 << 31) - 1, 1, 1), _desc(2, 1073741820, 1),
                                                _desc(2, 1073741820, 268435455), None, None, None, 268435455, 2),
     'lstm: scratch bytes'),
    ('norm_count', lambda: _C.instance_norm1d_plan(_desc(1 << 32, 1 << 32, 1), None, None, 1e-5), 'element count'),
]


@pytest.mark.parametrize('case', F03_CASES, ids=[c[0] for c in F03_CASES])
def test_f03_overflowing_geometry_is_refused_not_wrapped(case):
    _, call, frag = case
    with pytest.raises(ValueError) as ei:
        call()
    assert frag in str(ei.value), f'message {str(ei.value)!r} lacks {frag!r}'
