"""G1 contract: refusals raise the documented error; DLPack consumer rules; allocator; stream ordering; cuBLAS
configuration; fast-math canary.

Documented errors (README "Errors"): TypeError for a non-DLPack operand, a non-float32 dtype, a non-int attribute or a
bad padding type, a non-callable allocator, a foreign capsule; ValueError for a non-CUDA device, mismatched device
ordinals, non-contiguous / misaligned / empty / wrong-rank / wrong-shape operands, invalid attributes or output
length, an output overlapping another operand, a read-only or copied output, a consumed capsule, an unsupported
DLPack major version; RuntimeError for the default allocator without tensor_cuda.empty (naming set_allocator).

Skip guard (order): skip only when torch is not importable or there is no CUDA device; any other exception fails.
"""
import ctypes
import gc
import re

import pytest

try:
    import torch
except ImportError as e:
    pytest.skip(f'torch is not importable: {e}', allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip('no CUDA device', allow_module_level=True)

import numpy as np  # noqa: E402

import tc_signal as ts  # noqa: E402
from tc_signal import _tc_signal as _C  # noqa: E402

import tcs_census as C  # noqa: E402
import tcs_producer as P  # noqa: E402
from tcs_oracle import bits_equal, conv1d_f64, conv_transpose1d_f64, normwise  # noqa: E402


def _torch_alloc(shape):
    return torch.empty(shape, device='cuda', dtype=torch.float32)


@pytest.fixture(autouse=True)
def torch_allocator():
    prev = ts.set_allocator(_torch_alloc)
    yield
    ts.set_allocator(prev)


def _t(*shape):
    return torch.randn(*shape, device='cuda')


# ------------------------------------------------------------------------------------------------ refusals
def _c1(**kw):
    """conv1d with x [1, 8, 20], w [6, 8, 3], b [6] unless overridden."""
    x = kw.pop('x', None)
    w = kw.pop('w', None)
    b = kw.pop('b', 'default')
    x = _t(1, 8, 20) if x is None else x
    w = _t(6, 8, 3) if w is None else w
    b = _t(6) if isinstance(b, str) else b
    return ts.conv1d(x, w, b, **kw)


def _ct(**kw):
    """conv_transpose1d with x [1, 8, 20], w [8, 3, 4] (C_out 3), b [3], stride 2 unless overridden."""
    x = kw.pop('x', None)
    w = kw.pop('w', None)
    b = kw.pop('b', 'default')
    kw.setdefault('stride', 2)
    x = _t(1, 8, 20) if x is None else x
    w = _t(8, 3, 4) if w is None else w
    b = _t(3) if isinstance(b, str) else b
    return ts.conv_transpose1d(x, w, b, **kw)


def _overlap_out_is_x():
    x = _t(1, 8, 20)
    return ts.conv1d(x, _t(8, 8, 3), None, padding=1, out=x)


def _overlap_partial():
    st = torch.zeros(400, device='cuda')
    x = st[:160].view(1, 8, 20)
    out = st[100:220].view(1, 6, 20)
    return ts.conv1d(x, _t(6, 8, 3), None, padding=1, out=out)


def _overlap_out_is_w():
    w = _t(1, 8, 3)                       # out [1, 1, 8]: placed over w's memory
    x = _t(1, 8, 10)
    out = w.view(-1)[:8].view(1, 1, 8)
    return ts.conv1d(x, w, None, out=out)


def _overlap_convt_out_is_b():
    st = torch.zeros(1000, device='cuda')
    b = st[:3]
    out = st[:3 * 42].view(1, 3, 42)      # convT x [1,8,20] s2 K4 -> T_out 42
    return ts.conv_transpose1d(_t(1, 8, 20), _t(8, 3, 4), b, stride=2, out=out)


def _noncontig_x():
    return _c1(x=torch.randn(1, 20, 8, device='cuda').transpose(1, 2))


REFUSALS = [
    # (id, call, exception, message fragment)
    ('x_list', lambda: _c1(x=[[[1.0]]]), TypeError, 'DLPack producer'),
    ('x_numpy_cpu', lambda: _c1(x=np.zeros((1, 8, 20), np.float32)), ValueError, 'kDLCUDA'),
    ('x_cpu_tensor', lambda: _c1(x=torch.randn(1, 8, 20)), ValueError, 'kDLCUDA'),
    ('x_fp16', lambda: _c1(x=_t(1, 8, 20).half()), TypeError, 'float32'),
    ('w_fp64', lambda: _c1(w=_t(6, 8, 3).double()), TypeError, 'float32'),
    ('b_fp16', lambda: _c1(b=_t(6).half()), TypeError, 'float32'),
    ('x_noncontiguous', _noncontig_x, ValueError, 'C-contiguous'),
    ('x_2d', lambda: _c1(x=_t(8, 20)), ValueError, 'dimensions'),
    ('x_empty_T', lambda: _c1(x=torch.empty(1, 8, 0, device='cuda')), ValueError, 'empty operand'),
    ('w_bad_cin', lambda: _c1(w=_t(6, 7, 3)), ValueError, 'w.shape[1]'),
    ('groups_not_dividing_cin', lambda: _c1(w=_t(6, 2, 3), groups=3), ValueError, 'divisible'),
    ('groups_not_dividing_cout', lambda: _c1(w=_t(6, 2, 3), groups=4), ValueError, 'C_out'),
    ('b_wrong_len', lambda: _c1(b=_t(5)), ValueError, 'b has shape'),
    ('b_2d', lambda: _c1(b=_t(1, 6)), ValueError, 'dimensions'),
    ('out_wrong_shape', lambda: _c1(out=_t(1, 6, 17)), ValueError, 'expected'),
    ('out_fp16', lambda: _c1(out=_t(1, 6, 18).half()), TypeError, 'float32'),
    ('out_cpu', lambda: _c1(out=torch.empty(1, 6, 18)), ValueError, 'kDLCUDA'),
    ('out_is_x', _overlap_out_is_x, ValueError, 'overlaps'),
    ('out_partial_overlap', _overlap_partial, ValueError, 'overlaps'),
    ('out_over_w', _overlap_out_is_w, ValueError, 'overlaps'),
    ('out_len_le0', lambda: _c1(x=_t(1, 8, 2), w=_t(6, 8, 5)), ValueError, 'output length'),
    ('out_len_le0_dilation', lambda: _c1(x=_t(1, 8, 4), dilation=3), ValueError, 'output length'),
    ('stride0', lambda: _c1(stride=0), ValueError, 'stride'),
    ('dilation0', lambda: _c1(dilation=0), ValueError, 'dilation'),
    ('groups0', lambda: _c1(groups=0), ValueError, 'groups'),
    ('neg_pad', lambda: _c1(padding=(-1, 0)), ValueError, 'pad_begin'),
    ('neg_pad_end', lambda: _c1(padding=(0, -2)), ValueError, 'pad_end'),
    ('padding_str', lambda: _c1(padding='same'), TypeError, 'padding'),
    ('padding_len3', lambda: _c1(padding=(1, 1, 1)), ValueError, 'padding'),
    ('stride_bool', lambda: _c1(stride=True), TypeError, 'bool'),
    ('stride_float', lambda: _c1(stride=1.0), TypeError, 'stride'),
    ('convt_op_ge_max', lambda: _ct(output_padding=2), ValueError, 'output_padding'),
    ('convt_op_neg', lambda: _ct(output_padding=-1), ValueError, 'output_padding'),
    ('convt_w_bad_cin', lambda: _ct(w=_t(7, 3, 4)), ValueError, 'w.shape[0]'),
    ('convt_groups_not_dividing', lambda: _ct(w=_t(8, 3, 4), groups=3), ValueError, 'divisible'),
    ('convt_b_wrong_len', lambda: _ct(b=_t(4)), ValueError, 'b has shape'),
    ('convt_out_len_le0', lambda: _ct(x=_t(1, 8, 1), w=_t(8, 3, 1), stride=1, padding=(1, 1)), ValueError,
     'output length'),
    ('convt_out_wrong_shape', lambda: _ct(out=_t(1, 3, 41)), ValueError, 'expected'),
    ('convt_out_over_b', _overlap_convt_out_is_b, ValueError, 'overlaps'),
    ('convt_x_fp16', lambda: _ct(x=_t(1, 8, 20).half()), TypeError, 'float32'),
    ('convt_x_cpu', lambda: _ct(x=torch.randn(1, 8, 20)), ValueError, 'kDLCUDA'),
]


@pytest.mark.parametrize('case', REFUSALS, ids=[r[0] for r in REFUSALS])
def test_refusal(case):
    _, call, exc, frag = case
    with pytest.raises(exc, match=re.escape(frag)):
        call()
    torch.cuda.synchronize()            # no launch error left behind


def test_set_allocator_non_callable():
    with pytest.raises(TypeError, match='callable'):
        ts.set_allocator(42)


def test_default_allocator_names_set_allocator(monkeypatch):
    # The default allocator is tensor_cuda.empty (P1T-M1). Simulate the two ways it can be unavailable, independent of
    # which tensor_cuda is on PYTHONPATH (lead fix on voice-int: the original assumed a pre-M1 tensor_cuda and failed
    # once M1's core was present).
    import sys
    import types
    ts.set_allocator(None)
    monkeypatch.setitem(sys.modules, 'tensor_cuda', types.ModuleType('tensor_cuda'))      # importable, no `empty`
    with pytest.raises(RuntimeError, match='set_allocator'):
        _c1()
    with pytest.raises(RuntimeError, match='set_allocator'):
        _ct()
    monkeypatch.setitem(sys.modules, 'tensor_cuda', None)                                  # not importable
    with pytest.raises(RuntimeError, match='set_allocator'):
        _c1()


def test_default_allocator_uses_tensor_cuda_empty():
    # With M1's core on PYTHONPATH the default allocator returns TensorCUDA tensors and the conv matches the oracle.
    try:
        import tensor_cuda as tcore
    except ImportError:
        pytest.skip('tensor_cuda is not importable')
    if not hasattr(tcore, 'empty') or not hasattr(tcore, 'from_dlpack'):
        pytest.skip('this tensor_cuda predates P1T-M1 (no empty/from_dlpack)')
    ts.set_allocator(None)
    x = _t(1, 8, 20)
    w = _t(1, 8, 3)
    y = ts.conv1d(x, w, None, padding=1)
    assert hasattr(y, 'numpy') and not isinstance(y, torch.Tensor)          # a TensorCUDA tensor, not torch
    got = torch.from_numpy(np.ascontiguousarray(y.numpy())).double()
    ref = conv1d_f64(x, w, None, stride=1, pads=(1, 1), dilation=1, groups=1)
    assert normwise(got, ref) <= 1e-5


def test_allocator_wrong_shape_or_dtype():
    ts.set_allocator(lambda s: torch.empty(tuple(v + 1 for v in s), device='cuda'))
    with pytest.raises(ValueError, match='expected'):
        _c1()
    ts.set_allocator(lambda s: torch.empty(s, device='cuda', dtype=torch.float16))
    with pytest.raises(TypeError, match='float32'):
        _ct()


def test_dlpack_returns_non_capsule():
    with pytest.raises(TypeError, match='not a PyCapsule'):
        _c1(x=P.Producer(_t(160), (1, 8, 20), return_value=12345))


def test_foreign_capsule_name():
    new = ctypes.pythonapi['PyCapsule_New']
    new.restype = ctypes.py_object
    new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    keep = ctypes.c_int(0)
    cap = new(ctypes.addressof(keep), b'not_dlpack', None)
    with pytest.raises(TypeError, match='not a DLPack capsule'):
        _c1(x=cap)


def test_consumed_capsule_refused():
    x, w = _t(1, 8, 20), _t(6, 8, 3)
    cap = x.__dlpack__(stream=1)
    y = ts.conv1d(cap, w, None)          # a bare capsule is a valid operand
    torch.cuda.synchronize()
    assert normwise(y, conv1d_f64(x, w, None, stride=1, pads=(0, 0), dilation=1, groups=1)) <= C.GATE_N
    with pytest.raises(ValueError, match='already consumed'):
        ts.conv1d(cap, w, None)


@pytest.mark.parametrize('flags,frag', [(P.FLAG_READ_ONLY, 'read-only'), (P.FLAG_IS_COPIED, 'copy')],
                         ids=['read_only', 'is_copied'])
def test_written_operand_flags_refused(flags, frag):
    out = P.Producer(_t(1, 6, 18).view(-1), (1, 6, 18), flags=flags)
    with pytest.raises(ValueError, match=frag):
        _c1(out=out)


def test_device_ordinal_mismatch_refused():
    w = P.Producer(_t(6 * 8 * 3), (6, 8, 3), device_id=1)
    with pytest.raises(ValueError, match='same device'):
        _c1(w=w)


def test_non_cuda_device_type_refused():
    x = P.Producer(_t(160), (1, 8, 20), device_type=P.KDL_CPU)
    with pytest.raises(ValueError, match='kDLCUDA'):
        _c1(x=x)


def test_dlpack_major_version_refused():
    x = P.Producer(_t(160), (1, 8, 20), major=2)
    with pytest.raises(ValueError, match='major version'):
        _c1(x=x)


def test_misaligned_refused():
    x = P.Producer(_t(200), (1, 8, 20), extra_byte_offset=2)
    with pytest.raises(ValueError, match='aligned'):
        _c1(x=x)


def test_explicit_noncompact_strides_refused():
    x = P.Producer(_t(400), (1, 8, 20), strides=(160, 40, 1))
    with pytest.raises(ValueError, match='C-contiguous'):
        _c1(x=x)


# ------------------------------------------------------------------------------------------------ DLPack consumer rules
def _ref_conv1d(x, w, b, **kw):
    return conv1d_f64(x, w, b, stride=kw.get('stride', 1), pads=kw.get('padding', (0, 0)),
                      dilation=kw.get('dilation', 1), groups=kw.get('groups', 1))


def test_producer_kwargs_stream_1_and_copy_false():
    xb, wb, bb, ob = _t(160), _t(144), _t(6), _t(108)
    xp, wp, bp = P.Producer(xb, (1, 8, 20)), P.Producer(wb, (6, 8, 3)), P.Producer(bb, (6,))
    op = P.Producer(ob, (1, 6, 18))
    y = ts.conv1d(xp, wp, bp, out=op)
    assert y is op
    torch.cuda.synchronize()
    for p in (xp, wp, bp, op):
        assert len(p.calls) == 1
        assert p.calls[0]['stream'] == 1                     # the legacy default stream, DLPack int 1
        assert tuple(p.calls[0]['max_version']) == (1, 3)
    for p in (xp, wp, bp):
        assert p.calls[0]['copy'] is None                    # inputs: the producer may copy
    assert op.calls[0]['copy'] is False                      # outputs: must alias
    o = _ref_conv1d(xb.view(1, 8, 20), wb.view(6, 8, 3), bb)
    assert normwise(ob.view(1, 6, 18), o) <= C.GATE_N


def test_legacy_only_producer_fallback():
    xb, wb = _t(160), _t(144)
    xp = P.LegacyOnlyProducer(xb, (1, 8, 20))
    y = ts.conv1d(xp, P.LegacyOnlyProducer(wb, (6, 8, 3)), None)
    torch.cuda.synchronize()
    assert xp.calls == [{'stream': 1}]
    assert normwise(y, _ref_conv1d(xb.view(1, 8, 20), wb.view(6, 8, 3), None)) <= C.GATE_N


@pytest.mark.parametrize('mode', ['field', 'data'])
@pytest.mark.parametrize('versioned', [True, False], ids=['versioned', 'legacy'])
def test_byte_offset_honoured(mode, versioned):
    base = _t(1000)
    off = 37
    x = P.Producer(base, (1, 8, 20), offset_elems=off, byte_offset_mode=mode, versioned=versioned)
    wb = _t(6, 8, 3)
    y = ts.conv1d(x, wb, None, padding=(1, 1))
    torch.cuda.synchronize()
    ref = _ref_conv1d(base[off:off + 160].view(1, 8, 20), wb, None, padding=(1, 1))
    assert normwise(y, ref) <= C.GATE_N


def test_size1_dims_any_stride():
    base = _t(160)
    x = P.Producer(base, (1, 8, 20), strides=(987654, 20, 1))   # dim 0 has extent 1: its stride is free
    y = ts.conv1d(x, _t(6, 8, 3), None)
    xt = torch.randn(8, 1, 20, device='cuda').transpose(0, 1)    # torch view [1, 8, 20], strides (20, 20, 1)
    w2 = _t(6, 8, 3)
    y2 = ts.conv1d(xt, w2, None)
    torch.cuda.synchronize()
    assert y.shape == (1, 6, 18)
    assert normwise(y2, _ref_conv1d(xt.contiguous(), w2, None)) <= C.GATE_N


def test_explicit_compact_strides_accepted():
    base, wb = _t(160), _t(6, 8, 3)
    y = ts.conv1d(P.Producer(base, (1, 8, 20), strides=(160, 20, 1)), wb, None)
    torch.cuda.synchronize()
    assert normwise(y, _ref_conv1d(base.view(1, 8, 20), wb, None)) <= C.GATE_N


def test_bare_capsules_versioned_and_legacy():
    x, w, b = _t(1, 8, 20), _t(6, 8, 3), _t(6)
    out = torch.empty(1, 6, 18, device='cuda')
    caps = [x.__dlpack__(stream=1, max_version=(1, 3)), w.__dlpack__(stream=1), b.__dlpack__(stream=1, max_version=(1, 3))]
    ts.conv1d(*caps, out=out.__dlpack__(stream=1))
    torch.cuda.synchronize()
    assert normwise(out, _ref_conv1d(x, w, b)) <= C.GATE_N


def test_deleter_called_exactly_once_on_every_path():
    P.reset_stats()
    xb, wb, bb = _t(160), _t(144), _t(6)

    def prods():
        return P.Producer(xb, (1, 8, 20)), P.Producer(wb, (6, 8, 3)), P.Producer(bb, (6,))

    x, w, b = prods()
    ts.conv1d(x, w, b, out=P.Producer(_t(108), (1, 6, 18)))     # success
    x, w, b = prods()
    with pytest.raises(ValueError):
        ts.conv1d(x, w, b, stride=0)                                 # plan-phase refusal: capsules never consumed
    x, w, b = prods()
    with pytest.raises(ValueError):
        ts.conv1d(x, w, b, out=P.Producer(_t(200), (1, 6, 17)))     # run-phase refusal after consumption
    x, w, b = prods()
    with pytest.raises(ValueError):
        ts.conv1d(x, w, b, out=P.Producer(_t(108), (1, 6, 18), flags=P.FLAG_READ_ONLY))
    x, w, b = prods()
    with pytest.raises(ValueError):
        ts.conv1d(x, P.Producer(wb, (6, 8, 3), major=2), b)         # version refusal at peek
    x, w, b = prods()
    ts.conv_transpose1d(x, P.Producer(_t(8 * 3 * 4), (8, 3, 4)), None, stride=2)
    del x, w, b
    gc.collect()
    torch.cuda.synchronize()
    # P1T-FIX1 F01 contract: a call that reached its launch keeps its consumed managers until the legacy stream has
    # passed it; they are released at the next tc_signal call or by tc_signal.synchronize(). At least the last call's
    # x and w producers are still pending here (no tc_signal call has run since); synchronize() releases them.
    assert P.STATS['exported'] - P.STATS['deleted'] >= 2, P.STATS
    ts.synchronize()
    assert P.STATS['unknown_delete'] == 0, P.STATS
    assert P.STATS['exported'] == P.STATS['deleted'], P.STATS
    assert not P.LIVE


# ------------------------------------------------------------------------------------------------ allocator and API
def test_set_allocator_returns_previous():
    def a(s):
        return torch.empty(s, device='cuda')
    prev = ts.set_allocator(a)
    assert prev is _torch_alloc
    assert ts.set_allocator(prev) is a


def test_allocator_calls_and_out_identity():
    shapes = []

    def counting(s):
        shapes.append(tuple(s))
        return torch.empty(s, device='cuda')
    ts.set_allocator(counting)
    _c1()                                                     # out + im2col scratch
    assert shapes == [(1, 6, 18), (8 * 3 * 18,)]
    shapes.clear()
    ts.conv1d(_t(2, 8, 20), _t(6, 8, 1), None)                # direct path (K 1, stride 1, no pads): no scratch
    assert shapes == [(2, 6, 20)]
    shapes.clear()
    out = torch.empty(1, 6, 18, device='cuda')
    assert _c1(out=out) is out                                # out given: only the scratch is allocated
    assert shapes == [(8 * 3 * 18,)]
    shapes.clear()
    out = torch.empty(1, 3, 42, device='cuda')
    assert _ct(out=out) is out                                # conv_transpose1d: no scratch
    assert shapes == []


def test_padding_int_equals_tuple():
    x, w, b = _t(1, 8, 20), _t(6, 8, 3), _t(6)
    y1 = ts.conv1d(x, w, b, padding=2)
    y2 = ts.conv1d(x, w, b, padding=(2, 2))
    xt, wt = _t(1, 8, 20), _t(8, 3, 4)
    z1 = ts.conv_transpose1d(xt, wt, None, stride=2, padding=1)
    z2 = ts.conv_transpose1d(xt, wt, None, stride=2, padding=[1, 1])
    torch.cuda.synchronize()
    assert bits_equal(y1, y2) and bits_equal(z1, z2)


def test_adjacent_operands_allowed():
    st = torch.randn(160 + 108, device='cuda')
    x, out = st[:160].view(1, 8, 20), st[160:].view(1, 6, 18)
    w = _t(6, 8, 3)
    ref = _ref_conv1d(x.clone(), w, None)
    ts.conv1d(x, w, None, out=out)
    torch.cuda.synchronize()
    assert normwise(out, ref) <= C.GATE_N


# ------------------------------------------------------------------------------------------------ stream ordering
def test_stream_order_torch_write_then_tc_signal_read():
    """torch queues a long sleep then a write of x on the (shared) legacy default stream; tc_signal's conv, launched
    right after without any sync, must see the written values."""
    for trial in range(C.ORDER_PROBE_TRIALS):
        x = torch.zeros(1, 4, 64, device='cuda')
        w = torch.ones(2, 4, 3, device='cuda')
        torch.cuda.synchronize()
        torch.cuda._sleep(C.SLEEP_CYCLES)
        x.fill_(1.0 + trial)
        y = ts.conv1d(x, w, None, padding=(1, 1))
        torch.cuda.synchronize()
        ref = _ref_conv1d(torch.full((1, 4, 64), 1.0 + trial), w, None, padding=(1, 1))
        assert normwise(y, ref) == 0.0, f'trial {trial}: tc_signal read x before torch wrote it'


def test_stream_order_tc_signal_write_then_torch_read():
    """A conv of a few ms writes `out` (pre-filled with NaN); torch clones it right away without a sync: the clone must
    equal the finished result."""
    N, Cin, Cout, T, K = C.ORDER_PROBE_CONV
    x, w = _t(N, Cin, T), _t(Cout, Cin, K) / (Cin * K) ** 0.5
    for trial in range(C.ORDER_PROBE_TRIALS):
        out = torch.full((N, Cout, T - K + 1), float('nan'), device='cuda')
        torch.cuda.synchronize()
        ts.conv1d(x, w, None, out=out)
        early = out.clone()
        torch.cuda.synchronize()
        assert bits_equal(early, out), f'trial {trial}: torch read out before tc_signal finished'


# ------------------------------------------------------------------------------------------------ build / numerics config
def test_cublas_handle_config():
    cfg = _C.cublas_config(torch.cuda.current_device())
    assert cfg['default_math'] and cfg['atomics_not_allowed'] and cfg['legacy_stream'], cfg


def test_build_info():
    bi = _C.build_info()
    assert bi['cudart_version'] // 1000 == 12, bi            # the CUDA 12 runtime, shared with core
    assert bi['dlpack_consumer_stream'] == 1 and tuple(bi['dlpack_header_version']) == (1, 3), bi


def test_fast_math_canary_ieee_division_and_subnormals():
    """fp32 division compiled with the package's flags matches host IEEE division bitwise, and a subnormal quotient is
    not flushed to zero (--use_fast_math would give -prec_div=0 and -ftz=1)."""
    rng = np.random.default_rng(C.SEED)
    n = C.DIVIDE_PROBE_PAIRS
    xs = (rng.standard_normal(n) * np.exp2(rng.integers(-20, 20, n))).astype(np.float32)
    ys = (rng.standard_normal(n) * np.exp2(rng.integers(-20, 20, n))).astype(np.float32)
    tiny = np.float32(np.finfo(np.float32).tiny)
    xs[:4] = [tiny, tiny, -tiny, tiny * 3]
    ys[:4] = [2.0, 8.0, 4.0, 16.0]
    out = torch.empty(n, device='cuda')
    _C.fp32_divide_probe(torch.from_numpy(xs).cuda(), torch.from_numpy(ys).cuda(), out)
    torch.cuda.synchronize()
    got = out.cpu().numpy()
    want = xs / ys
    assert np.all(got[:4] != 0.0), f'subnormal quotients flushed to zero: {got[:4]}'
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), \
        f'{int(np.sum(got.view(np.uint32) != want.view(np.uint32)))} quotients differ from IEEE fp32 division'


def test_convt_matches_oracle_with_torch_tensor_inputs_mixed_with_producer_out():
    x, w, b = _t(2, 8, 20), _t(8, 3, 4), _t(3)
    ob = _t(2 * 3 * 42)
    ts.conv_transpose1d(x, w, b, stride=2, out=P.Producer(ob, (2, 3, 42)))
    torch.cuda.synchronize()
    ref = conv_transpose1d_f64(x, w, b, stride=2, pads=(0, 0), output_padding=0, dilation=1, groups=1)
    assert normwise(ob.view(2, 3, 42), ref) <= C.GATE_N
