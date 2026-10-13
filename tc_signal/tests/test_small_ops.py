"""G1 (P1T-M4) for the T6 small ops: tc_signal.upsample_nearest1d / reflection_pad1d / rint, bit-exact against
integer index maps and numpy's rint on the uint32 view (m4_oracle.py), on every census cell (m4_census.py), plus a
strided sweep over fp32 bit patterns for rint, the refusals and the contract.

Exact rule: bit patterns equal, NaN equal to NaN (LD9). The two gathers copy 32-bit words, so they are additionally held
to payload-exact equality. Outputs are pre-filled with a finite sentinel, so a skipped write cannot hide behind NaN.

Skip guard (as M2): skip only when torch is not importable or there is no CUDA device; any other exception fails.
"""
import re

import numpy as np
import pytest

try:
    import torch
except ImportError as e:  # torch absent: the only import that may skip
    pytest.skip(f'torch is not importable: {e}', allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip('no CUDA device', allow_module_level=True)

import tc_signal as ts  # noqa: E402  (a failure here is a test failure, never a skip)

import m4_census as C  # noqa: E402
import m4_oracle as O  # noqa: E402
import tcs_producer as P  # noqa: E402


@pytest.fixture(autouse=True)
def sentinel_allocator():
    prev = ts.set_allocator(lambda s: torch.full(s, C.EXACT_FILL, device='cuda', dtype=torch.float32))
    yield
    ts.set_allocator(prev)


def _dev(a):
    # torch.from_numpy keeps the bits (NaN payloads included); the H2D copy is a byte copy.
    return torch.from_numpy(np.ascontiguousarray(a)).cuda()


def _host(t):
    return t.cpu().numpy()


def _record(receipt, op, kind, name, attrs, a1, a2, o, strict):
    shape_ok = a1.shape == o.shape
    eq = (O.bits_equal_strict(a1, o) if strict else O.bits_equal_nan_aware(a1, o)) if shape_ok else False
    mism = (int(np.sum(a1.view(np.uint32) != o.view(np.uint32))) if strict else O.bit_mismatches_nan_aware(a1, o)) \
        if shape_ok else None
    det = O.bits_equal_strict(a1, a2)
    receipt.append({'op': op, 'kind': kind, 'cell': name, 'attrs': attrs, 'shape': list(a1.shape),
                    'bit_mismatches': mism, 'payload_exact_rule': strict, 'deterministic': det,
                    'pass': bool(shape_ok and eq and det)})
    assert shape_ok, f'shape {a1.shape} != oracle {o.shape}'
    assert eq, f'{mism} elements differ bitwise from the oracle'
    assert det, 'two calls differ'


def _data(name, shape, data):
    return O.special_data(name, shape) if data == 'special' else O.act_data(name, shape)


@pytest.mark.parametrize('cell', C.UPSAMPLE_CELLS, ids=[f'{c[1]}-{c[0]}' for c in C.UPSAMPLE_CELLS])
def test_upsample_census(cell, receipt):
    name, kind, shape, s, data = cell
    x = _data('up/' + name, shape, data)
    X = _dev(x)
    a1, a2 = _host(ts.upsample_nearest1d(X, s)), _host(ts.upsample_nearest1d(X, s))
    _record(receipt, 'upsample_nearest1d', kind, name, {'shape': list(shape), 'scale': s}, a1, a2,
            O.upsample_oracle(x, s), strict=True)


def _pads(shape, pads):
    T = shape[-1]
    return tuple(T - 1 if p == 'T-1' else p for p in pads)


@pytest.mark.parametrize('cell', C.PAD_CELLS, ids=[f'{c[1]}-{c[0]}' for c in C.PAD_CELLS])
def test_reflection_pad_census(cell, receipt):
    name, kind, shape, pads, data = cell
    pb, pe = _pads(shape, pads)
    x = _data('pad/' + name, shape, data)
    X = _dev(x)
    a1, a2 = _host(ts.reflection_pad1d(X, (pb, pe))), _host(ts.reflection_pad1d(X, [pb, pe]))
    _record(receipt, 'reflection_pad1d', kind, name, {'shape': list(shape), 'pads': [pb, pe]}, a1, a2,
            O.pad_oracle(x, pb, pe), strict=True)


@pytest.mark.parametrize('cell', C.RINT_CELLS, ids=[f'{c[1]}-{c[0]}' for c in C.RINT_CELLS])
def test_rint_census(cell, receipt):
    name, kind, shape, data = cell
    x = O.rint_data('rint/' + name, shape, data)
    X = _dev(x)
    a1, a2 = _host(ts.rint(X)), _host(ts.rint(X))
    _record(receipt, 'rint', kind, name, {'shape': list(x.shape), 'data': data}, a1, a2, O.rint_oracle(x), strict=False)


def test_rint_order_vector_values():
    """The order's named rint cases, value by value (signs of zero included)."""
    v = np.array([0.5, -0.5, 1.5, -1.5, 2.5, -2.5, -0.4, np.inf, -np.inf, C.TWO23 - 0.5, C.SUBNORMAL_MIN,
                  -C.SUBNORMAL_MIN], np.float32)
    want = np.array([0.0, -0.0, 2.0, -2.0, 2.0, -2.0, -0.0, np.inf, -np.inf, C.TWO23, 0.0, -0.0], np.float32)
    got = _host(ts.rint(_dev(v)))
    assert O.bits_equal_strict(got, want), (got, want)
    nan = _host(ts.rint(_dev(np.array([np.nan, -np.nan], np.float32))))
    assert np.all(np.isnan(nan))


def test_rint_bit_pattern_sweep(receipt):
    """Every RINT_SWEEP_STRIDE-th fp32 bit pattern (all exponents, both signs, NaNs, subnormals) vs numpy rint."""
    w = np.arange(0, 1 << 32, C.RINT_SWEEP_STRIDE, dtype=np.uint64).astype(np.uint32)
    x = w.view(np.float32)
    got = _host(ts.rint(_dev(x)))
    with np.errstate(invalid='ignore'):        # signalling NaNs among the patterns
        want = np.rint(x)                      # float32 rint, IEEE roundTiesToEven
    assert O.bits_equal_nan_aware(want, O.rint_oracle(x))   # float32 and float64 rint agree (oracle self-check)
    mism = O.bit_mismatches_nan_aware(got, want)
    receipt.append({'op': 'rint', 'kind': 'sweep', 'cell': f'bit_patterns_stride_{C.RINT_SWEEP_STRIDE}',
                    'attrs': {'n': int(x.size)}, 'bit_mismatches': mism, 'pass': mism == 0})
    assert mism == 0, f'{mism} of {x.size} patterns differ from numpy rint'


def test_rint_0d():
    x = torch.tensor(2.5, device='cuda')
    y = ts.rint(x)
    assert tuple(y.shape) == () and float(y.item()) == 2.0


# ------------------------------------------------------------------------------------------------ oracle self-checks
def test_pad_oracle_matches_onnx_formula_and_torch():
    for T, pb, pe in [(10, 0, 0), (10, 2, 3), (10, 9, 9), (2, 1, 1), (1, 0, 0), (9840, 1, 0)]:
        x = O.special_data(f'padchk/{T}/{pb}/{pe}', (2, 3, T))
        o = O.pad_oracle(x, pb, pe)
        idx = O.pad_index_formula(T, pb, pe)
        assert O.bits_equal_strict(o, x.view(np.uint32)[..., idx].view(np.float32))
        xf = O.act_data(f'padchk2/{T}', (2, 3, T))
        if pb or pe:
            t = torch.nn.functional.pad(torch.from_numpy(xf).double(), (pb, pe), mode='reflect').float().numpy()
            assert O.bits_equal_strict(O.pad_oracle(xf, pb, pe), t)


def test_upsample_oracle_matches_onnx_asymmetric_floor():
    """y[i] = x[floor(i / s)] computed through float64 coordinates as ONNX Resize does (asymmetric, floor, clamp)."""
    for T, s in [(82, 2), (164, 300), (13, 7), (1, 5)]:
        i = np.arange(T * s, dtype=np.float64)
        idx = np.clip(np.floor(i / float(s)), 0, T - 1).astype(np.int64)
        assert np.array_equal(idx, np.arange(T * s) // s)


def test_rint_oracle_matches_torch_round_f64():
    x = O.rint_data('rintchk', (2, 3, 1000), 'wide')
    v = np.concatenate([x.ravel(), np.array(C.RINT_ORDER_VECTOR, np.float32)])
    t = torch.round(torch.from_numpy(v).double()).float().numpy()
    assert O.bits_equal_nan_aware(O.rint_oracle(v), t)


# ------------------------------------------------------------------------------------------------ refusals
def _t(*shape):
    return torch.randn(*shape, device='cuda')


def _overlap(fn, out_len):
    def call():
        st = torch.zeros(4000, device='cuda')
        x = st[:2 * 4 * 10].view(2, 4, 10)
        out = st[10:10 + 2 * 4 * out_len].view(2, 4, out_len)
        return fn(x, out)
    return call


REFUSALS = [
    ('up_scale0', lambda: ts.upsample_nearest1d(_t(2, 4, 10), 0), ValueError, 'scale'),
    ('up_scale_neg', lambda: ts.upsample_nearest1d(_t(2, 4, 10), -2), ValueError, 'scale'),
    ('up_scale_float_integral', lambda: ts.upsample_nearest1d(_t(2, 4, 10), 2.0), TypeError, 'integer scales'),
    ('up_scale_float_fraction', lambda: ts.upsample_nearest1d(_t(2, 4, 10), 1.5), TypeError, 'integer scales'),
    ('up_scale_bool', lambda: ts.upsample_nearest1d(_t(2, 4, 10), True), TypeError, 'bool'),
    ('up_scale_overflow', lambda: ts.upsample_nearest1d(_t(2, 4, 10), 1 << 62), ValueError, 'overflows'),
    ('up_x_0d', lambda: ts.upsample_nearest1d(torch.tensor(1.0, device='cuda'), 2), ValueError, 'at least 1'),
    ('up_x_fp16', lambda: ts.upsample_nearest1d(_t(2, 4, 10).half(), 2), TypeError, 'float32'),
    ('up_x_cpu', lambda: ts.upsample_nearest1d(torch.randn(2, 4, 10), 2), ValueError, 'kDLCUDA'),
    ('up_x_list', lambda: ts.upsample_nearest1d([[1.0]], 2), TypeError, 'DLPack producer'),
    ('up_x_noncontig', lambda: ts.upsample_nearest1d(_t(2, 10, 4).transpose(1, 2), 2), ValueError, 'C-contiguous'),
    ('up_out_wrong_shape', lambda: ts.upsample_nearest1d(_t(2, 4, 10), 2, out=_t(2, 4, 21)), ValueError, 'expected'),
    ('up_out_is_x', lambda: (lambda x: ts.upsample_nearest1d(x, 1, out=x))(_t(2, 4, 10)), ValueError, 'overlaps'),
    ('up_out_overlap', _overlap(lambda x, o: ts.upsample_nearest1d(x, 2, out=o), 20), ValueError, 'overlaps'),
    ('pad_begin_eq_T', lambda: ts.reflection_pad1d(_t(2, 4, 10), (10, 0)), ValueError, 'must be < T'),
    ('pad_end_eq_T', lambda: ts.reflection_pad1d(_t(2, 4, 10), (0, 10)), ValueError, 'must be < T'),
    ('pad_T1_p10', lambda: ts.reflection_pad1d(_t(2, 4, 1), (1, 0)), ValueError, 'must be < T'),
    ('pad_neg', lambda: ts.reflection_pad1d(_t(2, 4, 10), (-1, 0)), ValueError, '>= 0'),
    ('pad_int', lambda: ts.reflection_pad1d(_t(2, 4, 10), 1), TypeError, 'pads'),
    ('pad_len3', lambda: ts.reflection_pad1d(_t(2, 4, 10), (1, 0, 0)), ValueError, 'pads'),
    ('pad_float', lambda: ts.reflection_pad1d(_t(2, 4, 10), (1.0, 0)), TypeError, 'pads[0]'),
    ('pad_x_0d', lambda: ts.reflection_pad1d(torch.tensor(1.0, device='cuda'), (0, 0)), ValueError, 'at least 1'),
    ('pad_x_fp16', lambda: ts.reflection_pad1d(_t(2, 4, 10).half(), (1, 0)), TypeError, 'float32'),
    ('pad_out_wrong_shape', lambda: ts.reflection_pad1d(_t(2, 4, 10), (1, 0), out=_t(2, 4, 10)), ValueError,
     'expected'),
    ('pad_out_overlap', _overlap(lambda x, o: ts.reflection_pad1d(x, (1, 0), out=o), 11), ValueError, 'overlaps'),
    ('rint_x_fp16', lambda: ts.rint(_t(2, 4, 10).half()), TypeError, 'float32'),
    ('rint_x_cpu', lambda: ts.rint(torch.randn(2, 4)), ValueError, 'kDLCUDA'),
    ('rint_x_list', lambda: ts.rint([1.5]), TypeError, 'DLPack producer'),
    ('rint_x_empty', lambda: ts.rint(torch.empty(0, device='cuda')), ValueError, 'empty operand'),
    ('rint_out_wrong_shape', lambda: ts.rint(_t(2, 4), out=_t(2, 5)), ValueError, 'expected'),
    ('rint_out_is_x', lambda: (lambda x: ts.rint(x, out=x))(_t(2, 4)), ValueError, 'overlaps'),
    ('rint_out_fp16', lambda: ts.rint(_t(2, 4), out=_t(2, 4).half()), TypeError, 'float32'),
]


@pytest.mark.parametrize('case', REFUSALS, ids=[r[0] for r in REFUSALS])
def test_small_op_refusal(case):
    _, call, exc, frag = case
    with pytest.raises(exc, match=re.escape(frag)):
        call()
    torch.cuda.synchronize()


# ------------------------------------------------------------------------------------------------ contract
def test_small_ops_out_identity_and_allocator_calls():
    shapes = []

    def counting(s):
        shapes.append(tuple(s))
        return torch.empty(s, device='cuda')
    ts.set_allocator(counting)
    x = _t(2, 4, 10)
    ts.upsample_nearest1d(x, 3)
    ts.reflection_pad1d(x, (2, 1))
    ts.rint(x)
    assert shapes == [(2, 4, 30), (2, 4, 13), (2, 4, 10)]
    shapes.clear()
    o1, o2, o3 = torch.empty(2, 4, 30, device='cuda'), torch.empty(2, 4, 13, device='cuda'), torch.empty_like(x)
    assert ts.upsample_nearest1d(x, 3, out=o1) is o1
    assert ts.reflection_pad1d(x, (2, 1), out=o2) is o2
    assert ts.rint(x, out=o3) is o3
    assert shapes == []


def test_small_ops_producer_kwargs_byte_offset_and_deleters():
    import gc
    P.reset_stats()
    base = _t(1000)
    off = 13
    xn = base[off:off + 80].view(2, 4, 10).cpu().numpy()
    for mode in ('field', 'data'):
        x = P.Producer(base, (2, 4, 10), offset_elems=off, byte_offset_mode=mode)
        ob = _t(2 * 4 * 20)
        op = P.Producer(ob, (2, 4, 20))
        assert ts.upsample_nearest1d(x, 2, out=op) is op
        torch.cuda.synchronize()
        assert x.calls[0]['stream'] == 1 and x.calls[0]['copy'] is None
        assert op.calls[0]['stream'] == 1 and op.calls[0]['copy'] is False
        assert O.bits_equal_strict(ob.view(2, 4, 20).cpu().numpy(), O.upsample_oracle(xn, 2))
        y = ts.reflection_pad1d(P.Producer(base, (2, 4, 10), offset_elems=off, byte_offset_mode=mode), (3, 2))
        z = ts.rint(P.Producer(base, (2, 4, 10), offset_elems=off, byte_offset_mode=mode))
        torch.cuda.synchronize()
        assert O.bits_equal_strict(y.cpu().numpy(), O.pad_oracle(xn, 3, 2))
        assert O.bits_equal_nan_aware(z.cpu().numpy(), O.rint_oracle(xn))
    with pytest.raises(ValueError):
        ts.reflection_pad1d(P.Producer(base, (2, 4, 10)), (10, 0))                       # plan-phase refusal
    with pytest.raises(ValueError):
        ts.rint(P.Producer(base, (2, 4, 10)), out=P.Producer(_t(100), (2, 4, 11)))     # run-phase refusal
    gc.collect()
    torch.cuda.synchronize()
    assert P.STATS['unknown_delete'] == 0, P.STATS
    assert P.STATS['exported'] == P.STATS['deleted'], P.STATS


def test_small_ops_stream_order_torch_write_then_tc_signal_read():
    for trial in range(C.ORDER_PROBE_TRIALS):
        x = torch.zeros(*C.ORDER_PROBE_SHAPE, device='cuda')
        torch.cuda.synchronize()
        torch.cuda._sleep(C.SLEEP_CYCLES)
        x.fill_(1.5 + 2 * trial)               # a tie: rint gives the even neighbour
        y = ts.rint(x)
        u = ts.upsample_nearest1d(x, 2)
        torch.cuda.synchronize()
        assert bool((y == float(np.rint(1.5 + 2 * trial))).all().item()), f'trial {trial}: rint read x early'
        assert bool((u == 1.5 + 2 * trial).all().item()), f'trial {trial}: upsample read x early'


def test_small_ops_tensorcuda_producer_and_default_allocator():
    """The production path: TensorCUDA tensors in, outputs from tc_signal's default allocator (tensor_cuda.empty)."""
    tc = pytest.importorskip('tensor_cuda')
    ts.set_allocator(None)
    x = O.special_data('tc/special', (2, 4, 9))
    X = tc.tensor(x, dtype='float32')
    u = ts.upsample_nearest1d(X, 300)
    p = ts.reflection_pad1d(X, (8, 8))
    r = ts.rint(X)
    for t in (u, p, r):
        assert type(t).__module__.startswith('tensor_cuda'), type(t)
    assert O.bits_equal_strict(np.asarray(u.numpy()), O.upsample_oracle(x, 300))
    assert O.bits_equal_strict(np.asarray(p.numpy()), O.pad_oracle(x, 8, 8))
    assert O.bits_equal_nan_aware(np.asarray(r.numpy()), O.rint_oracle(x))
