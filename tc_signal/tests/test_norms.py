"""G1 (P1T-M4) for the T5 norms: tc_signal.instance_norm1d / layer_norm / group_norm vs a float64 CPU oracle
(m4_oracle.py) on every census cell (m4_census.py), plus the refusals and the DLPack/allocator contract of the norms.

Per cell: exact output shape; every element written (NaN-prefilled allocator, outputs finite); normwise N <= 1e-5;
two calls bitwise equal; zero-variance cells return the bias exactly.

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
def nan_allocator():
    prev = ts.set_allocator(lambda s: torch.full(s, C.NAN_FILL, device='cuda', dtype=torch.float32))
    yield
    ts.set_allocator(prev)


def _dev(a):
    return None if a is None else torch.from_numpy(np.ascontiguousarray(a)).cuda()


def _call(op, x, s, b, arg, eps, **kw):
    if op == 'instance':
        return ts.instance_norm1d(x, s, b, eps=eps, **kw)
    if op == 'layer':
        return ts.layer_norm(x, s, b, axis=arg, eps=eps, **kw)
    return ts.group_norm(x, arg, s, b, eps=eps, **kw)


NORM_CELLS = [('kokoro',) + c for c in C.NORM_KOKORO] + [('edge',) + c for c in C.NORM_EDGE]


@pytest.mark.parametrize('cell', NORM_CELLS, ids=[f'{c[0]}-{c[1]}' for c in NORM_CELLS])
def test_norm_census(cell, receipt):
    kind, name, op, shape, arg, eps, data, affine = cell
    x, s, b = O.norm_data(name, op, shape, arg, data, affine)
    X, S, B = _dev(x), _dev(s), _dev(b)
    y1 = _call(op, X, S, B, arg, eps)
    y2 = _call(op, X, S, B, arg, eps)
    torch.cuda.synchronize()
    a1, a2 = y1.cpu().numpy(), y2.cpu().numpy()
    o = O.norm_oracle(op, x, s, b, arg, eps)
    shape_ok = a1.shape == o.shape
    finite = bool(np.all(np.isfinite(a1)))
    n = O.normwise(a1, o) if shape_ok else float('inf')
    det = O.bits_equal_strict(a1, a2)
    bias_exact = None
    if name in C.NORM_BIAS_EXACT:
        bias_exact = bool(np.array_equal(a1, O.bias_expected(op, shape, b)))
    ok = bool(shape_ok and finite and n <= C.GATE_N and det and bias_exact is not False)
    receipt.append({'op': op + '_norm' if op != 'layer' else 'layer_norm', 'kind': kind, 'cell': name,
                    'attrs': {'shape': list(shape), 'arg': arg, 'eps': eps, 'data': data, 'affine': affine},
                    'N': n, 'gate': C.GATE_N, 'deterministic': det, 'all_finite': finite,
                    'bias_exact': bias_exact, 'pass': ok})
    assert shape_ok, f'shape {a1.shape} != oracle {o.shape}'
    assert finite, 'non-finite output (an element was never written: the allocator pre-fills NaN)'
    assert n <= C.GATE_N, f'N {n:.3e} > gate {C.GATE_N:.0e}'
    assert det, f'two calls differ in {int(np.sum(a1.view(np.uint32) != a2.view(np.uint32)))} elements'
    assert bias_exact is not False, 'a zero-variance cell did not return the bias exactly'


def _L(c):
    return O.norm_geometry(c[1], c[2], c[3])[1]


# torch's F.instance_norm refuses a single spatial element (L = 1), so L = 1 cells are checked by the census test only
# (against the numpy oracle and the exact-bias rule).
ORACLE_CHECK = [c for c in C.NORM_KOKORO if c[2][-1] <= 3000] + [c for c in C.NORM_EDGE if c[5] != 'const' and _L(c) > 1]


@pytest.mark.parametrize('cell', ORACLE_CHECK, ids=lambda c: c[0])
def test_norm_oracle_matches_torch_f64(cell):
    """Oracle construction check: the numpy two-pass oracle equals torch's float64 F.instance_norm / F.layer_norm /
    F.group_norm (independent implementations) within ORACLE_CROSSCHECK_N normwise."""
    name, op, shape, arg, eps, data, affine = cell
    x, s, b = O.norm_data(name, op, shape, arg, data, affine)
    o = O.norm_oracle(op, x, s, b, arg, eps)
    xt = torch.from_numpy(x).double()
    st = None if s is None else torch.from_numpy(s).double()
    bt = None if b is None else torch.from_numpy(b).double()
    F = torch.nn.functional
    if op == 'instance':
        t = F.instance_norm(xt, weight=st, bias=bt, eps=eps)
    elif op == 'group':
        t = F.group_norm(xt, arg, weight=st, bias=bt, eps=eps)
    else:
        a = arg + len(shape) if arg < 0 else arg
        t = F.layer_norm(xt, tuple(shape[a:]), weight=st, bias=bt, eps=eps)
    assert O.normwise(t.numpy(), o) <= C.ORACLE_CROSSCHECK_N


# ------------------------------------------------------------------------------------------------ refusals
def _t(*shape):
    return torch.randn(*shape, device='cuda')


def _in(**kw):
    """instance_norm1d with x [2, 8, 20], scale [8], bias [8] unless overridden."""
    x = kw.pop('x', None)
    s = kw.pop('scale', 'default')
    b = kw.pop('bias', 'default')
    x = _t(2, 8, 20) if x is None else x
    s = _t(8) if isinstance(s, str) else s
    b = _t(8) if isinstance(b, str) else b
    return ts.instance_norm1d(x, s, b, **kw)


def _ln(**kw):
    """layer_norm with x [2, 6, 16], scale/bias [16], axis -1 unless overridden."""
    x = kw.pop('x', None)
    s = kw.pop('scale', 'default')
    b = kw.pop('bias', 'default')
    x = _t(2, 6, 16) if x is None else x
    s = _t(16) if isinstance(s, str) else s
    b = _t(16) if isinstance(b, str) else b
    return ts.layer_norm(x, s, b, **kw)


def _gn(**kw):
    """group_norm with x [2, 8, 20], 4 groups, scale/bias [8] unless overridden."""
    x = kw.pop('x', None)
    g = kw.pop('num_groups', 4)
    s = kw.pop('scale', 'default')
    b = kw.pop('bias', 'default')
    x = _t(2, 8, 20) if x is None else x
    s = _t(8) if isinstance(s, str) else s
    b = _t(8) if isinstance(b, str) else b
    return ts.group_norm(x, g, s, b, **kw)


def _out_is_x(fn):
    def call():
        x = _t(2, 8, 20)
        return fn(x=x, out=x)
    return call


def _out_over_scale():
    st = torch.zeros(1000, device='cuda')
    s = st[:8]
    out = st[:320].view(2, 8, 20)
    return _in(scale=s, out=out)


REFUSALS = [
    # (id, call, exception, message fragment)
    ('in_x_list', lambda: _in(x=[[[1.0]]]), TypeError, 'DLPack producer'),
    ('in_x_cpu', lambda: _in(x=torch.randn(2, 8, 20)), ValueError, 'kDLCUDA'),
    ('in_x_fp16', lambda: _in(x=_t(2, 8, 20).half()), TypeError, 'float32'),
    ('in_scale_fp64', lambda: _in(scale=_t(8).double()), TypeError, 'float32'),
    ('in_bias_fp16', lambda: _in(bias=_t(8).half()), TypeError, 'float32'),
    ('in_x_2d', lambda: _in(x=_t(8, 20)), ValueError, 'dimensions'),
    ('in_x_4d', lambda: _in(x=_t(2, 8, 20, 1)), ValueError, 'dimensions'),
    ('in_x_noncontig', lambda: _in(x=_t(2, 20, 8).transpose(1, 2)), ValueError, 'C-contiguous'),
    ('in_x_empty_T', lambda: _in(x=torch.empty(2, 8, 0, device='cuda')), ValueError, 'empty operand'),
    ('in_scale_len', lambda: _in(scale=_t(7)), ValueError, 'scale has shape'),
    ('in_bias_len', lambda: _in(bias=_t(9)), ValueError, 'bias has shape'),
    ('in_scale_2d', lambda: _in(scale=_t(1, 8)), ValueError, 'scale has shape'),
    ('in_scale_0d', lambda: _in(scale=torch.tensor(1.0, device='cuda')), ValueError, 'at least 1 dimension'),
    ('in_eps_neg', lambda: _in(eps=-1e-5), ValueError, 'eps'),
    ('in_eps_nan', lambda: _in(eps=float('nan')), ValueError, 'eps'),
    ('in_eps_inf', lambda: _in(eps=float('inf')), ValueError, 'eps'),
    ('in_eps_over_fp32', lambda: _in(eps=1e39), ValueError, 'fits fp32'),
    ('in_eps_str', lambda: _in(eps='1e-5'), TypeError, 'eps'),
    ('in_eps_bool', lambda: _in(eps=True), TypeError, 'eps'),
    ('in_out_wrong_shape', lambda: _in(out=_t(2, 8, 21)), ValueError, 'expected'),
    ('in_out_fp16', lambda: _in(out=_t(2, 8, 20).half()), TypeError, 'float32'),
    ('in_out_cpu', lambda: _in(out=torch.empty(2, 8, 20)), ValueError, 'kDLCUDA'),
    ('in_out_is_x', _out_is_x(_in), ValueError, 'overlaps'),
    ('in_out_over_scale', _out_over_scale, ValueError, 'overlaps'),
    ('in_scale_other_device', lambda: _in(scale=P.Producer(_t(8), (8,), device_id=1)), ValueError, 'same device'),
    ('ln_axis_too_big', lambda: _ln(axis=3), ValueError, 'axis'),
    ('ln_axis_too_small', lambda: _ln(axis=-4), ValueError, 'axis'),
    ('ln_axis_float', lambda: _ln(axis=-1.0), TypeError, 'axis'),
    ('ln_axis_bool', lambda: _ln(axis=True), TypeError, 'bool'),
    ('ln_scale_wrong_for_axis-2', lambda: _ln(axis=-2), ValueError, 'scale has shape'),
    ('ln_bias_wrong_shape', lambda: _ln(bias=_t(15)), ValueError, 'bias has shape'),
    ('ln_x_0d', lambda: _ln(x=torch.tensor(1.0, device='cuda'), scale=None, bias=None), ValueError,
     'at least 1 dimension'),
    ('ln_eps_neg', lambda: _ln(eps=-1.0), ValueError, 'eps'),
    ('ln_out_is_x', lambda: (lambda x: _ln(x=x, out=x))(_t(2, 6, 16)), ValueError, 'overlaps'),
    ('ln_out_wrong_shape', lambda: _ln(out=_t(2, 6, 17)), ValueError, 'expected'),
    ('ln_x_fp16', lambda: _ln(x=_t(2, 6, 16).half()), TypeError, 'float32'),
    ('gn_groups0', lambda: _gn(num_groups=0), ValueError, 'num_groups'),
    ('gn_groups_neg', lambda: _gn(num_groups=-2), ValueError, 'num_groups'),
    ('gn_groups_not_dividing', lambda: _gn(num_groups=3), ValueError, 'divisible'),
    ('gn_groups_float', lambda: _gn(num_groups=4.0), TypeError, 'num_groups'),
    ('gn_scale_per_group', lambda: _gn(scale=_t(4)), ValueError, 'scale has shape'),
    ('gn_x_2d', lambda: _gn(x=_t(8, 20)), ValueError, 'dimensions'),
    ('gn_eps_nan', lambda: _gn(eps=float('nan')), ValueError, 'eps'),
    ('gn_out_is_x', _out_is_x(_gn), ValueError, 'overlaps'),
    ('gn_x_cpu', lambda: _gn(x=torch.randn(2, 8, 20)), ValueError, 'kDLCUDA'),
]


@pytest.mark.parametrize('case', REFUSALS, ids=[r[0] for r in REFUSALS])
def test_norm_refusal(case):
    _, call, exc, frag = case
    with pytest.raises(exc, match=re.escape(frag)):
        call()
    torch.cuda.synchronize()            # no launch error left behind


# ------------------------------------------------------------------------------------------------ contract
def test_norm_out_identity_and_allocator_calls():
    shapes = []

    def counting(s):
        shapes.append(tuple(s))
        return torch.empty(s, device='cuda')
    ts.set_allocator(counting)
    _in()
    _ln()
    _gn()
    assert shapes == [(2, 8, 20), (2, 6, 16), (2, 8, 20)]
    shapes.clear()
    out = torch.empty(2, 8, 20, device='cuda')
    assert _in(out=out) is out and _gn(out=out) is out
    out2 = torch.empty(2, 6, 16, device='cuda')
    assert _ln(out=out2) is out2
    assert shapes == []


def test_norm_producer_kwargs_and_deleters():
    """Any DLPack producer: x/scale/bias get stream=1 and no copy request; the output gets copy=False; every exported
    tensor's deleter runs exactly once on success, on a plan-phase refusal and on a run-phase refusal."""
    import gc
    P.reset_stats()
    xb, sb, bb = _t(2 * 8 * 20), _t(8), _t(8)

    def prods():
        return P.Producer(xb, (2, 8, 20)), P.Producer(sb, (8,)), P.Producer(bb, (8,))

    x, s, b = prods()
    ob = _t(2 * 8 * 20)
    op = P.Producer(ob, (2, 8, 20))
    y = ts.instance_norm1d(x, s, b, out=op)
    assert y is op
    torch.cuda.synchronize()
    for p in (x, s, b):
        assert p.calls == [{'stream': 1, 'max_version': (1, 3), 'dl_device': None, 'copy': None}]
    assert op.calls[0]['stream'] == 1 and op.calls[0]['copy'] is False
    xn = xb.view(2, 8, 20).cpu().numpy()
    o = O.norm_oracle('instance', xn, sb.cpu().numpy(), bb.cpu().numpy(), None, 1e-5)
    assert O.normwise(ob.view(2, 8, 20).cpu().numpy(), o) <= C.GATE_N
    x, s, b = prods()
    with pytest.raises(ValueError):
        ts.layer_norm(x, s, b, axis=7)                                   # plan-phase refusal
    x, s, b = prods()
    with pytest.raises(ValueError):
        ts.group_norm(x, 4, s, b, out=P.Producer(_t(400), (2, 8, 21)))  # run-phase refusal after consumption
    x, s, b = prods()
    with pytest.raises(ValueError):
        ts.instance_norm1d(x, s, b, out=P.Producer(_t(320), (2, 8, 20), flags=P.FLAG_READ_ONLY))
    del x, s, b
    gc.collect()
    torch.cuda.synchronize()
    assert P.STATS['unknown_delete'] == 0, P.STATS
    assert P.STATS['exported'] == P.STATS['deleted'], P.STATS


@pytest.mark.parametrize('mode', ['field', 'data'])
def test_norm_byte_offset_honoured(mode):
    base = _t(2000)
    off = 37
    x = P.Producer(base, (2, 6, 16), offset_elems=off, byte_offset_mode=mode)
    s, b = _t(16), _t(16)
    y = ts.layer_norm(x, s, b, eps=1e-5)
    torch.cuda.synchronize()
    xn = base[off:off + 192].view(2, 6, 16).cpu().numpy()
    o = O.norm_oracle('layer', xn, s.cpu().numpy(), b.cpu().numpy(), -1, 1e-5)
    assert O.normwise(y.cpu().numpy(), o) <= C.GATE_N


def test_norm_adjacent_output_allowed():
    st = torch.randn(320 + 320, device='cuda')
    x, out = st[:320].view(2, 8, 20), st[320:].view(2, 8, 20)
    xn = x.cpu().numpy()
    ts.instance_norm1d(x, None, None, out=out)
    torch.cuda.synchronize()
    assert O.normwise(out.cpu().numpy(), O.norm_oracle('instance', xn, None, None, None, 1e-5)) <= C.GATE_N


def test_norm_stream_order_torch_write_then_tc_signal_read():
    """torch queues a long sleep then a write of x on the shared legacy stream; the norm, launched right after without
    a sync, must normalize the written values (row r = r + 1 + trial*j/L: a wrong read gives NaN or another answer)."""
    N, Ch, T = C.ORDER_PROBE_SHAPE
    ramp = torch.arange(T, dtype=torch.float32, device='cuda') / T
    for trial in range(C.ORDER_PROBE_TRIALS):
        x = torch.zeros(N, Ch, T, device='cuda')
        torch.cuda.synchronize()
        torch.cuda._sleep(C.SLEEP_CYCLES)
        x.copy_(ramp.expand(N, Ch, T) * (1.0 + trial))
        y = ts.instance_norm1d(x, None, None)
        torch.cuda.synchronize()
        xn = (ramp.expand(N, Ch, T) * (1.0 + trial)).cpu().numpy()
        o = O.norm_oracle('instance', xn, None, None, None, 1e-5)
        assert O.normwise(y.cpu().numpy(), o) <= C.GATE_N, f'trial {trial}: read x before torch wrote it'


def test_norm_tensorcuda_producer_and_default_allocator():
    """The production path: TensorCUDA tensors in, outputs from tc_signal's default allocator (tensor_cuda.empty)."""
    tc = pytest.importorskip('tensor_cuda')
    ts.set_allocator(None)
    for name, op, shape, arg, eps, data, affine in [C.NORM_KOKORO[0], C.NORM_KOKORO[19], C.NORM_KOKORO[23],
                                                    C.NORM_EDGE[3], C.NORM_EDGE[10]]:
        x, s, b = O.norm_data(name, op, shape, arg, data, affine)
        up = (lambda a: None if a is None else tc.tensor(a, dtype='float32'))
        y = _call(op, up(x), up(s), up(b), arg, eps)
        assert type(y).__module__.startswith('tensor_cuda'), type(y)
        assert O.normwise(np.asarray(y.numpy()), O.norm_oracle(op, x, s, b, arg, eps)) <= C.GATE_N, name
