"""G1 for tc_signal.lstm (P1T-M3, T4): numerics vs an independent fp64 NumPy oracle, determinism, refusals, contract.

Numerics (tcs_lstm_census.py): every edge cell, every Kokoro-shape cell and the 12 real fixture cases; per cell, for
each of Y, Y_h, Y_c: exact shape, every element finite (NaN-prefilled allocator), normwise N <= 2e-5, and two calls
bitwise equal (the second with a different allocator pre-fill). The oracle itself is cross-checked against torch fp64
nn.LSTM after the ONNX -> torch gate-order conversion. A transcendental probe checks the cell's sigmoid/tanh in ulps.

Skip guard: the module skips only when torch is not importable or there is no CUDA device; the real-fixture cells skip
only when Voice's fixtures directory is absent (it is present on this machine; G1's log shows them run). Any other
exception fails.

Prior art: mutation-tested parity suites (DeMillo, Lipton & Sayward 1978); normwise error (Higham 2002); ONNX LSTM spec
for the oracle; the DLPack consumer-rule checks reuse M2's test producer (tcs_producer.py). Ours: the cells and probe.
"""
import gc
import hashlib
import json
import os
import zlib

import numpy as np
import pytest

try:
    import torch
except ImportError as e:  # torch absent: the only import that may skip
    pytest.skip(f'torch is not importable: {e}', allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip('no CUDA device', allow_module_level=True)

import tc_signal as ts  # noqa: E402  (a failure here is a test failure, never a skip)
from tc_signal import _tc_signal as _C  # noqa: E402

import tcs_lstm_census as L  # noqa: E402
import tcs_producer as P  # noqa: E402
from tcs_lstm_oracle import lstm_onnx_f64, lstm_torch_f64, normwise, sigmoid_f64, ulp_error  # noqa: E402

OUTPUTS = ('Y', 'Y_h', 'Y_c')


def _fill_allocator(value):
    return lambda s: torch.full(s, value, device='cuda', dtype=torch.float32)


@pytest.fixture(autouse=True)
def nan_allocator():
    prev = ts.set_allocator(_fill_allocator(L.NAN_FILL))
    yield
    ts.set_allocator(prev)


def _rng(tag):
    return np.random.default_rng(L.SEED + zlib.crc32(tag.encode()) % 1_000_003)


def _D(direction):
    return 2 if direction == 'bidirectional' else 1


def _cell_data(cell):
    name, direction, T, N, I, H, bias, init, kind = cell
    g = _rng('lstm/' + name)
    D = _D(direction)
    large = kind == 'large_z'
    wg, rg, bs = (L.LARGE_W_GAIN, L.LARGE_R_GAIN, L.LARGE_B_STD) if large else (L.W_GAIN, L.R_GAIN, L.B_STD)
    d = {
        'x': (g.standard_normal((T, N, I)) * L.X_STD).astype(np.float32),
        'W': (g.standard_normal((D, 4 * H, I)) * (wg / np.sqrt(I))).astype(np.float32),
        'R': (g.standard_normal((D, 4 * H, H)) * (rg / np.sqrt(H))).astype(np.float32),
        'B': None, 'initial_h': None, 'initial_c': None,
    }
    if bias:
        b = g.standard_normal((D, 8 * H)) * bs
        if large:
            pins = np.arange(0, 8 * H, L.LARGE_Z_PIN_EVERY)
            b[:, pins] = L.LARGE_Z_BIAS * np.where(np.arange(pins.size) % 2 == 0, 1.0, -1.0)
        d['B'] = b.astype(np.float32)
    if init:
        d['initial_h'] = (g.standard_normal((D, N, H)) * L.STATE_STD).astype(np.float32)
        d['initial_c'] = (g.standard_normal((D, N, H)) * L.STATE_STD).astype(np.float32)
    return d, direction, H


def _edge(name):
    return next(c for c in L.EDGE_CELLS if c[0] == name)


def _cu(a):
    return None if a is None else torch.from_numpy(np.ascontiguousarray(a)).cuda()


def _call(d, direction, H, fill=None):
    prev = ts.set_allocator(_fill_allocator(fill)) if fill is not None else None
    try:
        return ts.lstm(_cu(d['x']), _cu(d['W']), _cu(d['R']), _cu(d['B']), hidden_size=H, direction=direction,
                       initial_h=_cu(d['initial_h']), initial_c=_cu(d['initial_c']))
    finally:
        if prev is not None:
            ts.set_allocator(prev)


def _bits_equal(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


def _mismatches(a, b):
    return int((a.contiguous().view(torch.int32) != b.contiguous().view(torch.int32)).sum().item())


def _check(receipt, kind, name, attrs, outs1, outs2, oracle, extra=None):
    torch.cuda.synchronize()
    per = {}
    ok = True
    for nm, y1, y2, o in zip(OUTPUTS, outs1, outs2, oracle):
        y = y1.cpu().numpy()
        shape_ok = tuple(y.shape) == tuple(o.shape)
        finite = bool(np.isfinite(y).all())
        n = normwise(y, o) if shape_ok else float('inf')
        det = _bits_equal(y1, y2)
        per[nm] = {'shape': list(y.shape), 'N': n, 'deterministic': det,
                   'mismatches': 0 if det else _mismatches(y1, y2), 'all_finite': finite}
        ok &= shape_ok and finite and n <= L.GATE_N and det
    rec = {'op': 'lstm', 'kind': kind, 'cell': name, 'attrs': attrs, 'N': max(p['N'] for p in per.values()),
           'gate': L.GATE_N, 'deterministic': all(p['deterministic'] for p in per.values()),
           'all_finite': all(p['all_finite'] for p in per.values()), 'outputs': per, 'pass': bool(ok)}
    if extra:
        rec.update(extra)
    receipt.append(rec)
    for nm, p in per.items():
        assert p['all_finite'], f'{nm}: non-finite output (an element was never written: the allocator pre-fills NaN)'
        assert p['N'] <= L.GATE_N, f'{nm}: N {p["N"]:.3e} > gate {L.GATE_N:.0e}'
        assert p['deterministic'], f'{nm}: two calls differ in {p["mismatches"]} elements'
    for nm, y1, o in zip(OUTPUTS, outs1, oracle):
        assert tuple(y1.shape) == tuple(o.shape), f'{nm}: shape {tuple(y1.shape)} != oracle {tuple(o.shape)}'


def _attrs(cell):
    name, direction, T, N, I, H, bias, init, kind = cell
    return dict(direction=direction, T=T, N=N, I=I, H=H, bias=bias, initial_state=init, kind=kind)


# ------------------------------------------------------------------------------------------------ census cells
CELLS = [('edge',) + c for c in L.EDGE_CELLS] + [('kokoro',) + c for c in L.KOKORO_SHAPE_CELLS]


@pytest.mark.parametrize('cell', CELLS, ids=[f'{c[0]}-{c[1]}' for c in CELLS])
def test_lstm_census(cell, receipt):
    kind, c = cell[0], cell[1:]
    d, direction, H = _cell_data(c)
    outs1 = _call(d, direction, H)
    outs2 = _call(d, direction, H, fill=L.SECOND_FILL)
    Y, Yh, Yc, zmax = lstm_onnx_f64(d['x'], d['W'], d['R'], d['B'], direction=direction, hidden_size=H,
                                    initial_h=d['initial_h'], initial_c=d['initial_c'], return_zmax=True)
    _check(receipt, kind, c[0], _attrs(c), outs1, outs2, (Y, Yh, Yc), extra={'z_absmax': zmax})
    if c[8] == 'large_z':
        assert zmax >= L.LARGE_Z_MIN_REACHED, f'census: max |z| {zmax:.1f} < {L.LARGE_Z_MIN_REACHED} (cell too tame)'


# ------------------------------------------------------------------------------------------------ real fixtures
def _fixture_cases():
    man = os.path.join(L.FIXTURES_DIR, 'manifest.json')
    if not os.path.exists(man):
        return []
    with open(man) as f:
        m = json.load(f)
    return [c for c in m['cases'] if c.get('op_type') == 'LSTM']


FIXTURE_CASES = _fixture_cases()


def test_lstm_fixture_manifest():
    """The real-data cells read exactly the fixtures Voice's runner checks (manifest sha256 and T4 case count)."""
    man = os.path.join(L.FIXTURES_DIR, 'manifest.json')
    if not os.path.exists(man):
        pytest.skip(f'Voice fixtures absent at {L.FIXTURES_DIR}')
    with open(man, 'rb') as f:
        assert hashlib.sha256(f.read()).hexdigest() == L.FIXTURES_MANIFEST_SHA256
    assert len(FIXTURE_CASES) == L.FIXTURES_T4_CASES


@pytest.mark.parametrize('case', FIXTURE_CASES, ids=[c['id'].replace('T4/', 'real-') for c in FIXTURE_CASES])
def test_lstm_real_fixture(case, receipt):
    cz = np.load(os.path.join(L.FIXTURES_DIR, case['case_file']))
    wz = np.load(os.path.join(L.FIXTURES_DIR, case['weights_file']))
    at = case['attrs']
    H, direction = int(at['hidden_size']), at.get('direction', 'forward')
    assert int(at.get('layout', 0)) == 0
    d = {'x': cz['x'], 'W': wz['W'], 'R': wz['R'], 'B': wz['B'],
         'initial_h': cz['initial_h'] if 'initial_h' in cz.files else None,
         'initial_c': cz['initial_c'] if 'initial_c' in cz.files else None}
    T, N = d['x'].shape[:2]
    seq = cz['sequence_lens'] if 'sequence_lens' in cz.files else None
    outs1 = ts.lstm(_cu(d['x']), _cu(d['W']), _cu(d['R']), _cu(d['B']), hidden_size=H, direction=direction,
                    initial_h=_cu(d['initial_h']), initial_c=_cu(d['initial_c']), sequence_lens=seq)
    outs2 = _call(d, direction, H, fill=L.SECOND_FILL)
    Y, Yh, Yc, zmax = lstm_onnx_f64(d['x'], d['W'], d['R'], d['B'], direction=direction, hidden_size=H,
                                    initial_h=d['initial_h'], initial_c=d['initial_c'], return_zmax=True)
    attrs = dict(direction=direction, T=int(T), N=int(N), I=int(d['x'].shape[2]), H=H, node=case['node'],
                 case_id=case['case_id'], sequence_lens=None if seq is None else seq.tolist())
    _check(receipt, 'real', case['id'], attrs, outs1, outs2, (Y, Yh, Yc),
           extra={'z_absmax': zmax, 'c_absmax': float(np.abs(Yc).max())})


# ------------------------------------------------------------------------------------------------ oracle cross-check
XCHECK_CELLS = list(L.EDGE_CELLS) + [c for c in L.KOKORO_SHAPE_CELLS if c[0] in L.XCHECK_KOKORO]


@pytest.mark.parametrize('cell', XCHECK_CELLS, ids=[c[0] for c in XCHECK_CELLS])
def test_lstm_oracle_matches_torch_f64(cell):
    """My NumPy ONNX-LSTM oracle == torch fp64 nn.LSTM after the gate-order conversion (i,o,f,c -> i,f,g,o)."""
    d, direction, H = _cell_data(cell)
    o = lstm_onnx_f64(d['x'], d['W'], d['R'], d['B'], direction=direction, hidden_size=H,
                      initial_h=d['initial_h'], initial_c=d['initial_c'])
    t = lstm_torch_f64(d['x'], d['W'], d['R'], d['B'], direction=direction, hidden_size=H,
                       initial_h=d['initial_h'], initial_c=d['initial_c'])
    for nm, a, b in zip(OUTPUTS, o, t):
        assert a.shape == b.shape, nm
        assert normwise(b, a) <= L.ORACLE_XCHECK_N, f'{nm}: oracle vs torch N {normwise(b, a):.3e}'


def test_lstm_oracle_gate_order_is_onnx():
    """The oracle is not symmetric in the gate order: swapping two ONNX gate blocks of W changes its output (guards
    against an oracle that would accept torch's order too)."""
    cell = _edge('bi_T33_N2')
    d, direction, H = _cell_data(cell)
    o = lstm_onnx_f64(d['x'], d['W'], d['R'], d['B'], direction=direction, hidden_size=H)
    W2 = d['W'].copy()
    W2[:, H:2 * H], W2[:, 2 * H:3 * H] = d['W'][:, 2 * H:3 * H], d['W'][:, H:2 * H]
    o2 = lstm_onnx_f64(d['x'], W2, d['R'], d['B'], direction=direction, hidden_size=H)
    assert normwise(o2[0], o[0]) > 1e-3


# ------------------------------------------------------------------------------------------------ transcendental probe
def test_lstm_transcendental_probe(receipt):
    """Isolates the cell's sigmoid and tanh (z = Wb exactly; registered thresholds in ulps, ledger 20:33)."""
    zs = np.linspace(*L.PROBE_SIGMOID_Z).astype(np.float32)
    zt = np.linspace(*L.PROBE_TANH_Z).astype(np.float32)
    Ps, Pt = zs.size, zt.size
    H = Ps + Pt
    Wb = np.zeros((4, H), np.float32)                        # ONNX gate rows i, o, f, c
    Wb[0, :Ps], Wb[3, :Ps] = zs, L.PROBE_TANH_ONE_Z             # sigmoid columns: Y_c = sigmoid(z) * tanh(20)
    Wb[0, Ps:], Wb[3, Ps:] = L.PROBE_SIGMOID_ONE_Z, zt          # tanh columns:    Y_c = sigmoid(60) * tanh(z)
    Wb[1, :] = L.PROBE_SIGMOID_ONE_Z                            # o = 1 exactly: Y_h = tanh(Y_c)
    B = np.zeros((1, 8 * H), np.float32)
    B[0, :4 * H] = Wb.reshape(-1)
    x = np.zeros((1, 1, 1), np.float32)
    W = np.zeros((1, 4 * H, 1), np.float32)
    R = np.zeros((1, 4 * H, H), np.float32)
    Y, Yh, Yc = ts.lstm(_cu(x), _cu(W), _cu(R), _cu(B), hidden_size=H, direction='forward')
    torch.cuda.synchronize()
    yc = Yc.cpu().numpy().reshape(-1).astype(np.float64)
    yh = Yh.cpu().numpy().reshape(-1).astype(np.float64)
    ref_c_sig = sigmoid_f64(zs.astype(np.float64)) * np.tanh(np.float64(L.PROBE_TANH_ONE_Z))
    ref_c_tanh = np.tanh(zt.astype(np.float64))
    ref_h = np.tanh(Yc.cpu().numpy().reshape(-1).astype(np.float32).astype(np.float64))
    u_sig = ulp_error(yc[:Ps], ref_c_sig)
    u_tanh = ulp_error(yc[Ps:], ref_c_tanh)
    u_h = ulp_error(yh, ref_h)
    rec = {'op': 'lstm', 'kind': 'probe', 'cell': 'transcendental_probe',
           'sigmoid_max_ulp': float(u_sig.max()), 'sigmoid_argmax_z': float(zs[int(u_sig.argmax())]),
           'tanh_max_ulp': float(u_tanh.max()), 'tanh_argmax_z': float(zt[int(u_tanh.argmax())]),
           'tanh_of_state_max_ulp': float(u_h.max()), 'sigmoid_limit_ulp': L.PROBE_SIGMOID_MAX_ULP,
           'tanh_limit_ulp': L.PROBE_TANH_MAX_ULP, 'all_finite': bool(np.isfinite(yc).all() and np.isfinite(yh).all())}
    rec['pass'] = bool(rec['all_finite'] and rec['sigmoid_max_ulp'] <= L.PROBE_SIGMOID_MAX_ULP
                       and rec['tanh_max_ulp'] <= L.PROBE_TANH_MAX_ULP
                       and rec['tanh_of_state_max_ulp'] <= L.PROBE_TANH_MAX_ULP)
    receipt.append(rec)
    assert rec['all_finite']
    assert rec['sigmoid_max_ulp'] <= L.PROBE_SIGMOID_MAX_ULP, rec
    assert rec['tanh_max_ulp'] <= L.PROBE_TANH_MAX_ULP, rec
    assert rec['tanh_of_state_max_ulp'] <= L.PROBE_TANH_MAX_ULP, rec


# ------------------------------------------------------------------------------------------------ refusals
def _small(direction='bidirectional', **over):
    """Torch CUDA operands for the contract shape, with overrides (value 'drop' removes an operand)."""
    T, N, I, H = L.CONTRACT_SHAPE
    D = _D(direction)
    g = _rng('lstm/contract')
    ops = {'x': torch.from_numpy(g.standard_normal((T, N, I)).astype(np.float32)).cuda(),
           'W': torch.from_numpy((g.standard_normal((D, 4 * H, I)) * 0.5).astype(np.float32)).cuda(),
           'R': torch.from_numpy((g.standard_normal((D, 4 * H, H)) * 0.5).astype(np.float32)).cuda(),
           'B': torch.from_numpy((g.standard_normal((D, 8 * H)) * 0.1).astype(np.float32)).cuda(),
           'initial_h': torch.from_numpy(g.standard_normal((D, N, H)).astype(np.float32)).cuda(),
           'initial_c': torch.from_numpy(g.standard_normal((D, N, H)).astype(np.float32)).cuda()}
    ops.update(over)
    return ops


def _lstm_small(direction='bidirectional', ops=None, **kw):
    T, N, I, H = L.CONTRACT_SHAPE
    o = _small(direction) if ops is None else ops
    kw.setdefault('hidden_size', H)
    return ts.lstm(o['x'], o['W'], o['R'], o['B'], direction=direction, initial_h=o['initial_h'],
                   initial_c=o['initial_c'], **kw)


def _t(*shape):
    return torch.randn(*shape, device='cuda')


def _T():
    return L.CONTRACT_SHAPE[0]


def _N():
    return L.CONTRACT_SHAPE[1]


def _H():
    return L.CONTRACT_SHAPE[3]


def _out_tuple(direction='bidirectional', **over):
    T, N, I, H = L.CONTRACT_SHAPE
    D = _D(direction)
    o = {'Y': _t(T, D, N, H), 'Y_h': _t(D, N, H), 'Y_c': _t(D, N, H)}
    o.update(over)
    return (o['Y'], o['Y_h'], o['Y_c'])


def _alias_out_over_x():
    T, N, I, H = L.CONTRACT_SHAPE
    st = torch.zeros(T * 2 * N * H + T * N * I, device='cuda')
    ops = _small(x=st[:T * N * I].view(T, N, I))
    out = (st[T * N * I // 2:T * N * I // 2 + T * 2 * N * H].view(T, 2, N, H), _t(2, N, H), _t(2, N, H))
    return _lstm_small(ops=ops, out=out)


def _same_yh_yc():
    T, N, I, H = L.CONTRACT_SHAPE
    s = _t(2, N, H)
    return _lstm_small(out=(_t(T, 2, N, H), s, s))


def _out_over_initial_h():
    ops = _small()
    T, N, I, H = L.CONTRACT_SHAPE
    return _lstm_small(ops=ops, out=(_t(T, 2, N, H), ops['initial_h'], _t(2, N, H)))


def _device_mismatch():
    ops = _small()
    ops['W'] = P.Producer(ops['W'], tuple(ops['W'].shape), device_id=1)
    return _lstm_small(ops=ops)


REFUSALS = [
    # (id, call, exception, message fragment)
    ('layout_1', lambda: _lstm_small(layout=1), ValueError, 'layout=1'),
    ('layout_bool', lambda: _lstm_small(layout=True), TypeError, 'layout'),
    ('layout_str', lambda: _lstm_small(layout='onnx'), TypeError, 'layout'),
    ('seq_lens_short', lambda: _lstm_small(sequence_lens=[_T() - 1] * _N()), ValueError, 'not supported'),
    ('seq_lens_mixed', lambda: _lstm_small(sequence_lens=[_T(), _T() - 1]), ValueError, 'not supported'),
    ('seq_lens_zero', lambda: _lstm_small(sequence_lens=[0] * _N()), ValueError, 'not supported'),
    ('seq_lens_shape', lambda: _lstm_small(sequence_lens=[[_T()] * _N()]), ValueError, 'shape'),
    ('seq_lens_len', lambda: _lstm_small(sequence_lens=[_T()] * (_N() + 1)), ValueError, 'shape'),
    ('seq_lens_float', lambda: _lstm_small(sequence_lens=[float(_T())] * _N()), TypeError, 'integers'),
    ('seq_lens_cuda_short', lambda: _lstm_small(sequence_lens=torch.full((_N(),), _T() - 1, dtype=torch.int32,
                                                                          device='cuda')), ValueError, 'not supported'),
    ('direction_bad', lambda: _lstm_small(direction='backward', ops=_small()), ValueError, 'direction'),
    ('direction_case', lambda: _lstm_small(direction='Forward', ops=_small()), ValueError, 'direction'),
    ('direction_none', lambda: _lstm_small(direction=None, ops=_small()), ValueError, 'direction'),
    ('direction_vs_W', lambda: _lstm_small(direction='forward', ops=_small()), ValueError, 'W.shape[0]'),
    ('hidden_size_mismatch', lambda: _lstm_small(hidden_size=_H() + 1), ValueError, '4*hidden_size'),
    ('hidden_size_float', lambda: _lstm_small(hidden_size=float(_H())), TypeError, 'hidden_size'),
    ('hidden_size_bool', lambda: _lstm_small(hidden_size=True), TypeError, 'hidden_size'),
    ('hidden_size_zero', lambda: _lstm_small(hidden_size=0), ValueError, 'hidden_size'),
    ('hidden_size_negative', lambda: _lstm_small(hidden_size=-4), ValueError, 'hidden_size'),
    ('W_bad_input_size', lambda: _lstm_small(ops=_small(W=_t(2, 4 * _H(), 7))), ValueError, 'W.shape[2]'),
    ('W_2d', lambda: _lstm_small(ops=_small(W=_t(4 * _H(), 3))), ValueError, 'dimensions'),
    ('R_bad', lambda: _lstm_small(ops=_small(R=_t(2, 4 * _H(), _H() + 1))), ValueError, 'R has shape'),
    ('R_one_direction', lambda: _lstm_small(ops=_small(R=_t(1, 4 * _H(), _H()))), ValueError, 'R has shape'),
    ('B_half', lambda: _lstm_small(ops=_small(B=_t(2, 4 * _H()))), ValueError, 'B has shape'),
    ('B_1d', lambda: _lstm_small(ops=_small(B=_t(8 * _H()))), ValueError, 'dimensions'),
    ('initial_h_bad_N', lambda: _lstm_small(ops=_small(initial_h=_t(2, _N() + 1, _H()))), ValueError,
     'initial_h has shape'),
    ('initial_c_bad_H', lambda: _lstm_small(ops=_small(initial_c=_t(2, _N(), _H() - 1))), ValueError,
     'initial_c has shape'),
    ('x_2d', lambda: _lstm_small(ops=_small(x=_t(_T(), 3))), ValueError, 'dimensions'),
    ('x_empty_T', lambda: _lstm_small(ops=_small(x=torch.empty(0, _N(), 3, device='cuda'))), ValueError,
     'empty operand'),
    ('x_fp16', lambda: _lstm_small(ops=_small(x=_t(_T(), _N(), 3).half())), TypeError, 'float32'),
    ('W_fp64', lambda: _lstm_small(ops=_small(W=_t(2, 4 * _H(), 3).double())), TypeError, 'float32'),
    ('B_fp16', lambda: _lstm_small(ops=_small(B=_t(2, 8 * _H()).half())), TypeError, 'float32'),
    ('x_cpu', lambda: _lstm_small(ops=_small(x=torch.randn(_T(), _N(), 3))), ValueError, 'kDLCUDA'),
    ('x_numpy', lambda: _lstm_small(ops=_small(x=np.zeros((_T(), _N(), 3), np.float32))), ValueError, 'kDLCUDA'),
    ('x_noncontiguous', lambda: _lstm_small(ops=_small(x=torch.randn(3, _N(), _T(), device='cuda').transpose(0, 2))),
     ValueError, 'C-contiguous'),
    ('x_list', lambda: _lstm_small(ops=_small(x=[[[1.0]]])), TypeError, 'DLPack producer'),
    ('x_none', lambda: _lstm_small(ops=_small(x=None)), TypeError, 'got None'),
    ('W_none', lambda: _lstm_small(ops=_small(W=None)), TypeError, 'got None'),
    ('device_mismatch', _device_mismatch, ValueError, 'same device'),
    ('out_not_tuple', lambda: _lstm_small(out=_t(_T(), 2, _N(), _H())), TypeError, '3-tuple'),
    ('out_len2', lambda: _lstm_small(out=_out_tuple()[:2]), ValueError, '3-tuple'),
    ('out_Y_shape', lambda: _lstm_small(out=_out_tuple(Y=_t(_T(), 1, _N(), _H()))), ValueError, 'expected'),
    ('out_Y_h_shape', lambda: _lstm_small(out=_out_tuple(Y_h=_t(2, _N(), _H() + 1))), ValueError, 'expected'),
    ('out_Y_c_rank', lambda: _lstm_small(out=_out_tuple(Y_c=_t(2 * _N() * _H()))), ValueError, 'dimensions'),
    ('out_Y_fp16', lambda: _lstm_small(out=_out_tuple(Y=_t(_T(), 2, _N(), _H()).half())), TypeError, 'float32'),
    ('out_Y_none', lambda: _lstm_small(out=_out_tuple(Y=None)), TypeError, 'DLPack producer'),
    ('out_Y_over_x', _alias_out_over_x, ValueError, 'overlaps'),
    ('out_Y_h_is_Y_c', _same_yh_yc, ValueError, 'overlaps'),
    ('out_Y_h_is_initial_h', _out_over_initial_h, ValueError, 'overlaps'),
    ('clip_kw', lambda: _lstm_small(clip=3.0), TypeError, 'clip'),
    ('peepholes_P_kw', lambda: _lstm_small(P=_t(2, 3 * _H())), TypeError, "'P'"),
    ('input_forget_kw', lambda: _lstm_small(input_forget=1), TypeError, 'input_forget'),
    ('activations_kw', lambda: _lstm_small(activations=['Sigmoid', 'Tanh', 'Tanh']), TypeError, 'activations'),
    ('hidden_size_missing', lambda: ts.lstm(_t(_T(), _N(), 3), _t(1, 4 * _H(), 3), _t(1, 4 * _H(), _H())),
     TypeError, 'hidden_size'),
]


@pytest.mark.parametrize('case', REFUSALS, ids=[r[0] for r in REFUSALS])
def test_lstm_refusal(case):
    _, call, exc, frag = case
    with pytest.raises(exc) as ei:
        call()
    assert frag in str(ei.value), f'message {str(ei.value)!r} lacks {frag!r}'


# ------------------------------------------------------------------------------------------------ contract
def test_lstm_sequence_lens_equal_T_accepted():
    """sequence_lens equal to T everywhere is accepted in every host and device form, and changes nothing."""
    ops = _small()
    ref = _lstm_small(ops=ops)
    T, N = _T(), _N()
    forms = [[T] * N, (T,) * N, np.full(N, T, np.int32), np.full(N, T, np.int64),
             torch.full((N,), T, dtype=torch.int32, device='cuda'), torch.full((N,), T, dtype=torch.int64)]
    for f in forms:
        got = _lstm_small(ops=ops, sequence_lens=f)
        for a, b in zip(ref, got):
            assert _bits_equal(a, b), type(f).__name__


def test_lstm_out_tuple_is_written_and_returned():
    out = _out_tuple()
    ret = _lstm_small(out=out)
    assert all(a is b for a, b in zip(ret, out))
    ref = _lstm_small()
    for a, b in zip(out, ref):
        assert _bits_equal(a, b)


def test_lstm_allocations_go_through_the_allocator():
    """Outputs and the one scratch operand come from the module allocator, with the planned shapes."""
    shapes = []

    def alloc(s):
        shapes.append(tuple(s))
        return torch.full(s, L.NAN_FILL, device='cuda', dtype=torch.float32)
    prev = ts.set_allocator(alloc)
    try:
        Y, Yh, Yc = _lstm_small()
    finally:
        ts.set_allocator(prev)
    T, N, I, H = L.CONTRACT_SHAPE
    D = 2
    assert shapes == [(T, D, N, H), (D, N, H), (D, N, H), (D * T * N * 4 * H + D * N * 4 * H,)]
    assert tuple(Y.shape) == (T, D, N, H) and tuple(Yh.shape) == (D, N, H) and tuple(Yc.shape) == (D, N, H)


def test_lstm_last_step_state_matches_Y():
    """Y_h is Y at each direction's last step: Y[T-1, 0] (forward) and Y[0, 1] (reverse), bitwise."""
    Y, Yh, Yc = _lstm_small()
    torch.cuda.synchronize()
    assert _bits_equal(Yh[0], Y[-1, 0]) and _bits_equal(Yh[1], Y[0, 1])


def test_lstm_cublas_handle_config():
    """The LSTM's cuBLAS handle: legacy default stream, CUBLAS_DEFAULT_MATH (not TF32), atomics not allowed."""
    c = _C.lstm_cublas_config(torch.cuda.current_device())
    assert c['legacy_stream'] and c['default_math'] and c['atomics_not_allowed'], c


def test_lstm_deleters_and_stream_every_path():
    """Every exported DLPack tensor is deleted exactly once on success, on a plan-time refusal, on a post-plan
    refusal (sequence_lens) and on a run-time refusal (bad out shape); every __dlpack__ call asked for stream=1."""
    T, N, I, H = L.CONTRACT_SHAPE
    base = _small()
    prods = {k: P.Producer(v, tuple(v.shape)) for k, v in base.items()}
    gc.collect()
    P.reset_stats()

    def call(**kw):
        return ts.lstm(prods['x'], prods['W'], prods['R'], prods['B'], hidden_size=kw.pop('hidden_size', H),
                       direction='bidirectional', initial_h=prods['initial_h'], initial_c=prods['initial_c'], **kw)
    outs = call()
    torch.cuda.synchronize()
    with pytest.raises(ValueError):
        call(hidden_size=H + 1)                                        # refused while planning
    with pytest.raises(ValueError):
        call(sequence_lens=[T - 1] * N)                                # refused after planning (capsules unconsumed)
    with pytest.raises(ValueError):
        call(out=_out_tuple(Y=_t(T, 2, N, H + 1)))                     # refused while running (capsules consumed)
    gc.collect()
    assert P.STATS['exported'] > 0
    assert P.STATS['deleted'] == P.STATS['exported'], P.STATS
    assert P.STATS['unknown_delete'] == 0, P.STATS
    for p in prods.values():
        assert p.calls and all(c['stream'] == 1 for c in p.calls)
    ref = ts.lstm(base['x'], base['W'], base['R'], base['B'], hidden_size=H, direction='bidirectional',
                  initial_h=base['initial_h'], initial_c=base['initial_c'])
    for a, b in zip(outs, ref):
        assert _bits_equal(a, b)


def test_lstm_tensor_cuda_producers_and_default_allocator(receipt):
    """The production path: TensorCUDA tensors in, tc_signal's default allocator (tensor_cuda.empty) for outputs and
    scratch, `.numpy()` download; same N gate as the census against the fp64 oracle, and two calls bitwise equal."""
    try:
        import tensor_cuda as tc
    except ImportError as e:
        pytest.skip(f'tensor_cuda is not importable: {e}')
    if not hasattr(tc, 'empty'):
        pytest.skip('this tensor_cuda has no empty() (P1T-M1 absent)')
    cell = _edge('bi_init_T40_N3_H20')                       # bidirectional, init state, H 20, T 40, N 3
    d, direction, H = _cell_data(cell)
    prev = ts.set_allocator(None)
    try:
        def up(a):
            return None if a is None else tc.tensor(np.ascontiguousarray(a), dtype='float32')
        args = [up(d[k]) for k in ('x', 'W', 'R', 'B')]
        outs = [ts.lstm(*args, hidden_size=H, direction=direction, initial_h=up(d['initial_h']),
                        initial_c=up(d['initial_c'])) for _ in range(2)]
    finally:
        ts.set_allocator(prev)
    o = lstm_onnx_f64(d['x'], d['W'], d['R'], d['B'], direction=direction, hidden_size=H,
                      initial_h=d['initial_h'], initial_c=d['initial_c'])
    host = [[np.ascontiguousarray(t.numpy()) for t in out] for out in outs]
    ns = [normwise(y, oo) for y, oo in zip(host[0], o)]
    det = all(a.tobytes() == b.tobytes() for a, b in zip(host[0], host[1]))
    receipt.append({'op': 'lstm', 'kind': 'tensor_cuda', 'cell': cell[0], 'N': max(ns), 'gate': L.GATE_N,
                    'deterministic': det, 'pass': bool(max(ns) <= L.GATE_N and det)})
    assert max(ns) <= L.GATE_N and det, (ns, det)
