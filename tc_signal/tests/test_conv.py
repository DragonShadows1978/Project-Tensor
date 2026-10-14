"""G1 numerics: tc_signal.conv1d / conv_transpose1d vs a torch float64 CPU oracle on every census cell (tcs_census.py).

Per cell: exact output shape, every element finite (outputs and scratch come from a NaN-prefilled allocator, so a
skipped write fails), normwise N <= 1e-5, and two calls bitwise equal.

Skip guard (order): skip only when torch is not importable or there is no CUDA device; any other exception fails.
"""
import zlib

import pytest

try:
    import torch
except ImportError as e:  # torch absent: the only import that may skip
    pytest.skip(f'torch is not importable: {e}', allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip('no CUDA device', allow_module_level=True)

import tc_signal as ts  # noqa: E402  (a failure here is a test failure, never a skip)

import tcs_census as C  # noqa: E402
from tcs_oracle import (bit_mismatches, bits_equal, conv1d_f64, conv_transpose1d_f64,  # noqa: E402
                        normwise)


@pytest.fixture(autouse=True)
def nan_allocator():
    prev = ts.set_allocator(lambda s: torch.full(s, C.NAN_FILL, device='cuda', dtype=torch.float32))
    yield
    ts.set_allocator(prev)


def _gen(tag):
    return torch.Generator().manual_seed(C.SEED + zlib.crc32(tag.encode()) % 1_000_003)


def _conv1d_data(tag, N, Cin, Cout, T, K, G, bias):
    g = _gen(tag)
    x = torch.randn(N, Cin, T, generator=g)
    w = torch.randn(Cout, Cin // G, K, generator=g) * (C.WEIGHT_SCALE / ((Cin // G) * K) ** 0.5)
    b = torch.randn(Cout, generator=g) if bias else None
    return x.cuda(), w.cuda(), (None if b is None else b.cuda())


def _convt_data(tag, N, Cin, Cout, T, K, G, bias):
    g = _gen(tag)
    x = torch.randn(N, Cin, T, generator=g)
    w = torch.randn(Cin, Cout // G, K, generator=g) * (C.WEIGHT_SCALE / ((Cin // G) * K) ** 0.5)
    b = torch.randn(Cout, generator=g) if bias else None
    return x.cuda(), w.cuda(), (None if b is None else b.cuda())


def _check(receipt, op, kind, name, attrs, y1, y2, o):
    torch.cuda.synchronize()
    shape_ok = tuple(y1.shape) == tuple(o.shape)
    finite = bool(torch.isfinite(y1).all().item())
    n = normwise(y1, o) if shape_ok else float('inf')
    det = bits_equal(y1, y2)
    receipt.append({'op': op, 'kind': kind, 'cell': name, 'attrs': attrs, 'shape': list(y1.shape), 'N': n,
                    'gate': C.GATE_N, 'deterministic': det, 'mismatches': 0 if det else bit_mismatches(y1, y2),
                    'all_finite': finite, 'pass': bool(shape_ok and finite and n <= C.GATE_N and det)})
    assert shape_ok, f'shape {tuple(y1.shape)} != oracle {tuple(o.shape)}'
    assert finite, 'non-finite output (an element was never written: the allocator pre-fills NaN)'
    assert n <= C.GATE_N, f'N {n:.3e} > gate {C.GATE_N:.0e}'
    assert det, f'two calls differ in {bit_mismatches(y1, y2)} elements'


CONV1D_CELLS = [('kokoro',) + c for c in C.CONV1D_KOKORO] + [('edge',) + c for c in C.CONV1D_EDGE]
CONVT_CELLS = [('kokoro',) + c for c in C.CONVT_KOKORO] + [('edge',) + c for c in C.CONVT_EDGE]


@pytest.mark.parametrize('cell', CONV1D_CELLS, ids=[f'{c[0]}-{c[1]}' for c in CONV1D_CELLS])
def test_conv1d_census(cell, receipt):
    kind, name, N, Cin, Cout, T, K, s, pads, d, G, bias = cell
    x, w, b = _conv1d_data('conv1d/' + name, N, Cin, Cout, T, K, G, bias)
    kw = dict(stride=s, padding=pads, dilation=d, groups=G)
    y1 = ts.conv1d(x, w, b, **kw)
    y2 = ts.conv1d(x, w, b, **kw)
    o = conv1d_f64(x, w, b, stride=s, pads=pads, dilation=d, groups=G)
    attrs = dict(N=N, C_in=Cin, C_out=Cout, T=T, K=K, stride=s, pads=list(pads), dilation=d, groups=G, bias=bias)
    _check(receipt, 'conv1d', kind, name, attrs, y1, y2, o)


@pytest.mark.parametrize('cell', CONVT_CELLS, ids=[f'{c[0]}-{c[1]}' for c in CONVT_CELLS])
def test_conv_transpose1d_census(cell, receipt):
    kind, name, N, Cin, Cout, T, K, s, pads, op, d, G, bias = cell
    x, w, b = _convt_data('convt/' + name, N, Cin, Cout, T, K, G, bias)
    kw = dict(stride=s, padding=pads, output_padding=op, dilation=d, groups=G)
    y1 = ts.conv_transpose1d(x, w, b, **kw)
    y2 = ts.conv_transpose1d(x, w, b, **kw)
    o = conv_transpose1d_f64(x, w, b, stride=s, pads=pads, output_padding=op, dilation=d, groups=G)
    attrs = dict(N=N, C_in=Cin, C_out=Cout, T=T, K=K, stride=s, pads=list(pads), output_padding=op, dilation=d,
                 groups=G, bias=bias)
    _check(receipt, 'conv_transpose1d', kind, name, attrs, y1, y2, o)


def _symmetric(cells):
    """Cells whose pads are symmetric (index 7 in both cell layouts), where torch's native padding applies."""
    return [c for c in cells if c[7][0] == c[7][1]]


@pytest.mark.parametrize('cell', _symmetric(C.CONV1D_EDGE), ids=lambda c: c[0])
def test_conv1d_oracle_matches_torch_native(cell):
    """The pad-then-conv oracle equals torch's own symmetric-padding conv1d in fp64 (oracle construction check)."""
    name, N, Cin, Cout, T, K, s, pads, d, G, bias = cell
    x, w, b = _conv1d_data('oracle/' + name, N, Cin, Cout, T, K, G, bias)
    o = conv1d_f64(x, w, b, stride=s, pads=pads, dilation=d, groups=G)
    nat = torch.nn.functional.conv1d(x.cpu().double(), w.cpu().double(), None if b is None else b.cpu().double(),
                                     stride=s, padding=pads[0], dilation=d, groups=G)
    assert o.shape == nat.shape and torch.allclose(o, nat, rtol=0, atol=1e-12)


@pytest.mark.parametrize('cell', _symmetric(C.CONVT_EDGE), ids=lambda c: c[0])
def test_conv_transpose1d_oracle_matches_torch_native(cell):
    """The full-then-crop oracle equals torch's own symmetric-padding conv_transpose1d (with output_padding) in fp64."""
    name, N, Cin, Cout, T, K, s, pads, op, d, G, bias = cell
    x, w, b = _convt_data('oracle/' + name, N, Cin, Cout, T, K, G, bias)
    o = conv_transpose1d_f64(x, w, b, stride=s, pads=pads, output_padding=op, dilation=d, groups=G)
    nat = torch.nn.functional.conv_transpose1d(x.cpu().double(), w.cpu().double(),
                                               None if b is None else b.cpu().double(), stride=s, padding=pads[0],
                                               output_padding=op, dilation=d, groups=G)
    assert o.shape == nat.shape and torch.allclose(o, nat, rtol=0, atol=1e-12)
