"""tc_signal T6 small ops (P1T-M4): upsample_nearest1d, reflection_pad1d, rint. Bit-exact by construction (32-bit word
gathers; rintf). Kernel notes: src/small_ops.cu. Operands are any fp32 CUDA DLPack producers (or capsules),
C-contiguous; the time axis is the last axis; the output comes from the module allocator unless `out` is given;
everything is queued on the legacy default stream. TensorCUDA core's `round` is untouched.

Prior art: ONNX Resize (nearest, asymmetric, floor), Pad (reflect) and Round specs; IEEE 754 roundTiesToEven / C99 rint;
torch reflection_pad1d. Ours: the Python wrappers.
"""
from __future__ import annotations

import operator
import sys

from . import _tc_signal as _C

__all__ = ['upsample_nearest1d', 'reflection_pad1d', 'rint']


def _alloc(shape):
    # The package's current allocator (tc_signal.set_allocator), read at call time.
    return sys.modules[__package__]._allocator(tuple(shape))


def _int(name, v):
    if isinstance(v, bool):
        raise TypeError(f'tc_signal: {name} must be an int, got bool')
    try:
        return operator.index(v)
    except TypeError:
        raise TypeError(f'tc_signal: {name} must be an int, got {type(v).__name__}') from None


def upsample_nearest1d(x, scale, *, out=None):
    """ONNX Resize nearest (asymmetric, floor) on the last axis at an INTEGER scale >= 1:
    y[..., i] = x[..., i // scale], output length T * scale. Non-integer scales are refused (TypeError). Returns `out`."""
    s = _int('scale (upsample_nearest1d takes integer scales only)', scale)
    if s < 1:
        raise ValueError(f'tc_signal: upsample_nearest1d: scale must be an integer >= 1, got {s}')
    xc, out_shape = _C.upsample_nearest1d_plan(x, s)
    if out is None:
        out = _alloc(out_shape)
    _C.upsample_nearest1d_run(xc, out, s)
    return out


def reflection_pad1d(x, pads, *, out=None):
    """ONNX Pad mode=reflect (torch reflection_pad1d) on the last axis: pads = (begin, end), each >= 0 and < T.
    Output length T + begin + end. Returns `out`."""
    if not isinstance(pads, (tuple, list)):
        raise TypeError(f'tc_signal: reflection_pad1d: pads must be a (begin, end) tuple or list, got '
                        f'{type(pads).__name__}')
    if len(pads) != 2:
        raise ValueError(f'tc_signal: reflection_pad1d: pads must be (begin, end), got {pads!r}')
    pb, pe = _int('pads[0]', pads[0]), _int('pads[1]', pads[1])
    if pb < 0 or pe < 0:
        raise ValueError(f'tc_signal: reflection_pad1d: pads must be >= 0, got ({pb}, {pe})')
    xc, out_shape = _C.reflection_pad1d_plan(x, pb, pe)
    if out is None:
        out = _alloc(out_shape)
    _C.reflection_pad1d_run(xc, out, pb, pe)
    return out


def rint(x, *, out=None):
    """ONNX Round: round half to even (rintf), sign of zero kept (rint(-0.4) = -0.0), +-Inf and NaN pass through.
    Any rank. Returns `out`."""
    xc, out_shape = _C.rint_plan(x)
    if out is None:
        out = _alloc(out_shape)
    _C.rint_run(xc, out)
    return out
