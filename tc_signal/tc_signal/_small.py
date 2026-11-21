"""tc_signal T6 small ops (P1T-M4): upsample_nearest1d, reflection_pad1d, rint. Bit-exact by construction (32-bit word
gathers; rintf). Kernel notes: src/small_ops.cu. Operands are any fp32 CUDA DLPack producers (or capsules),
C-contiguous; the time axis is the last axis; untracked outputs come from the module allocator unless `out` is given;
everything is queued on the legacy default stream. TensorCUDA core's `round` is untouched.

Prior art: ONNX Resize (nearest, asymmetric, floor), Pad (reflect) and Round specs; IEEE 754 roundTiesToEven / C99 rint;
torch reflection_pad1d. Ours: the Python wrappers.
"""
from __future__ import annotations

import math
import operator
import sys

from . import _tc_signal as _C
from . import _recording, _refuse_no_backward

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
    y[..., i] = x[..., i // scale], output length T * scale. Non-integer scales are refused (TypeError).
    Untracked calls return the supplied `out` or a tensor from the module allocator.

    Gradients. With grad mode enabled and a TensorCUDA input requiring grad, the output is a fresh TensorCUDA
    tensor that bypasses `set_allocator`. The tracked path requires TensorCUDA with BWD-R1's `attach_vjp`
    (d4285023 or a build carrying that hook), refuses `out=` (ValueError), and requires TensorCUDA operands.
    """
    s = _int('scale (upsample_nearest1d takes integer scales only)', scale)
    if s < 1:
        raise ValueError(f'tc_signal: upsample_nearest1d: scale must be an integer >= 1, got {s}')
    name = 'upsample_nearest1d'
    tc = _recording(name, out, x)
    if tc is not None:
        xc, out_shape = _C.upsample_nearest1d_plan(x, s)
        value = tc.empty(*out_shape, dtype='float32', device=x.device)
        _C.upsample_nearest1d_run(xc, value, s)
        value.mark_modified()
        shape = tuple(x.shape)
        count = math.prod(shape)

        def vjp(dy, needs_input_grad):
            if not needs_input_grad[0]:
                return (None,)
            if s == 1:
                return (dy,)
            # ONNX Resize (2018) and PyTorch nearest backward (2019).
            # The singleton trailing dimension selects core's serial
            # non-trailing reduction, summing repeats in ascending order.
            return (dy.reshape([count, s, 1]).sum([1]).reshape(shape),)

        return tc.attach_vjp(value, (x,), vjp, name=name)
    xc, out_shape = _C.upsample_nearest1d_plan(x, s)
    if out is None:
        out = _alloc(out_shape)
    _C.upsample_nearest1d_run(xc, out, s)
    return out


def _add_reflection(tc, dx, start, reflection):
    # Update only the interval touched by this reflection, preserving direct
    # copies (including signed zero) outside the interval.
    length = reflection.shape[-1]
    return tc.cat([
        dx.slice(-1, 0, start),
        dx.slice(-1, start, length) + reflection,
        dx.slice(-1, start + length, dx.shape[-1] - start - length),
    ], dim=-1)


def reflection_pad1d(x, pads, *, out=None):
    """ONNX Pad mode=reflect (torch reflection_pad1d) on the last axis: pads = (begin, end), each >= 0 and < T.
    Output length T + begin + end. Untracked calls return the supplied `out` or a tensor from the module allocator.

    Gradients. With grad mode enabled and a TensorCUDA input requiring grad, the output is a fresh TensorCUDA
    tensor that bypasses `set_allocator`. The tracked path requires TensorCUDA with BWD-R1's `attach_vjp`
    (d4285023 or a build carrying that hook), refuses `out=` (ValueError), and requires TensorCUDA operands.
    """
    if not isinstance(pads, (tuple, list)):
        raise TypeError(f'tc_signal: reflection_pad1d: pads must be a (begin, end) tuple or list, got '
                        f'{type(pads).__name__}')
    if len(pads) != 2:
        raise ValueError(f'tc_signal: reflection_pad1d: pads must be (begin, end), got {pads!r}')
    pb, pe = _int('pads[0]', pads[0]), _int('pads[1]', pads[1])
    if pb < 0 or pe < 0:
        raise ValueError(f'tc_signal: reflection_pad1d: pads must be >= 0, got ({pb}, {pe})')
    name = 'reflection_pad1d'
    tc = _recording(name, out, x)
    if tc is not None:
        xc, out_shape = _C.reflection_pad1d_plan(x, pb, pe)
        value = tc.empty(*out_shape, dtype='float32', device=x.device)
        _C.reflection_pad1d_run(xc, value, pb, pe)
        value.mark_modified()
        length = x.shape[-1]

        def vjp(dy, needs_input_grad):
            if not needs_input_grad[0]:
                return (None,)
            # ONNX Pad reflect (2017) and PyTorch reflection_pad1d backward
            # (2019) give the transpose of the sampling map. Here core
            # slices and flips gather in direct, left, then right order.
            dx = dy.slice(-1, pb, length) if pb or pe else dy
            if pb:
                dx = _add_reflection(tc, dx, 1, dy.slice(-1, 0, pb).flip([-1]))
            if pe:
                dx = _add_reflection(tc, dx, length - 1 - pe, dy.slice(-1, pb + length, pe).flip([-1]))
            return (dx,)

        return tc.attach_vjp(value, (x,), vjp, name=name)
    xc, out_shape = _C.reflection_pad1d_plan(x, pb, pe)
    if out is None:
        out = _alloc(out_shape)
    _C.reflection_pad1d_run(xc, out, pb, pe)
    return out


def rint(x, *, out=None):
    """ONNX Round: round half to even (rintf), sign of zero kept (rint(-0.4) = -0.0), +-Inf and NaN pass through.
    Any rank. Returns `out`."""
    _refuse_no_backward("rint", x, out=out)
    xc, out_shape = _C.rint_plan(x)
    if out is None:
        out = _alloc(out_shape)
    _C.rint_run(xc, out)
    return out
