"""tc_signal T5 norms (P1T-M4): instance_norm1d, layer_norm, group_norm.

fp32 statistics (shifted two-pass: shift by the row's first sample, mean of the shifted row, then the centred sum of
squares; biased variance as ONNX), a fixed block-tree reduction order (deterministic, no atomics), and
y = (x - mean) * (1 / sqrtf(var + eps)) * scale + bias. Kernel and numerics notes: src/norms.cu. Operands are any fp32
CUDA DLPack producers (or capsules), C-contiguous; the output comes from the module allocator (tc_signal.set_allocator)
unless `out` is given; everything is queued on the legacy default stream.

Prior art: Ulyanov, Vedaldi & Lempitsky (2016) instance norm; Ba, Kiros & Hinton (2016) layer norm; Wu & He (2018)
group norm; ONNX InstanceNormalization / LayerNormalization / GroupNormalization-21 (semantics); Welford (1962) and
Chan, Golub & LeVeque (1983) (variance algorithms, two-pass and shifted data). Ours: the Python wrappers.
"""
from __future__ import annotations

import math
import numbers
import operator
import sys

from . import _tc_signal as _C

__all__ = ['instance_norm1d', 'layer_norm', 'group_norm']


def _alloc(shape):
    # The package's current allocator (tc_signal.set_allocator), read at call time.
    return sys.modules[__package__]._allocator(tuple(shape))


def _eps(eps):
    if isinstance(eps, bool) or not isinstance(eps, numbers.Real):
        raise TypeError(f'tc_signal: eps must be a real number, got {type(eps).__name__}')
    e = float(eps)
    if not math.isfinite(e) or e < 0.0:
        raise ValueError(f'tc_signal: eps must be a finite number >= 0, got {e!r}')
    return e


def _int(name, v):
    if isinstance(v, bool):
        raise TypeError(f'tc_signal: {name} must be an int, got bool')
    try:
        return operator.index(v)
    except TypeError:
        raise TypeError(f'tc_signal: {name} must be an int, got {type(v).__name__}') from None


def instance_norm1d(x, scale=None, bias=None, *, eps=1e-5, out=None):
    """ONNX InstanceNormalization on x [N, C, T]: per (n, c), mean and biased variance over T (fp32 statistics);
    y = (x - mean) / sqrt(var + eps) * scale[c] + bias[c]. scale, bias: [C] or None (1 and 0). Returns `out`."""
    e = _eps(eps)
    xc, sc, bc, out_shape = _C.instance_norm1d_plan(x, scale, bias, e)
    if out is None:
        out = _alloc(out_shape)
    _C.instance_norm1d_run(xc, sc, bc, out, e)
    return out


def layer_norm(x, scale=None, bias=None, *, axis=-1, eps=1e-5, out=None):
    """ONNX LayerNormalization: normalize over the axes [axis, rank) with the biased variance (fp32 statistics, whatever
    eps: 1e-5 and 1e-12 both honoured); y = xhat * scale + bias with scale, bias of shape x.shape[axis:] or None.
    Returns `out`."""
    a = _int('axis', axis)
    e = _eps(eps)
    xc, sc, bc, out_shape = _C.layer_norm_plan(x, scale, bias, a, e)
    if out is None:
        out = _alloc(out_shape)
    _C.layer_norm_run(xc, sc, bc, out, a, e)
    return out


def group_norm(x, num_groups, scale=None, bias=None, *, eps=1e-5, out=None):
    """GroupNorm (ONNX GroupNormalization-21 / torch.nn.GroupNorm) on x [N, C, T]: mean and biased variance over each
    group's (C/num_groups, T) block (fp32 statistics); per-CHANNEL affine: scale, bias [C] or None. Returns `out`."""
    g = _int('num_groups', num_groups)
    e = _eps(eps)
    xc, sc, bc, out_shape = _C.group_norm_plan(x, g, scale, bias, e)
    if out is None:
        out = _alloc(out_shape)
    _C.group_norm_run(xc, g, sc, bc, out, e)
    return out
