"""tc_signal T5 norms (P1T-M4): instance_norm1d, layer_norm, group_norm.

fp32 statistics (shifted two-pass: shift by the row's first sample, mean of the shifted row, then the centred sum of
squares; biased variance as ONNX), a fixed block-tree reduction order (deterministic, no atomics), and
y = (x - mean) * (1 / sqrtf(var + eps)) * scale + bias. Kernel and numerics notes: src/norms.cu. Operands are any fp32
CUDA DLPack producers (or capsules), C-contiguous; untracked outputs come from the module allocator (tc_signal.set_allocator)
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
from . import _recording, _refuse_no_backward

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
    y = (x - mean) / sqrt(var + eps) * scale[c] + bias[c]. scale, bias: [C] or None (1 and 0).
    Untracked calls return the supplied `out` or a tensor from the module allocator.

    Gradients. The tracked path is selected when grad mode is enabled and any TensorCUDA operand requires grad.
    It returns a fresh TensorCUDA tensor that bypasses `set_allocator` and differentiates the requested
    slots among x, scale and bias. The affine VJP sums dy times the replayed normalized input (scale),
    or dy (bias), over time and then batch. The tracked path requires TensorCUDA with BWD-R1's
    `attach_vjp` (d4285023 or a build carrying that hook), and refuses `out=` (ValueError) and foreign
    tensor operands (TypeError).
    """
    e = _eps(eps)
    name = 'instance_norm1d'
    tc = _recording(name, out, x, scale, bias)
    if tc is not None:
        affine = tuple(p for p in (scale, bias) if p is not None)
        xc, sc, bc, out_shape = _C.instance_norm1d_plan(x, scale, bias, e)
        value = tc.empty(*out_shape, dtype='float32', device=x.device)
        _C.instance_norm1d_run(xc, sc, bc, value, e)
        value.mark_modified()

        def vjp(dy, needs_input_grad):
            needs_x = needs_input_grad[0]
            needs_scale = scale is not None and needs_input_grad[1]
            needs_bias = bias is not None and needs_input_grad[-1]
            dx = dscale = None
            # Ulyanov et al. (2016) instance normalization and Ioffe &
            # Szegedy (2015) normalization and affine chain rule. Here the
            # affine slots share the sidecar's shifted-statistics replay
            # with dx, and reduce time then batch using ordered core sums.
            # Unverified — lead to check: instance normalization 2016,
            # batch normalization 2015 affine scale bias gradients.
            # Last-use release follows tensor liveness (Chen et al., 2016,
            # training with sublinear memory). Here we drop Python references
            # after use while retaining the same fp32 expression order.
            if needs_x or needs_scale:
                shifted = x - x.slice(-1, 0, 1)
                centered = shifted - shifted.mean([-1], keepdim=True)
                del shifted
                rstd = ((centered * centered).mean([-1], keepdim=True) + e).sqrt().reciprocal()
                z = centered * rstd
                del centered
                if needs_scale:
                    dscale = (dy * z).sum([-1]).sum([0])
                if needs_x:
                    h = dy if scale is None else dy * scale.reshape([1, x.shape[1], 1])
                    centered_h = h - h.mean([-1], keepdim=True)
                    mean_hz = (h * z).mean([-1], keepdim=True)
                    del h
                    correction = z * mean_hz
                    del z, mean_hz
                    dx = rstd * (centered_h - correction)
            dbias = dy.sum([-1]).sum([0]) if needs_bias else None
            grads = [dx]
            if scale is not None:
                grads.append(dscale)
            if bias is not None:
                grads.append(dbias)
            return tuple(grads)

        return tc.attach_vjp(value, (x, *affine), vjp, name=name)
    xc, sc, bc, out_shape = _C.instance_norm1d_plan(x, scale, bias, e)
    if out is None:
        out = _alloc(out_shape)
    _C.instance_norm1d_run(xc, sc, bc, out, e)
    return out


def layer_norm(x, scale=None, bias=None, *, axis=-1, eps=1e-5, out=None):
    """ONNX LayerNormalization: normalize over the axes [axis, rank) with the biased variance (fp32 statistics, whatever
    eps: 1e-5 and 1e-12 both honoured); y = xhat * scale + bias with scale, bias of shape x.shape[axis:] or None.
    Returns `out`."""
    _refuse_no_backward("layer_norm", x, scale, bias, out=out)
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
    _refuse_no_backward("group_norm", x, scale, bias, out=out)
    g = _int('num_groups', num_groups)
    e = _eps(eps)
    xc, sc, bc, out_shape = _C.group_norm_plan(x, g, scale, bias, e)
    if out is None:
        out = _alloc(out_shape)
    _C.group_norm_run(xc, g, sc, bc, out, e)
    return out
