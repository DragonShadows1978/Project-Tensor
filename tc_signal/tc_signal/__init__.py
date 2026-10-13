"""tc_signal: TensorCUDA add-on for signal ops (P1T-M2: Conv1d T2, ConvTranspose1d T3).

A separate package next to `tensor_cuda/` (David, 2026-10-12: keep core free of these kernels). It consumes any DLPack
CUDA producer (TensorCUDA tensors once M1 lands, torch tensors for testing), computes in fp32 on the legacy default
stream, deterministically (no atomics, no fast-math, cuBLAS in default math mode), and allocates outputs and scratch
through the module allocator (`set_allocator`). Usage: `PYTHONPATH=<Project-Tensor>/tc_signal`, then
`import tc_signal`.

API (exact, order P1T-M2):
    conv1d(x, w, b=None, *, stride=1, padding=0, dilation=1, groups=1, out=None)
    conv_transpose1d(x, w, b=None, *, stride=1, padding=0, output_padding=0, dilation=1, groups=1, out=None)
    set_allocator(fn)   # fn(shape: tuple) -> fp32 CUDA DLPack tensor; returns the previous allocator

Semantics are ONNX Conv / ConvTranspose (1-D): x [N, C_in, T]; conv1d w [C_out, C_in/groups, K]; conv_transpose1d
w [C_in, C_out/groups, K]; b [C_out] or None; padding an int or (pad_begin, pad_end).

Prior art: ONNX Conv/ConvTranspose operator specs (semantics); DLPack (dmlc/dlpack v1.3) and the Python array API
data-interchange protocol (operand exchange); im2col (Chellapilla, Puri & Simard 2006) and gather-form transposed
convolution (Dumoulin & Visin 2016) for the kernels (see src/conv.cu). The pluggable allocator callback mirrors CuPy's
`cupy.cuda.set_allocator` (CuPy, 2017-), which also returns control of allocation to the caller; ours differs in that
the allocator returns a DLPack tensor, not a raw pointer.
"""
from __future__ import annotations

import operator

from . import _tc_signal as _C

__all__ = ['conv1d', 'conv_transpose1d', 'set_allocator']

_SET_ALLOCATOR_HINT = ('call tc_signal.set_allocator(fn), where fn(shape: tuple) returns a float32 CUDA tensor that '
                       'implements __dlpack__ (for example lambda s: torch.empty(s, device="cuda"))')


def _default_allocator(shape):
    """`tensor_cuda.empty(*shape, dtype="float32")` (added by P1T-M1). Raises RuntimeError naming set_allocator when
    tensor_cuda is not importable, lacks `empty`, or its tensors do not export DLPack yet."""
    try:
        import tensor_cuda
    except ImportError as e:
        raise RuntimeError(f'tc_signal: the default allocator needs tensor_cuda.empty, but tensor_cuda is not '
                           f'importable ({e}); {_SET_ALLOCATOR_HINT}') from e
    empty = getattr(tensor_cuda, 'empty', None)
    if empty is None:
        raise RuntimeError('tc_signal: the default allocator needs tensor_cuda.empty (P1T-M1), which this '
                           f'tensor_cuda does not have; {_SET_ALLOCATOR_HINT}')
    t = empty(*shape, dtype='float32')
    if not hasattr(t, '__dlpack__'):
        raise RuntimeError('tc_signal: tensor_cuda.empty returned a tensor without __dlpack__ (P1T-M1 DLPack export '
                           f'missing); {_SET_ALLOCATOR_HINT}')
    return t


_allocator = _default_allocator


def set_allocator(fn):
    """Set the allocator used for outputs (out=None) and im2col scratch: fn(shape: tuple) -> fp32 CUDA DLPack tensor of
    exactly that shape. Returns the previous allocator. `None` restores the default (tensor_cuda.empty)."""
    global _allocator
    if fn is not None and not callable(fn):
        raise TypeError(f'tc_signal.set_allocator: expected a callable or None, got {type(fn).__name__}')
    prev = _allocator
    _allocator = _default_allocator if fn is None else fn
    return prev


def _int(name, v):
    if isinstance(v, bool):
        raise TypeError(f'tc_signal: {name} must be an int, got bool')
    try:
        return operator.index(v)
    except TypeError:
        raise TypeError(f'tc_signal: {name} must be an int, got {type(v).__name__}') from None


def _pads(padding):
    if isinstance(padding, (tuple, list)):
        if len(padding) != 2:
            raise ValueError(f'tc_signal: padding must be an int or (pad_begin, pad_end), got {padding!r}')
        return _int('padding[0]', padding[0]), _int('padding[1]', padding[1])
    p = _int('padding', padding)
    return p, p


def conv1d(x, w, b=None, *, stride=1, padding=0, dilation=1, groups=1, out=None):
    """ONNX Conv, 1-D: y[n, o, t] = b[o] + sum_{c in group(o), k} w[o, c, k] * x_pad[n, c, t*stride + k*dilation].

    x [N, C_in, T], w [C_out, C_in/groups, K], b [C_out] or None (fp32 CUDA DLPack producers or capsules).
    Output [N, C_out, (T + pb + pe - dilation*(K-1) - 1)//stride + 1], written into `out` if given, else allocated
    through the module allocator. Returns `out`."""
    pb, pe = _pads(padding)
    s, d, g = _int('stride', stride), _int('dilation', dilation), _int('groups', groups)
    xc, wc, bc, out_shape, scratch_numel = _C.conv1d_plan(x, w, b, s, pb, pe, d, g)
    if out is None:
        out = _allocator(tuple(out_shape))
    scratch = _allocator((scratch_numel,)) if scratch_numel > 0 else None
    _C.conv1d_run(xc, wc, bc, out, scratch, s, pb, pe, d, g)
    return out


def conv_transpose1d(x, w, b=None, *, stride=1, padding=0, output_padding=0, dilation=1, groups=1, out=None):
    """ONNX ConvTranspose, 1-D (gather form, deterministic): x [N, C_in, T], w [C_in, C_out/groups, K] (ONNX layout),
    b [C_out] or None. Output [N, C_out, stride*(T-1) + output_padding + dilation*(K-1) + 1 - pb - pe], written into
    `out` if given, else allocated through the module allocator. Returns `out`."""
    pb, pe = _pads(padding)
    s, op, d, g = (_int('stride', stride), _int('output_padding', output_padding), _int('dilation', dilation),
                   _int('groups', groups))
    xc, wc, bc, out_shape = _C.conv_transpose1d_plan(x, w, b, s, pb, pe, op, d, g)
    if out is None:
        out = _allocator(tuple(out_shape))
    _C.conv_transpose1d_run(xc, wc, bc, out, s, pb, pe, op, d, g)
    return out


from ._norms import group_norm, instance_norm1d, layer_norm  # noqa: E402  (P1T-M4, T5)
from ._small import reflection_pad1d, rint, upsample_nearest1d  # noqa: E402  (P1T-M4, T6)
__all__ += ['instance_norm1d', 'layer_norm', 'group_norm', 'upsample_nearest1d', 'reflection_pad1d', 'rint']
