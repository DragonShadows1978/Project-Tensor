"""Public convolution backward functions (BWD-R4b item 2): dX, dW and dB of tc_signal's conv1d and conv_transpose1d.

    conv1d_backward(x, w, grad_out, *, stride=1, padding=0, dilation=1, groups=1, needs=(True, True, True))
    conv_transpose1d_backward(x, w, grad_out, *, stride=1, padding=0, output_padding=0, dilation=1, groups=1,
                              needs=(True, True, True))

Both return the tuple (dx, dw, db). `needs` is the ordered triple (dx, dw, db) of plain bools; a slot that is not
requested comes back as None. The keyword names and defaults are the forwards' (`conv1d` and `conv_transpose1d` in
tc_signal/__init__.py) minus `b` and `out`, plus `needs`.

Operands (all checked in Python, in this order, before anything reaches the native entry point):
  1. x, w and grad_out are tensor_cuda.Tensor objects (TypeError otherwise). A TensorCUDA tensor always exports compact
     C-contiguous strides (tensor_cuda/src/bindings.cpp, `contiguous_strides` in the DLPack export), so a
     non-contiguous operand can only arrive as a foreign producer, which this check refuses; the native entry point
     re-checks every operand's strides (src/dlpack_operand.cpp) as a second line.
  2. each dtype is float32 (TypeError otherwise).
  3. each is rank 3 with positive extents, and stride, padding, output_padding, dilation and groups satisfy the
     forward plan's rules (stride, dilation, groups >= 1; pads >= 0; output_padding < max(stride, dilation)); the
     channels divide by groups and w's layout matches (conv1d w [C_out, C_in/groups, K]; conv_transpose1d w
     [C_in, C_out/groups, K], ONNX layout); grad_out has exactly the forward output's shape [N, C_out, T_out]
     (ValueError otherwise; TypeError for a non-int attribute).
  4. BWD-R4b r4 (F05): the native run entry's own geometry and integer-limit checks, run without allocating through
     `_conv_backward_scratch` (every attribute in [lo, INT_MAX], fp32 element and byte limits, the exact upstream
     shape); ValueError otherwise. This runs before the all-off return, so an all-False `needs` cannot skip it.
  4b. BWD-R4b r6 (F06): each operand's element-0 address is non-NULL and 4-byte aligned (`Tensor.data_ptr`), the
     native run entry's address rule; ValueError otherwise. Residency is checked by the native run entry.
  5. needs is a length-3 tuple or list of bools (TypeError otherwise).
  6. all three live on one CUDA device and that device is the calling thread's current device (BWD-R4b r4, F10:
     TensorCUDA allocates on the current device only); ValueError otherwise, before any allocation. The caller makes
     the operands' device current. The device check comes last so that every other refusal is reachable, and tested,
     with CUDA hidden.
The native entry point (`_tc_signal._conv_backward_run`, BWD-R3) then runs its own geometry, stride, disjointness
and shape checks unchanged.

Outputs are fresh float32 TensorCUDA tensors on the operands' device, allocated with tensor_cuda.empty exactly as
tc_signal's tracked path allocates its VJP outputs (`_attach_conv` in __init__.py), with the same dW scratch sizing
(`_conv_backward_scratch`). They carry no autograd graph: these functions compute one first-order pullback and record
nothing, whether or not grad mode is on. dB is the bias gradient the forward would have had with a bias (the sum of
grad_out over batch and time); request it only when the forward had a bias.

Determinism: the kernels are BWD-R3's (strict fp32, no atomics, a fixed tile merge order, a private pedantic cuBLAS
handle on the legacy stream), so the same operand bytes in the same call layout (shapes, attributes and `needs`) give
the same output bytes on the same device, binary and driver. Requesting a different subset of slots does not change
the bytes of a requested slot. These functions wrap the private entry point, which stays as it is (Voice's
engine/kokoro_tc/tail_vjp.py calls it today).

Prior art: the convolution adjoints are textbook (Dumoulin and Visin, 2016, "A guide to convolution arithmetic for
deep learning": the adjoint of a strided transposed convolution is the strided convolution and vice versa); the
public-backward-function shape mirrors PyTorch's `torch.nn.grad.conv1d_input` / `conv1d_weight` (Paszke et al., 2019)
and its aten `convolution_backward(grad_output, input, weight, ..., output_mask)` (taken: one call returning the
requested (input, weight, bias) gradients under an output mask). Ours: the TensorCUDA-only operand contract, the check
order above, and the wrapping of BWD-R3's private kernels without a graph. Unverified - lead to check
"torch.ops.aten.convolution_backward output_mask", "torch.nn.grad.conv1d_weight", "Dumoulin Visin 2016".
"""
from __future__ import annotations

import operator
import sys

from . import _tc_signal as _C

__all__ = ['conv1d_backward', 'conv_transpose1d_backward']
# BWD-R4b r4 (F05): the no-allocation geometry preflight is the run entry's own `_conv_backward_scratch`, bound here at
# import. The r3 host tests patch the `_C` attributes to prove the native backward run and the scratch-sizing call are
# not reached before a refusal; the preflight is a separate, required step (proved by the r4 tests, which patch this
# name).
_PREFLIGHT = _C._conv_backward_scratch


def _tensor_module():
    tc = sys.modules.get('tensor_cuda')
    if tc is None:
        try:
            import tensor_cuda as tc
        except ImportError as e:
            raise TypeError(f'tc_signal: the public convolution backward functions need tensor_cuda ({e})') from e
    return tc


def _attr(fn, name, value, low):
    if isinstance(value, bool):
        raise TypeError(f'tc_signal: {fn}: {name} must be an int, got bool')
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f'tc_signal: {fn}: {name} must be an int, got {type(value).__name__}') from None
    if value < low:
        raise ValueError(f'tc_signal: {fn}: {name} must be >= {low}, got {value}')
    return value


def _padding(fn, padding):
    if isinstance(padding, (tuple, list)):
        if len(padding) != 2:
            raise ValueError(f'tc_signal: {fn}: padding must be an int or (pad_begin, pad_end), got {padding!r}')
        return _attr(fn, 'padding[0]', padding[0], 0), _attr(fn, 'padding[1]', padding[1], 0)
    p = _attr(fn, 'padding', padding, 0)
    return p, p


def _needs(fn, needs):
    if not isinstance(needs, (tuple, list)) or len(needs) != 3 or not all(isinstance(v, bool) for v in needs):
        raise TypeError(f'tc_signal: {fn}: needs must be a (dx, dw, db) triple of bools, got {needs!r}')
    return tuple(needs)


def _operands(fn, tc, x, w, grad_out):
    named = (('x', x), ('w', w), ('grad_out', grad_out))
    for name, t in named:
        if not isinstance(t, tc.Tensor):
            raise TypeError(f'tc_signal: {fn}: {name} must be a tensor_cuda.Tensor (C-contiguous by construction), '
                            f'got {type(t).__name__}')
    for name, t in named:
        if t.dtype != 'float32':
            raise TypeError(f'tc_signal: {fn}: {name} must be float32, got {t.dtype}')
    shapes = []
    for name, t in named:
        shape = tuple(int(d) for d in t.shape)
        if len(shape) != 3 or any(d <= 0 for d in shape):
            raise ValueError(f'tc_signal: {fn}: {name} must be rank 3 with positive extents, got {list(shape)}')
        shapes.append(shape)
    return shapes


def _current_cuda_device():
    """The calling thread's current CUDA device (cudaGetDevice through the CUDA 12 runtime), without allocating."""
    import ctypes

    runtime = ctypes.CDLL('libcudart.so.12')
    device = ctypes.c_int(-1)
    rc = runtime.cudaGetDevice(ctypes.byref(device))
    if rc != 0:
        raise RuntimeError(f'tc_signal: the current-device query cudaGetDevice failed (cudaError {rc}); '
                           f'nothing was allocated')
    return device.value


def _check_devices(fn, devices, current):
    """BWD-R4b r4 (F10): TensorCUDA allocates on the current device only (tensor_cuda.empty refuses another index), so
    every operand must be on one CUDA device and that device must be current; checked before any allocation."""
    if not devices[0].startswith('cuda') or len(set(devices)) != 1:
        raise ValueError(f'tc_signal: {fn}: x, w and grad_out must be on one CUDA device, got {devices}')
    if current is not None and devices[0] not in ('cuda', f'cuda:{current}'):
        raise ValueError(f'tc_signal: {fn}: the operands are on {devices[0]} but the current CUDA device is '
                         f'cuda:{current}; make their device current first (TensorCUDA allocates the gradients on the '
                         f'current device only); nothing was allocated')


def _device(fn, x, w, grad_out):
    devices = [str(t.device) for t in (x, w, grad_out)]
    _check_devices(fn, devices, None)
    _check_devices(fn, devices, _current_cuda_device())


def _data_pointer(t):
    return int(t.data_ptr())


def _check_alignment(fn, named_pointers):
    """BWD-R4b r6 (re-check 2 F06): TensorCUDA exposes each operand's element-0 address (`Tensor.data_ptr`), so the
    native run entry's address rule (non-NULL, 4-byte aligned: src/dlpack_operand.cpp) is checked here before any
    allocation and before the all-off return. Residency (that the memory is on the labelled device) is not visible to
    Python: the native run entry checks it through the DLPack export and refuses a mismatch before computing; with
    every gradient switched off nothing is computed."""
    for name, pointer in named_pointers:
        if pointer == 0 or pointer % 4:
            raise ValueError(f'tc_signal: {fn}: {name} element 0 is at address {pointer:#x}, which is NULL or not '
                             f'4-byte aligned; nothing was allocated')


def _native_preflight(fn, transposed, xs, ws, gs, attrs):
    """BWD-R4b r4 (F05): the native geometry and integer-limit checks of the run entry, without allocating:
    `_conv_backward_scratch` runs the run entry's own `backward_geom` (positive rank-3 shapes, fp32 byte and element
    limits, every attribute in [lo, INT_MAX], divisibility, output padding, the exact upstream shape) and returns the
    dW scratch sizes. Called before the all-off return and before any allocation."""
    try:
        return _PREFLIGHT(transposed, list(xs), list(ws), list(gs), attrs)
    except (TypeError, OverflowError) as e:
        raise ValueError(f'tc_signal: {fn}: an attribute or extent exceeds the native int64 range ({e})') from None


def _plan_conv1d(fn, xs, ws, s, pb, pe, d, g):
    n, c_in, t = xs
    c_out, c_group, k = ws
    if c_in % g:
        raise ValueError(f'tc_signal: {fn}: C_in ({c_in}) is not divisible by groups ({g})')
    if c_group != c_in // g:
        raise ValueError(f'tc_signal: {fn}: w.shape[1] ({c_group}) != C_in/groups ({c_in // g}); '
                         f'w is [C_out, C_in/groups, K]')
    if c_out % g:
        raise ValueError(f'tc_signal: {fn}: C_out ({c_out}) is not divisible by groups ({g})')
    num = t + pb + pe - d * (k - 1) - 1
    if num < 0:
        raise ValueError(f'tc_signal: {fn}: output length (T + pb + pe - dilation*(K-1) - 1)//stride + 1 is <= 0')
    return n, c_out, num // s + 1


def _plan_conv_transpose1d(fn, xs, ws, s, pb, pe, op, d, g):
    n, c_in, t = xs
    w_in, c_group, k = ws
    if op >= max(s, d):
        raise ValueError(f'tc_signal: {fn}: output_padding ({op}) must be < max(stride, dilation) ({max(s, d)})')
    if w_in != c_in:
        raise ValueError(f'tc_signal: {fn}: w.shape[0] ({w_in}) != C_in ({c_in}); w is [C_in, C_out/groups, K]')
    if c_in % g:
        raise ValueError(f'tc_signal: {fn}: C_in ({c_in}) is not divisible by groups ({g})')
    t_out = s * (t - 1) + op + d * (k - 1) + 1 - pb - pe
    if t_out <= 0:
        raise ValueError(f'tc_signal: {fn}: output length stride*(T-1) + output_padding + dilation*(K-1) + 1 - pb - pe '
                         f'= {t_out} is <= 0')
    return n, c_group * g, t_out


def _backward(fn, transposed, x, w, grad_out, stride, padding, output_padding, dilation, groups, needs):
    tc = _tensor_module()
    xs, ws, gs = _operands(fn, tc, x, w, grad_out)
    s = _attr(fn, 'stride', stride, 1)
    pb, pe = _padding(fn, padding)
    op = _attr(fn, 'output_padding', output_padding, 0)
    d = _attr(fn, 'dilation', dilation, 1)
    g = _attr(fn, 'groups', groups, 1)
    expected = (_plan_conv_transpose1d(fn, xs, ws, s, pb, pe, op, d, g) if transposed
                else _plan_conv1d(fn, xs, ws, s, pb, pe, d, g))
    if gs != expected:
        raise ValueError(f'tc_signal: {fn}: grad_out has shape {list(gs)}, expected the forward output '
                         f'{list(expected)}')
    attrs = (s, pb, pe, op, d, g)
    col, part = _native_preflight(fn, transposed, xs, ws, gs, attrs)
    _check_alignment(fn, [(name, _data_pointer(t)) for name, t in (('x', x), ('w', w), ('grad_out', grad_out))])
    need_x, need_w, need_b = _needs(fn, needs)
    _device(fn, x, w, grad_out)
    if not (need_x or need_w or need_b):
        return None, None, None
    device = x.device
    dx = tc.empty(*xs, dtype='float32', device=device) if need_x else None
    dw = tc.empty(*ws, dtype='float32', device=device) if need_w else None
    db = tc.empty(gs[1], dtype='float32', device=device) if need_b else None
    if not need_w:
        col, part = 0, 0
    scratch = [tc.empty(size, dtype='float32', device=device) if size else None for size in (col, part)]
    _C._conv_backward_run(transposed, x, w, grad_out, dx, dw, db, *scratch, attrs)
    for t in (dx, dw, db):
        if t is not None:
            t.mark_modified()
    return dx, dw, db


def conv1d_backward(x, w, grad_out, *, stride=1, padding=0, dilation=1, groups=1, needs=(True, True, True)):
    """(dx, dw, db) of y = conv1d(x, w, b, stride=, padding=, dilation=, groups=) for the upstream grad_out [N, C_out,
    T_out]. x [N, C_in, T], w [C_out, C_in/groups, K]. Slots not requested by `needs` are None. See the module
    docstring for the operand checks, allocation and determinism."""
    return _backward('conv1d_backward', False, x, w, grad_out, stride, padding, 0, dilation, groups, needs)


def conv_transpose1d_backward(x, w, grad_out, *, stride=1, padding=0, output_padding=0, dilation=1, groups=1,
                              needs=(True, True, True)):
    """(dx, dw, db) of y = conv_transpose1d(x, w, b, stride=, padding=, output_padding=, dilation=, groups=) for the
    upstream grad_out [N, C_out, T_out]. x [N, C_in, T], w [C_in, C_out/groups, K] (ONNX layout). Slots not requested by
    `needs` are None. See the module docstring for the operand checks, allocation and determinism."""
    return _backward('conv_transpose1d_backward', True, x, w, grad_out, stride, padding, output_padding, dilation,
                     groups, needs)
