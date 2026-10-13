"""tc_signal.lstm: ONNX-semantics LSTM (P1T-M3, T4), forward only, fp32, deterministic.

    lstm(x, W, R, B=None, *, hidden_size, direction="forward", initial_h=None, initial_c=None,
         sequence_lens=None, layout=0, out=None) -> (Y, Y_h, Y_c)

Semantics: ONNX LSTM, layout 0, default activations (Sigmoid, Tanh, Tanh), no clip, no peepholes, input_forget 0.
 * x [T, N, I]; W [D, 4H, I]; R [D, 4H, H]; B [D, 8H] = [Wb; Rb] or None (zeros); initial_h / initial_c [D, N, H] or
   None (zeros). Gate blocks along 4H are in the ONNX order i, o, f, c (torch's nn.LSTM uses i, f, g, o).
 * direction "forward" or "reverse" (D = 1) or "bidirectional" (D = 2: index 0 forward, index 1 reverse).
 * Returns Y [T, D, N, H], Y_h [D, N, H], Y_c [D, N, H]: new tensors from tc_signal's allocator (set_allocator; the
   default is tensor_cuda.empty), or the caller's `out=(Y, Y_h, Y_c)`, which is returned as given.
 * Every operand is any fp32 CUDA DLPack producer (TensorCUDA or torch tensors) or a DLPack capsule.

Refused (never silently approximated):
 * layout != 0 (ValueError); sequence_lens other than None or "every entry equals T" (ValueError; a non-integer
   sequence_lens is a TypeError). sequence_lens is only validated, never read by the kernel; a device tensor is copied
   to the host for that check (TensorCUDA `.numpy()`, torch `.cpu()`);
 * clip, peepholes (ONNX input P), input_forget and non-default activations: the signature has no such parameter, so
   passing one raises TypeError;
 * every shape mismatch, non-fp32 / non-CUDA / non-contiguous / misaligned operands, mismatched devices, an `out` that
   is not a 3-sequence, and any output or scratch aliasing another operand (M2's operand contract, dlpack_operand.h).

Execution: one `_tc_signal.lstm_plan` call peeks the inputs; this layer allocates Y, Y_h, Y_c and ONE scratch operand
(projected inputs G_x [D, T, N, 4H] then the per-step gate buffer G_h [D, N, 4H]) through the allocator; one
`_tc_signal.lstm_run` call then runs the whole recurrence in C++ (src/lstm.cu): one input-projection SGEMM per
direction (+ Wb), then a host C++ loop over the T steps, each one strided-batched SGEMM over the directions
(h_{t-1} . R^T) and one fused cell kernel. No Python loop over steps, no per-step DLPack capsules. Everything is queued
on the legacy default stream; the results are ready in stream order (a later TensorCUDA op or a `.numpy()` download
on the legacy stream sees them).

Prior art: the ONNX LSTM operator spec (semantics); Hochreiter & Schmidhuber 1997 (LSTM) and Gers, Schmidhuber &
Cummins 2000 (forget gate); Appleyard, Kocisky & Blunsom 2016 / the cuDNN RNN design (the input projection hoisted out
of the recurrence); the sign-split stable sigmoid (standard practice). Details and what is ours: src/lstm.cu.
The Python plan/allocate/run flow is M2's (tc_signal/__init__.py conv1d); no new prior art here.
"""
from __future__ import annotations

import operator
import sys

from . import _tc_signal as _C

# ONNX direction attribute -> the C++ direction code (lstm_bindings.cpp).
_DIRECTIONS = {'forward': 0, 'reverse': 1, 'bidirectional': 2}
# ONNX `layout` attribute: 0 = x [T, N, I] (the only supported layout).
_LAYOUT_SEQ_FIRST = 0
# `out` is (Y, Y_h, Y_c).
_N_OUTPUTS = 3


def _package():
    """The tc_signal package module, read at call time so set_allocator's current allocator is used."""
    return sys.modules[__name__.rpartition('.')[0]]


def _int(name, v):
    if isinstance(v, bool):
        raise TypeError(f'tc_signal.lstm: {name} must be an int, got bool')
    try:
        return operator.index(v)
    except TypeError:
        raise TypeError(f'tc_signal.lstm: {name} must be an int, got {type(v).__name__}') from None


def _host_array(seq):
    """sequence_lens as a host NumPy array: lists, tuples and NumPy arrays directly; TensorCUDA tensors through
    `.numpy()`; torch tensors through `.cpu().numpy()`."""
    import numpy as np
    if isinstance(seq, (list, tuple, np.ndarray)) or np.isscalar(seq):
        return np.asarray(seq)
    cpu = getattr(seq, 'cpu', None)
    if callable(cpu):                         # torch (CPU or CUDA)
        seq = cpu()
    to_numpy = getattr(seq, 'numpy', None)
    if callable(to_numpy):                    # TensorCUDA, torch CPU
        return np.asarray(to_numpy())
    return np.asarray(seq)


def _check_sequence_lens(sequence_lens, T, N):
    import numpy as np
    arr = _host_array(sequence_lens)
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f'tc_signal.lstm: sequence_lens must hold integers, got dtype {arr.dtype}')
    if arr.shape != (N,):
        raise ValueError(f'tc_signal.lstm: sequence_lens has shape {tuple(arr.shape)}, expected ({N},) = [batch]')
    if not bool(np.all(arr == T)):
        raise ValueError(f'tc_signal.lstm: sequence_lens {arr.tolist()} is not supported (refused): only None or every '
                         f'entry equal to T = {T} is implemented')


def lstm(x, W, R, B=None, *, hidden_size, direction='forward', initial_h=None, initial_c=None, sequence_lens=None,
         layout=0, out=None):
    """ONNX LSTM (layout 0, gates i, o, f, c; Sigmoid/Tanh/Tanh; no clip, no peepholes). See the module docstring.

    x [T, N, I], W [D, 4H, I], R [D, 4H, H], B [D, 8H] = [Wb; Rb] or None, initial_h / initial_c [D, N, H] or None.
    Returns (Y [T, D, N, H], Y_h [D, N, H], Y_c [D, N, H])."""
    H = _int('hidden_size', hidden_size)
    if not isinstance(direction, str) or direction not in _DIRECTIONS:
        raise ValueError(f'tc_signal.lstm: direction must be one of {sorted(_DIRECTIONS)}, got {direction!r}')
    lay = _int('layout', layout)
    if lay != _LAYOUT_SEQ_FIRST:
        raise ValueError(f'tc_signal.lstm: layout={lay} is not supported (refused); only ONNX layout 0 '
                         f'(x [T, N, I]) is implemented')
    if out is not None:
        if not isinstance(out, (tuple, list)):
            raise TypeError(f'tc_signal.lstm: out must be None or a 3-tuple (Y, Y_h, Y_c), got {type(out).__name__}')
        if len(out) != _N_OUTPUTS:
            raise ValueError(f'tc_signal.lstm: out must be a 3-tuple (Y, Y_h, Y_c), got {len(out)} entries')
    code = _DIRECTIONS[direction]
    (xc, wc, rc, bc, hc, cc, y_shape, state_shape, scratch_numel, T, N) = _C.lstm_plan(
        x, W, R, B, initial_h, initial_c, H, code)
    if sequence_lens is not None:
        _check_sequence_lens(sequence_lens, T, N)
    alloc = _package()._allocator
    if out is None:
        Y, Y_h, Y_c = alloc(tuple(y_shape)), alloc(tuple(state_shape)), alloc(tuple(state_shape))
    else:
        Y, Y_h, Y_c = out
    scratch = alloc((scratch_numel,))
    _C.lstm_run(xc, wc, rc, bc, hc, cc, Y, Y_h, Y_c, scratch, H, code)
    return Y, Y_h, Y_c
