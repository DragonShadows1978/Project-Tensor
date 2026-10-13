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

import atexit
import operator
import sys

from . import _tc_signal as _C

__all__ = ['conv1d', 'conv_transpose1d', 'set_allocator', 'synchronize']

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


def synchronize():
    """Wait until the legacy default stream has passed every tc_signal launch queued so far, then release every consumed
    operand tc_signal still holds (P1T-FIX1, F01 deferred release; see README "Streams and lifetimes").

    tc_signal keeps each call's consumed DLPack managers (inputs, outputs, scratch) alive until the legacy stream has
    passed that call, and releases completed ones at the start of the next tc_signal call. Call synchronize() to release
    them now (for example before measuring memory, or before handing a producer's memory back by other means). It
    synchronizes the legacy stream of the current device and of every device with pending operands (stream syncs, not a
    device-wide sync).

    Errors (P1T-FIX2, N1): a CUDA failure is never reported as success. RuntimeError, naming the CUDA error, when
    - the current-device query (cudaGetDevice) fails, for example on a machine with no usable CUDA device, even when
      nothing is pending;
    - a device's legacy stream cannot be synchronized (its pending operands are then deliberately leaked, never
      released early);
    - an earlier release that could not raise failed (the fallback sync or the bound drain inside a tc_signal call); such
      a failure is raised once, by the next synchronize().
    MemoryError if the error message itself cannot be allocated. Prior art: torch.cuda.synchronize /
    cupy.cuda.Stream.synchronize (API shape only).

    P1T-FIX5: (V5-02) A failed completion-event query in an earlier call's drain, and a failed completion-event
    record, are among the earlier failures raised here, exactly once.

    P1T-FIX6: (V6-01) synchronize() never waits for another thread. FIX5 made it wait for producer releases already
    running on other threads; a caller holding a lock that such a release needed then deadlocked (round-6 review). A
    release that another thread's tc_signal call already started is that call's release and finishes on that thread;
    synchronize() releases what is still pending. Only the interpreter-exit hook waits for other threads (see
    _release_pending_at_exit). (V6-03) If the RuntimeError's message cannot be allocated, MemoryError is raised and the
    CUDA failures it would have named are kept: the next synchronize() raises them, once. Arguments: none (any argument
    raises TypeError without any of its methods being called)."""
    _C.synchronize()


# P1T-FIX5 (round-5 review V5-01) knob: seconds the exit hook waits for producer callbacks still running on other
# threads before its first stderr diagnostic; another line follows each time the time waited doubles. It never bounds
# the wait (a bounded return let CPython finalize under a running deleter and abort: round-3 V3-01 in core, round-5
# V5-01 here). A callback normally takes microseconds. Same knob and schedule as core's _DLPACK_SHUTDOWN_WAIT_S.
_RELEASE_EXIT_DIAG_S = 5.0


def _release_pending_at_exit():
    """atexit hook (runs before CPython marks the runtime finalizing; GIL held).

    1. synchronize(): release whatever is still pending (P1T-FIX6: it does not wait for other threads; step 2 does).
       P1T-FIX2 (N1): an error is never swallowed silently; it is LOGGED to stderr (one line) and not raised (an
       exception from an atexit callback cannot be handled by anyone). Operands without a completion proof stay
       deliberately leaked; there is nothing else safe to do at exit.
    2. P1T-FIX5 (round-5 review V5-01): close the add-on's release admission for good, then wait, GIL released and
       with NO time bound, until no admitted producer callback of another thread is running: a deleter call (immediate
       release of a refused call, next-call drain, synchronize, bound drain, fallback sync), the drop of an unconsumed
       capsule's last reference (its destructor calls the producer's deleter), or a producer __dlpack__ /
       __dlpack_device__ call. Interpreter finalization therefore never starts under a running callback. From here on
       every later deleter call and capsule drop is deliberately leaked (counted in pending_stats()
       'deleters_leaked_after_close' / 'capsules_kept_after_close') and every later producer __dlpack__ call is refused
       (RuntimeError, 'callbacks_refused_after_close'): no add-on code enters Python for them. Callbacks on this
       thread's own stack are excluded (the hook may be called from inside one). After _RELEASE_EXIT_DIAG_S seconds a
       line goes to stderr, then another each time the time waited doubles.
       P1T-FIX6: this is the ONLY place tc_signal waits for another thread, and the wait's contract is: the exiting
       thread must not hold a lock that such an admitted callback needs (the rule CPython applies when it joins
       non-daemon threads at exit; atexit callbacks normally run with no application lock held).
       Consequence: a producer deleter, capsule destructor or __dlpack__ that never returns makes interpreter exit wait
       for it. That is the producer's contract violation (a DLPack deleter must return); waiting is the honest outcome,
       an early return aborts the process (rc -6, "FATAL: exception not rethrown"). Ctrl-C does not interrupt the wait;
       SIGTERM / SIGKILL still end the process.
    Never raises. Prior art: CPython's atexit, which itself prints (not raises) a callback's exception (taken:
    log-don't-raise); core's tensor_cuda._dlpack_shutdown_hook (P1T-FIX2 N5 / P1T-FIX3 V3-01), whose close-then-wait
    design and diagnostic schedule this reuses (see src/dlpack_operand.cpp, release lifecycle)."""
    try:
        _C.synchronize()
    except Exception as e:  # noqa: BLE001
        try:
            sys.stderr.write(f'tc_signal: synchronize() at exit failed ({type(e).__name__}: {e}); operands without a '
                             f'completion proof were deliberately leaked, not released\n')
            sys.stderr.flush()
        except Exception:  # noqa: BLE001  (stderr may already be closed at exit)
            pass
    try:
        try:
            first_diag_s = float(_RELEASE_EXIT_DIAG_S)
        except (TypeError, ValueError):
            first_diag_s = float('nan')  # an unusable knob must not skip the wait: diagnose at once, still wait
        r = _C._exit_close_admission(first_diag_s)
        if r['overran']:
            sys.stderr.write(f'tc_signal: the producer callback(s) admitted before exit finished after '
                             f'{r["waited_us"] / 1e6:.1f} s; interpreter exit continues\n')
            sys.stderr.flush()
    except Exception as e:  # noqa: BLE001
        try:
            sys.stderr.write(f'tc_signal: exit hook failed ({type(e).__name__}: {e})\n')
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass


atexit.register(_release_pending_at_exit)


from ._lstm import lstm  # noqa: E402,F401  (P1T-M3: tc_signal/_lstm.py)
__all__.append('lstm')


from ._norms import group_norm, instance_norm1d, layer_norm  # noqa: E402  (P1T-M4, T5)
from ._small import reflection_pad1d, rint, upsample_nearest1d  # noqa: E402  (P1T-M4, T6)
__all__ += ['instance_norm1d', 'layer_norm', 'group_norm', 'upsample_nearest1d', 'reflection_pad1d', 'rint']
