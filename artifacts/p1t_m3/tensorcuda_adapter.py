"""TensorCUDA adapter for KV-P1T-TESTS, M1 + M2 + M3 (T1, T2, T3, T4 real; T5-T6 NotImplementedError).

Provenance: a copy of the integration adapter `artifacts/p1t_int/tensorcuda_adapter.py` (M1 + M2, itself from
P1T-M1's adapter, itself a copy of Project-Voice's stub). P1T-M3 (2026-10-12) implemented `lstm` (T4); every other
function is unchanged. Run it with both packages on PYTHONPATH:

    cd /mnt/ForgeRealm/Project-Voice && PYTHONPATH=<fork>/tensor_cuda/tensor_cuda:<fork>/tensor_cuda:<fork>/tc_signal \
        python3 engine/tensor_requests/p1t_tests/run_p1t.py --adapter <fork>/artifacts/p1t_m3/tensorcuda_adapter.py \
        --items T1,T2,T3,T4 --fixtures artifacts/p1t/fixtures \
        --tolerances engine/tensor_requests/p1t_tests/tolerances_r2.json --out <report.json>

T2/T3 path, per the approved plan (`docs/P1T_TENSOR_ALT_PLAN.md` section 5, Voice LD51):
 * upload with `tensor_cuda.tensor` (host fp32 -> device fp32);
 * weight_storage 'fp16' (D6): numpy rounds w to fp16 (round to nearest even, as the runner's oracle), the fp16 tensor
   is uploaded and expanded on the device by TensorCUDA's `astype('float32')` (fp16 -> fp32 is exact);
 * `tc_signal.conv1d` / `conv_transpose1d` consume the TensorCUDA tensors through DLPack and allocate the output
   with `tensor_cuda.empty` (tc_signal's default allocator), all on the legacy default stream;
 * download with `Tensor.numpy()` (cudaMemcpy D2H on the legacy default stream).
T4 path (P1T-M3): the runner's layout must be 'onnx' (ONNX layout 0, x [T, batch, input]); x, W, R, B, initial_h and
initial_c are uploaded with `tensor_cuda.tensor`; one `tc_signal.lstm` call runs the whole recurrence (both
directions, every step) in C++ and allocates Y, Y_h, Y_c and its scratch with `tensor_cuda.empty`; the three outputs
are downloaded with `.numpy()`.
T5-T6 raise NotImplementedError until M4. T1 is M1's, unchanged.

Prior art: as in the M1 adapter (DLPack, the array API protocol); ONNX Conv/ConvTranspose/LSTM for T2-T4 semantics.
The glue is ours; no other prior art claimed.
"""
from __future__ import annotations

import numpy as np

import tensor_cuda as tc          # the fork's build on PYTHONPATH
import tc_signal as ts           # the fork's add-on on PYTHONPATH

_M4 = 'lands in the tc_signal add-on (M4; P1T_TENSOR_ALT_PLAN.md section 4), not in TensorCUDA core'
# The runner's only LSTM layout value (adapter_api.LSTM_LAYOUT): ONNX layout attribute 0.
_RUNNER_LSTM_LAYOUT = 'onnx'
_TC_SIGNAL_LAYOUT_SEQ_FIRST = 0

COVERAGE = {
    'T1': ('real', 'tc.from_dlpack / Tensor.__dlpack__ / __dlpack_device__ / data_ptr / fill_ (P1T-M1); allocation '
                   'via tc.empty (Amendment 1)'),
    'T2': ('real', 'tc_signal.conv1d on TensorCUDA tensors (P1T-M2), fp16 weights expanded by tc astype (D6)'),
    'T3': ('real', 'tc_signal.conv_transpose1d on TensorCUDA tensors (P1T-M2), fp16 weights expanded by tc astype'),
    'T4': ('real', 'tc_signal.lstm on TensorCUDA tensors (P1T-M3): ONNX LSTM, one call per LSTM, C++ step loop'),
    'T5': ('missing', 'NotImplementedError: instance_norm1d / layer_norm / group_norm ' + _M4),
    'T6': ('missing', 'NotImplementedError: upsample_nearest1d / reflection_pad1d / rint ' + _M4),
}


# ---------------------------------------------------------------------------------------------------- T2
def _up(a):
    """Host fp32 -> TensorCUDA fp32 on the device."""
    return tc.tensor(np.ascontiguousarray(np.asarray(a, dtype=np.float32)), dtype='float32')


def _up_opt(a):
    return None if a is None else _up(a)


def _weights(w, weight_storage):
    """D6: fp16 storage (numpy round-to-nearest-even), expanded exactly to fp32 by TensorCUDA on the device."""
    if weight_storage == 'fp32':
        return _up(w)
    if weight_storage == 'fp16':
        return tc.tensor(np.ascontiguousarray(np.asarray(w, dtype=np.float32).astype(np.float16)),
                         dtype='float16').astype('float32')
    raise ValueError(f'weight_storage {weight_storage!r}')


def _down(t):
    """TensorCUDA -> host fp32 numpy (D2H on the legacy default stream)."""
    return np.ascontiguousarray(t.numpy())


def conv1d(x, w, b, *, stride, pads, dilation, groups, weight_storage):
    xt, wt = _up(x), _weights(w, weight_storage)
    bt = None if b is None else _up(b)
    y = ts.conv1d(xt, wt, bt, stride=int(stride), padding=(int(pads[0]), int(pads[1])), dilation=int(dilation),
                  groups=int(groups))
    return _down(y)


def conv_transpose1d(x, w, b, *, stride, pads, output_padding, dilation, groups, weight_storage):
    xt, wt = _up(x), _weights(w, weight_storage)
    bt = None if b is None else _up(b)
    y = ts.conv_transpose1d(xt, wt, bt, stride=int(stride), padding=(int(pads[0]), int(pads[1])),
                            output_padding=int(output_padding), dilation=int(dilation), groups=int(groups))
    return _down(y)


# ---------------------------------------------------------------------------------------------------- T4
def lstm(x, W, R, B, *, direction, hidden_size, initial_h, initial_c, layout):
    if layout != _RUNNER_LSTM_LAYOUT:
        raise AssertionError(f'T4: the runner passed layout={layout!r}; this adapter implements only '
                             f'{_RUNNER_LSTM_LAYOUT!r} (ONNX layout 0)')
    Y, Y_h, Y_c = ts.lstm(_up(x), _up(W), _up(R), _up_opt(B), hidden_size=int(hidden_size), direction=str(direction),
                          initial_h=_up_opt(initial_h), initial_c=_up_opt(initial_c),
                          layout=_TC_SIGNAL_LAYOUT_SEQ_FIRST)
    return _down(Y), _down(Y_h), _down(Y_c)


# ---------------------------------------------------------------------------------------------------- T5
def instance_norm1d(x, scale, bias, *, eps):
    raise NotImplementedError('T5: instance_norm1d ' + _M4)


def layer_norm(x, scale, bias, *, axis, eps):
    raise NotImplementedError('T5: layer_norm ' + _M4)


def group_norm(x, scale, bias, *, groups, eps):
    raise NotImplementedError('T5: group_norm ' + _M4)


# ---------------------------------------------------------------------------------------------------- T6
def upsample_nearest1d(x, *, scale):
    raise NotImplementedError('T6a: upsample_nearest1d ' + _M4)


def reflection_pad1d(x, *, pads):
    raise NotImplementedError('T6b: reflection_pad1d ' + _M4)


def rint(x):
    raise NotImplementedError('T6c: rint ' + _M4)


# ---------------------------------------------------------------------------------------------------- T1
def t1_from_dlpack(obj):
    # tc.from_dlpack asks obj.__dlpack__(stream=1, max_version=(1, 3)): stream 1 = the legacy default stream, the
    # adapter's (TensorCUDA's) only stream, so the producer orders it after its pending work.
    return tc.from_dlpack(obj)


def t1_empty(shape, dtype):
    if dtype not in ('float32', 'float16'):
        raise ValueError(f't1_empty: dtype {dtype!r} (the contract allows float32 and float16)')
    return tc.empty(tuple(int(s) for s in shape), dtype=dtype)    # device allocation, no host round-trip


def t1_fill(t, value):
    t.fill_(float(value))                                            # queued on the legacy default stream


def t1_read(t):
    # Tensor.numpy(): cudaMemcpy D2H on the legacy default stream (TensorCUDA's stream only; not a device sync).
    return np.ascontiguousarray(t.numpy())


def t1_data_ptr(t):
    return int(t.data_ptr())


def t1_stream():
    return 0                                                             # legacy default stream (TensorCUDA's only)
