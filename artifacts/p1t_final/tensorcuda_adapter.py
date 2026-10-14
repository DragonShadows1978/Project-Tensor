"""TensorCUDA adapter for KV-P1T-TESTS: all of T1-T6 (P1T M1-M4, final integration on branch voice-int).

Provenance: Project-Voice's stub -> P1T-M1 (T1) -> lead integration (T2/T3, artifacts/p1t_int) -> P1T-M3 (T4,
artifacts/p1t_m3) and P1T-M4 (T5/T6, artifacts/p1t_m4); the lead merged the M3 and M4 function bodies unchanged into
this file on 2026-10-12. Run it with both packages on PYTHONPATH:

    cd /mnt/ForgeRealm/Project-Voice && PYTHONPATH=<fork>/tensor_cuda/tensor_cuda:<fork>/tensor_cuda:<fork>/tc_signal \
        python3 engine/tensor_requests/p1t_tests/run_p1t.py --adapter <fork>/artifacts/p1t_final/tensorcuda_adapter.py \
        --items T1,T2,T3,T4,T5,T6 --fixtures artifacts/p1t/fixtures \
        --tolerances engine/tensor_requests/p1t_tests/tolerances_r2.json --out <report.json>

Mapping: T1 -> TensorCUDA core DLPack (from_dlpack, empty, fill_, data_ptr); T2-T6 -> tc_signal.* on TensorCUDA tensors
(uploaded with tensor_cuda.tensor, outputs from tensor_cuda.empty, downloaded with Tensor.numpy()), legacy default
stream throughout; fp16 weight storage (T2/T3) is rounded by numpy (RNE) and expanded exactly by TensorCUDA astype (D6).

Prior art: as in the M1-M4 adapters (DLPack and the array API protocol; ONNX operator specs for every T2-T6 contract).
The glue is ours; no other prior art claimed.
"""
from __future__ import annotations

import numpy as np

import tensor_cuda as tc          # the fork's build on PYTHONPATH
import tc_signal as ts           # the fork's add-on on PYTHONPATH

# The runner's only LSTM layout value (adapter_api.LSTM_LAYOUT): ONNX layout attribute 0.
_RUNNER_LSTM_LAYOUT = 'onnx'
_TC_SIGNAL_LAYOUT_SEQ_FIRST = 0

COVERAGE = {
    'T1': ('real', 'tc.from_dlpack / Tensor.__dlpack__ / __dlpack_device__ / data_ptr / fill_ (P1T-M1); allocation '
                   'via tc.empty (Amendment 1)'),
    'T2': ('real', 'tc_signal.conv1d on TensorCUDA tensors (P1T-M2), fp16 weights expanded by tc astype (D6)'),
    'T3': ('real', 'tc_signal.conv_transpose1d on TensorCUDA tensors (P1T-M2), fp16 weights expanded by tc astype'),
    'T4': ('real', 'tc_signal.lstm on TensorCUDA tensors (P1T-M3): ONNX LSTM, one call per LSTM, C++ step loop'),
    'T5': ('real', 'tc_signal.instance_norm1d / layer_norm / group_norm on TensorCUDA tensors (P1T-M4), fp32 '
                   'shifted two-pass statistics'),
    'T6': ('real', 'tc_signal.upsample_nearest1d / reflection_pad1d / rint on TensorCUDA tensors (P1T-M4), bit-exact'),
}


# ---------------------------------------------------------------------------------------------------- T2
def _up(a):
    """Host fp32 -> TensorCUDA fp32 on the device."""
    return tc.tensor(np.ascontiguousarray(np.asarray(a, dtype=np.float32)), dtype='float32')


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


def lstm(x, W, R, B, *, direction, hidden_size, initial_h, initial_c, layout):
    if layout != _RUNNER_LSTM_LAYOUT:
        raise AssertionError(f'T4: the runner passed layout={layout!r}; this adapter implements only '
                             f'{_RUNNER_LSTM_LAYOUT!r} (ONNX layout 0)')
    Y, Y_h, Y_c = ts.lstm(_up(x), _up(W), _up(R), _up_opt(B), hidden_size=int(hidden_size), direction=str(direction),
                          initial_h=_up_opt(initial_h), initial_c=_up_opt(initial_c),
                          layout=_TC_SIGNAL_LAYOUT_SEQ_FIRST)
    return _down(Y), _down(Y_h), _down(Y_c)


# ---------------------------------------------------------------------------------------------------- T5
def _up_opt(a):
    return None if a is None else _up(a)


def instance_norm1d(x, scale, bias, *, eps):
    return _down(ts.instance_norm1d(_up(x), _up_opt(scale), _up_opt(bias), eps=float(eps)))


def layer_norm(x, scale, bias, *, axis, eps):
    return _down(ts.layer_norm(_up(x), _up_opt(scale), _up_opt(bias), axis=int(axis), eps=float(eps)))


def group_norm(x, scale, bias, *, groups, eps):
    return _down(ts.group_norm(_up(x), int(groups), _up_opt(scale), _up_opt(bias), eps=float(eps)))


# ---------------------------------------------------------------------------------------------------- T6
def upsample_nearest1d(x, *, scale):
    return _down(ts.upsample_nearest1d(_up(x), scale))


def reflection_pad1d(x, *, pads):
    return _down(ts.reflection_pad1d(_up(x), (int(pads[0]), int(pads[1]))))


def rint(x):
    return _down(ts.rint(_up(x)))


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
