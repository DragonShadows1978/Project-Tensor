"""TensorCUDA adapter for KV-P1T-TESTS, integration of M1 + M2 (T1, T2, T3 real; T4-T6 NotImplementedError).

Provenance: P1T-M1's adapter (`artifacts/p1t_m1/tensorcuda_adapter.py`, branch voice-m1 953d1df), itself a copy of
Project-Voice's stub. The lead (Project-Tensor) added T2/T3 on 2026-10-12 for the TensorCUDA-backed acceptance run.
Run it with both packages on PYTHONPATH:

    cd /mnt/ForgeRealm/Project-Voice && PYTHONPATH=<fork>/tensor_cuda/tensor_cuda:<fork>/tensor_cuda:<fork>/tc_signal \
        python3 engine/tensor_requests/p1t_tests/run_p1t.py --adapter <fork>/artifacts/p1t_int/tensorcuda_adapter.py \
        --items T1,T2,T3 --fixtures artifacts/p1t/fixtures \
        --tolerances engine/tensor_requests/p1t_tests/tolerances_r2.json --out <report.json>

T2/T3 path, per the approved plan (`docs/P1T_TENSOR_ALT_PLAN.md` section 5, Voice LD51):
 * upload with `tensor_cuda.tensor` (host fp32 -> device fp32);
 * weight_storage 'fp16' (D6): numpy rounds w to fp16 (round to nearest even, as the runner's oracle), the fp16 tensor
   is uploaded and expanded on the device by TensorCUDA's `astype('float32')` (fp16 -> fp32 is exact);
 * `tc_signal.conv1d` / `conv_transpose1d` consume the TensorCUDA tensors through DLPack and allocate the output
   with `tensor_cuda.empty` (tc_signal's default allocator), all on the legacy default stream;
 * download with `Tensor.numpy()` (cudaMemcpy D2H on the legacy default stream).
T4-T6 raise NotImplementedError until M3/M4. T1 is M1's, unchanged.

Prior art: as in the M1 adapter (DLPack, the array API protocol); ONNX Conv/ConvTranspose for T2/T3 semantics.
The glue is ours; no other prior art claimed.
"""
from __future__ import annotations

import numpy as np

import tensor_cuda as tc          # the fork's build on PYTHONPATH
import tc_signal as ts           # the fork's add-on on PYTHONPATH

_M2 = 'lands in the tc_signal add-on (M2-M4; P1T_TENSOR_ALT_PLAN.md section 4), not in TensorCUDA core'

COVERAGE = {
    'T1': ('real', 'tc.from_dlpack / Tensor.__dlpack__ / __dlpack_device__ / data_ptr / fill_ (P1T-M1); allocation '
                   'via tc.empty (Amendment 1)'),
    'T2': ('real', 'tc_signal.conv1d on TensorCUDA tensors (P1T-M2), fp16 weights expanded by tc astype (D6)'),
    'T3': ('real', 'tc_signal.conv_transpose1d on TensorCUDA tensors (P1T-M2), fp16 weights expanded by tc astype'),
    'T4': ('missing', 'NotImplementedError: lstm ' + _M2),
    'T5': ('missing', 'NotImplementedError: instance_norm1d / layer_norm / group_norm ' + _M2),
    'T6': ('missing', 'NotImplementedError: upsample_nearest1d / reflection_pad1d / rint ' + _M2),
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
    raise NotImplementedError('T4: lstm ' + _M2)


# ---------------------------------------------------------------------------------------------------- T5
def instance_norm1d(x, scale, bias, *, eps):
    raise NotImplementedError('T5: instance_norm1d ' + _M2)


def layer_norm(x, scale, bias, *, axis, eps):
    raise NotImplementedError('T5: layer_norm ' + _M2)


def group_norm(x, scale, bias, *, groups, eps):
    raise NotImplementedError('T5: group_norm ' + _M2)


# ---------------------------------------------------------------------------------------------------- T6
def upsample_nearest1d(x, *, scale):
    raise NotImplementedError('T6a: upsample_nearest1d ' + _M2)


def reflection_pad1d(x, *, pads):
    raise NotImplementedError('T6b: reflection_pad1d ' + _M2)


def rint(x):
    raise NotImplementedError('T6c: rint ' + _M2)


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
