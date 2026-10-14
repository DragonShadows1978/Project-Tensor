"""TensorCUDA adapter for KV-P1T-TESTS, M1 (T1 real; T2-T6 NotImplementedError until M2+).

Provenance: a copy of Project-Voice's stub
`/mnt/ForgeRealm/Project-Voice/engine/tensor_requests/p1t_tests/adapters/tensorcuda_stub.py`
(sha256 d0c5a596916333bce007773a92e5d8d896285707412492cfaf573ccd0819fc0b), changed by Project-Tensor's P1T-M1 seat
(order `orders/P1T_M1_DLPACK.md` item 7, amended by `orders/P1T_M1_AMENDMENT_1.md`). Run it with the fork build first
on PYTHONPATH:

    cd /mnt/ForgeRealm/Project-Voice && PYTHONPATH=<fork>/tensor_cuda/tensor_cuda:<fork>/tensor_cuda \
        python3 engine/tensor_requests/p1t_tests/run_p1t.py --adapter <fork>/artifacts/p1t_m1/tensorcuda_adapter.py \
        --items T1 --fixtures artifacts/p1t/fixtures \
        --tolerances engine/tensor_requests/p1t_tests/tolerances_r2.json --out <report.json>

What changed from the stub, and why:
 * T1 is real. TensorCUDA now has `Tensor.__dlpack__` / `__dlpack_device__` / `data_ptr` / `fill_` / `mark_modified`
   / `revision` (tensor_cuda/src/bindings.cpp), `tensor_cuda.from_dlpack` and `tensor_cuda.empty`
   (tensor_cuda/tensor_cuda/__init__.py). The stub's `hasattr` probes and NotImplementedError branches are gone.
 * Census fix, t1_empty: the stub allocated with `tc.zeros` (host numpy array + synchronous H2D upload). It now uses
   `tc.empty` (Amendment 1): a device allocation by the engine's allocator, no host round-trip, contents
   uninitialized. Every runner T1 case fills before it reads.
 * Census fix, t1_read: the stub cited `Tensor.numpy()` at nn.py:66, which is a use site. The definition is the C++
   binding `tensor_to_numpy` (bindings.cpp), which calls `NDArray::to_host` (kernels.cu): a `cudaMemcpy` D2H issued
   on the legacy default stream. It waits for TensorCUDA's stream (the legacy one) and returns only when the copy
   is done; it is not a device-wide synchronize, and it does not wait on non-blocking streams (torch's side streams
   are created non-blocking), so a missing cross-stream fence still shows up as stale data.
 * Census fix, t1_fill: the stub said there was no in-place fill (only `_C.scale_`). `Tensor.fill_(value)` now
   exists, queued on the legacy default stream through the engine's existing `fill_kernel`.
 * T2-T6: the stub's partial compositions over core ops (Conv1D-as-Conv2D, col2im, InstanceNorm2D/LayerNorm/GroupNorm
   compositions) are replaced by NotImplementedError. The approved plan (`docs/P1T_TENSOR_ALT_PLAN.md` in the Tensor
   voice fork, Voice LD51) puts T2-T6 in the separate add-on package `tc_signal` (M2-M4), and this order says to
   leave them raising until then. The stub's composition code is unchanged in Project-Voice's repository.

Prior art: the Python array API data-interchange protocol (`__dlpack__(stream=)`, `__dlpack_device__`; Consortium
for Python Data API Standards, 2021-2023) and DLPack (dmlc/dlpack, 2017-), as in the stub; the adapter only maps the
contract onto TensorCUDA's calls. No prior art beyond that is claimed.
"""
from __future__ import annotations

import numpy as np

import tensor_cuda as tc          # the fork's build on PYTHONPATH

_M2 = 'lands in the tc_signal add-on (M2-M4; P1T_TENSOR_ALT_PLAN.md section 4), not in TensorCUDA core'

COVERAGE = {
    'T1': ('real', 'tc.from_dlpack / Tensor.__dlpack__ / __dlpack_device__ / data_ptr / fill_ (P1T-M1); allocation '
                   'via tc.empty (Amendment 1)'),
    'T2': ('missing', 'NotImplementedError: conv1d ' + _M2),
    'T3': ('missing', 'NotImplementedError: conv_transpose1d ' + _M2),
    'T4': ('missing', 'NotImplementedError: lstm ' + _M2),
    'T5': ('missing', 'NotImplementedError: instance_norm1d / layer_norm / group_norm ' + _M2),
    'T6': ('missing', 'NotImplementedError: upsample_nearest1d / reflection_pad1d / rint ' + _M2),
}


# ---------------------------------------------------------------------------------------------------- T2
def conv1d(x, w, b, *, stride, pads, dilation, groups, weight_storage):
    raise NotImplementedError('T2: conv1d ' + _M2)


# ---------------------------------------------------------------------------------------------------- T3
def conv_transpose1d(x, w, b, *, stride, pads, output_padding, dilation, groups, weight_storage):
    raise NotImplementedError('T3: conv_transpose1d ' + _M2)


# ---------------------------------------------------------------------------------------------------- T4
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
