"""tc_signal adapter for KV-P1T-TESTS, TORCH-BACKED PRE-CHECK (P1T-M2, G2). Not the acceptance adapter.

Copied from Project-Voice engine/tensor_requests/p1t_tests/adapters/tensorcuda_stub.py (sha256
d0c5a596916333bce007773a92e5d8d896285707412492cfaf573ccd0819fc0b, read 2026-10-12) and changed per order P1T-M2
("Torch-backed adapter"):
 * conv1d (T2) and conv_transpose1d (T3) call tc_signal on torch CUDA tensors: host numpy -> torch upload -> for
   weight_storage 'fp16' the weights are rounded to fp16 and expanded exactly back to fp32 (torch.float16 then .float(),
   round-to-nearest-even like the runner's oracle `W.astype(np.float16).astype(np.float32)`; fp16 -> fp32 is exact) ->
   tc_signal with the torch allocator -> torch.cuda.synchronize() -> download.
 * every other function raises NotImplementedError (T1 is M1's; T4 is M3; T5/T6 are M4), and the stub's module-level
   `import tensor_cuda` and its TensorCUDA compositions are removed: this adapter must not depend on core.
The TensorCUDA-backed acceptance adapter (TensorCUDA tensors, tensor_cuda.empty allocator) comes after M1 merges.

Run (from /mnt/ForgeRealm/Project-Voice, read-only; outputs into the P1T-M2 worktree):
    PYTHONPATH=<worktree>/tc_signal python3 engine/tensor_requests/p1t_tests/run_p1t.py \
        --adapter <worktree>/artifacts/p1t_m2/tc_signal_torch_adapter.py --items T2,T3 \
        --fixtures /mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures \
        --tolerances engine/tensor_requests/p1t_tests/tolerances_r2.json --out <worktree>/artifacts/p1t_m2/<report>.json

Prior art: the stub's structure (Project-Voice); ONNX Conv/ConvTranspose specs for the semantics; D6 (fp16 storage,
fp32 math) is Voice's decision. The kernels' prior art is in tc_signal/src/conv.cu. Ours: the torch upload/download
glue (no prior art known to me).
"""
from __future__ import annotations

import numpy as np
import torch

import tc_signal as ts            # <Project-Tensor>/tc_signal on PYTHONPATH

COVERAGE = {
    'T1': ('missing', 'NotImplementedError: DLPack in core is M1 (tensor_cuda); this adapter is the tc_signal pre-check'),
    'T2': ('full', 'tc_signal.conv1d on torch CUDA tensors (ONNX Conv 1-D: stride, asymmetric pads, dilation, groups)'),
    'T3': ('full', 'tc_signal.conv_transpose1d on torch CUDA tensors (gather form, groups, output_padding, dilation)'),
    'T4': ('missing', 'NotImplementedError: BiLSTM is M3'),
    'T5': ('missing', 'NotImplementedError: norms are M4'),
    'T6': ('missing', 'NotImplementedError: upsample / reflection pad / rint are M4'),
}

_DEV = 'cuda'


def _alloc(shape):
    """tc_signal allocator: fp32 torch CUDA tensors (allocated on torch's current stream, the legacy default stream)."""
    return torch.empty(tuple(shape), device=_DEV, dtype=torch.float32)


ts.set_allocator(_alloc)


# ---------------------------------------------------------------------------------------------------- transfer
def _up(a):
    """Host fp32 -> torch CUDA fp32 (C-contiguous)."""
    return torch.from_numpy(np.ascontiguousarray(np.asarray(a, dtype=np.float32))).to(_DEV)


def _down(t):
    """Device -> host fp32 numpy (after torch.cuda.synchronize())."""
    return np.ascontiguousarray(t.cpu().numpy().astype(np.float32, copy=False))


def _weights(w, weight_storage):
    """D6: fp16 storage expanded to fp32 at load (round to fp16 with torch, then .float(), which is exact)."""
    wt = _up(w)
    if weight_storage == 'fp16':
        return wt.to(torch.float16).float().contiguous()
    if weight_storage != 'fp32':
        raise ValueError(f'weight_storage {weight_storage!r}')
    return wt


# ---------------------------------------------------------------------------------------------------- T2
def conv1d(x, w, b, *, stride, pads, dilation, groups, weight_storage):
    xt, wt = _up(x), _weights(w, weight_storage)
    bt = None if b is None else _up(b)
    y = ts.conv1d(xt, wt, bt, stride=int(stride), padding=(int(pads[0]), int(pads[1])), dilation=int(dilation),
                  groups=int(groups))
    torch.cuda.synchronize()
    return _down(y)


# ---------------------------------------------------------------------------------------------------- T3
def conv_transpose1d(x, w, b, *, stride, pads, output_padding, dilation, groups, weight_storage):
    xt, wt = _up(x), _weights(w, weight_storage)
    bt = None if b is None else _up(b)
    y = ts.conv_transpose1d(xt, wt, bt, stride=int(stride), padding=(int(pads[0]), int(pads[1])),
                            output_padding=int(output_padding), dilation=int(dilation), groups=int(groups))
    torch.cuda.synchronize()
    return _down(y)


# ---------------------------------------------------------------------------------------------------- T4
def lstm(x, W, R, B, *, direction, hidden_size, initial_h, initial_c, layout):
    raise NotImplementedError('T4: BiLSTM is P1T-M3 (tc_signal.lstm), not in this M2 pre-check adapter')


# ---------------------------------------------------------------------------------------------------- T5
def instance_norm1d(x, scale, bias, *, eps):
    raise NotImplementedError('T5: instance_norm1d is P1T-M4 (tc_signal), not in this M2 pre-check adapter')


def layer_norm(x, scale, bias, *, axis, eps):
    raise NotImplementedError('T5: layer_norm is P1T-M4 (tc_signal), not in this M2 pre-check adapter')


def group_norm(x, scale, bias, *, groups, eps):
    raise NotImplementedError('T5: group_norm is P1T-M4 (tc_signal), not in this M2 pre-check adapter')


# ---------------------------------------------------------------------------------------------------- T6
def upsample_nearest1d(x, *, scale):
    raise NotImplementedError('T6a: upsample_nearest1d is P1T-M4 (tc_signal), not in this M2 pre-check adapter')


def reflection_pad1d(x, *, pads):
    raise NotImplementedError('T6b: reflection_pad1d is P1T-M4 (tc_signal), not in this M2 pre-check adapter')


def rint(x):
    raise NotImplementedError('T6c: rint is P1T-M4 (tc_signal), not in this M2 pre-check adapter')


# ---------------------------------------------------------------------------------------------------- T1
def t1_from_dlpack(obj):
    raise NotImplementedError('T1: DLPack import is P1T-M1 (tensor_cuda.from_dlpack); torch-backed M2 pre-check')


def t1_empty(shape, dtype):
    raise NotImplementedError('T1: P1T-M1 (tensor_cuda); torch-backed M2 pre-check')


def t1_fill(t, value):
    raise NotImplementedError('T1: P1T-M1 (tensor_cuda Tensor.fill_); torch-backed M2 pre-check')


def t1_read(t):
    raise NotImplementedError('T1: P1T-M1 (tensor_cuda); torch-backed M2 pre-check')


def t1_data_ptr(t):
    raise NotImplementedError('T1: P1T-M1 (tensor_cuda Tensor.data_ptr); torch-backed M2 pre-check')


def t1_stream():
    raise NotImplementedError('T1: P1T-M1 (tensor_cuda); torch-backed M2 pre-check')
