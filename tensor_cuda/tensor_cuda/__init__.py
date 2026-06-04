"""Project Tensor — standalone CUDA tensor library (no PyTorch, no CuPy).

Phase 1 (the core engine) exposes a NumPy-friendly Tensor with reverse-mode
autograd backed entirely by hand-written CUDA kernels + cuBLAS. The C++ does the
work; this module is the thin Python ergonomics layer (factory helpers, dtype
plumbing, no_grad). See ROADMAP.md for the layers still being ported.
"""

from __future__ import annotations

import contextlib

import numpy as np

try:
    from . import _tensor_cuda as _C
except ImportError:  # pragma: no cover
    import _tensor_cuda as _C  # fallback when the .so sits on PYTHONPATH

Tensor = _C.Tensor

_NP_DTYPE = {
    "float32": np.float32,
    "float16": np.float16,
    "int64": np.int64,
    "bool": np.bool_,
    "uint8": np.uint8,
}


def tensor(data, *, device="cuda", dtype="float32", requires_grad=False):
    """Create a Tensor from array-like data."""
    arr = np.asarray(data, dtype=_NP_DTYPE.get(dtype, np.float32))
    arr = np.ascontiguousarray(arr)
    return _C.tensor(arr, device, requires_grad)


def _factory(np_fn):
    def make(*shape, device="cuda", dtype="float32", requires_grad=False):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        arr = np_fn(shape).astype(_NP_DTYPE.get(dtype, np.float32))
        return _C.tensor(np.ascontiguousarray(arr), device, requires_grad)
    return make


zeros = _factory(np.zeros)
ones = _factory(np.ones)
randn = _factory(lambda s: np.random.randn(*s))
rand = _factory(lambda s: np.random.rand(*s))


def from_numpy(arr, *, device="cuda", requires_grad=False):
    return _C.tensor(np.ascontiguousarray(arr), device, requires_grad)


def matmul(a, b):
    return _C.matmul(a, b)


def int4_linear(x, packed, scales, zeros, group_size=128):
    """INT4 group-quantized linear: y = x @ dequant(W)^T. Inference only."""
    return _C.int4_linear(x, packed, scales, zeros, group_size)


def int4_dequant(packed, scales, zeros, group_size=128, out_dtype="float16"):
    """Dequantize packed INT4 weight to a (K, N) transposed fp16/fp32 matrix."""
    return _C.int4_dequant(packed, scales, zeros, group_size, out_dtype)


def apa_selective_attention(q, k, kq, v, scale, zthr, is_causal=False):
    """Fused sparse selective APA attention: full-precision dot only on the keys
    the bulk/quantized pass selects (|bulk| >= mean+zthr*std), rest stay quantized.
    q,k,kq,v: (B,H,L,D)/(B,H,S,D). Inference only (no autograd)."""
    return _C.apa_selective_attention(q, k, kq, v, scale, zthr, is_causal)


def apa_blend_softmax(bulk, rank, zthr):
    """Fused APA blend+softmax over precomputed bulk/rank score matrices (..., S):
    per row thr = mean(|rank|)+zthr*std(|rank|); score = |rank|>=thr ? rank : bulk;
    returns softmax(score). Causal masking must be baked into bulk/rank as large
    negative scores by the caller. Pairs with cuBLAS bulk/rank matmuls."""
    return _C.apa_blend_softmax(bulk, rank, zthr)


def mse_loss(pred, target):
    return _C.mse_loss(pred, target)


def where(cond, x, y):
    return _C.where(cond, x, y)


def einsum(equation, *operands):
    from . import functional
    return functional.einsum(equation, *operands)


def embedding(weight, idx):
    if not isinstance(idx, Tensor):
        idx = _C.tensor(np.ascontiguousarray(np.asarray(idx, dtype=np.int64)), "cuda", False)
    return _C.embedding(weight, idx)


def cat(tensors, dim=0):
    return _C.cat(list(tensors), dim)


def stack(tensors, dim=0):
    return _C.stack(list(tensors), dim)


def cross_entropy(logits, target, *, device="cuda"):
    """Cross-entropy from logits. `target` may be int class labels (1D) or a
    float one-hot Tensor matching `logits`."""
    if isinstance(target, Tensor):
        return _C.cross_entropy(logits, target)
    labels = np.asarray(target).astype(np.int64).ravel()
    num_classes = logits.shape[-1]
    onehot = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    onehot[np.arange(labels.shape[0]), labels] = 1.0
    onehot = onehot.reshape(logits.shape)
    return _C.cross_entropy(logits, _C.tensor(onehot, device, False))


def save_checkpoint(path, model, **extra):
    """Save model parameters (+ optional extra scalars) to a .npz file."""
    sd = model.state_dict()
    payload = {f"model.{k}": v for k, v in sd.items()}
    for k, v in extra.items():
        payload[f"extra.{k}"] = np.array(v)
    np.savez(path, **payload)


def load_checkpoint(path, model):
    """Load parameters saved by save_checkpoint into `model`. Returns extras."""
    data = np.load(path, allow_pickle=True)
    sd = {k[len("model."):]: data[k] for k in data.files if k.startswith("model.")}
    model.load_state_dict(sd)
    return {k[len("extra."):]: data[k] for k in data.files if k.startswith("extra.")}


def weight_tie(src_module, src_attr, dst_module, dst_attr):
    """Tie two parameters to share one Tensor (e.g. embedding <-> LM head).

    Both modules then reference the same parameter object; gradients accumulate
    once and optimizers (which dedup by identity) update it once.
    """
    shared = getattr(src_module, src_attr)
    setattr(dst_module, dst_attr, shared)
    return shared


def checkpoint(fn, *inputs):
    """Gradient checkpointing.

    NOTE: the current engine builds the graph normally (this is a transparent,
    correct wrapper). True activation-recompute checkpointing needs a Python
    grad_fn hook and is tracked in ROADMAP; the API is provided for
    compatibility so models written against it run unchanged.
    """
    return fn(*inputs)


def synchronize():
    _C.synchronize()


def is_grad_enabled():
    return _C.is_grad_enabled()


@contextlib.contextmanager
def no_grad():
    prev = _C.is_grad_enabled()
    _C.set_grad_enabled(False)
    try:
        yield
    finally:
        _C.set_grad_enabled(prev)


from . import functional  # noqa: E402
from . import nn  # noqa: E402  (after _C and helpers are defined)
from . import optim  # noqa: E402
from . import quant  # noqa: E402

apa_quant_attention = quant.apa_quant_attention

__all__ = [
    "Tensor", "tensor", "from_numpy", "zeros", "ones", "randn", "rand",
    "matmul", "mse_loss", "cross_entropy", "where", "cat", "stack", "embedding",
    "synchronize", "no_grad", "is_grad_enabled", "nn", "optim", "functional",
    "quant", "apa_quant_attention", "save_checkpoint", "load_checkpoint",
    "weight_tie", "checkpoint", "einsum", "int4_linear", "int4_dequant",
    "apa_selective_attention",
]
__version__ = "0.1.0-phase1"
