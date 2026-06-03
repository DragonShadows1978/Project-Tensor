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

_NP_DTYPE = {"float32": np.float32, "float16": np.float16}


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


def mse_loss(pred, target):
    return _C.mse_loss(pred, target)


def where(cond, x, y):
    return _C.where(cond, x, y)


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

__all__ = [
    "Tensor", "tensor", "from_numpy", "zeros", "ones", "randn", "rand",
    "matmul", "mse_loss", "cross_entropy", "where", "cat", "stack", "embedding",
    "synchronize", "no_grad", "is_grad_enabled", "nn", "optim", "functional",
]
__version__ = "0.1.0-phase1"
