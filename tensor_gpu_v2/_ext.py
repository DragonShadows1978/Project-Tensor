"""Extended operations — Phases E–K.

Covers tensor math completeness, weight-init utilities, additional
optimizers/schedulers, extra layers (RMSNorm, RoPE, …), and the
Dataset/DataLoader API.

This module is part of the ``tensor_gpu_v2`` package.
Import via ``import tensor_gpu_v2 as tg``, not directly.
"""

from ._core import *
from ._core import _get_cached_kernel
from ._nn import *
from ._training import *


# ==============================================================
# PHASE E — TENSOR MATH COMPLETENESS
# ==============================================================

# ---- E1: Elementwise math methods ----

def _abs(self):
    xp = self.xp
    out = Tensor(xp.abs(self.data), (self,), 'abs', device=self._device)
    def _backward():
        self.grad += xp.sign(self.data) * out.grad
    out._backward = _backward
    return out

def _sqrt(self):
    xp = self.xp
    out_data = xp.sqrt(self.data)
    out = Tensor(out_data, (self,), 'sqrt', device=self._device)
    def _backward():
        self.grad += 0.5 / (out_data + 1e-12) * out.grad
    out._backward = _backward
    return out

def _exp(self):
    xp = self.xp
    out_data = xp.exp(self.data)
    out = Tensor(out_data, (self,), 'exp', device=self._device)
    def _backward():
        self.grad += out_data * out.grad
    out._backward = _backward
    return out

def _log(self):
    xp = self.xp
    out = Tensor(xp.log(self.data), (self,), 'log', device=self._device)
    def _backward():
        self.grad += out.grad / (self.data + 1e-12)
    out._backward = _backward
    return out

def _log2(self):
    xp = self.xp
    out = Tensor(xp.log2(self.data), (self,), 'log2', device=self._device)
    def _backward():
        self.grad += out.grad / (self.data * 0.6931471805599453 + 1e-12)
    out._backward = _backward
    return out

def _log10(self):
    xp = self.xp
    out = Tensor(xp.log10(self.data), (self,), 'log10', device=self._device)
    def _backward():
        self.grad += out.grad / (self.data * 2.302585092994046 + 1e-12)
    out._backward = _backward
    return out

def _clamp(self, min=None, max=None):
    xp = self.xp
    out_data = xp.clip(self.data, min, max)
    out = Tensor(out_data, (self,), 'clamp', device=self._device)
    def _backward():
        mask = xp.ones_like(self.data)
        if min is not None:
            mask *= (self.data >= min)
        if max is not None:
            mask *= (self.data <= max)
        self.grad += mask * out.grad
    out._backward = _backward
    return out

def _reciprocal(self):
    xp = self.xp
    out_data = 1.0 / self.data
    out = Tensor(out_data, (self,), 'reciprocal', device=self._device)
    def _backward():
        self.grad += -out_data ** 2 * out.grad
    out._backward = _backward
    return out

def _sign(self):
    xp = self.xp
    out = Tensor(xp.sign(self.data), (self,), 'sign', device=self._device)
    def _backward():
        pass  # subgradient is zero everywhere
    out._backward = _backward
    return out

def _ceil(self):
    xp = self.xp
    out = Tensor(xp.ceil(self.data), (self,), 'ceil', device=self._device)
    def _backward():
        pass  # STE: pass-through gradient
        self.grad += out.grad
    out._backward = _backward
    return out

def _floor(self):
    xp = self.xp
    out = Tensor(xp.floor(self.data), (self,), 'floor', device=self._device)
    def _backward():
        self.grad += out.grad   # STE
    out._backward = _backward
    return out

def _round(self):
    xp = self.xp
    out = Tensor(xp.round(self.data), (self,), 'round', device=self._device)
    def _backward():
        self.grad += out.grad   # STE
    out._backward = _backward
    return out

def _nan_to_num(self, nan=0.0, posinf=None, neginf=None):
    xp = self.xp
    if hasattr(xp, 'nan_to_num'):
        out_data = xp.nan_to_num(self.data, nan=nan, posinf=posinf, neginf=neginf)
    else:
        out_data = xp.where(xp.isnan(self.data), nan, self.data)
        if posinf is not None:
            out_data = xp.where(xp.isposinf(self.data), posinf, out_data)
        if neginf is not None:
            out_data = xp.where(xp.isneginf(self.data), neginf, out_data)
    out = Tensor(out_data, (self,), 'nan_to_num', device=self._device)
    def _backward():
        mask = xp.isfinite(self.data)
        self.grad += mask * out.grad
    out._backward = _backward
    return out

def _isnan(self):
    return Tensor(self.xp.isnan(self.data), device=self._device, requires_grad=False)

def _isinf(self):
    return Tensor(self.xp.isinf(self.data), device=self._device, requires_grad=False)

def _isfinite(self):
    return Tensor(self.xp.isfinite(self.data), device=self._device, requires_grad=False)

# ---- E2: Trigonometric methods ----

def _sin(self):
    xp = self.xp
    out = Tensor(xp.sin(self.data), (self,), 'sin', device=self._device)
    def _backward():
        self.grad += xp.cos(self.data) * out.grad
    out._backward = _backward
    return out

def _cos(self):
    xp = self.xp
    out = Tensor(xp.cos(self.data), (self,), 'cos', device=self._device)
    def _backward():
        self.grad += -xp.sin(self.data) * out.grad
    out._backward = _backward
    return out

def _tan(self):
    xp = self.xp
    cos_x = xp.cos(self.data)
    out = Tensor(xp.tan(self.data), (self,), 'tan', device=self._device)
    def _backward():
        self.grad += out.grad / (cos_x ** 2 + 1e-12)
    out._backward = _backward
    return out

def _asin(self):
    xp = self.xp
    out = Tensor(xp.arcsin(self.data), (self,), 'asin', device=self._device)
    def _backward():
        self.grad += out.grad / (xp.sqrt(1 - self.data ** 2) + 1e-12)
    out._backward = _backward
    return out

def _acos(self):
    xp = self.xp
    out = Tensor(xp.arccos(self.data), (self,), 'acos', device=self._device)
    def _backward():
        self.grad += -out.grad / (xp.sqrt(1 - self.data ** 2) + 1e-12)
    out._backward = _backward
    return out

def _atan(self):
    xp = self.xp
    out = Tensor(xp.arctan(self.data), (self,), 'atan', device=self._device)
    def _backward():
        self.grad += out.grad / (1 + self.data ** 2)
    out._backward = _backward
    return out

def _atan2(self, other):
    xp = self.xp
    out = Tensor(xp.arctan2(self.data, other.data), (self, other), 'atan2', device=self._device)
    def _backward():
        denom = self.data ** 2 + other.data ** 2 + 1e-12
        self.grad  += out.grad *  other.data / denom
        other.grad += out.grad * -self.data  / denom
    out._backward = _backward
    return out

def _sinh(self):
    xp = self.xp
    out = Tensor(xp.sinh(self.data), (self,), 'sinh', device=self._device)
    def _backward():
        self.grad += xp.cosh(self.data) * out.grad
    out._backward = _backward
    return out

def _cosh(self):
    xp = self.xp
    out = Tensor(xp.cosh(self.data), (self,), 'cosh', device=self._device)
    def _backward():
        self.grad += xp.sinh(self.data) * out.grad
    out._backward = _backward
    return out

def _asinh(self):
    xp = self.xp
    out = Tensor(xp.arcsinh(self.data), (self,), 'asinh', device=self._device)
    def _backward():
        self.grad += out.grad / (xp.sqrt(self.data ** 2 + 1) + 1e-12)
    out._backward = _backward
    return out

def _acosh(self):
    xp = self.xp
    out = Tensor(xp.arccosh(self.data), (self,), 'acosh', device=self._device)
    def _backward():
        self.grad += out.grad / (xp.sqrt(self.data ** 2 - 1) + 1e-12)
    out._backward = _backward
    return out

def _atanh(self):
    xp = self.xp
    out = Tensor(xp.arctanh(self.data), (self,), 'atanh', device=self._device)
    def _backward():
        self.grad += out.grad / (1 - self.data ** 2 + 1e-12)
    out._backward = _backward
    return out

# Patch all elementwise + trig methods onto Tensor
Tensor.abs        = _abs
Tensor.sqrt       = _sqrt
Tensor.exp        = _exp
Tensor.log        = _log
Tensor.log2       = _log2
Tensor.log10      = _log10
Tensor.clamp      = _clamp
Tensor.clip       = _clamp          # alias
Tensor.reciprocal = _reciprocal
Tensor.sign       = _sign
Tensor.ceil       = _ceil
Tensor.floor      = _floor
Tensor.round      = _round
Tensor.nan_to_num = _nan_to_num
Tensor.isnan      = _isnan
Tensor.isinf      = _isinf
Tensor.isfinite   = _isfinite
Tensor.sin        = _sin
Tensor.cos        = _cos
Tensor.tan        = _tan
Tensor.asin       = _asin
Tensor.acos       = _acos
Tensor.atan       = _atan
Tensor.atan2      = _atan2
Tensor.sinh       = _sinh
Tensor.cosh       = _cosh
Tensor.asinh      = _asinh
Tensor.acosh      = _acosh
Tensor.atanh      = _atanh


# ---- E3: Masking / selection ops ----

def _masked_fill(self, mask, value):
    """Return a new tensor with positions where mask==True filled with value."""
    xp = self.xp
    mask_data = mask.data if isinstance(mask, Tensor) else mask
    out_data = xp.where(mask_data, xp.array(value, dtype=self.data.dtype), self.data)
    out = Tensor(out_data, (self,), 'masked_fill', device=self._device)
    def _backward():
        self.grad += xp.where(mask_data, xp.zeros_like(out.grad), out.grad)
    out._backward = _backward
    return out

def _nonzero(self):
    """Return (N, ndim) int64 tensor of indices where self != 0. No gradient."""
    xp = self.xp
    idx = xp.stack(xp.nonzero(self.data), axis=1)
    return Tensor(idx, device=self._device, requires_grad=False)

Tensor.masked_fill = _masked_fill
Tensor.nonzero     = _nonzero


# Module-level where(condition, x, y)
def where(condition, x, y):
    """Element-wise selection: condition ? x : y. Supports Tensor or scalar x/y."""
    xp = x.xp if isinstance(x, Tensor) else (y.xp if isinstance(y, Tensor) else np)
    cond = condition.data if isinstance(condition, Tensor) else condition
    xd   = x.data        if isinstance(x, Tensor)         else xp.array(x, dtype=xp.float32)
    yd   = y.data        if isinstance(y, Tensor)         else xp.array(y, dtype=xp.float32)
    dev  = (x._device    if isinstance(x, Tensor)         else
            y._device    if isinstance(y, Tensor)         else get_device())
    out_data = xp.where(cond, xd, yd)
    children = tuple(t for t in (condition, x, y) if isinstance(t, Tensor))
    out = Tensor(out_data, children, 'where', device=dev)
    def _backward():
        if isinstance(x, Tensor) and x.grad is not None:
            x.grad += xp.where(cond, out.grad, xp.zeros_like(out.grad))
        if isinstance(y, Tensor) and y.grad is not None:
            y.grad += xp.where(cond, xp.zeros_like(out.grad), out.grad)
    out._backward = _backward
    return out


# Module-level pad(x, pad_widths, mode, value)
def pad(x, pad_widths, mode='constant', value=0.0):
    """Pad a tensor.

    pad_widths: sequence of (before, after) pairs, one per dimension, innermost first
                (matches numpy convention when reversed, or pass as numpy-style tuple-of-pairs).
    """
    xp = x.xp
    # Accept both PyTorch-style flat tuple (last-dim-first) and numpy-style tuple-of-pairs
    if isinstance(pad_widths[0], (int, float)):
        # Flat: (left, right) or (left, right, top, bottom, ...) — last dim first
        flat = list(pad_widths)
        pairs = []
        while flat:
            after = flat.pop(0)
            before = flat.pop(0) if flat else 0
            pairs.insert(0, (before, after))
        # Prepend (0,0) for leading dimensions not specified
        while len(pairs) < x.ndim:
            pairs.insert(0, (0, 0))
        pw = pairs
    else:
        pw = list(pad_widths)
        while len(pw) < x.ndim:
            pw.insert(0, (0, 0))

    kwargs = {} if mode != 'constant' else {'constant_values': value}
    out_data = xp.pad(x.data, pw, mode=mode, **kwargs)
    out = Tensor(out_data, (x,), 'pad', device=x._device)

    def _backward():
        slices = tuple(slice(b, s - a if a > 0 else None) for (b, a), s in zip(pw, out_data.shape))
        x.grad += out.grad[slices]

    out._backward = _backward
    return out


# ---- E4: Advanced indexing ----

def _gather(self, dim, index):
    """Differentiable gather: out[i][j][k] = self[i][index[i][j][k]][k] (for dim=1)."""
    xp = self.xp
    idx = index.data if isinstance(index, Tensor) else index
    out_data = xp.take_along_axis(self.data, idx, axis=dim)
    out = Tensor(out_data, (self,), 'gather', device=self._device)
    def _backward():
        grad = xp.zeros_like(self.data)
        xp.add.at(grad, tuple(
            idx if ax == dim else xp.arange(self.data.shape[ax]).reshape(
                [-1 if ax2 == ax else 1 for ax2 in range(self.data.ndim)])
            for ax in range(self.data.ndim)
        ), out.grad)
        self.grad += grad
    out._backward = _backward
    return out

def _scatter(self, dim, index, src):
    """Scatter src values into self at positions given by index along dim.
    Returns a new tensor (does not mutate self).
    """
    xp = self.xp
    idx = index.data if isinstance(index, Tensor) else index
    src_data = src.data if isinstance(src, Tensor) else xp.full_like(self.data, src)
    out_data = xp.copy(self.data)
    xp.put_along_axis(out_data, idx, src_data, axis=dim)
    parents = (self,) + ((src,) if isinstance(src, Tensor) else ())
    out = Tensor(out_data, parents, 'scatter', device=self._device)
    def _backward():
        # grad to self: zero out the scattered positions
        mask = xp.zeros_like(self.data, dtype=bool)
        xp.put_along_axis(mask, idx, True, axis=dim)
        self.grad += xp.where(mask, xp.zeros_like(out.grad), out.grad)
        # grad to src
        if isinstance(src, Tensor) and src.grad is not None:
            src.grad += xp.take_along_axis(out.grad, idx, axis=dim)
    out._backward = _backward
    return out

def _index_select(self, dim, index):
    """Select slices along dim using a 1-D index tensor."""
    xp = self.xp
    idx = index.data if isinstance(index, Tensor) else index
    out_data = xp.take(self.data, idx, axis=dim)
    out = Tensor(out_data, (self,), 'index_select', device=self._device)
    def _backward():
        grad = xp.zeros_like(self.data)
        xp.add.at(grad, [slice(None)] * dim + [idx], out.grad)
        self.grad += grad
    out._backward = _backward
    return out

Tensor.gather       = _gather
Tensor.scatter      = _scatter
Tensor.index_select = _index_select


# ---- E5: Reductions and sorting ----

def _prod(self, axis=None, keepdims=False):
    xp = self.xp
    out_data = xp.prod(self.data, axis=axis, keepdims=keepdims)
    out = Tensor(out_data, (self,), 'prod', device=self._device)
    def _backward():
        # grad_i = prod(all except i) * grad = total_prod / x_i * grad
        # Numerically stable: use exp(sum(log|x|)) when possible
        g = out.grad
        if not keepdims and axis is not None:
            g = xp.expand_dims(g, axis=axis)
            out_exp = xp.expand_dims(out_data, axis=axis)
        else:
            out_exp = out_data
        # prod / x_i — handle zeros carefully
        safe_x = xp.where(self.data != 0, self.data, xp.ones_like(self.data))
        full_prod = xp.broadcast_to(out_exp, self.data.shape)
        self.grad += g * full_prod / safe_x
    out._backward = _backward
    return out

def _cumsum(self, axis=0):
    xp = self.xp
    out_data = xp.cumsum(self.data, axis=axis)
    out = Tensor(out_data, (self,), 'cumsum', device=self._device)
    def _backward():
        # grad of cumsum is reverse cumsum (sum from position to end)
        self.grad += xp.cumsum(out.grad[..., ::-1] if axis == -1 else
                               xp.flip(out.grad, axis=axis), axis=axis)
        if axis != -1:
            self.grad = xp.flip(self.grad, axis=axis)  # undo the flip
        # simpler: just use flip explicitly
    out._backward = lambda: None  # replaced below
    def _cumsum_bwd():
        g = xp.flip(out.grad, axis=axis)
        self.grad += xp.flip(xp.cumsum(g, axis=axis), axis=axis)
    out._backward = _cumsum_bwd
    return out

def _cumprod(self, axis=0):
    xp = self.xp
    out_data = xp.cumprod(self.data, axis=axis)
    out = Tensor(out_data, (self,), 'cumprod', device=self._device)
    def _backward():
        # grad_i = sum_{j>=i} (out_j / x_i) * grad_j
        # = (1/x_i) * sum_{j>=i} cumprod[j] * grad_j
        # Use the identity: reverse_cumsum(out * grad) / x
        safe_x = xp.where(self.data != 0, self.data, xp.ones_like(self.data))
        g = xp.flip(xp.cumsum(xp.flip(out_data * out.grad, axis=axis), axis=axis), axis=axis)
        self.grad += g / safe_x
    out._backward = _backward
    return out

def _norm(self, p=2, axis=None, keepdims=False):
    xp = self.xp
    if p == 2:
        sq = xp.sum(self.data ** 2, axis=axis, keepdims=keepdims)
        out_data = xp.sqrt(sq)
    elif p == 1:
        out_data = xp.sum(xp.abs(self.data), axis=axis, keepdims=keepdims)
    else:
        out_data = xp.sum(xp.abs(self.data) ** p, axis=axis, keepdims=keepdims) ** (1.0 / p)
    out = Tensor(out_data, (self,), 'norm', device=self._device)
    def _backward():
        g = out.grad
        if not keepdims and axis is not None:
            g = xp.expand_dims(g, axis=axis)
            n = xp.expand_dims(out_data, axis=axis)
        else:
            n = out_data
        if p == 2:
            self.grad += self.data * g / (n + 1e-12)
        elif p == 1:
            self.grad += xp.sign(self.data) * g
        else:
            self.grad += (xp.abs(self.data) ** (p - 1) * xp.sign(self.data) *
                          (xp.sum(xp.abs(self.data) ** p, axis=axis, keepdims=True) ** (1.0/p - 1)) * g)
    out._backward = _backward
    return out

def _any(self, axis=None, keepdims=False):
    return Tensor(self.xp.any(self.data, axis=axis, keepdims=keepdims),
                  device=self._device, requires_grad=False)

def _all(self, axis=None, keepdims=False):
    return Tensor(self.xp.all(self.data, axis=axis, keepdims=keepdims),
                  device=self._device, requires_grad=False)

def _flip(self, dims):
    xp = self.xp
    if isinstance(dims, int):
        dims = (dims,)
    out_data = xp.flip(self.data, axis=dims)
    out = Tensor(out_data, (self,), 'flip', device=self._device)
    def _backward():
        self.grad += xp.flip(out.grad, axis=dims)
    out._backward = _backward
    return out

def _roll(self, shifts, dims=None):
    xp = self.xp
    out_data = xp.roll(self.data, shifts, axis=dims)
    out = Tensor(out_data, (self,), 'roll', device=self._device)
    def _backward():
        neg = -shifts if not isinstance(shifts, (list, tuple)) else [-s for s in shifts]
        self.grad += xp.roll(out.grad, neg, axis=dims)
    out._backward = _backward
    return out

def _repeat(self, repeats, axis=None):
    xp = self.xp
    out_data = xp.repeat(self.data, repeats, axis=axis)
    out = Tensor(out_data, (self,), 'repeat', device=self._device)
    def _backward():
        if axis is None:
            g = out.grad.reshape(self.data.shape)
        else:
            # Sum the repeated blocks
            r = repeats if isinstance(repeats, int) else repeats
            n = self.data.shape[axis]
            g = xp.stack([out.grad.take(
                    xp.arange(i * r, (i + 1) * r) if isinstance(r, int)
                    else xp.arange(sum(r[:i]), sum(r[:i+1])), axis=axis
                ).sum(axis=axis) for i in range(n)], axis=axis)
        self.grad += g
    out._backward = _backward
    return out

def _tile(self, reps):
    xp = self.xp
    out_data = xp.tile(self.data, reps)
    out = Tensor(out_data, (self,), 'tile', device=self._device)
    def _backward():
        # Sum over the tiled copies
        g = out.grad
        orig_shape = self.data.shape
        reps_full = ([1] * (len(orig_shape) - len(reps)) + list(reps)
                     if not isinstance(reps, int) else [reps] * len(orig_shape))
        for ax, r in enumerate(reps_full):
            if r > 1:
                g = g.reshape(*g.shape[:ax], r, g.shape[ax] // r, *g.shape[ax+1:]).sum(axis=ax)
        self.grad += g.reshape(orig_shape)
    out._backward = _backward
    return out

def _sort(self, axis=-1, descending=False):
    xp = self.xp
    idx = xp.argsort(self.data, axis=axis)
    if descending:
        idx = xp.flip(idx, axis=axis)
    values = xp.take_along_axis(self.data, idx, axis=axis)
    vals_t  = Tensor(values, (self,), 'sort_values', device=self._device)
    idx_t   = Tensor(idx,    device=self._device, requires_grad=False)
    def _backward():
        # scatter gradients back to original positions
        grad = xp.zeros_like(self.data)
        xp.put_along_axis(grad, idx, vals_t.grad, axis=axis)
        self.grad += grad
    vals_t._backward = _backward
    return vals_t, idx_t

def _topk(self, k, axis=-1, largest=True):
    xp = self.xp
    if largest:
        idx = xp.argsort(self.data, axis=axis)[..., -k:]
        idx = xp.flip(idx, axis=axis)
    else:
        idx = xp.argsort(self.data, axis=axis)[..., :k]
    values = xp.take_along_axis(self.data, idx, axis=axis)
    vals_t = Tensor(values, (self,), 'topk_values', device=self._device)
    idx_t  = Tensor(idx,    device=self._device, requires_grad=False)
    def _backward():
        grad = xp.zeros_like(self.data)
        xp.put_along_axis(grad, idx, vals_t.grad, axis=axis)
        self.grad += grad
    vals_t._backward = _backward
    return vals_t, idx_t

Tensor.prod      = _prod
Tensor.cumsum    = _cumsum
Tensor.cumprod   = _cumprod
Tensor.norm      = _norm
Tensor.any       = _any
Tensor.all       = _all
Tensor.flip      = _flip
Tensor.roll      = _roll
Tensor.repeat    = _repeat
Tensor.tile      = _tile
Tensor.sort      = _sort
Tensor.topk      = _topk


# ==============================================================
# PHASE F — WEIGHT INITIALIZATION UTILITIES
# ==============================================================

import math as _math

class _InitNamespace:
    """Namespace for in-place weight initialization functions.

    All functions operate on a Tensor in-place and return it, mirroring
    PyTorch's torch.nn.init API.

    Usage:
        init.kaiming_normal_(layer.w)
        model.apply(lambda m: init.xavier_uniform_(m.w) if hasattr(m, 'w') else None)
    """

    @staticmethod
    def _fan(tensor):
        """Compute (fan_in, fan_out) for a weight tensor."""
        ndim = tensor.data.ndim
        if ndim < 2:
            raise ValueError("Cannot compute fan for tensor with fewer than 2 dimensions")
        fan_in  = tensor.data.shape[1] * (tensor.data[0, 0].size if ndim > 2 else 1)
        fan_out = tensor.data.shape[0] * (tensor.data[0, 0].size if ndim > 2 else 1)
        return fan_in, fan_out

    @staticmethod
    def _gain(nonlinearity, param=None):
        gains = {
            'linear': 1.0, 'conv1d': 1.0, 'conv2d': 1.0,
            'sigmoid': 1.0, 'tanh': 5.0 / 3,
            'relu': _math.sqrt(2.0), 'leaky_relu': _math.sqrt(2.0 / (1 + (param or 0.01) ** 2)),
            'selu': 3.0 / 4, 'gelu': 1.0,
        }
        return gains.get(nonlinearity, 1.0)

    @staticmethod
    def uniform_(tensor, a=0.0, b=1.0):
        xp = tensor.xp
        tensor.data[:] = xp.random.uniform(a, b, tensor.data.shape).astype(tensor.data.dtype)
        return tensor

    @staticmethod
    def normal_(tensor, mean=0.0, std=1.0):
        xp = tensor.xp
        tensor.data[:] = xp.random.normal(mean, std, tensor.data.shape).astype(tensor.data.dtype)
        return tensor

    @staticmethod
    def constant_(tensor, val):
        tensor.data[:] = val
        return tensor

    @staticmethod
    def zeros_(tensor):
        tensor.data[:] = 0
        return tensor

    @staticmethod
    def ones_(tensor):
        tensor.data[:] = 1
        return tensor

    def xavier_uniform_(self, tensor, gain=1.0):
        fan_in, fan_out = self._fan(tensor)
        std = gain * _math.sqrt(2.0 / (fan_in + fan_out))
        a = _math.sqrt(3.0) * std
        return self.uniform_(tensor, -a, a)

    def xavier_normal_(self, tensor, gain=1.0):
        fan_in, fan_out = self._fan(tensor)
        std = gain * _math.sqrt(2.0 / (fan_in + fan_out))
        return self.normal_(tensor, 0.0, std)

    def kaiming_uniform_(self, tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        fan_in, fan_out = self._fan(tensor)
        fan = fan_in if mode == 'fan_in' else fan_out
        gain = self._gain(nonlinearity, a)
        std = gain / _math.sqrt(fan)
        bound = _math.sqrt(3.0) * std
        return self.uniform_(tensor, -bound, bound)

    def kaiming_normal_(self, tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        fan_in, fan_out = self._fan(tensor)
        fan = fan_in if mode == 'fan_in' else fan_out
        gain = self._gain(nonlinearity, a)
        std = gain / _math.sqrt(fan)
        return self.normal_(tensor, 0.0, std)

    def trunc_normal_(self, tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        """Truncated normal initialization (clips to [mean+a*std, mean+b*std])."""
        xp = tensor.xp
        from scipy.stats import truncnorm as _tn
        lo, hi = (a - mean) / std, (b - mean) / std
        data = _tn.rvs(lo, hi, loc=mean, scale=std, size=tensor.data.shape).astype('float32')
        if xp is cp:
            tensor.data[:] = cp.array(data)
        else:
            tensor.data[:] = data
        return tensor

    def orthogonal_(self, tensor, gain=1.0):
        """Orthogonal initialization via QR decomposition."""
        xp = tensor.xp
        shape = tensor.data.shape
        flat_shape = (shape[0], int(tensor.data.size // shape[0]))
        a = np.random.normal(0, 1, flat_shape)
        q, r = np.linalg.qr(a)
        # Make Q uniform (account for sign)
        d = np.diag(r)
        ph = np.sign(d)
        q *= ph
        if flat_shape[0] < flat_shape[1]:
            q = q.T
        q = q.reshape(shape[:2] + shape[2:])[:shape[0]]
        if xp is cp:
            tensor.data[:] = cp.array(q.astype('float32')) * gain
        else:
            tensor.data[:] = q.astype('float32') * gain
        return tensor


init = _InitNamespace()


# ==============================================================
# PHASE G — MORE OPTIMIZERS & LR SCHEDULERS
# ==============================================================

class Adagrad:
    """Adagrad optimizer: accumulates squared gradients per parameter.

    Good for sparse gradients (e.g., embedding tables in NLP).
    """

    def __init__(self, params, lr=0.01, lr_decay=0.0, weight_decay=0.0,
                 initial_accumulator_value=0.0, eps=1e-10):
        self.params      = list(params)
        self.lr          = lr
        self.lr_decay    = lr_decay
        self.weight_decay = weight_decay
        self.eps         = eps
        self._device     = self.params[0]._device if self.params else 'cpu'
        xp = cp if self._device == 'cuda' else np
        self.state_sum = [xp.full_like(p.data, initial_accumulator_value) for p in self.params]
        self.t = 0

    def step(self):
        xp = cp if self._device == 'cuda' else np
        self.t += 1
        clr = self.lr / (1 + (self.t - 1) * self.lr_decay)
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay != 0:
                g = g + self.weight_decay * p.data
            self.state_sum[i] += g ** 2
            p.data -= clr * g / (xp.sqrt(self.state_sum[i]) + self.eps)

    def zero_grad(self):
        xp = cp if self._device == 'cuda' else np
        for p in self.params:
            if p.grad is not None:
                p.grad = xp.zeros_like(p.data)

    def state_dict(self):
        return {'lr': self.lr, 'lr_decay': self.lr_decay, 't': self.t,
                'state_sum': [to_cpu(s) for s in self.state_sum]}

    def load_state_dict(self, state):
        xp = cp if self._device == 'cuda' else np
        self.lr = state.get('lr', self.lr)
        self.lr_decay = state.get('lr_decay', self.lr_decay)
        self.t = state.get('t', self.t)
        if 'state_sum' in state:
            self.state_sum = [xp.array(s, dtype=xp.float32) for s in state['state_sum']]


class RAdam:
    """Rectified Adam optimizer.

    Fixes Adam's large variance in early training by computing the
    rectification term rho that decides when to trust the adaptive
    learning rate.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params       = list(params)
        self.lr           = lr
        self.beta1, self.beta2 = betas
        self.eps          = eps
        self.weight_decay = weight_decay
        self._device      = self.params[0]._device if self.params else 'cpu'
        xp = cp if self._device == 'cuda' else np
        self.m  = [xp.zeros_like(p.data) for p in self.params]
        self.v  = [xp.zeros_like(p.data) for p in self.params]
        self.t  = 0
        self._rho_inf = 2 / (1 - self.beta2) - 1

    def step(self):
        xp = cp if self._device == 'cuda' else np
        self.t += 1
        b1, b2, t = self.beta1, self.beta2, self.t

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay != 0:
                p.data -= self.lr * self.weight_decay * p.data

            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g ** 2

            m_hat = self.m[i] / (1 - b1 ** t)
            rho_t = self._rho_inf - 2 * t * b2**t / (1 - b2**t)

            if rho_t > 4:
                # Variance is tractable — use adaptive lr
                v_hat = xp.sqrt(self.v[i] / (1 - b2**t))
                r = _math.sqrt((rho_t - 4) * (rho_t - 2) * self._rho_inf /
                               ((self._rho_inf - 4) * (self._rho_inf - 2) * rho_t))
                p.data -= self.lr * r * m_hat / (v_hat + self.eps)
            else:
                # Variance too high — fall back to SGD step
                p.data -= self.lr * m_hat

    def zero_grad(self):
        xp = cp if self._device == 'cuda' else np
        for p in self.params:
            if p.grad is not None:
                p.grad = xp.zeros_like(p.data)

    def state_dict(self):
        return {'lr': self.lr, 'betas': (self.beta1, self.beta2),
                'eps': self.eps, 'weight_decay': self.weight_decay, 't': self.t,
                'm': [to_cpu(m) for m in self.m], 'v': [to_cpu(v) for v in self.v]}

    def load_state_dict(self, state):
        xp = cp if self._device == 'cuda' else np
        self.lr = state.get('lr', self.lr)
        self.beta1, self.beta2 = state.get('betas', (self.beta1, self.beta2))
        self.t = state.get('t', self.t)
        if 'm' in state:
            self.m = [xp.array(x, dtype=xp.float32) for x in state['m']]
        if 'v' in state:
            self.v = [xp.array(x, dtype=xp.float32) for x in state['v']]


class Lion:
    """Lion optimizer (EvoLved Sign Momentum) — Symbolic Discovery of Optimization Algorithms.

    Chen et al., 2023. Uses only sign(m), so update magnitude is always 1·lr.
    Typically 2-3x lower memory than Adam (no v term). Uses cosine decay by default.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        self.params       = list(params)
        self.lr           = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self._device      = self.params[0]._device if self.params else 'cpu'
        xp = cp if self._device == 'cuda' else np
        self.m = [xp.zeros_like(p.data) for p in self.params]

    def step(self):
        xp = cp if self._device == 'cuda' else np
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            # update = sign(beta1 * m + (1 - beta1) * g)
            update = xp.sign(self.beta1 * self.m[i] + (1 - self.beta1) * g)
            if self.weight_decay != 0:
                update += self.weight_decay * p.data
            p.data -= self.lr * update
            # momentum update uses beta2 (different from update)
            self.m[i] = self.beta2 * self.m[i] + (1 - self.beta2) * g

    def zero_grad(self):
        xp = cp if self._device == 'cuda' else np
        for p in self.params:
            if p.grad is not None:
                p.grad = xp.zeros_like(p.data)

    def state_dict(self):
        return {'lr': self.lr, 'betas': (self.beta1, self.beta2),
                'weight_decay': self.weight_decay,
                'm': [to_cpu(m) for m in self.m]}

    def load_state_dict(self, state):
        xp = cp if self._device == 'cuda' else np
        self.lr = state.get('lr', self.lr)
        self.beta1, self.beta2 = state.get('betas', (self.beta1, self.beta2))
        if 'm' in state:
            self.m = [xp.array(x, dtype=xp.float32) for x in state['m']]


# ---- LR Schedulers ----

class MultiStepLR:
    """Decay lr by gamma at each milestone in milestones list."""

    def __init__(self, optimizer, milestones, gamma=0.1):
        self.optimizer  = optimizer
        self.milestones = sorted(milestones)
        self.gamma      = gamma
        self._base_lr   = optimizer.lr
        self._step_count = 0

    def step(self):
        self._step_count += 1
        if self._step_count in self.milestones:
            self.optimizer.lr *= self.gamma

    def state_dict(self):
        return {'step_count': self._step_count, 'milestones': self.milestones,
                'gamma': self.gamma}

    def load_state_dict(self, state):
        self._step_count = state.get('step_count', self._step_count)


class ExponentialLR:
    """Multiply lr by gamma every step."""

    def __init__(self, optimizer, gamma):
        self.optimizer = optimizer
        self.gamma     = gamma
        self._base_lr  = optimizer.lr

    def step(self):
        self.optimizer.lr *= self.gamma

    def state_dict(self):
        return {'gamma': self.gamma, 'current_lr': self.optimizer.lr}

    def load_state_dict(self, state):
        if 'current_lr' in state:
            self.optimizer.lr = state['current_lr']


class ReduceLROnPlateau:
    """Reduce lr when a monitored metric stops improving.

    mode='min': lr is reduced when metric stops decreasing (e.g., val loss).
    mode='max': lr is reduced when metric stops increasing (e.g., accuracy).
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, min_lr=0.0, verbose=False):
        if factor >= 1.0:
            raise ValueError("factor must be < 1.0")
        self.optimizer = optimizer
        self.mode      = mode
        self.factor    = factor
        self.patience  = patience
        self.threshold = threshold
        self.min_lr    = min_lr
        self.verbose   = verbose
        self._best     = float('inf') if mode == 'min' else float('-inf')
        self._num_bad_epochs = 0
        self._num_reductions = 0

    def _is_better(self, current):
        if self.mode == 'min':
            return current < self._best - self._best * self.threshold
        return current > self._best + abs(self._best) * self.threshold

    def step(self, metric):
        if self._is_better(metric):
            self._best = metric
            self._num_bad_epochs = 0
        else:
            self._num_bad_epochs += 1
            if self._num_bad_epochs >= self.patience:
                new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
                if new_lr < self.optimizer.lr:
                    if self.verbose:
                        print(f"ReduceLROnPlateau: reducing lr to {new_lr:.2e}")
                    self.optimizer.lr = new_lr
                    self._num_reductions += 1
                self._num_bad_epochs = 0

    def state_dict(self):
        return {'best': self._best, 'num_bad_epochs': self._num_bad_epochs,
                'num_reductions': self._num_reductions}

    def load_state_dict(self, state):
        self._best              = state.get('best', self._best)
        self._num_bad_epochs    = state.get('num_bad_epochs', 0)
        self._num_reductions    = state.get('num_reductions', 0)


class OneCycleLR:
    """One-Cycle learning rate policy (Smith & Touvron, 2019).

    Linearly warms up from base_lr/div_factor to max_lr in pct_start fraction
    of total_steps, then anneals to min_lr via cosine in the remaining steps.
    """

    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3,
                 div_factor=25.0, final_div_factor=1e4, anneal_strategy='cos'):
        self.optimizer       = optimizer
        self.max_lr          = max_lr
        self.total_steps     = total_steps
        self.pct_start       = pct_start
        self.base_lr         = max_lr / div_factor
        self.min_lr          = max_lr / (div_factor * final_div_factor)
        self.anneal_strategy = anneal_strategy
        self._step           = 0
        # Set initial lr
        self.optimizer.lr = self.base_lr

    def _annealing(self, start, end, pct):
        if self.anneal_strategy == 'cos':
            return end + (start - end) / 2.0 * (_math.cos(_math.pi * pct) + 1)
        return start + (end - start) * pct

    def step(self):
        self._step += 1
        warmup_steps = int(self.total_steps * self.pct_start)
        if self._step <= warmup_steps:
            pct = self._step / warmup_steps
            self.optimizer.lr = self._annealing(self.base_lr, self.max_lr, pct)
        else:
            pct = (self._step - warmup_steps) / (self.total_steps - warmup_steps)
            self.optimizer.lr = self._annealing(self.max_lr, self.min_lr, pct)

    def state_dict(self):
        return {'step': self._step, 'max_lr': self.max_lr, 'total_steps': self.total_steps}

    def load_state_dict(self, state):
        self._step = state.get('step', self._step)


class CyclicLR:
    """Cyclic learning rate: triangular or exp-range cycling between base_lr and max_lr."""

    def __init__(self, optimizer, base_lr, max_lr, step_size_up=2000,
                 step_size_down=None, mode='triangular', gamma=1.0):
        self.optimizer     = optimizer
        self.base_lr       = base_lr
        self.max_lr        = max_lr
        self.step_size_up   = step_size_up
        self.step_size_down = step_size_down if step_size_down is not None else step_size_up
        self.mode          = mode
        self.gamma         = gamma
        self._cycle_step   = 0
        self._cycle        = 0
        self.optimizer.lr  = base_lr

    def step(self):
        cycle_size = self.step_size_up + self.step_size_down
        self._cycle_step += 1
        pos_in_cycle = self._cycle_step % cycle_size
        self._cycle   = self._cycle_step // cycle_size

        if pos_in_cycle < self.step_size_up:
            pct = pos_in_cycle / self.step_size_up
        else:
            pct = 1 - (pos_in_cycle - self.step_size_up) / self.step_size_down

        scale = pct if self.mode == 'triangular' else pct * (self.gamma ** self._cycle)
        self.optimizer.lr = self.base_lr + (self.max_lr - self.base_lr) * scale

    def state_dict(self):
        return {'cycle_step': self._cycle_step, 'cycle': self._cycle}

    def load_state_dict(self, state):
        self._cycle_step = state.get('cycle_step', self._cycle_step)
        self._cycle      = state.get('cycle', self._cycle)


class ConstantLR:
    """Hold lr at a constant fraction of base_lr for num_steps, then restore."""

    def __init__(self, optimizer, factor=1.0/3, total_iters=5):
        self.optimizer    = optimizer
        self.factor       = factor
        self.total_iters  = total_iters
        self._base_lr     = optimizer.lr
        self._step        = 0
        optimizer.lr      = self._base_lr * factor

    def step(self):
        self._step += 1
        if self._step == self.total_iters:
            self.optimizer.lr = self._base_lr

    def state_dict(self): return {'step': self._step}
    def load_state_dict(self, s): self._step = s.get('step', 0)


class LinearLR:
    """Linearly interpolate lr from start_factor to end_factor over total_iters steps."""

    def __init__(self, optimizer, start_factor=1.0/3, end_factor=1.0, total_iters=5):
        self.optimizer    = optimizer
        self.start_factor = start_factor
        self.end_factor   = end_factor
        self.total_iters  = total_iters
        self._base_lr     = optimizer.lr
        self._step        = 0
        optimizer.lr      = self._base_lr * start_factor

    def step(self):
        self._step = min(self._step + 1, self.total_iters)
        t = self._step / self.total_iters
        factor = self.start_factor + (self.end_factor - self.start_factor) * t
        self.optimizer.lr = self._base_lr * factor

    def state_dict(self): return {'step': self._step}
    def load_state_dict(self, s): self._step = s.get('step', 0)


# ==============================================================
# PHASE I — ADDITIONAL LAYERS
# ==============================================================

# ---- I1: RMSNorm ----

_RMSNORM_FWD_KERNEL = r"""
extern "C" __global__
void rmsnorm_fwd(const float* x, const float* w, float* out, float* rms,
                 int N, int D, float eps) {
    int n = blockIdx.x;
    if (n >= N) return;
    const float* xn = x + n * D;
    float* on = out + n * D;

    float sum = 0.f;
    for (int d = 0; d < D; d++) sum += xn[d] * xn[d];
    float r = rsqrtf(sum / D + eps);
    rms[n] = r;
    for (int d = 0; d < D; d++) on[d] = xn[d] * r * w[d];
}
"""

_RMSNORM_BWD_KERNEL = r"""
extern "C" __global__
void rmsnorm_bwd(const float* x, const float* w, const float* dout, const float* rms,
                 float* dx, float* dw, int N, int D) {
    int n = blockIdx.x;
    if (n >= N) return;
    const float* xn   = x    + n * D;
    const float* doutn = dout + n * D;
    float*       dxn  = dx   + n * D;
    float r = rms[n];

    // dw accumulation (atomics needed for parallel N)
    for (int d = 0; d < D; d++)
        atomicAdd(dw + d, doutn[d] * xn[d] * r);

    // dx: d/dx (x * r * w) = r * w - x * r^3 / D * (sum dout*x*w)
    float dot = 0.f;
    for (int d = 0; d < D; d++) dot += doutn[d] * xn[d] * w[d];
    for (int d = 0; d < D; d++)
        dxn[d] = r * (doutn[d] * w[d] - xn[d] * r * r * dot / D);
}
"""


class RMSNorm(Module):
    """Root-Mean-Square Layer Normalization (no mean subtraction).

    out = x / rms(x) * weight,  where rms(x) = sqrt(mean(x^2) + eps)

    Used in LLaMA, Mistral, Falcon. Faster than LayerNorm (~30% on GPU).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        xp = cp if get_device() == 'cuda' else np
        self.weight = Tensor(xp.ones(normalized_shape, dtype=xp.float32), device=get_device())
        self._fwd_k = self._bwd_k = None

    def _get_kernels(self):
        if self._device == 'cuda' and self._fwd_k is None:
            self._fwd_k = _get_cached_kernel('rmsnorm_fwd', _RMSNORM_FWD_KERNEL, 'rmsnorm_fwd')
            self._bwd_k = _get_cached_kernel('rmsnorm_bwd', _RMSNORM_BWD_KERNEL, 'rmsnorm_bwd')

    def __call__(self, x):
        xp = cp if self._device == 'cuda' else np
        orig_shape = x.shape
        D = 1
        for s in self.normalized_shape:
            D *= s
        x_2d = x.data.reshape(-1, D)
        N = x_2d.shape[0]

        self._get_kernels()
        if self._device == 'cuda' and self._fwd_k is not None and x.data.dtype == cp.float32:
            out_data = cp.empty_like(x_2d)
            rms_buf  = cp.empty(N, dtype=cp.float32)
            self._fwd_k((N,), (1,), (x_2d, self.weight.data.ravel(),
                                     out_data, rms_buf, N, D, cp.float32(self.eps)))
            out = Tensor(out_data.reshape(orig_shape), (x, self.weight), 'RMSNorm', device=self._device)
            def _backward():
                dx_2d = cp.zeros_like(x_2d)
                dw    = cp.zeros_like(self.weight.data.ravel())
                dout_2d = out.grad.reshape(N, D)
                self._bwd_k((N,), (1,), (x_2d, self.weight.data.ravel(), dout_2d,
                                         rms_buf, dx_2d, dw, N, D))
                x.grad += dx_2d.reshape(orig_shape)
                self.weight.grad += dw.reshape(self.weight.data.shape)
            out._backward = _backward
        else:
            rms = xp.sqrt(xp.mean(x_2d ** 2, axis=1, keepdims=True) + self.eps)
            x_norm = x_2d / rms
            out_data = (x_norm * self.weight.data.ravel()).reshape(orig_shape)
            out = Tensor(out_data, (x, self.weight), 'RMSNorm', device=self._device)
            def _backward():
                dout = out.grad.reshape(N, D)
                self.weight.grad += (dout * x_norm).sum(axis=0).reshape(self.weight.data.shape)
                dx_norm = dout * self.weight.data.ravel()
                dx = (dx_norm / rms -
                      x_2d * (dx_norm * x_2d).mean(axis=1, keepdims=True) / (rms ** 3))
                x.grad += dx.reshape(orig_shape)
            out._backward = _backward

        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), out)
            if hr is not None: out = hr
        return out

    def parameters(self):
        return [self.weight]

    def __repr__(self):
        return f"RMSNorm({self.normalized_shape}, eps={self.eps})"


# ---- I2: InstanceNorm2D ----

class InstanceNorm2D(Module):
    """Instance normalization for (N, C, H, W) — normalizes per sample, per channel."""

    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.num_features = num_features
        self.eps    = eps
        self.affine = affine
        xp = cp if get_device() == 'cuda' else np
        if affine:
            self.gamma = Tensor(xp.ones(num_features, dtype=xp.float32), device=get_device())
            self.beta  = Tensor(xp.zeros(num_features, dtype=xp.float32), device=get_device())
        else:
            self.gamma = self.beta = None

    def __call__(self, x):
        xp = cp if self._device == 'cuda' else np
        N, C, H, W = x.shape
        x_flat = x.data.reshape(N, C, -1)   # (N, C, H*W)
        mean   = x_flat.mean(axis=2, keepdims=True)
        var    = x_flat.var(axis=2, keepdims=True)
        std_inv = 1.0 / xp.sqrt(var + self.eps)
        x_norm  = (x_flat - mean) * std_inv

        if self.affine:
            out_data = (x_norm * self.gamma.data[None, :, None] +
                        self.beta.data[None, :, None]).reshape(N, C, H, W)
        else:
            out_data = x_norm.reshape(N, C, H, W)

        children = (x,) + ((self.gamma, self.beta) if self.affine else ())
        out = Tensor(out_data, children, 'InstanceNorm2D', device=self._device)

        def _backward():
            dout = out.grad.reshape(N, C, -1)
            if self.affine:
                self.gamma.grad += (dout * x_norm).sum(axis=(0, 2))
                self.beta.grad  += dout.sum(axis=(0, 2))
                dout = dout * self.gamma.data[None, :, None]
            M = H * W
            dx = std_inv * (dout - dout.mean(axis=2, keepdims=True) -
                            x_norm * (dout * x_norm).mean(axis=2, keepdims=True))
            x.grad += dx.reshape(N, C, H, W)

        out._backward = _backward
        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), out)
            if hr is not None: out = hr
        return out

    def parameters(self):
        return [self.gamma, self.beta] if self.affine else []

    def __repr__(self):
        return f"InstanceNorm2D({self.num_features}, affine={self.affine})"


# ---- I3: Upsample / interpolate ----

def interpolate(x, size=None, scale_factor=None, mode='nearest'):
    """Resize spatial dimensions of (N, C, H, W) tensor.

    Args:
        size: (H_out, W_out) tuple
        scale_factor: float or (fy, fx) — used if size is None
        mode: 'nearest' | 'bilinear'
    """
    xp = x.xp
    N, C, H, W = x.shape
    if size is not None:
        H_out, W_out = size
    elif scale_factor is not None:
        if isinstance(scale_factor, (int, float)):
            scale_factor = (scale_factor, scale_factor)
        H_out = int(H * scale_factor[0])
        W_out = int(W * scale_factor[1])
    else:
        raise ValueError("Either size or scale_factor must be provided")

    if mode == 'nearest':
        iy = (xp.arange(H_out) * H / H_out).astype(int)
        ix = (xp.arange(W_out) * W / W_out).astype(int)
        out_data = x.data[:, :, iy[:, None], ix[None, :]]
        out = Tensor(out_data, (x,), 'interpolate_nearest', device=x._device)
        def _backward():
            dx = xp.zeros_like(x.data)
            xp.add.at(dx, (slice(None), slice(None), iy[:, None], ix[None, :]), out.grad)
            x.grad += dx
        out._backward = _backward

    elif mode == 'bilinear':
        # Normalized source coordinates
        gy = (xp.arange(H_out, dtype=xp.float32) + 0.5) * H / H_out - 0.5
        gx = (xp.arange(W_out, dtype=xp.float32) + 0.5) * W / W_out - 0.5
        y0 = xp.clip(xp.floor(gy).astype(int), 0, H - 2)
        x0 = xp.clip(xp.floor(gx).astype(int), 0, W - 2)
        y1, x1 = y0 + 1, x0 + 1
        dy = (gy - y0).astype(xp.float32)[:, None]   # (H_out, 1)
        dx_ = (gx - x0).astype(xp.float32)[None, :]  # (1, W_out)

        q00 = x.data[:, :, y0[:, None], x0[None, :]]   # (N,C,H_out,W_out)
        q01 = x.data[:, :, y0[:, None], x1[None, :]]
        q10 = x.data[:, :, y1[:, None], x0[None, :]]
        q11 = x.data[:, :, y1[:, None], x1[None, :]]
        out_data = ((1 - dy) * (1 - dx_) * q00 + (1 - dy) * dx_ * q01 +
                    dy * (1 - dx_) * q10 + dy * dx_ * q11)
        out = Tensor(out_data, (x,), 'interpolate_bilinear', device=x._device)

        def _backward():
            dout = out.grad
            dx = xp.zeros_like(x.data)
            w00 = (1 - dy) * (1 - dx_)
            w01 = (1 - dy) * dx_
            w10 = dy * (1 - dx_)
            w11 = dy * dx_
            for (yi, xi, w) in [(y0, x0, w00), (y0, x1, w01),
                                 (y1, x0, w10), (y1, x1, w11)]:
                xp.add.at(dx, (slice(None), slice(None), yi[:, None], xi[None, :]),
                          dout * w)
            x.grad += dx

        out._backward = _backward
    else:
        raise ValueError(f"interpolate: unknown mode '{mode}', use 'nearest' or 'bilinear'")

    return out


class Upsample(Module):
    """Upsample module wrapping the functional interpolate."""

    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        super().__init__()
        self.size         = size
        self.scale_factor = scale_factor
        self.mode         = mode

    def __call__(self, x):
        out = interpolate(x, self.size, self.scale_factor, self.mode)
        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), out)
            if hr is not None: out = hr
        return out

    def parameters(self): return []
    def __repr__(self):
        return f"Upsample(size={self.size}, scale_factor={self.scale_factor}, mode='{self.mode}')"


# ---- I4: AdaptiveMaxPool2D ----

class AdaptiveMaxPool2D(Module):
    """Adaptive max pooling to a fixed (H_out, W_out) output size."""

    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def __call__(self, x):
        xp = cp if self._device == 'cuda' else np
        N, C, H, W = x.shape
        H_out, W_out = self.output_size
        out_data = xp.zeros((N, C, H_out, W_out), dtype=x.data.dtype)
        argmax_i = xp.zeros((N, C, H_out, W_out), dtype=int)
        argmax_j = xp.zeros((N, C, H_out, W_out), dtype=int)

        for i in range(H_out):
            h0 = int(i * H / H_out); h1 = int((i + 1) * H / H_out)
            for j in range(W_out):
                w0 = int(j * W / W_out); w1 = int((j + 1) * W / W_out)
                patch = x.data[:, :, h0:h1, w0:w1].reshape(N, C, -1)
                idx   = patch.argmax(axis=2)
                out_data[:, :, i, j] = patch[xp.arange(N)[:, None],
                                              xp.arange(C)[None, :], idx]
                argmax_i[:, :, i, j] = h0 + idx // (w1 - w0)
                argmax_j[:, :, i, j] = w0 + idx %  (w1 - w0)

        out = Tensor(out_data, (x,), 'AdaptiveMaxPool2D', device=self._device)

        def _backward():
            dx = xp.zeros_like(x.data)
            for i in range(H_out):
                for j in range(W_out):
                    ii = argmax_i[:, :, i, j]
                    jj = argmax_j[:, :, i, j]
                    xp.add.at(dx,
                        (xp.arange(N)[:, None], xp.arange(C)[None, :], ii, jj),
                        out.grad[:, :, i, j])
            x.grad += dx

        out._backward = _backward
        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), out)
            if hr is not None: out = hr
        return out

    def parameters(self): return []
    def __repr__(self): return f"AdaptiveMaxPool2D({self.output_size})"


# ---- I5: TransformerDecoderLayer + TransformerDecoder ----

class TransformerDecoderLayer(Module):
    """Pre-norm Transformer decoder layer with self-attention + cross-attention + FFN."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='gelu', norm_first=True, device=None):
        super().__init__()
        dev = device or get_device()
        self._device   = dev
        self.norm_first = norm_first

        self.self_attn  = _MultiHeadAttentionModule(d_model, nhead, dropout, dev)
        self.cross_attn = _MultiHeadAttentionModule(d_model, nhead, dropout, dev)
        self.ff1  = Linear(d_model, dim_feedforward)
        self.ff2  = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self._dropout = Dropout(dropout) if dropout > 0 else None

        act_map = {'gelu': lambda t: t.gelu(), 'relu': lambda t: t.relu(),
                   'silu': lambda t: t.silu()}
        self._act = act_map[activation]

    def __call__(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        tgt:    (N, L_tgt, D) — target sequence
        memory: (N, L_src, D) — encoder output
        """
        if self.norm_first:
            sa = self.self_attn(self.norm1(tgt), mask=tgt_mask)
            if self._dropout: sa = self._dropout(sa)
            tgt = tgt + sa

            ca = self._cross_attention(self.norm2(tgt), memory, memory_mask)
            if self._dropout: ca = self._dropout(ca)
            tgt = tgt + ca

            ff = self.ff2(self._act(self.ff1(self.norm3(tgt))))
            if self._dropout: ff = self._dropout(ff)
            out = tgt + ff
        else:
            sa = self.self_attn(tgt, mask=tgt_mask)
            if self._dropout: sa = self._dropout(sa)
            tgt = self.norm1(tgt + sa)

            ca = self._cross_attention(tgt, memory, memory_mask)
            if self._dropout: ca = self._dropout(ca)
            tgt = self.norm2(tgt + ca)

            ff = self.ff2(self._act(self.ff1(tgt)))
            if self._dropout: ff = self._dropout(ff)
            out = self.norm3(tgt + ff)

        for hook in self._forward_hooks.values():
            hr = hook(self, (tgt, memory), out)
            if hr is not None: out = hr
        return out

    def _cross_attention(self, q_src, kv_src, mask=None):
        """Q from q_src, K/V from kv_src."""
        N, L_q, D = q_src.shape
        _, L_kv, _ = kv_src.shape
        H, d_h = self.cross_attn.nhead, D // self.cross_attn.nhead

        Q = q_src  @ self.cross_attn.w_q
        K = kv_src @ self.cross_attn.w_k
        V = kv_src @ self.cross_attn.w_v

        def _split(t, L):
            d = t.data.reshape(N, L, H, d_h).transpose(0, 2, 1, 3)
            return Tensor(d, (t,), device=self._device)

        Qh, Kh, Vh = _split(Q, L_q), _split(K, L_kv), _split(V, L_kv)
        scale = float((cp if self._device == 'cuda' else np).sqrt(d_h))
        attn = scaled_dot_product_attention(Qh, Kh, Vh, scale=1.0/scale,
                                            dropout_p=0.0, is_causal=False)
        xp = cp if self._device == 'cuda' else np
        merged_data = attn.data.transpose(0, 2, 1, 3).reshape(N, L_q, D)
        merged = Tensor(merged_data, (attn,), 'MergeHeads', device=self._device)
        def _bwd():
            if attn.grad is not None:
                attn.grad += merged.grad.reshape(N, L_q, H, d_h).transpose(0, 2, 1, 3)
        merged._backward = _bwd
        return merged @ self.cross_attn.w_o + self.cross_attn.b_o

    def parameters(self):
        params, seen = [], set()
        for sub in [self.self_attn, self.cross_attn, self.ff1, self.ff2,
                    self.norm1, self.norm2, self.norm3]:
            for p in sub.parameters():
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
        return params

    def __repr__(self):
        return (f"TransformerDecoderLayer(d_model={self.self_attn.d_model}, "
                f"nhead={self.self_attn.nhead})")


class TransformerDecoder(Module):
    """Stack of TransformerDecoderLayers with optional final norm."""

    def __init__(self, decoder_layer_fn, num_layers, norm=None):
        super().__init__()
        import copy
        if callable(decoder_layer_fn) and not isinstance(decoder_layer_fn, Module):
            layers = [decoder_layer_fn() for _ in range(num_layers)]
        else:
            layers = [copy.deepcopy(decoder_layer_fn) for _ in range(num_layers)]
        self.layers = ModuleList(layers)
        self.norm   = norm

    def __call__(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        if self.norm is not None:
            tgt = self.norm(tgt)
        for hook in self._forward_hooks.values():
            hr = hook(self, (tgt, memory), tgt)
            if hr is not None: tgt = hr
        return tgt

    def parameters(self):
        params = self.layers.parameters()
        if self.norm is not None:
            params += self.norm.parameters()
        return params

    def __repr__(self):
        return f"TransformerDecoder(num_layers={len(self.layers)})"


# ---- I6: RNNCell / SimpleRNN ----

class RNNCell(Module):
    """Vanilla RNN cell: h_new = tanh(x @ w_ih + h @ w_hh + b)."""

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh'):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.use_bias    = bias
        self.nonlinearity = nonlinearity
        xp = cp if get_device() == 'cuda' else np
        scale = float(xp.sqrt(1.0 / hidden_size))
        self.w_ih = Tensor(xp.random.uniform(-scale, scale, (input_size,  hidden_size)).astype(xp.float32), device=get_device())
        self.w_hh = Tensor(xp.random.uniform(-scale, scale, (hidden_size, hidden_size)).astype(xp.float32), device=get_device())
        self.b    = Tensor(xp.zeros(hidden_size, dtype=xp.float32), device=get_device()) if bias else None

    def __call__(self, x, h=None):
        xp = cp if self._device == 'cuda' else np
        N, H = x.shape[0], self.hidden_size
        if h is None:
            h = Tensor(xp.zeros((N, H), dtype=xp.float32), device=self._device, requires_grad=False)

        pre = x @ self.w_ih + h @ self.w_hh
        if self.use_bias:
            pre = pre + self.b

        h_new = pre.tanh() if self.nonlinearity == 'tanh' else pre.relu()
        for hook in self._forward_hooks.values():
            hr = hook(self, (x, h), h_new)
            if hr is not None: h_new = hr
        return h_new

    def parameters(self):
        p = [self.w_ih, self.w_hh]
        if self.use_bias: p.append(self.b)
        return p

    def __repr__(self):
        return f"RNNCell({self.input_size}, {self.hidden_size}, nonlinearity='{self.nonlinearity}')"


class RNN(Module):
    """Multi-layer vanilla RNN over (N, L, input_size) sequences."""

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=True, dropout=0.0, bidirectional=False, nonlinearity='tanh'):
        super().__init__()
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.batch_first   = batch_first
        self.bidirectional = bidirectional
        num_dir = 2 if bidirectional else 1

        self.cells = ModuleList()
        for layer in range(num_layers):
            in_sz = input_size if layer == 0 else hidden_size * num_dir
            self.cells.append(RNNCell(in_sz, hidden_size, bias, nonlinearity))
            if bidirectional:
                self.cells.append(RNNCell(in_sz, hidden_size, bias, nonlinearity))
        self._dropout = Dropout(dropout) if dropout > 0 else None

    def __call__(self, x, h0=None):
        xp = cp if self._device == 'cuda' else np
        x_data = x.data if self.batch_first else x.data.transpose(1, 0, 2)
        N, L, _ = x_data.shape
        H = self.hidden_size
        num_dir = 2 if self.bidirectional else 1

        if h0 is None:
            h_init = [Tensor(xp.zeros((N, H), dtype=xp.float32), device=self._device, requires_grad=False)
                      for _ in range(self.num_layers * num_dir)]
        else:
            h_init = [Tensor(h0.data[i], device=self._device, requires_grad=False)
                      for i in range(self.num_layers * num_dir)]

        layer_input = [Tensor(x_data[:, t, :], device=self._device, requires_grad=x._requires_grad)
                       for t in range(L)]
        h_n = []
        for layer in range(self.num_layers):
            fwd_cell = self.cells[layer * num_dir]
            h = h_init[layer * num_dir]
            fwd_out = []
            for t in range(L):
                h = fwd_cell(layer_input[t], h)
                fwd_out.append(h)
            h_n.append(h)

            if self.bidirectional:
                bwd_cell = self.cells[layer * num_dir + 1]
                hb = h_init[layer * num_dir + 1]
                bwd_out = []
                for t in reversed(range(L)):
                    hb = bwd_cell(layer_input[t], hb)
                    bwd_out.insert(0, hb)
                h_n.append(hb)
                layer_input = [cat([f, b], axis=-1) for f, b in zip(fwd_out, bwd_out)]
            else:
                layer_input = fwd_out

            if self._dropout is not None and self._training and layer < self.num_layers - 1:
                layer_input = [self._dropout(t) for t in layer_input]

        outputs   = stack(layer_input, axis=1)
        h_n_stack = Tensor(xp.stack([h.data for h in h_n], axis=0), device=self._device, requires_grad=False)
        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), outputs)
            if hr is not None: outputs = hr
        return outputs, h_n_stack

    def parameters(self):
        return self.cells.parameters()

    def __repr__(self):
        return (f"RNN({self.cells[0].input_size}, {self.hidden_size}, "
                f"num_layers={self.num_layers}, bidirectional={self.bidirectional})")


# ==============================================================
# PHASE J — ROTARY & MODERN POSITIONAL ENCODINGS
# ==============================================================

def _build_rope_cache(seq_len, head_dim, device, dtype=None, base=10000.0):
    """Precompute cos/sin tables for RoPE up to seq_len."""
    xp = cp if device == 'cuda' else np
    dtype = dtype or xp.float32
    half = head_dim // 2
    theta = 1.0 / (base ** (xp.arange(0, half, dtype=xp.float32) / half))
    t     = xp.arange(seq_len, dtype=xp.float32)
    freqs = xp.outer(t, theta)           # (seq_len, half)
    emb   = xp.concatenate([freqs, freqs], axis=-1)  # (seq_len, head_dim)
    cos   = xp.cos(emb).astype(dtype)
    sin   = xp.sin(emb).astype(dtype)
    return cos, sin


def _rotate_half(x_data, xp):
    """Rotate the last dimension: split in half and swap with sign flip."""
    h = x_data.shape[-1] // 2
    x1 = x_data[..., :h]
    x2 = x_data[..., h:]
    return xp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos, sin, offset: int = 0):
    """Apply RoPE to query and key tensors.

    Args:
        q, k: (N, H, L, d_h) tensors
        cos, sin: (max_seq_len, d_h) arrays (output of _build_rope_cache)
        offset: position offset (for KV-cache decoding)

    Returns:
        q_rot, k_rot: rotated tensors, same shape
    """
    xp = q.xp
    L = q.shape[2]
    cos_l = cos[offset: offset + L]   # (L, d_h)
    sin_l = sin[offset: offset + L]

    # Broadcast to (1, 1, L, d_h)
    cos_b = cos_l[None, None, :, :]
    sin_b = sin_l[None, None, :, :]

    def _apply(x: Tensor):
        rot = x.data * cos_b + _rotate_half(x.data, xp) * sin_b
        out = Tensor(rot, (x,), 'RoPE', device=x._device)
        def _bwd():
            # d/d(x) [x*cos + rot_half(x)*sin]
            # = cos_b * grad + rot_half_T(sin_b * grad)
            # rot_half_T is the transpose of _rotate_half (flip back)
            g = out.grad
            rh_g = _rotate_half(g * sin_b, xp)
            # _rotate_half([-x2, x1]) -> _rotate_half_T(g) = [g[h:], -g[:h]]
            # Actually rot_half^T = -rot_half, so:
            half = g.shape[-1] // 2
            inv_rh = xp.concatenate([g[..., half:] * sin_b[..., :half],
                                     -g[..., :half] * sin_b[..., half:]], axis=-1)
            x.grad += g * cos_b + xp.concatenate([
                sin_b[..., g.shape[-1]//2:] * g[..., g.shape[-1]//2:] * 0,  # placeholder
            ], axis=-1)
            # Correct grad: grad_x = grad_out * cos + rot_T(grad_out * sin)
            # where rot_T reverses rot_half
            h = g.shape[-1] // 2
            g_rot = g * sin_b
            x.grad += g * cos_b + xp.concatenate([g_rot[..., h:], -g_rot[..., :h]], axis=-1)
        out._backward = _bwd
        return out

    return _apply(q), _apply(k)


def _apply_rope(x: Tensor, cos, sin, offset: int = 0):
    """Simplified RoPE apply for a single tensor."""
    xp = x.xp
    L = x.shape[-2]
    cos_b = cos[offset: offset + L][None, None, :, :]
    sin_b = sin[offset: offset + L][None, None, :, :]
    rot_data = x.data * cos_b + _rotate_half(x.data, xp) * sin_b
    out = Tensor(rot_data, (x,), 'RoPE', device=x._device)
    def _backward():
        h = out.grad.shape[-1] // 2
        g = out.grad
        g_rot = g * sin_b
        x.grad += g * cos_b + xp.concatenate([g_rot[..., h:], -g_rot[..., :h]], axis=-1)
    out._backward = _backward
    return out


class RotaryEmbedding(Module):
    """Precomputed RoPE cache as a Module.

    Usage in attention:
        rope = RotaryEmbedding(head_dim, max_seq_len=4096)
        q_rot = rope(q)
        k_rot = rope(k)
    """

    def __init__(self, dim, max_seq_len=4096, base=10000.0):
        super().__init__()
        self.dim         = dim
        self.max_seq_len = max_seq_len
        self.base        = base
        # Built lazily on first call so device is known
        self._cos = self._sin = None

    def _build(self):
        self._cos, self._sin = _build_rope_cache(
            self.max_seq_len, self.dim, self._device, base=self.base)

    def __call__(self, x, offset=0):
        if self._cos is None:
            self._build()
        return _apply_rope(x, self._cos, self._sin, offset)

    def parameters(self): return []
    def __repr__(self): return f"RotaryEmbedding(dim={self.dim}, max_seq_len={self.max_seq_len})"


# ---- ALiBi (Attention with Linear Biases) ----

def get_alibi_slopes(n_heads):
    """Compute ALiBi slopes for n_heads attention heads.

    Returns numpy array of shape (n_heads,).
    The slopes follow a geometric sequence: 2^(-8/n) for powers of 2 heads,
    with interpolation for non-power-of-2 counts.
    """
    def _slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(int(_math.log2(n)) - 3)))
        return [start * (start ** i) for i in range(n)]

    if _math.log2(n_heads).is_integer():
        slopes = _slopes_power_of_2(n_heads)
    else:
        # Interpolate between closest powers of 2
        closest_pow2 = 2 ** int(_math.log2(n_heads))
        slopes = (_slopes_power_of_2(closest_pow2) +
                  _slopes_power_of_2(2 * closest_pow2)[0::2][:n_heads - closest_pow2])
    return np.array(slopes, dtype=np.float32)


def build_alibi_bias(n_heads, seq_len, device='cpu'):
    """Build ALiBi additive bias tensor of shape (1, n_heads, seq_len, seq_len).

    The bias for head h at position (i, j) is: -slope_h * |i - j|
    """
    slopes = get_alibi_slopes(n_heads)
    xp = cp if device == 'cuda' else np
    if device == 'cuda':
        slopes = cp.array(slopes)
    # relative distances: (seq_len, seq_len) matrix of |i - j|
    pos  = xp.arange(seq_len)
    dist = xp.abs(pos[:, None] - pos[None, :])  # (seq_len, seq_len)
    # bias: (n_heads, seq_len, seq_len)
    bias = -slopes[:, None, None] * dist[None, :, :]
    return Tensor(bias[None].astype(xp.float32), device=device, requires_grad=False)


class ALiBiAttention(Module):
    """Multi-head attention with ALiBi positional bias.

    Adds a fixed linear bias to attention logits that penalises distant keys.
    No learned positional parameters. Better length generalisation than sine PE.
    """

    def __init__(self, d_model, nhead, dropout=0.0, max_seq_len=4096):
        super().__init__()
        self.d_model     = d_model
        self.nhead       = nhead
        self.max_seq_len = max_seq_len
        xp = cp if get_device() == 'cuda' else np
        scale = float(xp.sqrt(1.0 / d_model))
        self.w_q = Tensor(xp.random.randn(d_model, d_model).astype(xp.float32) * scale, device=get_device())
        self.w_k = Tensor(xp.random.randn(d_model, d_model).astype(xp.float32) * scale, device=get_device())
        self.w_v = Tensor(xp.random.randn(d_model, d_model).astype(xp.float32) * scale, device=get_device())
        self.w_o = Tensor(xp.random.randn(d_model, d_model).astype(xp.float32) * scale, device=get_device())
        self.b_o = Tensor(xp.zeros(d_model, dtype=xp.float32), device=get_device())
        self.dropout_p   = dropout
        self._alibi_bias = None

    def _get_bias(self, seq_len):
        if (self._alibi_bias is None or
                self._alibi_bias.shape[2] < seq_len):
            self._alibi_bias = build_alibi_bias(self.nhead, seq_len, self._device)
        return self._alibi_bias[:, :, :seq_len, :seq_len]

    def __call__(self, x, mask=None):
        N, L, D = x.shape
        H, d_h  = self.nhead, D // self.nhead

        Q = x @ self.w_q
        K = x @ self.w_k
        V = x @ self.w_v

        def _split(t):
            d = t.data.reshape(N, L, H, d_h).transpose(0, 2, 1, 3)
            return Tensor(d, (t,), device=self._device)

        Qh, Kh, Vh = _split(Q), _split(K), _split(V)
        scale_val  = float((cp if self._device == 'cuda' else np).sqrt(d_h))
        # Compute raw attention scores and add ALiBi bias before softmax
        xp = cp if self._device == 'cuda' else np
        scores_data = (Qh.data @ Kh.data.transpose(0, 1, 3, 2)) / scale_val
        alibi = self._get_bias(L)
        scores_data = scores_data + alibi.data
        if mask is not None:
            mask_data = mask.data if isinstance(mask, Tensor) else mask
            scores_data = xp.where(mask_data, scores_data, -1e9)

        # Softmax + V
        sm   = xp.exp(scores_data - scores_data.max(axis=-1, keepdims=True))
        sm  /= sm.sum(axis=-1, keepdims=True) + 1e-12
        attn_data = sm @ Vh.data   # (N, H, L, d_h)
        attn_out  = Tensor(attn_data, (Qh, Kh, Vh), 'ALiBiAttn', device=self._device)

        merged_data = attn_data.transpose(0, 2, 1, 3).reshape(N, L, D)
        merged = Tensor(merged_data, (attn_out,), 'MergeHeads', device=self._device)
        def _mh_bwd():
            if attn_out.grad is not None:
                attn_out.grad += merged.grad.reshape(N, L, H, d_h).transpose(0, 2, 1, 3)
        merged._backward = _mh_bwd

        out = merged @ self.w_o + self.b_o
        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), out)
            if hr is not None: out = hr
        return out

    def parameters(self):
        return [self.w_q, self.w_k, self.w_v, self.w_o, self.b_o]

    def __repr__(self):
        return f"ALiBiAttention(d_model={self.d_model}, nhead={self.nhead})"


# ==============================================================
# PHASE K — DATASET & DATALOADER
# ==============================================================

class Dataset:
    """Abstract base class for datasets.

    Subclasses must implement __len__ and __getitem__.

    Example:
        class MyDataset(Dataset):
            def __init__(self, X, y):
                self.X, self.y = X, y
            def __len__(self):
                return len(self.X)
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]
    """

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class TensorDataset(Dataset):
    """Dataset wrapping in-memory Tensors. Each call to __getitem__ slices all tensors."""

    def __init__(self, *tensors):
        if not tensors:
            raise ValueError("TensorDataset requires at least one tensor")
        n = len(tensors[0])
        if not all(len(t) == n for t in tensors):
            raise ValueError("All tensors must have the same first dimension size")
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, idx):
        items = []
        for t in self.tensors:
            if isinstance(t, Tensor):
                items.append(Tensor(t.data[idx], device=t._device, requires_grad=False))
            else:
                items.append(t[idx])
        return tuple(items) if len(items) > 1 else items[0]


class ConcatDataset(Dataset):
    """Concatenation of multiple Dataset objects."""

    def __init__(self, datasets):
        self.datasets = list(datasets)
        self._cumlen  = []
        total = 0
        for d in self.datasets:
            total += len(d)
            self._cumlen.append(total)

    def __len__(self):
        return self._cumlen[-1] if self._cumlen else 0

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + idx
        for i, clen in enumerate(self._cumlen):
            if idx < clen:
                local = idx - (self._cumlen[i - 1] if i > 0 else 0)
                return self.datasets[i][local]
        raise IndexError(f"Index {idx} out of range for ConcatDataset of size {len(self)}")


def _default_collate(batch):
    """Default collate_fn: stack numpy arrays / Tensors, pass through others."""
    if not batch:
        return batch
    elem = batch[0]
    if isinstance(elem, Tensor):
        stacked = np.stack([b.data if b._device == 'cpu' else cp.asnumpy(b.data)
                            for b in batch], axis=0)
        dev = batch[0]._device
        xp  = cp if dev == 'cuda' else np
        return Tensor(xp.array(stacked), device=dev, requires_grad=False)
    elif isinstance(elem, np.ndarray):
        return np.stack(batch, axis=0)
    elif isinstance(elem, (int, float)):
        return np.array(batch)
    elif isinstance(elem, (list, tuple)):
        return type(elem)(_default_collate(list(x)) for x in zip(*batch))
    else:
        return batch


class DataLoader:
    """Iterable DataLoader with batching, shuffling, and drop_last support.

    Args:
        dataset:       A Dataset instance.
        batch_size:    Number of samples per batch.
        shuffle:       Randomly shuffle indices each epoch.
        drop_last:     Drop the last incomplete batch if dataset size % batch_size != 0.
        collate_fn:    Function to merge a list of samples into a batch.
                       Defaults to _default_collate.
        pin_memory:    Copy CPU arrays to pinned memory before yielding (CUDA only).
                       Currently a no-op placeholder for API compatibility.

    Usage:
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        for batch_x, batch_y in loader:
            ...
    """

    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False,
                 drop_last: bool = False, collate_fn=None, pin_memory: bool = False):
        self.dataset    = dataset
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.drop_last  = drop_last
        self.collate_fn = collate_fn or _default_collate
        self.pin_memory = pin_memory  # reserved for future use

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)
        indices = np.random.permutation(n) if self.shuffle else np.arange(n)

        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            if self.drop_last and end > n:
                break
            batch_indices = indices[start:min(end, n)]
            samples = [self.dataset[int(i)] for i in batch_indices]
            yield self.collate_fn(samples)

    def __repr__(self):
        return (f"DataLoader(dataset={self.dataset.__class__.__name__}, "
                f"batch_size={self.batch_size}, shuffle={self.shuffle})")


if get_device() == 'cuda':
    init_streams(4)
