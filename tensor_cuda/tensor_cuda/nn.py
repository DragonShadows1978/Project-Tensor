"""Neural-network layers — a thin Python wrapper over the C++/CUDA engine.

Modules are parameter containers; all compute (matmul, activations, embedding,
norm math) runs in C++ kernels via `tensor_cuda` ops. This mirrors the reference
`tensor_gpu_v2._nn` API.
"""

from __future__ import annotations

import math

import numpy as np

import tensor_cuda as tc


def parameter(array, *, device="cuda", dtype="float32"):
    return tc.tensor(array, device=device, dtype=dtype, requires_grad=True)


class Module:
    def __init__(self):
        object.__setattr__(self, "_params", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        if isinstance(value, tc.Tensor):
            if value.requires_grad:
                self._params[name] = value
            else:
                self._buffers[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def parameters(self):
        out = list(self._params.values())
        for m in self._modules.values():
            out.extend(m.parameters())
        return out

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode=True):
        object.__setattr__(self, "training", mode)
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def named_parameters(self, prefix=""):
        for n, p in self._params.items():
            yield prefix + n, p
        for n, m in self._modules.items():
            yield from m.named_parameters(prefix + n + ".")

    def state_dict(self, prefix=""):
        d = {}
        for n, p in self._params.items():
            d[prefix + n] = p.numpy()
        for n, b in self._buffers.items():
            d[prefix + n] = b.numpy()
        for n, m in self._modules.items():
            d.update(m.state_dict(prefix + n + "."))
        return d

    def load_state_dict(self, sd, prefix=""):
        for n in list(self._params):
            setattr(self, n, tc.tensor(sd[prefix + n], requires_grad=True))
        for n in list(self._buffers):
            setattr(self, n, tc.tensor(sd[prefix + n]))
        for n, m in self._modules.items():
            m.load_state_dict(sd, prefix + n + ".")

    def half(self):
        """Cast all parameters to fp16 (for mixed-precision training)."""
        for n in list(self._params):
            setattr(self, n, tc.tensor(self._params[n].numpy().astype("float16"),
                                       dtype="float16", requires_grad=True))
        for m in self._modules.values():
            m.half()
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        bound = 1.0 / math.sqrt(in_features)
        self.weight = parameter(
            np.random.uniform(-bound, bound, (out_features, in_features)))
        self.bias = parameter(np.zeros(out_features)) if bias else None

    def forward(self, x):
        w = self.weight if self.weight.dtype == x.dtype else self.weight.astype(x.dtype)
        out = tc.matmul(x, w, trans_b=True)
        if self.bias is not None:
            b = self.bias if self.bias.dtype == out.dtype else self.bias.astype(out.dtype)
            out = out + b
        return out


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = parameter(
            np.random.randn(num_embeddings, embedding_dim) * 0.02)

    def forward(self, idx):
        return tc.embedding(self.weight, idx)


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        dim = normalized_shape if isinstance(normalized_shape, int) else int(np.prod(normalized_shape))
        self.weight = parameter(np.ones(dim))
        self.bias = parameter(np.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean([-1], True)
        xc = x - mean
        var = (xc * xc).mean([-1], True)
        inv = (var + self.eps).pow(-0.5)
        return xc * inv * self.weight + self.bias


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (np.random.rand(*x.shape) > self.p).astype(np.float32) / (1 - self.p)
        return x * tc.tensor(mask, device=x.device.split(":")[0])


class ReLU(Module):
    def forward(self, x): return x.relu()


class GELU(Module):
    def forward(self, x): return x.gelu()


class SiLU(Module):
    def forward(self, x): return x.silu()


class Sigmoid(Module):
    def forward(self, x): return x.sigmoid()


class Tanh(Module):
    def forward(self, x): return x.tanh()


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.alpha = negative_slope

    def forward(self, x):
        return x.maximum(x * self.alpha)


class ELU(Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        pos = x.__gt__(0.0)
        return tc.where(pos, x, (x.exp().__add__(-1.0)) * self.alpha)


class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x): return x.softmax(self.dim)


class GeGLU(Module):
    """Gated GELU: split a 2*d projection into value and gate."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = Linear(in_dim, 2 * out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        h = self.proj(x)
        a = h.slice(-1, 0, self.out_dim)
        b = h.slice(-1, self.out_dim, self.out_dim)
        return a * b.gelu()


class FusedLinearGELU(Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.lin = Linear(in_f, out_f)

    def forward(self, x): return self.lin(x).gelu()


class FusedLinearSiLU(Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.lin = Linear(in_f, out_f)

    def forward(self, x): return self.lin(x).silu()


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
        for i, l in enumerate(layers):
            self._modules[str(i)] = l

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        self._list = []
        for m in (modules or []):
            self.append(m)

    def append(self, m):
        self._modules[str(len(self._list))] = m
        self._list.append(m)
        return self

    def __iter__(self):
        return iter(self._list)

    def __getitem__(self, i):
        return self._list[i]

    def __len__(self):
        return len(self._list)


class ModuleDict(Module):
    def __init__(self, modules=None):
        super().__init__()
        for k, m in (modules or {}).items():
            self._modules[k] = m
            object.__setattr__(self, k, m)

    def __getitem__(self, k):
        return self._modules[k]


class RMSNorm(Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = parameter(np.ones(dim))
        self.eps = eps

    def forward(self, x):
        ms = (x * x).mean([-1], True)
        w = self.weight if self.weight.dtype == x.dtype else self.weight.astype(x.dtype)
        return x * (ms + self.eps).pow(-0.5) * w


class MultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = Linear(embed_dim, embed_dim, bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias)

    def _split(self, x, B, L):
        return x.reshape([B, L, self.num_heads, self.head_dim]).transpose(1, 2)

    def forward(self, x, kv=None, attn_mask=None, is_causal=False):
        from . import functional as F
        kv = x if kv is None else kv
        B, L, _ = x.shape
        S = kv.shape[1]
        q = self._split(self.q_proj(x), B, L)
        k = self._split(self.k_proj(kv), B, S)
        v = self._split(self.v_proj(kv), B, S)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask, is_causal)
        out = out.transpose(1, 2).reshape([B, L, self.embed_dim])
        return self.out_proj(out)


class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, activation="gelu"):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.act = activation

    def _ff(self, x):
        h = self.linear1(x)
        h = h.gelu() if self.act == "gelu" else h.relu()
        return self.linear2(h)

    def forward(self, x, attn_mask=None, is_causal=False):
        x = self.norm1(x + self.self_attn(x, attn_mask, is_causal))
        x = self.norm2(x + self._ff(x))
        return x


# ------------------------------------------------------------- convolution
def _pair(x):
    return x if isinstance(x, (tuple, list)) else (x, x)


class Conv2D(Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.kh, self.kw = _pair(kernel_size)
        self.sh, self.sw = _pair(stride)
        self.ph, self.pw = _pair(padding)
        fan_in = in_ch * self.kh * self.kw
        bound = 1.0 / math.sqrt(fan_in)
        self.weight = parameter(np.random.uniform(
            -bound, bound, (out_ch, in_ch, self.kh, self.kw)))
        self.bias = parameter(np.zeros(out_ch)) if bias else None

    def forward(self, x):
        N = x.shape[0]
        K = self.in_ch * self.kh * self.kw
        cols = tc._C.im2col(x, self.kh, self.kw, self.sh, self.sw, self.ph, self.pw)
        L = cols.shape[-1]
        W2 = self.weight.reshape([self.out_ch, K])
        out = tc.matmul(cols.transpose(1, 2), W2.transpose(0, 1))  # (N, L, out)
        OH = (x.shape[2] + 2 * self.ph - self.kh) // self.sh + 1
        OW = (x.shape[3] + 2 * self.pw - self.kw) // self.sw + 1
        out = out.transpose(1, 2).reshape([N, self.out_ch, OH, OW])
        if self.bias is not None:
            out = out + self.bias.reshape([1, self.out_ch, 1, 1])
        return out


class DepthwiseConv2D(Module):
    """Per-channel conv (groups == channels), composed from im2col + reduce."""
    def __init__(self, channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.C = channels
        self.kh, self.kw = _pair(kernel_size)
        self.sh, self.sw = _pair(stride)
        self.ph, self.pw = _pair(padding)
        bound = 1.0 / math.sqrt(self.kh * self.kw)
        self.weight = parameter(np.random.uniform(-bound, bound, (channels, self.kh * self.kw)))
        self.bias = parameter(np.zeros(channels)) if bias else None

    def forward(self, x):
        N, C, H, W = x.shape
        cols = tc._C.im2col(x, self.kh, self.kw, self.sh, self.sw, self.ph, self.pw)
        L = cols.shape[-1]
        cols = cols.reshape([N, C, self.kh * self.kw, L])
        w = self.weight.reshape([1, C, self.kh * self.kw, 1])
        out = (cols * w).sum([2], False)  # (N, C, L)
        OH = (H + 2 * self.ph - self.kh) // self.sh + 1
        OW = (W + 2 * self.pw - self.kw) // self.sw + 1
        out = out.reshape([N, C, OH, OW])
        if self.bias is not None:
            out = out + self.bias.reshape([1, C, 1, 1])
        return out


class SeparableConv2D(Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depth = DepthwiseConv2D(in_ch, kernel_size, stride, padding)
        self.point = Conv2D(in_ch, out_ch, 1)

    def forward(self, x):
        return self.point(self.depth(x))


class ConvTranspose2D(Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.kh, self.kw = _pair(kernel_size)
        self.sh, self.sw = _pair(stride)
        self.ph, self.pw = _pair(padding)
        bound = 1.0 / math.sqrt(out_ch * self.kh * self.kw)
        self.weight = parameter(np.random.uniform(
            -bound, bound, (in_ch, out_ch, self.kh, self.kw)))
        self.bias = parameter(np.zeros(out_ch)) if bias else None

    def forward(self, x):
        N, Cin, H, W = x.shape
        Wt = self.weight.reshape([self.in_ch, self.out_ch * self.kh * self.kw])
        x2 = x.reshape([N, Cin, H * W]).transpose(1, 2)        # (N, HW, Cin)
        oc = tc.matmul(x2, Wt).transpose(1, 2)                 # (N, Cout*kh*kw, HW)
        OH = (H - 1) * self.sh - 2 * self.ph + self.kh
        OW = (W - 1) * self.sw - 2 * self.pw + self.kw
        out = tc._C.col2im(oc, N, self.out_ch, OH, OW,
                           self.kh, self.kw, self.sh, self.sw, self.ph, self.pw)
        if self.bias is not None:
            out = out + self.bias.reshape([1, self.out_ch, 1, 1])
        return out


class MaxPool2D(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kh, self.kw = _pair(kernel_size)
        s = stride if stride is not None else kernel_size
        self.sh, self.sw = _pair(s)
        self.ph, self.pw = _pair(padding)

    def forward(self, x):
        return tc._C.max_pool2d(x, self.kh, self.kw, self.sh, self.sw, self.ph, self.pw)


class AvgPool2D(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kh, self.kw = _pair(kernel_size)
        s = stride if stride is not None else kernel_size
        self.sh, self.sw = _pair(s)
        self.ph, self.pw = _pair(padding)

    def forward(self, x):
        return tc._C.avg_pool2d(x, self.kh, self.kw, self.sh, self.sw, self.ph, self.pw)


class BatchNorm2D(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps, self.momentum, self.C = eps, momentum, num_features
        self.weight = parameter(np.ones(num_features))
        self.bias = parameter(np.zeros(num_features))
        self.running_mean = tc.zeros(num_features)      # buffers (no grad)
        self.running_var = tc.ones(num_features)

    def forward(self, x):
        w = self.weight.reshape([1, self.C, 1, 1])
        b = self.bias.reshape([1, self.C, 1, 1])
        if self.training:
            mean = x.mean([0, 2, 3], True)
            var = x.var([0, 2, 3], True)
            mn = mean.numpy().ravel(); vr = var.numpy().ravel()
            rm = self.running_mean.numpy(); rv = self.running_var.numpy()
            self.running_mean = tc.tensor((1 - self.momentum) * rm + self.momentum * mn)
            self.running_var = tc.tensor((1 - self.momentum) * rv + self.momentum * vr)
        else:
            mean = self.running_mean.reshape([1, self.C, 1, 1]).detach()
            var = self.running_var.reshape([1, self.C, 1, 1]).detach()
        return (x - mean) * (var + self.eps).pow(-0.5) * w + b


class BatchNorm1D(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps, self.momentum, self.C = eps, momentum, num_features
        self.weight = parameter(np.ones(num_features))
        self.bias = parameter(np.zeros(num_features))
        self.running_mean = tc.zeros(num_features)
        self.running_var = tc.ones(num_features)

    def forward(self, x):
        # x: (N, C) or (N, C, L) -> normalize over N (and L)
        axes = [0] if x.ndim == 2 else [0, 2]
        wshape = [1, self.C] if x.ndim == 2 else [1, self.C, 1]
        w = self.weight.reshape(wshape); b = self.bias.reshape(wshape)
        if self.training:
            mean = x.mean(axes, True); var = x.var(axes, True)
            mn = mean.numpy().ravel(); vr = var.numpy().ravel()
            self.running_mean = tc.tensor((1 - self.momentum) * self.running_mean.numpy() + self.momentum * mn)
            self.running_var = tc.tensor((1 - self.momentum) * self.running_var.numpy() + self.momentum * vr)
        else:
            mean = self.running_mean.reshape(wshape).detach()
            var = self.running_var.reshape(wshape).detach()
        return (x - mean) * (var + self.eps).pow(-0.5) * w + b


class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.G, self.C, self.eps = num_groups, num_channels, eps
        self.weight = parameter(np.ones(num_channels))
        self.bias = parameter(np.zeros(num_channels))

    def forward(self, x):
        N, C = x.shape[0], x.shape[1]
        spatial = list(x.shape[2:])
        g = x.reshape([N, self.G, -1])
        mean = g.mean([2], True); var = g.var([2], True)
        g = (g - mean) * (var + self.eps).pow(-0.5)
        g = g.reshape([N, C] + spatial)
        wshape = [1, C] + [1] * len(spatial)
        return g * self.weight.reshape(wshape) + self.bias.reshape(wshape)


class InstanceNorm2D(Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.C, self.eps = num_features, eps
        self.weight = parameter(np.ones(num_features))
        self.bias = parameter(np.zeros(num_features))

    def forward(self, x):
        mean = x.mean([2, 3], True); var = x.var([2, 3], True)
        xn = (x - mean) * (var + self.eps).pow(-0.5)
        return xn * self.weight.reshape([1, self.C, 1, 1]) + self.bias.reshape([1, self.C, 1, 1])


class Conv1D(Module):
    """1D conv implemented as a height-1 Conv2D."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = Conv2D(in_ch, out_ch, (1, kernel_size), (1, stride), (0, padding), bias)

    def forward(self, x):  # x: (N, C, L)
        N, C, L = x.shape
        out = self.conv(x.reshape([N, C, 1, L]))
        return out.reshape([N, out.shape[1], out.shape[3]])


class AdaptiveAvgPool2D(Module):
    def __init__(self, output_size):
        super().__init__()
        self.out = output_size if isinstance(output_size, (tuple, list)) else (output_size, output_size)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        oh, ow = self.out
        if oh == 1 and ow == 1:
            return x.mean([2, 3], True)
        kh, kw = H // oh, W // ow
        return tc._C.avg_pool2d(x, kh, kw, kh, kw, 0, 0)


# ------------------------------------------------------------- recurrent
class RNNCell(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.ih = Linear(input_size, hidden_size)
        self.hh = Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        return (self.ih(x) + self.hh(h)).tanh()


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.ih = Linear(input_size, 4 * hidden_size)
        self.hh = Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, state):
        h, c = state
        g = self.ih(x) + self.hh(h)
        H = self.hidden_size
        i = g.slice(-1, 0, H).sigmoid()
        f = g.slice(-1, H, H).sigmoid()
        gg = g.slice(-1, 2 * H, H).tanh()
        o = g.slice(-1, 3 * H, H).sigmoid()
        c2 = f * c + i * gg
        h2 = o * c2.tanh()
        return h2, c2


class GRUCell(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.ih = Linear(input_size, 3 * hidden_size)
        self.hh = Linear(hidden_size, 3 * hidden_size)

    def forward(self, x, h):
        H = self.hidden_size
        xi, hi = self.ih(x), self.hh(h)
        r = (xi.slice(-1, 0, H) + hi.slice(-1, 0, H)).sigmoid()
        z = (xi.slice(-1, H, H) + hi.slice(-1, H, H)).sigmoid()
        n = (xi.slice(-1, 2 * H, H) + r * hi.slice(-1, 2 * H, H)).tanh()
        return (z * -1.0 + 1.0) * n + z * h


class _RNNBase(Module):
    """Iterates a cell over the time dimension (input (B, T, input_size))."""
    cell_cls = None
    has_cell_state = False

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = self.cell_cls(input_size, hidden_size)

    def forward(self, x):
        B, T, _ = x.shape
        dev = x.device.split(":")[0]
        h = tc.zeros(B, self.hidden_size, device=dev)
        c = tc.zeros(B, self.hidden_size, device=dev) if self.has_cell_state else None
        outs = []
        for t in range(T):
            xt = x.slice(1, t, 1).reshape([B, x.shape[2]])
            if self.has_cell_state:
                h, c = self.cell(xt, (h, c))
            else:
                h = self.cell(xt, h)
            outs.append(h)
        return tc.stack(outs, dim=1)  # (B, T, hidden)


class RNN(_RNNBase):
    cell_cls = RNNCell


class LSTM(_RNNBase):
    cell_cls = LSTMCell
    has_cell_state = True


class GRU(_RNNBase):
    cell_cls = GRUCell


# ------------------------------------------------------------- losses
class MSELoss(Module):
    def forward(self, pred, target): return tc.mse_loss(pred, target)


class L1Loss(Module):
    def forward(self, pred, target):
        return (pred - target).abs().mean()


class BCEWithLogitsLoss(Module):
    def forward(self, x, y):
        # max(x,0) - x*y + log(1 + exp(-|x|))
        term = x.relu() - x * y + (x.abs() * -1.0).exp().__add__(1.0).log()
        return term.mean()


class SmoothL1Loss(Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        d = (pred - target).abs()
        quad = d.pow(2.0) * (0.5 / self.beta)
        lin = d - 0.5 * self.beta
        small = d.__lt__(self.beta)  # detached 0/1
        return tc.where(small, quad, lin).mean()


class CosineEmbeddingLoss(Module):
    def __init__(self, margin=0.0, eps=1e-8):
        super().__init__()
        self.margin, self.eps = margin, eps

    def forward(self, x1, x2, y):
        dot = (x1 * x2).sum([-1])
        n1 = (x1 * x1).sum([-1]).pow(0.5)
        n2 = (x2 * x2).sum([-1]).pow(0.5)
        cos = dot / (n1 * n2 + self.eps)
        pos = cos * -1.0 + 1.0                      # 1 - cos
        neg = (cos + (-self.margin)).relu()         # max(0, cos - margin)
        mask = y.__gt__(0.0)                        # y == +1
        return tc.where(mask, pos, neg).mean()


class TripletMarginLoss(Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_ap = ((anchor - positive).pow(2.0)).sum([-1]).pow(0.5)
        d_an = ((anchor - negative).pow(2.0)).sum([-1]).pow(0.5)
        return (d_ap - d_an + self.margin).relu().mean()


class KLDivLoss(Module):
    """input = log-probabilities, target = probabilities (batchmean)."""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, log_q, p):
        term = p * ((p + self.eps).log() - log_q)
        return term.sum([-1]).mean()


class PositionalEncoding(Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = np.zeros((max_len, d_model), np.float32)
        pos = np.arange(max_len)[:, None]
        div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        self.pe = tc.tensor(pe)  # buffer (requires_grad=False)
        self.d_model = d_model

    def forward(self, x):
        L = x.shape[1]
        return x + self.pe.slice(0, 0, L).reshape([1, L, self.d_model])


class TransformerDecoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, activation="gelu"):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead)
        self.cross_attn = MultiheadAttention(d_model, nhead)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.act = activation

    def forward(self, x, memory, tgt_causal=True):
        x = self.norm1(x + self.self_attn(x, is_causal=tgt_causal))
        x = self.norm2(x + self.cross_attn(x, kv=memory))
        h = self.linear1(x)
        h = h.gelu() if self.act == "gelu" else h.relu()
        x = self.norm3(x + self.linear2(h))
        return x


class CrossEntropyLoss(Module):
    def forward(self, logits, target): return tc.cross_entropy(logits, target)
