"""Neural network modules, losses, optimizers, and gradient tools.

This module is part of the ``tensor_gpu_v2`` package.
Import via ``import tensor_gpu_v2 as tg``, not directly.
"""

from ._core import *                   # Tensor, get_device, no_grad, …
from ._core import _get_cached_kernel  # private kernel helper used by ops

# ==================== NEURAL NETWORK MODULES ====================

class _HookHandle:
    """Returned by register_*_hook; call .remove() to deregister."""
    __slots__ = ('_hooks', '_key')

    def __init__(self, hooks_dict, key):
        self._hooks = hooks_dict
        self._key = key

    def remove(self):
        self._hooks.pop(self._key, None)


class Module:
    """Base class for neural network modules."""

    def __init__(self):
        self._device = get_device()
        self._training = True
        self._forward_hooks = {}
        self._backward_hooks = {}
        self._hook_counter = 0

    def register_forward_hook(self, hook) -> _HookHandle:
        """Register a hook called after every forward pass.

        hook(module, input_tuple, output) -> None or modified output
        """
        key = self._hook_counter
        self._hook_counter += 1
        self._forward_hooks[key] = hook
        return _HookHandle(self._forward_hooks, key)

    def register_backward_hook(self, hook) -> _HookHandle:
        """Register a hook called after every backward pass on the output.

        hook(module, grad_input, grad_output) -> None
        Grad tensors are raw numpy/cupy arrays, not Tensor objects.
        """
        key = self._hook_counter
        self._hook_counter += 1
        self._backward_hooks[key] = hook
        return _HookHandle(self._backward_hooks, key)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def zero_grad(self):
        """Zero out gradients of all parameters."""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.fill(0)

    def parameters(self) -> List[Tensor]:
        """Return a list of all parameters in this module and its submodules."""
        params = []
        for _, p in self.named_parameters():
            params.append(p)
        return params

    def named_parameters(self) -> List[Tuple[str, Tensor]]:
        """Return an iterator over module parameters, yielding both name and parameter."""
        params = []
        # Use a set to avoid duplicates (e.g. if same parameter is assigned to multiple names)
        seen = set()
        
        for name in dir(self):
            if name.startswith('_'):
                continue
            try:
                attr = getattr(self, name)
                if isinstance(attr, Tensor) and attr._requires_grad:
                    if id(attr) not in seen:
                        params.append((name, attr))
                        seen.add(id(attr))
                elif isinstance(attr, Module) and attr is not self:
                    for child_name, child_param in attr.named_parameters():
                        full_name = f"{name}.{child_name}"
                        if id(child_param) not in seen:
                            params.append((full_name, child_param))
                            seen.add(id(child_param))
            except AttributeError:
                continue
        return params

    def named_modules(self, memo: set = None, prefix: str = '') -> List[Tuple[str, 'Module']]:
        """Return an iterator over all modules in the network, yielding both name and module."""
        if memo is None:
            memo = set()
        
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name in dir(self):
                if name.startswith('_'):
                    continue
                try:
                    attr = getattr(self, name)
                    if isinstance(attr, Module) and attr is not self:
                        submodule_prefix = prefix + ('.' if prefix else '') + name
                        for m in attr.named_modules(memo, submodule_prefix):
                            yield m
                except AttributeError:
                    continue

    def train(self):
        self._training = True
        for name in dir(self):
            attr = getattr(self, name)
            if isinstance(attr, Module):
                attr.train()
        return self

    def eval(self):
        self._training = False
        for name in dir(self):
            attr = getattr(self, name)
            if isinstance(attr, Module):
                attr.eval()
        return self

    def to(self, device: str):
        self._device = device
        # Move parameters of this module
        for p in self.parameters():
            new_p = p.to(device)
            p.data = new_p.data
            p.grad = new_p.grad
            p._device = device

        # Move registered buffers
        buffers = getattr(self, '_buffers', {})
        for buf_name, buf in list(buffers.items()):
            if buf is None:
                continue
            xp_new = cp if device == 'cuda' else np
            new_buf = xp_new.array(to_cpu(buf), dtype=buf.dtype)
            buffers[buf_name] = new_buf
            setattr(self, buf_name, new_buf)

        # Recursively move submodules
        for name in dir(self):
            if name.startswith('_'):
                continue
            try:
                attr = getattr(self, name)
                if isinstance(attr, Module) and attr is not self:
                    attr.to(device)
                elif isinstance(attr, (list, tuple)):
                    for item in attr:
                        if isinstance(item, Module):
                            item.to(device)
                elif isinstance(attr, dict):
                    for item in attr.values():
                        if isinstance(item, Module):
                            item.to(device)
            except AttributeError:
                continue
        return self

    def cuda(self):
        return self.to('cuda')

    def cpu(self):
        return self.to('cpu')

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def state_dict(self) -> dict:
        """Return state dictionary for saving."""
        state = {}
        for i, p in enumerate(self.parameters()):
            state[f'param_{i}'] = to_cpu(p.data)
        return state

    def load_state_dict(self, state: dict):
        """Load state dictionary."""
        xp = cp if self._device == 'cuda' else np
        params = self.parameters()
        for i, p in enumerate(params):
            key = f'param_{i}'
            if key in state:
                p.data = xp.array(state[key], dtype=p.data.dtype)
        for name, buf in getattr(self, '_buffers', {}).items():
            key = f'buffer_{name}'
            if key in state and buf is not None:
                buf[:] = (cp if self._device == 'cuda' else np).array(state[key])

    def register_buffer(self, name: str, tensor):
        """Register non-parameter state (moves with device, saved in state_dict)."""
        if not hasattr(self, '_buffers'):
            object.__setattr__(self, '_buffers', {})
        data = tensor.data if isinstance(tensor, Tensor) else tensor
        self._buffers[name] = data
        setattr(self, name, data)

    def apply(self, fn):
        """Recursively apply fn to every submodule (including self)."""
        for _, module in self.named_modules():
            fn(module)
        return self

    def requires_grad_(self, requires_grad: bool = True):
        """Set requires_grad for all parameters in this module."""
        for p in self.parameters():
            p._requires_grad = requires_grad
            if requires_grad and p.grad is None:
                xp = cp if p._device == 'cuda' else np
                p.grad = xp.zeros_like(p.data)
            elif not requires_grad:
                p.grad = None
        return self

    def freeze(self):
        """Disable gradients for all parameters (for transfer learning)."""
        return self.requires_grad_(False)

    def unfreeze(self):
        """Re-enable gradients for all parameters."""
        return self.requires_grad_(True)

    def summary(self, input_shape=None):
        """Print per-layer parameter count table."""
        lines = [f"{'Layer':<40} {'Type':<25} {'Params':>10}", "-" * 77]
        total = 0
        for name, mod in self.named_modules():
            if mod is self:
                continue
            direct = 0
            for attr_name in dir(mod):
                if attr_name.startswith('_'):
                    continue
                try:
                    attr = getattr(mod, attr_name)
                    if isinstance(attr, Tensor) and attr._requires_grad:
                        direct += attr.data.size
                except Exception:
                    pass
            total += direct
            lines.append(f"{name or '(root)':<40} {mod.__class__.__name__:<25} {direct:>10,}")
        lines.append("-" * 77)
        lines.append(f"{'Total trainable parameters':<65} {total:>10,}")
        print("\n".join(lines))
        return total


class Linear(Module):
    """Fully connected linear layer."""

    def __init__(self, nin, nout, bias=True):
        super().__init__()
        xp = cp if get_device() == 'cuda' else np
        scale = xp.sqrt(2.0 / nin)
        self.w = Tensor(xp.random.randn(nin, nout).astype(xp.float32) * scale, device=get_device())
        self.use_bias = bias
        if bias:
            self.b = Tensor(xp.zeros(nout, dtype=xp.float32), device=get_device())
        else:
            self.b = None

    def __call__(self, x):
        out = x @ self.w
        if self.use_bias:
            out = out + self.b
        return out

    def parameters(self):
        if self.use_bias:
            return [self.w, self.b]
        return [self.w]

    def __repr__(self):
        nin, nout = self.w.data.shape
        return f"Linear(in_features={nin}, out_features={nout}, bias={self.use_bias})"


class Conv2D(Module):
    """
    2D Convolutional layer using im2col.

    Supports both NCHW (default) and NHWC data formats.
    NHWC can be more efficient on modern GPU tensor cores.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 groups=1, bias=True, data_format='NCHW'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.use_bias = bias
        self.data_format = data_format

        if data_format not in ('NCHW', 'NHWC'):
            raise ValueError(f"data_format must be 'NCHW' or 'NHWC', got {data_format}")

        xp = cp if get_device() == 'cuda' else np
        scale = xp.sqrt(2.0 / (in_channels * kernel_size * kernel_size))

        if groups == 1:
            self.w = Tensor(xp.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(xp.float32) * scale, device=get_device())
        else:
            # Grouped convolution: weight shape is (out_channels, in_channels/groups, kernel_size, kernel_size)
            self.w = Tensor(xp.random.randn(out_channels, in_channels // groups, kernel_size, kernel_size).astype(xp.float32) * scale, device=get_device())

        if bias:
            self.b = Tensor(xp.zeros((out_channels,), dtype=xp.float32), device=get_device())
        else:
            self.b = None

    def __call__(self, x):
        xp = cp if self._device == 'cuda' else np

        # Handle NHWC input by converting to NCHW, computing, and converting back
        if self.data_format == 'NHWC':
            N, H, W, C = x.data.shape
            x_nchw_data = x.data.transpose(0, 3, 1, 2)
            # Preserve input dtype when bridging NHWC <-> NCHW.
            x_nchw = Tensor(
                x_nchw_data,
                device=self._device,
                requires_grad=x._requires_grad,
                dtype=x.data.dtype,
            )
            out_nchw = self._conv_standard(x_nchw) if self.groups == 1 else self._conv_grouped(x_nchw)
            out_data = out_nchw.data.transpose(0, 2, 3, 1)
            out = Tensor(out_data, (x,), 'Conv2D_NHWC', device=self._device, dtype=out_data.dtype)
            def _nhwc_backward():
                if out.grad is not None:
                    out_nchw.grad = out.grad.transpose(0, 3, 1, 2)
                    out_nchw._backward()
                    if x.grad is not None:
                        x.grad += x_nchw.grad.transpose(0, 2, 3, 1)
            out._backward = _nhwc_backward
            return out

        if self.groups == 1:
            return self._conv_standard(x)
        else:
            return self._conv_grouped(x)

    def _conv_standard(self, x):
        xp = cp if self._device == 'cuda' else np

        FN, C, HH, WW = self.w.data.shape
        N, C_in, H, W = x.data.shape

        H_out = (H + 2 * self.padding - HH) // self.stride + 1
        W_out = (W + 2 * self.padding - WW) // self.stride + 1

        x_cols = im2col_indices(x.data, HH, WW, self.padding, self.stride)
        w_col = self.w.data.reshape(FN, -1)

        out_col = w_col @ x_cols
        if self.use_bias:
            out_col = out_col + self.b.data.reshape(-1, 1)

        out_data = out_col.reshape(FN, H_out, W_out, N).transpose(3, 0, 1, 2)

        children = (x, self.w) if not self.use_bias else (x, self.w, self.b)
        out = Tensor(out_data, children, 'Conv2D', device=self._device)

        def _backward():
            dout = out.grad.transpose(1, 2, 3, 0).reshape(FN, -1)

            if self.use_bias:
                self.b.grad += xp.sum(dout, axis=1)

            dw_col = dout @ x_cols.T
            self.w.grad += dw_col.reshape(self.w.data.shape)

            dx_col = w_col.T @ dout
            dx_data = col2im_indices(dx_col, x.data.shape, HH, WW, self.padding, self.stride)
            x.grad += dx_data

        out._backward = _backward
        return out

    def _conv_grouped(self, x):
        """Optimized grouped convolution - batched matmul, single im2col."""
        xp = cp if self._device == 'cuda' else np

        N, C_in, H, W = x.data.shape
        G = self.groups
        C_in_g = C_in // G
        C_out_g = self.out_channels // G
        K = C_in_g * self.w.data.shape[2] * self.w.data.shape[3]  # C_in/G * HH * WW

        FN, _, HH, WW = self.w.data.shape
        H_out = (H + 2 * self.padding - HH) // self.stride + 1
        W_out = (W + 2 * self.padding - WW) // self.stride + 1

        # Reshape input to merge groups into batch: (N, G, C_in/G, H, W) -> (N*G, C_in/G, H, W)
        x_grouped = x.data.reshape(N, G, C_in_g, H, W)
        x_grouped = x_grouped.transpose(0, 1, 2, 3, 4).reshape(N * G, C_in_g, H, W)

        # Single im2col for all groups (batched)
        x_cols = im2col_indices(x_grouped, HH, WW, self.padding, self.stride)  # (K, N*G*H_out*W_out)

        # Reshape x_cols: (K, N*G*H_out*W_out) -> (G, K, N*H_out*W_out)
        x_cols = x_cols.reshape(K, N * G, H_out * W_out)
        x_cols = x_cols.transpose(1, 0, 2)  # (N*G, K, HW)
        x_cols = x_cols.reshape(N, G, K, H_out * W_out)  # (N, G, K, HW)
        x_cols = x_cols.transpose(1, 2, 0, 3)  # (G, K, N, HW)
        x_cols = x_cols.reshape(G, K, N * H_out * W_out)  # (G, K, N*HW)

        # Reshape weights: (G*C_out/G, C_in/G, HH, WW) -> (G, C_out/G, K)
        w_grouped = self.w.data.reshape(G, C_out_g, K)

        # Batched matmul: (G, C_out/G, K) @ (G, K, N*HW) -> (G, C_out/G, N*HW)
        out_cols = xp.matmul(w_grouped, x_cols)

        # Reshape to output: (G, C_out/G, N*HW) -> (N, G*C_out/G, H_out, W_out)
        out_cols = out_cols.reshape(G, C_out_g, N, H_out, W_out)  # (G, C_out/G, N, H, W)
        out_cols = out_cols.transpose(2, 0, 1, 3, 4)  # (N, G, C_out/G, H, W)
        out_data = out_cols.reshape(N, self.out_channels, H_out, W_out)

        if self.use_bias:
            out_data = out_data + self.b.data.reshape(1, -1, 1, 1)

        children = (x, self.w) if not self.use_bias else (x, self.w, self.b)
        out = Tensor(out_data, children, 'GroupedConv2D', device=self._device)

        # Cache for backward
        x_cols_cache = x_cols

        def _backward():
            # Reshape output gradient: (N, C_out, H, W) -> (G, C_out/G, N*HW)
            dout = out.grad.reshape(N, G, C_out_g, H_out, W_out)  # (N, G, C_out/G, H, W)
            dout = dout.transpose(1, 2, 0, 3, 4)  # (G, C_out/G, N, H, W)
            dout_flat = dout.reshape(G, C_out_g, N * H_out * W_out)  # (G, C_out/G, N*HW)

            if self.use_bias:
                # db: sum over all positions -> (C_out,)
                self.b.grad += dout_flat.sum(axis=2).reshape(-1)

            # dw: (G, C_out/G, N*HW) @ (G, N*HW, K) -> (G, C_out/G, K) -> (G*C_out/G, C_in/G, HH, WW)
            dw = xp.matmul(dout_flat, x_cols_cache.transpose(0, 2, 1))  # (G, C_out/G, K)
            self.w.grad += dw.reshape(self.out_channels, C_in_g, HH, WW)

            # dx: (G, K, C_out/G) @ (G, C_out/G, N*HW) -> (G, K, N*HW)
            dx_cols = xp.matmul(w_grouped.transpose(0, 2, 1), dout_flat)  # (G, K, N*HW)

            # Reshape dx_cols back for col2im: (G, K, N*HW) -> (K, N*G*HW)
            dx_cols = dx_cols.reshape(G, K, N, H_out * W_out)  # (G, K, N, HW)
            dx_cols = dx_cols.transpose(1, 2, 0, 3)  # (K, N, G, HW)
            dx_cols = dx_cols.reshape(K, N * G * H_out * W_out)  # (K, N*G*HW)

            # col2im to get input gradient
            dx_grouped = col2im_indices(dx_cols, (N * G, C_in_g, H, W), HH, WW, self.padding, self.stride)

            # Reshape back: (N*G, C_in/G, H, W) -> (N, C_in, H, W)
            dx_grouped = dx_grouped.reshape(N, G, C_in_g, H, W)
            x.grad += dx_grouped.reshape(N, C_in, H, W)

        out._backward = _backward
        return out

    def parameters(self):
        if self.use_bias:
            return [self.w, self.b]
        return [self.w]

    def __repr__(self):
        return (f"Conv2D({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, groups={self.groups}, bias={self.use_bias})")


class DepthwiseConv2D(Conv2D):
    """Depthwise convolution (groups=in_channels)."""

    def __init__(self, channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__(channels, channels, kernel_size, stride, padding,
                        groups=channels, bias=bias)


class SeparableConv2D(Module):
    """Depthwise separable convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.depthwise = DepthwiseConv2D(in_channels, kernel_size, stride, padding, bias=False)
        self.pointwise = Conv2D(in_channels, out_channels, 1, bias=bias)

    def __call__(self, x):
        return self.pointwise(self.depthwise(x))

    def parameters(self):
        return self.depthwise.parameters() + self.pointwise.parameters()

    def __repr__(self):
        ic = self.depthwise.in_channels
        oc = self.pointwise.out_channels
        ks = self.depthwise.kernel_size
        s = self.depthwise.stride
        p = self.depthwise.padding
        return f"SeparableConv2D({ic}, {oc}, kernel_size={ks}, stride={s}, padding={p})"


class MaxPool2D(Module):
    """2D Max Pooling layer using im2col."""

    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size

    def __call__(self, x):
        xp = cp if self._device == 'cuda' else np
        N, C, H, W = x.data.shape
        HH, WW = self.kernel_size, self.kernel_size
        stride = self.stride

        H_out = (H - HH) // stride + 1
        W_out = (W - WW) // stride + 1

        # Use im2col for general pooling support
        x_reshaped = x.data.reshape(N * C, 1, H, W)
        x_col = im2col_indices(x_reshaped, HH, WW, padding=0, stride=stride)
        
        max_idx = xp.argmax(x_col, axis=0)
        out_data = xp.max(x_col, axis=0).reshape(H_out, W_out, N, C).transpose(2, 3, 0, 1)

        out = Tensor(out_data, (x,), 'MaxPool2D', device=self._device)

        def _backward():
            dout = out.grad.transpose(2, 3, 0, 1).reshape(-1)
            dcol = xp.zeros_like(x_col)
            # Efficiently distribute gradient to max positions
            if xp == cp:
                # CuPy optimization
                dcol[max_idx, cp.arange(max_idx.size)] = dout
            else:
                dcol[max_idx, np.arange(max_idx.size)] = dout
            
            dx = col2im_indices(dcol, (N * C, 1, H, W), HH, WW, padding=0, stride=stride)
            if x.grad is not None:
                x.grad += dx.reshape(N, C, H, W)

        out._backward = _backward
        return out

    def __repr__(self):
        return f"MaxPool2D(kernel_size={self.kernel_size}, stride={self.stride})"


class AvgPool2D(Module):
    """2D Average Pooling layer using im2col."""

    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size

    def __call__(self, x):
        xp = cp if self._device == 'cuda' else np
        N, C, H, W = x.data.shape
        HH, WW = self.kernel_size, self.kernel_size
        stride = self.stride

        H_out = (H - HH) // stride + 1
        W_out = (W - WW) // stride + 1

        x_reshaped = x.data.reshape(N * C, 1, H, W)
        x_col = im2col_indices(x_reshaped, HH, WW, padding=0, stride=stride)
        
        out_data = xp.mean(x_col, axis=0).reshape(H_out, W_out, N, C).transpose(2, 3, 0, 1)

        out = Tensor(out_data, (x,), 'AvgPool2D', device=self._device)

        def _backward():
            dout = out.grad.transpose(2, 3, 0, 1).reshape(-1)
            dcol = xp.ones_like(x_col) * (dout / (HH * WW))
            
            dx = col2im_indices(dcol, (N * C, 1, H, W), HH, WW, padding=0, stride=stride)
            if x.grad is not None:
                x.grad += dx.reshape(N, C, H, W)

        out._backward = _backward
        return out

    def __repr__(self):
        return f"AvgPool2D(kernel_size={self.kernel_size}, stride={self.stride})"


class AdaptiveAvgPool2D(Module):
    """Adaptive 2D Average Pooling using im2col for performance."""

    def __init__(self, output_size, data_format='NCHW'):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
        self.data_format = data_format
        if data_format not in ('NCHW', 'NHWC'):
            raise ValueError(f"data_format must be 'NCHW' or 'NHWC', got {data_format}")

    def __call__(self, x):
        if self.data_format == 'NHWC':
            # Fast path for Global Average Pooling (1, 1) in NHWC
            if self.output_size == (1, 1):
                xp = x.xp
                out_data = xp.mean(x.data, axis=(1, 2), keepdims=True)
                out = Tensor(out_data, (x,), 'GlobalAvgPool2D_NHWC', device=self._device)
                def _gap_backward():
                    if out.grad is not None:
                        h, w = x.data.shape[1], x.data.shape[2]
                        x.grad += xp.broadcast_to(out.grad / (h * w), x.data.shape)
                out._backward = _gap_backward
                return out

            N, H, W, C = x.data.shape
            x_nchw_data = x.data.transpose(0, 3, 1, 2)
            x_nchw = Tensor(x_nchw_data, device=self._device, requires_grad=x._requires_grad, dtype=x.data.dtype)
            
            out_nchw = self._pool_nchw(x_nchw)
            
            out_data = out_nchw.data.transpose(0, 2, 3, 1)
            out = Tensor(out_data, (x,), 'AdaptiveAvgPool2D_NHWC', device=self._device, dtype=out_data.dtype)
            
            def _nhwc_backward():
                if out.grad is not None:
                    out_nchw.grad = out.grad.transpose(0, 3, 1, 2)
                    out_nchw._backward()
                    if x.grad is not None:
                        x.grad += x_nchw.grad.transpose(0, 2, 3, 1)
            out._backward = _nhwc_backward
            return out
            
        return self._pool_nchw(x)

    def _pool_nchw(self, x):
        xp = cp if self._device == 'cuda' else np
        N, C, H, W = x.data.shape
        out_h, out_w = self.output_size

        # Compute window sizes
        stride_h = H // out_h
        stride_w = W // out_w
        kernel_h = H - (out_h - 1) * stride_h
        kernel_w = W - (out_w - 1) * stride_w

        # Use im2col for faster execution
        x_reshaped = x.data.reshape(N * C, 1, H, W)
        x_col = im2col_indices(x_reshaped, kernel_h, kernel_w, padding=0, stride=(stride_h, stride_w))
        
        out_data = xp.mean(x_col, axis=0).reshape(out_h, out_w, N, C).transpose(2, 3, 0, 1)

        out = Tensor(out_data, (x,), 'AdaptiveAvgPool2D', device=self._device)

        def _backward():
            if x.grad is None:
                return
            
            dout = out.grad.transpose(2, 3, 0, 1).reshape(-1)
            dcol = xp.ones_like(x_col) * (dout / (kernel_h * kernel_w))
            
            dx = col2im_indices(dcol, (N * C, 1, H, W), kernel_h, kernel_w, padding=0, stride=(stride_h, stride_w))
            x.grad += dx.reshape(N, C, H, W)

        out._backward = _backward
        return out

    def __repr__(self):
        return f"AdaptiveAvgPool2D(output_size={self.output_size})"


class BatchNorm2D(Module):
    """2D Batch Normalization."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        xp = cp if get_device() == 'cuda' else np
        self.gamma = Tensor(xp.ones(num_features, dtype=xp.float32), device=get_device())
        self.beta = Tensor(xp.zeros(num_features, dtype=xp.float32), device=get_device())

        self.running_mean = xp.zeros(num_features, dtype=xp.float32)
        self.running_var = xp.ones(num_features, dtype=xp.float32)
        self._training = True

    def __call__(self, x):
        xp = cp if self._device == 'cuda' else np
        N, C, H, W = x.data.shape
        m = N * H * W

        if self._training:
            mean = x.data.mean(axis=(0, 2, 3))
            var = x.data.var(axis=(0, 2, 3))
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        std_inv = 1.0 / xp.sqrt(var.reshape(1, C, 1, 1) + self.eps)
        x_centered = x.data - mean.reshape(1, C, 1, 1)
        x_norm = x_centered * std_inv

        out_data = self.gamma.data.reshape(1, C, 1, 1) * x_norm + self.beta.data.reshape(1, C, 1, 1)

        out = Tensor(out_data, (x, self.gamma, self.beta), 'BatchNorm2D', device=self._device)
        self._cache = (x_norm, x_centered, std_inv, m)

        def _backward():
            dout = out.grad
            x_norm_c, x_centered_c, std_inv_c, m_c = self._cache

            self.gamma.grad += (dout * x_norm_c).sum(axis=(0, 2, 3))
            self.beta.grad += dout.sum(axis=(0, 2, 3))

            gamma_r = self.gamma.data.reshape(1, C, 1, 1)
            dx_norm = dout * gamma_r
            dvar = (dx_norm * x_centered_c * (-0.5) * (std_inv_c ** 3)).sum(axis=(0, 2, 3), keepdims=True)
            dmean = (dx_norm * (-std_inv_c)).sum(axis=(0, 2, 3), keepdims=True)
            dmean += dvar * (-2.0 / m_c) * x_centered_c.sum(axis=(0, 2, 3), keepdims=True)

            dx = dx_norm * std_inv_c
            dx += dvar * (2.0 / m_c) * x_centered_c
            dx += dmean / m_c

            x.grad += dx

        out._backward = _backward
        return out

    def parameters(self):
        return [self.gamma, self.beta]

    def __repr__(self):
        return f"BatchNorm2D({self.num_features}, eps={self.eps}, momentum={self.momentum})"


class FusedBatchNormReLU(Module):
    """
    Fused BatchNorm + ReLU for optimized inference.

    Single pass: y = max(0, (x - mean) / sqrt(var + eps) * gamma + beta)

    In inference mode, uses precomputed fused scale/shift parameters
    with a custom CUDA kernel for maximum performance.

    Target: >= 1.5x speedup over sequential BatchNorm2D + ReLU
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        xp = cp if get_device() == 'cuda' else np
        self.gamma = Tensor(xp.ones(num_features, dtype=xp.float32), device=get_device())
        self.beta = Tensor(xp.zeros(num_features, dtype=xp.float32), device=get_device())

        self.running_mean = xp.zeros(num_features, dtype=xp.float32)
        self.running_var = xp.ones(num_features, dtype=xp.float32)
        self._training = True

        # Precomputed parameters for inference
        self._fused_scale = None
        self._fused_shift = None

    def _precompute_fused_params(self):
        """Precompute scale and shift for inference mode."""
        xp = cp if self._device == 'cuda' else np
        # scale = gamma / sqrt(var + eps)
        # shift = beta - gamma * mean / sqrt(var + eps)
        inv_std = 1.0 / xp.sqrt(self.running_var + self.eps)
        self._fused_scale = (self.gamma.data * inv_std).astype(xp.float32)
        self._fused_shift = (self.beta.data - self.gamma.data * self.running_mean * inv_std).astype(xp.float32)

    def __call__(self, x):
        xp = cp if self._device == 'cuda' else np
        N, C, H, W = x.data.shape
        m = N * H * W

        if self._training:
            # Training mode: standard BatchNorm + ReLU (compute batch stats)
            mean = x.data.mean(axis=(0, 2, 3))
            var = x.data.var(axis=(0, 2, 3))
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            std_inv = 1.0 / xp.sqrt(var.reshape(1, C, 1, 1) + self.eps)
            x_centered = x.data - mean.reshape(1, C, 1, 1)
            x_norm = x_centered * std_inv

            out_data = self.gamma.data.reshape(1, C, 1, 1) * x_norm + self.beta.data.reshape(1, C, 1, 1)
            # Apply ReLU
            out_data = xp.maximum(0, out_data)

            out = Tensor(out_data, (x, self.gamma, self.beta), 'FusedBatchNormReLU', device=self._device)
            self._cache = (x_norm, x_centered, std_inv, m, out_data)

            def _backward():
                dout = out.grad
                x_norm_c, x_centered_c, std_inv_c, m_c, out_data_c = self._cache

                # Gradient through ReLU
                dout = dout * (out_data_c > 0).astype(xp.float32)

                # Gradient through BatchNorm
                self.gamma.grad += (dout * x_norm_c).sum(axis=(0, 2, 3))
                self.beta.grad += dout.sum(axis=(0, 2, 3))

                gamma_r = self.gamma.data.reshape(1, C, 1, 1)
                dx_norm = dout * gamma_r
                dvar = (dx_norm * x_centered_c * (-0.5) * (std_inv_c ** 3)).sum(axis=(0, 2, 3), keepdims=True)
                dmean = (dx_norm * (-std_inv_c)).sum(axis=(0, 2, 3), keepdims=True)
                dmean += dvar * (-2.0 / m_c) * x_centered_c.sum(axis=(0, 2, 3), keepdims=True)

                dx = dx_norm * std_inv_c
                dx += dvar * (2.0 / m_c) * x_centered_c
                dx += dmean / m_c

                x.grad += dx

            out._backward = _backward
            return out

        # Inference mode: use fused kernel
        if self._fused_scale is None:
            self._precompute_fused_params()

        total_size = N * C * H * W
        spatial_size = H * W

        if self._device == 'cuda' and x.data.dtype == cp.float32:
            # Use custom CUDA kernel
            out_data = cp.empty_like(x.data)
            block_size = 256
            grid_size = (total_size + block_size - 1) // block_size

            _fused_bn_relu_kernel(
                (grid_size,), (block_size,),
                (x.data.ravel(), self._fused_scale, self._fused_shift,
                 out_data.ravel(), total_size, C, spatial_size)
            )
            out_data = out_data.reshape(N, C, H, W)
        else:
            # CPU fallback or non-FP32
            scale = self._fused_scale.reshape(1, C, 1, 1)
            shift = self._fused_shift.reshape(1, C, 1, 1)
            out_data = xp.maximum(0, x.data * scale + shift)

        return Tensor(out_data, device=self._device, requires_grad=False)

    def train(self):
        """Switch to training mode and invalidate fused params."""
        super().train()
        self._fused_scale = None
        self._fused_shift = None
        return self

    def eval(self):
        """Switch to eval mode and precompute fused params."""
        super().eval()
        self._precompute_fused_params()
        return self

    def parameters(self):
        return [self.gamma, self.beta]

    def __repr__(self):
        return f"FusedBatchNormReLU({self.num_features}, eps={self.eps}, momentum={self.momentum})"


class LayerNorm(Module):
    """Layer Normalization."""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps

        xp = cp if get_device() == 'cuda' else np
        size = int(xp.prod(xp.array(normalized_shape)))
        self.gamma = Tensor(xp.ones(size, dtype=xp.float32), device=get_device())
        self.beta = Tensor(xp.zeros(size, dtype=xp.float32), device=get_device())

    def __call__(self, x):
        xp = cp if self._device == 'cuda' else np

        # Normalize over last len(normalized_shape) dimensions
        norm_dims = tuple(range(-len(self.normalized_shape), 0))

        mean = x.data.mean(axis=norm_dims, keepdims=True)
        var = x.data.var(axis=norm_dims, keepdims=True)

        x_norm = (x.data - mean) / xp.sqrt(var + self.eps)

        # Reshape gamma and beta to broadcast correctly
        shape = [1] * (x.ndim - len(self.normalized_shape)) + list(self.normalized_shape)
        gamma = self.gamma.data.reshape(shape)
        beta = self.beta.data.reshape(shape)

        out_data = gamma * x_norm + beta
        out = Tensor(out_data, (x, self.gamma, self.beta), 'LayerNorm', device=self._device)

        self._cache = (x_norm, mean, var, norm_dims)

        def _backward():
            x_norm_c, mean_c, var_c, dims = self._cache
            dout = out.grad

            n = 1
            for d in dims:
                n *= x.data.shape[d]

            shape = [1] * (x.ndim - len(self.normalized_shape)) + list(self.normalized_shape)
            gamma_reshaped = self.gamma.data.reshape(shape)

            self.gamma.grad += (dout * x_norm_c).sum(axis=tuple(range(x.ndim - len(self.normalized_shape))))
            self.beta.grad += dout.sum(axis=tuple(range(x.ndim - len(self.normalized_shape))))

            dx_norm = dout * gamma_reshaped
            std_inv = 1.0 / xp.sqrt(var_c + self.eps)

            dx = dx_norm * std_inv
            dx -= dx_norm.mean(axis=dims, keepdims=True) * std_inv
            dx -= x_norm_c * (dx_norm * x_norm_c).mean(axis=dims, keepdims=True) * std_inv

            x.grad += dx

        out._backward = _backward
        return out

    def parameters(self):
        return [self.gamma, self.beta]

    def __repr__(self):
        return f"LayerNorm({self.normalized_shape}, eps={self.eps})"


class GroupNorm(Module):
    """Group Normalization."""

    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        if num_groups <= 0:
            raise ValueError(f"num_groups must be positive, got {num_groups}")
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
            )
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        xp = cp if get_device() == 'cuda' else np
        self.gamma = Tensor(xp.ones(num_channels, dtype=xp.float32), device=get_device())
        self.beta = Tensor(xp.zeros(num_channels, dtype=xp.float32), device=get_device())

    def __call__(self, x):
        xp = cp if self._device == 'cuda' else np
        N, C, H, W = x.data.shape
        G = self.num_groups
        if C != self.num_channels:
            raise ValueError(f"GroupNorm expected {self.num_channels} channels, got {C}")

        # Reshape to (N, G, C//G, H, W)
        x_reshaped = x.data.reshape(N, G, C // G, H, W)

        mean = x_reshaped.mean(axis=(2, 3, 4), keepdims=True)
        var = x_reshaped.var(axis=(2, 3, 4), keepdims=True)

        x_norm = (x_reshaped - mean) / xp.sqrt(var + self.eps)
        x_norm = x_norm.reshape(N, C, H, W)

        out_data = self.gamma.data.reshape(1, C, 1, 1) * x_norm + self.beta.data.reshape(1, C, 1, 1)

        out = Tensor(out_data, (x, self.gamma, self.beta), 'GroupNorm', device=self._device)
        self._cache = (x_norm, x_reshaped, mean, var)

        def _backward():
            dout = out.grad
            x_norm_c, x_reshaped_c, mean_c, var_c = self._cache

            self.gamma.grad += (dout * x_norm_c).sum(axis=(0, 2, 3))
            self.beta.grad += dout.sum(axis=(0, 2, 3))

            # Gradient through normalization
            gamma_r = self.gamma.data.reshape(1, C, 1, 1)
            dx_norm = dout * gamma_r
            dx_norm_reshaped = dx_norm.reshape(N, G, C // G, H, W)

            n = (C // G) * H * W
            std_inv = 1.0 / xp.sqrt(var_c + self.eps)

            x_centered = x_reshaped_c - mean_c

            dx_reshaped = dx_norm_reshaped * std_inv
            dx_reshaped -= dx_norm_reshaped.mean(axis=(2, 3, 4), keepdims=True) * std_inv
            dx_reshaped -= x_centered * (dx_norm_reshaped * x_centered / (var_c + self.eps)).mean(axis=(2, 3, 4), keepdims=True) * std_inv

            x.grad += dx_reshaped.reshape(N, C, H, W)

        out._backward = _backward
        return out

    def parameters(self):
        return [self.gamma, self.beta]

    def __repr__(self):
        return f"GroupNorm(num_groups={self.num_groups}, num_channels={self.num_channels}, eps={self.eps})"


class Dropout(Module):
    """Dropout regularization."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self._mask = None

    def __call__(self, x):
        xp = cp if self._device == 'cuda' else np

        if not self._training or self.p == 0:
            return x

        # Handle p=1.0 edge case: all outputs are dropped, return zeros
        if self.p >= 1.0:
            return Tensor(xp.zeros_like(x.data), (x,), 'Dropout', device=self._device)

        mask = (xp.random.rand(*x.data.shape) > self.p).astype(xp.float32)
        scale = 1.0 / (1.0 - self.p)

        out = Tensor(x.data * mask * scale, (x,), 'Dropout', device=self._device)
        self._mask = mask

        def _backward():
            x.grad += out.grad * mask * scale
        out._backward = _backward

        return out

    def __repr__(self):
        return f"Dropout(p={self.p})"


class Embedding(Module):
    """Embedding layer for discrete inputs."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        xp = cp if get_device() == 'cuda' else np
        self.weight = Tensor(xp.random.randn(num_embeddings, embedding_dim).astype(xp.float32) * 0.01, device=get_device())
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def __call__(self, indices):
        xp = cp if self._device == 'cuda' else np

        if isinstance(indices, Tensor):
            idx = indices.data
        else:
            idx = xp.array(indices)

        idx = idx.astype(xp.int32)
        out_data = self.weight.data[idx]
        out = Tensor(out_data, (self.weight,), 'Embedding', device=self._device)

        def _backward():
            grad = xp.zeros_like(self.weight.data)
            if xp == cp:
                cupyx.scatter_add(grad, (idx.flatten(),), out.grad.reshape(-1, self.embedding_dim))
            else:
                np.add.at(grad, idx.flatten(), out.grad.reshape(-1, self.embedding_dim))
            self.weight.grad += grad

        out._backward = _backward
        return out

    def parameters(self):
        return [self.weight]

    def __repr__(self):
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"


# ==================== LOSS FUNCTIONS ====================

class MSELoss(Module):
    """Mean Squared Error loss."""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def __call__(self, y_pred, y_true):
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true, device=y_pred._device, requires_grad=False)

        diff = y_pred - y_true
        sq = diff ** 2

        if self.reduction == 'mean':
            return sq.mean()
        elif self.reduction == 'sum':
            return sq.sum()
        else:
            return sq

    def __repr__(self):
        return f"MSELoss(reduction='{self.reduction}')"


class CrossEntropyLoss(Module):
    """Cross Entropy loss with logits."""

    def __init__(self, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def __call__(self, logits, targets):
        xp = cp if logits._device == 'cuda' else np
        N = logits.data.shape[0]
        C = logits.data.shape[1]

        # Softmax
        shifted = logits.data - xp.max(logits.data, axis=1, keepdims=True)
        exp_logits = xp.exp(shifted)
        probs = exp_logits / xp.sum(exp_logits, axis=1, keepdims=True)

        # Handle targets
        targets_array = targets.data if isinstance(targets, Tensor) else xp.array(targets)
        targets_int = targets_array.astype(xp.int32)

        # Apply label smoothing
        if self.label_smoothing > 0:
            smooth_targets = xp.full_like(probs, self.label_smoothing / C)
            smooth_targets[xp.arange(N), targets_int] = 1 - self.label_smoothing + self.label_smoothing / C
            loss = -xp.sum(smooth_targets * xp.log(probs + 1e-10), axis=1)
        else:
            log_probs = xp.log(probs[xp.arange(N), targets_int] + 1e-10)
            loss = -log_probs

        if self.reduction == 'mean':
            loss_val = xp.mean(loss)
        elif self.reduction == 'sum':
            loss_val = xp.sum(loss)
        else:
            loss_val = loss

        out = Tensor(loss_val, (logits,), 'CrossEntropy', device=logits._device)

        def _backward():
            grad = probs.copy()
            if self.label_smoothing > 0:
                grad -= smooth_targets
            else:
                grad[xp.arange(N), targets_int] -= 1

            if self.reduction == 'mean':
                grad /= N

            logits.grad += grad

        out._backward = _backward

        return out

    def __repr__(self):
        return f"CrossEntropyLoss(reduction='{self.reduction}', label_smoothing={self.label_smoothing})"


class BCEWithLogitsLoss(Module):
    """Binary Cross Entropy with logits (numerically stable)."""

    def __init__(self, reduction='mean', pos_weight=None):
        super().__init__()
        self.reduction = reduction
        self.pos_weight = pos_weight

    def __call__(self, logits, targets):
        xp = cp if logits._device == 'cuda' else np

        if not isinstance(targets, Tensor):
            targets = Tensor(targets, device=logits._device, requires_grad=False)

        # Numerically stable BCE: max(x, 0) - x * t + log(1 + exp(-|x|))
        x = logits.data
        t = targets.data

        max_val = xp.maximum(x, 0)
        loss = max_val - x * t + xp.log(1 + xp.exp(-xp.abs(x)))

        if self.pos_weight is not None:
            pw = self.pos_weight
            loss = loss * (1 + (pw - 1) * t)

        if self.reduction == 'mean':
            loss_val = xp.mean(loss)
        elif self.reduction == 'sum':
            loss_val = xp.sum(loss)
        else:
            loss_val = loss

        out = Tensor(loss_val, (logits, targets), 'BCEWithLogits', device=logits._device)

        def _backward():
            sig = 1 / (1 + xp.exp(-x))
            grad = sig - t

            if self.pos_weight is not None:
                grad = grad * (1 + (self.pos_weight - 1) * t)

            if self.reduction == 'mean':
                grad /= x.size

            logits.grad += grad

        out._backward = _backward

        return out

    def __repr__(self):
        return f"BCEWithLogitsLoss(reduction='{self.reduction}')"


class L1Loss(Module):
    """L1 (Mean Absolute Error) loss."""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def __call__(self, y_pred, y_true):
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true, device=y_pred._device, requires_grad=False)

        xp = y_pred.xp
        diff = y_pred - y_true
        abs_diff = Tensor(xp.abs(diff.data), (diff,), 'abs', device=y_pred._device)

        def _abs_backward():
            diff.grad += xp.sign(diff.data) * abs_diff.grad
        abs_diff._backward = _abs_backward

        if self.reduction == 'mean':
            return abs_diff.mean()
        elif self.reduction == 'sum':
            return abs_diff.sum()
        return abs_diff

    def __repr__(self):
        return f"L1Loss(reduction='{self.reduction}')"


class SmoothL1Loss(Module):
    """Smooth L1 (Huber) loss."""

    def __init__(self, reduction='mean', beta=1.0):
        super().__init__()
        self.reduction = reduction
        self.beta = beta

    def __call__(self, y_pred, y_true):
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true, device=y_pred._device, requires_grad=False)

        xp = y_pred.xp
        diff = y_pred - y_true
        abs_diff = xp.abs(diff.data)

        # Huber loss: 0.5 * x^2 / beta if |x| < beta else |x| - 0.5 * beta
        loss_data = xp.where(abs_diff < self.beta,
                            0.5 * diff.data ** 2 / self.beta,
                            abs_diff - 0.5 * self.beta)

        out = Tensor(loss_data, (y_pred, y_true), 'SmoothL1', device=y_pred._device)

        def _backward():
            grad = xp.where(abs_diff < self.beta,
                           diff.data / self.beta,
                           xp.sign(diff.data))
            if self.reduction == 'mean':
                grad /= y_pred.size
            y_pred.grad += grad

        out._backward = _backward

        if self.reduction == 'mean':
            return out.mean()
        elif self.reduction == 'sum':
            return out.sum()
        return out

    def __repr__(self):
        return f"SmoothL1Loss(reduction='{self.reduction}', beta={self.beta})"


# ==================== OPTIMIZERS ====================

class SGD:
    """SGD optimizer with momentum and weight decay."""

    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self._device = self.params[0]._device if self.params else 'cuda'
        xp = cp if self._device == 'cuda' else np
        self.velocities = [xp.zeros_like(p.data) for p in self.params]

    def step(self):
        xp = cp if self._device == 'cuda' else np
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data

            if self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                if self.nesterov:
                    update = grad + self.momentum * self.velocities[i]
                else:
                    update = self.velocities[i]
            else:
                update = grad

            p.data -= self.lr * update

    def zero_grad(self):
        xp = cp if self._device == 'cuda' else np
        for p in self.params:
            if p.grad is not None:
                p.grad = xp.zeros_like(p.data)

    def state_dict(self) -> dict:
        """Return optimizer state for checkpointing."""
        return {
            'lr': self.lr,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'nesterov': self.nesterov,
            'velocities': [to_cpu(v) for v in self.velocities],
        }

    def load_state_dict(self, state: dict):
        """Load optimizer state from checkpoint."""
        xp = cp if self._device == 'cuda' else np
        self.lr = state.get('lr', self.lr)
        self.momentum = state.get('momentum', self.momentum)
        self.weight_decay = state.get('weight_decay', self.weight_decay)
        self.nesterov = state.get('nesterov', self.nesterov)
        if 'velocities' in state:
            self.velocities = [xp.array(v, dtype=xp.float32) for v in state['velocities']]


class Adam:
    """Adam optimizer with weight decay and gradient clipping."""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, amsgrad=False):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self._device = self.params[0]._device if self.params else 'cuda'
        xp = cp if self._device == 'cuda' else np
        self.m = [xp.zeros_like(p.data) for p in self.params]
        self.v = [xp.zeros_like(p.data) for p in self.params]
        if amsgrad:
            self.v_max = [xp.zeros_like(p.data) for p in self.params]
        self.t = 0

    def clip_grad_norm(self, max_norm=1.0) -> float:
        """Clip gradients by global norm. Returns the original norm."""
        xp = cp if self._device == 'cuda' else np
        total_norm = 0.0
        for p in self.params:
            if p.grad is not None:
                total_norm += float((p.grad ** 2).sum())
        total_norm = float(xp.sqrt(total_norm))

        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            for p in self.params:
                if p.grad is not None:
                    p.grad = p.grad * clip_coef

        return total_norm

    def step(self, clip_grad_norm=None):
        xp = cp if self._device == 'cuda' else np

        if clip_grad_norm is not None:
            self.clip_grad_norm(clip_grad_norm)

        self.t += 1

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad
            if self.weight_decay > 0:
                # AdamW-style weight decay
                p.data -= self.lr * self.weight_decay * p.data

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            if self.amsgrad:
                self.v_max[i] = xp.maximum(self.v_max[i], v_hat)
                denom = xp.sqrt(self.v_max[i]) + self.eps
            else:
                denom = xp.sqrt(v_hat) + self.eps

            p.data -= self.lr * m_hat / denom

    def zero_grad(self):
        xp = cp if self._device == 'cuda' else np
        for p in self.params:
            if p.grad is not None:
                p.grad = xp.zeros_like(p.data)

    def state_dict(self) -> dict:
        """Return optimizer state for checkpointing."""
        state = {
            'lr': self.lr,
            'betas': (self.beta1, self.beta2),
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'amsgrad': self.amsgrad,
            't': self.t,
            'm': [to_cpu(m) for m in self.m],
            'v': [to_cpu(v) for v in self.v],
        }
        if self.amsgrad:
            state['v_max'] = [to_cpu(vm) for vm in self.v_max]
        return state

    def load_state_dict(self, state: dict):
        """Load optimizer state from checkpoint."""
        xp = cp if self._device == 'cuda' else np
        self.lr = state.get('lr', self.lr)
        self.t = state.get('t', self.t)
        if 'betas' in state:
            self.beta1, self.beta2 = state['betas']
        self.eps = state.get('eps', self.eps)
        self.weight_decay = state.get('weight_decay', self.weight_decay)
        if 'm' in state:
            self.m = [xp.array(m, dtype=xp.float32) for m in state['m']]
        if 'v' in state:
            self.v = [xp.array(v, dtype=xp.float32) for v in state['v']]
        if self.amsgrad and 'v_max' in state:
            self.v_max = [xp.array(vm, dtype=xp.float32) for vm in state['v_max']]


class AdamW(Adam):
    """AdamW optimizer (Adam with decoupled weight decay)."""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr, betas, eps, weight_decay)


class RMSprop:
    """RMSprop optimizer."""

    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0):
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self._device = self.params[0]._device if self.params else 'cuda'
        xp = cp if self._device == 'cuda' else np
        self.square_avg = [xp.zeros_like(p.data) for p in self.params]
        if momentum > 0:
            self.momentum_buffer = [xp.zeros_like(p.data) for p in self.params]

    def step(self):
        xp = cp if self._device == 'cuda' else np

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data

            self.square_avg[i] = self.alpha * self.square_avg[i] + (1 - self.alpha) * (grad ** 2)

            avg = xp.sqrt(self.square_avg[i]) + self.eps

            if self.momentum > 0:
                self.momentum_buffer[i] = self.momentum * self.momentum_buffer[i] + grad / avg
                p.data -= self.lr * self.momentum_buffer[i]
            else:
                p.data -= self.lr * grad / avg

    def zero_grad(self):
        xp = cp if self._device == 'cuda' else np
        for p in self.params:
            if p.grad is not None:
                p.grad = xp.zeros_like(p.data)

    def state_dict(self) -> dict:
        """Return optimizer state for checkpointing."""
        state = {
            'lr': self.lr,
            'alpha': self.alpha,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'square_avg': [to_cpu(sa) for sa in self.square_avg],
        }
        if self.momentum > 0:
            state['momentum_buffer'] = [to_cpu(mb) for mb in self.momentum_buffer]
        return state

    def load_state_dict(self, state: dict):
        """Load optimizer state from checkpoint."""
        xp = cp if self._device == 'cuda' else np
        self.lr = state.get('lr', self.lr)
        self.alpha = state.get('alpha', self.alpha)
        self.eps = state.get('eps', self.eps)
        self.weight_decay = state.get('weight_decay', self.weight_decay)
        self.momentum = state.get('momentum', self.momentum)
        if 'square_avg' in state:
            self.square_avg = [xp.array(sa, dtype=xp.float32) for sa in state['square_avg']]
        if self.momentum > 0 and 'momentum_buffer' in state:
            self.momentum_buffer = [xp.array(mb, dtype=xp.float32) for mb in state['momentum_buffer']]


# ==================== GRADIENT CLIPPING ====================

def clip_grad_norm_(parameters: List[Tensor], max_norm: float,
                    norm_type: float = 2.0, error_if_nonfinite: bool = False) -> float:
    """
    Clip gradients by global norm (in-place).

    Handles NaN/Inf gradients gracefully by zeroing them before computing the norm.
    This prevents NaN propagation in the training loop while logging a warning.

    Args:
        parameters: Iterable of Tensors with gradients
        max_norm: Maximum allowed gradient norm
        norm_type: Type of p-norm (2.0 for L2, inf for max)
        error_if_nonfinite: If True, raise RuntimeError on NaN/Inf gradients

    Returns:
        Total gradient norm before clipping (0.0 if all gradients are NaN/Inf)
    """
    import warnings

    params = [p for p in parameters if p.grad is not None]
    if len(params) == 0:
        return 0.0

    xp = cp if params[0]._device == 'cuda' else np

    # Check for NaN/Inf gradients and handle them
    has_nan_inf = False
    for p in params:
        if xp.any(xp.isnan(p.grad)) or xp.any(xp.isinf(p.grad)):
            has_nan_inf = True
            if error_if_nonfinite:
                raise RuntimeError("Gradient contains NaN or Inf values")
            # Zero out non-finite gradients to prevent propagation
            p.grad = xp.where(xp.isfinite(p.grad), p.grad, xp.zeros_like(p.grad))

    if has_nan_inf:
        warnings.warn("clip_grad_norm_: Found NaN/Inf in gradients, zeroed them. "
                      "Consider reducing learning rate or checking loss computation.")

    # Recompute params list with potentially modified gradients
    params = [p for p in parameters if p.grad is not None]
    if len(params) == 0:
        return 0.0

    if norm_type == float('inf'):
        norms = [float(xp.abs(p.grad).max()) for p in params]
        total_norm = max(norms) if norms else 0.0
    else:
        total_norm = 0.0
        for p in params:
            param_norm = float((xp.abs(p.grad) ** norm_type).sum())
            total_norm += param_norm
        total_norm = total_norm ** (1.0 / norm_type)

    # Handle case where total_norm is still 0 (all gradients were zeroed)
    if total_norm == 0.0:
        return 0.0

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            p.grad = p.grad * clip_coef

    return total_norm


def clip_grad_value_(parameters: List[Tensor], clip_value: float):
    """
    Clip gradients by value (in-place).

    Args:
        parameters: Iterable of Tensors with gradients
        clip_value: Maximum absolute value for gradients
    """
    for p in parameters:
        if p.grad is not None:
            xp = cp if p._device == 'cuda' else np
            p.grad = xp.clip(p.grad, -clip_value, clip_value)


# ==================== LEARNING RATE SCHEDULERS ====================

class LRScheduler:
    """Base class for learning rate schedulers."""

    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        self.optimizer.lr = self.get_lr()

    def get_lr(self):
        raise NotImplementedError


class StepLR(LRScheduler):
    """Step learning rate decay."""

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.base_lr * (self.gamma ** (self.last_epoch // self.step_size))


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate schedule."""

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        import math
        return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2


class LinearWarmupCosineDecay(LRScheduler):
    """Linear warmup followed by cosine decay."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        import math
        step = self.last_epoch

        if step < self.warmup_steps:
            return self.base_lr * step / self.warmup_steps

        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)

        return self.min_lr + (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2


# ==================== MIXED PRECISION TRAINING ====================

class GradScaler:
    """
    Dynamic gradient scaler for mixed precision training.

    Automatically adjusts loss scale to prevent gradient overflow/underflow
    in FP16 training while maximizing numerical range utilization.
    
    Why use it?
    In mixed precision (FP16), small gradient values can underflow to zero,
    and large values can overflow to infinity. GradScaler multiplies the 
    loss by a 'scale' factor before backprop, keeping gradients within 
    representable range of FP16. It then 'unscales' them before updating weights.

    Example:
        scaler = GradScaler()
        for input, target in data:
            optimizer.zero_grad()
            
            # Forward pass with mixed precision context if available
            with tg.enable_mixed_precision():
                output = model(input)
                loss = criterion(output, target)
            
            # Scale loss and backward
            scaler.scale(loss).backward()
            
            # Unscale and step
            # step() internally calls unscale_(optimizer) if not already called
            # and only performs optimizer.step() if no Inf/NaN were found
            scaler.step(optimizer)
            
            # Update scale (increase if no overflow, decrease if overflow)
            scaler.update()
    """

    def __init__(self, init_scale: float = 65536.0, growth_factor: float = 2.0,
                 backoff_factor: float = 0.5, growth_interval: int = 2000,
                 min_scale: float = 1.0, max_scale: float = 2**24,
                 enabled: bool = True):
        """
        Args:
            init_scale: Initial loss scale value (must be > 0)
            growth_factor: Factor to multiply scale by on successful steps
            backoff_factor: Factor to multiply scale by on overflow
            growth_interval: Steps between scale growth attempts
            min_scale: Minimum allowed scale (prevents scale collapse, must be > 0)
            max_scale: Maximum allowed scale (prevents overflow)
            enabled: If False, scaler is a no-op (for BF16 training)

        Raises:
            ValueError: If init_scale <= 0, min_scale <= 0, or max_scale < init_scale
        """
        # Validate scale values to prevent division by zero and numerical issues
        if init_scale <= 0:
            raise ValueError(f"init_scale must be positive, got {init_scale}")
        if min_scale <= 0:
            raise ValueError(f"min_scale must be positive, got {min_scale}")
        if max_scale < init_scale:
            raise ValueError(f"max_scale ({max_scale}) must be >= init_scale ({init_scale})")

        self._scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.enabled = enabled

        self._growth_tracker = 0
        self._found_inf = False

        # Statistics
        self._overflow_count = 0
        self._step_count = 0
        self._consecutive_successes = 0

    def scale(self, loss: Tensor) -> Tensor:
        """Scale loss for backward pass."""
        if not self.enabled:
            return loss
        return loss * self._scale

    def unscale_(self, optimizer, zero_nan_grads: bool = True):
        """Unscale gradients. Check for inf/nan.

        Args:
            optimizer: Optimizer with params to unscale
            zero_nan_grads: If True, zero out NaN/Inf gradients after detection
                           to prevent them from corrupting optimizer state.
                           The step will still be skipped due to _found_inf.
        """
        if not self.enabled:
            return

        # Prevent double unscaling
        if getattr(optimizer, "_is_unscaled", False):
            return

        xp = cp if optimizer._device == 'cuda' else np
        inv_scale = 1.0 / self._scale

        self._found_inf = False
        for p in optimizer.params:
            if p.grad is not None:
                # First check for pre-existing NaN/Inf before unscaling
                if xp.any(xp.isnan(p.grad)) or xp.any(xp.isinf(p.grad)):
                    self._found_inf = True
                    if zero_nan_grads:
                        p.grad = xp.zeros_like(p.grad)
                    continue

                # Unscale the gradient
                p.grad = p.grad * inv_scale

                # Check for overflow after unscaling
                if xp.any(xp.isinf(p.grad)) or xp.any(xp.isnan(p.grad)):
                    self._found_inf = True
                    if zero_nan_grads:
                        p.grad = xp.zeros_like(p.grad)
        
        optimizer._is_unscaled = True

    def step(self, optimizer, clip_grad_norm=None):
        """Step optimizer if gradients are valid."""
        if not self.enabled:
            optimizer.step(clip_grad_norm=clip_grad_norm)
            return

        if not getattr(optimizer, "_is_unscaled", False):
            self.unscale_(optimizer)

        if self._found_inf:
            optimizer._is_unscaled = False
            return  # Skip update on overflow
        
        optimizer.step(clip_grad_norm=clip_grad_norm)
        optimizer._is_unscaled = False

    def update(self):
        """Update scale factor based on overflow status."""
        if not self.enabled:
            return

        self._step_count += 1

        if self._found_inf:
            self._overflow_count += 1
            self._consecutive_successes = 0
            self._scale = max(self._scale * self.backoff_factor, self.min_scale)
            self._growth_tracker = 0
        else:
            self._consecutive_successes += 1
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self._scale = min(self._scale * self.growth_factor, self.max_scale)
                self._growth_tracker = 0

    @property
    def scale_factor(self) -> float:
        """Current loss scale value."""
        return self._scale

    def get_scale(self) -> float:
        """Get current scale (alias for scale_factor)."""
        return self._scale

    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {
            'scale': self._scale,
            'growth_tracker': self._growth_tracker,
            'overflow_count': self._overflow_count,
            'step_count': self._step_count,
            'consecutive_successes': self._consecutive_successes,
        }

    def load_state_dict(self, state: dict):
        """Load state from checkpoint.

        Raises:
            ValueError: If checkpoint contains scale <= 0
        """
        loaded_scale = state.get('scale', self._scale)
        if loaded_scale <= 0:
            raise ValueError(f"Checkpoint contains invalid scale ({loaded_scale}), must be positive")
        self._scale = loaded_scale
        self._growth_tracker = state.get('growth_tracker', 0)
        self._overflow_count = state.get('overflow_count', 0)
        self._step_count = state.get('step_count', 0)
        self._consecutive_successes = state.get('consecutive_successes', 0)

    def get_statistics(self) -> dict:
        """Return training statistics."""
        return {
            'current_scale': self._scale,
            'total_steps': self._step_count,
            'overflow_count': self._overflow_count,
            'overflow_rate': self._overflow_count / max(1, self._step_count),
            'consecutive_successes': self._consecutive_successes,
        }


# ==================== GRADIENT ACCUMULATION ====================

class GradientAccumulator:
    """
    Context manager for gradient accumulation.

    Enables training with larger effective batch sizes by accumulating
    gradients over multiple forward/backward passes before updating weights.

    Usage:
        accumulator = GradientAccumulator(steps=4)
        for i, batch in enumerate(dataloader):
            loss = model(batch) * accumulator.loss_scale  # Scale loss
            loss.backward()

            if accumulator.step():  # Returns True every `steps` iterations
                optimizer.step()
                optimizer.zero_grad()

    The loss_scale property returns 1/steps to ensure gradients are properly
    averaged over the accumulation window.
    """

    def __init__(self, steps: int = 1, scale_loss: bool = True):
        """
        Args:
            steps: Number of accumulation steps before optimizer update
            scale_loss: Whether to auto-scale loss by 1/steps
        """
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        self.steps = steps
        self.scale_loss = scale_loss
        self._current_step = 0

    def __enter__(self):
        self._current_step = 0
        return self

    def __exit__(self, *args):
        pass

    def step(self) -> bool:
        """
        Increment step counter and return whether to update weights.

        Returns:
            True if optimizer should step (accumulated enough gradients)
        """
        self._current_step += 1
        if self._current_step >= self.steps:
            self._current_step = 0
            return True
        return False

    def should_step(self) -> bool:
        """Alias for step() - check if optimizer should update."""
        return self.step()

    def reset(self):
        """Reset step counter."""
        self._current_step = 0

    @property
    def current_step(self) -> int:
        """Current accumulation step (0 to steps-1)."""
        return self._current_step

    @property
    def loss_scale(self) -> float:
        """
        Loss scale factor for gradient accumulation.

        Multiply loss by this before backward() to ensure gradients
        are properly averaged over the accumulation window.
        """
        return 1.0 / self.steps if self.scale_loss else 1.0

    @property
    def is_accumulating(self) -> bool:
        """True if currently accumulating (not yet ready to step)."""
        return self._current_step < self.steps - 1


def accumulate_grad(steps: int = 1, scale_loss: bool = True) -> GradientAccumulator:
    """
    Convenience function to create a GradientAccumulator.

    Args:
        steps: Number of accumulation steps
        scale_loss: Whether to scale loss by 1/steps

    Returns:
        GradientAccumulator instance
    """
    return GradientAccumulator(steps, scale_loss)


# ==================== GRADIENT CHECKPOINTING ====================

def checkpoint(function: Callable, *args) -> Tensor:
    """
    Gradient checkpointing for memory-efficient training.

    Recomputes forward pass during backward pass instead of storing intermediate
    activations. This trades compute for memory, allowing training of much
    larger models that would otherwise not fit in GPU memory.
    
    Memory Complexity:
    - Standard: O(L) where L is the number of layers (all activations stored).
    - Checkpointed: O(sqrt(L)) if applied at regular intervals, or O(1) for
      the checkpointed block (only inputs and outputs stored).

    Note:
        - The function should not have side effects (e.g., modifying global state).
        - Random number generator state is NOT currently handled. If the
          function uses dropout or other stochastic operations, the results
          during recomputation might differ unless the seed is manually managed.
        - All input tensors that require gradients will have their gradients
          updated correctly.
        - The checkpointed function must return a Tensor or a collection of Tensors.

    Example:
        def expensive_block(x):
            return residual_connection(attention(x))
        
        # Standard way (stores all activations in attention)
        # y = expensive_block(x)
        
        # Checkpointed way (frees activations, recomputes during backward)
        y = checkpoint(expensive_block, x)

    Args:
        function: Callable that performs the forward pass of the block to checkpoint.
        *args: Input arguments to the function.

    Returns:
        Output tensor that tracks the checkpointed operation.
    """
    # Forward pass without gradient tracking
    with no_grad():
        output = function(*args)

    # Create wrapper tensor that triggers recomputation on backward
    class CheckpointFunction:
        saved_args = args
        saved_fn = function

    # Make output track gradients
    out = Tensor(output.data, tuple(a for a in args if isinstance(a, Tensor)),
                'checkpoint', device=output._device)

    def _backward():
        # Recompute forward pass with gradients
        with enable_grad():
            # Create fresh tensors from saved data
            new_args = []
            for arg in CheckpointFunction.saved_args:
                if isinstance(arg, Tensor):
                    new_arg = Tensor(arg.data, device=arg._device, requires_grad=arg._requires_grad)
                    new_arg.grad = arg.xp.zeros_like(arg.data) if arg._requires_grad else None
                    new_args.append(new_arg)
                else:
                    new_args.append(arg)

            # Recompute
            recomputed = CheckpointFunction.saved_fn(*new_args)
            
            if out.grad is not None:
                recomputed.backward(out.grad)

                # Copy gradients back
                for orig, new in zip(CheckpointFunction.saved_args, new_args):
                    if isinstance(orig, Tensor) and orig.grad is not None and new.grad is not None:
                        orig.grad += new.grad

    out._backward = _backward
    return out

