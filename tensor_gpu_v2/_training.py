"""Training infrastructure: checkpointing, profiling, and extended layers.

This module is part of the ``tensor_gpu_v2`` package.
Import via ``import tensor_gpu_v2 as tg``, not directly.
"""

from ._core import *
from ._core import _get_cached_kernel
from ._nn import *

# ==================== MODEL CHECKPOINTING ====================

import pickle
import os

CHECKPOINT_VERSION = '1.0'


def save_checkpoint(path: str, model, optimizer=None, scaler=None,
                   scheduler=None, epoch: int = 0, global_step: int = 0,
                   extra_state: dict = None):
    """
    Save complete training checkpoint.

    Args:
        path: File path to save checkpoint
        model: Model to save (Module with state_dict())
        optimizer: Optional optimizer with state_dict()
        scaler: Optional GradScaler with state_dict()
        scheduler: Optional LR scheduler
        epoch: Current epoch number
        global_step: Current global step
        extra_state: Additional state to save
    """
    checkpoint = {
        'version': CHECKPOINT_VERSION,
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': _serialize_state_dict(model.state_dict()),
    }

    if optimizer is not None and hasattr(optimizer, 'state_dict'):
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scaler is not None and hasattr(scaler, 'state_dict'):
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = {
            'last_epoch': scheduler.last_epoch,
            'base_lr': scheduler.base_lr,
        }

    if extra_state:
        checkpoint['extra_state'] = extra_state

    # Ensure directory exists
    dir_path = os.path.dirname(os.path.abspath(path))
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    # Save as pickle
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)


class SafeUnpickler(pickle.Unpickler):
    """Restricted unpickler for security."""
    def find_class(self, module, name):
        # Only allow safe modules/classes
        if module in ("numpy", "numpy.core.multiarray", "numpy.core.numeric", "numpy.dtype"):
            import numpy
            if module == "numpy.core.multiarray" and name == "_reconstruct":
                import numpy.core.multiarray
                return numpy.core.multiarray._reconstruct
            return getattr(numpy, name)
        if module in ("cupy", "cupy._core.core", "cupy._core.flags"):
            import cupy
            return getattr(cupy, name)
        if module == "builtins" and name in ("dict", "list", "set", "int", "float", "str", "bool", "complex", "tuple"):
            import builtins
            return getattr(builtins, name)
        # For tensor state dicts
        if module == "tensor_gpu_v2" or module == "__main__":
             if name == "Tensor":
                 return Tensor
             if name == "CHECKPOINT_VERSION":
                 return CHECKPOINT_VERSION
        
        raise pickle.UnpicklingError(f"Global '{module}.{name}' is forbidden")


def load_checkpoint(path: str, model, optimizer=None, scaler=None,
                   scheduler=None, strict: bool = True) -> dict:
    """
    Load checkpoint and restore model/optimizer state.

    Args:
        path: Checkpoint file path
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        scaler: Optional GradScaler to restore state
        scheduler: Optional LR scheduler to restore
        strict: If True, raise error on missing keys

    Returns:
        Dict with 'epoch', 'global_step', and any 'extra_state'
    """
    with open(path, 'rb') as f:
        unpickler = SafeUnpickler(f)
        checkpoint = unpickler.load()

    # Version check
    version = checkpoint.get('version', '0.0')
    if version != CHECKPOINT_VERSION:
        import warnings
        warnings.warn(f"Checkpoint version {version} differs from current {CHECKPOINT_VERSION}")

    # Load model
    model_state = _deserialize_state_dict(checkpoint['model_state_dict'], model._device)
    model.load_state_dict(model_state)

    # Load optimizer
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scaler
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # Load scheduler
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.last_epoch = checkpoint['scheduler_state_dict'].get('last_epoch', -1)

    return {
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'extra_state': checkpoint.get('extra_state', {}),
    }


def _serialize_state_dict(state_dict: dict) -> dict:
    """Convert GPU arrays to CPU for serialization."""
    serialized = {}
    for k, v in state_dict.items():
        if isinstance(v, cp.ndarray):
            serialized[k] = cp.asnumpy(v)
        elif isinstance(v, dict):
            serialized[k] = _serialize_state_dict(v)
        else:
            serialized[k] = v
    return serialized


def _deserialize_state_dict(state_dict: dict, device: str) -> dict:
    """Convert CPU arrays to appropriate device."""
    xp = cp if device == 'cuda' else np
    deserialized = {}
    for k, v in state_dict.items():
        if isinstance(v, np.ndarray):
            deserialized[k] = xp.array(v)
        elif isinstance(v, dict):
            deserialized[k] = _deserialize_state_dict(v, device)
        else:
            deserialized[k] = v
    return deserialized


# ==================== WEIGHT TYING ====================

class TiedWeight:
    """
    Wrapper for tied weight that accumulates gradients to source.

    Used internally by weight_tie() to manage gradient flow.
    """

    def __init__(self, source: Tensor, transpose: bool = False):
        self.source = source
        self.transpose = transpose

    @property
    def data(self):
        if self.transpose:
            return self.source.data.T
        return self.source.data

    @data.setter
    def data(self, value):
        # Prevent accidental overwrites
        pass

    @property
    def grad(self):
        return None  # Gradients go to source

    @property
    def shape(self):
        if self.transpose:
            return self.source.shape[::-1]
        return self.source.shape


def weight_tie(source: Tensor, target_module, target_attr: str,
               transpose: bool = False):
    """
    Tie weights between a source tensor and a module's parameter.

    The target parameter will use source's data (transposed if specified).
    Gradients computed at the target will accumulate to the source.

    Common use case: Tie embedding weights with output projection in LM:
        weight_tie(embedding.weight, lm_head, 'w', transpose=True)

    Args:
        source: Source tensor (the "master" copy)
        target_module: Module containing the target parameter
        target_attr: Attribute name of the target parameter
        transpose: If True, target uses source.T

    Raises:
        ValueError: If shapes are incompatible for tying

    Example:
        # In a language model
        embed = Embedding(vocab_size, embed_dim)
        lm_head = Linear(embed_dim, vocab_size)
        weight_tie(embed.weight, lm_head, 'w', transpose=True)
    """
    # Get target parameter
    target_param = getattr(target_module, target_attr)

    # Validate shape compatibility
    source_shape = source.shape
    target_shape = target_param.shape

    if transpose:
        expected_target_shape = source_shape[::-1] if len(source_shape) == 2 else source_shape
        if target_shape != expected_target_shape:
            raise ValueError(
                f"Shape mismatch for weight tying with transpose=True: "
                f"source.T shape {expected_target_shape} != target shape {target_shape}. "
                f"Source has shape {source_shape}."
            )
    else:
        if source_shape != target_shape:
            raise ValueError(
                f"Shape mismatch for weight tying: "
                f"source shape {source_shape} != target shape {target_shape}. "
                f"Use transpose=True if shapes should be transposed."
            )

    # Store original for gradient accumulation
    target_param._tied_source = source
    target_param._tied_transpose = transpose

    # Make data point to source
    if transpose:
        target_param.data = source.data.T
    else:
        target_param.data = source.data


def sync_tied_gradients(model):
    """
    After backward pass, accumulate gradients from tied targets to sources.

    Call this after loss.backward() if using weight tying.

    Args:
        model: Module containing tied weights
    """
    xp = cp if model._device == 'cuda' else np

    for param in model.parameters():
        if hasattr(param, '_tied_source') and param.grad is not None:
            source = param._tied_source
            if source.grad is not None:
                if param._tied_transpose:
                    source.grad += param.grad.T
                else:
                    source.grad += param.grad
                # Zero target grad to prevent double counting
                param.grad = xp.zeros_like(param.data)


# ==================== PROFILING ====================

class Profiler:
    """
    GPU profiling context manager.

    Tracks execution time, memory usage, and provides actionable insights.

    Usage:
        with Profiler() as prof:
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
        print(prof.report())
    """

    def __init__(self, name: str = 'default', sync: bool = True):
        """
        Args:
            name: Name for this profiling region
            sync: Whether to synchronize GPU before/after (recommended)
        """
        self.name = name
        self.sync = sync
        self._start_event = None
        self._end_event = None
        self._start_mem = 0
        self._end_mem = 0
        self._elapsed_ms = None

    def __enter__(self):
        if self.sync and get_device() == 'cuda':
            cp.cuda.Stream.null.synchronize()

        self._start_mem = _memory_pool.used_bytes()

        if get_device() == 'cuda':
            self._start_event = cp.cuda.Event()
            self._start_event.record()

        return self

    def __exit__(self, *args):
        if get_device() == 'cuda':
            self._end_event = cp.cuda.Event()
            self._end_event.record()

        if self.sync and get_device() == 'cuda':
            cp.cuda.Stream.null.synchronize()

        self._end_mem = _memory_pool.used_bytes()

        if get_device() == 'cuda' and self._start_event and self._end_event:
            self._elapsed_ms = cp.cuda.get_elapsed_time(
                self._start_event, self._end_event
            )

    def report(self) -> dict:
        """
        Generate profiling report.

        Returns:
            Dict with timing and memory statistics
        """
        mem_delta = (self._end_mem - self._start_mem) / 1024**2
        peak_mem = _memory_pool.total_bytes() / 1024**2

        report = {
            'name': self.name,
            'elapsed_ms': self._elapsed_ms,
            'memory_delta_mb': mem_delta,
            'peak_memory_mb': peak_mem,
            'start_memory_mb': self._start_mem / 1024**2,
            'end_memory_mb': self._end_mem / 1024**2,
        }

        return report

    def summary(self) -> str:
        """Get human-readable summary."""
        r = self.report()
        elapsed = r['elapsed_ms'] if r['elapsed_ms'] is not None else 0
        return (f"[{r['name']}] Time: {elapsed:.2f}ms | "
                f"Memory: {r['memory_delta_mb']:+.2f}MB | "
                f"Peak: {r['peak_memory_mb']:.1f}MB")


def profile(name: str = 'default') -> Profiler:
    """Create a profiling context manager."""
    return Profiler(name)


def benchmark(fn: Callable, *args, n_repeat: int = 10,
              n_warmup: int = 3, **kwargs) -> dict:
    """
    Benchmark a function with proper GPU synchronization.

    Args:
        fn: Function to benchmark
        *args: Arguments to pass to fn
        n_repeat: Number of benchmark iterations
        n_warmup: Number of warmup iterations (not counted)
        **kwargs: Keyword arguments to pass to fn

    Returns:
        Dict with timing statistics (mean, std, min, max in ms)
    """
    # Warmup
    for _ in range(n_warmup):
        fn(*args, **kwargs)
        if get_device() == 'cuda':
            cp.cuda.Stream.null.synchronize()

    # Benchmark
    times = []
    for _ in range(n_repeat):
        if get_device() == 'cuda':
            cp.cuda.Stream.null.synchronize()

        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record()
        fn(*args, **kwargs)
        end.record()

        if get_device() == 'cuda':
            cp.cuda.Stream.null.synchronize()

        times.append(cp.cuda.get_elapsed_time(start, end))

    times = np.array(times)
    return {
        'mean_ms': float(times.mean()),
        'std_ms': float(times.std()),
        'min_ms': float(times.min()),
        'max_ms': float(times.max()),
        'n_repeat': n_repeat,
    }


# ==================== UTILITY FUNCTIONS ====================

def cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """Concatenate tensors along dimension with gradient support."""
    if len(tensors) == 0:
        raise ValueError("Cannot concatenate empty list")
    if len(tensors) == 1:
        return tensors[0]

    device = tensors[0]._device
    xp = cp if device == 'cuda' else np

    out_data = xp.concatenate([t.data for t in tensors], axis=dim)
    out = Tensor(out_data, tuple(tensors), 'cat', device=device)

    sizes = [t.data.shape[dim] for t in tensors]

    def _backward():
        indices = []
        cumsum = 0
        for s in sizes[:-1]:
            cumsum += s
            indices.append(cumsum)
        grad_parts = xp.split(out.grad, indices, axis=dim)
        for t, g in zip(tensors, grad_parts):
            if t.grad is not None:
                t.grad += g

    out._backward = _backward
    return out


def stack(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """Stack tensors along new dimension."""
    if len(tensors) == 0:
        raise ValueError("Cannot stack empty list")

    expanded = [t.unsqueeze(dim) for t in tensors]
    return cat(expanded, dim=dim)


def chunk(tensor: Tensor, chunks: int, dim: int = 0) -> List[Tensor]:
    """Split tensor into chunks."""
    xp = tensor.xp
    chunk_size = (tensor.shape[dim] + chunks - 1) // chunks
    indices = list(range(chunk_size, tensor.shape[dim], chunk_size))

    data_chunks = xp.split(tensor.data, indices, axis=dim)
    result = []

    for i, data in enumerate(data_chunks):
        out = Tensor(data, (tensor,), f'chunk_{i}', device=tensor._device)

        start = i * chunk_size
        end = min(start + chunk_size, tensor.shape[dim])

        def _backward(s=start, e=end, out_chunk=out):
            if tensor.grad is not None:
                idx = [slice(None)] * tensor.ndim
                idx[dim] = slice(s, e)
                tensor.grad[tuple(idx)] += out_chunk.grad

        out._backward = _backward
        result.append(out)

    return result


def zeros(*shape, device=None, dtype=None, requires_grad=True) -> Tensor:
    """Create tensor of zeros."""
    device = device or get_device()
    dtype = dtype or get_default_dtype()
    xp = cp if device == 'cuda' else np
    return Tensor(xp.zeros(shape, dtype=dtype), device=device, requires_grad=requires_grad)


def ones(*shape, device=None, dtype=None, requires_grad=True) -> Tensor:
    """Create tensor of ones."""
    device = device or get_device()
    dtype = dtype or get_default_dtype()
    xp = cp if device == 'cuda' else np
    return Tensor(xp.ones(shape, dtype=dtype), device=device, requires_grad=requires_grad)


def randn(*shape, device=None, dtype=None, requires_grad=True) -> Tensor:
    """Create tensor with random normal values."""
    device = device or get_device()
    dtype = dtype or get_default_dtype()
    xp = cp if device == 'cuda' else np
    return Tensor(xp.random.randn(*shape).astype(dtype), device=device, requires_grad=requires_grad)


def rand(*shape, device=None, dtype=None, requires_grad=True) -> Tensor:
    """Create tensor with random uniform values [0, 1)."""
    device = device or get_device()
    dtype = dtype or get_default_dtype()
    xp = cp if device == 'cuda' else np
    return Tensor(xp.random.rand(*shape).astype(dtype), device=device, requires_grad=requires_grad)


def arange(start, stop=None, step=1, device=None, dtype=None, requires_grad=False) -> Tensor:
    """Create 1D tensor with evenly spaced values."""
    device = device or get_device()
    dtype = dtype or get_default_dtype()
    xp = cp if device == 'cuda' else np
    if stop is None:
        stop = start
        start = 0
    return Tensor(xp.arange(start, stop, step, dtype=dtype), device=device, requires_grad=requires_grad)


def linspace(start, stop, num, device=None, dtype=None, requires_grad=False) -> Tensor:
    """Create 1D tensor with linearly spaced values."""
    device = device or get_device()
    dtype = dtype or get_default_dtype()
    xp = cp if device == 'cuda' else np
    return Tensor(xp.linspace(start, stop, num, dtype=dtype), device=device, requires_grad=requires_grad)


def eye(n, m=None, device=None, dtype=None, requires_grad=False) -> Tensor:
    """Create identity matrix."""
    device = device or get_device()
    dtype = dtype or get_default_dtype()
    xp = cp if device == 'cuda' else np
    return Tensor(xp.eye(n, m, dtype=dtype), device=device, requires_grad=requires_grad)


# ==================== BACKWARDS COMPATIBILITY ====================

# Aliases for compatibility with tensor_gpu.py
class ConvTranspose2D(Module):
    """
    2D Transposed Convolutional layer (Deconvolution).
    
    Uses col2im for forward pass and im2col for backward pass.
    Supports NCHW and NHWC formats.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, data_format='NCHW'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.groups = groups
        self.use_bias = bias
        self.data_format = data_format

        if data_format not in ('NCHW', 'NHWC'):
            raise ValueError(f"data_format must be 'NCHW' or 'NHWC', got {data_format}")
            
        if in_channels % groups != 0:
            raise ValueError(f"in_channels must be divisible by groups, got {in_channels} and {groups}")
        if out_channels % groups != 0:
            raise ValueError(f"out_channels must be divisible by groups, got {out_channels} and {groups}")

        xp = cp if get_device() == 'cuda' else np
        scale = xp.sqrt(2.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        
        # Weight shape: (in_channels, out_channels // groups, kH, kW)
        self.w = Tensor(xp.random.randn(in_channels, out_channels // groups, self.kernel_size[0], self.kernel_size[1]).astype(xp.float32) * scale, device=get_device())
        
        if bias:
            self.b = Tensor(xp.zeros((out_channels,), dtype=xp.float32), device=get_device())
        else:
            self.b = None

    def __call__(self, x):
        if self.data_format == 'NHWC':
            N, H, W, C = x.shape
            # Convert NHWC to NCHW
            x_nchw_data = x.data.transpose(0, 3, 1, 2)
            x_nchw = Tensor(x_nchw_data, device=self._device, requires_grad=x._requires_grad, dtype=x.dtype)
            
            if self.groups == 1:
                out_nchw = self._forward_nchw(x_nchw)
            else:
                out_nchw = self._forward_grouped(x_nchw)
            
            # Convert NCHW back to NHWC
            out_data = out_nchw.data.transpose(0, 2, 3, 1)
            out = Tensor(out_data, (x,), 'ConvTranspose2D_NHWC', device=self._device, dtype=out_data.dtype)
            
            def _nhwc_backward():
                if out.grad is not None:
                    out_nchw.grad = out.grad.transpose(0, 3, 1, 2)
                    out_nchw._backward()
                    if x.grad is not None:
                        x.grad += x_nchw.grad.transpose(0, 2, 3, 1)
            out._backward = _nhwc_backward
            return out
            
        if self.groups == 1:
            return self._forward_nchw(x)
        else:
            return self._forward_grouped(x)

    def _forward_grouped(self, x):
        xp = cp if self._device == 'cuda' else np
        N, C_in, H_in, W_in = x.shape
        C_out = self.out_channels
        G = self.groups
        HH, WW = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        oph, opw = self.output_padding

        H_out = (H_in - 1) * sh - 2 * ph + HH + oph
        W_out = (W_in - 1) * sw - 2 * pw + WW + opw
        
        C_in_g = C_in // G
        C_out_g = C_out // G
        
        # Reshape weight: (G, C_in_g, C_out_g, HH, WW)
        w_grouped = self.w.data.reshape(G, C_in_g, C_out_g, HH, WW)
        w_grouped_flat = w_grouped.reshape(G, C_in_g, -1)
        w_grouped_T = w_grouped_flat.transpose(0, 2, 1)
        
        # Reshape input: (N, G, C_in_g, H_in, W_in) -> (G, C_in_g, H_in * W_in * N)
        x_grouped = x.data.reshape(N, G, C_in_g, H_in, W_in)
        x_grouped = x_grouped.transpose(1, 2, 3, 4, 0).reshape(G, C_in_g, -1)
        
        # Batched matmul: (G, K, C_in_g) @ (G, C_in_g, M * N) -> (G, K, M * N)
        out_grouped = xp.matmul(w_grouped_T, x_grouped)
        
        # Reshape out_grouped
        out_grouped = out_grouped.reshape(G, C_out_g, HH * WW, -1) 
        out_grouped = out_grouped.reshape(C_out, HH * WW, -1) 
        out_col = out_grouped.reshape(C_out * HH * WW, -1)
        
        # col2im
        out_data = col2im_indices(out_col, (N, C_out, H_out, W_out), HH, WW, padding=(ph, pw), stride=(sh, sw))
        
        if self.use_bias:
            out_data = out_data + self.b.data.reshape(1, -1, 1, 1)
            
        out = Tensor(out_data, (x, self.w) + ((self.b,) if self.use_bias else ()), 'GroupedConvTranspose2D', device=self._device)
        
        # Cache for backward
        x_grouped_cache = x_grouped
        w_grouped_cache = w_grouped
        
        def _backward():
            dout = out.grad
            if self.use_bias:
                self.b.grad += dout.sum(axis=(0, 2, 3))
                
            # im2col on dout
            dout_col = im2col_indices(dout, HH, WW, padding=(ph, pw), stride=(sh, sw))
            dout_grouped = dout_col.reshape(G, C_out_g * HH * WW, -1)
            
            # dw_grouped
            dw_grouped = xp.matmul(x_grouped_cache, dout_grouped.transpose(0, 2, 1))
            self.w.grad += dw_grouped.reshape(self.w.data.shape)
            
            if x.grad is not None:
                # dx_g
                w_g_flat = w_grouped_cache.reshape(G, C_in_g, -1)
                dx_grouped = xp.matmul(w_g_flat, dout_grouped)
                
                dx_grouped = dx_grouped.reshape(G, C_in_g, H_in, W_in, N)
                dx_grouped = dx_grouped.transpose(4, 0, 1, 2, 3).reshape(N, C_in, H_in, W_in)
                x.grad += dx_grouped

        out._backward = _backward
        return out

    def _forward_nchw(self, x):
        xp = cp if self._device == 'cuda' else np
        N, C_in, H_in, W_in = x.shape
        C_out = self.out_channels
        HH, WW = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        oph, opw = self.output_padding

        H_out = (H_in - 1) * sh - 2 * ph + HH + oph
        W_out = (W_in - 1) * sw - 2 * pw + WW + opw
        
        w_col = self.w.data.reshape(C_in, -1)
        x_col = x.data.transpose(1, 2, 3, 0).reshape(C_in, -1)
        out_col = w_col.T @ x_col
        
        out_data = col2im_indices(out_col, (N, C_out, H_out, W_out), HH, WW, padding=(ph, pw), stride=(sh, sw))
        
        if self.use_bias:
            out_data = out_data + self.b.data.reshape(1, -1, 1, 1)
            
        out = Tensor(out_data, (x, self.w) + ((self.b,) if self.use_bias else ()), 'ConvTranspose2D', device=self._device)
        
        def _backward():
            dout_col = im2col_indices(out.grad, HH, WW, padding=(ph, pw), stride=(sh, sw))
            
            if self.use_bias:
                self.b.grad += out.grad.sum(axis=(0, 2, 3))
                
            dw_col = x_col.reshape(C_in, -1) @ dout_col.T
            self.w.grad += dw_col.reshape(self.w.data.shape)
            
            if x.grad is not None:
                dx_col = w_col @ dout_col
                x.grad += dx_col.reshape(C_in, H_in, W_in, N).transpose(3, 0, 1, 2)

        out._backward = _backward
        return out

    def parameters(self):
        if self.use_bias:
            return [self.w, self.b]
        return [self.w]

    def __repr__(self):
        return (f"ConvTranspose2D({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, groups={self.groups}, bias={self.use_bias})")

# Initialize streams on import
# additions placeholder

# ==================== HOOK DISPATCH FOR EXISTING MODULES ====================

def _install_hook_dispatch(*classes):
    """Retroactively wrap __call__ in Module subclasses to run forward hooks."""
    for cls in classes:
        if '__call__' in cls.__dict__:
            orig = cls.__dict__['__call__']
            def _make_hooked(fn):
                def _hooked_call(self, *args, **kwargs):
                    result = fn(self, *args, **kwargs)
                    for hook in getattr(self, '_forward_hooks', {}).values():
                        hr = hook(self, args, result)
                        if hr is not None:
                            result = hr
                    return result
                return _hooked_call
            cls.__call__ = _make_hooked(orig)


# ==================== NEW TENSOR ACTIVATION METHODS ====================
# Patched onto Tensor class

def _elu(self, alpha=1.0):
    """ELU: alpha*(exp(x)-1) for x<0, x for x>=0."""
    xp = self.xp
    out_data = xp.where(self.data >= 0, self.data, alpha * (xp.exp(self.data) - 1))
    out = Tensor(out_data, (self,), 'ELU', device=self._device)
    def _backward():
        self.grad += xp.where(self.data >= 0, 1.0, out_data + alpha) * out.grad
    out._backward = _backward
    return out

def _celu(self, alpha=1.0):
    """CELU: max(0,x) + min(0, alpha*(exp(x/alpha)-1))."""
    xp = self.xp
    out_data = xp.maximum(0, self.data) + xp.minimum(0, alpha * (xp.exp(self.data / alpha) - 1))
    out = Tensor(out_data, (self,), 'CELU', device=self._device)
    def _backward():
        pos = (self.data >= 0).astype(self.data.dtype)
        neg_deriv = xp.exp(xp.minimum(self.data, 0) / alpha)
        self.grad += (pos + (1 - pos) * neg_deriv) * out.grad
    out._backward = _backward
    return out

def _mish(self):
    """Mish: x * tanh(softplus(x)) = x * tanh(ln(1+exp(x)))."""
    xp = self.xp
    sp = xp.log1p(xp.exp(self.data))
    tsp = xp.tanh(sp)
    out_data = self.data * tsp
    out = Tensor(out_data, (self,), 'Mish', device=self._device)
    def _backward():
        sig = 1 / (1 + xp.exp(-self.data))
        delta = tsp + self.data * (1 - tsp**2) * sig
        self.grad += delta * out.grad
    out._backward = _backward
    return out

def _hardswish(self):
    """Hard-Swish: x * ReLU6(x+3)/6."""
    xp = self.xp
    relu6 = xp.clip(self.data + 3, 0, 6)
    out_data = self.data * relu6 / 6.0
    out = Tensor(out_data, (self,), 'Hardswish', device=self._device)
    def _backward():
        grad = xp.where(self.data <= -3, 0.0,
               xp.where(self.data >= 3, 1.0,
                        (2 * self.data + 3) / 6.0))
        self.grad += grad * out.grad
    out._backward = _backward
    return out

def _log_sigmoid(self):
    """LogSigmoid: log(sigmoid(x)) = -softplus(-x)."""
    xp = self.xp
    out_data = -xp.log1p(xp.exp(-self.data))
    out = Tensor(out_data, (self,), 'LogSigmoid', device=self._device)
    def _backward():
        self.grad += (1 - 1 / (1 + xp.exp(-self.data))) * out.grad
    out._backward = _backward
    return out

Tensor.elu         = _elu
Tensor.celu        = _celu
Tensor.mish        = _mish
Tensor.hardswish   = _hardswish
Tensor.log_sigmoid = _log_sigmoid


# ==================== TENSOR UTILITY FUNCTIONS ====================

def zeros_like(t: Tensor, requires_grad: bool = False) -> Tensor:
    """Return a zero tensor with the same shape, dtype, and device as t."""
    xp = t.xp
    return Tensor(xp.zeros_like(t.data), device=t._device, requires_grad=requires_grad, dtype=t.dtype)

def ones_like(t: Tensor, requires_grad: bool = False) -> Tensor:
    """Return a ones tensor with the same shape, dtype, and device as t."""
    xp = t.xp
    return Tensor(xp.ones_like(t.data), device=t._device, requires_grad=requires_grad, dtype=t.dtype)

def full(shape, fill_value, device=None, dtype=None, requires_grad: bool = False) -> Tensor:
    """Return a tensor filled with fill_value."""
    device = device or get_device()
    dtype = dtype or get_default_dtype()
    xp = cp if device == 'cuda' else np
    return Tensor(xp.full(shape, fill_value, dtype=dtype), device=device, requires_grad=requires_grad)

def full_like(t: Tensor, fill_value, requires_grad: bool = False) -> Tensor:
    """Return a tensor filled with fill_value, same shape/dtype/device as t."""
    xp = t.xp
    return Tensor(xp.full_like(t.data, fill_value), device=t._device, requires_grad=requires_grad, dtype=t.dtype)

def empty(shape, device=None, dtype=None, requires_grad: bool = False) -> Tensor:
    """Return an uninitialized tensor."""
    device = device or get_device()
    dtype = dtype or get_default_dtype()
    xp = cp if device == 'cuda' else np
    return Tensor(xp.empty(shape, dtype=dtype), device=device, requires_grad=requires_grad)

def empty_like(t: Tensor, requires_grad: bool = False) -> Tensor:
    """Return an uninitialized tensor with same shape/dtype/device as t."""
    xp = t.xp
    return Tensor(xp.empty_like(t.data), device=t._device, requires_grad=requires_grad, dtype=t.dtype)

def randint(low, high, shape, device=None, dtype=None, requires_grad: bool = False) -> Tensor:
    """Return a tensor of random integers in [low, high)."""
    device = device or get_device()
    dtype = dtype or (cp.int64 if device == 'cuda' else np.int64)
    xp = cp if device == 'cuda' else np
    return Tensor(xp.random.randint(low, high, shape).astype(dtype), device=device, requires_grad=requires_grad)


# ==================== CONTAINER MODULES ====================

class Sequential(Module):
    """A sequential container that applies layers in order."""

    def __init__(self, *layers):
        super().__init__()
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            layers = layers[0]
        self._layers = list(layers)

    def __call__(self, x):
        for layer in self._layers:
            x = layer(x)
        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), x)
            if hr is not None:
                x = hr
        return x

    def parameters(self):
        params, seen = [], set()
        for layer in self._layers:
            if isinstance(layer, Module):
                for p in layer.parameters():
                    if id(p) not in seen:
                        params.append(p)
                        seen.add(id(p))
        return params

    def named_parameters(self):
        params, seen = [], set()
        for i, layer in enumerate(self._layers):
            if isinstance(layer, Module):
                for name, p in layer.named_parameters():
                    if id(p) not in seen:
                        params.append((f"{i}.{name}", p))
                        seen.add(id(p))
        return params

    def named_modules(self, memo=None, prefix=''):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for i, layer in enumerate(self._layers):
                if isinstance(layer, Module):
                    sub_prefix = f"{prefix}.{i}" if prefix else str(i)
                    yield from layer.named_modules(memo, sub_prefix)

    def train(self):
        self._training = True
        for layer in self._layers:
            if isinstance(layer, Module):
                layer.train()
        return self

    def eval(self):
        self._training = False
        for layer in self._layers:
            if isinstance(layer, Module):
                layer.eval()
        return self

    def to(self, device):
        self._device = device
        for layer in self._layers:
            if isinstance(layer, Module):
                layer.to(device)
        return self

    def __getitem__(self, idx):
        return self._layers[idx]

    def __len__(self):
        return len(self._layers)

    def __repr__(self):
        lines = [f"Sequential("]
        for i, layer in enumerate(self._layers):
            lines.append(f"  ({i}): {repr(layer)}")
        lines.append(")")
        return "\n".join(lines)


class ModuleList(Module):
    """Holds a list of modules. Parameters from all are registered."""

    def __init__(self, modules=None):
        super().__init__()
        self._modules = list(modules) if modules is not None else []

    def append(self, module):
        self._modules.append(module)
        return self

    def extend(self, modules):
        self._modules.extend(modules)
        return self

    def parameters(self):
        params, seen = [], set()
        for m in self._modules:
            if isinstance(m, Module):
                for p in m.parameters():
                    if id(p) not in seen:
                        params.append(p)
                        seen.add(id(p))
        return params

    def named_parameters(self):
        params, seen = [], set()
        for i, m in enumerate(self._modules):
            if isinstance(m, Module):
                for name, p in m.named_parameters():
                    if id(p) not in seen:
                        params.append((f"{i}.{name}", p))
                        seen.add(id(p))
        return params

    def named_modules(self, memo=None, prefix=''):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for i, m in enumerate(self._modules):
                if isinstance(m, Module):
                    sub = f"{prefix}.{i}" if prefix else str(i)
                    yield from m.named_modules(memo, sub)

    def train(self):
        self._training = True
        for m in self._modules:
            if isinstance(m, Module):
                m.train()
        return self

    def eval(self):
        self._training = False
        for m in self._modules:
            if isinstance(m, Module):
                m.eval()
        return self

    def to(self, device):
        self._device = device
        for m in self._modules:
            if isinstance(m, Module):
                m.to(device)
        return self

    def __getitem__(self, idx):
        return self._modules[idx]

    def __setitem__(self, idx, module):
        self._modules[idx] = module

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def __repr__(self):
        lines = ["ModuleList("]
        for i, m in enumerate(self._modules):
            lines.append(f"  ({i}): {repr(m)}")
        lines.append(")")
        return "\n".join(lines)


class ModuleDict(Module):
    """Holds a dict of named modules. Parameters from all are registered."""

    def __init__(self, modules=None):
        super().__init__()
        self._modules = dict(modules) if modules is not None else {}

    def parameters(self):
        params, seen = [], set()
        for m in self._modules.values():
            if isinstance(m, Module):
                for p in m.parameters():
                    if id(p) not in seen:
                        params.append(p)
                        seen.add(id(p))
        return params

    def named_parameters(self):
        params, seen = [], set()
        for key, m in self._modules.items():
            if isinstance(m, Module):
                for name, p in m.named_parameters():
                    if id(p) not in seen:
                        params.append((f"{key}.{name}", p))
                        seen.add(id(p))
        return params

    def named_modules(self, memo=None, prefix=''):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for key, m in self._modules.items():
                if isinstance(m, Module):
                    sub = f"{prefix}.{key}" if prefix else key
                    yield from m.named_modules(memo, sub)

    def train(self):
        self._training = True
        for m in self._modules.values():
            if isinstance(m, Module):
                m.train()
        return self

    def eval(self):
        self._training = False
        for m in self._modules.values():
            if isinstance(m, Module):
                m.eval()
        return self

    def to(self, device):
        self._device = device
        for m in self._modules.values():
            if isinstance(m, Module):
                m.to(device)
        return self

    def __getitem__(self, key):
        return self._modules[key]

    def __setitem__(self, key, module):
        self._modules[key] = module

    def __delitem__(self, key):
        del self._modules[key]

    def __contains__(self, key):
        return key in self._modules

    def __len__(self):
        return len(self._modules)

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()

    def __repr__(self):
        lines = ["ModuleDict("]
        for key, m in self._modules.items():
            lines.append(f"  ({key}): {repr(m)}")
        lines.append(")")
        return "\n".join(lines)


# ==================== 1D PRIMITIVES ====================

class Conv1D(Module):
    """1D convolution over (N, C_in, L) tensors."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by groups ({groups})")
        if out_channels % groups != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by groups ({groups})")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        xp = cp if get_device() == 'cuda' else np
        scale = float(xp.sqrt(2.0 / (in_channels * kernel_size)))
        self.w = Tensor(
            xp.random.randn(out_channels, in_channels // groups, kernel_size).astype(xp.float32) * scale,
            device=get_device())
        self.b = Tensor(xp.zeros(out_channels, dtype=xp.float32), device=get_device()) if bias else None

    def __call__(self, x):
        # x: (N, C_in, L)
        xp = cp if self._device == 'cuda' else np
        N, C_in, L = x.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding
        D = self.dilation
        L_out = (L + 2 * P - D * (K - 1) - 1) // S + 1

        # im2col for 1D: unfold input into columns
        x_padded_data = xp.pad(x.data, ((0, 0), (0, 0), (P, P)))
        # shape: (N, C_in, L_out, K)
        cols = xp.stack([x_padded_data[:, :, s * S: s * S + K * D: D] for s in range(L_out)], axis=2)
        # cols: (N, C_in, L_out, K) -> (N, C_in*K, L_out) for grouped matmul
        G = self.groups
        C_per_g = C_in // G
        cols = cols.reshape(N, G, C_per_g * K, L_out)
        w_data = self.w.data.reshape(G, self.out_channels // G, C_per_g * K)
        # out: (N, G, out//G, L_out)
        out_data = xp.einsum('ngkl,gok->ngol', cols, w_data).reshape(N, self.out_channels, L_out)

        if self.use_bias:
            out_data = out_data + self.b.data.reshape(1, -1, 1)

        children = (x, self.w) + ((self.b,) if self.use_bias else ())
        out = Tensor(out_data, children, 'Conv1D', device=self._device)

        def _backward():
            grad = out.grad  # (N, C_out, L_out)
            # bias grad
            if self.use_bias:
                self.b.grad += grad.sum(axis=(0, 2))
            # weight grad
            grad_g = grad.reshape(N, G, self.out_channels // G, L_out)
            self.w.grad += xp.einsum('ngol,ngkl->gok', grad_g, cols).reshape(self.w.data.shape)
            # input grad
            if x.grad is not None:
                w_g = self.w.data.reshape(G, self.out_channels // G, C_per_g * K)
                dx_cols = xp.einsum('gok,ngol->ngkl', w_g, grad_g).reshape(N, C_in * K, L_out)
                dx_pad = xp.zeros_like(x_padded_data)
                for s in range(L_out):
                    for k in range(K):
                        dx_pad[:, :, s * S + k * D] += dx_cols[:, k::K, s].reshape(N, C_in)
                x.grad += dx_pad[:, :, P: P + L] if P > 0 else dx_pad

        out._backward = _backward
        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), out)
            if hr is not None:
                out = hr
        return out

    def parameters(self):
        return [self.w, self.b] if self.use_bias else [self.w]

    def __repr__(self):
        return (f"Conv1D({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding})")


class MaxPool1D(Module):
    """1D max pooling over (N, C, L) tensors."""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def __call__(self, x):
        xp = cp if self._device == 'cuda' else np
        N, C, L = x.shape
        K, S, P = self.kernel_size, self.stride, self.padding
        L_out = (L + 2 * P - K) // S + 1
        x_pad = xp.pad(x.data, ((0, 0), (0, 0), (P, P)), constant_values=-xp.inf)
        # (N, C, L_out, K)
        windows = xp.stack([x_pad[:, :, s * S: s * S + K] for s in range(L_out)], axis=2)
        out_data = windows.max(axis=3)
        out = Tensor(out_data, (x,), 'MaxPool1D', device=self._device)

        def _backward():
            mask = (windows == out_data[:, :, :, None])
            mask = mask / mask.sum(axis=3, keepdims=True)
            dx_pad = xp.zeros_like(x_pad)
            for s in range(L_out):
                dx_pad[:, :, s * S: s * S + K] += mask[:, :, s, :] * out.grad[:, :, s: s + 1]
            x.grad += dx_pad[:, :, P: P + L] if P > 0 else dx_pad

        out._backward = _backward
        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), out)
            if hr is not None:
                out = hr
        return out

    def parameters(self):
        return []

    def __repr__(self):
        return f"MaxPool1D(kernel_size={self.kernel_size}, stride={self.stride})"


class AvgPool1D(Module):
    """1D average pooling over (N, C, L) tensors."""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def __call__(self, x):
        xp = cp if self._device == 'cuda' else np
        N, C, L = x.shape
        K, S, P = self.kernel_size, self.stride, self.padding
        L_out = (L + 2 * P - K) // S + 1
        x_pad = xp.pad(x.data, ((0, 0), (0, 0), (P, P)))
        windows = xp.stack([x_pad[:, :, s * S: s * S + K] for s in range(L_out)], axis=2)
        out_data = windows.mean(axis=3)
        out = Tensor(out_data, (x,), 'AvgPool1D', device=self._device)

        def _backward():
            dx_pad = xp.zeros_like(x_pad)
            for s in range(L_out):
                dx_pad[:, :, s * S: s * S + K] += out.grad[:, :, s: s + 1] / K
            x.grad += dx_pad[:, :, P: P + L] if P > 0 else dx_pad

        out._backward = _backward
        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), out)
            if hr is not None:
                out = hr
        return out

    def parameters(self):
        return []

    def __repr__(self):
        return f"AvgPool1D(kernel_size={self.kernel_size}, stride={self.stride})"


class BatchNorm1D(Module):
    """Batch normalization for (N, C) or (N, C, L) tensors."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        xp = cp if get_device() == 'cuda' else np
        if affine:
            self.gamma = Tensor(xp.ones(num_features, dtype=xp.float32), device=get_device())
            self.beta  = Tensor(xp.zeros(num_features, dtype=xp.float32), device=get_device())
        else:
            self.gamma = self.beta = None
        self.running_mean = xp.zeros(num_features, dtype=xp.float32)
        self.running_var  = xp.ones(num_features, dtype=xp.float32)
        self._cache = None

    def __call__(self, x):
        xp = cp if self._device == 'cuda' else np
        is_3d = x.ndim == 3  # (N, C, L)
        if is_3d:
            N, C, L = x.shape
            x_2d_data = x.data.transpose(0, 2, 1).reshape(N * L, C)
        else:
            x_2d_data = x.data  # (N, C)

        if self._training:
            mean = x_2d_data.mean(axis=0)
            var  = x_2d_data.var(axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            mean = self.running_mean
            var  = self.running_var

        std_inv = 1.0 / xp.sqrt(var + self.eps)
        x_norm  = (x_2d_data - mean) * std_inv

        if self.affine:
            out_2d = x_norm * self.gamma.data + self.beta.data
        else:
            out_2d = x_norm

        if is_3d:
            out_data = out_2d.reshape(N, L, C).transpose(0, 2, 1)
        else:
            out_data = out_2d

        self._cache = (x_norm, std_inv, mean, is_3d, x_2d_data.shape[0])
        children = (x,) + ((self.gamma, self.beta) if self.affine else ())
        out = Tensor(out_data, children, 'BatchNorm1D', device=self._device)

        def _backward():
            x_norm_, std_inv_, _, is_3d_, N_ = self._cache
            dout = out.grad
            if is_3d_:
                dout = dout.transpose(0, 2, 1).reshape(N_, -1)  # (N*L, C) — reuse N_ as batch dim
            if self.affine:
                self.gamma.grad += (dout * x_norm_).sum(axis=0)
                self.beta.grad  += dout.sum(axis=0)
                dout = dout * self.gamma.data
            dx_norm = dout
            N_eff = dx_norm.shape[0]
            dx = std_inv_ * (dx_norm - dx_norm.mean(0) - x_norm_ * (dx_norm * x_norm_).mean(0))
            if is_3d_:
                Norig, _, L_ = x.shape
                dx = dx.reshape(Norig, L_, -1).transpose(0, 2, 1)
            x.grad += dx

        out._backward = _backward
        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), out)
            if hr is not None:
                out = hr
        return out

    def parameters(self):
        return [self.gamma, self.beta] if self.affine else []

    def __repr__(self):
        return f"BatchNorm1D({self.num_features}, eps={self.eps}, momentum={self.momentum})"


# ==================== RECURRENT LAYERS ====================

class LSTMCell(Module):
    """Single-step LSTM cell: (x, (h, c)) -> (h_new, c_new)."""

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.use_bias    = bias
        xp = cp if get_device() == 'cuda' else np
        # Weight for [i, f, g, o] gates, shape (input_size, 4*hidden_size)
        scale = float(xp.sqrt(1.0 / hidden_size))
        self.w_ih = Tensor(xp.random.uniform(-scale, scale, (input_size,  4 * hidden_size)).astype(xp.float32), device=get_device())
        self.w_hh = Tensor(xp.random.uniform(-scale, scale, (hidden_size, 4 * hidden_size)).astype(xp.float32), device=get_device())
        if bias:
            self.b_ih = Tensor(xp.zeros(4 * hidden_size, dtype=xp.float32), device=get_device())
            self.b_hh = Tensor(xp.zeros(4 * hidden_size, dtype=xp.float32), device=get_device())
        else:
            self.b_ih = self.b_hh = None

    def __call__(self, x, state=None):
        # x: (N, input_size); state: ((N, H), (N, H)) or None
        xp = cp if self._device == 'cuda' else np
        N = x.shape[0]
        H = self.hidden_size
        if state is None:
            h = Tensor(xp.zeros((N, H), dtype=xp.float32), device=self._device, requires_grad=False)
            c = Tensor(xp.zeros((N, H), dtype=xp.float32), device=self._device, requires_grad=False)
        else:
            h, c = state

        gates = x @ self.w_ih + h @ self.w_hh
        if self.use_bias:
            gates = gates + self.b_ih + self.b_hh

        # split into 4 gates
        gi, gf, gg, go = [gates.data[:, k * H: (k + 1) * H] for k in range(4)]

        def _sigmoid(v):
            return xp.where(v >= 0, 1 / (1 + xp.exp(-v)), xp.exp(v) / (1 + xp.exp(v)))

        i_gate = _sigmoid(gi)
        f_gate = _sigmoid(gf)
        g_gate = xp.tanh(gg)
        o_gate = _sigmoid(go)

        c_new_data = f_gate * c.data + i_gate * g_gate
        h_new_data = o_gate * xp.tanh(c_new_data)

        parents = (x, self.w_ih, self.w_hh, h, c) + ((self.b_ih, self.b_hh) if self.use_bias else ())
        c_new = Tensor(c_new_data, parents, 'LSTMCell_c', device=self._device)
        h_new = Tensor(h_new_data, (c_new,) + parents, 'LSTMCell_h', device=self._device)

        def _backward():
            dh = h_new.grad                     # (N, H)
            dc_from_h = dh * o_gate * (1 - xp.tanh(c_new_data) ** 2)
            dc = c_new.grad + dc_from_h         # (N, H)

            do = dh * xp.tanh(c_new_data)
            di = dc * g_gate
            df = dc * c.data
            dg = dc * i_gate

            do_pre = do * o_gate * (1 - o_gate)
            di_pre = di * i_gate * (1 - i_gate)
            df_pre = df * f_gate * (1 - f_gate)
            dg_pre = dg * (1 - g_gate ** 2)

            d_gates = xp.concatenate([di_pre, df_pre, dg_pre, do_pre], axis=1)

            self.w_ih.grad += x.data.T @ d_gates
            self.w_hh.grad += h.data.T @ d_gates
            if self.use_bias:
                self.b_ih.grad += d_gates.sum(0)
                self.b_hh.grad += d_gates.sum(0)
            if x.grad is not None:
                x.grad += d_gates @ self.w_ih.data.T
            if h.grad is not None:
                h.grad += d_gates @ self.w_hh.data.T
            if c.grad is not None:
                c.grad += dc * f_gate

        h_new._backward = _backward
        c_new._backward = lambda: None  # handled inside h_new backward
        return h_new, c_new

    def parameters(self):
        p = [self.w_ih, self.w_hh]
        if self.use_bias:
            p += [self.b_ih, self.b_hh]
        return p

    def __repr__(self):
        return f"LSTMCell({self.input_size}, {self.hidden_size})"


class LSTM(Module):
    """Multi-layer LSTM over sequences (N, L, input_size) -> (N, L, hidden_size*num_dir)."""

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=True, dropout=0.0, bidirectional=False):
        super().__init__()
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.batch_first  = batch_first
        self.dropout_p    = dropout
        self.bidirectional = bidirectional
        num_dir = 2 if bidirectional else 1

        self.cells = ModuleList()
        for layer in range(num_layers):
            in_sz = input_size if layer == 0 else hidden_size * num_dir
            self.cells.append(LSTMCell(in_sz, hidden_size, bias))
            if bidirectional:
                self.cells.append(LSTMCell(in_sz, hidden_size, bias))

        self._dropout = Dropout(dropout) if dropout > 0 else None

    def __call__(self, x, state=None):
        # x: (N, L, C) if batch_first else (L, N, C)
        xp = cp if self._device == 'cuda' else np
        if not self.batch_first:
            x_data = x.data.transpose(1, 0, 2)
        else:
            x_data = x.data
        N, L, _ = x_data.shape
        H = self.hidden_size
        num_dir = 2 if self.bidirectional else 1

        if state is None:
            h0 = [Tensor(xp.zeros((N, H), dtype=xp.float32), device=self._device, requires_grad=False)
                  for _ in range(self.num_layers * num_dir)]
            c0 = [Tensor(xp.zeros((N, H), dtype=xp.float32), device=self._device, requires_grad=False)
                  for _ in range(self.num_layers * num_dir)]
        else:
            h0_raw, c0_raw = state
            h0 = [Tensor(h0_raw.data[i], device=self._device, requires_grad=False)
                  for i in range(self.num_layers * num_dir)]
            c0 = [Tensor(c0_raw.data[i], device=self._device, requires_grad=False)
                  for i in range(self.num_layers * num_dir)]

        layer_input = [Tensor(x_data[:, t, :], device=self._device, requires_grad=x._requires_grad)
                       for t in range(L)]

        h_n, c_n = [], []
        for layer in range(self.num_layers):
            fwd_cell = self.cells[layer * num_dir]
            h, c = h0[layer * num_dir], c0[layer * num_dir]
            fwd_out = []
            for t in range(L):
                h, c = fwd_cell(layer_input[t], (h, c))
                fwd_out.append(h)
            h_n.append(h); c_n.append(c)

            if self.bidirectional:
                bwd_cell = self.cells[layer * num_dir + 1]
                hb, cb = h0[layer * num_dir + 1], c0[layer * num_dir + 1]
                bwd_out = []
                for t in reversed(range(L)):
                    hb, cb = bwd_cell(layer_input[t], (hb, cb))
                    bwd_out.insert(0, hb)
                h_n.append(hb); c_n.append(cb)
                layer_input = [cat([f, b], axis=-1) for f, b in zip(fwd_out, bwd_out)]
            else:
                layer_input = fwd_out

            if self._dropout is not None and self._training and layer < self.num_layers - 1:
                layer_input = [self._dropout(t) for t in layer_input]

        # stack outputs along time axis
        outputs = stack(layer_input, axis=1)  # (N, L, H*num_dir)
        if not self.batch_first:
            # transpose to (L, N, H)
            outputs = Tensor(outputs.data.transpose(1, 0, 2), (outputs,), device=self._device)

        h_n_stack = Tensor(xp.stack([h.data for h in h_n], axis=0), device=self._device, requires_grad=False)
        c_n_stack = Tensor(xp.stack([c.data for c in c_n], axis=0), device=self._device, requires_grad=False)

        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), outputs)
            if hr is not None:
                outputs = hr
        return outputs, (h_n_stack, c_n_stack)

    def parameters(self):
        return self.cells.parameters()

    def __repr__(self):
        return (f"LSTM({self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
                f"bidirectional={self.bidirectional})")


class GRUCell(Module):
    """Single-step GRU cell: (x, h) -> h_new."""

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.use_bias    = bias
        xp = cp if get_device() == 'cuda' else np
        scale = float(xp.sqrt(1.0 / hidden_size))
        # weights for [r, z, n] gates
        self.w_ih = Tensor(xp.random.uniform(-scale, scale, (input_size,  3 * hidden_size)).astype(xp.float32), device=get_device())
        self.w_hh = Tensor(xp.random.uniform(-scale, scale, (hidden_size, 3 * hidden_size)).astype(xp.float32), device=get_device())
        if bias:
            self.b_ih = Tensor(xp.zeros(3 * hidden_size, dtype=xp.float32), device=get_device())
            self.b_hh = Tensor(xp.zeros(3 * hidden_size, dtype=xp.float32), device=get_device())
        else:
            self.b_ih = self.b_hh = None

    def __call__(self, x, h=None):
        xp = cp if self._device == 'cuda' else np
        N, H = x.shape[0], self.hidden_size
        if h is None:
            h = Tensor(xp.zeros((N, H), dtype=xp.float32), device=self._device, requires_grad=False)

        def _sigmoid(v):
            return xp.where(v >= 0, 1 / (1 + xp.exp(-v)), xp.exp(v) / (1 + xp.exp(v)))

        gates_x = x.data @ self.w_ih.data
        gates_h = h.data @ self.w_hh.data
        if self.use_bias:
            gates_x += self.b_ih.data
            gates_h += self.b_hh.data

        r = _sigmoid(gates_x[:, :H]      + gates_h[:, :H])
        z = _sigmoid(gates_x[:, H: 2*H]  + gates_h[:, H: 2*H])
        n = xp.tanh( gates_x[:, 2*H:]    + r * gates_h[:, 2*H:])

        h_new_data = (1 - z) * n + z * h.data
        parents = (x, self.w_ih, self.w_hh, h) + ((self.b_ih, self.b_hh) if self.use_bias else ())
        h_new = Tensor(h_new_data, parents, 'GRUCell', device=self._device)

        def _backward():
            dh_new = h_new.grad
            dz = dh_new * (h.data - n)
            dn = dh_new * (1 - z)
            dh_prev_from_z = dh_new * z

            dn_pre = dn * (1 - n ** 2)
            dz_pre = dz * z * (1 - z)

            dr_part = dn_pre * gates_h[:, 2*H:]  # contribution to r gate
            dr_pre  = dr_part * r * (1 - r)

            d_gates_x = xp.concatenate([dr_pre, dz_pre, dn_pre], axis=1)
            d_gates_hh_rz = xp.concatenate([dr_pre, dz_pre], axis=1)
            d_gates_hh_n  = dn_pre * r

            self.w_ih.grad += x.data.T @ d_gates_x
            self.w_hh.grad += h.data.T @ xp.concatenate([d_gates_hh_rz, d_gates_hh_n], axis=1)
            if self.use_bias:
                self.b_ih.grad += d_gates_x.sum(0)
                self.b_hh.grad += xp.concatenate([d_gates_hh_rz, d_gates_hh_n], axis=1).sum(0)
            if x.grad is not None:
                x.grad += d_gates_x @ self.w_ih.data.T
            if h.grad is not None:
                h.grad += (xp.concatenate([d_gates_hh_rz, d_gates_hh_n], axis=1) @ self.w_hh.data.T
                           + dh_prev_from_z)

        h_new._backward = _backward
        return h_new

    def parameters(self):
        p = [self.w_ih, self.w_hh]
        if self.use_bias:
            p += [self.b_ih, self.b_hh]
        return p

    def __repr__(self):
        return f"GRUCell({self.input_size}, {self.hidden_size})"


class GRU(Module):
    """Multi-layer GRU over sequences (N, L, input_size) -> (N, L, hidden_size*num_dir)."""

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=True, dropout=0.0, bidirectional=False):
        super().__init__()
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.batch_first  = batch_first
        self.dropout_p    = dropout
        self.bidirectional = bidirectional
        num_dir = 2 if bidirectional else 1

        self.cells = ModuleList()
        for layer in range(num_layers):
            in_sz = input_size if layer == 0 else hidden_size * num_dir
            self.cells.append(GRUCell(in_sz, hidden_size, bias))
            if bidirectional:
                self.cells.append(GRUCell(in_sz, hidden_size, bias))

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

        outputs = stack(layer_input, axis=1)
        if not self.batch_first:
            outputs = Tensor(outputs.data.transpose(1, 0, 2), (outputs,), device=self._device)

        h_n_stack = Tensor(xp.stack([h.data for h in h_n], axis=0), device=self._device, requires_grad=False)

        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), outputs)
            if hr is not None:
                outputs = hr
        return outputs, h_n_stack

    def parameters(self):
        return self.cells.parameters()

    def __repr__(self):
        return (f"GRU({self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
                f"bidirectional={self.bidirectional})")


# ==================== TRANSFORMER BLOCK ====================

class PositionalEncoding(Module):
    """Sinusoidal and learned positional encodings."""

    def __init__(self, d_model, max_len=4096, mode='sinusoidal', dropout=0.0):
        super().__init__()
        self.mode = mode
        self.dropout_p = dropout
        self._dropout = Dropout(dropout) if dropout > 0 else None
        xp = cp if get_device() == 'cuda' else np

        if mode == 'sinusoidal':
            pe = xp.zeros((max_len, d_model), dtype=xp.float32)
            position = xp.arange(0, max_len, dtype=xp.float32).reshape(-1, 1)
            div_term = xp.exp(xp.arange(0, d_model, 2, dtype=xp.float32) * (-xp.log(10000.0) / d_model))
            pe[:, 0::2] = xp.sin(position * div_term)
            pe[:, 1::2] = xp.cos(position * div_term[:d_model // 2])
            self._pe = pe  # (max_len, d_model), not a parameter
        else:  # learned
            self.pe = Tensor(xp.zeros((max_len, d_model), dtype=xp.float32), device=get_device())

    def __call__(self, x):
        # x: (N, L, D)
        xp = cp if self._device == 'cuda' else np
        L = x.shape[1]
        if self.mode == 'sinusoidal':
            pe_data = self._pe[:L]
            out_data = x.data + pe_data[None, :, :]
            out = Tensor(out_data, (x,), 'PosEnc', device=self._device)
            def _backward():
                x.grad += out.grad
            out._backward = _backward
        else:
            out = x + Tensor(self.pe.data[:L][None], device=self._device, requires_grad=False)
            out._op = 'PosEnc'
        if self._dropout is not None:
            out = self._dropout(out)
        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), out)
            if hr is not None:
                out = hr
        return out

    def parameters(self):
        return [self.pe] if self.mode == 'learned' else []

    def __repr__(self):
        return f"PositionalEncoding(mode='{self.mode}', dropout={self.dropout_p})"


class TransformerEncoderLayer(Module):
    """Pre-norm Transformer encoder layer: LN -> MHA -> residual -> LN -> FFN -> residual."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='gelu', norm_first=True, device=None):
        super().__init__()
        dev = device or get_device()
        self._device = dev
        self.norm_first = norm_first

        self.self_attn = _MultiHeadAttentionModule(d_model, nhead, dropout, dev)
        self.ff1       = Linear(d_model, dim_feedforward)
        self.ff2       = Linear(dim_feedforward, d_model)
        self.norm1     = LayerNorm(d_model)
        self.norm2     = LayerNorm(d_model)
        self._dropout  = Dropout(dropout) if dropout > 0 else None

        act_map = {'gelu': lambda t: t.gelu(), 'relu': lambda t: t.relu(),
                   'silu': lambda t: t.silu(), 'mish': lambda t: t.mish()}
        if activation not in act_map:
            raise ValueError(f"activation must be one of {list(act_map)}, got '{activation}'")
        self._act = act_map[activation]

    def __call__(self, x, mask=None):
        if self.norm_first:
            # Pre-norm
            attn_out = self.self_attn(self.norm1(x), mask=mask)
            if self._dropout:
                attn_out = self._dropout(attn_out)
            x = x + attn_out
            ff_in = self.norm2(x)
        else:
            attn_out = self.self_attn(x, mask=mask)
            if self._dropout:
                attn_out = self._dropout(attn_out)
            x = self.norm1(x + attn_out)
            ff_in = x

        ff_out = self.ff2(self._act(self.ff1(ff_in)))
        if self._dropout:
            ff_out = self._dropout(ff_out)

        out = x + ff_out if self.norm_first else self.norm2(x + ff_out)
        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), out)
            if hr is not None:
                out = hr
        return out

    def parameters(self):
        params, seen = [], set()
        for sub in [self.self_attn, self.ff1, self.ff2, self.norm1, self.norm2]:
            for p in sub.parameters():
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
        return params

    def __repr__(self):
        return (f"TransformerEncoderLayer(d_model={self.self_attn.d_model}, "
                f"nhead={self.self_attn.nhead}, norm_first={self.norm_first})")


class _MultiHeadAttentionModule(Module):
    """MHA wrapper that uses the library's multi_head_attention function."""

    def __init__(self, d_model, nhead, dropout=0.0, device=None):
        super().__init__()
        self.d_model = d_model
        self.nhead   = nhead
        self._device = device or get_device()
        xp = cp if self._device == 'cuda' else np
        scale = float(xp.sqrt(1.0 / d_model))
        # Fused QKV projection + output projection
        self.w_q  = Tensor(xp.random.randn(d_model, d_model).astype(xp.float32) * scale, device=self._device)
        self.w_k  = Tensor(xp.random.randn(d_model, d_model).astype(xp.float32) * scale, device=self._device)
        self.w_v  = Tensor(xp.random.randn(d_model, d_model).astype(xp.float32) * scale, device=self._device)
        self.w_o  = Tensor(xp.random.randn(d_model, d_model).astype(xp.float32) * scale, device=self._device)
        self.b_o  = Tensor(xp.zeros(d_model, dtype=xp.float32), device=self._device)
        self.dropout_p = dropout

    def __call__(self, x, mask=None):
        # x: (N, L, D)
        N, L, D = x.shape
        H = self.nhead
        d_h = D // H

        Q = x @ self.w_q
        K = x @ self.w_k
        V = x @ self.w_v

        # reshape to (N, H, L, d_h)
        def _split_heads(t):
            data = t.data.reshape(N, L, H, d_h).transpose(0, 2, 1, 3)
            return Tensor(data, (t,), device=self._device)

        Qh, Kh, Vh = _split_heads(Q), _split_heads(K), _split_heads(V)
        scale_val = float((cp if self._device == 'cuda' else np).sqrt(d_h))

        attn_out = scaled_dot_product_attention(Qh, Kh, Vh,
                                                scale=1.0 / scale_val,
                                                dropout_p=self.dropout_p if self._training else 0.0,
                                                is_causal=False)
        # merge heads: (N, H, L, d_h) -> (N, L, D)
        merged_data = attn_out.data.transpose(0, 2, 1, 3).reshape(N, L, D)
        merged = Tensor(merged_data, (attn_out,), 'MergeHeads', device=self._device)
        def _mh_bwd():
            if attn_out.grad is not None:
                attn_out.grad += merged.grad.reshape(N, L, H, d_h).transpose(0, 2, 1, 3)
        merged._backward = _mh_bwd

        out = merged @ self.w_o + self.b_o
        return out

    def parameters(self):
        return [self.w_q, self.w_k, self.w_v, self.w_o, self.b_o]

    def __repr__(self):
        return f"MultiHeadAttention(d_model={self.d_model}, nhead={self.nhead}, dropout={self.dropout_p})"


class TransformerEncoder(Module):
    """Stack of N TransformerEncoderLayers with optional final norm."""

    def __init__(self, encoder_layer_fn, num_layers, norm=None):
        """
        Args:
            encoder_layer_fn: callable () -> TransformerEncoderLayer, or a single
                              TransformerEncoderLayer to be deep-copied num_layers times.
            num_layers: number of sub-encoder-layers.
            norm: optional normalization module applied to the output.
        """
        super().__init__()
        import copy
        if callable(encoder_layer_fn) and not isinstance(encoder_layer_fn, Module):
            layers = [encoder_layer_fn() for _ in range(num_layers)]
        else:
            layers = [copy.deepcopy(encoder_layer_fn) for _ in range(num_layers)]
        self.layers = ModuleList(layers)
        self.norm = norm

    def __call__(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        if self.norm is not None:
            x = self.norm(x)
        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), x)
            if hr is not None:
                x = hr
        return x

    def parameters(self):
        params = self.layers.parameters()
        if self.norm is not None:
            params += self.norm.parameters()
        return params

    def __repr__(self):
        return f"TransformerEncoder(num_layers={len(self.layers)})"


# ==================== FUSED LINEAR + ACTIVATION ====================

# Fused CUDA kernels for linear+activation
_FUSED_LINEAR_GELU_KERNEL = r"""
extern "C" __global__
void fused_linear_gelu_fwd(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        float c = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        out[idx] = 0.5f * v * (1.0f + tanhf(c));
    }
}
extern "C" __global__
void fused_linear_gelu_bwd(const float* x, const float* dout, float* dx, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        float c = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        float tc = tanhf(c);
        float sech2 = 1.0f - tc * tc;
        float dc = 0.7978845608028654f * (1.0f + 3.0f * 0.044715f * v * v);
        dx[idx] = dout[idx] * (0.5f * (1.0f + tc) + 0.5f * v * sech2 * dc);
    }
}
"""

_FUSED_LINEAR_SILU_KERNEL = r"""
extern "C" __global__
void fused_linear_silu_fwd(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        float sig = 1.0f / (1.0f + expf(-v));
        out[idx] = v * sig;
    }
}
extern "C" __global__
void fused_linear_silu_bwd(const float* x, const float* dout, float* dx, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        float sig = 1.0f / (1.0f + expf(-v));
        dx[idx] = dout[idx] * sig * (1.0f + v * (1.0f - sig));
    }
}
"""


def _get_fused_gelu_kernels():
    fwd = _get_cached_kernel('fused_linear_gelu_fwd', _FUSED_LINEAR_GELU_KERNEL, 'fused_linear_gelu_fwd')
    bwd = _get_cached_kernel('fused_linear_gelu_bwd', _FUSED_LINEAR_GELU_KERNEL, 'fused_linear_gelu_bwd')
    return fwd, bwd


def _get_fused_silu_kernels():
    fwd = _get_cached_kernel('fused_linear_silu_fwd', _FUSED_LINEAR_SILU_KERNEL, 'fused_linear_silu_fwd')
    bwd = _get_cached_kernel('fused_linear_silu_bwd', _FUSED_LINEAR_SILU_KERNEL, 'fused_linear_silu_bwd')
    return fwd, bwd


class FusedLinearGELU(Module):
    """Linear layer followed by GELU, fused on CUDA to avoid extra memory round-trip."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        xp = cp if get_device() == 'cuda' else np
        scale = float(xp.sqrt(2.0 / in_features))
        self.w = Tensor(xp.random.randn(in_features, out_features).astype(xp.float32) * scale, device=get_device())
        self.use_bias = bias
        self.b = Tensor(xp.zeros(out_features, dtype=xp.float32), device=get_device()) if bias else None

    def __call__(self, x):
        lin = x @ self.w
        if self.use_bias:
            lin = lin + self.b

        xp = cp if self._device == 'cuda' else np
        if self._device == 'cuda' and lin.data.dtype == cp.float32:
            n = lin.data.size
            out_data = cp.empty_like(lin.data)
            bs = 256; gs = (n + bs - 1) // bs
            fwd_k, bwd_k = _get_fused_gelu_kernels()
            fwd_k((gs,), (bs,), (lin.data.ravel(), out_data.ravel(), n))
            out_data = out_data.reshape(lin.data.shape)
            out = Tensor(out_data, (lin,), 'FusedLinearGELU', device=self._device)
            def _backward():
                grad_in = cp.empty_like(lin.data)
                bwd_k((gs,), (bs,), (lin.data.ravel(), out.grad.ravel(), grad_in.ravel(), n))
                lin.grad += grad_in
            out._backward = _backward
        else:
            out = lin.gelu()
            out._op = 'FusedLinearGELU'

        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), out)
            if hr is not None:
                out = hr
        return out

    def parameters(self):
        return [self.w, self.b] if self.use_bias else [self.w]

    def __repr__(self):
        w = self.w.shape
        return f"FusedLinearGELU({w[0]}, {w[1]}, bias={self.use_bias})"


class FusedLinearSiLU(Module):
    """Linear layer followed by SiLU, fused on CUDA."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        xp = cp if get_device() == 'cuda' else np
        scale = float(xp.sqrt(2.0 / in_features))
        self.w = Tensor(xp.random.randn(in_features, out_features).astype(xp.float32) * scale, device=get_device())
        self.use_bias = bias
        self.b = Tensor(xp.zeros(out_features, dtype=xp.float32), device=get_device()) if bias else None

    def __call__(self, x):
        lin = x @ self.w
        if self.use_bias:
            lin = lin + self.b

        xp = cp if self._device == 'cuda' else np
        if self._device == 'cuda' and lin.data.dtype == cp.float32:
            n = lin.data.size
            out_data = cp.empty_like(lin.data)
            bs = 256; gs = (n + bs - 1) // bs
            fwd_k, bwd_k = _get_fused_silu_kernels()
            fwd_k((gs,), (bs,), (lin.data.ravel(), out_data.ravel(), n))
            out_data = out_data.reshape(lin.data.shape)
            out = Tensor(out_data, (lin,), 'FusedLinearSiLU', device=self._device)
            def _backward():
                grad_in = cp.empty_like(lin.data)
                bwd_k((gs,), (bs,), (lin.data.ravel(), out.grad.ravel(), grad_in.ravel(), n))
                lin.grad += grad_in
            out._backward = _backward
        else:
            out = lin.silu()
            out._op = 'FusedLinearSiLU'

        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), out)
            if hr is not None:
                out = hr
        return out

    def parameters(self):
        return [self.w, self.b] if self.use_bias else [self.w]

    def __repr__(self):
        w = self.w.shape
        return f"FusedLinearSiLU({w[0]}, {w[1]}, bias={self.use_bias})"


class GeGLU(Module):
    """Gated Linear Unit with GELU gate: output = first_half * GELU(second_half).

    Projects d_model -> 2*d_ff then gates; effectively maps d_model -> d_ff.
    Used in modern LLMs (PaLM, T5 v1.1) to improve quality over plain GELU FFN.
    """

    def __init__(self, d_model, d_ff, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_ff    = d_ff
        xp = cp if get_device() == 'cuda' else np
        scale = float(xp.sqrt(2.0 / d_model))
        # Single projection into 2*d_ff; first half is value, second half is gate
        self.w = Tensor(xp.random.randn(d_model, 2 * d_ff).astype(xp.float32) * scale, device=get_device())
        self.use_bias = bias
        self.b = Tensor(xp.zeros(2 * d_ff, dtype=xp.float32), device=get_device()) if bias else None

    def __call__(self, x):
        proj = x @ self.w
        if self.use_bias:
            proj = proj + self.b
        # Split into value and gate halves
        xp = cp if self._device == 'cuda' else np
        v_data = proj.data[..., :self.d_ff]
        g_data = proj.data[..., self.d_ff:]

        # GELU of gate half
        sqrt_2_pi = 0.7978845608028654
        g_cdf = 0.5 * (1 + xp.tanh(sqrt_2_pi * (g_data + 0.044715 * g_data**3)))
        gelu_g = g_data * g_cdf

        out_data = v_data * gelu_g
        out = Tensor(out_data, (proj,), 'GeGLU', device=self._device)

        def _backward():
            dout = out.grad
            # grad through gelu gate
            g3 = g_data ** 3
            inner = sqrt_2_pi * (g_data + 0.044715 * g3)
            tc = xp.tanh(inner)
            sech2 = 1 - tc ** 2
            d_inner = sqrt_2_pi * (1 + 3 * 0.044715 * g_data**2)
            d_gelu_g = 0.5 * (1 + tc) + 0.5 * g_data * sech2 * d_inner
            dg = dout * v_data * d_gelu_g
            dv = dout * gelu_g
            # combine and propagate
            d_proj = xp.concatenate([dv, dg], axis=-1)
            if proj.grad is not None:
                proj.grad += d_proj

        out._backward = _backward
        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), out)
            if hr is not None:
                out = hr
        return out

    def parameters(self):
        return [self.w, self.b] if self.use_bias else [self.w]

    def __repr__(self):
        return f"GeGLU(d_model={self.d_model}, d_ff={self.d_ff}, bias={self.use_bias})"


# ==================== FUSED ADAM CUDA KERNEL ====================

_FUSED_ADAM_KERNEL_CODE = r"""
extern "C" __global__
void fused_adam_step(float* param, float* m, float* v,
                     const float* grad, float lr,
                     float beta1, float beta2, float eps,
                     float bias_corr1, float bias_corr2,
                     float weight_decay, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = grad[idx];
        if (weight_decay != 0.0f) {
            param[idx] -= lr * weight_decay * param[idx];
        }
        float mi = beta1 * m[idx] + (1.0f - beta1) * g;
        float vi = beta2 * v[idx] + (1.0f - beta2) * g * g;
        m[idx] = mi;
        v[idx] = vi;
        float m_hat = mi / bias_corr1;
        float v_hat = vi / bias_corr2;
        param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}
"""


class FusedAdam:
    """Adam optimizer with a single fused CUDA kernel per parameter tensor.

    On CUDA this reads grad once and writes param+m+v in a single kernel launch,
    reducing memory bandwidth usage by ~4x vs. the sequential Adam implementation.
    Falls back to standard Adam on CPU.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        self.params       = list(params)
        self.lr           = lr
        self.beta1, self.beta2 = betas
        self.eps          = eps
        self.weight_decay = weight_decay
        self._device      = self.params[0]._device if self.params else 'cuda'
        xp = cp if self._device == 'cuda' else np
        self.m = [xp.zeros_like(p.data) for p in self.params]
        self.v = [xp.zeros_like(p.data) for p in self.params]
        self.t = 0
        self._kernel = None

    def _get_kernel(self):
        if self._kernel is None and self._device == 'cuda':
            self._kernel = _get_cached_kernel('fused_adam_step', _FUSED_ADAM_KERNEL_CODE, 'fused_adam_step')
        return self._kernel

    def step(self):
        self.t += 1
        bc1 = float(1 - self.beta1 ** self.t)
        bc2 = float(1 - self.beta2 ** self.t)
        kernel = self._get_kernel()

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if self._device == 'cuda' and kernel is not None and p.data.dtype == cp.float32:
                n = p.data.size
                bs = 256; gs = (n + bs - 1) // bs
                kernel((gs,), (bs,), (
                    p.data.ravel(), self.m[i].ravel(), self.v[i].ravel(),
                    p.grad.ravel(),
                    cp.float32(self.lr), cp.float32(self.beta1), cp.float32(self.beta2),
                    cp.float32(self.eps), cp.float32(bc1), cp.float32(bc2),
                    cp.float32(self.weight_decay), n
                ))
            else:
                # CPU fallback
                xp = np
                g = p.grad
                if self.weight_decay != 0.0:
                    p.data -= self.lr * self.weight_decay * p.data
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
                m_hat = self.m[i] / bc1
                v_hat = self.v[i] / bc2
                p.data -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        xp = cp if self._device == 'cuda' else np
        for p in self.params:
            if p.grad is not None:
                p.grad = xp.zeros_like(p.data)

    def state_dict(self):
        return {
            'lr': self.lr, 'betas': (self.beta1, self.beta2),
            'eps': self.eps, 'weight_decay': self.weight_decay, 't': self.t,
            'm': [to_cpu(x) for x in self.m],
            'v': [to_cpu(x) for x in self.v],
        }

    def load_state_dict(self, state):
        xp = cp if self._device == 'cuda' else np
        self.lr = state.get('lr', self.lr)
        self.beta1, self.beta2 = state.get('betas', (self.beta1, self.beta2))
        self.eps = state.get('eps', self.eps)
        self.weight_decay = state.get('weight_decay', self.weight_decay)
        self.t = state.get('t', self.t)
        if 'm' in state:
            self.m = [xp.array(x, dtype=xp.float32) for x in state['m']]
        if 'v' in state:
            self.v = [xp.array(x, dtype=xp.float32) for x in state['v']]


# ==================== ACTIVATION MODULES ====================

class ELU(Module):
    """ELU activation module."""
    def __init__(self, alpha=1.0):
        super().__init__(); self.alpha = alpha
    def __call__(self, x):
        out = x.elu(self.alpha)
        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), out)
            if hr is not None: out = hr
        return out
    def parameters(self): return []
    def __repr__(self): return f"ELU(alpha={self.alpha})"


class CELU(Module):
    """CELU activation module."""
    def __init__(self, alpha=1.0):
        super().__init__(); self.alpha = alpha
    def __call__(self, x):
        out = x.celu(self.alpha)
        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), out)
            if hr is not None: out = hr
        return out
    def parameters(self): return []
    def __repr__(self): return f"CELU(alpha={self.alpha})"


class Mish(Module):
    """Mish activation module."""
    def __init__(self): super().__init__()
    def __call__(self, x):
        out = x.mish()
        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), out)
            if hr is not None: out = hr
        return out
    def parameters(self): return []
    def __repr__(self): return "Mish()"


class Hardswish(Module):
    """Hard-Swish activation module."""
    def __init__(self): super().__init__()
    def __call__(self, x):
        out = x.hardswish()
        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), out)
            if hr is not None: out = hr
        return out
    def parameters(self): return []
    def __repr__(self): return "Hardswish()"


class LogSigmoid(Module):
    """LogSigmoid activation module."""
    def __init__(self): super().__init__()
    def __call__(self, x):
        out = x.log_sigmoid()
        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), out)
            if hr is not None: out = hr
        return out
    def parameters(self): return []
    def __repr__(self): return "LogSigmoid()"


class PReLU(Module):
    """Parametric ReLU with a learnable slope per channel (or shared)."""

    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        xp = cp if get_device() == 'cuda' else np
        self.weight = Tensor(xp.full(num_parameters, init, dtype=xp.float32), device=get_device())

    def __call__(self, x):
        xp = cp if self._device == 'cuda' else np
        a = self.weight.data
        # Broadcast slope to match x
        if x.ndim > 1 and a.shape[0] > 1:
            shape = [1] * x.ndim
            shape[1] = a.shape[0]
            a = a.reshape(shape)
        out_data = xp.where(x.data >= 0, x.data, a * x.data)
        out = Tensor(out_data, (x, self.weight), 'PReLU', device=self._device)

        def _backward():
            mask_pos = (x.data >= 0)
            x.grad += xp.where(mask_pos, 1.0, a) * out.grad
            da = (xp.where(mask_pos, 0.0, x.data) * out.grad)
            if self.weight.data.shape[0] == 1:
                self.weight.grad += da.sum(keepdims=False).reshape(1)
            else:
                # sum over all dims except channel dim 1
                axes = tuple(i for i in range(da.ndim) if i != 1)
                self.weight.grad += da.sum(axis=axes)

        out._backward = _backward
        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), out)
            if hr is not None: out = hr
        return out

    def parameters(self):
        return [self.weight]

    def __repr__(self):
        return f"PReLU(num_parameters={self.weight.data.shape[0]})"


# ==================== ADDITIONAL LOSS FUNCTIONS ====================

class KLDivLoss:
    """KL divergence loss: input should be log-probabilities, target probabilities.

    loss = target * (log(target) - input)  (element-wise), then reduced.
    Matches PyTorch's KLDivLoss with reduction='batchmean'.
    """

    def __init__(self, reduction='batchmean'):
        if reduction not in ('mean', 'batchmean', 'sum', 'none'):
            raise ValueError(f"reduction must be mean/batchmean/sum/none, got '{reduction}'")
        self.reduction = reduction

    def __call__(self, log_input: Tensor, target: Tensor) -> Tensor:
        xp = log_input.xp
        # KL(target || input) = target * (log(target) - log_input)
        # We clip target to avoid log(0)
        safe_target = xp.clip(target.data, 1e-8, None)
        loss_data = safe_target * (xp.log(safe_target) - log_input.data)

        if self.reduction == 'none':
            out_data = loss_data
        elif self.reduction == 'sum':
            out_data = loss_data.sum()
        elif self.reduction == 'mean':
            out_data = loss_data.mean()
        else:  # batchmean
            out_data = loss_data.sum() / log_input.data.shape[0]

        out = Tensor(out_data, (log_input,), 'KLDivLoss', device=log_input._device)

        def _backward():
            N = log_input.data.shape[0]
            grad = -safe_target
            if self.reduction == 'mean':
                grad = grad / log_input.data.size
            elif self.reduction == 'batchmean':
                grad = grad / N
            log_input.grad += grad * out.grad

        out._backward = _backward
        return out


class CosineEmbeddingLoss:
    """Loss for learning embeddings via cosine similarity.

    loss = 1 - cos(x1, x2)          if y == 1
    loss = max(0, cos(x1, x2) - m)  if y == -1
    """

    def __init__(self, margin=0.0, reduction='mean'):
        self.margin = margin
        self.reduction = reduction

    def __call__(self, x1: Tensor, x2: Tensor, y: Tensor) -> Tensor:
        xp = x1.xp
        # Normalise
        n1 = xp.sqrt((x1.data ** 2).sum(axis=1, keepdims=True)).clip(1e-8)
        n2 = xp.sqrt((x2.data ** 2).sum(axis=1, keepdims=True)).clip(1e-8)
        cos_sim = (x1.data / n1 * x2.data / n2).sum(axis=1)  # (N,)
        y_data  = y.data.ravel()

        loss_pos = 1 - cos_sim
        loss_neg = xp.maximum(0.0, cos_sim - self.margin)
        loss_data = xp.where(y_data == 1, loss_pos, loss_neg)

        if self.reduction == 'none':
            out_data = loss_data
        elif self.reduction == 'sum':
            out_data = loss_data.sum()
        else:
            out_data = loss_data.mean()

        out = Tensor(out_data, (x1, x2), 'CosineEmbeddingLoss', device=x1._device)

        def _backward():
            g = out.grad
            N = x1.data.shape[0]
            # Numerically stable grad: d(cos)/d(x1) = (x2/n2)/n1 - cos*x1/n1^2
            cos_col = cos_sim[:, None]
            x1n = x1.data / n1
            x2n = x2.data / n2
            d_cos_x1 = x2n / n1 - cos_col * x1n / n1
            d_cos_x2 = x1n / n2 - cos_col * x2n / n2

            sign = xp.where(y_data == 1, -1.0, xp.where(cos_sim > self.margin, 1.0, 0.0))
            scale = g / (N if self.reduction == 'mean' else 1.0)
            x1.grad += (sign[:, None] * d_cos_x1) * scale
            x2.grad += (sign[:, None] * d_cos_x2) * scale

        out._backward = _backward
        return out


class TripletMarginLoss:
    """Triplet loss: max(d(a,p) - d(a,n) + margin, 0).

    Inputs: anchor, positive, negative — all (N, D).
    """

    def __init__(self, margin=1.0, p=2, reduction='mean'):
        self.margin    = margin
        self.p         = p
        self.reduction = reduction

    def __call__(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        xp = anchor.xp
        eps = 1e-6

        def _dist(a, b):
            diff = a.data - b.data
            if self.p == 2:
                return xp.sqrt((diff ** 2).sum(axis=1) + eps)
            return xp.abs(diff).sum(axis=1)

        d_ap = _dist(anchor, positive)
        d_an = _dist(anchor, negative)
        loss_data = xp.maximum(0.0, d_ap - d_an + self.margin)

        if self.reduction == 'none':
            out_data = loss_data
        elif self.reduction == 'sum':
            out_data = loss_data.sum()
        else:
            out_data = loss_data.mean()

        out = Tensor(out_data, (anchor, positive, negative), 'TripletMarginLoss', device=anchor._device)

        def _backward():
            active = (loss_data > 0).astype(anchor.data.dtype)
            N = anchor.data.shape[0]
            scale = out.grad / (N if self.reduction == 'mean' else 1.0)

            def _l2_grad(a, b, d):
                return (a.data - b.data) / (d[:, None] + eps)

            if self.p == 2:
                g_ap = active[:, None] * _l2_grad(anchor, positive, d_ap) * scale
                g_an = active[:, None] * _l2_grad(anchor, negative, d_an) * scale
            else:
                g_ap = active[:, None] * xp.sign(anchor.data - positive.data) * scale
                g_an = active[:, None] * xp.sign(anchor.data - negative.data) * scale

            anchor.grad   += (g_ap - g_an)
            positive.grad += -g_ap
            negative.grad +=  g_an

        out._backward = _backward
        return out


# ==================== SPECTRAL NORMALIZATION ====================

class _SpectralNormHook:
    """Power-iteration spectral norm hook stored on a module's weight."""

    def __init__(self, module, name='w', n_power_iterations=1, eps=1e-12):
        self.name = name
        self.n_iter = n_power_iterations
        self.eps = eps
        W = getattr(module, name).data
        xp = cp if module._device == 'cuda' else np
        h, w = W.shape[0], int(W.size / W.shape[0])
        # random init for u and v
        self.u = xp.random.randn(h).astype(W.dtype)
        self.u /= xp.linalg.norm(self.u) + eps
        self.v = xp.random.randn(w).astype(W.dtype)
        self.v /= xp.linalg.norm(self.v) + eps

    def compute_weight(self, module):
        xp = cp if module._device == 'cuda' else np
        W = getattr(module, self.name).data
        W_mat = W.reshape(W.shape[0], -1)
        u, v = self.u, self.v
        for _ in range(self.n_iter):
            v = W_mat.T @ u
            v = v / (xp.linalg.norm(v) + self.eps)
            u = W_mat @ v
            u = u / (xp.linalg.norm(u) + self.eps)
        self.u, self.v = u, v
        sigma = float(u @ W_mat @ v)
        return W / sigma, sigma


def spectral_norm(module, name='w', n_power_iterations=1, eps=1e-12):
    """Apply spectral normalization to a parameter of the module.

    Wraps the module's forward pass so that on each call the weight is
    divided by its largest singular value (estimated via power iteration).

    Usage:
        layer = spectral_norm(Linear(64, 128))

    Returns the module with spectral norm applied in-place.
    """
    if not hasattr(module, name):
        raise ValueError(f"Module has no attribute '{name}'")

    hook = _SpectralNormHook(module, name, n_power_iterations, eps)

    # Store original weight under _name_orig
    orig_w = getattr(module, name)
    setattr(module, f'_{name}_orig', orig_w)
    module._sn_hook = hook

    # Monkey-patch __call__ to inject normalized weight before each forward
    orig_call = module.__call__

    def _sn_call(*args, **kwargs):
        W_norm, _ = hook.compute_weight(module)
        # Temporarily replace weight data
        real_w = getattr(module, name)
        orig_data = real_w.data
        real_w.data = W_norm
        try:
            result = orig_call(*args, **kwargs)
        finally:
            real_w.data = orig_data
        return result

    module.__call__ = _sn_call
    return module


# ==================== PIXEL SHUFFLE / UNSHUFFLE ====================

class PixelShuffle(Module):
    """Rearrange (N, C*r*r, H, W) -> (N, C, H*r, W*r) for sub-pixel upsampling."""

    def __init__(self, upscale_factor):
        super().__init__()
        self.r = upscale_factor

    def __call__(self, x):
        xp = cp if self._device == 'cuda' else np
        N, Crr, H, W = x.shape
        r = self.r
        C = Crr // (r * r)
        if C * r * r != Crr:
            raise ValueError(f"Input channels ({Crr}) must be divisible by r^2={r*r}")
        # (N, C, r, r, H, W) -> (N, C, H, r, W, r) -> (N, C, H*r, W*r)
        data = x.data.reshape(N, C, r, r, H, W)
        data = data.transpose(0, 1, 4, 2, 5, 3).reshape(N, C, H * r, W * r)
        out = Tensor(data, (x,), 'PixelShuffle', device=self._device)

        def _backward():
            grad = out.grad.reshape(N, C, H, r, W, r)
            grad = grad.transpose(0, 1, 3, 5, 2, 4).reshape(N, Crr, H, W)
            x.grad += grad

        out._backward = _backward
        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), out)
            if hr is not None: out = hr
        return out

    def parameters(self): return []
    def __repr__(self): return f"PixelShuffle(upscale_factor={self.r})"


class PixelUnshuffle(Module):
    """Reverse of PixelShuffle: (N, C, H*r, W*r) -> (N, C*r*r, H, W)."""

    def __init__(self, downscale_factor):
        super().__init__()
        self.r = downscale_factor

    def __call__(self, x):
        xp = cp if self._device == 'cuda' else np
        N, C, Hr, Wr = x.shape
        r = self.r
        H, W = Hr // r, Wr // r
        if H * r != Hr or W * r != Wr:
            raise ValueError(f"Spatial dims ({Hr},{Wr}) must be divisible by r={r}")
        # (N, C, H, r, W, r) -> (N, C, r, r, H, W) -> (N, C*r*r, H, W)
        data = x.data.reshape(N, C, H, r, W, r)
        data = data.transpose(0, 1, 3, 5, 2, 4).reshape(N, C * r * r, H, W)
        out = Tensor(data, (x,), 'PixelUnshuffle', device=self._device)

        def _backward():
            grad = out.grad.reshape(N, C, r, r, H, W)
            grad = grad.transpose(0, 1, 4, 2, 5, 3).reshape(N, C, Hr, Wr)
            x.grad += grad

        out._backward = _backward
        for hook in self._forward_hooks.values():
            hr = hook(self, (x,), out)
            if hr is not None: out = hr
        return out

    def parameters(self): return []
    def __repr__(self): return f"PixelUnshuffle(downscale_factor={self.r})"


# ==================== INSTALL HOOK DISPATCH ON EXISTING MODULES ====================

_install_hook_dispatch(
    Linear, Conv2D, DepthwiseConv2D, SeparableConv2D,
    MaxPool2D, AvgPool2D, AdaptiveAvgPool2D,
    BatchNorm2D, FusedBatchNormReLU, LayerNorm, GroupNorm,
    Dropout, Embedding, ConvTranspose2D,
)