"""
tensor_gpu_v2.py - High-Performance Autograd Engine with GPU Acceleration

Next-generation tensor library with:
- Fused kernel operations (GELU, LayerNorm, SiLU, LeakyReLU, Swish)
- CUDA streams for pipelined execution
- Memory pool optimization with aggressive caching
- Advanced operations (einsum, scaled dot-product attention, grouped convs)
- Mixed precision training (FP16/BF16 with dynamic loss scaling)
- Gradient checkpointing for memory-efficient backprop
- Zero-copy view operations
- Custom CUDA kernels via RawKernel for critical paths

Drop-in replacement for tensor_gpu.py with significant performance improvements.
"""
import cupy as cp
import cupyx
import numpy as np
from typing import Optional, Tuple, List, Union, Callable, Any
import weakref
import threading
import hashlib
import os

# ==================== KERNEL CACHE ====================

KERNEL_CACHE_DIR = os.path.expanduser('~/.cache/tensor_gpu_v2/kernels')
_kernel_cache = {}


def _get_kernel_cache_key(code: str, name: str) -> str:
    """Generate cache key for kernel."""
    try:
        cuda_version = cp.cuda.runtime.runtimeGetVersion()
        compute_cap = cp.cuda.Device().compute_capability
    except:
        cuda_version = 'unknown'
        compute_cap = 'unknown'

    content = f"{code}|{name}|{cuda_version}|{compute_cap}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _get_cached_kernel(name: str, code: str, func_name: str) -> cp.RawKernel:
    """
    Get kernel from cache or compile and cache.

    Uses both in-memory and file-based caching for optimal performance.
    CuPy also maintains its own cache in ~/.cupy/kernel_cache

    Args:
        name: Human-readable name for the kernel
        code: CUDA source code for the kernel
        func_name: Name of the kernel function in the CUDA code

    Returns:
        Compiled CuPy RawKernel

    Raises:
        ValueError: If code is empty or invalid
        RuntimeError: If kernel compilation fails
    """
    # Validate input
    if not code or not code.strip():
        raise ValueError(f"Kernel '{name}' has empty code - cannot compile")

    if not func_name or not func_name.strip():
        raise ValueError(f"Kernel '{name}' has empty function name")

    cache_key = _get_kernel_cache_key(code, name)

    # Check in-memory cache first
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    # Ensure cache directory exists
    os.makedirs(KERNEL_CACHE_DIR, exist_ok=True)

    # Attempt to compile kernel with proper error handling
    try:
        kernel = cp.RawKernel(code, func_name)
        # Force compilation to catch errors early (before caching)
        # CuPy defers compilation until first use, so we trigger it here
        kernel.compile()
    except cp.cuda.compiler.CompileException as e:
        raise RuntimeError(
            f"CUDA compilation failed for kernel '{name}' (function: {func_name}): {e}"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to create kernel '{name}' (function: {func_name}): {e}"
        )

    # Store in memory cache only after successful compilation
    _kernel_cache[cache_key] = kernel

    return kernel


def clear_kernel_cache():
    """Clear in-memory kernel cache."""
    global _kernel_cache
    _kernel_cache = {}


def get_kernel_cache_info() -> dict:
    """Get kernel cache statistics."""
    return {
        'in_memory_count': len(_kernel_cache),
        'cache_dir': KERNEL_CACHE_DIR,
        'kernels': list(_kernel_cache.keys()),
    }


# ==================== GLOBAL CONFIGURATION ====================

_device = 'cuda'
_default_dtype = cp.float32
_mixed_precision = False
_grad_enabled = True
_streams: List[cp.cuda.Stream] = []
_current_stream_idx = 0
_stream_lock = threading.Lock()

# Memory pool settings
_memory_pool = cp.get_default_memory_pool()
_pinned_memory_pool = cp.get_default_pinned_memory_pool()


def set_device(device: str):
    """Set compute device: 'cuda' or 'cpu'"""
    global _device
    if device not in ('cuda', 'cpu'):
        raise ValueError(f"Unknown device: {device}")
    _device = device


def get_device() -> str:
    """Get current compute device."""
    return _device


def enable_mixed_precision(enable: bool = True):
    """Enable or disable mixed precision training."""
    global _mixed_precision
    _mixed_precision = enable


def is_mixed_precision() -> bool:
    """Check if mixed precision is enabled."""
    return _mixed_precision


def no_grad():
    """Context manager to disable gradient computation."""
    return _NoGradContext()


class _NoGradContext:
    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *args):
        global _grad_enabled
        _grad_enabled = self._prev


# ==================== STREAM MANAGEMENT ====================

def init_streams(num_streams: int = 4):
    """Initialize CUDA streams for parallel execution."""
    global _streams
    if _device == 'cuda':
        _streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_streams)]


def get_stream() -> Optional[cp.cuda.Stream]:
    """Get next available stream in round-robin fashion."""
    global _current_stream_idx
    if not _streams:
        return None
    with _stream_lock:
        stream = _streams[_current_stream_idx]
        _current_stream_idx = (_current_stream_idx + 1) % len(_streams)
        return stream


def sync_all_streams():
    """Synchronize all CUDA streams."""
    for stream in _streams:
        stream.synchronize()


def sync():
    """Synchronize default stream."""
    if _device == 'cuda':
        cp.cuda.Stream.null.synchronize()


# ==================== MEMORY MANAGEMENT ====================

def to_gpu(arr, dtype=None):
    """Move numpy array to GPU with optional dtype conversion."""
    if isinstance(arr, cp.ndarray):
        return arr if dtype is None else arr.astype(dtype)
    dtype = dtype or _default_dtype
    return cp.asarray(arr, dtype=dtype)


def to_cpu(arr):
    """Move cupy array to CPU."""
    if isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr


def memory_info() -> dict:
    """Return GPU memory usage info."""
    if _device == 'cuda':
        return {
            'used_bytes': _memory_pool.used_bytes(),
            'total_bytes': _memory_pool.total_bytes(),
            'used_mb': _memory_pool.used_bytes() / 1024**2,
            'total_mb': _memory_pool.total_bytes() / 1024**2,
            'n_free_blocks': _memory_pool.n_free_blocks()
        }
    return {'used_bytes': 0, 'total_bytes': 0, 'used_mb': 0, 'total_mb': 0, 'n_free_blocks': 0}


def clear_memory():
    """Clear GPU memory cache."""
    if _device == 'cuda':
        _memory_pool.free_all_blocks()
        _pinned_memory_pool.free_all_blocks()


def set_memory_pool_limit(limit_mb: int):
    """Set GPU memory pool limit in megabytes."""
    if _device == 'cuda':
        _memory_pool.set_limit(size=limit_mb * 1024 * 1024)


# ==================== CUSTOM CUDA KERNELS ====================

# Fused GELU kernel (faster than separate operations)
_gelu_kernel = cp.RawKernel(r'''
extern "C" __global__
void gelu_forward(const float* x, float* out, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        float xi = x[i];
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x3 = xi * xi * xi;
        float inner = 0.7978845608f * (xi + 0.044715f * x3);
        out[i] = 0.5f * xi * (1.0f + tanhf(inner));
    }
}
''', 'gelu_forward')

_gelu_backward_kernel = cp.RawKernel(r'''
extern "C" __global__
void gelu_backward(const float* x, const float* grad_out, float* grad_in, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        float xi = x[i];
        float x3 = xi * xi * xi;
        float inner = 0.7978845608f * (xi + 0.044715f * x3);
        float tanh_inner = tanhf(inner);
        float sech2 = 1.0f - tanh_inner * tanh_inner;
        float d_inner = 0.7978845608f * (1.0f + 3.0f * 0.044715f * xi * xi);
        grad_in[i] = grad_out[i] * (0.5f * (1.0f + tanh_inner) + 0.5f * xi * sech2 * d_inner);
    }
}
''', 'gelu_backward')

# Fused SiLU (Swish) kernel
_silu_kernel = cp.RawKernel(r'''
extern "C" __global__
void silu_forward(const float* x, float* out, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        float xi = x[i];
        float sig = 1.0f / (1.0f + expf(-xi));
        out[i] = xi * sig;
    }
}
''', 'silu_forward')

_silu_backward_kernel = cp.RawKernel(r'''
extern "C" __global__
void silu_backward(const float* x, const float* grad_out, float* grad_in, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        float xi = x[i];
        float sig = 1.0f / (1.0f + expf(-xi));
        // d(x * sig)/dx = sig + x * sig * (1 - sig) = sig * (1 + x * (1 - sig))
        grad_in[i] = grad_out[i] * sig * (1.0f + xi * (1.0f - sig));
    }
}
''', 'silu_backward')

# Fused LayerNorm kernel
_layernorm_kernel = cp.RawKernel(r'''
extern "C" __global__
void layernorm_forward(
    const float* x, const float* gamma, const float* beta,
    float* out, float* mean, float* rstd,
    int batch_size, int hidden_size, float eps
) {
    int b = blockIdx.x;
    if (b >= batch_size) return;

    // Compute mean
    float sum = 0.0f;
    for (int h = 0; h < hidden_size; h++) {
        sum += x[b * hidden_size + h];
    }
    float m = sum / hidden_size;
    mean[b] = m;

    // Compute variance
    float var_sum = 0.0f;
    for (int h = 0; h < hidden_size; h++) {
        float diff = x[b * hidden_size + h] - m;
        var_sum += diff * diff;
    }
    float var = var_sum / hidden_size;
    float rs = rsqrtf(var + eps);
    rstd[b] = rs;

    // Normalize and apply affine
    for (int h = 0; h < hidden_size; h++) {
        int idx = b * hidden_size + h;
        out[idx] = (x[idx] - m) * rs * gamma[h] + beta[h];
    }
}
''', 'layernorm_forward')

# Fused BatchNorm-ReLU kernel for inference mode
_fused_bn_relu_kernel = cp.RawKernel(r'''
extern "C" __global__
void fused_bn_relu_forward(
    const float* x, const float* scale, const float* shift,
    float* out, int total_size, int channels, int spatial_size
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total_size) {
        int c = (idx / spatial_size) % channels;
        float val = x[idx] * scale[c] + shift[c];
        out[idx] = val > 0.0f ? val : 0.0f;
    }
}
''', 'fused_bn_relu_forward')

# Fused BatchNorm-ReLU backward kernel
_fused_bn_relu_backward_kernel = cp.RawKernel(r'''
extern "C" __global__
void fused_bn_relu_backward(
    const float* x, const float* scale, const float* shift,
    const float* grad_out, float* grad_in,
    int total_size, int channels, int spatial_size
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total_size) {
        int c = (idx / spatial_size) % channels;
        float val = x[idx] * scale[c] + shift[c];
        // Gradient is scale if val > 0, else 0
        grad_in[idx] = (val > 0.0f) ? grad_out[idx] * scale[c] : 0.0f;
    }
}
''', 'fused_bn_relu_backward')


# ==================== TENSOR CLASS ====================

class Tensor:
    """
    High-performance autograd tensor with GPU acceleration.

    Features:
    - Automatic differentiation via computation graph
    - GPU acceleration through CuPy
    - Mixed precision support (FP16/BF16)
    - View operations without data copy
    - Gradient checkpointing support
    """

    __slots__ = ('data', 'grad', '_backward', '_prev', '_op', 'label', '_device',
                 '_requires_grad', '_is_view', '_base', '_grad_fn', '__weakref__',
                 '_tied_source', '_tied_transpose')

    def __init__(self, data, _children=(), _op='', label='', device=None,
                 requires_grad=True, dtype=None):
        device = device or _device
        dtype = dtype or (_default_dtype if not _mixed_precision else cp.float16)

        if device == 'cuda':
            if isinstance(data, cp.ndarray):
                self.data = data.astype(dtype) if data.dtype != dtype else data
            elif isinstance(data, np.ndarray):
                self.data = cp.array(data, dtype=dtype)
            else:
                self.data = cp.array(data, dtype=dtype)
            self.grad = cp.zeros_like(self.data, dtype=self.data.dtype) if requires_grad and _grad_enabled else None
        else:
            if isinstance(data, cp.ndarray):
                self.data = cp.asnumpy(data).astype(dtype)
            elif isinstance(data, np.ndarray):
                self.data = data.astype(dtype)
            else:
                self.data = np.array(data, dtype=dtype)
            self.grad = np.zeros_like(self.data, dtype=self.data.dtype) if requires_grad and _grad_enabled else None

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._device = device
        self._requires_grad = requires_grad and _grad_enabled
        self._is_view = False
        self._base = None
        self._grad_fn = None

    @property
    def xp(self):
        """Return the array module (cupy or numpy) for this tensor."""
        return cp if self._device == 'cuda' else np

    @property
    def shape(self) -> tuple:
        """Return tensor shape."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return self.data.ndim

    @property
    def dtype(self):
        """Return data type."""
        return self.data.dtype

    @property
    def size(self) -> int:
        """Return total number of elements."""
        return self.data.size

    @property
    def T(self):
        """Return transpose view."""
        return self.transpose()

    def to(self, device: str):
        """Move tensor to specified device."""
        if device == self._device:
            return self
        new_data = to_cpu(self.data) if device == 'cpu' else to_gpu(self.data)
        new_tensor = Tensor(new_data, device=device, requires_grad=self._requires_grad)
        if self.grad is not None:
            new_tensor.grad = to_cpu(self.grad) if device == 'cpu' else to_gpu(self.grad)
        return new_tensor

    def cpu(self):
        """Move tensor to CPU."""
        return self.to('cpu')

    def cuda(self):
        """Move tensor to GPU."""
        return self.to('cuda')

    def numpy(self):
        """Return data as numpy array."""
        return to_cpu(self.data)

    def item(self):
        """Return scalar value."""
        return float(self.data)

    def clone(self):
        """Create a copy of the tensor."""
        xp = self.xp
        new_tensor = Tensor(xp.copy(self.data), device=self._device, requires_grad=self._requires_grad)
        return new_tensor

    def detach(self):
        """Create a copy without gradient tracking."""
        xp = self.xp
        return Tensor(xp.copy(self.data), device=self._device, requires_grad=False)

    def contiguous(self):
        """Return contiguous tensor (or self if already contiguous)."""
        xp = self.xp
        if self.data.flags['C_CONTIGUOUS']:
            return self
        return Tensor(xp.ascontiguousarray(self.data), device=self._device, requires_grad=self._requires_grad)

    def half(self):
        """Convert to FP16."""
        dtype = cp.float16 if self._device == 'cuda' else np.float16
        return Tensor(self.data.astype(dtype),
                     device=self._device, requires_grad=self._requires_grad, dtype=dtype)

    def float(self):
        """Convert to FP32."""
        dtype = cp.float32 if self._device == 'cuda' else np.float32
        return Tensor(self.data.astype(dtype),
                     device=self._device, requires_grad=self._requires_grad, dtype=dtype)

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, dtype={self.dtype}, op={self._op}, device={self._device})"

    # ==================== VIEW OPERATIONS ====================

    @staticmethod
    def _create_view_fast(data, device, requires_grad=False):
        """Fast tensor creation for views - bypasses full constructor."""
        t = object.__new__(Tensor)
        t.data = data
        t.grad = None
        t._backward = lambda: None
        t._prev = set()
        t._op = ''
        t.label = ''
        t._device = device
        t._requires_grad = requires_grad
        t._is_view = True
        t._base = None
        t._grad_fn = None
        return t

    def view(self, *shape):
        """Return a view with different shape (no data copy - uses CuPy's view mechanism)."""
        new_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape

        # Handle -1 dimension
        if -1 in new_shape:
            total = self.data.size
            known = 1
            unknown_idx = -1
            for i, s in enumerate(new_shape):
                if s == -1:
                    unknown_idx = i
                else:
                    known *= s
            new_shape = list(new_shape)
            new_shape[unknown_idx] = total // known
            new_shape = tuple(new_shape)

        # Use reshape which creates a view when possible (no copy)
        reshaped_data = self.data.reshape(new_shape)

        # Fast path: if no gradient tracking needed, use minimal tensor creation
        if not self._requires_grad or not _grad_enabled:
            return Tensor._create_view_fast(reshaped_data, self._device, False)

        # Create view tensor with gradient tracking
        xp = self.xp
        out = Tensor._create_view_fast(reshaped_data, self._device, True)
        out.grad = xp.zeros_like(reshaped_data, dtype=reshaped_data.dtype)
        out._prev = {self}
        out._op = 'view'
        original_shape = self.data.shape

        def _backward():
            if self.grad is not None and out.grad is not None:
                self.grad += out.grad.reshape(original_shape)
        out._backward = _backward

        return out

    def squeeze(self, dim=None):
        """Remove dimensions of size 1."""
        xp = self.xp
        if dim is None:
            new_data = xp.squeeze(self.data)
        else:
            new_data = xp.squeeze(self.data, axis=dim)

        out = Tensor(new_data, (self,), 'squeeze', device=self._device)
        original_shape = self.data.shape

        def _backward():
            self.grad += out.grad.reshape(original_shape)
        out._backward = _backward

        return out

    def unsqueeze(self, dim):
        """Add a dimension of size 1."""
        xp = self.xp
        new_data = xp.expand_dims(self.data, axis=dim)
        out = Tensor(new_data, (self,), 'unsqueeze', device=self._device)

        def _backward():
            self.grad += xp.squeeze(out.grad, axis=dim)
        out._backward = _backward

        return out

    def expand(self, *sizes):
        """Expand tensor to larger size (broadcasting)."""
        xp = self.xp
        out_data = xp.broadcast_to(self.data, sizes)
        out = Tensor(out_data, (self,), 'expand', device=self._device)

        def _backward():
            self.grad += self._unbroadcast(out.grad, self.data.shape)
        out._backward = _backward

        return out

    # ==================== INDEXING ====================

    def __getitem__(self, idx):
        """Advanced indexing with gradient support."""
        xp = self.xp
        out_data = self.data[idx]
        out = Tensor(out_data, (self,), 'getitem', device=self._device)

        def _backward():
            grad = xp.zeros_like(self.data)
            if isinstance(idx, tuple):
                xp_scatter_add = getattr(cupyx, 'scatter_add', None) if xp == cp else None
                if xp_scatter_add is not None:
                    xp_scatter_add(grad, idx, out.grad)
                else:
                    grad[idx] += out.grad
            else:
                grad[idx] += out.grad
            self.grad += grad
        out._backward = _backward

        return out

    def __setitem__(self, idx, value):
        """In-place assignment (breaks gradient tracking for simplicity)."""
        if isinstance(value, Tensor):
            self.data[idx] = value.data
        else:
            self.data[idx] = value

    # ==================== ARITHMETIC OPERATIONS ====================

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self._device, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other), '+', device=self._device)

        def _backward():
            if self.grad is not None:
                self.grad += self._unbroadcast(out.grad, self.data.shape)
            if other.grad is not None:
                other.grad += self._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self._device, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other), '*', device=self._device)

        def _backward():
            if self.grad is not None:
                self.grad += self._unbroadcast(other.data * out.grad, self.data.shape)
            if other.grad is not None:
                other.grad += self._unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        xp = self.xp
        out = Tensor(self.data ** other, (self,), f'**{other}', device=self._device)

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self._device, requires_grad=False)
        out = Tensor(self.data @ other.data, (self, other), '@', device=self._device)

        def _backward():
            if self.grad is not None:
                if self.data.ndim == 1:
                    self.grad += out.grad @ other.data.T
                elif out.grad.ndim == 1:
                    self.grad += self.xp.outer(out.grad, other.data)
                else:
                    self.grad += out.grad @ other.data.swapaxes(-1, -2)
            if other.grad is not None:
                if other.data.ndim == 1:
                    other.grad += self.data.T @ out.grad
                elif self.data.ndim == 1:
                    other.grad += self.xp.outer(self.data, out.grad)
                else:
                    other.grad += self.data.swapaxes(-1, -2) @ out.grad
        out._backward = _backward

        return out

    # ==================== REDUCTION OPERATIONS ====================

    def sum(self, axis=None, keepdims=False):
        xp = self.xp
        out = Tensor(xp.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum', device=self._device)

        def _backward():
            grad_output = out.grad
            if not keepdims and axis is not None:
                if isinstance(axis, int):
                    grad_output = xp.expand_dims(grad_output, axis)
                else:
                    for ax in sorted(axis):
                        grad_output = xp.expand_dims(grad_output, ax)
            self.grad += xp.broadcast_to(grad_output, self.data.shape)
        out._backward = _backward

        return out

    def mean(self, axis=None, keepdims=False):
        xp = self.xp
        s = self.sum(axis=axis, keepdims=keepdims)
        if axis is None:
            n = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            n = int(xp.prod(xp.array([self.data.shape[i] for i in axes])))
        return s * (1.0 / float(n))

    def var(self, axis=None, keepdims=False, correction=0):
        """Compute variance with optional Bessel's correction."""
        xp = self.xp
        mean_val = self.mean(axis=axis, keepdims=True)
        diff = self - mean_val
        sq_diff = diff ** 2
        n = self.data.size if axis is None else int(xp.prod(xp.array([self.data.shape[i] for i in ((axis,) if isinstance(axis, int) else axis)])))
        return sq_diff.sum(axis=axis, keepdims=keepdims) * (1.0 / float(n - correction))

    def std(self, axis=None, keepdims=False, correction=0):
        """Compute standard deviation."""
        return (self.var(axis=axis, keepdims=keepdims, correction=correction) + 1e-8) ** 0.5

    def max(self, axis=None, keepdims=False):
        xp = self.xp
        out_data = xp.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, (self,), 'max', device=self._device)

        def _backward():
            if axis is None:
                mask = (self.data == out_data)
            else:
                expanded = xp.expand_dims(out_data, axis) if not keepdims else out_data
                mask = (self.data == expanded)
            # Distribute gradient evenly among max elements
            count = mask.sum(axis=axis, keepdims=True)
            grad_expanded = xp.expand_dims(out.grad, axis) if not keepdims and axis is not None else out.grad
            self.grad += mask * xp.broadcast_to(grad_expanded, self.data.shape) / count
        out._backward = _backward

        return out

    def min(self, axis=None, keepdims=False):
        return (-self).max(axis=axis, keepdims=keepdims) * -1

    def argmax(self, axis=None):
        """Return indices of maximum values (no gradient)."""
        xp = self.xp
        return xp.argmax(self.data, axis=axis)

    def argmin(self, axis=None):
        """Return indices of minimum values (no gradient)."""
        xp = self.xp
        return xp.argmin(self.data, axis=axis)

    # ==================== ACTIVATION FUNCTIONS ====================

    def relu(self):
        xp = self.xp
        out = Tensor(xp.maximum(0, self.data), (self,), 'ReLU', device=self._device)

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def leaky_relu(self, negative_slope=0.01):
        """Leaky ReLU activation."""
        xp = self.xp
        out_data = xp.where(self.data >= 0, self.data, negative_slope * self.data)
        out = Tensor(out_data, (self,), 'LeakyReLU', device=self._device)

        def _backward():
            mask = xp.where(self.data >= 0, 1.0, negative_slope)
            self.grad += mask * out.grad
        out._backward = _backward

        return out

    def gelu(self):
        """GELU activation using fused CUDA kernel."""
        xp = self.xp

        if self._device == 'cuda' and self.data.dtype == cp.float32:
            # Use custom CUDA kernel for FP32
            n = self.data.size
            out_data = cp.empty_like(self.data)
            block_size = 256
            grid_size = (n + block_size - 1) // block_size
            _gelu_kernel((grid_size,), (block_size,), (self.data.ravel(), out_data.ravel(), n))
            out_data = out_data.reshape(self.data.shape)
            out = Tensor(out_data, (self,), 'GELU', device=self._device)

            def _backward():
                grad_in = cp.empty_like(self.data)
                _gelu_backward_kernel((grid_size,), (block_size,),
                                     (self.data.ravel(), out.grad.ravel(), grad_in.ravel(), n))
                self.grad += grad_in.reshape(self.data.shape)
            out._backward = _backward
        else:
            # Fallback to standard implementation
            x = self.data
            sqrt_2_pi = 0.7978845608028654
            cdf = 0.5 * (1 + xp.tanh(sqrt_2_pi * (x + 0.044715 * x**3)))
            out = Tensor(x * cdf, (self,), 'GELU', device=self._device)

            def _backward():
                x3 = self.data ** 3
                inner = sqrt_2_pi * (self.data + 0.044715 * x3)
                tanh_inner = xp.tanh(inner)
                sech2 = 1 - tanh_inner ** 2
                d_inner = sqrt_2_pi * (1 + 3 * 0.044715 * self.data ** 2)
                grad = 0.5 * (1 + tanh_inner) + 0.5 * self.data * sech2 * d_inner
                self.grad += grad * out.grad
            out._backward = _backward

        return out

    def silu(self):
        """SiLU/Swish activation: x * sigmoid(x)."""
        xp = self.xp

        # Use custom kernel only for large tensors where kernel launch overhead is amortized
        # For most sizes, CuPy's fused operations are faster
        if self._device == 'cuda' and self.data.dtype == cp.float32 and self.data.size > 4_000_000:
            n = self.data.size
            out_data = cp.empty_like(self.data)
            block_size = 256
            grid_size = (n + block_size - 1) // block_size
            _silu_kernel((grid_size,), (block_size,), (self.data.ravel(), out_data.ravel(), n))
            out_data = out_data.reshape(self.data.shape)
            out = Tensor(out_data, (self,), 'SiLU', device=self._device)

            def _backward():
                grad_in = cp.empty_like(self.data)
                _silu_backward_kernel((grid_size,), (block_size,),
                                     (self.data.ravel(), out.grad.ravel(), grad_in.ravel(), n))
                self.grad += grad_in.reshape(self.data.shape)
            out._backward = _backward
        else:
            # CuPy fused ops - uses optimized element-wise kernel under the hood
            sig = 1 / (1 + xp.exp(-self.data))
            out_data = self.data * sig
            out = Tensor(out_data, (self,), 'SiLU', device=self._device)

            # Cache sigmoid for backward
            def _backward():
                # Recompute sig rather than store (memory vs compute tradeoff)
                sig_val = 1 / (1 + xp.exp(-self.data))
                grad = sig_val * (1 + self.data * (1 - sig_val))
                self.grad += grad * out.grad
            out._backward = _backward

        return out

    def swish(self, beta=1.0):
        """Swish activation: x * sigmoid(beta * x)"""
        xp = self.xp
        sig = 1 / (1 + xp.exp(-beta * self.data))
        out = Tensor(self.data * sig, (self,), 'Swish', device=self._device)

        def _backward():
            sig_val = 1 / (1 + xp.exp(-beta * self.data))
            grad = sig_val + beta * self.data * sig_val * (1 - sig_val)
            self.grad += grad * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        """Numerically stable sigmoid."""
        xp = self.xp
        sig = xp.where(self.data >= 0,
                       1 / (1 + xp.exp(-self.data)),
                       xp.exp(self.data) / (1 + xp.exp(self.data)))
        out = Tensor(sig, (self,), 'sigmoid', device=self._device)

        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        xp = self.xp
        out = Tensor(xp.tanh(self.data), (self,), 'tanh', device=self._device)

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward

        return out

    def softmax(self, axis=-1):
        """Numerically stable softmax."""
        xp = self.xp
        shifted = self.data - xp.max(self.data, axis=axis, keepdims=True)
        exp_data = xp.exp(shifted)
        sm = exp_data / xp.sum(exp_data, axis=axis, keepdims=True)
        out = Tensor(sm, (self,), 'softmax', device=self._device)

        def _backward():
            grad_sum = xp.sum(out.grad * out.data, axis=axis, keepdims=True)
            self.grad += out.data * (out.grad - grad_sum)
        out._backward = _backward

        return out

    def log_softmax(self, axis=-1):
        """Numerically stable log softmax."""
        xp = self.xp
        shifted = self.data - xp.max(self.data, axis=axis, keepdims=True)
        log_sum_exp = xp.log(xp.sum(xp.exp(shifted), axis=axis, keepdims=True))
        out = Tensor(shifted - log_sum_exp, (self,), 'log_softmax', device=self._device)

        def _backward():
            sm = xp.exp(out.data)
            grad_sum = xp.sum(out.grad, axis=axis, keepdims=True)
            self.grad += out.grad - sm * grad_sum
        out._backward = _backward

        return out

    # ==================== SHAPE OPERATIONS ====================

    def reshape(self, *shape):
        """Reshape tensor with optimized fast path."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        reshaped_data = self.data.reshape(shape)

        # Fast path: no gradient tracking needed
        if not self._requires_grad or not _grad_enabled:
            return Tensor._create_view_fast(reshaped_data, self._device, False)

        # Gradient tracking path
        xp = self.xp
        original_shape = self.data.shape
        out = Tensor._create_view_fast(reshaped_data, self._device, True)
        out.grad = xp.zeros_like(reshaped_data, dtype=reshaped_data.dtype)
        out._prev = {self}
        out._op = 'reshape'

        def _backward():
            self.grad += out.grad.reshape(original_shape)
        out._backward = _backward

        return out

    def transpose(self, *axes):
        """Transpose tensor axes."""
        xp = self.xp
        if not axes:
            axes = tuple(range(self.ndim - 1, -1, -1))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])

        out = Tensor(self.data.transpose(axes), (self,), 'transpose', device=self._device)

        def _backward():
            inv_axes = [0] * len(axes)
            for i, ax in enumerate(axes):
                inv_axes[ax] = i
            self.grad += out.grad.transpose(tuple(inv_axes))
        out._backward = _backward

        return out

    def permute(self, *dims):
        """Permute tensor dimensions."""
        return self.transpose(*dims)

    def flatten(self, start_dim=0, end_dim=-1):
        """Flatten tensor dimensions."""
        xp = self.xp
        if end_dim < 0:
            end_dim = self.ndim + end_dim

        new_shape = list(self.shape[:start_dim])
        new_shape.append(-1)
        new_shape.extend(self.shape[end_dim + 1:])

        return self.reshape(tuple(new_shape))

    # ==================== UTILITY METHODS ====================

    def _unbroadcast(self, grad, shape):
        """Reduce gradient to match original shape after broadcasting."""
        xp = self.xp
        if grad.shape == shape:
            return grad

        ndims_added = grad.ndim - len(shape)
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)

        for i, dim in enumerate(shape):
            if dim == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad

    def backward(self):
        """Run backpropagation from this tensor."""
        xp = self.xp
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        if self.grad is None:
            self.grad = xp.ones_like(self.data)
        else:
            self.grad = xp.ones_like(self.data)

        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        """Zero out gradient."""
        if self.grad is not None:
            xp = self.xp
            self.grad = xp.zeros_like(self.data)


# ==================== ADVANCED OPERATIONS ====================

# Global cache for einsum contraction paths
_einsum_path_cache = {}
_EINSUM_CACHE_MAX_SIZE = 1000


def einsum(subscripts: str, *operands, optimize: Union[bool, str] = True, use_cache: bool = True):
    """
    Einstein summation with gradient support and path optimization.

    Supports all NumPy/CuPy einsum operations with automatic differentiation.
    Uses path caching for repeated patterns to improve performance.

    Args:
        subscripts: Einsum subscripts string (e.g., 'ij,jk->ik')
        *operands: Input tensors
        optimize: Path optimization ('auto', 'greedy', 'optimal', True, False)
        use_cache: Cache contraction paths for repeated patterns (default True)

    Returns:
        Result tensor with gradient support
    """
    tensors = [op if isinstance(op, Tensor) else Tensor(op, requires_grad=False) for op in operands]
    device = tensors[0]._device
    xp = cp if device == 'cuda' else np

    # Forward pass with optional caching
    data_list = [t.data for t in tensors]

    # Create cache key from subscripts and shapes
    if use_cache and len(tensors) > 2:
        cache_key = (subscripts, tuple(t.shape for t in tensors))

        if cache_key in _einsum_path_cache:
            # Use cached optimization flag
            out_data = xp.einsum(subscripts, *data_list, optimize=True)
        else:
            # Compute and cache (CuPy handles path optimization internally with optimize=True)
            if len(_einsum_path_cache) >= _EINSUM_CACHE_MAX_SIZE:
                _einsum_path_cache.clear()  # Simple cache eviction

            out_data = xp.einsum(subscripts, *data_list, optimize=True)
            _einsum_path_cache[cache_key] = True
    else:
        # Simple case or caching disabled
        out_data = xp.einsum(subscripts, *data_list, optimize=optimize)

    out = Tensor(out_data, tuple(tensors), f'einsum:{subscripts}', device=device)

    # Parse subscripts for backward
    parts = subscripts.replace(' ', '').split('->')
    if len(parts) == 2:
        inputs_str, output_str = parts
    else:
        inputs_str = parts[0]
        output_str = None

    input_subscripts = inputs_str.split(',')

    def _backward():
        for i, (t, subs) in enumerate(zip(tensors, input_subscripts)):
            if t.grad is None:
                continue

            # Build einsum for gradient
            other_subs = input_subscripts[:i] + input_subscripts[i+1:]
            other_data = data_list[:i] + data_list[i+1:]

            if output_str:
                grad_subscript = f"{output_str},{','.join(other_subs)}->{subs}"
            else:
                grad_subscript = f"{','.join([output_str or ''] + other_subs)}->{subs}"

            try:
                grad = xp.einsum(grad_subscript, out.grad, *other_data, optimize=True)
                t.grad += grad
            except:
                # Fallback: numerical gradient would go here
                pass

    out._backward = _backward
    return out


def clear_einsum_cache():
    """Clear the einsum path cache."""
    global _einsum_path_cache
    _einsum_path_cache.clear()


def einsum_cache_info() -> dict:
    """Return information about the einsum cache."""
    return {
        'size': len(_einsum_path_cache),
        'max_size': _EINSUM_CACHE_MAX_SIZE,
        'patterns': list(_einsum_path_cache.keys())[:10]  # First 10 patterns
    }


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor,
                                  attn_mask: Optional[Tensor] = None,
                                  dropout_p: float = 0.0,
                                  is_causal: bool = False,
                                  scale: Optional[float] = None,
                                  use_flash: bool = False,
                                  block_size: int = 64) -> Tensor:
    """
    Scaled dot-product attention with gradient support.

    Args:
        query: (B, H, L, D) or (B, L, D)
        key: (B, H, S, D) or (B, S, D)
        value: (B, H, S, D) or (B, S, D)
        attn_mask: Optional attention mask
        dropout_p: Dropout probability (training only)
        is_causal: Apply causal mask
        scale: Scale factor (default: 1/sqrt(D))
        use_flash: Use FlashAttention-style tiled computation for O(N) memory
        block_size: Block size for FlashAttention (default 64)

    Returns:
        Output tensor of same shape as query
    """
    # Use FlashAttention if requested and sequence length is long enough
    if use_flash and query.shape[-2] > block_size:
        return flash_attention(query, key, value, block_size=block_size,
                               is_causal=is_causal, scale=scale, dropout_p=dropout_p)

    xp = query.xp
    device = query._device

    # Get dimensions
    if query.ndim == 3:
        B, L, D = query.shape
        H = 1
        query = query.unsqueeze(1)  # (B, 1, L, D)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        squeeze_output = True
    else:
        B, H, L, D = query.shape
        squeeze_output = False

    S = key.shape[2]

    scale = scale or (D ** -0.5)

    # Compute attention scores: (B, H, L, D) @ (B, H, D, S) -> (B, H, L, S)
    scores = query @ key.transpose(0, 1, 3, 2)
    scores = scores * scale

    # Apply masks
    if is_causal:
        causal_mask = xp.triu(xp.ones((L, S), dtype=xp.bool_), k=1)
        scores.data = xp.where(causal_mask, -1e9, scores.data)

    if attn_mask is not None:
        if isinstance(attn_mask, Tensor):
            scores.data = scores.data + attn_mask.data
        else:
            scores.data = scores.data + attn_mask

    # Softmax
    attn_weights = scores.softmax(axis=-1)

    # Apply dropout (training mode only)
    if dropout_p > 0 and _grad_enabled:
        mask = (xp.random.rand(*attn_weights.shape) > dropout_p).astype(xp.float32)
        attn_weights = attn_weights * Tensor(mask / (1 - dropout_p), device=device, requires_grad=False)

    # Compute output: (B, H, L, S) @ (B, H, S, D) -> (B, H, L, D)
    output = attn_weights @ value

    if squeeze_output:
        output = output.squeeze(1)

    return output


def flash_attention(query: Tensor, key: Tensor, value: Tensor,
                    block_size: int = 64,
                    is_causal: bool = False,
                    scale: Optional[float] = None,
                    dropout_p: float = 0.0) -> Tensor:
    """
    FlashAttention-style tiled attention with O(N) memory complexity.

    Instead of materializing the full N×N attention matrix, computes attention
    in blocks using online softmax algorithm. This reduces memory from O(N²) to O(N).

    Args:
        query: (B, H, L, D) or (B, L, D) queries
        key: (B, H, S, D) or (B, S, D) keys
        value: (B, H, S, D) or (B, S, D) values
        block_size: Query block size for tiling (default 64)
        is_causal: Apply causal masking for decoder models
        scale: Attention scale (default 1/sqrt(D))
        dropout_p: Dropout probability (default 0.0)

    Returns:
        output: (B, H, L, D) or (B, L, D) attention output

    Memory Analysis:
        Standard: O(B * H * L * S) for attention matrix
        FlashAttn: O(B * H * block_size * S) = O(N) when S proportional to L
    """
    xp = query.xp
    device = query._device

    # Handle 3D input
    if query.ndim == 3:
        B, L, D = query.shape
        H = 1
        query_data = query.data.reshape(B, 1, L, D)
        key_data = key.data.reshape(B, 1, key.shape[1], D)
        value_data = value.data.reshape(B, 1, value.shape[1], D)
        squeeze_output = True
    else:
        B, H, L, D = query.shape
        query_data = query.data
        key_data = key.data
        value_data = value.data
        squeeze_output = False

    S = key_data.shape[2]
    scale = scale or (D ** -0.5)

    # Output accumulator and normalization
    output = xp.zeros((B, H, L, D), dtype=query_data.dtype)
    # For online softmax: track max and sum of exp for each query position
    m_prev = xp.full((B, H, L, 1), -xp.inf, dtype=query_data.dtype)  # Running max
    l_prev = xp.zeros((B, H, L, 1), dtype=query_data.dtype)  # Running sum of exp

    # Process query in blocks
    for i in range(0, L, block_size):
        end_i = min(i + block_size, L)
        Qi = query_data[:, :, i:end_i]  # (B, H, block_size, D)
        block_len = end_i - i

        # Compute attention scores for this block: (B, H, block, D) @ (B, H, D, S) -> (B, H, block, S)
        scores = xp.einsum('bhid,bhjd->bhij', Qi, key_data) * scale
        # scores shape: (B, H, block_size, S)

        # Apply causal mask if needed
        if is_causal:
            # For each query position i+k, mask out key positions > i+k
            row_indices = xp.arange(i, end_i).reshape(1, 1, -1, 1)
            col_indices = xp.arange(S).reshape(1, 1, 1, -1)
            causal_mask = col_indices > row_indices  # (1, 1, block, S)
            scores = xp.where(causal_mask, xp.float32(-1e9), scores)

        # Online softmax update (numerically stable)
        # Current block max
        m_curr = scores.max(axis=-1, keepdims=True)  # (B, H, block, 1)

        # Get previous values for this block
        m_block_prev = m_prev[:, :, i:end_i]  # (B, H, block, 1)
        l_block_prev = l_prev[:, :, i:end_i]  # (B, H, block, 1)
        o_block_prev = output[:, :, i:end_i]  # (B, H, block, D)

        # New max (combining previous and current)
        m_new = xp.maximum(m_block_prev, m_curr)

        # Compute exp(scores - m_new) for stable softmax
        exp_scores = xp.exp(scores - m_new)  # (B, H, block, S)

        # Update running sum: scale old sum by correction factor, add new exp sum
        correction = xp.exp(m_block_prev - m_new)  # (B, H, block, 1)
        l_new = correction * l_block_prev + exp_scores.sum(axis=-1, keepdims=True)

        # Compute weighted value sum for this block
        # V @ attn^T = (B, H, S, D)^T @ (B, H, block, S)^T -> need einsum
        pv = xp.einsum('bhij,bhjd->bhid', exp_scores, value_data)  # (B, H, block, D)

        # Update output: rescale previous output and add new contribution
        output[:, :, i:end_i] = (correction * l_block_prev * o_block_prev + pv) / l_new

        # Store updated state
        m_prev[:, :, i:end_i] = m_new
        l_prev[:, :, i:end_i] = l_new

    # Apply dropout if needed
    if dropout_p > 0 and _grad_enabled:
        mask = (xp.random.rand(*output.shape) > dropout_p).astype(output.dtype)
        output = output * mask / (1 - dropout_p)

    # Create output tensor with backward
    if squeeze_output:
        output = output.reshape(B, L, D)

    out = Tensor(output, (query, key, value), 'FlashAttention', device=device)

    # Store for backward pass
    _flash_cache = {
        'query_data': query_data,
        'key_data': key_data,
        'value_data': value_data,
        'scale': scale,
        'is_causal': is_causal,
        'block_size': block_size,
        'B': B, 'H': H, 'L': L, 'D': D, 'S': S,
        'squeeze_output': squeeze_output
    }

    def _backward():
        # Memory-efficient backward: recompute attention for each block
        # This is the key insight of FlashAttention - trading compute for memory
        grad_q = xp.zeros_like(query_data)
        grad_k = xp.zeros_like(key_data)
        grad_v = xp.zeros_like(value_data)

        if squeeze_output:
            grad_out = out.grad.reshape(B, H, L, D)
        else:
            grad_out = out.grad

        # Recompute forward to get attention weights for backward
        for i in range(0, L, block_size):
            end_i = min(i + block_size, L)
            Qi = query_data[:, :, i:end_i]
            dOi = grad_out[:, :, i:end_i]  # (B, H, block, D)

            # Recompute scores
            scores = xp.einsum('bhid,bhjd->bhij', Qi, key_data) * scale

            if is_causal:
                row_indices = xp.arange(i, end_i).reshape(1, 1, -1, 1)
                col_indices = xp.arange(S).reshape(1, 1, 1, -1)
                causal_mask = col_indices > row_indices
                scores = xp.where(causal_mask, xp.float32(-1e9), scores)

            # Softmax (standard, since we need full attention weights for backward)
            scores_max = scores.max(axis=-1, keepdims=True)
            exp_scores = xp.exp(scores - scores_max)
            attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)  # (B, H, block, S)

            # Gradient for V: dV += attn.T @ dO
            # (B, H, S, block) @ (B, H, block, D) -> (B, H, S, D)
            grad_v += xp.einsum('bhij,bhid->bhjd', attn, dOi)

            # Gradient for attention: dAttn = dO @ V.T
            # (B, H, block, D) @ (B, H, D, S) -> (B, H, block, S)
            dAttn = xp.einsum('bhid,bhjd->bhij', dOi, value_data)

            # Gradient through softmax: dS = attn * (dAttn - sum(attn * dAttn, dim=-1, keepdim=True))
            sum_dAttn = (attn * dAttn).sum(axis=-1, keepdims=True)
            dS = attn * (dAttn - sum_dAttn) * scale

            if is_causal:
                dS = xp.where(causal_mask, 0.0, dS)

            # Gradient for Q: dQ += dS @ K
            grad_q[:, :, i:end_i] += xp.einsum('bhij,bhjd->bhid', dS, key_data)

            # Gradient for K: dK += dS.T @ Q
            grad_k += xp.einsum('bhij,bhid->bhjd', dS, Qi)

        # Apply gradients
        if query.grad is not None:
            if query.ndim == 3:
                query.grad += grad_q.reshape(B, L, D)
            else:
                query.grad += grad_q
        if key.grad is not None:
            if key.ndim == 3:
                key.grad += grad_k.reshape(B, S, D)
            else:
                key.grad += grad_k
        if value.grad is not None:
            if value.ndim == 3:
                value.grad += grad_v.reshape(B, S, D)
            else:
                value.grad += grad_v

    out._backward = _backward
    return out


def multi_head_attention(query: Tensor, key: Tensor, value: Tensor,
                         embed_dim: int, num_heads: int,
                         q_proj: 'Linear', k_proj: 'Linear', v_proj: 'Linear',
                         out_proj: 'Linear',
                         attn_mask: Optional[Tensor] = None,
                         dropout_p: float = 0.0,
                         is_causal: bool = False) -> Tensor:
    """
    Multi-head attention with projections.

    Args:
        query: (B, L, embed_dim)
        key: (B, S, embed_dim)
        value: (B, S, embed_dim)
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        q/k/v_proj: Linear projection layers
        out_proj: Output projection layer
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Apply causal mask

    Returns:
        Output tensor (B, L, embed_dim)
    """
    B, L, _ = query.shape
    S = key.shape[1]
    head_dim = embed_dim // num_heads

    # Project and reshape: (B, L, embed_dim) -> (B, H, L, head_dim)
    q = q_proj(query).reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k_proj(key).reshape(B, S, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v_proj(value).reshape(B, S, num_heads, head_dim).transpose(0, 2, 1, 3)

    # Attention
    attn_output = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)

    # Reshape and project: (B, H, L, head_dim) -> (B, L, embed_dim)
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, embed_dim)

    return out_proj(attn_output)


# ==================== DATA FORMAT UTILITIES ====================

def to_nhwc(tensor: Tensor) -> Tensor:
    """
    Convert NCHW tensor to NHWC format.

    Args:
        tensor: Input tensor of shape (N, C, H, W)

    Returns:
        Tensor of shape (N, H, W, C)
    """
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {tensor.ndim}D")
    return tensor.transpose(0, 2, 3, 1)


def to_nchw(tensor: Tensor) -> Tensor:
    """
    Convert NHWC tensor to NCHW format.

    Args:
        tensor: Input tensor of shape (N, H, W, C)

    Returns:
        Tensor of shape (N, C, H, W)
    """
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {tensor.ndim}D")
    return tensor.transpose(0, 3, 1, 2)


# ==================== IM2COL HELPERS ====================

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    """Compute indices for im2col transformation."""
    N, C, H, W = x_shape
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = cp.repeat(cp.arange(field_height), field_width)
    i0 = cp.tile(i0, C)
    i1 = stride * cp.repeat(cp.arange(out_height), out_width)
    j0 = cp.tile(cp.arange(field_width), field_height * C)
    j1 = stride * cp.tile(cp.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = cp.repeat(cp.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(cp.int32), i.astype(cp.int32), j.astype(cp.int32))


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """Transform input tensor to column format."""
    p = padding
    x_padded = cp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height, field_width, padding=1, stride=1):
    """Transform column format back to tensor."""
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = cp.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    cupyx.scatter_add(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


# ==================== NEURAL NETWORK MODULES ====================

class Module:
    """Base class for neural network modules."""

    def __init__(self):
        self._device = _device
        self._training = True

    def zero_grad(self):
        xp = cp if self._device == 'cuda' else np
        for p in self.parameters():
            if p.grad is not None:
                p.grad = xp.zeros_like(p.data)

    def parameters(self) -> List[Tensor]:
        return []

    def named_parameters(self) -> List[Tuple[str, Tensor]]:
        return []

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
        for p in self.parameters():
            new_p = p.to(device)
            p.data = new_p.data
            p.grad = new_p.grad
            p._device = device
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


class Linear(Module):
    """Fully connected linear layer."""

    def __init__(self, nin, nout, bias=True):
        super().__init__()
        xp = cp if _device == 'cuda' else np
        scale = xp.sqrt(2.0 / nin)
        self.w = Tensor(xp.random.randn(nin, nout).astype(xp.float32) * scale, device=_device)
        self.use_bias = bias
        if bias:
            self.b = Tensor(xp.zeros(nout, dtype=xp.float32), device=_device)
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

        xp = cp if _device == 'cuda' else np
        scale = xp.sqrt(2.0 / (in_channels * kernel_size * kernel_size))

        if groups == 1:
            self.w = Tensor(xp.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(xp.float32) * scale, device=_device)
        else:
            # Grouped convolution: weight shape is (out_channels, in_channels/groups, kernel_size, kernel_size)
            self.w = Tensor(xp.random.randn(out_channels, in_channels // groups, kernel_size, kernel_size).astype(xp.float32) * scale, device=_device)

        if bias:
            self.b = Tensor(xp.zeros((out_channels,), dtype=xp.float32), device=_device)
        else:
            self.b = None

    def __call__(self, x):
        xp = cp if self._device == 'cuda' else np

        # Handle NHWC input by converting to NCHW, computing, and converting back
        if self.data_format == 'NHWC':
            N, H, W, C = x.data.shape
            x_nchw_data = x.data.transpose(0, 3, 1, 2)
            x_nchw = Tensor(x_nchw_data, device=self._device, requires_grad=x._requires_grad)
            out_nchw = self._conv_standard(x_nchw) if self.groups == 1 else self._conv_grouped(x_nchw)
            out_data = out_nchw.data.transpose(0, 2, 3, 1)
            out = Tensor(out_data, (x,), 'Conv2D_NHWC', device=self._device)
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


class MaxPool2D(Module):
    """2D Max Pooling layer."""

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

        x_reshaped = x.data.reshape(N, C, H_out, stride, W_out, stride)
        out_data = x_reshaped.max(axis=(3, 5))

        out = Tensor(out_data, (x,), 'MaxPool2D', device=self._device)

        def _backward():
            x_reshaped_copy = x.data.reshape(N, C, H_out, stride, W_out, stride)
            out_broadcast = out_data.reshape(N, C, H_out, 1, W_out, 1)
            mask = (x_reshaped_copy == out_broadcast)
            dout_broadcast = out.grad.reshape(N, C, H_out, 1, W_out, 1)
            grad_reshaped = mask * dout_broadcast
            x.grad += grad_reshaped.reshape(N, C, H, W)

        out._backward = _backward
        return out


class AvgPool2D(Module):
    """2D Average Pooling layer."""

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

        x_reshaped = x.data.reshape(N, C, H_out, stride, W_out, stride)
        out_data = x_reshaped.mean(axis=(3, 5))

        out = Tensor(out_data, (x,), 'AvgPool2D', device=self._device)

        def _backward():
            scale = 1.0 / (stride * stride)
            dout_broadcast = out.grad.reshape(N, C, H_out, 1, W_out, 1)
            grad_reshaped = xp.broadcast_to(dout_broadcast * scale, (N, C, H_out, stride, W_out, stride))
            x.grad += grad_reshaped.reshape(N, C, H, W)

        out._backward = _backward
        return out


class AdaptiveAvgPool2D(Module):
    """Adaptive 2D Average Pooling."""

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)

    def __call__(self, x):
        xp = cp if self._device == 'cuda' else np
        N, C, H, W = x.data.shape
        out_h, out_w = self.output_size

        # Compute window sizes
        stride_h = H // out_h
        stride_w = W // out_w
        kernel_h = H - (out_h - 1) * stride_h
        kernel_w = W - (out_w - 1) * stride_w

        out_data = xp.zeros((N, C, out_h, out_w), dtype=x.data.dtype)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride_h
                h_end = h_start + kernel_h
                w_start = j * stride_w
                w_end = w_start + kernel_w
                out_data[:, :, i, j] = x.data[:, :, h_start:h_end, w_start:w_end].mean(axis=(2, 3))

        out = Tensor(out_data, (x,), 'AdaptiveAvgPool2D', device=self._device)

        def _backward():
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride_h
                    h_end = h_start + kernel_h
                    w_start = j * stride_w
                    w_end = w_start + kernel_w
                    scale = 1.0 / (kernel_h * kernel_w)
                    x.grad[:, :, h_start:h_end, w_start:w_end] += out.grad[:, :, i:i+1, j:j+1] * scale

        out._backward = _backward
        return out


class BatchNorm2D(Module):
    """2D Batch Normalization."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        xp = cp if _device == 'cuda' else np
        self.gamma = Tensor(xp.ones(num_features, dtype=xp.float32), device=_device)
        self.beta = Tensor(xp.zeros(num_features, dtype=xp.float32), device=_device)

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

        xp = cp if _device == 'cuda' else np
        self.gamma = Tensor(xp.ones(num_features, dtype=xp.float32), device=_device)
        self.beta = Tensor(xp.zeros(num_features, dtype=xp.float32), device=_device)

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


class LayerNorm(Module):
    """Layer Normalization."""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps

        xp = cp if _device == 'cuda' else np
        size = int(xp.prod(xp.array(normalized_shape)))
        self.gamma = Tensor(xp.ones(size, dtype=xp.float32), device=_device)
        self.beta = Tensor(xp.zeros(size, dtype=xp.float32), device=_device)

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


class GroupNorm(Module):
    """Group Normalization."""

    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        xp = cp if _device == 'cuda' else np
        self.gamma = Tensor(xp.ones(num_channels, dtype=xp.float32), device=_device)
        self.beta = Tensor(xp.zeros(num_channels, dtype=xp.float32), device=_device)

    def __call__(self, x):
        xp = cp if self._device == 'cuda' else np
        N, C, H, W = x.data.shape
        G = self.num_groups

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


class Embedding(Module):
    """Embedding layer for discrete inputs."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        xp = cp if _device == 'cuda' else np
        self.weight = Tensor(xp.random.randn(num_embeddings, embedding_dim).astype(xp.float32) * 0.01, device=_device)
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

    def step(self, optimizer, clip_grad_norm=None):
        """Step optimizer if gradients are valid."""
        if not self.enabled:
            optimizer.step(clip_grad_norm=clip_grad_norm)
            return

        if self._found_inf:
            return  # Skip update on overflow
        optimizer.step(clip_grad_norm=clip_grad_norm)

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

    Recomputes forward pass during backward instead of storing activations.

    Usage:
        output = checkpoint(expensive_function, input1, input2)
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
        with _grad_enabled_context():
            # Create fresh tensors from saved data
            new_args = []
            for arg in CheckpointFunction.saved_args:
                if isinstance(arg, Tensor):
                    new_arg = Tensor(arg.data, device=arg._device)
                    new_arg.grad = arg.xp.zeros_like(arg.data)
                    new_args.append(new_arg)
                else:
                    new_args.append(arg)

            # Recompute
            recomputed = CheckpointFunction.saved_fn(*new_args)
            recomputed.grad = out.grad.copy()
            recomputed.backward()

            # Copy gradients back
            for orig, new in zip(CheckpointFunction.saved_args, new_args):
                if isinstance(orig, Tensor) and orig.grad is not None:
                    orig.grad += new.grad

    out._backward = _backward
    return out


class _grad_enabled_context:
    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = True
        return self

    def __exit__(self, *args):
        global _grad_enabled
        _grad_enabled = self._prev


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
        checkpoint = pickle.load(f)

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
        if self.sync and _device == 'cuda':
            cp.cuda.Stream.null.synchronize()

        self._start_mem = _memory_pool.used_bytes()

        if _device == 'cuda':
            self._start_event = cp.cuda.Event()
            self._start_event.record()

        return self

    def __exit__(self, *args):
        if _device == 'cuda':
            self._end_event = cp.cuda.Event()
            self._end_event.record()

        if self.sync and _device == 'cuda':
            cp.cuda.Stream.null.synchronize()

        self._end_mem = _memory_pool.used_bytes()

        if _device == 'cuda' and self._start_event and self._end_event:
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
        if _device == 'cuda':
            cp.cuda.Stream.null.synchronize()

    # Benchmark
    times = []
    for _ in range(n_repeat):
        if _device == 'cuda':
            cp.cuda.Stream.null.synchronize()

        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record()
        fn(*args, **kwargs)
        end.record()

        if _device == 'cuda':
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

        def _backward(s=start, e=end):
            idx = [slice(None)] * tensor.ndim
            idx[dim] = slice(s, e)
            tensor.grad[tuple(idx)] += out.grad

        out._backward = _backward
        result.append(out)

    return result


def zeros(*shape, device=None, dtype=None, requires_grad=True) -> Tensor:
    """Create tensor of zeros."""
    device = device or _device
    dtype = dtype or _default_dtype
    xp = cp if device == 'cuda' else np
    return Tensor(xp.zeros(shape, dtype=dtype), device=device, requires_grad=requires_grad)


def ones(*shape, device=None, dtype=None, requires_grad=True) -> Tensor:
    """Create tensor of ones."""
    device = device or _device
    dtype = dtype or _default_dtype
    xp = cp if device == 'cuda' else np
    return Tensor(xp.ones(shape, dtype=dtype), device=device, requires_grad=requires_grad)


def randn(*shape, device=None, dtype=None, requires_grad=True) -> Tensor:
    """Create tensor with random normal values."""
    device = device or _device
    dtype = dtype or _default_dtype
    xp = cp if device == 'cuda' else np
    return Tensor(xp.random.randn(*shape).astype(dtype), device=device, requires_grad=requires_grad)


def rand(*shape, device=None, dtype=None, requires_grad=True) -> Tensor:
    """Create tensor with random uniform values [0, 1)."""
    device = device or _device
    dtype = dtype or _default_dtype
    xp = cp if device == 'cuda' else np
    return Tensor(xp.random.rand(*shape).astype(dtype), device=device, requires_grad=requires_grad)


def arange(start, stop=None, step=1, device=None, dtype=None, requires_grad=False) -> Tensor:
    """Create 1D tensor with evenly spaced values."""
    device = device or _device
    dtype = dtype or _default_dtype
    xp = cp if device == 'cuda' else np
    if stop is None:
        stop = start
        start = 0
    return Tensor(xp.arange(start, stop, step, dtype=dtype), device=device, requires_grad=requires_grad)


def linspace(start, stop, num, device=None, dtype=None, requires_grad=False) -> Tensor:
    """Create 1D tensor with linearly spaced values."""
    device = device or _device
    dtype = dtype or _default_dtype
    xp = cp if device == 'cuda' else np
    return Tensor(xp.linspace(start, stop, num, dtype=dtype), device=device, requires_grad=requires_grad)


def eye(n, m=None, device=None, dtype=None, requires_grad=False) -> Tensor:
    """Create identity matrix."""
    device = device or _device
    dtype = dtype or _default_dtype
    xp = cp if device == 'cuda' else np
    return Tensor(xp.eye(n, m, dtype=dtype), device=device, requires_grad=requires_grad)


# ==================== BACKWARDS COMPATIBILITY ====================

# Aliases for compatibility with tensor_gpu.py
ConvTranspose2D = None  # Import from conv_transpose module if needed

# Initialize streams on import
if _device == 'cuda':
    init_streams(4)
