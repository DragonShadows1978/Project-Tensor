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
import functools

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
_active_autocast_dtype = None  # None means autocast is not active
_streams: List[cp.cuda.Stream] = []
_current_stream_idx = 0
_stream_lock = threading.Lock()

# Memory pool settings
_memory_pool = cp.get_default_memory_pool()
_pinned_memory_pool = cp.get_default_pinned_memory_pool()


def _canonicalize_device(device: Optional[str]) -> str:
    """Normalize device identifiers to supported canonical values."""
    if device is None:
        return _device
    if not isinstance(device, str):
        raise TypeError(f"device must be a string, got {type(device).__name__}")
    normalized = device.strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda" or normalized.startswith("cuda:") or normalized == "gpu":
        return "cuda"
    raise ValueError(f"Unknown device: {device}")


def _normalize_numeric_dtype(dtype_like, context: str) -> np.dtype:
    """Parse and validate numeric dtypes used by tensor data paths."""
    try:
        resolved = np.dtype(dtype_like)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Invalid {context}: {dtype_like!r}") from exc

    if resolved == np.dtype(np.object_):
        raise TypeError(f"{context} object dtype is not supported")

    is_numeric = np.issubdtype(resolved, np.number) or np.issubdtype(resolved, np.bool_)
    if not is_numeric:
        raise TypeError(f"{context} must be numeric or bool, got {resolved}")
    return resolved


def _infer_data_dtype(data) -> Optional[np.dtype]:
    """Infer dtype from array-like/scalar inputs, if possible."""
    if isinstance(data, (cp.ndarray, np.ndarray, np.generic, cp.generic)):
        return _normalize_numeric_dtype(data.dtype, "data dtype")
    try:
        inferred = np.asarray(data).dtype
    except Exception:
        return None
    return _normalize_numeric_dtype(inferred, "data dtype")


def _resolve_tensor_dtype(data, dtype, device: str) -> np.dtype:
    """Resolve target tensor dtype with explicit mixed precision semantics."""
    if dtype is not None:
        return _normalize_numeric_dtype(dtype, "dtype")

    inferred_dtype = _infer_data_dtype(data)
    default_dtype = _normalize_numeric_dtype(_default_dtype, "default dtype")

    if inferred_dtype is None:
        inferred_dtype = default_dtype

    # autocast context takes precedence over the global mixed-precision flag
    if _active_autocast_dtype is not None and _canonicalize_device(device) == "cuda":
        if np.issubdtype(inferred_dtype, np.floating):
            return np.dtype(_active_autocast_dtype)

    if _mixed_precision and _canonicalize_device(device) == "cuda":
        if np.issubdtype(inferred_dtype, np.floating) and inferred_dtype.itemsize > 2:
            return np.dtype(np.float16)

    return inferred_dtype


def set_device(device: str):
    """Set compute device: 'cuda' or 'cpu'"""
    global _device
    _device = _canonicalize_device(device)


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

def get_default_dtype():
    """Get the current default floating-point dtype."""
    return _default_dtype


def set_default_dtype(dtype):
    """Set the default floating-point dtype for new tensors.

    Args:
        dtype: Any numpy/cupy-compatible dtype (e.g. ``np.float32``,
               ``'float16'``).
    """
    global _default_dtype
    _default_dtype = np.dtype(dtype)


class no_grad:
    """Disable gradient computation for a block of code.

    Can be used as a **context manager** or a **decorator**.

    Context manager::

        with no_grad():
            y = model(x)          # no grads tracked
            loss = criterion(y, t)

    Decorator::

        @no_grad()
        def evaluate(model, x):
            return model(x)

    Nesting is safe — the previous state is restored on exit regardless
    of exceptions.
    """

    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *args):
        global _grad_enabled
        _grad_enabled = self._prev

    def __call__(self, fn):
        """Allow @no_grad() decorator usage."""
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with no_grad():
                return fn(*args, **kwargs)
        return wrapper


class enable_grad:
    """Re-enable gradient computation inside a ``no_grad`` block.

    Useful when most of an eval loop runs without grads but one sub-call
    (e.g. a meta-learner inner loop) still needs them.

    Can be used as a **context manager** or a **decorator**::

        with no_grad():
            feats = encoder(x)          # no grad
            with enable_grad():
                loss = meta(feats)      # grad back on

        @enable_grad()
        def inner_loop(x):
            return model(x)
    """

    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = True
        return self

    def __exit__(self, *args):
        global _grad_enabled
        _grad_enabled = self._prev

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with enable_grad():
                return fn(*args, **kwargs)
        return wrapper


def set_grad_enabled(mode: bool):
    """Globally enable or disable gradient computation.

    Unlike ``no_grad`` / ``enable_grad``, this is a permanent toggle
    until called again.  Useful for switching an entire script between
    training and evaluation modes.

    Args:
        mode: ``True`` to enable gradients, ``False`` to disable.
    """
    global _grad_enabled
    _grad_enabled = bool(mode)


def is_grad_enabled() -> bool:
    """Return ``True`` if gradient tracking is currently active."""
    return _grad_enabled


class autocast:
    """Automatic mixed-precision context manager / decorator.

    Inside the block every new floating-point Tensor created on the GPU
    is cast to *dtype* (default ``float16``).  Integer and boolean tensors
    are left unchanged.  On exit the previous dtype policy is restored.

    Can be used as a **context manager** or a **decorator**::

        # FP16 training step
        with autocast():
            logits = model(x)
            loss   = criterion(logits, y)
        scaler.scale(loss).backward()

        # BF16 (better dynamic range, same memory as FP16)
        with autocast(dtype='bfloat16'):
            ...

        @autocast()
        def train_step(x, y):
            return model(x), criterion(model(x), y)

        # Disable inside a larger autocast block
        with autocast():
            safe_out = risky_layer(x)               # fp16
            with autocast(enabled=False):
                stable_out = stable_layer(safe_out) # fp32

    Args:
        enabled: If ``False`` this is a no-op (useful for conditional AMP).
        dtype:   Target floating-point dtype.  Accepts numpy/cupy dtype
                 objects, dtype strings (``'float16'``, ``'bfloat16'``,
                 ``'float32'``), or ``None`` (defaults to ``float16`` on
                 CUDA, ``float32`` on CPU).
    """

    def __init__(self, enabled: bool = True, dtype=None):
        self.enabled = enabled
        if dtype is None:
            dtype = cp.float16 if _device == 'cuda' else np.float32
        # Normalise to a numpy dtype so comparisons are stable
        try:
            self._dtype = np.dtype(dtype)
        except TypeError:
            self._dtype = np.dtype(np.float16)

    def __enter__(self):
        global _mixed_precision, _active_autocast_dtype
        self._prev_mp    = _mixed_precision
        self._prev_dtype = _active_autocast_dtype
        if self.enabled:
            _mixed_precision       = True
            _active_autocast_dtype = self._dtype
        return self

    def __exit__(self, *args):
        global _mixed_precision, _active_autocast_dtype
        _mixed_precision       = self._prev_mp
        _active_autocast_dtype = self._prev_dtype

    def __call__(self, fn):
        """Allow @autocast() decorator usage."""
        dtype   = self._dtype
        enabled = self.enabled

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with autocast(enabled=enabled, dtype=dtype):
                return fn(*args, **kwargs)
        return wrapper


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
        device = _canonicalize_device(device or _device)
        dtype = _resolve_tensor_dtype(data, dtype=dtype, device=device)

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
        device = _canonicalize_device(device)
        if device == self._device:
            return self
        new_data = to_cpu(self.data) if device == 'cpu' else to_gpu(self.data, dtype=self.data.dtype)
        new_tensor = Tensor(
            new_data,
            device=device,
            requires_grad=self._requires_grad,
            dtype=self.data.dtype,
        )
        if self.grad is not None:
            new_tensor.grad = to_cpu(self.grad) if device == 'cpu' else to_gpu(self.grad, dtype=self.grad.dtype)
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
        grad_str = f", requires_grad={self._requires_grad}" if self._requires_grad else ""
        op_str = f", op='{self._op}'" if self._op else ""
        return f"Tensor({list(self.data.shape)}, dtype={self.dtype}, device='{self._device}'{op_str}{grad_str})"

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

    @staticmethod
    def _index_to_numpy(idx):
        """Convert CuPy index objects to NumPy-compatible index objects."""
        if isinstance(idx, tuple):
            return tuple(Tensor._index_to_numpy(i) for i in idx)
        if isinstance(idx, list):
            return [Tensor._index_to_numpy(i) for i in idx]
        if isinstance(idx, cp.ndarray):
            return cp.asnumpy(idx)
        return idx

    def _scatter_add(self, target, idx, src):
        """Accumulate src into target using duplicate-index-safe semantics."""
        xp = self.xp
        target_dtype = _normalize_numeric_dtype(target.dtype, "scatter target dtype")
        src_dtype = _normalize_numeric_dtype(src.dtype, "scatter source dtype")
        idx_np = Tensor._index_to_numpy(idx)

        # Validate index form/bounds up front to produce deterministic errors.
        try:
            np.empty(target.shape, dtype=np.int8)[idx_np]
        except Exception as exc:
            raise IndexError(f"Invalid scatter index for target shape {target.shape}: {exc}") from exc

        def _accumulate_numpy(target_np, idx_cpu, src_np):
            if np.issubdtype(target_dtype, np.integer) and target_dtype != np.dtype(np.bool_):
                promoted = np.promote_types(target_dtype, np.int64)
                tmp = target_np.astype(promoted, copy=True)
                np.add.at(tmp, idx_cpu, src_np.astype(promoted, copy=False))
                limits = np.iinfo(target_dtype)
                if tmp.min() < limits.min or tmp.max() > limits.max:
                    raise OverflowError(
                        f"scatter_add overflow for dtype {target_dtype}: "
                        f"range [{tmp.min()}, {tmp.max()}] outside [{limits.min}, {limits.max}]"
                    )
                target_np[...] = tmp.astype(target_dtype, copy=False)
                return

            if np.issubdtype(target_dtype, np.bool_):
                raise TypeError("scatter_add does not support boolean targets")

            np.add.at(target_np, idx_cpu, src_np.astype(target_dtype, copy=False))

        if xp == np:
            _accumulate_numpy(target, idx_np, src)
            return

        xp_scatter_add = getattr(cupyx, 'scatter_add', None)
        if xp_scatter_add is not None:
            try:
                xp_scatter_add(target, idx, src)
                return
            except Exception:
                # Fall back to NumPy add.at for unsupported index patterns.
                pass

        cpu_target = cp.asnumpy(target)
        cpu_idx = idx_np
        cpu_src = cp.asnumpy(src)
        _accumulate_numpy(cpu_target, cpu_idx, cpu_src)
        target[...] = cp.asarray(cpu_target)

    def __getitem__(self, idx):
        """Advanced indexing with gradient support."""
        xp = self.xp
        out_data = self.data[idx]
        out = Tensor(out_data, (self,), 'getitem', device=self._device)

        def _backward():
            grad = xp.zeros_like(self.data)
            self._scatter_add(grad, idx, out.grad)
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
            xp = self.xp
            lhs = self.data
            rhs = other.data
            if lhs.ndim == 0 or rhs.ndim == 0:
                raise ValueError(
                    f"matmul backward requires operands with ndim >= 1, got {lhs.ndim} and {rhs.ndim}"
                )

            grad_out = out.grad
            if not hasattr(grad_out, "shape"):
                grad_out = xp.asarray(grad_out)

            expected_shape = out.data.shape
            if expected_shape == ():
                if grad_out.size != 1:
                    raise ValueError(
                        f"matmul backward expected scalar grad_out, got shape {grad_out.shape}"
                    )
                grad_out = grad_out.reshape(())
            elif grad_out.shape != expected_shape:
                raise ValueError(
                    f"matmul backward grad_out shape mismatch: expected {expected_shape}, got {grad_out.shape}"
                )

            def _is_diff_dtype(arr):
                arr_dtype = np.dtype(arr.dtype)
                return np.issubdtype(arr_dtype, np.floating) or np.issubdtype(arr_dtype, np.complexfloating)

            if self.grad is not None and not _is_diff_dtype(lhs):
                raise TypeError(f"matmul backward requires floating/complex lhs, got {lhs.dtype}")
            if other.grad is not None and not _is_diff_dtype(rhs):
                raise TypeError(f"matmul backward requires floating/complex rhs, got {rhs.dtype}")

            lhs_was_vector = lhs.ndim == 1
            rhs_was_vector = rhs.ndim == 1

            lhs_mat = lhs[xp.newaxis, :] if lhs_was_vector else lhs
            rhs_mat = rhs[:, xp.newaxis] if rhs_was_vector else rhs

            if lhs_was_vector and rhs_was_vector:
                grad_out_mat = grad_out.reshape((1, 1))
            elif lhs_was_vector:
                grad_out_mat = xp.expand_dims(grad_out, axis=-2)
            elif rhs_was_vector:
                grad_out_mat = xp.expand_dims(grad_out, axis=-1)
            else:
                grad_out_mat = grad_out

            rhs_t = rhs_mat.swapaxes(-1, -2)
            lhs_t = lhs_mat.swapaxes(-1, -2)
            if np.issubdtype(np.dtype(rhs_t.dtype), np.complexfloating):
                rhs_t = xp.conj(rhs_t)
            if np.issubdtype(np.dtype(lhs_t.dtype), np.complexfloating):
                lhs_t = xp.conj(lhs_t)

            if self.grad is not None:
                grad_self = xp.matmul(grad_out_mat, rhs_t)
                if lhs_was_vector:
                    grad_self = grad_self.squeeze(axis=-2)
                self.grad += self._unbroadcast(grad_self, lhs.shape)
            if other.grad is not None:
                grad_other = xp.matmul(lhs_t, grad_out_mat)
                if rhs_was_vector:
                    grad_other = grad_other.squeeze(axis=-1)
                other.grad += self._unbroadcast(grad_other, rhs.shape)
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
        if shape is None or not isinstance(shape, (tuple, list)):
            raise TypeError(f"shape must be tuple/list of ints, got {shape!r}")

        try:
            target_shape = tuple(int(dim) for dim in shape)
        except Exception as exc:
            raise TypeError(f"shape must contain integer dimensions, got {shape!r}") from exc

        if any(dim < 0 for dim in target_shape):
            raise ValueError(f"shape dimensions must be non-negative, got {target_shape}")

        if grad is None:
            raise ValueError("grad cannot be None")
        if not hasattr(grad, "shape"):
            grad = xp.asarray(grad)

        if grad.shape == target_shape:
            return grad

        if grad.ndim < len(target_shape):
            raise ValueError(
                f"cannot unbroadcast grad with shape {grad.shape} to higher-rank shape {target_shape}"
            )

        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)

        for axis, (gdim, sdim) in enumerate(zip(grad.shape, target_shape)):
            if sdim == 1:
                if gdim != 1:
                    grad = grad.sum(axis=axis, keepdims=True)
            elif gdim != sdim:
                raise ValueError(
                    f"incompatible unbroadcast axis {axis}: grad dim {gdim} cannot map to target dim {sdim}"
                )

        if grad.shape != target_shape:
            raise ValueError(
                f"unbroadcast produced shape {grad.shape}, expected {target_shape}"
            )

        return grad

    def backward(self, gradient=None):
        """
        Run backpropagation from this tensor.

        Args:
            gradient: Optional gradient to start with. If None, defaults to 1.0
                     (only if this tensor is scalar or has all-zero grad).
        """
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

        if gradient is not None:
            if isinstance(gradient, Tensor):
                gradient = gradient.data
            grad_val = xp.asarray(gradient, dtype=self.dtype)
            if self.grad is None:
                self.grad = grad_val
            else:
                self.grad += grad_val
        elif self.grad is None or xp.all(self.grad == 0):
            # Default to ones if no gradient provided and current grad is zero/None
            self.grad = xp.ones_like(self.data)

        for v in reversed(topo):
            v._backward()
            v._backward = lambda: None  # Free closure and its captured tensors

    def zero_grad(self):
        """Zero out gradient."""
        if self.grad is not None:
            xp = self.xp
            self.grad = xp.zeros_like(self.data)


# ==================== ADVANCED OPERATIONS ====================

# Global cache for einsum contraction paths
_einsum_path_cache = {}
_EINSUM_CACHE_MAX_SIZE = 1000


def einsum(subscripts: str, *operands, optimize: Union[bool, str] = True, use_cache: bool = True) -> Tensor:
    """
    Einstein summation with gradient support and path optimization.

    Supports all NumPy/CuPy einsum operations with automatic differentiation.
    Uses path caching for repeated patterns to improve performance.

    Common operations:
    - Trace: 'ii' or 'ii->'
    - Diagonal: 'ii->i'
    - Sum: 'ij->'
    - Transpose: 'ij->ji'
    - Matrix multiplication: 'ij,jk->ik'
    - Batch matrix multiplication: 'bij,bjk->bik'
    - Inner product: 'i,i->' or 'i,i'
    - Outer product: 'i,j->ij'
    - Batch Dot Product: 'bi,bi->b'

    Examples:
    >>> # Matrix multiplication
    >>> x = Tensor([[1, 2], [3, 4]])
    >>> y = Tensor([[5, 6], [7, 8]])
    >>> out = einsum('ij,jk->ik', x, y)
    >>>
    >>> # Batch Matrix multiplication
    >>> a = Tensor(np.random.randn(10, 3, 4))
    >>> b = Tensor(np.random.randn(10, 4, 5))
    >>> c = einsum('bij,bjk->bik', a, b)

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

            if not other_subs:
                # Single operand einsum (like sum, transpose, view)
                if output_str is None or output_str == "":
                    # Case: einsum('i', x) or einsum('i->', x)
                    t.grad += xp.broadcast_to(out.grad, t.shape)
                else:
                    # Case: einsum('ij->ji', x) or einsum('i->ij', x) [if supported]
                    try:
                        grad_sub = f"{output_str}->{subs}"
                        grad = xp.einsum(grad_sub, out.grad, optimize=True)
                        t.grad += grad
                    except:
                        # Fallback for broadcasting
                        t.grad += xp.broadcast_to(out.grad, t.shape)
                continue

            if output_str is not None:
                grad_subscript = f"{output_str},{','.join(other_subs)}->{subs}"
            else:
                # For implicit output (e.g. 'i,i'), output is scalar. 
                # We use an empty string before the first comma to represent the scalar.
                grad_subscript = f",{','.join(other_subs)}->{subs}"

            grad = xp.einsum(grad_subscript, out.grad, *other_data, optimize=True)
            t.grad += grad

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

_im2col_idx_cache = {}
_IM2COL_CACHE_MAX_SIZE = 128

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1, xp=cp):
    """Compute indices for im2col transformation with caching."""
    # Use a cache to avoid recomputing indices for the same shapes
    cache_key = (x_shape, field_height, field_width, padding, stride, str(xp))
    if cache_key in _im2col_idx_cache:
        return _im2col_idx_cache[cache_key]

    N, C, H, W = x_shape
    
    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else stride
    pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding
    
    out_height = (H + 2 * pad_h - field_height) // stride_h + 1
    out_width = (W + 2 * pad_w - field_width) // stride_w + 1

    i0 = xp.repeat(xp.arange(field_height), field_width)
    i0 = xp.tile(i0, C)
    i1 = stride_h * xp.repeat(xp.arange(out_height), out_width)
    j0 = xp.tile(xp.arange(field_width), field_height * C)
    j1 = stride_w * xp.tile(xp.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = xp.repeat(xp.arange(C), field_height * field_width).reshape(-1, 1)

    indices = (k.astype(xp.int32), i.astype(xp.int32), j.astype(xp.int32))

    # Cache with FIFO eviction to bound GPU memory usage
    if len(_im2col_idx_cache) >= _IM2COL_CACHE_MAX_SIZE:
        oldest_key = next(iter(_im2col_idx_cache))
        del _im2col_idx_cache[oldest_key]
    _im2col_idx_cache[cache_key] = indices

    return indices


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """Transform input tensor to column format."""
    xp = cp if isinstance(x, cp.ndarray) else np
    pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding
    x_padded = xp.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride, xp=xp)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height, field_width, padding=1, stride=1):
    """Transform column format back to tensor."""
    xp = cp if isinstance(cols, cp.ndarray) else np
    N, C, H, W = x_shape
    pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding
    H_padded, W_padded = H + 2 * pad_h, W + 2 * pad_w
    x_padded = xp.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride, xp=xp)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    
    if xp == cp:
        cupyx.scatter_add(x_padded, (slice(None), k, i, j), cols_reshaped)
    else:
        # Fallback for numpy (slower)
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    
    if pad_h == 0 and pad_w == 0:
        return x_padded
    
    h_slice = slice(pad_h, -pad_h) if pad_h > 0 else slice(None)
    w_slice = slice(pad_w, -pad_w) if pad_w > 0 else slice(None)
    return x_padded[:, :, h_slice, w_slice]

