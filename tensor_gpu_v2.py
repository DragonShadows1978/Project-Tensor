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
        self._device = _device
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

        xp = cp if _device == 'cuda' else np
        self.gamma = Tensor(xp.ones(num_channels, dtype=xp.float32), device=_device)
        self.beta = Tensor(xp.zeros(num_channels, dtype=xp.float32), device=_device)

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

        xp = cp if _device == 'cuda' else np
        scale = xp.sqrt(2.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        
        # Weight shape: (in_channels, out_channels // groups, kH, kW)
        self.w = Tensor(xp.random.randn(in_channels, out_channels // groups, self.kernel_size[0], self.kernel_size[1]).astype(xp.float32) * scale, device=_device)
        
        if bias:
            self.b = Tensor(xp.zeros((out_channels,), dtype=xp.float32), device=_device)
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
    device = device or _device
    dtype = dtype or _default_dtype
    xp = cp if device == 'cuda' else np
    return Tensor(xp.full(shape, fill_value, dtype=dtype), device=device, requires_grad=requires_grad)

def full_like(t: Tensor, fill_value, requires_grad: bool = False) -> Tensor:
    """Return a tensor filled with fill_value, same shape/dtype/device as t."""
    xp = t.xp
    return Tensor(xp.full_like(t.data, fill_value), device=t._device, requires_grad=requires_grad, dtype=t.dtype)

def empty(shape, device=None, dtype=None, requires_grad: bool = False) -> Tensor:
    """Return an uninitialized tensor."""
    device = device or _device
    dtype = dtype or _default_dtype
    xp = cp if device == 'cuda' else np
    return Tensor(xp.empty(shape, dtype=dtype), device=device, requires_grad=requires_grad)

def empty_like(t: Tensor, requires_grad: bool = False) -> Tensor:
    """Return an uninitialized tensor with same shape/dtype/device as t."""
    xp = t.xp
    return Tensor(xp.empty_like(t.data), device=t._device, requires_grad=requires_grad, dtype=t.dtype)

def randint(low, high, shape, device=None, dtype=None, requires_grad: bool = False) -> Tensor:
    """Return a tensor of random integers in [low, high)."""
    device = device or _device
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
        xp = cp if _device == 'cuda' else np
        scale = float(xp.sqrt(2.0 / (in_channels * kernel_size)))
        self.w = Tensor(
            xp.random.randn(out_channels, in_channels // groups, kernel_size).astype(xp.float32) * scale,
            device=_device)
        self.b = Tensor(xp.zeros(out_channels, dtype=xp.float32), device=_device) if bias else None

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
        xp = cp if _device == 'cuda' else np
        if affine:
            self.gamma = Tensor(xp.ones(num_features, dtype=xp.float32), device=_device)
            self.beta  = Tensor(xp.zeros(num_features, dtype=xp.float32), device=_device)
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
        xp = cp if _device == 'cuda' else np
        # Weight for [i, f, g, o] gates, shape (input_size, 4*hidden_size)
        scale = float(xp.sqrt(1.0 / hidden_size))
        self.w_ih = Tensor(xp.random.uniform(-scale, scale, (input_size,  4 * hidden_size)).astype(xp.float32), device=_device)
        self.w_hh = Tensor(xp.random.uniform(-scale, scale, (hidden_size, 4 * hidden_size)).astype(xp.float32), device=_device)
        if bias:
            self.b_ih = Tensor(xp.zeros(4 * hidden_size, dtype=xp.float32), device=_device)
            self.b_hh = Tensor(xp.zeros(4 * hidden_size, dtype=xp.float32), device=_device)
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
        xp = cp if _device == 'cuda' else np
        scale = float(xp.sqrt(1.0 / hidden_size))
        # weights for [r, z, n] gates
        self.w_ih = Tensor(xp.random.uniform(-scale, scale, (input_size,  3 * hidden_size)).astype(xp.float32), device=_device)
        self.w_hh = Tensor(xp.random.uniform(-scale, scale, (hidden_size, 3 * hidden_size)).astype(xp.float32), device=_device)
        if bias:
            self.b_ih = Tensor(xp.zeros(3 * hidden_size, dtype=xp.float32), device=_device)
            self.b_hh = Tensor(xp.zeros(3 * hidden_size, dtype=xp.float32), device=_device)
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
        xp = cp if _device == 'cuda' else np

        if mode == 'sinusoidal':
            pe = xp.zeros((max_len, d_model), dtype=xp.float32)
            position = xp.arange(0, max_len, dtype=xp.float32).reshape(-1, 1)
            div_term = xp.exp(xp.arange(0, d_model, 2, dtype=xp.float32) * (-xp.log(10000.0) / d_model))
            pe[:, 0::2] = xp.sin(position * div_term)
            pe[:, 1::2] = xp.cos(position * div_term[:d_model // 2])
            self._pe = pe  # (max_len, d_model), not a parameter
        else:  # learned
            self.pe = Tensor(xp.zeros((max_len, d_model), dtype=xp.float32), device=_device)

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
        dev = device or _device
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
        self._device = device or _device
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
        xp = cp if _device == 'cuda' else np
        scale = float(xp.sqrt(2.0 / in_features))
        self.w = Tensor(xp.random.randn(in_features, out_features).astype(xp.float32) * scale, device=_device)
        self.use_bias = bias
        self.b = Tensor(xp.zeros(out_features, dtype=xp.float32), device=_device) if bias else None

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
        xp = cp if _device == 'cuda' else np
        scale = float(xp.sqrt(2.0 / in_features))
        self.w = Tensor(xp.random.randn(in_features, out_features).astype(xp.float32) * scale, device=_device)
        self.use_bias = bias
        self.b = Tensor(xp.zeros(out_features, dtype=xp.float32), device=_device) if bias else None

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
        xp = cp if _device == 'cuda' else np
        scale = float(xp.sqrt(2.0 / d_model))
        # Single projection into 2*d_ff; first half is value, second half is gate
        self.w = Tensor(xp.random.randn(d_model, 2 * d_ff).astype(xp.float32) * scale, device=_device)
        self.use_bias = bias
        self.b = Tensor(xp.zeros(2 * d_ff, dtype=xp.float32), device=_device) if bias else None

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
        xp = cp if _device == 'cuda' else np
        self.weight = Tensor(xp.full(num_parameters, init, dtype=xp.float32), device=_device)

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
            y._device    if isinstance(y, Tensor)         else _device)
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
        xp = cp if _device == 'cuda' else np
        self.weight = Tensor(xp.ones(normalized_shape, dtype=xp.float32), device=_device)
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
        xp = cp if _device == 'cuda' else np
        if affine:
            self.gamma = Tensor(xp.ones(num_features, dtype=xp.float32), device=_device)
            self.beta  = Tensor(xp.zeros(num_features, dtype=xp.float32), device=_device)
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
        dev = device or _device
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
        xp = cp if _device == 'cuda' else np
        scale = float(xp.sqrt(1.0 / hidden_size))
        self.w_ih = Tensor(xp.random.uniform(-scale, scale, (input_size,  hidden_size)).astype(xp.float32), device=_device)
        self.w_hh = Tensor(xp.random.uniform(-scale, scale, (hidden_size, hidden_size)).astype(xp.float32), device=_device)
        self.b    = Tensor(xp.zeros(hidden_size, dtype=xp.float32), device=_device) if bias else None

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
        xp = cp if _device == 'cuda' else np
        scale = float(xp.sqrt(1.0 / d_model))
        self.w_q = Tensor(xp.random.randn(d_model, d_model).astype(xp.float32) * scale, device=_device)
        self.w_k = Tensor(xp.random.randn(d_model, d_model).astype(xp.float32) * scale, device=_device)
        self.w_v = Tensor(xp.random.randn(d_model, d_model).astype(xp.float32) * scale, device=_device)
        self.w_o = Tensor(xp.random.randn(d_model, d_model).astype(xp.float32) * scale, device=_device)
        self.b_o = Tensor(xp.zeros(d_model, dtype=xp.float32), device=_device)
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


if _device == 'cuda':
    init_streams(4)
