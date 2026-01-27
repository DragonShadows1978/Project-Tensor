# Implementation Plan: tensor_gpu_v2.py Phase 4 - Production Hardening

## Mission Overview

Add production-ready features to tensor_gpu_v2.py for robust training workflows:
1. Dynamic Loss Scaling for Mixed Precision
2. Weight Tying / Parameter Sharing
3. Persistent Kernel Cache
4. Gradient Clipping Utilities
5. Model Checkpointing
6. Profiling Integration

**Target File**: `/home/vader/mini-mind-v2/workspace/OPAL/tensor_gpu_v2.py`

---

## Step 1: Gradient Clipping Utilities

### Description
Add standalone gradient clipping functions for use with any optimizer.

### Implementation
Add after the optimizers section (~line 2662):

```python
# ==================== GRADIENT CLIPPING ====================

def clip_grad_norm_(parameters: List[Tensor], max_norm: float,
                    norm_type: float = 2.0) -> float:
    """
    Clip gradients by global norm (in-place).

    Args:
        parameters: Iterable of Tensors with gradients
        max_norm: Maximum allowed gradient norm
        norm_type: Type of p-norm (2.0 for L2, inf for max)

    Returns:
        Total gradient norm before clipping
    """
    params = [p for p in parameters if p.grad is not None]
    if len(params) == 0:
        return 0.0

    xp = cp if params[0]._device == 'cuda' else np

    if norm_type == float('inf'):
        norms = [float(xp.abs(p.grad).max()) for p in params]
        total_norm = max(norms) if norms else 0.0
    else:
        total_norm = 0.0
        for p in params:
            param_norm = float((xp.abs(p.grad) ** norm_type).sum())
            total_norm += param_norm
        total_norm = total_norm ** (1.0 / norm_type)

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
```

### Tests
- Test with known gradient values and verify clipping
- Test with inf norm type
- Test with empty parameters list

---

## Step 2: Enhanced GradScaler (Dynamic Loss Scaling)

### Description
Enhance existing GradScaler with state_dict support, bounds, and statistics tracking.

### Implementation
Modify existing GradScaler class (lines 2734-2781):

```python
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
            init_scale: Initial loss scale value
            growth_factor: Factor to multiply scale by on successful steps
            backoff_factor: Factor to multiply scale by on overflow
            growth_interval: Steps between scale growth attempts
            min_scale: Minimum allowed scale (prevents scale collapse)
            max_scale: Maximum allowed scale (prevents overflow)
            enabled: If False, scaler is a no-op (for BF16 training)
        """
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

    def unscale_(self, optimizer):
        """Unscale gradients. Check for inf/nan."""
        if not self.enabled:
            return

        xp = cp if optimizer._device == 'cuda' else np
        inv_scale = 1.0 / self._scale

        self._found_inf = False
        for p in optimizer.params:
            if p.grad is not None:
                p.grad = p.grad * inv_scale
                if xp.any(xp.isinf(p.grad)) or xp.any(xp.isnan(p.grad)):
                    self._found_inf = True

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
        """Load state from checkpoint."""
        self._scale = state.get('scale', self._scale)
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
```

### Tests
- Test overflow detection and scale reduction
- Test scale growth after stable steps
- Test state_dict round-trip
- Test enabled=False pass-through

---

## Step 3: Optimizer state_dict Methods

### Description
Add state_dict/load_state_dict to optimizers for checkpointing support.

### Implementation
Add to SGD, Adam, AdamW, RMSprop classes:

```python
# In SGD class:
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
    if 'velocities' in state:
        self.velocities = [xp.array(v, dtype=xp.float32) for v in state['velocities']]


# In Adam class:
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
    if 'm' in state:
        self.m = [xp.array(m, dtype=xp.float32) for m in state['m']]
    if 'v' in state:
        self.v = [xp.array(v, dtype=xp.float32) for v in state['v']]
    if self.amsgrad and 'v_max' in state:
        self.v_max = [xp.array(vm, dtype=xp.float32) for vm in state['v_max']]
```

---

## Step 4: Model Checkpointing Functions

### Description
Add high-level save/load functions for complete training state.

### Implementation
Add new section after gradient checkpointing:

```python
# ==================== MODEL CHECKPOINTING ====================

import pickle
import os

CHECKPOINT_VERSION = '1.0'


def save_checkpoint(path: str, model: Module, optimizer=None, scaler=None,
                   scheduler=None, epoch: int = 0, global_step: int = 0,
                   extra_state: dict = None):
    """
    Save complete training checkpoint.

    Args:
        path: File path to save checkpoint
        model: Model to save
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
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # Save as pickle for simplicity
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(path: str, model: Module, optimizer=None, scaler=None,
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
```

---

## Step 5: Weight Tying

### Description
Add parameter sharing functionality for transformer-style weight tying.

### Implementation
Add new section:

```python
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

    Example:
        # In a language model
        embed = Embedding(vocab_size, embed_dim)
        lm_head = Linear(embed_dim, vocab_size)
        weight_tie(embed.weight, lm_head, 'w', transpose=True)
    """
    # Create tied weight wrapper
    tied = TiedWeight(source, transpose)

    # Replace target's data pointer
    target_param = getattr(target_module, target_attr)

    # Store original for gradient accumulation
    target_param._tied_source = source
    target_param._tied_transpose = transpose

    # Make data point to source
    if transpose:
        target_param.data = source.data.T
    else:
        target_param.data = source.data


def sync_tied_gradients(model: Module):
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
```

---

## Step 6: Profiling Integration

### Description
Add profiling context manager for GPU performance analysis.

### Implementation
Add new section:

```python
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
        return (f"[{r['name']}] Time: {r['elapsed_ms']:.2f}ms | "
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
```

---

## Step 7: Persistent Kernel Cache (Lower Priority)

### Description
Add file-based caching for compiled RawKernels to reduce cold-start time.

### Implementation
Add at the beginning of the file, after imports:

```python
import hashlib
import os

# Kernel cache directory
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
    """
    cache_key = _get_kernel_cache_key(code, name)

    # Check in-memory cache first
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    # Check file cache
    os.makedirs(KERNEL_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(KERNEL_CACHE_DIR, f"{name}_{cache_key}.bin")

    # File caching is complex with RawKernel, so just compile
    # CuPy handles its own caching in ~/.cupy/kernel_cache
    kernel = cp.RawKernel(code, func_name)

    # Store in memory cache
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
```

Then update kernel definitions to use the cache wrapper (optional optimization).

---

## Success Criteria

1. **Dynamic Loss Scaling**
   - [ ] Prevents overflow in FP16 training
   - [ ] state_dict() / load_state_dict() work correctly
   - [ ] Scale adjusts dynamically based on overflow frequency
   - [ ] Statistics tracking works

2. **Weight Tying**
   - [ ] Embedding-to-projection weight sharing works
   - [ ] Gradients accumulate correctly to shared weight
   - [ ] Memory usage reduced by sharing

3. **Kernel Cache**
   - [ ] In-memory caching reduces repeated compilation
   - [ ] Cache info function returns correct data

4. **Gradient Clipping**
   - [ ] clip_grad_norm_() clips to specified norm
   - [ ] clip_grad_value_() clips to specified value
   - [ ] Returns original norm for logging

5. **Checkpointing**
   - [ ] save_checkpoint() saves all training state
   - [ ] load_checkpoint() restores state correctly
   - [ ] Training can resume from checkpoint

6. **Profiling**
   - [ ] Profiler context manager measures GPU time
   - [ ] Memory tracking works correctly
   - [ ] benchmark() function provides reliable timings

---

## Files Modified

- `/home/vader/mini-mind-v2/workspace/OPAL/tensor_gpu_v2.py` - All changes

---

## Implementation Order

1. Gradient Clipping (simple, standalone)
2. Enhanced GradScaler (build on existing)
3. Optimizer state_dict methods (needed for checkpointing)
4. Checkpointing functions
5. Weight Tying
6. Profiling
7. Kernel Cache (if time permits)

---

## Test Plan

After implementation, create functional tests that:
1. Train a small model with FP16 and verify no overflow crashes
2. Save checkpoint, reload, and verify training continues correctly
3. Create model with weight tying and verify parameter count reduction
4. Profile a forward pass and verify timing is reported
5. Test gradient clipping with known gradients
