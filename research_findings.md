# Research Findings: tensor_gpu_v2.py Phase 4 - Production Hardening

## Date: 2026-01-18 (Cycle 4)

---

## Cycle 4 Context

Previous cycles delivered:
- **Cycle 1**: tensor_gpu_v2.py with 5.22x GELU speedup, 2.50x SiLU speedup, 14+ new features
- **Cycle 2**: Fixed Dropout p=1.0, half() dtype bugs; optimized grouped conv to 1.21x
- **Cycle 3**: FlashAttention (17.5x memory reduction), FusedBatchNormReLU (21.33x speedup), NHWC Conv2D, einsum caching, gradient accumulation

Cycle 4 focuses on production hardening:
1. Dynamic Loss Scaling for Mixed Precision
2. Weight Tying / Parameter Sharing
3. Persistent Kernel Cache
4. Gradient Clipping Utilities
5. Model Checkpointing
6. Profiling Integration

---

## 1. Dynamic Loss Scaling

### Current State in tensor_gpu_v2.py

The file already has a basic `GradScaler` class (lines 2734-2781) with:
- Initial scale (default 65536.0)
- Growth factor (2.0) and backoff factor (0.5)
- Growth interval (2000 iterations)
- `scale()`, `unscale_()`, `step()`, `update()` methods

### Improvements Needed

Based on [PyTorch mixed precision best practices](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/):

1. **Add overflow tracking statistics** - Track total overflows, consecutive successes
2. **Add state_dict() / load_state_dict()** - For checkpointing support
3. **Add min/max scale bounds** - Prevent runaway scaling
4. **BF16 support detection** - Skip scaling for BF16 (same exponent range as FP32)

### Implementation Approach

```python
class DynamicLossScaler:
    """Enhanced dynamic loss scaling with statistics tracking."""

    def __init__(self, init_scale=65536.0, growth_factor=2.0,
                 backoff_factor=0.5, growth_interval=2000,
                 min_scale=1.0, max_scale=2**24):
        self._scale = init_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        # Statistics
        self._overflow_count = 0
        self._consecutive_successes = 0

    def state_dict(self) -> dict:
        return {
            'scale': self._scale,
            'growth_tracker': self._growth_tracker,
            'overflow_count': self._overflow_count,
        }
```

### Sources
- [PyTorch AMP Documentation](https://docs.pytorch.org/docs/stable/amp.html)
- [NVIDIA Mixed Precision Training Guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
- [Loss Scaling for FP16 Stability](https://apxml.com/courses/how-to-build-a-large-language-model/chapter-20-mixed-precision-training-techniques/loss-scaling-techniques)

---

## 2. Weight Tying / Parameter Sharing

### Concept

Weight tying shares parameters between layers - commonly the input embedding and output projection in transformers. Benefits:
- Reduces parameter count
- Acts as regularization
- Maintains or improves performance

From [Weight Tying Explained](https://paperswithcode.com/method/weight-tying):
> The same weight matrix is used in three places: the input embedding layer in the encoder, the input embedding layer in the decoder, and the final linear layer before the softmax.

### Implementation Design

1. **weight_tie() function**: Links two Tensor parameters
2. **Gradient accumulation**: Both locations contribute gradients to shared weight
3. **Use case**: `weight_tie(embedding.weight, linear.w.T)`

```python
def weight_tie(source: Tensor, target: Tensor, transpose: bool = False):
    """
    Tie weights between two parameters.

    The target will share data with source (or source.T if transpose=True).
    Gradients accumulate to the source parameter.
    """
    if transpose:
        # For embedding -> linear projection tying
        target.data = source.data.T
        target._shared_with = source
        target._transpose = True
    else:
        target.data = source.data
        target._shared_with = source
        target._transpose = False
```

### Key Consideration

The tricky part is backward pass - when target.grad is computed, it needs to be accumulated back to source.grad (with transpose if needed).

### Sources
- [MartinLwx's Blog on Weight Tying](https://martinlwx.github.io/en/an-explanation-of-weight-tying/)
- [Papers With Code - Weight Tying](https://paperswithcode.com/method/weight-tying)
- [Beyond Weight Tying Paper](https://arxiv.org/abs/1808.10681)

---

## 3. Persistent Kernel Cache

### CuPy's Built-in Caching

From [CuPy Performance Documentation](https://docs.cupy.dev/en/stable/user_guide/performance.html):
> CuPy caches the kernel code sent to GPU device within the process, which reduces the kernel compilation time on further calls. The compiled code is also cached in the directory `${HOME}/.cupy/kernel_cache`.

### Our RawKernel Challenge

Our custom RawKernels (GELU, SiLU, LayerNorm, FusedBatchNormReLU) are compiled at module import time. CuPy already caches them, but:
1. First compilation still takes time
2. Cache invalidation can cause re-compilation
3. No easy way to share compiled kernels across environments

### Implementation Approach

Create a wrapper that:
1. Computes SHA256 hash of kernel source code
2. Stores compiled module in `~/.cache/tensor_gpu_v2/kernels/`
3. Checks cache before compilation
4. Falls back to normal compilation if cache miss

```python
import hashlib
import os
import pickle

KERNEL_CACHE_DIR = os.path.expanduser('~/.cache/tensor_gpu_v2/kernels')

def get_cached_kernel(name: str, code: str) -> cp.RawKernel:
    """Get kernel from cache or compile and cache."""
    os.makedirs(KERNEL_CACHE_DIR, exist_ok=True)

    # Hash includes code, CUDA version, device capability
    cache_key = hashlib.sha256(
        f"{code}:{cp.cuda.runtime.runtimeGetVersion()}:{cp.cuda.Device().compute_capability}".encode()
    ).hexdigest()

    cache_path = os.path.join(KERNEL_CACHE_DIR, f"{name}_{cache_key[:16]}.pkl")

    if os.path.exists(cache_path):
        # Load from cache
        ...
```

### Important Note

RawKernel compilation is actually quite fast (milliseconds), but for production systems with many kernels, the cumulative startup time matters. The real benefit is **consistency** - ensuring the same compiled code is used across runs.

### Sources
- [CuPy Environment Variables](https://docs.cupy.dev/en/v12.3.0/reference/environment.html)
- [NVIDIA CUDA JIT Caching](https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/)
- [CuPy Kernel Compilation DeepWiki](https://deepwiki.com/cupy/cupy/3.3-kernel-compilation-and-execution)

---

## 4. Gradient Clipping Utilities

### Current State

The `Adam` class already has a `clip_grad_norm()` method (lines 2558-2573). Need to:
1. Make it standalone (not optimizer-specific)
2. Add `clip_grad_value_()` for value clipping
3. Return the original norm for logging

### PyTorch API Reference

From [torch.nn.utils.clip_grad_norm_](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html):
- `parameters`: Iterable of Tensors
- `max_norm`: Maximum gradient norm
- `norm_type`: Type of p-norm (default 2, can be 'inf')
- Returns: Total norm of parameters

### Implementation

```python
def clip_grad_norm_(parameters: List[Tensor], max_norm: float,
                    norm_type: float = 2.0) -> float:
    """
    Clip gradients by global norm.

    Returns:
        Total norm before clipping
    """
    xp = cp if parameters[0]._device == 'cuda' else np

    if norm_type == float('inf'):
        total_norm = max(p.grad.max() for p in parameters if p.grad is not None)
    else:
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                total_norm += float((xp.abs(p.grad) ** norm_type).sum())
        total_norm = total_norm ** (1.0 / norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad *= clip_coef

    return total_norm


def clip_grad_value_(parameters: List[Tensor], clip_value: float):
    """Clip gradients by value."""
    xp = cp if parameters[0]._device == 'cuda' else np
    for p in parameters:
        if p.grad is not None:
            p.grad = xp.clip(p.grad, -clip_value, clip_value)
```

### Best Practices
- Start with max_norm=1.0 as baseline
- Monitor gradient norms before clipping
- Layer-specific clipping for transformers/RNNs

### Sources
- [GeeksforGeeks - Gradient Clipping in PyTorch](https://www.geeksforgeeks.org/deep-learning/gradient-clipping-in-pytorch-methods-implementation-and-best-practices/)
- [Neptune AI - Understanding Gradient Clipping](https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem)
- [PyTorch clip_grad.py Source](https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/clip_grad.py)

---

## 5. Model Checkpointing

### What to Save

From [PyTorch Saving/Loading Tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html):
> When saving a general checkpoint for resuming training, you must save more than just the model's state_dict. Also save optimizer's state_dict, epoch, latest recorded training loss.

### Checkpoint Contents

1. **Model weights** - state_dict for all modules
2. **Optimizer state** - momentum buffers, adaptive learning rates
3. **Training state** - epoch, global step, best loss
4. **Loss scaling state** - for mixed precision
5. **Scheduler state** - learning rate schedule position
6. **Version info** - for compatibility checking

### Implementation Design

```python
def save_checkpoint(path: str, model: Module, optimizer=None,
                   scaler=None, epoch: int = 0,
                   extra_state: dict = None):
    """
    Save complete training checkpoint.
    """
    checkpoint = {
        'version': '1.0',
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    if extra_state:
        checkpoint['extra_state'] = extra_state

    # Save with numpy for portability
    np.savez_compressed(path, **{k: to_cpu(v) if isinstance(v, cp.ndarray) else v
                                  for k, v in checkpoint.items()})


def load_checkpoint(path: str, model: Module, optimizer=None,
                   scaler=None, strict: bool = True):
    """
    Load checkpoint and restore state.

    Returns:
        dict with 'epoch' and any 'extra_state'
    """
    data = np.load(path, allow_pickle=True)

    model.load_state_dict(data['model_state_dict'].item())

    if optimizer and 'optimizer_state_dict' in data:
        optimizer.load_state_dict(data['optimizer_state_dict'].item())

    return {
        'epoch': int(data.get('epoch', 0)),
        'extra_state': data.get('extra_state', {})
    }
```

### Format Choice

Use `numpy.savez_compressed` for:
- Cross-platform compatibility
- Smaller file size
- Easy inspection

### Sources
- [PyTorch Lightning Checkpointing](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html)
- [Medium - Save and Load Models](https://medium.com/biased-algorithms/how-to-save-and-load-models-in-pytorch-330b5573189c)
- [PyTorch Forums - Resume Training](https://discuss.pytorch.org/t/loading-checkpoint-for-resume-training-without-loading-optimizer-state-dict/131450)

---

## 6. Profiling Integration

### CuPy Profiling Tools

From [CuPy Performance Documentation](https://docs.cupy.dev/en/stable/user_guide/performance.html):
> CuPy provides `cupyx.profiler.benchmark()` for timing the elapsed time of a Python function on both CPU and GPU.

Key insight: CPU timing utilities don't know about async GPU execution, so naive `time.time()` doesn't measure actual GPU time.

### Implementation Approach

Create a `profile()` context manager that:
1. Synchronizes GPU before/after
2. Records kernel launch times
3. Tracks memory allocation
4. Generates human-readable report

```python
import cupyx.profiler

class ProfilerContext:
    """Context manager for profiling GPU operations."""

    def __init__(self, sync: bool = True):
        self.sync = sync
        self.start_mem = None
        self.end_mem = None
        self.events = []

    def __enter__(self):
        if self.sync:
            cp.cuda.Stream.null.synchronize()
        self.start_mem = _memory_pool.used_bytes()
        self.start_time = cp.cuda.Event()
        self.start_time.record()
        return self

    def __exit__(self, *args):
        self.end_time = cp.cuda.Event()
        self.end_time.record()
        if self.sync:
            cp.cuda.Stream.null.synchronize()
        self.end_mem = _memory_pool.used_bytes()

    def report(self) -> dict:
        return {
            'elapsed_ms': cp.cuda.get_elapsed_time(self.start_time, self.end_time),
            'memory_delta_mb': (self.end_mem - self.start_mem) / 1024**2,
            'peak_memory_mb': _memory_pool.total_bytes() / 1024**2,
        }


def profile():
    """Create a profiling context manager."""
    return ProfilerContext()
```

### Advanced: NVTX Markers

For integration with NVIDIA Nsight:
```python
import cupy.cuda.nvtx as nvtx

def profile_region(name: str):
    """Mark a region for NVTX profiling."""
    class NVTXRegion:
        def __enter__(self):
            nvtx.RangePush(name)
        def __exit__(self, *args):
            nvtx.RangePop()
    return NVTXRegion()
```

### Sources
- [CuPy Performance Best Practices](https://docs.cupy.dev/en/stable/user_guide/performance.html)
- [NVIDIA CUPTI](https://developer.nvidia.com/cupti)
- [GPU Profiling Survey](https://eunomia.dev/blog/2025/04/21/gpu-profiling-under-the-hood-an-implementation-focused-survey-of-modern-accelerator-tracing-tools/)

---

## Key Takeaways from KB Learnings

From past missions:
1. **Training Efficiency & Optimization**: Combined optimization stack (gradient checkpointing + mixed precision + gradient accumulation) achieves ~75% total memory reduction - this validates our focus on these features
2. **RTX 3070 constraints**: 8GB VRAM at the boundary of feasibility - dynamic loss scaling and memory-efficient features critical

---

## Risk Mitigations

| Feature | Risk | Mitigation |
|---------|------|------------|
| Dynamic Loss Scaling | Infinite loop of overflows | Add min_scale bound, log warnings |
| Weight Tying | Gradient accumulation bugs | Comprehensive tests with known gradients |
| Kernel Cache | Stale cache after code changes | Include code hash in cache key |
| Gradient Clipping | NaN gradients | Check for inf/nan before clipping |
| Checkpointing | Version incompatibility | Store version, add migration paths |
| Profiling | Overhead in production | Make sync configurable |

---

## Implementation Order

1. **Gradient Clipping** - Simple, standalone functions (low risk)
2. **Enhanced GradScaler** - Build on existing implementation
3. **Checkpointing** - Requires optimizer state_dict changes
4. **Weight Tying** - Requires careful backward pass handling
5. **Profiling** - Can be added independently
6. **Kernel Cache** - Performance optimization, lower priority
