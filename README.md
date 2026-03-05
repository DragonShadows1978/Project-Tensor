# Project Tensor

A GPU-accelerated autograd tensor library built on CuPy. No PyTorch, no TensorFlow - just raw CUDA operations with automatic differentiation.

## What is this?

A from-scratch implementation of a deep learning tensor library with:
- Automatic differentiation (autograd)
- GPU acceleration via CuPy
- Modern transformer optimizations
- Production-ready features

Built across 4 development cycles, evolving from 826 lines (v1) to 3,835 lines (v2).

## Features

### Core Tensor Operations
- Full autograd with backward pass
- Broadcasting, reshape, transpose
- Matmul, conv2d, conv_transpose2d
- Batch normalization, layer normalization
- Dropout, embedding layers

### Activation Functions (Optimized)
- GELU (5.22x speedup over naive)
- SiLU/Swish (2.50x speedup)
- ReLU, Tanh, Sigmoid, Softmax

### Transformer Components
- **FlashAttention** - 17.5x memory reduction
- Multi-head attention
- **FusedBatchNormReLU** - 21.33x speedup
- Einsum with caching
- Weight tying for embeddings

### Training Utilities
- SGD, Adam, AdamW optimizers
- Learning rate schedulers
- Gradient clipping
- Gradient accumulation
- **Dynamic Loss Scaling** for mixed precision
- Model checkpointing
- Profiling integration

### Memory Optimizations
- NHWC convolution layout
- Grouped convolutions (1.21x speedup)
- Persistent kernel cache
- Half-precision (FP16) support

## Usage

```python
from tensor_gpu_v2 import Tensor, Linear, Conv2D, Adam

# Create tensors
x = Tensor.randn(32, 784, device='cuda')

# Build a simple network
linear1 = Linear(784, 256)
linear2 = Linear(256, 10)

# Forward pass
h = linear1(x).gelu()
out = linear2(h)

# Backward pass
loss = out.mean()
loss.backward()

# Optimize
optimizer = Adam([linear1.w, linear1.b, linear2.w, linear2.b], lr=1e-3)
optimizer.step()
```

## Requirements

- Python 3.8+
- CuPy (CUDA 11.x or 12.x)
- NumPy

```bash
pip install cupy-cuda11x numpy
```

## Development History

| Cycle | Features |
|-------|----------|
| 1 | Base autograd, optimized activations (GELU 5.22x, SiLU 2.50x), 14+ new ops |
| 2 | Dropout p=1.0 fix, half() dtype fix, grouped conv optimization (1.21x) |
| 3 | FlashAttention (17.5x memory), FusedBatchNormReLU (21.33x), NHWC Conv2D, einsum caching |
| 4 | Dynamic loss scaling, weight tying, kernel cache, gradient clipping, checkpointing |

## Philosophy

This library exists to prove that you don't need massive frameworks to do deep learning. A single file with 3,800 lines of focused code can train real models on real GPUs.

No abstraction layers. No plugin systems. No enterprise patterns. Just tensors, gradients, and CUDA.

## License

MIT
