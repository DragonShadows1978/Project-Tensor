# Project Tensor

A GPU-accelerated autograd tensor library built on CuPy. No PyTorch, no TensorFlow - just raw CUDA operations with automatic differentiation.

## What is this?

A from-scratch implementation of a deep learning tensor library with:
- Automatic differentiation (autograd)
- GPU acceleration via CuPy
- Modern transformer optimizations
- Production-ready features

Built across 4 development cycles, evolving from 826 lines (v1) to 3,835 lines (v2).

## Installation

### Requirements

- Python 3.9+
- CUDA toolkit 11.x or 12.x
- CuPy matching your CUDA version

### From source (recommended for development)

```bash
git clone <repo-url>
cd Project-Tensor

# Pick the CuPy variant that matches your CUDA toolkit:
pip install -e ".[cuda11]"   # CUDA 11.x
pip install -e ".[cuda12]"   # CUDA 12.x
```

The `-e` flag installs in **editable mode** — changes to the source files
take effect immediately without reinstalling.

### Production install

```bash
pip install ".[cuda12]"   # no -e flag; installs a fixed snapshot
```

### Manual CuPy install (if you need a specific CUDA sub-version)

```bash
pip install .                        # installs without cupy
pip install cupy-cuda12x             # or cupy-cuda11x, cupy-cuda120, etc.
```

See the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)
for the full list of available wheels.

### Verify the install

```python
import tensor_gpu_v2 as tg
print(tg.get_device())   # 'cuda'
x = tg.Tensor.randn(4, 4, device='cuda')
print(x)
```

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

### Mixed precision (AMP)

```python
from tensor_gpu_v2 import autocast, GradScaler

scaler = GradScaler()

with autocast():                 # tensors cast to float16 automatically
    logits = model(x)
    loss = criterion(logits, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Gradient control

```python
import tensor_gpu_v2 as tg

# Disable gradient tracking for inference
with tg.no_grad():
    predictions = model(x)

# As a decorator
@tg.no_grad()
def evaluate(model, loader):
    ...

# Re-enable inside a no_grad block
with tg.no_grad():
    features = backbone(x)
    with tg.enable_grad():
        loss = head(features)   # gradients back on just here
```

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
- SGD, Adam, AdamW, Adagrad, RMSProp optimizers
- Learning rate schedulers
- Gradient clipping
- Gradient accumulation
- **Dynamic Loss Scaling** for mixed precision (`GradScaler`)
- `no_grad` / `enable_grad` / `autocast` context managers + decorators
- Model checkpointing (`save_checkpoint` / `load_checkpoint`)
- Profiling integration

### Memory Optimizations
- NHWC convolution layout
- Grouped convolutions (1.21x speedup)
- Persistent kernel cache
- Half-precision (FP16) support

### TurboQuant
- `TurboQuantMSE` for unit-sphere vector quantization with stored vector norms
- `TurboQuantProd` for inner-product / attention-score approximation
- Lloyd-Max scalar codebook construction for beta-distributed sphere coordinates
- CPU/CuPy-compatible APIs that accept NumPy arrays, CuPy arrays, or `Tensor`

```python
import tensor_gpu_v2 as tg

quantizer = tg.TurboQuantMSE(dimension=128, bits=4, seed=2026)
encoding = quantizer.quantize(vectors)
reconstructed = quantizer.dequantize(encoding)

prod = tg.TurboQuantProd(dimension=128, bits=4, seed=2026)
key_encoding = prod.quantize(keys)
approx_scores = prod.attention_score(query, key_encoding)
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
| 5 | `no_grad`/`enable_grad`/`autocast` context managers, pip-installable package |
| 6 | TurboQuant MSE/product quantizers, validation harness, CPU-only import guard |

## Philosophy

This library exists to prove that you don't need massive frameworks to do deep learning. Focused, readable code can train real models on real GPUs.

No abstraction layers. No plugin systems. No enterprise patterns. Just tensors, gradients, and CUDA.

## License

MIT
