# Legacy `tensor_gpu_v2` CuPy API

> **Historical document.** This page preserves the installation, usage, and
> feature material for the original Python-autograd-on-CuPy implementation.
> It is not the current TensorCUDA install path. New users should build and
> import the [native `tensor_cuda` engine](../README.md#build-and-import-the-native-engine).

The repository-root `pyproject.toml` still describes this legacy package as
`tensor-gpu` and discovers `tensor_gpu_v2*`. Its optional dependencies select a
CuPy wheel for the installed CUDA toolkit.

## Historical installation

Requirements for this implementation, as enforced by the current project
metadata:

- Python 3.9+ for the packaged project metadata
- NumPy 1.21+
- CuPy matching CUDA 11.x or 12.x

The old README later repeated Python 3.8+; the inspected `pyproject.toml`
requires Python 3.9+, so 3.9+ is the inspect-verified requirement here.

Editable development install:

```bash
git clone <repo-url>
cd Project-Tensor
pip install -e ".[cuda11]"   # CUDA 11.x
pip install -e ".[cuda12]"   # CUDA 12.x
```

The `-e` flag installs the package in editable mode, so Python source changes
take effect without reinstalling.

Historical fixed-snapshot install:

```bash
pip install ".[cuda12]"
```

Manual dependency selection:

```bash
pip install .
pip install cupy-cuda12x      # or cupy-cuda11x / a matching wheel
```

Historical import check:

```python
import tensor_gpu_v2 as tg

print(tg.get_device())
x = tg.Tensor.randn(4, 4, device="cuda")
print(x)
```

## Historical tensor and training API

```python
from tensor_gpu_v2 import Adam, Linear, Tensor

x = Tensor.randn(32, 784, device="cuda")
linear1 = Linear(784, 256)
linear2 = Linear(256, 10)

h = linear1(x).gelu()
out = linear2(h)
loss = out.mean()
loss.backward()

optimizer = Adam(
    [linear1.w, linear1.b, linear2.w, linear2.b],
    lr=1e-3,
)
optimizer.step()
```

### Mixed precision

```python
from tensor_gpu_v2 import GradScaler, autocast

scaler = GradScaler()

with autocast():
    logits = model(x)
    loss = criterion(logits, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Gradient control

```python
import tensor_gpu_v2 as tg

with tg.no_grad():
    predictions = model(x)

@tg.no_grad()
def evaluate(model, loader):
    ...

with tg.no_grad():
    features = backbone(x)
    with tg.enable_grad():
        loss = head(features)
```

## Historical feature surface

### Core tensor operations

- Full autograd with backward pass
- Broadcasting, reshape, and transpose
- Matrix multiplication, `conv2d`, and `conv_transpose2d`
- Batch normalization and layer normalization
- Dropout and embedding layers

### Activations and transformer components

- GELU (recorded as 5.22x faster than the naive implementation)
- SiLU/Swish (recorded as 2.50x faster)
- ReLU, Tanh, Sigmoid, and Softmax
- FlashAttention (recorded as a 17.5x memory reduction)
- Multi-head attention
- Fused batch-normalization/ReLU (recorded as 21.33x faster)
- Einsum with caching and weight tying for embeddings

### Training utilities

- SGD, Adam, AdamW, Adagrad, and RMSProp optimizers
- Learning-rate schedulers
- Gradient clipping and accumulation
- Dynamic loss scaling through `GradScaler`
- `no_grad`, `enable_grad`, and `autocast` contexts and decorators
- Model checkpointing and profiling integration

### Memory work

- NHWC convolution layout
- Grouped convolution (recorded as 1.21x faster)
- Persistent kernel cache
- FP16 operation

## Historical TurboQuant API

The CuPy-era package exposed `TurboQuantMSE` for unit-sphere vector
quantization with stored norms and `TurboQuantProd` for approximate inner
products and attention scores. These APIs accepted NumPy arrays, CuPy arrays,
or legacy `Tensor` objects.

```python
import tensor_gpu_v2 as tg

quantizer = tg.TurboQuantMSE(dimension=128, bits=4, seed=2026)
encoding = quantizer.quantize(vectors)
reconstructed = quantizer.dequantize(encoding)

prod = tg.TurboQuantProd(dimension=128, bits=4, seed=2026)
key_encoding = prod.quantize(keys)
approx_scores = prod.attention_score(query, key_encoding)
```

## Historical development cycles

| Cycle | Recorded work |
|---|---|
| 1 | Base autograd, GELU 5.22x, SiLU 2.50x, and 14+ additional operations |
| 2 | Dropout `p=1.0`, `half()` dtype, and grouped-convolution 1.21x fixes |
| 3 | FlashAttention 17.5x memory, fused batch-normalization/ReLU 21.33x, NHWC convolution, and einsum caching |
| 4 | Dynamic loss scaling, weight tying, kernel cache, clipping, and checkpoints |
| 5 | `no_grad` / `enable_grad` / `autocast` contexts and package installation |
| 6 | TurboQuant quantizers, validation harness, and CPU-only import guard |

## Historical APA extension

The CuPy reference `apa_quant_attention` was useful for research, but its
per-call Python overhead made direct comparison with compiled attention
misleading. It was also ported into `apa_cuda/`, a PyTorch C++/CUDA extension
intended as a drop-in replacement for
`torch.nn.functional.scaled_dot_product_attention`. It used handwritten CUDA
kernels for fused quantize/gather and refinement score mixing, plus cuBLAS
GEMMs, with forward and backward paths in FP16 and FP32.

```python
from apa_attention import apa_scaled_dot_product_attention as apa_sdpa

out = apa_sdpa(q, k, v, is_causal=True)
```

See [`apa_cuda/README.md`](../apa_cuda/README.md) for that separate historical
extension. The current framework-independent implementation is the
[`tensor_cuda` package](../tensor_cuda/README.md).
