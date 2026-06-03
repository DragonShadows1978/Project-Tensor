# Project Tensor — standalone C++/CUDA engine

A from-scratch GPU tensor library with reverse-mode autograd, written in
**C++/CUDA with a thin Python wrapper** — **no PyTorch, no CuPy**. This is the
full rework of `tensor_gpu_v2` (which was CuPy + Python autograd) into raw CUDA.

> **Status: Phase 1 — the core engine.** Tensor, autograd, the essential
> kernels + cuBLAS matmul, and a trainable end-to-end loop are done. The
> remaining layers (full op surface, nn modules, optimizers, transformers,
> APA/TurboQuant) are ported in subsequent phases — see [ROADMAP.md](ROADMAP.md).

## Why standalone (not on libtorch)

The original library's whole point is "no massive frameworks — just tensors,
gradients, and CUDA." This rework keeps that: our own `Storage` → `NDArray` →
`Tensor` stack, our own autograd graph, hand-written CUDA kernels, and cuBLAS
only for GEMMs. Benchmarks against PyTorch are then genuinely independent.

## Build

Requires a CUDA toolkit (`nvcc`), CMake ≥ 3.18, and Python with NumPy.

```bash
cd tensor_cuda
./build.sh 86          # RTX 3070 = sm_86; or: cmake -B build && cmake --build build -j
PYTHONPATH=. python -m pytest tests -v
```

`build.sh` drops `_tensor_cuda.*.so` into the `tensor_cuda/` package, so
`PYTHONPATH=. python -c "import tensor_cuda"` works from this directory.

## Usage

```python
import numpy as np
import tensor_cuda as tc

x = tc.tensor(np.random.randn(64, 8), requires_grad=False)
W = tc.tensor(np.random.randn(8, 1) * 0.01, requires_grad=True)
y = tc.tensor(np.random.randn(64, 1))

for step in range(50):
    pred = tc.matmul(x, W)
    loss = tc.mse_loss(pred, y)
    W.zero_grad()
    loss.backward()
    W = tc.tensor(W.numpy() - 0.1 * W.grad.numpy(), requires_grad=True)  # SGD
    print(step, float(loss.numpy()))
```

(A proper `Module`/optimizer API replaces the manual SGD step in Phase 3–4.)

## Architecture

| Layer | File | Role |
|---|---|---|
| Memory + raw array | `include/tc/core.h`, `src/kernels.cu` | `Storage`, `NDArray`, all elementwise/reduction/shape kernels |
| Matmul | `src/matmul.cu` | cuBLAS batched GEMM (fp32/fp16) |
| Autograd | `include/tc/autograd.h`, `src/autograd.cpp` | `Tensor`/`Variable`, cycle-free reverse-mode engine |
| Ops | `include/tc/ops.h`, `src/ops.cpp` | differentiable ops + grad_fns |
| Bindings | `src/bindings.cpp`, `tensor_cuda/__init__.py` | pybind11 module + NumPy ergonomics |

See [ROADMAP.md](ROADMAP.md) for the full phase plan and the conventions every
new op/kernel/module follows.
