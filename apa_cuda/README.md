# APA-Quant Attention — C++/CUDA extension

A drop-in replacement for `torch.nn.functional.scaled_dot_product_attention`
backed by C++/CUDA kernels, so **APA-Quant attention can be benchmarked
head-to-head with PyTorch attention without Python-interpreter overhead**.

This is a port of `tensor_gpu_v2._core.apa_quant_attention` (a CuPy + custom
autograd implementation) to a PyTorch C++ extension. The reference is great for
research but its per-call Python/CuPy work — per-head quantization loops,
`xp.einsum`/`xp.partition` round-trips, `scipy.stats.norm.ppf`, and the tiled
block loop — dominates wall-clock time on short/medium sequences and makes an
APA-vs-PyTorch comparison unfair. This extension removes that overhead.

## Why this is a fair comparison

| Concern | How it's handled |
|---|---|
| Python interpreter in the hot path | **Eliminated.** Quantization, refinement, tiling, adaptive budgeting, autograd control flow all run in C++. |
| GEMM quality | Both APA and PyTorch's *math* SDPA backend dispatch matmuls to **cuBLAS** via ATen. The benchmark measures the *algorithm*, not BLAS. |
| Precision | fp16 and fp32 (`AT_DISPATCH_FLOATING_TYPES_AND_HALF`), matching PyTorch's fast fp16 path. |
| Forward + backward | Full `torch.autograd.Function`; trains end-to-end. |

## Design

```
apa_cuda/
  csrc/
    apa_kernels.cu   # custom CUDA kernels (the APA-specific fused work)
    apa.cpp          # C++ orchestration: forward, backward, pybind module
  apa_attention/
    quant_tables.py  # Lloyd-Max codebook + per-head rotations (one-time CPU, cached)
    ops.py           # torch.autograd.Function + drop-in API (thin wrapper)
    __init__.py
  setup.py           # CUDAExtension build
  benchmark.py       # APA vs F.scaled_dot_product_attention
  tests/test_parity.py
```

**Custom CUDA kernels** (`apa_kernels.cu`) cover the two APA-specific fused ops
that were the bulk of the Python overhead:

1. `quantize_gather` — per-element binary search against the Lloyd-Max boundary
   table (`searchsorted`, side="left") followed by a codebook gather.
2. `mix_scores` — the fused refinement decision
   `out = |ranking| >= thr ? ranking : bulk`, emitting the boolean refine mask
   the backward pass consumes.

**cuBLAS GEMMs** (via ATen `matmul`) handle the rotations, `Q·Kq`, `Q·K`, `P·V`
and the backward matmuls — exactly the path PyTorch's math SDPA backend uses.

**C++ control flow** implements the full reference algorithm: the dense path,
the flash-style online-softmax tiled path (for score matrices > 256 MB), exact
top-k refinement (`S ≤ 256`), z-score-threshold refinement (`S > 256`), the
adaptive per-head budget allocation, and an in-C++ inverse-normal-CDF (Acklam's
approximation, ~1e-9 vs `scipy.stats.norm.ppf`).

## Build

Requires a CUDA toolkit (`nvcc`) and a CUDA-enabled PyTorch.

```bash
cd apa_cuda
TORCH_CUDA_ARCH_LIST="8.6" pip install -e .   # RTX 3070 = sm_86
```

## Usage

```python
import torch
from apa_attention import apa_scaled_dot_product_attention as apa_sdpa

q = torch.randn(4, 8, 1024, 64, device="cuda", dtype=torch.float16)
k = torch.randn_like(q); v = torch.randn_like(q)

# Drop-in for F.scaled_dot_product_attention(q, k, v, is_causal=True):
out = apa_sdpa(q, k, v, is_causal=True, refine_percentile=0.15, bulk_bits=2)

# Or the full reference signature:
from apa_attention import apa_quant_attention
out = apa_quant_attention(q, k, v, bulk_bits=2, refine_percentile=0.15,
                          adaptive_heads=True)
```

## Benchmark

```bash
python benchmark.py --seqlen 1024 --heads 8 --dim 64 --dtype fp16
python benchmark.py --seqlen 4096 --dtype fp16 --causal --backward
```

## Test

```bash
python tests/test_parity.py        # or: pytest tests/test_parity.py
```

Parity is checked against a self-contained NumPy port of the algorithm covering
the top-k, z-score, full-precision, multi-bit and adaptive-heads paths, for both
forward and backward.

> **Note on backward / `gradcheck`.** APA's backward is deliberately an
> *approximation*: gradients are computed from the full-precision softmax
> weights, not from the quantized/mixed scores used in the forward (the bulk
> path is treated as a straight-through estimator for `grad_q`). This matches
> the reference exactly, so the parity test compares against the reference
> backward — `torch.autograd.gradcheck` is **not** applicable.

## Numerical notes & limitations

- **fp16 quantization** is computed in fp32 internally then cast back, which is
  marginally more accurate than the reference's in-dtype path; expect tiny
  (~1e-3) fp16 differences.
- **Threshold ties.** Near the refinement threshold, the Acklam inverse-CDF can
  differ from SciPy by ~1e-9, which can flip a small number of mask bits in the
  z-score path. The effect on the output is negligible (well within fp16 noise).
- **Future work — full fusion.** GEMMs currently round-trip through global
  memory between kernels. A single fully-fused flash-attention-style kernel
  (scores, refinement and `P·V` in one pass, never materializing the score
  matrix) would cut memory traffic further; the current structure is organized
  so that fusion can be added incrementally without changing the Python API.
```
