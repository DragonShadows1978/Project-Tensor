# Project Tensor

A from-scratch GPU tensor engine: a native CUDA C++ extension
(`tensor_cuda/`, CMake + nvcc, Release by default) with a full autograd
Python layer. No PyTorch, no TensorFlow in any runtime path. It serves
2B–20B-parameter LLMs interactively on consumer cards (RTX 3070 8GB /
4070 SUPER 12GB) and is the substrate for the
[GraftRepository](https://github.com/DragonShadows1978/GraftRepository)
memory system, APA selective attention, the HY3D image→3D pipeline, and
the ColdCast/Scorch rendering stack.

Discipline: every program in `docs/` is a plan (immutable) + ledger
(receipts); claims below name their receipt. What failed is listed next
to what worked.

## WHAT RUNS ON IT (receipts in docs/ here and in GraftRepository/docs/)

| Result | Number |
|---|---|
| MiniCPM3-4B (MLA, INT4) | decode 675 → **21.6 ms/token** (31×, fast stack: no_grad + pool + absorbed decode + fused GEMV/norm), parity-gated |
| Qwen3-4B / Qwen3.5-9B (GQA) | 9B at **25 tok/s**; port gates: logit parity, bit-identical state restore, APA zero-flip |
| Gemma-QAT 12B | exact q4_0 via symmetric-8 kernels, 31 tok/s; full 32K both phases on 12GB |
| GPT-OSS-20B (MoE) | resident MXFP4 experts + sink-APA; **96k context PASSES on a 12GB card** (128k OOMs — measured ceiling); four-way stack: MXFP4 + iSWA + INT8 KV + APA-selective |
| Trinity Nano (NoPE) | INT8 group-32 resident + fp32 compute, 8.8GiB with arena; NoPE dissolves the graft position-hole law |
| APA (selective attention) | r0.15 engaged; GQA selection rule externally validated (26B kv=2 on a collaborator's 32GB card: 64K alive where Flash OOMs); first non-causal site (DiT) E2E-equivalent at INT4/r0.15 (docs/APA_PAPER_DRAFT.md) |
| Quantized linears | native INT2/INT3/INT4/INT6/INT8 + MXFP4 experts + W8A16 group-32 fused tile-dequant (paint: 0.98–1.34× fp16, VRAM contract intact) |
| INT4 backprop | int4_linear VJP + backward grad-wave freeing — a 62-layer INT4 reader trains its gradient path on 8GB |
| GRM support ops | arena cache surgery, splice/inject hooks, CUDA route banks (GQA bridge 1.26–1.44× direct; MLA 1M-node route 2.22 ms — receipts in GraftRepository) |
| HY3D-TC (image→3D) | E2E functional: 14/14 watertight; UniPC bit-exact vs diffusers; software rasterizer pixel-exact vs the CUDA oracle; texture paint (F2) complete |
| Scorch/ColdCast ops | kernel renderer, dda_raycast, voxel pipeline primitives |
| Kernel-opt program | closed with receipts: fused GEMV decode path + wins ledgered in docs/KERNEL_OPT_* (Phase 5 negative — see below) |

## WHAT DOESN'T WORK / CLOSED NEGATIVE (same receipt discipline)

- **APA is a multi-KV-head thing** — MQA fails it structurally (coherent
  noise; KV slack pre-spent; D=512 blocks the fused kernel). Measured on
  the Gemma port; scope law: "saving memory on storage is not APA."
- **APA net cost where it doesn't select well**: on the 12B, +peak
  memory, 3.5× prefill, ppl +1.5–1.9% engaged — the selection rule
  (kv_heads≥2, bounded head_dim, long context) decides where it pays.
- **Adaptive Ladder Attention (ALA): works, doesn't pay** — six-version
  arc, correct per-site ladder policy; Amdahl ceiling 1.357× because the
  skeleton, not attention, is the wall. Closed.
- **CUDA-graph decode capture: parked** — +3% at gate; not worth the
  complexity (branch `kernel-opt-phase2-parked`).
- **fp16 threshold-ladder overflow law** — 76% of rows refine nothing in
  fp16; fp32 is NOT parity with fp16 and must re-gate separately.
- **Trinity INT4: dead** — group-size defect was fixed and INT4 stayed
  dead; INT8 group-32 is the operating mode.
- **128k on 12GB: OOM** — the GPT-OSS context ceiling sits in the
  96k–128k band; receipts bracket it.
- Engine gotchas with receipts: forgetting `no_grad` costs the 31×
  decode win; `_NP_DTYPE` silent-downcast bug class; Tensor compare ops
  take tensors, not scalars; a production `.so` shipped `-O0` for weeks
  (CMake now defaults Release — check your build flags).

## ENGINE MEASUREMENT LAWS (enforced by every gate here and downstream)

- **First-run effect**: the first forward of a process differs ≤0.5
  logit from all subsequent runs; warm runs are bit-identical — every
  same-process A/B warms up before capturing side A.
- **Matched-reference law**: compare against a reference produced under
  the same numerics (fp16 goldens are noise draws — gate by
  torch-own-spread vs an fp32 reference).
- **Teacher-force any equivalence comparison** past the first greedy
  divergence.
- **E2E-is-arbiter** for generative pipelines (DiT/HY3D): kernel-level
  deviation is judged by end-to-end output equivalence, not tensor
  diffs alone.

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

## APA-Quant attention — C++/CUDA (drop-in for PyTorch)

The CuPy reference `apa_quant_attention` is great for research, but its per-call
Python overhead makes it unfair to benchmark against PyTorch's compiled
attention. [`apa_cuda/`](apa_cuda/) ports it to a PyTorch C++/CUDA extension —
a drop-in replacement for `torch.nn.functional.scaled_dot_product_attention`
with hand-written CUDA kernels (fused quantize+gather and refinement score-mix)
plus cuBLAS GEMMs, forward + backward, fp16/fp32. See
[`apa_cuda/README.md`](apa_cuda/README.md).

```python
from apa_attention import apa_scaled_dot_product_attention as apa_sdpa
out = apa_sdpa(q, k, v, is_causal=True)   # drop-in for F.scaled_dot_product_attention
```

## Full C++/CUDA rework — `tensor_cuda/`

The complete framework-free rework of this library into a **standalone C++/CUDA
engine with a thin Python wrapper** (no PyTorch, no CuPy) lives in
[`tensor_cuda/`](tensor_cuda/). It has its own `Storage`→`NDArray`→`Tensor`
stack, a reverse-mode autograd engine, hand-written CUDA kernels (+ cuBLAS for
GEMMs), and a NumPy-friendly API:

```python
import tensor_cuda as tc
from tensor_cuda import nn, optim

model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))
opt = optim.Adam(model.parameters(), lr=1e-2)
loss = tc.cross_entropy(model(tc.tensor(X)), labels)
opt.zero_grad(); loss.backward(); opt.step()

out = tc.apa_quant_attention(q, k, v, bulk_bits=2, refine_percentile=0.15)  # APA, framework-free
```

Built in phases (see [`tensor_cuda/ROADMAP.md`](tensor_cuda/ROADMAP.md)):
autograd engine → op surface → nn.Module + layers → optimizers/schedulers →
attention/transformer → TurboQuant + APA → Conv/pool/BatchNorm, RNN/LSTM/GRU,
RoPE, AMP, checkpointing, indexing. Build with `cd tensor_cuda && ./build.sh 86`
(RTX 3070) and `PYTHONPATH=. python -m pytest tests`.

## Philosophy

This library exists to prove that you don't need massive frameworks to do deep learning. Focused, readable code can train real models on real GPUs.

No abstraction layers. No plugin systems. No enterprise patterns. Just tensors, gradients, and CUDA.

## License

Copyright (C) 2026 David Perry.

This repository is licensed under the GNU Affero General Public License
v3.0 — see [LICENSE](LICENSE). Any software derived from this code,
including software served over a network, must be released under the same
terms. **Commercial licensing outside the AGPL terms is available** —
contact `dave@ai-storyforge.com`.

The associated research papers are licensed CC BY 4.0 via their Zenodo
records.
