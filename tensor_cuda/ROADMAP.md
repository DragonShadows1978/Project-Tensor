# Full C++/CUDA rework — roadmap

Goal: port the entire `tensor_gpu_v2` library (CuPy + Python autograd) to a
**standalone C++/CUDA engine** with a thin Python wrapper — no PyTorch, no CuPy.
This is a large effort (~15–25k lines of C++/CUDA), built in coherent vertical
slices so each phase is buildable and testable on its own.

Tracking the source inventory: ~120 Tensor ops, ~50 nn modules, ~25
optimizers/schedulers, RNN/LSTM/GRU, transformer layers, RoPE/ALiBi, TurboQuant
+ APA attention, plus training infrastructure.

## Conventions (followed by every phase)

- **NDArray** (`core.h`) is the raw, autograd-free CUDA array every kernel acts
  on. **Tensor/Variable** (`autograd.h`) adds the autograd graph.
- Each differentiable op lives in `ops.cpp` as: build forward NDArray →
  `Tensor::from_op(out, inputs, name, grad_fn)`. The `grad_fn` captures **input
  Tensors** and **saved-forward NDArrays** only — never the output (keeps the
  graph an acyclic shared_ptr DAG).
- Kernels are dtype-generic via the `DISPATCH_FLOAT` macro (compute in fp32,
  store in the array dtype) so fp32 + fp16 share one path.
- Every phase ships kernels + bindings + a numerical test (finite-difference
  gradcheck for exact-grad ops; reference parity for approximate ones like APA).

## Status

### ✅ Phase 1 — Core engine (this commit)
- Device memory (`Storage`), `NDArray` (contiguous), dtype (fp32/fp16) + device.
- Reverse-mode autograd engine (iterative topo-sort, cycle-free closures).
- Kernels: broadcasting elementwise (add/sub/mul/div + scalar), unary
  (neg/exp/log/sqrt/relu/sigmoid/tanh/gelu/silu/recip/abs/sign), reductions
  (sum/max/`reduce_to`), permute/transpose, `ge` compare; cuBLAS batched matmul.
- Ops with exact grads: add/sub/mul/div, matmul, exp/log/sqrt, relu/sigmoid/
  tanh/gelu/silu, sum/mean, reshape/transpose, softmax, mse_loss.
- pybind11 bindings + NumPy-friendly `tensor_cuda` package; CMake build.
- Tests: forward, broadcasting, batched matmul, softmax, FD gradcheck, a
  converging linear-regression training loop.

### ✅ Phase 2 — Op surface (core breadth)
- pow, comparisons (gt/ge/lt/le/eq/ne, tensor + scalar), `where`, `masked_fill`.
- Reductions with grad: max/min along axis (tie-split subgradient), var, std.
- Shape: permute, transpose, squeeze, unsqueeze, expand/broadcast_to, flatten,
  cat, stack (+ grads via slice).
- log_softmax, cross_entropy (one-hot; Python wrapper accepts int labels).
- New kernels: pow, compare (bcast + scalar), 3-way `where`, reduce_min, cat/slice.
- **Deferred to Phase 2b** (added alongside the consumers that need them):
  gather/scatter/index_select (with Embedding), advanced `__getitem__`/setitem,
  einsum, sort/topk/argmax-as-index, cumsum/cumprod/prod, flip/roll/repeat/tile,
  trig/clamp/round family, strided (zero-copy) views.

### ⬜ Phase 3 — nn.Module system + core layers
`Module` base (parameters/buffers/state_dict/hooks), `Sequential`/`ModuleList`/
`ModuleDict`. Layers: Linear, Embedding, Dropout, LayerNorm/RMSNorm/GroupNorm/
BatchNorm{1,2}D, Conv1D/Conv2D (+ depthwise/separable/transpose, im2col kernels),
pooling (Max/Avg/Adaptive), losses (MSE/CE/BCE/L1/SmoothL1/KLDiv/…), activations
as modules. Parameter registration drives autograd leaves.

### ⬜ Phase 4 — Optimizers, schedulers, training infra
SGD/Adam/AdamW/RMSprop/Adagrad/RAdam/Lion/FusedAdam (in-place CUDA update
kernels), LR schedulers (Step/MultiStep/Exp/Cosine/OneCycle/Cyclic/Plateau/
warmup), grad clipping, GradScaler/autocast (AMP), gradient accumulation,
gradient checkpointing, weight tying, checkpoint save/load, profiler.

### ⬜ Phase 5 — Sequence & transformer models
RNN/LSTM/GRU (cells + layers), PositionalEncoding, RoPE, ALiBi,
scaled_dot_product_attention, flash_attention (online-softmax tiled kernel),
multi-head attention, TransformerEncoder/DecoderLayer, GeGLU, fused
Linear+GELU/SiLU.

### ⬜ Phase 6 — Quantization & APA
TurboQuant MSE/Prod (Lloyd-Max codebook builder + quantize/dequantize kernels),
`apa_quant_attention` (port from `apa_cuda/`, but on this engine's autograd
instead of a PyTorch extension). At this point the standalone library is a
self-contained, framework-free home for APA — fulfilling the original goal with
no PyTorch dependency at all.

## Build & test (each phase)
```bash
cd tensor_cuda && ./build.sh 86          # RTX 3070
PYTHONPATH=. python -m pytest tests -v
```
