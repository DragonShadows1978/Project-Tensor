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

### ✅ Phase 3 — nn.Module system + core layers (in progress → base done)
- `Module` base (param/buffer/submodule registration, parameters(), train/eval,
  zero_grad), `Sequential`. Layers: Linear, Embedding (+ scatter-add backward
  kernel), LayerNorm, Dropout, activation modules (ReLU/GELU/SiLU/Sigmoid/Tanh),
  losses (MSELoss, CrossEntropyLoss).
- Optimizers (folded in so the stack trains): SGD (+momentum/wd), Adam, AdamW —
  in-place CUDA step kernels.
- Tests: layer shapes, LayerNorm normalization, embedding grad, and full
  MLP classification + linear-regression training loops.
- **Remaining for Phase 3b**: RMSNorm/GroupNorm/BatchNorm{1,2}D, Conv1D/Conv2D
  (+ depthwise/separable/transpose, im2col kernels), pooling, more losses
  (BCE/L1/SmoothL1/KLDiv), ModuleList/ModuleDict, state_dict, hooks.

### ✅ Phase 4 — Optimizers, schedulers, grad clipping (core done)
- Optimizers: SGD/Adam/AdamW (Phase 3) + RMSprop, Adagrad (in-place CUDA step
  kernels). `clip_grad_norm_`. Schedulers: StepLR, CosineAnnealingLR,
  LinearWarmupCosineDecay.
- **Remaining for Phase 4b**: RAdam/Lion, MultiStep/Exp/OneCycle/Cyclic/Plateau,
  GradScaler + autocast (AMP), gradient accumulation/checkpointing, weight tying,
  checkpoint save/load, profiler.

### ✅ Phase 5 — Attention & transformer (core done)
- `functional.scaled_dot_product_attention` (causal + additive mask),
  `nn.MultiheadAttention`, `nn.RMSNorm`, `nn.TransformerEncoderLayer`. `abs` op.
- Tests: SDPA parity vs NumPy (dense + causal), MHA/encoder shapes, transformer
  training loop.
### ✅ Phase 5b — RNN family + RoPE + losses
- RNNCell/LSTMCell/GRUCell + RNN/LSTM/GRU (time-stepped), RoPE
  (`functional.rope_tables`/`apply_rotary`), losses L1/BCEWithLogits/SmoothL1.
- New ops (Phase 2b folded in): slice/narrow (+ grad via pad_into), sin, cos,
  reciprocal, clamp, elementwise maximum/minimum.
- **Still remaining**: ALiBi, fused flash-attention kernel, TransformerDecoder,
  GeGLU, fused Linear+GELU/SiLU; Conv/pooling (3b); AMP (4b); einsum/sort/topk/
  gather/scatter/cumsum (2b); strided views.

### ✅ Phase 6 — Quantization & APA (on the standalone engine)
- Lloyd-Max codebook builder + per-head rotations (numpy, cached);
  `apa_quantize_gather` CUDA kernel (searchsorted + codebook gather).
- `tc.apa_quant_attention` composed from engine ops: per-head quantized "bulk"
  scores + full-precision "ranking", z-score refinement, mixed softmax·V. The
  quantized-key path is detached, so exact autograd gives the same grad-routing
  as the reference (full key on refined positions, quantized key on bulk).
- Tests: forward parity vs a NumPy reference (dense, causal, multi-bit,
  full-precision), and finite backward.
- **This fulfills the original goal**: APA now runs on a self-contained,
  framework-free C++/CUDA stack with no PyTorch dependency.
- **Remaining for Phase 6b**: TurboQuantProd (inner-product estimator),
  exact top-k refinement path (needs sort/topk from Phase 2b), adaptive
  per-head budget, tiled/flash APA kernel, the reference's approximate backward
  as an opt-in.

## Cross-cutting remaining work (2b/3b/4b/5b)
Strided views; full indexing/einsum/sort/topk/cumsum/trig; Conv/RNN/pooling;
AMP/checkpointing; RoPE/ALiBi; fused flash kernels. Tracked above per phase.

## Build & test (each phase)
```bash
cd tensor_cuda && ./build.sh 86          # RTX 3070
PYTHONPATH=. python -m pytest tests -v
```
