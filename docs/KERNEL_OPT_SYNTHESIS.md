# Kernel Optimization Narrative Synthesis

The inv_f32d181e research set turned out to be three different things wearing
one label. A minority of it is directly usable: the llama.cpp MMQ extraction,
the CUTLASS design-pattern synthesis, the RoPE and softmax investigations that
actually read the live tree, and the launch-parameter and CUDA-graphs
clusters. A second slice did real analysis of the wrong codebase — the
Rust-port mission snapshot, not live tensor_cuda — so its dramatic numbers
(8–15× warp-divergence fixes) are hypotheses to re-verify, not findings. A
third slice is Haiku-fabricated filler and is excluded from evidence outright.

The live-tree map mattered more than the research: the engine has already
absorbed several of the corpus's headline recommendations (pooled async
allocator, fused RMSNorm, online-softmax APA kernel). What remains genuinely
open, in rough order of confidence: a per-token host sync in the decode
driver, hundreds of small launches per token with no CUDA graph, an O(S²)
mask materialization on the APA fast path, a two-pass standard softmax, an
int4 GEMV that ignores DP4A, and unfused SwiGLU elementwise pairs.

The plan therefore leads with receipts (harness, nsys/ncu, register table,
wrong-tree claim verification) and gates every phase on thresholds registered
up front. Numerics-moving changes — the DP4A/Q8_1 path above all — are held
at kernel-sweep validation and cannot be adopted without a model-bound phase,
mirroring the quant sweep's two-layer split. The plan stays the fixed intent;
the ledger holds the receipts.
