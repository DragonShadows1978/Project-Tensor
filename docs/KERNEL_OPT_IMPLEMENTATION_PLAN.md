# Kernel Optimization Implementation Plan

Status: DRAFT — becomes immutable at initial commit.
Source research: `inv_f32d181e/` (AtlasForge investigation, Sonnet lead / Haiku
subagents / Opus synthesis, 50-agent fan-out). Target: `tensor_cuda` inference
speed on RTX 4070 Super (Ada, SM 8.9, 56 SMs, 12 GB, ~504 GB/s).

## Objective

Increase tensor_cuda inference throughput — decode tok/s primary, prefill
secondary — on the models it actually serves (Qwen3.5-9B, MiniCPM3, Gemma 4),
using the inv_f32d181e research where it survives triage. No new library
dependencies (no CUTLASS, no FlashAttention, no TensorRT). No change ships
without parity receipts.

## House Rules

- This plan is immutable after its initial commit. Execution details go to
  `KERNEL_OPT_IMPLEMENTATION_LEDGER.md`; meaning goes to
  `KERNEL_OPT_SYNTHESIS.md`.
- Every claim names its evidence class: **kernel sweep** / **unit test** /
  **model perplexity** / **external literature** / **code inspection**.
- Kernel sweeps establish speed, memory shape, reconstruction error, and
  output deviation vs a dense reference — never model quality. Any change that
  moves numerics beyond existing unit-test tolerance requires a model-bound
  validation phase before adoption (same split as the quant sweep).
- Thresholds below are registered now, before any governed gate runs, and are
  not adjusted after seeing results.
- **APA is load-bearing and is not overwritten.** No workstream removes,
  replaces, or bypasses the APA kernels or changes their selection/refinement
  semantics. Attention-path work optimizes the existing kernels in place,
  gated on the existing APA parity tests; any change to APA behavior (not just
  speed) is out of scope for this plan.
- **Power constraint**: individual benchmark/profile runs bounded ≤10 min GPU
  draw; no sustained multi-hour sweeps until the electrical work is cleared.
- Roles: Fable session = planner/interpreter; implementation is delegated to
  Sonnet subagents per workstream.

## Source Triage (registered before use)

Reliability tiers assigned to the research set, fixed here:

- **Tier A (usable, live-tree or self-consistent external):**
  `llama_cpp_mmq_kernel_extraction.md` (real extracted code),
  `CUTLASS_CUTE_RESEARCH_SYNTHESIS.json`, `rope_kernel_investigation.json` /
  `rope_kernel_optimization_guide.md` (cite live `tensor_cuda/src/kernels.cu`),
  `softmax_kernel_optimization_research.json`, launch-parameter cluster
  (`CUDA_KERNEL_LAUNCH_PARAMETER_TUNING_SYNTHESIS.md` + Angle-4 files, with
  noted internal contradictions), CUDA graphs / sync / profiling cluster.
- **Tier B (wrong tree — analyzed the Rust-port mission snapshot at
  `AI-AtlasForge/workspace/Tensor_Rust_Port/mission_f860a512/tensor-rs/`, NOT
  live tensor_cuda):** `apa_kernel_analysis_structured.json`,
  `adaptive_precision_kernel_investigation.md`, warp-divergence cluster,
  `BROADCAST_STRIDE_OPTIMIZATION_RESEARCH.md`. Verified by spot-check:
  `apa_selective_softmax_tile_kernel` and `core_kernels.cu` do not exist in
  live `tensor_cuda/src/` (code inspection, 2026-07-07). Claims from this tier
  may be used only after Phase 0.3 confirms an analogous structure exists in
  the live tree.
- **Tier C (untrusted — templated/fabricated, Haiku-generated):**
  `TENSORRT_LLM_*` cluster, `BANDWIDTH_*` cluster,
  `GPU_MEMORY_HIERARCHY_RESEARCH_SUMMARY.md` speedup tables,
  `QUANTIZATION_BANDWIDTH_KEY_PAPERS.md`. Invented citations and internally
  inconsistent numbers. Concepts may inform design; no number from this tier
  may justify a gate or appear as evidence.
- **Known gap:** FP8-on-Ada research never executed (scripts unrun). Out of
  scope; a fresh investigation is a separate decision.

## Baseline Facts (code inspection, 2026-07-07)

- Decode loop is Python-driven, one full-model call per token; per-token
  device→host `.numpy()` + host argmax on the critical path
  (`GraftRepository/scripts/qwen35_generate.py:59`).
- Single legacy default stream everywhere; no CUDA graphs anywhere in-tree.
- Allocator already pooled: `cudaMallocAsync` behind `tc.set_alloc_pooling`,
  load/runtime split deliberate (`kernels.cu:99-182`). Allocator work is DONE;
  not a workstream.
- RMSNorm already fused single-kernel. RoPE separate elementwise kernel ×2 per
  layer. SwiGLU composes 2 elementwise launches. Generic elementwise unfused.
- Standard SDPA materializes full (B,H,L,S) scores + two-pass softmax;
  `USE_FUSED_SOFTMAX` defaults off. `apa_selective_kernel` is already
  online-softmax, O(L) memory, GQA-aware. Fast path `_cublas_blend_attention`
  (GraftRepository) materializes O(S²) bulk+rank score tensors plus a cached
  full additive causal mask on both (board item 4a; Chaosrabbit no-mask fix is
  the upstream candidate).
- `int4_gemv_kernel`: one warp per output column, `uchar4` packed reads,
  float FMA accumulation — no DP4A, no activation quantization.
- No benchmark harness inside tensor_cuda; e2e timing lives in GraftRepository
  scripts.

## Phase 0: Baseline And Receipts

0.1 **Microbench harness** (`tensor_cuda/bench/`): CUDA-event timing, N=100
    reps after 10-rep warmup, median + IQR, for: `causal_softmax_kernel`,
    `apa_blend_softmax(_sink)`, `apa_selective_kernel`, `int4_gemv_kernel`,
    `int4_gemm_fused_kernel`, `rope_kernel`, `rms_norm_kernel`, and the SwiGLU
    elementwise pair — at model-realistic shapes (Qwen3.5-9B and Gemma 4
    geometries; decode S ∈ {512, 2048, 8192}, prefill L ∈ {512, 2048}).
    Evidence class: kernel sweep.
0.2 **End-to-end + trace receipts**: decode tok/s via
    `GraftRepository/scripts/qwen35_generate.py` (N=5 runs, median);
    one `nsys` trace of a bounded decode session → kernel launches per token,
    launch/gap overhead share, top-5 kernels by time; targeted `ncu` on the
    top kernels (branch efficiency, warp efficiency, memory throughput vs
    peak, achieved occupancy). NVTX ranges added around decode-loop phases.
    Evidence class: kernel sweep.
0.3 **Tier-B verification**: inspect live `kernels.cu` for structures the
    wrong-tree analyses claim (divergent refine/selection branches in APA
    kernels; broadcast-offset div/mod hot paths; single-thread top-k
    ranking). Each claim gets a CONFIRMED-IN-LIVE-TREE / ABSENT verdict in
    the ledger. Evidence class: code inspection.
0.4 **Register/occupancy receipts**: `ptxas -v` register counts for every
    kernel in 0.1; recompute occupancy against Ada limits (56 SMs, 96 KB
    shared/SM, 16 blocks/SM, 64 warps/SM). Evidence class: kernel sweep.

## Registered Thresholds (fixed before any gate)

- **Parity gate (every change):** full existing test suite passes; benched
  kernel outputs match reference within the tolerance already used by that
  kernel's tests. A numerics-moving change (e.g. activation quantization)
  additionally reports max-abs output deviation and is BLOCKED from adoption
  pending a model-bound phase — kernel sweeps never claim model quality.
- **Kernel-level accept:** ≥15% median kernel-time reduction at the Phase-0
  shapes, no shape regressing >5%.
- **Loop-level accept:** ≥5% median end-to-end decode tok/s (N=5), no VRAM
  peak regression >5%.
- **Phase-2 entry gate:** proceed with CUDA graphs only if the Phase-0 nsys
  receipt shows ≥15% of decode-step wall time is launch/gap overhead.
- **Phase-5 entry gate:** proceed with elementwise fusion only if, after
  Phase 2, launch/gap overhead is still ≥10% of decode-step time.

## Phase 1: Decode-Loop Hygiene (effort S)

1.1 Device-side argmax for greedy decode: add/route an argmax kernel so the
    per-token transfer is one int, not the logits row; remove the implicit
    per-token sync. External-literature basis: sync-hierarchy numbers in the
    graphs/sync cluster (Tier A).
1.2 Audit bindings for synchronous `cudaMemcpy` and hidden legacy-stream
    syncs on the decode path; convert to async + pinned staging where they
    survive. Evaluate per-thread default stream flag.
1.3 Gate: loop-level accept threshold, plus before/after nsys gap comparison.

## Phase 2: CUDA Graph Capture Of The Decode Step (effort M/L)

Entry gate registered above. Batch=1, seq_len=1 fixed shape is the easy
capture case; ring-buffer KV cache gives stable addresses. Work: a
`tc`-level capture/replay API around one decode step; transient buffers must
come from the pool in capture-safe fashion or a pre-allocated workspace.
External literature (Tier A): 1.15–1.44× isolated launch-overhead effect (SET
paper) — the 30-40% vLLM number conflates fusion, do not cite it as target.
Gate: loop-level accept (≥5% tok/s); expect more if 0.2 shows launch share
≥25%.

## Phase 3: Attention Path (effort M/L)

3.1 Kill the O(S²) additive-mask materialization in the
    `_cublas_blend_attention` fast path: index-arithmetic causal bound inside
    `apa_blend_softmax` (the in-tree `USE_FUSED_SOFTMAX` SDPA path already
    demonstrates the pattern); evaluate the Chaosrabbit no-mask fix as the
    upstream implementation (board item 4a). Gate: kernel-level accept at
    S ≥ 2048; parity via existing APA tests.
3.2 `causal_softmax_kernel`: two-pass → single-pass online softmax with
    width-dispatched reduction (warp-shuffle only for S≤1024, current
    shared-mem tree above). External literature: 1.8–2.2× (Tier A, generic
    hardware — treat as direction, not target). Gate: kernel-level accept.
3.3 RoPE: (a) head-dim-aware launch config (S); (b) fusion into QKV epilogue
    (M) — only pursued if 0.2 shows RoPE launches are a measurable share.
    Attention-K-load fusion requiring unrotated-K cache layout is OUT OF
    SCOPE for this plan (architectural KV-layout change — separate decision).
3.4 APA kernel opts from Tier B (warp-vote early exit, prefilter, codebook
    shared-mem) — only workstreams whose target structure was CONFIRMED in
    0.3, each gated at kernel-level accept.

## Phase 4: Quantized Matmul (effort M)

4.1 DP4A path for `int4_gemv_kernel`: quantize activations to Q8_1-style
    blocks (fused half2 scale+sum), nibble-unpack via `0x0F0F0F0F` masks into
    `__dp4a` (llama.cpp MMQ extraction, Tier A). NUMERICS-MOVING: subject to
    the blocked-adoption rule — kernel sweep reports speed + output deviation
    only; adoption requires the model-bound phase.
4.2 Launch-config sweep for hot kernels (128/256/512 threads; grid-stride
    sizing at 2–4×56 blocks for flat elementwise) using the 0.1 harness plus
    0.4 register receipts. Adjust only where kernel-level accept is met.

## Phase 5: Elementwise Fusion (effort S/M, conditional)

Entry gate registered above. Candidates in order: SwiGLU (silu·up single
kernel), RMSNorm+residual-add. Gate: kernel-level accept AND a visible
loop-level effect (≥2% tok/s), else revert — fusion that only moves
microbenches is not adopted.

## Acceptance Criteria

- Phase 0 receipts (harness, nsys/ncu traces, ptxas table, Tier-B verdicts)
  committed before any optimization lands.
- Every adopted change: parity gate + its registered accept threshold, with
  receipts in the ledger.
- Numerics-moving changes explicitly held at "kernel-sweep-validated,
  model-bound validation pending" — never adopted silently.
- Final synthesis reports the cumulative decode tok/s multiplier per model
  with the receipt chain; no pre-registered overall multiplier exists, so
  none is claimed.
