# Kernel Optimization Implementation Ledger

This ledger is the execution record for the kernel optimization program. The
immutable implementation plan stays fixed; this file records what actually
happened.

## 2026-07-07 08:45 EDT

Action: Baseline orientation and research triage.

Repo state:
- Repository: `/mnt/ForgeRealm/Project-Tensor`
- Branch at orientation time: `codex/gpt-oss-mxfp4-kernel` (HEAD 1c2e8b0);
  program branch to be created at plan commit.
- Research set: `inv_f32d181e/` (~100 files, 137 MB), AtlasForge dashboard
  investigation, lead=sonnet / subagents=haiku / synthesis=opus,
  max_subagents=50.

Findings (research triage, 5 Sonnet readers over the full set):
- Tier A (usable): llama.cpp MMQ extraction, CUTLASS/CuTe synthesis, RoPE
  investigation (cites live `tensor_cuda/src/kernels.cu:3364`), softmax
  optimization research, launch-parameter cluster, CUDA graphs/sync/profiling
  cluster.
- Tier B (wrong tree): APA self-analysis, warp-divergence cluster, and
  broadcast-stride research analyzed the Rust-port mission snapshot
  (`AI-AtlasForge/workspace/Tensor_Rust_Port/mission_f860a512/tensor-rs/`),
  not live tensor_cuda. Spot-check receipts: `apa_selective_softmax_tile_kernel`
  has zero hits in live `kernels.cu`; `core_kernels.cu` exists only under the
  mission workspace. Their line numbers and measured baselines (3.04×) do not
  attach to the live tree.
- Tier C (untrusted): TENSORRT_LLM_* and BANDWIDTH_* clusters are
  templated/fabricated (invented citations, internally inconsistent numbers,
  a nonexistent `tensorrt_llm.functional.mlp` API). FP8 research never
  executed — no usable FP8 numbers exist in the set.

Findings (live-tree map, 1 Sonnet reader over tensor_cuda + drivers):
- Allocator already pooled (`cudaMallocAsync` behind `tc.set_alloc_pooling`,
  `kernels.cu:99-182`); the research's headline allocator recommendation is
  already implemented — dropped from scope.
- RMSNorm already fused; `apa_selective_kernel` already online-softmax O(L).
- Confirmed hot-path costs: per-token D2H `.numpy()` + host argmax
  (`GraftRepository/scripts/qwen35_generate.py:59`, verified by direct read);
  single legacy stream; no CUDA graphs; O(S²) score+mask materialization in
  `_cublas_blend_attention`; `int4_gemv_kernel` float-FMA without DP4A; no
  in-repo benchmark harness.

House-rule documents created:
- `docs/KERNEL_OPT_IMPLEMENTATION_PLAN.md` (DRAFT — pending David's review;
  immutable at commit)
- `docs/KERNEL_OPT_IMPLEMENTATION_LEDGER.md`
- `docs/KERNEL_OPT_SYNTHESIS.md`

Next action:
- David reviews the draft plan; on approval, create program branch and commit
  the house-rule baseline before any executable work.

## 2026-07-07 09:05 EDT

Action: Plan approved; APA-preservation constraint registered; baseline commit.

Repo state:
- Program branch `kernel-opt-house-rules` created from
  `codex/gpt-oss-mxfp4-kernel` HEAD (1c2e8b0) — deliberately includes the
  sink-aware APA attention kernels, which are inside this program's
  optimization surface.

Findings:
- David's approval condition: APA must not be overwritten; kernels may be
  optimized for speed. Added as a House Rule to the plan before its initial
  (immutability-establishing) commit.

Next action:
- Commit the three house-rule documents; then Phase 0 (harness + receipts)
  via Sonnet implementation agents.
