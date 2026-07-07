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

## 2026-07-07 09:20 EDT

Action: Benchmark matrix extended to GPT-OSS-20B; Phase 0 agents launched.

Repo state:
- Branch `kernel-opt-house-rules` at 121e783 (plan committed, immutable).

Findings:
- David directed attention to the GPT-OSS-20B implementation
  (`GraftRepository/core/gpt_oss20b_tc.py`): sink-aware APA blend softmax
  (`tc.apa_blend_softmax_sink`) + `resident_packed_mxfp4` expert mode — a
  20B MoE at near-full context in VRAM on the 12 GB card, no OOM.
- Execution detail, not a plan change: the plan's objective covers "the
  models it actually serves"; the program branch itself contains the
  GPT-OSS kernels (1c2e8b0, c9584df, 7cfd55e). GPT-OSS-20B geometry and the
  `mxfp4_gemv/gemm` + `apa_blend_softmax_sink` kernels are added to the
  Phase 0.1 bench matrix. All registered thresholds unchanged.
- Long-context note: the blend fast path is gated to S ≤ fast_max_seq
  (4096 default), so near-full-context GPT-OSS decode runs the O(L)
  `apa_selective_kernel` + mxfp4 expert GEMV — those two are the expected
  hot kernels for that workload, to be confirmed by 0.2 receipts.

Next action:
- Phase 0.1 (harness incl. GPT-OSS shapes), 0.3 (Tier-B live-tree
  verification), 0.4 (ptxas/occupancy receipts) running as Sonnet agents;
  0.2 (nsys/e2e) after the harness lands.

## 2026-07-07 09:55 EDT

Action: Phase 0.3 complete — Tier-B claim verdicts (Sonnet code inspection;
claims 1 and the mxfp4 observation spot-checked by lead at cited lines).

Findings (evidence class: code inspection):
- Claim 1 (per-key divergent refine branch) — CONFIRMED-IN-LIVE-TREE:
  `kernels.cu:1123` in `apa_selective_kernel` (`fabsf(bulk) >= thr` inside
  the strided key loop at :1117, threads=128); same pattern in the train
  forward (:1479) and backward (:1577, :1600) kernels. Phase 3.4's
  warp-vote / branchless-refinement candidates attach to a real structure.
- Claim 2 (thread-0-only O(S²) top-k) — ABSENT as described. `topk_kernel`
  (:2964) is one-thread-per-row selection scan; occupancy scales with
  batch·heads·rows. Partial analog only for degenerate outer==1 calls.
- Claim 3 (rank-based per-thread selection, S≤256) — ABSENT. `rank` in the
  blend kernels is a precomputed cuBLAS matrix, not in-kernel ranking.
- Claim 4 (broadcast-offset 64-bit div/mod helper) — ABSENT. No such helper
  in live tree; matmul.cu broadcasts via cuBLAS stride-0.
- Claim 5 (backward refine-mask branch) — CONFIRMED (:1600, :1617-1620);
  fwd/bwd recompute `sel` identically; structurally safe unless the two
  float recomputations drift (compiler reassociation risk noted).
- Claim 6 (codebook from global memory) — PARTIAL: `apa_qg_kernel`
  (:993-1001) does per-element global codebook/boundary reads, no shared
  staging (prefetch opportunity real); rotation is cuBLAS matmul in
  `quant.py:125`, not a bespoke kernel; no in-kernel Lloyd-Max.
- NEW live-tree observations (not from Tier B): `mxfp4_gemm_kernel` weight
  reads are uncoalesced across the warp (adjacent `col` lanes hit
  `blocks[(col*G+g)*16+…]`, stride G·16 bytes — spot-checked :2445,
  :2465-2468) and `scales` are re-read from global per element with no
  block-level reuse; `mxfp4_gemv_kernel` path is substantially better
  coalesced. `apa_selective_kernel` recomputes the bulk dot in both of its
  two passes (acknowledged tradeoff comment :1083-1087). Register tail
  waste when VD < DMAX (:1114).

Decision pending (David): the mxfp4 coalescing/scale-staging observation is
new work not enumerated in the immutable plan's Phase 4 items. Options:
treat as within Phase-4 intent (quantized-matmul optimization, bench matrix
already includes mxfp4) or defer to a follow-up plan. Not started either way.

Next action:
- Await 0.1 harness + 0.4 ptxas receipts; then 0.2 nsys/e2e baseline.

## 2026-07-07 10:10 EDT

Action: Phase 0.4 complete — ptxas/occupancy receipts (Sonnet; scratch
compile, existing build untouched).

Receipts:
- Command: nvcc (cuda-12.6) `-c src/kernels.cu -arch=sm_89 -O3
  --use_fast_math -Xptxas -v`; compiled clean, 238 instantiations.
- Artifacts: `artifacts/kernel_opt/ptxas_registers.json`,
  `artifacts/kernel_opt/ptxas_receipts.md`.

Findings (evidence class: kernel sweep — compile-time analytical, not ncu):
- Zero register spills across all 238 instantiations.
- All inference-path kernels at 100% analytical occupancy at their current
  block sizes. Sole sub-100%: `apa_selective_bwd_kernel` (75-83.3%,
  register-limited, 48-53 regs) — backward only, not on the inference path.
- BLIND SPOT registered: int4/intn/mxfp4 GEMV kernels use dynamic shared
  memory (K·sizeof(T), opt-in to 96 KB) invisible to ptxas -v; at large K
  the real residency may be 1 block/SM. Runtime ncu receipt (0.2) is
  authoritative for these.
- Interpretation for later phases: the research corpus's "register pressure
  is the dominant lever" prediction does NOT hold for this codebase —
  registers and analytical occupancy are already clean. Phase 4.2
  (launch-config sweep) is expected low-yield except for the dynamic-shmem
  GEMV question; the measured levers remaining are memory access patterns,
  divergence (0.3 findings), sync/launch overhead, and algorithmic passes.

Next action:
- Await 0.1 harness; then 0.2 nsys/e2e baseline with targeted ncu on the
  GEMV dynamic-shmem residency question.

## 2026-07-07 10:25 EDT

Action: Phase 0.1 complete — microbench harness (Sonnet) + full sweep (lead).

Receipts:
- Harness: `tensor_cuda/bench/{__init__,geometries,harness}.py`; geometries
  extracted from the real drivers (qwen35_tc, gemma4_tc, gpt_oss20b_tc) with
  source citations; VRAM pre-check raises recorded skips, never silent.
- Smoke: `artifacts/kernel_opt/kernel_microbench_smoke_20260707_100550.json`
  (46 shapes, 42.6s). Full: `kernel_microbench_full_20260707_101436.json`
  (146/146 entries, 0 skips, 453.1s wall — inside the ≤10 min bound; seed
  20260707, 10 warmup + 100 reps, perf_counter bracketed by synchronize).
- Existing parity tests re-run by the implementing agent: pass (bench is
  purely additive).

Findings (evidence class: kernel sweep):
- `apa_selective` decode at S=8192: 1.43 ms (gpt_oss20b geometries) to
  7.20 ms (gemma4 global MQA) per call — the dominant per-layer attention
  cost in the long-context regime where the blend path is gated off. This
  is the same kernel carrying the 0.3-confirmed divergent refine branch:
  inspection and measurement now point at the same target (Phase 3.4).
- `apa_selective` prefill L2048: 217–550 ms per call. Interpretation
  caveat: drivers route prefill at S≤fast_max_seq through the blend path
  (~0.7 ms at L512), so these numbers price the >4096 long-context prefill
  regime, not the common path.
- Quantized fused GEMM prefill at L2048: 137–214 ms (int4/intn2/intn3
  across models) — prefill-scale cost of the 16×16-tile fused kernels;
  two-stage-vs-fused crossover detail is in the artifact.
- Blend-family and causal softmax decode kernels are cheap at all swept S
  (≤0.28 ms); rope/rms_norm/swiglu decode-shape costs are launch-overhead
  scale (~0.12 ms) — consistent with launch/sync hygiene (Phases 1-2)
  mattering more than kernel math for short-context decode.
- Naming note: GEMV vs GEMM dispatch happens inside `int4_linear_fused` /
  `mxfp4_linear` on input row count; there are no separate *_gemv symbols.

Next action:
- Commit Phase 0.1/0.3/0.4 receipts; launch 0.2 (nsys decode trace +
  targeted ncu: GEMV dynamic-shmem residency, apa_selective at long S).

## 2026-07-07 11:15 EDT

Action: Phase 0.2 complete (Sonnet) — with a registered contamination caveat.

CONTAMINATION CAVEAT: all 0.2 runs shared the GPU with a concurrent Graft
Translation mission job (`gpt_oss20b_stream_forward_smoke.py`, 100% SM util,
left running — not ours to kill). Wall-clock-derived numbers (e2e tok/s,
gap share) are NOT valid registered baselines; ncu per-kernel
hardware-counter ratios are considered valid (kernel-replay isolation).
E2E + nsys must be re-run contention-free before gating Phase 1/2.

Receipts: `artifacts/kernel_opt/phase02_trace_receipts.md` (+ nsys-rep,
sqlite, ncu-rep/txt files, gitignored per repo policy). ncu needed sudo -E
for counter access.

Findings (evidence class: kernel sweep; ncu ratios trusted, wall-clock not):
- E2E (contaminated, lower bound): qwen35 decode median 55.1 tok/s, N=5.
- Launches/token ≈ 2,210 (int4_gemv exactly 201/token); gap/launch share
  ≈19% (lower-bound estimate). Per-token D2H argmax = 33 sync round-trips
  per 32-token run — Phase 1.1 target confirmed.
- int4_gemv (K=4096): 76% achieved occupancy, 5 blocks/SM (dynamic-shmem
  1-block fear REFUTED), 100% branch efficiency, 71.6% DRAM/peak, but 47%
  excessive shared-mem wavefronts (bank-conflict signal). NEW, not
  enumerated in plan Phase 4 items.
- apa_selective (S=8192 decode): 8.33% achieved occupancy from GRID
  UNDERFILL — decode launches only B·H·L=16 blocks on 56 SMs ("0.0 full
  waves"); branch efficiency 99.02% — the 0.3-confirmed divergent refine
  branch is NOT the measured bottleneck at decode. The high-leverage fix is
  key-dimension split (flash-decoding-style split-K + reduce), which is NOT
  among the enumerated 3.4 items. NEW.
- mxfp4_gemv (K=2880): 60.94% branch efficiency, ~16k divergent branches
  per launch — the mxfp4 nibble-decode path (16-way `mxfp4_value` switch,
  flagged "possible" in 0.3) is CONFIRMED divergent in practice. On the
  GPT-OSS decode path. NEW.

Interpretation: Phase 0 receipts collectively redirect the program. The
plan's enumerated kernel items (3.4 warp-vote divergence fix, 4.1 int4
DP4A) now look lower-leverage than three measured, non-enumerated targets:
apa_selective decode grid underfill, mxfp4_gemv branch divergence,
int4_gemv shared-mem bank conflicts. Phases 1-2 (hygiene, graphs) remain
valid as enumerated.

Decision pending (David):
1. Register a follow-up addendum plan (immutable once committed, own gates)
   for the three new measured targets, OR fold under a broad reading of
   Phase 3/4 intent. Lead recommends the addendum — keeps spec-is-law clean.
2. Scheduling of the contention-free e2e/nsys re-run (needs the Codex
   mission job finished or paused — coordination call, not ours).

Next action:
- Await David on both; meanwhile no optimization work starts (Phase 0
  gate: receipts first, which is now satisfied except the re-run).
