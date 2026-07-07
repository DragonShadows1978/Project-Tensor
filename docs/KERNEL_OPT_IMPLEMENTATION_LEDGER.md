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

## 2026-07-07 11:45 EDT

Action: David authorized implementation (session goal, 2026-07-07):
decisions to orchestrator's discernment; sole restriction = APA stays
two-pass bulk-bits → selective precision. Addendum 1 registered and
committed (immutable): workstreams A2 (mxfp4_gemv branchless decode),
A3 (int4_gemv bank conflicts), A1 (apa_selective split-K grid fill,
APA-invariant-bound), A4 (mxfp4_gemm coalescing), order A2→A3→A1→A4,
interleaved with base-plan Phase 1.1 (device argmax).

Repo state:
- GPU occupied by Codex Graft Translation capture (PID 2346865,
  gpt_oss20b_stream_forward_smoke.py, 96K capture, ~1h14m elapsed at
  check). Gating Validity Rule invoked: implementation + parity tests
  proceed under contention; ALL timing gates defer to an idle-GPU window.

Next action:
- Sonnet agents: A2+A3 (main tree), Phase 1.1 argmax (isolated worktree).
  A1 after A2/A3 land.

## 2026-07-07 12:40 EDT

Action: A2+A3 implemented (Sonnet), lead-verified, committed d917d35.

Findings (evidence class: unit test + code inspection; timing NOT gated):
- A2: branchless E2M1 decode, 16/16 bit-identical to the reference switch
  (exhaustive on-device test); reference kept under
  TC_MXFP4_REFERENCE_DECODE; gemm gets it free via shared helper (tile
  loads untouched — A4 still open). Warp-uniform scale read hoisted.
- A3: diagnosis from ncu aggregate + SASS reconstruction — lanes' private
  8-element windows collide 8-way on LDS.128; fix = 1-per-8 padding
  (stride 9), reaching the 4-way floor inherent to 128-bit lane loads.
  Shmem now K+K/8 (guard: K%8==0 launch condition — coupled, noted).
- Parity: suite 90 pass / 1 pre-existing fail (verified pre-existing via
  stash-rebuild-retest); int4 GEMV rel err 2.41e-07. ptxas: int4 regs
  unchanged, mxfp4 21→23 (not occupancy-limiting).
- Contended timing capture was uniformly ~90% off across UNTOUCHED kernels
  too — conclusively contention noise; discarded, not evidence. Kernel
  accept gates for A2/A3 remain OPEN pending idle GPU.

Next action:
- A1 (apa_selective split-K) agent on main tree; argmax worktree agent
  still running.

## 2026-07-07 13:20 EDT

Action: Phase 1.1 device argmax implemented (Sonnet, worktree), committed
d7f6d4a on branch `kernel-opt-argmax`. Merge into program branch DEFERRED
until A1 lands (A1 agent holds uncommitted kernels.cu state in main tree).

Findings (evidence class: unit test):
- `argmax_last_axis`: block-per-row, 256 threads, grid-stride +
  two-stage shuffle reduction, numpy first-occurrence tie-break enforced
  at every combine. 69/69 unit cases incl. decode shape (1, 152064).
- Token-parity receipt: 24-token greedy generation OLD (host argmax) vs
  NEW (device) — sequences IDENTICAL, decoded text SHA256-identical.
  Valid under contention.
- Suite: 159 pass, same single pre-existing failure (stash-verified).
- Indicative non-gating: isolated argmax 218.9µs vs 3325µs host path
  (~15×) on contended GPU; e2e delta unresolved under contention.
- GraftRepository `qwen35_generate.py` edited in place (uncommitted),
  guarded by hasattr fallback — inert until the new extension installs.
- Registered follow-up (not in any plan; execution detail): 4 other
  GraftRepository call sites share the host-argmax pattern; the two
  server scripts entangle host-side sampling — candidates for a later
  device-sampling pass, listed in the agent report, not edited.

Next action:
- Await A1; then merge kernel-opt-argmax, rebuild+install extension,
  re-run parity, commit.

## 2026-07-07 13:05 EDT

Action: Tip gating batch ran (sweep 146/146 idle-GPU, e2e ×5, nsys, ncu).
GATE VERDICTS WITHHELD — comparison self-invalidated by the lead.

Findings:
- Discovered asymmetry: the 10:14 baseline sweep ran during the lead's own
  0.3/0.4 agents' nvcc compiles (CPU saturation). Signature in the A/B:
  untouched kernels (int4_two_stage cuBLAS path, intn GEMVs, fused
  apa_selective at S=512 below the split-K heuristic) show +6–63% phantom
  gains; two opposite-direction outliers (causal_softmax gemma sliding
  prefill −52%). Sub-ms kernels are dispatch-overhead sensitive; CPU
  contention during the baseline inflated it. Timing A/B is NOT gate-grade
  in either direction. A failure is still a result: registered here.
- Contention-immune receipts that DO stand (ncu counter ratios):
  mxfp4_gemv branch efficiency 60.94% → 75.03% (A2 corroborated);
  int4_gemv branch efficiency 100% unchanged. apa split-K ncu probe
  errored (rc=2) — retry in clean session.
- Directional (not gating): apa_selective decode S=8192 +30–50%,
  S=32768 +34.7%, consistent with the split-K design prediction and the
  A1 agent's independent indicative numbers (1.33–1.83×).
- Tip e2e (56.5 median) vs baseline e2e (57.6 median): both batches had
  possible CPU contention from concurrent builds — same invalidation.

Next action:
- After A4 lands: ONE clean measurement session, nothing else running —
  baseline sweep + e2e (pt-baseline worktree, extension prebuilt), tip
  sweep + e2e, ncu apa-split retry, nsys launch-share for the Phase 2
  entry gate. All gates decided from that session only.

## 2026-07-07 14:05 EDT

Action: A4 CLOSED — NEGATIVE RESULT (gate failed, reverted unmerged).

Findings (evidence class: kernel sweep, clean window — GPU idle verified
before every timed run):
- Design: group-half staging (uint64 vectorized block-half + scale byte
  per tile column into shared, 16× fewer transactions, 16× fewer scale
  reads). Parity bit-identical (0.0 diff all dtypes/shapes); regs
  unchanged; 0 spills; occupancy unchanged.
- Timing: ALL shapes regressed +1.7–2.2% (gate_up/down × L512/L2048).
  The 16×16 tile's FMA loop dominates kernel time; weight-load
  transaction count was not the binding bottleneck. Staging overhead >
  redundancy saved. Full coalescing across cols is structurally
  impossible in the (col,g)-major layout without a weight-layout change
  (out of scope — layout changes alter the packed format contract).
- Disposition: change discarded (lived only in the A4 worktree; program
  branch never carried it). A failure is still a result.
- Note: agent initially saw phantom parity failure from Python salted
  hash() seeding in its own test script — root-caused as test bug, not
  kernel defect; receipts in agent transcript.

Next action:
- Clean measurement session running (both sweeps, both e2e batches, both
  nsys, ncu apa-split retry). All gates decided from it.

## 2026-07-07 14:40 EDT

Action: CORRECTION to the registered source triage (David's challenge:
"look THROUGH the json from the web proxy"). Lead inspected the raw
captures directly. The plan's Tier C label ("templated/fabricated") was
WRONG as written; this entry is the correcting record (plan itself is
immutable — corrections live here).

Findings (evidence class: code inspection of raw captures):
- artifacts/webproxy_json/ = 922 real proxy calls (815 WebSearch, 96
  WebFetch, 11 PaperFetch) across 56 subagent dirs, with full provenance
  (URLs, backends, cache keys, sha256). This is real research material,
  not hallucination.
- The failure mode is in the SYNTHESIS layer (Haiku subagent reports),
  not the source layer. Traced examples: report's
  `tensorrt_llm.functional.mlp` ← capture's REAL
  `tensorrt_llm.functional.gpt_attention` (garbled symbol); "Zhou,
  Greenfeld & Elhoushi (2016)" ← merge-mangle of real quantization
  authors (Zhou/DoReFa 2016, Elhoushi/DeepShift); "6-16x" ←
  synthesis-computed composite, not a fetched number.
- The reader agent's "fabricated" verdict was itself an over-claim, and
  the lead relayed it at full confidence without opening the captures —
  a triage-of-the-triage failure. Corrected operational rule: Tier C =
  UNRELIABLE SYNTHESIS OVER REAL SOURCES; load-bearing Tier-C claims are
  traced into webproxy_json/ and used at capture provenance, not
  discarded.
- "FP8 research never executed" also wrong at the source layer: 155
  captures contain FP8/E4M3 material incl. targeted Ada-FP8 spec
  queries. Only the analysis scripts died. An FP8-on-Ada assessment can
  be built from existing captures without new web work if wanted.

Next action:
- Board + memory corrected to match. Clean session still running; gate
  verdicts unaffected by this entry (no Tier-C number ever gated
  anything — that discipline held and stands).

## 2026-07-07 (clarification, David, verbatim-faithful)

APA invariant sharpened by David mid-program: "The concept is that
Bulk-Bits scores the z-score, and the Second Pass applies Full precision
to the refine percentile. THAT is the function... if that can be
optimized that's fine. But changing THAT function moves away from the
Hypothesis." Reading: the protected object is the FUNCTION (bulk-bits
scoring → z-score threshold → full precision on the refine percentile),
not any particular kernel structure; implementation rework for
effectiveness is explicitly permitted. A1's three-stage structure
complies (identical bulk dots, threshold, selection, refine; parity
receipts 0-diff at f32). All future attention workstreams inherit this
wording.
