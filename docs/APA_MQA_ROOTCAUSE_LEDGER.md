# APAMQ Ledger — APA × MQA Root-Cause Investigation

Append-only. Commands, file changes, results, failures, follow-up
decisions, as they happen. Plan: `docs/APA_MQA_ROOTCAUSE_PLAN.md`
(immutable).

## 2026-08-13 — Setup (lead)

- Recon: GPU idle (4070S, 257 MiB used). `kernels.cu:1053`
  TC_APA_MAXD = 512 ("bumped 256->512 for Gemma 4 global") — H-D
  registered from this.
- Plan + ledger + orders APAMQ_E1 (Project-Tensor) / APAMQ_E3
  (GraftRepository) authored and committed before dispatch.
- Dispatch: two Sol-max seats via codex-shim, parallel, flock GPU
  serialization. Sentinel waiters armed at dispatch.
- 21:15Z dispatched: E1 = SHIM-RUN 20260813T211541Z-791086
  (logs/apamq_e1_r1.log, Project-Tensor), E3 = SHIM-RUN
  20260813T211542Z-791603 (logs/apamq_e3_r1.log, GraftRepository).
  Both `-m gpt-5.6-sol`, SHIM_TIMEOUT_SECS=21600. Plan committed
  45ae082 (Project-Tensor) / order 4379e17 (GraftRepository) BEFORE
  dispatch.

## 2026-08-13 — E1 result (lead-run)

- E1 Sol seat: honest RED — codex sandbox exposes no GPU device
  (nvidia-smi driver fail, no /dev/nvidia*). Harness authored clean,
  0 fabricated cells (159 SKIPPED). Routing-ledger note appended.
- Lead ran the seat's harness under flock (logs/apamq_e1_leadrun.log):
  **160/160 cells OK, 0 OOM, 0 error.** Artifacts:
  artifacts/apamq_e1/{results.json,RESULTS.md}.
- **T1 / H-A: CONFIRMED.** Standard-path pool transient is IDENTICAL
  across kv ∈ {1,4,8,16} (D=128: 130→2050 MiB linear in S; D=512:
  136→2056 MiB), and fused-APA transient is IDENTICAL across kv and
  FLAT in S (2 MiB at D=128, 8 MiB at D=512, all S to 64K). Ratio
  deviation across kv ≈ 0%, far inside the ±20% threshold. The fused
  path's transient elimination (up to ~1025× at 64K prefill) is fully
  available at kv=1. KV-head count does not enter the memory
  economics. Evidence class: kernel sweep.
- SPEED (same sweep, secondary): fused is uniformly SLOWER than the
  cuBLAS standard path at these shapes — D=128 prefill ~5.8× at 64K
  (302 vs 52 ms), D=512 prefill ~16.6× (1031 vs 62 ms), decode
  D=512 kv=1 ~16× (9.3 vs 0.58 ms). Mechanism: the fused kernel
  computes dots on CUDA cores (warp-shuffle reductions); the standard
  path rides cuBLAS tensor cores. The O(L)-memory property is bought
  by giving up tensor cores. kv=1 is the FASTEST geometry for both
  paths at decode (smallest KV stream) — consistent with "MQA is the
  kernel's best case," refuting any kv=1 speed penalty.

## 2026-08-13 — E3 seat report (audit complete; GPU cells lead-run)

- E3 Sol seat: same sandbox-no-GPU wall (5th ledgered instance).
  Audit + harness delivered clean; protected-source SHA receipts
  clean; zero fabrication (all GPU rows marked NOT MEASURED).
- **T3 / H-D: CONFIRMED (static).** D=512 fused is kernel-legal
  (kernels.cu:1847-1850 dispatch arm; rejection only >512 at
  :1868-1869) but Gemma APA decode unconditionally dispatches
  _cublas_blend_attention (gemma4_tc.py:562-579). Fused decode is
  legal-but-unwired — June's "kernel-forbidden" clause is stale as a
  mechanism claim; parity was measured against this wiring.
- Branch census (audit): S=16K APA prefill = 4 standard + 7 blend +
  161 fused chunks; decode = blend always.
- kqb ring (static, bf16 mirror): 144 MiB as-built at S=16K
  (128 exact + 16 capacity), 272 MiB at 32K.
- 110MB item: strongest static suspect = blend live set, 3
  score-shaped bf16 tensors = exactly 90 MiB at (L,S)=(320,3072).
- **Static bound arithmetic (seat, cross-check pending live run):**
  at S=16K eliminable global score/softmax transients ≈ 64 MiB
  (adaptive chunking already caps standard-path chunks) vs exact-S
  kqb 128 MiB → net −65 MiB; at 32K net −129 MiB. If live numbers
  confirm: APA-as-designed is memory-NEGATIVE on this port because
  the port's adaptive chunking already bounds the very transients
  the fused path eliminates, while the bf16 kq ring mirrors the
  whole MQA cache.
- Lead-running the 6 GPU cells (4K/8K/16K × both modes) under flock;
  logs/apamq_e3_leadrun.log (GraftRepository).

## 2026-08-13 — Fix wave dispatched (David: "So lets fix it")

- Fix plan committed 2c0633e: docs/APA_MQA_FIX_PLAN.md (F-A engine
  int4-internal causal selective kernel + tensor-core rung; F-B Gemma
  fused-decode wiring + ring-drop scaffold; F-C quality ladder
  deferred until stack exists; gates G-A1/A2/B1/B2 registered).
- Key design input: noncausal DiT arc already has apa_int4_pack_kernel
  + apa_int4_sdpa_noncausal (kernels.cu:4594+) — in-kernel INT4 pack,
  NO external kq tensor. Causal/selective port of that design
  eliminates the kqb ring entirely (144→0 MiB resident @16K).
- 21:43Z dispatched, worktree-isolated (installed .so + main trees
  untouched while E3 live cells run): FA = SHIM-RUN
  20260813T214319Z-802383 (wt/apamq-fa, branch apamq-fa), FB =
  SHIM-RUN 20260813T214321Z-802926 (wt/apamq-fb, branch apamq-fb).
  Both Sol-max, 21600s leash, sentinels armed. GPU gates lead-run by
  design (sandbox has no GPU — 2 receipts today).

## 2026-08-13 — E3 live cells complete (lead-run) + T2 adjudication

- 6/6 cells complete (artifacts/apamq_e3/{standard,apa}_s{4096,8192,
  16384}.json, GraftRepository), one mode per process, pool + 1s nvml,
  kqb dtype assertion PASS. Measured A/B (peak abs MiB, APA−std):
  prefill +46/−24/−22 at 4/8/16K; decode +57/+97/+178. kqb resident
  48/80/144 MiB — decode penalty ≈ ring size + blend/quantize decode
  transients (35+7 MiB classes). Prefill wall APA/std: 1.01×/1.05×/
  1.13×; decode ms/tok +9% (50.8→55.3, 72.4→77.4, 114.1→124.8).
  Largest shared per-call class: sliding attention 109 MiB (both
  modes, APA-ineligible), global 109→136 MiB (APA adds blend chunks).
- **T2 ADJUDICATION: perfect-APA reclaimable at S=16K ≈ 180–190 MiB
  (ring 144 + APA decode transients) < 300 MiB registered threshold →
  verdict: PARITY WAS PREORDAINED BY THE ARCHITECTURE INSTANCE.**
  On this model/card nothing walls inside the trained window and the
  binding per-call transients live in APA-ineligible sliding layers;
  APA's own tax (the ring) was the only mode delta. KV-head count
  appears nowhere in the measured attribution — consistent with E1's
  T1. The fix wave flips APA to memory-neutral/slightly-positive
  (ring→0 via int4-internal kernel); it does not create extension on
  THIS model at these contexts because there is nothing to extend
  into — extension value lives where standard-path transients bind
  (E1: linear-in-S up to 2 GiB per call at 64K unchunked shapes).

## 2026-08-13 — FA/FB seats landed; lead-run GPU verification wave

- FB (run 802926): decode fused wiring + int4 no-ring scaffold to the
  pre-nailed ABI; incremental quantize preserved; diff lead-verified
  at the dispatch site; CPU receipts green. Parity tolerance
  registered by seat: exact greedy agreement over 64 steps @8K + max
  |Δlogit| ≤ 0.5.
- FA (run 802383): **F-A1 LANDED** — apa_selective_attention_int4
  (call-local symmetric-7 INT4 pack, ceil(D/2)+fp32-scale per key,
  fp32 in-register dequant, GQA/MQA, D≤512, monolithic+WCOOP+split-K
  decode, bottom-right causal + S<L contract rejection), built clean
  (arch 89) in the worktree; 11 skip-safe gates authored (tolerances:
  composed-ref 2e-3, bf16-kq cross-check 2e-2 — bf16 boundary
  removed, split-K 3e-3). **F-A2 tensor-core rung: honest STOP** —
  semantic blocker: staging K to bf16/TF32 for tensor cores perturbs
  threshold statistics/selection mask beyond the reassociation-only
  invariant; STOP doc docs/APAMQ_FA2_TENSOR_CORE_STOP.md (worktree).
- Lead-run GPU wave launched under flock: FA gate suite + extended E1
  sweep (int4_apa rows); FB parity gate on today's engine + int4
  ring-drop gate against FA's worktree-built engine (cross-worktree
  integration).

## 2026-08-13 — FA GPU verification (lead-run): 11/11 PASS + int4 sweep

- Gate suite: **11 passed** in 1.12s on the 4070S (worktree build).
  G-A1 CLOSED GREEN.
- Extended sweep 240/240 OK. int4_apa at D=512 kv=1 (pack included in
  timing): prefill transient 9→24 MiB at 4K→64K (vs standard 136→2056,
  fused-bf16kq flat 8) and **1.3–1.5× FASTER than the bf16-kq fused
  path** (37.9 vs 49.2 ms @4K; 745 vs 1089 ms @64K — packed keys halve
  bulk-pass memory traffic). Decode comparable to fused-bf16kq. Still
  ~12.5× cuBLAS standard at D=512 prefill — the F-A2 tensor-core gap
  stands as the remaining speed frontier.
- Net vs old path at the port level (once FB int4 mode is on): ring
  144 MiB → 0 resident, quantize-churn eliminated, and the fused call
  itself gets ~1.4× faster. Awaiting FB parity + ring-drop gates.

## 2026-08-13 — G-B1 RED (real finding) + FB-D1 diagnostic dispatched

- FB decode gate (lead-run, canonical engine, S=8192, 64 steps,
  512+512 calls censused): greedy tokens MATCH but max|Δlogit| =
  12.852391 vs registered 0.5 → **FAIL. Blend and fused implement the
  same selection rule at different precisions** (blend: cuBLAS bf16
  bulk/rank matrices + stats over bf16, apa_blend_softmax_kernel2;
  fused: fp32 in-register) → near-threshold selection flips. At
  D=512/bulk4 the flip population is non-trivial — first hard receipt
  consistent with the √D-noise quality mechanism. NOT merged
  default-ON; a failed registered gate is a result.
- INT4 ring-drop leg never ran the FA engine: gate hardcodes
  TC_ROOT=canonical at sys.path[0] (PYTHONPATH defeated). Fix in D1.
- FB-D1 dispatched on branch apamq-fb (order committed 1st):
  divergence attribution (selection-overlap + force-refine-all
  control), bulk_bits {4,6,8} ladder (doubles as F-C early probe),
  TENSOR_CUDA_ROOT override, int4 gate leg. SHIM-RUN
  20260813T221719Z-815096, Sol-max, sentinel armed. GPU legs
  lead-run.

## 2026-08-13 — FBD1 legs complete (lead-run): H-DIV REFUTED,
## divergence = blend bf16 SCORE MATERIALIZATION (force_all receipt)

- Bits ladder FLAT: max|Δlogit| 9.81/10.91/10.88 at bits 4/6/8; flip
  fraction ~0.00104 flat; Jaccard ~0.9932 flat; flip↔Δattention
  Pearson ≈ 0. Bulk-quant noise and selection flips are NOT the
  driver. **H-DIV (√D-noise → selection instability drives the
  divergence) REFUTED for this comparison** — honest failure recorded;
  the √D story survives only as unmeasured background, not as the
  explanation of anything measured today.
- **force_all control (refine=1.0, both paths exact-score every key,
  selection identical, thr Δ=0): attention out max|Δ| = 0.125,
  max|Δlogit| = 10.21.** With selection eliminated, the divergence
  persists → the named term (seat's bisect + lead concur): the blend
  MATERIALIZES score matrices in bf16 (bulk AND rank) and softmaxes
  bf16 scores; the fused kernel is fp32 in-register end-to-end. At
  Gemma magnitudes (scale=1.0, D=512 dots) bf16 absolute rounding is
  O(0.1)/score — the blend path itself is the noise source. All
  prior Gemma APA measurements (incl. June ppl +1.55–1.92%) ran
  blend segments; part of that regression may be blend rounding, not
  APA-the-mechanism. D=512+scale=1.0 amplifies this too — again a
  head-DIM effect, not head-count.
- Greedy tokens: 64/64 MATCH on EVERY leg (bits4/6/8, force_all,
  int4). Functional stability across all variants.
- int4 leg (FA engine, TENSOR_CUDA_ROOT override works):
  **ring_drop_pass = TRUE** (kqb never allocated), tokens MATCH;
  logits max|Δ| 7.37 vs blend (different bulk quantizer + no bf16
  materialization — not directly comparable; awaits ppl).
- **G-B1 ADJUDICATION: FAILED AS REGISTERED, stays failed** — no
  post-hoc tolerance adjustment. The gate was mis-specified: it
  measured the BLEND's bf16 rounding (force_all receipt), so it can
  never pass for any fp32-faithful implementation. The fused kernel
  separately holds its 11/11 fp32-reference gates (2e-3). Defaults
  stay un-flipped; merge decision deferred to G-C (ppl, registered
  below BEFORE running).
- **G-C REGISTERED (F-C leg, before any run):** engaged-scoring ppl
  A/B at S≥8K, arms = standard / apa-blend (status quo) / apa-fused-
  decode / apa-int4. Ship-gates: fused-decode arm relative ppl ≤
  blend arm + 0.25%; int4 arm ≤ blend arm + 0.25%. Informational:
  each arm vs standard. Defaults flip only on green; David
  adjudicates the ship decision either way.

## 2026-08-13 (late) — G-C arms in flight; FA2B forked (David's call)

- G-C lead-run in progress (logs/fc_arms.log, wt/apamq-fb): standard
  arm complete ~67 min; blend/fused/int4 arms queued; ETA ~midnight.
- David: fork isolated worktree, run the tensor-core experiment,
  stash if bad / merge if good. FA1 work committed to branch
  apamq-fa @ a3336aa (branch-commit only; main untouched); worktree
  wt/apamq-fa2 forked from it (branch apamq-fa2, disposable).
- FA2B order committed + dispatched: Lane 1 = bf16-MMA fp32-accum
  refine pass (exact — bf16 products are fp32-exact; reassociation
  class, drop-in legal); Lane 2 = int8-Q × int4-K INTEGER bulk pass
  via dp4a/IMMA (integer sums exact; new operating point, distinct
  mode, a3336aa path untouched; NO bf16 score materialization
  anywhere). Perf target per plan G-A2: ≤3× cuBLAS @D=512 prefill
  16K; decode bandwidth-advantage hypothesis to be measured. SHIM-RUN
  20260814T014527Z-838434, Sol-max, sentinel armed. GPU legs
  lead-run after G-C arms free the card.

## 2026-08-14 (early) — G-C RESULTS + FA2B RESULTS (both lead-run)

- **G-C table (25 docs, 51,200 scored L=1 tokens/arm, engagement
  censused 409,600 APA decode calls/arm):** standard ppl 33.450 /
  25.3 ms/tok; apa_blend 32.489 / 29.9; apa_fused 32.472 / 38.1;
  apa_int4 33.943 / 37.8.
- **G-C ADJUDICATION: apa_fused PASS** (−0.053% vs blend ± 0.118 sem,
  ceiling +0.25%); **apa_int4 FAIL** (+4.477% ± 1.155, ~4σ over) —
  the in-kernel symmetric-int4 bulk quantizer is worse than the
  port's table-based 4-bit at this operating point. Ring-drop mode
  stays flag-OFF; registered follow-up = quantizer parity (port
  tables in-kernel or int6/group scales), NOT a redesign.
- **SURPRISE (named, unreconciled):** ALL APA arms beat standard by
  ~2.9% ppl on this protocol — opposite sign from June's +1.55–1.92%
  A0 regression. Different protocols (this: 2048 true-decode tokens
  against 8K mounted ctx; A0: other shapes). Both stand in their own
  protocols; reconciliation = named open question. Also: the 12.85
  max-logit blend-vs-fused divergence does NOT reach ppl (−0.05%) —
  zero-mean where it counts.
- Sober note: fused decode passes quality but is slower than blend
  (38.1 vs 29.9 ms/tok) and still ring-bound — by itself it buys
  nothing on today's kernel; its value was contingent on fast/no-ring
  variants.
- **FA2B verification: G-A2 FAILED decisively — STASHED** (David's
  fork protocol: stash if bad). D=512 kv=1 prefill 16K: scalar int4
  182 ms; bf16-MMA lane 346 ms; dp4a lane 649 ms; cuBLAS standard
  11.7 ms (target ≤35). Root cause with receipt: one-query-per-block
  skeleton caps m16n16k16 MMA utilization at ~6% real rows; dp4a
  ALU throughput irrelevant because the loop remains memory/latency
  bound (consistent with the A5-era sectors/request finding).
  **Measured design law: tensor-coring the selective kernel requires
  a multi-query-tile (FlashAttention-style) skeleton — ≥16 real query
  rows per fragment — not instruction substitution.** Plus 1/21 gate
  RED: int4 pack tie-break convention mismatch (1 nibble in 1216,
  round-half divergence kernel-vs-numpy) — declare + align convention
  in any successor. Branch apamq-fa2 preserved @ cc59fa0, UNMERGED.
- Open decisions for David (morning): (1) merge FA1 engine kernel to
  Project-Tensor main as inert entry (11/11 gates, nothing calls it
  by default); (2) FB wiring merge posture — code-with-flags-OFF vs
  hold branch (fused-decode default currently ON in branch code —
  needs a seat one-liner to flip before any merge); (3) whether to
  order the multi-query-tile F-A2c redesign or park the speed
  program; (4) quantizer-parity round for the int4 mode (would
  re-arm G-C int4 re-run); (5) synthesis + docs law rewrite timing.
