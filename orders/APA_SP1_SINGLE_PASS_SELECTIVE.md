# APA-SP1 — Can the selective APA kernel's bulk pass and refine pass be one pass?

David, 2026-09-05: "a really hard thing is to figure out how to do the
bulk pass AND refine pass in a single pass instead of two — maybe not
even possible." This order is that question, end to end: theory,
kernel, receipts. A proof of impossibility with a receipt is a full
success. A single pass that is exactly equivalent is a full success.
A single pass under a DIFFERENT selection rule, honestly gated, is a
success if its gates say so. Fabricated equivalence is the only
failure.

YOUR WRITABLE TARGET is the worktree you were dispatched into
(`/mnt/ForgeRealm/Project-Tensor-wt-apa-sp1`, branch `apa-sp1` forked
from `main`). Edits, builds, and CPU tests AUTHORIZED. **Your sandbox
has NO GPU** (`cudaMalloc … no CUDA-capable device`): you build and
CPU-verify; every GPU gate is delivered as a runnable, leased, bounded
script plus a blocked-report, and the LEAD runs it on the card. Design
for that from the start.

## Context (read fully, in order)

1. `docs/APA.md` (mechanism §1: "a low-precision bulk pass ranks keys,
   a threshold selects the tail, and that tail is refined at full
   precision"; §5–6 boundaries and the certification rule).
2. `tensor_cuda/src/kernels.cu` — `apa_selective_kernel` (~1038–1350):
   **Pass 1** walks ALL keys computing `bulk_j = q·k_quant_j·scale` and
   accumulates `sum|bulk|`, `sum bulk²` → z-score threshold
   `thr = mean + z·std` (z from the host-side `_norm_ppf` of
   `refine_percentile`). **Pass 2** walks all keys again, recomputes
   `bulk_j`, and if `|bulk_j| ≥ thr` replaces it with the exact
   full-precision dot (refine), then online-softmax. Also the split-K
   family (~1355–1440: stats kernel = pass 1 over the full key range,
   partition kernels = pass 2), the sink variants, the decode-shaped
   variants, and the launcher dispatch. `include/tc/ops.h`, the Rust
   selective reference the comment names, and the tests
   `tests/test_apa_selective*.py`, `test_selector_accuracy.py`,
   `test_qtile_attention.py`, `test_apa_int4_sdpa_noncausal.py`.
3. `docs/KERNEL_OPT_IMPLEMENTATION_PLAN.md` (house rules: plan
   immutable, ledger, synthesis; **every claim names its evidence
   class; a kernel sweep establishes ONLY speed, memory shape,
   reconstruction error, and output deviation vs a dense reference —
   never model quality**), `docs/KERNEL_OPT_SYNTHESIS.md`,
   `docs/APAMQ_FA2_TENSOR_CORE_STOP.md` (a stopped rung: know why),
   `docs/QUANT_SWEEP_IMPLEMENTATION_PLAN.md` §house rules.
4. `/mnt/Shared/HOUSE_RULES.md` §8/§9 (seat laws). "APA is never
   overwritten": the existing `apa_selective_kernel` and its family
   stay byte-identical; anything new is a NEW kernel behind a NEW
   flag, default OFF.

## The question, precisely

The refine decision for key j depends on `thr`, and `thr` depends on
statistics over ALL keys' bulk scores. That is the dependency that
forces two walks. Three sub-questions, answer each with a receipt:

**Q1 — Exact single pass under the z-score rule.** Is there a single
walk over the keys that produces output bit-identical (or
float-equivalent within the existing tests' tolerance) to the two-pass
z-score kernel for all inputs? Either give the construction, or give
the argument why none exists (e.g. a two-input family where the
correct refine set for key 1 depends on key S). A clean impossibility
argument with a concrete counterexample the CPU reference can check is
a full answer.

**Q2 — Exact single pass under a different, order-safe rule.** Online
softmax already maintains a running max `m`. Consider selection rules
whose decision for key j depends only on prefix statistics
(`m`, running sums) available when j is visited, such that a decision
already made never becomes WRONG when later keys arrive (it may become
unnecessarily conservative — that costs refine work, not correctness).
Characterize which rules have this monotone property, pick one, and
show it is exactly single-pass. Register, BEFORE measuring, how its
selected-tail relates to the z-score tail (fraction refined, overlap).

**Q3 — Does it pay?** For the Q2 rule (and Q1 if it exists): a kernel
sweep on the registered shapes (prefill: L=S∈{512,2048,8192}, D∈{64,
128}, causal and non-causal, GQA where the existing tests use it;
decode: L=1, S∈{2048,8192,32768}) reporting speed vs the two-pass
kernel, memory shape, and **output deviation vs the dense fp32
reference** at matched refine fraction. Evidence class: kernel sweep.
Say the phrase "this establishes nothing about model quality" in the
report where the numbers are.

## Mission

1. **CPU reference first** (`tensor_cuda/tests/apa_sp1_reference.py`):
   a numpy emulator of the existing two-pass z-score kernel (pin it
   against the current kernel's own tests / the Rust reference at the
   tolerance they use), plus emulators of your candidate single-pass
   rule(s). All Q1/Q2 claims are checked here first, including the
   counterexample family for Q1 and a randomized equivalence sweep
   (≥ 10⁴ random (q, K) draws per shape class, seeded, registered).
2. **Kernel** (`tensor_cuda/src/kernels.cu` additions ONLY —
   `apa_selective_sp_kernel` + launcher + ops.h entry + a `TC_APA_SP`
   flag default OFF; the existing kernels byte-identical, test-pinned
   by hash). Same sink handling, same causal bounds, same GQA mapping.
   If a split-K variant is needed for decode shapes, say why and add
   it; if the single pass makes split-K unnecessary, say why.
3. **Gates**, all as runnable scripts, CPU ones run by you, GPU ones
   delivered for the lead:
   - G1 (CPU, you): pytest over the reference + existing APA tests
     green; existing-kernel hash pins; Q1 counterexample check; Q2
     monotone property check (a test that perturbs later keys and
     asserts earlier decisions never flip to wrong).
   - G2 (GPU, lead): bit/float equivalence of the new kernel vs its
     own CPU emulator on the registered shapes (seeded inputs).
   - G3 (GPU, lead): the Q3 sweep table, with the registered
     predictions scored.
   Provide `scripts/apa_sp1_lead_gpu.sh` (one leased, ≤10-min-bounded
   invocation per shape class; `/tmp/forge-gpu.lock` via the repo's
   flock convention; single GPU; ≥30 s gaps) and a `resume` mode.

## Registered predictions (lead, before any run)

- P1: Q1 is IMPOSSIBLE in general — the z-score threshold is a global
  statistic; a counterexample exists where the refine set for an early
  key flips on the value of the last key.
- P2: a running-max-relative rule (`refine iff bulk_j ≥ m_running − δ`)
  is monotone-safe and admits an exact single pass; at matched refine
  fraction its selected tail overlaps the z-score tail by ≥ 80% on
  prefill shapes, less on decode.
- P3: output deviation vs dense of the Q2 rule is within 2× the
  two-pass kernel's on ≥ half the registered shapes; speed-up
  ≥ 1.4× on prefill shapes (pass 1's full bulk walk is gone), smaller
  on decode where split-K already parallelizes it.
Register your own predictions alongside these BEFORE measuring, in
`artifacts/apa_sp1/registration.json` (IMMUTABLE; amendments in a
separate file citing its sha256).

## File boundary

Modify ONLY: `tensor_cuda/src/kernels.cu` (additions), `tensor_cuda/include/tc/ops.h`
(one entry), the Python binding/launcher file that exposes the existing
selective kernel (additions only; name it), `tensor_cuda/tests/`
(new files + the hash-pin test), `scripts/apa_sp1_*`,
`artifacts/apa_sp1/`, `docs/APA_SP1_LEDGER.md` (yours: commands,
results, failures, as they happen). READ-ONLY: every existing kernel
body, `docs/*PLAN*.md`, all other docs, all receipts. NO git (the lead
commits). NO subagents. NO background waits (foreground with explicit
`timeout`, every call under 10 minutes). Never kill a process you did
not start; multiple projects share this machine.

## Principles (binding)

Evidence classes on every number. Thresholds/δ registered before the
gate they govern, never adjusted after. Plan immutable (this order is
the plan); ledger as you go. RED honesty: "impossible, here is the
counterexample" and "possible under rule R, here is the cost" are both
results; "equivalent" without a bit-level receipt is not. Verify your
own claims against artifacts before writing them.

## Done (verbatim in the final message)

1. Q1 verdict with the counterexample (or construction) and the CPU
   check that pins it.
2. Q2: the rule, the monotone proof sketch, the CPU equivalence sweep
   result, the registered tail-overlap prediction vs measured.
3. Kernel: files/lines added; existing-kernel hash pins; the flag.
4. G1 results; G2/G3 delivered as blocked-reports with the exact lead
   commands.
5. Registration sha256; predictions (lead's and yours) hit/miss on
   everything measurable without a GPU.
6. Ledger path; deviations; residual risks; anything RED; process-
   safety acknowledgement; model id and effort.
