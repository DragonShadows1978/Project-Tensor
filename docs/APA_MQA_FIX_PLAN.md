# APA × MQA Fix Plan (APAMQ-F)

Date: 2026-08-13. Lead: Fable session. David's directive: "So lets fix it."
Successor to the (immutable) investigation plan
`docs/APA_MQA_ROOTCAUSE_PLAN.md`; grounded in its E1 result (T1
CONFIRMED: transient win KV-independent), the E3 audit (T3/H-D
CONFIRMED static: fused decode legal-but-unwired; kqb ring 144 MiB @16K
vs ~64 MiB eliminable transients — APA-as-built memory-negative), and
the E1 speed finding (fused kernel 6–17× slower: CUDA-core dots vs
cuBLAS tensor cores). E3 live attribution cells are in flight and will
refine magnitudes; they do not change the fix targets.

## What "fixed" means

APA on Gemma-4 12B (MQA kv=1, D=512 globals) runs with: (1) resident
memory ≤ standard mode (ring tax gone), (2) fused path engaged at
decode as well as prefill, (3) prefill/decode wall-clock within a
registered factor of standard, (4) no quality regression beyond the
registered gate at the chosen operating point. kv=1 was never the
blocker (E1); these are implementation fixes.

## Scope guard

K/V caches stay bf16. This plan eliminates/compresses APA's own
DERIVED scratch state (the kq bulk keys) — it does NOT touch stored
K/V residency. The declined B4/storage-quant arc stays declined.

## Workstreams

- **F-A (engine, Project-Tensor, hard):** causal/selective INT4-packed
  APA kernel family. New entry `apa_selective_attention_int4(q, k, v,
  scale, zthr, is_causal)` — signature mirrors `apa_selective_attention`
  MINUS the kq tensor: the kernel packs K to INT4 per key vector
  internally (reusing the `apa_int4_pack_kernel` / `apa_int4_dequant2`
  machinery from the noncausal DiT arc at kernels.cu:4594+), scores
  bulk from the packed form (fp32 dequant — higher-fidelity bulk than
  the current bf16-reconstructed ring), refines the tail from exact K,
  online softmax, bottom-right causal, GQA/MQA-aware. Ports then need
  NO kq ring at all. Includes decode split-K coverage mirroring the
  existing selective family's dispatch.
  Second rung (F-A2, speed): tensor-core/tiled bulk pass — attack the
  6–17× gap. Registered target ≤3× cuBLAS-standard wall at D=512
  prefill S=16K (from 16.6×); stretch = parity. Honest STOP if the
  rung doesn't land inside the leash — F-A1 correctness ships on its
  own value (memory-positive APA) regardless.
- **F-B (port, GraftRepository):** (1) wire Gemma APA DECODE to the
  fused path (the H-D fix) behind the same S>lever as prefill, with
  the blend retained as fallback; (2) integration scaffold for
  `apa_selective_attention_int4` behind an env flag
  (`GEMMA4_APA_INT4=1`), dropping the kqb ring when active; stubbed
  against the pre-nailed signature until F-A lands, then activated.
- **F-C (quality, registered, dispatched after F-A/F-B):** bulk-bits /
  operating-point ladder at D=512 with APA-engaged scoring (the E2′
  ladder; INT4-internal path replaces "bits" with pack-group/refine
  dials). Closes the +1.55–1.92% ppl question against the √D-noise
  hypothesis.

## Registered gates (before results)

- **G-A1:** int4-causal selective kernel matches (a) a composed fp32
  reference and (b) the existing bf16-kq selective kernel on
  rect-causal with-cache shapes including MQA kv=1 D=512 — selection
  behavior and output tolerance gates in the style of
  test_apa_selective.py + the EXP-APA-2 int4 conventions; explicit
  non-square S>L coverage (the 121→11M bug class).
- **G-A2:** perf rung target ≤3× cuBLAS standard at D=512 prefill
  16K; measured by the committed E1 harness (scripts/apamq_e1_sweep.py
  extended with the int4 path). Miss = honest report, not a fudge.
- **G-B1:** Gemma APA decode dispatches fused/int4 (branch census
  shows it); decode quality parity gate in the style of
  tests/gemma4_apa_incremental.py; decode peak ≤ standard + 20 MiB
  at S=16K.
- **G-B2:** with the ring dropped: APA-mode peak ≤ standard-mode peak
  at S=16K, one-mode-per-process, pool + nvml (the E3 harness
  measures it). This is the "memory-positive or neutral" gate that
  reverses E3's static −65 MiB.
- **G-C:** registered when F-C dispatches (needs the F-A/F-B stack to
  exist first).

## Discipline

Seats: Sol-max via codex-shim, one per workstream, PARALLEL, each in
its own git worktree branch (`apamq-fa` off Project-Tensor main,
`apamq-fb` off GraftRepository main) — the installed engine .so and
main-tree sources stay untouched while E3 live cells run. Codex
sandbox has no GPU (proven today, 2 receipts): seats deliver code +
CPU-side builds + test files; ALL GPU gates are lead-run. Lead
verifies, merges, commits; seats never run git. Evidence classes:
kernel gates = kernel sweep/unit test; port gates = instrumented port
measurement; F-C = model perplexity with engaged scoring.
