# APA-SP4G — the second model: single-pass APA inside Gemma 4 12B (QAT INT4), perplexity against standard and two-pass, margins on the global layers, ceilings, clean decode

David, 2026-09-07: "let's test on something else then. Gemma 4, that's
a good test subject." SP3 (`artifacts/apa_sp3/RESULTS.md`,
`/mnt/Shared/APA_SP3_Model_Test_Report_2026-09-07.md`) established on
MiniCPM3-4B (MLA, composite 96-dim keys) that the single-pass
running-max tail beats the two-pass z-score tail at equal refine
fraction because it selects softmax mass, that the provable δ is
vacuous at 4 bits, that δ transfers across length as a mass invariant,
and that in-model decode must be measured in the June configuration.
This order asks whether all of that survives a different architecture:
Gemma 4 12B, 48 layers = 40 sliding-window (1024, D=256, 8 KV heads)
+ 8 global MQA layers (1 KV head, D=512, K=V shared projection,
p-RoPE), qk-normed keys ("bulk 4 free" family), 16 query heads,
256K trained window. APA runs on the GLOBAL layers only.

YOUR WRITABLE TARGET is the worktree you were dispatched into
(`/mnt/ForgeRealm/Project-Tensor-wt-apa-sp4g`, branch `apa-sp4g` forked
from `apa-sp3`, so every `scripts/apa_sp3_*` module, the leased runner,
the fingerprint/amendment machinery and the SP3 receipts are present
read-only for reuse; copy or import, never edit them). Writable:
`scripts/apa_sp4g_*`, `tensor_cuda/tests/`, `artifacts/apa_sp4g/`,
`logs/`, `docs/APA_SP4G_LEDGER.md`, and, only if D=512 is not already
instantiated for BOTH SP kernels, the minimal hash-pinned addition to
the SP launchers (every pre-existing kernel body byte-identical;
`TC_APA_SP` default OFF). READ-ONLY: everything else here;
`/mnt/ForgeRealm/GraftRepository` (`core/gemma4_tc.py` the adapter,
`tests/gemma4_bulkbits_floor.py` the June protocol,
`tests/gemma4_kv_quant_apa_ppl.py`, `docs/GEMMA4_PORT_LEDGER.md`,
`docs/GEMMA4_APA_AUDIT_A1.md`); the models
(`/mnt/ForgeRealm/models/gemma-4-12B-it` tokenizer + bf16,
`/mnt/ForgeRealm/models/gemma-4-12B-it-qat/gemma-4-12b-it-qat-q4_0.gguf`
= the engine's QAT INT4 weights via `load_weights_qat`). Sandbox has
NO GPU: build, CPU-verify, blocked-report; the lead runs the card
(RTX 4070 SUPER 12 GB, `free` shows ~6 GB free / 58 GB available; the
harness reaps background tasks on low FREE memory, irrelevant to you).

## Adapter facts you must verify, not assume

- APA engages on global layers past `apa_min_context` (2048 in the
  adapter; the floor script forces 0 so APA runs on every window).
  State which you use per cell and why; the two-pass arm at S=2048
  windows needs 0 to exercise APA at all.
- The two-pass arm is the adapter's fused `apa_selective_attention` /
  `_int4` path at `bulk_bits=4, refine_percentile=0.15`
  (`core/gemma4_tc.py` ~596/609/682/712). Hook C/D at the SAME call
  site with the SAME q/k/kq/v tensors; if the V or K layout differs
  between arms (as B=blend/V64 vs C=V96 did on MiniCPM3) say so in the
  registration, not after.
- MQA: 1 KV head, 16 query heads, D=512. Verify the SP prefill kernel
  and the split-K SP kernel accept D=512 and KV=1 (cap ladder in
  `apa_sp1_1.cuh` has a 512 rung; the prefill SP launcher near
  `kernels.cu:7441` must be checked), pin with tests like SP3's
  `test_d96_*`.
- Decode: does the adapter re-quantize the global-layer keys every step
  (MiniCPM3's a6 finding) or keep `kq_count` incremental? Say which;
  if it re-quantizes, decode receipts are labelled adapter-bound.

## PROTOCOL-G (register before anything runs)

wikitext-2-raw-v1 test from the offline HF cache, joined as the floor
script joins it, tokenized with the `-it` tokenizer, canonical int64
stream saved + sha-pinned. Short rows: four consecutive 2048-token
windows scoring the last 1024 targets (the June floor protocol,
`WINDOW, SCORED, N_WINDOWS = 2048, 1024, 4`), fp64 log-softmax, one
feeding scheme for every arm (state it). Long rows: prefix 0, last-512
within the input at S ∈ {8192, 16384} and 32768 if it fits. Sliding
layers see ≤1024 keys and never touch APA; the ppl difference between
arms is the global layers' doing — say so in the table header.

## Arms

A standard (engine, QAT INT4). B two-pass bulk 4 r=0.15 (the June
setting). C single-pass, ONE global δ matched to B's realised fraction
on the global layers ±0.01 (SP3's rule). D single-pass refine-all
(must equal A; the in-model exactness pin at D=512/MQA). E single-pass
provable δ with e_q from THIS model's real global-layer keys. No torch
reference: 12B bf16 is 24 GB and does not fit; A is the reference and
say so. Bulk 8 secondary if wall permits.

## G2 margins — global layers only (8 layers, cheap)

Per global layer at S=2048 and S=8192, arms B and C: |bulk−exact|
mean/p99/p99.9/max on real keys, softmax mass on unrefined keys, max
relative weight of a skipped key, realised fraction. This is the test
of SP3's mechanism on a second architecture.

## G3 ceilings + clean decode

Ceiling grid {4096, 8192, 16384, 24576, 32768} per arm, prefill from
the pinned prefix, fit/OOM/rail + resident. Clean decode (the a6
lesson: June flags, pooling before load, no wrappers, no interposer,
no per-step host copies beyond the argmax) A/B/C at 2048 and 8192, 32
synced steps, ms/token; 32768 if it fits the rail.

## Registered predictions (lead) — register yours beside them BEFORE the card

- P1: D equals A within 0.005 ppl at D=512/MQA.
- P2: C ≤ B + 0.02 ppl at bulk 4 on the short rows, and the gap
  B − C is smaller than SP3's 0.118 (qk-normed keys make the bulk more
  accurate, so the tail choice matters less).
- P3: on every global layer, unrefined softmax mass under B ≥ 0.8 and
  under C ≤ 0.3 at matched fraction (the mechanism reproduces).
- P4: E refines ≥ 0.95 of keys (real-key e_q ≥ 0.5 logit units on
  512-dim keys).
- P5: single-pass ceiling ≥ two-pass ceiling ≥ standard ceiling on
  12 GB, and standard fails by 16384.

## Rules (binding)

Registration first (`artifacts/apa_sp4g/registration.json`, IMMUTABLE,
citing SP1/SP1.1/SP2/SP3 registration shas), amendments separate.
Leased runner `scripts/apa_sp4g_lead_gpu.sh list|run CELL|resume|summary`
on `/tmp/forge-gpu.lock --wait`, every job ≤ 290 s worker / 590 s
outer, 30 s cooldown, never kills anything; receipts create-only,
fingerprinted per kind as SP3 a4 does; `lead_commands.txt` in
dependency order with wall estimates; `RESULTS.md` writer with
evidence classes stated (model perplexity / kernel sweep) and the
sentence that G2/G3 rows establish nothing about model quality by
themselves. Prior Art Directive (HOUSE_RULES) applies: cite the June
Gemma 4 port ledger and floor protocol, BLASST, ThriftAttention,
FlashAttention-2, TurboQuant, and SP3. NO git. NO subagents. NO
background waits. Never kill a process you did not start. RED honesty.

## Done (verbatim)

1. Adapter facts verified (engage threshold used, call-site parity for
   B/C/D/E, D=512/MQA pins by test name, decode re-quantization answer).
2. PROTOCOL-G: token count + sha, feeding scheme, window scheme.
3. Registration sha; both prediction sets.
4. CPU gate results; GPU blocked-report with exact lead commands and
   per-job wall estimates (the 12B load alone: estimate it).
5. Prior art; deviations; residual risks; RED; process safety; model
   id and reasoning effort.
