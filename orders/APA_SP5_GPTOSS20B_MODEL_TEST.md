# APA-SP5 — the third architecture: single-pass APA inside GPT-OSS-20B (MoE, GQA 64/8 × D=64, attention sinks, 128-token sliding + full layers)

David, 2026-09-08: "let's try GPT-OSS-20B next." Seat: Opus 5 (this
order is dispatched to the `opus-max` seat; it has GPU access, unlike
the Codex sandbox, so it builds AND runs its own short gates on the
card under the lease rules below; the lead runs the long chain).

## What the first two architectures established (read the reports first)

- MiniCPM3-4B (`/mnt/Shared/APA_SP3_Model_Test_Report_2026-09-07.md`,
  branch `apa-sp3`): composite MLA keys, noisy 4-bit bulk (mean 0.25,
  max 2.9–105): the running-max single-pass tail beats the z-score
  two-pass tail by 0.12 ppl at equal 10 % fraction because it selects
  softmax mass (z-score leaves 92 % unrefined); provable δ vacuous;
  δ transfers across length as a mass invariant; refine-all == standard
  to 0.0005; APA is the only path past 4K prefill on 12 GB.
- Gemma 4 12B (`/mnt/Shared/APA_SP4G_Gemma4_Model_Test_Report_2026-09-08.md`,
  branch `apa-sp4g`): clean qk-normed 512-dim MQA keys (bulk mean 0.11):
  every tail is free; provable δ USABLE (69 % refined at standard's
  ppl); the model amplifies rounding-level differences ~30× so
  exactness is a PER-CALL property (passes at 1e-6) and model-level ppl
  carries a ±2.56 noise floor measured as standard-bf16 vs
  standard-fp32; no memory ceiling under 64K (standard fastest on
  prefill because its chunked path never materializes S×S).
- Principles to carry: (1) tail-choice penalty scales with bulk error;
  (2) provable δ is model-dependent; (3) measure bf16-vs-fp32
  sensitivity BEFORE any sub-percent ppl claim; (4) APA's memory lever
  exists only where the dense path materializes S×S; (5) exactness is
  gated per call on identical inputs, never at model level on
  numerically chaotic models.

## The model

`openai/gpt-oss-20b`, snapshot
`~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee`,
port `GraftRepository/core/gpt_oss20b_tc.py` (READ-ONLY): 24 layers,
64 query heads over 8 KV heads, D=64, alternating 128-token sliding
and full-attention layers, **learned attention sinks** (the engine's
`sink_attention_tc` / `apa_selective_attention_sink` paths), MoE with
resident MXFP4 experts, streamed forward
(`scripts/gpt_oss20b_stream_forward_smoke.py --score-ppl`,
`scripts/gpt_oss20b_realtext_ppl_gate.py` — a 64-token smoke, NOT a
protocol). The kernel-opt bench passed 96K context on this card
(2026-07-08) with sink-APA. APA scope is `full` (full-attention layers
only; sliding layers are bounded at 128 keys and stay standard).
This is the first model test with sinks in the softmax denominator
and with GQA 8:1 at D=64 — the shapes SPD1 benchmarked synthetically.

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor-wt-apa-sp5`
(branch `apa-sp5`, forked from `apa-sp4g` so every `scripts/apa_sp4g_*`
and `apa_sp3_*` module, the leased runner pattern, the fingerprint /
amendment machinery and the per-call capture code are yours to copy
or import — never edit them). Writable: `scripts/apa_sp5_*`,
`tensor_cuda/tests/`, `artifacts/apa_sp5/`, `logs/`,
`docs/APA_SP5_LEDGER.md`, and, ONLY if a sink/GQA shape is not already
instantiated on both SP kernels, the minimal hash-pinned launcher
addition (`TC_APA_SP` stays default OFF; every pre-existing kernel body
byte-identical). READ-ONLY: everything else here, GraftRepository, the
HF cache. The 4070 SUPER 12 GB is SHARED with other projects: `flock
--wait` on `/tmp/forge-gpu.lock`, ≤ 285 s worker / 590 s outer per
job, 30 s cooldown, never kill a process you did not start, one
foreground call < 10 minutes, no background waits. Claude Code reaps
YOUR background tasks on a page-cache quirk — do not use them.

## Mission (in this order; register before each gate)

0. **Cost of a window.** Measure the wall of one 1024-token window
   through the streamed forward on the card BEFORE registering the
   protocol; the MoE streamed path may be far slower than Gemma's
   (Gemma 12B: 2048-window ≈ 110 s + 75 s load). Choose PROTOCOL-O so
   every cell fits the 285 s rail: wikitext-2-raw-v1 test from the
   offline HF cache, the model's own tokenizer (harmony/o200k; no chat
   template), canonical int64 stream sha-pinned; N consecutive windows
   of W tokens scoring the last W/2 in fp64, W and N chosen from the
   measured wall (prefer W=2048, N=4; fall back to W=1024, N=4; state
   the choice and why); long rows prefix 0, last 512 within 8192 if it
   fits the rail, else registered non-fit.
1. **Sensitivity first (principle 3):** standard in bf16 vs standard
   with full-attention layers in fp32 on window 0. That number IS the
   noise floor for every model-level comparison in this order.
2. **Arms** (bulk 4 primary; bulk 8 secondary if wall permits), APA on
   full-attention layers only: A standard (sink attention); B two-pass
   `apa_selective_attention_sink` at the port's registered refine
   percentile (state it); C single-pass with ONE global δ matched to
   B's realised fraction on the full layers ±0.01; D single-pass
   refine-all (exact); E single-pass provable δ with e_q measured on
   THIS model's real keys. **Sinks:** state exactly how the sink logit
   enters each arm (standard folds it into the denominator; the SP
   entry takes `sinks` and folds it at the end) and pin it: a per-call
   comparison on one full layer with sinks on vs zeroed.
3. **Exactness, per call (principle 5):** identical-input comparison on
   one full layer for a prefill chunk and a cached short block (SP vs
   standard vs dense-fp32 NumPy, fp32 and bf16), max-abs and
   rel-Frobenius; gate: ≤ 1e-4 max-abs / ≤ 1e-5 relF in fp32. Model-
   level D vs A is reported against the floor from item 1, not gated.
4. **Margins on the full-attention layers** (real activations, bitwise
   replay via captured masks as SP4G a2 does), arms B and C at W and
   at 8192 if it fits: |bulk−exact| mean/p99/max, unrefined softmax
   mass, max relative weight of a skipped key, fraction — AND the sink
   mass per query (how much of the softmax goes to the sink), which no
   previous model had.
5. **Clean decode** (June flags, incremental key cache if the port has
   one — say whether it re-quantizes per step), A/B/C at 2048 and 8192,
   32 synced steps, ms/token.
6. **Ceiling** on 12 GB at {4096, 8192, 16384} per arm under the
   normal rail; register the long-lease rungs (24K–96K) as cells the
   lead can run later if David authorizes the 1,500 s lease again; the
   kernel-opt bench's 96K result is context.
7. Registered predictions (lead), yours beside them BEFORE the card:
   P1 per-call SP == standard ≤ 1e-5 relF in fp32 including sinks;
   P2 bulk error on D=64 GQA keys at 4 bits sits between MiniCPM3 and
   Gemma (mean 0.15–0.25) and the z-score tail leaves 50–80 % of mass
   unrefined; P3 C − B is negative or inside the floor; P4 E refines
   40–90 % (provable δ usable, as on Gemma); P5 the sink takes ≥ 10 %
   of softmax mass on a median full layer and the running-max rule's
   selection is unaffected by it (sink is not a key); P6 no OOM by 16K,
   standard fastest on prefill again unless the port materializes S×S
   on full layers (say which from the code before the card).

Registration first (`artifacts/apa_sp5/registration.json`, IMMUTABLE,
citing the SP3/SP4G registration shas), amendments separate,
create-only fingerprinted receipts, `lead_commands.txt` in dependency
order with measured-or-estimated walls, `RESULTS.md` writer with
evidence classes stated and the sentence that margins/ceilings/decode
establish nothing about model quality by themselves. You MAY run on
the card yourself: kernel pins, item 0, item 1, item 3, and single
cells of the others to validate plumbing — each under the lease rules;
leave the full chain to the lead's detached loop. Prior Art Directive
(HOUSE_RULES): cite the GPT-OSS sink attention (OpenAI 2025 model
card), BLASST, ThriftAttention, FlashAttention-2, TurboQuant, SP3/SP4G.
NO git (the lead commits). NO subagents. RED honesty; a finding is not
a failure.

## Done (verbatim in your final message)

1. Measured window wall; PROTOCOL-O as registered (W, N, token count,
   sha); the sensitivity floor from item 1.
2. Arm registry with the sink handling per arm; C δ rule; E e_q rule;
   kernel/shape pins by test name; what you ran on the card yourself
   (cell ids + results).
3. Registration sha; both prediction sets.
4. CPU/GPU gate results; `lead_commands.txt` for the chain with walls.
5. Prior art; deviations; residual risks; RED; process safety; model
   id and reasoning effort actually used.
