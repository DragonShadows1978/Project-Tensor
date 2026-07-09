# Trinity Nano (afmoe) Port — Implementation Plan

**Status: IMMUTABLE after initial commit.** Execution details and premise
corrections go in `docs/TRINITY_NANO_PORT_LEDGER.md`. Narrative meaning in
`docs/TRINITY_AFMOE_SYNTHESIS.md` (new wing). House rules per
`/mnt/Shared/HOUSE_RULES.md`.

## Intent

Port arcee-ai/Trinity-Nano-Preview (afmoe: 6.4B MoE, 1B active) to
tensor_cuda as `TrinityNano_TC`, then (separate order, GraftRepository)
drive it through APA and the GRM live loop. Chosen 2026-07-08 (David:
"biggest that fits" — Mini's 26B needs 13.8-14.6GB at INT4, does not fit
12.28GB resident; Nano fits with the entire 131k context in VRAM).

Config receipts (fetched 2026-07-08, cache JSONs in AtlasForge proxy):
56 layers (first 2 dense, 54 MoE: 128 experts moe_inter=256, 8+1/token,
sigmoid routing, route_scale 2.826), hidden 1024, 8Q/2KV heads,
head_dim 128, 3:1 sliding:full (global every 4th, window 2048),
max_position 131072, rope_theta 10000 (rope_scaling null), vocab 200192,
bf16, custom code (modeling_afmoe.py), muP-enabled.

## Registered predictions (frozen pre-port)

- **T1 (NoPE-graft):** IF P0 confirms full-attention layers apply no RoPE
  (NoPE, per the Trinity Large tech report): the GRM arena RoPE-hole law
  (GPT-OSS: clean ≤~115, collapse at 387) DOES NOT APPLY on those
  layers — grafts mount at any live_shift without generation collapse
  and without extend_rope. Confidence 0.7 conditional on NoPE
  confirmation. If NoPE holds and the prediction FAILS, that is a major
  finding about what the hole law actually measures (not position).
- **T2 (APA eligibility):** APA engages clean on the 14 full-attention
  layers (kv=2, head_dim 128, unbounded) per the multi-KV-head law and
  the external kv=2 validation; sliding layers are out of APA scope
  (bounded). Selective engagement per the ATTENTION POLICY geometry
  scanner rules.
- **T3 (full-context residency):** all-layers-resident INT4 weights +
  fp16 KV at 131,072 tokens fits 12,282 MiB (est. weights 3.5-5GB + KV
  ~2GB + workspace). If fp16 KV misses, INT8 KV (existing path) must
  close it. Receipt: nvidia-smi peak at full-context prefill.
- **T4 (parity law, binding as always):** TrinityNano_TC must reproduce
  the HF reference implementation's logits on identical input
  (deterministic engine ⇒ near-bit; gate registered at top-5 logit
  exact match minimum, max_abs_diff ledgered).

## Phases

- **P0 — Architecture scout (read-only, no GPU):** read
  modeling_afmoe.py + configuration_afmoe.py from the HF repo. Answers
  REQUIRED: (a) do full_attention layers apply RoPE or NoPE? (b) gated
  attention exact form (elementwise sigmoid gate location, per-channel?)
  (c) sliding layer RoPE handling; (d) MoE routing math (sigmoid score,
  route_norm, route_scale application, shared expert combine);
  (e) attention softmax details (QK-norm? zero-centered?); (f) dense
  layer 0-1 structure; (g) LICENSE. Deliverable: port map with
  file:line citations from the reference code.
- **P1 — Weights + reference capture:** download safetensors (~12.8GB),
  run HF reference (CPU or GPU bf16) on fixed probe prompts, capture
  logits + per-layer K/V for parity targets (stream_forward_smoke
  pattern from the GPT-OSS precedent).
- **P2 — TrinityNano_TC build:** model class on existing tensor_cuda
  primitives; new kernels ONLY where afmoe demands (gated attention
  epilogue; sigmoid router). INT4 weight path via existing quant
  kernels. Gates: T4 parity, incremental-decode==refeed, suite green.
- **P3 — Envelope + APA:** T3 residency receipt; APA selective
  engagement per T2 with the standard zero-flip/ppl gates at STANDARD
  vs APA-engaged.
- **P4 (separate order, GraftRepository):** GRM dialect wiring + T1
  NoPE-graft test — the arena width sweep replicated on Nano's full
  layers (the GPT-OSS law's experimental contrast).

Early-stop: if P0 finds the architecture diverges from config-implied
structure in a way that voids T1-T3 premises, stop and re-plan (this
plan stays immutable; correction goes to ledger).

Evidence classes as always: config receipt / code citation / gate run /
kernel bench / external report.
