# Trinity Nano Port — Implementation Ledger

Receipts for the Trinity Nano port. Plan: docs/TRINITY_NANO_PORT_PLAN.md
(immutable). Synthesis: docs/TRINITY_AFMOE_SYNTHESIS.md.

## 2026-07-08 (opening)

Work order opened (David: "Lets try that Arcee Trinity - Biggest version
that'll fit on the card"). Sizing receipt: Mini (26B-A3B, config fetched)
needs 13.8-14.6GB at INT4-with-scales vs 12,282 MiB card — not resident;
expert-streaming rejected (7-15ms/token PCIe tax + weeks of work). Nano
(6.4B-A1B, config fetched via Trinity-Nano-Preview; bare Trinity-Nano
401s) selected. Predictions T1-T4 frozen in plan before any code.

Next action: P0 architecture scout (dispatched: Grok, read-only).

## 2026-07-08 (P0 complete — NoPE CONFIRMED; no blockers; license clear)

Action: architecture scout (Grok, read-only, code citations) — full
findings docs/TRINITY_NANO_P0_FINDINGS.md (fold-in accepted).

- **T1 CONDITION HOLDS: full_attention layers are NoPE** — RoPE applied
  only when layer_types[i]=="sliding_attention" (modeling_afmoe.py ~326,
  374-376; no else branch). 14 full layers at indices 3,7,...,55.
- Port map (parity-critical): gated attention = sigmoid(gate_proj(h))
  ⊙ attn_out before o_proj, elementwise heads×head_dim; MoE router =
  sigmoid → topk(scores+expert_bias) → gather → route_norm → ×2.826 →
  shared expert + scatter_add (expert_bias in top-k SELECTION only);
  QK-Norm standard RMS (ones-init, NOT zero-centered); softmax fp32;
  scale 1/√128; muP embed ×√1024; untied lm_head; dual pre/post
  residual norms; dense layers 0-1 SwiGLU inter=3072; sliding RoPE
  θ=10000 no scaling, window 2048.
- License: OpenMDW-1.1 (permissive, retain notices). Download: 3 shards,
  12.24 GB. Early-stop rail NOT triggered.

Next action: weights download (started), then P1 reference capture.
