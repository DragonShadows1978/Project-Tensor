# Gemma-4 12B / MQA Adjudication

**Operative verdict:** `google/gemma-4-12B-it` is an **Engine port** and
**Parity-gated** TensorCUDA validation point. Its APA result is **APA negative**.
The APA mission closed with that verdict on 2026-07-04. GRM certification is
`unconfirmed` in Project-Tensor's public receipts.

This note is the required reconciliation between the earlier paper-draft result
and the later APA-engaged measurement. The lead-supplied adjudication in the
[DOCS1 order](orders/DOCS1_public_doc_rework.md#gemma-4--mqa-adjudication-lead-supplied-binding)
is controlling.

## Exact target

- Model: `google/gemma-4-12B-it`.
- Transformer: 48 layers — 40 sliding-window layers with window 1024 and
  8 global MQA layers.
- Global attention geometry: one KV head, head dimension 512, and a shared
  `K=V` projection.

## Engine-port verdict

The engine port is complete and gated:

- parity against the QAT ground truth: top-1 agreement 70/80, with the
  disagreements adjudicated as near-tie flips;
- bit-identical state gate;
- approximately 28.5 tokens/s serving on the consumer test card;
- exact QAT `q4_0` import through symmetric-8 INT4 kernels.

Post-hoc INT4 group-128 quantization from bf16-origin weights collapsed. The
failure was isolated to pure quantization noise, so exact QAT import is the only
supported weight route for this engine port.

## Why the earlier paper result is not operative

The historical [APA paper draft](APA_PAPER_DRAFT.md#41-the-agnosticism-table)
reported near-tie perplexity readings from a 2048-window sweep. That scoring
protocol never engaged APA during scoring. The row is preserved as historical
provenance, but it is not an APA evaluation and must not be used as the current
verdict.

The 2026-07-04 re-probe used APA-engaged scoring on a 12GB card. Both standard
and APA modes completed without OOM at the full 32K trained window. Against the
standard mode, APA as built on this model measured:

| Measure | APA delta |
|---|---:|
| Peak VRAM | +0.4% to +3.3% |
| Prefill time | approximately 3.5x |
| Decode latency | +8% ms/token |
| Perplexity | +1.55% to +1.92% |

This is a net cost, so the operative label is **APA negative**.

## Mechanism law

**APA is a multi-KV-head mechanism.** Its statistics need key population, which
the sliding windows of at most 1024 keys do not supply. Its economics need KV
head multiplicity, while one shared KV head makes quantization noise coherent
across all 16 query heads. The global head dimension of 512 also exceeds the
fused kernel's supported head dimension, and MQA already spent the available KV
slack during training. Gemma-4 12B therefore fails both selection axes and is an
accidental anti-APA architecture.

The scope law is equally binding: **saving memory on storage is not APA.** APA
controls precision in attention computation. KV or weight storage quantization
is a separate residency decision and must not be presented as an APA result.

## Selection boundary

APA should be selected for multi-KV-head, sufficiently-long-context GQA-family
sites and then certified per model. The external boundary receipt is a 26B,
two-KV-head GQA model on which a collaborator reported clean APA operation and
64K context surviving where baseline attention OOMed. That result is labeled
**External receipt** and was not locally reproduced.

## Final labels

| Surface | Label | Basis |
|---|---|---|
| TensorCUDA | **Engine port** + **Parity-gated** | QAT parity, state, serving, and exact-import gates above |
| APA | **APA negative** | Engaged 32K re-probe; quality and cost gates failed |
| GRM | `unconfirmed` | No full lifecycle receipt in this repository |
