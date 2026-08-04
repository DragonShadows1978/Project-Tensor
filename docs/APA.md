# Adaptive Precision Attention (APA)

## 1. Mechanism

> **Adaptive Precision Attention (APA) scores every key, but spends full
> precision only on the softmax-dominant tail.** The remaining keys retain
> low-precision scores and still participate in the exact softmax denominator;
> no keys are dropped. APA is implemented at the attention score/softmax
> boundary and has been evaluated across causal and bidirectional attention,
> MHA/[MQA](GEMMA4_MQA_ADJUDICATION.md)/GQA/MLA organization, MoE models, NoPE
> and sliding hybrids, and a visual diffusion transformer. Results are
> model-gated, and negative results define the supported envelope.

A low-precision bulk pass ranks keys, a threshold selects the tail, and that
tail is rescored at full precision before the merged scores enter softmax. The
value path remains full precision in the evaluated implementation. The
insertion point is architecture-neutral, while useful bulk bits, thresholding,
and refine policy remain model- and geometry-dependent. Every new target
therefore owes an engagement check plus quality, memory, and speed gates.

## 2. What APA is not

- It is not sparse attention: every key remains in the softmax denominator.
- It is not KV compression: the precision used for stored KV residency is a
  separate decision from score precision during attention computation.
- It is not GRM: APA changes attention-score computation, while GRM persists,
  routes, and mounts model-native attention state.
- It is not a universal benefit claim. A common insertion point does not imply
  that every model has a profitable operating point.

## 3. Evaluated attention families and modalities

The named targets are evaluation points, not a compatibility whitelist.
“Attention families” describes MHA, [MQA](GEMMA4_MQA_ADJUDICATION.md), GQA,
and MLA organization; it does not count models.

| Attention family or modality | Evaluation targets | Evidence class |
|---|---|---|
| MHA, causal text | GPT-2 | **APA negative** / abandoned PoC in the [historical paper](APA_PAPER_DRAFT.md#44-mha-the-abandoned-proof-of-concept) |
| [MQA + sliding, causal text](GEMMA4_MQA_ADJUDICATION.md) | [Gemma-4 12B](GEMMA4_MQA_ADJUDICATION.md) | **APA negative**, adjudicated from an engaged 32K re-probe |
| GQA/hybrid, causal text | Qwen3.5-9B; TinyLlama-1.1B; external 26B kv=2 | **APA positive**, **APA boundary**, and **External receipt**; see [matrix](SUPPORT_MATRIX.md) |
| MLA, causal text | MiniCPM3-4B; GRAPA-232M | **APA positive** historical evaluations, including native training; see [paper](APA_PAPER_DRAFT.md#41-the-agnosticism-table) |
| MoE text | OLMoE-1B-7B; GPT-OSS-20B; Trinity Nano | **APA boundary**, later **APA positive**, and **APA evaluated** / split results; see [matrix](SUPPORT_MATRIX.md) |
| NoPE/hybrid text | Trinity Nano | **APA evaluated** / **APA boundary**: engaged on 14/14 eligible full layers, but token identity failed in the [ledger](TRINITY_NANO_PORT_LEDGER.md#t2--apa-selective-engagement--s8192) |
| Bidirectional visual DiT | Hunyuan3D-2.0 shape transformer | **APA positive** E2E receipt in the [visual summary](APA_VISUAL_TOKENS_PRIMER.md#3-the-modality-jump-coldcast-2026-07-1314) |
| Bidirectional visual DiT + MoE | Hunyuan3D-2.1 | **Planned / unconfirmed**; the 2.0 result does not transfer |

## 4. Positive results

- MiniCPM3-4B's historical MLA evaluation reports 4-bit bulk scoring, a 0.10
  refine setting, and a 3K-to-32K context extension. GRAPA-232M supplies the
  separate “trained from birth” existence result. Both are preserved in the
  [paper's original evidence table](APA_PAPER_DRAFT.md#41-the-agnosticism-table).
- Qwen3.5-9B's historical GQA evaluation reports zero top-1 flips at its
  registered point; provenance remains the [paper draft and its evidence
  index](APA_PAPER_DRAFT.md#appendix-a--evidence-index-paths-for-the-authors-strip-before-submission).
- Hunyuan3D-2.0 is the non-causal, non-text E2E result: 14/14 meshes were
  watertight, and the promoted fused path measured 199.9 seconds versus 242.8
  seconds for the standard path. The [visual receipt summary](APA_VISUAL_TOKENS_PRIMER.md#3-the-modality-jump-coldcast-2026-07-1314)
  records the quality and operating-point details.
- GPT-OSS-20B is reported as a later long-context positive in the
  [visual/text receipt summary](APA_VISUAL_TOKENS_PRIMER.md#2-what-the-text-era-established-evidence-kernel-gates--perplexity--live-session-receipts).
  Its Project-Tensor kernel work is independently summarized in the
  [kernel report](KERNEL_OPT_SYNTHESIS.md).

## 5. Boundary and negative results

- TinyLlama-1.1B and OLMoE-1B-7B used a 2-bit bulk setting below their quality
  floors and ran slower at the recorded 1024-token point. These are informative
  **APA boundary** results, not product claims; see the
  [historical table](APA_PAPER_DRAFT.md#41-the-agnosticism-table).
- Trinity Nano engaged APA only on its 14 NoPE/full layers with no sliding-layer
  leakage, but 13 of 16 greedy tokens diverged. Mean-NLL stayed close and the
  policy was not tuned, so the ledger records a split evaluation rather than a
  positive gate; see the [T2 receipt](TRINITY_NANO_PORT_LEDGER.md#t2--apa-selective-engagement--s8192).
- [Gemma-4 12B / MQA](GEMMA4_MQA_ADJUDICATION.md) is the operative negative:
  +0.4–3.3% peak VRAM, about 3.5x prefill time, +8% decode ms/token, and
  +1.55–1.92% perplexity with APA engaged. The earlier near-tie paper row came
  from scoring that did not engage APA and is historical only.
- GPT-2/MHA demonstrated the early mechanism but paid about 8% perplexity and
  was abandoned; see the [paper's PoC account](APA_PAPER_DRAFT.md#44-mha-the-abandoned-proof-of-concept).
- Mistral-7B has a predicted bulk-bit floor but no closed gate in this repo, so
  its status remains **Planned / unconfirmed**.

## 6. Selection and certification rule

**APA is a multi-KV-head mechanism.** Current selection evidence favors
multi-KV-head GQA-family sites with sufficiently long context, enough key
population for stable score statistics, and a head dimension supported by the
fused path. [Gemma-4 12B's single shared MQA KV head](GEMMA4_MQA_ADJUDICATION.md)
fails the statistical and economic axes; storage savings are not an APA result.

The collaborator-reported 26B model with two KV heads is an **External
receipt**: APA remained clean and 64K context ran where baseline attention
OOMed. It is boundary evidence for the selection rule, not a local reproduction
or a blanket certification. Each model must separately close engagement,
quality, peak-memory, prefill, and decode gates at its intended operating point.

## 7. Evidence and implementation links

- [Master evidence matrix](SUPPORT_MATRIX.md)
- [Gemma-4 12B / MQA adjudication](GEMMA4_MQA_ADJUDICATION.md)
- [APA paper draft](APA_PAPER_DRAFT.md) — historical report; its Gemma row is
  superseded by the adjudication
- [Visual-token receipt summary](APA_VISUAL_TOKENS_PRIMER.md)
- [Kernel optimization synthesis](KERNEL_OPT_SYNTHESIS.md)
- [Kernel implementation ledger](KERNEL_OPT_IMPLEMENTATION_LEDGER.md)
- [`tensor_cuda` APA kernel tests](../tensor_cuda/tests/test_apa_selective.py)
