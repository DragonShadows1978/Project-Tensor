# Project-Tensor Support and Evidence Matrix

| Label | Meaning |
|---|---|
| **Engine port** | Model or pipeline loads and executes through TensorCUDA. |
| **Parity-gated** | Engine output was compared against a registered reference under a stated tolerance. |
| **APA evaluated** | APA was actually engaged and measured on the named target. |
| **APA positive** | The registered quality/cost gate passed at a stated operating point. |
| **APA boundary** | The experiment is informative but exposes a cost, quality, or geometry limit. |
| **APA negative** | The registered gate failed or the mode was abandoned. |
| **GRM adapter** | Model-specific capture/restore dialect code exists. |
| **GRM certified** | Deposit, route, mount, recall, persistence/restart, and relevant controls passed. |
| **External receipt** | Result was collaborator-reported and is not a locally reproduced gate. |
| **Planned / unconfirmed** | Code or a plan exists, but the required evaluation has not closed. |

## How to read this matrix

This is the public source of truth for evidence present in this repository as
of 2026-08-04. A linked status is no broader than its cited document. A cell is
`unconfirmed` when Project-Tensor has no in-repo Markdown receipt for that
surface, even if code, a downstream repository, or an older README mentions
it. In particular, a GRM mount demonstration is not **GRM certified** without
deposit, route, mount, recall, persistence/restart, and control receipts.

The named targets are validation points, not a compatibility whitelist.
TensorCUDA execution, APA evaluation, and GRM certification are independent
columns.

## Master matrix

| Target | Attention / modality | TensorCUDA | APA | GRM | Public note |
|---|---|---|---|---|---|
| [MiniCPM3-4B](APA_PAPER_DRAFT.md#41-the-agnosticism-table) | [MLA, causal text](APA_PAPER_DRAFT.md#41-the-agnosticism-table) | [**Engine port**; historical paper report](APA_PAPER_DRAFT.md#41-the-agnosticism-table) | [**APA positive**; historical paper report](APA_PAPER_DRAFT.md#41-the-agnosticism-table) | `unconfirmed` | [Original MLA evaluation target](APA_PAPER_DRAFT.md#41-the-agnosticism-table) |
| [GRAPA-232M](APA_PAPER_DRAFT.md#41-the-agnosticism-table) | [MLA, causal text](APA_PAPER_DRAFT.md#41-the-agnosticism-table) | [**Engine port**; native training runtime](APA_PAPER_DRAFT.md#6-native-training-apa-from-birth) | [**APA positive**; trained from birth](APA_PAPER_DRAFT.md#6-native-training-apa-from-birth) | `unconfirmed` | [Native-training existence report](APA_PAPER_DRAFT.md#6-native-training-apa-from-birth) |
| Qwen3-4B — `unconfirmed` | `unconfirmed` | `unconfirmed` | `unconfirmed` | `unconfirmed` | `unconfirmed` |
| [Qwen3.5-9B](APA_PAPER_DRAFT.md#41-the-agnosticism-table) | [GQA/hybrid causal text](APA_PAPER_DRAFT.md#41-the-agnosticism-table) | [**Engine port**; historical paper report](APA_PAPER_DRAFT.md#41-the-agnosticism-table) | [**APA positive**; historical zero-flip report](APA_PAPER_DRAFT.md#41-the-agnosticism-table) | `unconfirmed` | [Qwen3.5 receipt path recorded in the paper](APA_PAPER_DRAFT.md#appendix-a--evidence-index-paths-for-the-authors-strip-before-submission) |
| [Gemma-4 12B / MQA](GEMMA4_MQA_ADJUDICATION.md) | [MQA + sliding causal text](GEMMA4_MQA_ADJUDICATION.md#exact-target) | [**Engine port** + **Parity-gated**](GEMMA4_MQA_ADJUDICATION.md#engine-port-verdict) | [**APA negative**](GEMMA4_MQA_ADJUDICATION.md#why-the-earlier-paper-result-is-not-operative) | `unconfirmed` | [Operative 2026-07-04 adjudication](GEMMA4_MQA_ADJUDICATION.md) |
| [GPT-OSS-20B](APA_VISUAL_TOKENS_PRIMER.md#2-what-the-text-era-established-evidence-kernel-gates--perplexity--live-session-receipts) | [MoE + mixed causal attention](APA_VISUAL_TOKENS_PRIMER.md#2-what-the-text-era-established-evidence-kernel-gates--perplexity--live-session-receipts) | [**Engine port** workload; kernel receipt summary](KERNEL_OPT_SYNTHESIS.md) | [**APA positive**; later 96K report](APA_VISUAL_TOKENS_PRIMER.md#2-what-the-text-era-established-evidence-kernel-gates--perplexity--live-session-receipts) | `unconfirmed` | [96K reported; 128K is not promoted by this receipt](APA_VISUAL_TOKENS_PRIMER.md#2-what-the-text-era-established-evidence-kernel-gates--perplexity--live-session-receipts) |
| [Trinity Nano](TRINITY_NANO_PORT_LEDGER.md#2026-07-0809-t4-fp32-ab-disposition--semantic-parity-proven) | [NoPE/hybrid causal text](TRINITY_NANO_PORT_LEDGER.md#2026-07-0809-t4-fp32-ab-disposition--semantic-parity-proven) | [**Engine port** + **Parity-gated**](TRINITY_NANO_PORT_LEDGER.md#2026-07-0809-t4-fp32-ab-disposition--semantic-parity-proven) | [**APA evaluated** / **APA boundary**](TRINITY_NANO_PORT_LEDGER.md#t2--apa-selective-engagement--s8192) | `unconfirmed` | [Mount and recall evidence exists, but lifecycle certification is not established here](TRINITY_NANO_PORT_LEDGER.md#2026-07-09-t1-absolute-confirmed--nope-dissolves-the-arena-hole-law) |
| [Hunyuan3D-2.0](APA_VISUAL_TOKENS_PRIMER.md#3-the-modality-jump-hunyuan3d-transformer-replacement-2026-07-1314) | [Bidirectional visual DiT](APA_VISUAL_TOKENS_PRIMER.md#3-the-modality-jump-hunyuan3d-transformer-replacement-2026-07-1314) | [**Engine port**; complete runtime report](APA_VISUAL_TOKENS_PRIMER.md#3-the-modality-jump-hunyuan3d-transformer-replacement-2026-07-1314) | [**APA positive**; E2E visual receipt](APA_VISUAL_TOKENS_PRIMER.md#3-the-modality-jump-hunyuan3d-transformer-replacement-2026-07-1314) | `unconfirmed` | [The APA E2E receipt belongs to 2.0](APA_VISUAL_TOKENS_PRIMER.md#3-the-modality-jump-hunyuan3d-transformer-replacement-2026-07-1314) |
| Hunyuan3D-2.1 — `unconfirmed` | `unconfirmed` | `unconfirmed` | `unconfirmed` | `unconfirmed` | `unconfirmed`; never inherit the 2.0 APA receipt |
| [TinyLlama-1.1B](APA_PAPER_DRAFT.md#41-the-agnosticism-table) | [GQA causal text](APA_PAPER_DRAFT.md#41-the-agnosticism-table) | [**Planned / unconfirmed**; experimental paper target](APA_PAPER_DRAFT.md#41-the-agnosticism-table) | [**APA boundary**; sub-floor quality/cost result](APA_PAPER_DRAFT.md#42-the-honest-split-parity-vs-long-context-trade) | `unconfirmed` | [Mechanism experiment, not a general engine claim](APA_PAPER_DRAFT.md#42-the-honest-split-parity-vs-long-context-trade) |
| [OLMoE-1B-7B](APA_PAPER_DRAFT.md#41-the-agnosticism-table) | [MoE causal text](APA_PAPER_DRAFT.md#41-the-agnosticism-table) | [**Planned / unconfirmed**; experimental paper target](APA_PAPER_DRAFT.md#41-the-agnosticism-table) | [**APA boundary**; sub-floor quality/cost result](APA_PAPER_DRAFT.md#42-the-honest-split-parity-vs-long-context-trade) | `unconfirmed` | [Mechanism experiment, not a general engine claim](APA_PAPER_DRAFT.md#42-the-honest-split-parity-vs-long-context-trade) |
| [GPT-2](APA_PAPER_DRAFT.md#44-mha-the-abandoned-proof-of-concept) | [MHA causal text](APA_PAPER_DRAFT.md#44-mha-the-abandoned-proof-of-concept) | [**Planned / unconfirmed**; PoC only](APA_PAPER_DRAFT.md#44-mha-the-abandoned-proof-of-concept) | [**APA negative**; abandoned](APA_PAPER_DRAFT.md#44-mha-the-abandoned-proof-of-concept) | `unconfirmed` | [Early mechanism evidence only](APA_PAPER_DRAFT.md#44-mha-the-abandoned-proof-of-concept) |
| [DeepSeek-V2-Lite](QUANT_SWEEP_SYNTHESIS.md) | [MLA + MoE text](QUANT_SWEEP_SYNTHESIS.md) | [**Planned / unconfirmed**; adapter mention is not an execution gate](QUANT_SWEEP_SYNTHESIS.md) | `unconfirmed` | `unconfirmed` | [Kernel or adapter compatibility does not close model gates](QUANT_SWEEP_SYNTHESIS.md) |
| [Mistral-7B](APA_PAPER_DRAFT.md#43-the-bulk-bits-law) | [GQA causal text](APA_PAPER_DRAFT.md#43-the-bulk-bits-law) | [**Planned / unconfirmed**; port started](APA_PAPER_DRAFT.md#43-the-bulk-bits-law) | [**Planned / unconfirmed**; testing incomplete](APA_PAPER_DRAFT.md#appendix-b--open-measurement-worklist-gates-before-submission) | `unconfirmed` | [Predicted bulk floor is not a closed gate](APA_PAPER_DRAFT.md#43-the-bulk-bits-law) |
| [External 26B, kv=2](APA_VISUAL_TOKENS_PRIMER.md#2-what-the-text-era-established-evidence-kernel-gates--perplexity--live-session-receipts) | [GQA causal text](APA_VISUAL_TOKENS_PRIMER.md#2-what-the-text-era-established-evidence-kernel-gates--perplexity--live-session-receipts) | `unconfirmed` | [**External receipt**; positive boundary validation](APA_VISUAL_TOKENS_PRIMER.md#2-what-the-text-era-established-evidence-kernel-gates--perplexity--live-session-receipts) | `unconfirmed` | [Collaborator-reported; not locally reproduced](APA_VISUAL_TOKENS_PRIMER.md#2-what-the-text-era-established-evidence-kernel-gates--perplexity--live-session-receipts) |

## Scope notes

- “Engine port” says only that the model or pipeline executes through
  TensorCUDA; it does not imply APA benefit.
- “APA evaluated” requires proof that APA engaged during the measurement. The
  [Gemma-4 12B / MQA adjudication](GEMMA4_MQA_ADJUDICATION.md) exists because an
  earlier scoring protocol failed that requirement.
- “GRM certified” is intentionally absent from the rows above: no in-repo
  Project-Tensor document establishes the complete lifecycle gate.
- Hunyuan3D-2.0 owns the APA-positive visual E2E receipt. Hunyuan3D-2.1 remains
  `unconfirmed` for both its public TensorCUDA receipt and APA evaluation here.
