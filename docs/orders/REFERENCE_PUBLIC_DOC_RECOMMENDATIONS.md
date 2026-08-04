# GRM / APA / Project-Tensor Public Documentation Recommendations

**Status:** Recommended documentation changes; no runtime or public README changes authorized by this document  
**Prepared:** 2026-08-04  
**Repositories in scope:** GraftRepository, Project-Tensor, ColdCast  

## Executive recommendation

The public documentation currently contains enough evidence to establish a much
broader and more current body of work than a casual reader is likely to perceive.
The problem is not a shortage of receipts. It is that three different products
and three different compatibility claims are presented together:

1. **Project-Tensor compatibility:** a model or pipeline has been ported to the
   TensorCUDA runtime.
2. **APA evaluation:** the attention mechanism has been measured on a particular
   model, attention family, or modality, including positive and negative results.
3. **GRM certification:** graft harvest, persistence, routing, mounting, and
   recall have been gated end to end for a particular model dialect.

Those categories must be separated. A TensorCUDA model port is not automatically
an APA validation, and an APA validation is not automatically a GRM-certified
adapter. Conversely, the models named in the repositories are validation points,
not a whitelist of every architecture the underlying mechanism can address.

The recommended public story is:

> Project-Tensor is the independent CUDA tensor and inference engine. APA is a
> precision-allocation mechanism implemented at the attention score/softmax
> boundary. GRM is a routed persistent-memory runtime that operates on a frozen
> model's own attention state. ColdCast demonstrates that TensorCUDA and APA are
> not limited to text: Hunyuan3D's bidirectional shape DiT was ported to
> TensorCUDA and exercised end to end.

## What the current presentation gets wrong

### 1. It makes an evidence matrix look like a compatibility whitelist

Project-Tensor's opening table names roughly five LLM rows, then compresses all
of APA into one generic row. Hunyuan appears as `DiT` or `HY3D`, not as an
explicit Hunyuan3D transformer replacement. A skimming reader can therefore
conclude that the software supports only the handful of named models.

Every relevant README should include this sentence close to the top:

> **The named models below are validation points, not a compatibility
> whitelist.** Support status is reported separately for TensorCUDA execution,
> APA evaluation, and GRM end-to-end certification.

### 2. “Architecture” is being read as “model”

The APA paper discusses four attention families—MHA, MQA, GQA, and MLA—plus
MoE. Those are attention or model-organization families, not four model names.
The original paper matrix itself contains seven evaluated models, and later
receipts add GPT-OSS, Trinity/NoPE, an external 26B GQA boundary case, and
Hunyuan3D's non-causal visual DiT.

The README should say “attention families” every time this evidence is
summarized. It should never use an unqualified phrase such as “four
architectures” next to a model list.

### 3. The top and bottom of the Project-Tensor README describe different eras

The opening now correctly presents a native CUDA C++ engine with no PyTorch or
TensorFlow runtime path. The installation and usage sections then revert to the
older CuPy / `tensor_gpu_v2` package and API. That contradiction makes the
modern engine claims look aspirational even though the native implementation
and downstream ports exist.

The legacy CuPy material should be moved to a clearly labeled historical or
legacy document. The primary installation path must build and import
`tensor_cuda`, use the current package surface, and include a current smoke test.

### 4. APA's broad insertion point is overstated as universal success

“Architecture-agnostic by construction” has a defensible narrow meaning: APA
attaches at the QK-score/softmax operation rather than depending on a particular
model block layout. It must not be presented as “every transformer benefits at
the same operating point.” The repositories also report selection boundaries,
sub-floor quality trades, and an MQA failure mode.

Replace wording such as:

> APA composes with any transformer.

with:

> APA has an architecture-neutral insertion point at attention scoring, while
> useful operating points remain model- and geometry-dependent. Validate bulk
> precision and refine policy per model. Current selection evidence favors
> multi-KV-head, sufficiently long-context sites; negative and boundary results
> are listed alongside positive results.

### 5. The Gemma/MQA public story is internally inconsistent

The APA draft currently presents Gemma-4 MQA+sliding as a measured near-tie at a
specific operating point. The refreshed Project-Tensor README says MQA fails
APA structurally and reports a costly engaged mode. Both may reflect different
experiments, implementations, or promotion criteria, but a public reader cannot
infer that distinction.

Before the next public documentation release, reconcile these statements in one
short adjudication note that records:

- the exact Gemma model/configuration;
- whether the result was simulated, composed, fused, prefill, or decode;
- bulk bits and realized refine fraction;
- reference oracle and quality metric;
- memory and speed result;
- the final supported, experimental, or unsupported verdict.

Do not silently delete the earlier result. Explain why the later verdict is the
operative one.

## Recommended documentation architecture

Use progressive disclosure. The README sells and orients; the documentation
holds the evidence.

### Project-Tensor README

Target: approximately 100–150 lines before links and license.

Recommended order:

1. One-paragraph identity.
2. A five-minute installation and native-engine smoke test.
3. A compact “What runs today” table.
4. A separate “Research systems built on it” table for APA, GRM, and ColdCast.
5. Three or four representative performance receipts.
6. Honest limitations and hardware scope.
7. Links to the full model matrix, benchmark methodology, failure ledger, and
   historical CuPy API.

Do not place the full research ledger in the README. Do not interleave engine
features, model ports, research mechanisms, and negative experiments in one
table.

Suggested opening:

> **Project-Tensor is a from-scratch CUDA tensor and inference engine for local
> AI workloads.** Its current native runtime is a C++/CUDA extension with a
> Python tensor/autograd layer and no PyTorch or TensorFlow dependency in the
> inference path. It runs quantized LLMs and the complete ColdCast Hunyuan3D
> image-to-3D pipeline on consumer NVIDIA GPUs.

### APA overview

Create a short standalone `docs/APA.md`, or promote an equivalent concise page,
instead of requiring readers to interpret the full paper draft.

Recommended order:

1. Mechanism in four sentences.
2. What APA is not: not sparse attention, not KV compression, not GRM.
3. Validated attention-family and modality matrix.
4. Positive results.
5. Boundary and negative results.
6. Selection rule and per-model certification requirement.
7. Links to the paper, kernel gates, and receipts.

Suggested opening:

> **Adaptive Precision Attention (APA) scores every key, but spends full
> precision only on the softmax-dominant tail.** The remaining keys retain
> low-precision scores and still participate in the exact softmax denominator;
> no keys are dropped. APA is implemented at the attention score/softmax
> boundary and has been evaluated across causal and bidirectional attention,
> MHA/MQA/GQA/MLA organization, MoE models, NoPE and sliding hybrids, and a
> visual diffusion transformer. Results are model-gated, and negative results
> define the supported envelope.

### GraftRepository README

Target: approximately 120–180 lines, with only the strongest six to eight
receipts above the fold.

Recommended order:

1. Plain-language product identity.
2. A six-step “how it works” diagram or numbered flow.
3. User-visible outcome and constraints.
4. GRM certification matrix.
5. Minimal runnable example.
6. Current limitations: Project-Tensor dependency, model adapters, artifact
   storage, hardware/OS support, and research-grade setup.
7. Links to the full results ledger and architecture paper.

Suggested opening:

> **GRM gives a frozen local model persistent, routed memory without replaying
> the full conversation into its context window.** It captures the model's own
> attention state once, stores that state as a graft, routes relevant grafts for
> each request, and mounts them into a bounded live cache. The model weights are
> unchanged, old history does not repay tokenization or prefill, and the active
> VRAM budget remains bounded by the configured arena.

The current opening phrase “effectively unbounded” should remain only if it is
immediately paired with the operational qualification: bounded active residency
does not mean zero host/disk growth, universal recall, or unlimited addressable
storage.

### ColdCast README

ColdCast should remain product-focused, but its first screen should explicitly
state why it matters to Project-Tensor and APA:

> ColdCast replaces the Hunyuan3D shape-generation runtime with a torch-free
> TensorCUDA implementation. Hunyuan3D-2.0's bidirectional DiT was also the
> first non-causal, non-text end-to-end APA validation site.

Keep the Hunyuan3D-2.0 APA receipt distinct from the Hunyuan3D-2.1 TensorCUDA
port. Do not imply that APA has been E2E-certified on 2.1 until a corresponding
receipt exists.

## Required public status vocabulary

Use the following labels consistently in every matrix:

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

Never use “supported” without naming which of these surfaces is supported.

## Recommended master evidence matrix

Create `docs/SUPPORT_MATRIX.md` as the single public source of truth. The
following is a starting structure, not a final adjudication. Every cell must
link to a receipt, or say `unconfirmed`.

| Target | Attention / modality | TensorCUDA | APA | GRM | Public note |
|---|---|---|---|---|---|
| MiniCPM3-4B | MLA, causal text | Engine port | Positive | Certified | Original MLA validation target. |
| GRAPA-232M | MLA, causal text | Native training runtime | Positive, trained from birth | Not applicable | APA training existence proof. |
| Qwen3 / Qwen3.5 | GQA/hybrid text | Engine ports | Positive receipts on Qwen3.5 | Adapter/certification status per gate | Do not collapse distinct Qwen variants. |
| Gemma-4 12B | MQA + sliding text | Engine port | **Adjudication required** | Report exact status | Reconcile paper and current README. |
| GPT-OSS-20B | MoE + mixed attention | Engine port | Positive at registered long-context mode | E2E certified receipt | Name the four-way stack and 96K ceiling. |
| Trinity Nano | NoPE/hybrid text | Engine port | Evaluated/positive quality receipt | Certified receipt | Explain why NoPE changes GRM position handling. |
| Hunyuan3D-2.0 | Bidirectional visual DiT | Complete ColdCast port | Positive E2E visual receipt | Not applicable | Explicitly name Hunyuan and TensorCUDA replacement. |
| Hunyuan3D-2.1 | Bidirectional visual DiT + MoE | Complete engine port | Unconfirmed unless separately gated | Not applicable | Keep separate from the 2.0 APA result. |
| TinyLlama-1.1B | GQA text | Experimental | Boundary/sub-floor | Unconfirmed | Keep as honest quality/cost boundary. |
| OLMoE-1B-7B | MoE text | Experimental | Boundary/sub-floor | Unconfirmed | Separate mechanism test from product support. |
| GPT-2 | MHA text | PoC | Negative/abandoned | Not applicable | Evidence of early mechanism development, not current support. |
| DeepSeek-V2-Lite | MLA + MoE text | Engine/adapter work exists | State only the closed gate | State exact certification | Do not infer full support from kernel compatibility. |
| Mistral-7B | GQA text | Adapter exists | Port/testing incomplete in APA draft | State exact certification | Label unconfirmed until gates close. |
| External 26B, kv=2 | GQA text | External environment | External positive boundary receipt | Not claimed | Identify as collaborator-reported, not local reproduction. |

The matrix should have direct links to small human-readable receipt summaries.
Do not link a casual reader straight into multi-hundred-line ledgers unless no
summary exists.

## Specific edits by repository

### Project-Tensor

- Split the existing “WHAT RUNS ON IT” table into:
  - native engine ports;
  - downstream systems built on the engine;
  - research/evaluation targets.
- Replace the old CuPy installation path with the native `tensor_cuda` build.
- Move `tensor_gpu_v2` history and examples to `docs/LEGACY_TENSOR_GPU_V2.md`.
- Add explicit rows for ColdCast/Hunyuan3D-2.0 and Hunyuan3D-2.1.
- Link APA to the concise overview and full support matrix.
- Preserve the negative-results section, but move detailed incidents to a
  ledger and keep only current operative limits in the README.
- Reconcile the Gemma/MQA verdict before repeating the universal architecture
  claim.

### GraftRepository

- Replace the long top-level results dump with six to eight representative
  receipts: lossless mounting, bounded residency, persistence/restart, one MLA
  result, one GQA result, GPT-OSS E2E, and one honest negative.
- Move the complete gated-result table to `docs/RESULTS_INDEX.md`.
- Put the model/dialect support matrix before the quickstart.
- Distinguish “adapter exists” from “full GRM lifecycle certified.”
- Replace the MiniCPM-only quickstart with either:
  - a backend-neutral facade, once implemented; or
  - an explicitly labeled MiniCPM3 developer example plus links to other
    adapters.
- State clearly that the current runtime depends on Project-Tensor and is not
  yet a drop-in layer for arbitrary Hugging Face, llama.cpp, MLX, or vLLM
  models.
- Keep detailed research receipts, failures, and ledgers; change their location,
  not their honesty.

### APA paper and primer

- Update the paper's original seven-model table with a dated “subsequent
  validations” section rather than silently rewriting historical experiments.
- Add GPT-OSS, Trinity, Hunyuan3D-2.0, and the external 26B result with evidence
  class labels.
- State that Hunyuan3D-2.0 is the APA E2E visual receipt; Hunyuan3D-2.1 is a
  separate TensorCUDA port unless/until APA is gated there.
- Reconcile the MQA/Gemma result and update the selection law.
- Replace broad “any transformer” language with the narrower insertion-point
  claim and explicit operating-envelope language.
- Keep negative results in the main result story, not only in an appendix.

### ColdCast

- Add a brief architecture/status table for Hunyuan3D-2.0, 2mv, and 2.1.
- State which version has the E2E APA receipt.
- Link back to Project-Tensor's engine matrix and APA overview.
- Preserve the product-focused quickstart; do not turn this README into the APA
  paper.

## Recommended visual hierarchy

The first screen of each README should answer only four questions:

1. What is this?
2. Why is it different?
3. What can I run today?
4. Where is the evidence?

Use one compact diagram in the GRM README:

```text
document / prior turn
        │ harvest once
        ▼
model-native K/V graft ──► persistent repository
                                  │ route relevant memories
                                  ▼
current prompt ──────────► bounded live arena ──► frozen model response
```

Avoid stacking multiple slogans, large receipt tables, implementation layouts,
and research history before installation or usage.

## Implementation sequence

### Phase 0 — adjudicate claims

1. Build the master evidence matrix from existing receipts.
2. Resolve the Gemma/MQA contradiction.
3. Confirm every “certified” GRM row has the full lifecycle controls required by
   the chosen definition.
4. Separate local, external, simulated, kernel-only, and E2E evidence.

**Exit criterion:** every public claim maps to one status label and one receipt.

### Phase 1 — repair Project-Tensor identity

1. Replace the stale install and smoke test.
2. Split engine ports from research systems.
3. Move legacy CuPy material.
4. Add Hunyuan3D by name.

**Exit criterion:** a new user can build/import the current native engine by
following only the README, and cannot mistake the five headline rows for a
model whitelist.

### Phase 2 — publish concise APA and support pages

1. Create `docs/APA.md`.
2. Create `docs/SUPPORT_MATRIX.md`.
3. Link all repository READMEs to those pages.
4. Add dated post-paper results without rewriting original provenance.

**Exit criterion:** a reader can distinguish attention families from model
count and can see positive, boundary, and negative evaluations at a glance.

### Phase 3 — tighten the GRM README

1. Reduce above-the-fold receipts.
2. Add the six-step operational explanation and diagram.
3. Publish GRM-specific certification status.
4. Label the quickstart as developer-facing until the backend facade exists.

**Exit criterion:** an average technical reader can explain GRM correctly after
reading the first two screens and knows what is required to try it.

### Phase 4 — cross-repository consistency gate

Search every public document for these phrases and adjudicate each occurrence:

- `any transformer`
- `architecture-agnostic`
- `supported models`
- `four architectures`
- `unbounded` / `infinite context`
- `no PyTorch`
- `CuPy`
- `MQA`
- `Hunyuan` / `HY3D` / `DiT`

**Exit criterion:** the same term has the same qualified meaning in all three
repositories, and no current README contradicts another current README.

## Acceptance checks

The documentation refresh is complete when an unfamiliar reviewer can answer
all of the following without reading a ledger:

- Project-Tensor is the engine; APA and GRM are different systems built on it.
- “Four attention families” does not mean “four supported models.”
- Named model rows are validation points, not a universal whitelist.
- At least the original seven APA evaluation targets are visible, with later
  GPT-OSS, Trinity, external 26B, and Hunyuan receipts separately dated.
- The Hunyuan3D claim is correctly described as a torch-free TensorCUDA runtime
  port, not merely a wrapper.
- Hunyuan3D-2.0's APA E2E result is not accidentally attributed to 2.1.
- Positive, boundary, negative, external, and unconfirmed evidence are visibly
  different.
- A current Project-Tensor installation command imports the current native
  engine rather than the legacy CuPy package.
- GRM adapter presence is not presented as full lifecycle certification.
- “Effectively unbounded” is qualified as bounded active residency with growing
  external storage and model-dependent retrieval quality.

## Bottom line

The repositories do not need fewer receipts. They need a clearer evidence
taxonomy and a shorter entrance. The strongest corrective move is not to argue
that the reader failed to understand the work; it is to make the mistaken
“five old models” interpretation impossible from the first screen.
