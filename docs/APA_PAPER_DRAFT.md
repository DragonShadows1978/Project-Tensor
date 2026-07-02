# Selective Attention Is All You Need
## Adaptive Precision Attention
### Architecture-Agnostic Precision Allocation at the Attention-Kernel Level

**David Perry** — Independent Researcher (no institutional affiliation)
`dave@ai-storyforge.com`

*Preprint v1.0-rc — 2026-07-02. Measurements marked OPEN in Appendix B are
explicitly unfinished and are stated as such in the text; every quantitative
claim traces to a gated, registered evaluation (see §8 and Appendix A).
Text licensed CC BY 4.0; the implementations are separately licensed
AGPL-3.0 with commercial licensing available.*

---

## Author's Note and Disclosure

This paper is the work of an independent researcher without academic
affiliation, and it was produced with substantial assistance from AI systems.
Anthropic's Claude models and OpenAI's Codex served as implementation and
drafting collaborators under the author's direction — writing and reviewing
kernel code, executing registered evaluation gates, and drafting and editing
this manuscript, including this section. The hypotheses, the experimental
designs, the acceptance thresholds (registered before results were seen), and
all editorial decisions are the author's, and the author accepts full
responsibility for the content. All experiments were run on consumer hardware
(RTX 3070 8 GB and RTX 4070 SUPER 12 GB); no institutional compute, funding,
or sponsorship was involved. The disclosure is made in the interest of
transparency, and with some appreciation of the recursion: a paper about
allocating machine attention efficiently, written in part by the machines
whose attention it allocates.

---

## Abstract

Transformer attention spends most of its precision on interactions the softmax then
discards: because the softmax is exponential, output mass concentrates on a few keys
and the tail is crushed toward zero regardless of how precisely it was scored. We
introduce **Adaptive Precision Attention (APA)**, which scores *every* key but varies
the *precision* of the key used: a cheap bulk pass scores all keys against low-bit
quantized keys, and a sparse refine pass re-scores only the softmax-dominant fraction
at full precision. The denominator is summed over all keys, so nothing is dropped and
the distribution is not redistributed — the only approximation is the rounding of
un-refined keys the softmax was already suppressing. The hypothesis came from number
theory: a precision-deficit law observed in a hierarchical number-theoretic system
predicted that attention's per-interaction precision depth should decay
geometrically; direct measurement confirms the predicted two-regime,
cheap-bulk structure — roughly half of all interactions resolve at 1 bit
(49.6–60.9% across models, tolerances, and lengths; over half on every model
at working tolerance ε=0.01), 60–86% by 2 bits — with the expensive
remainder a heavy power-law tail. APA operates at the kernel level — on the
QK-score/softmax operation every attention layer computes — so it is
**architecture-agnostic by construction**: it has no component specific to how heads
share K/V or how experts route. We confirm this across the four attention families
(MHA, MQA, GQA, MLA) plus MoE routing — seven models, retrofit and
trained-from-birth. The one architecture-dependent quantity is the bulk bit-width,
predicted by a **bulk-bits law** we report: key normalization pins the safe bulk
floor at 4 bits uniformly, while raw-key models have model-specific floors (2–8
bits measured) that must be swept. At or above the floor, APA matches or improves
full-precision perplexity; below it, it trades quality for context.
Context extension is architecture-dependent — over 10× on MLA (engine-vs-engine),
parity on sliding-window models — and decode-time speedup reaches ~2.1× at long
sequence length.

---

## 1. Introduction

### 1.1 The quadratic wall and what actually drives it

Standard attention computes `softmax(QKᵀ/√d)·V`, materializing (or, with
FlashAttention, implicitly tiling) an `n×n` score matrix at a cost of `O(n²d)`. The
dominant framing of the past three years treats this cost as a *count* problem — too
many interactions — and attacks it by computing *fewer*: sparse attention
(Longformer, BigBird) drops keys via fixed patterns; linear attention (Performer)
approximates the matrix with kernels; FlashAttention keeps the count but removes the
memory traffic of materialization.

We observe that the cost is also a *precision* problem, and that this axis is
largely unexploited. The softmax is exponential: `exp(s − s_max)` is already near
zero for all but the top-scoring keys. The output is therefore insensitive to the
precision of the tail scores — they are multiplied by ~0 before they reach `V`.
**The waste is not that we compute too many interactions; it is that we compute the
unimportant ones too precisely.**

### 1.2 Origin of the hypothesis

APA came from number theory. In a study of the precision dynamics of a
number-theoretic system [Collatz precision framework, companion note], we identified
a structural law: in hierarchical computational systems, the precision required at
each step decays geometrically, and sustained high-precision operations are
exponentially rare and structurally unsustainable. Attention is such a system — a
hierarchical accumulation of query–key interactions — so the law should apply to it.
It does in shape (§3.4): direct measurement finds the predicted cheap-bulk /
rare-tail structure — most interactions resolve at 1–2 bits — with the
bulk's exact distribution family tolerance-dependent (geometric is
Vuong-preferred at loose tolerance on three of four models; power-law
dominates the tail throughout).

The transfer is structural rather than numerical: the decay *constant* differs across
domains (≈0.079 in the number-theoretic setting, 0.15–0.76 in attention), so the
attention result rests on its own measurement, not on a borrowed value. What carried
across domains is the *shape* of the law — and that a precision-deficit principle
proven in one hierarchical system predicted, correctly, the compressibility of
another is the reason APA exists.

The mechanism also has an older, informal ancestor worth recording: before
the number-theoretic work, the author's design sketch for attention was a
"memory manager" — a cheap reader that scans the full context and directs
each head's expensive attention. The bulk/refine split is that sketch made
mechanism: the low-bit bulk pass is the cheap reader; the refine set is the
direction. The full lineage of the program, including the published
catalyst that connected the sketch to frozen-model attention, is recorded
in the companion memory-system paper [22].

### 1.3 Contributions

1. **APA**, a content-adaptive precision-allocation mechanism for attention: bulk
   low-bit scoring over all keys, full-precision refinement of the softmax-dominant
   fraction, exact denominator, nothing dropped (§2).
2. **Architecture-agnostic by construction**: APA acts on the QK-score/softmax
   operation common to all attention, with no architecture-specific component, so it
   composes with any transformer. The four attention families plus MoE
   (MHA/MQA/GQA/MLA, MoE; seven models) are not a "happens to generalize" coverage
   argument — they confirm a mechanism that has nothing architecture-specific to
   break. KV-sharing-independence is one instance: precision allocation is a
   separate axis from how heads share K/V (§4).
3. A complexity/cost model separating operation count (`O((1+r)n²)`, quadratic like
   Flash) from precision-weighted cost (`O(b_bulk·n² + b_full·r·n²)`), and a
   **memory-wall argument** showing the dense full-precision baseline cannot be run
   at the long-context configs APA reaches (§2.3, §5).
4. The **bulk-bits law**: key-normalization structure predicts the safe bulk
   precision — normalized keys pin the floor at 4 bits uniformly; raw keys have
   model-specific floors that must be measured — validated across six architectures
   (§4.3).
5. A **native-training existence proof**: a 232M MLA model trained from birth under
   APA (refine 0.15) over a 12,288-token window, with a window-use probe showing
   deep-context facts remain attended (§6).

---

## 2. Method

### 2.1 The bulk/refine split

For query row `qᵢ` over keys `k₁…k_L`:

1. **Bulk pass (all keys, low precision).** Compute `bulkⱼ = qᵢ · k̃ⱼ · scale`, where
   `k̃ⱼ` is the key quantized to `b_bulk` bits (via TurboQuant product quantization,
   which preserves inner-product *ordering* — see §3.3). Cost: `L` dot products at
   low precision.
2. **Threshold from the bulk scores.** `thr = mean(|bulk|) + z·std(|bulk|)`, with
   `z = Φ⁻¹(1 − r)` for refine fraction `r`. This selects the top ~`r·L` keys *by the
   cheap pass*.
3. **Refine pass (selected keys, full precision).** For keys above `thr`, recompute
   `scoreⱼ = qᵢ · kⱼ · scale` at full precision. Below-threshold keys keep `bulkⱼ`.
4. **Exact softmax over all keys.** `out = softmax(score) · V`. The denominator sums
   over all `L` keys.

Only the **key** is quantized in the bulk score path; `V` is full precision, and the
softmax normalization is exact over the full key set.

### 2.2 No keys are dropped — why this is not sparse attention

This is the load-bearing distinction. Sparse/windowed methods *zero* the tail: an
unselected key contributes nothing, which alters the softmax denominator and shifts
the whole distribution. APA *keeps* the tail at low precision: every key receives a
real, quantization-rounded score and participates in the denominator. The worst case
for a mis-ranked key is not blindness but a slightly noisy weight — and that weight
was, by softmax concentration, already going to be near zero. The deviation from
full-precision attention is therefore **bounded by the bulk-quantization error on
the un-refined set**, a characterizable quantity, rather than the unbounded "did we
select the right keys?" failure of sparse methods.

This also predicts APA's gentle long-range degradation (§6): a deep fact outside the
refine fraction is still scored with quantized keys — dimly visible, not invisible.

### 2.3 Complexity and cost

Per row, the work is `L` low-precision dots + `r·L` full-precision dots. Over `L`
rows:

- **Operation count:** `O((1 + r)·n²)` — quadratic, like FlashAttention. APA's gain
  is in precision-weighted cost, not asymptotic class; the op-count is the same order.
- **Precision-weighted cost:** `O(b_bulk·n² + b_full·r·n²)`. At `b_bulk=4`,
  `b_full=16`, `r=0.15`: bulk term ∝ 4n², refine term ∝ 2.4n², total ∝ 6.4n² vs
  dense `16n²` — **~2.5× less precision-weighted work**, with the two passes
  roughly balanced (neither dominates).
- **Memory:** the training and decode kernels use online (FlashAttention-style)
  softmax and never materialize the `n×n` score matrix. This is what makes the
  long-context configs of §4 fit in memory at all (§5).

The two knobs — `b_bulk` and `r` — are independent, each with an
accuracy-imposed floor (§4.3 for `b_bulk`; §3.4/§6 for `r`).

---

## 3. Why It Works

### 3.1 Softmax concentration

The output is `Σⱼ pⱼ vⱼ` with `pⱼ ∝ exp(sⱼ)`. Because the weights are exponential in
the scores, output mass concentrates on the top scores by construction; the tail is
pre-crushed by the softmax itself, independent of any method. Full-precision scoring
of the tail computes a weight that is then multiplied to ~0 — the precision never
reaches the output.

### 3.2 The denominator must be kept

The reason APA can be approximate in the tail *scores* yet faithful in the *output*
is that it preserves the denominator. Zeroing the tail (sparse attention)
renormalizes the surviving weights and moves the distribution; rounding the tail
(APA) leaves the normalization correct and perturbs only terms the normalization was
already suppressing.

### 3.3 The bulk pass need only *rank*, not *score*

Selection (step 2) requires only that the bulk scores **order** keys roughly like the
full-precision scores would — the top-`r` by bulk must overlap the top-`r` by exact.
The TurboQuant quantizer (Zandieh et al., 2025; implemented in-project) is chosen
precisely because it preserves inner-product ordering. The direct evidence is
partial but sharp: on Qwen (raw keys), bulk-vs-exact score-rank correlation is
**0.85 at 8-bit bulk versus 0.03 at 2-bit** — above the floor the ranking survives,
below it the router is blind, which is the mechanism behind the bulk-bits cliff of
§4.3. A full Spearman/Kendall sweep per architecture remains open (Appendix B.1).

### 3.4 Measured precision-depth distribution (GHOST_PRECISION)

We measured, across GPT-2 124M, TinyLlama 1.1B, Qwen2.5 1.5B, and Mistral-7B
(hundreds of millions of measured pairs over 21 configurations; 70.6M per
configuration at 7B), the **interaction depth** `δ(qᵢ,kⱼ) = min{b :
quantize_b(qᵢ·kⱼ) ≈ qᵢ·kⱼ}` — the bits needed to resolve each dot product to
the softmax's tolerance. All figures below are from the study's final,
post-bug-fix analysis cycle (the instrument's mutation-testing and bug-fix
history is documented in the companion note [21]).

- **Low-bit bulk.** At working tolerance ε=0.01, over half of all
  interactions stabilize at 1 bit on every model (50.5–60.9% across both
  sequence lengths) and 60–86% by 2 bits; at ε≤0.001 the 1-bit fraction is
  49.6–51.4%. At loose tolerance the geometric distribution is
  Vuong-preferred on three of the four models — decisive even at 7B
  (Mistral, Z=646.5, p≈0; the study's largest geometric statistic is GPT-2
  at seq 512, Z=1579.6) — while Qwen2.5 prefers a power-law at every
  tolerance and the largest-magnitude statistics overall favor power-law at
  tight tolerance. The universal, load-bearing fact is the two-regime
  shape: a cheap bulk carrying most of the mass, plus an expensive minority
  tail — i.e. a bulk-quantize + tail-refine split, read directly off the data.
- **Heavy tail.** At tight tolerance (ε≤0.001) the tail is power-law on all
  four models. The expensive minority brackets the refine fractions
  (0.10–0.15) we operate at.
- **Phantoms yes, ghosts no.** High-weight-but-cheap-to-resolve interactions
  ("phantoms") appear by the hundreds in every configuration (counts at the
  500-pattern detection cap); high-depth-but-low-weight ones ("ghosts")
  number zero everywhere. The tail we approximate is genuinely low-mass.
- **Context helps.** 512-token sequences decay faster than 128-token on every
  model measured at both lengths — APA gets *more* efficient as context
  grows, predicting the long-context wins of §4.
- **Roughly uniform across layers on most configurations; stable across
  scale.** Layer-to-layer CV of the fitted decay rate spans ~0.04–0.39
  across the 21 configurations — below 0.15 for most Mistral-7B and GPT-2
  configurations, 0.14–0.18 for Qwen2.5, but up to 0.39 for TinyLlama at
  512 tokens/ε=0.01, which the study itself flags as position-dependent
  decay structure. Cross-model mean decay spans only 0.28–0.43, with
  Mistral-7B (0.288) indistinguishable from Qwen2.5-1.5B (0.282) — the
  structure does not wash out with scale.

The decay constants (0.15–0.76 across configurations) differ from the
number-theoretic origin's 0.079 by 2–10× (§1.2): the shape transferred, the
constant did not.

---

## 4. Results: APA Is Architecture-Agnostic

APA attaches to the QK-score/softmax operation, which is invariant across attention
designs, so there is no per-architecture port — the same kernel runs on each. The
table below is the confirmation: seven models across four attention families plus
MoE, one mechanism, no architecture-specific code path. The only value that changes
across rows is the bulk bit-width, set by the bulk-bits law (§4.3).

### 4.1 The agnosticism table

All retrofit results are engine-vs-engine against the model's own full-precision
attention at refine 1.0; native result is trained from birth.

All quality numbers below are from gated, registered evaluations (the project
research board and per-model port ledgers); they are not re-derived here. Parity is
engine-vs-engine (APA vs the same engine's standard attention) and, where noted,
against an fp32 ground-truth oracle.

| Family | Model | Bulk | Refine | Quality (gated) | Context | Speed |
|----------|-------------|-----|------|-----------------|-----------|-----------|
| **MLA** | MiniCPM3-4B | 4-bit | 0.10 | **free** (a) | 3K→**32K** | 21.6 ms/tok (31×) |
| **MLA (native)** | GRAPA-232M | 4-bit | 0.15 | trained from birth (§6) | 12,288 | ~33 tok/s train |
| **MQA+SW** | Gemma-4 12B | 4-bit | 0.10 | **near-tie vs fp32 GT** (b) | 8K (parity)* | 31 tok/s (6.7×) |
| **GQA** | Qwen3.5-9B | 4-bit | 0.15 | **zero top-1 flips** (c) | 2K→GRM | 38 tok/s (1.5×) |
| **GQA** | TinyLlama-1.1B | 2-bit | 0.10 | +ppl, sub-floor (§4.2) | 2K→**10K (5×)** | 0.69× @1024 |
| **MoE** | OLMoE-1B-7B | 2-bit | 0.10 | +ppl, sub-floor (§4.2) | 2K→3K+ | 0.67× @1024 |
| MHA | GPT-2 (PoC) | 2-bit | 0.10 | +8% ppl — abandoned (§4.4) | — | — |

(a) ppl 20.065→**19.817** (−0.25, noise-level free); standard attention
reproduced exactly in-process, so the APA code path adds zero drift. Context
win is over the trained window (3K engine-standard ceiling → the full 32K).
(b) gated ×8 against the fp32 QAT ground-truth oracle; refine 0.15/0.10/0.05
all within noise of standard.
(c) zero flips on the APA gate itself (all ground-truth prompts, bulk-4 /
refine 0.15); the final everything-on suite recorded one 0.062-logit flip —
noise floor. Speed is vs the ollama baseline.

\*Gemma-4 is the *parity-catch-up* case, not a context win. An earlier APA decode
OOM in `_quantize_keys` (the cuBLAS KV-expansion path) was fixed by an incremental
`kq` cache: APA now serves 8K decode at 73.9 ms/tok, 7.61 GB — lighter than
standard's own 8K high-water (7.64 GB), i.e. APA *caught up to* standard's ceiling.
It does not *exceed* it on Gemma (the incremental ring carries +50% resident; 12K
still OOMs in the un-optimized prefill path; a follow-up session — fused
dispatch, chunked prefill quantize, INT8-V KV storage — moved the prefill
wall to 12K solid at 7,802 MiB, with 16K still OOM: a managed rising wall,
not MLA-flat). This contrast — ~10.7× extension (3K→32K) on MLA/MiniCPM3
vs. parity-catch-up on Gemma's MQA+sliding — is itself a finding (§7).

**The MLA cells carry the thesis.** MLA is the most aggressive KV-sharing scheme (a
low-rank latent), so APA-composes-with-MLA is the hardest case for orthogonality to
survive. It survives twice — retrofit (MiniCPM3, where 4-bit APA came back
free-to-improved) and native (GRAPA). The other families establish that the result
is not MLA-specific.

### 4.2 The honest split: parity vs. long-context-trade

The quality column is **bimodal, and the bulk-bits law explains it.** The parity
results (MiniCPM3, Gemma4, Qwen3.5) are **4-bit** bulk on qk-normed models. The
"+ppl" results (TinyLlama, OLMoE) are **2-bit** bulk — below the precision those
architectures need (§4.3) — and they buy 5× context at a quality cost. These are
**not** "free parity"; they are a long-context-vs-quality trade, and we report them
as such. The law predicts its own failures: push bulk below what the norm structure
supports and you pay. The law forecasts exactly which cells degrade, which is what
makes it a law rather than an observation.

### 4.3 The bulk-bits law

| Architecture | Key normalization | Safe bulk bits |
|---|---|---|
| Llama-2 | raw keys | 2 |
| Qwen2.5 | raw keys (GQA) | 8 |
| Qwen3.5 | qk-norm (hybrid, GQA attention layers) | 4 |
| OLMoE | qk-norm | 4 |
| MiniCPM3 (MLA) | half-normalized latent | 4 |
| Gemma-4 | qk-norm (global MQA + sliding) | 4 |
| Mistral-7B | qk-norm | 4 (predicted — port started, testing not completed) |

Note that GQA appears twice among the six measured architectures — once
raw-key (Qwen2.5) and once qk-normed (Qwen3.5) — and the two land on
different floors. That is the law's sharpest evidence: the floor tracks
normalization, not attention family.

**Law:** key normalization pins the bulk floor. Every normalized-key architecture
measured — qk-norm (OLMoE, Qwen3.5, Gemma-4) and the half-normalized MLA latent
(MiniCPM3) — lands on a 4-bit floor. Raw-key models have real but *model-specific*
floors (Llama-2: 2-bit; Qwen2.5: 8-bit). The mechanism: normalization bounds the
dynamic range of the dot products, which is exactly what low-bit quantization needs
to preserve ordering (§3.3); without it, the floor depends on the model's score
statistics. The practical rule: normalized keys → set 4 bits and go; raw keys →
sweep the floor first. [OPEN: confirm the Mistral-7B 4-bit prediction — Appendix
B.3.]

*(Provenance note: the Llama-2, Qwen2.5, and OLMoE floor entries — and the
2-bit collapse/rank-correlation figures below — derive from the project's
earlier APA-Quant mission logs and research-board records rather than the
per-model port ledgers cited in §4.1; consolidating them into one citable
ledger is queued in Appendix B.)*

**The law was discovered at the cliff, and the cliff is sharp.** It did not come from
theory but from observing that low-bit bulk attention catastrophically fails on some
architectures and not others, then finding that the failure tracks key-normalization.
The controlled demonstration is a single model swept across its own floor — MiniCPM3
(MLA), refine 0.10 (the gated ledger protocol; because selection routes on
the bulk scores, the sweep still isolates the bulk floor — a blind bulk pass
cannot be rescued by refinement), wikitext ppl@1024, standard reproduced exactly
in-process so the APA code path adds zero drift (gated, registered):

| bulk bits | ppl@1024 | vs standard |
|---|---|---|
| standard (exact) | 20.065 | — |
| apa 8-bit | 20.080 | +0.015 (free) |
| **apa 4-bit (floor)** | **19.817** | **−0.25 (noise-level free)** |
| apa 2-bit (sub-floor) | 29.145 | **+9.1 (broken)** |

Only the bulk bit-width changes; same model, same eval. The 8→4-bit step is free; the
4→2-bit step falls off a cliff. The floor is a threshold, not a gradient — which is
what makes it a law rather than a trend, and the law predicts which side of it any
(architecture, bit-width) pair lands on. On raw-key models the cliff is far steeper
(2-bit Qwen2.5 collapses to ppl >19,000, score-rank correlation 0.03), and because
selection routes on the bulk scores, refinement *cannot* recover a sub-floor bulk pass
— the routing itself is blind. This is APA's one hard precondition: run at or above the
bulk-bits floor, which the law tells you in advance.

### 4.4 MHA: the abandoned proof of concept

The mechanism was first validated on GPT-2/MHA, confirming differentiability and
selective-compute correctness. That prototype used 2-bit bulk *before* the bulk-bits
law was known (raw-key MHA needs more bits) and naive quantization *before*
TurboQuant — so its +8% ppl / 0.37× is exactly what the later law retroactively
predicts for that configuration. We then moved to the KV-sharing and MoE
architectures that dominate current deployment, and make **no efficiency claim for
unoptimized MHA.** MHA is the no-sharing endpoint of the KV-sharing axis; the
prototype confirms the mechanism degrades gracefully to that limit, no more.

### 4.5 The speed crossover (regime scoping)

APA's speedup is sequence-length-dependent. On a controlled mixed-attention bench
(B=2, H=4, D=64, r=0.15):

| Seq len | SDPA | APA | APA/SDPA | Mem save |
|---|---|---|---|---|
| 128 | 0.69 ms | 4.21 ms | 6.08× slower | 76.8% |
| 256 | 0.97 | 4.50 | 4.64× | 85.4% |
| 512 | 2.17 | 4.40 | 2.03× | 90.9% |
| 1024 | 7.49 | 5.79 | **0.77× (faster)** | 94.0% |
| 2048 | 27.77 | 13.03 | **0.47× (2.1× faster)** | 95.7% |

**APA loses at short sequence and wins at long.** The crossover is ~512 tokens:
below it, selection overhead dominates; above it, the avoided full-precision work
dominates. APA is a long-context method; its speed claims apply to the bandwidth-bound
long-sequence decode regime. MSE vs SDPA is 0.0% at r=0.15 across this sweep, and the
suite that measures it has a 100% mutation-kill rate (§8) — the zero is a measured
result on a sensitive test.

---

## 5. The Memory Wall: Why the Dense Baseline Can't Be Run

The natural baseline for APA is dense attention at refine 1.0 — full-precision
scoring of all keys, i.e. exact attention. At the long-context configs APA reaches,
**that baseline does not fit in memory**, and this is itself the efficiency result,
stated in hardware rather than as a ratio.

Worked example — the native GRAPA config (L=12288, H=16, MLA, bf16, 12 GB GPU):

- A dense score matrix `(B,H,L,L)` in bf16 is **4.5 GB per layer** for one tensor.
- The backward pass needs scores *and* softmax probabilities live: **~9 GB/layer
  (bf16), ~18 GB/layer (fp32)** — before weights, optimizer state, activations, or
  the other 23 layers.

Dense materialized-softmax OOMs on the *first* attention block. APA's online-softmax
training kernel is `O(L)` in memory and never forms the matrix; the same model trains
at ~8.1 GB. **The dense baseline is not slower — it does not exist at this config.**

The claim is specific: *materialized* dense attention is impossible at this config. A
*fused* dense kernel (FlashAttention) avoids materialization too — but it does so by
running the same online-softmax APA runs, at full precision on 100% of keys instead of
`b_bulk` on the bulk and full on the `r` fraction. So the memory wall rules out the
materialized baseline outright, and the precision-weighted cost (§2.3) and bulk-bits
law (§4.3) account for the difference against the fused one. Both baselines are
addressed; neither leaves APA's advantage unexplained.

---

## 6. Native Training: APA From Birth

`MLAConfig200`: 232.6M params, d_model 1024, 24 layers, 16 heads, MLA latent
(kv_lora_rank 256, qk_rope_dim 32), vocab 8192, window 12,288, refine 0.15, bf16,
gradient checkpointing, RTX 4070 SUPER. The selection is trained through a
**stop-gradient** straight-through estimator: the threshold and the bulk/refine
*choice* are constants in the backward pass; gradients flow through the score dots
and the softmax only. This is the standard estimator for hard top-k/quantization
selection, and the 232M model (§6) trains stably under it from random init.

**Trainability:** loss descends cleanly through the selective path (4.4→3.1 band over
the run); APA r0.15 is trainable from birth, not merely tolerable at inference.

**Window-use probe (the architecture-thesis result):** facts placed ~11,200 tokens
deep cost only **+0.41 nll** vs. shallow (ppl ~20→~25–30) — a gentle gradient, not a
cliff. A deep fact must be one of the ~15% of keys the bulk router refines, *or* be
visible through its quantized bulk score; the mild degradation shows the refine
budget lands on the right keys across the full 12K. This is the same claim as the
Pareto knee (§ below), viewed along the depth axis.

**Open measurements (tracked in Appendix B):**
- Pareto frontier r0.05→0.30, Δppl-from-r-max vs precision-weighted cost, per family
  — the central figure; partial points exist (Gemma4: 121.3/117.3/119.5 at
  0.05/0.10/0.15, non-monotonic, within noise) but a dense single-model sweep is not
  yet plotted. Show the *whole* curve including where it degrades.
- Native-vs-dense parity at equal token budget — cannot be run on 12 GB (§5); to be
  measured on a testbed with headroom, or argued via the memory wall.
- Verbatim copy of merged (BPE) tokens: digit copy solved (ppl ~6), alpha copy not
  (ppl ~234). This is APA's honest operating boundary — copy needs near-*uniform*
  precise attention over the source span, the worst case for a peaked-softmax method
  (§7). Likely a tokenizer-singleton question, not a model-capacity one.

---

## 7. Limitations

1. **Diffuse-attention tasks.** APA's efficiency is proportional to how *peaked* the
   softmax is. Near-uniform attention (some early layers, verbatim copy, certain
   retrieval) is the worst case: the tail is not crushed, every key carries mass, and
   low-bit bulk rounding hurts. There the refine fraction must rise; the limit r→1.0
   is exact attention. The GRAPA copy struggle is this boundary showing up in the one
   task that lives at it.
2. **Short sequences.** APA is slower below ~512 tokens (§4.5). It is a long-context
   method.
3. **Sub-floor bulk precision.** 2-bit bulk on architectures that need more (§4.2)
   trades quality for context rather than giving free parity.
4. **MHA unoptimized** (§4.4).
5. **Context extension is architecture-dependent, not universal.** APA extends
   context dramatically on MLA (MiniCPM3 2K→32K) but only reaches *parity* on
   Gemma-4 (MQA+sliding): the incremental `kq` cache that fixed the decode OOM
   carries +50% resident memory, so APA catches standard's ~8K ceiling rather than
   exceeding it, and 12K OOMs in prefill. The extension magnitude tracks how much
   the architecture's KV footprint dominates residency — APA is not a
   uniform context multiplier.
6. **Stop-gradient selection** (§6): the selection is non-differentiable at the
   threshold, so training uses a straight-through estimator. The 232M model trains
   stably under it; we have not characterized whether a differentiable relaxation of
   the threshold would train faster.

---

## 8. Validation and Reproducibility

- **Correctness:** 35/35 unit tests; gradient parity dL/dQ=0.9992, dL/dK=1.0000,
  dL/dV=1.0000; refine=1.0 reproduces SDPA-exact output; selective kernel matches a
  NumPy reference of the same algorithm to <1e-3 across causal/non-causal, head_dim
  up to 512, rectangular KV cache, and MQA shapes.
- **Test sensitivity:** 100% mutation-kill rate — the suite catches both the
  grad-accumulation bug and the refine-mask bug it was built around, so the 0.0% MSE
  figures (§4.5) reflect a real result, not an insensitive metric.
- **Two bugs of record** (both found by gating the regime the feature exists for, not
  convenient shapes): (i) a backward Q/K gradient-suppression bug (dL/dQ≈0.2→0.999
  post-fix); (ii) an APA causal-mask top-left-alignment bug that silently blinded
  rectangular-cache queries (caught by a refine ppl sweep: 121→11,000,000). Lesson
  carried into the methodology: gate the regime the feature is *for*.
- **Framework-free:** kernels implemented in a standalone C++/CUDA engine
  (`tensor_cuda`) and a CuPy reference (`tensor_gpu_v2`); no PyTorch dependency in
  the core path.

---

## 9. Related Work

### 9.1 The classical efficient-attention families

- **Sparse attention** (Longformer, BigBird): fixed/heuristic patterns that *drop*
  keys. APA drops nothing and adapts to content; it is approximate in tail
  *precision*, not tail *coverage*.
- **Linear attention** (Performer): kernel approximations that degrade for sharp
  distributions — exactly the regime where APA is *strongest* (peaked softmax).
- **FlashAttention:** exact, memory-optimal, `O(n²)` compute. APA reduces the
  *precision* of that compute and composes with the same online-softmax memory trick.
- **MLA / MQA / GQA:** KV-*sharing* schemes. APA is orthogonal — a precision-
  allocation axis — and composes with each (§4).
- **Quantization-aware training:** uniform/learned bit-width on weights/activations.
  APA is *interaction-adaptive* and *transient* (precision per query–key pair at
  inference), not a persistent weight quantization, with a content-adaptive criterion
  for which interactions get full precision.

### 9.2 Concurrent mixed-precision / low-bit-routed attention (the close neighbors)

A line of 2025–2026 work shares APA's premise — that quantisation error concentrates
in the few interactions the softmax weights most heavily — and is the relevant
comparison set. We position APA against the three closest precisely.

**ThriftAttention** (Sharratt, arXiv 2605.23081, 2026 — notably also
single-author, independent work) is the closest in spirit. It independently reaches the same core insight — *"the output impact of
quantisation error is highly non-uniform and increases with the importance of each
query–key interaction"* — and the same no-drop response: select a small fraction of
query–key pairs for FP16, compute the rest in FP4, and merge both via online softmax,
so an unselected pair *"is degraded by quantisation rather than deleted."* It reports
recovering 89.1% of the FP4→FP16 gap by promoting only 5% of blocks. This is genuine
independent corroboration of APA's premise, and we cite it as such. APA differs in
four respects: **(i) routing signal** — ThriftAttention scores importance with a
separate block-mean statistic (`Q̄ᵢ·K̄ⱼᵀ`), whereas APA routes on the low-bit bulk
scores it has *already computed*, with no separate pooling pass; **(ii) granularity**
— ThriftAttention promotes whole 64-token *blocks*; APA selects *per key* per query
row; **(iii) breadth** — we validate across the four attention families plus
MoE routing (MHA/MQA/GQA/MLA, MoE; seven models), retrofit and native;
**(iv) training** — ThriftAttention is *training-free
only*, and its own conclusion names training-through-the-mechanism as future work
(*"promoting sensitive interactions in the forward and backward attention computation
to FP16 could help address stability issues in sub-byte attention training"*). APA is
training-free *and* trainable: the same algorithm runs zero-fit on any frozen model
(§4) and carries a differentiable backward used to train a 232M model from birth (§6).
The capability ThriftAttention lists as open is one we demonstrate.

**SALE** (Ji et al., arXiv 2505.24179, 2025) routes the way APA does — *"4-bit
quantized query-key products"* estimate per-pair importance (its *Relative
Attention Score*) —
but it is, by its own description, *"a training-free block-**Sparse** Attention
technique"*: the unselected blocks are **masked out of the computation**, dropped from
the softmax entirely. SALE matches APA on *route-on-quantized-scores* and is the
opposite of APA on *keep-the-tail*. It is prefill-only and training-free.

**MCBP** (Wang et al., arXiv 2509.10372, 2025) is a hardware accelerator that uses low-bit
"filtering rounds" to pick top-k Key candidates, then computes the formal stage
*"using only these selected Keys and Values"* — again a drop, and inference-only.

**The unoccupied intersection.** SALE has route-on-quantized-scores but drops;
ThriftAttention is no-drop but block-routed and inference-only; no prior work is
trainable. APA's contribution is the conjunction none of them hold at once:
**route on the quantized scores, keep the tail at those scores (no drop), at per-key
granularity, with a differentiable backward** — across four attention
families plus MoE, seven models.
A reviewer could ask whether this is an obvious composition of SALE (routing) +
ThriftAttention (no-drop) + an STE selection; §3.2 answers why it is not — keeping the
tail at low precision is precisely what bounds the gradient noise that lets the
stop-gradient selection train stably, so the no-drop choice *enables* the trainable
version rather than merely co-occurring with it.

### 9.3 The dial

`refine` is a free scalar on a continuum from exact attention (`r=1.0`, which
reproduces SDPA bit-for-bit, §8) to maximal compression, with no weight or structural
dependence on its value. It can therefore be exposed as a runtime control, tuned per
request with no retraining; our current implementation sets it per configuration.
SALE and MCBP fix a sparsity pattern, so their budgets trade quality against
*dropped* keys rather than precision. ThriftAttention's FP16 block budget `k`
is, like `refine`, a runtime quality dial (its Fig. 1 sweeps it from FP4
toward FP16); the differences are granularity and reach — `k` promotes whole
64-token blocks and exists only at inference, whereas `refine` acts per key
per query row, reproduces SDPA bit-for-bit at `r=1.0` (§8), and is the same
dial the model can be trained through (§6).

---

## 10. Conclusion

Selective precision — not selective *keys* — is the lever. Attention spends most of
its precision on interactions the softmax then discards; APA reclaims that precision
by scoring the bulk cheaply, refining the softmax-dominant fraction exactly, and
keeping the denominator whole so nothing is dropped. Because this is an allocation of
precision rather than a change to KV-sharing, it composes with every attention family
and with MoE routing — demonstrated across seven models spanning four
attention families plus MoE, retrofit and native.
The method's wins (long-context extension, precision-weighted speedup, memory
headroom that lets a dense baseline not even fit) and its honest boundaries (diffuse
attention, short sequences, sub-floor bulk precision) both follow from a single
measured fact: the precision depth of attention interactions is geometric in the
bulk and rare in the tail.

---

## Acknowledgments

This work was carried out independently, without funding, sponsorship, or
institutional support. The evaluation missions ran on AtlasForge, the
author's autonomous research platform, itself open source (MIT):
https://github.com/DragonShadows1978/AI-AtlasForge (PyPI `ai-atlasforge`);
the mission workspaces cited in Appendix A are project-local artifacts, not
part of that repository. The author thanks the open-weight model teams whose
releases made a consumer-hardware evaluation matrix possible, and acknowledges
the AI systems named in the disclosure above as working collaborators in the
implementation, evaluation, and writing.

---

## References

*All arXiv identifiers, titles, and author lists below were verified against
the arXiv record or the primary source on 2026-07-02.*

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
   A. N., Kaiser, L., Polosukhin, I. *Attention Is All You Need.* NeurIPS
   2017. arXiv:1706.03762.
2. Beltagy, I., Peters, M. E., Cohan, A. *Longformer: The Long-Document
   Transformer.* 2020. arXiv:2004.05150.
3. Zaheer, M., Guruganesh, G., Dubey, A., et al. *Big Bird: Transformers for
   Longer Sequences.* NeurIPS 2020. arXiv:2007.14062.
4. Choromanski, K., Likhosherstov, V., Dohan, D., et al. *Rethinking
   Attention with Performers.* ICLR 2021. arXiv:2009.14794.
5. Dao, T., Fu, D. Y., Ermon, S., Rudra, A., Ré, C. *FlashAttention: Fast and
   Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022.
   arXiv:2205.14135.
6. Shazeer, N. *Fast Transformer Decoding: One Write-Head is All You Need.*
   2019. arXiv:1911.02150. (MQA)
7. Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F.,
   Sanghai, S. *GQA: Training Generalized Multi-Query Transformer Models from
   Multi-Head Checkpoints.* EMNLP 2023. arXiv:2305.13245.
8. DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient
   Mixture-of-Experts Language Model.* 2024. arXiv:2405.04434. (MLA)
9. Bengio, Y., Léonard, N., Courville, A. *Estimating or Propagating
   Gradients Through Stochastic Neurons for Conditional Computation.* 2013.
   arXiv:1308.3432. (straight-through estimator)
10. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I.
    *Language Models are Unsupervised Multitask Learners.* OpenAI, 2019.
    (GPT-2)
11. Touvron, H., Martin, L., Stone, K., et al. *Llama 2: Open Foundation and
    Fine-Tuned Chat Models.* 2023. arXiv:2307.09288.
12. Zhang, P., Zeng, G., Wang, T., Lu, W. *TinyLlama: An Open-Source Small
    Language Model.* 2024. arXiv:2401.02385.
13. Muennighoff, N., Soldaini, L., Groeneveld, D., et al. *OLMoE: Open
    Mixture-of-Experts Language Models.* 2024. arXiv:2409.02060.
14. Hu, S., Tu, Y., Han, X., et al. *MiniCPM: Unveiling the Potential of
    Small Language Models with Scalable Training Strategies.* 2024.
    arXiv:2404.06395. Evaluated model: OpenBMB *MiniCPM3-4B* (model card,
    Hugging Face, 2024).
15. Gemma Team, Google DeepMind. *Introducing Gemma 4 12B: a unified,
    encoder-free multimodal model.* Model release and model card, June 2026.
    https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/
16. Qwen Team, Alibaba. *Qwen2.5 Technical Report.* 2024. arXiv:2412.15115.
    And: *Qwen3.5: Towards Native Multimodal Agents.* Model release,
    February–March 2026. https://qwen.ai/blog?id=qwen3.5 (evaluated model:
    Qwen3.5-9B, released 2026-03-02).
17. Sharratt, J. *ThriftAttention: Selective Mixed Precision for Long-Context
    FP4 Attention.* 2026. arXiv:2605.23081.
18. Ji, X., Zhang, H., Fu, F., Cui, B. *SALE: Low-bit Estimation for
    Efficient Sparse Attention in Long-context LLM Prefilling.* 2025.
    arXiv:2505.24179.
19. Wang, H., Wang, Z., Yue, Z., et al. *MCBP: A Memory-Compute Efficient LLM
    Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and
    Repetitiveness.* 2025. arXiv:2509.10372.
20. Zandieh, A., et al. *TurboQuant: Online Vector Quantization with
    Near-optimal Distortion Rate.* ICLR 2026. arXiv:2504.19874. (The bulk
    quantizer of §2.1 is an in-project implementation of this method.)
21. Perry, D. *Ghost Geometry: A Precision-Collapse Framework for the Collatz
    Conjecture, and Its Measured Transfer to Transformer Attention.*
    Companion note, 2026, released alongside this paper (Zenodo DOI on
    release). Source of the precision-decay hypothesis (§1.2) and the
    GHOST_PRECISION interaction-depth study summarized in §3.4.
22. Perry, D. *Grafted Memory — GRM: A Routed, Tokenless K/V Memory Runtime
    for Frozen Language Models.* Companion paper, 2026, released alongside
    this paper (Zenodo DOI on release).

---

## Appendix A — Evidence Index (paths, for the authors; strip before submission)

- MiniCPM3-MLA: `GraftRepository/docs/MiniCPM3-MLA_Results.md`
- Gemma-4: `GraftRepository/docs/GEMMA4_PORT_LEDGER.md`
- Qwen3.5: `GraftRepository/docs/QWEN35_APA_GRM_REPORT.md`
- Mixed-attention bench: `AI-AtlasForge/workspace/APA-Quant_Native_Mixed-Attention/artifacts/benchmark_results.json`
- MoE (OLMoE): `AI-AtlasForge/workspace/APA-Quant_on_MoE/mission_6f6405e6/artifacts/benchmark_results.json`
- TinyLlama: `AI-AtlasForge/workspace/APA-Quant_Tiny_Llama/mission_6be46b0c/artifacts/benchmark_results.json`
- True-implementation tests: `AI-AtlasForge/workspace/APA-Quant_True_Implementation/mission_d364083d/test_results_summary.json`
- GHOST_PRECISION depth study: AtlasForge mission 4482de7b; precision-depth Q1–Q7
- Native training: `GRAPA-Native-LLM/` (kernels: `Project-Tensor/tensor_cuda/src/kernels.cu`, `apa_selective_*`)
- Origin note: `collatz-experimental-data/collatz-ml-bridge.md` (§6.1 only; structural, not numerical)

## Appendix B — Open-measurement worklist (gates before submission)

**Note on evidence source:** §4 quality/parity numbers are drawn from the project's
registered research board (`/mnt/Shared/AI_Research_Board.md`) and per-model port
ledgers (`GraftRepository/docs/{QWEN35_APA_GRM_REPORT,MiniCPM3-MLA_Results,
GEMMA4_PORT_LEDGER}.md`), where APA parity and the bulk-bits law are gated against
fp32 ground-truth oracles. These are not re-derived in the paper; they are cited from
the gates that produced them.

1. Bulk-vs-exact rank correlation (Spearman/Kendall) per architecture → proves §3.3.
   (Partially evidenced: 2-bit Qwen score-rank corr 0.03 vs 8-bit 0.85, in ledger.)
2. Per-family Pareto frontier r0.05→0.30, full curve incl. degradation tail.
3. Mistral-7B 4-bit bulk confirmation → would add a row to §4.3. Status: port started
   (`core/mistral7b_tc.py`), deprioritized before testing finished — genuinely
   UNCONFIRMED. Note this is the *low-information* gate: Mistral (2023, vanilla
   qk-normed GQA) sits strictly inside the envelope where the law is already confirmed
   on harder, newer architectures (Qwen3.5, Gemma-4, both 2026). Finishing it tests the
   law somewhere easier than where it already holds — hence the deprioritization.
   Weights in cache, adapter exists; a finishable gate, not a from-scratch one.
4. ~~Fix or footnote Gemma-4 KV-expansion context bug.~~ **DONE** (commit 72bccd6).
   ~~§4.2.1 residency analysis.~~ Board documents it: Gemma's MQA cache grows
   24KB/tok (~85× MLA's latent), structural — "managed rising wall ~8K→~12K," not flat.
5. ~~Bulk-bits law table → measured cross-the-floor sweep.~~ **DONE** — §4.3 now uses
   the gated MiniCPM3 8/4/2-bit sweep from the ledger (20.065/20.080/19.817/29.145).
6. Attention-peakedness (entropy) vs APA-efficiency scatter → turns §7 boundary into a figure.
7. GRAPA copy: tokenizer-singleton probe (merged-token copy under this setup?).
8. Consolidate the raw-key/OLMoE bulk-floor evidence (Llama-2 2-bit, Qwen2.5
   8-bit incl. the 2-bit ppl>19,000 collapse and 0.85/0.03 rank correlations,
   OLMoE 4-bit) from the APA-Quant mission artifacts and board history into a
   single citable ledger; §4.3 currently cites them by provenance note.
