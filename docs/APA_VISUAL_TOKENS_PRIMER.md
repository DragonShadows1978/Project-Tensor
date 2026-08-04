# APA → Visual Tokens: A Primer

**Thesis: APA is an attention primitive, not a text trick.** Its premise
is a distributional claim about attention score vectors — nothing in it
references language, causality, or positional scheme. As of 2026-07,
that claim has receipts on both sides of the modality line: causal text
attention (GPT-OSS, MiniCPM3, Qwen3.5, Trinity) and bidirectional
visual attention (the Hunyuan3D-2.0 shape DiT in the torch-free TensorCUDA transformer replacement).

David's formulation, verbatim (2026-07-15, before the visual receipts
existed): *"It's an ATTENTION mechanism regardless of WHAT kind of
attention it is. Bottom line is if there IS a bulk tail, then APA
SHOULD work... if there's a forward pass over tokens, be they letters
or pixels? It should work."* The experiments below tested that, and it
held.

## 1. The premise (IME lineage)

APA descends from IME: two non-coincident measurement geometries
produce a geometric bulk plus a heavy tail. Applied to attention: for a
given query, the score vector over keys concentrates its post-softmax
mass on a small high-score tail. The bulk of keys contributes almost
nothing to the normalized output — so the bulk can be *scored against
low-precision keys*, and only the tail needs exact re-scoring.

Mechanism (engine implementation):

- Keys stored/scored in INT4 (symmetric, grouped) for the bulk pass.
- A refine fraction **r** (nominal 0.15) selected per score row by the
  engine quantile rule: threshold = mean + z·std of the bulk scores
  (Gaussian-quantile z for the target fraction).
- Selected keys re-scored at full precision; non-selected keys keep
  their bulk scores; V stays full precision; softmax is exact over the
  merged scores.

The load-bearing assumption is only: *the score distribution has a
bulk-plus-tail shape, and the tail is where softmax mass lives.* That
is a statement about attention geometry, not about what the tokens
mean.

## 2. What the text era established (evidence: kernel gates + perplexity + live-session receipts)

- Works across causal LLM families: GPT-OSS-20B (four-way stack, 96k
  context on a 12GB card), MiniCPM3, Qwen3.5-9B (zero-flip port gate),
  Trinity/NoPE (21% teacher-forcing flips, NLL-neutral).
- **Structural law, not modality law — [Gemma-4 12B / MQA](GEMMA4_MQA_ADJUDICATION.md)
  negative result:** APA is a multi-KV-head thing. [MQA on this target](GEMMA4_MQA_ADJUDICATION.md)
  fails it structurally — one KV head means
  the "bulk" has no slack to spend; the failure is coherent noise, not
  degradation. Selection rule: kv_heads ≥ 2, plus geometry conditions
  (head_dim bound, sufficiently long context, sufficient S). Externally
  boundary-validated (26B, kv=2, collaborator-reported).
- Scope law: "saving memory on storage is not APA." APA is a *compute/
  precision* asymmetry on live attention, not a KV-compression scheme.

## 3. The modality jump: Hunyuan3D transformer replacement (2026-07-13/14)

ColdCast is a personal project (not publicly available) that tests APA on Project-Tensor outside text LLMs by replacing Hunyuan3D's shape transformer with a torch-free TensorCUDA implementation.

**EXP-APA-1 — first bidirectional APA site ever.** Hunyuan3D-2.0 shape DiT
(3072 latent tokens, joint attention with conditioner tokens; fully
non-causal), INT4 bulk + r=0.15 exact refine (evidence: gate receipts
+ E2E generations):

- Single DiT step: max deviation 6.5e-3, cosine 0.99998.
- 4-step trajectory: drift 0.54, cosine 0.87 — deviations *compound*
  mid-trajectory.
- 50-step full generation: **visually equivalent mesh** (+0.26% faces,
  watertight).

This receipt belongs to Hunyuan3D-2.0. It does not establish APA evaluation on
Hunyuan3D-2.1, whose APA status remains `unconfirmed`.

The compounding-then-converging shape produced a law: **the full
sampling trajectory is an attractor.** Short-trajectory fixtures are a
fragile instrument for quantization deltas (a coarse-probe fixture
collapsed a mesh from 38 near-threshold cells despite 99.9% ISO
agreement); E2E is the honest arbiter for this class. This is the
visual-domain analogue of not judging a text model by teacher-forcing
flips (Trinity's APA-on-NoPE receipts said the same thing: 21% TF
flips, NLL-neutral).

**The promotion train (K1→K3, evidence: kernel gates + E2E + operator
eyeball):**

- Q-tile WMMA skeleton made fused APA *faster than the standard path*:
  APA-on E2E 199.9s vs 242.8s standard at merge time — on visual
  tokens, the INT4 bulk is a net speed win, not just a memory trade.
- Realized refine fraction 0.158–0.167 across different input images
  (nominal 0.15) — the quantile rule transfers to visual score
  distributions nearly unchanged. 2 of 48 blocks ran conservative
  (0.23–0.24) on non-Gaussian score rows: deviation in the safe
  direction (over-refining, never under).
- Quality re-gate: 6/7 image pairs within 1.79% face delta, 14/14
  watertight. The boundary pair (2.07%) *added coherent detail*
  (embossed relief on a sign back) rather than corrupting — operator
  eyeball was the closing gate.

## 4. What actually changed when tokens became pixels

The mechanism ported unchanged. The *instruments* did not:

1. **Score magnitudes are domain-dependent.** The fp16 threshold
   ladder — adequate at text magnitudes — overflowed at DiT magnitudes
   (up to 76% of rows saturated to +inf and refined nothing). Fix: fp32
   shifted-Welford statistics (K2). Notable honesty point: EXP-APA-1's
   quality had held *even while refining fewer rows than designed* —
   the mechanism was more robust than reported, not less.
2. **Gating philosophy shifts from stepwise to E2E.** Diffusion
   compounds per-step deviations mid-trajectory and then re-converges;
   any fp32 threshold change makes fused/composed refine masks stop
   bit-tracking (lattice-tie degeneracy law). Only full-generation
   A/Bs mean anything.
3. **r is a dial, not a constant.** Different architectures prefer
   different r (0.10–0.25 range observed across ports). A dynamic-r
   (bulk-score-conditioned) frontier was analyzed and came back a
   measured null — fixed-z holds for now; the per-architecture ideal-r
   question stays open.

Everything else — the INT4 grouping, the quantile rule, exact softmax
over merged scores, the multi-KV-head prerequisite — carried across
without modification.

## 5. Current frontier (registered, not run)

- **APA-on-paint**: the texture UNet's multiview attention (~24k-token
  joint sequences at 6 views, more under the 9-view arc) is the next
  registered site — larger sequences than the shape DiT, different
  attention limbs (MDA / PoseRoPE / reference / DINO cross-attention).
  Open experiment row.
- Non-causal APA at other sites: any bulk-tail attention qualifies by
  the thesis; each new site owes its own magnitude/threshold and E2E
  gates (the fp16-ladder lesson generalizes: port the mechanism, re-
  derive the instrument).

## 6. Honest boundary of the claim

APA is *not* free and *not* universal: it costs peak memory on some
ports, it fails [Gemma-4 12B's MQA geometry](GEMMA4_MQA_ADJUDICATION.md),
prefill-heavy regimes can pay 3.5×, and engaged-mode perplexity deltas of
+1.5–1.9%
exist on text ports that chose it for memory reasons. The claim this
primer makes is narrower and better: **wherever attention has a
multi-KV-head bulk-tail score geometry — text, mesh latents, and by
registered hypothesis any pixel/voxel token stream — the same
INT4-bulk / exact-tail primitive works, with domain-specific
instrument calibration.** Two modalities down; the paint UNet is next.

---
*Receipts: EXP-APA-1/2 and K1–K3 in the runtime-replacement project's
internal ledger (not publicly available); engine main (Q-tile/Welford kernels, 2026-07-14), [Gemma-4 12B adjudication](GEMMA4_MQA_ADJUDICATION.md)
(multi-KV law), Trinity T1 (NoPE), GPT-OSS 96k receipts. Evidence classes named
inline per house rules.*
