# APA × MQA Root-Cause Plan (APAMQ)

Date: 2026-08-13. Lead: Fable session. Question raised by David.
Plan is IMMUTABLE after initial commit (house rules). Ledger:
`docs/APA_MQA_ROOTCAUSE_LEDGER.md`.

## Question

Why did APA on Gemma-4 12B local (MQA globals: 1 KV head × D=512, K=V shared
projection) deliver only ceiling-parity with standard attention — and slower
(3.5× prefill, +8% decode) — instead of the memory extension seen on
MiniCPM3/MLA and the external 26B kv=2 GQA receipt? Specifically: is
parity a property of single-KV-head attention (the closed mission's
"APA is a multi-KV-head thing" law), or of this architecture instance
(40/48 sliding layers, D=512, 6.8GB body) plus this implementation
(kqb ring +50% resident, fused-kernel wiring gaps)?

## Prior record (anchors, not re-litigated)

- `docs/GEMMA4_MQA_ADJUDICATION.md` — operative APA-negative verdict.
- A0 re-probe 2026-07-04 (12GB card): no OOM either mode at full 32K;
  APA +0.4–3.3% peak, 3.5× prefill, +8% decode ms/tok.
- June 2026-06-13 root-cause (conf 0.88): incremental-kq deficit (fixed),
  fused kernel forbidden at D=512 (TC_APA_MAXD=256 then), no KV storage
  quant. 3070-era walls were 8GB artifacts.
- GraftRepository `docs/GEMMA4_APA_AUDIT_A1.md` — open ~110MB unexplained
  allocator high-water at 4K.
- RECON 2026-08-13 (this session): `tensor_cuda/src/kernels.cu:1053` now
  reads `constexpr int TC_APA_MAXD = 512; // bumped 256->512 for Gemma 4
  global` — the June "kernel-forbidden at D=512" clause may be stale.

## Hypotheses

- **H-A (KV-independence of the transient win):** the fused
  `apa_selective_attention` path's peak-transient advantage over the
  standard path scales with q-heads × S and is independent of KV-head
  count — i.e. fully available at kv=1.
- **H-B (layer mix):** Gemma's binding memory term lives in
  APA-ineligible places (sliding-layer prefill transients, 6.8GB
  weights); even perfect APA on the 8 global layers moves little.
- **H-C (implementation overhead):** kqb ring (+50% KV resident) plus
  quantize transients cancel whatever the eligible layers save.
- **H-D (stale kernel cap):** with TC_APA_MAXD=512, the fused path may
  now be legal on Gemma decode but simply unwired/unexercised in the
  port.

## Experiments

- **E1 — MQA-geometry kernel transient sweep** (Project-Tensor, order
  `orders/APAMQ_E1_mqa_transient_sweep.md`). Evidence class: **kernel
  sweep** — speed + memory shape ONLY, no model-quality claims.
  Standard vs fused-APA attention at kv ∈ {1, 4, 8, 16} × q=16,
  D ∈ {128, 512}, S up to 64K, prefill-chunk and decode shapes.
- **E3 — Gemma-4 local peak attribution + perfect-APA upper bound**
  (GraftRepository, order `orders/APAMQ_E3_gemma_peak_attribution.md`).
  Evidence class: **instrumented port measurement** — memory/speed
  attribution ONLY, no quality claims. Per-phase, per-layer-class peak
  attribution standard vs APA; component sizes vs S (kqb ring, quantize
  transients, score transients, mask caches); decode-path audit under
  TC_APA_MAXD=512; the A1 ~110MB item; arithmetic inputs for the
  perfect-APA reclaimable bound.
- **E2 — coherent-noise decorrelation ablation** (quality axis):
  REGISTERED, NOT DISPATCHED. Contingent follow-up; the current
  question is memory/speed, not perplexity.

## Registered thresholds (before results, per house rules)

- **T1 (E1 / H-A):** H-A CONFIRMED if the fused-vs-standard
  peak-transient ratio at kv=1 is within ±20% (relative) of the same
  ratio at kv=4 and kv=8 for matched (q=16, D, S) cells. Larger
  deviation = KV-head count genuinely enters the memory economics;
  report the direction.
- **T2 (E3 / H-B vs H-C):** verdict "parity preordained by the
  architecture instance" if the perfect-APA reclaimable bound
  (eliminable global-layer score/softmax transients minus irreducible
  APA state) is < 300 MiB at S=16K on the 12GB card. Verdict
  "unrealized win — implementation tickets follow" if > 1 GiB.
  Between: boundary; judgment deferred to synthesis, named as such.
- **T3 (E3 / H-D):** if `apa_selective` engages on Gemma global decode
  at D=512 without error at S ≥ 8K, the June "fused kernel forbidden"
  clause is STALE; successor docs get a dated correction (the
  adjudication itself is not edited).

## Out of scope

- KV/weight STORAGE quantization (B4, prefill-into-rings) — declined by
  David 2026-07-04; scope law "saving memory on storage is not APA."
- Reopening the Gemma product decision. This is a why-investigation;
  any law revision is David's call on the synthesis.

## Seats & discipline

Codex Sol (`-m gpt-5.6-sol`) via codex-shim, two parallel seats, GPU
serialized by `flock -w 7200 /tmp/forge-gpu.lock` (operator right of
way). Engine sources and the built .so are READ-ONLY for both seats
(runtime-shared dependency). Lead commits all work; seats never run
git. Every claim names its evidence class. Synthesis + board update
follow results.
