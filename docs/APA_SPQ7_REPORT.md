# APA-SPQ7 report: model-level A/B of the APA-SP operating point (Qwen3-1.7B, Qwen3-4B; wikitext-2 test)

Seat Sonnet 5.5 high. Order `orders/APA_SPQ7_MODEL_AB_PERPLEXITY.md`; ledger `docs/APA_SPQ7_LEDGER.md`; registration SHA-256
`0c1528a2679848ade715a90e376070fce39c3856a528fb3a1965cddd63bcb181` (`artifacts/apa_spq7/registration.json`, written before any evaluation).
Evidence class: end-to-end model evaluation of a torch fp32 SIMULATION of APA-SP semantics in every attention layer (teacher-forced, every query row over its causal prefix).
Scope: 10 non-overlapping 8192-token wikitext-2 test windows per model (indices 0,4,...,32,35; 81,910 scored tokens per model). It does NOT cover kernel speed (SPQ6), generation quality beyond next-token metrics, other
models/tasks, long-context retrieval, or the bf16 output rounding of the `tc` kernel (known extra error source, about 8e-4 relFro per layer output, NOT simulated here).
Reference "exact" = the unmodified model with its own bf16 sdpa attention. Every APA path computes attention in fp32 (TF32 off) and rounds the layer output to bf16, so even refine-everything (delta=inf, gate G1) differs from the reference at the bf16-rounding floor: mean KL about 8e-4..1e-3 nats, top-1 agreement 0.987..0.989, |dPPL| <= 0.16%. Differences below that floor are not detectable here.

## Verdict first
- G1 PASS (both models): delta=inf on test window 0: Qwen3-1.7B dPPL -0.004%, KL 1.01e-3, top-1 0.9867; Qwen3-4B dPPL +0.072%, KL 8.3e-4, top-1 0.9894 (registered: |dPPL|<=0.5%, KL<=3e-3, top-1>=0.97).
- G2 PASS: Qwen3-1.7B layer 11 (SPQ3 captured q/k/v, 16 heads x 4 configs = 64 cells), max |refined-fraction difference| 0.0 pp, max deviation ratio 1.009 (`artifacts/apa_spq7/G2_result.json`).
- TQ4 bulk with row carry preserves model quality at the operating points tested. At delta 8.1: dPPL +0.28% (1.7B), -0.07% (4B); at SP2 per-head eps=1e-2: -0.11% / -0.03% (inside the bf16 floor, i.e. no detectable change).
- The cost is small but visible at delta 8.1 on 1.7B: top-1 agreement 0.962, KL 0.0136 (13x the floor), single-window dPPL up to +1.01%. Keeping layer 0 exact removes it (KL 1.1e-3, top-1 0.986).
- INT4 sym-7 bulk at delta 8.1 BREAKS the model: dPPL +179% (1.7B), +12.9% (4B). Even with layer 0 exact: +5.0% (1.7B), +0.01% (4B).
- Predictions: P1 HELD, P2 HELD, P3 MISSED as registered (the 3x refined-fraction clause failed on 4B: 2.6x; the PPL clause held but only by 0.07 / 0.06 pp, inside the noise), P4 HELD, P5 HELD.


### Qwen3-1.7B (bf16; 10 test windows x 8192 tokens, 81910 scored tokens; PPL_exact = 14.4578)

| operating point | PPL | dPPL % pooled | dPPL % per-window min..max | top-1 agree w/ exact | mean KL (nats) | refined fraction |
|---|---|---|---|---|---|---|
| tq4_row_d8p1 | 14.4987 | +0.283 | -0.41..+1.01 | 0.9619 | 0.01355 | 0.1603 |
| tq4_row_d10 | 14.4683 | +0.073 | -0.21..+0.53 | 0.9719 | 0.00694 | 0.3027 |
| tq4_row_e1em1 | 14.4418 | -0.111 | -0.19..+0.03 | 0.9860 | 0.00105 | 0.3322 |
| tq4_row_e1em2 | 14.4425 | -0.106 | -0.25..+0.10 | 0.9871 | 0.00102 | 0.4755 |
| tq4_part_d8p1 | 14.5084 | +0.350 | -0.29..+1.19 | 0.9656 | 0.01106 | 0.4828 |
| int4_row_d8p1 | 40.3421 | +179.033 | +123.98..+326.14 | 0.5969 | 1.29953 | 0.1991 |
| tq4_row_d8p1_L0x | 14.4373 | -0.142 | -0.28..+0.01 | 0.9863 | 0.00111 | 0.1521 |
| tq4_row_d10_L0x | 14.4439 | -0.097 | -0.22..+0.02 | 0.9871 | 0.00101 | 0.2940 |
| tq4_row_e1em1_L0x | 14.4494 | -0.058 | -0.17..+0.10 | 0.9862 | 0.00103 | 0.3076 |
| tq4_row_e1em2_L0x | 14.4435 | -0.099 | -0.20..-0.00 | 0.9867 | 0.00100 | 0.4562 |
| tq4_part_d8p1_L0x | 14.4417 | -0.111 | -0.28..-0.02 | 0.9866 | 0.00106 | 0.4840 |
| int4_row_d8p1_L0x | 15.1817 | +5.007 | +1.25..+7.20 | 0.8742 | 0.10915 | 0.1695 |
| g1_inf (window 0 only, 8191 tokens) | 16.4060 | -0.004 | - | 0.9867 | 0.00101 | 1.000 |

### Qwen3-4B (bf16; 10 test windows x 8192 tokens, 81910 scored tokens; PPL_exact = 11.1739)

| operating point | PPL | dPPL % pooled | dPPL % per-window min..max | top-1 agree w/ exact | mean KL (nats) | refined fraction |
|---|---|---|---|---|---|---|
| tq4_row_d8p1 | 11.1659 | -0.072 | -0.29..+0.15 | 0.9875 | 0.00100 | 0.1772 |
| tq4_row_d10 | 11.1759 | +0.018 | -0.10..+0.19 | 0.9879 | 0.00089 | 0.3243 |
| tq4_row_e1em1 | 11.1647 | -0.082 | -0.16..-0.01 | 0.9873 | 0.00086 | 0.1642 |
| tq4_row_e1em2 | 11.1703 | -0.032 | -0.18..+0.17 | 0.9877 | 0.00085 | 0.3083 |
| tq4_part_d8p1 | 11.1726 | -0.011 | -0.20..+0.13 | 0.9876 | 0.00093 | 0.4633 |
| int4_row_d8p1 | 12.6156 | +12.902 | +9.70..+17.67 | 0.8770 | 0.16643 | 0.1736 |
| tq4_row_d8p1_L0x | 11.1688 | -0.045 | -0.16..+0.05 | 0.9876 | 0.00085 | 0.1676 |
| tq4_row_d10_L0x | 11.1742 | +0.003 | -0.09..+0.15 | 0.9877 | 0.00086 | 0.3151 |
| tq4_row_e1em1_L0x | 11.1642 | -0.086 | -0.24..+0.10 | 0.9875 | 0.00089 | 0.1404 |
| tq4_row_e1em2_L0x | 11.1694 | -0.040 | -0.17..+0.14 | 0.9881 | 0.00084 | 0.2886 |
| tq4_part_d8p1_L0x | 11.1742 | +0.003 | -0.09..+0.09 | 0.9873 | 0.00084 | 0.4595 |
| int4_row_d8p1_L0x | 11.1751 | +0.010 | -0.28..+0.85 | 0.9723 | 0.00608 | 0.1639 |
| g1_inf (window 0 only, 8191 tokens) | 13.3188 | +0.072 | - | 0.9894 | 0.00083 | 1.000 |

(`g1_inf` rows: window 0 only, 8,191 tokens, so their PPL differs from the 10-window pooled PPL; the refined fraction there is 1.0 by construction. Refined fraction of `_L0x` rows is over layers 1..N-1 only: layer 0 is fully dense there.)
Paired window noise (mean +- s.e. of per-window dPPL%, 10 windows): 1.7B tq4_row_d8p1 +0.284 +- 0.156; tq4_row_e1em2 -0.106 +- 0.033; tq4_part_d8p1 +0.351 +- 0.156; int4_row +185 +- 21 (mean of per-window values; pooled +179). 4B tq4_row_d8p1 -0.072 +- 0.047; tq4_row_e1em2 -0.032 +- 0.028; tq4_part_d8p1 -0.011 +- 0.034; int4_row +12.9 +- 0.8.
Per-window and per-layer numbers: `artifacts/apa_spq7/report/summary.json`; raw per (model, window, point): `artifacts/apa_spq7/eval/<model>/w<i>_<op>.json`.

## Predictions vs registered thresholds
| | registered | 1.7B | 4B | verdict |
|---|---|---|---|---|
| P1 | TQ4 row eps=1e-2 dPPL <= +1% both | -0.106% | -0.032% | HELD |
| P2 | TQ4 row d8.1 dPPL <= +3% both | +0.283% | -0.072% | HELD |
| P3 | part d8.1 dPPL >= row d8.1 AND part refined >= 3x row | +0.350 vs +0.283 (ratio 3.01x) | -0.011 vs -0.072 (ratio 2.62x) | MISSED as registered (fraction clause fails on 4B). PPL clause alone: held, but the gaps (0.07 pp, 0.06 pp) are below one s.e. of the paired windows, so "no better" is not distinguishable from "equal". Partition refines 2.6-3.0x more keys at no measurable benefit. The lead's "~5x" came from SPQ3's prefill grid; in the model the layer-0/early layers (where row already refines 40-51%) compress the ratio. |
| P4 | INT4 d8.1 worse than TQ4 d8.1 both | +179% vs +0.28% | +12.9% vs -0.07% | HELD (by orders of magnitude) |
| P5 | L0-exact cuts the worst base point's dPPL by >= 20% rel., every window | INT4: +179% -> +5.0% (97%) | INT4: +12.9% -> +0.01% (99.9%) | HELD. Worst point is INT4 on both. For the recommended TQ4 point: 1.7B +0.28% -> -0.14% (dPPL improves in 8 of 10 paired windows, worsens by <=0.24 pp in the other 2; KL 0.0136 -> 0.0011); 4B -0.07% -> -0.05% (no detectable effect). |

## Recommended operating point (registered rule: DeltaPPL<=+1% on both models, mean top-1>=0.97, then lowest mean refined fraction)
**TQ4 bulk, row carry, delta 8.1, layer 0 exact** (`tq4_row_d8p1_L0x`): dPPL -0.142% (1.7B) / -0.045% (4B), KL 1.11e-3 / 8.5e-4 (at the floor), top-1 0.986 / 0.988, refined fraction 15.2% / 16.8% of layers 1..N-1 (mean 16.0%; layer 0 dense).
Runner-up that needs no special-casing: **TQ4 row, SP2 per-head eps=1e-1** (`tq4_row_e1em1`): dPPL -0.11% / -0.08%, KL at the floor, mean refined fraction 24.8% (33% on 1.7B, 16% on 4B). The SP2 per-head e_q rule self-protects the early layers: it refines 99.5-99.9% of layer 0 (calibration e_q there is large), so it needs no explicit L0-exact rule; plain delta 8.1 refines only 40-51% of layer 0, which is where the 1.7B degradation (+0.28%, KL 13x floor) comes from.
Plain TQ4 row delta 8.1 (no L0x) does NOT meet the registered rule (1.7B top-1 0.962 < 0.97) although its dPPL is small; TQ4 partition buys nothing and refines ~48%; delta 10 refines ~31% and sits between. Do not use INT4 as the bulk tier.
Caveat: bf16 `tc` kernel output rounding (~8e-4 relFro/layer) is not in this simulation; one text corpus (wikitext-2), two models, one window size (8192).

## Deviations / honest residuals
- G1 tolerance was sized from a pre-registration probe on TRAIN text (not an evaluation window), disclosed in the registration; the order gave no numeric tolerance.
- N = 10 windows (order: >= 4 and >= 64K tokens). Windows registered by token-id SHA-256.
- The `_L0x` variant was run for all 6 base points (order: one variant) so P5 could be judged on the worst point per the registered variant rule.
- e_q calibration uses ONE 8192-token window of wikitext-2 train (disjoint from test). Per-head max over one window is a tail statistic; no test of how often eval pairs exceed it.
- Memory: Qwen3-4B needs APA_RB=32 (row block) on the 12 GB card (peak 10.2 GB); RB is a memory knob only (results are block-size independent).
- The very first eval invocation was auto-backgrounded by the Claude tool (120 s default) but ran under the same flock/timeout; no result affected.
- RED items: none of the gates failed. P3 missed as registered (see above).
- Unused file: `scripts/apa_spq7_spq3lib_ref.py` is a verbatim copy of SPQ3's lib, not imported.

## Prior art
- Refine rule: BLASST (Yuan, Shinn et al., arXiv 2512.12087, 2025) running-max softmax-threshold criterion; APA-SP1/SP2 (this repo) adapt it to per-key precision refinement; delta = ln(1/eps)+2 e_q from `docs/APA_SP2_DELTA_DERIVATION.md`. Taken: the rule. Ours: the model-level A/B harness.
- Bulk tier: TurboQuant MSE stage (Zandieh et al., arXiv 2504.19874; via `quant.py`), INT4 per-key absmax (textbook). Carries: BLASST online-softmax carry vs flash-decoding split-K (Dao et al., 2023).
- Metrics: perplexity on non-overlapping windows (standard HF recipe), next-token top-1 agreement and KL(exact||approx): standard, no prior art claimed. Per-layer/per-head e_q calibration of delta: SP2 (this repo). Nothing algorithmically new in this order.
