# APA-SPQ3 report: APA-SP on real attention (Qwen3-4B, Qwen3-1.7B; real post-RoPE Q/K/V; S=8192)

Seat: Claude Sonnet 5.5 high. Order `orders/APA_SPQ3_REAL_QK_OPERATING_POINT.md`; ledger `docs/APA_SPQ3_LEDGER.md`; registration SHA-256
`c361d5b964e27d5fd502486f30354a390ed3d90f2599001547315851c041f7d3`. Evidence class: measurement on captured activations (one wikitext-2-raw sequence of 8192 tokens, 6 layers per model,
all query heads, causal). Scope: it does NOT cover model quality (no perplexity or generations), kernel speed, other text/lengths/models, MLA/MQA. Full per-layer tables: `artifacts/apa_spq3/report/tables.md`.

## Verdict first
- G1 capture integrity: **PASS 12/12** (relFro vs the model's own sdpa output 1.6e-3..1.7e-3 = bf16 rounding). G2: INT4 sym-7 **PASS** with a corrected probe (registered probe was defective; deviation disclosed); TQ4 strict
  bitwise reconstruction **RED** (engine fp32 reduction order, max 3.8e-6 abs, indices inferred identical).
- The synthetic operating point (delta 0.5-4) does NOT transfer. On real attention the SP error reaches the bf16-rounding floor only at **delta about 8-10** (TQ4), i.e. a refine fraction of ~10-45% depending on carry.
- **Row carry beats partition carry decisively at equal refined fraction** (TQ4 relFro at fraction 0.10: median 2.4e-3 row vs 1.8e-2 partition). The partition carry refines 4-30x more keys at the same delta.
- Early layers (layer 0 of both models, Qwen3-1.7B layer 5) are not served by any registered delta (relFro 4e-2..0.5 at delta 8.1; e_q 14-40).
- The self-calibrated maximum e_q (SP2's table rule) is vacuous on real attention: pooled e_q 22-39 gives delta >= 50 and refines 100% of keys.
- Predictions: P1 NOT CONFIRMED (quantitatively: far below synthetic 3% only for TQ4 row carry), P2 CONFIRMED (but small for row carry), P3 NOT CONFIRMED (the row carry cuts decode refine ~30x, not 2x),
  P4 CONFIRMED, P5 CONFIRMED, P6 CONFIRMED.

## Registered predictions, filed against the registered thresholds
```
## Verdicts against the registered predictions

P1 (median over 12 cells of prefill relFro at refined fraction 0.10 <= 1.0% for each fmt x carry): **NOT CONFIRMED**. TQ4/row: med@0.10=0.00244, med@0.15=0.00144, cells outside grid fraction range=0; TQ4/part: med@0.10=0.0182, med@0.15=0.0133, cells outside grid fraction range=0; INT4/row: med@0.10=0.0112, med@0.15=0.00734, cells outside grid fraction range=0; INT4/part: med@0.10=0.0488, med@0.15=0.0332, cells outside grid fraction range=0
P2 (median 64x64 zero-refined-tile share at refine~0.10 > 0.01): **CONFIRMED**. TQ4/row: 0.07562; TQ4/part: 0.2976; INT4/row: 0.07445; INT4/part: 0.2467
P3 (median row/partition decode refined-fraction ratio over cells x fmts x delta in {2,2.75,4} in [0.35,0.65]): median=0.03003, min=0.003075, max=0.8513 -> **NOT CONFIRMED**
P4 (pooled e_q TQ4 < INT4 per model): Qwen3-4B: TQ4 22.08 vs INT4 39.9; Qwen3-1.7B: TQ4 31.13 vs INT4 32.37 -> **CONFIRMED**
P5 (median prefill refined fraction of SP2 eps=1e-2 pooled e_q > 0.5): TQ4: 1; INT4: 1 -> **CONFIRMED**
P6 (median per-cell Spearman TQ4 >= INT4): TQ4: 0.9895; INT4: 0.9407 -> **CONFIRMED**
```
P1 per cell (prefill relFro at refined fraction 0.10, interpolated log-log on the grid):

| fmt | carry | -4B L0 | -4B L7 | -4B L14 | -4B L21 | -4B L28 | -4B L35 | -1.7B L0 | -1.7B L5 | -1.7B L11 | -1.7B L16 | -1.7B L22 | -1.7B L27 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TQ4 | row | 1.8e-01 | 1.4e-03 | 1.9e-03 | 1.5e-03 | 8.4e-04 | 2.7e-03 | 3.2e-01 | 1.0e-02 | 2.4e-03 | 1.7e-03 | 3.2e-03 | 2.5e-03 |
| TQ4 | part | 1.9e-01 | 1.4e-02 | 1.5e-02 | 9.5e-03 | 1.3e-02 | 2.5e-02 | 3.3e-01 | 1.4e-01 | 9.9e-03 | 1.1e-02 | 2.1e-02 | 2.8e-02 |
| INT4 | row | 9.5e-01 | 1.7e-03 | 9.1e-03 | 5.4e-03 | 3.0e-03 | 3.2e-02 | 8.3e-01 | 5.0e-01 | 2.5e-03 | 2.5e-03 | 1.3e-02 | 6.4e-02 |
| INT4 | part | 9.6e-01 | 2.1e-02 | 5.6e-02 | 3.2e-02 | 3.0e-02 | 1.3e-01 | 8.3e-01 | 7.9e-01 | 1.4e-02 | 1.9e-02 | 4.2e-02 | 3.0e-01 |

Reading P1: P1 as registered ("<= 1.0% for each fmt x carry") fails for TQ4/partition (1.8%), INT4/row (1.1%) and INT4/partition (4.9%); only TQ4/row (0.24%) passes. Relative to the synthetic ~3%, TQ4/row is
12x lower, TQ4/partition 1.7x lower, INT4/partition worse than the synthetic number in the median. Early-layer cells dominate the high end (0.18-0.96).

## Recommended operating point for real attention (reasoning from the tables; not a model-quality claim)
Median over the 9 cells that are not early layers ("REST"; full table `artifacts/apa_spq3/report/summary_medians.md`, copied below):
- **TQ4 bulk, row carry, delta about 8.1-9** (epsilon-equivalent: delta = ln(1/eps)+2 e_q with the PER-HEAD e_q median 2.0-3.4, eps 1e-1 to 1e-2). At delta=8.1: refine 9.6%, relFro 2.0e-3 (1.9x the bf16 P-rounding baseline 1.0e-3),
  decode refine 6.2%, relFro 2.6e-3; top-1 attention target agreement 1.0000 at every delta. The deviation drops to the bf16 floor (1.0x) at delta 8.3-10.1 in the non-early layers (TQ4 row, crossing table) and to 0.5x at 9.0-11.4; refine rises from ~10% at delta 8.1 to ~46% at delta 12 (median curve), so the 1x point sits between.
- With per-head e_q the SP2 rule at **eps = 1e-2** gives refine 21% and relFro 9.7e-4 (0.88x baseline); eps = 1e-1 gives refine 6.7% at 3.2x baseline. **eps = 1e-2 (per-head e_q) is the recommended certified setting;** eps 1e-1 is the cheap, noisier alternative.
- Do NOT calibrate e_q as the model-wide maximum: pooled e_q is set by rare sink/outlier pairs (head max 3-4 vs p99.9 about 1-2 in mid layers; layer 0 up to 22-31), so delta >= 50 and nothing is skipped.
- Keep early layers dense (or give them their own delta above 12): layer 0 of both models and 1.7B layer 5 do not reach even 2x baseline by delta 12.
- INT4 sym-7 as bulk is clearly worse: at delta 8.1 row carry its REST relFro is 5.1e-3 (4.7x baseline) vs TQ4 2.0e-3; Spearman 0.94 vs 0.99 (median over cells). Use TQ4.
- Row vs partition carry reading: row carry (global running max over the keys seen so far) is the effective one; partition carry is a strict superset in refined keys (SP2 containment) but buys little accuracy per refined key.

## What the tile-occupancy result means for a sparse-refine prefill (reasoning from the table below)
At refine fraction 0.10 (delta calibrated per layer, TQ4): share of tiles with ZERO refined keys, median over cells: 16x16 row 0.33 / partition 0.45; 64x64 **row 0.076 / partition 0.30**; 128x64 row 0.046 / partition 0.27;
GQA-group-shared 64x64 row 0.022 / partition 0.23. P2 ("nonzero at 64x64") holds, unlike the synthetic 0; but with the carry that is accurate (row) a 64x64 tile-skip applies to only 3-13% of tiles in the mid/late layers (up to 27% in Qwen3-1.7B layer 5; 1-3%
in layer 0) and almost none when the mask is shared across a GQA group. Partition carry frees 25-40% of 64x64 tiles in mid/late layers, but it needs a different delta for the same accuracy and is ~7x worse at the same refined fraction.
Consequence: tile-granular skipping of the refine pass is worth little; a useful sparse refine must work at key granularity (compaction/gather of refined keys) or at 16x16 granularity (a third of tiles free at row carry). Single-sequence caveat applies.

## Bulk ranking (paper Appendix B.1 open item)
Median per-row Spearman (bulk vs exact over visible keys), median over cells: TQ4 0.989, INT4 0.941 (layer 0: TQ4 0.78-0.88, INT4 0.48-0.59; mid layers TQ4 0.99, INT4 0.92-0.98). Top-r% overlap: see table
(TQ4 mid layers 0.90-0.93, INT4 0.72-0.87). Comparator: the two-pass z-score rule at realised refine 0.11 gives relFro 4.9e-2..1.9e-1 in mid/late layers on the sampled rows, 2-80x worse than the SP row rule at a similar refined fraction (about 25x typical)
(sample rows differ: 256 rows vs all rows; indicative only).

## Median table (REST = 9 non-early cells)
Median over cells of prefill refined fraction / relFro / relFro÷bf16-P-baseline.  ALL = 12 cells; REST = 9 cells excluding the three early layers (4B L0, 1.7B L0, 1.7B L5) whose e_q is 3-10x larger.

| fmt | carry | config | ALL frac | ALL relFro | ALL dev/base | REST frac | REST relFro | REST dev/base | decode frac (REST) | decode relFro (REST) |
|---|---|---|---|---|---|---|---|---|---|---|
| TQ4 | row | grid_6 | 0.027 | 8.85e-03 | 7.47 | 0.021 | 6.50e-03 | 5.87 | 0.013 | 7.87e-03 |
| TQ4 | row | grid_8.1 | 0.102 | 3.01e-03 | 2.52 | 0.096 | 2.03e-03 | 1.88 | 0.062 | 2.59e-03 |
| TQ4 | row | grid_12 | 0.469 | 2.14e-04 | 0.19 | 0.457 | 1.03e-04 | 0.10 | 0.414 | 1.54e-04 |
| TQ4 | row | sp2head_eps0.1 | 0.085 | 3.03e-03 | 2.86 | 0.067 | 3.57e-03 | 3.18 | 0.057 | 3.98e-03 |
| TQ4 | row | sp2head_eps0.01 | 0.252 | 6.95e-04 | 0.68 | 0.214 | 9.73e-04 | 0.88 | 0.187 | 1.08e-03 |
| TQ4 | row | sp2head_eps0.001 | 0.523 | 1.05e-04 | 0.11 | 0.460 | 1.75e-04 | 0.15 | 0.407 | 2.08e-04 |
| TQ4 | row | sp2pool_eps0.1 | 1.000 | 0.00e+00 | 0.00 | 1.000 | 0.00e+00 | 0.00 | 1.000 | 0.00e+00 |
| TQ4 | part | grid_6 | 0.304 | 5.16e-03 | 4.50 | 0.320 | 4.10e-03 | 3.97 | 0.370 | 2.22e-03 |
| TQ4 | part | grid_8.1 | 0.470 | 1.61e-03 | 1.51 | 0.469 | 1.23e-03 | 1.09 | 0.572 | 7.57e-04 |
| TQ4 | part | grid_12 | 0.716 | 1.02e-04 | 0.09 | 0.713 | 5.90e-05 | 0.05 | 0.764 | 5.23e-05 |
| TQ4 | part | sp2head_eps0.1 | 0.443 | 1.80e-03 | 1.74 | 0.415 | 2.16e-03 | 1.90 | 0.484 | 1.07e-03 |
| TQ4 | part | sp2head_eps0.01 | 0.606 | 3.73e-04 | 0.41 | 0.594 | 5.95e-04 | 0.47 | 0.671 | 3.14e-04 |
| TQ4 | part | sp2head_eps0.001 | 0.764 | 5.08e-05 | 0.06 | 0.732 | 9.83e-05 | 0.08 | 0.789 | 6.55e-05 |
| TQ4 | part | sp2pool_eps0.1 | 1.000 | 0.00e+00 | 0.00 | 1.000 | 0.00e+00 | 0.00 | 1.000 | 0.00e+00 |
| INT4 | row | grid_6 | 0.036 | 2.39e-02 | 25.63 | 0.036 | 1.68e-02 | 15.53 | 0.023 | 1.50e-02 |
| INT4 | row | grid_8.1 | 0.123 | 8.39e-03 | 9.10 | 0.112 | 5.13e-03 | 4.74 | 0.094 | 4.52e-03 |
| INT4 | row | grid_12 | 0.486 | 7.95e-04 | 0.86 | 0.466 | 4.68e-04 | 0.43 | 0.404 | 5.67e-04 |
| INT4 | row | sp2head_eps0.1 | 0.708 | 1.56e-04 | 0.16 | 0.494 | 3.82e-04 | 0.39 | 0.461 | 3.32e-04 |
| INT4 | row | sp2head_eps0.01 | 0.855 | 2.07e-05 | 0.02 | 0.730 | 6.12e-05 | 0.06 | 0.679 | 5.90e-05 |
| INT4 | row | sp2head_eps0.001 | 0.941 | 1.15e-06 | 0.00 | 0.883 | 6.47e-06 | 0.01 | 0.844 | 7.42e-06 |
| INT4 | row | sp2pool_eps0.1 | 1.000 | 0.00e+00 | 0.00 | 1.000 | 0.00e+00 | 0.00 | 1.000 | 0.00e+00 |
| INT4 | part | grid_6 | 0.293 | 1.33e-02 | 13.87 | 0.283 | 1.08e-02 | 10.93 | 0.313 | 6.48e-03 |
| INT4 | part | grid_8.1 | 0.464 | 3.97e-03 | 4.17 | 0.461 | 3.05e-03 | 2.82 | 0.509 | 1.56e-03 |
| INT4 | part | grid_12 | 0.722 | 2.77e-04 | 0.26 | 0.716 | 1.94e-04 | 0.23 | 0.760 | 1.31e-04 |
| INT4 | part | sp2head_eps0.1 | 0.833 | 9.08e-05 | 0.09 | 0.702 | 2.29e-04 | 0.25 | 0.748 | 1.46e-04 |
| INT4 | part | sp2head_eps0.01 | 0.923 | 1.11e-05 | 0.01 | 0.844 | 3.30e-05 | 0.04 | 0.855 | 2.50e-05 |
| INT4 | part | sp2head_eps0.001 | 0.970 | 6.23e-07 | 0.00 | 0.943 | 3.05e-06 | 0.00 | 0.936 | 2.99e-06 |
| INT4 | part | sp2pool_eps0.1 | 1.000 | 0.00e+00 | 0.00 | 1.000 | 0.00e+00 | 0.00 | 1.000 | 0.00e+00 |


## e_q
## e_q (max |bulk - exact| scaled-logit error over all causal pairs)

| model | layer | fmt | e_q layer-max | head min | head median | head max | mean err (pooled heads' mean) | p99.9 (sample, median head) |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | 0 | TQ4 | 22.08 | 7.198 | 14.77 | 22.08 | 1.472 | 6.944 |
| Qwen3-4B | 0 | INT4 | 39.9 | 12.82 | 27.39 | 39.9 | 3.484 | 15.89 |
| Qwen3-4B | 7 | TQ4 | 3.749 | 1.497 | 1.941 | 3.749 | 0.2152 | 0.8501 |
| Qwen3-4B | 7 | INT4 | 6.303 | 2.053 | 2.883 | 6.303 | 0.3861 | 1.517 |
| Qwen3-4B | 14 | TQ4 | 3.137 | 1.837 | 2.476 | 3.137 | 0.2801 | 1.258 |
| Qwen3-4B | 14 | INT4 | 8.632 | 4.642 | 7.148 | 8.632 | 0.8364 | 3.872 |
| Qwen3-4B | 21 | TQ4 | 3.516 | 1.596 | 2.371 | 3.516 | 0.2514 | 1.078 |
| Qwen3-4B | 21 | INT4 | 7.813 | 3.253 | 4.694 | 7.813 | 0.5873 | 2.194 |
| Qwen3-4B | 28 | TQ4 | 3.802 | 1.892 | 2.362 | 3.802 | 0.2841 | 1.192 |
| Qwen3-4B | 28 | INT4 | 5.98 | 2.695 | 4.704 | 5.98 | 0.5747 | 2.338 |
| Qwen3-4B | 35 | TQ4 | 4.003 | 2.084 | 3.365 | 4.003 | 0.4142 | 1.892 |
| Qwen3-4B | 35 | INT4 | 13.72 | 6.097 | 10.51 | 13.72 | 1.338 | 5.649 |
| Qwen3-1.7B | 0 | TQ4 | 31.13 | 7.066 | 21.35 | 31.13 | 1.789 | 8.906 |
| Qwen3-1.7B | 0 | INT4 | 31.01 | 11.75 | 21.16 | 31.01 | 2.997 | 12.12 |
| Qwen3-1.7B | 5 | TQ4 | 14.35 | 4.396 | 10.29 | 14.35 | 1.23 | 5.471 |
| Qwen3-1.7B | 5 | INT4 | 32.37 | 10.15 | 23.33 | 32.37 | 3.561 | 14.17 |
| Qwen3-1.7B | 11 | TQ4 | 4.779 | 2.183 | 2.804 | 4.779 | 0.2693 | 1.201 |
| Qwen3-1.7B | 11 | INT4 | 7.72 | 3.228 | 4.611 | 7.72 | 0.479 | 2.26 |
| Qwen3-1.7B | 16 | TQ4 | 3.855 | 1.838 | 2.456 | 3.855 | 0.2382 | 1.083 |
| Qwen3-1.7B | 16 | INT4 | 6.951 | 3.54 | 5.059 | 6.951 | 0.5066 | 2.222 |
| Qwen3-1.7B | 22 | TQ4 | 3.501 | 1.524 | 2.438 | 3.501 | 0.2712 | 1.127 |
| Qwen3-1.7B | 22 | INT4 | 6.937 | 2.741 | 3.772 | 6.937 | 0.5278 | 2.122 |
| Qwen3-1.7B | 27 | TQ4 | 4.012 | 1.998 | 3.096 | 4.012 | 0.3965 | 1.791 |
| Qwen3-1.7B | 27 | INT4 | 14.22 | 6.167 | 10.57 | 14.22 | 1.317 | 5.729 |

Pooled (model, format) e_q (the SP2-style table value): Qwen3-4B TQ4: 22.08; Qwen3-4B INT4: 39.9; Qwen3-1.7B TQ4: 31.13; Qwen3-1.7B INT4: 32.37




## delta at which relFro reaches 2x / 1x / 0.5x of the bf16 P-rounding baseline (prefill; decode and the logit-rounding baseline are in tables.md)
| model | layer | fmt | carry | rows | baseline | 2x | 1x | 0.5x |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | 0 | TQ4 | row | prefill | P (0.00111) | >12 | >12 | >12 |
| Qwen3-4B | 0 | TQ4 | part | prefill | P (0.00111) | >12 | >12 | >12 |
| Qwen3-4B | 0 | INT4 | row | prefill | P (0.00111) | >12 | >12 | >12 |
| Qwen3-4B | 0 | INT4 | part | prefill | P (0.00111) | >12 | >12 | >12 |
| Qwen3-4B | 7 | TQ4 | row | prefill | P (0.00138) | 7.97 | 9.09 | 10.3 |
| Qwen3-4B | 7 | TQ4 | part | prefill | P (0.00138) | 6.59 | 8.13 | 9.13 |
| Qwen3-4B | 7 | INT4 | row | prefill | P (0.00138) | 8.01 | 9.07 | 10.2 |
| Qwen3-4B | 7 | INT4 | part | prefill | P (0.00138) | 7.12 | 8.37 | 9.38 |
| Qwen3-4B | 14 | TQ4 | row | prefill | P (0.00105) | 7.52 | 8.55 | 9.39 |
| Qwen3-4B | 14 | TQ4 | part | prefill | P (0.00105) | 6.74 | 8.01 | 8.81 |
| Qwen3-4B | 14 | INT4 | row | prefill | P (0.00105) | 9.69 | 10.9 | >12 |
| Qwen3-4B | 14 | INT4 | part | prefill | P (0.00105) | 8.99 | 9.96 | 11 |
| Qwen3-4B | 21 | TQ4 | row | prefill | P (0.000984) | 7.24 | 8.3 | 9.03 |
| Qwen3-4B | 21 | TQ4 | part | prefill | P (0.000984) | 6.5 | 7.54 | 8.44 |
| Qwen3-4B | 21 | INT4 | row | prefill | P (0.000984) | 8.97 | 9.99 | 11.1 |
| Qwen3-4B | 21 | INT4 | part | prefill | P (0.000984) | 8.45 | 9.33 | 10.3 |
| Qwen3-4B | 28 | TQ4 | row | prefill | P (0.00108) | 7.98 | 8.8 | 9.64 |
| Qwen3-4B | 28 | TQ4 | part | prefill | P (0.00108) | 7.08 | 8.24 | 9.01 |
| Qwen3-4B | 28 | INT4 | row | prefill | P (0.00108) | 9.33 | 10.5 | 11.7 |
| Qwen3-4B | 28 | INT4 | part | prefill | P (0.00108) | 8.55 | 9.55 | 10.7 |
| Qwen3-4B | 35 | TQ4 | row | prefill | P (0.00126) | 8.68 | 9.67 | 10.8 |
| Qwen3-4B | 35 | TQ4 | part | prefill | P (0.00126) | 7.96 | 8.93 | 9.93 |
| Qwen3-4B | 35 | INT4 | row | prefill | P (0.00126) | >12 | >12 | >12 |
| Qwen3-4B | 35 | INT4 | part | prefill | P (0.00126) | 11.3 | >12 | >12 |
| Qwen3-1.7B | 0 | TQ4 | row | prefill | P (0.00114) | >12 | >12 | >12 |
| Qwen3-1.7B | 0 | TQ4 | part | prefill | P (0.00114) | >12 | >12 | >12 |
| Qwen3-1.7B | 0 | INT4 | row | prefill | P (0.00114) | >12 | >12 | >12 |
| Qwen3-1.7B | 0 | INT4 | part | prefill | P (0.00114) | >12 | >12 | >12 |
| Qwen3-1.7B | 5 | TQ4 | row | prefill | P (0.00137) | 11.5 | >12 | >12 |
| Qwen3-1.7B | 5 | TQ4 | part | prefill | P (0.00137) | 11.1 | >12 | >12 |
| Qwen3-1.7B | 5 | INT4 | row | prefill | P (0.00137) | >12 | >12 | >12 |
| Qwen3-1.7B | 5 | INT4 | part | prefill | P (0.00137) | >12 | >12 | >12 |
| Qwen3-1.7B | 11 | TQ4 | row | prefill | P (0.000995) | 7.76 | 8.69 | 9.56 |
| Qwen3-1.7B | 11 | TQ4 | part | prefill | P (0.000995) | 7.06 | 8.19 | 8.96 |
| Qwen3-1.7B | 11 | INT4 | row | prefill | P (0.000995) | 7.64 | 8.61 | 9.55 |
| Qwen3-1.7B | 11 | INT4 | part | prefill | P (0.000995) | 7.09 | 8.17 | 8.98 |
| Qwen3-1.7B | 16 | TQ4 | row | prefill | P (0.000866) | 7.89 | 8.76 | 9.6 |
| Qwen3-1.7B | 16 | TQ4 | part | prefill | P (0.000866) | 6.69 | 7.86 | 8.71 |
| Qwen3-1.7B | 16 | INT4 | row | prefill | P (0.000866) | 8.46 | 9.56 | 10.8 |
| Qwen3-1.7B | 16 | INT4 | part | prefill | P (0.000866) | 7.57 | 8.64 | 9.66 |
| Qwen3-1.7B | 22 | TQ4 | row | prefill | P (0.000862) | 8.81 | 9.59 | 10.4 |
| Qwen3-1.7B | 22 | TQ4 | part | prefill | P (0.000862) | 8.15 | 8.86 | 9.64 |
| Qwen3-1.7B | 22 | INT4 | row | prefill | P (0.000862) | 11 | >12 | >12 |
| Qwen3-1.7B | 22 | INT4 | part | prefill | P (0.000862) | 8.97 | 9.84 | 10.8 |
| Qwen3-1.7B | 27 | TQ4 | row | prefill | P (0.00119) | 9.02 | 10.1 | 11.4 |
| Qwen3-1.7B | 27 | TQ4 | part | prefill | P (0.00119) | 8.11 | 9.05 | 10.1 |
| Qwen3-1.7B | 27 | INT4 | row | prefill | P (0.00119) | >12 | >12 | >12 |
| Qwen3-1.7B | 27 | INT4 | part | prefill | P (0.00119) | >12 | >12 | >12 |

## Tile occupancy at refined fraction 0.10 (all cells; 0.05 and 0.15 in tables.md)
| model | layer | fmt | carry | target | delta* | achieved | 16x16 | 64x64 | 128x64 | 64x64 GQA-OR |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | 0 | TQ4 | row | 0.1 | 3.288 | 0.0997 | 0.1423 | 0.0257 | 0.0154 | 0.0040 |
| Qwen3-4B | 0 | TQ4 | part | 0.1 | 2.749 | 0.0994 | 0.1113 | 0.0137 | 0.0079 | 0.0024 |
| Qwen3-4B | 0 | INT4 | row | 0.1 | 1.500 | 0.0998 | 0.4118 | 0.2143 | 0.1914 | 0.1000 |
| Qwen3-4B | 0 | INT4 | part | 0.1 | 1.056 | 0.0999 | 0.3953 | 0.1866 | 0.1639 | 0.0869 |
| Qwen3-4B | 7 | TQ4 | row | 0.1 | 8.871 | 0.0971 | 0.2641 | 0.0263 | 0.0085 | 0.0015 |
| Qwen3-4B | 7 | TQ4 | part | 0.1 | 3.253 | 0.0999 | 0.4433 | 0.3392 | 0.3073 | 0.2332 |
| Qwen3-4B | 7 | INT4 | row | 0.1 | 8.652 | 0.0999 | 0.2396 | 0.0184 | 0.0047 | 0.0009 |
| Qwen3-4B | 7 | INT4 | part | 0.1 | 3.402 | 0.0992 | 0.4328 | 0.3080 | 0.2690 | 0.1844 |
| Qwen3-4B | 14 | TQ4 | row | 0.1 | 7.667 | 0.0999 | 0.3720 | 0.1284 | 0.0858 | 0.0343 |
| Qwen3-4B | 14 | TQ4 | part | 0.1 | 3.152 | 0.0994 | 0.4285 | 0.2869 | 0.2521 | 0.1756 |
| Qwen3-4B | 14 | INT4 | row | 0.1 | 7.282 | 0.0975 | 0.3122 | 0.0798 | 0.0470 | 0.0179 |
| Qwen3-4B | 14 | INT4 | part | 0.1 | 3.629 | 0.0997 | 0.3868 | 0.2164 | 0.1771 | 0.1105 |
| Qwen3-4B | 21 | TQ4 | row | 0.1 | 7.674 | 0.0998 | 0.3275 | 0.0702 | 0.0437 | 0.0106 |
| Qwen3-4B | 21 | TQ4 | part | 0.1 | 3.902 | 0.0993 | 0.4953 | 0.2430 | 0.1875 | 0.1151 |
| Qwen3-4B | 21 | INT4 | row | 0.1 | 7.427 | 0.0980 | 0.3175 | 0.0650 | 0.0401 | 0.0093 |
| Qwen3-4B | 21 | INT4 | part | 0.1 | 4.147 | 0.1000 | 0.4505 | 0.1884 | 0.1387 | 0.0784 |
| Qwen3-4B | 28 | TQ4 | row | 0.1 | 8.931 | 0.0968 | 0.3514 | 0.1073 | 0.0802 | 0.0556 |
| Qwen3-4B | 28 | TQ4 | part | 0.1 | 3.751 | 0.0996 | 0.5717 | 0.3965 | 0.3512 | 0.3018 |
| Qwen3-4B | 28 | INT4 | row | 0.1 | 8.738 | 0.0990 | 0.3475 | 0.1122 | 0.0851 | 0.0600 |
| Qwen3-4B | 28 | INT4 | part | 0.1 | 4.035 | 0.0995 | 0.5386 | 0.3429 | 0.2974 | 0.2476 |
| Qwen3-4B | 35 | TQ4 | row | 0.1 | 8.466 | 0.0975 | 0.4006 | 0.1305 | 0.0852 | 0.0205 |
| Qwen3-4B | 35 | TQ4 | part | 0.1 | 2.563 | 0.0998 | 0.4636 | 0.3874 | 0.3705 | 0.3099 |
| Qwen3-4B | 35 | INT4 | row | 0.1 | 8.156 | 0.0960 | 0.3456 | 0.0991 | 0.0638 | 0.0223 |
| Qwen3-4B | 35 | INT4 | part | 0.1 | 3.255 | 0.0999 | 0.4511 | 0.3441 | 0.3159 | 0.2294 |
| Qwen3-1.7B | 0 | TQ4 | row | 0.1 | 4.158 | 0.0999 | 0.1434 | 0.0103 | 0.0033 | 0.0030 |
| Qwen3-1.7B | 0 | TQ4 | part | 0.1 | 3.584 | 0.0995 | 0.1021 | 0.0041 | 0.0014 | 0.0013 |
| Qwen3-1.7B | 0 | INT4 | row | 0.1 | 1.683 | 0.0994 | 0.3100 | 0.0535 | 0.0366 | 0.0190 |
| Qwen3-1.7B | 0 | INT4 | part | 0.1 | 1.450 | 0.0995 | 0.2960 | 0.0382 | 0.0242 | 0.0088 |
| Qwen3-1.7B | 5 | TQ4 | row | 0.1 | 9.150 | 0.0983 | 0.5153 | 0.2724 | 0.2104 | 0.1857 |
| Qwen3-1.7B | 5 | TQ4 | part | 0.1 | 3.917 | 0.0993 | 0.4656 | 0.3563 | 0.3308 | 0.3187 |
| Qwen3-1.7B | 5 | INT4 | row | 0.1 | 8.793 | 0.0993 | 0.5705 | 0.3684 | 0.3187 | 0.3021 |
| Qwen3-1.7B | 5 | INT4 | part | 0.1 | 3.762 | 0.0996 | 0.4969 | 0.3864 | 0.3662 | 0.3589 |
| Qwen3-1.7B | 11 | TQ4 | row | 0.1 | 7.384 | 0.0978 | 0.2744 | 0.0492 | 0.0246 | 0.0161 |
| Qwen3-1.7B | 11 | TQ4 | part | 0.1 | 3.731 | 0.0996 | 0.3750 | 0.1676 | 0.1234 | 0.1112 |
| Qwen3-1.7B | 11 | INT4 | row | 0.1 | 7.238 | 0.0974 | 0.2296 | 0.0373 | 0.0186 | 0.0114 |
| Qwen3-1.7B | 11 | INT4 | part | 0.1 | 3.863 | 0.0992 | 0.3495 | 0.1267 | 0.0864 | 0.0782 |
| Qwen3-1.7B | 16 | TQ4 | row | 0.1 | 7.915 | 0.0975 | 0.3387 | 0.0810 | 0.0473 | 0.0376 |
| Qwen3-1.7B | 16 | TQ4 | part | 0.1 | 3.366 | 0.0994 | 0.4503 | 0.2774 | 0.2325 | 0.2247 |
| Qwen3-1.7B | 16 | INT4 | row | 0.1 | 7.824 | 0.0983 | 0.3202 | 0.0691 | 0.0395 | 0.0343 |
| Qwen3-1.7B | 16 | INT4 | part | 0.1 | 3.633 | 0.0998 | 0.4283 | 0.2427 | 0.1980 | 0.1841 |
| Qwen3-1.7B | 22 | TQ4 | row | 0.1 | 8.090 | 0.0956 | 0.2808 | 0.0542 | 0.0302 | 0.0227 |
| Qwen3-1.7B | 22 | TQ4 | part | 0.1 | 3.012 | 0.0992 | 0.4197 | 0.3082 | 0.2834 | 0.2567 |
| Qwen3-1.7B | 22 | INT4 | row | 0.1 | 7.552 | 0.0987 | 0.2828 | 0.0553 | 0.0296 | 0.0289 |
| Qwen3-1.7B | 22 | INT4 | part | 0.1 | 3.262 | 0.0998 | 0.3910 | 0.2508 | 0.2198 | 0.2118 |
| Qwen3-1.7B | 27 | TQ4 | row | 0.1 | 8.793 | 0.0980 | 0.4168 | 0.1568 | 0.1130 | 0.0733 |
| Qwen3-1.7B | 27 | TQ4 | part | 0.1 | 2.492 | 0.0996 | 0.4742 | 0.4011 | 0.3884 | 0.3843 |
| Qwen3-1.7B | 27 | INT4 | row | 0.1 | 8.922 | 0.0965 | 0.3686 | 0.1152 | 0.0777 | 0.0477 |
| Qwen3-1.7B | 27 | INT4 | part | 0.1 | 3.114 | 0.0992 | 0.4797 | 0.3809 | 0.3602 | 0.3472 |

## Bulk ranking
## Bulk ranking vs exact (256 sampled rows per head, all heads)

| model | layer | fmt | Spearman mean | Spearman median | Spearman p10 | top5% overlap | top10% overlap | top15% overlap | z-score comparator realised frac (r=.05/.10/.15) | z-score relFro (sample rows) |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | 0 | TQ4 | 0.8748 | 0.8829 | 0.8077 | 0.5785 | 0.6439 | 0.6885 | 0.074/0.119/0.158 | 6.61e-01/6.42e-01/6.24e-01 |
| Qwen3-4B | 0 | INT4 | 0.5739 | 0.5948 | 0.3377 | 0.2068 | 0.3114 | 0.3976 | 0.077/0.129/0.171 | 1.05e+00/1.04e+00/1.03e+00 |
| Qwen3-4B | 7 | TQ4 | 0.9927 | 0.9933 | 0.9890 | 0.9096 | 0.9212 | 0.9297 | 0.062/0.106/0.148 | 7.42e-03/5.93e-03/3.57e-03 |
| Qwen3-4B | 7 | INT4 | 0.9762 | 0.9790 | 0.9586 | 0.8400 | 0.8596 | 0.8748 | 0.063/0.107/0.148 | 2.02e-02/1.68e-02/7.52e-03 |
| Qwen3-4B | 14 | TQ4 | 0.9893 | 0.9904 | 0.9831 | 0.8977 | 0.9102 | 0.9197 | 0.062/0.112/0.158 | 1.16e-01/1.05e-01/9.64e-02 |
| Qwen3-4B | 14 | INT4 | 0.9220 | 0.9222 | 0.8841 | 0.7215 | 0.7552 | 0.7801 | 0.064/0.113/0.159 | 3.99e-01/3.69e-01/3.40e-01 |
| Qwen3-4B | 21 | TQ4 | 0.9928 | 0.9940 | 0.9872 | 0.9110 | 0.9231 | 0.9322 | 0.065/0.112/0.156 | 5.69e-02/4.91e-02/4.39e-02 |
| Qwen3-4B | 21 | INT4 | 0.9621 | 0.9760 | 0.9179 | 0.8104 | 0.8341 | 0.8521 | 0.066/0.112/0.156 | 1.56e-01/1.35e-01/1.21e-01 |
| Qwen3-4B | 28 | TQ4 | 0.9890 | 0.9901 | 0.9821 | 0.8967 | 0.9118 | 0.9215 | 0.062/0.109/0.154 | 8.61e-02/7.02e-02/5.71e-02 |
| Qwen3-4B | 28 | INT4 | 0.9584 | 0.9603 | 0.9339 | 0.8017 | 0.8283 | 0.8465 | 0.063/0.110/0.154 | 1.77e-01/1.51e-01/1.33e-01 |
| Qwen3-4B | 35 | TQ4 | 0.9656 | 0.9671 | 0.9491 | 0.8211 | 0.8420 | 0.8571 | 0.055/0.105/0.153 | 2.00e-01/1.95e-01/1.90e-01 |
| Qwen3-4B | 35 | INT4 | 0.7942 | 0.8005 | 0.7160 | 0.5476 | 0.6014 | 0.6390 | 0.056/0.107/0.156 | 6.37e-01/6.25e-01/6.11e-01 |
| Qwen3-1.7B | 0 | TQ4 | 0.7764 | 0.7750 | 0.6359 | 0.4996 | 0.5656 | 0.6123 | 0.069/0.113/0.153 | 5.40e-01/4.90e-01/4.48e-01 |
| Qwen3-1.7B | 0 | INT4 | 0.4662 | 0.4779 | 0.2075 | 0.2224 | 0.3067 | 0.3780 | 0.053/0.102/0.154 | 8.40e-01/8.32e-01/8.23e-01 |
| Qwen3-1.7B | 5 | TQ4 | 0.9198 | 0.9386 | 0.8355 | 0.6842 | 0.7322 | 0.7679 | 0.054/0.108/0.161 | 8.64e-01/8.52e-01/8.37e-01 |
| Qwen3-1.7B | 5 | INT4 | 0.6046 | 0.6414 | 0.3649 | 0.3395 | 0.4294 | 0.4934 | 0.050/0.107/0.164 | 1.27e+00/1.26e+00/1.26e+00 |
| Qwen3-1.7B | 11 | TQ4 | 0.9919 | 0.9927 | 0.9876 | 0.9043 | 0.9183 | 0.9270 | 0.059/0.106/0.151 | 6.86e-03/4.38e-03/3.40e-03 |
| Qwen3-1.7B | 11 | INT4 | 0.9752 | 0.9765 | 0.9639 | 0.8362 | 0.8582 | 0.8734 | 0.059/0.106/0.152 | 1.21e-02/8.16e-03/6.30e-03 |
| Qwen3-1.7B | 16 | TQ4 | 0.9930 | 0.9936 | 0.9885 | 0.9164 | 0.9270 | 0.9344 | 0.063/0.110/0.154 | 1.44e-02/9.05e-03/6.28e-03 |
| Qwen3-1.7B | 16 | INT4 | 0.9668 | 0.9729 | 0.9371 | 0.8294 | 0.8490 | 0.8631 | 0.064/0.110/0.154 | 2.58e-02/1.65e-02/1.14e-02 |
| Qwen3-1.7B | 22 | TQ4 | 0.9866 | 0.9890 | 0.9746 | 0.8849 | 0.8999 | 0.9106 | 0.063/0.109/0.154 | 7.73e-02/7.34e-02/7.07e-02 |
| Qwen3-1.7B | 22 | INT4 | 0.9533 | 0.9592 | 0.9140 | 0.7860 | 0.8116 | 0.8306 | 0.063/0.110/0.154 | 1.65e-01/1.53e-01/1.45e-01 |
| Qwen3-1.7B | 27 | TQ4 | 0.9641 | 0.9660 | 0.9479 | 0.8115 | 0.8314 | 0.8473 | 0.057/0.107/0.154 | 1.66e-01/1.53e-01/1.44e-01 |
| Qwen3-1.7B | 27 | INT4 | 0.7046 | 0.7101 | 0.6123 | 0.4626 | 0.5110 | 0.5523 | 0.056/0.106/0.154 | 5.35e-01/5.18e-01/5.10e-01 |



## Deviations, residuals, RED items
- G2-TQ4 strict bitwise reconstruction RED (see ledger entry 3). G2-INT4 registered probe was defective; corrected probe passes (disclosed deviation).
- Amendment 001: first occupancy run invalid (delta* bug), rerun after fix (ledger entry 6). eq/curves/rank unaffected.
- S=16384 not attempted. Decode = the last 64 rows of the causal prefill (same keys), not a separate KV-cache run; decode statistics are therefore from 64 query rows x heads per layer.
- Exact scores are fp64 dots of the bf16 values; bulk in fp32. Registered tolerances never changed; predictions filed as written.
- Not covered: model-level quality; the refine kernel's speed; other text; other S. Every number is one sequence.
- One short unlocked GPU touch before the lock discipline began: a ~1 s fp64-GEMM timing probe (disclosed).
