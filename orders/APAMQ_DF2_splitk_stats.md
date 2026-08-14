# APAMQ-DF2 — Split-K the Stats Pass (the isolated 90% of decode time)

YOUR WRITABLE TARGET is this git worktree (Project-Tensor branch
`apamq-df`) — same grants and boundaries as APAMQ_DF1_decode_floor.md
(read it). Run first, report after. Sandbox has NO GPU; lead runs GPU
legs. RED honesty.

## Measured diagnosis (lead-run, 2026-08-14, logs/df1_bench_gpu.log)

17/17 gates green. Stage-resolved decode at D=512, KVH=1, S=64K:
pack 0.006 ms (V1 GREEN), split 0.9–5.6 ms, merge 0.006 ms — and
**stats 8.8–10.7 ms in EVERY variant (~90% of pipeline)**. Best combo
10.26 ms vs the registered 1.0 ms rail. Root cause: the stats kernel
launches ONE BLOCK PER QUERY ROW (16 blocks vs 56 SMs) and walks all
S keys serially; V2 cached its OUTPUT but nothing parallelized its
COMPUTATION. Secondary observation: V3 (P=4) interacts badly with V2
at 64K (split 3.49 ms vs 0.94 at P=32) — partition sizing needs to be
per-stage, not global.

## Task

1. **Split-K stats**: partition keys across blocks exactly like the
   split stage (sum and sumsq of |bulk·scale| are exact partitionable
   reductions); tiny second-stage reduce produces mean/var/thr per
   row. Bit-concern: fp32 partial-sum reassociation only — same class
   as the existing split-K, same tolerance (3e-3 vs monolithic).
   When V2 is on, the stats pass should ALSO emit the bulk-score
   cache (one read of packed K total for stats+split combined).
2. **Per-stage partition sizing**: stats-P and split-P chosen
   independently to fill ~one 56-SM wave each at the actual
   (rows, S); fix the V2×V3 interaction (cached-bulk split wants
   more blocks, not fewer).
3. **Gates**: FA1 11 stay green; DF1 variant-equivalence gates stay
   green; add stats-splitK-vs-monolithic equivalence (3e-3); extend
   scripts/apamq_df1_bench.py so the new path appears as a variant
   column (`v4` or similar) and BEST_TARGET reflects it.
4. **Registered rail (unchanged): pipeline ≤ 1.0 ms at D=512, KVH=1,
   S=64K.** Report the expected best combo; the lead measures. A
   miss with the stats stage at split-parity (~1 ms-class pipeline)
   is still a strong result — report honestly either way.

## Done

Final message verbatim: diff summary + line counts, build command +
result, CPU-check/pytest outputs, the per-stage partition policy you
implemented, exact lead commands, deviations. No GPU numbers.
