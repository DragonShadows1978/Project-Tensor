# APAMQ-DF3 — Micro-Round: Stats Partition Layout P=4 → at-least-wave

Same worktree, grants, boundaries as APAMQ_DF1/DF2 orders. Sandbox has
NO GPU. This is a MICRO-round: minimal diff, no scope beyond the two
items below.

## Measured fact (lead-run, logs/df2_bench_gpu.log)

19/19 gates green. Decisive cell (D=512 KVH=1 S=64K decode,
v1_v2_v4): pipeline 4.206 ms (from 10.26). Residual = stats 2.935 ms
at stats_P=4 (wave layout: 64 blocks). The split stage at P=32 does
comparable per-key work in 0.999 ms — the stats loop is latency-bound
and wants oversubscription, not one exact wave.

## Tasks

1. Switch the v4 stats planner to
   `apa_int4_at_least_wave_partition_layout` (picks max(fixed-2048-key,
   wave) → P=32 here). Keep the reduce stage sized accordingly.
2. Add env override `TC_APA_STATS_PART_KEYS` (integer keys/partition,
   0/unset = planner default) so the lead can sweep the knob without
   further seat rounds. Clamp sanely; document in the bench header.
3. Gates: the 19 stay green (planner change is layout-only —
   reassociation class, existing 3e-3 equivalence gates cover it).
   Bench column already reports stats_P — no bench changes needed
   beyond the env passthrough if any.

Rail unchanged: ≤1.0 ms pipeline at the decisive cell; report nothing
you cannot measure — the lead runs.

## Done

Final message verbatim: the exact diff (it should be small), build
result, CPU checks, lead commands (unchanged from DF2 + one env-sweep
example), deviations.
