# APA-EB3 report: variant B decode latency, bitwise identical

**Registered speed verdict: WIN.** On real S = 8192 cells (non-early median per model), cold L2, RTX 4070 SUPER
12 GB, optimized / variant B is:
- Qwen3-1.7B: **0.827** (x-B 1.209), x-flash **1.240**;
- Qwen3-4B: **0.848** (x-B 1.179), x-flash **1.194**.

Bulk tier on every line: TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate. The output is
**bitwise identical** to today's variant B on every cell, row and shape tested (G1). That was checked in-process and
against the separately built baseline binary.

Registration SHA-256: `36e0117985bc7ab0127cd0617ad6a023f824d2ee79c472276b2ed2b58c67e2ad`.

- **G0: PASS.** Both binaries built (baseline `05413962…`, EB3 `3fa9c21d…`). One real cell end to end (1.7B L11, 64
  rows x 4 configurations): 256/256 bitwise, finite. Receipt `artifacts/apa_eb3/g0/result.json`.
- **G1: PASS.** 12 real cells x 64 causal-prefix rows x {carry row, partition} x {sinks off, on}, plus 10 synthetic
  shapes (H16/H32 x S 2048, 8000, 8192, 32768, 65536): 3,200 row-configurations.
  - Each row-configuration compares bitwise: the production output, a poisoned-arena diagnostic output, the rotated
    query, mask, bulk scores and prefix.
  - The baseline binary's own outputs match on every cell.
  - G = 2 and G = 4 both covered; S not a multiple of C on S = 8000 and on 63 of 64 real prefix rows.
  - Receipt `artifacts/apa_eb3/g1/result.json`.
- **G2: PASS.** With the feature present but not selected, the EB3 build reproduces the legacy hashes EB2b
  reproduced:
  - 72 r2 keys (= a6752ea);
  - 32 INT4-tc keys (= SPQ6 build);
  - 136/136 SPQ6 hashes;
  - 48 capture-file hashes.

  The five default variant-B kernels are SASS-identical between the two binaries. Receipt
  `artifacts/apa_eb3/g2/result.json`.
- **G3: PASS, 6/6 mutants killed by G1.** Mutant 3 (counter fires one CTA early) is a data race and was killed
  weakly on real data (9 of 1,024 real output checks: 2 cells x 256 row-configurations x {production, poisoned
  diagnostic}; 32/32 at S = 2048). Receipt
  `artifacts/apa_eb3/g3/result.json`.
- **G4: PASS** (measurement gate). 20 cells, in one process per cell, interleaved flash / variant B / EB3, cold L2,
  SPQ6/EB2b protocol. Profiles for 1.7B L11 and 4B L14, plus a baseline-binary cross-check. Receipt
  `artifacts/apa_eb3/g4/result.json`.
- **G5: PASS.**
  - Launches per decode call: 4 -> 3.
  - Device bytes per call: unchanged (2 allocations instead of 3).
  - Extra device memory: the persistent merge-ticket buffer, 4,096 B once per process.
  - Receipt `artifacts/apa_eb3/g5/result.json`.

Predictions: LC2 is a MISS. The rotation is 4.1 / 4.5 µs, not ≤ 3 µs. All other lead predictions (LC1, LC3–LC6) and
all seat predictions (S1–S8) are HITs. See the caveats on the stated mechanisms of LC5 and LC6 below.

## What changed (and what did not)

The default path is untouched. EB3 is a separate opt-in entry, `tensor_cuda.apa_eb3(…, variant=0)`, compiled only
with `-DTC_APA_EB3=1`. The default path's sources equal the pre-edit snapshot, apart from added flag-guarded blocks,
and its kernels are SASS-identical (`artifacts/apa_eb3/build/sass_identity.json`). No fp32 operation on any output
element changes its operands or its order. The registered path (`tensor_cuda/src/apa_spo/eb3.cuh`, knobs in
`eb3_tuning.cuh`) changes four things:

1. **Query rotation.** `eb3::rotate_q` stages 16 rows of R per CTA through padded shared memory and serves all G
   heads of a KV head from them. Each output keeps `spq1::rotate`'s sequential `__fmaf_rn` chain over i = 0…D−1.
   The kernel also issues an L2 prefetch of the packed codes and norms the bulk kernel will read.
   10.0 → 4.1 µs (1.7B L11) and 9.9 → 4.5 µs (4B L14). Without the prefetch it is 3.5 / 3.9 µs (ablation A2).
2. **Bulk, swap-AB tile.** `eb3::bulk_swap` computes the same `mma.sync.m16n8k16` dot products transposed: the
   centroids are A (16 keys in M) and the query hi/lo are B (G heads in N). Each output element keeps:
   - the same 16 products;
   - the same k-slot of every dimension;
   - the same three MMAs per step (hi·hi, hi·lo_q, lo_c·hi), in the same order into one fp32 accumulator;
   - the same `(acc·norm)·scale`.

   Only the element's position in the tile differs, and M is no longer 87%/75% zero padding, so there are half
   the MMAs. Standalone it runs 16.3 / 16.1 µs against 17.7 µs (A4). In the registered path it also issues the L2
   prefetch of `select_pv`'s V rows when codes + norms + V fit in half the L2. That keeps DRAM busy, and is why its
   CUPTI time is 23.5 µs there.
3. **select_pv.** It is spq4's kernel, S1–S4 verbatim, with these changes:
   - shared arrays sized by the compile-time G instead of MAXG = 8 (12.6 / 16.2 KB instead of 23.5 KB);
   - `__launch_bounds__(128, 5)`, so all 256 CTAs at S = 8192 fit in one wave;
   - an L2 prefetch of the chunk's V rows at CTA start;
   - the Flash-Decoding merge fused into the last CTA of each (batch, KV head) group. Every CTA calls
     `__threadfence`, then takes a self-resetting `atomicInc` ticket with wrap NC−1. The last CTA merges with
     `spo2::merge_kernel<bf16>`'s exact arithmetic.
4. **Launches.** The merge launch is gone. PDL was not usable (see Findings).

## Predictions (registered; full lead text quoted from the order)

| ID | Result | Registered prediction | Evidence |
|---|---|---|---|
| LC1 | HIT | G1 bitwise identity holds on every cell and shape. | G1 PASS: 3,200 row-configurations, 12 real x 64 + 10 synthetic, cross-binary pass |
| LC2 | **MISS** | the query-rotation phase falls from about 10 µs to ≤ 3 µs, or disappears into the bulk kernel. | rotate_q CUPTI 4.128 µs (1.7B L11), 4.529 µs (4B L14); 3.55 / 3.94 µs without the code prefetch (A2) |
| LC3 | HIT | real 8K non-early median optimized / B ≤ 0.90 on both models. Rotation and merge are about 14 of about 81 µs. | 1.7B 0.8270; 4B 0.8481 |
| LC4 | HIT | at 8K real, optimized / flash ≤ 0.88 on both models (x-flash ≥ 1.14). | 1.7B 0.8063 (x-flash 1.240); 4B 0.8375 (x-flash 1.194) |
| LC5 | HIT | occupancy tuning alone moves `select_pv` by less than 10%. Its latency floor is its 256-CTA chunk structure, which this order may not change. | occupancy-only select_pv (A1) vs baseline select_pv: −3.93% (1.7B L11), −2.38% (4B L14) |
| LC6 | HIT | at 2K synthetic, the relative gain is larger than at 8K, because launch overhead is a bigger share. | EB3/B at S2048: 0.778 (H16), 0.816 (H32); at S8192: 0.879, 0.896 |
| S1 | HIT | G1 bitwise identity holds on every real cell, row, carry mode and sinks setting, and on every synthetic shape (including S=8000), same-process and against the baseline binary. | as LC1 |
| S2 | HIT | The query-rotation phase (CUPTI, cold L2) is between 3.0 and 5.0 us on both profiled cells: LC2 misses its 3 us bound because the rotation kernel also issues the bulk codes/norms L2 prefetch and pays one cold DRAM round trip for R. | 4.128 / 4.529 µs |
| S3 | HIT | Real S=8192 non-early median optimized / variant B is <= 0.88 on both models (registered verdict WIN). | 0.8270 / 0.8481 |
| S4 | HIT | Real S=8192 non-early median optimized / flash is <= 0.85 on Qwen3-1.7B and <= 0.90 on Qwen3-4B. | 0.8063 / 0.8375 |
| S5 | HIT | Occupancy-only select_pv (ablation A1: G-sized shared arrays, 5 CTAs/SM) changes the select_pv CUPTI time by less than 10% versus the baseline select_pv on both profiled cells (agrees with LC5). | −3.93% / −2.38% |
| S6 | HIT | At S=2048 synthetic, optimized / variant B is lower (larger gain) than at S=8192 synthetic, for both H16 and H32. | as LC6 |
| S7 | HIT | At S=65536 synthetic, optimized / variant B is between 0.93 and 1.00 for both H16 and H32 (DRAM-bound, synthetic refine fraction near 1). | 0.963 (H16), 0.970 (H32) |
| S8 | HIT | Launches per decode call fall from 4 to 3; the only extra device memory is the persistent ticket buffer (4096 bytes per process); per-call bytes are unchanged (the rotated query moves into the arena). | 4 → 3 on all six G5 shapes; extra bytes/call 0; persistent 4,096 B |

Mechanism caveats (reasoning from the profile; the registered results stand as stated):
- **LC6:** the relation holds, but launch overhead is not the cause. Queued kernels on this GPU leave gaps of about
  0.4 µs. At 2K the fixed saving from the rotation (about 6.5 µs) is simply a larger share of a 46 µs call.
- **LC5:** the 10% bound holds for occupancy alone. The stated floor ("its 256-CTA chunk structure") is not
  supported: with C and the chunk structure unchanged, the memory schedule moved `select_pv` from 48.9 to
  37.4 µs, including the fused merge, at 1.7B L11. The kernel is DRAM-bound, not chunk-latency-bound.
- **Registration defect:** `registration.json` truncates LC3 and LC5 at the order's line wrap (see Findings). The
  text above is the full order text.

## Speed (cold L2, RTX 4070 SUPER 12 GB, B = 1, D = VD = 128, C = 256, δ 8.1 SP row carry; kernel evidence)

x-B = B/EB3 and x-flash = flash/EB3. Real cells time the last 8 query rows with all 8,192 keys visible, as in
SPQ6/EB2b. Synthetic cells use one seeded query, with the synthetic refine fraction about 1 (SP's worst case).
Every cell interleaves flash, variant B and EB3 in one process, with 7 x 50 calls per arm and the median of sample
medians. CUDA-event resolution is about 1.02 µs here.

| Cell class | EB3 µs | variant B µs | flash µs | EB3/B | x-B | EB3/flash | x-flash | refined fraction | L2 | Bulk tier |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B_L00 (early, excluded from median) | 83.952 | 98.304 | 81.920 | 0.854 | 1.171 | 1.025 | 0.976 | 0.469 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-1.7B_L05 (early, excluded from median) | 65.536 | 78.848 | 81.920 | 0.831 | 1.203 | 0.800 | 1.250 | 0.042 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-1.7B_L11 | 66.560 | 79.872 | 81.920 | 0.833 | 1.200 | 0.813 | 1.231 | 0.131 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-1.7B_L16 | 66.560 | 80.896 | 81.920 | 0.823 | 1.215 | 0.813 | 1.231 | 0.107 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-1.7B_L22 | 64.512 | 78.848 | 81.920 | 0.818 | 1.222 | 0.788 | 1.270 | 0.021 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-1.7B_L27 | 65.536 | 78.848 | 81.920 | 0.831 | 1.203 | 0.800 | 1.250 | 0.021 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-4B_L00 (early, excluded from median) | 92.160 | 104.448 | 81.920 | 0.882 | 1.133 | 1.125 | 0.889 | 0.487 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-4B_L07 | 68.448 | 80.704 | 81.920 | 0.848 | 1.179 | 0.836 | 1.197 | 0.045 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-4B_L14 | 71.680 | 83.968 | 81.920 | 0.854 | 1.171 | 0.875 | 1.143 | 0.106 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-4B_L21 | 70.656 | 82.944 | 81.920 | 0.852 | 1.174 | 0.863 | 1.159 | 0.076 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-4B_L28 | 67.552 | 79.872 | 81.920 | 0.846 | 1.182 | 0.825 | 1.213 | 0.031 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-4B_L35 | 68.608 | 81.328 | 81.920 | 0.844 | 1.185 | 0.838 | 1.194 | 0.044 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| **Qwen3-1.7B S=8192 non-early median (4 layers)** | 66.048 | 79.360 | 81.920 | **0.827** | **1.209** | **0.806** | **1.240** | per-layer | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| **Qwen3-4B S=8192 non-early median (5 layers)** | 68.608 | 81.328 | 81.920 | **0.848** | **1.179** | **0.838** | **1.194** | per-layer | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| synthetic H16/KV8 S=2048 | 35.840 | 46.080 | 31.744 | 0.778 | 1.286 | 1.129 | 0.886 | 1.000 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| synthetic H16/KV8 S=8192 | 96.256 | 109.504 | 82.096 | 0.879 | 1.138 | 1.172 | 0.853 | 1.000 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| synthetic H16/KV8 S=32768 | 359.424 | 377.856 | 297.984 | 0.951 | 1.051 | 1.206 | 0.829 | 1.000 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| synthetic H16/KV8 S=65536 | 714.752 | 742.400 | 582.656 | 0.963 | 1.039 | 1.227 | 0.815 | 1.000 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| synthetic H32/KV8 S=2048 | 36.752 | 45.056 | 31.744 | 0.816 | 1.226 | 1.158 | 0.864 | 1.000 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| synthetic H32/KV8 S=8192 | 98.176 | 109.568 | 82.192 | 0.896 | 1.116 | 1.194 | 0.837 | 1.000 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| synthetic H32/KV8 S=32768 | 374.784 | 391.632 | 299.008 | 0.957 | 1.045 | 1.253 | 0.798 | 1.000 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| synthetic H32/KV8 S=65536 | 747.520 | 771.072 | 582.656 | 0.969 | 1.032 | 1.283 | 0.779 | 1.000 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |

The 32K/64K synthetic gain is small (0.95–0.97) because those cells refine about 100% of keys. Exact K and V are
then 128 MB or more from DRAM, and the cross-kernel V prefetch is disabled by its L2 budget. Real long-context
operating points remain unmeasured, as in EB2b.

Baseline-arm cross-check (G4x): variant B / flash measured inside the baseline binary's own process is 0.9877 (1.7B
L11) and 1.0374 (4B L14). The EB3 binary's G4 process gave 0.9750 and 1.0250. They agree to within one event tick,
which together with the SASS identity shows the in-process variant-B arm is the baseline
(`artifacts/apa_eb3/g4x_table.md`).

## Profile, before and after (CUPTI per-kernel medians, cold L2, 20 repetitions)

| Stage | 1.7B L11 before (variant B) | 1.7B L11 after (EB3) | 4B L14 before | 4B L14 after |
|---|---|---|---|---|
| query rotation | 10.048 | 4.128 | 9.920 | 4.529 |
| bulk | 17.744 | 23.455 (swap-AB, carries the V prefetch) | 17.745 | 23.616 |
| exact K + V + PV (`select_pv`) | 48.864 | 37.423 (merge fused) | 50.368 | 39.985 (merge fused) |
| merge | 3.744 | — (fused) | 3.744 | — (fused) |
| sum of kernel medians | 80.399 | 65.006 | 81.776 | 68.129 |
| flash (split-K + merge) | 80.257 | — | 80.403 | — |

Bulk tier for every non-flash row: TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate.
The flash rows are bf16 dense.

Descriptive ablations, same binary and cell, sum of kernel medians in µs (1.7B L11 / 4B L14;
`artifacts/apa_eb3/profile_table.md`):

| Ablation | 1.7B L11 | 4B L14 |
|---|---|---|
| baseline | 80.40 | 81.78 |
| A1 occupancy-only select_pv | 78.42 | 80.53 |
| A2 rotation only | 74.08 | 75.81 |
| A3 rotation + tuned select_pv (V prefetch), separate merge | 70.39 | 72.64 |
| A4 + swap-AB bulk + fused merge, no cross-kernel prefetch | 67.46 | 70.64 |
| A7 rotation prefetches codes + V instead | 66.16 | 69.47 |
| A8 registered without the fused merge | 66.27 | 68.45 |
| **registered** | **65.01** | **68.13** |
| A5 registered + refined-K prefetch | 65.12 | 68.08 |
| A6 baseline-footprint clones | 80.07 | 81.60 |

The ablation sums are descriptive; only the registered path is gated. Warm-L2 diagnostic: the registered path is
36–39 µs warm and 65–71 µs cold. The call stays DRAM-bound: about 23.5 MB at S = 8192, where a 256 MiB streaming
sum reaches about 423 GB/s, giving a floor near 56 µs.

## Achieved occupancy (CTA timelines; hardware counters are admin-only on this machine)

| Cell | Configuration | Kernel | Achieved occupancy (warps/48) | Mean resident CTAs/SM | Peak resident CTAs/SM | CTAs started after the first CTA ended |
|---|---|---|---|---|---|---|
| 1.7B L11 | before (baseline footprint) | bulk | 0.195 | 2.35 | 3 | 88 |
| 1.7B L11 | after (registered) | bulk (swap-AB) | 0.268 | 3.22 | 4 | 32 |
| 1.7B L11 | before (baseline footprint) | select_pv | 0.262 | 3.15 | 4 | 32 |
| 1.7B L11 | occupancy-only (A1) | select_pv | 0.327 | 3.92 | 5 | 0 |
| 1.7B L11 | after (registered) | select_pv | 0.314 | 3.77 | 5 | 0 |
| 4B L14 | before (baseline footprint) | bulk | 0.196 | 2.35 | 3 | 88 |
| 4B L14 | after (registered) | bulk (swap-AB) | 0.263 | 3.16 | 4 | 32 |
| 4B L14 | before (baseline footprint) | select_pv | 0.255 | 3.06 | 4 | 32 |
| 4B L14 | occupancy-only (A1) | select_pv | 0.317 | 3.81 | 5 | 0 |
| 4B L14 | after (registered) | select_pv | 0.316 | 3.79 | 5 | 0 |

Method:
- Every CTA records its SM id, clock64 and %globaltimer (32 ns resolution) at start and end.
- Achieved occupancy = Σ(CTA duration × 4 warps) / (56 SMs × makespan × 48 warps).
- "Before" uses instruction-copies of the baseline kernels at the baseline footprint: bulk 154 registers;
  select_pv 86 registers with MAXG-sized shared arrays (the baseline has 88 registers, 23,456 B).
- Instrumented launches are not the timed path; their outputs were checked bitwise (`profile/*/result.json`).
- Static census: `profile/*/result.json` `resources_static`. Baseline select_pv: 88 registers, 23,456 B shared, 4
  CTAs/SM. EB3 select_pv: 92 registers, 12,604 / 16,244 B, 5 CTAs/SM. Baseline bulk: 154 registers, 3 CTAs/SM.
  EB3 swap-AB bulk: 107/109 registers, 4 CTAs/SM.

## Footprint (G5)

| Cell | Launches before | Launches after | Bytes/call before | Bytes/call after | Extra bytes/call | Persistent ticket bytes (once per process) |
|---|---|---|---|---|---|---|
| Qwen3-1.7B L11 (S 8192) | 4 | 3 | 935,936 | 935,936 | 0 | 4,096 |
| Qwen3-4B L14 (S 8192) | 4 | 3 | 1,871,872 | 1,871,872 | 0 | 4,096 |
| syn H16 S2048 | 4 | 3 | 243,200 | 243,200 | 0 | 4,096 |
| syn H32 S2048 | 4 | 3 | 486,400 | 486,400 | 0 | 4,096 |
| syn H16 S65536 | 4 | 3 | 7,401,472 | 7,401,472 | 0 | 4,096 |
| syn H32 S65536 | 4 | 3 | 14,802,944 | 14,802,944 | 0 | 4,096 |

- Per-call bytes are the arena plus the rotated query plus the output. Before, the rotated query was its own
  allocation; now it lives inside the arena. Allocations per call: 3 → 2.
- The ticket buffer has one unsigned per (batch, KV head) and a minimum of 1,024 entries. It is zeroed once, and
  every completed launch returns it to zero through the `atomicInc` wrap, so it never needs a reset launch.
- The tickets assume one stream and one device, like the engine's legacy default stream. Two concurrent EB3
  decodes on different streams would share tickets. That is out of scope here, and it is documented at the code
  site.

## D1 arm (optional, non-bitwise)

Not built. Every adopted change is bitwise, and the registered verdict did not depend on a non-bitwise idea. No D1
speed or error numbers exist.

## Findings and residuals

1. **PDL is unavailable on this GPU (order premise).** `griddepcontrol.wait` does not assemble for sm_89: ptxas
   says "Modifier '.wait' requires .target sm_90 or higher". The host attribute is accepted but gives no safe early
   start, so PDL was not used. The ledger has the receipt.
2. **Registration defect.** `registration.json` `lead_predictions_verbatim` truncates LC3 and LC5 at the order's
   line wrap. No threshold is lost, and the full text is pinned through `order_sha256`, which equals the order
   file's SHA-256. The registration was not amended: this is not a crash, and the file is create-only. The
   generated `artifacts/apa_eb3/prediction_table.md` carries the truncated text; this report quotes the full text.
3. **Mutant 3 is killed only weakly.** It is a data race (the counter fires one CTA early) and was killed in 9 of
   1,024 real output checks (1.7B L11: 4 production + 3 diagnostic of 256; 4B L14: 2 diagnostic of 256),
   32/32 at S = 2048, and 0/16 at H32 S = 8000. G1 would only catch a similar real defect probabilistically. This
   is a named test-strength residual; a deterministic test would need a forced-delay hook in the last CTA.
4. **Mutant 6 shows why G1 poisons.** In production mode a reused, identical output buffer let 37 of 256 rows
   match by accident; the poisoned diagnostic call caught 256/256.
5. **LC2 MISS (receipt above).** The rotation costs 4.1–4.5 µs with the code prefetch and 3.5–3.9 µs without it.
   Its cost is one cold DRAM round trip for R plus a 128-step dependent FMA chain that this order forbids
   reassociating.
6. **The registered path is about 10 µs above the DRAM floor** (65–71 µs cold against a floor of about 56 µs;
   36–39 µs warm). Not pursued: a rotation fused into the bulk through cross-CTA waits (deadlock-sensitive) and
   deeper V pipelines.
7. **The CUDA-event tick (about 1.02 µs) quantizes every G4 number**, as in EB2b. Medians of 7 sample medians, all
   samples retained, no best-run selection.
8. **The G3 batch outlived the tool's 10-minute foreground limit** and was moved to the background by the harness.
   It was not parked: its output was collected on completion, and every step ran under its own lease.
9. **Development disclosure.** Dev timings (3 x 30 calls) on 1.7B L11, 4B L14 and synthetic cells chose the knobs
   before registration (ledger, Development). G4 is the first full registered measurement.

## Prior art

- **TurboQuant (Zandieh et al. 2025, arXiv:2504.19874):** the rotate-the-query identity and the TQ4 format,
  unchanged from SPQ1/SPQ6. Ours: only the rotation schedule.
- **Shared-memory staging with a padded tile** (NVIDIA CUDA C++ Best Practices Guide, coalescing; M. Harris 2013,
  "An Efficient Matrix Transpose in CUDA C/C++"). Taken for rotate_q.
- **Swap-AB operand transposition for small-M tensor-core tiles** (CUTLASS-style). FlashInfer's GQA M-packing (Ye
  et al. 2025), which SPQ4 inherited, is the opposite choice. Unverified as a single citation; lead to check:
  "CUTLASS swap AB small M", "swapAB GEMM tensor core decode". Ours: applying it to the TQ4 LUT bulk tile with the
  k-slot permutation and the three-term bf16 split (Henry et al. 2019; Ootomo & Yokota 2022, inherited) held fixed,
  bitwise.
- **Flash-Decoding (Dao, Haziza, Massa, Sizov 2023):** split-K and merge, arithmetic unchanged.
- **Threadfence reduction / last-block pattern** (NVIDIA CUDA Samples `threadFenceReduction`; CUDA C++ Programming
  Guide memory-fence example). Taken. Ours: per-(batch, KV head) tickets self-reset by the `atomicInc` wrap, and a
  last-CTA G-head merge with the ordered chain.
- **Programmatic dependent launch** (CUDA C++ Programming Guide; PTX `griddepcontrol`, sm_90+). Evaluated, not
  used.
- **Software prefetching** (Mowry, Lam, Gupta 1992; PTX `prefetch.global.L2`). Taken. Ours: the cross-kernel
  schedule under an L2-capacity budget.
- **Occupancy / `__launch_bounds__`** (CUDA C++ Programming Guide). Taken.
- **BLASST (Yuan et al. 2025, arXiv:2512.12087):** the running-max criterion, inherited unchanged in the SP rule.
- **Mutation testing** (DeMillo, Lipton, Sayward 1978). Taken.
- **SHA-256 manifests** (NIST FIPS 180-4).
- The SPQ6/EB2b cold-L2 protocol and receipt plumbing are inherited (Project-Tensor 2026).
- No prior art known to me for the exact assembly, the CTA-timeline occupancy integral, or the receipt layout. No
  novelty claimed. Code-site annotations are in `eb3.cuh`, `eb3_tuning.cuh` and the EB3 blocks of `spq4.cuh`,
  `bindings.cpp`, `ops.h` and `__init__.py`; the ledger carries the same.

## Scope

- **Evidence class:** author-run CUDA kernel correctness (bitwise reference-pair, same-process and cross-binary),
  compile-time mutation tests, and cold-L2 kernel micro-benchmarks. RTX 4070 SUPER 12 GB, native driver 595.99.02,
  CUDA 12.6.
- **Bulk tier on every APA line:** TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate
  (variant B).
- **Exact tier:** bf16 K refined by the SP row-carry rule at δ 8.1 with a GQA union gather; bf16 V; C = 256.
- **Shapes:** captured Qwen3-1.7B (H16/KV8) and Qwen3-4B (H32/KV8) at D = 128, one wikitext sequence, S = 8192.
  G1 uses the last 64 causal-prefix rows; G4 uses the last 8 query rows with full visibility. Synthetic H16/KV8 and
  H32/KV8 at 2K–64K, plus 8000 for G1.
- **Does not cover:**
  - model-level quality (unaffected if G1 holds, which it does);
  - prefill;
  - other GPUs;
  - CUDA-graph capture in a model loop (AB-1);
  - the tk8 / EB value-precision path (option 2);
  - multi-stream concurrent use of the ticket buffer;
  - any production promotion.
- **Author checks are not a blind review.** No service, production, engine or other-worktree writes; no git; no
  subagents; no deletes; no process kills.

## Every file written

All paths are relative to the worktree. Edited existing files (additive, flag-guarded): `tensor_cuda/src/apa_spo/spq4.cuh`, `tensor_cuda/src/bindings.cpp`, `tensor_cuda/include/tc/ops.h`, `tensor_cuda/tensor_cuda/__init__.py`. New source files: `tensor_cuda/src/apa_spo/eb3.cuh`, `tensor_cuda/src/apa_spo/eb3_tuning.cuh`. Also written, outside the worktree, in the session scratchpad `<local scratch>` (temporary, not repo files, OS-wiped): the PDL probe (`scratchpad/pdl/pdl.cu`, `pdlw.cu` and binaries), the register-pressure probe (`scratchpad/regs/bulk_mb*.cu/.o`), report fragments (`scratchpad/speed.md`, `report_head.md`, `report_mid.md`) and the harness's background-task output file for the G3 batch (`tasks/bxyvt5efb.output`, copied to `logs/apa_eb3_g3_batch.log`). 352 worktree files (artifacts/apa_eb3/written_files.txt):

```text
artifacts/apa_eb3/build/dev1/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb3/build/dev1/bindings.o
artifacts/apa_eb3/build/dev1/device_link.o
artifacts/apa_eb3/build/dev1/kernels.o
artifacts/apa_eb3/build/dev1/module.sha256
artifacts/apa_eb3/build/dev2/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb3/build/dev2/bindings.o
artifacts/apa_eb3/build/dev2/device_link.o
artifacts/apa_eb3/build/dev2/kernels.o
artifacts/apa_eb3/build/dev2/module.sha256
artifacts/apa_eb3/build/dev3/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb3/build/dev3/bindings.o
artifacts/apa_eb3/build/dev3/device_link.o
artifacts/apa_eb3/build/dev3/kernels.o
artifacts/apa_eb3/build/dev3/module.sha256
artifacts/apa_eb3/build/main/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb3/build/main/bindings.o
artifacts/apa_eb3/build/main/device_link.o
artifacts/apa_eb3/build/main/kernels.o
artifacts/apa_eb3/build/main/module.sha256
artifacts/apa_eb3/build/mutant1/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb3/build/mutant1/bindings.o
artifacts/apa_eb3/build/mutant1/device_link.o
artifacts/apa_eb3/build/mutant1/kernels.o
artifacts/apa_eb3/build/mutant1/module.sha256
artifacts/apa_eb3/build/mutant2/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb3/build/mutant2/bindings.o
artifacts/apa_eb3/build/mutant2/device_link.o
artifacts/apa_eb3/build/mutant2/kernels.o
artifacts/apa_eb3/build/mutant2/module.sha256
artifacts/apa_eb3/build/mutant3/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb3/build/mutant3/bindings.o
artifacts/apa_eb3/build/mutant3/device_link.o
artifacts/apa_eb3/build/mutant3/kernels.o
artifacts/apa_eb3/build/mutant3/module.sha256
artifacts/apa_eb3/build/mutant4/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb3/build/mutant4/bindings.o
artifacts/apa_eb3/build/mutant4/device_link.o
artifacts/apa_eb3/build/mutant4/kernels.o
artifacts/apa_eb3/build/mutant4/module.sha256
artifacts/apa_eb3/build/mutant5/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb3/build/mutant5/bindings.o
artifacts/apa_eb3/build/mutant5/device_link.o
artifacts/apa_eb3/build/mutant5/kernels.o
artifacts/apa_eb3/build/mutant5/module.sha256
artifacts/apa_eb3/build/mutant6/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb3/build/mutant6/bindings.o
artifacts/apa_eb3/build/mutant6/device_link.o
artifacts/apa_eb3/build/mutant6/kernels.o
artifacts/apa_eb3/build/mutant6/module.sha256
artifacts/apa_eb3/build/sass_identity.json
artifacts/apa_eb3/build_baseline/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb3/build_baseline/bindings.o
artifacts/apa_eb3/build_baseline/device_link.o
artifacts/apa_eb3/build_baseline/kernels.o
artifacts/apa_eb3/build_baseline/module.sha256
artifacts/apa_eb3/dev/dev1_profile/result.json
artifacts/apa_eb3/dev/dev1_smoke/result.json
artifacts/apa_eb3/dev/dev1_timing/result.json
artifacts/apa_eb3/dev/dev2_smoke/result.json
artifacts/apa_eb3/dev/dev2_timing/result.json
artifacts/apa_eb3/dev/dev3_smoke/result.json
artifacts/apa_eb3/dev/dev3_timing/result.json
artifacts/apa_eb3/footprint_table.md
artifacts/apa_eb3/g0/Qwen3-1.7B_L11/result.json
artifacts/apa_eb3/g0/result.json
artifacts/apa_eb3/g1/Qwen3-1.7B_L00/result.json
artifacts/apa_eb3/g1/Qwen3-1.7B_L05/result.json
artifacts/apa_eb3/g1/Qwen3-1.7B_L11/result.json
artifacts/apa_eb3/g1/Qwen3-1.7B_L16/result.json
artifacts/apa_eb3/g1/Qwen3-1.7B_L22/result.json
artifacts/apa_eb3/g1/Qwen3-1.7B_L27/result.json
artifacts/apa_eb3/g1/Qwen3-4B_L00/result.json
artifacts/apa_eb3/g1/Qwen3-4B_L07/result.json
artifacts/apa_eb3/g1/Qwen3-4B_L14/result.json
artifacts/apa_eb3/g1/Qwen3-4B_L21/result.json
artifacts/apa_eb3/g1/Qwen3-4B_L28/result.json
artifacts/apa_eb3/g1/Qwen3-4B_L35/result.json
artifacts/apa_eb3/g1/result.json
artifacts/apa_eb3/g1/syn_H16_KV8_S2048/result.json
artifacts/apa_eb3/g1/syn_H16_KV8_S32768/result.json
artifacts/apa_eb3/g1/syn_H16_KV8_S65536/result.json
artifacts/apa_eb3/g1/syn_H16_KV8_S8000/result.json
artifacts/apa_eb3/g1/syn_H16_KV8_S8192/result.json
artifacts/apa_eb3/g1/syn_H32_KV8_S2048/result.json
artifacts/apa_eb3/g1/syn_H32_KV8_S32768/result.json
artifacts/apa_eb3/g1/syn_H32_KV8_S65536/result.json
artifacts/apa_eb3/g1/syn_H32_KV8_S8000/result.json
artifacts/apa_eb3/g1/syn_H32_KV8_S8192/result.json
artifacts/apa_eb3/g1_baseline_binary/Qwen3-1.7B_L00/result.json
artifacts/apa_eb3/g1_baseline_binary/Qwen3-1.7B_L00/rows.npz
artifacts/apa_eb3/g1_baseline_binary/Qwen3-1.7B_L05/result.json
artifacts/apa_eb3/g1_baseline_binary/Qwen3-1.7B_L05/rows.npz
artifacts/apa_eb3/g1_baseline_binary/Qwen3-1.7B_L11/result.json
artifacts/apa_eb3/g1_baseline_binary/Qwen3-1.7B_L11/rows.npz
artifacts/apa_eb3/g1_baseline_binary/Qwen3-1.7B_L16/result.json
artifacts/apa_eb3/g1_baseline_binary/Qwen3-1.7B_L16/rows.npz
artifacts/apa_eb3/g1_baseline_binary/Qwen3-1.7B_L22/result.json
artifacts/apa_eb3/g1_baseline_binary/Qwen3-1.7B_L22/rows.npz
artifacts/apa_eb3/g1_baseline_binary/Qwen3-1.7B_L27/result.json
artifacts/apa_eb3/g1_baseline_binary/Qwen3-1.7B_L27/rows.npz
artifacts/apa_eb3/g1_baseline_binary/Qwen3-4B_L00/result.json
artifacts/apa_eb3/g1_baseline_binary/Qwen3-4B_L00/rows.npz
artifacts/apa_eb3/g1_baseline_binary/Qwen3-4B_L07/result.json
artifacts/apa_eb3/g1_baseline_binary/Qwen3-4B_L07/rows.npz
artifacts/apa_eb3/g1_baseline_binary/Qwen3-4B_L14/result.json
artifacts/apa_eb3/g1_baseline_binary/Qwen3-4B_L14/rows.npz
artifacts/apa_eb3/g1_baseline_binary/Qwen3-4B_L21/result.json
artifacts/apa_eb3/g1_baseline_binary/Qwen3-4B_L21/rows.npz
artifacts/apa_eb3/g1_baseline_binary/Qwen3-4B_L28/result.json
artifacts/apa_eb3/g1_baseline_binary/Qwen3-4B_L28/rows.npz
artifacts/apa_eb3/g1_baseline_binary/Qwen3-4B_L35/result.json
artifacts/apa_eb3/g1_baseline_binary/Qwen3-4B_L35/rows.npz
artifacts/apa_eb3/g1_baseline_binary/syn_H16_KV8_S2048/result.json
artifacts/apa_eb3/g1_baseline_binary/syn_H16_KV8_S2048/rows.npz
artifacts/apa_eb3/g1_baseline_binary/syn_H16_KV8_S32768/result.json
artifacts/apa_eb3/g1_baseline_binary/syn_H16_KV8_S32768/rows.npz
artifacts/apa_eb3/g1_baseline_binary/syn_H16_KV8_S65536/result.json
artifacts/apa_eb3/g1_baseline_binary/syn_H16_KV8_S65536/rows.npz
artifacts/apa_eb3/g1_baseline_binary/syn_H16_KV8_S8000/result.json
artifacts/apa_eb3/g1_baseline_binary/syn_H16_KV8_S8000/rows.npz
artifacts/apa_eb3/g1_baseline_binary/syn_H16_KV8_S8192/result.json
artifacts/apa_eb3/g1_baseline_binary/syn_H16_KV8_S8192/rows.npz
artifacts/apa_eb3/g1_baseline_binary/syn_H32_KV8_S2048/result.json
artifacts/apa_eb3/g1_baseline_binary/syn_H32_KV8_S2048/rows.npz
artifacts/apa_eb3/g1_baseline_binary/syn_H32_KV8_S32768/result.json
artifacts/apa_eb3/g1_baseline_binary/syn_H32_KV8_S32768/rows.npz
artifacts/apa_eb3/g1_baseline_binary/syn_H32_KV8_S65536/result.json
artifacts/apa_eb3/g1_baseline_binary/syn_H32_KV8_S65536/rows.npz
artifacts/apa_eb3/g1_baseline_binary/syn_H32_KV8_S8000/result.json
artifacts/apa_eb3/g1_baseline_binary/syn_H32_KV8_S8000/rows.npz
artifacts/apa_eb3/g1_baseline_binary/syn_H32_KV8_S8192/result.json
artifacts/apa_eb3/g1_baseline_binary/syn_H32_KV8_S8192/rows.npz
artifacts/apa_eb3/g2/result.json
artifacts/apa_eb3/g2_legacy/apa_fxtc1/g1.json
artifacts/apa_eb3/g2_legacy/apa_fxtc1/g1_captures_sha.json
artifacts/apa_eb3/g2_legacy/apa_fxtc1/g1_hash_fxtc1.json
artifacts/apa_eb3/g2_legacy/apa_fxtc1/registration.json
artifacts/apa_eb3/g2_legacy/apa_fxtc1/registration.sha256
artifacts/apa_eb3/g2_legacy/apa_spq4/g1_hash_a6752ea_ref.json
artifacts/apa_eb3/g2_legacy/apa_spq6/g1_hash_eb3.json
artifacts/apa_eb3/g2_legacy/apa_spq6/g1_hash_spq6.json
artifacts/apa_eb3/g2_legacy/apa_spq6/registration.json
artifacts/apa_eb3/g2_legacy/apa_spq6/registration.sha256
artifacts/apa_eb3/g2_legacy/source_hashes_eb2b.txt
artifacts/apa_eb3/g3/baseline/result.json
artifacts/apa_eb3/g3/mutant1/result.json
artifacts/apa_eb3/g3/mutant2/result.json
artifacts/apa_eb3/g3/mutant3/result.json
artifacts/apa_eb3/g3/mutant4/result.json
artifacts/apa_eb3/g3/mutant5/result.json
artifacts/apa_eb3/g3/mutant6/result.json
artifacts/apa_eb3/g3/result.json
artifacts/apa_eb3/g4/Qwen3-1.7B_L00/result.json
artifacts/apa_eb3/g4/Qwen3-1.7B_L05/result.json
artifacts/apa_eb3/g4/Qwen3-1.7B_L11/result.json
artifacts/apa_eb3/g4/Qwen3-1.7B_L16/result.json
artifacts/apa_eb3/g4/Qwen3-1.7B_L22/result.json
artifacts/apa_eb3/g4/Qwen3-1.7B_L27/result.json
artifacts/apa_eb3/g4/Qwen3-4B_L00/result.json
artifacts/apa_eb3/g4/Qwen3-4B_L07/result.json
artifacts/apa_eb3/g4/Qwen3-4B_L14/result.json
artifacts/apa_eb3/g4/Qwen3-4B_L21/result.json
artifacts/apa_eb3/g4/Qwen3-4B_L28/result.json
artifacts/apa_eb3/g4/Qwen3-4B_L35/result.json
artifacts/apa_eb3/g4/result.json
artifacts/apa_eb3/g4/syn_H16_KV8_S2048/result.json
artifacts/apa_eb3/g4/syn_H16_KV8_S32768/result.json
artifacts/apa_eb3/g4/syn_H16_KV8_S65536/result.json
artifacts/apa_eb3/g4/syn_H16_KV8_S8192/result.json
artifacts/apa_eb3/g4/syn_H32_KV8_S2048/result.json
artifacts/apa_eb3/g4/syn_H32_KV8_S32768/result.json
artifacts/apa_eb3/g4/syn_H32_KV8_S65536/result.json
artifacts/apa_eb3/g4/syn_H32_KV8_S8192/result.json
artifacts/apa_eb3/g4x/Qwen3-1.7B_L11/result.json
artifacts/apa_eb3/g4x/Qwen3-4B_L14/result.json
artifacts/apa_eb3/g4x_table.md
artifacts/apa_eb3/g5/Qwen3-1.7B_L11/result.json
artifacts/apa_eb3/g5/Qwen3-4B_L14/result.json
artifacts/apa_eb3/g5/result.json
artifacts/apa_eb3/g5/syn_H16_KV8_S2048/result.json
artifacts/apa_eb3/g5/syn_H16_KV8_S65536/result.json
artifacts/apa_eb3/g5/syn_H32_KV8_S2048/result.json
artifacts/apa_eb3/g5/syn_H32_KV8_S65536/result.json
artifacts/apa_eb3/immutable_inputs.json
artifacts/apa_eb3/occupancy_table.md
artifacts/apa_eb3/path_copies.diff
artifacts/apa_eb3/path_copies_legacy.diff
artifacts/apa_eb3/prediction_table.md
artifacts/apa_eb3/profile/Qwen3-1.7B_L11/result.json
artifacts/apa_eb3/profile/Qwen3-4B_L14/result.json
artifacts/apa_eb3/profile_table.md
artifacts/apa_eb3/registration.json
artifacts/apa_eb3/registration.sha256
artifacts/apa_eb3/source_before.sha256
artifacts/apa_eb3/source_before/tensor_cuda_include_tc_ops.h
artifacts/apa_eb3/source_before/tensor_cuda_src_apa_spo_merge.cuh
artifacts/apa_eb3/source_before/tensor_cuda_src_apa_spo_packed.cuh
artifacts/apa_eb3/source_before/tensor_cuda_src_apa_spo_packed_launcher.cuh
artifacts/apa_eb3/source_before/tensor_cuda_src_apa_spo_spq4.cuh
artifacts/apa_eb3/source_before/tensor_cuda_src_apa_spo_spq6.cuh
artifacts/apa_eb3/source_before/tensor_cuda_src_bindings.cpp
artifacts/apa_eb3/source_before/tensor_cuda_tensor_cuda___init__.py
artifacts/apa_eb3/speed_table.md
artifacts/apa_eb3/summary.json
docs/APA_EB3_LEDGER.md
docs/APA_EB3_REPORT.md
logs/apa_eb3_bdump_Qwen3-1.7B_L0.log
logs/apa_eb3_bdump_Qwen3-1.7B_L11.log
logs/apa_eb3_bdump_Qwen3-1.7B_L16.log
logs/apa_eb3_bdump_Qwen3-1.7B_L22.log
logs/apa_eb3_bdump_Qwen3-1.7B_L27.log
logs/apa_eb3_bdump_Qwen3-1.7B_L5.log
logs/apa_eb3_bdump_Qwen3-4B_L0.log
logs/apa_eb3_bdump_Qwen3-4B_L14.log
logs/apa_eb3_bdump_Qwen3-4B_L21.log
logs/apa_eb3_bdump_Qwen3-4B_L28.log
logs/apa_eb3_bdump_Qwen3-4B_L35.log
logs/apa_eb3_bdump_Qwen3-4B_L7.log
logs/apa_eb3_bdump_batch.log
logs/apa_eb3_bdump_syn_H16_S2048.log
logs/apa_eb3_bdump_syn_H16_S32768.log
logs/apa_eb3_bdump_syn_H16_S65536.log
logs/apa_eb3_bdump_syn_H16_S8000.log
logs/apa_eb3_bdump_syn_H16_S8192.log
logs/apa_eb3_bdump_syn_H32_S2048.log
logs/apa_eb3_bdump_syn_H32_S32768.log
logs/apa_eb3_bdump_syn_H32_S65536.log
logs/apa_eb3_bdump_syn_H32_S8000.log
logs/apa_eb3_bdump_syn_H32_S8192.log
logs/apa_eb3_build_baseline.log
logs/apa_eb3_build_dev1.log
logs/apa_eb3_build_dev2.log
logs/apa_eb3_build_dev3.log
logs/apa_eb3_build_main.log
logs/apa_eb3_build_mutant1.log
logs/apa_eb3_build_mutant2.log
logs/apa_eb3_build_mutant3.log
logs/apa_eb3_build_mutant4.log
logs/apa_eb3_build_mutant5.log
logs/apa_eb3_build_mutant6.log
logs/apa_eb3_cpu_tests.log
logs/apa_eb3_dev1_profile.log
logs/apa_eb3_dev1_smoke.log
logs/apa_eb3_dev1_timing.log
logs/apa_eb3_dev2_smoke.log
logs/apa_eb3_dev2_timing.log
logs/apa_eb3_dev3_smoke.log
logs/apa_eb3_dev3_timing.log
logs/apa_eb3_finalize.log
logs/apa_eb3_g0.log
logs/apa_eb3_g1_Qwen3-1.7B_L0.log
logs/apa_eb3_g1_Qwen3-1.7B_L11.log
logs/apa_eb3_g1_Qwen3-1.7B_L16.log
logs/apa_eb3_g1_Qwen3-1.7B_L22.log
logs/apa_eb3_g1_Qwen3-1.7B_L27.log
logs/apa_eb3_g1_Qwen3-1.7B_L5.log
logs/apa_eb3_g1_Qwen3-4B_L0.log
logs/apa_eb3_g1_Qwen3-4B_L14.log
logs/apa_eb3_g1_Qwen3-4B_L21.log
logs/apa_eb3_g1_Qwen3-4B_L28.log
logs/apa_eb3_g1_Qwen3-4B_L35.log
logs/apa_eb3_g1_Qwen3-4B_L7.log
logs/apa_eb3_g1_batch.log
logs/apa_eb3_g1_syn_H16_S2048.log
logs/apa_eb3_g1_syn_H16_S32768.log
logs/apa_eb3_g1_syn_H16_S65536.log
logs/apa_eb3_g1_syn_H16_S8000.log
logs/apa_eb3_g1_syn_H16_S8192.log
logs/apa_eb3_g1_syn_H32_S2048.log
logs/apa_eb3_g1_syn_H32_S32768.log
logs/apa_eb3_g1_syn_H32_S65536.log
logs/apa_eb3_g1_syn_H32_S8000.log
logs/apa_eb3_g1_syn_H32_S8192.log
logs/apa_eb3_g2_fxtc1_captures.log
logs/apa_eb3_g2_fxtc1_compare.log
logs/apa_eb3_g2_fxtc1_hashes.log
logs/apa_eb3_g2_spq6_hashes.log
logs/apa_eb3_g3_baseline.log
logs/apa_eb3_g3_batch.log
logs/apa_eb3_g3_mutant1.log
logs/apa_eb3_g3_mutant2.log
logs/apa_eb3_g3_mutant3.log
logs/apa_eb3_g3_mutant4.log
logs/apa_eb3_g3_mutant5.log
logs/apa_eb3_g3_mutant6.log
logs/apa_eb3_g4_Qwen3-1.7B_L0.log
logs/apa_eb3_g4_Qwen3-1.7B_L11.log
logs/apa_eb3_g4_Qwen3-1.7B_L16.log
logs/apa_eb3_g4_Qwen3-1.7B_L22.log
logs/apa_eb3_g4_Qwen3-1.7B_L27.log
logs/apa_eb3_g4_Qwen3-1.7B_L5.log
logs/apa_eb3_g4_Qwen3-4B_L0.log
logs/apa_eb3_g4_Qwen3-4B_L14.log
logs/apa_eb3_g4_Qwen3-4B_L21.log
logs/apa_eb3_g4_Qwen3-4B_L28.log
logs/apa_eb3_g4_Qwen3-4B_L35.log
logs/apa_eb3_g4_Qwen3-4B_L7.log
logs/apa_eb3_g4_syn_H16_S2048.log
logs/apa_eb3_g4_syn_H16_S32768.log
logs/apa_eb3_g4_syn_H16_S65536.log
logs/apa_eb3_g4_syn_H16_S8192.log
logs/apa_eb3_g4_syn_H32_S2048.log
logs/apa_eb3_g4_syn_H32_S32768.log
logs/apa_eb3_g4_syn_H32_S65536.log
logs/apa_eb3_g4_syn_H32_S8192.log
logs/apa_eb3_g4x_Qwen3-1.7B_L11.log
logs/apa_eb3_g4x_Qwen3-4B_L14.log
logs/apa_eb3_g5_Qwen3-1.7B_L11.log
logs/apa_eb3_g5_Qwen3-4B_L14.log
logs/apa_eb3_g5_syn_H16_S2048.log
logs/apa_eb3_g5_syn_H16_S65536.log
logs/apa_eb3_g5_syn_H32_S2048.log
logs/apa_eb3_g5_syn_H32_S65536.log
logs/apa_eb3_measure_g4x.log
logs/apa_eb3_measure_g5.log
logs/apa_eb3_measure_profile.log
logs/apa_eb3_measure_real.log
logs/apa_eb3_measure_syn.log
logs/apa_eb3_profile_Qwen3-1.7B_L11.log
logs/apa_eb3_profile_Qwen3-4B_L14.log
logs/apa_eb3_registration.log
logs/apa_eb3_sass_identity.log
logs/apa_eb3_synthesis.log
scripts/apa_eb3_admin.py
scripts/apa_eb3_batch.sh
scripts/apa_eb3_build.sh
scripts/apa_eb3_dev.py
scripts/apa_eb3_fxtc1_common.py
scripts/apa_eb3_fxtc1_g1.py
scripts/apa_eb3_fxtc1_gpu.sh
scripts/apa_eb3_fxtc1_lib.py
scripts/apa_eb3_gates.py
scripts/apa_eb3_gpu.sh
scripts/apa_eb3_lib.py
scripts/apa_eb3_measure.sh
scripts/apa_eb3_mutants.sh
scripts/apa_eb3_profile.py
scripts/apa_eb3_sass.py
scripts/apa_eb3_spq1_common.py
scripts/apa_eb3_spq6_common.py
scripts/apa_eb3_spq6_g1.py
scripts/apa_eb3_spq6_gpu.sh
scripts/apa_eb3_spq6_lib.py
scripts/apa_eb3_timing.py
tests/test_apa_eb3.py
tensor_cuda/src/apa_spo/eb3.cuh
tensor_cuda/src/apa_spo/eb3_tuning.cuh
tensor_cuda/src/apa_spo/spq4.cuh
tensor_cuda/src/bindings.cpp
tensor_cuda/include/tc/ops.h
tensor_cuda/tensor_cuda/__init__.py
```

## Spec ledger

- G0 build and smoke: **met**. `artifacts/apa_eb3/g0/result.json`
  (`g0/Qwen3-1.7B_L11/result.json`, `build_baseline/module.sha256`, `build/main/module.sha256`).
- G1 bitwise identity (real 12 x 64 x 4 configurations, synthetic incl. S=8000, G=2/4, sinks off/on, rotated q,
  cross-binary): **met**. `artifacts/apa_eb3/g1/result.json`, `artifacts/apa_eb3/g1/*/result.json`,
  `artifacts/apa_eb3/g1_baseline_binary/*/`.
- G2 legacy hashes, feature off (72 r2, 32 INT4-tc, 136 SPQ6, 48 capture files): **met**.
  `artifacts/apa_eb3/g2/result.json`, `artifacts/apa_eb3/g2_legacy/`, `artifacts/apa_eb3/build/sass_identity.json`.
- G3 at least 6 compile-time mutants, each killed by G1: **met (6/6; mutant 3 killed weakly)**.
  `artifacts/apa_eb3/g3/result.json`, `artifacts/apa_eb3/g3/mutant{1..6}/result.json`.
- Registration after G0-G3 and before G4, create-only, with SHA-256: **met**. `artifacts/apa_eb3/registration.json`,
  `artifacts/apa_eb3/registration.sha256`.
- Registration holds the lead predictions verbatim: **not fully met**. LC3 and LC5 are truncated at the order's line
  wrap; the full text is pinned through `order_sha256`. Receipts: `artifacts/apa_eb3/registration.json`,
  `docs/APA_EB3_LEDGER.md` (Registration).
- Registration holds the timing protocol, seat predictions S1-S8, every source/script hash and both binary hashes:
  **met**. `artifacts/apa_eb3/registration.json`.
- G4 speed (real per layer and non-early medians, synthetic 2K/8K/32K/64K, before/after per-phase profile per
  model): **met**. `artifacts/apa_eb3/g4/result.json`, `artifacts/apa_eb3/g4/*/result.json`,
  `artifacts/apa_eb3/profile/*/result.json`, `artifacts/apa_eb3/g4x/*/result.json`.
- G5 footprint (extra device memory and launches per call, before/after): **met**. `artifacts/apa_eb3/g5/result.json`,
  `artifacts/apa_eb3/g5/*/result.json`.
- Registered speed verdict: **met, WIN**. `artifacts/apa_eb3/summary.json`.
- Build receipts: **met**. `logs/apa_eb3_build_{baseline,main,mutant1..6,dev1..3}.log`,
  `artifacts/apa_eb3/build*/**/module.sha256`.
- Per-cell JSON for G0-G5: **met**. `artifacts/apa_eb3/{g0,g1,g3,g4,g4x,g5,profile}/*/result.json`,
  `artifacts/apa_eb3/g2_legacy/`.
- Before-and-after profile tables, including achieved occupancy: **met**. `artifacts/apa_eb3/profile_table.md`,
  `artifacts/apa_eb3/occupancy_table.md`, this report.
- Mutation receipts: **met**. `artifacts/apa_eb3/g3/`, `logs/apa_eb3_g3_*.log`, `logs/apa_eb3_build_mutant*.log`.
- Prediction table: **met**. This report (full text); `artifacts/apa_eb3/prediction_table.md`, which carries the
  truncated LC3/LC5 text.
- Speed table, x-B and x-flash per cell class with the bulk-tier precision on each line: **met**. This report and
  `artifacts/apa_eb3/speed_table.md`.
- D1 arm (optional): **not built (optional; no receipt)**.
- `docs/APA_EB3_REPORT.md`, verdict first, with G0-G5, the prediction table, the speed table, the before-and-after
  profile, the D1 status, Prior art and Scope: **met**. `docs/APA_EB3_REPORT.md`.
- `docs/APA_EB3_LEDGER.md`: **met**. `docs/APA_EB3_LEDGER.md`.
- Prior art at the code sites, in the ledger and in the report: **met**. `tensor_cuda/src/apa_spo/eb3.cuh`,
  `eb3_tuning.cuh`, the EB3 blocks in `spq4.cuh`, `bindings.cpp`, `ops.h` and `__init__.py`; the ledger; this
  report.
- Path-only copies of the EB2b scripts, with each diff recorded: **met**. `artifacts/apa_eb3/path_copies.diff`,
  `artifacts/apa_eb3/path_copies_legacy.diff`, ledger.
- Additive, flag-guarded edits only; the default path byte-identical in behaviour: **met**.
  `tests/test_apa_eb3.py` (`logs/apa_eb3_cpu_tests.log`), `artifacts/apa_eb3/build/sass_identity.json`,
  `artifacts/apa_eb3/g2/result.json`.
- Baseline binary never rebuilt after registration: **met**. `artifacts/apa_eb3/final_audit.json` (registered
  binary hashes re-checked).
- Rails (RSS < 16 GB, `artifacts/apa_eb3` < 3 GB): **met** (peak RSS 1.99 GB; artifacts about 435 MB).
  `artifacts/apa_eb3/final_audit.json`, per-receipt `resources`.
- No git writes, no subagents, no deletes, writes only in the path list, GPU only under flock/timeout, native
  driver: **met**. `artifacts/apa_eb3/final_audit.json`, ledger.
- List of every file written: **met**. `artifacts/apa_eb3/written_files.txt`, this report.
