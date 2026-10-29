# APA-EB4 report: a cheap tk8 decision on the EB3 path

**Verdicts (registered rules).** Only **T-fresh** (the exact tk8 rule, the control) was built: the P1 offline gate left
**T-stale (1/9)** and **T-thr (4/9)** not QUALIFIED at 8K, and the order sends kernel work to QUALIFIED arms only.
- **T-fresh speed at 8K: WIN.** Real 8K non-early median T-fresh / EB3 = **0.8562** (Qwen3-1.7B, x-EB3 1.168) and **0.8769** (Qwen3-4B, x-EB3 1.140); x-flash 1.481 / 1.404.
- **T-fresh LONG at 32K: WIN.** Real 32K non-early median T-fresh / EB3 = **0.7665** / **0.7838**; x-flash 1.819 / 1.670.
- **T-stale, T-thr:** no speed or LONG verdict (not built). Their P1 failures are the result.

Scope of every speed number: RTX 4070 SUPER 12 GB; bulk tier TQ4, 4.25 bits per element stored, computed bf16 hi/lo
with fp32 accumulate (variant B); V per 64-key block bf16 or int8 (EB2b format) by tk8 at EB2b's f_L; cold L2; B = 1,
D = VD = 128, C = 256; real cells time the last 8 query rows with all S keys visible. Kernel evidence only, no model
quality claim.

Registration SHA-256: `071f7fb5139f6fd326d1fbc7f738835b73d699c236beab8ab5bd4c78a431421b`.

- **P0 PASS:** real 32K captures for both models (4B reached the full 32,768 tokens), manifests with hashes.
- **P1:** T-fresh 9/9 (8K) and 9/9 (32K); T-stale 1/9 and 8/9; T-thr 4/9 and 7/9. θ_L frozen before any kernel
  timing (`theta.json` SHA-256 `c8cb8dcea28169f2a74301b10a5074059a4f9f0608203f940757ce3d1eeb054f`).
- **G0 PASS**, **G1 PASS**, **G2 PASS**, **G3 PASS (9/10 killed; mutant 7 survived, mutant 8 weak)**, **G4 PASS**
  (measurement gate), **G5 PASS**.
- **Predictions:** LD1 MISS, LD2 MISS, LD3 HIT, LD4 MISS, LD5 MISS, LD6 NOT TESTED (T-thr not built), LD7 HIT; S1–S9
  HIT.
- **Finding about EB3's receipts:** every `outs` array in EB3's `g1_baseline_binary/*/rows.npz` is all zero (uint16
  truncation of fp32-returned bf16 bits in EB3's driver), so EB3's cross-binary *output* comparison was vacuous. EB3's
  in-process comparisons and cross-binary digests were real, and the lead's independent 33/33 check stands. EB4's G1.1
  and G2 supplement re-establish the cross-binary output identity with full 32-bit receipts (details below).

## What was built

`tensor_cuda.apa_eb4(q, k, v, vi, vs, packed, scale, delta, fraction, ...)`, compiled only with `-DTC_APA_EB4=1`
(`tensor_cuda/src/apa_spo/eb4.cuh`, knobs in `eb4_tuning.cuh`; additive guarded blocks in `spq4.cuh`, `bindings.cpp`,
`ops.h`, `__init__.py`). Three launches, as EB3:
1. `eb3::rotate_q` with EB3's codes+norms L2 prefetch: the same kernel EB3 launches.
2. `eb4::bulk_stats`: EB3's swap-AB bulk tile instruction for instruction for scores and chunk maxima, plus EB2b's
   per-(row, chunk) statistics in the epilogue (`cz`, `blog`; EB2b's fp32 order) from shared-memory-staged scores, and
   an L2 prefetch of the chunk's int8 V and scales.
3. `eb4::select_mixed`: EB3's select (S1–S3 and the fused merge verbatim). Before S1, each CTA computes the tk8 decision
   for its own 4 blocks with EB2b's arithmetic: per-head lz as an ordered fma chain fed by warp shuffles, then every
   block's score as `max_g __expf(blog - lz)`. Each own block's stable descending rank comes from comparison counting,
   so the flags equal EB2b's bitonic flags bitwise. The CTA-start prefetch follows the flags. S4 loads each V row as
   bf16 or as int8 + scale, dequantizing with `eb2b::value8` into the unchanged bf16 tile and PV mma.

No key is dropped; only int8 blocks' V operand changes. With every flag forced to bf16 the output is EB3's, bitwise. A
development alternative that runs the choose in the bulk kernel's last CTA (bitonic sort, as EB2b) is kept in the
source under a tuning knob. It was not registered: its serial sort added 7.7 µs at 8K and 20 µs at 32K.

## P0: real 32K captures

`scripts/apa_eb4_capture.py` is a copy of SPQ3's capture script with only the order's changes; the diff is
`artifacts/apa_eb4/p0/capture_copy.diff`. Text: wikitext-2-raw-v1 train, chars[0:400000], no special tokens, first
32,768 ids. The first 8,192 ids equal SPQ3's (`token_ids_sha256` 1685f7be…). Pinned snapshots. Stock sdpa attention
(`is_causal`, mask None). One full-sequence forward per model with `use_cache=False`:
- 1.7B: 5.0 s, peak 5.21 GB CUDA;
- 4B: 10.1 s, peak 10.65 GB CUDA.

Each stores full k/v and the last 256 rows of q/o. Manifests:
- `artifacts/apa_eb4/p0/captures/Qwen3-1.7B/manifest.json` (SHA-256 f1fa3dd3af919fde8a93703a6b4def350fb8cd6248605e1ea53a545c0ee275d9, 817,892,352 B);
- `.../Qwen3-4B/manifest.json` (bbc8a8e2ef0e2dc970952d6b044cc4f9fcde711e50d0cd5f826181fb425a547b, 830,475,264 B).

Receipt `artifacts/apa_eb4/p0/result.json`: PASS, all 48 file hashes re-verified. Descriptive: the 32K captures' first
8,192 K/V rows are not bitwise equal to the SPQ3 8K captures beyond 1.7B L0. Their relative Frobenius difference is
0.3–1.5%, because the 32K forward runs different GEMM and flash schedules (reasoning).

## P1: offline quality gate (CPU, fp64, E2a arithmetic)

Method: path-only copies of the E2a chain (`artifacts/apa_eb4/p1/path_copies.diff`). The trimmed `heads_rows` (E1b
`heads`, 64-row window as a parameter) reproduces E2a's stored arrays bitwise on all 12 8K cells (mass, positions,
counts, d_error, tk8_error, tk8 flags, tk8_bits_union, d_bits_union). Rows:
- eval = S−64..S−1;
- calibration = S−128..S−65;
- T-stale's row p uses row p−1. Its newest-block rule fires on the first eval row (8128 = 127·64; 32704 = 511·64).

Band (E2a): mean ≤ 1.01× SP and p99 ≤ 1.05× SP. Receipts: `artifacts/apa_eb4/p1/cells/<cell>/`,
`p1/summary.json`, `p1/table.md`.

### Qualification

| Arm | 8K non-early qualifying | QUALIFIED (≥7/9) | 32K non-early qualifying | LONG-qualified (≥7/9) | built |
|---|---|---|---|---|---|
| T-fresh | 9/9 | QUALIFIED | 9/9 | yes | yes (control, built whatever it scores) |
| T-stale | 1/9 | not qualified | 8/9 | yes | no (not QUALIFIED; kernel work goes to QUALIFIED arms only) |
| T-thr | 4/9 | not qualified | 7/9 | yes | no (not QUALIFIED; kernel work goes to QUALIFIED arms only) |

### θ_L (frozen; `artifacts/apa_eb4/p1/theta.json`, SHA-256 `c8cb8dcea28169f2a74301b10a5074059a4f9f0608203f940757ce3d1eeb054f`)

Grid: 121 points, `10**linspace(-6, 0, 121)`. θ_L is the largest grid θ meeting the band on the 8K calibration rows,
applied unchanged at 32K. All 12 layers calibrated (no fallback).

| Layer | θ_L | calibrated | calibration rows mean/SP | p99/SP | bf16-block fraction (calibration) |
|---|---|---|---|---|---|
| Qwen3-1.7B L00 | 0.199526 | yes | 1.0098 | 1.0000 | 0.0112 |
| Qwen3-1.7B L05 | 0.177828 | yes | 1.0092 | 1.0000 | 0.0149 |
| Qwen3-1.7B L11 | 0.0112202 | yes | 1.0094 | 1.0012 | 0.0979 |
| Qwen3-1.7B L16 | 0.0199526 | yes | 1.0099 | 1.0033 | 0.0541 |
| Qwen3-1.7B L22 | 0.0398107 | yes | 1.0097 | 1.0035 | 0.0240 |
| Qwen3-1.7B L27 | 0.125893 | yes | 1.0091 | 0.9971 | 0.0137 |
| Qwen3-4B L00 | 0.01 | yes | 1.0100 | 1.0007 | 0.2225 |
| Qwen3-4B L07 | 0.0446684 | yes | 1.0098 | 0.9998 | 0.0247 |
| Qwen3-4B L14 | 0.00794328 | yes | 1.0096 | 1.0001 | 0.1395 |
| Qwen3-4B L21 | 0.0141254 | yes | 1.0095 | 0.9983 | 0.1193 |
| Qwen3-4B L28 | 0.0398107 | yes | 1.0093 | 1.0018 | 0.0438 |
| Qwen3-4B L35 | 0.177828 | yes | 1.0085 | 1.0002 | 0.0151 |

The calibration mean ratio sits at 1.0085–1.0100 on every layer: the rule picks the largest θ in the band, so θ_L lies
on the band edge by construction. On fresh evaluation rows, 5 of 9 non-early cells land just over it (1.0100–1.0153).
That is why T-thr is not QUALIFIED at 8K.

### Per-cell table (eval rows; ratios to SP's own error; Q = qualifies)

| Cell | S | f_L | θ_L | SP mean | SP p99 | T-fresh mean/p99 ×SP | T-fresh Q | T-stale mean/p99 ×SP | T-stale Q | T-thr mean/p99 ×SP | T-thr Q | bf16 fraction fresh/stale/thr | V bits/elem fresh/stale/thr | stale=fresh flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B L0 (early) | 8192 | 0.01 | 0.1995 | 0.052032 | 0.72597 | 1.0087/0.9999 | yes | 1.0102/1.0000 | no | 1.0100/0.9999 | no | 0.0156/0.0157/0.0113 | 8.349/8.349/8.316 | 0.9872 |
| Qwen3-1.7B L5 (early) | 8192 | 0.02 | 0.1778 | 0.021062 | 0.13016 | 1.0006/1.0000 | yes | 1.0016/1.0000 | yes | 1.0131/1.0003 | no | 0.0234/0.0236/0.0142 | 8.404/8.404/8.337 | 0.9929 |
| Qwen3-1.7B L11 | 8192 | 0.25 | 0.01122 | 0.0026126 | 0.022024 | 1.0045/1.0006 | yes | 1.0290/1.0018 | no | 1.0100/1.0046 | no | 0.2500/0.2501/0.1086 | 10.165/10.166/9.065 | 0.8603 |
| Qwen3-1.7B L16 | 8192 | 0.1 | 0.01995 | 0.0034164 | 0.023428 | 1.0071/0.9985 | yes | 1.0393/0.9926 | no | 1.0114/0.9984 | no | 0.1016/0.1017/0.0578 | 9.013/9.014/8.674 | 0.9411 |
| Qwen3-1.7B L22 | 8192 | 0.05 | 0.03981 | 0.0066488 | 0.024321 | 1.0072/0.9982 | yes | 1.0115/0.9988 | no | 1.0111/0.9984 | no | 0.0547/0.0548/0.0241 | 8.649/8.650/8.414 | 0.9700 |
| Qwen3-1.7B L27 | 8192 | 0.01 | 0.1259 | 0.008291 | 0.033157 | 1.0052/0.9999 | yes | 1.0069/0.9999 | yes | 1.0083/1.0044 | yes | 0.0156/0.0157/0.0135 | 8.346/8.347/8.337 | 0.9990 |
| Qwen3-4B L0 (early) | 8192 | 0.25 | 0.01 | 0.015279 | 0.17786 | 1.0076/0.9998 | yes | 1.0087/0.9999 | yes | 1.0075/0.9991 | yes | 0.2500/0.2501/0.2146 | 10.165/10.166/9.890 | 0.8722 |
| Qwen3-4B L7 | 8192 | 0.05 | 0.04467 | 0.0070847 | 0.052178 | 1.0046/0.9984 | yes | 1.0203/0.9987 | no | 1.0101/0.9988 | no | 0.0547/0.0548/0.0236 | 8.646/8.647/8.407 | 0.9599 |
| Qwen3-4B L14 | 8192 | 0.25 | 0.007943 | 0.0022747 | 0.016683 | 1.0053/1.0013 | yes | 1.0170/0.9975 | no | 1.0070/0.9981 | yes | 0.2500/0.2501/0.1684 | 10.165/10.166/9.530 | 0.9051 |
| Qwen3-4B L21 | 8192 | 0.25 | 0.01413 | 0.0018406 | 0.014885 | 1.0043/1.0000 | yes | 1.0352/1.0031 | no | 1.0153/0.9997 | no | 0.2500/0.2501/0.1237 | 10.165/10.166/9.183 | 0.8898 |
| Qwen3-4B L28 | 8192 | 0.05 | 0.03981 | 0.0050911 | 0.021327 | 1.0067/1.0012 | yes | 1.0167/1.0004 | no | 1.0086/1.0060 | yes | 0.0547/0.0548/0.0394 | 8.646/8.647/8.529 | 0.9824 |
| Qwen3-4B L35 | 8192 | 0.01 | 0.1778 | 0.0089134 | 0.040823 | 1.0075/0.9999 | yes | 1.0101/0.9999 | no | 1.0063/0.9999 | yes | 0.0156/0.0157/0.0149 | 8.345/8.346/8.342 | 0.9988 |
| Qwen3-1.7B L0 (early) | 32768 | 0.01 | 0.1995 | 0.068899 | 0.99585 | 1.0031/0.9998 | yes | 1.0041/0.9998 | yes | 1.0076/0.9998 | yes | 0.0117/0.0117/0.0031 | 8.334/8.334/8.269 | 0.9911 |
| Qwen3-1.7B L5 (early) | 32768 | 0.02 | 0.1778 | 0.020969 | 0.10765 | 1.0000/1.0000 | yes | 1.0000/0.9999 | yes | 1.0075/1.0008 | yes | 0.0215/0.0215/0.0037 | 8.409/8.409/8.273 | 0.9914 |
| Qwen3-1.7B L11 | 32768 | 0.25 | 0.01122 | 0.0032101 | 0.022158 | 0.9999/1.0000 | yes | 1.0083/0.9994 | yes | 1.0033/1.0007 | yes | 0.2500/0.2500/0.0181 | 10.182/10.182/8.383 | 0.8525 |
| Qwen3-1.7B L16 | 32768 | 0.1 | 0.01995 | 0.0043972 | 0.027494 | 1.0001/1.0001 | yes | 1.0101/1.0010 | no | 1.0054/0.9986 | yes | 0.1016/0.1016/0.0115 | 9.031/9.031/8.333 | 0.9331 |
| Qwen3-1.7B L22 | 32768 | 0.05 | 0.03981 | 0.009006 | 0.02814 | 1.0006/0.9996 | yes | 1.0015/0.9996 | yes | 1.0047/1.0009 | yes | 0.0508/0.0508/0.0057 | 8.637/8.637/8.289 | 0.9741 |
| Qwen3-1.7B L27 | 32768 | 0.01 | 0.1259 | 0.010868 | 0.03861 | 1.0003/1.0007 | yes | 1.0005/1.0007 | yes | 1.0053/1.0012 | yes | 0.0117/0.0117/0.0034 | 8.333/8.334/8.272 | 0.9947 |
| Qwen3-4B L0 (early) | 32768 | 0.25 | 0.01 | 0.020461 | 0.20306 | 1.0013/1.0000 | yes | 1.0016/1.0000 | yes | 1.0079/1.0005 | yes | 0.2500/0.2500/0.0446 | 10.182/10.182/8.589 | 0.9300 |
| Qwen3-4B L7 | 32768 | 0.05 | 0.04467 | 0.010321 | 0.082085 | 1.0002/0.9999 | yes | 1.0070/1.0001 | yes | 1.0055/0.9996 | yes | 0.0508/0.0508/0.0056 | 8.636/8.637/8.287 | 0.9543 |
| Qwen3-4B L14 | 32768 | 0.25 | 0.007943 | 0.0023119 | 0.013962 | 1.0001/1.0000 | yes | 1.0005/1.0000 | yes | 1.0048/0.9998 | yes | 0.2500/0.2500/0.0238 | 10.182/10.182/8.427 | 0.8990 |
| Qwen3-4B L21 | 32768 | 0.25 | 0.01413 | 0.0014541 | 0.0094175 | 1.0003/1.0000 | yes | 1.0020/1.0005 | yes | 1.0227/1.0025 | no | 0.2500/0.2500/0.0303 | 10.182/10.182/8.478 | 0.8673 |
| Qwen3-4B L28 | 32768 | 0.05 | 0.03981 | 0.0028875 | 0.012882 | 1.0017/1.0010 | yes | 1.0094/1.0021 | yes | 1.0294/1.0069 | no | 0.0508/0.0508/0.0193 | 8.637/8.637/8.393 | 0.9822 |
| Qwen3-4B L35 | 32768 | 0.01 | 0.1778 | 0.013638 | 0.093035 | 1.0008/1.0001 | yes | 1.0015/1.0001 | yes | 1.0054/1.0003 | yes | 0.0117/0.0117/0.0037 | 8.334/8.334/8.273 | 0.9957 |

T-stale: flags agree with T-fresh on 0.9465 of 8K blocks pooled
(cell minimum 0.8603, 1.7B L11, f 0.25). Check: T-stale's row-p flags equal T-fresh's row-(p−1) flags on
every cell (rows 1..63), and the newest block reads bf16 on row 0. V bits per key element: 16 for bf16, 8.25 for
int8 + an fp32 scale per token (E2a `value_costs`).

## G0–G5

- **G0 PASS** (`artifacts/apa_eb4/g0/result.json`): main build `8f180672…e332`; Qwen3-1.7B L11 at 8K, 64 rows end
  to end. Results: T-fresh production and poisoned-diagnostic outputs finite and equal; forced bf16 equal to EB3;
  decisions 0/65,536 off P1.
- **G1 PASS** (`artifacts/apa_eb4/g1/result.json`, per cell `g1/<cell>/`):
  1. Forced bf16 vs EB3's binary (3fa9c21d, own process; dumps `g1/eb3_binary/`): 24 real cells (12 at 8K, 12 at
     32K) × 64 rows × sinks {off, on} = 3,072 row-configurations, **0 mismatches**. Also 0 vs in-process EB3.
  2. T-fresh vs EB2b's binary tk8 (7261c99d, own process; dumps `g1/eb2b_binary/`): 12 real 8K cells × 64 rows,
     **0 mismatches** on the output and on every decision intermediate EB2b exposes (flags, scores, logz, cz, blog, cm).
  3. Offline flags injected vs in-kernel flags: 12,288 (row, KV group) pairs. 12,287 agree, all bitwise equal; 1 group disagrees (the
     G2 near-tie).
  4. Auxiliary: production == poisoned diagnostic; loader flags == choose flags; GPU quantizer == `quant_ref` on every
     cell (and EB2b's binary quantizer too).
- **G2 PASS** (`artifacts/apa_eb4/g2/result.json`):
  - Decision parity: **2 / 3,932,160** decisions (8K 0/786,432; 32K 2/3,145,728), rate 5.09e-07 ≤ 1e-4. Both
    mismatches are one fp32 exact tie at the cutoff: 4B L14 32K, row 32748, KV head 7, blocks 91 and 374.
    - fp32 scores: both 8.444292e-05.
    - fp64 scores: 8.444273e-05 and 8.444359e-05; relative distance from the fp64 cutoff 1.0e-5 and 0.
    - Witnessed as near-ties under the registered rule (≤ 1e-4).
    - fp32-vs-fp64 decision-score gap, scores > 1e-6: median 8.7e-6, p99 6.9e-5, p99.9 2.3e-4. 1,433 of 3.33M blocks
      exceed 1e-3 (max 0.22). A diagnostic found 1 of 8,388,608 TQ4 coordinate codes differing between the GPU packer
      and the CPU reconstruction per 8K cell (reasoning for the tail).
  - No regression with TC_APA_EB4 off (`build/noeb4`):
    - legacy hashes: 72 r2 (= a6752ea), 32 INT4-tc (= SPQ6 build), 136/136 SPQ6, 48 capture files;
    - EB3's G1 driver (path-only copy) on its 12 real + 10 synthetic cells: PASS, 10,400 cross-binary digests equal. Its
      `outs` comparison is vacuous, see the finding above;
    - supplement g2x: 6,144 bitwise output comparisons of the noeb4 default variant-B path and of `apa_eb3`
      against EB3's binary, 0 mismatches;
    - SASS: all 705 kernels of EB3's object are identical in noeb4. In main, all 705 are identical (the 5 default and
      the 9 registered variant-0 kernels included) and 9 new eb4 kernels are added.
- **G3 PASS** (`artifacts/apa_eb4/g3/result.json`): 10 compile-time mutants (`-DAPA_EB4_MUTANT=k`, separate builds,
  sources never edited). Kill = any G1.1/G1.2/G1.3/G2 failure on 1.7B L11 and 4B L14 at 8K and 1.7B L11 at 32K.
  Unmutated baseline not killed. **9/10 killed** (≥ 6 required).

| # | Mutant | Result | Kill evidence |
|---|---|---|---|
| 1 | V loader ignores flags (all bf16) | **killed** | Qwen3-1.7B_L11_S8192: G1.2_out 64, used_equals_choose 64; G2 mismatches 49152; Qwen3-4B_L14_S8192: G1.2_out 64, used_equals_choose 64; G2 mismatches 49152; Qwen3-1.7B_L11_S32768: used_equals_choose 64; G2 mismatches 196608 |
| 2 | V loader ignores flags (all int8) | **killed** | Qwen3-1.7B_L11_S8192: G1.1_forced_vs_eb3_sinks_off 64, G1.1_forced_vs_eb3_sinks_on 64, G1.1_forced_vs_inprocess_eb3 64, G1.2_out 64, used_equals_choose 64; G2 mismatches 16384; Qwen3-4B_L14_S8192: G1.1_forced_vs_eb3_sinks_off 64, G1.1_forced_vs_eb3_sinks_on 64, G1.1_forced_vs_inprocess_eb3 64, G1.2_out 64, used_equals_choose 64; G2 mismatches 16384; Qwen3-1.7B_L11_S32768: G1.1_forced_vs_eb3_sinks_off 64, G1.1_forced_vs_eb3_sinks_on 64, used_equals_choose 64; G2 mismatches 65536 |
| 3 | int8 scale dropped | **killed** | Qwen3-1.7B_L11_S8192: G1.2_out 64; G2 mismatches 0; Qwen3-4B_L14_S8192: G1.2_out 64; G2 mismatches 0; Qwen3-1.7B_L11_S32768: no check failed; G2 mismatches 0 |
| 4 | int8 scale taken from the neighbouring token | **killed** | Qwen3-1.7B_L11_S8192: G1.2_out 64; G2 mismatches 0; Qwen3-4B_L14_S8192: G1.2_out 64; G2 mismatches 0; Qwen3-1.7B_L11_S32768: no check failed; G2 mismatches 0 |
| 5 | tk8 cutoff off by one (n-1) | **killed** | Qwen3-1.7B_L11_S8192: G1.2_out 64, G1.2_flags 64; G2 mismatches 512; Qwen3-4B_L14_S8192: G1.2_out 64, G1.2_flags 64; G2 mismatches 512; Qwen3-1.7B_L11_S32768: no check failed; G2 mismatches 512 |
| 6 | GQA score from head 0 only | **killed** | Qwen3-1.7B_L11_S8192: G1.2_out 64, G1.2_flags 64, G1.2_scores 64; G2 mismatches 5728; Qwen3-4B_L14_S8192: G1.2_out 64, G1.2_flags 64, G1.2_scores 64; G2 mismatches 5838; Qwen3-1.7B_L11_S32768: no check failed; G2 mismatches 18046 |
| 7 | T-fresh's tie order reversed | **survived** | Qwen3-1.7B_L11_S8192: no check failed; G2 mismatches 0; Qwen3-4B_L14_S8192: no check failed; G2 mismatches 0; Qwen3-1.7B_L11_S32768: no check failed; G2 mismatches 0 |
| 8 | lz chain order reversed within each 32-chunk group | **killed** | Qwen3-1.7B_L11_S8192: G1.2_scores 34, G1.2_logz 34; G2 mismatches 0; Qwen3-4B_L14_S8192: G1.2_scores 62, G1.2_logz 62; G2 mismatches 0; Qwen3-1.7B_L11_S32768: no check failed; G2 mismatches 0 |
| 9 | loader block boundary off by one | **killed** | Qwen3-1.7B_L11_S8192: G1.2_out 64; G2 mismatches 0; Qwen3-4B_L14_S8192: G1.2_out 64; G2 mismatches 0; Qwen3-1.7B_L11_S32768: no check failed; G2 mismatches 0 |
| 10 | statistics sum only half of each block | **killed** | Qwen3-1.7B_L11_S8192: G1.2_out 64, G1.2_flags 64, G1.2_scores 64, G1.2_logz 64, G1.2_cz 64, G1.2_blog 64; G2 mismatches 7666; Qwen3-4B_L14_S8192: G1.2_out 64, G1.2_flags 64, G1.2_scores 64, G1.2_logz 64, G1.2_cz 64, G1.2_blog 64; G2 mismatches 5070; Qwen3-1.7B_L11_S32768: no check failed; G2 mismatches 26864 |

  - Mutant 7 **survived**: no exact fp32 tie at the cutoff occurs in the kill cells. In the one tie G2 found, the
    reversed order would agree with the fp64 reference. This is a named test-strength residual.
  - Mutant 8 is **weak**: only G1.2's bitwise logz/scores catch it (34/64 and 62/64 rows); outputs, flags and
    decisions are unchanged.
  - Mutants 3, 4 and 9 are caught only by G1.2's output at 8K. At 32K there is no EB2b reference, and G1.3 shares the
    loader.
- **G4 PASS** (measurement gate; `artifacts/apa_eb4/g4/result.json`): all 24 real and 4 synthetic cells valid. Speed
  tables below.
- **G5 PASS** (`artifacts/apa_eb4/g5/result.json`): footprint below.

## Predictions (registered; full text)

| ID | Result | Registered prediction (full text) | Evidence |
|---|---|---|---|
| LD1 | MISS | T-stale's flags agree with T-fresh's on at least 98% of blocks at 8K, and T-stale qualifies on at least 8 of 9 non-early 8K cells. | stale/fresh flag agreement pooled 8K 0.9465 (min cell 0.8603); T-stale qualifies 1/9 |
| LD2 | MISS | T-thr qualifies on at least 7 of 9 non-early 8K cells, and its mean bf16-block fraction is within 1.5× of tk8's f_L on at least 6 of 9. | T-thr qualifies 4/9 at 8K; bf16 fraction within [f/1.5, 1.5f] on 4/9 (<= 1.5f on 9/9) |
| LD3 | HIT | applied unchanged at 32K, the 8K f_L (T-fresh) and θ_L (T-thr) still qualify on at least 6 of 9 non-early 32K cells each. | 32K: T-fresh 9/9, T-thr 7/9 |
| LD4 | MISS | on EB3, T-fresh is still no faster than EB3: arm / EB3 ≥ 0.97 at 8K on both models. The pre-pass eats the V saving. | 8K non-early median T-fresh/EB3: 1.7B 0.8562, 4B 0.8769 |
| LD5 | MISS | T-thr is the fastest built arm at 8K on both models. Its verdict is PARTIAL, not WIN: arm / EB3 between 0.88 and 0.96 on both models. | T-thr is not a built arm (not QUALIFIED in P1: 4/9 non-early 8K cells), so it is not the fastest built arm; no T-thr speed exists |
| LD6 | NOT TESTED | T-thr's gain over EB3 is larger at 32K than at 8K on both models, because more of the time is V bytes. | T-thr not built (P1 rail); counted as not hit. Descriptive: T-fresh's gain over EB3 is larger at 32K than at 8K on both models (1.7B 0.856 -> 0.767; 4B 0.877 -> 0.784) |
| LD7 | HIT | EB3 itself is faster than flash on real 32K cells, non-early median x-flash ≥ 1.0 on both models. The real refine fraction stays near 10%, unlike the synthetic 32K cells. | real 32K non-early median EB3 x-flash: 1.7B 1.391, 4B 1.304. Mechanism note: per-head refined fraction at 32K non-early 0.0053-0.0506, below the stated ~10% |
| S1 | HIT | Real 8K non-early median T-fresh / EB3 lies in [0.84, 0.92] for Qwen3-1.7B. | 0.8562 |
| S2 | HIT | Real 8K non-early median T-fresh / EB3 lies in [0.87, 0.93] for Qwen3-4B. | 0.8769 |
| S3 | HIT | T-fresh's registered 8K speed verdict is WIN or PARTIAL (not NONE). | WIN |
| S4 | HIT | Real 32K non-early median T-fresh / EB3 is <= 0.90 on both models (LONG verdict WIN; T-fresh qualifies 9/9 at 32K in P1). | 0.7665 / 0.7838; LONG verdict WIN |
| S5 | HIT | EB3 is faster than flash on real 32K cells: non-early median EB3 / flash <= 0.80 on both models (x-flash >= 1.25). | EB3/flash 0.7189 / 0.7666 |
| S6 | HIT | Real 32K non-early median T-fresh / flash <= 0.70 on both models (x-flash >= 1.43). | T-fresh/flash 0.5497 / 0.5989 |
| S7 | HIT | In the cold CUPTI profile at 8K (1.7B L11, 4B L14), T-fresh's bulk kernel (bulk_stats) is faster than EB3's bulk_swap, and T-fresh's select kernel is slower than EB3's select_pv. | Qwen3-1.7B_L11_S8192: bulk 18.45 vs 23.42, select 38.37 vs 37.62 us; Qwen3-4B_L14_S8192: bulk 20.16 vs 23.49, select 41.06 vs 40.42 us |
| S8 | HIT | G5: 3 launches per decode call (equal to EB3); per-call device bytes exceed EB3's by less than 2% on every G5 cell; stored bytes 712 per key per KV head (tk8, unchanged from EB2b). | launches 3,3,3,3,3,3; extra bytes/call max 1.22%; stored 712 B (tk8) |
| S9 | HIT | Synthetic 32K (H16/KV8 and H32/KV8, refine fraction about 1): T-fresh at f = 0.05 / EB3 <= 0.88 on both shapes. | 0.8519 / 0.8626 |

## Speed

Cold L2, RTX 4070 SUPER 12 GB, B = 1, D = VD = 128, C = 256, δ 8.1 SP row carry, f = registered f_L. µs are medians of
7 sample medians × 50 calls; the CUDA-event tick is about 1.02 µs. The EB3 arm is variant 0 inside the EB4 binary
(SASS-identical to EB3's binary). It reads 65.54 / 70.66 µs at 1.7B L11 / 4B L14 against EB3's own G4 66.56 / 71.68,
one tick apart. x-EB3 = EB3/T-fresh, x-flash = flash/T-fresh; the last data column is EB3's own x-flash. Early layers
are excluded from the medians.

| Cell class | S | f_L | bf16 blocks | refined | T-fresh µs | EB3 µs | flash µs | T-fresh/EB3 | x-EB3 | T-fresh/flash | x-flash | EB3 x-flash | L2 | Bulk tier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B L0 (early, excluded) | 8192 | 0.01 | 0.0156 | 0.4693 | 73.568 | 81.920 | 81.920 | 0.8980 | 1.114 | 0.8980 | 1.114 | 1.000 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-1.7B L5 (early, excluded) | 8192 | 0.02 | 0.0234 | 0.0420 | 55.296 | 64.320 | 81.920 | 0.8597 | 1.163 | 0.6750 | 1.481 | 1.274 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-1.7B L11 | 8192 | 0.25 | 0.2500 | 0.1315 | 60.416 | 65.536 | 81.920 | 0.9219 | 1.085 | 0.7375 | 1.356 | 1.250 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-1.7B L16 | 8192 | 0.1 | 0.1016 | 0.1071 | 56.320 | 65.248 | 81.920 | 0.8632 | 1.159 | 0.6875 | 1.455 | 1.256 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-1.7B L22 | 8192 | 0.05 | 0.0547 | 0.0211 | 54.272 | 63.904 | 81.920 | 0.8493 | 1.177 | 0.6625 | 1.509 | 1.282 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-1.7B L27 | 8192 | 0.01 | 0.0156 | 0.0214 | 53.248 | 63.824 | 81.920 | 0.8343 | 1.199 | 0.6500 | 1.538 | 1.284 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-4B L0 (early, excluded) | 8192 | 0.25 | 0.2500 | 0.4871 | 86.016 | 90.112 | 81.920 | 0.9545 | 1.048 | 1.0500 | 0.952 | 0.909 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-4B L7 | 8192 | 0.05 | 0.0547 | 0.0449 | 58.368 | 66.560 | 81.920 | 0.8769 | 1.140 | 0.7125 | 1.404 | 1.231 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-4B L14 | 8192 | 0.25 | 0.2500 | 0.1058 | 65.536 | 70.656 | 81.920 | 0.9275 | 1.078 | 0.8000 | 1.250 | 1.159 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-4B L21 | 8192 | 0.25 | 0.2500 | 0.0758 | 63.488 | 68.608 | 81.920 | 0.9254 | 1.081 | 0.7750 | 1.290 | 1.194 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-4B L28 | 8192 | 0.05 | 0.0547 | 0.0313 | 56.624 | 65.536 | 81.824 | 0.8640 | 1.157 | 0.6920 | 1.445 | 1.249 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-4B L35 | 8192 | 0.01 | 0.0156 | 0.0438 | 58.240 | 67.584 | 81.920 | 0.8617 | 1.160 | 0.7109 | 1.407 | 1.212 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| **Qwen3-1.7B non-early median (4 layers)** | 8192 | per layer | | | 55.296 | 64.576 | 81.920 | **0.8562** | **1.168** | **0.6750** | **1.481** | 1.269 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| **Qwen3-4B non-early median (5 layers)** | 8192 | per layer | | | 58.368 | 67.584 | 81.920 | **0.8769** | **1.140** | **0.7125** | **1.404** | 1.212 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-1.7B L0 (early, excluded) | 32768 | 0.01 | 0.0117 | 0.3444 | 217.600 | 270.336 | 296.960 | 0.8049 | 1.242 | 0.7328 | 1.365 | 1.098 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-1.7B L5 (early, excluded) | 32768 | 0.02 | 0.0215 | 0.0153 | 162.816 | 212.992 | 296.960 | 0.7644 | 1.308 | 0.5483 | 1.824 | 1.394 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-1.7B L11 | 32768 | 0.25 | 0.2500 | 0.0164 | 173.056 | 217.088 | 296.960 | 0.7972 | 1.254 | 0.5828 | 1.716 | 1.368 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-1.7B L16 | 32768 | 0.1 | 0.1016 | 0.0176 | 162.656 | 214.016 | 297.024 | 0.7600 | 1.316 | 0.5476 | 1.826 | 1.388 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-1.7B L22 | 32768 | 0.05 | 0.0508 | 0.0053 | 161.792 | 212.992 | 296.960 | 0.7596 | 1.316 | 0.5448 | 1.835 | 1.394 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-1.7B L27 | 32768 | 0.01 | 0.0117 | 0.0064 | 163.840 | 211.952 | 296.960 | 0.7730 | 1.294 | 0.5517 | 1.813 | 1.401 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-4B L0 (early, excluded) | 32768 | 0.25 | 0.2500 | 0.3377 | 269.312 | 312.832 | 297.568 | 0.8609 | 1.162 | 0.9050 | 1.105 | 0.951 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-4B L7 | 32768 | 0.05 | 0.0508 | 0.0149 | 178.176 | 228.352 | 297.872 | 0.7803 | 1.282 | 0.5982 | 1.672 | 1.304 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-4B L14 | 32768 | 0.25 | 0.2500 | 0.0215 | 182.624 | 229.856 | 297.744 | 0.7945 | 1.259 | 0.6134 | 1.630 | 1.295 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-4B L21 | 32768 | 0.25 | 0.2500 | 0.0506 | 190.464 | 237.568 | 297.216 | 0.8017 | 1.247 | 0.6408 | 1.560 | 1.251 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-4B L28 | 32768 | 0.05 | 0.0508 | 0.0170 | 178.176 | 227.328 | 297.520 | 0.7838 | 1.276 | 0.5989 | 1.670 | 1.309 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| Qwen3-4B L35 | 32768 | 0.01 | 0.0117 | 0.0066 | 174.080 | 224.256 | 297.792 | 0.7763 | 1.288 | 0.5846 | 1.711 | 1.328 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| **Qwen3-1.7B non-early median (4 layers)** | 32768 | per layer | | | 163.248 | 213.504 | 296.960 | **0.7665** | **1.305** | **0.5497** | **1.819** | 1.391 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| **Qwen3-4B non-early median (5 layers)** | 32768 | per layer | | | 178.176 | 228.352 | 297.744 | **0.7838** | **1.276** | **0.5989** | **1.670** | 1.304 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |

Synthetic cells (descriptive only; refine fraction ≈ 1, SP's worst case; one seeded query; no θ sweep because T-thr is
not built):

| Synthetic cell (refine ~1) | EB3 µs | flash µs | f | bf16 blocks | T-fresh µs | T-fresh/EB3 | x-flash | L2 | Bulk tier |
|---|---|---|---|---|---|---|---|---|---|
| syn_H16_KV8_S8192 | 95.008 | 81.920 | 0.01 | 0.0156 | 86.800 | 0.9136 | 0.944 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| syn_H16_KV8_S8192 | 95.008 | 81.920 | 0.05 | 0.0547 | 86.016 | 0.9054 | 0.952 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| syn_H16_KV8_S8192 | 95.008 | 81.920 | 0.25 | 0.2500 | 90.112 | 0.9485 | 0.909 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| syn_H16_KV8_S32768 | 359.424 | 296.960 | 0.01 | 0.0117 | 304.128 | 0.8462 | 0.976 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| syn_H16_KV8_S32768 | 359.424 | 296.960 | 0.05 | 0.0508 | 306.176 | 0.8519 | 0.970 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| syn_H16_KV8_S32768 | 359.424 | 296.960 | 0.25 | 0.2500 | 317.936 | 0.8846 | 0.934 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| syn_H32_KV8_S8192 | 96.256 | 81.920 | 0.01 | 0.0156 | 88.064 | 0.9149 | 0.930 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| syn_H32_KV8_S8192 | 96.256 | 81.920 | 0.05 | 0.0547 | 88.064 | 0.9149 | 0.930 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| syn_H32_KV8_S8192 | 96.256 | 81.920 | 0.25 | 0.2500 | 91.136 | 0.9468 | 0.899 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| syn_H32_KV8_S32768 | 372.736 | 296.960 | 0.01 | 0.0117 | 319.488 | 0.8571 | 0.929 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| syn_H32_KV8_S32768 | 372.736 | 296.960 | 0.05 | 0.0508 | 321.536 | 0.8626 | 0.924 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |
| syn_H32_KV8_S32768 | 372.736 | 296.960 | 0.25 | 0.2500 | 333.824 | 0.8956 | 0.890 | cold | TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate |

## Profile (CUPTI per-kernel medians, µs, 20 repetitions; "before" = EB3, "after" = T-fresh)

| Cell | Stage | EB3 cold | T-fresh cold | EB3 warm | T-fresh warm | flash cold |
|---|---|---|---|---|---|---|
| Qwen3-1.7B_L11_S32768 | query rotation | 6.752 | 6.721 | 8.480 | 8.800 |  |
| Qwen3-1.7B_L11_S32768 | bulk (+statistics) | 39.088 | 38.816 | 38.608 | 38.752 |  |
| Qwen3-1.7B_L11_S32768 | select + exact K + V + PV + fused merge | 167.870 | 125.857 | 171.594 | 126.930 |  |
| Qwen3-1.7B_L11_S32768 | sum of kernel medians | 213.710 | 171.393 | 218.681 | 174.481 | 297.373 |
| Qwen3-1.7B_L11_S8192 | query rotation | 4.128 | 4.064 | 2.847 | 2.944 |  |
| Qwen3-1.7B_L11_S8192 | bulk (+statistics) | 23.424 | 18.448 | 12.960 | 12.688 |  |
| Qwen3-1.7B_L11_S8192 | select + exact K + V + PV + fused merge | 37.615 | 38.367 | 21.151 | 24.560 |  |
| Qwen3-1.7B_L11_S8192 | sum of kernel medians | 65.167 | 60.880 | 36.959 | 40.193 | 80.734 |
| Qwen3-4B_L14_S32768 | query rotation | 7.136 | 7.040 | 8.544 | 8.960 |  |
| Qwen3-4B_L14_S32768 | bulk (+statistics) | 38.736 | 42.303 | 39.521 | 42.639 |  |
| Qwen3-4B_L14_S32768 | select + exact K + V + PV + fused merge | 183.150 | 139.327 | 187.682 | 143.231 |  |
| Qwen3-4B_L14_S32768 | sum of kernel medians | 229.023 | 188.671 | 235.747 | 194.830 | 297.729 |
| Qwen3-4B_L14_S8192 | query rotation | 4.528 | 4.480 | 3.264 | 3.297 |  |
| Qwen3-4B_L14_S8192 | bulk (+statistics) | 23.488 | 20.160 | 13.280 | 14.176 |  |
| Qwen3-4B_L14_S8192 | select + exact K + V + PV + fused merge | 40.416 | 41.055 | 22.080 | 25.776 |  |
| Qwen3-4B_L14_S8192 | sum of kernel medians | 68.432 | 65.695 | 38.624 | 43.249 | 80.559 |

Reading (reasoning from the table):
- At 8K, T-fresh's bulk kernel is faster than EB3's: it prefetches about half the V bytes (int8 + scales), and the
  statistics epilogue costs little.
- Its select kernel is slightly slower cold (+0.6–0.8 µs) and 3.4–3.7 µs slower warm: the in-CTA choose plus the int8
  dequant.
- The cold sum still drops 2.7–4.3 µs because fewer V bytes cross DRAM.
- At 32K, select drops by 42–44 µs (−24 to −25%), where V bytes dominate.

## Footprint (G5)

| Cell | Launches EB3 → EB4 | Bytes/call EB3 | Bytes/call EB4 | Extra | Persistent tickets (merge shared / choose) | Quant per appended token µs (8 KV) | Quant per 8192-token prefill µs | Append / decode |
|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B_L11_S32768 | 3 → 3 | 3,706,880 | 3,751,936 | 45,056 (1.22%) | 4,096 / 0 | 3.49 | 64.51 | 0.0199 |
| Qwen3-1.7B_L11_S8192 | 3 → 3 | 935,936 | 947,200 | 11,264 (1.20%) | 4,096 / 0 | 4.05 | 64.51 | 0.0670 |
| Qwen3-4B_L14_S32768 | 3 → 3 | 7,413,760 | 7,499,776 | 86,016 (1.16%) | 4,096 / 0 | 3.42 | 64.51 | 0.0187 |
| Qwen3-4B_L14_S8192 | 3 → 3 | 1,871,872 | 1,893,376 | 21,504 (1.15%) | 4,096 / 0 | 3.81 | 64.51 | 0.0572 |
| syn_H16_KV8_S32768 | 3 → 3 | 3,706,880 | 3,751,936 | 45,056 (1.22%) | 4,096 / 0 | 3.07 | 64.51 | 0.0100 |
| syn_H32_KV8_S32768 | 3 → 3 | 7,413,760 | 7,499,776 | 86,016 (1.16%) | 4,096 / 0 | 3.94 | 64.51 | 0.0122 |

- **Stored bytes per key per KV head:**
  - flash 512 (bf16 K 256 + bf16 V 256);
  - EB3 580 (codes 64 + K norm 4 + bf16 K 256 + bf16 V 256);
  - **T-fresh 712** (+ int8 V 128 + V scale 4), unchanged from EB2b's tk8. Both V copies stay resident.
- **Scratch per call:** EB3's arena plus `cz` (4·rows·NC), `blog` (4·rows·NB) and `flags` (B·KVH·NB): +1.15–1.22%.
- **Launches per call:** 3, as EB3.
- **Persistent state:** no new buffer. The choose tickets are unused at the registered choose site; the merge tickets
  are EB3's 4,096 B.
- **Quantization per appended token:** standalone, EB2b's kernel, 8 KV heads: 3.1–4.1 µs, 1–7% of a decode call.

## Findings and residuals

1. **The P1 rail decided the build.** T-stale fails the mean band on 8/9 non-early 8K cells (up to 1.039× SP), because
   its flags (the previous row's top blocks) disagree with T-fresh's on 5.4% of 8K blocks (14% at 1.7B L11, f 0.25). T-thr fails 5/9 by
   ≤ 0.0053 over the 1.01 band, because θ_L is calibrated to the band edge. Both qualify at 32K (8/9, 7/9); there
   the arms' error ratios to SP sit closer to 1 (observation, not explained here). T-thr also keeps only 0.3–3.0% of blocks bf16 there (non-early). Not built, so not timed.
2. **The decision is now cheap enough on EB3.** T-fresh's exact choose, run redundantly in every select CTA, costs about
   2 µs warm (development profile: forced-bf16 select vs EB3's select). Against EB2b's 8 µs pre-pass, it turns the V-byte saving into real time: 0.856 / 0.877 × EB3 at 8K,
   0.767 / 0.784 at 32K. LD4 ("pre-pass eats the V saving") does not hold on these real cells (MISS).
3. **EB3's receipts carry a vacuous cross-binary output comparison** (see the verdict section). EB4's G1.1 and g2x are
   non-vacuous replacements, and every EB4 dump asserts non-zero content.
4. **Harness defect before registration:** the first G1 batch was RED because EB4's own `dump3` had the same uint16
   truncation. Fixed, failed receipts moved to `scratch_retired/g1_uint16_truncation/`; no kernel change.
5. **LD7 mechanism:** EB3 beats flash at 32K (x-flash 1.39 / 1.30). The real 32K per-head refined fraction is 0.5–5.1%
   on non-early layers, below the "~10%" the prediction states.
6. **Residuals:**
   - mutant 7 survived and mutant 8 is weak (G3);
   - T-fresh at 32K has no bitwise EB2b reference (G1.2 is 8K only, as ordered);
   - the merge and choose tickets keep EB3's single-stream contract (LEAD_NOTES item 1);
   - stored bytes rise 23% over EB3 (712 vs 580);
   - the in-CTA choose is computed redundantly by every CTA. That is cheap at NB ≤ 512 but grows with S, and 64K
     real cells were not measured.

## Prior art

- **tk8 rule:** ThriftAttention-style block promotion (Sharratt 2026, arXiv 2605.23081) and Calver (arXiv 2605.20868)
  per-block INT8→FP16 promotion from mass, inherited via E2a/EB2b. Taken unchanged.
- **Stale selection (T-stale):** the "persistence of importance" observation in Scissorhands (Liu et al. 2023) and
  accumulated past attention scores in H2O (Zhang et al. 2023). Taken as the idea behind the arm, which was not built.
- **Mass threshold vs top-k (T-thr):** top-p attention pruning in Twilight (2025). Unverified, lead to check: "Twilight
  top-p attention pruning 2025". T-thr is a per-block mass threshold, not a cumulative top-p. Not built.
- **Running-max criterion:** BLASST (Yuan et al. 2025, arXiv 2512.12087), inherited in S1.
- **Split-K and merge:** Flash-Decoding (Dao, Haziza, Massa, Sizov 2023), inherited in EB3's fused merge.
- **Last-CTA pattern:** NVIDIA CUDA Samples `threadFenceReduction` (EB3's merge; the unregistered bulk-choose alternative).
- **Per-token int8 V:** KIVI (Liu et al. 2024, arXiv 2402.02750); the EB2b format.
- **Statistics:** online-normalizer partial sums (Milakov & Gimelshein 2018), in EB2b's arithmetic.
- **Sorting and ranking:**
  - Batcher (1968) bitonic sort with an index tie-break: EB2b, and the development alternative here.
  - Rank by comparison counting (Knuth, TAOCP Vol. 3 §5.2, "comparison counting"; Muller & Preparata 1975 parallel
    enumeration sort). Unverified as citations, lead to check: "comparison counting sort Knuth", "Muller Preparata 1975".
    Ours: each CTA ranks only its own 4 blocks, redundantly, instead of a global sort.
- **Prefetch:** software L2 prefetch (Mowry, Lam, Gupta 1992; PTX `prefetch.global.L2`), EB3's helper.
- **TQ4 and the bulk path:** TurboQuant (Zandieh et al. 2025) via SPQ1/SPQ6/EB3, unchanged.
- **Methods:** mutation testing (DeMillo, Lipton, Sayward 1978); SHA-256 manifests (NIST FIPS 180-4).
- **Capture:** the transformers AttentionInterface extension point (Hugging Face).
- **Ours, no prior art known to me:** the assembly (statistics in the swap-AB bulk epilogue, redundant per-CTA exact
  rank in select, flag-driven prefetch), the 121-point θ calibration receipt and the near-tie tolerance. No novelty
  claimed. The code-site annotations are in `eb4.cuh`, `eb4_tuning.cuh`, the EB4 blocks of
  `spq4.cuh`/`bindings.cpp`/`ops.h`/`__init__.py`, and `scripts/apa_eb4_offline.py`; the ledger carries the same.

## Scope

- **Evidence classes:**
  - offline CPU fp64 numerics (P1, E2a arithmetic);
  - author-run CUDA kernel correctness (bitwise reference pairs, same-process and cross-binary);
  - compile-time mutation tests;
  - cold-L2 kernel micro-benchmarks.
- **Platform:** RTX 4070 SUPER 12 GB, native driver 595.99.02, CUDA 12.6.
- **Bulk tier on every APA line:** TQ4, 4.25 bits per element stored, computed bf16 hi/lo with fp32 accumulate.
- **Exact tier:** bf16 K refined by the SP row-carry rule at δ 8.1 with a GQA union gather. V per 64-key block, bf16 or
  int8 (EB2b format, per-token absmax/127, ties-to-even, fp32 scale), by tk8 at EB2b's f_L.
- **Data:**
  - real Qwen3-1.7B (H16/KV8) and Qwen3-4B (H32/KV8), D 128, one wikitext-2 sequence, S 8192 (SPQ3 captures) and
    32768 (EB4 captures);
  - G1/G2 on the last 64 causal-prefix rows; G4 on the last 8 rows with all keys visible;
  - synthetic H16/H32 at 8K and 32K, descriptive.
- **Does not cover:**
  - model-level quality (AB-1);
  - prefill;
  - other GPUs;
  - KV-cache write integration (quantization timed standalone);
  - EB3's ticket-buffer hardening;
  - 64K real cells;
  - T-stale and T-thr kernels (not built, rail);
  - the optional max-logit-gap T-thr variant (not evaluated);
  - any production promotion.
- **Process:** author checks are not a blind review. No git, subagents, deletes, process kills or writes outside the
  order's paths. The only exceptions are Python byte-code caches (`scripts/__pycache__/`: `apa_eb4_admin`, `apa_eb4_lib`
  .pyc; gitignored, exempt) and an empty `artifacts/apa_eb4/build/cuda_cache/`.

## Every file written

Paths are relative to the worktree, 889 entries, from `artifacts/apa_eb4/written_files.txt` plus this
report. Edited existing files (additive, TC_APA_EB4-guarded): `tensor_cuda/src/apa_spo/spq4.cuh`,
`tensor_cuda/src/bindings.cpp`, `tensor_cuda/include/tc/ops.h`, `tensor_cuda/tensor_cuda/__init__.py`. Outside the
order's path list: `scripts/__pycache__/apa_eb4_admin.cpython-312.pyc` and `scripts/__pycache__/apa_eb4_lib.cpython-312.pyc`
(Python caches). Nothing was written outside the worktree: the session scratchpad holds only this report's generator,
`<local scratch>`.

```text
artifacts/apa_eb4/build/dev1/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb4/build/dev1/bindings.o
artifacts/apa_eb4/build/dev1/device_link.o
artifacts/apa_eb4/build/dev1/kernels.o
artifacts/apa_eb4/build/dev1/module.sha256
artifacts/apa_eb4/build/dev2/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb4/build/dev2/bindings.o
artifacts/apa_eb4/build/dev2/device_link.o
artifacts/apa_eb4/build/dev2/kernels.o
artifacts/apa_eb4/build/dev2/module.sha256
artifacts/apa_eb4/build/dev3/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb4/build/dev3/bindings.o
artifacts/apa_eb4/build/dev3/device_link.o
artifacts/apa_eb4/build/dev3/kernels.o
artifacts/apa_eb4/build/dev3/module.sha256
artifacts/apa_eb4/build/dev4/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb4/build/dev4/bindings.o
artifacts/apa_eb4/build/dev4/device_link.o
artifacts/apa_eb4/build/dev4/kernels.o
artifacts/apa_eb4/build/dev4/module.sha256
artifacts/apa_eb4/build/dev5/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb4/build/dev5/bindings.o
artifacts/apa_eb4/build/dev5/device_link.o
artifacts/apa_eb4/build/dev5/kernels.o
artifacts/apa_eb4/build/dev5/module.sha256
artifacts/apa_eb4/build/main/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb4/build/main/bindings.o
artifacts/apa_eb4/build/main/device_link.o
artifacts/apa_eb4/build/main/kernels.o
artifacts/apa_eb4/build/main/module.sha256
artifacts/apa_eb4/build/mutant1/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb4/build/mutant1/bindings.o
artifacts/apa_eb4/build/mutant1/device_link.o
artifacts/apa_eb4/build/mutant1/kernels.o
artifacts/apa_eb4/build/mutant1/module.sha256
artifacts/apa_eb4/build/mutant10/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb4/build/mutant10/bindings.o
artifacts/apa_eb4/build/mutant10/device_link.o
artifacts/apa_eb4/build/mutant10/kernels.o
artifacts/apa_eb4/build/mutant10/module.sha256
artifacts/apa_eb4/build/mutant2/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb4/build/mutant2/bindings.o
artifacts/apa_eb4/build/mutant2/device_link.o
artifacts/apa_eb4/build/mutant2/kernels.o
artifacts/apa_eb4/build/mutant2/module.sha256
artifacts/apa_eb4/build/mutant3/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb4/build/mutant3/bindings.o
artifacts/apa_eb4/build/mutant3/device_link.o
artifacts/apa_eb4/build/mutant3/kernels.o
artifacts/apa_eb4/build/mutant3/module.sha256
artifacts/apa_eb4/build/mutant4/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb4/build/mutant4/bindings.o
artifacts/apa_eb4/build/mutant4/device_link.o
artifacts/apa_eb4/build/mutant4/kernels.o
artifacts/apa_eb4/build/mutant4/module.sha256
artifacts/apa_eb4/build/mutant5/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb4/build/mutant5/bindings.o
artifacts/apa_eb4/build/mutant5/device_link.o
artifacts/apa_eb4/build/mutant5/kernels.o
artifacts/apa_eb4/build/mutant5/module.sha256
artifacts/apa_eb4/build/mutant6/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb4/build/mutant6/bindings.o
artifacts/apa_eb4/build/mutant6/device_link.o
artifacts/apa_eb4/build/mutant6/kernels.o
artifacts/apa_eb4/build/mutant6/module.sha256
artifacts/apa_eb4/build/mutant7/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb4/build/mutant7/bindings.o
artifacts/apa_eb4/build/mutant7/device_link.o
artifacts/apa_eb4/build/mutant7/kernels.o
artifacts/apa_eb4/build/mutant7/module.sha256
artifacts/apa_eb4/build/mutant8/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb4/build/mutant8/bindings.o
artifacts/apa_eb4/build/mutant8/device_link.o
artifacts/apa_eb4/build/mutant8/kernels.o
artifacts/apa_eb4/build/mutant8/module.sha256
artifacts/apa_eb4/build/mutant9/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb4/build/mutant9/bindings.o
artifacts/apa_eb4/build/mutant9/device_link.o
artifacts/apa_eb4/build/mutant9/kernels.o
artifacts/apa_eb4/build/mutant9/module.sha256
artifacts/apa_eb4/build/noeb4/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_eb4/build/noeb4/bindings.o
artifacts/apa_eb4/build/noeb4/device_link.o
artifacts/apa_eb4/build/noeb4/kernels.o
artifacts/apa_eb4/build/noeb4/module.sha256
artifacts/apa_eb4/build/sass_main_vs_eb3.json
artifacts/apa_eb4/build/sass_noeb4_vs_eb3.json
artifacts/apa_eb4/dev/dev1_profile/result.json
artifacts/apa_eb4/dev/dev1_smoke/result.json
artifacts/apa_eb4/dev/dev1_timing/result.json
artifacts/apa_eb4/dev/dev2_profile/result.json
artifacts/apa_eb4/dev/dev2_smoke/result.json
artifacts/apa_eb4/dev/dev3_profile/result.json
artifacts/apa_eb4/dev/dev3_smoke/result.json
artifacts/apa_eb4/dev/dev3_timing/result.json
artifacts/apa_eb4/dev/dev4_profile_warm/result.json
artifacts/apa_eb4/dev/dev4_smoke/result.json
artifacts/apa_eb4/dev/dev4_timing/result.json
artifacts/apa_eb4/dev/dev5_timing/result.json
artifacts/apa_eb4/dev/timing_dryrun/g4/Qwen3-1.7B_L11_S8192/result.json
artifacts/apa_eb4/dev/timing_dryrun/g4/syn_H16_KV8_S8192/result.json
artifacts/apa_eb4/dev/timing_dryrun/g5/Qwen3-4B_L14_S8192/result.json
artifacts/apa_eb4/dev/timing_dryrun/profile/Qwen3-4B_L14_S32768/result.json
artifacts/apa_eb4/footprint_table.md
artifacts/apa_eb4/g0/Qwen3-1.7B_L11_S8192/result.json
artifacts/apa_eb4/g0/Qwen3-1.7B_L11_S8192/rows.npz
artifacts/apa_eb4/g0/result.json
artifacts/apa_eb4/g1/Qwen3-1.7B_L00_S32768/result.json
artifacts/apa_eb4/g1/Qwen3-1.7B_L00_S32768/rows.npz
artifacts/apa_eb4/g1/Qwen3-1.7B_L00_S8192/result.json
artifacts/apa_eb4/g1/Qwen3-1.7B_L00_S8192/rows.npz
artifacts/apa_eb4/g1/Qwen3-1.7B_L05_S32768/result.json
artifacts/apa_eb4/g1/Qwen3-1.7B_L05_S32768/rows.npz
artifacts/apa_eb4/g1/Qwen3-1.7B_L05_S8192/result.json
artifacts/apa_eb4/g1/Qwen3-1.7B_L05_S8192/rows.npz
artifacts/apa_eb4/g1/Qwen3-1.7B_L11_S32768/result.json
artifacts/apa_eb4/g1/Qwen3-1.7B_L11_S32768/rows.npz
artifacts/apa_eb4/g1/Qwen3-1.7B_L11_S8192/result.json
artifacts/apa_eb4/g1/Qwen3-1.7B_L11_S8192/rows.npz
artifacts/apa_eb4/g1/Qwen3-1.7B_L16_S32768/result.json
artifacts/apa_eb4/g1/Qwen3-1.7B_L16_S32768/rows.npz
artifacts/apa_eb4/g1/Qwen3-1.7B_L16_S8192/result.json
artifacts/apa_eb4/g1/Qwen3-1.7B_L16_S8192/rows.npz
artifacts/apa_eb4/g1/Qwen3-1.7B_L22_S32768/result.json
artifacts/apa_eb4/g1/Qwen3-1.7B_L22_S32768/rows.npz
artifacts/apa_eb4/g1/Qwen3-1.7B_L22_S8192/result.json
artifacts/apa_eb4/g1/Qwen3-1.7B_L22_S8192/rows.npz
artifacts/apa_eb4/g1/Qwen3-1.7B_L27_S32768/result.json
artifacts/apa_eb4/g1/Qwen3-1.7B_L27_S32768/rows.npz
artifacts/apa_eb4/g1/Qwen3-1.7B_L27_S8192/result.json
artifacts/apa_eb4/g1/Qwen3-1.7B_L27_S8192/rows.npz
artifacts/apa_eb4/g1/Qwen3-4B_L00_S32768/result.json
artifacts/apa_eb4/g1/Qwen3-4B_L00_S32768/rows.npz
artifacts/apa_eb4/g1/Qwen3-4B_L00_S8192/result.json
artifacts/apa_eb4/g1/Qwen3-4B_L00_S8192/rows.npz
artifacts/apa_eb4/g1/Qwen3-4B_L07_S32768/result.json
artifacts/apa_eb4/g1/Qwen3-4B_L07_S32768/rows.npz
artifacts/apa_eb4/g1/Qwen3-4B_L07_S8192/result.json
artifacts/apa_eb4/g1/Qwen3-4B_L07_S8192/rows.npz
artifacts/apa_eb4/g1/Qwen3-4B_L14_S32768/result.json
artifacts/apa_eb4/g1/Qwen3-4B_L14_S32768/rows.npz
artifacts/apa_eb4/g1/Qwen3-4B_L14_S8192/result.json
artifacts/apa_eb4/g1/Qwen3-4B_L14_S8192/rows.npz
artifacts/apa_eb4/g1/Qwen3-4B_L21_S32768/result.json
artifacts/apa_eb4/g1/Qwen3-4B_L21_S32768/rows.npz
artifacts/apa_eb4/g1/Qwen3-4B_L21_S8192/result.json
artifacts/apa_eb4/g1/Qwen3-4B_L21_S8192/rows.npz
artifacts/apa_eb4/g1/Qwen3-4B_L28_S32768/result.json
artifacts/apa_eb4/g1/Qwen3-4B_L28_S32768/rows.npz
artifacts/apa_eb4/g1/Qwen3-4B_L28_S8192/result.json
artifacts/apa_eb4/g1/Qwen3-4B_L28_S8192/rows.npz
artifacts/apa_eb4/g1/Qwen3-4B_L35_S32768/result.json
artifacts/apa_eb4/g1/Qwen3-4B_L35_S32768/rows.npz
artifacts/apa_eb4/g1/Qwen3-4B_L35_S8192/result.json
artifacts/apa_eb4/g1/Qwen3-4B_L35_S8192/rows.npz
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-1.7B_L00_S8192/result.json
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-1.7B_L00_S8192/rows.npz
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-1.7B_L05_S8192/result.json
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-1.7B_L05_S8192/rows.npz
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-1.7B_L11_S8192/result.json
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-1.7B_L11_S8192/rows.npz
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-1.7B_L16_S8192/result.json
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-1.7B_L16_S8192/rows.npz
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-1.7B_L22_S8192/result.json
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-1.7B_L22_S8192/rows.npz
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-1.7B_L27_S8192/result.json
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-1.7B_L27_S8192/rows.npz
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-4B_L00_S8192/result.json
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-4B_L00_S8192/rows.npz
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-4B_L07_S8192/result.json
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-4B_L07_S8192/rows.npz
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-4B_L14_S8192/result.json
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-4B_L14_S8192/rows.npz
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-4B_L21_S8192/result.json
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-4B_L21_S8192/rows.npz
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-4B_L28_S8192/result.json
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-4B_L28_S8192/rows.npz
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-4B_L35_S8192/result.json
artifacts/apa_eb4/g1/eb2b_binary/Qwen3-4B_L35_S8192/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L00_S32768/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L00_S32768/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L00_S8192/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L00_S8192/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L05_S32768/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L05_S32768/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L05_S8192/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L05_S8192/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L11_S32768/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L11_S32768/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L11_S8192/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L11_S8192/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L16_S32768/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L16_S32768/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L16_S8192/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L16_S8192/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L22_S32768/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L22_S32768/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L22_S8192/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L22_S8192/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L27_S32768/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L27_S32768/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L27_S8192/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-1.7B_L27_S8192/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L00_S32768/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L00_S32768/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L00_S8192/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L00_S8192/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L07_S32768/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L07_S32768/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L07_S8192/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L07_S8192/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L14_S32768/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L14_S32768/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L14_S8192/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L14_S8192/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L21_S32768/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L21_S32768/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L21_S8192/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L21_S8192/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L28_S32768/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L28_S32768/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L28_S8192/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L28_S8192/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L35_S32768/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L35_S32768/rows.npz
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L35_S8192/result.json
artifacts/apa_eb4/g1/eb3_binary/Qwen3-4B_L35_S8192/rows.npz
artifacts/apa_eb4/g1/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-1.7B_L00_S32768/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-1.7B_L00_S8192/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-1.7B_L05_S32768/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-1.7B_L05_S8192/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-1.7B_L11_S32768/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-1.7B_L11_S8192/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-1.7B_L16_S32768/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-1.7B_L16_S8192/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-1.7B_L22_S32768/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-1.7B_L22_S8192/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-1.7B_L27_S32768/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-1.7B_L27_S8192/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-4B_L00_S32768/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-4B_L00_S8192/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-4B_L07_S32768/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-4B_L07_S8192/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-4B_L14_S32768/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-4B_L14_S8192/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-4B_L21_S32768/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-4B_L21_S8192/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-4B_L28_S32768/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-4B_L28_S8192/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-4B_L35_S32768/result.json
artifacts/apa_eb4/g2/noeb4_vs_eb3_binary/Qwen3-4B_L35_S8192/result.json
artifacts/apa_eb4/g2/result.json
artifacts/apa_eb4/g2_legacy/apa_fxtc1/g1.json
artifacts/apa_eb4/g2_legacy/apa_fxtc1/g1_captures_sha.json
artifacts/apa_eb4/g2_legacy/apa_fxtc1/g1_hash_fxtc1.json
artifacts/apa_eb4/g2_legacy/apa_fxtc1/registration.json
artifacts/apa_eb4/g2_legacy/apa_fxtc1/registration.sha256
artifacts/apa_eb4/g2_legacy/apa_spq4/g1_hash_a6752ea_ref.json
artifacts/apa_eb4/g2_legacy/apa_spq6/g1_hash_eb4noeb4.json
artifacts/apa_eb4/g2_legacy/apa_spq6/g1_hash_spq6.json
artifacts/apa_eb4/g2_legacy/apa_spq6/registration.json
artifacts/apa_eb4/g2_legacy/apa_spq6/registration.sha256
artifacts/apa_eb4/g2_legacy/eb3_g1/Qwen3-1.7B_L00/result.json
artifacts/apa_eb4/g2_legacy/eb3_g1/Qwen3-1.7B_L05/result.json
artifacts/apa_eb4/g2_legacy/eb3_g1/Qwen3-1.7B_L11/result.json
artifacts/apa_eb4/g2_legacy/eb3_g1/Qwen3-1.7B_L16/result.json
artifacts/apa_eb4/g2_legacy/eb3_g1/Qwen3-1.7B_L22/result.json
artifacts/apa_eb4/g2_legacy/eb3_g1/Qwen3-1.7B_L27/result.json
artifacts/apa_eb4/g2_legacy/eb3_g1/Qwen3-4B_L00/result.json
artifacts/apa_eb4/g2_legacy/eb3_g1/Qwen3-4B_L07/result.json
artifacts/apa_eb4/g2_legacy/eb3_g1/Qwen3-4B_L14/result.json
artifacts/apa_eb4/g2_legacy/eb3_g1/Qwen3-4B_L21/result.json
artifacts/apa_eb4/g2_legacy/eb3_g1/Qwen3-4B_L28/result.json
artifacts/apa_eb4/g2_legacy/eb3_g1/Qwen3-4B_L35/result.json
artifacts/apa_eb4/g2_legacy/eb3_g1/syn_H16_KV8_S2048/result.json
artifacts/apa_eb4/g2_legacy/eb3_g1/syn_H16_KV8_S32768/result.json
artifacts/apa_eb4/g2_legacy/eb3_g1/syn_H16_KV8_S65536/result.json
artifacts/apa_eb4/g2_legacy/eb3_g1/syn_H16_KV8_S8000/result.json
artifacts/apa_eb4/g2_legacy/eb3_g1/syn_H16_KV8_S8192/result.json
artifacts/apa_eb4/g2_legacy/eb3_g1/syn_H32_KV8_S2048/result.json
artifacts/apa_eb4/g2_legacy/eb3_g1/syn_H32_KV8_S32768/result.json
artifacts/apa_eb4/g2_legacy/eb3_g1/syn_H32_KV8_S65536/result.json
artifacts/apa_eb4/g2_legacy/eb3_g1/syn_H32_KV8_S8000/result.json
artifacts/apa_eb4/g2_legacy/eb3_g1/syn_H32_KV8_S8192/result.json
artifacts/apa_eb4/g2_legacy/source_hashes_eb3.txt
artifacts/apa_eb4/g3/mutant0/result.json
artifacts/apa_eb4/g3/mutant1/result.json
artifacts/apa_eb4/g3/mutant10/result.json
artifacts/apa_eb4/g3/mutant2/result.json
artifacts/apa_eb4/g3/mutant3/result.json
artifacts/apa_eb4/g3/mutant4/result.json
artifacts/apa_eb4/g3/mutant5/result.json
artifacts/apa_eb4/g3/mutant6/result.json
artifacts/apa_eb4/g3/mutant7/result.json
artifacts/apa_eb4/g3/mutant8/result.json
artifacts/apa_eb4/g3/mutant9/result.json
artifacts/apa_eb4/g3/result.json
artifacts/apa_eb4/g4/Qwen3-1.7B_L00_S32768/result.json
artifacts/apa_eb4/g4/Qwen3-1.7B_L00_S8192/result.json
artifacts/apa_eb4/g4/Qwen3-1.7B_L05_S32768/result.json
artifacts/apa_eb4/g4/Qwen3-1.7B_L05_S8192/result.json
artifacts/apa_eb4/g4/Qwen3-1.7B_L11_S32768/result.json
artifacts/apa_eb4/g4/Qwen3-1.7B_L11_S8192/result.json
artifacts/apa_eb4/g4/Qwen3-1.7B_L16_S32768/result.json
artifacts/apa_eb4/g4/Qwen3-1.7B_L16_S8192/result.json
artifacts/apa_eb4/g4/Qwen3-1.7B_L22_S32768/result.json
artifacts/apa_eb4/g4/Qwen3-1.7B_L22_S8192/result.json
artifacts/apa_eb4/g4/Qwen3-1.7B_L27_S32768/result.json
artifacts/apa_eb4/g4/Qwen3-1.7B_L27_S8192/result.json
artifacts/apa_eb4/g4/Qwen3-4B_L00_S32768/result.json
artifacts/apa_eb4/g4/Qwen3-4B_L00_S8192/result.json
artifacts/apa_eb4/g4/Qwen3-4B_L07_S32768/result.json
artifacts/apa_eb4/g4/Qwen3-4B_L07_S8192/result.json
artifacts/apa_eb4/g4/Qwen3-4B_L14_S32768/result.json
artifacts/apa_eb4/g4/Qwen3-4B_L14_S8192/result.json
artifacts/apa_eb4/g4/Qwen3-4B_L21_S32768/result.json
artifacts/apa_eb4/g4/Qwen3-4B_L21_S8192/result.json
artifacts/apa_eb4/g4/Qwen3-4B_L28_S32768/result.json
artifacts/apa_eb4/g4/Qwen3-4B_L28_S8192/result.json
artifacts/apa_eb4/g4/Qwen3-4B_L35_S32768/result.json
artifacts/apa_eb4/g4/Qwen3-4B_L35_S8192/result.json
artifacts/apa_eb4/g4/result.json
artifacts/apa_eb4/g4/syn_H16_KV8_S32768/result.json
artifacts/apa_eb4/g4/syn_H16_KV8_S8192/result.json
artifacts/apa_eb4/g4/syn_H32_KV8_S32768/result.json
artifacts/apa_eb4/g4/syn_H32_KV8_S8192/result.json
artifacts/apa_eb4/g5/Qwen3-1.7B_L11_S32768/result.json
artifacts/apa_eb4/g5/Qwen3-1.7B_L11_S8192/result.json
artifacts/apa_eb4/g5/Qwen3-4B_L14_S32768/result.json
artifacts/apa_eb4/g5/Qwen3-4B_L14_S8192/result.json
artifacts/apa_eb4/g5/result.json
artifacts/apa_eb4/g5/syn_H16_KV8_S32768/result.json
artifacts/apa_eb4/g5/syn_H32_KV8_S32768/result.json
artifacts/apa_eb4/p0/capture_copy.diff
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L00_k.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L00_o.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L00_q.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L00_v.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L05_k.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L05_o.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L05_q.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L05_v.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L11_k.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L11_o.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L11_q.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L11_v.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L16_k.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L16_o.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L16_q.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L16_v.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L22_k.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L22_o.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L22_q.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L22_v.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L27_k.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L27_o.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L27_q.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/L27_v.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/manifest.json
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/token_ids.npy
artifacts/apa_eb4/p0/captures/Qwen3-1.7B/token_ids_first8192.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L00_k.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L00_o.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L00_q.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L00_v.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L07_k.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L07_o.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L07_q.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L07_v.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L14_k.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L14_o.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L14_q.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L14_v.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L21_k.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L21_o.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L21_q.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L21_v.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L28_k.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L28_o.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L28_q.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L28_v.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L35_k.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L35_o.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L35_q.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/L35_v.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/manifest.json
artifacts/apa_eb4/p0/captures/Qwen3-4B/token_ids.npy
artifacts/apa_eb4/p0/captures/Qwen3-4B/token_ids_first8192.npy
artifacts/apa_eb4/p0/result.json
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L00_S32768/result.json
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L00_S32768/rows.npz
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L00_S8192/result.json
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L00_S8192/rows.npz
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L05_S32768/result.json
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L05_S32768/rows.npz
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L05_S8192/result.json
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L05_S8192/rows.npz
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L11_S32768/result.json
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L11_S32768/rows.npz
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L11_S8192/result.json
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L11_S8192/rows.npz
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L16_S32768/result.json
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L16_S32768/rows.npz
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L16_S8192/result.json
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L16_S8192/rows.npz
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L22_S32768/result.json
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L22_S32768/rows.npz
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L22_S8192/result.json
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L22_S8192/rows.npz
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L27_S32768/result.json
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L27_S32768/rows.npz
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L27_S8192/result.json
artifacts/apa_eb4/p1/cells/Qwen3-1.7B_L27_S8192/rows.npz
artifacts/apa_eb4/p1/cells/Qwen3-4B_L00_S32768/result.json
artifacts/apa_eb4/p1/cells/Qwen3-4B_L00_S32768/rows.npz
artifacts/apa_eb4/p1/cells/Qwen3-4B_L00_S8192/result.json
artifacts/apa_eb4/p1/cells/Qwen3-4B_L00_S8192/rows.npz
artifacts/apa_eb4/p1/cells/Qwen3-4B_L07_S32768/result.json
artifacts/apa_eb4/p1/cells/Qwen3-4B_L07_S32768/rows.npz
artifacts/apa_eb4/p1/cells/Qwen3-4B_L07_S8192/result.json
artifacts/apa_eb4/p1/cells/Qwen3-4B_L07_S8192/rows.npz
artifacts/apa_eb4/p1/cells/Qwen3-4B_L14_S32768/result.json
artifacts/apa_eb4/p1/cells/Qwen3-4B_L14_S32768/rows.npz
artifacts/apa_eb4/p1/cells/Qwen3-4B_L14_S8192/result.json
artifacts/apa_eb4/p1/cells/Qwen3-4B_L14_S8192/rows.npz
artifacts/apa_eb4/p1/cells/Qwen3-4B_L21_S32768/result.json
artifacts/apa_eb4/p1/cells/Qwen3-4B_L21_S32768/rows.npz
artifacts/apa_eb4/p1/cells/Qwen3-4B_L21_S8192/result.json
artifacts/apa_eb4/p1/cells/Qwen3-4B_L21_S8192/rows.npz
artifacts/apa_eb4/p1/cells/Qwen3-4B_L28_S32768/result.json
artifacts/apa_eb4/p1/cells/Qwen3-4B_L28_S32768/rows.npz
artifacts/apa_eb4/p1/cells/Qwen3-4B_L28_S8192/result.json
artifacts/apa_eb4/p1/cells/Qwen3-4B_L28_S8192/rows.npz
artifacts/apa_eb4/p1/cells/Qwen3-4B_L35_S32768/result.json
artifacts/apa_eb4/p1/cells/Qwen3-4B_L35_S32768/rows.npz
artifacts/apa_eb4/p1/cells/Qwen3-4B_L35_S8192/result.json
artifacts/apa_eb4/p1/cells/Qwen3-4B_L35_S8192/rows.npz
artifacts/apa_eb4/p1/path_copies.diff
artifacts/apa_eb4/p1/qualification.md
artifacts/apa_eb4/p1/summary.json
artifacts/apa_eb4/p1/table.md
artifacts/apa_eb4/p1/theta.json
artifacts/apa_eb4/p1/theta.sha256
artifacts/apa_eb4/path_copies.diff
artifacts/apa_eb4/path_copies_eb3_gates.diff
artifacts/apa_eb4/path_copies_legacy.diff
artifacts/apa_eb4/prediction_table.md
artifacts/apa_eb4/profile/Qwen3-1.7B_L11_S32768/result.json
artifacts/apa_eb4/profile/Qwen3-1.7B_L11_S8192/result.json
artifacts/apa_eb4/profile/Qwen3-4B_L14_S32768/result.json
artifacts/apa_eb4/profile/Qwen3-4B_L14_S8192/result.json
artifacts/apa_eb4/profile_table.md
artifacts/apa_eb4/registration.json
artifacts/apa_eb4/registration.sha256
artifacts/apa_eb4/scratch_retired/debug/dbg_codes.py
artifacts/apa_eb4/scratch_retired/debug/dbg_g11.py
artifacts/apa_eb4/scratch_retired/dryrun_first/g5/Qwen3-1.7B_L11_S8192/result.json
artifacts/apa_eb4/scratch_retired/dryrun_first/g5/syn_H32_KV8_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g0/Qwen3-1.7B_L11_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g0/Qwen3-1.7B_L11_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L00_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L00_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L00_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L00_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L05_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L05_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L05_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L05_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L11_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L11_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L11_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L11_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L16_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L16_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L16_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L16_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L22_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L22_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L22_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L22_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L27_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L27_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L27_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-1.7B_L27_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L00_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L00_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L00_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L00_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L07_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L07_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L07_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L07_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L14_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L14_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L14_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L14_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L21_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L21_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L21_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L21_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L28_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L28_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L28_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L28_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L35_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L35_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L35_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/Qwen3-4B_L35_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L00_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L00_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L00_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L00_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L05_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L05_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L05_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L05_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L11_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L11_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L11_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L11_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L16_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L16_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L16_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L16_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L22_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L22_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L22_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L22_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L27_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L27_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L27_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-1.7B_L27_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L00_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L00_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L00_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L00_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L07_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L07_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L07_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L07_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L14_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L14_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L14_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L14_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L21_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L21_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L21_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L21_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L28_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L28_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L28_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L28_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L35_S32768/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L35_S32768/rows.npz
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L35_S8192/result.json
artifacts/apa_eb4/scratch_retired/g1_uint16_truncation/g1/eb3_binary/Qwen3-4B_L35_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/p1_dev/Qwen3-1.7B_L11_S8192/result.json
artifacts/apa_eb4/scratch_retired/p1_dev/Qwen3-1.7B_L11_S8192/rows.npz
artifacts/apa_eb4/scratch_retired/sass_first_attempt/sass_main_vs_eb3.json
artifacts/apa_eb4/scratch_retired/sass_first_attempt/sass_noeb4_vs_eb3.json
artifacts/apa_eb4/source_before.sha256
artifacts/apa_eb4/source_before/tensor_cuda_include_tc_ops.h
artifacts/apa_eb4/source_before/tensor_cuda_src_apa_spo_eb2b.cuh
artifacts/apa_eb4/source_before/tensor_cuda_src_apa_spo_eb2b_values.cuh
artifacts/apa_eb4/source_before/tensor_cuda_src_apa_spo_eb3.cuh
artifacts/apa_eb4/source_before/tensor_cuda_src_apa_spo_eb3_tuning.cuh
artifacts/apa_eb4/source_before/tensor_cuda_src_apa_spo_spq4.cuh
artifacts/apa_eb4/source_before/tensor_cuda_src_apa_spo_spq6.cuh
artifacts/apa_eb4/source_before/tensor_cuda_src_bindings.cpp
artifacts/apa_eb4/source_before/tensor_cuda_tensor_cuda___init__.py
artifacts/apa_eb4/speed_table.md
artifacts/apa_eb4/speed_table_synthetic.md
artifacts/apa_eb4/summary.json
docs/APA_EB4_LEDGER.md
docs/APA_EB4_REPORT.md
logs/apa_eb4_build_dev1.log
logs/apa_eb4_build_dev2.log
logs/apa_eb4_build_dev3.log
logs/apa_eb4_build_dev4.log
logs/apa_eb4_build_dev5.log
logs/apa_eb4_build_main.log
logs/apa_eb4_build_mutant1.log
logs/apa_eb4_build_mutant10.log
logs/apa_eb4_build_mutant2.log
logs/apa_eb4_build_mutant3.log
logs/apa_eb4_build_mutant4.log
logs/apa_eb4_build_mutant5.log
logs/apa_eb4_build_mutant6.log
logs/apa_eb4_build_mutant7.log
logs/apa_eb4_build_mutant8.log
logs/apa_eb4_build_mutant9.log
logs/apa_eb4_build_noeb4.log
logs/apa_eb4_cpu_tests.log
logs/apa_eb4_dev1_profile.log
logs/apa_eb4_dev1_smoke.log
logs/apa_eb4_dev1_timing.log
logs/apa_eb4_dev2_profile.log
logs/apa_eb4_dev2_smoke.log
logs/apa_eb4_dev3_profile.log
logs/apa_eb4_dev3_smoke.log
logs/apa_eb4_dev3_timing.log
logs/apa_eb4_dev4_profile_warm.log
logs/apa_eb4_dev4_smoke.log
logs/apa_eb4_dev4_timing.log
logs/apa_eb4_dev5_timing.log
logs/apa_eb4_dryrun_g4_--H_16_--S_8192.log
logs/apa_eb4_dryrun_g4_--model_Qwen3-1.7B_--layer_11_--S_8192.log
logs/apa_eb4_dryrun_g5_--H_32_--S_32768.log
logs/apa_eb4_dryrun_g5_--model_Qwen3-1.7B_--layer_11_--S_8192.log
logs/apa_eb4_dryrun_g5_b.log
logs/apa_eb4_dryrun_profile_--model_Qwen3-4B_--layer_14_--S_32768.log
logs/apa_eb4_dump2b_Qwen3-1.7B_L0_S8192.log
logs/apa_eb4_dump2b_Qwen3-1.7B_L11_S8192.log
logs/apa_eb4_dump2b_Qwen3-1.7B_L16_S8192.log
logs/apa_eb4_dump2b_Qwen3-1.7B_L22_S8192.log
logs/apa_eb4_dump2b_Qwen3-1.7B_L27_S8192.log
logs/apa_eb4_dump2b_Qwen3-1.7B_L5_S8192.log
logs/apa_eb4_dump2b_Qwen3-4B_L0_S8192.log
logs/apa_eb4_dump2b_Qwen3-4B_L14_S8192.log
logs/apa_eb4_dump2b_Qwen3-4B_L21_S8192.log
logs/apa_eb4_dump2b_Qwen3-4B_L28_S8192.log
logs/apa_eb4_dump2b_Qwen3-4B_L35_S8192.log
logs/apa_eb4_dump2b_Qwen3-4B_L7_S8192.log
logs/apa_eb4_dump3_Qwen3-1.7B_L0_S32768.log
logs/apa_eb4_dump3_Qwen3-1.7B_L0_S8192.log
logs/apa_eb4_dump3_Qwen3-1.7B_L11_S32768.log
logs/apa_eb4_dump3_Qwen3-1.7B_L11_S8192.log
logs/apa_eb4_dump3_Qwen3-1.7B_L16_S32768.log
logs/apa_eb4_dump3_Qwen3-1.7B_L16_S8192.log
logs/apa_eb4_dump3_Qwen3-1.7B_L22_S32768.log
logs/apa_eb4_dump3_Qwen3-1.7B_L22_S8192.log
logs/apa_eb4_dump3_Qwen3-1.7B_L27_S32768.log
logs/apa_eb4_dump3_Qwen3-1.7B_L27_S8192.log
logs/apa_eb4_dump3_Qwen3-1.7B_L5_S32768.log
logs/apa_eb4_dump3_Qwen3-1.7B_L5_S8192.log
logs/apa_eb4_dump3_Qwen3-4B_L0_S32768.log
logs/apa_eb4_dump3_Qwen3-4B_L0_S8192.log
logs/apa_eb4_dump3_Qwen3-4B_L14_S32768.log
logs/apa_eb4_dump3_Qwen3-4B_L14_S8192.log
logs/apa_eb4_dump3_Qwen3-4B_L21_S32768.log
logs/apa_eb4_dump3_Qwen3-4B_L21_S8192.log
logs/apa_eb4_dump3_Qwen3-4B_L28_S32768.log
logs/apa_eb4_dump3_Qwen3-4B_L28_S8192.log
logs/apa_eb4_dump3_Qwen3-4B_L35_S32768.log
logs/apa_eb4_dump3_Qwen3-4B_L35_S8192.log
logs/apa_eb4_dump3_Qwen3-4B_L7_S32768.log
logs/apa_eb4_dump3_Qwen3-4B_L7_S8192.log
logs/apa_eb4_g0.log
logs/apa_eb4_g1_Qwen3-1.7B_L0_S32768.log
logs/apa_eb4_g1_Qwen3-1.7B_L0_S8192.log
logs/apa_eb4_g1_Qwen3-1.7B_L11_S32768.log
logs/apa_eb4_g1_Qwen3-1.7B_L11_S8192.log
logs/apa_eb4_g1_Qwen3-1.7B_L16_S32768.log
logs/apa_eb4_g1_Qwen3-1.7B_L16_S8192.log
logs/apa_eb4_g1_Qwen3-1.7B_L22_S32768.log
logs/apa_eb4_g1_Qwen3-1.7B_L22_S8192.log
logs/apa_eb4_g1_Qwen3-1.7B_L27_S32768.log
logs/apa_eb4_g1_Qwen3-1.7B_L27_S8192.log
logs/apa_eb4_g1_Qwen3-1.7B_L5_S32768.log
logs/apa_eb4_g1_Qwen3-1.7B_L5_S8192.log
logs/apa_eb4_g1_Qwen3-4B_L0_S32768.log
logs/apa_eb4_g1_Qwen3-4B_L0_S8192.log
logs/apa_eb4_g1_Qwen3-4B_L14_S32768.log
logs/apa_eb4_g1_Qwen3-4B_L14_S8192.log
logs/apa_eb4_g1_Qwen3-4B_L21_S32768.log
logs/apa_eb4_g1_Qwen3-4B_L21_S8192.log
logs/apa_eb4_g1_Qwen3-4B_L28_S32768.log
logs/apa_eb4_g1_Qwen3-4B_L28_S8192.log
logs/apa_eb4_g1_Qwen3-4B_L35_S32768.log
logs/apa_eb4_g1_Qwen3-4B_L35_S8192.log
logs/apa_eb4_g1_Qwen3-4B_L7_S32768.log
logs/apa_eb4_g1_Qwen3-4B_L7_S8192.log
logs/apa_eb4_g2_eb3g1_Qwen3-1.7B_L0.log
logs/apa_eb4_g2_eb3g1_Qwen3-1.7B_L11.log
logs/apa_eb4_g2_eb3g1_Qwen3-1.7B_L16.log
logs/apa_eb4_g2_eb3g1_Qwen3-1.7B_L22.log
logs/apa_eb4_g2_eb3g1_Qwen3-1.7B_L27.log
logs/apa_eb4_g2_eb3g1_Qwen3-1.7B_L5.log
logs/apa_eb4_g2_eb3g1_Qwen3-4B_L0.log
logs/apa_eb4_g2_eb3g1_Qwen3-4B_L14.log
logs/apa_eb4_g2_eb3g1_Qwen3-4B_L21.log
logs/apa_eb4_g2_eb3g1_Qwen3-4B_L28.log
logs/apa_eb4_g2_eb3g1_Qwen3-4B_L35.log
logs/apa_eb4_g2_eb3g1_Qwen3-4B_L7.log
logs/apa_eb4_g2_eb3g1_syn_H16_S2048.log
logs/apa_eb4_g2_eb3g1_syn_H16_S32768.log
logs/apa_eb4_g2_eb3g1_syn_H16_S65536.log
logs/apa_eb4_g2_eb3g1_syn_H16_S8000.log
logs/apa_eb4_g2_eb3g1_syn_H16_S8192.log
logs/apa_eb4_g2_eb3g1_syn_H32_S2048.log
logs/apa_eb4_g2_eb3g1_syn_H32_S32768.log
logs/apa_eb4_g2_eb3g1_syn_H32_S65536.log
logs/apa_eb4_g2_eb3g1_syn_H32_S8000.log
logs/apa_eb4_g2_eb3g1_syn_H32_S8192.log
logs/apa_eb4_g2_fxtc1_captures.log
logs/apa_eb4_g2_fxtc1_compare.log
logs/apa_eb4_g2_fxtc1_hashes.log
logs/apa_eb4_g2_spq6_hashes.log
logs/apa_eb4_g2x_Qwen3-1.7B_L0_S32768.log
logs/apa_eb4_g2x_Qwen3-1.7B_L0_S8192.log
logs/apa_eb4_g2x_Qwen3-1.7B_L11_S32768.log
logs/apa_eb4_g2x_Qwen3-1.7B_L11_S8192.log
logs/apa_eb4_g2x_Qwen3-1.7B_L16_S32768.log
logs/apa_eb4_g2x_Qwen3-1.7B_L16_S8192.log
logs/apa_eb4_g2x_Qwen3-1.7B_L22_S32768.log
logs/apa_eb4_g2x_Qwen3-1.7B_L22_S8192.log
logs/apa_eb4_g2x_Qwen3-1.7B_L27_S32768.log
logs/apa_eb4_g2x_Qwen3-1.7B_L27_S8192.log
logs/apa_eb4_g2x_Qwen3-1.7B_L5_S32768.log
logs/apa_eb4_g2x_Qwen3-1.7B_L5_S8192.log
logs/apa_eb4_g2x_Qwen3-4B_L0_S32768.log
logs/apa_eb4_g2x_Qwen3-4B_L0_S8192.log
logs/apa_eb4_g2x_Qwen3-4B_L14_S32768.log
logs/apa_eb4_g2x_Qwen3-4B_L14_S8192.log
logs/apa_eb4_g2x_Qwen3-4B_L21_S32768.log
logs/apa_eb4_g2x_Qwen3-4B_L21_S8192.log
logs/apa_eb4_g2x_Qwen3-4B_L28_S32768.log
logs/apa_eb4_g2x_Qwen3-4B_L28_S8192.log
logs/apa_eb4_g2x_Qwen3-4B_L35_S32768.log
logs/apa_eb4_g2x_Qwen3-4B_L35_S8192.log
logs/apa_eb4_g2x_Qwen3-4B_L7_S32768.log
logs/apa_eb4_g2x_Qwen3-4B_L7_S8192.log
logs/apa_eb4_g3_mutant0.log
logs/apa_eb4_g3_mutant1.log
logs/apa_eb4_g3_mutant10.log
logs/apa_eb4_g3_mutant2.log
logs/apa_eb4_g3_mutant3.log
logs/apa_eb4_g3_mutant4.log
logs/apa_eb4_g3_mutant5.log
logs/apa_eb4_g3_mutant6.log
logs/apa_eb4_g3_mutant7.log
logs/apa_eb4_g3_mutant8.log
logs/apa_eb4_g3_mutant9.log
logs/apa_eb4_g4_Qwen3-1.7B_L0_S32768.log
logs/apa_eb4_g4_Qwen3-1.7B_L0_S8192.log
logs/apa_eb4_g4_Qwen3-1.7B_L11_S32768.log
logs/apa_eb4_g4_Qwen3-1.7B_L11_S8192.log
logs/apa_eb4_g4_Qwen3-1.7B_L16_S32768.log
logs/apa_eb4_g4_Qwen3-1.7B_L16_S8192.log
logs/apa_eb4_g4_Qwen3-1.7B_L22_S32768.log
logs/apa_eb4_g4_Qwen3-1.7B_L22_S8192.log
logs/apa_eb4_g4_Qwen3-1.7B_L27_S32768.log
logs/apa_eb4_g4_Qwen3-1.7B_L27_S8192.log
logs/apa_eb4_g4_Qwen3-1.7B_L5_S32768.log
logs/apa_eb4_g4_Qwen3-1.7B_L5_S8192.log
logs/apa_eb4_g4_Qwen3-4B_L0_S32768.log
logs/apa_eb4_g4_Qwen3-4B_L0_S8192.log
logs/apa_eb4_g4_Qwen3-4B_L14_S32768.log
logs/apa_eb4_g4_Qwen3-4B_L14_S8192.log
logs/apa_eb4_g4_Qwen3-4B_L21_S32768.log
logs/apa_eb4_g4_Qwen3-4B_L21_S8192.log
logs/apa_eb4_g4_Qwen3-4B_L28_S32768.log
logs/apa_eb4_g4_Qwen3-4B_L28_S8192.log
logs/apa_eb4_g4_Qwen3-4B_L35_S32768.log
logs/apa_eb4_g4_Qwen3-4B_L35_S8192.log
logs/apa_eb4_g4_Qwen3-4B_L7_S32768.log
logs/apa_eb4_g4_Qwen3-4B_L7_S8192.log
logs/apa_eb4_g4_syn_H16_S32768.log
logs/apa_eb4_g4_syn_H16_S8192.log
logs/apa_eb4_g4_syn_H32_S32768.log
logs/apa_eb4_g4_syn_H32_S8192.log
logs/apa_eb4_g5_Qwen3-1.7B_L11_S32768.log
logs/apa_eb4_g5_Qwen3-1.7B_L11_S8192.log
logs/apa_eb4_g5_Qwen3-4B_L14_S32768.log
logs/apa_eb4_g5_Qwen3-4B_L14_S8192.log
logs/apa_eb4_g5_syn_H16_S32768.log
logs/apa_eb4_g5_syn_H32_S32768.log
logs/apa_eb4_p0_Qwen3-1.7B.log
logs/apa_eb4_p0_Qwen3-4B.log
logs/apa_eb4_p1_Qwen3-1.7B_L0_S32768.log
logs/apa_eb4_p1_Qwen3-1.7B_L0_S8192.log
logs/apa_eb4_p1_Qwen3-1.7B_L11_S32768.log
logs/apa_eb4_p1_Qwen3-1.7B_L11_S8192.log
logs/apa_eb4_p1_Qwen3-1.7B_L16_S32768.log
logs/apa_eb4_p1_Qwen3-1.7B_L16_S8192.log
logs/apa_eb4_p1_Qwen3-1.7B_L22_S32768.log
logs/apa_eb4_p1_Qwen3-1.7B_L22_S8192.log
logs/apa_eb4_p1_Qwen3-1.7B_L27_S32768.log
logs/apa_eb4_p1_Qwen3-1.7B_L27_S8192.log
logs/apa_eb4_p1_Qwen3-1.7B_L5_S32768.log
logs/apa_eb4_p1_Qwen3-1.7B_L5_S8192.log
logs/apa_eb4_p1_Qwen3-4B_L0_S32768.log
logs/apa_eb4_p1_Qwen3-4B_L0_S8192.log
logs/apa_eb4_p1_Qwen3-4B_L14_S32768.log
logs/apa_eb4_p1_Qwen3-4B_L14_S8192.log
logs/apa_eb4_p1_Qwen3-4B_L21_S32768.log
logs/apa_eb4_p1_Qwen3-4B_L21_S8192.log
logs/apa_eb4_p1_Qwen3-4B_L28_S32768.log
logs/apa_eb4_p1_Qwen3-4B_L28_S8192.log
logs/apa_eb4_p1_Qwen3-4B_L35_S32768.log
logs/apa_eb4_p1_Qwen3-4B_L35_S8192.log
logs/apa_eb4_p1_Qwen3-4B_L7_S32768.log
logs/apa_eb4_p1_Qwen3-4B_L7_S8192.log
logs/apa_eb4_profile_Qwen3-1.7B_L11_S32768.log
logs/apa_eb4_profile_Qwen3-1.7B_L11_S8192.log
logs/apa_eb4_profile_Qwen3-4B_L14_S32768.log
logs/apa_eb4_profile_Qwen3-4B_L14_S8192.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-1.7B_L0_S32768.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-1.7B_L0_S8192.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-1.7B_L11_S32768.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-1.7B_L11_S8192.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-1.7B_L16_S32768.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-1.7B_L16_S8192.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-1.7B_L22_S32768.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-1.7B_L22_S8192.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-1.7B_L27_S32768.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-1.7B_L27_S8192.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-1.7B_L5_S32768.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-1.7B_L5_S8192.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-4B_L0_S32768.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-4B_L0_S8192.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-4B_L14_S32768.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-4B_L14_S8192.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-4B_L21_S32768.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-4B_L21_S8192.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-4B_L28_S32768.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-4B_L28_S8192.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-4B_L35_S32768.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-4B_L35_S8192.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-4B_L7_S32768.log
logs/retired_g1_uint16/apa_eb4_dump3_Qwen3-4B_L7_S8192.log
logs/retired_g1_uint16/apa_eb4_g0.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-1.7B_L0_S32768.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-1.7B_L0_S8192.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-1.7B_L11_S32768.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-1.7B_L11_S8192.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-1.7B_L16_S32768.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-1.7B_L16_S8192.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-1.7B_L22_S32768.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-1.7B_L22_S8192.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-1.7B_L27_S32768.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-1.7B_L27_S8192.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-1.7B_L5_S32768.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-1.7B_L5_S8192.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-4B_L0_S32768.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-4B_L0_S8192.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-4B_L14_S32768.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-4B_L14_S8192.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-4B_L21_S32768.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-4B_L21_S8192.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-4B_L28_S32768.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-4B_L28_S8192.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-4B_L35_S32768.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-4B_L35_S8192.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-4B_L7_S32768.log
logs/retired_g1_uint16/apa_eb4_g1_Qwen3-4B_L7_S8192.log
scripts/apa_eb4_admin.py
scripts/apa_eb4_batch.sh
scripts/apa_eb4_build.sh
scripts/apa_eb4_capture.py
scripts/apa_eb4_dev.py
scripts/apa_eb4_eb1_lib.py
scripts/apa_eb4_eb1_quant.py
scripts/apa_eb4_eb1_run.py
scripts/apa_eb4_eb1b_run.py
scripts/apa_eb4_eb2a_lib.py
scripts/apa_eb4_eb2a_run.py
scripts/apa_eb4_eb3_gates.py
scripts/apa_eb4_eb3_lib.py
scripts/apa_eb4_fxtc1_common.py
scripts/apa_eb4_fxtc1_g1.py
scripts/apa_eb4_fxtc1_gpu.sh
scripts/apa_eb4_fxtc1_lib.py
scripts/apa_eb4_gates.py
scripts/apa_eb4_gpu.sh
scripts/apa_eb4_legacy_rows.sh
scripts/apa_eb4_lib.py
scripts/apa_eb4_measure.sh
scripts/apa_eb4_mutants.sh
scripts/apa_eb4_offline.py
scripts/apa_eb4_p0_check.py
scripts/apa_eb4_p1_batch.sh
scripts/apa_eb4_p1_summary.py
scripts/apa_eb4_sass.py
scripts/apa_eb4_spq1_common.py
scripts/apa_eb4_spq6_common.py
scripts/apa_eb4_spq6_g1.py
scripts/apa_eb4_spq6_gpu.sh
scripts/apa_eb4_spq6_lib.py
scripts/apa_eb4_synth.py
scripts/apa_eb4_timing.py
tensor_cuda/include/tc/ops.h
tensor_cuda/src/apa_spo/eb4.cuh
tensor_cuda/src/apa_spo/eb4_tuning.cuh
tensor_cuda/src/apa_spo/spq4.cuh
tensor_cuda/src/bindings.cpp
tensor_cuda/tensor_cuda/__init__.py
tests/test_apa_eb4.py
```

## Spec ledger

- P0 real 32K captures (both models, full 32K, first 8,192 ids = SPQ3), manifests with hashes: **met**. `artifacts/apa_eb4/p0/result.json`, `p0/captures/*/manifest.json`.
- P1 offline gate, 12 layers × {8K, 32K}, E2a metric and band, per-cell JSON: **met**. `artifacts/apa_eb4/p1/cells/*/result.json`, `p1/summary.json`.
- P1 θ_L calibrated on 8K calibration rows, ≥ 64-point log grid 1e-6..1, frozen before kernel timing with SHA-256: **met**. `artifacts/apa_eb4/p1/theta.json`, `theta.sha256`.
- P1 per-cell bf16-block fraction, V bits per key element, T-stale/T-fresh agreement: **met**. `artifacts/apa_eb4/p1/table.md`.
- P1 QUALIFIED status per arm; kernel work only for QUALIFIED arms (+ T-fresh control): **met** (T-fresh built; T-stale, T-thr not built). `artifacts/apa_eb4/p1/qualification.md`.
- Max-logit-gap T-thr variant (optional, descriptive): **not done** (optional; no receipt).
- G0 build and smoke: **met**. `artifacts/apa_eb4/g0/result.json`, `logs/apa_eb4_build_main.log`.
- G1.1 forced bf16 == EB3 binary, 24 real cells × 64 rows, sinks off/on: **met**. `artifacts/apa_eb4/g1/result.json`, `g1/eb3_binary/`.
- G1.2 T-fresh == EB2b binary tk8, 12 real 8K cells × 64 rows: **met**. `artifacts/apa_eb4/g1/result.json`, `g1/eb2b_binary/`.
- G1.3 injected offline flags == in-kernel output where flags agree: **met**. `artifacts/apa_eb4/g1/result.json`.
- G2 decision parity ≥ 1 − 1e-4 with near-tie witnesses (fp32 and fp64 scores): **met** (2/3,932,160, both witnessed). `artifacts/apa_eb4/g2/result.json`.
- G2 no regression with TC_APA_EB4 off: legacy hashes (72/32/136/48): **met**. `artifacts/apa_eb4/g2_legacy/apa_fxtc1/g1.json`, `g2_legacy/apa_spq6/g1_hash_eb4noeb4.json`.
- G2 no regression: the outputs in EB3's `g1_baseline_binary` rows: **met for digests, vacuous for outputs**. EB3's saved `outs` are all zero. It is supplemented by a real cross-binary output comparison (g2x, 6,144/6,144 equal). `artifacts/apa_eb4/g2_legacy/eb3_g1/`, `g2/noeb4_vs_eb3_binary/`.
- G2 SASS identity of EB3's 5 default and registered variant-0 kernels: **met** (all 705 kernels, noeb4 and main). `artifacts/apa_eb4/build/sass_noeb4_vs_eb3.json`, `build/sass_main_vs_eb3.json`.
- G3 ≥ 6 compile-time mutants each killed by G1 or G2, weak kills stated: **met** (9/10; mutant 7 survived, mutant 8 weak). `artifacts/apa_eb4/g3/result.json`.
- Registration after G0–G3 and before G4, create-only, with SHA-256: **met**. `artifacts/apa_eb4/registration.json`, `registration.sha256`.
- Registration contents: timing protocol, θ_L and its SHA-256, P1 verdicts, LD1–LD7 in full (not line-wrapped), S1–S9, every source/script hash, every binary hash: **met**. `artifacts/apa_eb4/registration.json`.
- Crash amendments: **none used**.
- G4 real 8K and 32K per layer (arm/EB3, arm/flash, non-early medians), one process, interleaved, cold L2: **met**. `artifacts/apa_eb4/g4/*/result.json`, `speed_table.md`.
- G4 synthetic f sweep {0.01, 0.05, 0.25} (descriptive): **met**; θ sweep: **not applicable** (T-thr not built). `artifacts/apa_eb4/speed_table_synthetic.md`.
- G4 per-phase CUPTI profile for each built arm and EB3, one real cell per model at 8K and 32K: **met**. `artifacts/apa_eb4/profile/*/result.json`, `profile_table.md`.
- G4 T-stale steady-state timing: **not applicable** (T-stale not built).
- G5 stored bytes, scratch per call, launches per call, quantization per appended token: **met**. `artifacts/apa_eb4/g5/result.json`, `footprint_table.md`.
- Registered speed and LONG verdicts per built arm: **met** (T-fresh WIN / WIN). `artifacts/apa_eb4/summary.json`.
- Prediction table with full text: **met**. `artifacts/apa_eb4/prediction_table.md`, this report.
- Development timings before registration disclosed: **met**. `docs/APA_EB4_LEDGER.md` (Development), `artifacts/apa_eb4/dev/`.
- `docs/APA_EB4_REPORT.md` (verdict first; P0, P1, G0–G5, predictions, speed 8K/32K, profiles, footprint, Prior art, Scope): **met**. This file.
- `docs/APA_EB4_LEDGER.md`: **met**.
- Prior art at code sites, ledger and report: **met**. `tensor_cuda/src/apa_spo/eb4.cuh`, `eb4_tuning.cuh`, `scripts/apa_eb4_offline.py`, ledger, this report.
- Path-only copies with diffs recorded: **met**. `artifacts/apa_eb4/path_copies.diff`, `path_copies_legacy.diff`, `path_copies_eb3_gates.diff`, `p0/capture_copy.diff`, `p1/path_copies.diff`.
- Additive, flag-guarded source edits; EB3 variant 0 SASS-identical in this build: **met**. `tests/test_apa_eb4.py` (`logs/apa_eb4_cpu_tests.log`), `artifacts/apa_eb4/build/sass_main_vs_eb3.json`.
- Rails (RSS < 16 GB; `artifacts/apa_eb4` < 4 GB, 2.66 GB): **met**. `artifacts/apa_eb4/final_audit.json`, per-receipt `resources`.
- No git writes, no subagents, no deletes, GPU only under flock/timeout, native driver: **met**. `artifacts/apa_eb4/final_audit.json`, ledger.
- List of every file written: **met**. `artifacts/apa_eb4/written_files.txt`, this report.
