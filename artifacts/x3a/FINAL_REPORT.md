# TX3-X3a final report: exact integer z-score statistics for APA (seat: Claude Opus 5.5, CPU only)

Order: `orders/TX3_X3A_EXACT_ZSTATS.md` (immutable). Ledger: `artifacts/x3a/LEDGER.md`. Worktree branch
`x3a-exact-zstats` (base production main b4da646). No git, no GPU, no subagents. Nothing deleted outside `/tmp`
(one seat file binned with `delete-me`).

## Verdict

- **T-G1 build: PASS.** Clean build from an empty build directory with the copied TX3-F1 helper: rc 0, 95 s.
  The `.so` sha256 is `eab547e99433ea68404773b7a4ec9494b5518595798ffe3005d909bfdad5a29d` (14,305,384 bytes).
- **T-G2 SASS: PASS.**
  - Linked cubin: 563/563 production functions identical (501,496 instructions), plus 103 new.
  - `-all`: 1,128/1,128 identical (1,003,200 instructions), plus 1 new cubin.
  - Power check: a one-token device mutant is flagged on exactly its 13 instantiations (26 under `-all`).
  - Reaching this took three structural fixes (first build: 75 production functions moved; see T-G2).
- **T-G3 CPU tests: PASS for X3a. The full suite does NOT pass on CPU, and that state is pre-existing.**
  - X3a CPU tests: 23/23 fresh. Tooling power checks: 15/15.
  - The full suite on CPU has per-test outcomes **identical** to the pristine baseline build: 593 passed,
    450 skipped, 1,541 failed, 15 errors, plus 1 pre-existing collection error.
  - The failures are GPU tests without a CPU skip guard and stale source pins. They fail the same way on the
    production sources.
- **T-G4 staged GPU lanes: READY, CPU-audited.**
  - 8 lines: G0, B0, X1a, X1b, X2, X3, X4, VERDICT. The registered counts are below.
  - L48 audit pass A: the staged verdict FAILS, as it must (no device).
  - Audit pass B: the staged verdict FAILS on constructed captures, as it must. The same verdict with `--audit`
    PASSES (the control).
- **Registered acceptance (T-G1 to T-G4 pass on CPU and the staged lines are ready): ACCEPT.** The lead's GPU run
  decides X1 (0 mismatches), X3 (≤ 3% per cell) and X4 (no regression).
- **One finding against the order's premise (ledger 10:30): D3's "a = |bulk| (fp32, as now)" cannot give X1's
  equality "across WCOOP".**
  - Production's per-key |bulk| itself depends on WCOOP at DMAX 64. WCOOP=true uses lane FMA chains plus a
    butterfly; WCOOP=false uses one serial FMA chain.
  - Resolution: the statistic's |bulk| is pinned to the canonical (WCOOP lane) order. This equals "as now" in every
    WCOOP=true launch (proved from production SASS). It differs only where production runs WCOOP=false (D=64 decode).
  - The pass-2 decisions are untouched.
- **Production bug found, not fixed (the order forbids changing existing device code).**
  - Location: `gemm_apa.cu:306-307` (`scale_stats_count`) reads `red[0]` and has thread 0 overwrite it with no
    barrier in between (a write-after-read race across warps).
  - Effect: only `counts[row]`, which nothing reads.
  - `thresholds[row]` is correct.
- **CPU-reconstructed witness (the GPU decides).**
  - Under the exact statistic, head 3 / query 1137 has thr bits `0x3fc7c230`. That equals key 737's canonical
    |bulk| bits, so the key is refined.
  - SP1.1's CUDA-order reconstruction of production gave thr `0x3fc7c232` (skip).

## D2 census (file:line at b4da646; every z-score statistic site of the APA selectors)

| # | Site (kernel) | Accumulate / thr lines | Entry points | Status |
|---|---|---|---|---|
| 1 | `apa_selective_kernel` pass 1 (fused) | kernels.cu:1625-1659 (sum/sumsq 1635, 1643; var 1658; thr 1659) | `apa_selective_attention` (fused branch) | **CONVERTED** to `apa_x3a_selective_exact_kernel<T,DMAX,WCOOP,SINK=0>` (apa_x3a.cu:195-385). Dispatch is at kernels.cu:2439-2449. |
| 2 | `apa_selective_stats_kernel` (split-K pass-1 stats) | kernels.cu:1892-1926 | `apa_selective_attention` and `_sink`, both through `apa_selective_splitk_dispatch` | **CONVERTED** to `apa_x3a_stats_partial_kernel` (grid rows×P) plus `apa_x3a_stats_finalize_kernel` (apa_x3a.cu:392-478). Launcher: apa_x3a.cu:481-500 and 550-575. Dispatch: kernels.cu:2349-2358. |
| 3 | `apa_selective_sink_kernel` pass 1 | kernels.cu:2490-2531 | `apa_selective_attention_sink` (fused) | **CONVERTED** (same template, SINK=1). Dispatch is at kernels.cu:2750-2759. |
| 4 | `scale_stats_count` (gemm_apa statistics) | gemm_apa.cu:287-315 | `apa_gemm_selective_attention(_cached)`, debug entries | **CONVERTED** to `scale_stats_count_x3a` (apa_x3a.cu:853-949). The flag branch is in `make_bulk` (gemm_apa.cu:628-640). |
| 5 | `apa_selective_fwd_train_kernel` pass 1 | kernels.cu:2811-2843 | `apa_selective_fwd_train`, `apa_selective_train` | **Not converted.** This is the training forward, and its saved thr feeds the backward. No training operand is in the census. It is a successor. B0 uses it as the observation route for the production fused pass-1 thr (the same statements as #1). |
| 6 | bk4 tensor-core training forward | kernels.cu:8538 (accumulate), 8584-8585 (thr) | `apa_selective_fwd_bk4`, `_fwd_train_variant` | **Not converted.** This is the BP-KERNEL-4 training variant; out of census. |
| 7 | `apa_blend_softmax_kernel2` | kernels.cu:3726-3748 | `apa_blend_softmax` (cuBLAS blend path) | **Not converted.** Its input is a materialized bulk tensor reduced over one fixed 256-thread tree, so it has no launch-configuration axis. It has MASK_LIM sentinel semantics and is not in the census. Successor. |
| 8 | `apa_blend_softmax_sink_kernel` | kernels.cu:3885-3920 | `apa_blend_softmax_sink` | **Not converted** (as #7). |
| 9 | `apa_selective_int4_kernel` pass 1 | kernels.cu:5958-5994 | `apa_selective_attention_int4` | **Not converted.** This is the INT4 packed-code family (APAMQ), with its own dot and its own registered gates. It is not in the census (fp32 kq). Successor. |
| 10 | `apa_selective_int4_stats_splitk_kernel` + `_stats_reduce_kernel` (DF2 V4) | kernels.cu:6124-6159, 6165-6196 | INT4 split-K | **Not converted** (as #9). This is production's only partition-dependent fp32 statistic (documented at 6163), so it is the prime successor. |
| 11 | `apa_selective_int4_stats_kernel` (DF1) | kernels.cu:6229-6268 | INT4 V4-off oracle | **Not converted** (as #9). |
| 12 | `apa_int4_sdpa_noncausal_kernel` | kernels.cu:6576-6627 | `apa_int4_sdpa_noncausal` | **Not converted.** Its registered semantics are composed fp16-ladder parity (EXP-APA-2, saturation included). Integer stats would break the K-EQ parity contract. |
| 13 | `attn_noncausal_f16_kernel` | kernels.cu:6930-6986 (Welford), 6989-7046 (ladder16) | INT4 non-causal f16 v2 | **Not converted.** Its registered semantics are shifted Welford (EXP-APA-4) or the ladder, a different formula. |
| - | `apa_sp2_baseline_thresholds` (apa_sp2.cuh:95-115) | - | diagnostic | Not a site: it launches #2's production kernel. |
| - | SP / SP1.1 kernels (apa_sp1_1.cuh) | - | `apa_selective_attention_sp` | No z-statistic (out of scope per the order). |

## D3 arithmetic statement

The source is `tensor_cuda/src/apa_x3a_exact.h`, compiled into the device module and, unchanged, into the CPU
harness.

1. **Statistic input.**
   - a = |bulk| in fp32, where bulk = fp32(dot · scale).
   - The dot is in the **canonical order**. Lane l (0..31) accumulates `fma(q[d], k[d], ·)` over d = l, l+32, …
     (< D), starting from +0. Then the shfl_down butterfly 16, 8, 4, 2, 1 runs, and lane 0's value is used.
   - Explicit `__fmaf_rn` / `__fadd_rn` / `__fmul_rn` are used.
   - WCOOP=true kernels compute this natively. WCOOP=false kernels emulate it serially (`canon_dot_serial`, 32
     register partials).
   - Production WCOOP=true is the same order: `FFMA R2, R2, R3, RZ`, then an FFMA chain at stride 0x80
     (`artifacts/apa_sp1/sp1_1_baseline_sass.txt:457-477`).
2. **Rounding to the grid: E0 = 40.**
   - q = RNE(a · 2^40) as an unsigned 64-bit integer. The fp32 product `a·2^40` is exact (a power-of-two scaling,
     < 2^60). It is converted with `__float2ull_rn`.
   - Why 40:
     - Every a ≥ 2^-17 is on the grid exactly (its 24-bit significand fits), so the moments are the exact moments
       of the fp32 values for all |bulk| ≥ 7.6e-6.
     - Smaller a are rounded once, with error ≤ 2^-41 absolute. That is 2^-18 ulp of a thr near 1, so it is
       negligible.
     - With |bulk| < 2^20 and N ≤ 2^17, E0 = 40 gives S1 < 2^77 and S2 < 2^137: two and three u64 limbs.
     - It is the resolution MUTATIONS X3 proposed.
     - It was derived from the stated bounds, not tuned on results (L20).
3. **Accumulation and limb layout (u64, little-endian).**
   - S1 = Σq is held in limbs (s1[0], s1[1]); S2 = Σq² in (s2[0], s2[1], s2[2]).
   - q² is the 128-bit product (`q*q`, `__umul64hi(q,q)`).
   - Add-with-carry is written with explicit carry compares.
   - Block reduction:
     - per-lane partials go through a warp shuffle reduction, then one slot per warp, summed by thread 0;
     - split-K adds P per-partition partials, merged by the finalize kernel;
     - the gemm twin does the same at 256 threads.
   - Under WCOOP, lane (it mod 32) keeps key `it` in a register and accumulates once per 32 keys. Integer addition
     is associative, so the result is the same.
4. **Bound proof.**
   - For 0 ≤ a < 2^20, the largest in-range fp32 is 2^20 − 2^-4. So q ≤ 2^60 − 2^36 < 2^60.
   - Hence S1 < 2^17·2^60 = 2^77 (s1[1] < 2^13) and S2 < 2^17·2^120 = 2^137 (s2[2] < 2^9).
   - **The limbs cannot wrap for any u64 q and N < 2^63.** Each addend's top limb is 0. A top limb therefore grows
     by at most one carry per addition, plus the partner's top limb and one carry per merge.
   - Finalize: N·S2 < 2^154 and S1² < 2^154 fit three limbs. Dv = N·S2 − S1² ≥ 0 by Cauchy-Schwarz.
5. **Fallback.** Any of the following makes the row take production's fp32 statistic:
   - `kFlagRange`: some a ≥ 2^20 or not finite (NaN fails the `<`);
   - `kFlagCount`: N < 1 or N > 2^17;
   - `kFlagBound`: an end-of-row check (s1[1] ≥ 2^13, s2[2] ≥ 2^9, or a borrow in Dv); never expected.

   The fallback statistic is a verbatim copy of the production pass-1 body, run at the production launch shape
   (same block, WCOOP, thread count). It is counted per row (probe outputs) and in module-local device counters
   (`apa_x3a_fallback_rows()`). In the gemm twin, the copy adds the barrier production lacks; every thread then uses
   thread 0's value, which equals production's `thresholds[row]`.
6. **Threshold.** Each step is one IEEE binary64 round-to-nearest-even operation, in this fixed order:

   | Step | Operation | Rounding |
   |---|---|---|
   | s1d | RN64(S1) | one rounding |
   | dvd | RN64(Dv) | one rounding |
   | n, n2 | (double)N, n·n | exact |
   | meang | RN64(s1d / n) | `__ddiv_rn` |
   | varg | RN64(dvd / n2) | `__ddiv_rn` |
   | sdg | RN64(sqrt(max(varg, 0))) | `__dsqrt_rn` |
   | mean, sd | meang·2^-40, sdg·2^-40 | exact |
   | t | RN64((double)z · sd) | `__dmul_rn` |
   | thr64 | RN64(mean + t) | `__dadd_rn` |
   | thr | RN32(thr64) | `__double2float_rn`, the one rounding to fp32 |

   - RN64 of a ≤192-bit integer takes the top 64 bits and ORs every lower bit into bit 0 (sticky; 11 bits below the
     binary64 round bit). One `__ull2double_rn` then gives the correct decision, followed by an exact `ldexp`.
   - The `_rn` intrinsics forbid FMA contraction. The host harness uses `-ffp-contract=off`.
7. **Emulator** (`tensor_cuda/tests/apa_x3a_reference.py`, NumPy plus Python ints).
   - It computes the same canonical dot. fp32 FMA uses binary64 + TwoSum + round-to-odd, which is the correctly
     rounded fp32 FMA. A constructed double-rounding witness shows that naive binary64→fp32 fails and the emulator
     does not.
   - Then the same integers (20-bit digit sums) and the same fp64 sequence (Python floats, `math.sqrt`).
   - It equals a `fractions.Fraction` oracle and the shared C++ header bit for bit (T-G3).
8. **Edges, all tested in T-G3.**
   - Grid ties round half to even: (k+½)·2^-40 maps to 0, 2, 2, 4, 4, 6, 6, 8.
   - Exact at 2^-17 and above; rounded below it.
   - 2^20 − 2^-4 is in range; 2^20 is out of range.
   - 2^17 keys at the top value give S2 in (2^137 − 2^115, 2^137).
   - S = 1 and all-equal rows: var = 0 exactly, so thr = |bulk|.
   - All-zero: thr = 0.
   - NaN, inf, 3e6 and N = 2^17+1 take the fallback.

## T-G1 receipts (build)

| Build | Command | rc | Wall | sha256 | Bytes |
|---|---|---|---|---|---|
| **final (T-G1)** | `bash artifacts/x3a/build_engine.sh tensor_cuda artifacts/x3a/build/final`, clean build dir | 0 | 95 s | `eab547e99433ea68404773b7a4ec9494b5518595798ffe3005d909bfdad5a29d` | 14,305,384 |
| pristine control (b4da646 sources, scratch copy) | same command on `/tmp/x3a_base1/tensor_cuda` | 0 | 72 s | `87403e85459212e925dbfa78e4192489369a6b808abe9cfb3c14d0b4be203be3` | 9,582,576 |
| power-check mutant | same command on `/tmp/x3a_mutant/tensor_cuda` | 0 | 93 s | `1c04612701f124dd2f015fccf4f49b8ce9b8e09612a9ce3f5b1ef0cba98d2b03` | 14,305,384 |
| r1 to r6 (development) | incremental | 0 | 1-90 s | r1 `08c0f9c1…`, r2 `fd77e1cf…`, r3 `080ba66d…`, r4/r5 `df3c52d3…` (r5 stale: see T-G2), r6 `c90d5244…` | - |

- **Toolchain:** nvcc 12.6 V12.6.85, gcc 13.3.0, cmake 3.28.3, Python 3.12.3, offline pybind11 v2.12.0.
- **The copied helper:**
  - It is `artifacts/x3a/build_engine.sh`.
  - Only two changes from TX3-F1's: the pybind11 source is read from `/tmp/x3a_deps`, and TMPDIR defaults to
    `/tmp/x3a_tmp`.
- **Build change:** `tensor_cuda/CMakeLists.txt` gains the object library `tc_x3a`:
  - its CUDA_SEPARABLE_COMPILATION is OFF;
  - its object goes to the host link only, through `$<HOST_LINK:$<TARGET_OBJECTS:tc_x3a>>`;
  - `LINK_DEPENDS` is set, because without it the module was not relinked after a source change (r5).
- **Warnings:** one compiler warning, the pre-existing `kernels.cu(3510) lane declared but never referenced`. The
  pristine build has it at line 3490.
- **Receipts:** `artifacts/x3a/build/*/{result,toolchain,source_sha256}.txt`.

## T-G2 receipts (SASS)

- **Tool:** `artifacts/x3a/tools/sass_compare.py`, TX3-F1's comparator.
  - Two additions: `--allow-new` (new functions are listed, never failed) and `diff-all` (`-all` cubins matched by
    function-name set, because the new module's fatbin shifts cubin ordinals).
  - Normalization is TX3-F1's: module-ID symbols only.
  - cuobjdump 12.6 (the build toolkit).

| Comparison | Production functions | Identical | Instructions | Verdict | Receipt |
|---|---|---|---|---|---|
| production vs pristine control, linked | 563 | 563 | 501,496 | IDENTICAL | sass/production_vs_base1_linked.txt |
| **production vs final, linked** | 563 | 563 | 501,496 | **PRODUCTION_FUNCTIONS_IDENTICAL** (+103 new) | sass/production_vs_final_linked.txt |
| **production vs final, -all** | 1,128 | 1,128 | 1,003,200 | **PRODUCTION_FUNCTIONS_IDENTICAL** (+1 new cubin, 103 functions) | sass/production_vs_final_all.txt |
| power: production vs mutant, linked | 563 | 550 | 501,496 | DIFFERENT: exactly the 13 `apa_selective_stats_kernel` instantiations | sass/production_vs_mutant_linked.txt |
| power: final vs mutant, linked | 666 | 653 | 696,776 | DIFFERENT (the same 13) | sass/final_vs_mutant_linked.txt |
| power: production vs mutant, -all | 1,128 | 1,102 | 1,003,200 | DIFFERENT (13 linked + 13 relocatable) | sass/production_vs_mutant_all.txt |

**How the identity was reached** (ledger 11:10 to 12:20; receipts in `sass/`). The development builds failed three
times, and each failure had a different cause.

1. **r1: new kernels inside kernels.cu.** 75 production functions differed.
   - Mechanism: nvlink lays out a kernel's shared arrays by size. The tie order between equal-size arrays (`qsh` and
     `osh`, 0x200 bytes each) flipped once the device link held new symbols, for example
     `STS [R6.X4]` → `STS [R6.X4+0x200]`.
   - The relocatable cubin was identical, so the change happened at device-link time.
   - gemm_apa.cu's relocatable "differences" were only cuobjdump column padding.
2. **r2/r3: the new code moved to a non-rdc module.** The 4 `bk3::backward` instantiations still differed.
   - First cause: CMake still passed the object to `-dlink`.
   - Second cause, after the object moved to the host link only: symbol order. My B0 helper referenced production
     stats-kernel instantiations earlier than production's first use. That moved their symbols (0x2e7 → 0x4fa) and
     flipped bk3's five 0x1000-byte shared-array ties.
   - A pristine rebuild matched 563/563, which shows the flip was caused by my change and is not build noise.
   - Fix: the helper moved to the end of kernels.cu and launches only instantiations production already emits.
3. **r4 onward: identical.**

## T-G3 receipts (CPU tests)

- **The command form (L01):** `CUDA_VISIBLE_DEVICES= TMPDIR=/tmp/x3a_tmp OMP/MKL/OPENBLAS_NUM_THREADS=1
  PYTHONPATH=tensor_cuda python3 -B -m pytest -p no:cacheprovider -q …`
- **`tensor_cuda/tests/test_apa_x3a_cpu.py`, fresh on the final build: 23/23 passed.**
  - 15 passed in 65 s (`cpu/T-G3_x3a_cpu_fast.{xml,txt}`).
  - 8 passed in 345 s (`cpu/T-G3_x3a_cpu_random.{xml,txt}`).
- **What the 23 tests check:**
  - **Every operand** (10^4 random rows in 8 cells, the 8,192-row SP1.1 witness tensor, 8 constructed edge rows):
    the emulator's S1 and S2, and the rational mean and var, equal a `fractions.Fraction` oracle. The oracle's var is
    the two-pass definition, a different formula from N·S2 − S1².
  - The emulator's thr equals an independent NumPy-float64 D3 sequence **and** the shared C++ header compiled for
    the host, bit for bit, on every row.
  - **The witness:** the rebuilt tensor reproduces SP1.1's CUDA-order |bulk| bits at all 5 receipt witness keys.
  - **The FMA emulator:** fp32 FMA equals exact rational rounding on 20,000 random triples. Two constructed
    double-rounding witnesses: naive binary64→fp32 is wrong, the emulator is right, for c > 0 and c < 0.
  - **The canonical dot:** NumPy equals C++ `fmaf` on 20,000 rows, D ∈ {1, 33, 64, 96, 128}.
  - **The limbs:** wide-integer → binary64 equals Python `float(int)` on 5,000+ random values and every RNE tie
    pattern from 2^54 to 2^191. Sequential and pairwise-tree limb sums equal Python ints, including full-u64 carries.
  - **Edges:** every edge behaves as stated (D3 item 8).
  - **The flag parser** of the built module: unset or "0" is OFF, "1" is ON, anything else raises.
  - **The header constants** equal the emulator's.
- **Tooling power checks** (`artifacts/x3a/gpu/test_x3a_tooling_cpu.py`): 15/15, 119 s
  (`cpu/tooling_power_checks.{xml,txt}`).
  - The control verdict passes.
  - Each of these one-mutation copies of the audit run must make the verdict FAIL with its rule tag:
    - one thr bit;
    - one fallback flag;
    - fallback vs production;
    - a gemm score;
    - the real-entry fallback count;
    - an X2 sample count;
    - X3 at +5%;
    - a constant B0 array;
    - a missing capture;
    - a JUnit skip;
    - the flag ON at session start;
    - the wrong binary;
    - a real verdict on constructed captures.
  - lane.sh refusals, the label resolution and the unmasked status are also checked.
- **Full CPU suite: RED as a literal bar, with no regression.**
  - Run: `tensor_cuda/tests` (74 files in 2 chunks) plus `tests/`, on the final build and on the pristine baseline
    build with the same files (`cpu/full_suite_chunk{1,2,3}.*`, `cpu/baseline_suite_chunk{1,2,3}.*`,
    `cpu/suite_comparison.txt`).
  - Per-test outcomes are identical over 2,599 test records: 593 passed, 450 skipped, 1,541 failed, 15 errors in
    both.
  - There is also 1 pre-existing collection error, `test_selector_accuracy.py`, which parses `sys.argv` at import.
  - The failures are GPU tests that have no CPU skip guard, plus stale source pins.
  - The source-reading tests re-run in a scratch copy holding the **production** sources give identical outcomes
    (`cpu/source_pin_tests_production_sources.xml`).
  - **Pin census** (`cpu/source_pin_census.txt`):
    - Every registration that pins kernels.cu or bindings.cpp is already stale on production.
    - My edits additionally invalidate still-valid gemm_apa.cu pins (SP3, SP4G, SP5, SPD1, MF2.2 probes,
      BP-KERNEL-2/3/4) and CMakeLists.txt pins (BP-KERNEL-2/3/4).
    - Every one of those registrations also pins an already-stale file, so no check that passes on production now
      fails.

## T-G4: staged GPU lines and registered per-lane counts

- **The file:** `artifacts/x3a/gpu/STAGED_GPU_COMMANDS.sh`.
  - Run each line in order, as written, from any shell. `RUNS` defaults to
    `$W/artifacts/x3a/gpu_runs/run1`, a fresh directory.
  - The lead's environment must expose the GPU.
  - Neither `TC_APA_EXACT_STATS` nor `X3A_DEVICE` may be set.
- **The lines** (each lane has `forge-gpu-run --max 600` and `timeout 540`):
  ```
  G0       python3 -B artifacts/x3a/gpu/g0.py $RUNS                      (fresh dir, sha256 pin eab547e9…, governor offset)
  B0       pytest test_apa_x3a_gpu.py -k test_b0_                         22 tests, 22 captures   (measurement)
  X1a      pytest test_apa_x3a_gpu.py -k test_x1_exact                    17 tests, 17 captures   (rule)
  X1b      pytest -k "test_x1_real_entry or test_x1_sink or test_x1_gemm" 30 tests, 30 captures   (rule)
  X2       pytest -k test_x2_                                             48 tests, 48 captures   (measurement)
  X3       pytest -k test_x3_                                              6 tests,  6 captures   (rule: <= 3% each)
  X4       pytest test_apa_selective.py test_apa_selective_splitk.py test_apa_value_dim.py test_apamq_sb1.py test_apa_sp2.py
                                                                          69 tests, all passed     (rule)
  VERDICT  python3 -B artifacts/x3a/gpu/verdict.py $RUNS                  (CPU; exit status = verdict)
  ```
- **The registry:** `artifacts/x3a/gpu/test_registry.txt`, sha256
  `33525b199c19ab8423f347fcec4014630a724ccf4ce36c64707b1b5af3581832`, pinned in `verdict.py`.
  - It was generated from real collection of the staged lines (`test_registry.py`).
  - Every outcome is "passed"; no registered skips.
- **Registered per-lane rules (the verdict, fail-closed):**
  - **X1 (0 mismatches).** Every exact configuration's thr equals the NumPy emulator's on every exact-path row. The
    configurations:
    - fused WCOOP 1/0;
    - stats WCOOP 1/0 × threads 128/256 × parts 1/2/4/8;
    - the real entry;
    - the sink;
    - the gemm twin.

    The emulator is recomputed by the verdict from the regenerated operands. The sets are the 10^4 random rows, the
    witness tensor and the 8 edge rows.
  - **Fallback counts (L04).** Exactly 1 row in every configuration for `edge_overflow_forcing`, and exactly 1 for
    `gemm_overflow_forcing`. 0 for every other set. The fallback rows' thr equals production's at the same launch
    shape (L32).
  - **X3.** For each of 6 cells: overhead = median(ON)/median(OFF) − 1 ≤ 0.03. The cells are selective prefill
    D64/D128 (L=S=8192, H=8, KVH=2, causal, fused), selective decode D64/D128 (L=1, H=32, KVH=8, S=8192, split-K)
    and gemm_apa prefill D64/D128 (BF16).
    - The arms are interleaved ABBA in the same lane, with the transient pool ON.
    - The plan: prefill 2 warm + 8 rounds × 1 call; decode 5 warm + 40 rounds × 20 calls.
  - **X4.** The JUnit identities and outcomes equal the registry; all 69 pass.
- **Measurements (no pass rule):**
  - B0: production thr mismatch counts between every pair of production configurations, and against the X3a
    emulator and the SP1 fp32 formula. The configurations are the fused (fwd_train) dispatch and the stats kernel at
    every production instantiation (D=64: WCOOP 0/1; D=128: WCOOP 1) × 128/256 threads.
  - X1: the native-order integers (mode 2).
  - X2: differing refine decisions / visible keys over the 48 SP1 regression shapes. The verdict re-derives the
    decision counts and the exact thr on 16 sample rows per shape (a hard rule).
  - Predictions P17-P19 are printed.
- **The protocol (TX3-F1C lessons):**
  - G0 creates a fresh run directory with a sha256 pin.
  - lane.sh resolves `@X3A_LABEL@` to a run/lane/attempt label and exits with the test's status.
  - x3a_lane_plugin refuses a session that starts with the flag set. It records the flag as env and as the C++
    parse at start and end (L53), the device mode and the imported `.so`.
  - The verdict binds governor rows by label, checks chronology, JUnit identities against the registry, the binary
    and the lane state, and recomputes every count from the captures (L44).
- **The L48 CPU audit** (`artifacts/x3a/gpu/cpu_audit.py`; receipts in `artifacts/x3a/audit/`). It runs the
  staged text as written, through the real lane.sh, with a governor stand-in that uses the real governor's parser
  and row writer.
  - **Pass A** (passthrough, no device): every lane skips "CUDA unavailable". X4's existing suites fail on CPU. The
    staged VERDICT **FAILS** (132 failures), as it must.
  - **Pass B** (constructed device answers; everything after the device is real):
    - all 7 lane lines exit 0;
    - the staged VERDICT **FAILS** (128 failures, every capture "device constructed");
    - the same verdict with `--audit` **PASSES**: 123/123 captures, 0 failures. This is the control.
- **Expected walls:** not measured (the seat has no GPU). Each lane is bounded at 540 s by `timeout` and 600 s by
  the governor. The verdict took about 1 minute of CPU time on the audit run.
- **What the lead must know about git:** `.gitignore` drops `*.json` and `*.log`. The run directories carry
  `run_manifest.json`, `lane_state.json`, `loaded_binary.json` and `run.log`, the TX3-F1 protocol names. Use
  `git status --ignored` when committing receipts. The committed registry is `.txt` for this reason.

## Prior art

- **Exact fixed-point accumulation** (Kulisch long accumulator; U. Kulisch, 1980s, "Computer Arithmetic and
  Validity").
  - Taken: round each summand once onto a fixed grid and add exactly in a wide integer.
  - Ours: E0 = 40 from the APA bounds, u64 limbs and the fallback rule.
- **Reproducible summation** (Demmel and Nguyen, ReproBLAS; ARITH 2013, IEEE TC 2015).
  - Same goal: bitwise-equal results for any order or partition.
  - Different method: they pre-round against data-dependent extractors in floating point.
- **Variance cancellation** (Higham 2002, "Accuracy and Stability of Numerical Algorithms"; Welford 1962;
  Chan-Golub-LeVeque 1983).
  - Taken: the diagnosis.
  - Not taken: their recurrences. Here N·S2 − S1² is formed exactly, so there is no cancellation to control.
- **FMA emulation:** round-to-odd (Boldo and Melquiond, IEEE TC 2008) and TwoSum (Knuth, TAOCP vol. 2;
  Møller 1965). Taken as published for the NumPy emulator.
- **Shuffle reductions** (NVIDIA, Luitjens 2014): the canonical order is production's.
- **Guard/round/sticky rounding** (IEEE 754; Muller et al., Handbook of Floating-Point Arithmetic).
- **Split-K partials then finalize:** APAMQ DF2 V4's stats layout. Taken: the layout. Ours: integer partials.
- **Lane protocol, verdict, registry, SASS comparator and governor stand-in:** TX3-F1/F1C, taken with names
  changed. The additions are ours: `--allow-new`, `diff-all`, lane state and the constructed device.
- **Test doubles:** Meszaros 2007. **Mutation testing:** DeMillo, Lipton and Sayward 1978.
- **Mine, as far as I know (no prior art known to me):**
  - pinning the statistic's |bulk| to one canonical dot order;
  - register-slot staging of the WCOOP keys.
- **Unverified in the sandbox. Lead to check these search terms:**
  - "Kulisch long accumulator exact dot product";
  - "ReproBLAS Demmel Nguyen reproducible summation";
  - "Boldo Melquiond rounding to odd FMA emulation";
  - "TwoSum Moller Knuth";
  - "nvlink shared memory variable layout order";
  - "CMake HOST_LINK TARGET_OBJECTS".

## Disclosures (rules broken or near misses, however small)

1. **Ledger time stamps.** The entries stamped 10:30 to 14:00 were written as estimates, not read from the clock,
   and they ran up to about 2.7 h late. The ledger appends a correction table built from file mtimes. Every later
   stamp was read with `date`.
2. **Scratch over its cap.** `/tmp/x3a_tmp` reached 9.9 GB (the order caps `/tmp/x3a_*` at 5 GB).
   - Cause: the tooling mutants copied a roughly 600 MB audit run into pytest's tmp_path.
   - It was cleared (under `/tmp`, exempt) and the capture size was cut. The audit was then redone in
     `/tmp/x3a_audit2` (69 MB).
   - Current scratch total: about 1.8 GB.
3. **A delete-guard refusal.** One Bash heredoc was refused by the delete guard because it contained the text
   `os.remove` inside a string I was replacing. Nothing was deleted. The edit was redone with the Edit tool.
4. **Two test-design errors on my side,** both fixed before the final receipts: the first double-rounding witness's
   c < 0 case, and one power-check tag string.
5. **Development builds r1 to r6** were incremental, and r5 was stale (not relinked). The T-G1 build is clean.
6. **Removals.**
   - `tensor_cuda/src/apa_x3a_kernels.cuh` was my own first-draft file from this session. It was moved to DELETE_ME
     with `delete-me` (id `20261028-104008_x3a-seat_apa_x3a_kernels.cuh`).
   - Under `/tmp` (exempt): the superseded first audit root and pytest tmp trees.
7. **No git, GPU, subagents or background waits.** I killed no process. I edited no receipts committed by others.
   - Every command ran in the foreground under an explicit timeout below 10 minutes.
   - The sanitization of `$HOME` and `$USER` touched only my own receipt files under `artifacts/x3a`.
8. **Images rewritten by an existing test.** The T-G3 full-suite runs (mine and the baseline's, same files) executed
   `test_terrain_render.py`, which rewrites 13 images in `artifacts/` (wo12g_*, wo8a_*, wo8c_*, wo9ae_*,
   wo9e_*.ppm). Their content is byte-identical to production's copies (sha256 compared); only mtimes changed. They
   are not my deliverables and there is nothing to restore.

## Disposable scratch (all under /tmp, about 1.8 GB)

- `/tmp/x3a_deps`: a pybind11 v2.12.0 copy for the offline build. Needed only to rebuild.
- `/tmp/x3a_tmp`: harness binaries, registry dry-runs, constructed-run captures (`cap_try`), SASS and ELF dumps,
  and pytest tmp.
- `/tmp/x3a_base1`: a pristine b4da646 rebuild (the SASS and CPU-suite baseline).
- `/tmp/x3a_mutant`: the power-check mutant build.
- `/tmp/x3a_oldbuild_r1_r6`: the incremental build directory and the r6 `.so`.
- `/tmp/x3a_suitebase`: a worktree copy with production sources (the source-pin control).
- `/tmp/x3a_audit2`: the final L48 audit runs. The tooling tests default to its `B_run`.
- In the worktree, git-ignored: `tensor_cuda/build/`, and the T-G1 `.so` (the binary the GPU lanes pin).

## Every file written

All in the worktree, except `/tmp` scratch and the one DELETE_ME move.

- **Source, modified** (host-only diffs vs b4da646 for the `.cu`/`.cpp` files): `tensor_cuda/src/kernels.cu`,
  `tensor_cuda/src/gemm_apa.cu`, `tensor_cuda/src/bindings.cpp`, `tensor_cuda/CMakeLists.txt`.
- **Source, new:** `tensor_cuda/src/apa_x3a.cu`, `tensor_cuda/src/apa_x3a_exact.h`, `tensor_cuda/src/apa_x3a.h`.
- **Binary, new (git-ignored):** `tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`
  (sha256 `eab547e9…`).
- **Tests, new:** `tensor_cuda/tests/apa_x3a_reference.py`, `apa_x3a_operands.py`, `apa_x3a_device.py`,
  `test_apa_x3a_cpu.py`, `test_apa_x3a_gpu.py`.
- **Gate tooling, new** (`artifacts/x3a/gpu/`): `STAGED_GPU_COMMANDS.sh`, `g0.py`, `lane.sh`, `expected_binary.py`,
  `binary_record.py`, `x3a_lane_plugin.py`, `test_registry.py`, `test_registry.txt`, `verdict.py`, `cpu_audit.py`,
  `test_x3a_tooling_cpu.py`, `synthetic/fake_governor/forge-gpu-run`, `synthetic/x3a_constructed_outcome.py`, and
  `registry_receipts/` (7 files).
- **Tools, new:** `artifacts/x3a/build_engine.sh`, `artifacts/x3a/tools/sass_compare.py`,
  `artifacts/x3a/harness/x3a_host_harness.cpp`.
- **Receipts, new:**
  - `artifacts/x3a/FINAL_REPORT.md` and `artifacts/x3a/LEDGER.md`.
  - `artifacts/x3a/build/`: 43 files (r1-r6, final, mutant_powercheck, pristine_control). `build.log` and
    `preseed.log` are git-ignored.
  - `artifacts/x3a/sass/`: 27 files. `production_vs_r3_all.txt` is the ordinal-keyed comparison that showed why
    `diff-all` was needed.
  - `artifacts/x3a/cpu/`: 21 files.
  - `artifacts/x3a/audit/`: 14 files.
- **Moved to DELETE_ME:** `tensor_cuda/src/apa_x3a_kernels.cuh` (my draft).
