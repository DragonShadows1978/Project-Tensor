<!-- Filed by the lead verbatim from the r1 seat's final message (the harness refused the seat's own write). -->

RED: G1 fails on PF-I3, so I stopped there under L35 and did not run G2 to G5, G7, the rest of G6, or the speed line. PF-I3's 1e-5 cap is smaller than the difference between prefill and decode outputs even when every key is refined, so it cannot hold as written. Everything else measured in G1 passes. By the registered rule the round is not ACCEPT, and only the lead can amend PF-I3.

The harness refused my write of `artifacts/apa_p1pf2/REPORT.md` ("Subagents should return findings as text"), so this message is the report; please file it verbatim. Wall time was about 1 h 40 min (14:32 to 16:12 EST). GPU lines took 3,696 s, about 1.03 h of the 5 h cap.

## Why PF-I3 cannot hold
- **The kernel matches dense prefill, not decode.** PF-I2 shows the new kernel's output is bitwise equal to the dense prefill kernel's when every key is refined (4,848 of 4,848 cases).
- **Dense prefill and packed decode accumulate differently.** Prefill runs an fp32 online softmax with `__expf` and a per-key rescale. Decode sums chunks in double for fp32 and then merges partials. bf16 outputs are also rounded to bf16 at the final store.
- **The resulting gap exceeds the cap with identical decisions.** Prefill against decode with every key refined, worst row (relative), from the G1 parts:

| dtype | S 255 | S 256 | S 257 | S 2047 | S 2049 | S 8192 |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 3.96e-6 | 4.89e-6 | 4.94e-6 | 1.12e-5 | 1.28e-5 | 3.91e-5 |
| bf16 | 2.68e-3 | 3.38e-3 | 2.82e-3 | 3.05e-3 | 3.37e-3 | 2.94e-3 |

- **The gap is accumulation order, not decisions.** At the registered delta the worst rows are about the same: fp32 S 8192 4.11e-5, bf16 S 2049 3.17e-3.
- **Result:** τ = min(4 × 3.38e-3, 1e-5) = 1e-5. 131,705 of 1,398,320 rows are above τ.
- **Informative only, not a verdict and not registered:** with the cap removed, τ would be 1.35e-2 and 0 rows would exceed it. An fp32-only τ would be 1.56e-4, also with 0 rows above.
- **Options for the lead:** a per-dtype τ without the cap, or a cap that scales with the key count.

## Gates
- **G1: FAIL.**
  - **PF-I1 (decisions against packed decode, bitwise): PASS.**
    - Masks: 1,398,320 oracle rows and 21,912,774,758 decisions, 0 rows unequal. Mask tails beyond each row's visible keys are 0 on every row (0 nonzero). No prefill mask is uninformative.
    - Exact-tie rows: 32 of 32 pass (threshold equal to the bulk score, one ulp above and one ulp below).
    - Store-prefix premise: a store filled with only the first smax keys has the same codes and norms as the larger store's prefix (all 16 checks).
    - Bulk bits on Build PFH: 21,912,774,758 values on uint32 views, 0 rows unequal.
  - **PF-I1n (non-finite inputs): PASS.**
    - 192 of 192 cases: NaN, +Inf and −Inf in k, a 1e20 k element that overflows the norm, NaN in q, NaN in v.
    - Prefill masks equal the decode oracle's on every row.
    - The stored norm is NaN in all 128 k cases.
    - The non-finite row sets equal dense SP prefill's in all 192 cases (7,032 rows on each side), and 0 rows are finite where dense is non-finite.
  - **PF-I2 (output against the dense prefill kernel, every key refined): PASS.** 4,848 of 4,848 cases bitwise, 3,749,744,640 elements, all refined on both sides and informative. The registered fallback was not needed.
  - **PF-I3: FAIL**, as above.
- **G2: NOT RUN** (stopped). `scripts/apa_p1pf2_g2.py` is written and has not been run.
- **G3: NOT RUN** (stopped).
- **G4: NOT RUN** (stopped).
- **G5: NOT RUN** (stopped).
- **G6: INCOMPLETE.** The object-code part passes (below). LF1, P1-D L1/L2/L4/L6 and the golden and pin checks were not run.
- **G7: NOT RUN.** The 16 mutants are registered in `MUTANTS.json`; none was staged, built or run.
- **G8 (scope): PASS so far.**
  - Writes are only in the writable set: 3 tracked files changed, all inside their allowed regions, plus 1 new kernel file. `__init__.py`, `kernels.cu` and `include/` are unchanged (checked with `git diff --stat`).
  - GPU use was only through `forge-gpu-run`, each line after a dry run that returned 78.
  - Git use was read-only. Nothing was deleted.

## Diff summary
4 product files, +324/−11 (`git diff --stat` +58/−11 on the 3 tracked files, plus the new 266-line kernel file). Prior art is annotated at each code site.
- **New `tensor_cuda/src/apa_spo/packed_prefill.cuh` (266 lines):**
  - The kernel `spq_pf::prefill<T,CAP,DIAG,SEQ>` is the dense prefill kernel's source with one substitution: each key's bulk score comes from its 4-bit codes in rotated space, using packed decode's per-pair arithmetic. Lane order is used for D128, and for D64 when G is not 4 or 8 and VD > 64. Otherwise one sequential fused chain runs, computed 32 keys per warp.
  - The query row is rotated in the kernel, in `spq1::rotate`'s order.
  - A bulk dump exists only under `TC_APA_P1_TEST_HOOKS`.
  - The host entry `apa_spq_prefill_run` raises every refusal before any allocation or launch.
  - A global-scope bridge `apa_p1pf2_packed_prefill` follows eb3.cuh's pattern, because `tc/ops.h` is outside the writable set.
- **`packed_launcher.cuh` (+6):** an include line only.
- **`bindings.cpp` (+19):** a `packed_prefill` binding inside the `_tensor_cuda_spo2` module, declared at block scope.
- **`apa_packed.py` (+44/−11):** the L > 1 branch replaces the R-E refusal and calls `spo2.packed_prefill`. The route record is backend `spq_packed_prefill_tq4_row` with the pinned packed delta and e_q. The docstring is updated.
- No knob added. No existing kernel, test or numeric result changed.

## Modules and SASS (with the registered mappings)

| Build | Module sha256 |
|---|---|
| Build F3 (before-build) | `2e557a23eaf228512a2f1a2f0629401d91d02d05705087661be5c48849c0f3e9` |
| Build PF (`build_PF`, PASS, 205 s, `source_unchanged_during_build: true`) | `c3c27f8651d82390aac89d62ef704e20dade16d38e9ac8f6cf0e3fb456f4f831` |
| Build PFH (hooks, `build_PFH`, PASS) | `e54b39c355ff584ed1f20b27f410819641c933fbf1b4c1c6a35e4dbe6dd0d4ee` |
| Mutant builds | none built |

- **Registrations:**
  - `EXPECTED.json` sha256 `ef2aa16a40fb61a851ab531eb516192b53bbe8aac88d0e6ead880da02bf69b3e`. Written once, before the first PF build, and never amended.
  - `REGISTRATION.json` sha256 `3a201384382366cc7a3ae1da7bfec83fbf97e1ea05d5e2a4d524e4fa8e7d60da`, written before any GPU line.
  - `MUTANTS.json` sha256 `4771bd3ac41d3c3e44110c5cae2dafdf660503678b82d618c4716d8dcde311cd`.
- **`SASS_COMPARISON_PF.json` (sha256 `781b2a3a98351a139252a3c864e83c2ad1be5b1832bffe6c81a6bee2869e42ee`): PASS.**
  - 813 kernels: 781 equal, 0 changed, 0 removed, 32 added. The 32 added are exactly the registered `void tc::spq_pf::prefill<{float,__nv_bfloat16}, {64,128,256,512}, {false,true}, {false,true}>`.
  - Unexpected differences: 0. Registered but not seen: 0. Symbol renames among paired kernels after mapping: 260.
- **Mappings, applied in names and bodies:**
  - Path-derived names: `<letters><digits>_GLOBAL__N__<8hex>_`, `_GLOBAL__N__<8hex>_`, `__nv_static_<digits>__<8hex>_`.
  - Build directory: `_cu_<8hex>[_<n>]`.
  - The mapped keys were asserted unique per build.
- **Receipt checks:**
  - A fresh F3 dump equals F3's own dump (`e1b49ad9…`). PF dump sha256 `78df2d2c62234af33b583b4741b1daa55facb97ceeab018fb9c9f972db624756`.
  - Flags are equal to F3's. The adapter replacements are equal after normalizing the worktree name; the raw strings embed each worktree's path, so the build receipt records `same_adapter_replacements_as_F3: false`.

## G1's per-case counts (`gpu/G1/RESULT.json` `726d9a51…`, `gpu/G1H/RESULT.json` `1d5e3e7a…`)

| Class | Cases | Oracle rows | Decisions | Mask rows unequal | PF-I2 bitwise cases | PF-I2 elements |
|---|---:|---:|---:|---:|---:|---:|
| grid D64 fp32 | 1,152 | 329,406 | 5,280,751,968 | 0 | 1,152 | 611,297,280 |
| grid D64 bf16 | 1,152 | 329,401 | 5,279,632,534 | 0 | 1,152 | 611,297,280 |
| grid D128 fp32 | 1,152 | 329,404 | 5,281,100,158 | 0 | 1,152 | 1,222,594,560 |
| grid D128 bf16 | 1,152 | 329,405 | 5,279,640,778 | 0 | 1,152 | 1,222,594,560 |
| D64/VD128 fp32 | 96 | 38,016 | 393,491,400 | 0 | 96 | 36,495,360 |
| D64/VD128 bf16 | 96 | 38,016 | 393,491,400 | 0 | 96 | 36,495,360 |
| D128/VD256 fp32 | 12 | 1,168 | 1,166,630 | 0 | 12 | 1,495,040 |
| D128/VD256 bf16 | 12 | 1,168 | 1,166,630 | 0 | 12 | 1,495,040 |
| D128/VD512 fp32 | 12 | 1,168 | 1,166,630 | 0 | 12 | 2,990,080 |
| D128/VD512 bf16 | 12 | 1,168 | 1,166,630 | 0 | 12 | 2,990,080 |

Bulk-bit rows unequal on Build PFH: 0 in every class. The D64/VD128 sub-grid reaches the D64 lane order, and VD 256/512 reach the CAP 256/512 instantiations; both are additions to the design grid.

## G2's per-layer table
Not run (stopped).

## G3's per-model, per-population table
Not run (stopped).

## G4's refusal rows
Not run (stopped). fp16 is registered as a refusal row, not a grid case, because the packed store refuses fp16 K.

## G5's memory figures
Not run (stopped). The method is registered in `REGISTRATION.json` g5.

## Mutant table
All 16 are registered in `artifacts/apa_p1pf2/mutants/MUTANTS.json`, each with its one change, the case subset to run, its expected failing assertion and its executed-line note. None was staged, built or run. m13 is to be built on Build PFH, because its named kill is the hooks bulk bits.

| # | Name | Expected failing assertion | Result |
|---|---|---|---|
| 1 | m01_partition_restart | pf_i1_mask_equal | not run |
| 2 | m02_gt_for_ge | pf_i1_ties | not run |
| 3 | m03_dense_delta | g4_route_record_packed_delta | not run |
| 4 | m04_causal_bound | pf_i1_mask_equal | not run |
| 5 | m05_kv_head_mod | pf_i1_mask_equal | not run |
| 6 | m06_norm_dropped | pf_i1_mask_equal | not run |
| 7 | m07_nibble_swap | pf_i1_mask_equal | not run |
| 8 | m08_rotation_transposed | pf_i1_mask_equal | not run |
| 9 | m09_s_stride | pf_i1_mask_equal | not run |
| 10 | m10_refined_keeps_bulk | pf_i2_equal | not run |
| 11 | m11_exact_from_codes | pf_i2_equal | not run |
| 12 | m12_unrefined_dropped | pf_i3_rows_within_tau | not run |
| 13 | m13_rotation_order | pf_i1_bulk_equal | not run |
| 14 | m14_sinks_omitted | pf_i2_equal | not run |
| 15 | m15_silent_fallback | g5_no_key_shaped_transient | not run |
| 16 | m16_nan_norm_as_zero | pf_i1n_ok | not run |

## Speed line
Not run (stopped).

## Each GPU line's result
Receipts are in `artifacts/apa_p1pf2/gpu/`. Every line ran through `forge-gpu-run --min 30` after a dry run that returned 78, with the inner job in its own scope (MemoryMax 12G). The device was an RTX 4070 SUPER with 12 GB.

| Line | Exit | Wall | Log sha256 | What it did |
|---|---|---:|---|---|
| G1a_r1 (PF, cases 0–834) | 0 | 249 s | `48f6fdeb…` | part complete |
| G1b_r1 (PF, from 835) | 1 | 450 s | `de3ee51d…` | CUDA OOM at about case 1,935; no part saved |
| G1b_r2 (PF, 835–4170) | 0 | 1,471 s | `417043a1…` | part complete |
| G1c_r1 (PF, 4171–4847 plus ties, prefix store and PF-I1n) | 0 | 416 s | `85d0f564…` | part complete |
| G1Ha_r1 (PFH, 0–4847 plus ties and prefix store) | 0 | 1,110 s | `140de486…` | complete |

No line exited 124 or 76.

## Deviations
1. The harness refused the `REPORT.md` write, so this message is the report.
2. **Own-tool slips (L56a):**
   - One foreground wait had no explicit timeout; the harness moved it to the background, and it completed.
   - One wait used an unanchored `pgrep` that matched its own shell, which cost about 580 s idle. Later waits use an anchored pattern.
3. **Script exceptions (L56b), two repairs (the limit is 3):**
   - The SASS dump's TMPDIR did not exist, so cuobjdump exited 1. The 114-byte partial dump was renamed, not deleted, to `sass/F3.sass.failed_r1_no_tmpdir` and its log to `logs/sass_PF_r1_failed.out`. The rerun passed.
   - G1b_r1 ran out of memory with three full masks alive at once. Fixed by checking one mask at a time, freeing caches after large cases, and saving the part on an exception. Its work after case 835 was rerun in G1b_r2.
4. **Registered case choices (they add cases and narrow no gate):**
   - Two VD sub-grids added.
   - fp16 moved to refusal rows.
   - Tie rows adapted from the block-tie fixture to SP threshold ties, using an fp32 delta search.
   - The hooks bulk check runs each oracle row as an L = 1 causal prefill where the full bulk dump would exceed 2 GiB.
   - PF-I1's oracle uses the same store's first smax keys, with that premise checked on 16 cases.
5. **The L35 stop:** I completed both G1 lines so every G1 clause has a verdict, then stopped before any other gate.
6. Build PF's `same_adapter_replacements_as_F3` is false only because the strings embed each worktree's path; they are equal after normalizing the worktree name.

## Every file written
- **Product:**
  - `tensor_cuda/src/apa_spo/packed_prefill.cuh` (new)
  - `tensor_cuda/src/apa_spo/packed_launcher.cuh`
  - `tensor_cuda/src/bindings.cpp`
  - `tensor_cuda/tensor_cuda/apa_packed.py`
- **Scripts** (all run as `python3 -B -P`; none shares a standard-library module name):
  - `scripts/apa_p1pf2_build.py`
  - `scripts/apa_p1pf2_gpu.py`
  - `scripts/apa_p1pf2_line.py`
  - `scripts/apa_p1pf2_sass.py`
  - `scripts/apa_p1pf2_mutants.py`
  - `scripts/apa_p1pf2_g2.py` (not run)
- **No `tensor_cuda/tests/test_apa_p1pf2_*.py` was written.**
- **`artifacts/apa_p1pf2/` (647 MB, under the worktree root `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1pf2`):**
  - `INSTRUMENTS.log`
  - `EXPECTED.json` and `.sha256`
  - `REGISTRATION.json` and `.sha256`
  - `mutants/MUTANTS.json`
  - `build_PF/` and `build_PFH/`
  - `arms/PF/` and `arms/PFH/`
  - `sass/F3.sass`, `sass/PF.sass`, `sass/F3.sass.failed_r1_no_tmpdir`
  - `SASS_COMPARISON_PF.json`
  - `gpu/G1/`: `part_00000.json`, `part_00835.json`, `part_04171.json`, `RESULT.json`, `exception_g1_1794859776672260559.txt`
  - `gpu/G1H/`: `part_00000.json`, `RESULT.json`
  - `gpu/` line receipts: `*.start`, `*.log`, `*.exit`, `*_dry_*.log`, `*_precheck_*.log`
  - `logs/`
- **Scratch left (leave-and-list):** `/tmp/apap1pf2_{main,build,sass,gpu,tmp}`.

## Prior art
- **TurboQuant** (Zandieh et al., 2025, arXiv 2504.19874): the rotated 4-bit codes. Taken as the stored form.
- **Asymmetric distance computation in product quantization** (Jégou, Douze and Schmid, 2011): scoring a full-precision query against codes without reconstructing. Taken as the idea; here the query is rotated rather than scored through a per-subspace table.
- **BLASST** (Yuan et al., 2025, arXiv 2512.12087): the running-maximum refine threshold. Taken unchanged.
- **Online softmax** (Milakov and Gimelshein, 2018, arXiv 1805.02867) and **FlashAttention** (Dao et al., 2022): the dense prefill kernel's streaming softmax. Inherited unchanged.
- **Mutation testing** (DeMillo, Lipton and Sayward, 1978): the 16 registered mutants.
- **IEEE 754 round-to-nearest subtraction:** the tie-delta construction.
- **P1-D/F1/F2/F3 plumbing** (bit views, line runners, build and mutant staging): imported or restated.
- **Ours, no prior art known to me:**
  - the packed decode's per-pair bulk arithmetic placed inside the dense prefill structure, which makes prefill decisions equal decode's bit for bit;
  - the 32-keys-per-warp form for the sequential D64 order;
  - the threshold-tie construction.
- **Unverified — lead to check.** Search terms: "attention scores from quantized keys without dequantization rotated query"; "TurboQuant 2504.19874"; "BLASST 2512.12087"; "Milakov Gimelshein online normalizer".

G1 FAIL (PF-I3 FAIL; PF-I1, PF-I1n, PF-I2 PASS) · G2 NOT RUN · G3 NOT RUN · G4 NOT RUN · G5 NOT RUN · G6 INCOMPLETE (SASS PASS; LF1/L1/L2/L4/L6 not run) · G7 NOT RUN (0 of 16 run) · G8 PASS so far — verdict RED, not ACCEPT.
