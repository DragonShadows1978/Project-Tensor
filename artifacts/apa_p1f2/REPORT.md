<!-- Filed by the lead verbatim from the seat's final message (the harness refused the seat's own write of this file). -->

**APA-P1-F2 is RED as registered: G6 (SASS) fails, and G1–G5 and G7–G9 pass.** All eight REV2 findings are fixed. Finite results are bitwise unchanged (LF1, 6,922 of 6,922 arrays). The packed route no longer hides a non-finite row that dense SP makes non-finite (LF2), and all 6 mutants are killed.

G6 fails for one reason. Build F was compiled in `tensor_fork_apa_wt_p1d` and Build F2 in `tensor_fork_apa_wt_p1f2`. nvcc puts a hash of the source path into the names of internal-linkage kernels, so 260 kernels appear as removed in F and added in F2. My `EXPECTED.json` did not anticipate this, and it cannot be amended. A supplementary comparison (not a gate) maps those path hashes; it shows exactly the three registered `spq1::prepare` kernels differ out of 781. You need to rule on G6 (D1).

The harness refused my write of `artifacts/apa_p1f2/REPORT.md` ("Subagents should return findings as text"), so this message is the report. Wall time was about 2 h 25 min. GPU lines took 1,075 s in total.

## Gates
- **G1 finite identity: PASS.** LF1 compared 6,922 of 6,922 arrays bitwise, Build F vs Build F2. Both sides are informative on all 6,922, and the key sets are identical. It used F1's LF1 cases unchanged: P1-D's L1 grid on the public route, and every raw packed entry on REV1's rule shapes, with carry and delta passed explicitly and equal on both sides.
- **G2 non-finite: PASS.** LF2 passed 1,139 of 1,139 gated checks.
  - G2a, NaN row-set equality: 132/132.
  - G2b, containment (never hides): 471/471.
  - G2c, no all-zero row where dense is non-finite: 471/471.
  - L19: 54/54 cases informative, 10/10 declared all-finite cases were all-finite, 13 single-row cases declared.
  - F1's G2b misses are gone. For example, +Inf at k[0,1,700,64] now gives {4,5,6,7} on every tq4 entry, where dense gives {6}; F1 gave {}.
  - Rows non-finite in an entry but finite in dense (allowed, the loud direction) are listed per case in `gpu/LF2/RESULT.json` `loud_rows`. They occur on ±Inf in k (tq4 and int4 entries), on NaN and +Inf sinks (row 2), and on all-negative logits (rows 0–3, every entry). The FLT_MAX scale is refused on the public route by the table lookup, before any device work.
- **G3 refusals: PASS.** LF3 passed 40 of 40 rows. S4's R1, R3 and R8 tests pass (part of 31/31). `BOUNDARY.json` has 47 rows, each giving F2's refusal and Build F's behaviour.
- **G4 append: PASS.** LF4 passed 134 of 134. At both D64 and D128, all 65 steps are bitwise equal in codes, norms and route output. Filling to N=256 and appending once more raises a ValueError with 0 launches; the codes and norms are bitwise unchanged and visible_rows stays at 256.
- **G5 NITs: PASS.**
  - R6: `ops.h` now declares the 16-argument signature. The compile probe fails on the base tree (`cannot deduce "auto" type`) and compiles on this tree.
  - R7: the S4 synthetic-dump tests pass, and the F2 comparison itself ran through the R7 script.
- **G6 SASS: FAIL (as registered).**
  - The 3 listed kernels changed as expected and 518 are equal.
  - 260 kernels are removed and 260 added, none of them listed. All are internal-linkage kernels (217 in kernels.cu, plus conv, gemm_apa, terrain, paint_*, dda and terrain_objects). The only difference is the path hashes `__nv_static_<n>__<8 hex>_` and `_GLOBAL__N__<8 hex>_`.
  - `expected_but_not_seen` is empty.
  - Supplementary result (`SASS_PATHMAP_SUPPLEMENT.json`, using P1-D's `apa_p1d_sass_a1.read` imported): 781 kernels, 0 present on one side only, and the differing set equals the registered changed list.
- **G7 mutants: PASS.** 6 of 6 killed, each by its named assertion.
- **G8 no regression: PASS.**
  - CPU: 744 IDs, identical collections. There are 8 differences, all of them the 8 R1 intended changes registered in `EXPECTED.json`; 0 unintended.
  - LF6: 44 files and 2,078 IDs (LF6a 1,376, LF6b 702), 0 differences, collections equal.
- **G9 scope: PASS.**
  - Writes only in the writable set; the sha256 diff against the round-start map shows 15 changed files, all listed below.
  - GPU only through forge-gpu-run.
  - Git use was read-only: `git archive`, `diff`, `ls-files` and `status`.
  - Nothing was deleted, and nothing was written into the p1d worktree (a `find -newer` check came back empty).
  - The REPORT.md write was refused by the harness.

## Diff summary
15 tracked files, +366/−76.
- **R1 (`apa_packed.py`)**
  - New `PACKED_PINNED_ENTRIES`, the two pinned e_q values.
  - New `check_pinned_margins()`: it requires the exact type `PackedRegisteredMargins`, a sha256 equal to `PACKED_TABLE_SHA256`, and entries with the same keys whose values are floats bit-equal to the pins.
  - It is called in `load()`, in `lookup()`, and in `packed_attention` right after the type check, before any device work.
  - `load(expected_sha256=…)` refuses any digest other than the pin. `packed_attention` calls the class's own `lookup`, so a subclass cannot substitute its own.
  - No experimental table type was kept.
- **R2 (`packed.cuh` `spq1::prepare`)**
  - A key row with any non-finite element now stores a quiet NaN norm.
  - The shared `norm` that the unit vector divides by is unchanged, so the rotated output is bit-identical and finite rows are unchanged.
  - The census found this to be the only TQ4 key quantizer of the packed route; `pack_tq` copies the norm verbatim.
- **R3**
  - `launcher.cuh` gains `apa_spo_require_positive_extents`, `apa_spo_require_int_range`, `apa_spo_product`, `apa_spo_ceil_div` and `apa_spo_require_key_range`, all raising `std::invalid_argument` (ValueError) that names the operand.
  - `apa_spq1_buffer_check` refuses a capacity above INT_MAX. It is the one check every packed attention and append entry runs first.
  - The rotate grid, the prepare grid and the pack grid and B*KV products are checked.
  - The attention entries (core, spq4 tc, spq6 tc, eb3) check S + 2047 ≤ INT_MAX (int partition and chunk arithmetic).
  - `eb2b_values` checks numel/8 + 127.
  - Python: `check_native_range()` runs in `PackedSpKeys.__init__` and `append` (before the generation moves) and in `ApaSpq1Workspace.__init__` and `append` (before `zeros()`, integer capacity required).
  - Nothing is widened; everything out of range is refused.
- **R4**
  - Raw `packed_rotate`, `packed_prepare`, `packed_append`, `eb2b_quantize` and `eb2b_values` now refuse non-positive extents with ValueError before any launch. `int4_stats` already did since F1.
  - EB3 refuses `timeline=True` at G ∉ {2,4} before the lock, the arena and any launch.
- **R5 (comments only)**: in `merge.cuh` and the fused merges in `eb3.cuh` and `eb4.cuh`, the claim is now narrowed to finite intermediate scores and denominator. Overflow, NaN or +Inf sinks, and ±Inf queries over all-negative logits are named as IEEE propagation, the loud direction.
- **R6**
  - `ops.h:328` now has the current signature.
  - The spo2 binding calls `tc::apa_eb2b_run` directly; F1's global bridge `apa_p1f1_eb2b_run` is removed (host code only).
  - The hooks-only comment in `eb3.cuh` was updated (comment only).
- **R7 (`scripts/apa_p1f1_sass.py`)**
  - `compare_builds()` takes the builds, the artifacts directory and the expectation file as arguments.
  - It records `expected_active` (path and sha256) next to `expected_json_sha256`.
  - `symbol_suffix_renames` counts only paired kernels; additions and removals are counted separately (`symbol_only_in_base` and `symbol_only_in_new`).
  - It maps `_cu_<8 hex>` with an optional `_<n>`.
  - F1's filed receipts are untouched.
- **R8 (`apa_sp2.py`, `quantize_sp2_keys` only)**
  - New `capacity=None` keyword-only argument, forwarded to `PackedSpKeys` when the result is packed.
  - Dense storage raises ValueError on any capacity other than None: requested by argument, by environment, or by the default's recorded fallback. The check runs before any device work.
  - With `capacity=None`, every result is as before.

## Modules and SASS
| Build or receipt | Module sha256 |
|---|---|
| Build F (checked before every use) | `5c17147715ffc422b0aa7ba8fae826c554a4caff63b1231c22ce580867efc243` |
| Build F2 (`build_F2`) | `6f03b2cdd5d7f9bee4456ed34759c6c710cccfab9a50f9ac0fc6a678c076f579` |
| `build_F2_r2` (no-op refresh receipt, same module, source map current; see D3) | same as Build F2 |
| Build F2H | not built: no code compiled only with hooks changed |
| Mutant m02 (`build_r3`) | `26b6c9ee79a69a31be7a6eb4bdc6fa0cfb982fbb118c473632c0b5522285f68e` |
| Mutant m04 (`build`) | `98cf265c664bff9a3bde900ecd9f56c9b13a2306f25043e3a5b9f6141d6681b8` |
| Mutant m05 (`build`) | `efb74aadb077c50bb24bebc0cee7d7217783e3b4a88f50a3d59b9ab4fc676974` |
| Mutants m01, m03, m06 (Python only) | Build F2's module |

Each mutant build recompiled only `kernels.cu.o`, then device-linked and linked.

| File | sha256 |
|---|---|
| `EXPECTED.json` (written once, before the first F2 build) | `a221c22355fa5a7848b1584543e57bd3db96347e1702cf96b5f62058238cac2f` |
| `SASS_COMPARISON_F2.json` | `1c3190eddb0b30a4e6d956fe09d823a0b6880bfb140ebc8a327ebb2154ac34ab` |
| `SASS_PATHMAP_SUPPLEMENT.json` | `c1c34297afabbf5acada26c232edaa7b8c44fcf0a37cda718c02d65a5e32b46d` |

SASS against the expected list:
- Expected changed: `void tc::spq1::prepare<float>`, `<__half>`, `<__nv_bfloat16>`. All three changed.
- Expected added and removed: none.
- Not on the list: 260 removed and 260 added, all path-hash pairs, so **G6 FAIL**.
- Renames counted under R7's rules: 0 paired renames; 260 only in F and 260 only in F2.

## Censuses
- **`TQ4_CENSUS.json` (R2):**
  - The only TQ4 key quantizer is `spq1::prepare`, which owns the norm and rotation, together with `spq1::pack_tq`, which owns the codes and copies the norm.
  - `eb3.cuh`, `spq4.cuh` and `spq6.cuh`, and also `eb4.cuh` and `eb2b.cuh`, only read codes and norms. `packed_launcher.cuh` only launches the kernels.
  - The census does not widen R2.
- **`INDEX_CENSUS.json` (R3):**
  - 16 rows: every int that receives the capacity (`int capacity` / `int dst_s` in the spq1, spq4, spq6, eb3 and eb4 kernels), the rotate, prepare and pack grids, the B*KV product, S+2047 and nc*256 in the attention launchers, and the eb2b values index.
  - L32: the readers of the packed key store were found by grep; every kernel reads rows j < visible S.
  - The census changed what the ruling covers: it added the S+2047 key range and the eb2b_values index.
- **`HELPER_CENSUS.json` (R4):**
  - 15 spo2 `m.def`s. Raw helpers: rotate, prepare, append, int4_stats, eb2b_quantize, eb2b_values.
  - The 5 attention entries already refuse empty extents before any launch with RuntimeError; I kept that (D2).
  - EB3 timeline templates exist for G 2 and 4 only; G 3, 5, 6, 7 and 8 are now refused before launch.

## CPU and per-ID results
- **S4 (`test_apa_p1f2_cpu.py`): 31/31 pass** (`cpu/S4.json`).
  - R1: 9 hand-built table cases, a subclass, entries or sha replaced after load, load override, and the pinned positive control.
  - R1 also restates on the pinned table the coverage of the 8 policy tests R1 retires: delta, causal, sinks and r2 forwarding at D64 and D128, the refusal order, prefill, stale generation, and a packed table with dense Kq.
  - R3: capacity at and one past the bound, prepare-grid product at and past the bound, append product, factory capacity, and the workspace in tq4 and int4.
  - R8: signature, forwarding, None unchanged, dense refusal, invalid capacities.
  - R4: source order (each refusal comes before the first launch).
  - R6: the declaration, plus the compile probe on base and after.
  - R7: synthetic dumps and the optional-suffix mapping.
- **Per ID (`CPU_G3_COMPARISON.json`):** 744 IDs. Before (Build F arm on `/tmp/apap1f2_treeF0`): PASSED 465, FAILED 125, ERROR 15, SKIPPED 139. After (Build F2 arm on this tree): PASSED 457, FAILED 133, ERROR 15, SKIPPED 139.
  - The 8 differences are exactly the registered intended changes. Ruling R1: `packed_table()` loads a synthetic table with its own digest, which is now refused.
  - The 8 IDs, all in `test_apa_p1d_policy.py`: `test_stale_generation_is_refused`, `test_packed_table_on_dense_kq_is_refused`, `test_packed_route_refusals_before_launch`, `test_packed_route_forwards_delta_causal_sinks_and_r2[64]` and `[128]`, `test_packed_loader_checks_its_own_registration_quantizer_and_entries`, `test_routes_are_reported_for_quantize_and_attention`, `test_prefill_on_packed_keys_is_refused_naming_dense`.
- **LF6:** 2,078 IDs, 0 differences.

## Mutants (6/6 killed; `MUTANTS.json` sha256 `8c536224600ddb33ad61bb94f47249630b5171d9bfda4922202b01f3e3c01fde`, written before any mutant ran)
| Mutant | Kind | Line | Named assertion that failed |
|---|---|---|---|
| m01_margin_pin_unchecked | Python | S4 | `test_r1_hand_built_table_refused_at_lookup_and_at_attention_before_any_device_call[wrong_sha_pinned_entries]`: DID NOT RAISE |
| m02_tq4_inf_norm_kept | device | LF2 | `G2b containment (never hides): r2_tq4 bfloat16 k_posinf_p1 vs dense_entry` (19 checks failed in total) |
| m03_capacity_unbounded | Python | S4 | `test_r3_packed_capacity_at_the_bound_is_accepted_one_past_is_refused_before_allocation`: DID NOT RAISE |
| m04_raw_extent_unchecked | binding | LF3 | `LF3 raw helper zero extent raises ValueError, no launch: rotate_q_B0` (got a RuntimeError, CUDA invalid configuration, 1 launch) |
| m05_eb3_timeline_g_late | device | LF3 | `LF3 EB3 timeline=True G=3 raises before any launch` (got a RuntimeError after 2 launches) |
| m06_factory_capacity_dropped | Python | S4 and LF4 | `test_r8_capacity_is_forwarded_to_packed_keys` (capacity 100 ≠ 164); `LF4 D64: factory capacity forwarded` |

## GPU lines
Every dry run returned 78. Each line ran once (r1): there were no retries and no exit 76. Forge exit equals the inner timeout exit, which was 0 every time.

| Line | Seconds | Result | Log sha256 (prefix) |
|---|---|---|---|
| LF1 | 19 | PASS, 6,922/6,922 | 762d992c21341abb |
| LF2 | 2 | PASS, 1,139/1,139 | 51b50a98c43ec41e |
| LF3 | 29 | PASS, 40/40, plus 32 Build F boundary rows in child processes | 4634f115bfc3e8f2 |
| LF4 | 2 | PASS, 134/134 | f0cfb88789012e1e |
| LF6a | 373 | PASS, 1,376 IDs, 0 differences | 80d994c209b3c6f4 |
| LF6b | 643 | PASS, 702 IDs, 0 differences | 484fafcf353cdab8 |
| MUT | 6 | PASS, 4/4 | 5f185d0072869544 |

Build F's behaviour on LF3's inputs (`BOUNDARY.json`):
- Raw helpers: zero-grid CUDA "invalid configuration" errors after a launch, 8 rows.
- EB3 timeline at G 3, 5, 6, 7 and 8: a RuntimeError after 2 launches.
- Hand-built margin tables: silently accepted, 5 rows.
- Capacity 2**31: a TypeError on the factory, and a 2 TiB host allocation failure (MemoryError) in PackedSpKeys and ApaSpq1Workspace.
- Native fake-capacity rows: recorded from the source, not run on Build F (D10).

## Deviations
- **D1 (G6).** This is my registration miss: the path-derived kernel-name hashes differ between the two worktrees. I did not amend `EXPECTED.json`. The supplementary path-mapped comparison shows that only the registered kernels differ. You need to rule on G6.
- **D2 (R4 interpretation).** By "raw spo2 helper" I took the non-attention entries. The attention entries (run, packed_run, eb2b_run, eb3_run, eb4_run) already refuse empty extents before any launch, with the RuntimeError F1 R11 chose for dense parity; I left that unchanged. If you want ValueError there too, it is a one-line change per entry.
- **D3.** I edited a refusal message in `apa_packed.py` (Python only) while Build F2 was compiling. That receipt records `source_unchanged_during_build: false`. The `build_F2_r2` refresh is a no-op rebuild with an identical module and the current source map; every GPU precheck used it.
- **D4.** The harness refused REPORT.md.
- **D5 (tool slips, L56a).**
  - My LF2 wait command had no explicit Bash timeout, so the harness moved it to the background; I then waited in the foreground. The line itself ran detached and was not affected.
  - Several `pgrep -f` waits matched my own shell (F1's D17), so they ran to their timeout. No result was affected.
- **D6 (script repairs, L56b; at most 3 per script, counted per script).**
  - The S4 test file had a dict TypeError (1).
  - Build script, mutant relocation: repair 1 moved the object directories (attempt `build` FAIL); repair 2 rewrote `.rsp` files (attempt `build_r2` FAIL, `build_r3` PASS). The failed attempt directories are left in place.
  - GPU script: precheck now handles the receipt for Python-only mutants (1).
- **D7.** R3's census added the key-range checks (S+2047) on the attention entries and the eb2b_values index check. These go beyond the order's explicit list and are refusal-only.
- **D8.** Under R8, a capacity combined with the default's dense fallback (2-bit) also raises, rather than being dropped silently.
- **D9.** For L19, the dense side of k +Inf at (0,0,100,5) was declared all-finite before the run, by a sign rule computed from the inputs (F1's known case). The r01 cases and the exact r02 case were declared single-row.
- **D10.** LF3's two native fake-capacity rows (a DLPack re-import with capacity 2**31) ran on F2 only. Build F's behaviour is recorded from the source, because on Build F those rows would index outside the buffers.
- **D11.** `PackedSpKeys` and `ApaSpq1Workspace` refuse zero B or KV only when rows would actually be packed (L55). An empty S with a reserved capacity stays valid.
- **D12 (residual).** A finite key row whose squared norm overflows fp32 still stores an Inf norm. That path is unchanged (finite rows unchanged, L55/LF1) and is not covered by G2.

## Prior art
- **IEEE 754-2008** NaN and overflow propagation (R2, R5).
- **TurboQuant** (Zandieh et al., 2025): the rotated 4-bit codes and norm, unchanged. The NaN-norm carry is ours, mirroring F1's INT4 sentinel (Project-Tensor, 2026).
- **Online-softmax merge** (Milakov and Gimelshein, 2018; FlashAttention, Dao et al., 2022): unchanged; R5 only restates the claim.
- **SHA-256 pinning** (NIST FIPS 180-4): R1. Checking the pin at use is ours.
- **Capacity reservation as in `std::vector::reserve`**, and **ISO C++17 [conv.integral]** narrowing: R3 and R8.
- **CUDA Programming Guide** grid-dimension limits: R3 and R4. The census itself: no prior art known to me.
- **Mutation testing** (DeMillo, Lipton and Sayward, 1978); **test doubles** (Meszaros, 2007).
- **make** timestamp-driven incremental rebuild (Feldman, 1979): the mutant builds. Relocating the build tree: no prior art known to me.
- **P1-D/F1 line machinery, P1-A1 path-hash mapping, P1-Q append check** (Project-Tensor, 2026): imported.
- Unverified, lead to check. Search terms: "nvcc __nv_static anonymous namespace hash source path", "cmake relocate build tree object path absolute source".

## Files written
Under `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1f2/`:
- **Modified:**
  - `tensor_cuda/include/tc/ops.h`
  - `tensor_cuda/src/apa_spo/{eb2b,eb3,eb4,launcher,merge,packed,packed_launcher,spq4,spq6}.cuh`
  - `tensor_cuda/src/bindings.cpp`
  - `tensor_cuda/tensor_cuda/{__init__,apa_packed,apa_sp2}.py`
  - `scripts/apa_p1f1_sass.py`
- **New code:**
  - `tensor_cuda/tests/test_apa_p1f2_cpu.py`
  - `scripts/apa_p1f2_{build,census,sass,sass_pathmap,gpu,line,mutants,cpu_g3,boundary}.py`
  - None of these names shadows a standard-library module (checked).
- **`artifacts/apa_p1f2/` (about 1.0 GB):**
  - registrations and censuses: `BASE_SHA256_MAP.txt`, `INSTRUMENTS.log`, `EXPECTED.json`, `TQ4_CENSUS.json`, `INDEX_CENSUS.json`, `HELPER_CENSUS.json`, `BOUNDARY.json`
  - SASS: `SASS_COMPARISON_F2.json`, `SASS_PATHMAP_SUPPLEMENT.json`, `sass/{F,F2}.sass`
  - builds: `build_F2/` (88 files), `build_F2_r2/`, `arms/F2/`
  - CPU: `cpu/{S4.json,S4.log,S4_RESULT.json,CPU_STAGE.log}`, `cpu_g3/{before,after}.{json,log}`, `CPU_G3_COMPARISON.json`
  - GPU: `gpu/{LF1,LF2,LF3,LF4,LF6a,LF6b,MUT}/` plus each line's `.start`, `.log`, `.exit`, dry-run log and precheck logs
  - mutants: `mutants/{MUTANTS.json,MUTANT_RESULTS.json}` and `mutants/m0{1..6}_*/` (tree, MUTANT.json, arm, cpu/ or build*/)
- **Scratch, left in place for you to clear:** `/tmp/apap1f2_{tmp,build,pytest,gpu,shim,r6,g3,l6,mut}` and `/tmp/apap1f2_treeF0` (about 945 MB, git archive d01118b4).
