<!-- Filed by the lead verbatim from the seat's final message (the harness refused the seat's own write). -->

All three REV3 findings are fixed on the packed default, and gates G1 to G7 all pass. By the registered rule the round is ACCEPT, pending your on-disk check and the Codex review.

The harness refused my write of `artifacts/apa_p1f3/REPORT.md` ("Subagents should return findings as text"), so this message is the report; please file it verbatim. Wall time was about 1 h 15 min (06:20 to 07:35 EST). GPU lines took 1,265 s in total.

## Gates
- **G1, finite identity: PASS.** LF1 compared 6,922 of 6,922 arrays bitwise, Build F2 against Build F3. Every array is informative on both sides, and the key sets are identical. The cases are F2's LF1 cases unchanged (F1's `lf1_arm`, imported).
- **G2, non-finite: PASS.** LF2 passed 1,265 of 1,265 gated checks.
  - F2's inherited cases: 1,139 checks, all pass.
  - The new overflow cases: 126 checks, all pass. A key row with one finite element of 1e20 is now non-finite on every TQ4 reader (r2_tq4, tc_tq4, eb2b, eb3, eb4) and on the public default route. These rows are 0–3, the same rows dense SP returns non-finite with quantized keys, and the same rows the dense opt-out returns. This holds at D64 and D128, with q and −q.
  - The stored norm of the overflow key is NaN in all 10 cases.
  - REV3's exact r01 case now gives packed row {0}, matching dense {0}. Before, packed was finite there.
  - L19: every overflow case is informative or declared. The dense entry with exact keys was declared all-finite before the run, and it was.
- **G3, refusals: PASS.**
  - S4 passed 34 of 34 (R1′, R2′ and R3′).
  - LF3 passed 18 of 18. Every refusal had 0 launches, and the controls were bitwise equal after the refusals.
  - `BOUNDARY.json` has 18 rows, each with Build F2's behaviour on the same input.
- **G4, SASS: PASS.** Only the 3 registered kernels differ; detail under "Modules and SASS".
- **G5, mutants: PASS.** All 3 are killed, each by its named assertion.
- **G6, no regression: PASS.**
  - CPU: 775 IDs over 23 files, collections equal, 0 differences. Both sides: 496 passed, 125 failed, 15 errors, 139 skipped (the failures and errors are the same on both sides).
  - LF6: 2,078 IDs (LF6a 1,376, LF6b 702), 0 differences.
  - No intended changes were registered, and none occurred.
- **G7, scope: PASS.**
  - Against the round-start sha256 map, only the 9 product files changed; the new files are the 10 listed below plus `artifacts/apa_p1f3/`.
  - Nothing under `artifacts/apa_p1f2`, `apa_p1_rev3` or `apa_p1d` is newer than the round start (0 files).
  - GPU use was only through `forge-gpu-run`.
  - Git use was read-only (`archive`, `show`, `diff`, `status`, `ls-files`). Nothing was deleted.

## Diff summary
9 tracked files, +111/−13. Every edit is annotated at its code site with its ruling and prior art.
- **R2′ (`packed.cuh`, `spq1::prepare`):**
  - The stored norm is now `(nonfinite || !isfinite(norm)) ? qNaN : norm`, so it is NaN for any row whose computed norm is non-finite, including a sum of squares that overflows from finite elements.
  - The shared `norm` that the unit vector divides by is unchanged, so the rotated output and codes are bit for bit as before.
  - F2's comment that an overflowing row keeps its Inf norm is replaced.
  - L55 from the code: the square root of a finite, non-negative sum is finite, so only rows whose computed norm is non-finite change.
- **R1′ (`apa_packed.py`, `lookup`):** after `check_pinned_margins` passes, `lookup` returns `PACKED_PINNED_ENTRIES[key]` and never reads the caller's mapping again. Seat check: `packed_attention` reaches margin values only through `lookup`; its own check reads them for validation only.
- **R3′:**
  - `launcher.cuh` gains an `APA_P1F3` guard block:
    - `ApaSpoOperand` and `apa_spo_require_attention_extents`: an extent of q, k or v above INT_MAX raises a ValueError naming the entry, the operand, the axis and the extent. A zero or negative extent is left to the existing checks, which keep F1's RuntimeError.
    - `apa_spo_require_at_most`, for bounded int64 products.
  - The extent guard is the first statement of all 7 packed attention entries: `apa_spq4_run` (the `packed_run` dispatcher), `apa_spq1_run`, `apa_spq4_tc`, `apa_spq6_tq4_tc`, `apa_eb2b_run`, `apa_eb3_run` and `apa_eb4_run`.
  - EB2b: B*H is now formed by `apa_spo_product({q.shape[0],q.shape[1]})` in int64, before the narrowing `int B=…`. It is checked against INT_MAX/4 (the same bound as before) and refused with a ValueError naming the operand. The signed int term is removed from the old geometry check.
  - Census "like sites": `N = S - start` is now formed in int64 in `apa_spq1_prepare` and `apa_spq1_pack`. A large negative start could overflow before its refusal. All defined inputs get the same refusals as before.

## Modules and SASS
| Build | Module sha256 |
|---|---|
| Build F2 (before-build, checked before every use) | `6f03b2cdd5d7f9bee4456ed34759c6c710cccfab9a50f9ac0fc6a678c076f579` |
| Build F3 (`build_F3`) | `2e557a23eaf228512a2f1a2f0629401d91d02d05705087661be5c48849c0f3e9` |
| m02 mutant (incremental, only `kernels.cu.o` rebuilt) | `781029589a00bbd90205a6a2f3cfab4feecfe03dd726b4925dc2ff8bc5054186` |
| m03 mutant (incremental, only `kernels.cu.o` rebuilt) | `9af99ddc6a744257aef21e15cd2437316c16bd10c61f55d6f9f8eaac06b94a2b` |
| m01 mutant (Python only) | Build F3's module |

Build F3 recompiled 13 objects, with `source_unchanged_during_build: true` and `TC_APA_PACKED=ON`. F3H was not built: no code compiled only with the test hooks changed.

Registrations:
- `EXPECTED.json` sha256 `1aae94624586c404031c489b639e7cae6ef033a385ef74b284094f78b21a23f4`. It was written once, before the first F3 build and before any GPU line, and is unchanged.
- `SASS_COMPARISON_F3.json` sha256 `c4aa66c0…`.

SASS against the expected list:
- 781 kernels per build: 778 equal, 3 changed, 0 added, 0 removed.
- The 3 changed kernels are exactly the listed ones: `void tc::spq1::prepare<float>`, `<__half>` and `<__nv_bfloat16>`.
- Nothing differs that is not on the list, and every listed kernel differs.
- There were 14 paired `_cu_` suffix renames, mapped.

The non-code sources of difference I registered:
1. **Source path:** equal, and not mapped. Both builds compiled this worktree. The path-derived names (`_GLOBAL__N__…`, `__nv_static_…`) came out identical: 18 of 18.
2. **Build directory:** the bijective `_cu_<8 hex>` plus optional `_<n>` mapping.
3. **TMPDIR and cmake_source location:** not mapped separately; they can only reach names through `_cu_`.
4. **Compiler, flags and dump tool:** equal, checked from the receipts. The `p1d_flags` are equal, and the adapter replacements are equal once normalized (see D3). A fresh dump of Build F2 reproduces F2's own dump, sha256 `3b7efd2a…`.

## Product census (`PRODUCT_CENSUS.json`, `e1cc1a5a…`)
- 15 range-checked host-product rows in `src/apa_spo`, starting from `rg -n 'numeric_limits<int>'` plus F2's range helpers.
- 3 were not int64 before the check and narrowing, and are fixed:
  - EB2b B*H (eb2b.cuh:158–159 in the committed tree);
  - prepare's `N`;
  - pack's `N`.
- 12 were already int64: spq4, spq6, eb3 and eb4 B*H; the core check (int64 division); the key range; buffer capacity; rotate B*H; pack B*KV; int4_stats; eb2b_quantize; eb2b_values.
- All 15 are int64 now, and all 7 attention entries start with the named-extent guard.
- `OPERAND_CENSUS.json` covers the one norm store (`packed.cuh:123`) and every read of the margin mapping in `apa_packed.py` (15 lines). Only `check_pinned_margins` reads values at use time.

## CPU and per-ID results
- **S4** (`test_apa_p1f3_cpu.py`, Build F3): 34 of 34 pass.
  - R1′: a stateful mapping at `lookup` (D64/D128) gets the pinned value with only 2 reads. At attention, for 0, 1, 2, 3, 4, 5 and 8 pinned reads, the call is either refused before any device call or receives the pinned delta and e_q. The loaded pinned table is unchanged, and the margin-read census holds.
  - R2′: source form of the norm store.
  - R3′: host-level fake. The real guard blocks and EB2b's real prologue are compiled with g++ (-O0 -fwrapv) against a stub NDArray. Results: at the bound, no refusal; one past, ValueError naming apa_eb2b and B*H; 65536×65536, the same ValueError; extent 2^31 on k axis 2 and q axis 0, ValueError naming entry, operand and axis; q B=0, the RuntimeError. Plus a source-order test for all 7 entries.
- **Per ID** (`CPU_G3_COMPARISON.json`):
  - Before is Build F2's arm on `/tmp/apap1f3_tree0` (git archive 25f59b9c); after is Build F3's arm on this worktree.
  - 775 IDs; the inherited CPU G3 set is 689 of them.
  - `test_apa_p1d_policy.py` 40/40, `test_apa_p1f1_cpu.py` 15/15 and `test_apa_p1f2_cpu.py` 31/31 pass on both sides.
  - 0 differences.

## Mutants
`MUTANTS.json` sha256 `a824c1cc…` was written before any mutant was staged, built or run. `MUTANT_RESULTS.json` sha256 `17365ac5…` records 3 of 3 killed.

| Mutant | Kind | Line | Named assertion that failed (evidence the mutated line ran) |
|---|---|---|---|
| m01_lookup_rereads_mapping | Python | S4 | `test_r1p_stateful_mapping_at_lookup_returns_the_pinned_value[64]`: "lookup returned 0.0, not the pinned 1.010646104812622; mapping item reads = 3" (4 other S4 IDs also fail) |
| m02_overflow_norm_kept | device | LF2 overflow cases | `G2b containment (never hides): public_default float32 r01_ovf_qneg vs dense_opt_out`, among 28 failures; the stored norm is `inf` in all 10 overflow records |
| m03_eb2b_product_int32 | host | S4 and LF3 | S4 `test_r3p_eb2b_product_that_wraps_a_signed_int_raises_value_error_naming_it` got `('none','')`; LF3 `LF3 R3' row eb2b_BH_2p32` got "ValueError: apa_spq1_rotate (packed_rotate): B*H … = 4294967296", 0 launches (the wrapped product passed EB2b's check) |

## GPU lines
Every dry run returned 78. Each line ran once: no retries and no exit 76. The forge exit equals the inner timeout exit, which was 0 every time.

| Line | Seconds | Result | Log sha256 |
|---|---|---|---|
| LF1 | 19 | PASS, 6,922/6,922 bitwise, all informative | `ad5f7fdd…` |
| LF2 | 3 | PASS, 1,265/1,265 (1,139 inherited + 126 overflow) | `61aa32a8…` |
| LF3 | 16 | PASS, 18/18; plus 15 Build F2 boundary rows, each in its own child | `a0447c25…` |
| LF6a | 512 | PASS, 22 files, 1,376 IDs, 0 differences | `ca594287…` |
| LF6b | 709 | PASS, 22 files, 702 IDs, 0 differences | `ce18adb5…` |
| MUT | 8 | PASS, 2/2 (m02, m03) | `3908ee81…` |

Build F2's behaviour on LF3's inputs (`BOUNDARY.json`):
- B*H one past the bound: RuntimeError (geometry).
- B*H = 2^32: refused only by packed_rotate's ValueError.
- S = 2^31 on packed_run: the capacity ValueError.
- q B = 2^31, and S = 2^31 on eb2b/eb3/eb4: RuntimeError.
- Zero extents: the same RuntimeError as Build F3.
- The stateful mapping ran with e_q 0.0. On this input its output was still bitwise equal to the pinned table's.

## Deviations
- **D1.** The harness refused the REPORT.md write.
- **D2 (L56a, tool slip).** My LF2 wait used `pgrep -f 'apa_p1f3_line.py LF2'`, which matched my own shell, so the wait ran to its 590 s timeout. The line itself had exited 0. This cost about 10 minutes; later waits used the `[a]pa…` pattern. LF6b's first wait also hit 590 s because that line ran 709 s; I waited again.
- **D3 (receipt slip).** Build F3's receipt says `same_adapter_replacements_as_F2: false`. That is a tuple-versus-JSON-list artifact of my comparison; normalized, the replacements are equal, and so is the P1-D script sha. Nothing was re-registered.
- **D4 (L64).** The imported F1/F2 machinery reads, read-only and exactly as F2 did:
  - REV1's `repro.py` in `tensor_fork_apa_wt_apa2pprod`, to build F2's LF1/LF2 cases;
  - the production dense table (sha256 `5b6f8bfb…`, checked).

  The A1 file list came from this worktree's copy. You may want to rule on whether L64 was meant to exclude these reads.
- **D5 (R3′ scope).**
  - The named-extent guard is also in `apa_spq1_run`, `apa_spq4_tc` and `apa_spq6_tq4_tc`. From Python these are reached only through the dispatcher, whose guard fires first; `apa_spq1_run` is also declared in ops.h for native callers.
  - The unpacked d1 `run` entry keeps its RuntimeError for extents above INT_MAX, because it is not a packed entry.
  - The `N` int64 changes are census "like sites".
- **D6.** m03's kill rule was registered as "S4 or LF3", the order's wording. Both killed it.
- **D7.** I registered LF1 stricter than F2's code: every array must be informative, not only the outputs. It passed.
- **D8.** Free disk fell from 121 GB to 91 GB during the round. This round wrote about 0.8 GB to `artifacts/apa_p1f3` and about 2.6 GB of scratch; the rest came from other activity on the disk. The 40 GB rail was never reached.
- **No script repairs (L56b):** none of my scripts raised.

## Prior art
- **IEEE 754-2008:** overflow to Inf and NaN propagation (R2′).
- **TurboQuant (Zandieh et al., 2025):** the norm, rotation and codes, unchanged. The NaN carry keyed on the computed norm is ours, extending F1's INT4 sentinel and F2's R2.
- **Time-of-check to time-of-use (MITRE CWE-367):** R1′ uses the validated value. Applying it to the pinned table is ours.
- **ISO C++17** signed overflow ([expr]/4) and narrowing ([conv.integral]), and ordinary checked multiplication (R3′). No prior art known to me for the census itself.
- **Mutation testing (DeMillo, Lipton and Sayward, 1978)** and **test doubles (Meszaros, 2007).**
- **F1/F2/P1-D build, SASS, line and DLPack re-import machinery (Project-Tensor 2026):** imported.
- **make's incremental rebuild (Feldman, 1979).**
- Unverified, lead to check. Search terms: "CWE-367 check use immutable snapshot", "TurboQuant key norm overflow fp32".

## Files written
In `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1f2/`:
- **Modified:**
  - `tensor_cuda/src/apa_spo/{eb2b,eb3,eb4,launcher,packed,packed_launcher,spq4,spq6}.cuh`
  - `tensor_cuda/tensor_cuda/apa_packed.py`
- **New code:**
  - `tensor_cuda/tests/test_apa_p1f3_cpu.py`
  - `scripts/apa_p1f3_{register,build,sass,cpu_g3,mutants,gpu,line,census,boundary}.py`
  - All run as `python3 -B -P`; none shares a standard-library module name (checked).
- **`artifacts/apa_p1f3/`:**
  - registrations and receipts: `BASE_SHA256_MAP.txt`, `INSTRUMENTS.log`, `EXPECTED.json`
  - builds: `build_F3/`, `arms/F3/`
  - SASS: `sass/{F2,F3}.sass`, `SASS_COMPARISON_F3.json`, `SASS_CHECKS_F3.json`
  - censuses: `PRODUCT_CENSUS.json`, `OPERAND_CENSUS.json`
  - `BOUNDARY.json`
  - CPU: `cpu/{S4.json,S4.log,S4_RESULT.json}`, `cpu_g3/{before,after}.{json,log}`, `CPU_G3_COMPARISON.json`
  - mutants: `mutants/{MUTANTS.json,MUTANT_RESULTS.json}` and `mutants/m0{1,2,3}_*/` (tree, MUTANT.json, arm, cpu/ or build/)
  - GPU: `gpu/{LF1,LF2,LF3,LF6a,LF6b,MUT}/`, each line's `_r1.{start,log,exit}`, `_dry_r1.log`, the precheck logs, and `gpu/CPU_STAGE.log`
- **Scratch, left in place for you to clear (about 2.6 GB):** `/tmp/apap1f3_{build,g3,gpu,l6,mut,shim,tmp,tree0,treeF0}`. Nothing was deleted.
