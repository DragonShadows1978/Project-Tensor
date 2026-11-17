<!-- Filed by the lead: the harness refused the seat's write of this file, so the seat returned the report as its final
message. Copied verbatim, except that HTML-escaped comparison signs are restored. -->

**APA-P1-F1: all twelve REV1 defects are fixed and their tests pass, but G7 fails as registered. G1–G6 and G8–G10 pass; G2b is NOT HELD.** The final Build F's SASS differs from Build B in exactly the intended kernels. G7 fails because those are not the kernels my first registration (`EXPECTED.json`) listed, and I re-registered only after seeing attempt 1's result. You need to rule on whether the attempt-2 registration stands (D1). The harness refused my `artifacts/apa_p1f1/REPORT.md` write ("Subagents should return findings as text"), so this message is the report. Wall time was about 4 h 30 min; the GPU was held for about 22 minutes in total.

## Gates
| Gate | Verdict | Receipt |
|---|---|---|
| G1 finite identity | **PASS** | LF1 r2: 6,922/6,922 arrays bitwise equal, Build B vs F; both sides informative on all 6,922. Covers P1-D's L1 grid (512 cases × 4 routes), every raw packed entry on REV1's rule shapes (both carries, deltas 2.0 and 0.0, with and without sinks, diagnostics on and off), EB3 variants 0–8, EB3 at S=70000, the d1 baseline dump, the tq4/int4 workspace codes and scales, and `eb2b_quantize` vi/vs. |
| G2 non-finite | **PASS** | LF2 r2: NaN row-set equality 72/72; "no all-zero row where dense is non-finite" 120/120, plus 24/24 in LF2S. Caveat: LF2's `RESULT.json` status reads FAIL. The only cause is my L19 check on two registered cases (+Inf at k[0,0,100,5], bf16 and fp32): every KV-group-0 head has q₅ < 0 there, so the dense output is finite everywhere. The supplementary case LF2S is informative under L19 and passes. |
| **G2b** | **NOT HELD** | +Inf in k only; the +Inf-in-q cases match dense. Differing rows (entry vs dense): k_inf_rev1: r2_int4 and tc_int4 give {0,1,2,3} vs {}. k_inf_more: tq4 entries (r2_tq4, tc_tq4, EB2b, EB3, EB4) give {} vs {6}; int4 entries give {4,5,6,7} vs {6}. LF2S (key 100, dim 0): tq4 entries give {2,3} vs {0,2,3}; int4 entries give {0,1,2,3}. d1 matches dense everywhere. Cause: int4 turns any non-finite key into NaN scores. For tq4, Inf·acc takes its sign from the rotated-code dot product, not from q·k. Per the order this changes no code; you rule. |
| G3 alignment | **PASS** | LF3 r2: 38/38. All 32 census rows refuse with a ValueError naming the entry, the operand and the alignment, with 0 launches; the aligned rerun is bitwise equal. The 6 acceptance rows (dense q/k/v, dense G1 k, r2 public route q/v) are bitwise equal to their aligned copy. |
| G4 refusals | **PASS** | LF4 r2: 63/63, including the launch-counter positive control. S4: 15/15. |
| G5 tickets | **PASS** | LF5: 11/11 on Build FH. EB3 and EB4 forced timeouts raise; the slot reads [0]; the next fused merge is bitwise equal to a fresh process; S3's knob behaviour holds. |
| G6 NITs | **PASS** | R9: two loader tests on real binaries. R10: one-definition test. R11: LF4 checks (empty extents, TC_APA_SP, dots, carry) plus the EB2b cursor check now `!=`. R12: preprocessor test with and without the define. |
| **G7 SASS** | **FAIL (as registered)** | Attempt 1 (module 86fb1a58) differs from B in 124 changed, 6 removed and 3 added kernels. 96 of those differences are unlisted, and they differ only in static shared-memory offsets. The final Build F (attempt 2, module 5c171477) differs in 28 changed and 3 added kernels, 750 equal, 0 unlisted. That matches `EXPECTED_SASS_F_r2.json` exactly, but `EXPECTED.json` listed 6 removals that attempt 2 does not make. |
| G8 mutants | **PASS** | 11/11, each killed by its named assertion. |
| G9 regression | **PASS** | CPU: 729 IDs (P1's G3 set of 689 plus P1-D's 40 policy tests), 0 differences, identical collections. LF6: 44 files, 2,078 IDs, 0 differences. No intended changes were listed and none were observed. |
| G10 scope | **PASS** | Writes only in the writable set, `artifacts/apa_p1f1/` and `/tmp/apap1f1_*`. GPU only through forge-gpu-run. Only read-only git (diff, archive, ls-files). Nothing deleted: superseded receipts were renamed into `attempts/`. Exception: REPORT.md was refused by the harness. |

## Diff summary
13 tracked files, +541/−101.
- **`merge.cuh`, and the fused merges in `eb3.cuh` and `eb4.cuh` (R1):** the guard is now `den!=0` instead of `den>0`. The denominator is a sum of non-negative l·e terms, so on finite inputs the two guards differ only for a NaN denominator.
- **`packed.cuh` (R1):** new `spq1::pack_int4_rows`, a copy of the kernels.cu int4 packer (which is read-only) whose only addition is storing a NaN scale for a key row with any non-finite element. The tq4 prepare already carries NaN or Inf into the stored norm; that is shown in a code comment, with no code change.
- **`eb2b.cuh` (R1):** `quantize` stores a NaN scale for a token row with any non-finite value.
- **`launcher.cuh`:**
  - shared guards: `apa_spo_require_aligned` (R2), `apa_spo_require_extents` (R11), `apa_spo_knob` (R7);
  - `apa_spq1_core_check`, which runs every check before the first launch, including from `apa_spq1_run` before its rotation;
  - fp32 with `dots='tensor'` now raises;
  - the alignment census for the staged G=1 path.
- **`spq4.cuh` / `spq6.cuh`:** extents and alignment checks. `apa_spq6_tq4_tc` now takes the rotation and rotates the query only after all checks (R11). `APA_FXTC1_VARIANT` is read only in the hooks build; the default build refuses a non-default value, and the variant kernels stay instantiated but unreachable.
- **`eb2b.cuh` (R4):** `apa_eb2b_run(..., delta, carry, ...)` validates both, with the partition layout or a whole-row partition. There is no launch-site literal. The cursor check is now `!=`. Alignment checks were added to the run and values entries.
- **`eb3.cuh`:**
  - `#error` under `CUDA_API_PER_THREAD_DEFAULT_STREAM` (R12);
  - on a test-hook timeout, `early_counter_finish(hook, ctr, slots)` zeroes the merge slot before throwing (R5);
  - a hooks-only forced-timeout mode, `APA_P1_EARLY_COUNTER=timeout`;
  - a hooks-only ticket accessor;
  - alignment, extents and knob checks.
- **`eb4.cuh`:** alignment, extents and knob checks.
- **`packed_launcher.cuh` (R6):** `int4_stats` validates rank, dtype, device, extents, KVH > 0 and H % KVH, raising ValueError.
- **`bindings.cpp`, inside the spo2 module only:**
  - `eb3_run`/`eb4_run` default `carry="partition"` (R3);
  - `eb2b_run` takes required keyword arguments `delta` and `carry`;
  - a hooks-only `_p1_test_merge_tickets`.
- **`__init__.py`, packed region only:**
  - the R9 loader: an `RTLD_NOLOAD` probe for `PyInit__tensor_cuda_spo2`; a failed load in a packed build raises at import with the real cause; a non-packed build gets clear "built without TC_APA_PACKED" stubs, raised before any allocation;
  - the workspace exact-K identity (R8);
  - the dead `_apa_spq1_entry` is removed (R10);
  - `apa_eb2b(*, delta, carry)` (R4);
  - `apa_eb3`/`apa_eb4` default to `carry="partition"` (R3).
- **`apa_packed.py` (R8):** the shared `bind_exact_k`, `is_exact_k_source` and `check_exact_k`, used by both PackedSpKeys and ApaSpq1Workspace.
- **`test_apa_p1_exact_tie.py:70`:** gains `delta=8.1, carry="row"`. This is the only change to an existing test.

## Modules and SASS
- Build B: `8728e67e26027dbafe8a61592decb38e68171fa7b5634239ce0af7897c4e8c4e`, checked before every use.
- Build F attempt 1 (`build_F`): `86fb1a58eaa1240449ca387b86418d20784dc9f1db8258315219b8da97c501eb`, superseded.
- **Build F, final (`build_F_r2`; `build_F_r3` is a no-op refresh receipt):** `5c17147715ffc422b0aa7ba8fae826c554a4caff63b1231c22ce580867efc243`
- **Build FH (`-DTC_APA_PACKED=ON` plus S3's appended `target_compile_definitions(_tensor_cuda PRIVATE TC_APA_P1_TEST_HOOKS=1)`):** `19b4a1783e981302522e79654c278e0e3c462606c2b3687c11876395756be2b6`
- Mutant modules:
  - m01 `218350b975a9c933…`
  - m02 `c4989cf794095abd…`
  - m03 `2297a037183f55ae…`
  - m04 `0f3622fb9ed750a9…`
  - m05 `21970cc77f4045c9…`
  - m07 `e234ab5161828c7d…`
  - m08 (hooks build) `4d7bcb88c200788b…`
  - m09 `9850265d2223b9ba…`
  - m10 `99db8b9095e1e5cf…`
- `EXPECTED.json` sha256: `8ee14b874eaf3cf156d7ba060b06ca450897131e5f17ac5a20460d16590af6a1` (written before the first build and the first GPU line).
- `EXPECTED_SASS_F_r2.json` sha256: `30c58ef0bbbd96034f41d2f040bc43fa3ab0d036ec7356bb6a1c4c5433235802` (written before attempt 2).
- SASS against the expected list:
  - final: 28 changed (spo2 merge ×3, the 18 FUSE=true `eb3::select_pv`, `eb4::select_mixed` ×6, `eb2b::quantize`), 3 added (`spq1::pack_int4_rows` ×3), 750 equal, 0 unlisted;
  - attempt 1: the listed 28 changed, 6 removed and 3 added, plus 96 unlisted. 90 of those differ only in the offsets of shared-memory operands; 6 (bk3/bk4) differ only in `UMOV` immediates that carry shared-memory base addresses. No other opcode differs.

## Censuses
- **Alignment (`ALIGNMENT_CENSUS.json`):** 32 rows across 9 launchers, each with its vector width and file:line load site.
  - d1 staged G=1: kq, k, v at 16 B, plus a fp32 k row.
  - packed r2 at G=1: k, v at 16 B.
  - SPQ4 and SPQ6: q at 8 B; codes, k, v at 16 B.
  - EB2b: codes, k, v at 16 B; q, vi at 8 B.
  - `eb2b_values`: v at 16 B; vi at 8 B.
  - EB3: q at 8 B; k, v, codes, rotation at 16 B.
  - EB4: rotation, codes, k, v at 16 B; q, vi at 8 B.

  Not vector-loaded, and still accepted: sinks, scales, codebook, vs, inject, flags; the spo4, spo5 and baseline paths; r2 q and codes; the dense entries. On Build B all 32 census rows fault with a sticky misaligned-address error (BND line).
- **Knobs (`KNOB_CENSUS.json`):**
  - `TC_APA_SP` is kept as the feature switch.
  - `APA_FXTC1_VARIANT` (spq6) and `TC_APA_SPQ1_REFERENCE` are compiled out of the default build; a non-default value raises ValueError naming the variable.
  - The `APA_FXTC1_VARIANT` guard in EB3/EB4 is unchanged; it already refused non-default values.
  - `APA_P1_EARLY_COUNTER` is refused in the default build when set to anything but 0.
  - `TC_APA_SP_STORAGE` (P1-D's R-A storage policy) is kept and flagged for your ruling.

## CPU and per-ID results
- **S4 (`tensor_cuda/tests/test_apa_p1f1_cpu.py`):** 15/15 pass.
  - R3: defaults and binding text.
  - R4: required keyword arguments, forwarding, no literal.
  - R8: six identity tests.
  - R9: the production non-packed module gives the clear error with 0 allocations; Build F loads; a simulated load failure raises ImportError at import, naming the initializer and the cause.
  - R10.
  - R12: with and without the define.

  The existing P1 structural tests (tickets, early counter) also pass: 45 passed, 4 GPU-only skipped.
- **CPU G3 plus P1-D policy:** before (arm B_v2 with the full f01201cf tree) and after (arm F with this tree): 729 IDs, 0 differences. Both arms: PASSED 450, FAILED 125, ERROR 15, SKIPPED 139 (historical failures, unchanged).
- **LF6 per ID:** LF6a 22 files, 1,376 IDs, 0 differences. LF6b 22 files, 702 IDs, 0 differences.

## Mutants (11/11 killed)
| Mutant | Line | Named assertion that failed |
|---|---|---|
| m01_spo2_merge_guard | LF2 | G2 NaN row-set equality: r2_tq4 bfloat16 q_nan_rev1 |
| m02_eb3_merge_guard | LF2 | G2 NaN row-set equality: eb3 bfloat16 q_nan_rev1 |
| m03_eb4_merge_guard | LF2 | G2 NaN row-set equality: eb4 bfloat16 q_nan_rev1 |
| m04_int4_scale_nan_dropped | LF2 | G2 NaN row-set equality: r2_int4 bfloat16 k_nan_rev1 |
| m05_alignment_check_passes | LF3 eb3_q | LF3: misaligned call raises ValueError before any launch… (got "DID NOT RAISE", then "misaligned address" and a broken context) |
| m06_eb3_default_row | S4 | apa_eb3 default carry must be partition |
| m07_eb2b_delta_ignored | LF4 | LF4 EB2b delta changes the selected count… |
| m08_ticket_reset_dropped | LF5 | LF5 eb3: ticket slot reads zero after the forced timeout |
| m09_int4_stats_unchecked | LF4 | LF4 int4_stats KVH=0 raises ValueError and the child exits 0 |
| m10_knob_silently_ignored | LF4 | LF4 knob APA_FXTC1_VARIANT=0 on tc_tq4 raises naming it |
| m11_workspace_identity_dropped | S4 | DID NOT RAISE |

`MUTANTS.json` sha256 starts `8796341fbc8ba195` and was written before any mutant was staged or run.

## GPU lines
Every dry run returned 78. The forge exit is passed through from the inner timeout.

| Line | Attempts | Result |
|---|---|---|
| LF1 | r1: exit 1. The launch-counter shim segfaulted (repair 1). r2: exit 0. | PASS |
| LF2 | r1: exit 76. r2: exit 1. | 76 = lock not acquired, retried unchanged after 5 min. r2's RESULT status is FAIL only from the L19 miss; G2 holds; G2b NOT HELD. |
| LF2S | r1: exit 0 | Supplementary informative +Inf-in-k case: zero-row check and L19 pass. G2b NOT HELD. |
| LF3 | r1: exit 76. r2: exit 0. | PASS, 38/38 |
| LF4 | r1: exit 1. In-process launch counter absent (repair 2); every refusal itself raised correctly. r2: exit 0. | PASS, 63/63 |
| LF5 | r1: exit 0 | PASS, 11/11 |
| MUT | r1: exit 0 | 9/9 GPU mutants killed |
| BND | r1: exit 0 | Build B boundary record, no gate |
| LF6a | r1: exit 0 | PASS |
| LF6b | r1: exit 0 | PASS |

Each line's `.start`, `.log`, `.exit` (with the log's sha256) and its dry-run log are in `artifacts/apa_p1f1/gpu/`.

## Deviations
- **D1 (G7).** In attempt 1, compiling out the unreachable FXTC1 variant kernels moved the static shared-memory placement in 96 unrelated kernels. I kept those instantiations, registered `EXPECTED_SASS_F_r2.json`, and then built attempt 2. `EXPECTED.json` and `SASS_COMPARISON_F_r1.json` are kept unchanged.
- **D2.** P1-D's build script is imported, and its adapter and flags are taken from its source via AST. Its `main()` cannot be called: it asserts the arm is A or B and that TMPDIR starts with `/tmp/apap1d_`.
- **D3.** `ops.h` is read-only, so R4's new signature reaches the binding through a global-namespace bridge (`apa_p1f1_eb2b_run`). The binding declares it at block scope; `eb2b.cuh` defines it by closing and reopening namespace tc. The hooks-only `apa_p1f1_merge_tickets` uses the same pattern. The old 14-argument `tc::apa_eb2b_run` declared in `ops.h` now has no definition and no caller.
- **D4.** A Python message changed after the attempt-2 build, so `build_F_r3` is a no-op incremental rebuild receipt; the module is identical.
- **D5.** LF1 r1: the shim's `RTLD_NEXT` lookup cannot see the RTLD_LOCAL libcudart. The shim now takes the loaded libcudart's handle, and LF1's children run without the shim.
- **D6.** LF4 r1 ran without the launch counter; LF4 now re-execs itself under `LD_PRELOAD`.
- **D7.** Two exit-76 retries (LF2, LF3), each unchanged and at least 5 minutes after the 76.
- **D8.** The L19 miss above. LF2S added one case chosen by a rule from the inputs alone, after the result was seen.
- **D9.** fp32 operands (d1 fp32 k, rotation) are misaligned by one element, 4 bytes, because a DLPack import must be element aligned.
- **D10.** My edit adding the F supplementary case to the BND line never applied (the AfterImage hook blocked it and I did not retry), so BND ran Build B only and LF2S ran as its own line.
- **D11.** CPU G3 attempt 1 used a partial git archive of the Build B tree and hit a collection error; I re-staged the full archive. Attempt 1 is in `attempts/`.
- **D12.** The S4 receipt's first run lacked the arm on PYTHONPATH.
- **D13.** The default build now refuses `APA_P1_EARLY_COUNTER`. The hooks build gains a test-only `timeout` mode.
- **D14.** `TC_APA_SP_STORAGE` was kept; it needs your ruling.
- **D15.** The R2 refusal also applies to the r2 public route at G=1, where k and v really are vector-loaded. At G>1 the route accepts misaligned operands unchanged (LF3 acceptance rows).
- **D16.** The new `int4_stats` checks raise ValueError, including for H % KVH ≠ 0, which used to be RuntimeError; G9 shows no test affected.
- **D17.** Several waits matched my own shell with `pgrep -f` and simply ran to their timeout; no result was affected.
- **D18.** REPORT.md was refused by the harness.

Script repairs under L56(b): 2.

## Prior art
- **IEEE 754-2008:** NaN propagation, and maxNum, which `fmaxf` implements and which drops NaN; this is the mechanism behind R1. Taken unchanged.
- **Online-softmax merge** (Milakov and Gimelshein, 2018; FlashAttention, Dao et al., 2022; Flash-Decoding, 2023): arithmetic unchanged; only the zero guard changed.
- **CUDA C++ Programming Guide,** vector-type natural alignment: the basis of R2. Refusing on the host before the launch is ours.
- **Symmetric absmax INT4** (Project-Tensor `apa_int4_pack_rows_kernel`) and the **KIVI** per-token int8 V format (Liu et al., 2024): taken unchanged. The non-finite carry is ours.
- **PyTorch version counters** (Paszke et al., 2019): the identity idea. The helper shared by both owners is ours.
- **CUDA Samples threadFenceReduction tickets:** the reset on the hook's failure path is ours.
- **Mutation testing** (DeMillo, Lipton and Sayward, 1978); **test doubles** (Meszaros, 2007).
- **POSIX dlopen/dlsym and LD_PRELOAD interposition:** used for the loader probe and the launch counter. The repo's TX3-F4X interposer is the in-house precedent.
- **ISO C++ block-scope function declaration:** the bridge.
- **nvlink static shared-memory placement shifting when template instantiations are removed:** no prior art known to me.
- **The forced-timeout hook mode:** no prior art known to me.
- Unverified, lead to check. Search terms: "nvlink shared memory layout relocatable device code symbol order", "RTLD_NEXT RTLD_LOCAL dependency interposition", "KIVI 2402.02750".

## Files written
All paths are under `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1d/`.
- **Modified:**
  - `tensor_cuda/src/apa_spo/{eb2b,eb3,eb4,launcher,merge,packed,packed_launcher,spq4,spq6}.cuh`
  - `tensor_cuda/src/bindings.cpp`
  - `tensor_cuda/tensor_cuda/__init__.py`
  - `tensor_cuda/tensor_cuda/apa_packed.py`
  - `tensor_cuda/tests/test_apa_p1_exact_tie.py`
- **New:**
  - `tensor_cuda/tests/test_apa_p1f1_cpu.py`
  - `scripts/apa_p1f1_{build,gpu,line,sass,mutants,cpu_g3,boundary}.py`
- **`artifacts/apa_p1f1/` (1.6 GB):**
  - registrations and censuses: `INSTRUMENTS.log`, `EXPECTED.json`, `EXPECTED_SASS_F_r2.json`, `ALIGNMENT_CENSUS.json`, `KNOB_CENSUS.json`, `BOUNDARY.json`
  - builds: `build_F/` (attempt 1), `build_F_r2/`, `build_F_r3/` (refresh receipt), `build_FH/`, `arms/{F,FH}/`
  - SASS: `sass/{B,F_r2}.sass`, `sass_r1/F.sass`, `SASS_COMPARISON_F_r1.json`, `SASS_COMPARISON_F_r2.json`
  - CPU: `cpu/{S4.json,S4.log,S4_r1_no_pythonpath.log}`, `cpu_g3/`, `CPU_G3_COMPARISON.json`
  - mutants: `mutants/{MUTANTS.json,MUTANT_RESULTS.json,CPU_MUTANT_RESULTS.json,m01…m11/}` (tree, build and arm per mutant)
  - GPU: `gpu/` (LF1–LF6b, LF2S, MUT, BND receipts)
  - superseded attempts: `attempts/`, `gpu/attempts_r1/`
- **Scratch, left in place and listed (cleanup by you):**
  - `/tmp/apap1f1_{tmp,build,mutbuild,pytest,mut,g3,gpu,l6}`
  - `/tmp/apap1f1_shim`, `/tmp/apap1f1_shim_r1`, `/tmp/apap1f1_shim_r2`
  - `/tmp/apap1f1_treeB` (about 933 MB), `/tmp/apap1f1_treeB_partial`
  - Python caches (exempt)
