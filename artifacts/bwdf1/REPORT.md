# BWD-F1 report

Filed by the lead from the seat's final message, verbatim (the harness refuses report-file writes from Agent-tool
seats). Seat Claude Opus 5.5; order `orders/BWDF1_ZERO_EXTENT_AND_AUDIT_FIXES.md` (62026a267) with Amendment 1
(00f70f96c). The seat's first report was RED on G4's CPU half; the lead ruled (a) and the seat applied it; this is the
updated report.

---

# BWD-F1 report (with Amendment 1, Part C), updated after lead ruling (a)

**Verdict: the seat's part passes.** G1, G4's CPU half, the CPU mutants (G5) and G6 pass. G2, G3, G4's GPU half and m03's GPU line are staged, and every staged line dry-runs with exit 78. The round is accepted only once your GPU lines pass.

I applied ruling (a):
- The two IDs are listed as intended H1 changes, marked as listed after the after-run, by the ruling, 2026-11-23.
- Only the assertion of the call at `test_bwd_r3_paths.py:60-61` changed. It now expects RuntimeError "tc_signal: {op}: out cannot require grad"; the test's other two calls are unchanged.
- I re-ran G4's CPU half on both builds. The two listed IDs are the only differences, so `G4_CPU_COMPARE_R2.json` is PASS.

I did not write `REPORT.md`: this environment tells seats not to write report files, so this message is the report.

## Done

**G1 to G6**
- **G1: PASS.** 1561 tests passed on Build E with `CUDA_VISIBLE_DEVICES` empty (`g1/G1_E.xml`). The refusal rows check the shapes in the message and use ENG3's counter to show 0 CUDA calls, with a positive control.
  - 817 Z2 refusal rows (0 against an extent above 1). Ops: the 11 binary ops; `where` in each position; `masked_fill`; `expand`; `axpy_`; `einsum`. All three float dtypes, ranks 1 to 4 and across ranks, both operand orders.
  - 550 rows of 0 against 1: each returns NumPy's shape with no CUDA call.
  - The guard's host harness rows, and the Z3 routing row.
  - Every H3 message reachable before device work: `attach_vjp`'s value error, matmul backward's first refusal (2 rows), tc_signal's backward entry (5 rows).
  - H1: the 5 tracked ops, plus `rint` as the reference row and the controls.
  - 67 Part C rows.
  - The same tests on Build B: 1384 failed and 19 errors on the core file, 10 failed on the tc_signal file.
- **G2: STAGED.**
- **G3: STAGED.**
- **G4: CPU half PASS** after the ruling. **GPU half STAGED.**
- **G5:** CPU mutants **PASS** (6 of 6 killed). m03's GPU line **STAGED**.
- **G6: PASS.** Every write is in the writable set (checked with `git status --ignored`); the test edit is in by the ruling. No GPU use, no git writes, nothing deleted. One rule slip is under deviation 6.

**Z1 census summary** (`CENSUS.json`)
- 11 product sites:
  - S1 `broadcast_shape` (kernels.cu:1028) used the max rule, so 0 against 1 gave 1. Fixed.
  - S2 `bcast_strides`: unchanged.
  - S3 `ew_binary`, S4 `compare`, S5 `where_nd`: now carry the guard; `where_nd` names all three shapes.
  - S6 `broadcast_to`: shapes are now checked before its zeros allocation.
  - S7 `reduce_to`: fixed (Z3).
  - S8 matmul's batch rule: unchanged. It never gives a non-NumPy extent and refuses a strict subset.
  - S9 `einsum`: a per-index check now runs before its permutes.
  - S10 `axpy_` and S11 `masked_fill`: refusals now come first. These two were first listed under public ops and promoted to sites after their edits.
- 19 public ops, with call paths.
- `reduce_to` callers: ops.cpp:873, ops.cpp:1240, autograd.cpp:90.
- Consumers, by grep only: Voice not found (`model.py:792` pads ids, so T is at least 2). GRAPA not found.

**Z2 classes and witnesses** (`L55.json`, compiled from each build's own source text)
- **0 against 1, `[4,0]` and `[4,1]`:**
  - Build B: output `[4,1]`; operand a's strides `[0,1]`; every element is read at offset 0 of an empty buffer. That is a read at address 0x0, which matches the lead's memcheck.
  - Build E: output `[4,0]`, nothing launched.
- **0 against an extent above 1, `[4,0]` and `[4,3]`:** Build B says "shapes not broadcastable". Build E says "shapes not broadcastable: [4, 0] and [4, 3] (extents 0 and 3 at output axis 1)".
- **Equal, `[4,3]` and `[4,3]`:** the same output and strides on B and E.
- **1 against an extent above 1, `[4,1]` and `[4,3]`:** the same output and strides on B and E (operand a's strides `[1,0]`).

**Diff summary** (`source.diff`, sha256 cf42bd0a…; 9 files, +441/−73)
- kernels.cu:
  - the NumPy shape rule and the three-shape helper;
  - the dispatch guard, in a host-only block marked BWDF1-HOST;
  - the `broadcast_to`, `axpy_` and `reduce_to` changes;
  - Part C's `check_copy_operands`, the slice and pad_into checks, the write_rows terms, the row ops' checks split from their runs, and the batched pair ops checking every pair before any work.
- core.h: two helper declarations.
- ops.cpp: the masked_fill precheck, the stack check, and the H3 messages.
- tensor_cuda bindings.cpp: the H3 value message and the H2 docstring.
- functional.py: einsum's per-index check.
- tc_signal: H1 in `__init__.py`, the H3 messages in its bindings.cpp, and H2/H4/H5 in the README. The second "Gradients" section now points to the first.
- `tc_signal/tests/test_bwd_r3_paths.py:60-61`: the one assertion changed by ruling (a).

**Every module sha256**

| Build | Module sha256 |
|---|---|
| tc B | 70cd17528077d2bbfc2875ec5059213cd56f897985520c30c022c8bba0e10b45 |
| ts B | 857432c55bf0de21309debdf5716c6e05176e113590fd41314d67c59ede11e0c |
| tc E | 8b84e5d872e382d3f29360bdf7b7cb253978d41c8a450ee250d168e9999d7f30 |
| ts E | 1ab3757f55c88c0c693bd1ce33d0363892493f0609c4eaaa4715d2906f5dd8d8 |
| m01 | 45bd820927653450a9066af3085eca8b94e68ab08a1fbde07a146ee48cd34bf9 |
| m02 | 456455ad9a4adc8a2100bd0b9b7f6ed7221994bf2673a1d9a5a558e65c807530 |
| m03 | 9e733b34386b0d3b0e9ac73496061ea40f35660fcf743b2838e740378dfdac28 |
| m05 | 387bb9c580a76f561fc0a2c4f27d66cb235273ab048a1ee2b9376106c47992eb |
| m06 | d53a9a48cc70c24ec15c716bc2de76e8f96c178b9ac68d2654303245f074d9a1 |

Superseded and never used for a gate: tc E ba1e04b2…, m01 141d13b2…, m02 fd09bbfd…, m03 27d99b0d…. m04 is Python only and uses E's modules.

**G4's CPU per-ID result and the intended changes**
- Re-run after the ruling (series `cpu_B_r2` and `cpu_E_r2`; `G4_CPU_COMPARE_R2.json`): **PASS**, 4656 IDs per build, both series complete.
  - Build B: 1645 passed, 2176 failed, 820 skipped, 15 error.
  - Build E: 1647 passed, 2174 failed, 820 skipped, 15 error.
  - Differences: exactly the two intended IDs, failed on B and passed on E. 0 unintended.
  - Collection errors, the same in both: `test_selector_accuracy.py` and `tests/test_bp_kernel_2.py`.
  - No test wrote into the worktree. `test_terrain_render.py` and this round's own tests were excluded.
- The runner leaves `test_bwd_r3_paths.py` as it is in the worktree for both builds, so both run the new assertion.
- Against the first runs (`cpu_B`, `cpu_E`; `G4_CPU_COMPARE.json`, FAIL with 2 unintended), only those two IDs moved, in both builds.
- Intended changes (`INTENDED_CHANGES.json`):
  - Before the after-run: none.
  - After the after-run, by lead ruling (a), 2026-11-23, as H1 intended changes: `tc_signal/tests/test_bwd_r3_paths.py::test_host_real_planner_and_no_grad[conv1d]` and `[conv_transpose1d]`. Old assertion: BufferError matching "CUDA". New: H1's RuntimeError "tc_signal: {op}: out cannot require grad".

**Mutant table** (`MUTANTS.json` written before any run; results in `MUTANT_RESULTS.json`)

| Mutant | What fails | Result |
|---|---|---|
| m01_max_rule | the 0-against-1 shape rows (the guard refuses on the host) and the harness rows: 559 failed | KILLED |
| m02_no_empty_operand_guard | the 5 guard harness rows (`raised` is None) | KILLED |
| m03_reduce_gt1 | the CPU Z3 row ("reshape: element count mismatch") | KILLED on CPU; GPU line g3_m03 STAGED |
| m04_h1_dropped | the 5 tracked-op H1 rows (BufferError instead of the refusal) | KILLED |
| m05_cat_no_extent_check | 4 rows: both Voice cases, the shorter-later case, stack's shape case | KILLED |
| m06_cat_no_dtype_check | 2 rows: cat's dtype case, stack's dtype case | KILLED |

**Each staged GPU line with its dry-run exit**
- 35 lines in `GPU_LINES.json`. Each has a pre-lease gate that checks module sha256 and write-once paths before `forge-gpu-run`.
- All 35 dry-ran once, exactly as written: **exit 78**, no write-once path created (`gpu_dry/DRY_RUN.json`).

| Lines | Memcheck | `--min` / timeout |
|---|---|---|
| g2_f01_B, g2_f01_E, g2_f03_B, g2_f03_E (the audit's scripts, only the module pin and output path changed) | yes | 240 / 500s |
| g2_direct_B, g2_direct_E | yes | 300 / 500s |
| copy_B, copy_E (Part C) | yes | 240 / 500s |
| g3_B, g3_E, g3_m03_reduce_gt1 | no | 300 / 500s |
| g4_B_01 to g4_B_12, g4_E_01 to g4_E_12 | no | 480 / 500s |

- After the lines run, three CPU follow-ups give the verdicts: `bwdf1_gpu_post.py line <line>`, `controls` and `g4`.
- The post-device code was driven on constructed results first: 43 checks PASS, plus 15 readbacks of each dtype on empty and one-element shapes, PASS.
- For the GPU G4 half: both builds run the new assertion, so the same two IDs should differ, and `g4` reads them as intended from `INTENDED_CHANGES.json`.

**Deviations**
1. Builds use ENG4's configuration (Release, CUDA arch 89, -j4), not CMake's default architecture list.
2. My test row table was repaired twice (L56(b)): a rank-0 operand construction, a non-empty CUDA operand, a span that was valid, and rope tables that needed D = 0. A third fix was a wrong expected substring.
3. The first Build B CPU suite run aborted because `before/` lacked functional.py. The worktree was restored to Build E and the empty part directory was moved aside, not deleted.
4. The C1 census was written after the C2 edits that were made from reading it. S10 and S11 were promoted to site rows after their edits.
5. One command that built the five mutants passed the tool's 600 s limit and was moved to the background. I waited for its completion notice and ran no other CPU-busy work meanwhile.
6. I ran `bwdf1_step_g3.py E` once directly, outside a lease. It refused at its line 2, before any import or CUDA call. This is a rule slip with no GPU effect (lead noted).
7. Home paths in suite logs and receipts were rewritten as `$HOME`.
8. My Z4 search missed H1's effect on an existing test. It was listed after the after-run by lead ruling (a), and the one assertion was changed by that ruling.

**Every file written**
- Product files:
  - tensor_cuda: `src/kernels.cu`, `src/ops.cpp`, `src/bindings.cpp`, `include/tc/core.h`, `tensor_cuda/functional.py`.
  - tc_signal: `tc_signal/__init__.py`, `src/bindings.cpp`, `README.md`.
- Existing test changed by the ruling: `tc_signal/tests/test_bwd_r3_paths.py`.
- Installed modules: Build E's two modules in the worktree packages.
- New tests: `tensor_cuda/tests/test_bwdf1_zero_extent.py`, `tensor_cuda/tests/test_bwdf1_copy_ops.py`, `tc_signal/tests/test_bwdf1_hardening.py`.
- Scripts: `scripts/bwdf1_{build, suite, make_mutants, mutant_results, gpu_common, gpu_lines, gpu_post, step_g2_direct, step_g3, step_copy, step_g4, freeze_pkgs, l55, boundary, post_readback_cpu}.py`.
- Under `artifacts/bwdf1/`:
  - `INSTRUMENTS.log`, `CENSUS.json`, `INTENDED_CHANGES.json`, `MUTANTS.json`, `MUTANT_RESULTS.json`, `BOUNDARY.json`, `L55.json`, `GPU_LINES.json`, `G4_CPU_COMPARE.json`, `G4_CPU_COMPARE_R2.json`, `POST_SELFTEST.txt`, `POST_READBACK_DRIVE.txt`, `source.diff`, `LEDGER.md`;
  - `before/` (now with the test file's before copy), `E_src/`, `build/`, `g1/`;
  - `g4/`: `cpu_B`, `cpu_E`, `cpu_B_r2`, `cpu_E_r2`, `cpu_B.aborted_missing_before_file`, `L64_OPEN_AUDIT_cpu_E.json`;
  - `gpu_lines/{B,E}/`, `gpu_dry/`, `mutants/`, and an empty `gpu/`.
- Build directories: `tensor_cuda/build-bwdf1/` (439 MB) and `tc_signal/build-bwdf1/` (19 MB).
- Logs: `logs/bwdf1_*` (11 files, including `bwdf1_g4_cpu_B_r2.log` and `bwdf1_g4_cpu_E_r2.log`).
- Scratch: `/tmp/bwdf1_*`. I would remove it (it is not deleted).

**C1 census summary** (`CENSUS.json` amendment_1_c1_copy_ops)

| Op | What Build B checked |
|---|---|
| `cat_nd` | only a non-empty list |
| `stack` | nothing |
| `slice_nd` | nothing |
| `pad_into` | nothing |
| `write_rows` | everything except device, start ≥ 0 and the zero-divisor case |
| `export_rows`, `splice_rows`, `evict_rows` | every term |
| `export_rope_rows` | everything except sin's rank and rows, and the table device |
| the 5 batched pair ops | each pair only inside the loop, after earlier pairs had already run |

Index-driven ops (scatter, gather, embedding) and `rope_apply`'s table gaps were left unchanged and are listed as residuals. The Voice consumer at `grad.py:301` is report-only.

**C2 refusal classes and witnesses** (`L55.json`, `BOUNDARY.json`)
- **Rank:** `cat([2,3], [2,3,1])`. Build B ignores operand 1's third extent; Build E refuses.
- **Non-cat extent:**
  - Voice case `[1,512,0]` with `[1,1,1]` on dim 1: Build B allocates `[1,513,0]` and operand 1's copy writes past the end; Build E refuses.
  - Voice case `[1,3,5]`, `[1,2,6]`, `[1,4,6]`: Build B's third operand writes past the end; Build E refuses.
- **Dtype:** float32 with float16. Build B reads the float16 operand as 4-byte elements, past its end; Build E refuses.
- **Device:** host with CUDA. Build B launches with any pointer; Build E refuses.
- **Dim range:** dim 2 on rank 2. Build B indexes the shape vector out of range on the host; Build E refuses.

Measured on Build B: cat-dim-low, stack-dim-low and slice-dim-low abort the process with heap corruption, and cat-rank0 segfaults.

**C3 rows**
- **G1:** 67 refusal rows, all PASS with 0 CUDA calls (`g1/records_E/g1_copy_ops.json`).
- **Staged memcheck line:** copy_B and copy_E, dry-run exit **78**. Twelve cases: Voice's two plus one mismatch per other op.
- **m05:** KILLED (4 of 4 rows). **m06:** KILLED (2 of 2 rows).

## Prior art
- NumPy's broadcasting rule (`numpy.broadcast_shapes`) is the Z2 rule.
- PyTorch's `sum_to_size` and NumPy's empty sum give the `reduce_to` rule.
- `numpy.concatenate` and `numpy.stack` give the cat and stack rules. The device and dtype terms are mine.
- H1 follows PyTorch's refusal of `out=` tensors that require grad. Unverified, lead to check: "out= arguments don't support automatic differentiation".
- ENG3's LD_PRELOAD call counter (ltrace-style interposition) proves no CUDA call.
- Mutation testing: DeMillo, Lipton and Sayward (1978).
- L64's audit hook uses PEP 578.
- The dispatch guard's check and the exact-text host harness: no prior art known to me. Unverified, lead to check: "empty operand non-empty output broadcast check".
- ENG4's build, suite and staged-line scripts and BWD-AUD1's witness scripts were copied and adapted.

Paths:
- Worktree: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_bwdf1`
- Receipts: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_bwdf1/artifacts/bwdf1/` (ledger: `LEDGER.md`)
