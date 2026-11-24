# BWD-F2 report

Filed by the lead from the seat's final message, verbatim (the harness refuses report-file writes from Agent-tool
seats). Seat Claude Opus 5.5; order `orders/BWDF2_REV1_FIXES.md` at b475c61e2.

---

Everything the seat can do on the CPU passed, except K5's writable-set clause, which I report as FAIL (deviation 1 explains it). K1, K3's CPU half and the 4 CPU mutants pass. K2, K3's GPU half, R11's restaged lines and the 2 GPU mutants are staged, and all 53 lines dry-ran with exit 78. The round is accepted only once your GPU lines pass.

## Done

**K1 to K5**
- **K1: PASS.** All 46 CPU rows hold on Build F2 (`K1_RESULTS.json`).
  - The 35 rows that name a defect all fail on Build E. The 11 control rows behave the same on Build E.
  - Every refusal shows 0 CUDA calls on ENG3's counter. Each row runs in its own child process, because Build E crashes on some rows (SIGSEGV on `f09-getitem-rank0`).
  - BWD-F1's G1 still passes: 1522 tests (1561 minus the 39 duplicate IDs that R10 removes), plus the 46 K1 rows, 1568 passed and 0 failed (`g1/G1K1_F2.xml`). The only rows that changed are R1's four, listed in INTENDED_CHANGES_F2.
- **K2: STAGED.** 18 K2 lines (f01, f04, f07, vals, count against B, E and F2, plus the n03, n04 and n05 mutant lines). All dry-ran with exit 78.
- **K3: CPU half PASS; GPU half STAGED** (24 g4 lines).
- **K4:** the 4 CPU mutants are **KILLED**: n01, n02, n03, n06. The GPU mutants n04 and n05 (and n03's GPU read) are **STAGED**.
- **K5: FAIL on one clause.**
  - The suite runner I copied from BWD-F1 wrote Build B's copies of three files outside the writable set during the cpu_B run: `tc_signal/tc_signal/__init__.py`, `tc_signal/src/bindings.cpp` and `tc_signal/README.md`.
  - All three were restored to HEAD's bytes afterwards (sha-checked; git shows them unmodified). The staged g4 lines will do the same when you run them.
  - Everything else in K5 passes: no GPU use, no git writes, nothing deleted, every other write inside the writable set.

**Per-finding closure table** (fixed lines are in `kernels.cu` unless another file is named)

| Finding | Fix (file:line) | Test | Result |
|---|---|---|---|
| F01 BLOCKER sin span | shared `rope_tables_check` :8700, through `rope_apply_check` :8724; swap preflight :8903; arena goes through swap | K1 `f01-*` (4 rows plus a control); GPU `k2_f01_*` | CPU: PASS on F2, fails on E. GPU: staged |
| F02 BLOCKER rank ceiling | `check_meta_rank` :1114, at 14 sites (census below), including stack's rank after unsqueeze :1705 | K1 `f02-*` (11 rows plus a control) | CPU: PASS on F2, fails on E |
| F03 BLOCKER no-op members | dtype/device only for members with elements :1728, :1732; `write_rows` :8397; the RoPE device term only when the kernel reads :8700 | K1 `f03-*`; R1 rows in `test_bwdf1_copy_ops.py`; GPU `k2_vals` r9/f03 | CPU: PASS on F2, fails on E. GPU values: staged |
| F04 BLOCKER ring overflow | host `start %= CAP` :8418; overflow-free kernel wrap :8375 | GPU `k2_f04_*` | Staged (the CPU cannot reach it) |
| F05 MAJOR late RoPE checks | `export_row_pair` checks both sides first :8839; float launch dtype in the shared validator | K1 `f05-*`; GPU `k2_count` f05 rows | CPU: PASS. Work-before-refusal: GPU, staged |
| F06 MAJOR weak ordering rows | GPU ordering rows (first pair allocates and launches); CPU zero-size cat/stack controls | GPU `k2_count` f06 rows; K1 `f03-stack/cat-empty-*` | CPU controls PASS. Ordering: staged |
| F07 BLOCKER condition dtype | condition loaded in its own dtype :1562, :1630; x/y dtypes must match before allocation :1615 | K1 `f07-*`; GPU `k2_f07_*` | CPU: PASS. Condition read: staged |
| F08 MAJOR unchecked sums | cat sum :1755; splice extent :8603; arena tail start :9003; `product_dims` checks for an empty extent first :8493; `reduce_impl` :1409 | K1 `f08-*` | CPU: PASS. The `write_rows` and reduce products are not visible on the CPU (silent wrap) |
| F09 MAJOR late dtype / getitem | slice :5730; pad_into :6035; `bindings.cpp:250` | K1 `f09-*`; GPU `k2_count` f09 | CPU: PASS. The cudaMalloc counter row: staged |
| F10 MINOR test duplication | `test_bwdf1_zero_extent.py`: equal pairs generated once, axpy_ size rows apart, Z3 credited to m03 | G1 | PASS |
| F11 MINOR messages | `normalize_dim` adds context :8507; `with_index` adds "[pair i of n]" :8519; arena names the layer | K1 `f11-*` | PASS on F2, fails on E |
| F12 BLOCKER axpy_ rows | must map onto param, leading size-one axes allowed :5668 | K1 `f12-*`; GPU `k2_vals` | CPU: PASS. GPU values: staged |
| F13 MINOR einsum | `functional.py:241` | K1 `f13-einsum-shorter-term` | PASS on F2, fails on E |

**R2 array census summary** (`CENSUS_F2.json`; found by declared size, not by function name)
- 7 specs are indexed by axis and sized TC_MAX_DIMS = 8: DimSpec, ReduceSpec (plus `reduce_kernel`'s local `coord`), PermSpec, Dim3Spec, DimCopySpec, GatherSpec and FlipSpec.
- 10 dispatch functions consume them, and every one checks the rank before writing its spec:
  - ew_binary, compare (DimSpec);
  - reduce_impl (ReduceSpec);
  - permute (PermSpec);
  - where_nd (Dim3Spec);
  - cat_nd, through check_copy_operands; slice_nd; pad_into (DimCopySpec);
  - gather_nd, scatter_add_nd (GatherSpec);
  - flip_nd (FlipSpec).
- 4 more checks sit before work that comes ahead of the spec:
  - `broadcast_to`, before its zeros allocation;
  - `axpy_`, before `ew_scalar` launches;
  - `ops.cpp` `masked_fill`, before its fill;
  - stack's rank after unsqueeze (nd + 1), inside check_copy_operands, before the unsqueeze.
- Count by role: 15 product sites, 1 test file, 0 consumers, 18 harness scripts.
- The other 300 fixed arrays don't depend on rank: xyz geometry (size 3), the 8 inputs of gated_delta_step, `kl/ql/sl[8]` (bounded by Dk ≤ 256), cuBLASLt's `heur[8]`, accumulator widths, and shared-memory reduction buffers.
- The index-driven ops (gather, scatter) got only the rank ceiling. Their index checks remain a residual.

**Diff summary** (`source_F2.diff` sha256 161d77dd…, against fbe04d8f9: 5 files, +372/−79)
- kernels.cu (+416 changed lines): the R1 to R8 and R10 fixes above, plus the helpers `check_meta_rank`, `rope_tables_check`, `rope_apply_check`, `with_index`, `checked_add3` and a checked `product_dims`. `where_kernel` is now templated on the condition's dtype.
- core.h: declares `check_meta_rank` (+3).
- ops.cpp: the rank check in masked_fill (+11).
- bindings.cpp: the getitem axis check (+12).
- functional.py: the einsum preflight (+9/−1).
- Tests (`tests_F2.diff`): `test_bwdf1_copy_ops.py` and `test_bwdf1_zero_extent.py` changed (+75/−25). New file: `test_bwdf2_fixes.py`.

**Every module sha256**

| Build | sha256 |
|---|---|
| tc B | 70cd17528077d2bbfc2875ec5059213cd56f897985520c30c022c8bba0e10b45 |
| ts B | 857432c55bf0de21309debdf5716c6e05176e113590fd41314d67c59ede11e0c |
| tc E | 8b84e5d872e382d3f29360bdf7b7cb253978d41c8a450ee250d168e9999d7f30 |
| ts E | 1ab3757f55c88c0c693bd1ce33d0363892493f0609c4eaaa4715d2906f5dd8d8 |
| tc F2 | 5ff79dee0336e47171f193c6b1c9a69bbe5512cd1a03b0456328e1ae981eff4a (source map 47c077c5…) |
| ts F2 | 6787d405b592e14915f80735fdf5ee39953a1448522c9f856aeef00e14afc1ff (source map 7505dc09…) |
| n01 | ec8f6504a3d1a956a44347eb5c48e4dd43b3b772abb61c679886878eb6dcbcef |
| n02 | 23dbe20115a4ec936a1bd86f1ddb34859054ea59bdcf3dd9fbb48996281a20a6 |
| n03 | ea4636216c24e0b176dcb6b5e0405dd2a25078d4e30b26f279bf90600d2a5e0c |
| n04 | 0c3e7fa285a285c8474560c2437e216a838ac739826b0e6bdee2d3df30d613e1 |
| n05 | b93271da2a964212092f2cec1432a558dfbd30109d54df8f676bfe2aafc8c995 |
| n06 | d5d2edeb57210c81c45c9c62a0fc9f375a8dd681b3a8d5380161165c9a6503d7 |
| m03 (BWD-F1's, restaged) | 9e733b34386b0d3b0e9ac73496061ea40f35660fcf743b2838e740378dfdac28 |

**K3's CPU per-ID result and the intended changes**
- Result: **PASS** (`K3_CPU_COMPARE.json`). 4656 IDs per build, both series complete.
  - Build B: 1645 passed, 2176 failed, 820 skipped, 15 errors.
  - Build F2: 1647 passed, 2174 failed, 820 skipped, 15 errors.
- The only differences are BWD-F1's two listed H1 IDs (`test_bwd_r3_paths.py::test_host_real_planner_and_no_grad[conv1d|conv_transpose1d]`, failed on B and passed on F2). 0 unintended.
- Both builds have the same collection errors. No test wrote into the worktree.
- L64: `g4/L64_OPEN_AUDIT_cpu_F2.json` (9586 distinct paths outside the worktree).
- Intended changes (`INTENDED_CHANGES_F2.json`, written before any test after-run): under R12, four BWD-F1 copy-op rows move from refusals to "not refused" (cat-device, stack-device, write_rows-device, export_rope_rows-table-device). No production test changed. The R9 and R10 test edits are listed separately in the same file.

**Mutant table** (`MUTANTS_F2.json` written before any mutant tree existed; results in `MUTANT_RESULTS_F2.json`)

| Mutant | Killing assertion | Result |
|---|---|---|
| n01_dtype_check_all_members | the 3 `f03` rows that expect the device step; the mutant refuses with "operand 1 has dtype float16/bfloat16" | KILLED (3/3) |
| n02_no_rank_ceiling_stack | `f02-stack-rank8-to-9` and `f02-cat-rank9-empty`; the mutant reaches "CUDA error at cat" | KILLED (2/2) |
| n03_sin_span_unchecked | the 3 `f01` short-sin rows; the mutant returns instead of refusing | KILLED (3/3); GPU line `k2_f01_n03` STAGED |
| n04_ring_no_modulo | `k2_f04`: buf must equal NumPy's ring write and memcheck must report 0 | STAGED (`k2_f04_n04`) |
| n05_condition_as_data_dtype | `k2_f07`: the bool-condition rows must equal np.where bitwise | STAGED (`k2_f07_n05`) |
| n06_axpy_broadcast_accepted | `f12-axpy-named-case` and `f12-axpy-trailing-broadcast`; the mutant reaches "CUDA error at ew_scalar" | KILLED (2/2) |

**Each staged GPU line with its dry-run exit** (`GPU_LINES_F2.json`, `gpu_dry/DRY_RUN.json`)
- All 53 lines dry-ran once, exactly as written: **exit 78**. No write-once path was created.
- Each line's gate checks module sha256s and write-once paths before `forge-gpu-run`. Every step script has the `--require-held` lease check at line 2 (L125), listed in `lease_check_lines`.
- R11, BWD-F1's lines with Build E replaced by F2:
  - g2_f01_{B,F2}, g2_f03_{B,F2}: memcheck, `--min` 240;
  - g2_direct_{B,F2}: memcheck, 300;
  - copy_{B,F2}: memcheck, 240;
  - g3_{B,F2,m03_reduce_gt1}: 300;
  - g4_{B,F2}_01 to 12: 480.
- K2: k2_f01_{B,E,F2,n03}, k2_f04_{B,E,F2,n04}, k2_f07_{B,E,F2,n05} and k2_vals_{B,E,F2} under memcheck; k2_count_{B,E,F2} without memcheck (it uses the counter). All `--min` 240.
- Every line uses `--max 540` and an inner `timeout 500s`.
- After the lines run, three CPU follow-ups give the verdicts: `bwdf2_gpu_post.py line <line>`, `controls` (B against F2, bitwise) and `g4`.
- I drove the post-device code on constructed results first: `POST_SELFTEST_F2.txt` 46 checks PASS, and `POST_READBACK_DRIVE_F2.txt` 18 readbacks PASS (3 dtypes × 6 shape classes, empty and one-element included).

**Deviations**
1. K5 (above): three tc_signal files outside the writable set were written temporarily by the copied suite runner, then restored.
2. L56(b): I repaired the K1 harness twice. First to run one child per row, after Build E's crash hid the other rows; second to record tuple results.
3. Builds use BWD-F1's configuration: Release, CUDA arch 89, -j4.
4. The CPU cannot hold a CUDA operand with elements. So F06's ordering witnesses, and every value check, are GPU rows. The CPU later-pair rows are labelled as refusal rows with no ordering claim.
5. I rewrote home paths as `$HOME` in the suite receipts and logs.
6. Residuals, not fixed:
   - index validation in gather and scatter;
   - `reduce_impl` axis values;
   - einsum terms longer than their operand's rank, which can still reach `permute` with extra dims;
   - the F08 product overflows in `write_rows` and `reduce_impl` give the same result through a silent wrap, so the CPU can't show them;
   - `where`'s `default:` branch is unreachable, since every DType is handled.

**Every file written**
- Product files: `tensor_cuda/src/kernels.cu`, `src/ops.cpp`, `src/bindings.cpp`, `include/tc/core.h`, `tensor_cuda/functional.py`.
- Installed modules: Build F2's two modules in the worktree packages.
- Tests:
  - new: `tensor_cuda/tests/test_bwdf2_fixes.py`;
  - changed: `tensor_cuda/tests/test_bwdf1_copy_ops.py`, `tensor_cuda/tests/test_bwdf1_zero_extent.py`.
- Scripts: `scripts/bwdf2_{build, freeze_pkgs, k1, make_mutants, mutant_results, suite, k3_compare, gpu_common, gpu_lines, gpu_post, post_readback_cpu, boundary, census, step_g2_direct, step_g3, step_copy, step_g4, step_k2}.py`.
- Under `artifacts/bwdf1/f2/`:
  - `INSTRUMENTS.log`, `INTENDED_CHANGES_F2.json`, `K1_RESULTS.json`, `MUTANTS_F2.json`, `MUTANT_RESULTS_F2.json`, `K3_CPU_COMPARE.json`;
  - `BOUNDARY_F2.json`, `CENSUS_F2.json`, `L55_F2.json`, `GPU_LINES_F2.json`, `POST_SELFTEST_F2.txt`, `POST_READBACK_DRIVE_F2.txt`;
  - `source_F2.diff`, `tests_F2.diff`, `LEDGER_F2.md`;
  - directories: `before/`, `F2_src/`, `build/`, `g1/`, `g4/` (cpu_B, cpu_F2, L64 audit), `mutants/` (diffs and runs), `gpu_lines/{B,F2}/`, `gpu_dry/`.
- Build directories: `tensor_cuda/build-bwdf1/{F2, n01…n06}` (63 MB each) and `tc_signal/build-bwdf1/F2` (9.5 MB).
- Logs: `logs/bwdf2_k3_cpu_B.log`, `logs/bwdf2_k3_cpu_F2.log`.
- Scratch: `/tmp/bwdf2_*`, including the `/tmp/bwdf2_src_*` trees. Nothing was deleted. The n0* build directories must stay until your GPU mutant lines have run.

## Prior art
- NumPy's `concatenate` and `stack` (shape rules for every operand, empty ones included), `where` and `putmask` (the condition is truth-tested in its own dtype), and its broadcasting rules: taken as the rules. Mine: dtype and device are checked only for operands whose bytes are read (R1).
- RoPE: Su et al., 2021, "RoFormer". It sets the absolute-position table read that the shared validator bounds. Mine: the shared validator and its term order.
- Checked arithmetic: GCC's `__builtin_add_overflow` and `__builtin_mul_overflow` (2015), and P1T-FIX2's rule of checking for a zero extent before multiplying. Unverified, lead to check CERT INT32-C: "CERT INT32-C signed overflow checked addition".
- Ring wrap: the standard overflow-free circular-buffer wrap. Unverified, lead to check: "circular buffer index overflow modulo".
- axpy_ (R6): PyTorch's in-place broadcast rule. Unverified, lead to check: "output with shape doesn't match the broadcast shape".
- getitem: NumPy's IndexError "too many indices".
- Rank ceiling: the engine's own TC_MAX_DIMS refusals (P1T-FIX1 F03). No prior art known to me for the per-consumer placement.
- Mutation testing: DeMillo, Lipton and Sayward (1978). Call counting: ENG3's LD_PRELOAD interposition (ltrace-style).
- BWD-F1's build, suite, staged-line and post scripts were copied and adapted.

Paths:
- Worktree: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_bwdf1`
- Receipts: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_bwdf1/artifacts/bwdf1/f2/` (ledger: `LEDGER_F2.md`)

---

## Lead notes (2026-11-23)

- Lead checks on disk: `K3_CPU_COMPARE.json` PASS, 0 unintended; `K1_RESULTS.json` 46 rows; `g1/G1K1_F2.xml` 1,568
  tests, 0 failures; installed tc module `5ff79dee`; the three tc_signal files show unmodified in git.
- **K5 ruling:** the clause fails as registered and is recorded so, with no rerun. The writes came from the arm-swapping
  suite runner the order let the seat copy; all three files were restored byte for byte, and the product is unaffected
  (precedent: ENG3 G5). A lead order fault in part: the order's writable set did not list the files that runner swaps.
  Future orders that let a seat copy an arm-swapping suite runner list those files as temporarily written and restored.
