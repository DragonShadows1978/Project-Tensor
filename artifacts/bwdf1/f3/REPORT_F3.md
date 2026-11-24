# BWD-F3 report

Filed by the lead from the seat's final message, verbatim (L127: a Claude seat's final message is its report). Seat
Claude Opus 5.5; order `orders/BWDF3_REV2_FIXES.md` at 400a8389f.

---

P1, P3's CPU half, the four CPU mutants and P5 pass. P2, P3's GPU half and the p04 GPU mutant line are staged: all 66 lines dry-ran with exit 78. The round is accepted only once your GPU lines pass.

## Done

**P1 to P5**
- **P1: PASS** (`P1_RESULTS.json`). 57 CPU rows, each in its own child process, with CUDA_VISIBLE_DEVICES empty and ENG3's call counter.
  - 33 E1-E4 rows: every one fails on Build F2 (5ff79dee) and holds on Build F3.
  - 12 control rows hold on both builds.
  - 12 D1 rows hold on F3 and are recorded as domain refusals, with F2's and B's answers.
  - Every refusal row shows 0 CUDA calls. The counter's positive control sees 4 calls.
  - BWD-F1's G1, BWD-F2's K1 and P1 on F3: **1625 passed, 0 failed** (`g1/G1K1P1_F3.xml`). The only rows that changed are the 8 listed in `INTENDED_CHANGES_F3.json`.
- **P2: STAGED.** 13 P2 lines (e1, e2 and e3 under memcheck on B, F2 and F3; count on B, F2, F3 and p04), all dry-run exit 78.
- **P3:**
  - CPU half **PASS** (`P3_CPU_COMPARE.json`).
  - GPU half **STAGED** (24 g4 lines, all dry-run exit 78).
- **P4:** p01, p02, p03 and p04 are **KILLED** on the CPU. p04's GPU line is **STAGED**.
- **P5: PASS.**
  - Every write is in the writable set. No GPU use, no git writes, nothing deleted.
  - L128: the suite runner's temporary writes were restored and checked by sha256 after each run (`l128/CHECKS.json`):
    - pre_cpu_B → post_cpu_B: PASS;
    - post_cpu_B → post_cpu_F3: PASS;
    - post_cpu_F3 → end_of_round: PASS;
    - every snapshot matches Build F3's source map, and both installed modules are F3's.

**Per-finding closure table** (lines are `kernels.cu` unless another file is named)

| Finding | Fix or ruling | Test | Result |
|---|---|---|---|
| F01 BLOCKER, empty RoPE span refused | E1: the span term applies only when the launch reads (:8812). The same validator serves rope_apply, the exports, the swap and the arena. | P1 `e1-*` (9 rows, 3 controls); GPU `p2_e1_*` | CPU PASS. GPU staged |
| F02 BLOCKER, `[T, D, 1]` table refused | E2: `rope_table_layout_ok` (:8797, used at :8810) accepts rank 2, or extra extents that are all 1 | P1 `e2-*` (4 rows, 4 controls); GPU `p2_e2_*` (bitwise against the same call with `[T, D]` tables) | CPU PASS. GPU staged |
| F03 BLOCKER, axpy_ empty broadcast | D1 ruling. The check is unchanged (:5719) | P1 `d1-axpy-empty-destination-shape` | Domain refusal (holds on F3) |
| F04 BLOCKER, rank-9 empty operands | D1 ruling. `check_meta_rank` is unchanged (15 call sites) | P1 `d1-rank9-*` (7 rows) | Domain refusal |
| F05 BLOCKER, where with mixed dtypes | D1 ruling, plus code: x/y dtype is now checked on every call (:1666), including an empty output | P1 `d1-where-*` (2 rows); K1 `f07-where-empty-mixed-data` changed | Domain refusal |
| F06 BLOCKER, reduction axes | E3: `normalize_reduce_axes` (:1326) in reduce_impl (:1442) and reduce_arg (:5872); `ops::mean` (ops.cpp:1062) | P1 `e3-*` (17 rows, 4 controls); GPU `p2_e3_*` and the count line | CPU PASS. GPU staged |
| F07 MAJOR, late byte bound | E4: `preflight_out_bound` (:8602) runs the constructor's own `ndarray_checked_nbytes` (:539) in 5 preflights (splice at :8682) | P1 `e4-*` (3 rows, 1 control); GPU `p2_count_*` e4 rows | CPU PASS. Ordering: GPU staged |
| F08 MINOR, table dtype checked when nothing is read | D1 ruling. The term still runs on every call (:8808, unchanged) | P1 `d1-rope-table-dtype-*` (2 rows) | Domain refusal |

**D1 domain-refusal rows** (Build B's answer is the one recorded in `BOUNDARY_F3.json`)

| Row | Build B | Build F2 | Build F3 |
|---|---|---|---|
| d1-where-empty-mixed-dtype | returned `[0]` | returned `[0]` | refused |
| d1-where-all-true-mixed-dtype | reached `CUDA error at where` | refused | refused |
| d1-rank9-binary-empty, -compare-, -where-, -expand-, -masked_fill- | each returned the empty result (0 calls) | refused | refused |
| d1-rank9-axpy-empty | returned None (1 call) | refused | refused |
| d1-rank9-sum-empty | `cudaMalloc failed` | refused | refused |
| d1-axpy-empty-destination-shape | returned None (1 call) | refused | refused |
| d1-rope-table-dtype-empty, -swap-empty | refused | refused | refused |

**Diff summary**
- Product: `source_F3.diff` (sha256 2220560d…), against e4ad8381a: 3 files, +138/−15.
  - `kernels.cu`:
    - the NDArray constructor's byte bound is moved, unchanged, into `ndarray_checked_nbytes`;
    - `normalize_reduce_axes` is called by reduce_impl and reduce_arg;
    - where_nd refuses a dtype mismatch on every call;
    - `rope_table_layout_ok`, and the span term now under `reads`;
    - `preflight_out_bound` in export_rows_check, splice_rows_check, evict_rows_check, export_rope_rows_check and rope_apply_check.
  - `ops.cpp`: mean checks its axes before computing its count.
  - `core.h`: two declarations.
- Tests: `tests_F3.diff` (sha256 8a14f01f…).
  - `test_bwdf2_fixes.py`: 6 rows changed.
  - `test_bwdf1_copy_ops.py`: 2 rows moved to a new test ID.
  - New file: `test_bwdf3_fixes.py`.

**Every module sha256**

| Build | sha256 |
|---|---|
| tc B | 70cd17528077d2bbfc2875ec5059213cd56f897985520c30c022c8bba0e10b45 |
| ts B | 857432c55bf0de21309debdf5716c6e05176e113590fd41314d67c59ede11e0c |
| tc F2 | 5ff79dee0336e47171f193c6b1c9a69bbe5512cd1a03b0456328e1ae981eff4a |
| ts F2 | 6787d405b592e14915f80735fdf5ee39953a1448522c9f856aeef00e14afc1ff |
| tc F3 | 36d023b3ce2b474907f1fd33c2e1b0906e2048314c7e59abdf4711ff351a04fb (source map 71d1e539…) |
| ts F3 | b67d8e048f3549cf7bda81277b2f1cb6cf9da28220aae7031d1511d2f02bfd24 (source map 7505dc09…, the same as F2's) |
| p01 | 0a6c8ab9b73da3629a004926806dcbab5368a7cab0e9acb13832571563d07207 |
| p02 | 786b33442911b12e522b4a9d4725d16b081a67e975d7c5467a8acaf44a027398 |
| p03 | 2c4326973fd70eb855288c555572351c6945e0fc0b44edb18c0e61ca0a50ac9b |
| p04 | 3d6d954b69a6ba6df7d6234d5a8e70aa7f49e551309b01f7c101a5c812de281c |

The restaged lines also load Build E, n03, n04, n05 and m03 under their existing receipts.

**P3's CPU per-ID result and the intended changes**
- **PASS.** 4656 IDs per build, both series complete, the same collection errors.
  - Build B: 1645 passed, 2176 failed, 820 skipped, 15 errors.
  - Build F3: 1647 passed, 2174 failed, 820 skipped, 15 errors.
- The only differences are BWD-F1's two H1 IDs (`test_bwd_r3_paths.py::test_host_real_planner_and_no_grad[conv1d|conv_transpose1d]`, failed on B and passed on F3). 0 unintended. No test wrote into the worktree.
- L64: 9586 distinct paths outside the worktree.
- Intended changes (`INTENDED_CHANGES_F3.json`, written at 04:37, before any after-run). No production test changed.
  - E1, K1 rows now accepted: f01-rope_apply-short-sin, f01-swap-short-sin, f01-arena-short-sin, f01-rope_apply-negative-pos0, f05-export_row_pair-rope-side.
  - D1, K1 row now refused: f07-where-empty-mixed-data.
  - E1, copy-op rows: export_rope_rows-pos-negative and export_rope_rows-past-cos move to `test_cpu_e1_empty_export_span_not_refused`.

**Mutant table** (`MUTANTS_F3.json` was written at 04:46, before any mutant tree existed; results in `MUTANT_RESULTS_F3.json`)

| Mutant | Killing assertion | Result |
|---|---|---|
| p01_empty_span_refused | e1 empty-span rows (rope_apply, export, swap). The mutant refuses with "position past table end" | KILLED (3/3) |
| p02_trailing_singleton_table_refused | e2 rows (`sin [2,2,1]`, `cos [2,2,1,1]`, export `sin [8,2,1]`). The mutant refuses with "tables must be (T, D)" | KILLED (3/3) |
| p03_duplicate_axis_accepted | e3-sum-duplicate, e3-mean-duplicate, e3-sum-empty-duplicate. The mutant reaches the kernel, or returns | KILLED (3/3) |
| p04_splice_bound_after_work | e4 splice, swap and arena rows. The mutant's refusal is the constructor's bare text, with no pair named | CPU KILLED (3/3). GPU `p2_count_p04…` STAGED |

**Each staged GPU line with its dry-run exit** (`GPU_LINES_F3.json`, `gpu_dry/DRY_RUN.json`)
- Every line below dry-ran once, exactly as written, with `FORGE_GPU_DRY_RUN=1`: **exit 78**. No write-once path was created, and `f3/gpu/` does not exist yet.
- Restaged lines:
  - g2_f01_{B,F3}, g2_f03_{B,F3}, copy_{B,F3}: memcheck, `--min` 240;
  - g2_direct_{B,F3}: memcheck, 300;
  - g3_{B,F3,m03_reduce_gt1}: 300;
  - g4_B_01…12 and g4_F3_01…12: 480;
  - k2_f01_{B,E,F3,n03_sin_span_unchecked}, k2_f04_{B,E,F3,n04_ring_no_modulo}, k2_f07_{B,E,F3,n05_condition_as_data_dtype}, k2_vals_{B,E,F3}: memcheck, 240;
  - k2_count_{B,E,F3}: 240, no memcheck.
- P2 lines:
  - p2_e1_{B,F2,F3}, p2_e2_{B,F2,F3}, p2_e3_{B,F2,F3}: memcheck with `--target-processes all`, 240;
  - p2_count_{B,F2,F3,p04_splice_bound_after_work}: 240, no memcheck.
- Every line uses `--max 540` and an inner `timeout 500s`. Before the lease, each gate checks the module sha256s and the write-once paths (the paths are listed per line in `write_once_paths`).
- Every step script has its `--require-held` check at line 2 (`lease_check_lines`).
- I drove the post-device code on constructed results first: `POST_SELFTEST_F3.txt` 102 checks PASS (the P2 verdicts require the exact row IDs), and `POST_READBACK_DRIVE_F3.txt` 29 rows PASS.
- After the lines run: `scripts/bwdf3_gpu_post.py line <line>`, then `controls` (B against F3, bitwise), then `g4`.

**Deviations**
1. Tool slip (L56(a)): the first G1/K1/P1 run had no `--basetemp` parent directory. I re-ran it; the slip's xml is kept beside the result.
2. L56(b): `test_bwdf3_fixes.py` was repaired once. In the first version the E4 pair rows failed on F2 because of the span check, not the late byte bound, so I gave their rope side D = 0.
3. One entry in `l128/CHECKS.json` reads FAIL: start_of_round → pre_cpu_B. It spans the deliberate install of F3's two modules, so it is not a restore check. Every restore check passes.
4. Readings for you to confirm:
   - D1's "a RoPE table's floating dtype" read as the table and data dtypes must match.
   - axpy_ keeps BWD-F2's R6 (leading size-one axes allowed), because a stricter reading would change accepted CUDA results, which the order forbids.
   - E2 applies to both tables, so `cos [T, D, 1]` is now accepted (Build B refused it).
   - E1 also lets an empty launch pass with a negative pos0, which matches Build B.
5. Residuals, not fixed (out of scope): cumsum_nd's axis check (cumsum is a scan, not a reduction) and the confirm's six successors, handed back:
   - gather/scatter index validation;
   - flip axis values;
   - malformed einsum or permutation;
   - mean's unchecked geometry (the E3 mean fix covers its axes);
   - legacy write_rows/splice_rows dtype and device policy;
   - K2 verdict coverage.
6. Home paths in the suite receipts and logs are rewritten as `$HOME`.

**Every file written**
- Product: `tensor_cuda/src/kernels.cu`, `tensor_cuda/src/ops.cpp`, `tensor_cuda/include/tc/core.h`. Build F3's two modules are installed in the worktree packages.
- Tests:
  - changed: `tensor_cuda/tests/test_bwdf2_fixes.py`, `tensor_cuda/tests/test_bwdf1_copy_ops.py`;
  - new: `tensor_cuda/tests/test_bwdf3_fixes.py`.
- Scripts: `scripts/bwdf3_{build, freeze_pkgs, p1, make_mutants, mutant_results, suite, l128, p3_compare, gpu_common, gpu_lines, gpu_post, post_readback_cpu, p2_cases, p2_drive, step_p2, step_g2_direct, step_g3, step_copy, step_g4, step_k2, boundary, census}.py`.
- Under `artifacts/bwdf1/f3/`:
  - `INSTRUMENTS.log`, `INTENDED_CHANGES_F3.json`, `P1_RESULTS.json`, `MUTANTS_F3.json`, `MUTANT_RESULTS_F3.json`, `P3_CPU_COMPARE.json`;
  - `BOUNDARY_F3.json`, `CENSUS_F3.json`, `L55_F3.json`, `GPU_LINES_F3.json`, `POST_SELFTEST_F3.txt`, `POST_READBACK_DRIVE_F3.txt`;
  - `source_F3.diff`, `tests_F3.diff`, `LEDGER_F3.md`;
  - directories: `before/`, `F3_src/`, `build/`, `g1/`, `g4/` (cpu_B, cpu_F3, L64 audit), `l128/`, `mutants/` (diffs and runs), `gpu_lines/{B,F3}/`, `gpu_dry/`.
- Build directories: `tensor_cuda/build-bwdf1/{F3, p01…p04}` (63 MB each) and `tc_signal/build-bwdf1/F3` (9.5 MB).
- Logs: `logs/bwdf3_p3_cpu_B.log`, `logs/bwdf3_p3_cpu_F3.log`.
- Scratch: `/tmp/bwdf3_tmp`, `/tmp/bwdf3_src_{F3,p01…p04}`.
- Nothing deleted. Items that can go (leave-and-list):
  - the four `/tmp/bwdf3_src_p0*` trees;
  - `tensor_cuda/build-bwdf1/p01…p03` (their mutant runs are done);
  - p04's build must stay until its GPU line has run.

## Prior art
- **NumPy's reduction-axis rule** (`normalize_axis_tuple` / `normalize_axis_index`, "axis N is out of bounds for array of dimension M", "duplicate value in axis"): taken as the rule and the message texts. Mine: one helper shared by every engine entry, and the operand context added to the message.
- **RoPE** (Su et al., 2021, "RoFormer"): the absolute-position table read that the validator bounds. Mine: checking the span only when the launch reads, and the trailing-singleton layout rule (no specific prior art known to me).
- **Byte bound:** the engine's ENG3-F1 constructor bound, reused unchanged. No prior art known to me for running it in the preflight. CERT INT30-C/INT32-C: unverified, lead to check ("CERT INT32-C signed overflow checked addition").
- **where dtype rule:** NumPy promotes x and y to one dtype; this engine has no promotion, so a mismatch is refused. No other prior art known to me.
- **Mutation testing:** DeMillo, Lipton and Sayward (1978). **Call counting:** ENG3's LD_PRELOAD interposition (ltrace-style).
- BWD-F2's scripts were copied and adapted.

Paths:
- Worktree: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_bwdf1`
- Receipts and ledger: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_bwdf1/artifacts/bwdf1/f3/` (`LEDGER_F3.md`)

---

## Lead notes (2026-11-24)

- Lead checks on disk: `g1/G1K1P1_F3.xml` 1,625 tests, 0 failures; `P3_CPU_COMPARE.json` PASS, 0 unintended; P1 57
  rows; installed tc module `36d023b3`; the three tc_signal files unmodified in git; `l128/CHECKS.json`'s one
  differing entry is the deliberate module install, as reported.
- The seat's four readings (deviation 4) are confirmed: the table dtype must match the data dtype (the kernel reads the
  tables in the data's dtype); `axpy_` keeps R6; E2 for both tables (a new acceptance, no accepted result changed);
  an empty launch with a negative pos0 passes, as Build B did.
