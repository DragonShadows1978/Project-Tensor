# BWD-F2 ledger (seat: Claude Opus 5.5, Agent tool)

Order: `orders/BWDF2_REV1_FIXES.md` (b475c61e2). Worktree `tensor_fork_apa_wt_bwdf1`, branch `bwdf1`. Every command ran
with `CUDA_VISIBLE_DEVICES` empty and `TMPDIR` under `/tmp/bwdf2_*`. No GPU use, no compute-sanitizer, no step script or
GPU function called directly, no git writes, nothing deleted. Times are UTC (2026-11-24).

## Receipts, in order

- 02:49 read the order, HOUSE_RULES.md, AGENTS.md, the review (`tensor_fork_apa_wt_bwdf1rev1/artifacts/bwdf1_rev1/
  REPORT.md`, findings F01-F13), BWD-F1's REPORT.md and LEDGER.md, BWD-F1's scripts and tests.
- 02:56 L01 -> `INSTRUMENTS.log`: cmake 3.28.3, nvcc 12.6.85, gcc 13.3.0, pytest 9.0.2, Python 3.12.3 + NumPy 2.4.2,
  `forge-gpu-run --status --min 30 --max 540` rc 75 (busy), `forge-gpu-lease-check --require-held` outside a lease rc 3.
  Disk 232 GB free, MemAvailable 34.6 GB. compute-sanitizer not run (order).
- 02:57 L99: each finding's cited line re-read at fbe04d8f9 (kernels.cu 8656, 1667, 1639, 8257, 8595, 1563, 1659, 5599,
  8353, 5544; test_bwdf1_copy_ops.py:150; test_bwdf1_zero_extent.py:113; functional.py:234): all match the review's
  readings (5599 is the output-shape line, the allocation is 5600, as the review's text says).
- 02:57-03:04 censuses (L82 grep, then by declared size) and edits: kernels.cu (check_meta_rank and its 14 call sites;
  reduce_impl emptiness-first count; where_nd condition dispatch + data dtype check; check_copy_operands R1 + R2; cat_nd
  checked sum, strides before allocation; axpy_ R6; slice_nd / pad_into R2 + R8; gather / scatter / flip R2; write_rows
  R1 device term, R4 host modulo + overflow-free kernel wrap; product_dims emptiness-first checked; normalize_dim
  context; with_index; checked_add3; splice_rows_check checked output extent; rope_tables_check + rope_apply_check
  shared by rope_apply, export_rope_rows_check, export_row_pair, export_row_pairs, swap preflight, arena; arena checked
  tail_start and layer context), core.h (check_meta_rank), ops.cpp (masked_fill rank check before the fill),
  bindings.cpp (getitem axis check), functional.py (einsum preflight of shorter terms).
- 03:04-03:09 Build F2: tree `/tmp/bwdf2_src_F2` (git archive HEAD + this round's 5 engine files), tc configure + build
  (own scope, MemoryMax=12G, OOMPolicy=continue, CPUWeight=50, nice 19, -j4): tc 5ff79dee..., ts 6787d405...; frozen
  packages under `<pkg>/build-bwdf1/F2/pkg` (sha-checked).
- 03:10 CPU smoke on F2's frozen package (no GPU, no test): refusals and R1 acceptances as designed.
- 03:12 `INTENDED_CHANGES_F2.json` written before any test after-run (R12: four BWD-F1 copy rows; no production
  test).
- 03:12-03:17 `tensor_cuda/tests/test_bwdf2_fixes.py` (K1, 46 rows) and `scripts/bwdf2_k1.py`. First K1 run: Build E's
  child died (SIGSEGV) on a row and hid the others -> the harness now runs one child per row (L56(b) repair 1 of the
  test harness). A second repair recorded tuple results instead of reading `.shape` of a tuple (repair 2). Final:
  F2 46/46 hold; every one of the 35 defect rows fails on E; the 11 controls behave the same on E
  (`K1_RESULTS.json`).
- 03:15 BWD-F1's tests updated (R12 rows moved to `R1_ROWS`; R9 labels; R10: equal pairs once, axpy_ size rows apart,
  Z3 attribution); before copies in `before/`.
- 03:17 G1 + K1 on F2: 1568 passed, 0 failed (`g1/G1K1_F2.xml`; 1522 G1 + 46 K1; G1 = BWD-F1's 1561 minus the 39
  duplicated IDs).
- 03:18 `MUTANTS_F2.json` written before any mutant tree existed; trees and diffs (`mutants/*.diff`); builds n01-n06
  03:19-03:38 (two per tool call, each under 300 s); frozen packages.
- 03:40 CPU mutants (`MUTANT_RESULTS_F2.json`): n01 KILLED (3/3), n02 KILLED (2/2), n03 KILLED (3/3), n06 KILLED (2/2),
  each killing test's failure text showing the mutated line's own behaviour. n04, n05 STAGED (GPU).
- 03:41-03:52 K3 CPU: `scripts/bwdf2_suite.py` series cpu_B and cpu_F2 (own scope, MemoryMax=12G): 4656 IDs each;
  B 1645 passed / 2176 failed / 820 skipped / 15 error; F2 1647 / 2174 / 820 / 15; differences: exactly BWD-F1's two
  listed H1 IDs -> `K3_CPU_COMPARE.json` PASS. L64 audit `g4/L64_OPEN_AUDIT_cpu_F2.json` (9586 distinct paths outside
  the worktree). No test wrote into the worktree. The worktree ends on Build F2 (modules and files sha-checked).
- 03:53-03:58 staging: `scripts/bwdf2_gpu_common.py`, restaged step scripts (`bwdf2_step_{g2_direct,g3,copy,g4}.py`),
  audit copies under `f2/gpu_lines/{B,F2}/`, `bwdf2_step_k2.py` (modes f01, f04, f07, vals, count), `bwdf2_gpu_lines.py`
  (53 lines) and `bwdf2_gpu_post.py`. `GPU_LINES_F2.json` written; gates checked on 5 lines (rc 0).
- 03:59 L48: `POST_SELFTEST_F2.txt` 46 checks PASS; `POST_READBACK_DRIVE_F2.txt` 18 readbacks (3 dtypes x 6 shape
  classes incl. empty and one-element) PASS on F2's package.
- 04:00 dry run, each line once, exactly as written, `FORGE_GPU_DRY_RUN=1`: 53 / 53 exit 78; no write-once path
  created; `artifacts/bwdf1/f2/gpu/` not created (`gpu_dry/DRY_RUN.json`).
- 04:01-04:03 `BOUNDARY_F2.json` (Build B measured per row, one child per row), `CENSUS_F2.json`, `L55_F2.json`,
  `source_F2.diff` (5 files, +372/-79), `tests_F2.diff`.
- 04:05 home paths in the suite receipts and logs rewritten as `$HOME`; scope check (below).

## Deviations

1. K5 scope: `scripts/bwdf2_suite.py` (BWD-F1's runner, copied) installs each arm's FILES into the worktree for the
   suite run, including `tc_signal/tc_signal/__init__.py`, `tc_signal/src/bindings.cpp` and `tc_signal/README.md`, which
   are not in this order's writable set. They held Build B's copies during the cpu_B run and were restored to Build F2's
   (= HEAD's) bytes afterwards (sha-checked; git shows them unmodified). The staged g4 lines do the same when run.
2. L56(b): the K1 harness was repaired twice (one child per row; tuple results).
3. Build configuration is BWD-F1's (Release, CUDA arch 89, -j4).
4. The CPU cannot hold a CUDA operand with elements, so F06's ordering witnesses and every value check are GPU rows
   (k2_count, k2_vals); the CPU later-pair rows are labelled as refusal rows with no ordering claim.

## Prior art (also at each code site)

- NumPy's concatenate / stack (shape rules for every operand, empty included), where / putmask (condition truth-tested
  in its own dtype), broadcasting; taken as the rules. Ours: the bytes-read condition for dtype/device (R1).
- RoPE: Su et al., 2021, "RoFormer" (cos/sin tables indexed by absolute position): the read pattern the shared
  validator bounds. Ours: the shared validator and its term order.
- Checked arithmetic: GCC `__builtin_add_overflow` / `__builtin_mul_overflow` (2015); CERT INT30-C / INT32-C
  (unverified -- lead to check: "CERT INT32-C signed overflow checked addition"); P1T-FIX2's zero-first numel_of rule.
- Ring wrap: the standard overflow-free circular-buffer wrap (unverified -- lead to check: "circular buffer index
  overflow modulo").
- axpy_ R6: PyTorch's in-place broadcast rule (an in-place op may not change the destination's shape; unverified
  wording -- lead to check: "output with shape doesn't match the broadcast shape").
- __getitem__: NumPy's IndexError "too many indices for array".
- Rank ceiling: the engine's own TC_MAX_DIMS refusals (P1T-FIX1 F03); no specific prior art known to me for the
  per-consumer placement.
- Mutation testing: DeMillo, Lipton and Sayward (1978). Call counter: ENG3's LD_PRELOAD interposition (ltrace-style).
- Scripts: BWD-F1's build / suite / staged-line / post scripts (this repo), copied and adapted.
