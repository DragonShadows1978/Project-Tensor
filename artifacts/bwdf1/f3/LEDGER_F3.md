# BWD-F3 ledger (seat: Claude Opus 5.5, Agent tool)

Order: `orders/BWDF3_REV2_FIXES.md` (400a8389f). Worktree `tensor_fork_apa_wt_bwdf1`, branch `bwdf1`. Every command ran
with `CUDA_VISIBLE_DEVICES` empty and `TMPDIR` under `/tmp/bwdf3_*`. No GPU use, no compute-sanitizer, no step script
or GPU function called directly, no git writes, nothing deleted. Times are UTC (2026-11-24).

## Receipts, in order

- 04:24 read the order, HOUSE_RULES.md, AGENTS.md, the confirm (`tensor_fork_apa_wt_bwdf1rev2/artifacts/bwdf1_rev2/
  findings.json`, sha256 9a687dad2a6d7e8a4928807872e9c4f92e767a8556c6a1dbdaf07154814bd5df, as the order states),
  BWD-F2's REPORT_F2.md and LEDGER_F2.md, BWD-F2's scripts and tests.
- 04:27 L01 -> `INSTRUMENTS.log`: cmake 3.28.3, nvcc 12.6.85, gcc 13.3.0, pytest 9.0.2, Python 3.12.3 + NumPy 2.4.2,
  `forge-gpu-run --status --min 30 --max 540` rc 75 (busy), `forge-gpu-lease-check --require-held` outside a lease
  rc 3. Disk 227 GB free, MemAvailable 34.5 GB. compute-sanitizer not run (order).
- 04:28 L99: HEAD 400a8389f differs from e4ad8381a only by the order file. Each finding's cited line re-read at
  e4ad8381a: kernels.cu 8708, 8706, 5668, 1236, 1615, 1410, 1416, 1434, 1336, 535, 8603, 8615, 8898, 8704, 8693, 8711,
  5648; bindings.cpp 1982; test_bwdf2_fixes.py 106, 179; BWDF2_REV1_FIXES.md:38; BOUNDARY_F2.json 163, 526. All match
  the confirm's readings. None differs.
- 04:28 L82 census grep, then the censuses by behaviour (`CENSUS_F3.json`): RoPE table reads (2 kernels), ReduceSpec
  writers (reduce_impl only) and every reduction entry, pair preflights.
- 04:29-04:31 edits: core.h (declarations of `ndarray_checked_nbytes` and `normalize_reduce_axes`); kernels.cu
  (constructor bound factored into `ndarray_checked_nbytes`; `normalize_reduce_axes` used by reduce_impl and
  reduce_arg; where_nd D1; `rope_table_layout_ok` E2 and the span under `reads` E1; `preflight_out_bound` in
  export_rows_check, splice_rows_check, evict_rows_check, export_rope_rows_check, rope_apply_check E4); ops.cpp
  (ops::mean validates its axes first, E3).
- 04:31-04:35 Build F3: tree `/tmp/bwdf3_src_F3` (git archive HEAD + this round's engine files), each step in its own
  scope (MemoryMax=12G, OOMPolicy=continue, CPUWeight=50, nice 19, -j4): tc 36d023b3..., ts b67d8e04... (ts sources
  identical to F2's, source map 7505dc09...; the module bytes differ because the build path differs). Frozen packages
  under `<pkg>/build-bwdf1/F3/pkg` (sha-checked).
- 04:36 CPU smoke on F3, F2 and B packages (no test). F2 and B aborted on `softmax(-3)` (free(): invalid pointer,
  exit 134), the F06 host corruption.
- 04:37 `INTENDED_CHANGES_F3.json` written, before any test after-run: E1 moves 5 K1 rows and 2 G1 copy-op rows to
  "not refused"; D1 moves 1 K1 row to "refused"; no production test predicted.
- 04:38-04:43 tests: K1 and G1 rows edited per the list; `test_bwdf3_fixes.py` (57 P1 rows) and `scripts/bwdf3_p1.py`.
  First P1 run: 57/57 by the bar, but the E4 pair rows failed on F2 for the E1 reason (rope tables with 0 rows), not
  the late byte bound. The rope side was changed to D = 0 (tables cover every span), so Build F2's failure is the late
  bound alone (L56(b): repair 1 of the P1 table). Second run 04:43: PASS, 57 rows (33 fix rows fail on F2 and hold on
  F3; 12 controls hold on both; 12 D1 rows hold on F3), F3's counter control sees 4 calls (`P1_RESULTS.json`).
- 04:44 G1 + K1 + P1 on F3: first attempt was a tool slip (the `--basetemp` parent did not exist: 1621 setup errors;
  kept as `g1/G1K1P1_F3_slip1_no_basetemp_parent.xml`). Re-run 04:45: 1625 passed, 0 failed (`g1/G1K1P1_F3.xml`:
  1432 zero-extent, 67 copy-op, 23 tc_signal hardening, 46 K1, 57 P1).
- 04:46 `MUTANTS_F3.json` written before any mutant tree existed; trees and diffs (`mutants/*.diff`, 1 line each);
  builds p01-p04 04:46-04:58; frozen.
- 04:59-05:00 CPU mutants (`MUTANT_RESULTS_F3.json`): p01 KILLED (3/3), p02 KILLED (3/3), p03 KILLED (3/3),
  p04 KILLED (3/3 CPU; its GPU line staged). Each killing test's failure text is the mutated line's own behaviour.
- 05:00 L128 snapshots (`l128/`): `start_of_round`, then Build F3 installed into the worktree (`F3_src` snapshot +
  install), `pre_cpu_B`. The start_of_round -> pre_cpu_B comparison shows 2 differences, the two modules (F2 -> F3);
  that is the install, not a restore check.
- 05:01-05:06 P3 CPU, series cpu_B (own scope, MemoryMax=12G): 4656 IDs, 1645 passed / 2176 failed / 820 skipped /
  15 error. L128 check pre_cpu_B -> post_cpu_B: PASS (50 paths, 0 differences).
- 05:07-05:12 P3 CPU, series cpu_F3 (L64 audit on): 4656 IDs, 1647 / 2174 / 820 / 15. L128 check post_cpu_B ->
  post_cpu_F3: PASS. `P3_CPU_COMPARE.json` PASS: the only differences are BWD-F1's two listed H1 IDs; 0 unintended;
  same collection errors. No test wrote into the worktree. L64: 9586 distinct paths outside the worktree.
- 05:13-05:18 staging: `bwdf3_gpu_common.py`, the restaged steps (`bwdf3_step_{g2_direct,g3,copy,g4,k2}.py`), audit
  copies under `f3/gpu_lines/{B,F3}/`, `bwdf3_step_p2.py` + `bwdf3_p2_cases.py` (P2 modes e1, e2, e3, count; one
  child per row), `bwdf3_gpu_lines.py` (66 lines), `bwdf3_gpu_post.py`. `GPU_LINES_F3.json` written.
- 05:16 P2 CPU geometry drive on F3 (`gpu_dry/P2_GEOMETRY_DRIVE_F3.json`; not a gate).
- 05:19 L48: `POST_SELFTEST_F3.txt` 102 checks PASS (every P2 mode built from the case table, with exact row IDs,
  plus BWD-F2's restaged paths); `POST_READBACK_DRIVE_F3.txt` 29 rows PASS.
- 05:20 dry run, each line once, exactly as written, `FORGE_GPU_DRY_RUN=1`: 66/66 exit 78; no write-once path
  created; `artifacts/bwdf1/f3/gpu/` not created (`gpu_dry/DRY_RUN.json`).
- 05:21-05:23 `BOUNDARY_F3.json`, `CENSUS_F3.json`, `L55_F3.json`, `source_F3.diff` (3 files, +138/-15),
  `tests_F3.diff`; home paths in the suite receipts and logs rewritten as `$HOME`; L128 `end_of_round` vs
  post_cpu_F3: PASS. The worktree ends on Build F3 (sources = F3's source map, both modules F3's).

## Deviations

1. L56(a) tool slip: the first G1/K1/P1 run had no `--basetemp` parent directory; re-run, the slip's xml kept.
2. L56(b): `test_bwdf3_fixes.py` repaired once (the E4 pair rows' rope side, so Build F2 fails for the E4 reason).
3. The start_of_round -> pre_cpu_B L128 comparison is labelled FAIL in `l128/CHECKS.json`: it compares across the
   deliberate install of Build F3's modules (2 differences, the modules). Every restore check (B -> F3 runs, and the
   end of round) is PASS.
4. Readings recorded for the lead: D1's "a RoPE table's floating dtype" read as the table / data dtype equality
   (always checked; the launch-dtype term stays under `reads`, BWD-F2's R1); D1's axpy_ rule kept BWD-F2's R6
   (leading size-one axes allowed), since a stricter reading would change accepted CUDA results the order forbids
   changing; E2 applied to both tables (cos [T, D, 1] is now accepted, Build B refused it); E1 applied to every span
   term, so a negative pos0 on an empty launch passes (Build B's result).
5. Residuals, not fixed (out of scope): cumsum_nd's axis (a scan, not a reduction); the confirm's six successors.

## Prior art (also at each code site)

- NumPy's reduction-axis rule: `normalize_axis_tuple` / `normalize_axis_index`, AxisError "axis 8 is out of bounds
  for array of dimension 1", ValueError "duplicate value in axis". Taken as the rule and the texts. Ours: one helper
  shared by every engine entry, the operand context, and pybind11's mapping (std::out_of_range -> IndexError,
  std::invalid_argument -> ValueError; NumPy's AxisError subclasses both IndexError and ValueError).
- RoPE: Su et al., 2021, "RoFormer: Enhanced Transformer with Rotary Position Embedding": absolute-position cos/sin
  tables, the read pattern the validator bounds. Ours: the span bound only when the launch reads, and the
  trailing-singleton layout rule (no specific prior art known to me; NumPy's squeeze of size-one axes keeps the same
  contiguous buffer).
- Byte bound: the engine's own ENG3-F1 constructor bound, moved into `ndarray_checked_nbytes` unchanged and run in the
  preflights. The check-then-act split: no specific prior art known to me. Subtraction-based bounds: CERT
  INT30-C/INT32-C (unverified -- lead to check: "CERT INT32-C signed overflow checked addition").
- where's dtype rule (D1): NumPy's where promotes x and y to one dtype; the engine has no implicit promotion, so the
  mismatch is refused. No other prior art known to me.
- Mutation testing: DeMillo, Lipton and Sayward (1978). Call counter: ENG3's LD_PRELOAD interposition (ltrace-style).
- Scripts: BWD-F2's build, freeze, suite, staged-line, step, post and mutant scripts (this repo), copied and adapted.
