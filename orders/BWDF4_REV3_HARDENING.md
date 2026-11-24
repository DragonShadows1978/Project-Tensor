# BWD-F4: BWD-F1-REV3's four hardening items (one fix each, lead-verified, no re-review)

Follows: BWDF3_REV2_FIXES.md (branch `bwdf1`) and BWDF1_REV3_CONFIRM.md (branch `bwdf1rev3`)
ID reuse: BWD-F1's fix rounds (F2, F3, F4) share the BWD-F1 id

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_bwdf4`
(branch `bwdf4`, from `bwdf1` at `2117296ea` = BWD-F3, at the commit carrying this order). This is a separate worktree
because the lead's GPU lines for BWD-F3 run in `tensor_fork_apa_wt_bwdf1`. Edits, a build and CPU runs are AUTHORIZED,
limited to:
- `tensor_cuda/src/ops.cpp` (`mean`), `tensor_cuda/src/kernels.cu` (`rope_table_layout_ok` and the comment above
  `preflight_out_bound`), and `tensor_cuda/tests/test_bwdf1_copy_ops.py` (one comment);
- the new `tensor_cuda/tests/test_bwdf4_hardening.py`, new `scripts/bwdf4_*.py`, everything under `artifacts/bwdf1/f4/`;
- the build directory `tensor_cuda/build-bwdf4/` and the module it installs into this worktree's package;
- `logs/bwdf4_*`, and scratch under `/tmp/bwdf4_*`. Point `TMPDIR` there for every command.

**Read-only:** every other file here; every other worktree (BWD-F3's receipts are in `tensor_fork_apa_wt_bwdf1`, the
confirm's in `tensor_fork_apa_wt_bwdf1rev3/artifacts/bwdf1_rev3/`); the production repository.
**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively; search inside the roots named here.
**GPU:** none of any kind. This round is CPU only; it stages no GPU line.

## Why

The read-only confirm of BWD-F3 (BWD-F1-REV3, Codex; `findings.json` sha256 `107ea408...`) closed all 8 earlier
findings with 0 blockers. It left 4 hardening items, which the house rule settles with one fix each, verified by the
lead, with no re-review:
- **F01 (MAJOR):** `ops::mean` (`ops.cpp:1064`) multiplies the selected extents in signed int64 before the checked
  reducer runs. An empty tensor reshaped to `[2^32, 2^32, 0, 0]` makes `mean([0, 1, 2])` compute 2^64.
- **F02 (MINOR):** `rope_table_layout_ok` (`kernels.cu:8797`) accepts a table of rank above 8, such as
  `[2,2,1,1,1,1,1,1,1]`. Ruling D1 sets the engine's rank ceiling at TC_MAX_DIMS (8).
- **F03 (NIT):** the comment above `preflight_out_bound` (`kernels.cu:8596-8600`) says only splice can fire the bound.
  A reshaped empty view (`[0, 2^62, 2]`, uint8) passed to `export_rows` fires it too.
- **F04 (NIT):** a comment in `test_bwdf1_copy_ops.py` (about line 270) calls the accepted `pos0 < 0`, `len = 0` export
  "Build B behaviour". Build B's `export_rope_rows` refused `pos0 < 0` unconditionally; only `rope_apply` lacked the
  check.

## Rulings (binding)

- **H1 (F01).** `mean` computes its count without signed overflow. Decide emptiness first (any selected extent 0 gives
  count 0), then multiply with a checked product, and refuse an overflowing product before any work. The result for
  every input that does not overflow is unchanged.
- **H2 (F02).** `rope_table_layout_ok` refuses a table whose rank is above TC_MAX_DIMS, as D1 requires.
- **H3 (F03, F04).** Correct the two comments to say what the code does. Comments only.

## Gates

- **G1, CPU only** (`CUDA_VISIBLE_DEVICES` empty), on Build F4. These rows each fail on Build F3 (`36d023b3`) and pass
  on Build F4, with 0 CUDA calls on ENG3's counter where they refuse:
  - H1: the reviewer's `[2^32, 2^32, 0, 0]` case, and one overflowing non-empty-axis case;
  - H2: a rank-9 table.
  Controls: a valid `mean` over each rank 1 to 4 gives Build F3's value; a rank-8 `[T, D, 1, ...]` table is still
  accepted.
- **G2, no regression.** BWD-F1's, BWD-F2's and BWD-F3's own test files (`test_bwdf1_zero_extent.py`,
  `test_bwdf1_copy_ops.py`, `test_bwdf2_fixes.py`, `test_bwdf3_fixes.py`, `tc_signal/tests/test_bwdf1_hardening.py`)
  pass on Build F4 on CPU, as on Build F3.
- **G3, mutants.** `h01_mean_unchecked_product` (H1) and `h02_table_rank_unbounded` (H2), each named in
  `artifacts/bwdf1/f4/MUTANTS_F4.json` before it runs, each failing its G1 row.
- **G4, scope.** Writes only in the writable set. No GPU use. No git writes, nothing deleted.

**Registered acceptance:** G1 to G4 pass. The lead verifies the diff, and the change joins `bwdf1` by merge after the
BWD-F3 GPU lines finish.

## Builds

- **Build F3:** BWD-F3 as receipted (`tc` `36d023b3...`, in `tensor_fork_apa_wt_bwdf1/tensor_cuda/build-bwdf1/F3/`).
- **Build F4:** this round's tree (`tensor_cuda` only; `tc_signal` is unchanged). Build it the way BWD-F3 did
  (`scripts/bwdf3_build.py` in `tensor_fork_apa_wt_bwdf1`; copy what you reuse into `scripts/bwdf4_*.py`).

## Mechanism & scope

- **Measures:** whether the four hardening items are fixed with results unchanged.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_bwdf1rev3/artifacts/bwdf1_rev3/REPORT.md`
  (the confirm).
- **Exercises:** `mean`'s count, the table layout predicate, two comments; Builds F3 and F4 plus two mutant builds; CPU
  tests with the call counter.
- **Does not cover:** the confirm's six successors; GPU lines (none needed: every row is CPU-reachable); performance;
  promotion.
- **Required evidence:** the diff against `2117296ea`; Build F4's receipt with module sha256; G1 and G2 results;
  `MUTANTS_F4.json` and `MUTANT_RESULTS_F4.json`.
- **Operand census:** `mean`'s axis and count path; every caller of `rope_table_layout_ok`.
- **Baseline isolation:** Build F3 for unchanged results and for "defect present".
- **Forbidden configuration:** any change to a result that does not overflow; edits outside the writable set; GPU use;
  git writes; deleting anything yourself.

## Machine rules

- **CPU:** at most one CPU-busy process of yours at a time, at `nice -n 19`. The build and any job over 1 GB run in their
  own scope: `systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue -p CPUWeight=50 ...`, started only with
  `MemAvailable` at or above 12 GB; builds at most `-j4` (another seat builds on this machine).
- **Time:** every command under 30 minutes, the build under 60. The round's ceiling is 2 hours of wall time.
- **Disk:** stop if `/mnt/ForgeRealm` has under 40 GB free.
- **GPU:** none. Never call a step script or GPU function, never run `compute-sanitizer`, never remove or override
  `CUDA_VISIBLE_DEVICES` in a child.

## Registration checklist (order-lint)

- **L01:** cmake, nvcc, gcc, pytest and python3 with NumPy, each run with `--version` first and recorded.
- **L02, L03, L07, L14, L15, L18, L32, L37, L45, L48, L53, L79, L111, L114, L117, L125, L128:** n/a (no GPU line, no
  share, nothing visual, no gain gate, no fitted threshold, no owner requirement behind an option, no bounded buffer, no
  knob, no staged line, nothing for another project's run, no storage design, no job over 1 GB beyond the build in its
  scope, no gradient change, no step script, no arm-swapping runner: G2 runs the test files directly on Build F4).
- **L04:** no fallback; H1 and H2 refuse.
- **L08:** the build under `tensor_cuda/build-bwdf4/`, receipts under `artifacts/bwdf1/f4/`, scratch under `/tmp/bwdf4_*`.
- **L09, L10, L19:** G1's controls compare `mean`'s value with Build F3's per element on same-width unsigned views, each
  side checked nonzero and non-constant first.
- **L12:** inherited gates: the three rounds' test files (G2).
- **L16:** the cheapest passing work would special-case the reviewer's one shape. Excluded by the second, non-empty-axis
  overflow row and `h01`.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`.
- **L22, L25:** cases defined by this order; each build in its own process.
- **L34:** a Claude seat (Sonnet 5.5, high effort, Agent tool). Hardening is lead-verified, no re-review.
- **L35:** if a clause cannot hold as written, report RED for that clause and finish the rest.
- **L38:** no existing test assertion changes; F04's edit is a comment.
- **L40:** pytest with `-p no:cacheprovider`, `--basetemp` and `TMPDIR` under `/tmp/bwdf4_*`.
- **L41, L47:** each mutant names its failing G1 row in `MUTANTS_F4.json` before it runs; show the killing test executes
  the mutated line.
- **L55:** for H1, show the shape, the count Build F3 computes (the overflow) and Build F4's refusal point.
- **L56:** (a) tool slips go under Deviations: re-run and continue. (b) At most 3 repairs of each of your scripts.
- **L58:** G1 and G2 run with `CUDA_VISIBLE_DEVICES` empty.
- **L60:** H1's and H2's refusals are recorded in `artifacts/bwdf1/f4/BOUNDARY_F4.json` with Build F3's behaviour.
- **L64:** reads are this worktree, BWD-F3's build tooling and module, the confirm's receipts, and ENG3's counter
  source.
- **L65:** leave-and-list form for any file you would remove.
- **L67:** one tensor_cuda build (about 63 MB) and two mutant builds. Free space well above the 40 GB stop.
- **L69:** unchanged results -> Build F3; refusals -> this order.
- **L74:** plain wording.
- **L76:** the lead writes nothing in the writable set while the seat runs.
- **L77:** both mutants can fail and both are run.
- **L81:** the confirm's cases are its readings (source records: file:line and the shapes it names; nothing run); G1
  measures each on Build F3 first.
- **L82:** `grep -n "rope_table_layout_ok\|Tensor mean" tensor_cuda/src/*.cu tensor_cuda/src/*.cpp` lists the callers.
- **L89:** n/a. No guard keyed on a name.
- **L90:** roles: product (two functions), test (the round's own file plus one comment), harness. Count per role.
- **L99:** the lead read `ops.cpp:1058-1066`, `kernels.cu:8795-8800` and `kernels.cu:8596-8600` at `2117296ea` before
  sealing.
- **L102:** no generated file or registry depends on the writable files.
- **L103:** G1 and G2 load only the module built from this worktree; record its sha256 and source map.
- **L104:** no ratio bar.
- **L109:** helpers run as `python3 -B -P`; none shares a standard-library module name.
- **L118:** consumers of the changed checks: `mean`'s callers and the RoPE paths, all covered by G2's test files.
- **L119:** hardening: one fix each, lead-verified, no re-review.
- **L127:** a Claude seat: your final message is the report, and the lead files it as `artifacts/bwdf1/f4/REPORT_F4.md`.
  Your JSON receipts and logs are written as usual.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes (read-only git is allowed). NO subagents. No background waits; each command under an explicit timeout.
- No deletes, in the leave-and-list form. Never kill a process you did not start.
- **Prior Art Directive:** annotate each code site and the report. Known: GCC's `__builtin_mul_overflow`; NumPy's
  rule that an empty reduction axis gives count 0. Write "no prior art known to me" where none applies.

## Done

Your final message contains, verbatim:
- G1 to G4, each PASS or FAIL;
- a 4-row closure table (finding, fix file:line, test, result);
- the diff summary;
- every module sha256;
- the mutant table;
- the deviations;
- every file written.
