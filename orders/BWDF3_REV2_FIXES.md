# BWD-F3: fix BWD-F1-REV2's remaining defects (RoPE spans and table layout, reduction axes, a late byte bound), rule the domain conflicts, and restage (implementing round)

Follows: BWDF2_REV1_FIXES.md (this branch)
ID reuse: BWD-F1's fix rounds (F2, F3) share the BWD-F1 id

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_bwdf1`
(branch `bwdf1`, at the commit carrying this order; BWD-F2 is `e4ad8381a`). Edits, builds and CPU runs are AUTHORIZED,
limited to:
- `tensor_cuda/src/kernels.cu`, `ops.cpp`, `tensor_cuda/include/tc/*.h`: only the fixes ruled below;
- BWD-F1 and BWD-F2's tests (`tensor_cuda/tests/test_bwdf1_*.py`, `test_bwdf2_fixes.py`), new
  `tensor_cuda/tests/test_bwdf3_*.py`;
- `scripts/bwdf2_*.py`, new `scripts/bwdf3_*.py`, everything under `artifacts/bwdf1/f3/`;
- the build directories `tensor_cuda/build-bwdf1/` and `tc_signal/build-bwdf1/` (new arms `F3*`, `p0*`) and the modules
  they install into this worktree's packages;
- `logs/bwdf3_*`, and scratch under `/tmp/bwdf3_*`. Point `TMPDIR` there for every command.
- **Temporarily written and restored (L128):** BWD-F2's arm-swapping suite runner installs each build's copies of the
  engine sources (`tensor_cuda/src/**`, `tensor_cuda/include/**`, `tensor_cuda/tensor_cuda/*.py`,
  `tc_signal/tc_signal/__init__.py`, `tc_signal/src/bindings.cpp`, `tc_signal/README.md`) into this worktree and
  restores them. Judge scope by sha256 after each restore and record the check.

**Read-only:** every other file here (BWD-F1 and BWD-F2's receipts included); the reviews
(`tensor_fork_apa_wt_bwdf1rev1/artifacts/bwdf1_rev1/`, `tensor_fork_apa_wt_bwdf1rev2/artifacts/bwdf1_rev2/`, commit
`4dfd24f9b`); every other worktree; the production repository; the consumer roots (grep only).
**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively; search inside the roots named here.
**GPU:** you run no GPU work of any kind. You stage GPU lines and dry-run them; the lead runs them.

## Why

BWD-F2 fixed BWD-F1-REV1's 13 findings (`e4ad8381a`). Its read-only confirm (BWD-F1-REV2, Codex;
`artifacts/bwdf1_rev2/findings.json` sha256 `9a687dad...`) found 8 more: 6 BLOCKER, 1 MAJOR, 1 MINOR. The provider's
content filter stopped it after `findings.json` was complete, so it has no REPORT.md. Four of the blockers are refusals
of empty or ill-formed requests that Build B answered with a no-op, and the reviewer asked the lead to resolve them.
Two are real defects in F2's own checks: an empty RoPE span refused (F01), and a contiguous `[T, D, 1]` sine table
refused although Build B computed it correctly (F02). One is an inherited memory defect in the reduction loops F2
rewrote (F06). The MAJOR is a byte bound checked too late (F07).

## Lead rulings (binding)

- **D1, the engine's input domain (closes F03, F04, F05, F08 by ruling).** The following are refused before any work,
  even where Build B returned a value for an empty no-op:
  - a rank above TC_MAX_DIMS (8). The engine documents and enforces that limit elsewhere (P1T-FIX1 F03);
  - operand dtypes that differ where the op's contract requires them to match (`where`'s x and y; a RoPE table's
    floating dtype);
  - an in-place destination whose shape differs from the broadcast result (`axpy_`).
  Record each such row in `BOUNDARY_F3.json` as a "domain refusal", with Build B's answer. These are not regressions.
  Copy-op members keep BWD-F2's R1, where dtype and device are checked only for members whose bytes are copied,
  because `concatenate`/`stack` accept mixed empty members.
- **E1, RoPE spans (F01).** The validator bounds the actual read span `[pos0, pos0 + L)` in both tables. An empty span
  (L = 0 or no rows) reads nothing and passes, whatever the tables' lengths. This holds for `rope_apply`, the exports
  and the swap and arena paths.
- **E2, table layout (F02).** A RoPE table is accepted when its storage is a contiguous `[T, D]` layout: rank 2, or rank
  above 2 with every extra extent 1. Other ranks are refused.
- **E3, reduction axes (F06).** Every reduction entry, including the `mean` wrapper and its axis handling in `ops.cpp`,
  normalizes and range-checks each axis and refuses duplicates before indexing any shape or writing any spec slot. Follow
  NumPy's rule ("duplicate value in axis").
- **E4, byte bound (F07).** `splice_rows_check`, and every pair preflight, check the output's byte bound
  (`ndarray_checked_nbytes` or the equivalent) before any pair's work.
- **E5, restage.** BWD-F2's 53 lines were never run (the lead stopped the runner). Restage them against Build F3 under
  `artifacts/bwdf1/f3/gpu/`, plus a GPU row for each E1-E4 case only the GPU can show.
- **E6.** Existing tests change only to follow D1 and E1-E4, each listed in `artifacts/bwdf1/f3/INTENDED_CHANGES_F3.json`
  before any after-run. No knob. The six successors in the confirm are handed back (listed in the report).

## Gates

- **P1, CPU only** (`CUDA_VISIBLE_DEVICES` empty), on Build F3. A row for each E1-E4 case, failing on Build F2
  (`5ff79dee`) and passing on Build F3. A row for each D1 domain refusal, recorded as such. Refusals show 0 CUDA calls.
  BWD-F2's K1 and BWD-F1's G1 still pass, except rows D1 and E1-E4 change, each listed.
- **P2, GPU rows and lines, staged.** E1-E4 cases only the GPU can show, and E5's restaged lines.
- **P3, no regression.** Production's test suite per test ID, Build B (`70cd1752` and `857432c5`) against Build F3, on CPU
  and as staged GPU lines. Identical except BWD-F1's two listed H1 IDs and E6's list.
- **P4, mutants.** Each fails its named assertion in `artifacts/bwdf1/f3/MUTANTS_F3.json`, written before any mutant
  runs:
  - `p01_empty_span_refused` (E1);
  - `p02_trailing_singleton_table_refused` (E2);
  - `p03_duplicate_axis_accepted` (E3);
  - `p04_splice_bound_after_work` (E4).
- **P5, scope.** Writes only in the writable set, with L128's temporary writes restored and sha-checked. No GPU use. No
  git writes, nothing deleted.

**Registered acceptance:** the seat's part passes when P1, P3's CPU half, the CPU mutants and P5 pass, and P2, P3's GPU
half and the GPU mutants are staged and dry-run clean. The round is ACCEPTED when the lead's GPU lines pass. Then a
read-only Codex confirm scoped to the F3 diff.

## Builds

- **Build B:** BWD-F1's base as receipted (`tc` `70cd1752...`, `ts` `857432c5...`).
- **Build F2:** BWD-F2 as receipted (`tc` `5ff79dee...`, `ts` `6787d405...`).
- **Build F3:** this round's tree, both packages.

## Mechanism & scope

- **Measures:** whether F01, F02, F06 and F07 are fixed, D1's domain refusals are recorded, and results on accepted CUDA
  inputs are unchanged.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_bwdf1rev2/artifacts/bwdf1_rev2/findings.json`
  (the confirm, which has no REPORT.md) and `artifacts/bwdf1/f2/REPORT_F2.md` (in this worktree).
- **Exercises:** the ruled fixes; Builds B, F2 and F3 plus mutant builds; CPU tests with the call counter; staged GPU
  lines.
- **Does not cover:** the confirm's successors (gather/scatter indices, flip axes, malformed einsum, legacy write_rows and
  splice policy, the K2 verdict); ENG5's host-operand guards (its own lane); multi-GPU behaviour; performance;
  promotion.
- **Required evidence:** a per-finding closure table (8 rows: finding, fix or ruling, test, result); `BOUNDARY_F3.json`
  with D1's domain rows; the diff against `e4ad8381a`; build receipts with module sha256s; P1 results;
  `MUTANTS_F3.json` and `MUTANT_RESULTS_F3.json`; `INTENDED_CHANGES_F3.json`; P3's CPU per-ID result; staged GPU lines
  with dry-run exits; the L128 sha checks.
- **Operand census:** every RoPE path's read span; every reduction entry and its axis handling; every pair preflight's
  byte bound; every D1 row.
- **Baseline isolation:** Build B for unchanged behaviour and P3; Build F2 for "defect present".
- **Forbidden configuration:**
  - a refusal of an empty RoPE span;
  - an axis written to a spec before it is range-checked;
  - any change to an accepted CUDA result beyond the ruled fixes;
  - edits outside the writable set; git writes; deleting anything yourself.

## Machine rules

- **CPU:** at most one CPU-busy process of yours at a time, at `nice -n 19`. Builds and any job over 1 GB run in their
  own scope: `systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue -p CPUWeight=50 ...`, started only with
  `MemAvailable` at or above 12 GB; builds at most `-j4` (another seat builds on this machine).
- **Time:** every command under 30 minutes, builds under 60. The round's ceiling is 5 hours of wall time.
- **Disk:** stop if `/mnt/ForgeRealm` has under 40 GB free.
- **GPU:** none of any kind for you. Never call a step script or GPU function directly, never run `compute-sanitizer`,
  never remove or override `CUDA_VISIBLE_DEVICES` in a child. A smoke of a staged line is its `FORGE_GPU_DRY_RUN=1` dry
  run only. Stage lines as BWD-F2 did (`scripts/bwdf2_gpu_lines.py` form, `--min` ≤ 480, `--max 540`, inner `timeout` ≤
  500 s). Every step script starts with
  `if subprocess.run(["forge-gpu-lease-check", "--require-held"]).returncode != 0: raise SystemExit("not inside a held forge-gpu-run lease")`
  before it imports the engine or reads `CUDA_VISIBLE_DEVICES` (order-lint L125, GW8).

## Registration checklist (order-lint)

- **L01:** cmake, nvcc, gcc, pytest, python3 with NumPy, `forge-gpu-run` (`--status --min 30 --max 540`), each run first
  and recorded.
- **L02:** GPU lines carry an explicit `--min`; each lane's smoke is its dry run.
- **L03, L07, L14, L15, L18, L32, L37, L53, L111, L117:** n/a (no share, nothing visual, no gain gate, no fitted
  threshold, no owner requirement behind an option, no bounded buffer, no knob, nothing for another project's run, no
  storage design, no gradient change).
- **L04:** no fallback; E3 and D1 refuse, they do not repair.
- **L08:** builds under the two build directories (`F3*`, `p0*`), receipts under `artifacts/bwdf1/f3/`, scratch under
  `/tmp/bwdf3_*`.
- **L09:** P3 compares outcomes per test ID; value comparisons are per element on same-width unsigned views or against
  NumPy.
- **L10:** results on accepted CUDA inputs held to identity with Build B beyond the ruled fixes.
- **L12:** inherited gates: production's suite per ID, BWD-F1's G1 and BWD-F2's K1; intended changes listed first (E6).
- **L16:** the cheapest passing work would patch the named cases only. Excluded by E1's read-span rule applied to every
  RoPE path, E3 applied to every reduction entry including `mean`, E4 to every pair preflight, and `p01`-`p04`.
- **L19:** every bitwise comparison first checks each side is nonzero and non-constant.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`.
- **L22, L25:** cases and seeds defined by this order and `MUTANTS_F3.json`; each build in its own process.
- **L34:** a Claude seat (Opus 5.5, high effort, Agent tool); the confirm that follows is Codex.
- **L35:** registrations may fix cases and seeds, never narrow a gate. If a clause cannot hold as written, report RED
  for that clause and finish the rest.
- **L38:** existing tests change only as E6 allows, each listed.
- **L40:** pytest with `-p no:cacheprovider`, `--basetemp` and `TMPDIR` under `/tmp/bwdf3_*`; restore the tracked
  `artifacts/wo*.ppm` files `test_terrain_render.py` writes, or exclude that file and list it.
- **L41:** each mutant names its failing assertion before it runs; show each killing test executes the mutated line.
- **L45:** each staged line records `forge-gpu-run`'s exit, the inner exit and the log sha256; 124 is never a pass.
- **L47:** mutant expectations come from this order.
- **L48:** staged lines run as written from the worktree root through the real `forge-gpu-run`; drive each line's
  post-device code on constructed results of each shape class it will see, empty and one-element included, and list
  those steps; each GPU row runs in its own child, so one crash cannot hide the rest.
- **L55:** for one witness per E1-E4 and per D1 class, show the inputs, Build F2's path and Build F3's check position.
- **L56:** (a) tool slips go under Deviations: re-run and continue. (b) At most 3 repairs of each of your scripts.
- **L58:** P1 runs with `CUDA_VISIBLE_DEVICES` empty; no child of yours removes or overrides it.
- **L60:** every new or changed refusal is in `BOUNDARY_F3.json` with its terms, the rows that trigger each, and Build B's
  behaviour; D1 rows are labelled "domain refusal" with Build B's answer.
- **L64:** reads are this worktree, the two reviews' receipts, ENG3's counter source and the consumer roots (grep only);
  re-record the inherited suite's reads for Build F3's CPU run.
- **L65:** leave-and-list form for any file you would remove.
- **L67:** BWD-F2's arms were about 63 MB each. Free space at sealing: 73 GB on `/mnt/ForgeRealm` (`df -h`, 2026-11-23
  20:36). Stop below 40 GB.
- **L69:** one reference per class: restored behaviour -> Build B; review cases -> the confirm's stated behaviour,
  re-measured on Build F2; accepted CUDA results -> Build B per ID; domain refusals -> D1.
- **L74:** plain wording.
- **L76:** the lead writes nothing in the writable set while the seat runs.
- **L77:** every mutant can fail and each is run (GPU ones staged).
- **L79:** `FORGE_GPU_DRY_RUN=1` returns 78 without a device check; once per staged line.
- **L81:** the confirm's cases are its readings (source records: file:line and the shapes it names; nothing run). P1 and
  P2 measure each on Build F2 before claiming it closed.
- **L82:** `grep -n "rope_tables_check\|rope_apply_check\|reduce_impl\|ReduceSpec\|splice_rows_check\|product_dims" tensor_cuda/src/*.cu tensor_cuda/src/*.cpp`
  starts the censuses.
- **L89:** evasion: a reduction entry or RoPE path under another name. Excluded by censusing reductions by their spec
  write and RoPE paths by their table reads, not by name; controls `p01`, `p03`.
- **L90:** roles: product (engine), test, consumer (report-only), harness. Count per role in the census.
- **L99:** the rulings rest on the confirm's findings (`findings.json` `9a687dad...`, 8 rows with file:line evidence),
  which the lead read before sealing (F04 and F06 in full). Seat check that runs first: re-read each finding's evidence
  at `e4ad8381a` and record any that do not match.
- **L102:** no generated file or registry depends on the writable files.
- **L103:** each staged GPU line refuses, before the lease, a module whose sha256 differs from its receipt; receipts
  carry the full-tree source map.
- **L104:** no ratio bar; every gate is exact.
- **L109:** helpers run as `python3 -B -P`; none shares a standard-library module name.
- **L114:** builds in their own scope (`MemoryMax=12G`); staged lines set no RLIMIT_AS; peak host bytes listed.
- **L118:** consumers of the changed checks: P1's rows, BWD-F1's G1, BWD-F2's K1, P3's suite.
- **L119:** blocking findings closed with tests (F01, F02, F06) or by ruling D1 (F03, F04, F05), then the confirm; F07
  and F08 with one fix or ruling each, lead-verified. Every write-once path of a staged line is listed so a retry can
  move it.
- **L125:** every staged step script starts with `forge-gpu-lease-check --require-held`; cite its file:line.
- **L127:** a Claude seat: your final message is the report, and the lead files it as `artifacts/bwdf1/f3/REPORT_F3.md`.
  Your JSON receipts and logs are written as usual.
- **L128:** see the writable set: the arm-swapping suite runner's temporary writes are listed and judged by sha256 after
  restore.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes (read-only git is allowed). NO subagents. No background waits; each command under an explicit timeout.
- No deletes, in the leave-and-list form. Never kill a process you did not start.
- **Prior Art Directive:** annotate each code site and the report. Known: NumPy's reduction axis rules
  (`normalize_axis_tuple`, "duplicate value in axis"); RoPE (Su et al., 2021); mutation testing (DeMillo, Lipton and
  Sayward, 1978). Write "no prior art known to me" where none applies.

## Done

Your final message contains, verbatim:
- P1 to P5, each PASS, FAIL or STAGED;
- the per-finding closure table (8 rows);
- the D1 domain-refusal rows;
- the diff summary;
- every module sha256;
- P3's CPU per-ID result and the intended changes;
- the mutant table;
- each staged GPU line with its dry-run exit;
- the deviations;
- every file written.
