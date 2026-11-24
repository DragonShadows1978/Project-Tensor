# BWD-F2: fix BWD-F1-REV1's findings (copy and row ops, rank ceiling, where/masked_fill, axpy_), and restage BWD-F1's GPU lines (implementing round)

Follows: BWDF1_ZERO_EXTENT_AND_AUDIT_FIXES.md and BWDF1_AMENDMENT_1_COPY_OPS.md (this branch)
ID reuse: BWD-F1's fix rounds (F2, ...) share the BWD-F1 id

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_bwdf1`
(branch `bwdf1`, at the commit carrying this order; BWD-F1's round is `fbe04d8f9`). Edits, builds and CPU runs are
AUTHORIZED, limited to:
- `tensor_cuda/src/kernels.cu`, `ops.cpp`, `bindings.cpp`, `tensor_cuda/include/tc/*.h`, `tensor_cuda/tensor_cuda/functional.py`:
  only the fixes ruled below, and host helpers for them;
- BWD-F1's tests (`tensor_cuda/tests/test_bwdf1_*.py`, `tc_signal/tests/test_bwdf1_hardening.py`), new
  `tensor_cuda/tests/test_bwdf2_*.py`; existing tests only as R12 allows;
- `scripts/bwdf1_*.py`, new `scripts/bwdf2_*.py`, everything under `artifacts/bwdf1/f2/`;
- the build directories `tensor_cuda/build-bwdf1/` and `tc_signal/build-bwdf1/` (new arms `F2*`, `n0*`), and the modules
  they install into this worktree's packages;
- `logs/bwdf2_*`, and scratch under `/tmp/bwdf2_*`. Point `TMPDIR` there for every command.

**Read-only:** every other file here (BWD-F1's receipts under `artifacts/bwdf1/` included); the review
(`tensor_fork_apa_wt_bwdf1rev1/artifacts/bwdf1_rev1/`, commit `f40bef8c4`); every other worktree; the production
repository; the consumer roots `/mnt/ForgeRealm/Project-Voice/engine` and `/mnt/ForgeRealm/GRAPA-Native-LLM` (grep only).
**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively; search inside the roots named here.
**GPU:** you run no GPU work of any kind. You stage GPU lines and dry-run them; the lead runs them.

## Why

BWD-F1 fixed zero-extent broadcasting and made the copy ops check their operands (`fbe04d8f9`). Its cross-model review
(BWD-F1-REV1, Codex, read-only, complete; `artifacts/bwdf1_rev1/findings.json` sha256 `fc26ea52...`) found 13 defects:
6 BLOCKER, 4 MAJOR, 3 MINOR.
- One is a regression of BWD-F1's (F03): a dtype and device check on `cat`/`stack` members whose bytes are never
  read.
- The rest are inherited defects in the code BWD-F1 touched: rank-9 metadata written into 8-slot arrays (F02), a RoPE
  sine span read past its table (F01), a ring index that overflows (F04), `where`/`masked_fill` reading a bool
  condition as the data dtype (F07), `axpy_` installing values from the wrong rows (F12), plus late dtype checks,
  unchecked sums, weak tests and messages.

## Lead rulings (binding)

- **R1 (F03), check what is read.** The same principle as ENG5-F1. Rank and extent checks stay for every operand,
  because they decide the output shape. Dtype and device checks apply only to operands whose bytes are copied (a
  non-zero copy). The output dtype and device follow Build B for empty work. Uniform dtype and device are not made an
  API constraint for no-op members.
- **R2 (F02), the rank ceiling.** Every consumer of fixed-size metadata arrays (the copy specs, binary, compare, where,
  reduction and pad dispatch) refuses an effective rank above its array size before any allocation or metadata write.
  That includes `stack`'s rank after unsqueeze, and empty operands. Census every such array and its size.
- **R3 (F01, F05), RoPE.** One shared validator checks the actual read span of both the cosine and the sine table,
  using subtraction-based bounds, plus the tables' shape, device and supported floating dtype. It is used by
  `rope_apply`, the export and swap preflights and the arena path. A pair op validates both sides before any work.
- **R4 (F04), ring index.** Reduce `start` modulo the capacity on the host when a copy will run, and use overflow-free
  wraparound in the kernel.
- **R5 (F07), condition dtype.** `where` and `masked_fill` validate the condition's dtype and dispatch its loads on its
  own dtype, or refuse an incompatible condition before any allocation (including before `masked_fill` builds its
  fill). The data operands' dtypes must match, or be refused, before allocation.
- **R6 (F12), `axpy_`.** The update must map element for element onto the destination's shape. Refuse a broadcast that
  does not, before any work. Leading size-one differences that map exactly stay accepted.
- **R7 (F08), checked arithmetic.** Checked sums and checked non-zero products before allocation or any pair's work, in
  the copy preflights, `splice_rows`, the arena span and `write_rows`. Validate the arena span before computing
  `tail_start`. Decide emptiness before multiplying.
- **R8 (F09), dtype before allocation.** `slice_nd` and `pad_into` check the launch dtype before allocating, when a
  copy will run. `__getitem__` checks each axis before reading its extent.
- **R9 (F06), tests.** The later-pair ordering rows use a first pair that would allocate or launch, and show that a
  bad later pair refuses before that work. Add passing zero-size `cat`/`stack` controls, the R1 cases among them.
- **R10 (F10, F11, F13), minor.** Generate equal operands once and label the `axpy_` size refusals separately.
  Attribute the CPU `reduce_to` failure to `m03`. Messages keep the dim-error context and add the operand position and
  the pair or layer index. `einsum` preflights the label extents its factor construction actually uses.
- **R11, restage.** BWD-F1's 35 staged GPU lines were never run (the lead stopped the runner; Build E is superseded).
  Restage them against Build F2 under `artifacts/bwdf1/f2/gpu/`, add one GPU row for each fix that only the GPU can
  show, and keep Build B as the baseline.
- **R12.** Existing tests change only to follow R1's restored Build B behaviour, each listed in
  `artifacts/bwdf1/f2/INTENDED_CHANGES_F2.json` before any after-run. No knob.

## Gates

- **K1, CPU only** (`CUDA_VISIBLE_DEVICES` empty), on Build F2. One row per finding's named case where the CPU can
  reach it, each shown failing on Build E (`fbe04d8f9`) where Build E reaches it, and passing on Build F2. Refusals
  show 0 CUDA calls with ENG3's counter. BWD-F1's G1 still passes, except rows changed by R1 and listed.
- **K2, GPU rows, staged.** The cases only the GPU can show (F01's sine read, F04's wrap, F07's condition read, F12's
  values), under memcheck where memory is involved. Defect on Build E, absent on Build F2.
- **K3, no regression.** Production's test suite per test ID, Build B (`70cd1752` and `857432c5`, BWD-F1's base)
  against Build F2, on CPU and as staged GPU lines. Identical except BWD-F1's two listed H1 IDs and R12's list.
- **K4, mutants.** Each fails its named assertion in `artifacts/bwdf1/f2/MUTANTS_F2.json`, written before any mutant
  runs:
  - `n01_dtype_check_all_members` (R1);
  - `n02_no_rank_ceiling_stack` (R2);
  - `n03_sin_span_unchecked` (R3);
  - `n04_ring_no_modulo` (R4);
  - `n05_condition_as_data_dtype` (R5);
  - `n06_axpy_broadcast_accepted` (R6).
  Each runs on CPU where its row is CPU-reachable, else as a staged GPU line.
- **K5, scope.** Writes only in the writable set. No GPU use. No git writes, nothing deleted.

**Registered acceptance:** the seat's part passes when K1, K3's CPU half, the CPU mutants and K5 pass, and K2, K3's GPU
half, R11's restaged lines and the GPU mutants are staged and dry-run clean. The round is ACCEPTED when the lead's GPU
lines pass. Then a read-only Codex confirm of the F2 diff against the 13 findings.

## Builds

- **Build B:** BWD-F1's base as receipted (`tc` `70cd1752...`, `ts` `857432c5...`).
- **Build E:** BWD-F1 as receipted (`tc` `8b84e5d8...`, `ts` `1ab3757f...`).
- **Build F2:** this round's tree, both packages.

## Mechanism & scope

- **Measures:** whether the review's 13 findings are closed, with results on accepted CUDA inputs unchanged and BWD-F1's
  zero-extent fixes intact.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_bwdf1rev1/artifacts/bwdf1_rev1/REPORT.md`
  (the review) and `artifacts/bwdf1/REPORT.md` (BWD-F1's report, in this worktree).
- **Exercises:** the ruled fixes; Builds B, E and F2 plus mutant builds; CPU tests with the call counter; staged GPU
  lines (K2, K3, R11, mutants).
- **Does not cover:** ENG5's host-operand guards (its own lane; R8's `__getitem__` axis check is a metadata check, not
  a host guard); the index-driven ops BWD-F1 listed as residuals (scatter, gather, embedding); multi-GPU behaviour;
  performance; promotion.
- **Required evidence:** a per-finding closure table (13 rows: finding, fix file:line, test, CPU or GPU result); the R2
  census of fixed-size arrays; `BOUNDARY_F2.json`; the diff against `fbe04d8f9`; build receipts with module sha256s;
  K1 results; `MUTANTS_F2.json` and `MUTANT_RESULTS_F2.json`; `INTENDED_CHANGES_F2.json`; K3's CPU per-ID result;
  staged GPU lines with dry-run exits.
- **Operand census:** every finding's named case; every fixed-size metadata array and its consumers; every RoPE read
  path; every copy op's members by "bytes read or not".
- **Baseline isolation:** Build B for K3 and for R1's restored behaviour; Build E for "defect present".
- **Forbidden configuration:**
  - a dtype or device check on a member whose bytes are not read (R1);
  - a refusal after an allocation or CUDA call;
  - any change to an accepted CUDA result beyond the ruled fixes;
  - edits outside the writable set; git writes; deleting anything yourself.

## Machine rules

- **CPU:** at most one CPU-busy process of yours at a time, at `nice -n 19`. Builds and any job over 1 GB run in their
  own scope: `systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue -p CPUWeight=50 ...`, started only with
  `MemAvailable` at or above 12 GB; builds at most `-j4` (another seat builds on this machine).
- **Time:** every command under 30 minutes, builds under 60. The round's ceiling is 6 hours of wall time.
- **Disk:** stop if `/mnt/ForgeRealm` has under 40 GB free.
- **GPU:** none of any kind for you. Never call a step script or GPU function directly (BWD-F1's deviation 6 must not
  recur), never run `compute-sanitizer`, never remove or override `CUDA_VISIBLE_DEVICES` in a child. A smoke of a staged
  line is its `FORGE_GPU_DRY_RUN=1` dry run only. Stage lines as BWD-F1 did (`scripts/bwdf1_gpu_lines.py` form,
  `--min` ≤ 480, `--max 540`, inner `timeout` ≤ 500 s). Every step script starts with
  `if subprocess.run(["forge-gpu-lease-check", "--require-held"]).returncode != 0: raise SystemExit("not inside a held forge-gpu-run lease")`
  before it imports the engine or reads `CUDA_VISIBLE_DEVICES` (order-lint L125, GW8).

## Registration checklist (order-lint)

- **L01:** cmake, nvcc, gcc, pytest, python3 with NumPy, `forge-gpu-run` (`--status --min 30 --max 540`), each run
  first and recorded.
- **L02:** GPU lines carry an explicit `--min`; each lane's smoke is its dry run.
- **L03, L07, L14, L15, L18, L32, L37, L53, L111, L117:** n/a (no share, nothing visual, no gain gate, no fitted
  threshold, no owner requirement behind an option, no bounded buffer, no knob, nothing for another project's run, no
  storage design, no gradient change).
- **L04:** no fallback; R2-R8 refuse, they do not repair.
- **L08:** builds under the two build directories (`F2*`, `n0*`), receipts under `artifacts/bwdf1/f2/`, scratch under
  `/tmp/bwdf2_*`.
- **L09:** K3 compares outcomes per test ID; value comparisons are per element on same-width unsigned views or against
  NumPy.
- **L10:** results on accepted CUDA inputs held to identity with Build B, beyond the ruled fixes.
- **L12:** inherited gates: production's suite per ID and BWD-F1's G1; intended changes listed first (R12).
- **L16:** the cheapest passing work would patch the named cases only (one shape each). Excluded by R2's census of
  every fixed-size array, R3's shared validator over every RoPE path, R1's members-by-bytes census, and `n01`-`n06`.
- **L19:** every bitwise comparison first checks each side is nonzero and non-constant.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`.
- **L22, L25:** cases and seeds defined by this order and `MUTANTS_F2.json`; each build in its own process.
- **L34:** a Claude seat (Opus 5.5, high effort, Agent tool); the confirm that follows is Codex.
- **L35:** registrations may fix cases and seeds, never narrow a gate. If a clause cannot hold as written, report RED
  for that clause and finish the rest.
- **L38:** existing tests change only as R12 allows, each listed.
- **L40:** pytest with `-p no:cacheprovider`, `--basetemp` and `TMPDIR` under `/tmp/bwdf2_*`; restore the tracked
  `artifacts/wo*.ppm` files `test_terrain_render.py` writes, or exclude that file and list it.
- **L41:** each mutant names its failing assertion before it runs; show each killing test executes the mutated line.
- **L45:** each staged line records `forge-gpu-run`'s exit, the inner exit and the log sha256; 124 is never a pass.
- **L47:** mutant expectations come from this order.
- **L48:** staged lines run as written from the worktree root through the real `forge-gpu-run`; drive each line's
  post-device code on constructed results of each shape class it will see, empty and one-element included, and list
  those steps; no trailing command masks a line's exit.
- **L55:** for one witness per ruling R1-R8, show the inputs, Build E's path and Build F2's check position.
- **L56:** (a) tool slips go under Deviations: re-run and continue. (b) At most 3 repairs of each of your scripts.
- **L58:** K1 runs with `CUDA_VISIBLE_DEVICES` empty; no child of yours removes or overrides it.
- **L60:** every new or changed refusal is in `BOUNDARY_F2.json` with its terms, the rows that trigger each, and Build
  B's behaviour for the same operands.
- **L64:** reads are this worktree, the review's receipts, ENG3's counter source and the consumer roots (grep only); the
  inherited suite reads what BWD-F1's open audit recorded; re-record it for Build F2's CPU run.
- **L65:** leave-and-list form for any file you would remove.
- **L67:** BWD-F1's build directories were 439 MB and 19 MB. Free space at sealing: 73 GB on `/mnt/ForgeRealm`
  (`df -h`, 2026-11-23 20:36). Stop below 40 GB.
- **L69:** one reference per class: restored no-op behaviour -> Build B; review cases -> the review's stated behaviour,
  re-measured on Build E; accepted CUDA results -> Build B per ID; refusals -> this order.
- **L74:** plain wording.
- **L76:** the lead writes nothing in the writable set while the seat runs.
- **L77:** every mutant can fail and each is run (GPU ones staged).
- **L79:** `FORGE_GPU_DRY_RUN=1` returns 78 without a device check; once per staged line.
- **L81:** the review's cases are its readings (source records: file:line and the shapes it names; nothing run). K1
  and K2 measure each on Build E before claiming it closed.
- **L82:** `grep -n "DimCopySpec\|DimSpec\|\[8\]\|kMaxDims\|rope_apply\|ring\|where_nd\|masked_fill\|axpy_" tensor_cuda/src/*.cu tensor_cuda/src/*.cpp`
  starts the censuses.
- **L89:** evasion: a fixed-size array consumer under another name. Excluded by R2's census of arrays by declared size,
  not by function name; control `n02`.
- **L90:** roles: product (engine), test, consumer (report-only), harness. Count per role in the census.
- **L99:** the rulings rest on the review's findings (`findings.json` `fc26ea52...`, 13 rows with file:line evidence),
  which the lead read in full before sealing. Seat check that runs first: re-read each finding's evidence at
  `fbe04d8f9` and record any that do not match.
- **L102:** no generated file or registry depends on the writable files.
- **L103:** each staged GPU line refuses, before the lease, a module whose sha256 differs from its receipt; receipts
  carry the full-tree source map.
- **L104:** no ratio bar; every gate is exact.
- **L109:** helpers run as `python3 -B -P`; none shares a standard-library module name.
- **L114:** builds in their own scope (`MemoryMax=12G`); staged lines set no RLIMIT_AS; peak host bytes listed.
- **L118:** consumers of the changed checks: K1's rows, BWD-F1's G1, K3's suite.
- **L119:** blocking findings (F01-F04, F07, F12) are closed with tests and the confirm; the others with one fix each,
  lead-verified. Every write-once path of a staged line is listed so a retry can move it.
- **L125:** every staged step script starts with `forge-gpu-lease-check --require-held`; cite its file:line.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes (read-only git is allowed). NO subagents. No background waits; each command under an explicit timeout.
- No deletes, in the leave-and-list form. Never kill a process you did not start.
- **The report:** your final message is the report; the lead files it as `artifacts/bwdf1/f2/REPORT_F2.md` verbatim.
- **Prior Art Directive:** annotate each code site and the report. Known: NumPy's `concatenate`, `stack`, `where` and
  `putmask` rules; the RoPE formulation (Su et al., 2021, "RoFormer"); mutation testing (DeMillo, Lipton and Sayward,
  1978). Write "no prior art known to me" where none applies.

## Done

Your final message contains, verbatim:
- K1 to K5, each PASS, FAIL or STAGED;
- the per-finding closure table (13 rows);
- the R2 array census summary;
- the diff summary;
- every module sha256;
- K3's CPU per-ID result and the intended changes;
- the mutant table;
- each staged GPU line with its dry-run exit;
- the deviations;
- every file written.
