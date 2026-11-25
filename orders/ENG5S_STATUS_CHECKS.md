# ENG5-S: ENG5's status checks alone, without the host-operand guards (implementing round)

Follows: ENG5_HOST_GUARD_AND_STATUS.md, ENG5_F1_REV1_FIXES.md, ENG5_F2_REV2_FIXES.md (branch `eng5`)
ID reuse: ENG5-S is the status half of ENG5, split out by the lead's sealed fallback

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng5s`
(branch `eng5s`, from `eng4` at `fbd335ae7` = production main `f7282b1d` plus ENG4, at the commit carrying this order).
Edits, builds and CPU runs are AUTHORIZED, limited to:
- `tensor_cuda/src/kernels.cu`, `tensor_cuda/src/apa_spo/packed_prefill.cuh`, `tensor_cuda/include/tc/core.h`: only the
  nine status checks below and the helper they share;
- new `tensor_cuda/tests/test_eng5s_*.py`, new `tensor_cuda/tests/support/eng5s_*.c` (copy ENG5's injection shim
  `tensor_cuda/tests/support/eng5_cuda_shim.c` from the `eng5` branch, read-only there), new `scripts/eng5s_*.py`,
  everything under `artifacts/eng5s/`;
- the build directory `tensor_cuda/build-eng5s/` and the module it installs into this worktree's package;
- `logs/eng5s_*`, and scratch under `/tmp/eng5s_*`. Point `TMPDIR` there for every command.

**Read-only:** every other file here; the `eng5` worktree (`tensor_fork_apa_wt_eng5`) and its receipts, which you read
and copy from; every other worktree; the production repository.
**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively; search inside the roots named here.
**GPU:** none of any kind. You stage GPU lines and dry-run them; the lead runs them.

## Why

ENG5 had two parts. Part H made every compute entry refuse host operands. Three cross-model reviews kept finding
blockers in each fix round's own guard code, so under the lead's sealed fallback (ledger, 2026-11-23 23:24) Part H is
parked for a later engine round. Part S makes CUDA calls whose failure used to pass silently check their own returned
status. It was stable through the reviews: REV1 found more such sites, F1 checked them, and REV2 confirmed them closed.
F1's status-injection GPU lines passed (`f1g3_F1` PASS; mutants `m05` and `m06` killed). This round takes Part S alone
onto ENG4's tree, so it can be promoted.

## The nine sites (lead's reading at `fbd335ae7`)

| Site | Call | Today |
|---|---|---|
| `kernels.cu:108` (`cuda_sync`) | `cudaDeviceSynchronize` | status discarded |
| `kernels.cu:185` (allocator OOM retry) | `cudaStreamSynchronize(0)` drain | status discarded |
| `kernels.cu:773` (`NDArray::zeros`) | `cudaMemset` | status discarded |
| `kernels.cu:885-897` (`NDArray::from_host`) | `cudaMemcpy` | checked only through `cuda_check_last` (the last-error slot) |
| `kernels.cu:898-908` (`NDArray::to_host`) | `cudaMemcpy` | the same |
| `kernels.cu:909-916` (`NDArray::clone`) | `cudaMemcpy` | status discarded |
| `kernels.cu:918-924` (`NDArray::to`) | `cudaMemcpy` | status discarded |
| `kernels.cu:5395-5401` (`axpy_`) | `cudaMemcpy` | status discarded |
| `apa_spo/packed_prefill.cuh:244` | `cudaMemset` (NaN fill) | status discarded |

## Rulings (binding)

- **S1.** Each of the nine checks its own returned status and raises `RuntimeError` naming the entry, the CUDA call and
  the error name (`cudaGetErrorName`). Take the form from ENG5-F1's `check_cuda_status` (`eng5` branch,
  `tensor_cuda/src/kernels.cu`; read it there). Do not read or clear the last-error slot for these calls. The OOM
  retry's drain is classed apart from cleanup: a failed drain raises instead of continuing to the retry.
- **S2.** Nothing else changes: no host-operand guard, no empty-operand rule, no metadata check. Results for every call
  whose CUDA status is success are unchanged.
- **S3.** If a site's code differs from the table at `fbd335ae7`, stop and report it before editing.

## Gates

- **G1, CPU only** (`CUDA_VISIBLE_DEVICES` empty). The ENG5 status rows that reach no CUDA call on host operands keep
  Build B's behaviour, and the change is confined to the nine sites (a source-map diff against Build B listing exactly
  the files above).
- **G2, GPU line, failure injection.** Port ENG5-F2's status-injection lines (`scripts/eng5_f2_gpu_step.py`, `f2g3_*`,
  on the `eng5` branch): the injection shim makes each of the nine calls fail in turn. On Build B, record what happens;
  on Build S, each raises naming its entry and error. Stage `g3_B`, `g3_S`, and the mutants below.
- **G3, no regression.** Production's test suite per test ID, Build B (`fbd335ae7`, ENG4's tip) against Build S, on CPU,
  and staged as GPU lines. Identical outcomes.
- **G4, mutants.** Each named in `artifacts/eng5s/MUTANTS.json` before it runs and killed by its G2 row:
  `s01_clone_status_dropped`, `s02_to_status_dropped`, `s03_zeros_status_dropped`, `s04_from_host_last_error_only`.
- **G5, scope.** Writes only in the writable set. No GPU use. No git writes, nothing deleted.

**Registered acceptance:** G1, G3's CPU half and G5 pass, and G2, G3's GPU half and the mutant lines are staged and
dry-run clean. The round is ACCEPTED when the lead's GPU lines pass. Then a read-only Codex confirm of this diff. Its
code is a subset of what REV2 already closed, so the confirm checks only that the subset is faithful.

## Builds

- **Build B:** this branch as committed (`fbd335ae7`).
- **Build S:** this round's tree.

Build both under `tensor_cuda/build-eng5s/` the way ENG5 built its arms (`scripts/eng5_build.py` on the `eng5` branch;
copy what you reuse into `scripts/eng5s_*.py`). Each receipt records the module sha256 and a sha256 map of every engine
source.

## Mechanism & scope

- **Measures:** whether the nine CUDA calls raise on failure with their own status, with every success result
  unchanged.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng5/artifacts/eng5/f1/REPORT_F1.md`
  (F1's status work) and `artifacts/eng5/f1/STATUS_CENSUS_F1.json` there (the census of 397 sites).
- **Exercises:** the nine sites; Builds B and S plus four mutant builds; CPU suite; staged injection and suite lines.
- **Does not cover:** Part H (host-operand guards, parked); BWD-F1's broadcasting and copy-op work (its own lane);
  multi-GPU behaviour; performance; promotion.
- **Required evidence:** the diff against `fbd335ae7`; build receipts with module sha256s; G1's result; G3's CPU per-ID
  result; `MUTANTS.json` and `MUTANT_RESULTS.json`; staged GPU lines with dry-run exits.
- **Operand census:** the nine sites above.
- **Baseline isolation:** Build B for unchanged results and the injection baseline.
- **Forbidden configuration:** any change outside the nine sites and their helper; reading or clearing the last-error
  slot at these sites; edits outside the writable set; git writes; deleting anything yourself.

## Machine rules

- **CPU:** at most one CPU-busy process of yours at a time, at `nice -n 19`. Builds and any job over 1 GB run in their
  own scope: `systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue -p CPUWeight=50 ...`, started only with
  `MemAvailable` at or above 12 GB; builds at most `-j4`.
- **Time:** every command under 30 minutes, builds under 60. The round's ceiling is 4 hours of wall time.
- **Disk:** stop if `/mnt/ForgeRealm` has under 40 GB free.
- **GPU:** none of any kind for you. Never call a step script or GPU function directly, never run `compute-sanitizer`,
  never remove or override `CUDA_VISIBLE_DEVICES` in a child. A smoke of a staged line is its `FORGE_GPU_DRY_RUN=1` dry
  run only. Stage lines in ENG5-F2's form (`--min` ≤ 480, `--max 540`, inner `timeout` ≤ 500 s); each step script and
  each child starts with
  `if subprocess.run(["forge-gpu-lease-check", "--require-held"]).returncode != 0: raise SystemExit("not inside a held forge-gpu-run lease")`
  before it imports the engine (L125). Each GPU row runs in its own child.

## Registration checklist (order-lint)

- **L01:** cmake, nvcc, gcc, pytest, python3 with NumPy, `forge-gpu-run` (`--status --min 30 --max 540`), each run first
  and recorded.
- **L02:** GPU lines carry an explicit `--min`; each lane's smoke is its dry run.
- **L03, L07, L14, L15, L18, L32, L37, L53, L55, L60, L89, L111, L117:** n/a (no share, nothing visual, no gain gate,
  no fitted threshold, no owner requirement behind an option, no bounded buffer, no knob, nothing for another project,
  no exact rule needing construction beyond G2's injection rows, no new refusal of an input, no name-keyed guard, no
  storage design, no gradient change).
- **L04:** no fallback; a failed call raises.
- **L08:** builds under `tensor_cuda/build-eng5s/`, receipts under `artifacts/eng5s/`, scratch under `/tmp/eng5s_*`.
- **L09, L10, L19:** G3 compares outcomes per test ID; success results held to identity with Build B; any value
  comparison checks each side nonzero and non-constant first.
- **L12:** inherited gates: production's suite per ID.
- **L16:** the cheapest passing work would check only the four ENG5 sites. Excluded by the nine-site table, G2's row per
  site and the four mutants.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`.
- **L22, L25:** cases defined by this order; each build in its own process.
- **L34:** a Claude seat (Opus 5.5, high effort, Agent tool); the confirm that follows is Codex.
- **L35:** if a clause cannot hold as written, report RED for that clause and finish the rest.
- **L38:** no existing test changes.
- **L40:** pytest with `-p no:cacheprovider`, `--basetemp` and `TMPDIR` under `/tmp/eng5s_*`; restore the tracked
  `artifacts/wo*.ppm` files `test_terrain_render.py` writes, or exclude that file and list it.
- **L41, L47:** each mutant names its failing G2 row in `MUTANTS.json` before it runs; expectations come from this order.
- **L45:** each staged line records `forge-gpu-run`'s exit, the inner exit and the log sha256; 124 is never a pass.
- **L48:** staged lines run as written from the worktree root through the real `forge-gpu-run`; drive the verdict code on
  constructed results (each site raising, returning, a dead child) before staging, and list those steps.
- **L56:** (a) tool slips go under Deviations: re-run and continue. (b) At most 3 repairs of each of your scripts.
- **L58:** CPU runs with `CUDA_VISIBLE_DEVICES` empty; no child of yours removes or overrides it.
- **L64:** reads are this worktree, the `eng5` worktree's scripts, shim and receipts, and ENG3's counter source.
- **L65:** leave-and-list form for any file you would remove.
- **L67:** one build per arm (about 60 MB each) plus four mutants. Free space at sealing above 70 GB on
  `/mnt/ForgeRealm`; stop below 40 GB.
- **L69:** success results and the injection baseline -> Build B; refusals on failure -> S1.
- **L74:** plain wording.
- **L76:** the lead writes nothing in the writable set while the seat runs.
- **L77:** every mutant can fail and each is run (as staged GPU lines).
- **L79:** `FORGE_GPU_DRY_RUN=1` returns 78 without a device check; once per staged line.
- **L81:** the site table is the lead's reading at `fbd335ae7` (source records: file:line and the call); S3 makes the
  seat re-read it first. F1's GPU results are cited only to say the injection harness works (`f1g3_F1` PASS,
  `m05`/`m06` killed, lead runs `b73745eec`).
- **L82:** `grep -n "cudaMemcpy\|cudaMemset\|cudaDeviceSynchronize\|cudaStreamSynchronize" tensor_cuda/src/kernels.cu tensor_cuda/src/apa_spo/packed_prefill.cuh`
  starts the census.
- **L90:** roles: product (the nine sites), test (the new file), harness (scripts). Count per role.
- **L99:** the lead read the nine sites at `fbd335ae7` before sealing (lines in the table).
- **L102:** no generated file or registry depends on the writable files.
- **L103:** each staged GPU line refuses, before the lease, a module whose sha256 differs from its receipt.
- **L104:** no ratio bar.
- **L109:** helpers run as `python3 -B -P`; none shares a standard-library module name.
- **L114:** builds in their own scope (`MemoryMax=12G`); staged lines set no RLIMIT_AS; peak host bytes listed.
- **L118:** consumers of the changed calls: every caller of the nine sites, all covered by G3's suite.
- **L119:** every write-once path of a staged line is listed so a retry can move it.
- **L125:** every step script and child starts with `forge-gpu-lease-check --require-held`; cite its file:line.
- **L127:** a Claude seat: your final message is the report, and the lead files it as `artifacts/eng5s/REPORT.md`.
- **L128:** if you copy ENG5's arm-swapping suite runner, list the engine files it swaps as temporarily written and
  restored, and judge scope by sha256 after each restore.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes (read-only git is allowed). NO subagents. No background waits; each command under an explicit timeout.
- No deletes, in the leave-and-list form. Never kill a process you did not start.
- **Prior Art Directive:** annotate each code site and the report. Known: the CUDA runtime's returned-status model
  versus `cudaGetLastError`; fault injection by library interposition (LFI, Marinescu and Candea, 2009; unverified,
  lead to check); mutation testing (DeMillo, Lipton and Sayward, 1978).

## Done

Your final message contains, verbatim:
- G1 to G5, each PASS, FAIL or STAGED;
- the nine-site table with each fix's file:line;
- the diff summary;
- every module sha256;
- G3's CPU per-ID result;
- the mutant table;
- each staged GPU line with its dry-run exit;
- the deviations;
- every file written.
