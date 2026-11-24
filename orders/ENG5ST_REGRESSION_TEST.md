# ENG5-S-T: a standalone production regression test for ENG5-S's nine status checks

Follows: ENG5S_STATUS_CHECKS.md (this branch) and ENG5S_REV1_CONFIRM.md (branch `eng5srev1`)
ID reuse: ENG5-S's rounds share the ENG5 id

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng5s`
(branch `eng5s`, at the commit carrying this order). Edits and CPU runs are AUTHORIZED there, limited to:
- the new test file `tensor_cuda/tests/test_eng5s_status_checks.py`;
- new `scripts/eng5st_*.py`;
- everything under `artifacts/eng5s/t/` (new);
- `logs/eng5st_*`, and scratch under `/tmp/eng5st_*`. Point `TMPDIR` there for every command.

**Read-only:**
- every product file;
- every other test file, including `tensor_cuda/tests/test_eng5s_status.py` and
  `tensor_cuda/tests/support/eng5s_cuda_shim.c` (use the shim as it is);
- ENG5-S's scripts and receipts, and the receipted builds in `tensor_cuda/build-eng5s/` (B, S, s01-s04);
- every other worktree; the production repository.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively; search inside this worktree.
**GPU:** none of any kind. You stage GPU lines and dry-run them; the lead runs them.

## Why

ENG5-S made nine CUDA calls check their own returned status. Its evidence is a differential harness:
`test_eng5s_status.py` compares the receipted Build B and Build S packages under `tensor_cuda/build-eng5s/`, and the
GPU injection lines run under `scripts/eng5s_gpu_step.py`.
- Those builds are git-ignored, so in production `test_eng5s_status.py` cannot run. It stays a fork-lane harness, and
  the promotion tree leaves it out.
- Production still needs a test that fails if any of the nine checks is removed, as BWD's tests and ENG4's
  `test_eng4_matmul.py` do for their changes.
- BWD's tests show the form: they compile the committed `tests/support/cudart_count.c` at test time and test the
  installed `tensor_cuda` only.

## The test (`tensor_cuda/tests/test_eng5s_status_checks.py`)

**Self-contained.**
- It imports the installed `tensor_cuda` only.
- It reads no build directory, receipt, artifact or script.
- It compiles `tests/support/eng5s_cuda_shim.c` at test time into `tmp_path`, finding the compiler as BWD's tests do.
  With no compiler, it skips, naming the missing compiler.

**Rows, each in its own child process with the shim preloaded** (`ENG5_INJECT` arms one call, as the GPU lines do):
- **CPU lane** (`CUDA_VISIBLE_DEVICES` empty; skip on the GPU lane with a reason naming the lane):
  - `tc.synchronize` raises `RuntimeError` in the S1 form (`cuda_sync: cudaDeviceSynchronize failed: cudaError...`);
  - the OOM retry: with the allocation injected to fail with an out-of-memory error and the drain injected to fail,
    allocation raises in the S1 form naming the OOM retry and the drain, through both `cudaMalloc` and
    `cudaMallocAsync` where the CPU lane reaches them (as `G1_OOM_DRAIN_CPU.json` did).
- **GPU lane** (a GPU visible; skip on the CPU lane with a reason naming the lane): the eleven injection rows of
  `eng5s_gpu_step.py`'s `G2_CASES` and its empty `axpy_` control-only row. For each row:
  - an uninjected success control first, whose result equals a value computed on the host;
  - then the injected call, which must fire (`eng5s_fired`) and raise `RuntimeError` in the S1 form naming that entry
    and call;
  - the dump fill (site 9) needs a hooks build, so it is excluded from the production test, with a one-line comment
    naming the reason. Its evidence stays the lead's `g3_S` and `g3_B` lines.

**Message checks** use the S1 regular expression from `test_eng5s_status.py` (copy it; do not import that file).

## Gates

- **T1, CPU:**
  - With the installed module (Build S), the new file's CPU rows pass and its GPU rows skip with a lane reason.
  - The CPU rows run against Build B's package (`tensor_cuda/build-eng5s/B/pkg` first on `PYTHONPATH`, with its
    module sha256 checked against `artifacts/eng5s/build/BUILD_B.json`), and each fails. This shows the test detects
    the missing checks.
  - Record both runs as junit XML under `artifacts/eng5s/t/`.
- **T2, GPU lines, staged:** one line per arm, running the new file under pytest with that arm's package first on
  `PYTHONPATH`. The arms are S, B and the four mutants s01-s04.
  - **Expected:** on S every GPU row passes. On B every injected row fails. Each mutant fails the row(s) of its own
    site (s01 clone, s02 both directions of `to`, s03 zeros, s04 from_host) and passes every other row: KILLED.
  - **Form:** `forge-gpu-run --min 120 --max 540 --label eng5st-<arm> -- timeout 480s python3 -B -P
    scripts/eng5st_gpu_step.py <arm>`.
  - **The step script:**
    - checks the arm's module sha256 before the lease;
    - starts with `forge-gpu-lease-check --require-held`;
    - runs pytest as a child, inside the lease;
    - writes under `artifacts/eng5s/t/gpu/<arm>/`, refusing to start if that directory exists;
    - writes the verdict, from the junit XML per row.
  - Dry-run each line once with `FORGE_GPU_DRY_RUN=1` (exit 78), and record them in `artifacts/eng5s/t/GPU_LINES_T.json`.
- **T3, the verdict code:** drive the step's verdict on constructed junit files (all pass; one row failed; a mutant
  failing its own row; a mutant failing a different row; a missing row; an empty file), each giving its expected
  verdict.
- **T4, scope:** writes only in the writable set; no product change; no other test file changed; no GPU use; no git
  writes; nothing deleted.

**Registered acceptance:** T1, T3 and T4 pass, and the lead's T2 lines give PASS on S, the defect on B, and KILLED
for s01-s04.

## Mechanism & scope

- **Measures:** whether a self-contained production test catches the removal of ENG5-S's status checks.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng5s/artifacts/eng5s/REPORT.md`
  (the nine sites, the shim, the GPU rows) and `scripts/eng5s_gpu_step.py` (the injection rows).
- **Exercises:** the new test on the CPU lane against Builds S and B; the verdict drive; dry runs.
- **Does not cover:** site 9 in the production test (it needs a hooks build); the host-operand work (parked); the
  merged-tree check (next, by the lead).
- **Operand census:** the nine sites; the eleven injection rows and the empty `axpy_` control; the four mutants.
- **Baseline isolation:** Build B for "defect present"; Build S for the fix; s01-s04 for the kills.
- **Required evidence:** the test file; T1's two junit files; `GPU_LINES_T.json` with dry-run exits; the T3 drive
  results; `COPY_DEPS.json` listing what was copied from `eng5s_gpu_step.py` and `test_eng5s_status.py`.
- **Forbidden configuration:** edits outside the writable set; reading build directories or receipts from the test
  file; GPU use; git writes; deleting anything yourself.

## Machine rules

- At most one CPU-busy process of yours at a time, at `nice -n 19`; every command under an explicit `timeout`.
- No GPU of any kind: never call a step script outside its dry run, never run `compute-sanitizer`, never remove or
  override `CUDA_VISIBLE_DEVICES` in a child of yours. A smoke of a staged line is its `FORGE_GPU_DRY_RUN=1` dry run.
- **Ceiling:** 2 hours of wall time.

## Registration checklist (order-lint)

- **L01:** python3, pytest, gcc and `forge-gpu-run` (`FORGE_GPU_DRY_RUN=1`, exit 78), each run first and recorded.
- **L02:** every staged line carries an explicit `--min`; each line's smoke is its dry run.
- **L08:** receipts under `artifacts/eng5s/t/`, scratch under `/tmp/eng5st_*`.
- **L16:** the cheapest pass would be a test whose rows all skip. Excluded: T1 requires the CPU rows to run and fail
  on Build B, and T2 requires the GPU rows to fail on B and kill s01-s04.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`.
- **L34:** a Claude seat (Sonnet 5.5, high effort, Agent tool). The work is an LD_PRELOAD injection test, which has
  stopped Codex seats at the provider's filter.
- **L35:** if a clause cannot hold as written, report RED for that clause and finish the rest.
- **L40:** pytest runs with `-p no:cacheprovider`, `--basetemp` and `TMPDIR` under `/tmp/eng5st_*`.
- **L41:** each mutant's kill is judged on its own site's row, and every other row must pass on that mutant.
- **L45:** each staged line records `forge-gpu-run`'s exit, the inner exit and the junit sha256; 124 is never a pass.
- **L48:** staged lines run as written from the worktree root through the real `forge-gpu-run`. T3 drives the verdict
  on constructed results.
- **L56:** (a) tool slips go under Deviations: re-run and continue. (b) At most 3 repairs of each of your scripts.
- **L58:** every CPU command runs with `CUDA_VISIBLE_DEVICES` empty; no child of yours removes or overrides it. The
  test's own GPU rows run only on the GPU lane.
- **L64:** reads are this worktree's files named above.
- **L65:** leave-and-list form for any file you would remove.
- **L69:** one reference per class: Build B for the defect; Build S for the fix; s01-s04 for the kills; host-computed
  values for the success controls.
- **L76:** the lead writes nothing in the writable set while the seat runs.
- **L79:** `FORGE_GPU_DRY_RUN=1` returns 78 without a device check; once per staged line.
- **L81:** the failures come from ENG5-S's receipts: `gpu/g3_B/VERDICT.json` (Build B returned under every fired
  injection) and `g1/G1_OOM_DRAIN_CPU.json`. Nothing else is assumed.
- **L89:** n/a. No guard keyed on a name.
- **L90:** roles: one test file (test), the step and verdict scripts (harness). Product is read-only.
- **L99:** the lead read, before sealing:
  - `test_eng5s_status.py:1-60`, which reads `tensor_cuda/build-eng5s/<arm>/pkg` and `artifacts/eng5s/build/BUILD_<arm>.json`;
  - `eng5s_gpu_step.py`'s `G2_CASES`;
  - `git ls-tree 75c85de6 tensor_cuda/tests/support/`, which lists only `cudart_count.c`, so the new test brings the
    ENG5-S shim into production.
- **L103:** each staged line refuses, before the lease, a module whose sha256 differs from its receipt.
- **L109:** helpers run as `python3 -B -P`; none shares a standard-library module name.
- **L118:** the consumers of the new test: the production suite (it runs on both lanes) and the next merged-tree check.
- **L125:** the step script starts with `forge-gpu-lease-check --require-held`; cite its file:line. The test file is a
  production test: it does not call the lease check, and its children run inside the step's lease.
- **L127:** a Claude seat: your final message is the report, and the lead files it as `artifacts/eng5s/t/REPORT_T.md`.
- **L03, L07, L09, L10, L14, L19, L25, L32, L47, L53, L55, L60, L77, L104, L111, L114, L117, L128:** n/a. A test port:
  no share, nothing visual, no suite comparison, no identity gate across builds, no gain gate, no fitted threshold, no
  owner option, no bounded buffer, no knob, no new refusal, no ratio bar, no storage design, no job over 1 GB, no
  gradient, no arm-swapping runner (each arm is imported from its own package directory).

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes (read-only git is allowed). NO subagents. No background waits; each command under an explicit timeout.
- No deletes, in the leave-and-list form. Never kill a process you did not start.
- **Prior Art Directive:** annotate the test file and the report. Known: ENG3's LD_PRELOAD counter and ENG5's
  injection shim (this repo); library-level fault injection (LFI, Marinescu and Candea, 2009); mutation testing
  (DeMillo, Lipton and Sayward, 1978).

## Done

Your final message contains, verbatim:
- T1 to T4, each PASS or FAIL;
- the test's rows, with each one's lane and what it asserts;
- each staged line with its dry-run exit;
- the deviations;
- every file written.
