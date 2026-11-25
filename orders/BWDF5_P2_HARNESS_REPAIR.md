# BWD-F5: repair BWD-F3's P2 GPU harness (every child died under memcheck) and restage the P2 rows on the final build

Follows: BWDF3_REV2_FIXES.md (branch `bwdf1`) and BWDF4_REV3_HARDENING.md (this branch)
ID reuse: BWD-F1's fix rounds (F2-F5) share the BWD-F1 id

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_bwdf4`
(branch `bwdf4`: BWD-F3 `2117296ea` plus BWD-F4 `529eea9de`, at the commit carrying this order). Edits and CPU runs are
AUTHORIZED, limited to:
- new `scripts/bwdf5_*.py` (copy BWD-F3's `scripts/bwdf3_step_p2.py`, `bwdf3_p2_cases.py`, `bwdf3_gpu_common.py`,
  `bwdf3_gpu_lines.py`, `bwdf3_gpu_post.py` and whatever else the P2 path needs into `bwdf5_*` names and change only
  the copies);
- everything under `artifacts/bwdf1/f5/`;
- `logs/bwdf5_*`, and scratch under `/tmp/bwdf5_*`. Point `TMPDIR` there for every command.

**Read-only:** every product and test file; BWD-F3's and BWD-F4's scripts and receipts; every other worktree
(`tensor_fork_apa_wt_bwdf1` holds BWD-F3's GPU line results and the lead's runner, which is active: do not touch it);
the production repository. No build is needed. Use the modules already built and receipted: Build B and Build F2 in
`tensor_fork_apa_wt_bwdf1/tensor_cuda/build-bwdf1/` and Build F4 in this worktree's `tensor_cuda/build-bwdf4/F4/`.
**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively; search inside the roots named here.
**GPU:** none of any kind. You stage GPU lines and dry-run them; the lead runs them.

## Why

The lead ran BWD-F3's staged lines (log `tensor_fork_apa_wt_bwdf1/logs/bwdf3_lead_runner.out`). 33 of the 42 short
lines exited 0. All 9 P2 lines (`p2_e1_*`, `p2_e2_*`, `p2_e3_*` on Builds B, F2 and F3) exited 255, and every row in
every one of them is `CHILD-DIED` with "child exit 255" and empty stderr, the control rows included.
- The lines run `compute-sanitizer --tool memcheck --target-processes all` around the step script.
- The step script runs each row in a child process, which re-executes the same script with `--child`. The child's
  line 2 runs `forge-gpu-lease-check --require-held` before anything else.
- Each line's memcheck log holds exactly two lines: `COMPUTE-SANITIZER` and
  `Error: Target application terminated before first instrumented API call`.
So the children die before their first CUDA call, and the P2 evidence (E1-E3 on the GPU: RoPE read spans, the
`[T, D, 1]` table values bitwise against `[T, D]`, reduction axes under memcheck) does not exist yet.

## Rulings (binding)

- **A1, diagnose from the code and receipts.** Write `artifacts/bwdf1/f5/DIAGNOSIS.md`: the child launch path, what
  `--target-processes all` does to it, and the most likely cause of exit 255 with empty stderr. Cite the files and the
  lead's logs. You cannot run the sanitizer; say what is inferred and what is read.
- **A2, a harness that cannot fail this way.** Run each row's child under its own memcheck instead of wrapping the
  whole step: the step (inside the lease, no sanitizer) runs `compute-sanitizer --tool memcheck --log-file <row log>
  python3 -B -P <script> --child ...` per row, and the verdict reads each row's memcheck log. Each child still checks
  the lease first (`--require-held`), and the parent records a child's death as a row. If A1 shows a simpler correct
  fix, use it and say why.
- **A3, restage the P2 rows on the final build.** The same rows as BWD-F3's P2 (`bwdf3_p2_cases.py`), on Builds B, F2
  and F4 (F4 replaces F3: it is F3 plus the hardening, and is the tree that will be promoted). Write each line to
  `artifacts/bwdf1/f5/gpu/<line>/`, refusing to start if that directory exists. Add one canary line that runs a single
  control row on Build F4 first, so the lead sees at once whether children survive under the sanitizer.
- **A4, drive it on CPU.** Drive the new verdict code on constructed results (a dead child, a memcheck log with 0
  errors, one with errors, an empty log, a missing log), and run the step's CPU path end to end with the sanitizer
  call replaced by a plain run, so every row's child starts, refuses cleanly on the CPU lane and is recorded.

## Gates

- **G1:** `DIAGNOSIS.md` written (A1).
- **G2:** the CPU drive (A4) passes: every constructed case gives its expected verdict, and every row's child is recorded.
- **G3:** the canary line and the P2 lines on B, F2 and F4 are staged in `artifacts/bwdf1/f5/GPU_LINES_F5.json` and each
  dry-runs once with `FORGE_GPU_DRY_RUN=1` (exit 78). Each step script and each child starts with
  `forge-gpu-lease-check --require-held` (L125). Each line checks its modules' sha256 before the lease.
- **G4, scope:** writes only in the writable set; no product or test change; no GPU use; no git writes; nothing deleted.

**Registered acceptance:** G1 to G4 pass. The round's GPU evidence comes from the lead's run of the staged lines.

## Mechanism & scope

- **Measures:** whether a repaired harness can produce BWD-F3's P2 evidence on the final build.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_bwdf1/scripts/bwdf3_step_p2.py`
  (the failing harness) and `tensor_fork_apa_wt_bwdf1/artifacts/bwdf1/f3/REPORT_F3.md` (P2's intent).
- **Exercises:** the P2 step and verdict code, copied and repaired; CPU drives; dry runs.
- **Does not cover:** product code; BWD-F3's other lines (they exited 0); the G4 suite lines; promotion.
- **Operand census:** every P2 row in `bwdf3_p2_cases.py`; every way a child can end (exit, signal, timeout, sanitizer
  refusal).
- **Baseline isolation:** Build B for Build B's behaviour; Build F2 for "defect present"; Build F4 for the fix.
- **Required evidence:** `DIAGNOSIS.md`; the drive results; `GPU_LINES_F5.json` with dry-run exits; `COPY_DEPS.json`
  listing each copied script and what changed.
- **Forbidden configuration:** edits outside the writable set; running the sanitizer or any GPU work; touching
  `tensor_fork_apa_wt_bwdf1`; git writes; deleting anything yourself.

## Machine rules

- At most one CPU-busy process of yours at a time, at `nice -n 19`; every command under an explicit `timeout`.
- No GPU of any kind: never run `compute-sanitizer`, never call a step script outside its dry run, never remove or
  override `CUDA_VISIBLE_DEVICES` in a child of yours. A smoke of a staged line is its `FORGE_GPU_DRY_RUN=1` dry run.
- Stage lines in BWD-F3's form (`--min` 240, `--max 540`, inner `timeout 500s`).
- **Ceiling:** 2 hours of wall time.

## Registration checklist (order-lint)

- **L01:** python3, pytest, gcc (if the counter shim is rebuilt) and `forge-gpu-run` (`--status --min 30 --max 540`),
  each run first and recorded. `compute-sanitizer --version` only.
- **L02:** every staged line carries an explicit `--min`; each lane's smoke is its dry run.
- **L03, L07, L09, L10, L14, L15, L18, L19, L25, L32, L37, L41, L47, L53, L55, L60, L77, L104, L111, L114, L117:** n/a
  (a harness repair: no share, nothing visual, no comparison or identity gate run by the seat, no gain gate, no fitted
  threshold, no owner requirement behind an option, no bounded buffer, no knob, no mutant, nothing for another project,
  no new refusal, no ratio bar, no storage design, no job over 1 GB, no gradient).
- **L08:** receipts under `artifacts/bwdf1/f5/`, scratch under `/tmp/bwdf5_*`.
- **L16:** the cheapest passing repair would stage the same wrapper again. Excluded by A2 (per-row sanitizer) and the
  canary line, which shows the lead the outcome on one row first.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`.
- **L34:** a Claude seat (Sonnet 5.5, high effort, Agent tool).
- **L35:** if a clause cannot hold as written, report RED for that clause and finish the rest.
- **L40:** n/a: no pytest run of the suite.
- **L45:** each staged line records `forge-gpu-run`'s exit, the inner exit and each row's memcheck log sha256; 124 is
  never a pass.
- **L48:** staged lines run as written from the worktree root through the real `forge-gpu-run`; A4 drives the
  post-device code on constructed results, empty and one-element shapes included.
- **L56:** (a) tool slips go under Deviations: re-run and continue. (b) At most 3 repairs of each of your scripts.
- **L58:** every CPU command runs with `CUDA_VISIBLE_DEVICES` empty; no child of yours removes or overrides it.
- **L64:** reads are this worktree, `tensor_fork_apa_wt_bwdf1`'s scripts, logs, receipts and module directories (read
  only), and ENG3's counter source.
- **L65:** leave-and-list form for any file you would remove.
- **L67:** receipts are small (logs and JSON); no build.
- **L69:** one reference per class: B's behaviour -> Build B; the defect -> Build F2; the fix -> Build F4.
- **L74:** plain wording.
- **L76:** the lead writes nothing in the writable set while the seat runs.
- **L79:** `FORGE_GPU_DRY_RUN=1` returns 78 without a device check; once per staged line.
- **L81:** the failure is the lead's run (source records: the runner log's exit 255 per line, the rows' CHILD-DIED with
  empty stderr, the two-line memcheck logs); nothing else is assumed.
- **L82:** `grep -n "subprocess\|--child\|target-processes\|memcheck" tensor_fork_apa_wt_bwdf1/scripts/bwdf3_*.py`
  starts the census.
- **L89:** n/a. No guard keyed on a name.
- **L90:** roles: harness only. Product and tests are read-only.
- **L99:** the lead read `bwdf3_step_p2.py:1-12, 84-118`, the runner log and `p2_e1_F3`'s output and memcheck log before
  sealing.
- **L102:** no generated file or registry depends on the writable files.
- **L103:** each staged line refuses, before the lease, a module whose sha256 differs from its receipt.
- **L109:** helpers run as `python3 -B -P`; none shares a standard-library module name.
- **L118:** the consumers of the repaired harness are the P2 rows only.
- **L119:** every write-once path of a staged line is listed so a retry can move it.
- **L125:** the step and every child start with `forge-gpu-lease-check --require-held`; cite the file:line of each.
- **L127:** a Claude seat: your final message is the report, and the lead files it as `artifacts/bwdf1/f5/REPORT_F5.md`.
- **L128:** n/a. No arm-swapping runner is used: each arm's module is loaded from its own package directory.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes (read-only git is allowed). NO subagents. No background waits; each command under an explicit timeout.
- No deletes, in the leave-and-list form. Never kill a process you did not start.
- **Prior Art Directive:** annotate each code site and the report. Known: compute-sanitizer's `--target-processes` and
  `--log-file` options (NVIDIA's documentation); one process per test row (pytest-forked style). Write "no prior art
  known to me" where none applies.

## Done

Your final message contains, verbatim:
- G1 to G4, each PASS or FAIL;
- the diagnosis, in a few lines;
- each staged line with its dry-run exit, the canary first;
- the deviations;
- every file written.
