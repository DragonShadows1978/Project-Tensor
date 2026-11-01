# GN1 amendment A6: fix every GN1-REV5 finding under a stated threat model (pinned programs, pinned environment, robust scorer)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_gn1`
(branch `gn1`, at the commit carrying this amendment). Edits and CPU runs there are AUTHORIZED, limited to:
- `artifacts/gn1/a6/`: the new tooling copy, arms, receipts, ledger and report;
- `logs/`, and scratch under `/tmp/gn1_a6_*`. Point `TMPDIR` at it for every command.

**Read-only:**
- the product (`nn.py` at `cf922891`'s bytes) and the test file;
- `artifacts/gn1/` outside `a6/`: every earlier round and review, all frozen;
- the pinned module and `tensor_cuda/build-gn1/`, with no rebuild;
- every other worktree, and production.

Read-only git is allowed. You may run the installed `forge-gpu-run` only on CPU children, with its summary, window and
lock redirected to scratch as REV2 to REV5 did. Never touch the real lock or window, and leave
`/mnt/ForgeRealm/grapa_run7` strictly alone.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.
Search inside this worktree, by path.

## Why

GN1-REV5 (`artifacts/gn1/rev5/REPORT.md` and `findings.json`, commit `eca1f2f9`) is a Claude cross-model re-review.
It found A5 (`24ddbff3`) fixed R4-F01 to R4-F04, and confirmed the kill-grace ordering and every earlier sound area.
It rejected A5 on these findings:
- **R5-F01 (MINOR):** the scorer never checks which runner ran. A wrapper at `command[0]` can apply `--max 900`.
- **R5-F02 (MINOR):** PATH and the `FORGE_GPU_*` environment can change the applied bound, or the window and lock the
  runner honours, and the scorer cannot see it. Example: a `timeout` on PATH that shadows the real one.
- **R5-F03 (NIT):** the scorer crashes on deeply nested JSON (`RecursionError`).
- **R5-F04 (NIT):** `raw_exit` normalisation is not recomputed by the scorer.
- **R5-F05 (NIT):** after the child exits, the seal hash and the temporary write are unprotected.

The findings per round have gone 11, 11, 6, 4, 5. Each round has widened the adversary the tooling is held against.
This round states the threat model explicitly (D0), fixes every finding within it, and asks the next review to judge
against it.

## Design (binding; the lead chose these)

- **D0, the threat model.** These are what the GPU verdict must defend against.
  - **In scope:**
    - **(a) receipts:** any edit, truncation, forgery or malformation of any receipt or log a line writes, by any
      process after the fact;
    - **(b) staged lines:** any change to a staged line's argv, in any spelling;
    - **(c) programs and environment:** a substitute or wrapper at the runner, at `timeout`, at the Python interpreter,
      or at any program a line resolves through PATH, and any `FORGE_GPU_*` or PATH value that differs from the staged
      one;
    - **(d) interruptions:** every interruption path the installed runner and `timeout` can produce.
  - **Out of scope:**
    - a coordinated rewrite of every receipt and log together with the seal, which no unsigned scheme can refuse;
    - root-level tampering with the installed runner's file between the seal and the run, beyond the sha256 check D1
      adds;
    - kernel-level process hiding;
    - a GPU or driver defect.
  - **Reviews:** report anything outside this model as an observation, not a finding.
- **D1, pinned programs (R5-F01, R5-F02).**
  - **The staged script** sets `PATH` to a fixed value and calls `forge-gpu-run`, `timeout` and `python3` by absolute
    path. It unsets `FORGE_GPU_SUMMARY`, `FORGE_GPU_WINDOW_FILE` and `FORGE_GPU_LOCK` for every non-audit line, and
    sets `FORGE_GPU_KILL_GRACE` and the poll interval as A5 does.
  - **The seal** records the absolute path and sha256 of each program: the installed runner (`ad6c0a22…e107`),
    `timeout` and the interpreter.
  - **The parent records, for each line:**
    - the resolved path and sha256 of the runner, of `timeout` and of the interpreter that actually ran;
    - the environment the child received, restricted to `PATH` and every `FORGE_GPU_*` variable.
  - **The scorer** accepts a line only when `command[0]` is the pinned runner path, and every recorded program and
    environment value equals the sealed or staged value.
  - **CPU controls** run in a declared control mode, recorded in the receipt, that can never establish acceptance, just
    as audit mode cannot. That keeps stand-ins usable for controls.
- **D2, the scorer never crashes (R5-F03).** Both of the scorer's read sites, and every place the scorer and the
  parent parse JSON, catch `Exception`, `RecursionError` included, and record a reason.
- **D3, exit normalisation recomputed (R5-F04).** The scorer requires an integer `raw_exit`, with `exit == raw if raw
  >= 0 else 128 - raw`. Otherwise it records a reason.
- **D4, the post-exit receipt (R5-F05).** `seal_sha256` is computed inside the protected block, with a recorded reason
  if that fails. If writing the receipt fails, the parent retries once with a minimal error receipt. If that also
  fails, the scorer's "incomplete" handling is the documented fallback.

## Gates

- **G1, scope.**
  - `nn.py` and the test file are byte-identical to `cf922891`.
  - Every earlier round and review folder is byte-identical before and after.
  - The `a5` to `a6` tooling diff holds only D1 to D4 hunks, each mapped.
- **G2, CPU checks.** The earlier reference checks pass. Both arms import and collect with the GPU hidden, with the
  same IDs as A5.
- **G3, the module.** The pin and source map are unchanged, and every arm's module copy matches the pin.
- **G4, hostile controls.** Each control runs through `a6/`'s real parent and scorer. Each must meet its target, and its
  deciding rule is named. The set:
  - A5's 283 targets;
  - REV5's reproductions: `Q2_wrapped_runner_command0`, `Q2_path_shadow_inner_timeout`, the three deep-nesting cases,
    the `raw_exit` edits, and the post-exit seal failure;
  - one control per D0(c) item: a substitute or wrapper for the runner, `timeout` and the interpreter, a PATH shadow,
    and each `FORGE_GPU_*` override;
  - a CPU control-mode root, which must never be accepted.
- **G5, mutants on CPU.** Unchanged from A5, re-verified against the new tooling.
- **G6, staging.** The staged script, the line expectations, the arm regeneration verified against the seal, and one
  audit-mode line. An honest run of every label passes the scorer with the real runner, using control-mode stand-ins
  only where the order allows them.

**Registered acceptance:** ACCEPT when G1 to G6 pass. The lead's GPU run follows a clean GN1-REV6, judged against D0.

## Lead predictions (registered here, before any data)

- **P117:** every G4 control meets its target, REV5's reproductions and every D0(c) substitution included.
- **P118:** an honest run of every staged line with the real installed runner passes the scorer under D1's pins.

## Mechanism & scope

- **Measures:** whether GN1's GPU verdict reports only what the pinned programs, run under the staged environment,
  actually did, within D0's threat model.
- **Mechanism doc:**
  - `artifacts/gn1/rev5/REPORT.md`;
  - `orders/GN1_A5_REV4_FIXES.md`;
  - `artifacts/gn1/a5/REPORT.md`.
- **Exercises:** `a6/`'s parent, child and scorer under hostile inputs and substituted programs, with the real runner
  redirected, plus the regeneration.
- **Does not cover:**
  - D0's out-of-scope items;
  - GPU execution;
  - the product;
  - performance.
- **Required evidence:**
  - the `a5` to `a6` diff with its hunk map;
  - the sha256 maps;
  - a per-finding fix table;
  - the G4 table;
  - the program pins;
  - the staged script.
- **Operand census:**
  - every REV5 finding and reproduction;
  - every program a line resolves;
  - every `FORGE_GPU_*` variable the runner reads, listed from its source;
  - PATH.
- **Baseline isolation:** the product is frozen at `cf922891`. Earlier receipts are frozen.
- **Forbidden configuration:**
  - any edit outside the writable list;
  - any product or test change;
  - symlinks in arms;
  - a rebuild;
  - touching the real GPU lock or window;
  - GPU use;
  - git writes;
  - deleting anything yourself.

## Budget (L42)

- **Units:** A5 took 2,583 s.
- **Ceiling:** 2 h. Every single command stays under 20 minutes.
- If a gate cannot finish inside the ceiling, stop and report `RED-BUDGET` with what was done.

## Registration checklist (order-lint)

- **L01:** the instruments all exist:
  - A5's tooling;
  - REV5's scripts;
  - the installed forge-gpu-run;
  - `/usr/bin/timeout`;
  - the system `python3`;
  - the pinned module;
  - NumPy 2.4.2;
  - pytest 9.0.2.
- **L02:** the smoke is G2 and G4 on CPU. The lead's GPU run is the first device execution.
- **L03:** n/a. The mutants stand in for a coverage bar.
- **L04:** n/a. There is no builder with fallback rungs. The "fallback" in D4 is the scorer's existing handling of an
  incomplete receipt.
- **L05:** the owner is the GPU tooling under `artifacts/gn1/`.
- **L06:** n/a. No size threshold.
- **L07:** n/a. Nothing visual.
- **L08:** the staged lines and the scorer run from `a6/`'s own directory.
- **L09:**
  - commands are compared as exact argv lists;
  - programs by absolute path and sha256;
  - environment values as exact strings;
  - exits as integers after normalisation.
- **L12:** the rules this round changes are program and environment binding, scorer robustness, exit recomputation
  and post-exit receipts. G4 re-runs every earlier control.
- **L14:** n/a. No gain gate.
- **L15:** P117 and P118 are predictions, scored in the report.
- **L16:** the cheapest passing work special-cases REV5's two substitutions. G4's control for every D0(c) item
  excludes it.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`. Name the runner
  generically.
- **L25:** threads at 1.
- **L26:** the lead touches nothing in this worktree while the seat runs.
- **L33:** every staged GPU line runs through the pinned `forge-gpu-run` with `--max` and an inner timeout. This round
  runs the installed runner only on CPU children, redirected to scratch.
- **L34:** a Codex seat (gpt-6.1-sol, xhigh), on CPU. GN1-REV6 is a Claude seat.
- **L38:** n/a. No test changes.
- **L40:** outputs go under `artifacts/gn1/a6/` or `/tmp/gn1_a6_*`.
- **L41:** the mutants are unchanged from A5, with witness reach as REV5 verified (54/54).
- **L42:** see Budget.
- **L44:** the scorer recomputes every verdict, cause, exit normalisation, bound check and pin check from the raw
  receipts and the runner's log.
- **L45:** each line counts only with the parent's record of the child's exit, written after the child exits, together
  with the runner's own outcome, the child's start marker, and the resolved programs.
- **L47:** the findings and reproductions are REV5's receipts under `rev5/`. The runner's pin `ad6c0a22…e107` is
  REV5's reading of the installed runner. Requirements on new code are marked as requirements.
- **L48:** G4 runs on CPU before any GPU line.
- **L55:** exact equalities: argv lists, absolute paths, sha256 digests, environment strings and exit integers.
- **L58:** every CPU run sets `CUDA_VISIBLE_DEVICES` empty.
- **L59:** the seal is this order's commit. `START.json` is written by the GPU run's parent before its first line.
- **L64:** reads: the worktree, earlier receipts, and the installed runner's and coreutils' sources and help. The
  read-only list permits them.
- **L65:** leave-and-list form used (Rules).
- **L67:** the GPU lines write a few MB each. The lead's runner stops below 25 GB free on `/mnt/ForgeRealm` and 20 GB
  on `/`.
- **L69:** one reference governs each class:
  - GroupNorm values: the float64 reference;
  - tracked files: `cf922891`;
  - the module: round 1's pin;
  - line commands and bounds: `gpu_lines.json`;
  - programs and environment: the seal.
- **L70:** n/a. No numeric bound between device computations.
- **L71:** n/a. No input evidence is derived.
- **L72:** the round runs its own tooling copy under `a6/`. Its write targets are under `a6/`, `/tmp/gn1_a6_*` and the
  redirected runner scratch. The registered invocations are the gate scripts, the controls and the audit-mode line,
  each recorded at launch.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes. NO subagents. No background waits. Run everything synchronously, each command under an explicit
  timeout.
- **No deletes by you,** in the leave-and-list form. Anything that must go stays where it is and is listed under
  `## Done` for the lead to bin. Scratch goes under `/tmp` (exempt). Never set `DELETE_ME_ROOTS`. The code under test
  may manage its own files inside the writable set.
- Never kill a process you did not start. GPU never.
- **Read-only inspection** is always allowed. How long your harness waits between polls is not governed by this order.
- A failed gate is a result. Report it with receipts.
- Prior Art Directive (AGENTS.md): as A5. Add POSIX `execvp` PATH lookup and Linux procfs (`/proc/<pid>/exe`,
  `cmdline`, `environ`).
- **Write early:** create `artifacts/gn1/a6/REPORT.md` and `artifacts/gn1/a6/LEDGER.md` within your first few steps.

## Done

Your final message contains, verbatim:
- G1 to G6, each PASS or FAIL, with receipts;
- the per-finding fix table;
- the tooling diff summary;
- the program pins;
- the G4 table;
- P117 and P118, each HELD or NOT HELD;
- every file written, and anything left for the lead to bin.
