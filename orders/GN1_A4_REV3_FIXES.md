# GN1 amendment A4: fix every GN1-REV3 finding (start evidence, timeout causes, atomic results, runtime seal, equivalences, bound binding)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_gn1`
(branch `gn1`, at the commit carrying this amendment). Edits and CPU runs there are AUTHORIZED, limited to:
- `artifacts/gn1/a4/`: the new tooling copy, arms, mutants, receipts, ledger and report;
- `logs/`, and scratch under `/tmp/gn1_a4_*`. Point `TMPDIR` at it for every command.

**Read-only:**
- the product (`nn.py` stays at `cf922891`'s bytes) and the test file;
- `artifacts/gn1/` outside `a4/`: round 1, A1, A2, A3 and REV1 to REV3, all frozen;
- the pinned module and `tensor_cuda/build-gn1/`, with no rebuild;
- every other worktree, and production.

Read-only git is allowed. You may run the installed `forge-gpu-run` only on CPU children, with its summary, window and
lock redirected to scratch as REV2 and REV3 did. Never touch the real lock or window.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.
Search inside this worktree, by path.

## Why

GN1-REV3 (`artifacts/gn1/rev3/REPORT.md` and `findings.json`, commit `a5f0676a`) is a Claude cross-model re-review.
It rejected A3 (`cf922891`, receipts `a3/`) on 3 MINOR and 3 NIT findings, R3-F01 to R3-F06. The kill rule and D5
validation are sound. Each finding has a reproduction under `rev3/`, through REV3's `parent_wrap.py` mirror.

REV3 also noted one item outside its scope: the scorer does not tie each line's recorded bounds to `gpu_lines.json`.
The lead rules that it is fixed here too (D7).

House rule: every finding is fixed, NITs included, and the fix is reviewed again.

## Design (binding; the lead chose these)

- **D1, the tooling copy.** Copy `a3/`'s tooling to `a4/`, and make every change there. `a3/` stays frozen. The
  product and the test file do not change in this round.
- **D2, start evidence (R3-F01, R3-F03).**
  - **The marker:** `run_gpu_tests.py` writes a start marker as its first action, before it imports the engine,
    pytest or NumPy. The write is atomic: a temporary file, then `os.replace`.
  - **Not admitted:** an attempt is not admitted only when there is no start evidence of any kind:
    - no marker and no `invocation.json`;
    - no runner row of `ran`, `timeout` or `window-closed`;
    - no `junit.xml`.
  - **Otherwise** it is admitted and evaluated. If it lacks its result, it fails, so a started attempt is never
    retried until it passes.
  - An outer timeout during the runner's lock wait, with no row and no marker, is therefore not admitted, and its
    label may be retried.
- **D3, timeout causes (R3-F02).**
  - A row `timeout` or `window-closed` logged with 143 or 137, where the recorded exit is 124 or 137, means the
    outer bound fired during the runner's kill grace: its cause is `outer-timeout`.
  - `evaluate()` never raises. Every problem it finds becomes a recorded reason, and the line fails closed.
  - The scorer recomputes the cause.
- **D4, atomic results (R3-F04).** The child writes `child_result.json` atomically. `evaluate()` reads it inside its
  `try`. A truncated or unreadable result is a recorded reason, not a crash.
- **D5, the runtime seal (R3-F05).**
  - **Recording:** after `pytest.main`, the child records the origin path and sha256 of every module in
    `sys.modules` that has a file.
  - **Sealing:** extend the seal with the modules a CPU stand-in execution of each suite imports at run time, in
    addition to the collection trace.
  - **At GPU time:**
    - any recorded module that is sealed but whose bytes differ from the seal fails the line (tamper);
    - a module the seal never saw (a path only the real device reaches) is listed in the receipt as `unsealed`, does
      not by itself fail the line, and is reported by the scorer for the lead.
  - **The rule in writing:** state this in `SEAL_IMPORT_TRACE.md`.
- **D6, equivalence records (R3-F06).**
  - Record both `numpy.bool_`-clause deletions (`nn.py:518` and `:526`) as sealed equivalences. Each carries the
    `operator.index` proof and its NumPy-version dependency: the version, plus a live check of
    `operator.index(numpy.bool_(True))` in the arm's interpreter at run time.
  - Stage both, with an expected pass, as V3 is. If the live check finds that `operator.index` accepts
    `numpy.bool_`, the equivalence no longer holds and the line fails.
  - Keep the negation mutants as reach checks.
- **D7, bound binding (REV3's out-of-scope note).** The scorer checks every line against its row in `gpu_lines.json`:
  - the parent's recorded outer bound;
  - the runner's `--max` and `--min`;
  - the inner timeout;
  - the expected exits.

  A mismatch fails that line.

## Gates

- **G1, scope.**
  - `nn.py` and the test file are byte-identical to `cf922891`.
  - `a3/`, `rev1/`, `rev2/` and `rev3/` are byte-identical before and after.
  - The `a3` to `a4` tooling diff holds only D2 to D7 hunks, each mapped.
- **G2, CPU checks.** All earlier reference checks pass. Both arms import and collect with the GPU hidden, and the IDs
  equal A3's.
- **G3, the module.** The pin and source map are unchanged, and every arm's module copy matches the pin.
- **G4, hostile controls.** Each control is run through `a4/`'s real parent and scorer, with the installed runner
  redirected to scratch where REV3 used it. Each must meet its target, and the deciding rule is named.
  - **Inherited:** A3's 107 targets.
  - **REV3's reproductions:**
    - B6: lock held past the outer bound, so not admitted, and the retry is admitted;
    - B7, B7b and B8: cause `outer-timeout`, with no parent crash;
    - C7b: a started child with no result is admitted and fails, and is never retried to a pass;
    - E5: a truncated result is a recorded reason.
  - **One control per D5 case:** a sealed module whose bytes are tampered fails; an unsealed module is listed but
    passes.
  - **D6:** the live `operator.index` check.
  - **One control per D7 field:** each mismatch fails.
- **G5, mutants on CPU.** The mutant list now includes the two equivalence-recorded deletions. Every mutant is
  single-site, with a call-phase trace. The deletions pass, as their equivalence records require.
- **G6, staging.** The staged script, every line's expected exit, the arm regeneration verified against the seal, and
  one audit-mode line. An honest run of every label passes the scorer, shown with stand-in receipts.

**Registered acceptance:** ACCEPT when G1 to G6 pass. The lead's GPU run follows a clean GN1-REV4.

## Lead predictions (registered here, before any data)

- **P108:** every G4 control meets its target, REV3's reproductions included.
- **P109:** the CPU stand-in's runtime census adds at least `numpy.random` and `numpy.testing` modules to the seal, and
  no sealed module differs at the end of the round.

## Mechanism & scope

- **Measures:** whether GN1's GPU verdict reports only real kills, real regressions and real runner outcomes, under
  every interruption the runner can produce, with the code each line ran sealed at run time.
- **Mechanism doc:**
  - `artifacts/gn1/rev3/REPORT.md`;
  - `orders/GN1_A3_REV2_FIXES.md`;
  - `artifacts/gn1/a3/REPORT.md`.
- **Exercises:**
  - `a4/`'s parent, child and scorer under hostile inputs, with the real installed runner redirected;
  - the runtime import census;
  - the equivalence lines;
  - the arm regeneration.
- **Does not cover:**
  - GPU execution;
  - the product, which is unchanged;
  - other layers;
  - performance.
- **Required evidence:**
  - the `a3` to `a4` diff with its hunk map;
  - the sha256 maps;
  - a per-finding fix table, R3-F01 to R3-F06 and D7;
  - the G4 table;
  - the runtime seal census;
  - the equivalence records;
  - the staged script.
- **Operand census:**
  - every REV3 finding and reproduction;
  - every runner outcome, including interrupts during the lock wait and during the kill grace;
  - every runtime module;
  - every `gpu_lines.json` field.
- **Baseline isolation:** the product is frozen at `cf922891`. The earlier receipts are frozen.
- **Forbidden configuration:**
  - any edit outside the writable list;
  - any product or test change;
  - symlinks in arms;
  - a rebuild;
  - widening a tolerance;
  - touching the real GPU lock or window;
  - GPU use;
  - git writes;
  - deleting anything.

## Budget (L42)

- **Units:** A3 took 3,073 s.
- **Ceiling:** 2 h. Every single command stays under 20 minutes.
- If a gate cannot finish inside the ceiling, stop and report `RED-BUDGET` with what was done.

## Registration checklist (order-lint)

- **L01:** the instruments all exist:
  - A3's tooling;
  - REV3's scripts and mirror method;
  - the installed forge-gpu-run;
  - the pinned module;
  - NumPy 2.4.2;
  - pytest 9.0.2.
- **L02:** the smoke is G2 and G4 on CPU. The lead's GPU run is the first device execution.
- **L03:** n/a. The mutants stand in for a coverage bar.
- **L05:** the owners:
  - the GPU tooling under `artifacts/gn1/` (F01 to F05, D7);
  - the mutant registry (F06).
- **L06:** n/a. No size threshold.
- **L07:** n/a. Nothing visual.
- **L08:** the staged lines and the scorer run from `a4/`'s own directory.
- **L09:**
  - JUnit outcomes are compared ID by ID;
  - files and modules are compared by sha256;
  - exits are integers;
  - runner outcomes are exact tokens;
  - bounds are integers compared with `gpu_lines.json`.
- **L12:** the rules this round changes:
  - admission;
  - timeout attribution;
  - result reading;
  - the seal's scope;
  - the equivalence registry;
  - bound binding.

  G4 re-runs every earlier control.
- **L14:** n/a. No gain gate.
- **L15:** P108 and P109 are predictions, scored in the report.
- **L16:** the cheapest passing work special-cases REV3's exact inputs. G4's control per runner outcome, per D5 case
  and per D7 field excludes it.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`. Name the runner
  generically.
- **L25:** threads at 1.
- **L26:** the lead touches nothing in this worktree while the seat runs.
- **L33:** every staged GPU line runs through `forge-gpu-run` with `--max` and an inner timeout. This round runs the
  installed runner only on CPU children, redirected to scratch.
- **L34:** a Codex seat (gpt-6.1-sol, xhigh), on CPU. GN1-REV4 is a Claude seat.
- **L38:** n/a. No test changes.
- **L40:** outputs go under `artifacts/gn1/a4/` or `/tmp/gn1_a4_*`.
- **L41:** every mutant's witness reaches its site in the call phase. The equivalence lines carry a live proof.
- **L42:** see Budget.
- **L44:** the scorer recomputes every verdict, timeout cause and bound check from the raw receipts and the runner's
  log.
- **L45:** each line counts only with the parent's record of the child's exit, written after the child exits,
  together with the runner's own outcome and the child's start marker.
- **L47:** the findings and reproductions are REV3's receipts under `rev3/`. The runner's lines (lock wait 255-278,
  grace 183-186 and 205-218, row 302) are re-read in D2 and D3. Requirements on new code are marked as requirements.
- **L48:** G4 runs on CPU before any GPU line.
- **L55:** exact equalities: test IDs, exit integers, runner tokens, bound integers and sha256 digests.
- **L58:** every CPU run sets `CUDA_VISIBLE_DEVICES` empty.
- **L59:** the seal is this order's commit. `START.json` is written by the GPU run's parent before its first line.
- **L64:** reads: the worktree, the earlier receipts, the installed runner's source, and NumPy. The read-only list
  permits them.
- **L65:** leave-and-list form used (Rules).
- **L67:** the GPU lines write a few MB each. The lead's runner stops below 25 GB free on `/mnt/ForgeRealm` and 20 GB
  on `/`.
- **L69:** one reference governs each class:
  - GroupNorm values: the float64 reference;
  - tracked files: `cf922891`;
  - the module: round 1's pin;
  - line bounds: `gpu_lines.json`.
- **L70:** n/a. No bound in this round separates rounding from a defect between two device computations. The
  GroupNorm tolerances compare device output with a float64 NumPy reference, are unchanged since A1, and were
  derived in A2 (D8). This round changes only tooling.
- **L71:** n/a. No input evidence is derived. The runtime seal records imported modules; caches are excluded by
  construction, since only module origins are recorded.
- **L72:** the round runs its own copied tooling under `a4/`. Every write target is under `a4/`, `/tmp/gn1_a4_*` or
  the redirected runner scratch. The registered invocations are the gate scripts, the controls and the audit-mode
  line, each recorded at launch.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes. NO subagents. No background waits. Run everything synchronously, each command under an explicit
  timeout.
- **No deletes,** in the leave-and-list form. Anything that must go stays where it is and is listed under `## Done`
  for the lead to bin. Scratch goes under `/tmp` (exempt). Never set `DELETE_ME_ROOTS`.
- Never kill a process you did not start. GPU never.
- A failed gate is a result. Report it with receipts.
- **Read-only inspection** (reading, listing, hashing, parsing, read-only git) is always allowed. How long your
  harness waits between polls is not governed by this order.
- Prior Art Directive (AGENTS.md): as A3. Add atomic file replacement (POSIX `rename(2)`) and the `sys.modules`
  origin census.
- **Write early:** create `artifacts/gn1/a4/REPORT.md` and `artifacts/gn1/a4/LEDGER.md` within your first few steps.

## Done

Your final message contains, verbatim:
- G1 to G6, each PASS or FAIL, with receipts;
- the per-finding fix table;
- the tooling diff summary;
- the G4 table;
- the runtime seal census;
- the equivalence records;
- P108 and P109, each HELD or NOT HELD;
- every file written, and anything left for the lead to bin.
