# GN1 amendment A3: fix every GN1-REV2 finding (runner outcomes, kill rule, validation, mutants, seal)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_gn1`
(branch `gn1`, at the commit carrying this amendment). Edits and CPU runs there are AUTHORIZED, limited to:
- `tensor_cuda/tensor_cuda/nn.py`, the `GroupNorm` class only;
- `tensor_cuda/tests/test_groupnorm_gn1.py`;
- `artifacts/gn1/a3/`: the new tooling copy, arms, mutants, receipts, ledger and report;
- `logs/`, and scratch under `/tmp/gn1_a3_*`. Point `TMPDIR` at it for every command.

**Read-only:**
- `artifacts/gn1/` outside `a3/`: round 1, A1, A2, REV1 and REV2, all frozen;
- the pinned module and `tensor_cuda/build-gn1/`, with no rebuild;
- every other file in this worktree, every other worktree, and production.

Read-only git is allowed. You may run the installed `forge-gpu-run` only as REV2 did, redirected to scratch with a CPU
child (`rev2/rev2_forge.py`). It must never touch a GPU or the real window and lock files.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.
Search inside this worktree, by path.

## Why

GN1-REV2 (`artifacts/gn1/rev2/REPORT.md` and `findings.json`, commit `ed3ac28d`) is a Claude cross-model re-review.
It rejected A2 (`bd30498a`, receipts `a2/`) on 4 MINOR and 7 NIT findings, R2-F01 to R2-F11. REV1's two MAJORs are
closed. R1-F08 and R1-F11 are not fixed (R2-F01, R2-F04). Each finding has a reproduction under `rev2/`. House rule:
every finding is fixed, NITs included, and the fix is reviewed again.

## Design (binding; the lead chose these)

- **D1, the tooling copy.** Copy `a2/`'s tooling (the scripts, the stand-in, the arm and mutant builders, the seal) to
  `a3/`. Make every change there. `a2/` stays frozen.
- **D2, the lead's ruling on R2-F02: option (a).** A mutant's engine refusal of a registered valid input is a kill
  when all of these hold:
  - the input is valid, and the same run root's candidate focus line has passed it;
  - the error text carries no device pattern;
  - the refusal happens in a GroupNorm call phase.

  The test wrapper keeps re-raising an engine refusal as an AssertionError. That AssertionError records the original
  exception class and message, and the evidence reads both. Replace REV1's two synthetic G4 controls
  (`Generic_RuntimeError_not_kill`, `Unregistered_TypeError_not_kill`) with controls that run the real test file
  through the real plugin on the CPU stand-in engine:
  - an engine RuntimeError with a device pattern must be vetoed;
  - one without a device pattern, on an input the candidate passed, counts as a kill;
  - one on an input the candidate did not pass is refused.

  Document the rule in D2's text, in the evidence code, and in the report.
- **D3, refusal origin (R2-F03).** A "GroupNorm refusal" is classified by the raising frame
  (`excinfo.traceback[-1]`), which must be GroupNorm's own `__init__` or `forward`. Add a control in which a callee
  raises inside `__init__`.
- **D4, runner outcomes (R2-F01, R2-F04, R2-F07).**
  - Recognise every outcome the installed forge-gpu-run can log. List them from its source, with line numbers.
  - **Not admitted:** with no `child_result.json`, these mean not admitted:
    - exit 75 with `no-window` or `rechecked-out`;
    - exit 76 with `no-lock`.
  - **Cause of exit 124 or 137:** record the cause as one of `outer-timeout`, `runner-deadline`, `window-closed` or
    `inner-timeout`, by REV2's rule. The scorer recomputes the cause.
  - **Interrupted attempts:** an attempt directory without `parent_result.json` is incomplete, not fatal.
    - If its `child_result.json` exists, evaluate that attempt, so a failure is never retried until it passes.
    - If the child never started, mark it not admitted and allow a retry.
  - Add a G4 control for every runner outcome. Drive the real installed runner the way REV2 did.
- **D5, validation (R2-F05, R2-F06).**
  - **Counts:** accept any value `operator.index` accepts, except `bool` and `numpy.bool_`, and store it as `int`.
  - **Parameters:** refuse with `ValueError` a `weight` or `bias` whose `ndim != 1` or whose `shape[0] != C`. Do it in
    `forward`, before any tensor op.
  - **Tests:** NumPy integer counts are accepted, `numpy.bool_` is refused, and each wrong parameter shape is refused.
  - **Mutants:** one per new decision.
  - **PyTorch:** where its behaviour can be observed, record it.
- **D6, mutant evidence (R2-F08, R2-F09).**
  - Witness traces cover the test's call phase only.
  - M14 is recorded as reached through default-argument binding.
  - Register V4, the fast-path condition `dtype == "float32"` at `nn.py:562` (or its new line), and stage its GPU line.
- **D7, guards and seal (R2-F10, R2-F11).**
  - The TMPDIR guard compares resolved paths.
  - The seal pins every module a sealed test imports, found by import tracing of the collected regression suite, not
    by a hand list. That includes `scripts/apa_om135_core.py`. The child verifies the pins.

## Gates

- **G1, scope.**
  - Relative to `0e1dea3`, the diff is `GroupNorm` plus the test file.
  - Relative to `bd30498a`, the `nn.py` diff is D5 only.
  - `a2/`, `rev1/` and `rev2/` are byte-identical before and after.
- **G2, CPU checks.** Every earlier reference check passes, and so does every new D5 case. Both arms import and
  collect with the GPU hidden. List the IDs.
- **G3, the module.** The pin and source map are unchanged, and every arm's module copy matches the pin.
- **G4, hostile controls.** Each is run through `a3/`'s real parent and scorer. Each must meet its target, and the
  deciding rule is named:
  - A2's 48 controls, with the two replaced by D2's real-test controls. List each adaptation;
  - REV2's cases B1 to B5, C7, P1 and P2, through the real installed runner where REV2 used it;
  - one control per runner outcome;
  - REV1's H1 to H6.
- **G5, mutants on CPU.** The mutant list, now with V4 and the D5 mutants. Each has a call-phase trace, or the
  recorded default-binding reach for M14. Each must load and differ.
- **G6, staging.**
  - The staged script and each line's expected exit.
  - The arm regeneration, verified against the seal.
  - One audit-mode line.
  - A valid run of every label passes the scorer, shown with stand-in receipts like REV2's.

**Registered acceptance:** ACCEPT when G1 to G6 pass. The lead's GPU run follows a clean GN1-REV3.

## Lead predictions (registered here, before any data)

- **P94:** every G4 control meets its target. That includes all of REV2's reproductions.
- **P95:** V4 and every D5 mutant are killed on the CPU stand-in by registered witnesses.

## Mechanism & scope

- **Measures:** whether GN1's GPU verdict reports only real kills, real regressions and real runner outcomes, and
  whether GroupNorm accepts exactly its domain.
- **Mechanism doc:**
  - `artifacts/gn1/rev2/REPORT.md`;
  - `orders/GN1_A2_REV1_FIXES.md`;
  - `artifacts/gn1/a2/REPORT.md`.
- **Exercises:**
  - the amended class and tests on CPU;
  - `a3/`'s parent, child and scorer under hostile inputs, including the real installed runner redirected to scratch;
  - the mutant traces;
  - the arm regeneration;
  - the import-traced seal.
- **Does not cover:**
  - GPU execution;
  - other layers;
  - the engine's float16 pow backward;
  - `reshape(-1)`;
  - performance.
- **Required evidence:**
  - the diffs against `0e1dea3` and `bd30498a`;
  - the sha256 maps;
  - a per-finding fix table, R2-F01 to R2-F11, each row with its receipt;
  - the G4 table;
  - the runner-outcome list with its source lines;
  - the mutant table;
  - the seal's import trace;
  - the staged script.
- **Operand census:**
  - every REV2 finding and reproduction;
  - every runner outcome;
  - every D5 decision;
  - every mutant;
  - every module imported by the sealed suites.
- **Baseline isolation:**
  - the baseline arm is `0e1dea3`'s package with the pinned module copy;
  - `a2/`, `rev1/` and `rev2/` are frozen.
- **Forbidden configuration:**
  - any edit outside the writable list;
  - symlinks in arms;
  - a rebuild;
  - widening a tolerance;
  - touching the real GPU window or lock;
  - GPU use;
  - git writes;
  - deleting anything.

## Budget (L42)

- **Units:** A2 took 2,775 s. REV2 needed about 30 minutes for its runner probes.
- **Ceiling:** 2 h. Every single command stays under 20 minutes.
- If a gate cannot finish inside the ceiling, stop and report `RED-BUDGET` with what was done.

## Registration checklist (order-lint)

- **L01:** the instruments all exist:
  - A2's tooling;
  - REV1's and REV2's scripts;
  - the installed forge-gpu-run;
  - the pinned module;
  - NumPy;
  - pytest 9.0.2;
  - PyTorch 2.11, which REV2 observed.
- **L02:** the smoke is G2 and G4 on CPU. The lead's GPU run is the first device execution.
- **L03:** n/a. The mutants stand in for a coverage bar.
- **L05:** the owners:
  - the GPU tooling under `artifacts/gn1/` (F01, F04, F07, F10, F11);
  - the tests and mutants (F02, F03, F08, F09);
  - the product's GroupNorm (F05, F06).
- **L06:** n/a. No size threshold.
- **L07:** n/a. Nothing visual.
- **L08:** the staged lines and the scorer run from `a3/`'s own directory.
- **L09:**
  - JUnit outcomes are compared ID by ID;
  - files are compared by sha256 maps;
  - exits are integers;
  - runner outcomes are exact log tokens.
- **L12:** the rules this round changes:
  - the kill rule (D2, a lead ruling);
  - refusal origin;
  - runner-outcome handling;
  - validation;
  - the seal's scope.

  G4 re-runs every earlier control.
- **L14:** n/a. No gain gate.
- **L15:** P94 and P95 are predictions, scored in the report.
- **L16:** the cheapest passing work special-cases REV2's exact inputs. G4's per-outcome controls and real-test
  controls, D7's traced seal, and the D5 mutants exclude it.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`. Name the runner
  generically, as REV2 did.
- **L25:** threads at 1.
- **L26:** the lead touches nothing in this worktree while the seat runs.
- **L33:** every staged GPU line runs through `forge-gpu-run` with `--max` and an inner timeout (D4, G6). This round
  runs the installed runner only on CPU children, with its summary, window and lock redirected to scratch as REV2 did.
  It never takes the real lock or window.
- **L34:** a Codex seat (gpt-6.1-sol, xhigh) on CPU. GN1-REV3 is a Claude seat.
- **L38:** the test file only.
- **L40:** outputs go under `artifacts/gn1/a3/` or `/tmp/gn1_a3_*`.
- **L41:** every mutant's witness reaches its site in the call phase (D6).
- **L42:** see Budget.
- **L44:** the scorer recomputes every verdict and every timeout cause from the raw receipts and the runner's log.
- **L45:** each line counts only with the parent's record of the child's exit, written after the child exits,
  together with the runner's own outcome.
- **L47:** the findings, reproductions and observed PyTorch behaviour are REV2's receipts under `rev2/`, and the runner
  lines REV2 cites (281, 285) are re-read in D4. Requirements on new code are marked as requirements.
- **L48:** G4 runs on CPU before any GPU line.
- **L55:** exact equalities: test IDs, dtype strings, exit integers, runner tokens and sha256 digests.
- **L58:** every CPU run sets `CUDA_VISIBLE_DEVICES` empty.
- **L59:** the seal is this order's commit. `START.json` is written by the GPU run's parent before its first line.
- **L64:** reads: the worktree, the earlier receipts, the installed runner's source, NumPy and PyTorch. The read-only
  list permits them.
- **L65:** leave-and-list form used (Rules).
- **L67:** module copies add about 10 MB per arm. The GPU lines write a few MB each. The lead's runner stops below
  25 GB free on `/mnt/ForgeRealm` and 20 GB on `/`.
- **L69:** one reference governs each class:
  - GroupNorm values: the float64 reference;
  - tracked files: `0e1dea3`;
  - the module: round 1's pin;
  - earlier receipts: their sha256 maps (G1).

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes. NO subagents. No background waits. Run everything synchronously, each command under an explicit
  timeout.
- **No deletes,** in the leave-and-list form. Anything that must go stays where it is and is listed under `## Done`
  for the lead to bin. Scratch goes under `/tmp` (exempt). Never set `DELETE_ME_ROOTS`.
- Never kill a process you did not start. GPU never.
- A failed gate is a result. Report it with receipts.
- Prior Art Directive (AGENTS.md): as A2. Add Python's `operator.index` (PEP 357).
- **Write early:** create `artifacts/gn1/a3/REPORT.md` and `artifacts/gn1/a3/LEDGER.md` within your first few steps.

## Done

Your final message contains, verbatim:
- G1 to G6, each PASS or FAIL, with receipts;
- the per-finding fix table;
- the `nn.py` diff against `bd30498a`;
- the G4 table;
- the runner-outcome list;
- the mutant table;
- the seal's import trace summary;
- P94 and P95, each HELD or NOT HELD;
- every file written, and anything left for the lead to bin.
