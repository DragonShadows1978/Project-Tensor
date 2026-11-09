# GN1 amendment A2: fix every GN1-REV1 finding (GPU supervision, scorer, mutants, input validation, tolerances, arms)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_gn1`
(branch `gn1`, at the commit carrying this amendment). Edits and CPU runs there are AUTHORIZED, limited to:
- `tensor_cuda/tensor_cuda/nn.py`, the `GroupNorm` class only;
- `tensor_cuda/tests/test_groupnorm_gn1.py`;
- `artifacts/gn1/a2/`: the new tooling copy, arms, mutants, receipts, ledger and report;
- `logs/`, and scratch under `/tmp/gn1_a2_*`. Point `TMPDIR` at it for every command.

**Read-only:**
- `artifacts/gn1/` outside `a2/`: round 1, A1 and REV1, all frozen;
- the pinned module and `tensor_cuda/build-gn1/` (no rebuild; no C++ or CUDA change);
- every other file in this worktree, every other worktree, and production.

Read-only git is allowed.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.
Search inside this worktree, by path.

## Why

GN1-REV1 (`orders/GN1_REV1_REVIEW.md`; receipts `artifacts/gn1/rev1/`, commit `b2dccd67`) is a Claude cross-model
review. It rejected round 1 plus A1 with 11 findings, R1-F01 to R1-F11. Read `rev1/REPORT.md` and `rev1/findings.json`
first. Each finding has a reproduction, and its hostile cases H1 to H6 are scripted in `rev1/hostile_scorer.py`.

The math was found correct. The defects are in the GPU supervision and the scorer, the mutants' granularity, two
untested decisions, missing input validation, the tolerance write-up, and the arms. House rule: every finding is fixed,
NITs included, and the fix is reviewed again.

## Design (binding; the lead chose these)

- **D1, the tooling copy.** Copy `artifacts/gn1/a1/`'s tooling (the staged script, `run_gpu_line.py`,
  `run_gpu_tests.py`, `score_gpu.py`, `start_gpu_run.py`, the stand-in, the arm and mutant builders) to
  `artifacts/gn1/a2/`. Make every change there. `a1/` stays frozen.
- **D2, a kill is a real failure (R1-F01, R1-F02).** A mutant counts as killed only when all of these hold:
  - **Real failures:** at least one GroupNorm test outside `test_reference_*` fails in its call phase, with an
    `AssertionError` or a registered refusal (`ValueError` or `TypeError` from GroupNorm).
  - **No device fault:** no test anywhere in the session failed with a device-pattern error. The device patterns are
    at least:
    - `CUDA error at`;
    - `cudaMallocAsync failed`;
    - `out of memory`;
    - `illegal memory access`;
    - `unspecified launch failure`;
    - `launch timed out`;
    - every pattern round 1 already had.

    A session with any such failure is a void line, never a kill.
  - **Its binding holds:**
    - the child summary's arm and suite equal the parent's own arguments;
    - its `nn.py` and engine sha256 equal the seal and the pin;
    - the JUnit outcomes the parent reads itself equal the summary's reports, ID by ID.
- **D3, the scorer re-verifies (R1-F03).** `score_gpu.py` refuses unless every one of these checks out:
  - `START.json` and its seal;
  - each line's child result and its D2 binding;
  - each exit against `gpu_lines.json`'s expected exit;
  - the run-root guard the parent and child use.

  A stored parent receipt is compared with this recomputation, never trusted on its own.
- **D4, single-site mutants (R1-F04, R1-F05, R1-F06).**
  - Re-cut every mutant to one edit site. Each site gets its own mutant, with a registered witness that reaches it on
    CPU (a line trace like REV1's `shim_trace.py`). Where a multi-line edit is the smallest runnable form, say why.
  - **New mutants:** REV1's V1 (drop the bias dtype check), V2 (drop the weight dtype check) and V3 (upcast whenever
    `dtype != "float32"`), plus one mutant per new D5 validation.
  - **The bar:** every mutant is killed, or carries an equivalence proof.
- **D5, input validation in GroupNorm (R1-F06, R1-F05).**
  - **`__init__`:** refuse with `ValueError` any `num_groups` or `num_channels` that is not a positive integer (`bool`
    excluded), before the divisibility check.
  - **`forward`:**
    - refuse with `ValueError` when `x.ndim < 2` or `x.shape[1] != C`;
    - refuse with `TypeError`, naming the dtype, when `x.dtype` is not float32, float16 or bfloat16;
    - a parameter whose dtype is not one of those three is refused the same way.
  - **Tests:** each refusal is tested. Mixed weight and bias dtypes are tested on float32 inputs: float32 weight with
    float16 bias, and float16 weight with float32 bias.
  - Report what PyTorch does in each case where you can verify it. Mark anything you cannot "unverified — lead to
    check".
- **D6, the regression rule (R1-F07).**
  - **Candidate exit:** the candidate regression line accepts child exit 0 or 1.
  - **It passes when:**
    - no ID that passed on baseline fails, errors or goes missing on the candidate;
    - every candidate failure also failed on baseline, with the same exception class.
  - The order's own rule is "no regression", not "everything passes".
- **D7, retries (R1-F08).** A lease refusal (exit 75 or 76) is recorded as a "not admitted" attempt in a numbered
  subdirectory, and the label may be retried in the same run root. Only an admitted attempt counts. The scorer uses
  the last admitted attempt and lists every attempt.
- **D8, tolerance derivations (R1-F09).**
  - **What changes:** rewrite `TOLERANCES.md` under `a2/` with a real first-order bound for each budget, not
    multipliers.
  - **Each bound:** states its powf accuracy assumption (name the CUDA Math API figure, or mark it "unverified — lead
    to check") and records the margin on the worst-conditioned case. REV1 reports 0.907 of budget.
  - **Numbers:** they do not change unless the bound shows one is unsound. If it does, stop and report it before any
    GPU run. Never widen a budget to fit.
- **D9, arms (R1-F10).**
  - **No symlinks:** every arm holds a byte copy of the pinned module, checked by sha256 when the arm is built and
    again when the child starts.
  - **Regeneration:** a script under `a2/` rebuilds every arm deterministically from the commit and the pin, and
    verifies the result against the seal. The script is the committed record. The arms are not committed.
- **D10, the recorded cause (R1-F11).** The parent reads forge-gpu-run's own outcome from its log and records it:
  timeout, window-closed, or the child's exit. A 124 is never labelled only as "timeout".

## Gates

- **G1, scope.**
  - Relative to `0e1dea3`: the diff is `GroupNorm` plus the test file.
  - Relative to `0c6a4244`: the `nn.py` diff is D5 only.
  - `a1/`, `rev1/` and round 1's receipts are byte-identical before and after, by sha256 maps.
- **G2, CPU checks.**
  - All earlier reference checks pass.
  - Every new refusal and mixed-dtype case is checked.
  - Both arms import and collect with the GPU hidden. List the IDs.
- **G3, the module.** The pin and source map are unchanged. Every arm's module copy matches the pin.
- **G4, hostile controls.** Each is run through `a2/`'s real parent and scorer on CPU, with a stand-in for
  forge-gpu-run. Each must be rejected by the rule it targets:
  - REV1's H1, H1b, H2, H3, H4 and H5, with REV1's scripts adapted to `a2/` paths. List each adaptation;
  - H2 with each D2 device pattern;
  - a kill counted only from `test_reference_*`;
  - a JUnit and summary that disagree on one ID;
  - a line whose engine copy differs from the pin by one byte;
  - a retry after a refusal, which must be admitted.

  REV1's controls that were correctly rejected (H2c, H3b, H6) are still rejected.
- **G5, mutants on CPU.** The mutant list, the witness reach for each mutant (a trace), and the validation that each
  mutant loads and differs.
- **G6, staging.** The staged script, every line's expected exit, and one audit-mode line.

**Registered acceptance:** ACCEPT when G1 to G6 pass. The lead's GPU run follows a clean GN1-REV2.

## Lead predictions (registered here, before any data)

- **P86:** every G4 control is rejected by its targeted rule, and REV1's H2c, H3b and H6 are still rejected.
- **P87:** D8's bounds keep every A1 tolerance number unchanged.

## Mechanism & scope

- **Measures:** whether GN1's GPU verdict can only report what happened (real kills, real regressions, bound to the
  line that ran), and whether GroupNorm refuses every input outside its domain with a clear error.
- **Mechanism doc:**
  - `artifacts/gn1/rev1/REPORT.md` and `findings.json`;
  - `orders/GN1_GROUPNORM_FIX.md`;
  - `orders/GN1_A1_HALF_PRECISION.md`;
  - `artifacts/gn1/a1/REPORT.md`.
- **Exercises:**
  - the amended class and tests on CPU;
  - `a2/`'s parent, child and scorer under hostile inputs;
  - the mutant builder and its traces;
  - the arm regeneration script.
- **Does not cover:**
  - GPU execution;
  - other layers;
  - the engine's float16 pow backward (FP16-POW1);
  - `reshape(-1)`;
  - performance.
- **Required evidence:**
  - the diffs against `0e1dea3` and `0c6a4244`;
  - the sha256 maps;
  - the per-finding fix table (R1-F01 to R1-F11, each with its receipt);
  - the G4 control table;
  - the mutant table with traces;
  - the D8 derivations;
  - the regeneration script and its verification;
  - the staged script.
- **Operand census:**
  - every REV1 finding;
  - H1 to H6 and the new controls;
  - every D2 device pattern;
  - every mutant;
  - every D5 refusal;
  - both mixed-dtype configurations.
- **Baseline isolation:** the baseline arm is `0e1dea3`'s package with the pinned module (now a copy). `a1/` and
  `rev1/` are frozen.
- **Forbidden configuration:**
  - any edit outside the writable list;
  - symlinks in any arm;
  - a rebuild, or any C++ or CUDA change;
  - widening a tolerance;
  - a kill counted from a device fault or a reference test;
  - GPU use;
  - git writes;
  - deleting anything.

## Budget (L42)

- **Units:** round 1 took 1,300 s and A1 took 1,286 s. REV1's hostile runs ran in minutes on CPU.
- **Ceiling:** 2 h. Every single command stays under 20 minutes.
- If a gate cannot finish inside the ceiling, stop and report `RED-BUDGET` with what was done.

## Registration checklist (order-lint)

- **L01:** the instruments all exist: A1's tooling, REV1's scripts and receipts, the pinned module, NumPy, and pytest
  9.0.2.
- **L02:** the smoke is G2 and G4 on CPU. The lead's GPU run is the first device execution.
- **L03:** n/a. No coverage bar; the mutants stand in for one.
- **L05:** the owners:
  - the GPU tooling under `artifacts/gn1/` (F01-F03, F07, F08, F10, F11);
  - the tests and mutants (F04, F05, F09);
  - the product's GroupNorm (F06).
- **L06:** no size threshold.
- **L07:** n/a. Nothing visual.
- **L08:** the staged lines and the scorer run from `a2/`'s own directory.
- **L09:**
  - JUnit outcomes are compared with the summary ID by ID, as exact strings;
  - files are compared by sha256 maps;
  - exits are integers compared with `gpu_lines.json`.
- **L12:** the rules this round changes are the kill rule, the regression rule, the scorer's checks, the retry rule
  and the arm layout. G4 compares REV1's controls before and after.
- **L14:** n/a. No gain gate.
- **L15:** P86 and P87 are predictions, scored in the report.
- **L16:** two cheapest passing fixes:
  - special-casing REV1's exact hostile inputs. G4's new controls (every device pattern, the reference-only kill, the
    one-byte engine change) exclude it;
  - dropping a check to make a control pass. D2's conjunction and G4's REV1 rejections exclude it.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`. Name local tool paths
  generically.
- **L25:** threads at 1.
- **L26:** the lead touches nothing in this worktree while the seat runs.
- **L34:** a Codex seat (gpt-6.1-sol, xhigh), on CPU. GN1-REV2 is a Claude seat.
- **L38:** the test file only.
- **L40:** outputs go under `artifacts/gn1/a2/` or `/tmp/gn1_a2_*`.
- **L30:** the hashes this order binds to, and their receipts:
  - the module pin: `artifacts/gn1/engine_pin.json` (sha256 `c9b7dc61…`), restated in `a1/G3_pin.json`;
  - the source map: `artifacts/gn1/engine_sources_after.json` (map sha256 `98ef59cd…`);
  - the baseline package: `a1/base_sha256.json`, which maps `0e1dea3`'s files;
  - the frozen receipts: G1 builds fresh maps of `a1/`, `rev1/` and round 1 at the start, and compares with them at
    the end.
- **L41:** every mutant's witness is traced to its edit site on CPU before the mutant counts (D4, G5).
- **L42:** see Budget.
- **L44:** D3. The scorer recomputes every verdict from the child results, the JUnit, the seal and the expected
  exits. A disagreeing stored receipt is refused (G4).
- **L45:** each line counts only with the parent's record of the child's exit and timeout, written after the child
  exits, with forge-gpu-run's own outcome (D10).
- **L47:** the findings and their reproductions are REV1's receipts (`rev1/hostile_results.json`,
  `shim_trace_results.json`, `decisions_probe_results.json`, `numerics_results.json`). REV1 measured the 0.907
  margin. Requirements on new code (the D5 refusals, the D2 rule) are marked as requirements.
- **L48:** G4 runs every control on CPU before any GPU line.
- **L55:** exact equalities: test IDs, dtype strings, sha256 digests and exit integers. Values use the registered
  tolerances.
- **L58:** every CPU run sets `CUDA_VISIBLE_DEVICES` empty.
- **L59:** the seal is this order's commit. `START.json` is written by the GPU run's parent before its first line,
  and the scorer checks it (D3).
- **L64:** reads: the worktree, REV1's and A1's receipts, NumPy. The read-only list permits them.
- **L65:** leave-and-list form used (Rules).
- **L67:** the module copies add about 10 MB per arm, about 0.3 GB in total. The GPU lines write a few MB each. The
  lead's runner stops below 25 GB free on `/mnt/ForgeRealm` and 20 GB on `/`.
- **L69:** one reference governs each class:
  - GroupNorm values: the float64 reference;
  - every other tracked file: `0e1dea3`;
  - the module: the round 1 pin;
  - the receipts of round 1, A1 and REV1: their sha256 maps (G1).

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes. NO subagents. No background waits. Run everything synchronously, each command under an explicit
  timeout.
- **No deletes,** in the leave-and-list form. Anything that must go stays where it is and is listed under `## Done`
  for the lead to bin. Scratch goes under `/tmp` (exempt). Never set `DELETE_ME_ROOTS`.
- Never kill a process you did not start. GPU never.
- A failed gate is a result. Report it with receipts.
- Prior Art Directive (AGENTS.md):
  - mutation adequacy and the reachability-infection-propagation model (DeMillo, Lipton and Sayward 1978;
    Offutt/Voas);
  - Higham's error analysis for D8;
  - PyTorch GroupNorm's argument checks.

  Name each at the code site, in the ledger and in the report. Mark anything unverifiable "unverified — lead to
  check".
- **Write early:** create `artifacts/gn1/a2/REPORT.md` and `artifacts/gn1/a2/LEDGER.md` within your first few steps.

## Done

Your final message contains, verbatim:
- G1 to G6, each PASS or FAIL, with receipts;
- the per-finding fix table;
- the `nn.py` diff against `0c6a4244`;
- the G4 control table;
- the mutant table with each witness's trace;
- the D8 bounds and their margins;
- the staged lines with their expected exits;
- P86 and P87, each HELD or NOT HELD;
- every file written, and anything left for the lead to bin.
