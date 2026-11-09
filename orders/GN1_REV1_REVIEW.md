# GN1-REV1: cross-model review of the GroupNorm fix (round 1 and A1) before its GPU run

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_gn1/artifacts/gn1/rev1/`
(branch `gn1`, at the commit carrying this order). Writing there, CPU runs, and scratch under `/tmp/gn1_rev1_*` are
AUTHORIZED. Point `TMPDIR` at the scratch directory for every command.

**Read-only:** everything else in this worktree, including the product, the tests, round 1's receipts
(`artifacts/gn1/`) and A1's receipts (`artifacts/gn1/a1/`). Every other worktree and production are read-only too.
Read-only git is allowed.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.
Search inside this worktree, by path.

## What you review

A Codex seat (GPT-6.1 Sol) fixed `nn.GroupNorm` in two rounds:
- **Round 1:** `orders/GN1_GROUPNORM_FIX.md`, commit `50b3418f`, receipts `artifacts/gn1/`.
- **A1:** `orders/GN1_A1_HALF_PRECISION.md`, commit `0c6a4244`, receipts `artifacts/gn1/a1/`.

Before the change, every GroupNorm call failed: `reshape([N, G, -1])`, and `NDArray::reshape` does not infer `-1`. The
fix uses an explicit extent and refuses `C % G != 0`. A1 computes in float32 for float16 and bfloat16 inputs and
parameters.

The lead's GPU run (G5) comes after your review. It executes A1's staged lines:
- the baseline and candidate focus lines;
- the regression lines over every test file that imports `tensor_cuda.nn`;
- 18 mutants.

## Questions (answer each, with receipts)

1. **Correctness of the code.** Read the diff against `0e1dea3` (`artifacts/gn1/a1/changes_vs_base.patch`).
   - Is it Group Normalization (Wu and He, 2018) in every dtype and parameter configuration it accepts?
   - Is any accepted input wrong, or refused when it should not be? Examples: rank 2, `G = C`, a non-float input,
     float16 `x` with bfloat16 parameters, parameters that require no gradient.
   - Does the float32 path really equal round 1's?
2. **Independence of the oracle.** Is the test file's float64 NumPy reference independent of the engine's outputs, in
   both the forward and the backward VJP?
   - Re-derive the backward formula yourself.
   - Check one case by hand or with your own NumPy script.
3. **Tolerances.** Re-check the derivations in `artifacts/gn1/TOLERANCES.md` and `artifacts/gn1/a1/TOLERANCES.md`.
   - Is any tolerance loose enough that a wrong implementation would pass? For each mutant M01 to M18, compute or
     bound the error it causes on its predicted witness, and compare it with that witness's tolerance.
   - Is any tolerance too tight for honest float32, float16 or bfloat16 arithmetic on these operands? Bound the honest
     error yourself on at least the worst-conditioned case.
4. **Mutants.** Is each mutant a single, compilable edit that changes behaviour on some registered input? Name any
   decision in the code that no mutant covers and no test pins.
5. **Supervision and scoring.** Read `artifacts/gn1/a1/STAGED_GPU_COMMANDS.sh`, `run_gpu_line.py`,
   `run_gpu_tests.py`, `score_gpu.py` and `start_gpu_run.py`.
   - Can a line count as PASS, or a mutant as killed, without the real outcome the order requires? Examples: a kill
     counted from a setup, collection or device error; a parent that trusts the child's own summary; a timeout
     misread.
   - Construct at least one hostile input per question on CPU (forged receipts, edited JUnit, a stand-in child) and
     run it through the real scorer.
6. **Arms.** Do the arms isolate exactly one difference, `nn.py`, with the same module? Could an arm import the
   worktree's package or another arm's? Check `sys.path` and module resolution in the child command, and run one
   import probe per arm class on CPU.
7. **Scope.** Is anything outside GroupNorm changed? Is anything the orders required missing?

## Method

- Work on CPU with `CUDA_VISIBLE_DEVICES` empty. GPU never.
- **A finding** has an ID (`R1-Fnn`), a severity (BLOCKER, MAJOR, MINOR or NIT), the file and line, a reproduction or
  a precise argument, and the fix you suggest. "Looks fine" is not a finding. A question you checked and found sound
  is reported as checked, with what you ran.
- Report every real defect, NITs included. The house rule is no waivers: every finding is fixed and reviewed again.

## Mechanism & scope

- **Measures:** whether the GN1 fix is correct, whether its tests, tolerances and mutants can detect a wrong
  GroupNorm, and whether its GPU supervision can only report outcomes that happened.
- **Mechanism doc:**
  - `orders/GN1_GROUPNORM_FIX.md`;
  - `orders/GN1_A1_HALF_PRECISION.md`;
  - `artifacts/gn1/REPORT.md`;
  - `artifacts/gn1/a1/REPORT.md`.
- **Exercises:** the code, the tests' reference on CPU, the tolerance derivations, the mutants, and the staged
  scripts with hostile inputs on CPU.
- **Does not cover:**
  - GPU execution (G5);
  - the engine's other operations;
  - other normalization layers;
  - performance.
- **Required evidence:**
  - `findings.json` and `REPORT.md`;
  - the scripts behind every computation or hostile run, with their outputs;
  - each mutant-versus-tolerance bound;
  - the honest-error bound for the worst-conditioned case.
- **Operand census:**
  - the A1 diff against `0e1dea3`;
  - every test ID class (dtype, rank, G, eps, parameter configuration);
  - M01 to M18;
  - the five staged-tooling scripts;
  - the four arm classes (baseline, candidate, mutant, audit).
- **Baseline isolation:** the review reads frozen receipts, and its own runs write only under `rev1/` and
  `/tmp/gn1_rev1_*`.
- **Forbidden configuration:**
  - any write outside `rev1/` and `/tmp/gn1_rev1_*`;
  - editing the product, the tests or any receipt;
  - GPU use;
  - git writes;
  - deleting anything.

## Registration checklist (order-lint)

- **L01:** the instruments are the receipts named above, NumPy and pytest 9.0.2. All exist.
- **L05:** findings are attributed to the product (`nn.py`), the test file, or the GPU tooling under `artifacts/gn1/`.
- **L07:** n/a. Nothing visual.
- **L09:** the identity claims under review are compared by their own rules:
  - "the float32 path equals round 1's": the operation sequence in `float32_identity.json`, checked against the two
    `nn.py` texts;
  - "arms differ only in `nn.py`": sha256 maps of each arm's package files, symlink targets resolved;
  - "other tracked files unchanged": sha256 maps from `tracked_before.json` and `tracked_after.json`.

  Re-run each comparison, or state why the receipt suffices.
- **L17:** n/a. Nothing visual. "Looks fine" in Method refers to review wording, not images.
- **L41:** question 4 asks whether each mutant's registered witness reaches its edited line. For each mutant, show
  that its witness ID executes the edited line on CPU: a trace, or an argument from the call path. Do this before
  counting the mutant as covered.
- **L16:** the cheapest passing review reads the reports and agrees. Questions 2, 3 and 5 require your own
  computation and hostile runs, with receipts.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`.
- **L25:** threads at 1.
- **L26:** the lead touches nothing in this worktree while you run.
- **L34:** a Claude seat (Opus 5.5) reviews Codex's work, which makes the review cross-model.
- **L40:** outputs go under `artifacts/gn1/rev1/` or `/tmp/gn1_rev1_*`.
- **L42:** budget 75 minutes. If it runs out, report what was done and what was not.
- **L45:** your hostile runs record each child's exit after it exits.
- **L58:** `CUDA_VISIBLE_DEVICES` is empty for every run.
- **L65:** n/a. You are a Claude Agent-tool seat, and the delete guard covers you. Delete nothing anyway. Scratch goes
  under `/tmp`.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes. NO subagents. No edits outside `artifacts/gn1/rev1/`. Never kill a process you did not start.
- Prior Art Directive: name your sources for any method you use (finite differences, error bounds, mutation testing).
- **Write early:** create `artifacts/gn1/rev1/REPORT.md` and `artifacts/gn1/rev1/findings.json` in your first steps.

## Done

Your final message contains:
- the verdict: ACCEPT (no findings) or REJECT (with the findings);
- every finding with its ID, severity, file:line, reproduction and suggested fix;
- for each of questions 1 to 7, what you checked and ran;
- the files you wrote.
