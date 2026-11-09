# GN1 amendment A1: GroupNorm computes in float32 for float16 and bfloat16, so low-variance groups no longer overflow

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_gn1`
(branch `gn1`, at the commit carrying this amendment). Edits, CPU runs and engine builds there are AUTHORIZED, limited
to:
- `tensor_cuda/tensor_cuda/nn.py`, the `GroupNorm` class only;
- `tensor_cuda/tests/test_groupnorm_gn1.py`;
- `artifacts/gn1/a1/`: every receipt, script, reference, mutant, arm, staged GPU line, ledger and report of this
  round;
- `logs/`, and scratch under `/tmp/gn1_a1_*`. Point `TMPDIR` at it for every command.

**Read-only:**
- round 1's receipts under `artifacts/gn1/` outside `a1/`;
- the pinned module and `tensor_cuda/build-gn1/`. The engine is not rebuilt: no C++ or CUDA source changes;
- every other file in this worktree, every other worktree, and production.

Read-only git is allowed.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.
Search inside this worktree, by path.

## Why

Round 1 (`orders/GN1_GROUPNORM_FIX.md`, receipts `artifacts/gn1/`, commit `50b3418f`) fixed the `-1` reshape and passed
G1 to G4. Its `AMENDMENT_01.md` found a float16 defect and left it open:
- **The defect:** the backward of `(var + eps).pow(-0.5)` materializes `(var + eps)**(-1.5)` in the input dtype
  (`ops.cpp` pow backward, `x**(p-1)`). In float16 that exceeds 65504 whenever `var + eps` is below about 6.2e-4,
  because 65504^(-2/3) ≈ 6.2e-4.
- **Its reach:** this is not only the singleton-group case round 1 excluded. Any group whose variance is below about
  6e-4 (near-constant activations, a common case in training) gives inf, then NaN, gradients.
- **Why it matters:** `Module.half()` (`nn.py:81`) casts every parameter to float16 for mixed-precision training, so
  this is a supported path.
- **The house rule:** no waivers for real defects. PyTorch's GroupNorm keeps its statistics in float32 for half
  inputs (unverified — lead to check: "PyTorch group_norm acc_type half").

## Design (binding; the lead chose these)

- **D1, compute in float32.** In `GroupNorm.forward`:
  - when `x` is float16 or bfloat16, cast it to float32 with the differentiable cast (`astype`, `ops::cast`,
    `ops.cpp:1060-1063`, whose backward casts the gradient back to the source dtype);
  - cast `weight` and `bias` to float32 when they are not float32;
  - compute the statistics, the normalization and the affine in float32;
  - cast the output back to `x`'s dtype.
  - **Float32 inputs with float32 parameters:** the computation is unchanged from round 1, with no casts. Show the
    float32 IDs' code path is byte-for-byte the same sequence of operations.
  - **Gradients:** each parameter's gradient arrives in that parameter's own dtype, and `x`'s gradient in `x`'s dtype.
- **D2, tests.** Update `test_groupnorm_gn1.py`:
  - **Restore** round 1's two excluded float16 default-eps singleton backward cases.
  - **Add** non-singleton groups with variance 1e-4 and 1e-6 at the default eps, forward and backward, for float16 and
    bfloat16. These are the cases that overflow without D1.
  - **Add** both parameter configurations for float16 and bfloat16 inputs:
    - the module's own float32 parameters (the default);
    - parameters cast by `Module.half()`.
  - **Assert** that the output dtype equals the input dtype, and each gradient's dtype equals its leaf's dtype. NaN
    and inf never pass.
  - **Tolerances:** re-derive each low-precision one for D1's path, before any GPU run: input already represented,
    float32 compute (round 1's float32 budget), then one rounding to the output dtype. Gradients are rounded once to
    their leaf's dtype. Round 1's float32 tolerances and IDs stay as they are.
- **D3, mutants.** Keep round 1's 14 mutants, re-derived on D1's code, and add:
  - **M15:** compute in the input dtype (no upcast);
  - **M16:** the output not cast back;
  - **M17:** the parameters not cast (float16 parameters with a float32 `x`);
  - **M18:** the upcast applied to `x` but not to the parameters.

  Each must be killed, or carry an equivalence proof.
- **D4, staging.** Re-stage under `artifacts/gn1/a1/` with round 1's structure:
  - baseline focus and candidate focus;
  - baseline and candidate regression over the files that import `tensor_cuda.nn`;
  - one line per mutant.

  The same pinned module serves every arm. `RUNS` has no default. Each line names its expected exit.

## Gates

- **G1, scope.** Relative to `0e1dea3`, the diff is `GroupNorm` in `nn.py` plus the test file. Relative to round 1
  (`50b3418f`), the `nn.py` diff is D1 only. Every other tracked file is byte-identical, shown by sha256 maps.
- **G2, CPU checks.**
  - Round 1's reference checks still pass.
  - The new low-variance cases' references are checked against central differences in float64.
  - Show by arithmetic that each new case overflows float16 under M15's path: `(var + eps)**(-1.5) > 65504`.
  - Both arms import and collect with the GPU hidden. List the IDs.
- **G3, the module.** The pinned module (`c9b7dc61…`) and its source map are unchanged. No rebuild.
- **G4, staging.** The staged script, its expected exits, the mutant list (M01 to M18) and one CPU audit-mode line.
- **G5 (the lead's, after a clean cross-model review):** the candidate passes, the regression lanes show no
  regression, and every mutant dies or has a proof.

**Registered acceptance of this round:** ACCEPT when G1 to G4 pass.

## Lead predictions (registered here, before any data)

- **P84:** with D1, every float16 and bfloat16 case passes at re-derived tolerances no looser than round 1's, and
  M15 is killed by the new float16 low-variance backward cases.
- **P85:** the float32 IDs are unchanged in number and tolerance.

## Mechanism & scope

- **Measures:** whether GroupNorm is numerically sound for every dtype it accepts, including low-variance groups in
  float16 and bfloat16 under mixed-precision use, and whether its tests catch the loss of each D1 decision.
- **Mechanism doc:**
  - `artifacts/gn1/AMENDMENT_01.md`;
  - `artifacts/gn1/REPORT.md`;
  - `orders/GN1_GROUPNORM_FIX.md`;
  - `nn.py:81` (`Module.half`);
  - `ops.cpp:1060-1063` (cast).
- **Exercises:** the amended class, the updated tests, the NumPy reference on CPU, and the staged lines in audit mode.
- **Does not cover:**
  - the engine's float16 pow-backward materialization, which the lead tickets separately;
  - other normalization layers (BatchNorm1D, InstanceNorm2D), also a lead follow-up;
  - `-1` in `reshape`;
  - GPU execution;
  - performance.
- **Required evidence:**
  - the diff against round 1 and against `0e1dea3`;
  - the sha256 maps;
  - the overflow arithmetic;
  - the tolerance derivations;
  - the dtype and parameter-configuration table;
  - the mutant list;
  - the staged script with its expected exits;
  - the audit-mode line.
- **Operand census:**
  - round 1's census;
  - variances 1e-4 and 1e-6 at the default eps;
  - float32 parameters and `.half()` parameters;
  - M01 to M18.
- **Baseline isolation:** the baseline arm is `0e1dea3`'s package, with the same pinned module.
- **Forbidden configuration:**
  - any edit outside the writable list;
  - an engine rebuild or any C++ or CUDA change;
  - changing round 1's float32 tolerances;
  - a tolerance chosen after GPU data;
  - GPU use;
  - git writes;
  - deleting anything.

## Budget (L42)

- **Units:** round 1 took 1,300 s in all, the engine build included. This round has no build.
- **Ceiling:** 90 minutes. Every single command stays under 20 minutes.
- If a gate cannot finish inside the ceiling, stop and report `RED-BUDGET` with what was done.

## Registration checklist (order-lint)

- **L01:** the instruments all exist: round 1's pinned module, arms tooling and test file, NumPy, and pytest 9.0.2.
- **L02:** the smoke is G2's import and collection on CPU. The lead's G5 is the first device run.
- **L03:** n/a. No coverage bar; the mutants stand in for one.
- **L05:** the owner is the product (`nn.GroupNorm`).
- **L06:** no size threshold. The tolerances come from derivations.
- **L07:** n/a. Nothing visual.
- **L08:** the staged lines run from the tooling's own directory.
- **L09:**
  - values are compared to the float64 reference at the stated tolerances;
  - dtypes are compared as exact strings;
  - files are compared by sha256 maps.
- **L12:** the only rule changed is the compute dtype for low-precision inputs and parameters. The float32 path is
  unchanged (D1, P85).
- **L14:** n/a. No gain gate.
- **L15:** P84 and P85 are predictions, scored in the report.
- **L16:** the cheapest passing work tests only the singleton case, or loosens the half-precision tolerances until
  they pass. D2's low-variance cases, G2's overflow arithmetic, M15 and the ban on tolerances chosen after the fact
  exclude it.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`.
- **L25:** threads at 1.
- **L26:** the lead touches nothing in this worktree while the seat runs.
- **L32:** n/a. No bounded buffer. "Overflow" here means float16's finite range, which G2 checks arithmetically.
- **L34:** a Codex seat (gpt-6.1-sol, xhigh), on CPU. A Claude seat does the cross-model review afterwards.
- **L52:** this round forbids an engine rebuild. Every lane still has what it needs:
  - every arm and mutant runs on round 1's pinned module (`c9b7dc61…`, `artifacts/gn1/engine_pin.json`), because no
    C++ or CUDA source changes;
  - G2's CPU checks need only NumPy and the Python package;
  - the GPU lines reuse the same module.

  No lane needs a build.
- **L38:** one test file, the new one from round 1. No other test changes.
- **L40:** outputs go under `artifacts/gn1/a1/` or `/tmp/gn1_a1_*`.
- **L41:** every staged line names its expected exit. Every mutant receipt names the test that killed it.
- **L42:** see Budget.
- **L45:** each staged line counts only with the parent's record of the child's exit and timeout flag, written after
  the child exits.
- **L47:** the expected values and their receipts:
  - the overflow bound: `AMENDMENT_01.md`, and G2 recomputes it;
  - `Module.half`: `nn.py:81-88`;
  - the cast's backward: `ops.cpp:1060-1063`;
  - the pinned module: `artifacts/gn1/engine_pin.json`.
- **L48:** G2 runs on CPU before any GPU line.
- **L55:** exact equalities: dtypes as strings, IDs, and sha256 digests. Numeric comparisons use stated tolerances.
  The overflow claim is exact integer and float arithmetic, recomputed in G2.
- **L58:** every CPU run sets `CUDA_VISIBLE_DEVICES` empty.
- **L59:** the seal is this order's commit. The lead's GPU run writes a start record before the first line.
- **L64:** reads: the worktree, round 1's receipts, NumPy. The read-only list permits them.
- **L65:** leave-and-list form used (Rules).
- **L67:** the GPU lines write a few MB each, under 1 GB in total. The lead's runner stops below 25 GB free on
  `/mnt/ForgeRealm` and 20 GB on `/`.
- **L69:** one reference governs each class:
  - GroupNorm values: the float64 reference;
  - every other tracked file: `0e1dea3`'s bytes;
  - the module: round 1's pin.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes. NO subagents. No background waits. Run everything synchronously, each command under an explicit
  timeout.
- **No deletes,** in the leave-and-list form. Anything that must go stays where it is and is listed under `## Done`
  for the lead to bin. Scratch goes under `/tmp` (exempt). Never set `DELETE_ME_ROOTS`.
- Never kill a process you did not start. GPU never.
- A failed gate is a result. Report it with receipts.
- Prior Art Directive (AGENTS.md), at the code site, in the ledger and in the report:
  - PyTorch's float32 accumulation for half-precision normalization;
  - Micikevicius et al. (2018), "Mixed Precision Training";
  - round 1's sources.

  Mark anything unverifiable "unverified — lead to check".
- **Write early:** create `artifacts/gn1/a1/REPORT.md` and `artifacts/gn1/a1/LEDGER.md` within your first few steps.

## Done

Your final message contains, verbatim:
- G1 to G4, each PASS or FAIL, with receipts;
- the `nn.py` diff against round 1;
- the new and changed test IDs, each with its tolerance and derivation;
- the overflow arithmetic;
- the mutant list M01 to M18;
- the staged lines with their expected exits;
- P84 and P85, each HELD or NOT HELD;
- every file written, and anything left for the lead to bin.
