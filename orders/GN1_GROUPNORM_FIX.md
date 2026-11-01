# GN1: fix nn.GroupNorm (it fails on every call) and give it tests that check values, not shapes

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_gn1`
(branch `gn1`, from production main `0e1dea3`). Edits, CPU runs and engine builds there are AUTHORIZED, limited to:
- `tensor_cuda/tensor_cuda/nn.py`, the `GroupNorm` class only;
- a new test file, `tensor_cuda/tests/test_groupnorm_gn1.py`;
- `artifacts/gn1/`: every receipt, script, reference, mutant, staged GPU line, ledger and report;
- the engine build directory `tensor_cuda/build-gn1/` and the built module it installs into the worktree's package;
- `logs/`, and scratch under `/tmp/gn1_*`. Point `TMPDIR` at it for every command.

**Read-only:** every other file in this worktree (every other test included), every other worktree, and the
production repository `/mnt/ForgeRealm/Project-Tensor`. Read-only git is allowed everywhere.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.
Search inside this worktree, by path.

## Why

- **The defect:** `nn.GroupNorm.forward` (`tensor_cuda/tensor_cuda/nn.py:517-526`) calls `x.reshape([N, self.G, -1])`.
  `ops::reshape` reaches `NDArray::reshape` (`tensor_cuda/src/kernels.cu`, the refusal at line 759), which has never
  inferred `-1`. Every call
  therefore fails with `reshape: element count mismatch`. That is the only `-1` reshape in the package.
- **Found 10-31** in TX3's suite run: `tensor_cuda/tests/test_ext_phase7.py::test_norms_and_conv1d` fails on production
  and on the TX3 candidate alike. TX3 did not cause it.
- **Test validity:** that test asserts only the output shape (line 39), so even a working GroupNorm would be unchecked.
  This project's bar for infrastructure code is passing tests plus mutation tests that show the tests can fail.

## Design (binding; the lead chose these)

- **D1, the fix.** Only the `GroupNorm` class changes.
  - `__init__` refuses `num_channels % num_groups != 0` with a `ValueError` that names both values. PyTorch's
    `nn.GroupNorm` refuses this case too.
  - `forward` reshapes to `[N, G, (C // G) * prod(spatial)]`, an explicit extent, never `-1`.
  - **Unchanged:** the constructor's signature, its parameters (`weight` ones, `bias` zeros, `eps`), and the output
    shape.
  - **Statistics:** biased variance over each group, as in Group Normalization (Wu and He, 2018) and PyTorch. Check
    which variance `var` computes in this engine (find its definition and cite it). If it is unbiased, correct for
    it.
  - `NDArray::reshape` is not changed. Teaching it `-1` is a separate feature; list it as a follow-up.
- **D2, the engine.** Build `_tensor_cuda` from this worktree's sources into `tensor_cuda/build-gn1/`:
  - `cmake -S tensor_cuda -B tensor_cuda/build-gn1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89`
    `-DFETCHCONTENT_SOURCE_DIR_PYBIND11=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src`
    `-DFETCHCONTENT_FULLY_DISCONNECTED=ON`;
  - then `cmake --build tensor_cuda/build-gn1 -j4`.
  - **Pin it:** record the module's sha256, the commit, and a sha256 map of the sources it was built from.
  - **Both arms use this one module.** The baseline arm is `0e1dea3`'s Python package, and the candidate arm is
    `gn1`'s. They differ only in `nn.py`, so a test difference between the arms comes from the fix alone.
  - The sandbox has no GPU. Building needs none.
- **D3, the tests (`test_groupnorm_gn1.py`).** Values are checked against a float64 NumPy reference written in the test
  file, never against shapes alone.
  - **Forward:** inputs of shape `(N, C)`, `(N, C, L)`, `(N, C, H, W)` and `(N, C, D, H, W)`. Groups `G = 1`, `G = C`
    and at least one value in between. `N = 1` and `N > 1`. Non-default `weight`, `bias` and `eps`.
  - **Backward:** gradients of a scalar loss (a fixed random projection of the output) with respect to `x`, `weight`
    and `bias`, against the reference's analytic gradients. If autograd does not reach a parameter, say so and test
    what it does reach.
  - **Refusal:** `GroupNorm(4, 6)` raises `ValueError`.
  - **Dtypes:** float32 is required. Census which other dtypes the engine's `mean`, `var`, `pow` and the elementwise
    ops support for this path. Test each supported one at its own tolerance, or record why it is not supported.
  - **Tolerances:** state each one, with its derivation, before any GPU run. For example: float32 rounding over a
    group of n elements, on inputs with a group variance bounded away from zero, and the condition of the gradient.
    A tolerance is never tuned to observed results.
  - The existing `test_ext_phase7.py::test_norms_and_conv1d` is left as it is. It must pass on the candidate.
- **D4, mutation.** Write the mutant list before running it.
  - **The mutants:** at least one per decision the tests should pin:
    - unbiased instead of biased variance;
    - `eps` added outside the square root;
    - `eps` ignored;
    - a reduction over the wrong axis;
    - group extent `C` instead of `C // G`;
    - groups formed channel-interleaved instead of contiguous;
    - `weight` not applied;
    - `bias` not applied;
    - `weight` and `bias` swapped;
    - the divisibility refusal removed;
    - a broadcast shape off by one dimension.

    Add any other mutant you can name.
  - **Execution:** each mutant is the fixed `nn.py` with one edit, run against `test_groupnorm_gn1.py` on the GPU by
    the lead.
  - **The bar:** every mutant is killed. A survivor counts only with an equivalence proof, a reason the mutant cannot
    change any output. The lead and the cross-model reviewer check every survivor.
- **D5, the GPU lines (the lead runs them).** Stage `artifacts/gn1/STAGED_GPU_COMMANDS.sh`. Each line goes through
  `forge-gpu-run` with `--max` and an inner timeout, from the tooling's own directory, and writes under a run root
  that the lead passes as `RUNS`. There is no default. The lines are:
  - **(a)** the baseline arm, running `test_groupnorm_gn1.py` and `test_ext_phase7.py`. The GroupNorm tests must fail
    with the reshape error, and the other tests in that file must pass.
  - **(b)** the candidate arm, on the same two files: everything passes.
  - **(c)** the candidate arm, on every test file that imports `tensor_cuda.nn`, compared with the baseline arm on the
    same files. Any test that passes on the baseline and fails on the candidate is a regression.
  - **(d)** one line per mutant.

  A parent records each line's exit and timeout flag after the child exits (L45). Name the expected exit of every
  line before the run.

## Gates

- **G1, scope.** `git diff 0e1dea3 -- . ':!artifacts/gn1' ':!orders' ':!logs'` touches only `GroupNorm` in `nn.py`
  and the new test file. Every other tracked file is byte-identical, shown by a sha256 map from before and after.
- **G2, CPU checks.**
  - The NumPy reference matches hand-computed values on at least two constructed inputs whose group mean and variance
    are exact in binary.
  - Its analytic gradients match central finite differences in float64.
  - The test file imports and collects, and lists its IDs.
  - `python3 -c "import tensor_cuda.nn"` succeeds on both arms with the GPU hidden. Report it if the engine refuses
    to import without a device.
- **G3, the build.** D2's commands exit 0. The module's sha256 and the source map are recorded, and
  `strings`/`nm` show the expected module symbols. Recording CPU-only evidence is enough here.
- **G4, staging.** The D5 script exists, every line has its expected exit and bounds, and the D4 mutant list is
  written. One line runs in audit mode on CPU, with a stand-in for `forge-gpu-run`, to show the plumbing.
- **G5 (the lead's GPU run, after this round):**
  - D5 (a) fails as stated;
  - (b) passes;
  - (c) shows no regression;
  - (d) kills every mutant, or each survivor has a proof.

**Registered acceptance of this round:** ACCEPT when G1 to G4 pass. G5 is the lead's, after a clean cross-model review.

## Lead predictions (registered here, before any data)

- **P81:** this engine's `var` is unbiased by default, so D1 needs a correction factor or an explicit biased call.
- **P82:** every D4 mutant except possibly "eps ignored" is killed. "eps ignored" survives unless some test has a
  group variance near `eps`. The registered tests should include one.
- **P83 (G5):** (a) fails only on GroupNorm IDs, (b) and (c) pass, and all D4 mutants die.

## Mechanism & scope

- **Measures:** whether `nn.GroupNorm` computes Group Normalization's forward and backward correctly on the engine,
  and whether its tests fail when any one of its decisions is wrong.
- **Mechanism doc:**
  - `tensor_cuda/tensor_cuda/nn.py` (`GroupNorm`, lines 510-526);
  - `tensor_cuda/tests/test_ext_phase7.py:35-42`;
  - Wu and He, "Group Normalization" (ECCV 2018);
  - PyTorch's `torch.nn.GroupNorm` documentation, for the refusal and the biased variance.
- **Exercises:** the fixed class, the new tests, the NumPy reference on CPU, the engine build, and the staged GPU
  lines in audit mode.
- **Does not cover:**
  - `-1` inference in `reshape` (a follow-up);
  - other normalization layers;
  - GPU execution in this round (the lead's G5);
  - performance;
  - `affine=False` and other PyTorch options this class never had.
- **Required evidence:**
  - the diff;
  - the sha256 maps;
  - the reference checks;
  - the tolerance derivations;
  - the dtype census;
  - the build receipt;
  - the mutant list;
  - the staged script with its expected exits;
  - the audit-mode line.
- **Operand census:**
  - ranks 2 to 5;
  - G at 1, C and in between;
  - N at 1 and above;
  - each supported dtype;
  - `eps` default and non-default, including one group variance near `eps`;
  - every D4 mutant.
- **Baseline isolation:** the baseline arm is `0e1dea3`'s package with the same pinned module. Production is
  read-only.
- **Forbidden configuration:**
  - any edit outside the writable list;
  - changing `NDArray::reshape` or any other op;
  - editing any existing test;
  - a tolerance chosen after seeing GPU results;
  - GPU use;
  - git writes;
  - deleting anything.

## Budget (L42)

- **Units:** an engine build took about 50 s in BP-KERNEL-1. A test file with about 100 IDs collects in seconds.
- **Ceiling:** 2 h for the round. Every single command stays under 20 minutes.
- If a gate cannot finish inside the ceiling, stop and report `RED-BUDGET` with what was done.

## Registration checklist (order-lint)

- **L01:** the instruments all exist:
  - the worktree at `0e1dea3`;
  - CMake and the CUDA 12.6 toolkit, as used for earlier builds;
  - production's pybind11 sources;
  - NumPy;
  - pytest 9.0.2.
- **L02:** the pre-registration smoke is G2's import and collection on CPU. The lead's G5 is the first device run.
- **L03:** n/a. No coverage bar; D4's mutants stand in for one.
- **L05:** the owners: the product (`nn.GroupNorm`) for the defect, and the test suite for the shape-only test.
- **L06:** no size threshold. The tolerances come from D3's derivations, not data.
- **L07:** n/a. Nothing visual.
- **L08:** the staged lines run from the tooling's own directory.
- **L09:**
  - values are compared to the float64 reference with `numpy.allclose` at the stated tolerances;
  - test IDs are exact strings;
  - files are compared by sha256 maps.
- **L12:** the only rule changed is GroupNorm's behaviour, from "always raises" to D1. No other operation's semantics
  change (G1).
- **L14:** n/a. No gain gate.
- **L15:** P81 to P83 are predictions, scored in the report (P83 by the lead).
- **L16:** the cheapest passing work is a fix whose tests compare against the implementation itself, or check only
  shapes. D3's independent float64 reference, G2's hand-computed and finite-difference checks, and D4's mutants
  exclude it.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`.
- **L25:** threads at 1, except the `-j4` build.
- **L26:** the lead touches nothing in this worktree while the seat runs.
- **L34:** a Codex seat (gpt-6.1-sol, xhigh), on CPU. The cross-model review that follows goes to a Claude seat.
- **L38:** one new test file. No existing test changes.
- **L40:** outputs go under `artifacts/gn1/`, `tensor_cuda/build-gn1/` or `/tmp/gn1_*`.
- **L41:** every staged line names its expected exit. Every mutant receipt names the test that killed it.
- **L42:** see Budget.
- **L45:** each staged line's outcome counts only with the parent's record of the child's exit and timeout flag,
  written after the child exits. A line that writes PASS and then exits non-zero is RED.
- **L47:** the expected values and their receipts:
  - the failing line: `nn.py:520` at `0e1dea3` (read it);
  - the shape-only assertion: `test_ext_phase7.py:39`;
  - the failure in TX3's run: W5 in
    `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_tx3f1m/artifacts/tx3_f1m/LEAD_RERUN1_B.md`
    (branch `tx3-f1m`);

  Requirements on new code (the `ValueError`, the explicit extent) are marked as requirements in D1.
- **L48:** G2 runs the reference and the plumbing on CPU before any GPU line.
- **L55:** exact equalities: test IDs and sha256 digests. The reference checks in G2 use constructed inputs whose
  mean and variance are exact in binary, so their expected values are exact by construction. Everything else uses
  stated tolerances.
- **L58:** every CPU run sets `CUDA_VISIBLE_DEVICES` empty.
- **L59:** the seal is this order's commit. The lead's GPU run writes a start record (`date` output and `git log -1`
  of the order) before the first line.
- **L64:** reads: the worktree, production's pybind11 sources (for the build), NumPy, and the CUDA toolkit. The
  read-only list permits all of them.
- **L65:** leave-and-list form used (Rules).
- **L67:** the GPU lines run about 20 test sessions and write a few MB of JUnit and logs each, under 1 GB in total.
  The lead's runner stops below 25 GB free on `/mnt/ForgeRealm` and 20 GB on `/`.
- **L69:** one reference governs each class:
  - GroupNorm values: the float64 NumPy reference (D3);
  - every other tracked file: `0e1dea3`'s bytes (G1);
  - the module: D2's pin.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes. NO subagents. No background waits. Run everything synchronously, each command under an explicit
  timeout.
- **No deletes,** in the leave-and-list form. Anything that must go stays where it is and is listed under `## Done`
  for the lead to bin. Scratch goes under `/tmp` (exempt). Never set `DELETE_ME_ROOTS`.
- Never kill a process you did not start. GPU never.
- A failed gate is a result. Report it with receipts.
- Prior Art Directive (AGENTS.md):
  - Wu and He (2018), Group Normalization;
  - PyTorch `nn.GroupNorm`, for its refusal and its biased variance;
  - DeMillo, Lipton and Sayward (1978), "Hints on Test Data Selection", for mutation testing;
  - the finite-difference gradient check.

  Name each at the code site, in the ledger and in the report. Mark anything unverifiable "unverified — lead to
  check".
- **Write early:** create `artifacts/gn1/REPORT.md` and `artifacts/gn1/LEDGER.md` within your first few steps.

## Done

Your final message contains, verbatim:
- G1 to G4, each PASS or FAIL, with receipts;
- the `nn.py` diff;
- the test file's ID list, with each tolerance and its derivation;
- the dtype census;
- the mutant list;
- the staged script's lines with their expected exits;
- P81 and P82, each HELD or NOT HELD;
- every file written, and anything left for the lead to bin.
