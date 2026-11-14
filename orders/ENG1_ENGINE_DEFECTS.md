# ENG1: six queued tensor_cuda defects, fixed with value-checking tests and mutants (CPU round; GPU lines staged)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng1`
(branch `eng1`, from `gn1` at `1680764d`, which is production main `0e1dea3` plus the accepted GN1 GroupNorm fix).
Edits, CPU runs and engine builds there are AUTHORIZED, limited to:
- `tensor_cuda/src/kernels.cu`: `NDArray::reshape` (E1), the four zero-extent entries (E4), and nothing else in that
  file;
- `tensor_cuda/src/ops.cpp`: `pow_scalar`'s backward (E2), and any zero-extent entry that lives there (E4);
- `tensor_cuda/src/conv.cu`: `make_dims` (E5);
- `tensor_cuda/tests/test_terrain_render.py`: only the lines that write `.ppm` files (E6);
- the new test file `tensor_cuda/tests/test_eng1.py`;
- `artifacts/eng1/`: every receipt, reference, mutant, staged GPU line, ledger and report;
- the build directory `tensor_cuda/build-eng1/` and the built module it installs into this worktree's package;
- `logs/`, and scratch under `/tmp/eng1_*`. Point `TMPDIR` at it for every command.

**Read-only:** every other file in this worktree, `artifacts/gn1/` included. You may copy its tooling into
`artifacts/eng1/`. Every other worktree, and the production repository `/mnt/ForgeRealm/Project-Tensor`, are
read-only too. Read-only git is allowed.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.
Search inside this worktree, by path.

## Why

These are open defects in Tensor's register. Each was found by an earlier round and left out of that round's scope:
- **E1, RESHAPE-NEG1.** `NDArray::reshape` (`tensor_cuda/src/kernels.cu:757-760`) never infers `-1`; it refuses with
  `reshape: element count mismatch`. `Shape` is `std::vector<int64_t>` (`include/tc/core.h:47`), so `-1` reaches it.
  GN1 removed the only `-1` caller in the package. PyTorch-style inference is still missing, and any user code that
  passes `-1` fails.
- **E2, FP16-POW1.** `pow_scalar`'s backward (`tensor_cuda/src/ops.cpp:854-859`) computes `p * x^(p-1)` in the input
  dtype. For float16 and small x with p < 0, `x^(p-1)` overflows: with p = -0.5, `(1e-5)^-1.5 ≈ 3.2e7`, against
  float16's maximum of 65,504. GN1's amendment A1 found this through `(var + eps).pow(-0.5)` and avoided it only inside
  GroupNorm. LayerNorm-style code, `RMSNorm`, `BatchNorm1D` and `InstanceNorm2D` (`nn.py:136`, `289`, `482`, `507`,
  `601`) still use the same composition.
- **E3, NORM-HALF.** The half-precision behaviour of those norm layers has never been measured.
- **E4, DIV0-LAST1.** Host code divides by a zero last extent in `rms_norm`, `causal_softmax`, `apa_blend_softmax` (and
  its sink variant, if it shares the division) and `topk_nd`. That is undefined behaviour in the source, which today
  compiles to an empty result. Found by TX3-F3-REV1, question Q5:
  `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_tx3f3/artifacts/tx3_f3_rev1/REVIEW.md:118`.
  Its line numbers refer to TX3's candidate, so find the same entries on this branch and cite your own lines.
- **E5, POOL-OHW1.** `make_dims` (`tensor_cuda/src/conv.cu:126`) never checks `OH > 0` and `OW > 0`. A kernel larger
  than the padded input gives negative output extents, and im2col, col2im and the pools then return a wrongly shaped
  result. Found by the same review, Q6 (`REVIEW.md:119`).
- **E6, TERRAIN-TEST-WRITES.** `tensor_cuda/tests/test_terrain_render.py` writes tracked or ignored
  `artifacts/wo*.ppm` files into the repository on every run, at lines 2885-2886, 3866, 3912, 3941, 3960 and any other
  `_write_ppm(repo_root / "artifacts" ...)` site. A test must not write into the source tree.

## Design (binding; the lead chose these)

- **E1, reshape.** `NDArray::reshape` accepts at most one `-1`, and infers it as `numel / product(other extents)`.
  - **It refuses,** each with its own message:
    - two or more `-1`;
    - any other negative extent;
    - a `-1` when the product of the other extents is 0 (PyTorch refuses this case too);
    - a non-integer inferred extent.
  - **Everything else is unchanged:** storage sharing, and the existing refusal message for a plain count mismatch.
  - Check, and cite, whether any Python-side shape check rejects `-1` before C++ sees it. If one does, change only that
    check, and only to pass a single `-1` through.
- **E2, pow backward.** For float16 and bfloat16 inputs, compute `g * p * x^(p-1)` in float32 per element, then cast
  the product to the input dtype once. For float32 and float64 the result must be bit-identical to today's: same
  operations, same order.
  - Forward `pow` is unchanged.
  - If the engine has no float32 elementwise path that fits, use the nearest existing ops (`astype`, `ew_pow`, `nmul`)
    and say so.
- **E3, the half-precision census (measurement only).** Write staged GPU tests for `LayerNorm`, `RMSNorm`,
  `BatchNorm1D`, `InstanceNorm2D` and `GroupNorm`:
  - forward and backward in float16 and bfloat16, against a float64 NumPy reference;
  - including one case with group or channel variance near `eps`;
  - at tolerances derived from each dtype's rounding.

  No norm class changes in this round. The census result, after the lead's GPU run, decides any follow-up. E2 already
  removes the backward overflow these layers share.
- **E4, empty extents.** `rms_norm`, `causal_softmax`, `apa_blend_softmax` (and the sink variant, if it divides) and
  `topk_nd` return an empty result of the correct shape and dtype when the last extent or the row count is 0. They do
  this before any division and with no kernel launch.
  - `topk_nd` with `k > last extent` keeps its existing refusal. `k = 0` returns empty values and indices.
  - State the PyTorch behaviour you match for each, citing the documentation.
- **E5, pooling dims.** `make_dims` refuses `OH <= 0` or `OW <= 0` with an error that names the input extents, the
  kernel, the stride and the padding. This covers every caller: im2col, col2im, and the max and average pools forward
  and backward.
- **E6, test writes.** Each `.ppm` write goes to pytest's `tmp_path`, or a `tmp_path_factory` directory, instead of
  `repo_root / "artifacts"`. The images' content and every assertion are unchanged. No other line of the file changes.
- **The engine build:**
  - `cmake -S tensor_cuda -B tensor_cuda/build-eng1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89`
    `-DFETCHCONTENT_SOURCE_DIR_PYBIND11=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src`
    `-DFETCHCONTENT_FULLY_DISCONNECTED=ON`;
  - then `nice -n 19 cmake --build tensor_cuda/build-eng1 -j2`.
  - Use the CUDA 12.6 toolkit, as GN1 did (its round 1 first picked the system's 12.0; see GN1's ledger).
  - Pin the module's sha256, the commit, and a sha256 map of its sources.
  - Building needs no GPU.
- **The arms.**
  - **Baseline:** `gn1`'s sources, built the same way into `tensor_cuda/build-eng1-base/`.
  - **Candidate:** this branch.
  - Unlike GN1, the arms differ in C++ as well, so each arm has its own module, and both are pinned.
- **The tests (`tensor_cuda/tests/test_eng1.py`).** Values are checked against float64 NumPy references written in
  the test file, never against shapes alone. Every refusal is tested by its message.
  - **E1:** valid `-1` in each position, for ranks 1 to 4; each refusal; and storage sharing, shown by a write
    through the view.
  - **E2:** gradients of `x.pow(p)` for p in {-1.5, -0.5, 0.5, 2, 3}, in float16, bfloat16, float32 and float64,
    including small x. Float32 and float64 must be bit-identical to the baseline arm.
  - **E3:** the census above.
  - **E4:** each op at zero last extent and zero rows: shape, dtype, no error. `topk` with k = 0, and with k too
    large.
  - **E5:** each caller with a too-large kernel refuses. A valid pooling still matches the reference.
  - **E6:** running the terrain test file leaves `git status --porcelain --ignored` of `artifacts/` unchanged.
  - **Tolerances** are stated with their derivations before any GPU run, and are never tuned to results.
- **Mutation.** Write the mutant list before running it, with at least one mutant per decision:
  - E1: two `-1` accepted; inferred extent off by one; `-1` accepted with a zero product; a negative extent other
    than `-1` accepted;
  - E2: the half path computed in the input dtype; the exponent `p` instead of `p - 1`; the cast before the multiply;
  - E4: an empty result with the wrong shape; the guard placed after the division;
  - E5: `>= 0` instead of `> 0`; the check on `OH` only;
  - E6: one write left in the source tree.

  Each mutant is one edit, built or applied, and run against `test_eng1.py` (E6's against the terrain file) on the
  GPU by the lead. Every mutant is killed. A survivor counts only with an equivalence proof.
- **The GPU lines (the lead runs them).**
  - Stage `artifacts/eng1/STAGED_GPU_COMMANDS.sh` by adapting GN1's accepted A8 tooling (`artifacts/gn1/a8/`:
    `run_gpu_line.py`, `run_gpu_tests.py`, the scorer, `gpu_lines.json`). Copy it into `artifacts/eng1/` and ledger
    every change.
  - **The lines:**
    - (a) the baseline arm on `test_eng1.py`, where the defect tests fail as predicted per ID;
    - (b) the candidate on `test_eng1.py`, where everything passes;
    - (c) the candidate against the baseline on every existing test file that exercises reshape, pow, the norms, the
      E4 ops, conv or pooling, and on the terrain file: no test that passes on the baseline fails on the candidate;
    - (d) one line per mutant.
  - Name each line's expected exit, and have the parent record the child's exit and timeout flag after the child
    exits.

## Gates

- **G1, scope.** `git diff 1680764d -- . ':!artifacts/eng1' ':!orders' ':!logs'` touches only the functions and
  lines listed under the writable target, plus the new test file. Every other tracked file is byte-identical, shown by
  sha256 maps from before and after.
- **G2, CPU checks.**
  - Each NumPy reference matches hand-computed values on constructed inputs.
  - The E2 reference gradient matches float64 central differences.
  - The test files import and collect, and list their IDs.
  - `python3 -c "import tensor_cuda"` succeeds on both arms with `CUDA_VISIBLE_DEVICES` empty. If the module refuses
    to import without a device, report it.
- **G3, the build.** Both arms' builds exit 0. The module sha256s and source maps are recorded, and `nm` shows the
  expected symbols.
- **G4, staging.** The staged script exists, every line has its expected exit and bounds, and the mutant list is
  written. One line runs in audit mode on CPU with GN1's `forge_gpu_run` stand-in, to show the plumbing.
- **G5 (the lead's GPU run, after this round):** (a) fails as predicted, (b) passes, (c) shows no regression, and
  (d) kills every mutant. The E3 census is reported, not gated.

**Registered acceptance of this round:** ACCEPT when G1 to G4 pass. G5 is the lead's, after a cross-model review.
NIT findings in that review are fixed once and closed without a further review round (owner, 11-03: less procedure).

## Lead predictions (registered here, before any data)

- **P201:** no Python-side check rejects `-1` before `NDArray::reshape`, so E1 is a C++-only change.
- **P202:** in float16, the baseline's `pow` backward overflows to inf for p = -0.5 at x ≤ 1e-4. The candidate gives
  finite values within tolerance. Float32 and float64 are bit-identical across the arms.
- **P203 (G5):** the E3 census finds at least one norm class whose float16 forward exceeds a derived tolerance for
  variance near `eps`.

## Mechanism & scope

- **Measures:** whether reshape infers `-1` correctly and refuses what PyTorch refuses; whether `pow`'s backward stays
  finite and correct in half precision without changing float32 or float64; whether the four ops handle empty
  extents; whether pooling refuses impossible geometry; whether the terrain tests stop writing into the tree; and how
  the norm layers behave in half precision.
- **Mechanism doc:**
  - `tensor_cuda/src/kernels.cu` (`NDArray::reshape`, lines 757-760), `tensor_cuda/src/ops.cpp` (`pow_scalar`,
    854-859), `tensor_cuda/src/conv.cu` (`make_dims`, 126) and `tensor_cuda/tensor_cuda/nn.py` (the norm classes);
  - the TX3-F3-REV1 review (`REVIEW.md:118-119`, path above);
  - GN1's order and amendment A1 (`orders/GN1_GROUPNORM_FIX.md`, `orders/GN1_A1_HALF_PRECISION.md`);
  - PyTorch's documentation for `reshape`/`view`, `pow`, `softmax`, `topk` and `nn.MaxPool2d`/`AvgPool2d`, for the
    refusals and the empty-extent semantics.
- **Exercises:** the changed functions, the new tests, the NumPy references on CPU, both engine builds, and the
  staged GPU lines in audit mode.
- **Does not cover:**
  - changing any norm class (E3 only measures);
  - GPU execution in this round (G5 is the lead's);
  - TX3's candidate changes. This branch is production plus GN1, and TX3 will meet these hunks at its own merge;
  - performance;
  - promotion.
- **Required evidence:**
  - the diff;
  - the sha256 maps;
  - the reference checks;
  - the tolerance derivations;
  - the build receipts for both arms;
  - the mutant list;
  - the staged script with its expected exits;
  - the audit-mode line;
  - the report and ledger.
- **Operand census:**
  - E1: ranks 1–4 with every `-1` position, and each refusal;
  - E2: 5 exponents × 4 dtypes, small x included;
  - E3: 5 norm classes × 2 half dtypes × forward and backward, including variance near `eps`;
  - E4: 4 or 5 ops × {zero last extent, zero rows}, and topk k = 0 and k too large;
  - E5: each pooling and im2col caller, valid and too large;
  - E6: the terrain file;
  - every mutant.
- **Baseline isolation:** the baseline arm is `gn1`'s sources with its own pinned module. Production is read-only.
- **Forbidden configuration:**
  - any edit outside the writable list;
  - changing a norm class;
  - changing any op other than those named;
  - editing any other existing test;
  - a tolerance chosen after seeing GPU results;
  - GPU use, or creating a CUDA context;
  - git writes;
  - deleting anything.

## Budget (L42)

- **Unit times:** GN1's engine build took about 50 s at `-j4` on an idle machine. This machine is loaded (12 cores,
  load about 26), so allow about 10 minutes per build at `-j2`. There are two arms and one build per mutant that
  edits C++. Mutant builds may be staged for the GPU run instead of built now; say which.
- **Ceiling:** 3.5 h for the round. Every single command stays under 20 minutes.
- **If a gate cannot finish inside the ceiling,** stop and report `RED-BUDGET` with what was done. Cut first: the E3
  census lines, then mutant pre-builds.

## Registration checklist (order-lint)

- **L01:** the instruments exist:
  - the worktree at `1680764d`;
  - CMake and the CUDA 12.6 toolkit, as used by GN1;
  - production's pybind11 sources;
  - NumPy;
  - pytest;
  - GN1's A8 tooling under `artifacts/gn1/a8/`.
- **L02:** the pre-registration smoke is G2's import and collection on CPU. The lead's G5 is the first device run.
- **L03:** n/a. No coverage bar. The mutants stand in for one.
- **L05:** the owners: the engine (E1, E2, E4, E5), the test suite (E6), and the norm layers (E3, measured only).
- **L06:** no size threshold. Tolerances come from derivations, not data.
- **L07:** n/a. Nothing visual is judged. The terrain images' content is unchanged, and only their path moves.
- **L09:**
  - values are compared to float64 references with `numpy.allclose` at the stated tolerances;
  - float32 and float64 backward identity is compared as uint32 or uint64 views;
  - test IDs are exact strings;
  - files are compared by sha256 maps.
- **L12:** the rules that change are reshape (`-1` accepted), the half-precision `pow` backward, the empty-extent
  results, and the pooling refusals. No other operation changes (G1).
- **L14:** n/a. No gain gate.
- **L15:** P201–P203 are predictions, scored in the report (P203 by the lead).
- **L16:** the cheapest passing work is tests that check shapes, or compare against the implementation itself. The
  float64 references, the value checks, refusal-message checks and the mutants exclude it.
- **L19:** float32 and float64 identity arrays are compared as uint32 and uint64. References are checked nonzero and
  non-constant.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`.
- **L25:** threads at 1, except the `-j2` builds.
- **L26:** the lead touches nothing in this worktree while the seat runs.
- **L34:** a Claude seat (Opus 5.5, an Agent-tool seat), on CPU. The cross-model review that follows goes to a Codex
  seat.
- **L38:** one new test file, plus E6's path-only edit to `test_terrain_render.py`. No other test changes.
- **L40:** outputs go under `artifacts/eng1/`, the two build directories, or `/tmp/eng1_*`. E6 removes the one
  existing writer into the tree.
- **L41:** every staged line names its expected exit. Every mutant receipt names the test that killed it, and shows
  that the test executes the mutated line.
- **L42:** see Budget.
- **L45:** each staged line's outcome counts only with the parent's record of the child's exit and timeout flag,
  written after the child exits. GN1's A8 tooling does this. Keep that property.
- **L47:** the expected values and their receipts:
  - `kernels.cu:759` (the refusal);
  - `ops.cpp:854-859`;
  - `conv.cu:126`;
  - `core.h:47`;
  - `nn.py:136/289/482/507/601`;
  - `test_terrain_render.py:2885`;
  - REV1's `REVIEW.md:118-119`.

  The lead read each of them at `1680764d` on 11-09.
- **L48:** G2 runs the references and the plumbing on CPU before any GPU line.
- **L55:** exact equalities: test IDs, sha256 digests, and the float32 and float64 backward identity, which holds by
  construction because the same operations run in the same order. Name one constructed input on which a reordering
  would change bits.
- **L58:** every CPU run sets `CUDA_VISIBLE_DEVICES` empty.
- **L59:** the seal is this order's commit. The lead's GPU run writes a start record before the first line.
- **L64:** reads: the worktree, production's pybind11 sources, the CUDA toolkit, NumPy, and the TX3-F3 review (read
  only).
- **L65:** n/a. You are a Claude Agent-tool seat, and the delete guard covers you. Delete nothing anyway.
- **L67:** the GPU lines run about 30 test sessions, a few MB each. The lead's runner stops below 25 GB free on
  `/mnt/ForgeRealm`.
- **L69:** one reference governs each class:
  - values: the float64 NumPy references;
  - the float32 and float64 backward: the baseline arm's bits;
  - every other tracked file: `1680764d`'s bytes;
  - the modules: their pins.
- **L82:** `grep -rn "reshape(" tensor_cuda/tensor_cuda` lists the Python callers. E1 must not change any of their
  results. Name each caller that passes an explicit shape, and show that its path is unchanged.
- **L87:** n/a. No seeded sampling is registered. Test inputs are fixed constructions.
- **L92:** n/a for kinks. E2's float64 central-difference check is on `x^p` for x > 0, which is smooth. No piecewise
  op is on the checked path.
- **L93:** n/a. No unit sample is drawn by graph position.
- **L32:** n/a. No bounded buffer, cap or log is added or relied on. E4 changes behaviour only at extent 0.
- **L46:** E6's "writes nothing into the tree" claim gets a baseline first.
  - Line (c) records `git status --porcelain --ignored -- artifacts/` before and after the terrain file, on both arms.
  - The baseline arm must show the `.ppm` writes, which proves the check detects them. The candidate must show none.
  - The seat runs this record once on CPU, on the baseline's collection only (no device), to show it parses.
- **L80:** G1 diffs against `1680764d`, the base this branch starts from. The dispatch commit adds only this order,
  which `':!orders'` excludes, so the diff base and the dispatch commit agree on every other path. Name the dispatch
  commit's hash in the report.
- **L90:** the treatment per operand, as E4 sets it:
  - `rms_norm`: empty result at D = 0 or rows = 0;
  - `causal_softmax`: empty at S = 0 or rows = 0;
  - `apa_blend_softmax`: empty at S = 0 or rows = 0;
  - the sink variant: the same, only if it shares the division. The seat reads it and states which;
  - `topk_nd`: empty at last extent 0 with k = 0, and the existing refusal for k > extent.

  No operand outside this list changes. E5 applies one rule to every `make_dims` caller, and the seat lists those
  callers.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes. NO subagents. Run every command under an explicit timeout.
- **No deletes,** in the leave-and-list form. Anything that must go stays where it is and is listed under `## Done`.
  Scratch goes under `/tmp` (exempt).
- Never kill a process you did not start. GPU never: no CUDA context, `CUDA_VISIBLE_DEVICES` empty everywhere.
- **CPU courtesy:** the machine is shared with KA1's G4, KA2's seat, two other Claude seats and GRAPA's training run.
  Use `nice -n 19` for builds and tests.
- A failed gate is a result. Report it with receipts.
- **Prior Art Directive (AGENTS.md):** PyTorch's reshape, pow, softmax, topk and pooling semantics; mixed-precision
  accumulation (Micikevicius et al., 2018); DeMillo, Lipton and Sayward (1978) for mutation testing. Name each at the
  code site, in the ledger and in the report. Mark anything unverifiable "unverified — lead to check".
- **Write early:** create `artifacts/eng1/REPORT.md` and `artifacts/eng1/LEDGER.md` within your first few steps.

## Done

Your final message contains, verbatim:
- G1 to G4, each PASS or FAIL, with receipts;
- the full source diff;
- the test files' ID lists, with each tolerance and its derivation;
- the mutant list, and which mutants are pre-built and which are staged;
- the staged script's lines with their expected exits;
- P201 and P202, each HELD or NOT HELD;
- every file written, and anything left for the lead to bin.
