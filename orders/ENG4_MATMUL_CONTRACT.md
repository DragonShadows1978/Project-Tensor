# ENG4: the matmul forward refuses every operand shape it cannot compute (implementing round)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng4`
(branch `eng4`, from production main `f7282b1d`, at the commit carrying this order). Edits, builds and CPU runs are
AUTHORIZED, limited to:
- `tensor_cuda/src/matmul.cu` (the forward `tc::matmul`), plus host helpers you add beside it;
- `tensor_cuda/src/ops.cpp`, only `matmul_backward`'s operand check and the `Tensor matmul` refusal, so that forward
  and backward share one operand predicate;
- `tensor_cuda/include/tc/*.h`, only declarations of helpers you add;
- `tensor_cuda/src/bindings.cpp`, only `numpy_dtype` (N1 below);
- `tensor_cuda/tests/test_bwd_r1_matmul.py`, only the tests listed under M7;
- the new `tensor_cuda/tests/test_eng4_matmul.py`, new `scripts/eng4_*.py`, everything under `artifacts/eng4/`;
- the build directory `tensor_cuda/build-eng4/` and the built module it installs into this worktree's package;
- `logs/eng4_*`, and scratch under `/tmp/eng4_*`. Point `TMPDIR` there for every command.

**Read-only:** every other file here, every other worktree, the production repository, and the consumer roots
`/mnt/ForgeRealm/Project-Voice/engine` and `/mnt/ForgeRealm/GRAPA-Native-LLM` (grep only, for M1's consumer list).
**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively; search inside the roots named here.
**GPU:** you run no GPU work. You stage GPU lines and dry-run them; the lead runs them (see Machine rules).

## Why

The backward-pass review (BWD-REV1 re-check of r3, finding BWDR3-F02, filed as engine item ENG-MM1) found that the
matmul forward reads out of bounds. The lead's reading of production `f7282b1d` (`tensor_cuda/src/matmul.cu:25-83`)
found that the forward never checks B's batch dimensions:
- `b_batched = (ndb == nda)` (`matmul.cu:42`). With equal ranks, B's batch stride is `Kb * N` over A's batch count.
  - A smaller B batch reads past B's storage. This is ENG-MM1.
  - A larger B batch silently uses only B's first matrices.
  - An equal count with a different batch shape pairs matrices silently.
- With unequal ranks and `ndb > 2`, the stride is 0, so only B's first stored matrix is used. The backward refuses
  this later (`ops.cpp:829-835`, "forward uses stride 0 and only the first B matrix"), but the forward has already
  returned a wrong result.
- The forward checks neither that A and B are on the same device nor that `M`, `N`, `K` and the batch count fit in
  `int` before casting them (`matmul.cu:61-77`). The backward checks both (`ops.cpp:786-794`).

A second recorded defect is folded in as N1: `numpy_dtype` (`bindings.cpp:77-85`) maps every NumPy dtype it does not
list to float32 and reads the buffer's bytes as float32.

## Lead rulings (binding)

- **M1, census first.** Write `artifacts/eng4/CENSUS.json` with:
  - every call path into the forward `tc::matmul` (C++ callers, the binding at `bindings.cpp:2075`, the Python
    `matmul` at `__init__.py:340`, and every in-repo Python caller: `__init__.py`, `apa_packed.py`, `functional.py`,
    `nn.py`, `quant.py`, and any others your grep finds), each with file:line and the operand ranks it passes, where
    the code fixes them;
  - for the consumer roots (read-only grep for `matmul(` and ` @ `): each call site and its operand ranks where the
    code fixes them, else "dynamic";
  - every existing test that calls matmul with batch dimensions, with its rank combination.
  - If any product call path (not a test) passes a combination M3 refuses, stop and report RED with the call site.
    That is a lead ruling, not a seat decision.
- **M2, the accepted set.** The forward accepts exactly these operand combinations. They are the ones the backward
  already differentiates (BWD, now in production):
  - (a) `ndb == 2`: B is one matrix broadcast over A's batch (today's stride 0);
  - (b) `nda == ndb > 2` and A's batch shape equals B's batch shape, dimension by dimension;
  - (c) `nda != ndb`, `ndb > 2`, and every leading B extent is 1: B is one matrix (today's stride 0). The output shape
    stays A's batch shape plus `[M, N]`, as today. `test_gpu_unequal_rank_singleton_b_broadcast` pins this.
- **M3, refusals.** Every other combination raises `RuntimeError` before any CUDA call, with a message that names both
  shapes. That covers equal ranks with unequal batch shapes, including a B batch of all ones under a larger A batch,
  and unequal ranks with any B batch extent other than 1. No new broadcasting is added in this round.
- **M4, operand checks the backward already has.** The forward also refuses, before any CUDA call:
  - A and B on different devices (including one host and one CUDA operand);
  - `M`, `N`, `K` or the batch count above `INT_MAX`.
  Two host operands keep today's behaviour; that is HOST-COMPUTE-HMM, out of scope.
- **M5, one predicate.** The forward and `matmul_backward` use one shared operand predicate. The backward's count-based
  batch check (`ops.cpp:798-801`) becomes the shape check of M2(b). The `Tensor matmul` refusal (`ops.cpp:829-835`)
  stays as a guard and becomes unreachable through the forward. Say so in a code comment.
- **M6, accepted results unchanged.** For every accepted combination, the forward's output is bitwise equal to Build B's
  (G3). Zero-size operands (`M`, `N`, `K` or batch = 0) give the same result as Build B and match `numpy.matmul`'s
  shape rule under M2. Record what each gives.
- **M7, tests that pin the old behaviour.** `test_bwd_r1_matmul.py::test_gpu_refuses_forward_first_batch_only` calls a
  forward that M3 now refuses. Change it to assert the forward refusal, in this form: same parametrization, the forward
  raises `RuntimeError` naming the shapes, and no gradient is created. Any other existing test that asserts behaviour M3
  refuses is changed the same way. List each in `artifacts/eng4/INTENDED_CHANGES.json` with the old and new assertion.
  No other existing test changes.
- **N1, `numpy_dtype`.** It refuses a NumPy dtype it does not list, naming the dtype, instead of returning float32.
  First list in the census every caller of `numpy_dtype` and `tensor_from_numpy`, and whether the Python wrapper casts
  before the call. Public results for listed dtypes stay unchanged (G3).
- **M8.** No knob is added.

## Gates

- **G1, CPU only** (`CUDA_VISIBLE_DEVICES` empty). Through the public Python API, with host (`device="cpu"`) operands:
  - every M3 class refuses, with the shapes in the message;
  - every M4 size class refuses. Build the witness with zero-element operands, so nothing large is allocated: for
    example `M = 2**31` with `N = 0`;
  - N1: `tc._C.tensor_from_numpy` on a float64, an int32 and a uint16 array refuses, and each listed dtype keeps its
    Build B result.
- **G2, GPU line, ENG-MM1 witness.** On Build B, the ENG-MM1 case (A `[4, 2, 3]`, B `[2, 3, 4]`, float32 CUDA) under
  `compute-sanitizer --tool memcheck` reports an out-of-bounds read. On Build E it refuses before any kernel runs, and
  memcheck reports 0 errors. If `compute-sanitizer` fails its preflight, record that and keep the Build E half.
- **G3, GPU line, accepted results unchanged.** For each M2 class (a, b, c), each dtype (float32, float16, bfloat16),
  `trans_b` false and true, and `alpha` 1 and 0.5, plus the zero-size cases: Build E's output bits equal Build B's,
  compared per element on same-width unsigned views. Inputs are seeded; the seeds are recorded.
- **G4, no regression.** Production's test suite, per test ID, Build B against Build E, on CPU, and staged as GPU lines.
  Outcomes are identical except the intended changes listed in `INTENDED_CHANGES.json` before the after-run, each with
  its ruling.
- **G5, mutants** (validity of the new tests). Each one fails its named assertion in `artifacts/eng4/MUTANTS.json`,
  written before any mutant runs:
  - `m01_count_not_shape`: the shared predicate compares batch counts, not shapes (G1 equal-count case);
  - `m02_no_device_check`: the forward drops the device check (G1, mixed host and CUDA operands, staged GPU line);
  - `m03_no_intmax`: the forward drops the `INT_MAX` checks (G1);
  - `m04_ones_any_rank`: the predicate accepts all-ones B batches at equal ranks (G1, M3 case);
  - `m05_numpy_dtype_fallback`: `numpy_dtype` returns float32 again (G1, N1).
- **G6, scope.** Writes only in the writable set. No GPU use. No git writes, nothing deleted.

**Registered acceptance:** the seat's part passes when G1, G5 (CPU-killable mutants) and G6 pass, and G2, G3, G4's GPU
half and `m02` are staged and dry-run clean. The round is ACCEPTED when the lead's GPU lines pass too. Then a
cross-model (Codex) review follows. This is an infrastructure change: promotion needs the owner's verbatim approval and
the review receipt.

## Builds

- **Build B:** this branch as committed (production `f7282b1d`), default configuration.
- **Build E:** this round's tree.

Build both out of tree with the engine's CMake under `tensor_cuda/build-eng4/`, the way ENG3 built its arms. ENG3's
`artifacts/eng3/LEDGER.md` records the commands, in `tensor_fork_apa_wt_eng3`. Each build receipt records the module
sha256 and a sha256 map of every engine source (`tensor_cuda/src/`, `tensor_cuda/include/`, `tensor_cuda/tensor_cuda/`).

## Mechanism & scope

- **Measures:** whether the matmul forward computes only the operand combinations it computes correctly and refuses
  the rest before any device work, with accepted results bitwise unchanged; and whether `numpy_dtype` refuses dtypes
  it does not list.
- **Mechanism doc:** `tensor_cuda/src/matmul.cu` (the forward, `:25-83`) and `tensor_cuda/src/ops.cpp`
  (`matmul_backward` `:783-822`, `Tensor matmul` `:825-851`); `tensor_cuda/include/tc/core.h:9-11` states that
  NDArrays are always contiguous, so no stride check is needed.
- **Exercises:** the forward and backward operand checks; `numpy_dtype`; Builds B and E plus mutant builds; CPU tests
  with no visible GPU; staged GPU lines for G2, G3, G4 and `m02`.
- **Does not cover:**
  - new broadcasting (equal-rank all-ones B batches, or NumPy's full broadcasting);
  - host-tensor compute (HOST-COMPUTE-HMM);
  - the current-device check the backward has (single GPU here; record it in the report as a gap);
  - performance;
  - promotion.
- **Required evidence:** `CENSUS.json`; the diff; Build B, Build E and mutant build receipts with module sha256s;
  G1's CPU result; `MUTANTS.json` and `MUTANT_RESULTS.json`; `INTENDED_CHANGES.json`; G4's CPU per-ID before and after;
  the staged GPU lines with their dry-run exits; `REPORT.md`.
- **Operand census:** M1's call paths and consumer sites; M2 and M3's rank and batch classes; G3's dtype, `trans_b`,
  `alpha` and zero-size grid.
- **Baseline isolation:** Build B is the reference for G3 and G4. Production is read-only.
- **Forbidden configuration:**
  - any change to an accepted combination's result;
  - a refusal that runs after a CUDA call;
  - new broadcasting;
  - edits outside the writable set;
  - git writes;
  - deleting anything yourself.

## Machine rules

- **CPU:** at most one CPU-busy process of yours at a time, at `nice -n 19`. Builds and any job over 1 GB run in their
  own scope: `systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue -p CPUWeight=50 ...`. Start one only
  with `MemAvailable` at or above 12 GB.
- **Time:** every command under 30 minutes, except builds (under 60). Long commands run detached and are waited on in
  the foreground with `timeout <n> tail --pid`. The round's ceiling is 6 hours of wall time.
- **Disk:** stop if `/mnt/ForgeRealm` has under 40 GB free.
- **GPU:** GRAPA's run 9 holds the GPU, with short cooldown gaps between its legs. You run no GPU work. Stage each GPU
  line as one command in `artifacts/eng4/GPU_LINES.json`, in this form:
  `forge-gpu-run --min <measured or estimated seconds, at most 480> --max 540 --label eng4-<line> -- timeout <≤500>s ...`.
  Each line must fit in 500 seconds; split any that would not. Dry-run each once with `FORGE_GPU_DRY_RUN=1` (returns
  78) and record the exit. Each line writes its result under `artifacts/eng4/gpu/<line>/`, and refuses to start if
  that directory exists.

## Registration checklist (order-lint)

- **L01:** the instruments are cmake, nvcc, pytest, python3 with NumPy, `compute-sanitizer` and `forge-gpu-run`. Run
  each with `--version` (or `--status --min 30 --max 540`) first. For `compute-sanitizer`, also run a no-op through
  it.
- **L02:** GPU lines carry an explicit `--min`.
- **L03:** n/a. No share is gated.
- **L04:** no fallback is added. M3 refuses; it does not reshape or copy.
- **L07:** n/a. No prediction is registered.
- **L08:** builds go to `tensor_cuda/build-eng4/`, receipts to `artifacts/eng4/`, and scratch to `/tmp/eng4_*`.
- **L09:** G3 compares per element on same-width unsigned views, after dtype and shape checks. G4 compares test
  outcomes per ID. Logs and timings are never compared.
- **L10:** accepted results are held to identity with Build B (G3).
- **L12:** the inherited gates are production's test suite per ID. Intended changes are listed before the after-run,
  each with its ruling (expected: M7's test).
- **L15:** n/a. No threshold is fitted.
- **L16:** the cheapest passing work would refuse only the ENG-MM1 case (smaller B batch at equal ranks) and leave the
  larger-batch, equal-count and unequal-rank cases silent. Excluded by G1, which covers every M3 class, and by `m01`
  and `m04`.
- **L18:** n/a. No owner requirement is implemented behind an option.
- **L19:** every bitwise comparison first checks that each side is nonzero and non-constant, at full width.
- **L21:** the remote is public. No personal names, no raw quotes, and home paths as `$HOME/...`.
- **L22:** G3's grid is defined in this order; its inputs use recorded seeds.
- **L25:** G3 runs each build in its own process, with the same inputs from the same seeds, compared afterwards.
- **L32:** n/a. No bounded buffer and no sizing figure.
- **L34:** a Claude seat (Opus 5.5, high effort, Agent tool). The review that follows is Codex, so it is cross-model.
- **L35:** your registrations (`MUTANTS.json`, the census, the intended-change list) may fix cases and seeds. They may
  not remove, narrow or reclassify anything a gate covers. If a clause cannot hold as written, stop and report RED.
- **L37:** n/a. No knob, lever or spec key is added (M8).
- **L38:** existing tests change only as M7 allows, each listed.
- **L40:** pytest runs with `-p no:cacheprovider`, `--basetemp` and `TMPDIR` under `/tmp/eng4_*`. The engine's
  `test_terrain_render.py` writes tracked `artifacts/wo*.ppm` files (TERRAIN-TEST-WRITES). Restore those files
  afterwards from git (read-only git: `git show HEAD:<path>` into the file), or exclude that file and list it.
- **L41:** each mutant names its expected failing assertion in `MUTANTS.json`, written before it runs. Show that each
  killing test executes the mutated line.
- **L45:** each staged GPU line records `forge-gpu-run`'s exit, the inner `timeout` exit and the log's sha256. Exit 124
  is a timeout, never a pass.
- **L47:** the mutants' expected failures come from this order, never from this round's runs.
- **L48:** every GPU line is staged to run as written, from the worktree root, through the real `forge-gpu-run`.
- **L55:** M2 holds only if the predicate sees the same shapes the stride arithmetic uses. For one witness per class
  (a), (b) and (c), show the shapes, the predicate's verdict and the strides the forward then uses.
- **L56:** (a) a slip in running your own tools goes under Deviations: re-run and continue. (b) An exception in your own
  scripts is repaired at most 3 times, each recorded.
- **L58:** G1 runs with `CUDA_VISIBLE_DEVICES` empty.
- **L60:** for each refusal class, record in `artifacts/eng4/BOUNDARY.json` the call, the refusal, and Build B's
  behaviour (on CPU where Build B reaches no CUDA call; otherwise "GPU line <name>").
- **L64:** reads are this worktree, ENG3's ledger in `tensor_fork_apa_wt_eng3` for the build commands, and the two
  consumer roots (grep only).
- **L65:** leave-and-list form for any file you would remove.
- **L67:** builds go to `tensor_cuda/build-eng4/`. Stop if `/mnt/ForgeRealm` has under 40 GB free (about 47 GB at
  sealing; check before each build).
- **L69:** one reference per operand class:
  - accepted results -> Build B (G3);
  - refusals -> this order's M3 and M4 classes (G1);
  - test outcomes -> Build B per ID (G4);
  - the ENG-MM1 read -> memcheck on Build B (G2).
- **L74:** plain wording. These are engine operand checks and their tests.
- **L76:** the lead writes nothing in the writable set while the seat runs.
- **L77:** every mutant can fail, and each is run (`m02` as a staged GPU line).
- **L79:** `FORGE_GPU_DRY_RUN=1` returns 78 without a device check. Use it once per staged line.
- **L82:** `grep -n "matmul" tensor_cuda/tensor_cuda/*.py tensor_cuda/src/*.cpp tensor_cuda/src/*.cu` starts M1's
  census.
- **L99:** the rulings rest on these lead readings at `f7282b1d`, run before sealing:
  - `matmul.cu:25-83`: the forward checks rank ≥ 2, dtype and inner dimension only; `b_batched = (ndb == nda)`;
    `sB = b_batched ? Kb * N : 0`; `M`, `N`, `K` and the batch count are cast to `int` unchecked;
  - `ops.cpp:783-822`: `matmul_backward` checks same-device CUDA, `INT_MAX` and the batch count (`numel`), not the
    batch shape;
  - `ops.cpp:825-851`: `Tensor matmul`'s backward refuses unequal ranks with a non-singleton B batch;
  - `core.h:9-11`: NDArrays are always contiguous, row-major;
  - `bindings.cpp:77-85`: `numpy_dtype` returns float32 for unlisted dtypes;
  - `test_bwd_r1_matmul.py:157` (`test_gpu_refuses_forward_first_batch_only`) and `:168`
    (`test_gpu_unequal_rank_singleton_b_broadcast`) pin the old refusal and class (c).

  Seat check that runs first: on Build B with a CUDA device unavailable, the ENG-MM1 shapes on host operands reach
  the cuBLAS call (no refusal). Record what happens. That is the defect's CPU reach; G2 shows the read on the GPU.
- **L102:** no generated file or registry depends on the writable files.
- **L103:** each staged GPU line refuses, before the lease, a module whose sha256 differs from its build receipt.
- **L104:** no ratio or fraction bar. Every gate is exact (per element, per test ID, per refusal); slack 0 = 0 events.
- **L105:** n/a.
- **L109:** your helpers run as `python3 -B -P` (or `-I`). List every `.py` you wrote, and confirm none shares a
  standard-library module name.
- **L111:** n/a. No storage design is emulated.
- **L53:** n/a. Nothing is built or proven for another project's run. GRAPA's run 9 is named only because it holds
  the GPU.
- **L81:** G2's expectation comes from the lead's code reading (L99: `sB = Kb * N` over A's batch count), not from a
  witness, and G2 records it itself on Build B. Source records: the BWDR3-F02 finding (BWD-REV1 re-check of r3,
  receipts `447a67b5`) recorded the class only (equal ranks, smaller B batch, forward reads out of bounds). It is cited
  for provenance, and no gate relies on a value it did not record. Accepted-result expectations come from Build B,
  recorded by this round (G3).
- **L90:** roles: product (in-repo C++ and Python callers), test (existing tests), consumer (Voice and GRAPA sites,
  read-only), and harness (your scripts). Record the count per role in `CENSUS.json`. Treatment:
  - product callers are not edited; each is classified as accepted or refused, and a refused one stops the round
    (M1);
  - tests change only as M7 allows;
  - consumer sites are report-only;
  - harness: n/a.
- **L114:** big jobs run in the seat's own systemd scope (a Claude Agent seat can start one; `MemoryMax=12G`). None
  needs anything more. L114 GPU: staged lines set no RLIMIT_AS (nor `ulimit -v` or `prlimit --as`). The lead runs them
  in a scope with `MemoryMax=12G` and `OOMPolicy=continue`. `GPU_LINES.json` lists each line's expected peak host
  bytes.
- **L117:** n/a. No gradient consumer's trained operands are involved; the backward's numerics are unchanged (only its
  operand check, M5).
- **L118:** the census of consumers of the changed check is M1 (every call path, consumer site and test that reaches
  matmul).
- **L119:** every write-once path a staged line creates is listed in `GPU_LINES.json`, so a retry can move it first.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes (read-only git is allowed). NO subagents. No background waits: run everything synchronously, each
  command under an explicit timeout.
- No deletes, in the leave-and-list form. Never kill a process you did not start.
- **The report:** write `artifacts/eng4/REPORT.md` early and keep it current. If the write is refused, say so; your
  final message is then the report, and the lead files it verbatim.
- **Prior Art Directive:** annotate each code site and the report. Known: NumPy's `matmul` broadcasting rules; BLAS and
  cuBLAS strided-batched GEMM semantics (cuBLAS documentation, including the zero stride); mutation testing (DeMillo,
  Lipton and Sayward, 1978). Write "no prior art known to me" where none applies.

## Done

Your final message contains, verbatim:
- G1 to G6, each PASS, FAIL or STAGED;
- the census summary and the accepted and refused classes, each with its witness;
- the diff summary;
- every module sha256;
- G4's CPU per-ID result and the intended changes;
- the mutant table;
- each staged GPU line with its dry-run exit;
- the deviations;
- every file written.
