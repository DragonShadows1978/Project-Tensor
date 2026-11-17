# ENG3-F3: fix ENG3-REV3's findings, then close (Codex; CPU round with builds; GPU lines staged)

Follows: ENG3_F2_DIAGNOSTICS.md
ID reuse: ENG3-F3 is the third fix round of the same ENG3 change, as F1 and F2 were

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng3` (branch
`eng3`, at the commit carrying this order; ENG3-F2 is `60e28eb7`). Edits, CPU runs and engine builds are AUTHORIZED,
limited to:
- `tensor_cuda/src/bindings.cpp`: only `core_device` (the device parser of the private `_core_*` hooks), its CUDA query
  error path, and the hooks' rank-refusal messages (lines 2654 to 2735 at `60e28eb7`);
- `tensor_cuda/tests/test_eng3_host.py`, and new `tensor_cuda/tests/test_eng3_f3*.py` with any child helper they need;
- new `scripts/eng3f3_*.py`, `artifacts/eng3f3/` (new), the build directory `tensor_cuda/build-eng3f3/`, `logs/eng3f3_*`;
- scratch under `/tmp/eng3f3_*`. Point `TMPDIR` there.

**Read-only:**
- everything else here, including `kernels.cu` and the earlier build folders;
- `tensor_fork_apa_wt_eng3rev3` (the review: `artifacts/eng3_rev3/findings.json`, `REPORT.md` and its receipts);
- every other worktree;
- production.

Never walk `/`, `/mnt` or `/mnt/Shared` recursively.

## Why and rulings

ENG3-REV3 (a Claude cross-model review of F1 and F2; receipts in `tensor_fork_apa_wt_eng3rev3/artifacts/eng3_rev3/`)
found no BLOCKER or MAJOR. It found 3 MINOR and 3 NIT. The owner approved ENG3's merge on 11-17, conditional on all the
testing and mutation testing passing. Real findings are fixed, not waived. The lead ruled:
- **ENG3R3-F01, closed by ruling, with a test.**
  - A zero-extent tensor whose nominal byte product overflows is refused. This is an intended change: it matches
    NumPy ("array is too big") and the engine's P1T-FIX2 rule.
  - Add a shipped test that pins the refusal: `astype` of `tc.empty((0, 2**62), "uint8")` and `_core_zeros([0, 2**62])`
    raise OverflowError naming the shape.
  - Add one comment at the constructor's bound citing this ruling.
  - No product behaviour changes.
- **ENG3R3-F02, fix.** For bare `"cuda"`, `core_device` queries `cudaGetDevice` and labels the current device, as the
  public `tensor_empty` parser does (`bindings.cpp:1270-1283`).
- **ENG3R3-F03, fix.** Move F1's three killing tests from `artifacts/eng3f1/tools/eng3f1_checks.py` into
  `tensor_cuda/tests`, with their child helper:
  - `test_f1_cpu_memoryerror`;
  - `test_f1_cpu_rank_negative`;
  - `test_f1_gpu_device_index`, which skips cleanly with no GPU.
- **ENG3R3-F04 (NIT), fix once.** A failing CUDA query in `core_device` raises RuntimeError naming the call and the
  CUDA error, as public `tc.empty` does. It no longer raises "invalid device".
- **ENG3R3-F05 (NIT), fix once.** The rank messages use `std::to_string(TC_MAX_DIMS)`. Mark the unreachable
  negative-extent loops in `_core_clone` and `_core_to` with a one-line comment as defensive (the constructor refuses
  first). Do not delete them.
- **ENG3R3-F06 (NIT), fix once.** The R4 child test loads the soname `libcudart.so.12`, not an absolute toolkit path.
  So does the moved `test_f1_gpu_device_index`.

Nothing else changes.

## Gates

- **G1, CPU** (`CUDA_VISIBLE_DEVICES` empty), on Build H, this round's build:
  - every new and moved test passes;
  - every other test in `test_eng3_host.py` passes as on Build G;
  - with no CUDA device, bare `"cuda"` and `"cuda:0"` in a `_core_*` hook raise RuntimeError naming `cudaGetDevice`
    (F02 and F04 together: the bare spelling now queries the device).
- **G2, regression:** the CPU suite per test ID on Build H against Build G (`build-eng3f2/source_G`, `5fcd179b…`).
  - Identical, except for the new and moved test IDs, and the changed message rows of F04, each listed.
  - The GPU lines are staged for the lead: ENG3-F2's `g3-g1` and `g3-g2` forms on Build H against Build G, each with its
    expected exit. Each line runs `forge-gpu-run --max 540` with an explicit `--min`, and its launcher sets its full
    environment.
- **G3, mutants.** Each is built incrementally, and its killing test executes the edit:
  - `f3m01_bare_cuda_no_query`: F02 reverted. It must fail G1's no-device test for bare `"cuda"`.
  - `f3m02_query_error_as_invalid`: F04 reverted. It must fail the same test for `"cuda:0"`.
  - The moved `test_f1_cpu_memoryerror` must fail on F1's existing mutant module `f1m02_malloc_unchecked` (its build is
    in `build-eng3f1/`), run from `tensor_cuda/tests`.
- **G4, scope:** writes only in the writable set; no git writes; nothing deleted; no GPU.

**Registered acceptance:** ACCEPT when G1, G3 and G4 pass, and G2 passes on CPU and in the lead's GPU lines. A narrow
cross-model review of this round's diff follows: the ENG3-REV3 reviewer, resumed. Then ENG3 is promoted under the
owner's approval.

## Mechanism & scope

- **Measures:** whether REV3's findings are fixed or closed as ruled, with nothing else changed.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng3rev3/artifacts/eng3_rev3/findings.json`,
  with this order's rulings.
- **Exercises:** `core_device` and the hook messages; the moved and new tests; Build H and two mutant builds; CPU tests;
  staged GPU lines.
- **Does not cover:**
  - the constructor's byte rule (F01 is closed by ruling);
  - HOST-COMPUTE-HMM;
  - `Device::from_string`;
  - rank-9 public tensors (REV3 noted them as out of scope);
  - promotion.
- **Operand census:** `core_device`, its three device-taking callers (`_core_full`, `_core_zeros`, `_core_to`), the
  rank messages, and the three moved tests.
- **Baseline isolation:** Build G is the reference.
- **Required evidence:**
  - the diff;
  - Build H and mutant receipts with module sha256s;
  - G1 to G3 results;
  - the staged lines with expected exits;
  - `REPORT.md`.
- **Forbidden configuration:** any change other than the rulings above; GPU use; git writes; deletes.

## Seat

Codex gpt-6.1-sol, high effort.

**Rules:** read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is
immutable.
- No git writes, no subagents, no background waits; every command under a `timeout` (builds under 60 minutes).
- **No deletes:** never `rm`, `rmdir`, `unlink`, `git clean`, `find -delete` or an inline delete call. Leave and list.
- Never call `forge-gpu-run` without `FORGE_GPU_DRY_RUN=1`.
- Builds use at most `-j4` under `nice -n 19`. Stop if MemAvailable falls below 12 GB or `/mnt/ForgeRealm` below 40 GB
  free.
- Your final message is the report; also write `artifacts/eng3f3/REPORT.md`.
- **Prior art:** follow `AGENTS.md`.

## Registration checklist (order-lint)

- **L01:** cmake, nvcc, pytest and python3, each run with `--version` first.
- **L02:** staged GPU lines carry an explicit `--min`.
- **L03:** n/a. No share is gated.
- **L09:** G2 compares per test ID. Logs and timings are never compared.
- **L10:** n/a. G2 holds outcomes to Build G.
- **L12:** inherited: ENG3-F2's G2 per-ID identity, with the listed exceptions.
- **L14:** n/a. No gain gate.
- **L16:** the cheapest pass would move the tests without running them. Excluded: G3 runs each moved killing test
  against its mutant, from `tensor_cuda/tests`.
- **L19:** n/a. No array identity is compared.
- **L32:** n/a.
- **L35:** your registrations may fix cases. They may not narrow a gate.
- **L38:** existing tests in `test_eng3_host.py` are not edited, except F06's soname line.
- **L40:** pytest runs with `-p no:cacheprovider`, `--basetemp` and `TMPDIR` under `/tmp/eng3f3_*`.
- **L41:** each mutant's killing test executes the mutated line.
- **L45:** the lead records each staged line's exits and log sha256.
- **L47:** the mutants' expected failures come from this order.
- **L48:** staged lines run as written, from the worktree root, with the environment they state.
- **L56:** (a) tool slips go under Deviations; (b) at most 3 repairs of your own scripts.
- **L58:** CPU commands run with `CUDA_VISIBLE_DEVICES` empty.
- **L65:** no deletes.
- **L67:** builds go to `tensor_cuda/build-eng3f3/`. Stop below 40 GB free.
- **L69:** one reference: test outcomes come from Build G (G2).
- **L75:** the three tests are copied into `tensor_cuda/tests`. Their old file under `artifacts/eng3f1/tools/` stays in
  place, unchanged and readable, so nothing that reads it breaks.
- **L79:** staged lines' dry runs stop before the lease. Each launcher states its full environment.
- **L82:** `grep -n 'core_device' tensor_cuda/src/bindings.cpp` finds the parser (2660) and its three callers (2697, 2709,
  2727), all in lines 2654 to 2735. They are writable. Every other line of `bindings.cpp` is read-only.
- **L99:** the lead's readings:
  - REV3's `findings.json` lists F01 to F06, with F02 at `bindings.cpp:2680`;
  - REV3's report gives the three moved tests' names;
  - Build G's sha is `5fcd179b98b0749af0d75414d35ae0fc8519bf200364c30c743904a90fc1a711`.
- **L103:** each staged line refuses, before the lease, a module whose sha256 differs from its build receipt.
- **L104:** n/a.
- **L109:** helpers run as `python3 -B -P`; none shares a standard-library module name.
- **L111:** n/a.

## Done

Your final message gives, verbatim:
- G1, G3 and G4, each PASS or FAIL;
- G2's CPU result, with its listed exceptions;
- the staged GPU lines exactly as the lead should run them, with their expected exits;
- the diff summary;
- the module sha256s;
- the deviations;
- every file written.
