# ENG3-F1: fix ENG3-REV1's five findings (Codex; CPU round with a build; GPU lines staged for the lead)

Follows: ENG3_HOST_TENSOR_CORE.md

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng3`
(branch `eng3`, at the commit carrying this order; ENG3's code and receipts are `9c6a8f5e`). Edits, CPU runs and engine
builds are AUTHORIZED, limited to:
- `tensor_cuda/src/kernels.cu`, only the `Storage` allocating constructor and the `NDArray` constructor
  (`kernels.cu:174-197` and `:531-534` at `9c6a8f5e`), and the ENG3 host block's comments (F03);
- `tensor_cuda/src/bindings.cpp`, only the four private `_core_*` entries (`bindings.cpp:2657-2684` at `9c6a8f5e`);
- `tensor_cuda/tests/test_eng3_host.py` (ENG3's own test file), and new `tensor_cuda/tests/test_eng3f1_*.py`;
- new `scripts/eng3f1_*.py`, everything under `artifacts/eng3f1/` (new), and the build directory
  `tensor_cuda/build-eng3f1/`;
- `logs/eng3f1_*`, and scratch under `/tmp/eng3f1_*`. Point `TMPDIR` there for every command.

**Read-only:** every other file here, including `artifacts/eng3/` and `tensor_cuda/build-eng3/` (Build B and Build E);
`tensor_fork_apa_wt_eng3rev1` (the review's receipts); every other worktree; the production repository.
**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively; search inside the named worktrees.

## Why

ENG3 (`9c6a8f5e`) gave host tensors CUDA-free core paths. Its blind Codex review (ENG3-REV1, receipts `cecee864` in
`tensor_fork_apa_wt_eng3rev1`) returned RED with 1 MAJOR and 4 MINOR findings. The lead confirmed F02 on the GPU:
`cuda:2147483647` yields `cuda:0` tensors. Read `artifacts/eng3_rev1/findings.json` and `REPORT.md` there first.

## Lead rulings (binding)

- **R1 (F01, MAJOR).** The allocation byte size is `numel_of(shape) * dtype_size(dtype)`, formed today with no overflow
  check (`kernels.cu:533`). Host `std::malloc` failure is not checked (`kernels.cu:194`).
  - Form the byte size with a checked multiply. A product that overflows `size_t` (or `int64`) raises before any
    allocation, with an error that names the shape and dtype.
  - A failed host `malloc` (null for a nonzero size) raises `MemoryError` (via `std::bad_alloc`) before any write.
  - This applies to host and device storage alike. A device allocation whose byte size used to wrap is now refused,
    which is the one intended change to device behaviour.
  - The private hooks also validate their shape: every extent ≥ 0, and the rank within what `Shape` and the engine's
    kernels accept. Find that limit from the code, or refuse rank above 8 if no limit exists, and say which.
- **R2 (F02, MINOR).** The private hooks parse device strings strictly:
  - exactly `"cpu"`, `"cuda"`, or `"cuda:N"` where N is the current device, the public rule at `__init__.py:193`;
  - anything else raises `ValueError` naming the hook and the string, before any allocation.

  `Device::from_string` itself is not changed in this round: it is production behaviour used elsewhere, and the
  public Python API guards it. Record in the report that it drops an explicit CUDA index.
- **R3 (F03, MINOR).** The NaN claims in code comments and the report become measured statements:
  - the rules were measured on sm_89 builds, over the probe's census (all 2,046 half NaNs; 1,099 float NaN words);
  - host and device NaN bits agree on that census and that architecture;
  - no claim is made for other architectures. The default CMake also builds sm_90.

  No code change beyond comments.
- **R4 (F04, MINOR).** State the LD_PRELOAD counter's scope in its test docstring and in the report: it counts calls
  through the module's global libcudart imports after import. It cannot see direct-handle, driver-API or
  other-library calls. Also add a second, independent check: the host-path tests also pass in a process where the
  CUDA runtime cannot initialise (`CUDA_VISIBLE_DEVICES` empty), any CUDA error fails the test, and
  `cudaGetLastError` is clean after the host calls where it can be queried.
- **R5 (F05, MINOR).** Every identity comparison in `test_eng3_host.py` checks both operands informative (nonzero and
  non-constant, at full width) before comparing, per L19, including the L55 constructed instances and `full`'s
  constant outputs. Where a constant output is the point of the test (`full`), compare against an independently built
  NumPy constant and check that reference's bits, then say so.
- Nothing else changes.

## Gates

- **G1, CPU** (`CUDA_VISIBLE_DEVICES` empty, Build F, this round's build): ENG3's `test_eng3_host.py` CPU rows, plus new
  tests, all pass:
  - `_core_full([2**62], …)`, `_core_zeros([2**62])` and a byte-size overflow for every dtype raise before allocating;
    no process dies;
  - an unallocatable but valid shape raises `MemoryError` in a child process with an address-space limit
    (REV1's receipt 028 method), and the child exits 0;
  - the R2 strings refuse (malformed, `cpu:garbage`, `cpuBAD`, `cuda:2147483647`), and a valid `cuda`/`cpu` string
    still works where no GPU is needed;
  - R5's guards are present and executed.
- **G2, GPU, staged for the lead:** ENG3's G2 (host vs device cast bits) and G1b (bool numpy) lines re-run on Build F,
  plus R2's device-index refusal on the GPU. Expected: as ENG3's, plus R2 refuses `cuda:2147483647`.
- **G3, regression, staged for the lead:** ENG3's G3 suite on Build F against Build E, per test ID, CPU and GPU.
  Identical except a listed set of intended changes, each with its ruling (expected: none, or tests that allocated a
  wrapping size).
- **G4, mutants** (named assertions in `artifacts/eng3f1/MUTANTS.json`, written before any mutant runs):
  - `f1m01_bytes_unchecked`: the checked multiply removed (G1's overflow test);
  - `f1m02_malloc_unchecked`: the null check removed (G1's `MemoryError` child test);
  - `f1m03_device_lenient`: the strict device parse removed (G1's R2 test).
  Mutants are built incrementally in a copied tree under `tensor_cuda/build-eng3f1/`.
- **G5, scope:** writes only in the writable set. No git writes, nothing deleted, no GPU.

**Registered acceptance:** ACCEPT when G1, G4 and G5 pass and the staged G2/G3 lines pass when the lead runs them. A
narrow review of this diff follows only if a finding is MAJOR or above; NITs and MINORs close with this round.

## Builds

- **Build E** is ENG3's (`tensor_cuda/build-eng3/E`, module `9f070b6f…`), read-only.
- **Build F** is this round's tree. Build it out of tree under `tensor_cuda/build-eng3f1/F` with the engine's CMake, as
  ENG3 did (its `artifacts/eng3/build/` receipts record the commands).
  - Use at most `-j4`, under `nice -n 19`.
  - Your sandbox has no `systemd-run`, so record peak memory with `/usr/bin/time -v`, and stop if `MemAvailable` falls
    below 12 GB.
  - Each build receipt records the module sha256 and a sha256 map of every engine source.
- Stage each GPU line as a script that checks the module sha before the lease and runs on CPU up to its first device
  call, with a receipt (REV2's lesson). Give each line exactly as the lead should run it:
  `forge-gpu-run --min 30 --max <≤1800> --label eng3f1-<line> -- timeout <≤1750>s ...`, with its expected exit.

## Mechanism & scope

- **Measures:** whether ENG3-REV1's five findings are closed without changing any device result except refusing
  wrapping allocations.
- **Mechanism doc:** `orders/ENG3_HOST_TENSOR_CORE.md` and its two amendments, with ENG3-REV1's `findings.json`.
- **Exercises:** the allocation constructors, the four private hooks, the test file, Build F plus three mutant builds,
  CPU tests, and staged GPU lines.
- **Does not cover:** `Device::from_string` outside the hooks; the 172 compute entries (HOST-COMPUTE-HMM); a CPU
  backend; performance; promotion.
- **Operand census:** every allocating constructor; the four hooks; every identity comparison in `test_eng3_host.py`.
- **Baseline isolation:** Build E is the reference for G3. ENG3's receipts are read-only.
- **Required evidence:** the diff; Build F and mutant build receipts with module sha256s; G1's CPU result; the staged
  G2/G3 lines with expected exits and their CPU-prefix receipts; `MUTANTS.json` and `MUTANT_RESULTS.json`;
  `REPORT.md`.
- **Forbidden configuration:**
  - any device-result change other than refusing wrapping sizes;
  - a change to `Device::from_string`;
  - edits outside the writable set;
  - GPU use;
  - git writes;
  - deleting anything yourself.

## Seat

Codex gpt-6.1-sol, xhigh effort (SC's 09:20 rule: fixes go to Codex). ENG3 was Claude's, so the review was
cross-model. This round's own changes are small; the lead verifies on disk and runs the GPU lines.

**Rules:**
- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- No git writes (read-only git is allowed). No subagents. No background waits. Every command under 30 minutes (builds
  under 60), with `timeout`.
- **No GPU.** Never call `forge-gpu-run` without `FORGE_GPU_DRY_RUN=1`.
- **No deletes.** Never run `rm`, `rmdir`, `unlink`, `git clean`, `find -delete` or an inline delete call. Leave and
  list.
- Never kill a process you did not start.
- **Your final message is the report.** Also write `artifacts/eng3f1/REPORT.md`.
- **Prior art:** follow the directive in `AGENTS.md` (checked arithmetic, ISO C++17 `[expr]/4`; CWE-190 integer
  overflow; CWE-252 unchecked return value).

## Registration checklist (order-lint)

- **L01:** the instruments are cmake, nvcc, pytest, python3 with NumPy, and `forge-gpu-run` (dry run only). Run each
  with `--version` first.
- **L02:** staged GPU lines carry an explicit `--min`.
- **L03:** n/a. No share is gated.
- **L04:** no fallback. R1 and R2 refuse.
- **L07:** n/a. No prediction.
- **L09:** G3 compares per test ID; any bitwise check uses same-width unsigned views after dtype and shape checks. No
  build-identity gate is registered.
- **L10:** device results are held to Build E by G3. Casts are held to the device by G2.
- **L12:** the inherited gates are ENG3's G1, G1b, G2 and G3, unchanged, plus R1's intended refusal of wrapping sizes.
- **L14:** n/a. There is no gain gate.
- **L32:** n/a. No bounded buffer and no sizing figure. "Overflow" in R1 is integer overflow of a byte count.
- **L69:** one reference per operand class:
  - device results and test outcomes -> Build E (G3);
  - host casts -> the device cast on the same bits (G2, ENG3's form);
  - refusals -> this order's rulings (G1).

  "Nothing else changes" is checked by the diff.
- **L16:** the cheapest passing work would check only the hooks and leave the constructor unchecked. Excluded: R1's
  test calls the public and private paths, and `f1m01` removes the constructor's check.
- **L19:** R5 is this lesson; every identity comparison checks both sides first.
- **L21:** the remote is public. No personal names, no raw quotes, and home paths as `$HOME/...`.
- **L25:** G2 and G3 keep ENG3's same-process and same-options rules.
- **L35:** your registrations may fix cases and seeds. They may not narrow a gate. If a clause cannot hold, stop and
  report RED.
- **L38:** only ENG3's own test file changes, for R5.
- **L40:** pytest runs with `-p no:cacheprovider`, `--basetemp` and `TMPDIR` under `/tmp/eng3f1_*`.
- **L41:** each mutant's killing test must execute the mutated line (a counter or a log line).
- **L42:** one full build took 110 s at `-j6` in ENG3. Four builds (F and three mutants) fit easily in the 6-hour
  ceiling.
- **L45:** the lead records each staged line's `forge-gpu-run` exit, inner exit and log sha256.
- **L47:** mutants' expected failures come from this order.
- **L48:** staged lines run as written, from the worktree root.
- **L55:** R1's check: show one constructed shape per dtype whose byte size overflows, and one that is exactly at the
  limit and does not.
- **L56:** (a) a slip in running your own tools goes under Deviations; (b) an exception in your own scripts is repaired
  at most 3 times, each recorded.
- **L58:** CPU commands run with `CUDA_VISIBLE_DEVICES` empty.
- **L60:** each new refusal is recorded in `artifacts/eng3f1/BOUNDARY.json` with Build E's behaviour on the same input
  (from the review's receipts where it would crash; never crash the parent process).
- **L64:** reads are this worktree, the review worktree and ENG3's build folder.
- **L65:** no deletes, as stated.
- **L67:** builds go to `tensor_cuda/build-eng3f1/`. Stop if `/mnt/ForgeRealm` has under 40 GB free.
- **L79:** staged lines' dry runs stop before the lease.
- **L82:** the writable product sites are named at the top, by line, at `9c6a8f5e`.
- **L99:** the lead's readings at `9c6a8f5e`:
  - `kernels.cu:533`: `numel_of(shape_) * dtype_size(dtype_)` passed to `Storage`;
  - `kernels.cu:194`: `ptr = std::malloc(nbytes);` with no check;
  - `kernels.cu:63-67`: `Device::from_string` sets index 0 for any non-`cpu` string;
  - `bindings.cpp:2657-2684`: the four hooks, which check only negative extents;
  - the lead's GPU run confirmed F02 (`lead_eng3rev1_gpu_repro.json` in the review worktree).
- **L103:** each staged line refuses, before the lease, a module whose sha256 differs from its build receipt.
- **L104:** n/a. No ratio bar.
- **L109:** your helpers run as `python3 -B -P`; none shares a standard-library module name.
- **L111:** n/a.

## Done

Your final message gives, verbatim: G1, G4 and G5, each PASS or FAIL; the staged G2 and G3 lines exactly as the lead
should run them, with expected exits; the diff summary; every module sha256; the mutant table; the deviations; and every
file written.
