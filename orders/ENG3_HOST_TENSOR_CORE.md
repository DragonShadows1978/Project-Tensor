# ENG3: host tensors without a GPU: the NDArray core's host paths never call CUDA (implementing round)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng3`
(branch `eng3`, from production main `2478a4d`, at the commit carrying this order). Edits, builds, CPU runs and GPU runs
through `forge-gpu-run` are AUTHORIZED, limited to:
- `tensor_cuda/src/kernels.cu`, only the NDArray core section (the constructors, `empty`, `zeros`, `full`,
  `from_host`, `to_host`, `clone`, `to`, `astype`, and any other core member your census lists), plus host helpers
  you add beside them;
- `tensor_cuda/include/tc/core.h`, only declarations of host helpers you add;
- `tensor_cuda/tensor_cuda/__init__.py`, only if a census row needs a Python-side change, each ledgered;
- the new `tensor_cuda/tests/test_eng3_host.py`, new `scripts/eng3_*.py`, everything under `artifacts/eng3/`;
- the build directory `tensor_cuda/build-eng3/` and the built module it installs into this worktree's package;
- `logs/eng3_*`, and scratch under `/tmp/eng3_*`. Point `TMPDIR` there for every command.

**Read-only:** every other file here, every other worktree, the production repository, Project-Voice and GRAPA.
**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively; search inside this worktree.

## Why

DS1-A5 found two engine defects on 11-14, and they are open in production (`2478a4d`):
- **HOST-TENSOR-NEEDS-DEVICE.** `NDArray::from_host` calls `cuda_check_last` after a host `memcpy`
  (`kernels.cu:679-687`). With no visible GPU, no tensor can be built, host or device. Every CPU test of this engine
  needs stubs for that reason.
- **HOST-ASTYPE-HMM.** `astype` on a host tensor launches `cast_kernel` on `malloc`'d host memory
  (`kernels.cu:727-756`). It works here only because this machine's driver enables HMM. On a driver without HMM it
  faults.

The lead's reading found the same pattern in more core members of a host tensor:
- `zeros` calls `cudaMemset` (`kernels.cu:577`);
- `full` launches `fill_kernel` (`kernels.cu:580-587`);
- `clone` calls `cudaMemcpy(..., cudaMemcpyDeviceToDevice)` (`kernels.cu:695-700`);
- `to`, between two host devices, goes through `clone` (`kernels.cu:701-708`).

## Lead rulings (binding)

- **E1, census first.** List every NDArray core member, and every public Python entry, that a host (`device="cpu"`)
  tensor can reach and that calls a CUDA API or launches a kernel on its storage. Give file:line and what it calls.
  Write `artifacts/eng3/HOST_CENSUS.json`. The members named under Why are the starting list, not the census.
- **E2, the host data path.** For a host tensor, every data-movement and creation member in the census has a host path
  that calls no CUDA API: `from_host`, `to_host`, `zeros`, `full`, `clone`, `to` (host to host) and `astype`. It uses
  `std::memcpy`, `std::memset` or a CPU loop. Device tensors keep today's path, unchanged. A host-to-device or
  device-to-host `to` keeps its `cudaMemcpy`.
- **E3, host `astype` matches the device bit for bit.** For every dtype pair the engine supports, the host cast gives
  the same bits as the device kernel on the same values. That means the same rounding (round-to-nearest-even for
  float→half/bfloat16), the same NaN bits, the same overflow-to-Inf, and, for uint8, the same `rintf` round and [0, 255]
  clamp. Derive each rule from the device code (`ld<T>`, `st<T>`, `cast_kernel`, `cast_to_u8_kernel`,
  `cast_from_u8_kernel`), not from a library.
- **E4, any other census member** that a host tensor reaches and that launches device work gets a clear refusal on a
  host tensor, the way `fill_` already refuses (`kernels.cu:643`: "CUDA arrays only"), naming the member. It is not
  given a host implementation in this round.
- **E5.** Device-tensor results stay bitwise unchanged (G3). No knob is added.

## Gates

- **G1, CPU only** (`CUDA_VISIBLE_DEVICES` empty, so any CUDA call fails). Through the public Python API, on every
  supported dtype:
  - `tc.tensor(..., device="cpu")`, `.numpy()`, `zeros`, `full`, `clone`, `.to("cpu")` and `.astype(...)` all work;
  - each returns values equal to a numpy reference built by E3's rules;
  - E4's members refuse on a host tensor.
- **G2, host `astype` equals device `astype` bitwise** (GPU line). For every supported dtype pair, on a value census:
  - ±0, the smallest and largest subnormals and normals of each source type;
  - ±Inf, and NaN (the quiet NaN pattern your engine produces, plus one with a payload);
  - rounding ties of each destination type, and values just past each destination's overflow threshold;
  - for uint8: −1, −0.5, 0.5, 1.5, 254.5, 255.5 and 256;
  - 10,000 seeded random finite values per source type.

  Compare the host tensor's bits with the device tensor's bits (copied to host), per element, on same-width unsigned
  views.
- **G3, no regression.** Production's test suite, per test ID, before (Build B, this branch as committed) and after
  (Build E): identical outcomes, except a listed set of intended changes, each with its ruling. Run it on CPU and as a
  GPU line. Also add a bitwise check of the device paths of every E2 member on one shape per dtype (Build B against
  Build E).
- **G4, mutants** (validity of the new tests), each failing its named assertion in `artifacts/eng3/MUTANTS.json`,
  written before any mutant runs:
  - `m01_from_host_checks_cuda`: the host path of `from_host` calls `cuda_check_last` again (G1);
  - `m02_host_astype_truncates`: the host float→bfloat16 cast truncates instead of rounding to nearest even (G2);
  - `m03_host_clone_cudamemcpy`: host `clone` goes back to `cudaMemcpy` (G1);
  - `m04_u8_no_clamp`: the host uint8 cast drops the clamp (G2).
- **G5, scope.** Writes only in the writable set. GPU only through `forge-gpu-run`. No git writes, nothing deleted.

**Registered acceptance:** ACCEPT when G1 to G5 pass. Then the lead verifies on disk, and a cross-model (Codex) review
follows. This is an infrastructure change: promotion needs the owner's verbatim approval and the review receipt.

## Builds

- **Build B:** this branch as committed (production `2478a4d`), default configuration.
- **Build E:** this round's tree.

Both are built out of tree with the engine's CMake under `tensor_cuda/build-eng3/`, as ENG2 built its two arms. ENG2's
`artifacts/eng2/LEDGER.md` records the commands, in `tensor_fork_apa_wt_eng2`. Each build receipt records the module
sha256 and a sha256 map of every engine source (`tensor_cuda/src/`, `tensor_cuda/include/`,
`tensor_cuda/tensor_cuda/`).

## Mechanism & scope

- **Measures:** whether a host tensor's core members work with no GPU and no HMM, bit-exact to the device casts, with
  device results unchanged.
- **Mechanism doc:** `tensor_cuda/src/kernels.cu` (NDArray core section, `kernels.cu:531-760`) and `tensor_cuda/include/tc/core.h`; there is no separate design doc for the core (none exists).
- **Exercises:** the NDArray core members in the census; host cast code; Builds B and E plus mutant builds; CPU tests
  with no visible GPU; GPU lines for G2 and G3.
- **Does not cover:**
  - host implementations of compute ops (E4 refuses them);
  - a CPU backend;
  - performance;
  - DLPack import and export of host tensors (census them; change them only if they call CUDA on host storage, then
    ledger it);
  - promotion.
- **Required evidence:** `HOST_CENSUS.json`; the diff; Build B, Build E and mutant build receipts with module sha256s;
  G1's CPU result; G2's per-pair result; G3's per-ID before/after and the device-path check; `MUTANTS.json` and
  `MUTANT_RESULTS.json`; each GPU line's `RESULT.json`, log, start and exit files; `REPORT.md`.
- **Operand census:** every core member and Python entry a host tensor reaches (E1); every dtype pair (E3); G2's value
  census.
- **Baseline isolation:** Build B is the reference for G3. Production is read-only.
- **Forbidden configuration:**
  - any change to a device-tensor result;
  - a host path that calls a CUDA API;
  - a library cast in place of E3's derived rules;
  - edits outside the writable set;
  - git writes;
  - deleting anything yourself.

## Machine rules

- **CPU:** at most one CPU-busy process of yours at a time, at `nice -n 19`. Builds and any job over 1 GB run in their
  own scope: `systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue -p CPUWeight=50 ...`. Start one only
  with `MemAvailable` at or above 12 GB. Another Tensor seat (P1-F3) may be building at the same time; never touch its
  worktree or processes.
- **Time:** every command under 30 minutes, except builds (under 60). Long commands run detached and are waited on in
  the foreground with `timeout <n> tail --pid`. The round's ceiling is 8 hours of wall time, with GPU use at most 1.5
  hours.
- **Disk:** stop if `/mnt/ForgeRealm` has under 40 GB free.
- **GPU:** only through `forge-gpu-run`, one line at a time:
  `forge-gpu-run --min 30 --max <≤1800> --label eng3-<line> -- timeout <≤1750>s ...`. Do one dry run with
  `FORGE_GPU_DRY_RUN=1` (returns 78) before each line. Exit 76 means nothing ran: retry unchanged after at least 5
  minutes, at most 6 times.

## Registration checklist (order-lint)

- **L01:** the instruments are cmake, nvcc, pytest, python3 with NumPy, and `forge-gpu-run`. Run each with `--version`
  (or `--status --min 30 --max 600`) first.
- **L02:** GPU lines carry an explicit `--min`.
- **L03:** n/a. No share is gated.
- **L04:** no fallback is added. E4 refuses; it does not copy to the device.
- **L07:** n/a. No prediction is registered.
- **L08:** builds go to `tensor_cuda/build-eng3/`, receipts to `artifacts/eng3/`, and scratch to `/tmp/eng3_*`.
- **L09:** G2 compares per element on same-width unsigned views, after dtype and shape checks. G3 compares test
  outcomes per ID. Logs and timings are never compared. The build comparison in G3 compares module outputs, not
  binaries, so no non-code build difference enters a gate.
- **L10:** the casts are deterministic stages, held to identity (G2). Device results are held to identity (G3).
- **L12:** the inherited gates are production's test suite per ID. Intended changes are listed before the after-run,
  each with its ruling (expected: tests that relied on HMM host casts or skipped for lack of a GPU).
- **L15:** n/a. No threshold is fitted.
- **L16:** the cheapest passing work would fix `from_host` and `astype` only and leave `zeros`, `full` and `clone`
  calling CUDA. Excluded by G1, which runs every E2 member with no visible GPU, and by `m03`.
- **L18:** n/a. No owner requirement is implemented behind an option. Host paths are not optional; device paths are
  unchanged.
- **L32:** n/a. No bounded buffer and no sizing figure. "Overflow" in E3 is floating-point overflow of a cast.
- **L37:** n/a. No knob, lever or spec key is added (E5).
- **L19:** every bitwise comparison first checks that each side is nonzero and non-constant, at full width.
- **L21:** the remote is public. No personal names, no raw quotes, and home paths as `$HOME/...`.
- **L22:** G2's value census is defined in this order; the random part uses a recorded seed.
- **L25:** G2 runs both sides in one process: the host cast on CPU and the device cast on the GPU, on the same input
  bits. G3 runs each build in its own process kind, with the same pytest options.
- **L34:** a Claude seat (Opus 5.5, high effort, Agent tool), because the round builds and runs GPU lines. The review
  that follows is Codex, so it is cross-model.
- **L35:** your registrations (`MUTANTS.json`, the census, the intended-change list) may fix cases and seeds. They may
  not remove, narrow or reclassify anything a gate covers. If a clause cannot hold as written, stop and report RED.
- **L38:** existing tests are not changed. A test that cannot pass after E2 is an intended change, listed with its
  ruling.
- **L40:** pytest runs with `-p no:cacheprovider`, `--basetemp` and `TMPDIR` under `/tmp/eng3_*`. The engine's
  `test_terrain_render.py` writes tracked `artifacts/wo*.ppm` files (TERRAIN-TEST-WRITES). Run the suite with those
  files restored afterwards from git (read-only git: `git show HEAD:<path>` into the file), or exclude that file and
  list it.
- **L41:** each mutant names its expected failing assertion in `MUTANTS.json`, written before it runs. Show that each
  killing test executes the mutated line.
- **L45:** each GPU line records `forge-gpu-run`'s exit, the inner `timeout` exit and the log's sha256. Exit 124 is a
  timeout, never a pass.
- **L47:** the mutants' expected failures come from this order, never from this round's runs.
- **L48:** every GPU line runs as written, from the worktree root, through the real `forge-gpu-run`.
- **L55:** E3's identity holds only if the host code reproduces each device rounding step exactly. For one constructed
  value per rule (a float→bfloat16 tie, a float→half subnormal, a NaN payload, a uint8 .5 tie), show the device rule
  from the code and the host result, then measure G2.
- **L56:** (a) a slip in running your own tools goes under Deviations: re-run and continue. (b) An exception in your own
  scripts is repaired at most 3 times, each recorded.
- **L58:** G1 runs with `CUDA_VISIBLE_DEVICES` empty. GPU commands run only inside `forge-gpu-run`.
- **L60:** for each E4 refusal, record in `artifacts/eng3/BOUNDARY.json` the call, the refusal, and Build B's behaviour.
- **L64:** reads are this worktree, and ENG2's ledger in `tensor_fork_apa_wt_eng2` for the build commands.
- **L65:** leave-and-list form for any file you would remove.
- **L67:** builds go to `tensor_cuda/build-eng3/`. Stop if `/mnt/ForgeRealm` has under 40 GB free (121 GB at
  sealing).
- **L69:** one reference per operand class:
  - host casts -> the device cast on the same bits (G2);
  - device results -> Build B (G3);
  - test outcomes -> Build B per ID (G3).
- **L74:** plain wording. These are engine data paths and their tests.
- **L76:** the lead writes nothing in the writable set while the seat runs.
- **L77:** every mutant can fail, and each is run.
- **L79:** `FORGE_GPU_DRY_RUN=1` returns 78 without a device check. Use it once per line before the real run.
- **L82:** `grep -n "cuda\|<<<" tensor_cuda/src/kernels.cu` over lines 531-760 starts E1's census.
- **L99:** the rulings rest on these lead readings, run before sealing, at `2478a4d`:
  - `kernels.cu:679-687`: `from_host` calls `cuda_check_last("from_host")` after a host `memcpy`;
  - `kernels.cu:727-756`: `astype` launches `cast_kernel` and the u8 kernels with no device check;
  - `kernels.cu:575-578` (`zeros`, `cudaMemset`), `:580-587` (`full`, `fill_kernel`), `:695-700` (`clone`,
    `cudaMemcpyDeviceToDevice`) and `:701-708` (`to`);
  - `kernels.cu:643`: `fill_` refuses non-CUDA arrays;
  - `__init__.py:58`: `tensor(data, *, device="cuda", ...)` accepts a device argument.

  Seat check that runs first: that `tc.tensor([1.0], device="cpu")` fails with no visible GPU on Build B (the defect,
  reproduced). If it does not, report that and continue with the census.
- **L102:** no generated file or registry depends on the writable files.
- **L103:** each GPU line refuses, before the lease, a module whose sha256 differs from its build receipt.
- **L104:** no ratio or fraction bar. Every gate is exact (per element, per test ID); slack 0 = 0 events.
- **L105:** n/a.
- **L109:** your helpers run as `python3 -B -P` (or `-I`). List every `.py` you wrote, and confirm none shares a
  standard-library module name.
- **L111:** n/a. No storage design is emulated.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes (read-only git is allowed). NO subagents. No background waits: run everything synchronously, each
  command under an explicit timeout.
- No deletes, in the leave-and-list form. Never kill a process you did not start.
- **The report:** write `artifacts/eng3/REPORT.md` early and keep it current. If the write is refused, say so; your
  final message is then the report, and the lead files it verbatim.
- **Prior Art Directive:** annotate each code site and the report. Known: IEEE 754-2008 conversions and
  round-to-nearest-even; the bfloat16 format (Google Brain); CUDA's `__float2half_rn` and `__float2bfloat16_rn`
  semantics (CUDA Math API); mutation testing (DeMillo, Lipton and Sayward, 1978). Write "no prior art known to me"
  where none applies.

## Done

Your final message contains, verbatim:
- G1 to G5, each PASS or FAIL;
- the census (summary) and every E4 refusal;
- the diff summary;
- every module sha256;
- G2's per-pair result;
- G3's per-ID result and intended changes;
- the mutant table;
- each GPU line's result;
- the deviations;
- every file written.
