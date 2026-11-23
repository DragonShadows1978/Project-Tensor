# ENG4-F3: the two new zero fills check cudaMemset's own status (ENG4-REV3 F01)

Follows: ENG4_F2_REV2_HARDENING.md
ID reuse: ENG4's fix rounds (F1, F2, F3) continue the ENG4 lane and share its id

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng4f3`
(branch `eng4f3`, from `9b7eb1796`, at the commit carrying this order). Edits, builds and CPU runs are AUTHORIZED,
limited to:
- `tensor_cuda/src/matmul.cu` (the forward's K = 0 path, plus one helper beside it);
- `tensor_cuda/src/ops.cpp` (only the empty-batch shared-B branch of `Tensor matmul`'s gradient);
- `tensor_cuda/include/tc/core.h` (only the helper's declaration);
- `tensor_cuda/tests/test_eng4_matmul.py`;
- new `scripts/eng4_f3_*.py`, everything under `artifacts/eng4/f3/`, `tensor_cuda/build-eng4/` in this worktree,
  `logs/eng4_f3_*`, scratch under `/tmp/eng4f3_*`.

**Read-only:** everything else here; `tensor_fork_apa_wt_eng4` (the lead's GPU suite lines are running there now; never
write in it, and never run anything from it); `tensor_fork_apa_wt_eng4rev3/artifacts/eng4_rev3/`; the production
repository. Build B and F2 module copies may be read from `tensor_fork_apa_wt_eng4/tensor_cuda/build-eng4/{B,F2}/` by
path, checked by sha256.

## GPU: you run none, not even as a smoke

You run no GPU work at all: no staged line, no step function, no compute-sanitizer (it removes
`CUDA_VISIBLE_DEVICES` from the program it runs, so it reaches the GPU), and nothing that removes or overrides
`CUDA_VISIBLE_DEVICES`. A smoke of a staged line is its `FORGE_GPU_DRY_RUN=1` dry run (exit 78) only.

## Why

ENG4-REV3 (Codex, run 20261123T151754Z-103659, clean exit; `findings.json` sha256 `f1243879`) confirmed every REV1 and
REV2 finding closed, with no blocker. It raised two MINOR error-path findings:
- **F01 (this round):** the forward's K = 0 path (`matmul.cu`) and the gradient's zero dB (`ops.cpp`) call
  `NDArray::zeros`, which discards `cudaMemset`'s returned status. They then read the thread's last-error slot with
  `cuda_check_last`. So a failed fill is seen only through the slot, and an unrelated earlier error can be raised
  under the fill's label. The gradient branch checks the slot even when B has no elements and no fill ran.
- **F02 (not this round):** the first gradient accumulation clones `zero_b` through a core copy whose `cudaMemcpy`
  status is unchecked (`kernels.cu`). That is an inherited core defect, recorded for a later engine round.

## The fix (binding)

- **Z1.** Add one helper beside the forward, for example
  `NDArray zeros_checked(const Shape&, DType, Device, const char* where)`, declared in `core.h`:
  - it allocates the array as `NDArray::zeros` does;
  - zero bytes: no fill and no CUDA call;
  - host: `std::memset`;
  - CUDA: `cudaMemset`, raising on a non-success *return* status and naming `where` and the status.

  It does not read or clear the last-error slot. `NDArray::zeros` itself is unchanged.
- **Z2.** The forward's K = 0 path and the gradient's empty-batch zero dB both use it, in place of
  `NDArray::zeros` plus `cuda_check_last`.
- **Z3.** Nothing else changes. No knob.

## Gates

- **K1, CPU** (`CUDA_VISIBLE_DEVICES` empty), on F3: every test in `test_eng4_matmul.py` that passes on F2 passes on F3.
  This includes the host K = 0 and host empty-batch zero-gradient tests, and a host case with an empty B (no fill).
- **K2, clean checkout,** as F2's J2.
- **K3, CPU suite per ID,** F2 against F3: identical outcomes.
- **K4, the pending-error witness.** Find whether a pending, non-sticky CUDA error can be left in the slot through the
  engine's public API or its own runtime, for example a call that fails cleanly and returns.
  - If one exists, stage a GPU line: with that error pending, a K = 0 matmul raises "matmul zero fill" on F2 and returns
    zeros on F3.
  - If none exists, say so with the reading, and record the status check as verified by reading only.
- **K5, a mutant** `m11_f3_status_ignored` (the helper ignores `cudaMemset`'s status), written in
  `artifacts/eng4/f3/MUTANTS_F3.json` before any run. It can fail only if a fill can be made to fail. If no test can see
  it, record it as unkillable by test, with the reason.
- **K6, staged GPU lines** in `artifacts/eng4/f3/GPU_LINES_F3.json`, in F2's form, dry-run once (78) each:
  - `own_F3`: `test_eng4_matmul.py` on the GPU;
  - `g3z_F3`: the G3 zero-size cells and the 24 empty-batch gradient cases, against NumPy and zeros;
  - K4's witness, if one exists.

  Each line's step refuses unless forge-gpu-run is an ancestor process, as F2's `lease_guard()` does.
- **K7, scope:** writes only in the writable set; no GPU use of any kind; no git writes; nothing deleted.

**Registered acceptance:** the seat's part passes when K1 to K3, K5 (as registered) and K7 pass and K6 is staged and
dry-run clean. The lead verifies, runs the lines, and closes REV3-F01. No re-review.

## Mechanism & scope

- **Measures:** whether both new fills check `cudaMemset`'s own status, without reading the last-error slot, and
  without changing any result.
- **Mechanism doc:** `orders/ENG4_F2_REV2_HARDENING.md` (Y2, Y3) and this order (Z1 to Z3).
- **Exercises:** the helper, the forward's K = 0 path, the gradient's empty-batch branch; Builds F2 (copied, by sha)
  and F3, plus one mutant arm; CPU tests; a clean-checkout copy; staged GPU lines.
- **Does not cover:** `NDArray::zeros` and other core members (REV3-F02's copy is a separate item); the forward's
  current-device check; HOST-COMPUTE-HMM; performance; promotion.
- **Required evidence:** the F3 diff against `9b7eb1796`; build receipts with module sha256s; K1 to K5 results;
  `GPU_LINES_F3.json` with dry-run exits; `artifacts/eng4/f3/REPORT_F3.md`.
- **Operand census:** both fill sites; zero, host and CUDA sizes; the K4 search over public API calls that can leave a
  pending error.
- **Baseline isolation:** Build F2 for results (K3).
- **Forbidden configuration:** changes to `NDArray::zeros` or other core members; any result change; any GPU use;
  writes in `tensor_fork_apa_wt_eng4`; git writes; deleting anything yourself.

## Machine rules

One CPU-busy process at a time at `nice -n 19`. Builds and jobs over 1 GB run in their own scope with
`MemoryMax=12G` and `OOMPolicy=continue`, started only with `MemAvailable` ≥ 12 GB. Stop under 40 GB free on
`/mnt/ForgeRealm`. Ceiling: 2 hours of wall time.

## Registration checklist (order-lint)

- **L01:** the instruments are cmake, nvcc, pytest, python3 with NumPy, and `forge-gpu-run` (dry runs only). Run each
  with `--version` (or `--status --min 30 --max 540`) first.
- **L02:** GPU lines carry an explicit `--min`.
- **L04:** no fallback. A failed fill raises.
- **L06:** n/a. No count threshold; `MemAvailable` ≥ 12 GB is a memory floor.
- **L09:** `g3z_F3` compares against NumPy: empty arrays by shape and dtype, zeros by value.
- **L10:** results are held to identity with F2 (K3 per test ID; `g3z_F3` by value).
- **L12:** inherited gates: F2's J1 tests and the suite per ID.
- **L16:** the cheapest passing work would wrap the existing `cuda_check_last` in a helper. Excluded: Z1 requires the
  helper to check `cudaMemset`'s return status and not read the slot. The diff is read for that, and K4 tests it where
  a pending error can be made.
- **L18:** n/a. No owner requirement behind an option.
- **L19:** zero-value checks use L09's rule.
- **L21:** public remote: no personal names, no raw quotes, home paths as `$HOME/...`.
- **L25:** `g3z_F3` runs F3 alone against NumPy; no cross-build bitwise comparison.
- **L32:** n/a. No bounded buffer.
- **L35:** registrations may fix cases; they may not narrow a gate. If a clause cannot hold, stop and report RED.
- **L37:** n/a. No knob.
- **L38:** existing tests unchanged except `test_eng4_matmul.py`, listed.
- **L40:** pytest with `-p no:cacheprovider`, `--basetemp` and `TMPDIR` under `/tmp/eng4f3_*`.
- **L41:** the mutant names its expected failing assertion before it runs, or is registered as unkillable with the
  reason.
- **L45:** staged lines record `forge-gpu-run`'s exit, the inner `timeout` exit and the log's sha256.
- **L47:** the mutant's expected failure comes from this order.
- **L48:** each staged line runs as written, through its `run` wrapper, from this worktree's root. K1 drives each
  line's post-device steps on constructed returns, listed in `artifacts/eng4/f3/POST_DEVICE_STEPS.json`.
- **L55:** Z1 holds only if both sites pass the same out shape, dtype and device as before. Show both call sites and
  their arguments.
- **L56:** (a) slips go under Deviations; (b) an exception in your own scripts is repaired at most 3 times, each
  recorded.
- **L58:** CUDA_VISIBLE_DEVICES overrides: census `grep -n "CUDA_VISIBLE_DEVICES" scripts/eng4_f3_*.py`, list each site
  and its guard (inside forge-gpu-run only, behind the ancestor check), and never run one outside it. No
  compute-sanitizer.
- **L67:** this round adds two module arms (46–65 MB each, measured) plus receipts, under 0.3 GB. Stop at 40 GB free,
  checked before each build.
- **L76:** the lead writes nothing in this worktree while the seat runs.
- **L77:** the mutant is run, or registered as unkillable with its reason.
- **L79:** `FORGE_GPU_DRY_RUN=1` returns 78. Use it once per staged line.
- **L69:** test outcomes -> Build F2 per test ID (K3); excluded from `g3z_F3`. Zero-size results and the empty-batch
  gradients -> NumPy and exact zeros (`g3z_F3`); excluded from K3's identity clause, which compares outcomes only.
- **L81:** REV3-F01's claim comes from source reading (its `evidence_class`). Source records: the code paths only, no
  run. The order asserts nothing beyond them, and K4 measures where it can.
- **L99:** the lead's readings at `9b7eb1796`:
  - `matmul.cu`: the K = 0 path is `NDArray zeros = NDArray::zeros(...)`, then
    `if (a.device.is_cuda()) cuda_check_last("matmul zero fill")`;
  - `ops.cpp`: the empty-batch branch is `NDArray zero_b = NDArray::zeros(...)`, then
    `if (zero_b.device.is_cuda()) cuda_check_last("matmul backward zero gradient")`;
  - `kernels.cu`: `NDArray::zeros` calls `cudaMemset` without keeping its status.
- **L103:** each staged line refuses, before the lease, a module whose sha256 differs from its build receipt.
- **L104:** no ratio bar; every gate is exact.
- **L109:** helpers run as `python3 -B -P`; none shares a standard-library module name.
- **L114:** big jobs in the seat's own scope (Claude seat). Staged lines set no RLIMIT_AS; the lead runs them in a
  `MemoryMax=12G` scope; `GPU_LINES_F3.json` lists each line's expected peak host bytes.
- **L118:** the helper's consumers are the two sites named; nothing else calls it.
- **L119:** n/a. The diff is product code and its test.

## Rules

- This order is immutable. NO git writes (read-only git, including `git archive` for K2, is allowed). NO subagents. No
  background waits; every command under an explicit timeout.
- No deletes, in the leave-and-list form. Never kill a process you did not start.
- **Report:** write `artifacts/eng4/f3/REPORT_F3.md`. If the write is refused, your final message is the report.
- **Prior Art Directive:** annotate each code site and the report. Known: the CUDA Runtime API's per-call return status
  and its separate last-error slot (CUDA 12.6 Runtime API); TensorCUDA's existing per-call status checks (`ops.cpp`).

## Done

Your final message gives, verbatim: K1 to K7 (PASS, FAIL or STAGED); the diff summary; every module sha256; the mutant
entry; the K4 finding; each staged line with its dry-run exit; the L58 census; deviations; every file written.
