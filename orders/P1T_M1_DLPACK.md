# P1T-M1: DLPack interop in TensorCUDA core (T1) for Project-Voice

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_m1`
(branch `voice-m1`, based on production main `9df5bef`). Edits, builds and GPU runs inside it are AUTHORIZED.

## Authority

David, 2026-10-12, in the Project-Tensor session, verbatim:
- "Voice needs Tensor Work done... But I don't want to build it directly INTO the core of TensorCUDA - It needs to be
  separate I guess. I don't want to bloat the kernels if we don't have too."
- "I'd follow your reccomended option": core gets only T1 (DLPack); T2–T6 go to a separate add-on package, which is
  NOT part of this order.
- "Make an alternative plan and mail it basck seeking approval for it". The plan is
  `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/docs/P1T_TENSOR_ALT_PLAN.md` (fork 326fdfe). Project-Voice
  APPROVED it on 2026-10-12 (their lane decision LD51) with amendment A1, which is in scope here.

## Background

Project-Voice is porting Kokoro-82M to run natively on TensorCUDA. Its own CUDA kernels (SineGen, STFT, iSTFT) and the
add-on package must read and write TensorCUDA device memory without copies. Today the Python layer has no
`data_ptr`, DLPack or `__cuda_array_interface__`, and TensorCUDA queues all work on the legacy default stream.
Project-Voice wrote and blind-reviewed acceptance tests for T1. They check:
- zero-copy pointer identity and write-through in both directions;
- device `(kDLCUDA=2, ordinal)`;
- owner lifetime after the producer drops its last reference;
- `__dlpack__(stream=)` ordering in both directions, with torch (CUDA) as the peer, over 20 trials each.

## Mechanism & scope

- **Measures:**
  - whether TensorCUDA tensors can be exported to, and imported from, any DLPack CUDA producer zero-copy, with correct
    owner lifetime and stream ordering;
  - whether this costs any existing behaviour: the FAILED/ERROR set of the core suite, and the alloc/free microbench.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Voice/research/kernels/TENSOR_IMPL_PLAN_P1T.md` §5 "T1: DLPack" (the
  spec), as amended by `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/docs/P1T_TENSOR_ALT_PLAN.md` §3. The
  protocol itself is DLPack v1.3 (`dlpack.h`) plus the Python array API standard's `__dlpack__`/`__dlpack_device__`
  stream contract.
- **Exercises:**
  - `tensor_cuda/include/tc/core.h` (Storage: external memory with a release callback);
  - `tensor_cuda/src/kernels.cu` (the Storage constructor/destructor region, and a host wrapper for an in-place fill
    that reuses the existing `fill_kernel` at :217);
  - `tensor_cuda/src/bindings.cpp`;
  - `tensor_cuda/tensor_cuda/__init__.py`;
  - a vendored `tensor_cuda/include/dlpack/dlpack.h`;
  - Project-Voice's runner `run_p1t.py --items T1`, through your adapter copy.
- **Does not cover:**
  - T2–T6 and the add-on package;
  - `__cuda_array_interface__`;
  - CPU-device DLPack export or import beyond a clear error;
  - multi-GPU;
  - non-contiguous import (refused, not supported);
  - autograd through imported tensors;
  - core's GroupNorm/LayerNorm;
  - edits to existing tests;
  - any kernel change.
- **Required evidence:**
  - the G1 report JSON and its verdict line;
  - the G2 suite log plus the compare output;
  - the G3 diff stat and `__global__` count;
  - the G4 mutant table (mutant, build path, .so sha256, test_dlpack result, G1 verdict, failure counts);
  - the G5 medians;
  - the built .so sha256 and size;
  - the `revision` reader list with file:line citations.
- **Operand census:**
  - **Dtypes:** fp32 and fp16 (required), plus any exact-match extras you choose.
  - **Shapes:** Voice's T1 cases (read the runner), and in `test_dlpack.py` at least 0-d, 1-d, 3-d [1, 512, 1193],
    a non-zero-offset view (a reshape or slice of a larger Storage, if the Python layer exposes one) and a size-1
    dimension with an arbitrary stride on import.
  - **Producers and consumers:** torch 2.11 cu130, on the default stream and on a side `torch.cuda.Stream()`.
- **Baseline isolation:**
  - G2 compares against the fixed 52-id baseline log named below.
  - G5 compares production's installed engine (d8fda9a8) with your build, in fresh processes, alternating order.
  - Mutants are built and run outside the worktree.
- **Forbidden configuration:**
  - `--use_fast_math` or any change to existing compile flags;
  - TF32 math modes or `NVIDIA_TF32_OVERRIDE`;
  - the retired 595.84 driver override;
  - any new `__global__` kernel;
  - any change to `round`/`U_ROUND` or to existing op semantics;
  - writing outside the target or `/tmp`.

## Spec (implement exactly; if Voice's tests demand otherwise, the tests win and you report the difference)

1. **`Tensor.__dlpack__(*, stream=None, max_version=None, dl_device=None, copy=None)`.**
   - It returns a PyCapsule named `"dltensor"`, or `"dltensor_versioned"` when `max_version >= (1, 0)`.
   - It is zero-copy. The data pointer plus `byte_offset` addresses element 0, and the NDArray `offset` is honoured.
   - The capsule's manager holds a `shared_ptr<Storage>`, so the memory outlives the producer's last Python reference.
   - The capsule follows the DLPack rules: the destructor of an unconsumed capsule calls the deleter, and a consumed
     capsule is renamed `used_*`.
   - **Stream contract:**
     - `stream=None` means the consumer is on the legacy default stream.
     - `stream=-1` means no synchronization.
     - Any other value is a consumer stream handle. Record a `cudaEvent` on the legacy stream and call
       `cudaStreamWaitEvent(consumer, event)` before returning. Legacy-to-legacy (None/1) needs nothing. For 2
       (per-thread default) and real handles you must wait.
   - `copy=True` and `dl_device` other than this device raise `BufferError`/`ValueError`, as the standard says.
2. **`Tensor.__dlpack_device__()`** returns `(2, ordinal)` for CUDA tensors. CPU tensors may raise `BufferError` with
   a clear message.
3. **`tensor_cuda.from_dlpack(obj)`** (also accept a bare capsule):
   - It calls `obj.__dlpack__(stream=1)`, so the consumer is the legacy default stream.
   - It wraps the pointer without copying, in a Storage that keeps the DLPack manager alive. When the Storage dies:
     1. **First** make sure TensorCUDA work queued on that memory has finished (sync the legacy stream, or wait on an
        event; justify your choice).
     2. **Then** call the producer's deleter with the GIL held.
     3. Never return the memory to TensorCUDA's pool and never `cudaFree` it.
   - It refuses, with a clear error: non-contiguous strides (size-1 dimensions may have any stride), non-CUDA devices,
     and dtypes TensorCUDA does not have. It never silently converts.
   - fp32 and fp16 are required. Other dtypes with an exact TensorCUDA match are allowed unless Voice's tests require
     their refusal.
4. **`Tensor.data_ptr()`** returns the device address of element 0, as an int.
5. **`Tensor.fill_(value)`** fills in place, queued on the legacy default stream, through the EXISTING `fill_kernel`.
   It adds no new `__global__`.
6. **Amendment A1 (Project-Voice):**
   - Expose `Tensor.mark_modified()`. It bumps `Storage::revision`.
   - Expose a read-only `Tensor.revision` if you need it for tests.
   - In the report, list every core op or cache that READS `Storage::revision` (file:line), so Project-Voice knows
     when an external in-place writer must call `mark_modified()`.
   - Document the rule in the `__dlpack__` docstring.
7. **Adapter:** copy
   `/mnt/ForgeRealm/Project-Voice/engine/tensor_requests/p1t_tests/adapters/tensorcuda_stub.py` to
   `artifacts/p1t_m1/tensorcuda_adapter.py`.
   - Make `t1_*` real.
   - Leave T2–T6 raising `NotImplementedError` (M2+ fills them).
   - Fix anything the census got wrong, and say what.
   - `t1_read` must synchronize only TensorCUDA's stream, never the whole device.

## Where to work and how to run

- **GPU:** every GPU command runs as `flock -w 900 /tmp/forge-gpu.lock timeout 600 …`.
  - Project-Voice released the lock at 19:20:48 EDT and has nothing queued, but always take the lock anyway.
  - If a lock wait times out, retry.
  - Use the native driver only (595.99.02). Never set the 595.84 override.
- **Build:**
  - `cd <target>/tensor_cuda && ./build.sh 89`. No GPU is needed. It drops `_tensor_cuda*.so` into
    `<target>/tensor_cuda/tensor_cuda/`.
  - Record the .so sha256 and size.
  - For mutant builds, copy the tree or build dir to `/tmp/p1t_m1_mut_<x>/` and build there. Never in the worktree.
- **Package path:** `PYTHONPATH=<target>/tensor_cuda/tensor_cuda:<target>/tensor_cuda`, plus `PYTHONDONTWRITEBYTECODE=1`.
- **Voice's acceptance run (G1).** Run it from `/mnt/ForgeRealm/Project-Voice`, which is READ-ONLY to you; every
  output goes to your target:
  ```
  cd /mnt/ForgeRealm/Project-Voice && flock -w 900 /tmp/forge-gpu.lock timeout 600 env PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=<target>/tensor_cuda/tensor_cuda:<target>/tensor_cuda python3 engine/tensor_requests/p1t_tests/run_p1t.py \
    --adapter <target>/artifacts/p1t_m1/tensorcuda_adapter.py --items T1 \
    --fixtures /mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures \
    --tolerances engine/tensor_requests/p1t_tests/tolerances_r2.json --out <target>/artifacts/p1t_m1/p1t_T1_report.json
  ```
  Read the runner's T1 section and `adapter_api.py` before you design anything. Do not edit any Project-Voice file.
  If you believe one of their tests is wrong, report the case and evidence; the lead mails Voice.
- **Full core suite (G2):**
  - Command:
    `cd <target> && flock -w 900 /tmp/forge-gpu.lock timeout 600 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<target>/tensor_cuda/tensor_cuda:<target>/tensor_cuda python3 -m pytest tensor_cuda/tests -q -p no:cacheprovider --ignore=tensor_cuda/tests/test_selector_accuracy.py`
    Log it to `logs/`.
  - Compare with
    `python3 /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/scripts/apa_mf2_2_g1_compare.py /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/logs/apa_mf1_g1_baseline_unchanged.log <your log>`.
    That script's "MF2.2 touched" line is irrelevant here. The id sets must be equal.
- **New core tests:** `tensor_cuda/tests/test_dlpack.py`.
  - They use torch as the peer.
  - The skip guard is narrow: skip ONLY when torch is not importable or no CUDA device is present. Any other exception
    in the probe FAILS.
  - Cover:
    - export and import pointer identity;
    - write-through both ways;
    - lifetime both ways (drop the producer, `gc.collect()`, then read);
    - stream ordering with a torch side stream, both directions;
    - the refusals (non-contiguous, CPU, unsupported dtype);
    - `data_ptr`, `fill_` and `mark_modified`.
  - Default wall time is ≤ 10 s.
- **Microbench (G5, informational):** 100,000 small alloc/free cycles plus 100,000 `fill_`-sized ops, comparing
  production's installed engine (`/mnt/ForgeRealm/Project-Tensor/tensor_cuda/tensor_cuda/`, read-only) with your
  build. Report the medians of 5 runs.

## Gates

- **G1 (acceptance):** Voice's runner with `--items T1` and `tolerances_r2.json` exits 0 with verdict PASS. Every
  lifetime and stream trial passes (20/20 each).
- **G2 (no regression):** the full core suite's FAILED/ERROR set equals the 52 baseline ids, and `test_dlpack.py`
  passes.
- **G3 (scope):**
  - `git diff 9df5bef --stat` shows only the paths under "Exercises", `tensor_cuda/include/tc/ops.h`
    (declarations), `tensor_cuda/CMakeLists.txt` (include path, if needed), `tensor_cuda/tests/test_dlpack.py`,
    `docs/P1T_M1_*.md`, `artifacts/p1t_m1/**` and `logs/**`.
  - `git diff 9df5bef -- '*.cu' '*.cuh' | grep -c '__global__'` prints 0.
  - `dlpack.h` keeps its license header.
- **G4 (teeth):** build these mutants outside the worktree, and run `test_dlpack.py` plus the G1 command on each:
  - (a) `from_dlpack` copies into a new Storage instead of aliasing;
  - (b) the export manager holds a raw pointer, not the Storage;
  - (c) export ignores `stream` (no event wait);
  - (d) the import Storage calls the deleter without the stream sync.

  (a) and (b) must FAIL. For (c) and (d), report the failure counts. A race that does not reproduce is an honest RED
  for that mutant, with your explanation, not a reason to weaken anything.
- **G5 (cost, informational):** the microbench above. Flag any median slowdown above 2%.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` first. Your order file is immutable.
- NO git (the lead commits). NO subagents. No background waits: run every gate synchronously to completion.
- No deletes anywhere: `rm` and the like are blocked. Leave unneeded files in `/tmp`, which is exempt.
- Never kill a process you did not start.
- Production `/mnt/ForgeRealm/Project-Tensor` (outside `Fork/`), `/mnt/ForgeRealm/Project-Voice`, and every other
  fork or worktree are READ-ONLY.
- RED honesty: a failed gate is a result. Report it with receipts; never weaken a gate or a test to pass.
- Prior Art Directive (AGENTS.md): annotate at the code site, in the ledger and in the report. Known prior art:
  - dmlc/dlpack (2017+, v1.3 header);
  - the Python array API standard's DLPack stream protocol (2021+);
  - the PyTorch and CuPy `from_dlpack` owner-keepalive pattern.
- If the AFTERIMAGE memory hook denies a Write, read what it shows you and retry the Write. Do not route around it
  through the shell. If a report Write is refused outright, put the full report in your final message.

## Done

- Write `docs/P1T_M1_REPORT.md` (gate table, API signatures, design choices for lifetime and stream sync with code
  citations, the `revision` reader list, adapter changes, prior art) and `docs/P1T_M1_LEDGER.md` (timestamped
  steps).
- Your final message must contain, verbatim:
  - the gate table;
  - the G1 verdict line and report path;
  - the G2 compare output;
  - the G3 diff stat and `__global__` count;
  - the G4 mutant table;
  - the G5 medians;
  - the built .so sha256;
  - every deviation from this order, with its reason.
