# P1T-M1 report: DLPack interop in TensorCUDA core (T1), plus Amendment 1 (`tensor_cuda.empty`)

Seat: Claude Opus 5.5 (Agent-tool implementation seat), 2026-10-12. Order: `orders/P1T_M1_DLPACK.md` (fork bc812f4) and
`orders/P1T_M1_AMENDMENT_1.md` (fork f5da728). Worktree: `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_m1`
(branch voice-m1, base 9df5bef). Ledger: `docs/P1T_M1_LEDGER.md`. Evidence class for every gate: suite / end-to-end runs
on the RTX 4070 SUPER, driver 595.99.02 (native), under `/tmp/forge-gpu.lock`.

**Verdict.** G1 PASS, G2 PASS, G3 PASS (emulated without git; the lead runs the literal git command), G4: (a), (b) and
(c) FAIL on both instruments as required; (d) FAILs `test_dlpack.py` (after a test fix, disclosed below) but PASSES
Voice's G1 runner, 0/20 in every trial, so (d) is not detected by G1. G5 FLAGGED: the alloc/free rows came out at
+2.26% and +4.95% in the registered run, and +2.82% for the pooled row in a repeat, measured under a load average of
about 26 on 12 cores. Amendment 1: 7/7 tests pass; `empty` takes 0.27 ms and `zeros` takes 10.04 ms at
[1, 512, 7160] fp32.

## Gate table

| Gate | Criterion | Result | Receipt |
|---|---|---|---|
| G1 | Voice `run_p1t.py --items T1`, tolerances_r2: exit 0, PASS, every lifetime/stream trial 20/20 | **PASS**. Exit 0, 11/11 cases; import_lifetime, export_lifetime, export_stream and import_stream each 0/20 failed trials. Run r1 at 19:38 and r2 at 20:09, same .so | `artifacts/p1t_m1/p1t_T1_report.json`, `logs/p1t_m1_g1_r2.log` (r1: `logs/p1t_m1_g1_r1.log`) |
| G2 | full core suite FAILED/ERROR set == the 52 baseline ids; `test_dlpack.py` passes | **PASS**. 52 == 52, none new, none fixed; test_dlpack 37 passed | `logs/p1t_m1_g2_suite_r2.log`, `logs/p1t_m1_g2_compare_r2.txt`, `logs/p1t_m1_test_dlpack_r4.log` |
| G3 | diff only in allowed paths; `__global__` count 0; dlpack.h keeps its license header | **PASS**, emulated without git: only allowed paths, plus the lead's two order files; `__global__` lines in the `.cu` diff = 0 (the file total stays 79); dlpack.h is upstream v1.3, byte for byte | `artifacts/p1t_m1/g3_scope.txt`, `artifacts/p1t_m1/g4/scope.py` |
| G4 | (a), (b) must FAIL; (c), (d) report failure counts | (a) FAIL / FAIL; (b) FAIL / FAIL; (c) FAIL / FAIL, export_stream 18/20; (d) test_dlpack FAIL (r4), **G1 PASS, 0/20 on all trials** | `artifacts/p1t_m1/g4/` |
| G5 | informational; flag any median slowdown > 2% | **FLAGGED**. r2 (registered): alloc_free +2.26%, alloc_free_pooled +4.95%, fill-sized -7.36%. r3 (repeat): -2.00%, +2.82%, -1.80%. Load average 25.9-27.0 on 12 cores | `logs/p1t_m1_g5_r2.log`, `logs/p1t_m1_g5_r3.log` |
| A1 | `empty` test + empty-vs-zeros timing | **PASS** 7/7; empty 0.2717 ms vs zeros 10.0376 ms (median of 20) | `logs/p1t_m1_amendment1_tests.log`, `logs/p1t_m1_g5_r2.log` |

## G1

Verdict line, from the final run (r2, 20:09, `logs/p1t_m1_g1_r2.log`): `T1: 11 cases, 6.9 s` and summary
`"verdict": "PASS", "exit_code": 0`, counts PASS 11 / FAIL 0 / SKIP 0; tolerance files tolerances.json `bccfdb33...`
and tolerances_r2.json `1716832057...`. Report: `artifacts/p1t_m1/p1t_T1_report.json` (adapter sha256 `6f4590c0...`).
Per case: device/{fp32,fp16}, export_alias/{fp32,fp16}, import_alias/{fp32,fp16} and roundtrip_shapes PASS (pointer
identical); import_lifetime, export_lifetime, export_stream and import_stream PASS with 0/20 failed trials each.

## G2 compare output (final tree, r2)

```
G1 compare (APA-MF2.2): baseline = /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/logs/apa_mf1_g1_baseline_unchanged.log; now = /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_m1/logs/p1t_m1_g2_suite_r2.log
summary: 37 failed, 2028 passed, 47 skipped, 2 warnings, 15 errors in 262.26s (0:04:22)
baseline ids 52 now ids 52
baseline ids that no longer fail: 0 []
new failing ids: 0 []
new MF2.2 test files with any FAILED/ERROR: 0
G1 PASS
```
The script prints its own label ("G1 PASS"), and its MF2.2 line does not apply to this order. Run r1, on the first
test file, gave the same result (`logs/p1t_m1_g2_compare_r1.txt`).

## G3 (emulated; see deviation D1)

This seat is not allowed to run git. `artifacts/p1t_m1/g4/scope.py` compares every file in the worktree (skipping
gitignored build products) with production's checkout, whose tracked files are at 9df5bef and clean. The final
output is in `artifacts/p1t_m1/g3_scope.txt`, which is reproduced verbatim in the final message. Every path falls
under the allowed list: the "Exercises" paths, `tensor_cuda/tests/test_dlpack.py`, `docs/P1T_M1_*.md`,
`artifacts/p1t_m1/**` and `logs/**`. The two exceptions are `orders/P1T_M1_DLPACK.md` and
`orders/P1T_M1_AMENDMENT_1.md`, which are the lead's own commits. `ops.h` and `CMakeLists.txt` are unchanged: the
`include/` directory is already on the include path.

`__global__` count: `diff -U3 -p <production>/tensor_cuda/src/kernels.cu <worktree>/tensor_cuda/src/kernels.cu |
grep -c __global__` gives **0**. kernels.cu is the only `.cu`/`.cuh` file that changed. `grep -c __global__` over
the whole file gives 79 in both versions.

dlpack.h: upstream dmlc/dlpack tag v1.3, `include/dlpack/dlpack.h`, fetched through the AtlasForge web proxy from the
GitHub contents API. Its git blob sha1 recomputes to `9252a44833ee55e0092035c57d34dcab928c2e8d`, the API's `sha`, and
its sha256 is `62052bef24bd69bb7b52e95428f9e99bd7951783850c36c5f2a41e71fd63c2fc`. It is vendored byte for byte, so
the header `Copyright (c) 2017 -  by Contributors` is kept. The repository license is Apache-2.0. The copy shipped
with torch was not used, because it is wrapped in a torch-only `#if !defined(TORCH_STABLE_ONLY)` guard and its
comments differ from the tag in 8 places.

Lead: please also run the literal commands, `git diff 9df5bef --stat` and
`git diff 9df5bef -- '*.cu' '*.cuh' | grep -c '__global__'`.

## G4 mutant table

Each mutant was built from an rsync copy of `tensor_cuda/` with exactly one asserted edit
(`artifacts/p1t_m1/g4/make_mutants.py`; the diffs are in `artifacts/p1t_m1/g4/mut_*.diff`). Each was built with
`./build.sh 89` (exit 0) and run by `artifacts/p1t_m1/g4/run_mutant.sh`, all outside the worktree. The test_dlpack
column uses the final test file (r4, sha256 `e816254f...`). Voice's G1 does not depend on that file, so its r1 runs
stand.

| Mutant | Edit | Build path | .so sha256 | test_dlpack.py | G1 verdict | G1 failure counts |
|---|---|---|---|---|---|---|
| (a) import copies into a new Storage | `bindings.cpp` `tensor_from_dlpack_capsule`: `NDArray::empty` + D2D copy, capsule left unconsumed | `/tmp/p1t_m1_mut_a/tensor_cuda` | `b1e2e33797170af75d2c4fb0d0b74af49404eb286b4fd8b366e4cb42d5b970ef` | **FAIL**: 17 failed, 20 passed (import identity x6, capsule protocol, producer-stream, deleter-once, size-1 stride, byte_offset, empty round-trip x6) | **FAIL** (exit 1) | import_alias/float32 and /float16 FAIL (pointer_identical False, both write-throughs False); lifetime/stream 0/20 each |
| (b) export manager holds a raw `Storage*` | `bindings.cpp` `DLPackExportCtx::owner` | `/tmp/p1t_m1_mut_b/tensor_cuda` | `e80056d2a625a7fd8188c31e0517a7b8637f093f5612415cfb1c544f815514e4` | **FAIL**: 3 failed (unconsumed_capsule_releases, export_lifetime, import_deleter_called_once) | **FAIL** (exit 1) | export_lifetime **20/20** failed; import_lifetime, export_stream and import_stream 0/20 |
| (c) export ignores `stream` (no event wait; argument checks kept) | `bindings.cpp` `dlpack_fence_consumer` returns before the event record | `/tmp/p1t_m1_mut_c/tensor_cuda` | `2404d5ede1ebc10194079da9bf3ee45b17f95ffe19b4e159ed8710187e1e810f` | **FAIL**: 1 failed (export_stream_ordering, trial 0) | **FAIL** (exit 1) | export_stream **18/20** failed; the others 0/20 |
| (d) import Storage calls the deleter without the stream sync | `kernels.cu` `~Storage`: `cudaStreamSynchronize(0)` removed | `/tmp/p1t_m1_mut_d/tensor_cuda` | `758ff27846429305dd2a2b38a75fbe50ae2d97bc06e5755e07bb90d81062717e` | r1 (first test file): **37 PASSED, survived**. r4 (final file): **FAIL**, 1 failed (import_release_drains_engine_stream_before_deleter, trial 0) | **PASS** (exit 0): not detected | 0/20 in import_lifetime, export_lifetime, export_stream and import_stream |

Why (d) survived r1 (diagnosed with receipts in `artifacts/p1t_m1/g4/diag_d*.py`; ledger 19:58-19:59):
- On the real build, `del u` takes about 7.9 ms (the drain) and leaves the legacy stream idle. On the mutant it
  takes 0.02 ms and the legacy stream is still busy, so the mutant really does call the deleter early.
- My test called `gc.collect()` between the release and torch's reuse of the block. With torch loaded, that call
  stalls the host for 57-79 ms, longer than the ~8 ms sleep, so the queued engine write had finished before the
  block was reused.
- Fix: no `gc.collect()` in that window (refcounting frees the Tensor right away), plus kernel warm-ups. The
  assertions and trial count are unchanged. The first attempt at the fix ran the unchanged file because the
  AFTERIMAGE hook refused the Edit (r3, void); r4 is the real fix.

Why Voice's G1 cannot see (d): none of its T1 cases has TensorCUDA work still queued on imported memory when the
TensorCUDA tensor dies. `import_lifetime` calls `t1_read` (which completes engine work) before `del t`, and the
`gc.collect()` that follows would close the window anyway. This is an honest RED for G1's teeth on (d), not a defect
in the build. The lead may want to mail Voice a candidate case (the shape of
`test_import_release_drains_engine_stream_before_deleter`). I did not edit Voice's tests.

## G5 (informational)

Script: `artifacts/p1t_m1/g5_microbench.py`. It runs fresh processes for 5 runs, alternating which engine goes
first. Production's engine is .so `d8fda9a8...` (read-only, `PYTHONDONTWRITEBYTECODE=1`); mine is `09aa9076...`.
Production has neither `fill_` nor `empty`, so the cross-engine pair uses calls both engines have:
- alloc_free: `y = x * 1.0; del y` on 256 fp32 elements.
- fill-sized: `_C.scale_(t, 1.0)`, the same one-block launch shape that `fill_` uses.

Registered run r2 (`logs/p1t_m1_g5_r2.log`), seconds per 100,000 calls:
```
alloc_free_s: mine median 1.1260 s (11.260 us/call); prod median 1.1011 s (11.011 us/call); mine/prod - 1 = +2.26%  FLAG (> 2%)
alloc_free_pooled_s: mine median 0.7873 s (7.873 us/call); prod median 0.7501 s (7.501 us/call); mine/prod - 1 = +4.95%  FLAG (> 2%)
fill_sized_scale_s: mine median 0.3573 s (3.573 us/call); prod median 0.3857 s (3.857 us/call); mine/prod - 1 = -7.36%
fill__s: mine median 0.3941 s (3.941 us/call)
```

Repeat run r3, an addition that does not replace r2 (`logs/p1t_m1_g5_r3.log`): alloc_free -2.00%, alloc_free_pooled
+2.82% (FLAG), fill-sized -1.80%, fill_ 0.3732 s.

Run r1 is void for its alloc row: same-dtype `float()` allocates nothing, because `ops.cpp:1059` returns the input.

Conditions: load average 25.9-27.0 on 12 cores from other sessions' python and delve processes, with about ±14%
run-to-run spread in the alloc rows.

Reasoning (not a measurement): on this path the change adds one predictable branch to `~Storage` and 16 bytes to
`Storage`, which should cost nanoseconds against cycles of 7.5-11 µs. The pooled flag repeated and the raw one
flipped sign. Not resolved under this load; the lead may re-run G5 on an idle machine.

## Built engine

`tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`: sha256
`09aa9076d40922dac50521735579d01db2e96cf21058b8e6fc6ae6e27b67e83d`, 9,487,976 bytes.
- Build: `cd tensor_cuda && ./build.sh 89`, CUDA 12.6 (`logs/p1t_m1_build_r1.log`).
- Flags unchanged: `-O3 -DNDEBUG -std=c++17 --generate-code=arch=compute_89,...`. No fast-math, no TF32 and no
  `--default-stream` flag.

## API (as built)

- `Tensor.__dlpack__(*, stream=None, max_version=None, dl_device=None, copy=None) -> PyCapsule`
  (`bindings.cpp:349` impl, `:635` binding; keyword-only)
  - Returns `"dltensor"`, or `"dltensor_versioned"` (version 1.3, flags 0) when `max_version[0] >= 1`.
  - `data` is the Storage base pointer and `byte_offset` is `offset * itemsize` (`:393`), so data + byte_offset is
    element 0. Strides are C-contiguous.
  - Raises `BufferError` for CPU tensors, for `copy=True` and for a foreign `dl_device`. Raises `ValueError` for
    `stream=0` and for streams below -1.
  - The docstring states the mark_modified rule.
- `Tensor.__dlpack_device__() -> (2, ordinal)`; `BufferError` on CPU (`bindings.cpp:425`).
- `tensor_cuda.from_dlpack(obj)`: an object with `__dlpack__`, or a bare capsule (`__init__.py:84`; C++
  `bindings.cpp:451`).
- `Tensor.data_ptr() -> int`: the address of element 0 (`bindings.cpp:641`).
- `Tensor.fill_(value) -> self` (`bindings.cpp:646`; host wrapper `kernels.cu:282-292` around the existing
  `fill_kernel`, which is at `kernels.cu:244` now and was at :217 before).
  - Asynchronous, queued on the legacy default stream.
  - float32, float16 and bfloat16 only (RuntimeError otherwise); CUDA only.
  - Bumps the revision.
- `Tensor.mark_modified()` (`bindings.cpp:654`) and read-only `Tensor.revision` (`bindings.cpp:657`).
- Amendment 1: `tensor_cuda.empty(*shape, dtype="float32", device="cuda")` (`__init__.py:128`; C++ `_C.empty`,
  `bindings.cpp:573`, `:1411`).
- `__all__` gains `"from_dlpack", "empty"` next to `from_numpy` (`__init__.py:1310`).

## Design: lifetime

- **Export keepalive.** The manager context `DLPackExportCtx` owns a `std::shared_ptr<Storage>`
  (`bindings.cpp:280-281`, set at `:386`), so the memory outlives the producer Tensor's last Python reference.
  Its deleter deletes the ctx (`:288-293`).
- **Capsule destructors** (`:297-312`). An unconsumed capsule (still named `dltensor*`) calls the deleter, with the
  Python error state saved and restored. A consumed one (`used_*`) does nothing.
- **Import.**
  - `tensor_from_dlpack_capsule` validates first, without consuming:
    - version major == 1;
    - not read-only;
    - device kDLCUDA, on the current device;
    - an exact dtype;
    - ndim at most 8;
    - C-contiguous (a size-1 dimension may have any stride);
    - `byte_offset` a multiple of the element size.
  - It then wraps the pointer in `Storage(external_ptr, nbytes, dev, release, manager)` (`core.h:63-81`,
    `kernels.cu:182-185`) and only then renames the capsule `used_*` (`bindings.cpp:559`).
  - Every refusal leaves the capsule unconsumed, so its own destructor returns the memory to the producer.
  - `NDArray.offset = byte_offset / itemsize`, so a non-zero producer `byte_offset` survives a re-export.
- **Release order** (`kernels.cu:186-207`):
  1. If `external_release` is set, `cudaStreamSynchronize(0)` drains the engine stream (`:205`).
  2. Then `external_release(ctx)` (`:206`) calls the producer's deleter with the GIL held: `PyGILState_Ensure`, at
     `bindings.cpp:435-448`. It skips the call (a leak, not a crash) if the interpreter has been finalized.
  3. It then returns, so the memory is never `cudaFree`-d and never pooled.
- **Why a stream sync rather than an event.**
  - The engine keeps no per-Storage last-use event, so "everything queued on the engine stream so far" is the only
    sound fence. An event recorded at destruction and then waited on covers exactly the same work, at three API
    calls instead of one.
  - A deferred release through a host callback is ruled out: CUDA forbids CUDA API calls in host callbacks, and
    torch's deleter calls into its caching allocator.
  - Cost: the host blocks for the duration of the queued work. This happens only when imported memory is released,
    and measures 7.9 ms behind a 10 ms sleep (diag_d).

## Design: stream contract

- **Export** (`dlpack_fence_consumer`, `bindings.cpp:320-347`):
  - None, 1 and -1 need nothing.
  - 0 and values below -1 raise ValueError.
  - 2 maps to `cudaStreamPerThread`. Any other value is a `cudaStream_t`. For both, `cudaEventRecord(ev,
    cudaStreamLegacy)` and then `cudaStreamWaitEvent(consumer, ev)` (`:337-338`); the event is destroyed right after
    the wait is enqueued.
  - Every engine kernel launches on stream 0, the legacy stream, because CMake passes no `--default-stream` flag.
- **Import:** `from_dlpack` calls `obj.__dlpack__(stream=1, max_version=(1, 3))`. On a TypeError it retries with
  `stream=1` alone (`__init__.py:120-124`). The producer orders the legacy stream after its own pending work.
- **Reads:** `t1_read` is `Tensor.numpy()`, a `cudaMemcpy` D2H on the legacy stream. It waits for TensorCUDA's stream
  only, not for non-blocking streams and not for the device.

## `Storage::revision` readers (A1)

Grep of every `revision` use in `tensor_cuda/` (src, include, Python):

| Reader | Site | What it caches |
|---|---|---|
| `ops::terrain_render` (density_filter 1 or 2) | `tensor_cuda/src/ops.cpp:397-402` (reads `materials.data().revision()`, compares with and stores `cache.source_revision`) | the filtered u8 density field for the materials tensor; the cache lives per Variable at `tensor_cuda/include/tc/autograd.h:31-34` (`TerrainDensityCacheEntry`) and `:46` (`terrain_density_cache[2]`) |

That is the only reader.
- **Writers that bump it:** optimizer steps and `axpy_`/`scale_` (`kernels.cu:1002-1054`, `4758-4789`),
  `write_rows` (`kernels.cu:7432`), the new `fill_` (`kernels.cu:291`) and `mark_modified()`.
- **Not readers:** the Python caches `functional.py:17` (causal masks), `:77` (guard constants), `:242` (RoPE tables)
  and `quant.py:135` (selection rows) are keyed on shape, dtype and device and built from TensorCUDA's own constants,
  so an external write cannot make them stale.
- **Caller-held derived tensors:** `apa_gemm_selective_attention_cached` takes `k_codes` and `k_scales` from the
  caller. Those are derived tensors the caller holds, not a revision cache; if K is rewritten externally, the caller
  must re-quantize.
- **Rule for Voice** (it is in the `__dlpack__` docstring): after an external in-place write, call `mark_modified()`
  before handing the tensor to a TensorCUDA op. Today that matters only for `terrain_render` with `density_filter`;
  calling it when it is not needed is harmless.

## Adapter changes (`artifacts/p1t_m1/tensorcuda_adapter.py`, sha256 6f4590c0...)

The adapter is a copy of Voice's stub (sha256 d0c5a596...):
- `t1_*` are real:
  - `t1_from_dlpack` is `tc.from_dlpack`.
  - `t1_empty` is `tc.empty`.
  - `t1_fill` is `Tensor.fill_`.
  - `t1_read` is `numpy()`.
  - `t1_data_ptr` is `data_ptr()`.
  - `t1_stream` returns 0.
- T2-T6 now raise NotImplementedError naming tc_signal (M2-M4).
- Census fixes:
  1. `t1_empty` used `tc.zeros`, a host numpy array plus a synchronous H2D copy. It now uses `tc.empty`.
  2. `t1_read` cited `nn.py:66`, which is a use site. The definition is `tensor_to_numpy` (bindings.cpp) →
     `NDArray::to_host`: a `cudaMemcpy` D2H on the legacy stream, which is not a device sync and does not wait on
     non-blocking streams.
  3. "No in-place fill": `fill_` now exists.
  4. The stub's partial T2/T3/T5 compositions over core ops are replaced by NotImplementedError, per order item 7 and
     the approved alt plan. The originals remain in Voice's repository.

## Amendment 1: device-side `tensor_cuda.empty`

- Bound `NDArray::empty` with no fill and no host round trip; its contents are uninitialized (the docstring says
  so). `requires_grad` is False.
- Shape conventions are those of `zeros`: `empty(2, 3)`, `empty((2, 3))`, `empty([2, 3])`, numpy ints.
- Negative dimensions raise ValueError. An unknown dtype string raises, where `zeros` silently falls back to float32.
- Test result (`logs/p1t_m1_amendment1_tests.log`, verbose):
```
tensor_cuda/tests/test_dlpack.py::test_empty_shape_dtype_and_dlpack_roundtrip[0d-float32] PASSED
tensor_cuda/tests/test_dlpack.py::test_empty_shape_dtype_and_dlpack_roundtrip[0d-float16] PASSED
tensor_cuda/tests/test_dlpack.py::test_empty_shape_dtype_and_dlpack_roundtrip[7-float32] PASSED
tensor_cuda/tests/test_dlpack.py::test_empty_shape_dtype_and_dlpack_roundtrip[7-float16] PASSED
tensor_cuda/tests/test_dlpack.py::test_empty_shape_dtype_and_dlpack_roundtrip[1x512x7160-float32] PASSED
tensor_cuda/tests/test_dlpack.py::test_empty_shape_dtype_and_dlpack_roundtrip[1x512x7160-float16] PASSED
tensor_cuda/tests/test_dlpack.py::test_empty_shape_conventions_match_zeros PASSED
======================= 7 passed, 30 deselected in 2.22s =======================
```
  Each parametrized case checks shape, dtype, `requires_grad` False and device cuda. It also checks that
  `torch.from_dlpack(e).data_ptr() == e.data_ptr()` and that `tc.from_dlpack` of that torch view gives the same
  pointer back.
- Timing line (`logs/p1t_m1_g5_r2.log`; each call is timed with `tc.synchronize()` inside the timer, and the free
  happens outside it):
```
Amendment 1: [1, 512, 7160] fp32, median of 20 (after 2 warm-ups): empty 0.2717 ms, zeros 10.0376 ms, zeros/empty = 36.9x
```
  Repeat r3: empty 0.2522 ms, zeros 7.4468 ms (29.5x).
- **Finding (D4):** `zeros()` and `zeros(())` return shape **(1,)**, not (): `_factory` passes the 0-d host array
  through `np.ascontiguousarray`, which promotes it to 1-d. That is pre-existing behavior. `empty()` and `empty(())`
  return the requested 0-d shape, because the amendment's census lists () and asks for "the requested shape".
  `test_empty_shape_conventions_match_zeros` pins this difference, so the lead can choose otherwise.

## Deviations from the order, with reasons

- **D1. G3 was computed without git.** The dispatch rules forbid git. I ran a non-git emulation (a diff against
  production's checkout at 9df5bef). The lead should run the literal `git diff 9df5bef --stat` and the `__global__`
  grep.
- **D2. Rule slips: two git invocations.**
  - 19:28: `git ls-remote https://github.com/pybind/pybind11.git refs/tags/v2.12.0`, a read-only network probe for
    the build's FetchContent.
  - 19:38: `git -C /mnt/ForgeRealm/Project-Voice status --short`, a read-only check that G1 wrote nothing into
    Project-Voice. It printed nothing.
  - Neither changed anything, and no other git command was run.
- **D3. `from_dlpack` also passes `max_version=(1, 3)`.** It passes it together with `stream=1`, retrying without it
  on TypeError. This gives versioned capsules, so a version check and the read-only flag are available. `stream=1`
  is always passed. `test_from_dlpack_asks_producer_for_legacy_stream` pins the exact arguments.
- **D4. `empty` returns 0-d for an empty shape, while `zeros` returns (1,).** See Amendment 1.
- **D5. Additions beyond the spec, all refusals or protocol details, none silent:**
  - read-only DLPack imports are refused;
  - a misaligned `byte_offset` is refused;
  - another CUDA device is refused (multi-GPU is out of scope);
  - `stream=0` and values below -1 raise ValueError (0 is ambiguous under the protocol);
  - `copy=True` and a foreign `dl_device` raise BufferError (the order allowed BufferError or ValueError);
  - export is allowed for `requires_grad=True` tensors (unspecified; torch refuses them).
- **D6. Extra import dtypes.** Exact-match bfloat16, int64, uint8 and bool are accepted on import and export, as the
  order permits. Voice's tests require no refusal of them.
- **D7. Adapter.** T2-T6 compositions were replaced by NotImplementedError, and `t1_empty` uses `tc.empty`.
- **D8. G4: one test was fixed after the mutant run.** It is disclosed in full above. The table reports both r1
  (survived) and r4 (killed). Nothing was loosened.
- **D9. G5 workload and runs.**
  - The cross-engine workload uses `x * 1.0` and `_C.scale_`, because production has neither `empty` nor `fill_`.
  - r1's alloc row is void (its workload allocated nothing).
  - r3 is an extra repeat; r2 is the registered result.
- **D10. G1 report JSON overwritten.** It was overwritten by the final r2 run; r1's summary is in
  `logs/p1t_m1_g1_r1.log`.
- **D11. Census: there is no non-zero-offset view.** The Python layer exposes none (`reshape` shares storage at
  offset 0, and `slice` copies). Instead:
  - a reshape view is tested for sharing;
  - a ctypes-built DLPack producer supplies a real non-zero `byte_offset` on import, which is checked through a
    re-export;
  - the same producer supplies a size-1 dimension with stride 12345.
- **D12. Files the repository's `.gitignore` excludes.** `.gitignore` excludes `*.json` and `*.log`, so the lead
  needs `git add -f` for `artifacts/p1t_m1/p1t_T1_report.json`, `artifacts/p1t_m1/g4/*.json` and `logs/p1t_m1_*.log`.
  The mutant logs in `g4/` are named `*.log.txt`.

## Residuals (not claimed)

- G1 does not detect mutant (d), as above.
- Export side: when a consumer calls our deleter, an engine-owned Storage may be freed at once.
  - In the default (raw) mode, `cudaFree` synchronizes the device.
  - With pooling on, `cudaFreeAsync` on the legacy stream does not wait on a consumer's non-blocking stream.
  - DLPack makes the consumer responsible for finishing its stream work before it releases. This is untested.
- A bare capsule import cannot be stream-ordered; the caller must already have ordered the producer's work.
- If Python has already been finalized at release time, the producer's deleter is skipped (a leak, not a crash).
- The G5 flags are unresolved, as above.
- These are author-run tests; no blind red team has been run (House Rules §8).

## Prior art

- DLPack: dmlc/dlpack, 2017 onward. The v1.3 header is vendored verbatim; the data layout, capsule naming and deleter
  contract come from it.
- The Python array API standard's data-interchange protocol: `__dlpack__(stream=, max_version=, dl_device=, copy=)`
  and `__dlpack_device__`, Consortium for Python Data API Standards, 2021-2023. The stream-integer semantics come
  from it.
- PyTorch's ATen DLConvertor and `torch.utils.dlpack.from_dlpack`, and CuPy's `from_dlpack`. Taken: the owner
  keepalive (the manager holds the producer's storage, and the consumer holds the manager until its tensor dies),
  renaming the capsule `used_*` on consumption, and the versioned-then-legacy retry.
- The CUDA event-record plus `cudaStreamWaitEvent` cross-stream fence: the standard idiom.
- Project-Voice's T1 runner (`run_p1t.py`, 2026-10), for the sleep-then-write race and the same-size reuse-bait
  allocation.
- DeMillo, Lipton and Sayward (1978), mutation testing, for the registered mutants.
- Mine:
  - the TensorCUDA wiring: a `shared_ptr<Storage>` owner on export, and an external-release Storage that drains the
    engine stream before the deleter, placed in `~Storage`;
  - the import-release-ordering race test, which kills mutant (d);
  - the ctypes DLPack producer used for the offset, stride and flag cases.
- Unverified, lead to check: whether the drain-before-deleter rule on the consumer side is spelled out in the DLPack
  or array API documents. Search terms: "DLPack deleter stream synchronization consumer", "array API from_dlpack
  deleter stream".

## Scratch left in /tmp (cannot delete; exempt)

- `/tmp/p1t_m1_before/`: originals of the touched files.
- `/tmp/p1t_m1_scratch/`: smoke and diagnostic scripts, and the decoded upstream header.
- `/tmp/p1t_m1_mut_{a,b,c,d}/`: mutant trees, builds and logs.
- Receipts needed for review are copied into `artifacts/p1t_m1/g4/`.
