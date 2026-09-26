# PT-RING-1 implementation ledger

Registered CPU-only implementation; GPU acceptance BLOCKED by the order.
Immutable plan: `orders/PT_RING_1_ASYNC_COPY.md`. House rules read. No git,
subagents, production lock access, live-engine imports/builds/edits, or GPU use.
Branch/HEAD are the lead's supplied identity. All work is in this fork.

## Registration before gates

`artifacts/pt_ring_1/REGISTRATION.json` records baseline source hashes, original
order hash, named tests, timing thresholds and lead-only gate protocol. Original
sources and binary were copied before editing. Existing campaign receipts and
registrations remain immutable; this order gets an additive source seal.

Evidence class: source inspection. NDArrays are contiguous; transpose/permute
materialize and reshape shares storage. Allocations and all engine kernels use
the legacy stream. Async copies must retain actual Storage objects, not merely
Python Tensor handles whose data may later be replaced. D2D methods will be
explicit `memcpy` (default) and `kernel` (experimental comparison).

Evidence class: host inspection. `ulimit -l` = 8220136 KiB = 8417419264 bytes,
less than the approximate 8.817 GB host ring. CUDA driver enforcement remains
unmeasured. An explicit engine-owned pinned-byte budget will reject requests
before CUDA above its limit; driver allocation errors propagate without retry
or pageable fallback. No driver call latency guarantee is inferred from this.

## Prior art

NVIDIA CUDA 12.6 (2024), streams/events/page-locked memory and memcpy, supplies
the mechanism; verified primary runtime documentation:
https://docs.nvidia.com/cuda/archive/12.6.3/cuda-runtime-api/group__CUDART__MEMORY.html
and https://docs.nvidia.com/cuda/archive/12.6.3/cuda-runtime-api/stream-sync-behavior.html.
PyTorch (2016 onward; pin-memory tutorial and DCP async-save recipe, 2024) supplies
the API and completion/lifetime idioms; CheckFreq (Mohan et al., FAST 2021)
supplies two-phase checkpoint context. NVIDIA Apex multi_tensor_apply (2018
onward) is the batched pointer-list/chunk copy precedent, not a new algorithm.
The contribution here is integration with this engine's legacy stream, checked
byte-copy contracts, explicit lifetime retention and run-7-specific gates.
Further primary links and scope are recorded in the API note/final report.

## Execution notes

Inspection commands read the order, house rules, core/autograd headers, bindings,
allocation implementation, existing PT-DET-2 CPU tests/ledger and prior build
receipt. Initial guessed paths `tensor_cuda/__init__.py`, `include/ndarray.h`,
`tensor_cuda/build/CMakeCache.txt`, and `scripts/pt_det_2_slot.py` did not exist;
actual package is `tensor_cuda/tensor_cuda`, header is `include/tc/core.h`,
existing build is `build-tf32`, and DET-2 uses the DET-1 slot. No gate was run.

## CPU checkpoint schema

Evidence class: read-only checkpoint inspection, no engine imported. Restricted
NumPy-only pickle loading of the specified snapshot yielded 434 model entries,
338 Adam m and 338 Adam v: **1110 tensors / 2939091552 bytes**. Three slots need
**8817274656 bytes**. `artifacts/pt_ring_1/GEOMETRY.json` records every shape,
dtype and byte count, source file size and SHA256
`dac512303b6528f6fd6b40cca7b68a54cf237a1b976bf8057f1474dbd74c06bd`.
NumPy emitted its `numpy.core is deprecated and has been renamed to numpy._core`
warning during inspection; it was not suppressed. No checkpoint values were
changed or imported into the live engine.

## Implementation and build 01

Evidence class: CPU compilation. Added shared shape/alias validation headers,
an additive PT-RING-1 section after all existing `kernels.cu` bytes, bindings,
and Python aliases. C++ pending requests own actual Storage and pinned buffers
until completion; no background threads or Python references in CUDA work.
The default uses C++ memcpy batching; an explicit single-kernel comparison uses
pinned metadata, one descriptor H2D, one byte-copy kernel and async metadata free.

`artifacts/pt_ring_1/build_01/receipt.json` and `build.log` record exact CMake
argv/environment/source pins. Verbatim: `BUILD_RC 0 ELAPSED 64.25963601842523`.
CUDA was hidden. Inherited unused-variable / nvlink archive warnings remain
visible. No live-engine build or import occurred.

## CPU baseline 01

Evidence class: author CPU tests. Exact files:
`tests/test_pt_ring_1_cpu.py` and `tests/test_pt_det_2_cpu.py`.
`artifacts/pt_ring_1/cpu_01/pytest.log`: **`121 passed in 2.43s`**, `CPU_RC 0`.
The new tests compile the shared shape/alias contract to a CPU-only bridge,
compare 2000 interval cases with an independent quadratic oracle, exercise
binding/invalid/empty contracts without CUDA calls, preserve original kernel
bytes, and reject fictional incomplete/slow/nonfinite receipts. This is not
blind review, CUDA correctness, or a full engine-suite result.

## Amendment 001 and implementation review

Separate `artifacts/pt_ring_1/AMENDMENT_001.json` fixes exact byte geometry,
interleaved median timing statistics, default memcpy selection, seven lead
lanes, one-shot events, alias policy, limits and bounded create-only receipts
before those GPU gates run. Original order and registration remain untouched.

Source review prompted cache revision invalidation before submission (so a
partial CUDA failure cannot leave a derived cache apparently current). Golden
inputs are cast before becoming leaves because existing backward frees
non-leaf gradients. `empty_like` was added to allocate independent staging
without pageable intermediates and handle rank-zero/empty dtypes. These are
implementation details within the registered staging work, not threshold changes.

The existing selector benchmark prepends a live path and parses `sys.argv[1]`
at collection. The suite worker preloads this fork and calls programmatic pytest
with neutral `sys.argv`; an audit hook refuses live engine file/import access.
No existing tests, assertions or campaign pins are edited. Prior art: PEP 578
(Python 3.8, 2019), pytest's public main API, and PyTorch empty_like (2016 onward).
The GPU harness reports any remaining historical-suite failure, never suppresses
it. A 10-second receipt reserve stays inside the 900-second total cap.

## Final build, tests and seal

Evidence class: CPU compilation. Final build receipt
`artifacts/pt_ring_1/build_02/receipt.json`: **`BUILD_02_RC 0 ELAPSED
60.77024424355477`**. Compiler input pins were unchanged during compilation;
the final seal verifies them again. Existing warnings are retained in `build.log`.

Evidence class: author CPU tests. `final_cpu/pytest.log`: `126 passed in 2.36s`.
`final_cpu/gpu_collection.log`: `121 tests collected in 0.11s`, **zero GPU tests
executed**. A final harness timing-expression correction uses two nonnegative
elapsed intervals relative to a common origin for the negative hazard control.
After sealing the source, `handoff/pytest.log` reran the same two named CPU files:
**`126 passed in 2.39s`**. No full 60-file battery was run.

Evidence class: CPU guard checks. Each of correctness, hazard, overlap, ring,
limits, goldens, engine_suite and all returned code 2 and created no requested
output directory. Verbatim: `BLOCKED: lead GPU slot required; --lead-gpu missing
or CUDA_VISIBLE_DEVICES empty`. Exact argv/results are in
`artifacts/pt_ring_1/handoff/receipt.json`.

Evidence class: source/artifact inspection. `SOURCE_AUDIT.json` confirms only
`kernels.cu`, `bindings.cpp`, and the package `__init__.py` changed among existing
engine sources; all previous kernel bytes are an identical prefix. The baseline
binary SHA256 `69b154cd2e9a5a7cbe288e1419d963697e5add57d2b4ee827c4c78a273ff2c5c`
matches the earlier PT-DET-2 receipt, and all overlapping source pins match.
Checkpoint entries are all float32, ranks 1/2/3 = 459/627/24, no empty tensors;
the separate GPU matrix covers the other dtypes/ranks/empty cases.

Active additive seal: `artifacts/pt_ring_1/SOURCE_MANIFEST.json`, SHA256
`d938b3cf3a6cd7aa34a3dfe4ea4011406a4762501a6783f8543ab4996bf88f1a`.
Its files remained unchanged across final tests and all handoff checks. Old
orders, registrations and source manifests were not edited.

## Done

**Seat deliverables complete: bindings/Python API, CPU build, named CPU checks,
GPU gates prepared, API recipe, source seal and blocked report. GPU acceptance
remains BLOCKED.** See `docs/PT_RING_1_REPORT.md` for synthesis and
`artifacts/pt_ring_1/BLOCKED_REPORT.json` for machine-readable gate disposition.
The original host-capture overhead and three-state affordability are **not
claimed fixed**. No claim of GPU bitwise correctness, overlap, <=1% cost,
driver ring allocation, full engine-suite success, or golden parity is made.

No live engine was edited/rebuilt/imported; no GPU work, git, subagents,
background jobs/waits, production-lock operation or outside-process signal
occurred. Foreground compiler/pytest commands were monitored through the tool's
foreground process handles; no shell background jobs were launched. All edits,
builds, receipts and temporary files are in the authorized fork. Author checks
do not replace the lead's slot or independent verification.

Prior art completion: code comments, the API note and final report identify
NVIDIA CUDA 12.6 (2024), PyTorch (2016 onward; 2024 tutorials), NVIDIA Apex
(project 2018 onward), CheckFreq (Mohan et al., FAST 2021), related Check-N-Run /
Gemini / DCP context, pybind11/Jakob (2015), PEP 578 (2019), ctypes (2006), C++
binary search, POSIX, SHA256/NIST (2001), and local receipt conventions (2026).
Primary CUDA/PyTorch/Apex/checkpoint sources were checked with the web tool.
The local contribution is integration and gates, not a novel copy algorithm.
