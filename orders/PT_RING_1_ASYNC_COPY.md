# PT-RING-1 — REGISTERED 2026-09-26 10:50 EDT by the lead (Fable 5.1). Seat: GPT-6 Astra at max. WRITABLE TARGET: this fork worktree `/mnt/ForgeRealm/wt/pt-tf32` (branch tf32-fast-path, HEAD = PT-DET-2 f2c686e) — edits and builds AUTHORIZED here; the live engine `/mnt/ForgeRealm/Project-Tensor` is never edited, rebuilt or imported; no GPU in the seat (CUDA_VISIBLE_DEVICES=""); no git; no subagents; never kill processes you did not start; never touch /tmp/forge-gpu.lock; RED honesty. A CPU-only build + blocked-report; the lead runs the GPU gates in a slot. Name the test files you run (a full 60-file battery times out the seat). The draft below (written by the CC47-B seat) is the specification.

# PT-RING-1 — asynchronous device→host copies on a side stream into pinned host memory (+ a bound device copy), so a host ring of 3 training states becomes affordable

**DRAFT — written by the CC47-B seat (Opus 5.5) on 2026-09-26; NOT registered. The lead registers it (date, seat,
worktree, slot).** Proposed seat: GPT-6 Astra (engine). Proposed writable target: a new engine worktree/branch from the
engine run 7 uses (Project-Tensor `tensor_cuda` for v2; the `tf32-fast-path` fork for v3) — the lead names it. Live
engine never touched; no GPU in the seat; the lead runs the slot under `/tmp/forge-gpu.lock`.

## Why (GRAPA CC47 / CC47-B receipts)
- CC47's gradient-norm tripwire keeps the last R = 3 exact training states (weights + buffers + Adam m/v, 2.939 GB each,
  1,110 tensors) in host memory. The only device→host path is `Tensor.numpy()` (`bindings.cpp tensor_to_numpy`): a fresh
  pageable `py::array` + a synchronous `cudaMemcpy` on the legacy default stream with the GIL held
  (`kernels.cu NDArray::to_host`). Measured host half 0.44–1.75 s per capture (fresh pageable pages fault in), estimated
  0.9–1.2 s per step = **18–23 % of a 5.09 s step** — not armable (`docs/CC47_LEDGER.md` §4).
- CC47-B therefore keeps ONE state on the device and copies it device→device per step. The engine binds no
  `clone()`/`copy_()` (`NDArray::clone` exists in C++ only; `ops::cast` returns the tensor itself for its own dtype), so
  the copy is `tensor_cuda.write_rows(dst, src, 0)` — the KV-ring writer, repurposed: one launch per tensor
  (1,110 launches per step), int64 index arithmetic per element, ≥ 2-D only (1-D/0-D need reshape views), float/uint8
  only, raises under grad (`docs/CC47B_LEDGER.md`). It works (bitwise) but it is a workaround, and it keeps only one
  state: the host ring's t-3 and t-2 are gone.
- With a pinned host ring and an asynchronous copy on a side stream, the 2.94 GB D2H (≈ 0.12 s at ≈ 25 GB/s pinned on
  PCIe 4.0 x16) overlaps the 5 s step: a host ring of 3 costs the device→device staging copy (≈ 12 ms) plus launch
  overhead per step.

## Build (engine; Python surface in `tensor_cuda/__init__.py`, bindings in `src/bindings.cpp`, kernels/allocations in `src/kernels.cu`)
1. **Pinned host buffers.** `tc.pinned_empty(shape, dtype)` → a host buffer allocated with `cudaHostAlloc`
   (default flags; report whether `cudaHostAllocPortable` is needed), freed with `cudaFreeHost` when the last Python
   reference goes, exposing a zero-copy numpy view (buffer protocol / `py::array` with a capsule owner). Dtypes: the
   engine's (float32, float16, bfloat16 as raw 2-byte, uint8, int64, bool).
2. **Streams and events.** `tc.Stream(non_blocking=True)` (`cudaStreamCreateWithFlags(cudaStreamNonBlocking)` — the
   engine launches everything on the LEGACY default stream, which implicitly synchronises with blocking streams, so the
   copy stream MUST be non-blocking to overlap), `tc.Event()` (`cudaEventDisableTiming` unless asked), `record(stream)`,
   `stream.wait(event)`, `event.query()`, `event.synchronize()`, and a way to record on / wait from the legacy stream
   the engine's kernels use.
3. **Asynchronous D2H.** `tc.copy_to_host_async(tensors, buffers, stream, after=event)`: for every pair, a
   `cudaMemcpyAsync(DeviceToHost)` on `stream` after `stream.wait(after)`; returns immediately with the GIL RELEASED;
   returns an event recorded after the last copy. One call for the whole list (1,110 tensors), not one Python call each.
   Refuse a pageable destination (async into pageable memory silently degrades to a staged synchronous copy).
4. **Device copy (the CC47-B workaround's replacement).** Bind `Tensor.copy_(src)` (in place, same shape/dtype,
   `cudaMemcpyAsync` DeviceToDevice on the current stream; bitwise) and a multi-tensor form
   `tc.copy_many_(dsts, srcs)` (one launch over pointer/size arrays, or one `cudaMemcpyAsync` each from C++ — measure
   both). No autograd (in-place; refuse under grad like `write_rows`, or document why it is safe on leaf data).
5. **The hazard, as a documented recipe + a test.** The optimizer updates parameters and Adam m/v IN PLACE
   (`adam_step`). A D2H straight from the live tensors would race the next step. The recipe GRAPA will use:
   after `opt.step()` on the legacy stream → `copy_many_(device_staging, live)` (≈ 12 ms, legacy stream) → record
   `e_staged` → the copy stream waits `e_staged` and D2Hs the staging into host slot `i` → record `e_host[i]`; before
   the NEXT `copy_many_` into the staging, the legacy stream waits the previous `e_host` (so the staging is never
   overwritten while it is being read). The host slot is readable once `e_host[i]` completes. Provide the wait primitive
   and prove the recipe in a test (below).
6. **Accounting.** Pinned bytes held (process total) queryable; `cudaMemGetInfo` bound (`tc.mem_get_info()` → free,
   total) so GRAPA stops reaching into libcudart through ctypes (CC47-B `grapa/tripwire.py device_mem_info`).

## Gates (the seat builds them; the lead runs the GPU half in a slot)
- Bitwise round trip for every dtype, every rank 0–4, empty tensors, non-contiguous inputs refused or handled (say
  which); -0.0 and NaN payloads preserved.
- Overlap proof: a 2.94 GB D2H on the copy stream concurrent with a synthetic compute loop on the legacy stream; the
  compute loop's wall time within 3 % of its solo time; the copy's events show it ran concurrently (event timestamps).
- Hazard proof: the step-then-stage-then-D2H recipe above with an in-place update of the live tensors right after the
  stage; the host slot equals the pre-update state bitwise; without the wait primitive the test shows the race (or
  explains why it cannot on this hardware).
- Per-step cost of a ring of 3 at the run-7 geometry (1,110 tensors, 2.939 GB, from
  `/mnt/ForgeRealm/grapa_snapshots/run7_step32055_prespike.ckpt`, read-only): host-side time per step and device-time
  added to the legacy stream; target ≤ 1 % of a 5.09 s step. Receipt create-only.
- Pinned-memory limits: 3 × 2.94 GB = 8.8 GB pinned on the 62 GB host (non-pageable; check `ulimit -l` and the driver's
  behaviour; state what happens at the limit — refuse, never hang).
- Existing kernels and numerics untouched: the engine's own suite green; bf16/fp32 goldens byte-identical; source pins
  updated only for the files this order changes.

## Deliverables
Bindings + Python surface + tests; an engine ledger entry (`## Done` with receipts); the GRAPA-facing API note (the
recipe in 5, the exact calls) so a GRAPA order (CC47-C, lead) can switch `--tripwire-ring host` to pinned async slots
fed from the CC47-B device staging.

## Prior art (Prior Art Directive — unverified, lead to check)
- CUDA streams, events, pinned (page-locked) memory and `cudaMemcpyAsync` overlap: NVIDIA CUDA C++ Programming Guide
  ("Asynchronous Concurrent Execution", "Page-Locked Host Memory") and Best Practices Guide ("Asynchronous and
  Overlapping Transfers with Computation"). Taken: the mechanism entirely.
- PyTorch: `Tensor.pin_memory()`, `tensor.to('cpu', non_blocking=True)` on a side `torch.cuda.Stream`, and
  `Tensor.record_stream` / event-based lifetime; `torch._foreach_copy_` and NVIDIA Apex `multi_tensor_apply`
  (one launch over many tensors). Taken: the API shape.
- Checkpoint pipelines that snapshot to device/host memory and persist asynchronously: CheckFreq (Mohan et al., FAST
  2021: snapshot() then persist()), Check-N-Run (Eisenman et al., NSDI 2022), Gemini (Wang et al., SOSP 2023), and
  asynchronous distributed checkpointing in PyTorch DCP (`async_save`, 2024). Taken: stage on the device, drain to
  pinned host memory off the critical path.
- Ours: the concrete recipe for the engine's legacy-stream world (a non-blocking copy stream + the staging wait), the
  gates, and the run-7 geometry target.
- Search terms: "CUDA cudaMemcpyAsync pinned memory overlap non-blocking stream legacy default stream",
  "PyTorch pin_memory non_blocking record_stream", "CheckFreq FAST 2021 snapshot persist", "Check-N-Run NSDI 2022",
  "Gemini SOSP 2023 in-memory checkpoint", "PyTorch DCP async_save".

## Boundaries (proposed; the lead's registration binds)
- WRITABLE TARGET: the engine worktree the lead names. CPU-only in the seat; the GPU gates run in the lead's slot.
  No git; no subagents; never touch run 7, its engine paths or `/tmp/forge-gpu.lock`.
