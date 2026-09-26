# PT-RING-1 report

**Implemented and built in the fork; 126 CPU tests passed. GPU acceptance is
BLOCKED by the registered CPU-only order.** The 18–23% capture overhead and
affordability of a three-state host ring are **not claimed fixed**.

Author baseline only; no blind review. No GPU execution, live-engine
edit/build/import, git, subagents, production-lock access, or unowned process
signals occurred. The original order and previous campaign receipts are intact.

## Delivered

`tensor_cuda/src/bindings.cpp`, `tensor_cuda/tensor_cuda/__init__.py`, two new
headers under `tensor_cuda/include/tc/`, and an additive section in
`tensor_cuda/src/kernels.cu` provide pinned buffers with NumPy owners, streams,
one-shot events, batched async D2H, guarded bitwise device copies, direct staging
allocation, explicit pinned accounting/limits, cleanup and device memory info.

Default D2D uses C++ memcpy batching. The explicitly selected kernel alternative
uses one pointer/size descriptor transfer and one chunked copy kernel. Both
are built for measurement; neither has a measured speed claim. Whole-list
validation precedes enqueue. C++ pending ownership protects storage through
completion without a background worker. The caller must still order writes and
protect CPU readers from ring-slot reuse.

The exact GRAPA calls and dependency recipe are in
[`PT_RING_1_API.md`](PT_RING_1_API.md). Engine operations still use the legacy
stream. The API is scoped to logical device 0; default host allocation flags
suffice for that single-context scope. Pageable destinations are refused.
Noncontiguous NumPy inputs use the engine's existing materialization; its
NDArrays are contiguous. bf16 pinned views expose raw uint16 bits.

## Evidence and residuals

| Evidence class / gate | Result | Receipt |
| --- | --- | --- |
| CPU compilation, final rebuild | PASS, 60.77024424355477 s; inherited warnings retained | `artifacts/pt_ring_1/build_02/receipt.json`, `build.log` |
| Author CPU tests | **126 passed in 2.39s** | `artifacts/pt_ring_1/handoff/pytest.log`, `receipt.json` |
| GPU test discovery | **121 collected; zero executed** | `artifacts/pt_ring_1/final_cpu/gpu_collection.log` |
| Source inspection | Only three pre-existing engine files changed; all prior kernel bytes preserved | `artifacts/pt_ring_1/SOURCE_AUDIT.json` |
| Checkpoint schema inspection | 1110 tensors, **2939091552 bytes/state**, **8817274656 bytes/ring** | `artifacts/pt_ring_1/GEOMETRY.json` |
| CPU-hidden handoff guards | All seven lanes plus `all` exit 2 before engine import/output creation | `artifacts/pt_ring_1/handoff/receipt.json` |
| Dtype/payload/rank/lifetime correctness | **BLOCKED**, not executed on GPU | `tensor_cuda/tests/test_async_copy.py` prepared |
| Overlap, hazard and <=1% ring cost | **BLOCKED**, neither D2D method measured | `scripts/pt_ring_1.py` prepared |
| Full pinned ring / driver-limit behavior | **BLOCKED**; only host RLIMIT and deterministic software-budget refusal tested | `artifacts/pt_ring_1/BLOCKED_REPORT.json` |
| Engine GPU suite / fp32 and bf16 goldens | **BLOCKED** | Preserved baseline `.so` and comparison lane prepared |

Exact CPU test files run: **`tests/test_pt_ring_1_cpu.py`** and
**`tests/test_pt_det_2_cpu.py`**. `tensor_cuda/tests/test_async_copy.py` was only
collected. The full engine suite was **not run** in this seat. The CPU interval
test checks 2000 cases against an independent quadratic oracle; contract and
receipt counterexamples are finite author checks, not adversarial certification.

Host `RLIMIT_MEMLOCK` is 8417419264 bytes, below the requested ring. The API's
12 GiB engine-owned cap rejects larger requests before calling CUDA; driver
errors propagate without retry/fallback. Driver pinning availability and time
at this scale remain unmeasured. A driver hang cannot be ruled out by source
reasoning: the lead harness enforces a worker timeout and records BLOCKED.

The preserved baseline binary matches the prior PT-DET-2 build receipt exactly,
and all overlapping baseline source pins match. Existing kernel source bytes
are unchanged, but that is not a GPU numerical parity result. The prepared
goldens cover finite matmul/GELU/reduction/backward in fp32 and bf16; broader
model and every-kernel equivalence are outside that finite check. Historical
campaign pins are not weakened; any full-suite failures remain visible.

## Handoff

The lead supplies an authorized exclusive GPU slot. No lock operation is in the
harness. Fresh receipts only:

```bash
cd /mnt/ForgeRealm/wt/pt-tf32
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
  python -B scripts/pt_ring_1.py --lead-gpu --gate all \
  --out artifacts/pt_ring_1/lead_01
```

Sequential, stops at first non-GREEN lane. 900-second total cap including a
10-second receipt reserve; 90 seconds per non-suite lane (both golden processes
together), 300 seconds for the engine suite. Use individual gate choices in the
API note when the lead splits slots. Tests are named in every process receipt.
No GPU result has been prefilled or inferred from CPU success.

Additive seal: `artifacts/pt_ring_1/SOURCE_MANIFEST.json`, SHA256
`d938b3cf3a6cd7aa34a3dfe4ea4011406a4762501a6783f8543ab4996bf88f1a`.
The immutable plan, registration, protocol amendment, exact geometry, build,
binary, tests and harness are all sealed. Only this order's changed/new source
pins differ; historical manifests remain unchanged.

## Prior art

- **NVIDIA CUDA 12.6 (2024):** the pinned-memory, async copy, stream/event and
  timing mechanisms are taken from the
  [runtime API](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-runtime-api/group__CUDART__MEMORY.html).
- **PyTorch (2016 onward; 2024 tutorial):** API shape and event-bound lifetime
  are taken from [pinned/non-blocking transfer](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
  and `record_stream` idioms. **NVIDIA Apex (project 2018 onward):** chunked
  pointer-list batching is prior art, verified in
  [multi_tensor_apply](https://github.com/NVIDIA/apex/blob/master/csrc/multi_tensor_apply.cuh).
- **CheckFreq, Mohan et al., FAST 2021:** two-phase checkpoint context,
  [primary publication](https://www.microsoft.com/en-us/research/publication/checkfreq-frequent-fine-grained-dnn-checkpointing/).
  Check-N-Run (Eisenman et al., NSDI 2022), Gemini (Wang et al., SOSP 2023), and
  PyTorch DCP async-save (2024) are related context, not implementations imported
  here. Primary links and boundaries are in the API note.
- **pybind11/Jakob 2015, Python ctypes 2006, PEP 578 2019, NIST SHA256 2001,
  standard C++ binary search and POSIX:** established ownership, audit, checking
  and receipt mechanisms reused. The contribution is this engine's integration,
  contracts and run-7 gates; no new copy/checkpoint algorithm is claimed.

The same attribution appears at code sites and in the implementation ledger.
