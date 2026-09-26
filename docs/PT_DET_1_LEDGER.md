# PT-DET-1 implementation ledger

Status: implementation in progress; GPU certification BLOCKED by order.

Immutable plan: `orders/PT_DET_1_EMBED_BWD.md`. House rules read before work.
Registration written before implementation or gates: `artifacts/pt_det_1/REGISTRATION.json`, SHA256 `0b1af9f61695c5cf14ae6b0ecedac255dfebc4f18ac9821ed250751d79d48033`. Amendments will be separate files.

## Initial source findings

Evidence class: read-only source/evidence inspection. Embedding backward uses FP32 atomicAdd; real CC46 batches exist at the registered archive path. CC46-C reports only emb.weight differed at step one. Existing PT-TF32-4 runner lacks a repro lane. The registered CUB candidate remains provisional until GPU measurements; no cost claim is possible on this CPU-only seat. Dense one-hot cost is 34.36 billion multiply-accumulates, or 68.72 GFLOP at two FLOP/MAC.

All build/tests use CUDA_VISIBLE_DEVICES empty; writes are inside this fork; no git, subagents, live engine import, or production-lock operation.

## Prior art

NVIDIA CUB stable radix sort/segmented reduction (CUDA 12.6, 2024): use stable token grouping. PyTorch 1.9 (2021) deterministic index_add: take the opt-in API pattern. Demmel and Nguyen (2013), Fast Reproducible Floating-Point Summation: motivation for reproducibility only; fixed-order FP64 accumulation here is not their order-independent reproducible accumulator. CC46-B/C (2026): reuse replay, receipt and first-step digest mechanics. PT-DET-1 adds the embedding mode arms and required certification gate. Primary-source verification and final code locations follow below.

## Amendment 001: slot composition (before gate execution)

The dedicated PT-DET-1 slot is capped at 1500 seconds (80 embedding, 1400 replay, 20 reserve). The historical TF32 slot already budgets 1035 seconds for its full sequence. `AMENDMENT_001.json` requires a verified same-build PT-DET-1 replay receipt before any subsequent TF32 slot GPU work, plus a named required lane. It does not alter the old 1200-second slot or any numeric gate. Historical manifests remain intact; the new manifest verifies the additive build.

## Build 01

Evidence class: CPU-only compilation. `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -B scripts/pt_det_1_build.py`. Receipt line verbatim:

```text
BUILD_RC=0 SOURCE_PINS_UNCHANGED=True RECEIPT=/mnt/ForgeRealm/wt/pt-tf32/artifacts/pt_det_1/build_01/receipt.json
```

Existing warnings remain: unused `lane` at kernels.cu:3056; nvlink skipped incompatible librt.a/libpthread.a/libdl.a archives. No warning suppression, engine import, or device operation. Full transcript in build_01/build.log.

## CPU baseline and replay compatibility finding

`cpu_tests_01/pytest.log` verbatim: `41 passed in 1.42s`; `PT_DET_1 CPU_TESTS_RC=0`. Author baseline only.

Amendment 002 registers the mandatory cruise precision-record adapter before testing it. Source `cc46_trace.py:515` compares complete records, including engine identity. Replaying arm a on the rebuilt engine would otherwise fail before training. The adapter permits only old-to-verified-new engine identity in precision records; it rejects policy drift, and the actual checkpoint/logs keep the new identity. No controller/config/receipt file is edited.

## CPU real-batch arithmetic

Evidence class: CPU execution of the shared C++ reduction; synthetic FP32 dY, real CC46 token IDs. `cpu_real_01/summary.json` records x_32055, x_32083, x_32110; repeated-position fractions 0.981689453125, 0.981689453125, 0.98193359375. Five byte-identical results per batch. Receipt verbatim:

```text
PT_DET_1 CPU_EMBED GREEN batches 3 max_rel_L2 2.5424602395914753e-08
```

CPU medians are 23.245, 23.216, 23.413 ms; these are NOT CUDA latency or a cost-gate reading.

## CPU tests 02: retained RED fixture failure

The first expanded suite included a fixture that redirected the legacy artifact root without pinning the real registration reader. This test-fixture error happened while enumerating blocked lanes, before any GPU/import/lock path. Corrected by supplying the unchanged already-read registration to that fixture. Assertions and gate rules are unchanged. Full transcript: `cpu_tests_02/pytest.log`. Verbatim:

```text
1 failed, 45 passed in 2.15s
PT_DET_1 CPU_TESTS_RC=1
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/ForgeRealm/wt/pt-tf32/artifacts/pt_det_1/cpu_tests_02/tmp/test_legacy_slot_cannot_reach_0/pt4/AMENDMENT_001.json'
```

## RED containment finding and Amendment 003

The bridge g++ invocations in the earlier CPU tests/real-batch checks did not override TMPDIR; environment inspection returned `TMPDIR=None`. GCC scratch was not explicitly confined and may have used transient files in `/tmp`. No past syscall trace exists, so exact transient paths cannot be certified. This corrects the earlier blanket assertion that every write stayed in the fork. The CUDA build did set an in-fork TMPDIR. No GPU or production-lock operation occurred.

The two CPU bridge call sites now set TMPDIR explicitly to in-fork scratch. Subsequent checks also set the parent process TMPDIR. Earlier receipts and seal are retained; the final code uses create-only `SOURCE_MANIFEST_002.json`. This is a containment correction, not a changed gate threshold.

## Final implementation and audit

New production source: `tensor_cuda/src/embed_deterministic.cu` and
`tensor_cuda/include/tc/deterministic_embed.h`. Modified declarations,
bindings, CMake source list, and the opt-in branch at
`tensor_cuda/src/kernels.cu:4689`. The original atomic kernel and OFF body
are source-byte-identical; `ops.cpp` is unchanged. Stable token/position
sorting precedes fixed-position-order FP64 accumulation with one output
writer. CPU and CUDA use the same scalar reduction. Default OFF; environment
initializer is exactly TC_DET_EMBED_BWD=1; explicit thread-local setter wins.
The bindings also expose a raw backward hook for the full-dispatch benchmark.

New harness files: `scripts/pt_det_1.py`, `scripts/pt_det_1_slot.py`,
`scripts/pt_det_1_build.py`. New tests: `tests/test_pt_det_1_cpu.py`,
`tests/pt_det_1_cpu_bridge.cpp`, `tests/pt_det_1_host_contract.cpp`.
The certification runner and manifest adapter change only
`scripts/pt_tf32_4_slot.py` and `scripts/pt_tf32_4.py`.

The required lane rederives all four pairs from hashed underlying receipts
and rejects missing/partial/nonfinite/duplicate rows, tensor schema changes,
ULP and signed-zero differences, stale source/build identity and log tampering.
First-step gradient hashes and full-precision norms retain CC46 mechanics.
The inherited slot checks the required receipt BEFORE any GPU work and does
not offer an opt-out. The dedicated PT-DET-1 slot is <=1500 s; the prior
1200-second TF32 slot still has its original per-lane GPU budgets.

The source audit is `docs/PT_DET_1_AUDIT.md`, with machine-readable source
hashes/sites in `artifacts/pt_det_1/ATOMIC_SITE_RECEIPT.json`. Other contended
backwards: attention a/b/d, general gather/top-k, convolution/average/max
pooling. Neither registered live route selects those attention variants;
the loss gather has one destination per distinct token position, so its
current use has no collisions. Norms, Adam and graph accumulation have fixed
ownership/order in source. Vendor GEMMs still need empirical replay.
No other backward implementation was fixed.

## Final prior art record

Code sites: `deterministic_embed.h:9`, `embed_deterministic.cu:15` and `:34`,
`kernels.cu:4691`, `bindings.cpp:606`, and harness module/function docstrings.
NVIDIA CUB/Merrill stable radix sort and sorted segmented reduction are taken
(installed CUDA 12.6, 2024; header copyright Merrill 2011/NVIDIA 2011-2023).
The installed NVIDIA header at
`/usr/local/cuda-12.6/include/cub/device/device_radix_sort.cuh:109` states
stability. The web documentation URL returned 404 and the versioned GitHub
URL returned a cache miss; local primary source supplies the verification.

PyTorch 1.9 (2021), https://pytorch.org/blog/pytorch-1-9-released/,
was checked as the opt-in deterministic index_add precedent. Demmel and
Nguyen (ARITH 2013), https://www.acsel-lab.com/arithmetic/arith21/papers/p54.pdf,
was checked for reproducible-summation context. Their order-independent
accumulator is NOT implemented. Ours is this engine integration, owner-row
FP64 reduction, shared CPU arithmetic, four embedding-mode replay arms and
required lane. CC46-B/C/PT-TF32 (2026), SHA256 (NIST 2001) and POSIX process
facilities supply the replay, receipt and deadline mechanisms. These are
adaptations of prior art, not claims of a new summation/sorting algorithm.

## Done

**Built; final author CPU checks pass. GPU gates remain BLOCKED. Earlier
unconfined compiler-scratch handling is retained as a RED containment
finding, not erased by the successful rerun.** No original training-divergence
or <=2x GPU-cost success is claimed.

Final active seal: `SOURCE_MANIFEST_002.json`, SHA256
`b04c8ea7aaddbc5df67d342b979bbf62d760d2ebedf4aae1a319d1424151c376`.
Initial `SOURCE_MANIFEST.json` remains historical, SHA256
`fadca79c8555d682313cefdb046c7639e946c3a1ee6667dcab4f0c7847bd8d0a`.
The order registration remains unchanged. All three amendments are separate.

Binary: `tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`,
10,193,936 bytes, SHA256
`0d4e1c5c9bfd9d7b38b9e43bba6f6dbf154b4b1b31b5ef74bf4fecc7d3e15a7e`.
Build 01 took 74.97754774801433 s; CPU compilation evidence only.

Final commands used CUDA_VISIBLE_DEVICES empty, PYTHONDONTWRITEBYTECODE=1,
TMPDIR=/mnt/ForgeRealm/wt/pt-tf32/artifacts/pt_det_1/final_tmp,
OPENBLAS_NUM_THREADS=2 and OMP_NUM_THREADS=2. Final test receipt stores the
exact pytest argv and environment. The code itself now also confines both
CPU compiler scratch locations.

Receipts verbatim (`build_01/receipt.json`, `final_cpu/pytest.log`,
`final_cpu/real_batches.log`, `final_handoff/*.log`):

```text
BUILD_RC=0 SOURCE_PINS_UNCHANGED=True RECEIPT=/mnt/ForgeRealm/wt/pt-tf32/artifacts/pt_det_1/build_01/receipt.json
59 passed in 2.62s
PT_DET_1 FINAL_CPU_RC=0
PT_DET_1 CPU_EMBED GREEN batches 3 max_rel_L2 2.5424602395914753e-08
PT_DET_1 FINAL_REAL_BATCHES_RC=0
PT_DET_1 FINAL_HANDOFF repro rc=2 expected=2
PT_DET_1 BLOCKED BLOCKED: lead GPU slot required; CUDA_VISIBLE_DEVICES is empty or --lead-gpu absent
PT_DET_1 FINAL_HANDOFF embedding rc=2 expected=2
PT_DET_1 BLOCKED BLOCKED: lead GPU slot required; CUDA_VISIBLE_DEVICES is empty or --lead-gpu absent
PT_DET_1 FINAL_HANDOFF slot rc=2 expected=2
PT_DET_1 SLOT BLOCKED BLOCKED: lead GPU slot required; CUDA_VISIBLE_DEVICES is empty or --lead-gpu absent
PT_DET_1 FINAL_HANDOFF plan rc=0 expected=0
PT_DET_1 FINAL_HANDOFF sequence rc=0 expected=0
```

The 59 checks are 51 PT-DET-1 checks and 8 inherited certification checks.
Real-token CPU results: 3 batches x 5 bitwise-identical outputs; maximum
relative L2 2.5424602395914753e-8. CUDA ms and the <=2x ratio are unmeasured.
The CUB choice is provisional until the GPU measurements; dense one-hot
SGEMM is documented only (34.36 billion MACs / 68.72 GFLOP, no timing claim).

Current blocked report: `artifacts/pt_det_1/BLOCKED_REPORT_002.json`.
No GPU calls, live-engine import/edit/rebuild, git, subagents, or production
lock operations were executed. These author checks are not blind review.
Earlier temporary-file containment is qualified explicitly in Amendment 003.

Lead sequence (already-held exclusive production lock inherited as FD 9;
no acquisition/wait by this seat or this script):

```bash
cd /mnt/ForgeRealm/wt/pt-tf32
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
  python -B scripts/pt_det_1_slot.py --lead-gpu --lock-fd 9 \
  --out artifacts/pt_det_1/lead_slot_01
```

Embedding 80 s, repro 1400 s (eight full 30-step runs), reserve 20 s:
global maximum 1500 s. A timeout is BLOCKED, never a shorter gate. This cap
is not a promise the workload will fit. Four arm predictions remain ON =
bitwise and OFF = differs. A bitwise OFF control is NOT_RECURRED and blocks
attribution. Existing TF32 model-lane receipts retain older engine pins;
fresh additive model registration is still needed for those optional lanes.

Narrative report: `docs/PT_DET_1_REPORT.md`. No lead GPU result is fabricated.
