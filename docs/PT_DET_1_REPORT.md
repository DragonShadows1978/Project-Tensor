# PT-DET-1 report

**Implemented and rebuilt in the fork; 59 final CPU checks passed. GPU
certification is BLOCKED. An earlier compiler-scratch containment miss is
recorded RED.** The original training divergence and the ≤2× GPU cost bar
are **not claimed fixed**.

Plan: [immutable order](../orders/PT_DET_1_EMBED_BWD.md).
Evidence: [ledger](PT_DET_1_LEDGER.md), [atomic audit](PT_DET_1_AUDIT.md),
[current blocked report](../artifacts/pt_det_1/BLOCKED_REPORT_002.json).

## Implementation

`tensor_cuda/src/embed_deterministic.cu` implements an opt-in backward:
stable CUB radix sort of token/position pairs, followed by one writer per
token/feature. Each writer sums ascending original positions in FP64, then
rounds once to FP32 before the existing weight-dtype conversion. Sorting
stays on the GPU without ID readback or a cached token list; this reduction
uses no float atomic. Allocation follows the engine's existing
default-stream allocator policy. CPU and CUDA share the scalar reduction
in `tensor_cuda/include/tc/deterministic_embed.h`.

Enable through `tc._C.set_deterministic_embed_bwd(True)` or
`TC_DET_EMBED_BWD=1`. The setting is thread-local, read at backward dispatch;
the setter overrides the environment initializer for that thread.
`tc._C.get_deterministic_embed_bwd()` reports it. Default is OFF. A source
comparison test proves the original atomic kernel body and OFF dispatch
body remain byte-identical; this is not a comparison of old/new SASS.
Only embedding backward changed. FP16/BF16 gradient dispatch compiled, but
its GPU execution is unmeasured; the model gate uses FP32 master gradients.

This is a **candidate pending measurement**, not a timing-selected winner.
The alternative dense one-hot product requires 34,359,738,368 MACs, or
68,719,476,736 FLOP counting multiplication and addition separately, plus a
128 MiB FP32 one-hot matrix at the registered dimensions. These are size
calculations, not timings. A sparse one-hot product reduces to the same
grouped gather/reduce operation; no separate sparse library path or dense
SGEMM implementation was added.

## CPU evidence

The final suite covers the actual shared C++ reduction, empty/unique/repeated
and cancellation cases, ragged widths, environment/setter behavior through
both the compiled host entry point and Python bindings, exact byte/text
diffs, malformed/incomplete/nonfinite logs and checkpoints, all four replay
receipt structures, receipt tampering, the strict engine-identity replay
adapter, and mandatory certification-lane handling. It consists of 51
PT-DET-1 checks plus 8 affected inherited certification checks.

Real CC46 IDs come from the SHA-verified `scratch_cc46/batches.npz`; each
4097-token array drops its last target token. The independent reference is
FP64 `numpy.add.at`, with seeded synthetic FP32 upstream gradients.

| Batch | Repeated-position fraction `1 - unique/L` | Five CPU calls bitwise | Relative L2 to FP64 |
|---|---:|---|---:|
| `x_32055` | 0.981689453125 | yes | 2.502350979251485e-8 |
| `x_32083` | 0.981689453125 | yes | 2.5424602395914753e-8 |
| `x_32110` | 0.98193359375 | yes | 2.5282777480651434e-8 |

Hashes and CPU milliseconds are in
[the final CPU batch receipt](../artifacts/pt_det_1/cpu_real_final/summary.json).
**Atomic GPU ms: unmeasured. Deterministic GPU ms: unmeasured. Ratio:
unmeasured.** CPU timing does not satisfy the registered CUDA cost bar.

## Replay and certification

`scripts/pt_det_1.py repro --steps 30` runs 32,055 → 32,085 twice per arm,
with a fresh trainer process each time. The checkpoint, seed and trainer
argv are identical within each pair except output paths. All four arms use
this rebuilt fork and the v3 checkpoint/corpus. BF16 removes the two
precision-policy flags. No live engine is imported.

| Arm | Registered prediction |
|---|---|
| v3, deterministic embedding ON | bitwise |
| BF16, deterministic embedding ON | bitwise |
| v3, deterministic embedding OFF | differs |
| BF16, deterministic embedding OFF | differs |

The gate requires complete finite logs, exact loss/gnorm text, matching
batch cursors, actual precision-mode records, final checkpoint geometry,
all model tensor dtype/shape/bytes, Adam moments, first-step parameter
gradient hashes, and full-precision norm records. Source/config/engine
hashes are pinned and rechecked. Missing or partial runs cannot pass.
A control that repeats bitwise is `NOT_RECURRED`, not an invented failure
or a successful treatment attribution.

CC46's cruise receipt is replayed without running sensors or writing live
receipts. Amendment 002 translates only the historical engine identity
after checking that every other v3 precision-record field is unchanged.
Actual checkpoint and log provenance still identifies the rebuilt engine.

The harness accepts an **inherited, already exclusively locked descriptor**;
it verifies `/proc/self/fdinfo` and never opens/acquires/unlocks the production
lock. This seat exercised only hidden-CUDA guards, which return before
descriptor inspection. Training checkpoints/logs/cache/scratch go into the
fork. Complete checkpoints are digested and removed by default, one at a
time; `--keep-ckpts` is explicit. Timeout cleanup signals only children the
harness created, and preserves logs/receipts.

`scripts/pt_tf32_4_slot.py` now requires `--det-repro <repro/summary.json>`
for the same source seal and binary, verifies its underlying receipts before
GPU work, and records `pt_det_1_repro` as a mandatory lane. Amendment 001
keeps the inherited slot's 1200-second budget intact: the replay runs in
its own bounded slot, then its receipt is verified by future certification.
Historical registrations/manifests are retained. The optional inherited
GRAPA model lanes still pin the older binary and need a fresh additive
model registration before they can execute on this build; their old results
are not reused as certification of the new binary.

## Lead slot, at most 25 minutes

Run only in the lead's authorized GPU slot, with its existing exclusive
lock inherited as FD 9. Choose a fresh output directory:

```bash
cd /mnt/ForgeRealm/wt/pt-tf32
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
  python -B scripts/pt_det_1_slot.py --lead-gpu --lock-fd 9 \
  --out artifacts/pt_det_1/lead_slot_01
```

1. Embedding lane: **80 seconds**. Three real batches, five repeated results,
   FP64 reference error ≤1e-6, three warmups and ten interleaved timed calls
   per mode; every median deterministic/atomic ratio must be ≤2. Timing
   includes backward allocations, sorting, zero fill and dtype conversion.
2. Required replay lane: **1400 seconds**, all eight 30-step processes and
   byte/text comparisons. ON arms must repeat and OFF controls must differ.
3. Receipt/cleanup reserve: **20 seconds**. Global deadline **1500 seconds**.

This is a hard cap, not a measured completion estimate. Startup, checkpoint
I/O and 240 training steps must fit; a timeout stays BLOCKED and never
reduces the number of steps. The CPU-only sequence is saved in
[final_handoff/sequence.log](../artifacts/pt_det_1/final_handoff/sequence.log).
The runner does not wait for, seize, or release a GPU slot.

## Done

The built binary is
`tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`,
SHA256 `0d4e1c5c9bfd9d7b38b9e43bba6f6dbf154b4b1b31b5ef74bf4fecc7d3e15a7e`.
The final implementation seal is
[SOURCE_MANIFEST_002.json](../artifacts/pt_det_1/SOURCE_MANIFEST_002.json),
SHA256 `b04c8ea7aaddbc5df67d342b979bbf62d760d2ebedf4aae1a319d1424151c376`.
Registration remains
`0b1af9f61695c5cf14ae6b0ecedac255dfebc4f18ac9821ed250751d79d48033`.

Receipts verbatim:

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

The earlier expanded suite had a test-fixture error (`1 failed, 45 passed
in 2.15s`); its transcript remains intact. No assertion was relaxed.

**RED containment record:** earlier host test-bridge compilations inherited
`TMPDIR` unset and may have used transient `/tmp` scratch. Exact past paths
were not traced; full write containment cannot be claimed for those runs.
Amendment 003 confines both compiler call sites, and the final 59 checks
and three-batch CPU rerun used explicit in-fork scratch. The CUDA build
already used in-fork scratch. No git, subagents, live engine edit/build/import,
GPU work, or production-lock operation was performed.

GPU repeatability, CUDA precision, cost, four training replay pairs,
sanitizers and blind verification remain unmeasured. This is an author
implementation handoff, not an engine certification.

## Prior art

- **CUB / Duane Merrill / NVIDIA (2011 onward; CUDA 12.6 used here, 2024):**
  stable radix sort and the standard sorted segmented scatter-add pattern
  are taken. Local stability evidence is the installed CUB header at line
  109. PT-DET-1 contributes the engine integration, fixed position-order
  FP64 owner reduction, and shared CPU arithmetic; it claims no new sorting
  or summation algorithm.
- **[PyTorch 1.9 (2021)](https://pytorch.org/blog/pytorch-1-9-released/):**
  opt-in deterministic indexing is the API precedent, including `index_add`.
- **[Demmel and Nguyen (ARITH 2013)](https://www.acsel-lab.com/arithmetic/arith21/papers/p54.pdf):**
  reproducible-summation context is taken; their order-independent
  accumulator is not implemented. Fixed-order FP64 summation here has a
  narrower claim.
- **CC46-B/C and PT-TF32 (2026), SHA256 (NIST 2001), POSIX process/lock
  facilities:** replay, digests, source seals and bounded owned-child
  execution are reused. PT-DET-1 adds four embedding-mode arms, strict
  coverage/diff checks, the narrow provenance adapter and required lane.

These annotations also appear at the code sites and in the ledger. PyTorch
and the Demmel/Nguyen paper were checked through primary web sources; CUB
stability was checked in the installed NVIDIA header. No cross-device or
cross-toolchain reproducibility theorem is claimed.
