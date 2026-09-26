# PT-TF32-4 report

**Rebuilt and ready for the lead's slot. All 78 current CPU tests pass;
GPU certification remains BLOCKED.** No GPU, git, subagents, live-engine
writes or lock operations were used.

dK retains its L2 bar and now has a separately derived extreme-value bar,
0.00490652734648425–0.005009733474008307 across the four shapes. All 12 old
dK comparisons fit it. This is a conditional sub-Gaussian **peak-scale**
engineering model: RMS alone does not prove the per-element scale assumption.
The registration states that limitation and fixes the risk/margin before
historical comparison. The old RED receipts remain untouched.

GEMM now gates actual HMMA/TF32/FP32 capability flags and FP64 rel-L2 <=1e-3
for every shape/direction/allocation policy; speedup is retained diagnostically.
All 54 old cells, including dWeight, meet those flags and accuracy criteria.
Removed the invalid capability-8 query that returned status 7; the supported
NUMERICAL_IMPL_FLAGS query now records its actual status and byte count.

The dV padding tests use input-derived per-element probability-rounding and
accumulation intervals across all 40 fixtures. This accounts for TF32 midpoint
jumps without selecting failing shapes/coordinates or using observed errors
as tolerances. A separate amendment accounts for NVIDIA's unspecified WMMA
rounding. dQ/dK's small-shape tolerances and the independent FP64 edge tests
remain unchanged. **Rebuilt-native edge passes are not claimed.**

The [ledger](PT_TF32_4_LEDGER.md) contains complete derivations, assumptions,
prior-art attribution, historical comparisons, commands and exact failures.
[REGISTRATION.json](../artifacts/pt_tf32_4/REGISTRATION.json) and
[AMENDMENT_001.json](../artifacts/pt_tf32_4/AMENDMENT_001.json) are immutable.
[BLOCKED_REPORT.json](../artifacts/pt_tf32_4/BLOCKED_REPORT.json) pins the
unmeasured gates to the final binary and manifest. No blind review or
training-stability result is claimed.

## Done

Verbatim receipts:

```text
BUILD_RC=0 SOURCE_PINS_UNCHANGED=True RECEIPT=/mnt/ForgeRealm/wt/pt-tf32/artifacts/pt_tf32_4/build_01/receipt.json
HOST_DESCRIPTOR_RC 0 CASES 27
78 passed in 1.08s
60 tests collected in 0.12s
3 failed, 146 passed in 3.26s
E           ValueError: fork binary drift
E           ValueError: fork binary drift
E       ValueError: fork engine drift
PRESERVED 898 OF 902 UNEXPECTED_CHANGES []
HISTORICAL_RECEIPTS_BYTE_IDENTICAL True
BF16_FP16_MATMUL_SOURCE_BYTE_IDENTICAL True
KERNELS_CU_AND_OBJECT_BYTE_IDENTICAL True
ATTENTION_TF32_SOURCE_BYTE_IDENTICAL True
```

The three RED historical CPU checks refer to immutable PT-TF32-1/2/3 binary
pins. The current registration/manifest passes; historical assertions are
preserved. Eight large legacy dumps are stat-pinned only, explicitly separate
from byte-hash preservation claims. Full build, CPU, collection and preservation
receipts are under `artifacts/pt_tf32_4/`.

Final binary SHA256:
`cbb11e8c5cc7c6ffba4e603183310e830d03df31ef1ef4390740d1c91166b3c5`.

The lead supplies one exclusive GPU slot and runs the following command;
this seat did not execute it:

```bash
cd /mnt/ForgeRealm/wt/pt-tf32
CUDA_VISIBLE_DEVICES=0 python3 -B scripts/pt_tf32_4_slot.py \
  --lead-gpu --out artifacts/pt_tf32_4/lead_slot_01
```

Sequence: dispatch (15 s), units (30), GEMM (60), attention x4
(35/35/90/90), memcheck/racecheck/synccheck (30/60/30): **475 s** total
budgets, with a **1200 s** global deadline. Add `--include-model` to that
single invocation for fresh noise/onset/healthy/control/step-time lanes;
the complete budget is then **1035 s**, under the same deadline. Complete
argv arrays: [required lanes](../artifacts/pt_tf32_4/LEAD_SEQUENCE.json),
[with model lanes](../artifacts/pt_tf32_4/LEAD_SEQUENCE_WITH_MODEL.json).
Use a fresh output directory on subsequent runs.

Default gradient/checkpoint dumps: **0 bytes**. The 8-GiB space rail,
pipe/RAM transport and ENOSPC/timeout BLOCKED handling are preserved.

## Prior art

Taken: Gumbel (1958), [Statistics of Extremes](https://doi.org/10.7312/gumb92958),
David & Nagaraja (2003), [Order Statistics](https://doi.org/10.1002/0471722162.ch4),
for Gaussian maxima; Higham (2002),
[rounding/error bounds](https://epubs.siam.org/doi/10.1137/1.9780898718027.fm);
NVIDIA TF32 (2020),
[CUDA 12.6 exp/WMMA](https://docs.nvidia.com/cuda/archive/12.6.2/cuda-c-programming-guide/index.html#intrinsic-functions),
[PTX 8.5 accumulation limitations](https://docs.nvidia.com/cuda/archive/12.6.2/parallel-thread-execution/index.html#warp-level-matrix-instructions-wmma-mma),
and [cuBLASLt flags](https://docs.nvidia.com/cuda/archive/12.6.2/cublas/index.html#cublasltmatmulalgocapgetattribute)
(2024); Williams/Waterman/Patterson (2009),
[Roofline](https://digital.library.unt.edu/ark:/67531/metadc934195/).
Existing PT-TF32/CC39/CC41/BP harnesses, NumPy, pytest, POSIX subprocess and
SHA256 supply the fixtures, exact transport and receipts.

Ours: the explicit scale/risk registration, grouped dV interval placement,
supported-query repair, actual-shape receipt checks and bounded slot. No
novelty claim for these underlying methods. Book bibliography was verified,
not an inaccessible numbered theorem. No prior art known to me for the
unchanged user's exact four-arm cosine calibration rule.
