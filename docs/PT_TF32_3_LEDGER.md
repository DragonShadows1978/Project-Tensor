# PT-TF32-3 implementation ledger

**BUILD + CPU HANDOFF COMPLETE; certification BLOCKED by the no-GPU order.**
Immutable plan: `orders/PT_TF32_3_CLOSE.md`. Read `/mnt/Shared/HOUSE_RULES.md`
before work. No git, subagents, GPU use, live-engine imports or lock operations.
All writes, including compiler temporaries, stay in this fork.

## Registration

`artifacts/pt_tf32_3/REGISTRATION.json` and its SHA256 sidecar were created
before implementation or CPU gates. All numerical/speed thresholds are inherited
unchanged from PT-TF32-2. A new storage/sanitizer protocol implements this order.
`BASELINE_SHA256.json` records historical receipts, source and compiled objects;
small editable sources are also copied into `baseline/`. Historical receipts
are immutable, including old RED verdicts and obsolete source/binary seals.

## Inherited evidence and diagnosis

Evidence class: slot-10 artifacts and source reasoning, not a new GPU run.
`lead_slot_01/gpu_units.log`: `2 failed, 54 passed in 0.91s`.
Both failures are dV (3968 and 4480 elements), not dK:
`Max absolute difference among violations: 3.32072377e-05` and
`Max absolute difference among violations: 2.30306759e-05`.
Their CPU model rounds probabilities from FP64 selected scores, whereas native
selected scores accumulate FP32 and use `__expf`; a TF32 midpoint is discontinuous.
This is a hypothesis to test, not yet a native root-cause certification.

Attention case 1 downstream dK normalized max error is
`0.002203109868528126` vs bound `0.0020796940929215554`; case 2 is
`0.004581897533772324` vs `0.0020830459898043387`. Same-native-state dK
passes. Forward selection has respectively 2 and 18 flips. The saved FP32
threshold uses sequential FP32 sums and sums of squares over up to 4096 keys;
its error can change whole selected-only dK contributions even when forward
out/LSE and the aggregate flip rate pass. No mask is saved for production
backward: both owners recompute against the saved threshold.

GEMM case 0 reports algo `21`, flags `262658` (HMMA + TF32 input + FP32
accumulator), speedup `1.3257044982268866`. Therefore the proposed silent
CUDA-core fallback is not established by this receipt. Source already calls
Lt heuristics with FAST_TF32. The event interval includes raw cudaMalloc output
allocation; model training enables allocation pooling. Both require direct
diagnostics in the new slot, without changing the >=5x gate or engine defaults.

Healthy/control/step_time each contain verbatim
`OSError: [Errno 28] No space left on device`: INVALID/BLOCKED, not numerical
RED. Each sanitizer completed with its own zero-error summary while pytest
returned 1. The new runner will record those distinct outcomes separately.

## Prior art

- NVIDIA TF32 (2020), CUDA/cuBLASLt 12.6 (2024): taken WMMA rounding, explicit
  compute descriptors, heuristics, capability/alignment queries and CUDA events.
  [Official cuBLAS documentation](https://docs.nvidia.com/cuda/archive/12.6.2/cublas/)
  verified this session. Its deprecated MATHMODE_IMPL attribute is absent from
  the installed header; the implementation must receipt API availability.
- Ootomo and Yokota (2022), arXiv:2203.03341: existing residual product
  decomposition; Dao (2023), FlashAttention-2: existing online softmax/VJP.
  Higham (2002): floating-point error and quantization-boundary reasoning.
  Any numerical additions are local integration, not novel underlying methods.
- Python subprocess/POSIX pipes, NumPy (Harris et al. 2020), SHA256 (NIST 2001):
  taken process isolation, exact array transport, hashes and sufficient dot/norm
  statistics. Ours: bounded ephemeral model results and fail-closed slot receipts.
  The four-arm cosine rule is inherited from the user; no prior art known to me
  for that exact calibration rule.

## Execution receipts

Entries below are appended as work completes. No native failure is claimed
fixed until the lead repeats its gate; no blind verification is claimed.

### CPU diagnostic harness correction

First author diagnostic failed before writing its result, not a numerical
pass: `RuntimeWarning: overflow encountered in scalar subtract`, followed by
`OverflowError: (34, 'Numerical result out of range')`. The exponent-bit field
was an unsigned NumPy scalar; convert it to Python int before subtracting 127.
This is a harness bug correction, with no threshold or kernel changes.

### Saved-state treatment and edge justification

`CPU_DIAGNOSIS.json` / `cpu_diagnosis.log` hold the controlled CPU experiment.
All three arms use identical rounded bulk scores; only moment accumulation
and threshold storage differ. Thus it isolates statistics, and does **not**
emulate native WMMA accumulation or certify treatment of native dK.

```text
THRESHOLD 2048 96 {'fp32_sum': 5, 'fp64_sum': 0, 'fp64_sum_fp32_store': 1}
THRESHOLD 2048 128 {'fp32_sum': 3, 'fp64_sum': 1, 'fp64_sum_fp32_store': 1}
THRESHOLD 4096 96 {'fp32_sum': 21, 'fp64_sum': 2, 'fp64_sum_fp32_store': 5}
THRESHOLD 4096 128 {'fp32_sum': 18, 'fp64_sum': 1, 'fp64_sum_fp32_store': 3}
```

Implemented FP64 sum/square accumulation in `attention_tf32.cu`; the final
threshold is rounded once to FP32 before forward compares, then saved in the
existing FP32 tensor for backward. No ABI, mask layout, selection rule, LSE
layout, or default/BF16 path changes. Prior art: wider-precision reductions
(Higham, 2002), taken; local saved-state placement is ours. The remaining
CPU flips are an explicit residual. **Downstream dK not claimed fixed.**

The two edge failures are justified by TF32 midpoint sensitivity, not removed
from the suite. For a normal probability in exponent bin e, adjacent TF32
values differ by `2^(e-10)`; changing one rounded probability gives
`Delta dV[b,kh,j,d] = +/-2^(e-10)*RN_TF32(dO[b,h,i,d])` before FP32 accumulation.
Near a midpoint, FP32 score/exp rounding can trigger that jump with arbitrarily
small input error. Cancellation in a near-zero dV element makes the fixed
absolute-plus-relative element tolerance incompatible with every such jump.
This is rounding-model reasoning (NVIDIA TF32 2020; Higham 2002), not a
larger budget or assertion suppression.

| L/S | h/i/j | dV coordinate (kh,j,d) | One-bin magnitude | Observed mismatch magnitude |
|---|---|---|---:|---:|
| 17/31 | 3/2/1 | (1,1,30) | 2.4646520614624023e-05 | 2.4646520614624e-05 |
| 17/31 | 3/2/1 | (1,1,42) | 3.319978713989258e-05 | 3.32072377e-05 |
| 33/35 | 3/18/0 | (1,0,22) | 2.3052096366882324e-05 | 2.30306759e-05 |

CPU fixture probabilities round to an FP32 number exactly at the TF32 midpoint
at both witness sites. The jumps match the receipt errors within 2.2e-8.
Native saved arrays were not retained in slot 10, so the exact native rounding
branch cannot be replayed here. The inherited assertions stay unchanged;
their failures remain possible and must remain RED if repeated. Two added
GPU tests independently apply the **existing** FP64 precision budget, without
overriding the legacy lane verdict. **Legacy edge passes not claimed.**

### GEMM dispatch and output allocation

Real host `cublasLtMatmulDescGetAttribute` checks FAST_TF32 (readback `77`),
both transpose flags, and layouts. The host-only self-check covers all 27
shape/direction cells and prints algorithm -1 / device_selected 0 when CUDA is
hidden. Device algorithm selection cannot be performed on CPU alone; the
lead's first lane calls the same planner with `cublasLtMatmulAlgoGetHeuristic`.
Actual matmul receipts include its own pointer/stride alignments (>=16 bytes),
algorithm, workspace, descriptor readback and numerical implementation flags.

The installed 12.6 header lacks deprecated `CUBLASLT_ALGO_CAP_MATHMODE_IMPL`.
The compatibility query uses legacy attribute slot 8, recording API status and
value; unsupported never masquerades as successful readback. That historical
numeric mapping is **unverified — lead to check** `CUDA 11 cublasLt.h
CUBLASLT_ALGO_CAP_MATHMODE_IMPL = 8`. The supported NUMERICAL_IMPL_FLAGS query
is mandatory and rejects anything lacking HMMA + TF32 + FP32 accumulation.

Source diagnosis: the old raw allocation inside the event interval can hide
kernel acceleration; this is not yet a measured causal attribution. Only
explicit TF32 outputs now use `cudaMallocAsync` and the existing pooled Storage
destructor on stream 0. There is no global allocator/default toggle. The old
FP16/BF16 branch text is byte-identical. Prior art: Kini and Hemstad/NVIDIA,
[stream-ordered allocation (2021)](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/),
taken allocation/lifetime rules; ours is placement on the opt-in TF32 output.

GEMM now measures both global-pool-off (SGEMM raw, TF32 output pooled) and
model-pool-on (both pooled). Both must clear unchanged >=5x and <=1e-3 bars;
the added model measurement cannot hide a RED original-policy measurement.
No speed claim is made. Diagnostic map copying was moved out of the GEMM hot
path before final handoff; the explicit getter alone copies the cached map.

### Storage and slot harness

New entry points use `pt_tf32_3*`; historical scripts and receipts remain intact.
Full FP32 gradients cross an anonymous pipe into parent RAM by default. Exact
FP64 cosines still compare every element. The immutable calibration receipt
keeps all six dot/norm statistics, four content hashes and four child receipt
hashes; holdouts validate these before running. This replaces the old dependency
on permanent gradient archives. `--keep-grads` opts into NPZ/checkpoint retention.

The model child adapts only its private CC39 module's output calls; checkpoint
saves are intercepted only in the isolated timing child. Trainer/checkpoint
loads, parameter updates, cursor checks and policy code remain read-only.
The fork engine is installed/verified **before** importing the checkpoint
module, which itself imports tensor_cuda. The trainer's historical text still
mentions a final checkpoint; `checkpoint_output.json` records whether each
save was actually suppressed. No retained checkpoint is implied by that text.

All slot lanes and model children check 8 GiB free before launch. Low space,
runtime ENOSPC and timeouts produce BLOCKED receipts. Sanitizer zero-error
summaries yield GREEN independently of pytest returncode; the unit returncode
is retained, and missing summaries/zero executed coverage cannot pass.
`--print-sequence` prints default dump bytes `0`, optional dump estimate,
and lane budgets `1705` seconds under a global `1800`-second deadline.
Prior art for pipes/statistics/receipts is listed above and at each code site.

### Build and initial author checks

```text
BUILD_RC=0 SOURCE_PINS_UNCHANGED=True RECEIPT=/mnt/ForgeRealm/wt/pt-tf32/artifacts/pt_tf32_3/build_01/receipt.json
35 passed in 1.15s
2 failed, 111 passed in 2.22s
E           ValueError: fork binary drift
E       ValueError: fork engine drift
```

The last two failures are historical PT-TF32-2 and PT-TF32-1 engine pins,
respectively. They are retained RED, not repinned or skipped. The new order's
exact-state registration passes. After removing diagnostic hot-path copies,
build 02 and `GRAPA_REGISTRATION_002.json` are separate new receipts; build 01
and its first GRAPA registration remain immutable evidence. No tolerances changed.

## Done

Final author receipts, verbatim:

```text
BUILD_RC=0 SOURCE_PINS_UNCHANGED=True RECEIPT=/mnt/ForgeRealm/wt/pt-tf32/artifacts/pt_tf32_3/build_02/receipt.json
GRAPA_REGISTERED 8f88e759a8f8714bac16a4edb71dbc7a686bd912230338ec2aaa9115aeee386b
36 passed in 1.19s
60 tests collected in 0.13s
SEALED 50040ecba989f53400896b9d72b59ab9be87302a2fa30232e0798ccb24a1734f
MANIFEST_VERIFIED 50040ecba989f53400896b9d72b59ab9be87302a2fa30232e0798ccb24a1734f
PRESERVED 656 OF 668 UNEXPECTED_CHANGES []
BF16_FP16_GEMM_BRANCH_BYTE_IDENTICAL True
KERNELS_CU_AND_OBJECT_BYTE_IDENTICAL True
```

Evidence files are under `artifacts/pt_tf32_3/`:
`build_02/receipt.json`, `build_console_02.log`, `cpu_tests_02.log`,
`gpu_collection.log`, `SOURCE_MANIFEST.json`, `PRESERVATION_RECEIPT.json`,
`host_dispatch_02.log`, `BLOCKED_REPORT.json`, `LEAD_SEQUENCE_FINAL.json`.
The 36 new CPU tests include an actual 1.2 MB cross-process pipe, exact array
hashes, low-space non-launch, own-child timeout, ENOSPC classification, all
three real zero-error slot-10 sanitizer logs with rc=1, output interception,
frozen scalar calibration tamper detection, native host descriptor readback,
fork registration, GPU-hidden entrypoint rejection and midpoint witnesses.
These are author tests, not blind verification.

Final fork binary SHA256:
`a1bf6ce6b5212b83f4652547fd87e3508ae824f9ab461850680c2081320f8fbe`.
All historical artifact JSON/log/registration files included in the 668-file
baseline are unchanged. The 12 expected changes are five C++/CUDA/header source
files, the test fixture's explicit generation selector, five compiled objects
and the rebuilt extension. `kernels.cu` and its compiled object are identical;
the FP16/BF16 GEMM branch text is identical. Runtime BF16 parity remains a GPU
gate. The historical suite's two binary-pin failures above remain RED.

### Lead's single slot, at most 30 minutes

The lead supplies an exclusive GPU slot externally. These commands are
provided only; this seat did not run them. The runner does not touch the GPU
lock, start parallel lanes, or kill any pre-existing process. Use a fresh output
directory if `lead_slot_01` already exists.

```bash
cd /mnt/ForgeRealm/wt/pt-tf32
CUDA_VISIBLE_DEVICES='' python3 -B scripts/pt_tf32_3_slot.py \
  --print-sequence --out artifacts/pt_tf32_3/lead_slot_01
CUDA_VISIBLE_DEVICES=0 python3 -B scripts/pt_tf32_3_slot.py \
  --lead-gpu --out artifacts/pt_tf32_3/lead_slot_01
```

| Lane | Budget (seconds) |
|---|---:|
| Host API device dispatch self-check (before timing) | 30 |
| All 60 GPU units, including unchanged legacy edge assertions | 75 |
| 27 GEMM cells in each of two allocation policies (54 total) | 100 |
| Attention 0 / 1 / 2 / 3 | 75 / 75 / 140 / 140 |
| memcheck / racecheck / synccheck | 75 / 100 / 75 |
| New four-arm healthy noise floor | 220 |
| Onset / healthy / control, fresh holdouts | 100 / 100 / 100 |
| Both 20-step timing arms | 300 |
| Total lane budgets / global deadline | 1705 / 1800 |

Default full-gradient/checkpoint dump total: **0 bytes**. Small token-batch,
JSON/text and CUDA-cache files are excluded and identified by print-sequence.
`--keep-grads` explicitly requests retained NPZ/checkpoint files; print-sequence
then reports the conservative total estimate from pinned reference sizes.
The complete command array is `LEAD_SEQUENCE_FINAL.json`. All unrun gates at
the deadline get BLOCKED receipts. Healthy/control require this new binary's
new frozen noise floor; the old 0.99843 bar is not silently reused.

`BLOCKED_REPORT.json` is the certification result for this seat. **Not claimed
fixed:** downstream dK, legacy edge passes, >=5x GEMM, attention speed,
healthy/control/step-time gates, runtime BF16 parity and long-term stability.
No GPU ran. The edge rounding explanation does not change the unit lane's
verdict; the lead must review any repeated RED in its original context.
