# PT-TF32-2 implementation ledger

**DELIVERED — fork rebuilt; 112 CPU checks pass, one historical binary-pin
check remains RED. All new GPU certification remains BLOCKED.**
Immutable plan: `orders/PT_TF32_2_CERTIFY.md`. House rules read before work.
No git, subagents, GPU operations, live-engine imports, or lock operations.
All generated files, caches and compiler temporaries are confined to this fork.

## Registration and inherited evidence

New registration, before implementation/gates:
`artifacts/pt_tf32_2/REGISTRATION.json`, SHA256
`0d0c02b1c5c5f488b9598ddd69aebe1670bc8e499cca8381feb48ed378763cbd`.
`BASELINE_SHA256.json` pins 432 historical files and both immutable orders.
Original attention/matmul/GPU test source is copied into `baseline/`.
Old registrations and RED results are not revised or reclassified.

Read slot-9 `lead_gemm/case_00.json`, all attention/state summary receipts,
and source. Load-bearing verbatim examples from `lead_attention_0/case_00.json`:
`"relative_L2": 0.016164940440447136` (isolated dK),
`"relative_L2": 0.016001361895631706` (downstream dK),
`"relative_L2": 0.0013338882683235221` (old dK bound),
`"flips": 1863`, `"rate": 2.7760863304138184e-05`.
GEMM: `"speedup": 1.3304865517649636`, medians SGEMM
`0.7826879918575287` ms / TF32 `0.5882720053195953` ms.
The lead's order reports seven edge failures and memcheck zero errors; those
transcripts were not found as standalone files in the supplied `lead_*` dirs.
This distinction is retained rather than inventing missing native evidence.

## Diagnosis before edits

Evidence class: source reasoning, pending CPU ablation/native confirmation.
Both backward owners implement the correct selected-only dK formula and
bottom-right causal/grouped ownership. However, `scores(qs,kqs)` rounds inputs
before the hard predicate, whereas the FP64 oracle does not. dK changes by a
whole contribution when a predicate flips; dQ changes between K and KQ and is
less sensitive. A small global mask flip rate is not a gradient error bound.
The selected dS/Q WMMA remains a secondary rounding site, not yet isolated as
the root cause. No native defect is claimed fixed without the lead gate.

The `E[x*x]-E[x]^2` threshold variance and frozen threshold comparisons are
ill-conditioned at singleton rows. The old FP64 reference itself can disagree
with `a` there (receipt LSE max_abs `0.050635944674309666`). The new independent
oracle uses centered variance and the singleton identity; the isolated lane
uses independent FP64 state cast to the actual FP32 API. It does not reuse a
numerically inconsistent `a` state. Downstream compares full independent FP64
forward/VJP and separately reports VJP on native saved state.

## Prior art

- NVIDIA Ampere TF32 (2020), CUDA/cuBLASLt 12.6 (2024): WMMA, explicit TF32
  conversions, FAST_TF32, heuristic search/workspace and HMMA capability flags.
  Taken APIs; ours is fork-local dispatch and capability filtering/caching.
  [NVIDIA TF32](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/),
  [cuBLAS](https://docs.nvidia.com/cuda/archive/12.6.2/cublas/index.html).
- Ootomo and Yokota (2022), [arXiv:2203.03341](https://arxiv.org/abs/2203.03341):
  taken input residual decomposition for tensor-core multiplication. Ours is
  a local three-product adaptation at APA selection/cancellation sites. This
  is not their full compensated accumulator scheme or an FP32 equivalence claim.
- Dao et al. (2022), Dao (2023), FlashAttention/FlashAttention-2, and local
  BP-KERNEL-2/3/4 (2026): taken online softmax, output-dot VJP and owner tiles.
- Higham (2002), [*Accuracy and Stability of Numerical Algorithms*](https://nhigham.com/accuracy-and-stability-of-numerical-algorithms/): standard
  unit-roundoff model; author's book page checked during this seat.
  The independent RMS propagation and event counts are explicitly engineering
  assumptions for these fixtures, not a worst-case theorem. The bound is
  `3*(sqrt(events)*2^-11/sqrt(3)+sqrt(max_dimension)*2^-24)`, registered before
  implementation measurements. No allowance for hard-predicate errors.
- Centered population variance and singleton softmax identities are standard
  statistics/calculus, no new algorithm claimed. The old test's tolerance is
  unchanged. NumPy (Harris et al., 2020), pytest (Krekel et al., 2004), SHA256
  (NIST, 2001) and CUDA event timing are taken testing/receipt methods.
- Healthy-state cosine spread and `1-3*spread`: user order (2026); no prior
  art known to me for this exact rule. Spread includes deterministic
  across-policy bias and within-arm repeats, then is frozen before holdouts.

## Execution receipts

Entries below are appended as implementation and checks complete.

### 1. dK diagnosis and treatment

`CPU_DIAGNOSIS.json` reproduces the error class in a B=1/H=4/KVH=2,
L=S=512/D=96/VD=64 fixture with the registered seed. This first ablation
also changes singleton handling between arms; it is **not** a clean isolation
of just the predicate. `CPU_DIAGNOSIS_002.json` controls that confound by
holding singleton handling OFF for every arm:

| CPU arm | dK relative-L2 | Selection flips |
|---|---:|---:|
| Original TF32 product sites | 0.010012751525230686 | 29 |
| Only predicate uses full precision | 0.00046814694494103505 | 0 |
| Only dO·V uses full precision | 0.010010259569968226 | 29 |
| Only outer accumulation operands use full precision | 0.010006225910986988 | 29 |
| Corrected bulk/dO·V, singleton treatment OFF | 0.00029896899716708473 | 2 |

This CPU treatment effect isolates the predicate as the dominant dK error
source for this fixture and refutes outer-product rounding as its main cause.
The final row still has dV `0.004786876328431096` from two singleton selection
disagreements. With the explicit singleton identity ON, the first diagnosis
records dK `0.00029896942600306627`, dV `0.00027169498529855174`, and zero flips.
These are **FP64-accumulator CPU models**, not native measurements, and not
proof that all seven native regressions are fixed.

`attention_tf32.cu` now uses three TF32 products for bulk selection scores,
dO·V (before subtraction of D_i), and forward P·V (whose output forms D_i).
High and residual cross-products have separate FP32 accumulators. Final
gradient outer products retain ordinary TF32 inputs/FP32 accumulation. Padding,
grouped-head ownership, selected full-operand Q·K and detached KQ are retained.
The omitted residual×residual product is O(u²); this is not an assertion of
exact FP32 multiplication. Prior art: Ootomo/Yokota (2022), as cited above.

### 2. Edge cases

One-visible-key rows now select their only pair, store V exactly, use p=1,
and produce exactly zero dQ/dK. This removes threshold equality and softmax
subtraction cancellation at that mathematically degenerate row. The new
CPU oracle uses centered variance; the native threshold uses an explicit
zero-variance singleton branch and retains the existing reduction elsewhere.
Prior art: standard population variance and singleton softmax derivative.

The original GPU tests keep `rtol=3e-3`, `atol=2e-5`, and forward rel-L2
`3e-3`. Only the loader/source seal and CPU product-placement model change;
there is no tolerance increase. Independent complete FP64 gates and additional
singleton/grouped/selection regressions prevent the updated model from being
the sole oracle. The lead must run all 56 cases; none ran here.
`AMENDMENT_001.json` corrects a registration counting typo: 40 parameter cases
(2×5×4), nine original integration cases, seven new cases. Thresholds and
shapes are unchanged; the original JSON is untouched.

### 3. Tensor-core GEMM dispatch

`matmul.cu`'s explicit FP32 TF32 branch calls `tf32_gemm.cu`. Model-aligned
shapes use cuBLASLt FAST_TF32 with a 32 MiB per-thread/device workspace, a
bounded 128-plan cache, actual pointer/batch-stride alignment, and a heuristic
filter requiring HMMA + TF32 input + FP32 accumulator capability bits.
The diagnostic getter returns algorithm ID, numerical flags and workspace
bytes. A missing eligible model algorithm throws; it cannot quietly count an
SGEMM fallback as a speed result. No GPU was available to inspect the old
cuBLAS algorithm, so **old tensor-core non-engagement is a lead hypothesis,
not established by the 1.33× timing alone**.

`AMENDMENT_002.json` registers a tail-safe WMMA route for ragged M/N/K before
GPU execution. It avoids assuming Lt supports every odd alignment. ID -2
identifies this route; registered model performance results require ID ≥0.
The fallback pads only shared 16×8 / 8×16 operand tiles, keeps FP32 storage,
supports transposed B and zero-stride broadcast, and uses TF32 WMMA throughout.
No FMA/SIMT precision fallback was added. Prior art: NVIDIA WMMA/cuBLASLt,
with local bounded-cache/containment integration as described above.

FP16/BF16 source branches and `kernels.cu` remain unchanged (existing CPU
source checks pass). The default GEMM mode remains off, and backward mode
capture remains intact. New native behavior still requires GPU validation.

### 4. Precision registration and independent oracles

With 10 fraction bits, round-to-nearest has u=2^-11 and a uniform-error RMS
model σ=u/√3. Two rounded operands give approximately √2 σ per dot product.
The registration budgets two events for LSE, four for output/dV, six for
dQ/dK, plus √max(L,S,D,VD)·2^-24 for FP32 arithmetic; the acceptance bound is
three times that expected error. At L=2048 this is about 0.001204 (LSE),
0.001700 (out/dV), 0.002080 (dQ/dK); L=4096 adds only the registered FP32 term.
Both relative-L2 and max-absolute normalized by reference peak must pass.
Zero-reference arrays require exact zero; nonfinite or empty data cannot pass.
These are statistical engineering assumptions for the normal fixtures;
they do not bound adversarial cancellation or hard-mask discontinuities.
No term was estimated from the candidate's errors or adjusted after a gate.

The isolated backward gate freezes independent FP64 state cast to the actual
FP32 API and uses the FP64 output-dot VJP on that same state. The full
downstream gate compares native forward/backward to complete independent FP64
forward/VJP. A third same-native-state VJP diagnostic must also pass. This
corrects both the unlike-precision 2× rule and the singleton oracle issue
without rewriting or declaring GREEN any old receipt.

Noise calibration runs four fresh healthy-state child processes in order
none/policy/policy/none. Spread is the largest cosine distance among all six
pairs, including deterministic policy bias, not merely repeat nondeterminism.
The create-only `NOISE_FLOOR.json` pins every input and freezes bar=1−3·spread
before separate healthy/control holdouts. Zero spread yields bar=1; missing,
zero-norm, nonfinite, shape/key-mismatched or vacuous spread fails closed.
Onset cosine ≥0.99, onset rel-L2 ≤0.1, time ≤1.3× bf16 and ≤6.5 s remain.

### 5. Build, CPU checks and static evidence

Commands, all with `CUDA_VISIBLE_DEVICES=''`:

```text
python3 -B scripts/pt_tf32_2_build.py
python3 -B scripts/pt_tf32_2_grapa.py prepare
TC_TF32_GEMM=0 python3 -B -m pytest -q tests/test_pt_tf32_2_cpu.py tests/test_pt_tf32_cpu.py tests/test_pt_tf32_grapa.py -p no:cacheprovider --basetemp=artifacts/pt_tf32_2/cpu_tmp_02
python3 -B -m pytest --collect-only -q tests/test_pt_tf32_gpu.py tests/test_pt_tf32_2_gpu.py -p no:cacheprovider --basetemp=artifacts/pt_tf32_2/collect_tmp
python3 -B scripts/pt_tf32_2.py blocked
python3 -B scripts/pt_tf32_2.py seal
```

Builds 01 and 02 succeeded; build 02 adds ragged WMMA. Offline CMake uses
the existing in-fork pybind11 source, CUDA 12.6 and SM89; it performs no git
operation or network fetch. Compiler caches/temporaries stay under this fork.
The first new GRAPA registration pins build 01 and is retained unchanged;
`GRAPA_REGISTRATION_002.json` is the new create-only build-02 experiment pin.
No thresholds changed between them.

Tests cover residual arithmetic, controlled dK ablation, singleton identities,
finite differences with a frozen APA mask, detached KQ and grouped gradients,
independent tiled ownership, precision bounds, cosine calibration/tampering,
deadline termination of an own CPU child, native CPU host rejection guards,
fork isolation and CUDA-hidden GPU-command rejection. The combined final run
has **112 passed, one failed**. The failure is retained verbatim below: the
historical PT-TF32-1 GRAPA check correctly rejects a changed binary. No skip,
suppression, old pin update or assertion change conceals it. The new
PT-TF32-2 registration test passes. These are author tests, not blind review.

`SASS_RECEIPT_002.json` finds HMMA.*.F32.TF32 in both backward owners (56
static instructions each), both forward variants (36 each), and ragged GEMM
(4). Resources show zero local memory/stack for these kernels; backward shared
memory is 46,144 bytes, forward 43,392 bytes, ragged GEMM 2,048 bytes.
This is compile evidence, **not proof of Lt runtime dispatch or speed**.
The first SASS counter incorrectly searched for HMMA.1688 while SM89 emits
HMMA.1684; the first receipt is retained and the second corrects parsing of
the same dump. No GPU rerun occurred.

### 6. Exact lead GPU sequence — one slot, at most 30 minutes

The lead must supply its already-authorized exclusive slot. This command does
not acquire/read/write the GPU lock. Use a fresh output directory; existing
receipts deliberately cause refusal. Do not invoke it from this CPU-only seat.

```bash
cd /mnt/ForgeRealm/wt/pt-tf32
CUDA_VISIBLE_DEVICES=0 TC_TF32_GEMM=0 PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/pt_tf32_2_slot.py --lead-gpu --out artifacts/pt_tf32_2/lead_slot_01
```

`artifacts/pt_tf32_2/LEAD_SEQUENCE.json` contains the exact child argument
arrays, in this order. Times are per-lane ceilings; a single global deadline
also applies and leaves time to terminate only process groups the runner made.

| Lane | Maximum seconds | Required evidence |
|---|---:|---|
| GPU units | 75 | All 56 cases pass, no tolerance changes |
| GEMM | 100 | All 27 shape/direction cells ≤1e-3, ≥5×, Lt HMMA flags |
| Attention 0, 1 | 75 each | D96/D128 at L2048: all numerical/mask/timing gates |
| Attention 2, 3 | 140 each | D96/D128 at L4096: all numerical/mask/timing gates |
| memcheck | 75 | All 56 cases; explicit zero-error summary |
| racecheck | 100 | All 56 cases; explicit zero-error/hazard summary |
| synccheck | 75 | All 56 cases; explicit zero-error summary |
| Healthy noise floor | 220 | Four independent processes, six distances, frozen bar |
| Onset | 100 | Same exact registered state and FP64 target |
| Healthy holdout | 100 | Fresh pair against previously frozen bar |
| Control holdout | 100 | Fresh pair against same frozen bar |
| Step time | 300 | Both registered 20-step runs, cursor/loss checks and timing |

The per-lane ceilings sum to 1,675 seconds; total deadline is 1,800 seconds.
Finite RED results are recorded while independent lanes continue within the
deadline. Timeouts/unrun lanes are BLOCKED, and unavailable calibration blocks
both dependent holdouts. No timeout, missing sanitizer summary or empty run
can count GREEN. Each lane writes a receipt and the runner writes
`SLOT_SUMMARY.json`; no background waits, detached training or automatic retry.

### 7. Blocked-report and residuals

`artifacts/pt_tf32_2/BLOCKED_REPORT.json` marks all 18 certification items
BLOCKED/unmeasured: GEMM accuracy, tensor-core dispatch, GEMM speed; attention
forward, isolated backward, downstream backward, selection and speed; GPU
units; memcheck, racecheck, synccheck; model noise floor, onset, healthy,
control and step time; blind lead verification.

**Not claimed fixed:** native dK, all seven native edge failures, ≥5× GEMM,
the new healthy/control gates or training stability. Extra score/P·V products
can change attention time; only the lead's unchanged timing gates decide.
The original slot-9 onset/timing successes describe the original binary;
they are not transferred to this build. This CPU/build handoff is complete;
GPU certification and blind verification belong to the lead.

## Done

Verbatim final command receipts:

```text
BUILD_RC=0 SOURCE_PINS_UNCHANGED=True RECEIPT=/mnt/ForgeRealm/wt/pt-tf32/artifacts/pt_tf32_2/build_02/receipt.json
1 failed, 112 passed in 2.34s
FAILED tests/test_pt_tf32_grapa.py::test_grapa_registered_geometry_and_engine_are_fork
E       ValueError: fork engine drift
56 tests collected in 0.12s
PROTECTED_FILES 432 CHANGED []
SEALED 4e00f4dcb36bf249662b8e57fbe1892c7382e021f564bcd798c2803d4a2929fa
MANIFEST_VERIFIED 55 SOURCE_FILES a90ed65584e0ec8db30d986c14923c7180954669a7ef29fd943d97b83c907302
GRAPA_VERIFIED GRAPA_REGISTRATION_002.json
```

Binary: `tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`.
Build source pins and binary hash are in `build_02/receipt.json`; source seal
is `SOURCE_MANIFEST.json`. Final historical-file audit and deliverable checksums
are in `FINAL_AUDIT.json` and `SHA256SUMS` under `artifacts/pt_tf32_2/`.
