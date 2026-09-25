# PT-TF32-1 implementation ledger

Status: **BUILT, author CPU checks PASS; GPU correctness, speed and model gates
BLOCKED / UNPROVEN. Existing archival suite remains RED.** Not claimed fixed:
the onset precision event or the 6.5 s step target. Order:
`orders/PT_TF32_1_FAST_PRECISE_PATH.md`, immutable.

## Registration and boundaries

Read HOUSE_RULES.md, AGENTS.md, the order, BP-KERNEL-2/3/4 source and ledgers,
CC39-B / CC41 ledgers and the board through 2026-09-25 13:45. Registration is
`artifacts/pt_tf32_1/REGISTRATION.json` with a separate SHA256. Amendments, if
needed, get separate files. Baselines precede edits. No git or subagents.
All execution uses `CUDA_VISIBLE_DEVICES=''`; no GPU lock, live-engine import,
live build, run directory writes or process signals. Local pybind11 dependency
source was copied read-only into this fork; CMake will use the fork copy offline.

## Design and source findings (reasoning, not GPU evidence)

- `src/matmul.cu` is the only fp32 cuBLAS input GEMM site. `nn.Linear` calls
  `ops::matmul`; its dInput/dWeight call the same NDArray matmul. The two
  `gemm_apa.cu` GEMM calls have BF16 operands with FP32 output; its cuBLASLt
  call has INT8 inputs and INT32 accumulation. Those are outside the fp32 switch.
- Add a default-off, thread-local `set_tf32_gemm(bool)` / getter, initialized
  from exact `TC_TF32_GEMM=1`. FP32 uses FAST_TF32 with FP32 output/accumulation
  when enabled. No handle math-mode mutation can leak into BF16 calls.
- Separate h_tf32 / g1_tf32 code leaves the BF16 kernel bodies untouched.
  FP32 shared Q/K preserve selected scalar exact-score computation. WMMA
  m16n16k8 rounds **fragment operands** to TF32; p/dS storage stays FP32.
  This is not full-mantissa FP32 multiplication. A 2x spread relative to fp32
  a can fail even if relative error is small. The registered gate stays strict.
- Preserve visible-key population-variance z threshold, absolute >= selection,
  bottom-right causality, saved lse/thr/O, detached KQ, grouped-head owner sums,
  D/VD <=128, and zero padding. Reuse dead shared buffers to fit under 48 KiB.
- Read-only model source says d_ff=1792 and qk_nope=64; rope=64 means total
  D=128. Test requested FFN=4096 and D=96 too; do not confuse VD=64 with D.
- Initial checkpoint forward uses no-grad inference APA. Training-forward
  routing alone does not accelerate it; integration must cover this explicitly.

## Prior art

- NVIDIA Ampere TF32 (2020), CUDA 11 WMMA and cuBLAS: taken tensor-core format,
  conversion intrinsics, fragment API and FAST_TF32 compute mode. Verified in
  [NVIDIA's TF32 article](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/),
  [CUDA guide](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-c-programming-guide/index.html),
  [cuBLAS documentation](https://docs.nvidia.com/cuda/archive/13.0.2/cublas/index.html).
- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) and
  [FlashAttention-2 (Dao, 2023)](https://arxiv.org/abs/2307.08691): taken tiled
  ownership, recomputation, online softmax and VJP/output-dot structure through
  the Project-Tensor BP-KERNEL-2/3/4 (2026) implementations. Their APA selection
  rule is preserved, not invented here.
- Ours: opt-in FP32 integration, TF32 fragment adaptation, buffer-lifetime
  reuse, strict guards, and this order's gate/bench plumbing. No novelty claim
  for TF32, tiling, GEMM routing or selective attention.
- NumPy (Harris et al., 2020), pytest (Krekel et al., 2004), CUDA events
  (NVIDIA, 2007+), SHA-256 (NIST, 2001): taken testing, timing and receipt
  techniques. Dates unverified — lead to check those names. The exact numeric
  thresholds are the order's experimental rules; no prior art known to me for
  those particular acceptance thresholds.

## Execution receipts

Append build, CPU checks, source hashes, blocked gates and lead commands below.

Build attempt 01: RED, `BUILD_RC=2`. Verbatim compiler error:
`At end of source: error: expected a "}"` at the opening `namespace tc`.
The new translation unit omitted its final namespace brace. Fixed that brace;
receipt and failed build log are retained in `artifacts/pt_tf32_1/build_01/`.
This was a mechanical compile error, not a numerical-gate result.

Build attempt 02: RED, `BUILD_RC=2`. Verbatim:
`error: identifier "__float_to_tf32" is undefined`.
CUDA 12.6's `crt/mma.h:96` places it in `nvcuda::wmma`; qualified the name.
That header explicitly implements `cvt.rna.tf32.f32` (nearest, ties away).
Receipt retained in `artifacts/pt_tf32_1/build_02/`. No gate was run or changed.

Build attempt 03: PASS (CPU compile/link only), `[100%] Built target _tensor_cuda`.
CPU suite attempt 01: `1 failed, 63 passed in 1.36s`. Verbatim failure:
`RuntimeError: CUDA error at from_host: no CUDA-capable device is detected`.
The existing `NDArray::from_host` unconditionally calls `cuda_check_last` even
for a CPU tensor. GPU visibility was empty. The native guard test now constructs
CPU buffers directly in a tiny C++ executable, avoiding that unrelated transfer
API. The same rejection assertions remain; the engine's default core is unchanged.

The first standalone host-test link failed because the pybind11 module hides
internal symbols: `undefined reference to tc::apa_selective_fwd_tf32(...)`.
The successful executable links the exact CMake object set instead; the recipe
is now in `scripts/pt_tf32_build.py`. Receipt:
`HOST_CONTRACT: 6 expected rejections; no CUDA API called`.
Boundary slip: that first direct `c++` command used the compiler's default
temporary directory (`/tmp/ccTYI9OE.o` in its error), outside the order's fork-only
writes. Subsequent compiler invocations set TMPDIR inside the fork. No live or
GRAPA files were written; the transient compiler file was compiler-managed.

Author CPU suite attempt 02: `64 passed in 1.33s`. Existing BP suites were run
unchanged with `--campaign-receipts`, including archival checks. First run:
`9 failed, 119 passed, 1 skipped in 1.78s`. Seven available historical fixture
files were then copied read-only from the canonical artifact tree into this
fork; hashes are in `legacy_fixture_copies.json`. No live engine import/build.

## Final implementation

`tensor_cuda/src/attention_tf32.cu` contains separate h_tf32 and g1_tf32
implementations. `kernels.cu` is byte-identical to the initial fork, including
all a/f/g1/g2/h kernels. The FP16/BF16 branches in `matmul.cu` are byte-identical
to the saved baseline. Setters remain default a/a and TF32 GEMM remains off
unless selected by the API or exact environment value `TC_TF32_GEMM=1`.

The FP32 GEMM arm uses `cublasGemmStridedBatchedEx` with CUDA_R_32F input/output
and `CUBLAS_COMPUTE_32F_FAST_TF32`. It handles the same trans_b, alpha, batches
and broadcast strides as SGEMM. FP32 GEMM backward captures its forward mode
and restores the caller's mode with an RAII guard. API overrides the environment
initialization for the current thread; changing the environment after that
thread's first access does not reset its mode. No cuBLAS handle mode is changed.

Attention uses m16n16k8 WMMA fragments, explicitly converts fragment elements
with `wmma::__float_to_tf32`, and retains original FP32 Q/K for selected scalar
scores. p/dS buffers, saved O/lse/thr, output gradients and accumulators are
FP32. Gradient ownership, selected-only dK, KQ stop-gradient and grouped heads
are inherited from g1. No fallback changes the requested variant. Bad rank,
geometry, dtype, device or scalar inputs fail; TF32 attention requires SM80+.
The delivered binary is specifically built for SM89; compilation is not a
claim of execution on other architectures.

Static binary inspection (`resources.log`, `tf32_sass.log`, `sass_receipt.json`):

| Compiled function | Registers/thread | Shared bytes | Local bytes | TF32 HMMA instructions in code |
|---|---:|---:|---:|---:|
| h_tf32 ordinary forward | 94 | 43,392 | 0 | 20 |
| h_tf32 diagnostic forward | 128 | 43,392 | 0 | 20 |
| g1_tf32 query owner | 110 | 46,144 | 0 | 56 |
| g1_tf32 key owner | 80 | 46,144 | 0 | 56 |

These are static resource/instruction counts, not executed instruction counts,
occupancy measurements or timing evidence. The original build warning about
unused `lane` and the nvlink static-library warnings are retained in build logs.

## GRAPA integration

Do not select h_tf32/g1_tf32 globally for a mixed-dtype model: they intentionally
reject BF16. In `grapa/fwd_precision.py`, an explicit TF32 policy needs:

```python
# Taken: CC41 scoped precision policy (2026); NVIDIA TF32 (2020).
# Ours: mappings for the fork's explicit FP32 variants.
FP32_BWD_FALLBACK = {'g1': 'g1_tf32', 'g2': 'g1_tf32'}
FP32_FWD_FALLBACK = {'h': 'h_tf32'}
# Inside fp32_apa_variants(tc), save get_tf32_gemm(), set_tf32_gemm(True),
# and restore it in finally, alongside the existing variant restoration.
```

The context must wrap both the initial block call and checkpoint replay, as
CC41 already does. Keep g1/h outside that context. The fork's explicitly
selected h_tf32 also covers `apa_selective_attention`, so no-grad initial
forwards and replay use the same new forward algorithm. Matmul backward
captures the GEMM setting; APA backward already captures its variant.
Simply enabling GEMMs in the initial forward and restoring them before replay
would be insufficient. The existing CC41 policy records continue to report
the selected fallback mappings.

Delivered `scripts/pt_tf32_grapa.py` supplies this policy as a process-local
adapter, without editing GRAPA. It imports the fork package by absolute path,
pins the binary and GRAPA Python sources, and invokes the existing CC39 child
body against a **new** registration, `GRAPA_REGISTRATION.json`. The historical
CC41 registration/engine identity is retained as provenance; it is never
rewritten or claimed equivalent to the new engine. State runs reproduce the
registered checkpoint and token hashes. Healthy/control comparisons use a
fresh no-policy arm in the same invocation. Timing loads the registered
checkpoint independently for each 20-step arm, checks all cursors and the
no-policy losses, and uses CC41's logged median seconds/step. Log times have
the inherited two-decimal resolution. The timing arms are sequential rather
than interleaved training trajectories.

The adapter does not open the GPU lock or stop a live run. The lead must
schedule it inside an authorized idle slot. All new logs/checkpoints/gradients
go under the specified fork output directory. A state invocation has two
foreground children capped at 600 seconds each; timing has the same caps.
The adapter itself and every model/GPU path remain GPU-unexecuted here.

## Registered lead gates and expected numbers

The following are **acceptance targets**, not measurements or assurances.
Registration `REGISTRATION.json` is immutable; `AMENDMENT_001.json` records
fixture/layout and no-grad-routing details without changing any threshold.
`SOURCE_MANIFEST.json` pins the final source and binary before GPU gates.

| Gate | Required result | Seat status |
|---|---|---|
| GEMM, all 9 shapes × forward/dInput/dWeight | TF32 vs full CPU FP64 relative L2 <= 0.001; SGEMM/TF32 median >= 5 | BLOCKED |
| Attention forward O and lse | Each max-abs and relative-L2 <= 2 × fresh fp32 a distance to FP64; zero means zero | BLOCKED |
| Attention backward dQ/dK/dV | Same strict 2× rule, on both a saved state and downstream h_tf32 state | BLOCKED |
| Selection | Absolute >= z-score rule; <= 0.005 flip fraction over all pairs vs a; visible-pair rate also reported | BLOCKED |
| Attention speed | h_tf32/h(BF16) <= 2 and g1_tf32/g1(BF16) <= 2; a/f fp32 also measured | BLOCKED |
| Native unit suite, existing CUDA tests | All assertions pass | BLOCKED |
| memcheck, racecheck, synccheck | Zero reported errors | BLOCKED |
| Onset, blocks 0–10 | Cosine >= 0.99 and relative L2 <= 0.1 vs registered FP64 gradient | BLOCKED |
| Healthy/control, blocks 0–10 | Cosine >= 0.999 vs same-invocation no-policy arm | BLOCKED |
| 20-step throughput | Median <= 6.5 s AND <= 1.3 × fresh BF16 median | BLOCKED |
| Blind lead verification | Lead-dispatched verification; author tests do not substitute | BLOCKED |

GEMM widths include 1024, requested FFN 4096, actual FFN 1792, MLA 256/768,
kv_a 320 and vocab 8192, at M=4096. Attention has B=1/H=KVH=16, L=S=2048/4096,
VD=64, refine=0.15, and both D=96 (64+32) and D=128 (64+64). Smaller grouped-head,
tail-width, causal/noncausal, transpose/broadcast and invalid-input tests are
in the unit suites. Inputs are synthetic normal FP32 data; the KQ quarter-step
prefix plus exact RoPE suffix is a fixture, not the model quantizer. The
exact-state GRAPA lane is therefore a separate mandatory check.

FP64 forward and frozen-state backward references and the exact gate functions
are imported from the unchanged BP-KERNEL-2/4 scripts. Backward references
preserve saved lse/thr and do not renormalize probabilities. Timings use three
warmups and ten samples with alternating arm order in one session. On numerical
RED, timings may be retained as diagnostics, but `timing_counts=false` and the
verdict cannot be GREEN. Exceptions/timeouts are not passes.

## Exact commands for the lead

Run from this fork. These GPU commands were **not executed by the seat**.
They require an idle slot arranged by the lead; the commands never acquire or
touch `/tmp/forge-gpu.lock`. Keep `NVIDIA_TF32_OVERRIDE` unset (or nonzero);
the GEMM/model harness rejects an override of zero.

```bash
cd /mnt/ForgeRealm/wt/pt-tf32
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/mnt/ForgeRealm/wt/pt-tf32/tensor_cuda
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2

# CPU verification of delivered source/binary and model-source pins.
CUDA_VISIBLE_DEVICES='' python3 -B -c "import sys; sys.path.insert(0,'scripts'); import pt_tf32_1 as t; import pt_tf32_grapa as g; t.verify_manifest(); g.registered(); print('PINS_OK')"

# Author CPU suite; use a fresh basetemp name for each invocation.
CUDA_VISIBLE_DEVICES='' TC_TF32_GEMM=0 python3 -B -m pytest -q tests/test_pt_tf32_cpu.py tests/test_pt_tf32_grapa.py -p no:cacheprovider --basetemp=artifacts/pt_tf32_1/lead_cpu_tmp

# GPU units and the relevant shipped selective-attention regression suites.
CUDA_VISIBLE_DEVICES=0 TC_TF32_GEMM=0 PT_TF32_LEAD_GPU=1 timeout --kill-after=15s 1200s python3 -B -m pytest -q tests/test_pt_tf32_gpu.py tensor_cuda/tests/test_apa_value_dim.py tensor_cuda/tests/test_apa_selective.py tensor_cuda/tests/test_apa_phase6.py -p no:cacheprovider --basetemp=artifacts/pt_tf32_1/lead_gpu_tmp

# Full FP64 GEMM references plus interleaved SGEMM/TF32 timings, 27 cells.
CUDA_VISIBLE_DEVICES=0 timeout --kill-after=15s 1200s python3 -B scripts/pt_tf32_1.py gemm --lead-gpu --out artifacts/pt_tf32_1/lead_gemm

# One attention geometry per bounded invocation: 2048/D96, 2048/D128,
# 4096/D96, 4096/D128. Each includes FP64 gates and kernel benchmarks.
CUDA_VISIBLE_DEVICES=0 timeout --kill-after=15s 1200s python3 -B scripts/pt_tf32_1.py attention --case 0 --lead-gpu --out artifacts/pt_tf32_1/lead_attention_0
CUDA_VISIBLE_DEVICES=0 timeout --kill-after=15s 1200s python3 -B scripts/pt_tf32_1.py attention --case 1 --lead-gpu --out artifacts/pt_tf32_1/lead_attention_1
CUDA_VISIBLE_DEVICES=0 timeout --kill-after=15s 1200s python3 -B scripts/pt_tf32_1.py attention --case 2 --lead-gpu --out artifacts/pt_tf32_1/lead_attention_2
CUDA_VISIBLE_DEVICES=0 timeout --kill-after=15s 1200s python3 -B scripts/pt_tf32_1.py attention --case 3 --lead-gpu --out artifacts/pt_tf32_1/lead_attention_3

# Run each sanitizer separately, within its own authorized slot.
CUDA_VISIBLE_DEVICES=0 PT_TF32_LEAD_GPU=1 timeout --kill-after=15s 1200s /usr/local/cuda-12.6/bin/compute-sanitizer --tool memcheck --error-exitcode 86 python3 -B -m pytest -q tests/test_pt_tf32_gpu.py -p no:cacheprovider --basetemp=artifacts/pt_tf32_1/memcheck_tmp
CUDA_VISIBLE_DEVICES=0 PT_TF32_LEAD_GPU=1 timeout --kill-after=15s 1200s /usr/local/cuda-12.6/bin/compute-sanitizer --tool racecheck --error-exitcode 86 python3 -B -m pytest -q tests/test_pt_tf32_gpu.py -p no:cacheprovider --basetemp=artifacts/pt_tf32_1/racecheck_tmp
CUDA_VISIBLE_DEVICES=0 PT_TF32_LEAD_GPU=1 timeout --kill-after=15s 1200s /usr/local/cuda-12.6/bin/compute-sanitizer --tool synccheck --error-exitcode 86 python3 -B -m pytest -q tests/test_pt_tf32_gpu.py -p no:cacheprovider --basetemp=artifacts/pt_tf32_1/synccheck_tmp

# After kernel correctness is GREEN, run onset first; stop on a RED gate.
# Each state runs fresh none and 0-10 arms; checkpoint/batch/engine pins checked.
CUDA_VISIBLE_DEVICES=0 timeout --kill-after=15s 1200s python3 -B scripts/pt_tf32_grapa.py state --state onset --lead-gpu --out artifacts/pt_tf32_1/lead_onset
CUDA_VISIBLE_DEVICES=0 timeout --kill-after=15s 1200s python3 -B scripts/pt_tf32_grapa.py state --state healthy --lead-gpu --out artifacts/pt_tf32_1/lead_healthy
CUDA_VISIBLE_DEVICES=0 timeout --kill-after=15s 1200s python3 -B scripts/pt_tf32_grapa.py state --state control --lead-gpu --out artifacts/pt_tf32_1/lead_control

# After all three model states pass: none and 0-10, 20 steps each.
CUDA_VISIBLE_DEVICES=0 timeout --kill-after=15s 1200s python3 -B scripts/pt_tf32_grapa.py timing --lead-gpu --out artifacts/pt_tf32_1/lead_step_time
```

Output directories are create-only. Preserve failed/incomplete receipts; use a
new directory for a separately registered retry. The harness reports each
case incrementally. The final SOURCE_MANIFEST is immutable; a source/binary
change needs a separately registered successor, not a hidden repin.
To reproduce the local build: `CUDA_VISIBLE_DEVICES='' python3 -B scripts/pt_tf32_build.py`.
This creates another build receipt and the host guard executable. A changed
binary will intentionally fail the existing source manifest until registered
separately by the lead.

## Optional forward-FP32/backward-BF16 split — deferred design

Not implemented: the order permits it only after (1)+(2) miss the step target,
and no GPU timing was authorized. Preserve FP32 initial forward, FP32 checkpoint
replay and the FP32 saved state; add an explicit per-block backward compute
policy so each VJP consumes casts of its saved operands and dO in BF16, keeps
tensor-core accumulation in FP32, and converts results to the parent tensor's
gradient dtype. FP32 master parameter gradients/optimizer states remain FP32.
Checkpoint replay must re-enter the forward policy while constructing that
backward graph. Casting just the block output/dX does not change the internal
GEMM/attention VJPs and is not this split.

Prior art: [Mixed Precision Training, Micikevicius et al., ICLR 2018](https://arxiv.org/abs/1710.03740)
for low-precision compute with FP32 master state; CC39-B (2026) for the specific
forward/backward precision ablation, and CC41 for block-scoped replay. Taken:
the mixed-precision separation; ours: proposed scope/plumbing in this engine.
The code-site design annotation is in `pt_tf32_grapa.install_policy`.
This remains a design, not a validated implementation.

## Done

Authorized CPU/build deliverables are complete. **No GPU gate is claimed run,
passed, or fixed.** The literal blocked report for every GPU lane is
`artifacts/pt_tf32_1/BLOCKED_REPORT.json` (all `measured=false`). In particular,
TF32's reduced input mantissa may violate the stringent fp32-a spread gate and
may fail the sensitive onset. Build/SASS/CPU evidence cannot settle either.

Final build receipt `artifacts/pt_tf32_1/build_05/receipt.json`, verbatim lines:

```text
[100%] Built target _tensor_cuda
BUILD_RC=0 SOURCE_PINS_UNCHANGED=True RECEIPT=/mnt/ForgeRealm/wt/pt-tf32/artifacts/pt_tf32_1/build_05/receipt.json
```

Final author checks, verbatim (logs `cpu_tests_final.log`, `mutation_final.log`,
`gpu_collection_final.log`; collection does not execute GPU tests):

```text
70 passed in 1.27s
MUTATION GREEN: killed=10/10, errors=0, source_unchanged=True
```

The 70 tests include the host executable's six native rejection checks. Its
final output is `HOST_CONTRACT: 6 expected rejections; no CUDA device operations`.
Library loading may register CUDA fatbinaries; the test makes no CUDA device
calls. The earlier wording “no CUDA API called” was too broad and was corrected.
The final GPU collection contains **49 tests, all unexecuted**. Mutation scope
is only author CPU conversion/model/gate logic, never native CUDA or blind
coverage; final receipts are `MUTATION_REGISTRATION_002.json` and
`MUTATION_RECEIPT_002.json`.

Existing BP suite after provisioning, verbatim:

```text
7 failed, 121 passed, 1 skipped in 2.95s
```

No original assertion or marker was edited. Five failures are pre-existing:
three BP-KERNEL-1 checks require the pruned `grapa-bp1` receipt, the BP-KERNEL-2
companion requires its pruned import path, and BP-KERNEL-1's old full-ops-prefix
check was already false in the saved initial fork. Two are **new frozen-pin
failures caused by this authorized implementation**: BP-KERNEL-4's exact
backward-setter text changes to admit g1_tf32, and CMake gains the new CUDA file.
Verbatim distinguishing errors:

```text
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/ForgeRealm/wt/grapa-bp1/artifacts/bp_census_1/receipt.json'
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/ForgeRealm/wt/grapa-bp1/scripts/bp_census_1.py'
ValueError: pin drift: tensor_cuda/CMakeLists.txt
```

Full failing node IDs/assertions are in `legacy_tests_02.log`; before/after
source predicates in `legacy_pin_disposition.json` establish that classification.
The one module skip is the repository's existing pruned-worktree classification;
its companion failure was deliberately included by `--campaign-receipts`.
**The full existing suite is RED, not represented as passing.**

Primary SHA256 receipts (complete source/test/doc/build list in
`artifacts/pt_tf32_1/SHA256SUMS`):

```text
42d72654dd75aacd23ed8b46750d04930166bad4db761c3f0bff6747fe7d8e06  artifacts/pt_tf32_1/REGISTRATION.json
d002c2cf25c71d664d475d2abe33e8a1916e3044a101cfff175fc2517494b706  artifacts/pt_tf32_1/SOURCE_MANIFEST.json
74bb57f7a6b3458007de89b35fc1313d3d679f2b751b2ec57da1b74215d488b2  artifacts/pt_tf32_1/GRAPA_REGISTRATION.json
cadcdaf11777215ef3d8b17dbf10d3bedaf262e3c4c8ccf4e01536d3bb98687d  tensor_cuda/src/attention_tf32.cu
196beae9e04e8297df52835eee692e0c7f2865c484db4838eea6ca7267097ceb  tensor_cuda/src/kernels.cu (UNCHANGED)
7954022c0d83d00676e8ee237f0925e082fa906de2066f98262477cdf872ef29  tensor_cuda/src/matmul.cu
c99edf3263e938c83d1fc8ce8312d43e602c1057b3182bd896fe4d764a1c4806  tensor_cuda/src/ops.cpp
5d7b1bd37fa47ef5dac84de67cb4546acdbac0b901a3879db274a851f207b783  tensor_cuda/src/bindings.cpp
0ce976a59fa035590612320eb3c7fdd999b59ce8afef9c00d786aca28ffea51d  tensor_cuda/include/tc/tf32.h
e6d4afcb8794af3704615436e5214cfb74f58c6e9f3f133e187f82eaa2aada56  tensor_cuda/CMakeLists.txt
16e3cb6362721cf5fbff9e9c65fcf5e7f582a2519013e1ed5de3e450145f3cbf  tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
```

No GPU work, no git, no subagents, no process signals, no GPU-lock access and
no live/GRAPA changes. The transient compiler-directory slip and all failed
build/test attempts are retained above. The lead owns GPU validation, blind
verification, any successor registration and any deployment decision.
