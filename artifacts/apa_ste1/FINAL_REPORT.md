# APA-DK-STE1 final report
CPU DELIVERY COMPLETE. S-G1 through S-G4 PASS within the scopes below.
L-G1 through L-G6 are completely STAGED, NOT RUN. Engine use requires the
lead's GPU PASS and cross-model review. No production promotion performed.
Additional literal all-specialization SASS identity diagnostic is RED for
seven address-immediate-only mismatches; this is disclosed, not suppressed.

## Results and claim boundaries

| Gate | Result | Evidence / limit |
|---|---|---|
| S-G1 census | PASS | Complete source entry/kernel/dispatch and operand census below; actual GRAPA MLA head path and run7 rope64 registration inspected read-only. |
| S-G2 CPU reference | PASS | 23 task CPU tests, receipts/cpu_artifacts.log/xml. Float64 on/off central differences (eps1e-6, atol2e-7/rtol2e-5); refine-all off=on=dense (atol2e-12); source operands, switch parsing, actual manifest/input bytes, staged command bounds. This is the task suite, not all repository GPU tests. |
| S-G3 build/init | PASS | receipts/BUILD_RECEIPT.json, build_final.log and SWITCH_IMPORT.json. CUDA12.6.85/GCC13.3/pybind11 2.12.0/SM89; CPU import of staged package succeeds. Unset/0/1 print exactly once; empty/2/-1/01/true/space variants refuse initialization with clear TC_APA_DK_STE message. |
| S-G4 mutation | PASS (staged killing check) | SELECTED_ONLY_MUTANT.patch disables all STE specializations while preserving switch print. Four CPU derivative cases reject its off-equivalent key1 gradient (one registered mutant). GPU L-G3 kills it via exact-zero off / nonzero on dK at a never-selected visible key. Production mutant so was not built or GPU-executed. |
| L-G1..L-G6 | STAGED NOT RUN | GPU_COMMANDS.sh and GPU_PASS_RULES.md below; serialized input files present and pinned. No GPU was used by this seat. |

CPU test command, run from the target with a smoke reference and a successful
build/import smoke before the suite:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR="$PWD/artifacts/apa_ste1/scratch" timeout --foreground 120s python3 -B -m pytest artifacts/apa_ste1/tests -p no:cacheprovider -q --junitxml=artifacts/apa_ste1/receipts/cpu_artifacts.xml
```

Result: `23 passed in 3.07s` (unit/suite evidence). An earlier development
source-normalizer test failed because its parser removed if-constexpr
parentheses: `1 failed, 17 passed in 2.36s`; preserved in cpu_final.log/xml,
then corrected without changing product arithmetic or numerical tolerances.
The initial cuobjdump exact-name filter found no mangled function and emitted
warnings; full dumps replaced that inspection (both retained).

Synthetic fixtures cover real registered geometry, not GRAPA activation data:
small1=(B1,H2,L7,S11,D32,VD16,KVH2), noncausal; small2=(B1,H4,L17,S19,D32,
VD16,KVH2), causal GQA; GRAPA=(B1,H16,L4096,S4096,D128,VD64,KVH16), causal.
BF16 plus explicit FP32 TF32 cases. The learned quantizer, trainer and FFN gates
were not evaluated. No model-quality, hypothesis confirmation or certification
claim is made. This switch defaults off and is only an experiment arm.

## Census (verbatim)

# Selective-training census (source inspection / reasoning)
Baseline kernels.cu SHA-256 is
f1951814eae4f8183c3a2d3cd9b86eda9a5e0df74def7f07e5393e3c3f67fb2b,
matching the immutable engine's source. Survey covered all branches before
implementation; scratch/original holds the four original modified source files.
Line numbers below refer to the final source.

| Entry point / dispatcher | Targets and conditions |
|---|---|
| Python `tc.apa_selective_train` -> binding -> `tc::ops::apa_selective_train` (`ops.cpp:778`) | Differentiable public entry. Forward getter defaults `a`; backward getter defaults `a`. Captures q/k/v, detached kq, saved lse/thr; non-a backward also captures out. |
| Python/binding `apa_selective_fwd_train` -> `tc::apa_selective_fwd_train` (`kernels.cu:2639`) | Base forward `apa_selective_fwd_train_kernel<T,DMAX,WCOOP>` (`:2350`), 128 threads/query row; DMAX=64/128/256/512. WCOOP=false only cap<=64 and decode-shaped L=1, rows<2*SM_COUNT (default56); otherwise true. |
| Binding `apa_selective_fwd_train_variant`, ops forward switch | `a` -> base; `h` -> `tc::apa_selective_fwd_train_variant` (`:8220`) -> `tc::apa_selective_fwd_bk4` (`:8193`) -> `bk4::forward<DIAG=false>` (`:8065`), BF16 D/VD<=128; `h_tf32` -> `tc::apa_selective_fwd_tf32` (`attention_tf32.cu:407`) -> `pt_tf32::forward<DIAG=false>` (`:229`), FP32 SM80+. |
| Binding `bp_kernel_4_diagnostic` | `tc::apa_selective_fwd_bk4` or `tc::apa_selective_fwd_tf32` with DIAG=true. `a` -> base plus `bk4::a_mask` / `pt_tf32::a_mask`; `h` / `h_tf32` -> corresponding forward<true>. Masks/clocks diagnostic only. |
| Python/binding `apa_selective_bwd` -> `tc::apa_selective_bwd` (`kernels.cu:2684`) | `apa_selective_bwd_kernel<T,DMAX,DK_STE>` (`:2530`): query-owned dQ, dtype dK/dV atomics, explicit rowdot recomputation. 128 threads; DMAX=64/128/256/512. |
| Binding `apa_selective_bwd_variant` -> `tc::apa_selective_bwd_variant` (`ops.cpp:1159`) | `a` -> base; `b/c/d` -> bk1; `f/f_pass_a` -> bk2; `g1/g2` -> bk3; `g1_tf32` -> TF32. Validation/branches unchanged. |
| `tc::apa_selective_bwd_bk1_cuda` (`kernels.cu:2856`) | `apa_selective_bwd_bk1_kernel<T,DMAX,WRITE_KV,OUTPUT_DOT,DK_STE>` (`:2736`). b=true/false (FP32 dK/dV atomics, recomputed rowdot); c=false/false (deletion diagnostic); d=true/true (saved-output dot rowdot). Final casts unchanged off. |
| `tc::apa_selective_bwd_bk2_cuda` (`:2980`) | Query prepass reuses bk1<WRITE_KV=false,OUTPUT_DOT=(f),DK_STE=false>, writes saved_rowdot. Key pass `apa_selective_bwd_bk2_kv_kernel<T,DMAX,DK_STE>` (`:2915`), one block per batch/KV-head/key, fixed FP32 reductions, no atomics. f uses output dot; f_pass_a recomputes rowdot. |
| `tc::apa_selective_bwd_bk3_cuda` (`:3207`) | BF16 `bk3::backward<KEY,DENSE,DK_STE>` (`:3076`), D/VD<=128. g1 DENSE=false selective scalar exact scores; g2=true dense exact MMA scores. Query half KEY=false,STE=false; key/value half KEY=true,STE=switch. Owner tiles, BF16 coefficient MMA, saved-output rowdot. |
| `tc::apa_selective_bwd_tf32` (`attention_tf32.cu:447`) | FP32 `pt_tf32::backward<KEY,DK_STE>` (`:87`), D/VD<=128, SM80+. Query half false/false; key/value half true/switch. Existing TF32 coefficient products, saved-output rowdot, singleton identity unchanged. |
| Backward training setter `bp_kernel_2_set_variant` | Accepts a/b/d/f/f_pass_a/g1/g2/g1_tf32. c is bare diagnostic only and cannot be selected for autograd. Forward setter accepts a/h/h_tf32. No TC environment variable changes either getter. |

This is the complete kernel/launcher graph for this op: three training-forward
kernel families, two diagnostic-mask kernels, five backward kernel families
(base, bk1, bk2, bk3, TF32). Inference selective/sink/split-K/int4/SP and composed
blend-softmax/GEMM attention are separate APIs, not dispatched by this op.

GRAPA's specified call dispatches base forward
`apa_selective_fwd_train_kernel<__nv_bfloat16,128,true>` and base backward
`apa_selective_bwd_kernel<__nv_bfloat16,128,STE>`, grid=65536, block=128,
KVH=16, group=1. Reason: B=1,H=16,L=S=4096,D=128,VD=64 means cap=128;
L>1 rules out decode; both variant getters initialize to a. zthr is the
engine `_norm_ppf(0.85)` cast to float, exactly as bindings already do.

MLA confirmation (read-only real model code): GRAPA `grapa/attention_mla.py`
sets qk=nope+rope at :58; kv_b reshapes to B,L,H,nope+vdim at :103;
k_nope and v are per-head, k_pe broadcasts to H, and cat assembles k_h with H
heads at :119-120. Therefore KVH=H=16 after up-projection, not one latent head.
`grapa/attention.py:66-72` selects tc.apa_selective_train for causal grad-enabled
refine<1 with kq supplied. `/mnt/ForgeRealm/grapa_run7/registration.json` records
checkpoint-carried qk_rope_dim=64 (overriding the repo's older default32);
nope64+rope64 gives D128 and VD64. No trainer/registration file was changed.

`TC_DETERMINISTIC=1` / legacy `TC_DET_EMBED_BWD=1` initialize one shared policy
in embed_deterministic.cu:21-25. Only embedding and gather/scatter use it.
They do not redirect APA. Base/bk1 still use atomics; bk2/bk3/TF32 have owners.
Neither global APA determinism nor FFN gate behavior is established here.

# Operand census and off-path argument

| Touched site | Only on change | Off-path identity reasoning |
|---|---|---|
| base backward + launcher | Template DK_STE makes dK guard `sel || DK_STE`. | false folds exactly to original `if(sel)`; same dK cast, q/scale/dscore, dQ/dV, grids and reductions. All 12 off SASS specializations have exact original instruction encodings. |
| bk1 backward + launcher | Same guard in WRITE_KV paths; diagnostic c adds only unselected dK, using FP32 scratch and one cast. | false preserves WRITE_KV=false elimination for c, original pointers, allocations, zeroing/casts and guard for b/d. Query prepass of bk2 always instantiates STE=false. |
| bk2 KV backward + launcher | `selected || DK_STE` admits unselected ds*q to existing local ak. | false recovers original guard; owner/thread traversal and fixed reduction trees unchanged. Query prepass and saved_rowdot unchanged. |
| bk3 backward + launcher | Only KEY=true coefficient sel receives ds for unselected keys. | false yields original selected?ds:0; query half is the unchanged false specialization. p, ds, bulkds, value coefficients and WMMA order unchanged. |
| TF32 backward + launcher | Same KEY-only coefficient admission. | false restores original coefficient; query half stays false; singleton, probability, ds and dV coefficient arithmetic unchanged. |
| ops switch helper + extension init | Strict process-wide frozen getenv parse; once-only stderr receipt; host dispatch before loops. | Unset/0 selects false; no CUDA work, forward edits or launch-shape changes. Invalid strings fail during import before use. |
| restored deterministic support files | None: exact base source copies missing from delivered worktree. | Hashes equal base build's source pins; no algorithm modification. |

`test_source.py` checks every forward body byte-for-byte and backward operand
bodies after substituting STE=false. Static SASS evidence: 75/78 off and 15/19
forward specializations have identical encodings. The remaining 3/4 differ
only in immediates of MOV/UMOV/LDS/STS address instructions; arithmetic/register
instruction sequence and instruction count match. This is consistent with
changed shared allocation (reasoning), not literal encoding identity. The raw
all-specialization SASS identity diagnostic remains RED. Runtime equal output
bytes require L-G1/L-G2; atomic ordering cannot be proved by source equivalence.

# Prior art
Bengio, Leonard and Courville (2013) STE: identity derivative for detached
quantizer, integrated here only into key accumulation. Dao et al. (2022)
FlashAttention: saved-lse backward recomputation already in this engine;
Dao (2023) FlashAttention-2 owner partitioning already in bk2/bk3. NVIDIA
CUDA/WMMA/Nsight systems: existing kernels and static/timing instrumentation.
Unverified - lead to check these names/years. No new estimator/attention
algorithm claimed; switch integration and evidence harness are this order.

## Diff summary and off-path identity argument (verbatim)

# Diff summary
Four existing source files changed, one header added; three missing baseline
support files restored unchanged from the base engine's read-only source.

- `kernels.cu`: annotate mechanism; add a compile-time DK_STE boolean to base,
  bk1, bk2, bk3 backwards; launch false/true specializations on the host. Key
  contribution is admitted for unselected keys when on. bk2 query prepass and
  bk3 query half explicitly remain false. The diagnostic c keeps selected dK
  and dV zero and adds only unselected dK when on.
- `attention_tf32.cu`: equivalent KEY-only STE coefficient and host dispatch;
  query half remains false. Includes the otherwise easy-to-miss TF32 path.
- `ops.cpp` + `include/tc/apa_dk_ste.h`: strict cached process opt-in, unset/0 off,
  1 on, every other string errors. Prints exactly
  `tensor_cuda: TC_APA_DK_STE=<0|1>` once to stderr.
- `bindings.cpp`: call the helper at extension initialization, before exposing
  any tensor use. No Python package behavior changed.
- Restored `embed_deterministic.cu`, `gather_deterministic.cu`, and
  `deterministic_gather.h`: hashes exactly match base build source pins.

No forward body, selection, threshold, lse, p, dScore, dQ or dV arithmetic was
changed. The false specialization recovers original guards, pointer choices,
allocations, casts, launches and fixed reduction orders. CENSUS.md contains
per-operand proofs; SOURCE_DIFF.patch contains the complete source diff.
Default base backward has 12/12 exact instruction encodings vs 69b154cd.
Across all families, literal SASS equality is RED for 7 address-immediate-only
mismatches (3 off, 4 forward); 75/78 off and 15/19 forward match exactly.
Runtime byte equality remains a lead gate. No byte-identical runtime claim is
made from these CPU/static checks, particularly for atomic accumulation order.

Prior art: Bengio, Leonard and Courville (2013) STE; Dao et al. (2022)
FlashAttention saved-state recomputation already present. Unverified - lead to
check. Integration is key-only; no new estimator or attention algorithm claim.

## Build and staged engine

Built so SHA-256: `af9ffce4733628caac5b6bd7a345de86c66c1f67d5150a53d7454b3d4bcb3760`.
Baseline so SHA-256: `69b154cd2e9a5a7cbe288e1419d963697e5add57d2b4ee827c4c78a273ff2c5c`.
Exact outer build command (both smoke and final source builds succeeded):

```bash
CUDA_VISIBLE_DEVICES= timeout --foreground 1050s bash artifacts/apa_ste1/build.sh
```

Build script (verbatim; CMakeLists.txt unchanged):

```bash
#!/usr/bin/env bash
# Prior art: CMake/Project-Tensor build (2026), unchanged toolchain; staging ours.
set -euo pipefail
cd /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_ste1
export CUDA_VISIBLE_DEVICES=
export TMPDIR="$PWD/artifacts/apa_ste1/scratch"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDACXX=/usr/local/cuda-12.6/bin/nvcc
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
timeout --foreground 120s cmake -S tensor_cuda -B artifacts/apa_ste1/scratch/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc -DCMAKE_CXX_COMPILER=/usr/bin/c++ -DFETCHCONTENT_SOURCE_DIR_PYBIND11="$PWD/artifacts/apa_ste1/scratch/vendor/pybind11" -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_BUILD_RPATH=/usr/local/cuda-12.6/targets/x86_64-linux/lib
timeout --foreground 900s cmake --build artifacts/apa_ste1/scratch/build -j4
```

The exact baseline build receipt was inspected at
`/mnt/ForgeRealm/Project-Tensor/WorkTree/tensor_wt_det_cert/artifacts/pt_det_1/build_02/receipt.json`;
it names the immutable 69b154cd so and same CUDA/compiler/architecture/vendor
version. readelf confirms the same RUNPATH and GCC version in the staged so.
Package files match base manifest hashes; only the extension differs. The
engine folder is lab-local `artifacts/apa_ste1/engine/`; lead publishes later.
`source_commit` is the user-supplied f2c686e, not independently queried via git.
All source bytes are hashed in the manifest. Three baseline support files were
missing from delivered checkout and restored unchanged with provenance in
receipts/restored_source.json. No alternative implementation was substituted.

## Staged GPU commands (verbatim; NOT executed)

Run the following script as written. It contains the exact individual command
lines, serialized execution, thread pinning and an EXIT receipt verifier. It
requires a fresh lead/ capture directory; no implicit retries or deletions.

```bash
#!/usr/bin/env bash
# Lead only. Exact staged commands; never run by CPU seat.
set -euo pipefail
cd /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_ste1
export TMPDIR="$PWD/artifacts/apa_ste1/scratch"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export FORGE_GPU_RUN_LOG="$PWD/artifacts/apa_ste1/lead/governor.tsv"
finish_receipt() {
  local command_rc=$?
  trap - EXIT
  set +e
  CUDA_VISIBLE_DEVICES= timeout --foreground 120s python3 -B artifacts/apa_ste1/gpu_gate.py verify
  local verification_rc=$?
  if [[ "$command_rc" -ne 0 ]]; then
    exit "$command_rc"
  fi
  exit "$verification_rc"
}
trap finish_receipt EXIT
forge-gpu-run --min 300 --max 600 --label ste1-base-small -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which base --switch 0 --suite small --label base-small
forge-gpu-run --min 300 --max 600 --label ste1-arm-off-small -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which arm --switch 0 --suite small --label arm-off-small
CUDA_VISIBLE_DEVICES= timeout --foreground 60s python3 -B artifacts/apa_ste1/gpu_gate.py calibrate
forge-gpu-run --min 300 --max 600 --label ste1-arm-on-small -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which arm --switch 1 --suite small --label arm-on-small
forge-gpu-run --min 300 --max 600 --label ste1-arm-repeat-small -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which arm --switch 1 --suite small --label arm-repeat-small
forge-gpu-run --min 300 --max 600 --label ste1-arm-off-all-small -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which arm --switch 0 --suite small --label arm-off-all-small --all
forge-gpu-run --min 300 --max 600 --label ste1-arm-on-all-small -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which arm --switch 1 --suite small --label arm-on-all-small --all
forge-gpu-run --min 300 --max 600 --label ste1-base-grapa -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which base --switch 0 --suite grapa --label base-grapa
forge-gpu-run --min 300 --max 600 --label ste1-arm-off-grapa -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which arm --switch 0 --suite grapa --label arm-off-grapa
forge-gpu-run --min 300 --max 600 --label ste1-arm-on-grapa -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which arm --switch 1 --suite grapa --label arm-on-grapa
forge-gpu-run --min 300 --max 600 --label ste1-arm-repeat-grapa -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which arm --switch 1 --suite grapa --label arm-repeat-grapa
forge-gpu-run --min 300 --max 600 --label ste1-arm-off-all-grapa -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which arm --switch 0 --suite grapa --label arm-off-all-grapa --all
forge-gpu-run --min 300 --max 600 --label ste1-arm-on-all-grapa -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which arm --switch 1 --suite grapa --label arm-on-all-grapa --all
forge-gpu-run --min 300 --max 600 --label ste1-timing-0 -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py profile --switch 0
forge-gpu-run --min 300 --max 600 --label ste1-timing-1 -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py profile --switch 1
```

## GPU pass rules and budgets (verbatim)

# Staged GPU pass rules (no GPU execution by seat)
Run GPU_COMMANDS.sh as written, in a clean lead/ receipt directory. Every label
is unique in this registration. Existing paths fail closed; preserve failed
runs and register a separate run directory/labels before retrying, never delete.
All inputs exist in inputs/ and hashes are in INPUTS.json. Staged package is
engine/tensor_cuda/tensor_cuda. Baseline is the immutable 69b154cd engine.
The command uses the current installed forge-gpu-run, with --min 300 --max 600;
the command itself is bounded by timeout --foreground 300s. Refused windows
are RED/missing, never a pass. No command here has been run on a GPU by seat.

L-G1: base/off uint16 BF16 and uint32 FP32 views equal bytes on all six tensors,
on GRAPA geometry and both small geometries. Includes train autograd (small),
all valid diagnostic variants except zero-output c; includes h/h_tf32 forwards. No masks or tolerances.
L-G2: on/off exact out,lse,thr on those cases; input bytes also equal.
L-G3: each on output/gradient matches float64 NumPy with tolerances frozen in
CALIBRATION.json BEFORE any selective-on capture. E is the pooled measured
base/off maxabs error per dtype and tensor across two small shapes/variants;
tolerance=max(8E,1e-5). Recompute calibration from raw base tensors and require
identity to the saved calibration. Every variant must give exactly zero off dK
and nonzero on dK at key 1, whose kq is all zero and threshold strictly positive.
This is the selected-only mutant's killing check. Variant c uses its deletion
ablation reference and contributes no zero tensors to bitwise gates.
L-G4: zthr=-1e6, all-selected fixtures, on/off all six tensors equal bytes.
Recheck stored thresholds are negative on all multi-key rows; the pinned
singleton dot must also meet the saved threshold. This proves every visible
pair selected without materializing a GRAPA-size pairwise mask.
The large fixture pins q(query 0) and kq(key 0) to .25 so the singleton bulk dot
is exact in either reduction order. No forward/threshold production code change.
L-G5: two fresh on captures equal bytes for all six tensors, deterministic mode.
Existing atomic variants can fail: TC_DETERMINISTIC does not control APA atomics.
A failure remains RED. This harness never changes dispatch to obtain equality.
L-G6: two kernel samples per arm after one warmup; Nsight CUDA records include
only apa_selective_bwd_kernel captured between cudaProfilerStart/Stop. Every
duration finite positive, exactly two kernel launches, report median and ratio.
CUDA-event backward wrapper samples are reported separately. No speed threshold;
P2 (1.3-2x) is a prediction only. Whole model/optimizer step unavailable from the
engine-only entry point; no registered checkpoint/optimizer fixture was supplied;
GRAPA times its step. Do not label backward-wrapper timing a training step.

Each capture's PASS only checks capture health. The gate verdict is recomputed
by verify from stored raw tensors/profiler records, not copied from that flag.
A GPU gate PASS additionally requires each dependency's governor TSV row to be
unique with outcome=ran and rc=0. If a command times out or does not run, it
cannot count even if partial files exist. Final GPU_RECEIPT.json contains PASS
for each L-G1..L-G6 and the combined PASS.

Work/time estimates (reasoning, no GPU timing evidence): small job processes
24 cases with <=4*17*19 pairs/head geometry, <=32 feature width: under ~10^8
scalar operations, estimated 2-25 s including CUDA cold start/I/O. Large job
has 16*4096*4097/2=134250496 visible query-key pairs, 128/64 feature widths:
roughly 1e11 scalar arithmetic operations and order 1e11 bytes of traffic per
backward, including up to ~2.58e10 BF16 atomics on. Estimate 10-120 s per capture.
Timing job: 1 warmup + 2 sampled backwards, one forward, Nsight export: estimate
30-240 s. These are scheduling estimates, not measured guarantees; 300 s is a
budget, and any timeout is RED. Jobs run sequentially, never concurrently.

Prior art: Bengio et al. 2013 STE; Dao et al. 2022 FlashAttention VJP;
NVIDIA CUDA events/Nsight Systems (years unverified - lead to check);
NIST FIPS 180-4 (2015) SHA-256. Integration/fixtures/receipt rules are ours.

## Engine manifest (verbatim)

```json
{
  "base_engine_dir": "/mnt/ForgeRealm/engines/tf32-det-69b154cd2e9a",
  "base_engine_so_sha256": "69b154cd2e9a5a7cbe288e1419d963697e5add57d2b4ee827c4c78a273ff2c5c",
  "build_command": "CUDA_VISIBLE_DEVICES= timeout --foreground 1050s bash artifacts/apa_ste1/build.sh",
  "build_log": "/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_ste1/artifacts/apa_ste1/receipts/build_final.log",
  "build_script_sha256": "53e76e19917950f857d5a115710035d06a353359ccddaed7290a0ea1fb909b1a",
  "default": 0,
  "engine_dir": "/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_ste1/artifacts/apa_ste1/engine",
  "engine_path": "/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_ste1/artifacts/apa_ste1/engine/tensor_cuda",
  "files": {
    "__init__.py": "0f0a24a4ccbaff5294ae1577a7c2b7d93d8b0a57577bb165bb1f859268084c5c",
    "_tensor_cuda.cpython-312-x86_64-linux-gnu.so": "af9ffce4733628caac5b6bd7a345de86c66c1f67d5150a53d7454b3d4bcb3760",
    "apa_sp2.py": "018a26b6bf7acfd85bc23e5084d0367952e1b2c81898da8585c8ecda302a9f07",
    "functional.py": "418ee285bdeefacbd9d5a1ccaf39700b8e83ea339377230c79f1d4480a1e3d9c",
    "nn.py": "4659cdfa382c0166543a555ef5101e0e3b7ceb8d930bb822be95a0324263517d",
    "optim.py": "41693c324d5ea2f081ec7777ce8165fbc65d04a7b557fce27ff6d9b9a24457ee",
    "quant.py": "f5ceea6a862d7950217441c8aae3b5b957c99c922412ef9489f464f62c72b00a",
    "quantization/__init__.py": "d29a048409906800e68bbed774978764045451b4451eb2d31ec9d5343293658f",
    "quantization/affine.py": "8bf31258f8635fa4948d08805fb4909ea6a36189d40f0ee2cc589b796c5e3943"
  },
  "installed_at": "2026-10-25T02:04:18.936150+00:00",
  "name": "apa-dk-ste",
  "package": "/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_ste1/artifacts/apa_ste1/engine/tensor_cuda/tensor_cuda",
  "prior_art": "Bengio, Leonard and Courville (2013) STE identity derivative; Dao et al. (2022) FlashAttention recomputation (existing); NIST FIPS 180-4 (2015) SHA-256; unverified - lead to check",
  "restored_source_receipt": "/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_ste1/artifacts/apa_ste1/receipts/restored_source.json",
  "rollback": "default off; immutable base engine remains unchanged; no installation outside lab fork",
  "runpath": [
    "/usr/local/cuda-12.6/targets/x86_64-linux/lib"
  ],
  "schema": "cc42-engine-manifest-v1",
  "so": "_tensor_cuda.cpython-312-x86_64-linux-gnu.so",
  "so_path": "/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_ste1/artifacts/apa_ste1/engine/tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so",
  "so_sha256": "af9ffce4733628caac5b6bd7a345de86c66c1f67d5150a53d7454b3d4bcb3760",
  "source": {
    "package": "/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_ste1/tensor_cuda/tensor_cuda",
    "path": "/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_ste1/tensor_cuda"
  },
  "source_commit": "f2c686e",
  "source_commit_provenance": "user order; no git invoked; kernels baseline hash verified f1951814eae4f8183c3a2d3cd9b86eda9a5e0df74def7f07e5393e3c3f67fb2b",
  "source_file_hashes": {
    "tensor_cuda/CMakeLists.txt": "c88ef53161580ecd6bd7eb88620cc155f903ab73de963544d8ce3eb31d92f175",
    "tensor_cuda/include/tc/apa_dk_ste.h": "10fbbd3c26d9b874c493b0e0d74161436fc84481cb77e27cbe5787edb5cd51e4",
    "tensor_cuda/include/tc/autograd.h": "22c5bc0123e40e506181302fa091cad0861b0a05b202d428c253421866a6b32c",
    "tensor_cuda/include/tc/bp_op_timing.h": "fcdcd19bcbb5e2935c8cbb898bd47a802456d1da65f1d0748fa955124932b5da",
    "tensor_cuda/include/tc/core.h": "1945ab470b2513c22cafc32180590036483f22925169f386432d0bac6c7ba860",
    "tensor_cuda/include/tc/deterministic_embed.h": "14e4f566f5590a003fcd35aabf68bc10e190978d072d6ed8aba3c4f30c4e8453",
    "tensor_cuda/include/tc/deterministic_gather.h": "0bab127a26b3c775024ea8ffa0c5c5e629b2cf2609f42611c120d70b68aff95c",
    "tensor_cuda/include/tc/ops.h": "dd0ef60f15573d6ceec98828e76ee79846da18aedf63916db2d9e0aab12f437e",
    "tensor_cuda/include/tc/tf32.h": "8574889d13e110c2e2ad44a31d825be664d5e8043e7f6ee5110a2b1ce39af155",
    "tensor_cuda/src/apa_sp1_1.cuh": "74248ccb1579efd10d2c7a64473030b1358166875a753cc9f1339829f8afe48a",
    "tensor_cuda/src/apa_sp2.cuh": "f186c255c7f1aec743e83c884d1ed8eb558630116cdbb5e7295352c6c26aa5ad",
    "tensor_cuda/src/attention_tf32.cu": "b5e1e00172a0fdfa013e61fb723604380edb94e19edee625c71c7ba4628569ae",
    "tensor_cuda/src/autograd.cpp": "833bbf8c6aa9b308b655dbc54e6908a79fe5db5702c5bb7f207f9f860487e188",
    "tensor_cuda/src/bindings.cpp": "be8c5dd259feee3bbbd07d911e684859c136f0aa39611b1f8cd0f7fda81f20db",
    "tensor_cuda/src/conv.cu": "1cda2a6e147dc3dd599ea57e89b3423ff68464de8c837f51ed8d433dc18fe0a8",
    "tensor_cuda/src/dda.cu": "58289183587f1aa936321d85de232d03e71922845851cead862c8a526974d6d1",
    "tensor_cuda/src/embed_deterministic.cu": "44e7182a5e00fec534f681dabc4349c3bef433612f4e7355e0cb2731f70529b7",
    "tensor_cuda/src/gather_deterministic.cu": "05ce1d3e814521823f8f77bdff347c632112f4ba3f290246eaf9fa77870b4758",
    "tensor_cuda/src/gemm_apa.cu": "4827910cc9383b4c6cf7505b12da99eb6e54cc8792a3a43d492a58de8eee25f6",
    "tensor_cuda/src/kernels.cu": "4b537cc7b2e0dce992f66240c03770c403d8888414485e824c53675c3d366649",
    "tensor_cuda/src/matmul.cu": "5ec0119a4a9bc53f1bc2724d6a0176ce2a3dad8b94a32c1b928af4e15b7ab68a",
    "tensor_cuda/src/ops.cpp": "ed041cfb06c1994cc1198fedfc6f27b35bb94daffe3a885c32d81b7fca6a75dc",
    "tensor_cuda/src/paint_bake.cu": "5c5ea305df2da870b03dedd44d698728d6a13ecdf1d07e8ef1f7bcd8c4d13f50",
    "tensor_cuda/src/paint_inpaint.cu": "77fd450425aeec42b5d0dbb29069dfbd4c50aef067382c48801cae9c3b33cce7",
    "tensor_cuda/src/paint_raster.cu": "cd6da34283999d2624087d81420c00f427f143ebadb163f3baad158dc199cab7",
    "tensor_cuda/src/terrain.cu": "11bd7c7b3f6d0a80a6b627e48dede0e68869ffc03af8d20d7112e68b56f97d73",
    "tensor_cuda/src/terrain_objects.cu": "401b56b58f747878106b0b333d2b589be72fa32cb1f7824046653e0a27b267a0",
    "tensor_cuda/src/tf32_gemm.cu": "fcce9972e8fa63d5d06a6be29982691d1761820ac1ff14dbd21a7734d2193027"
  },
  "source_references_checked": [
    "/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_ste1/tensor_cuda",
    "/mnt/ForgeRealm/wt/pt-det-cert/tensor_cuda"
  ],
  "status": "CPU-built; GPU gates NOT RUN; cross-model review pending; not approved for training or promotion",
  "switch": "TC_APA_DK_STE",
  "toolchain": {
    "cuda": "12.6",
    "cuda_architectures": [
      "89"
    ],
    "cxx": "/usr/bin/c++",
    "gcc_version": "13.3.0",
    "nvcc": "/usr/local/cuda-12.6/bin/nvcc",
    "pybind11": "2.12.0"
  },
  "variant": "apa-dk-ste"
}
```

## Prior art

- Bengio, Leonard and Courville (2013), straight-through estimator: identity
  derivative through detached key quantization; this work adds an opt-in key
  accumulation only. Unverified - lead to check: Bengio 2013 straight through
  estimator, arXiv 1308.3432.
- Dao et al. (2022), FlashAttention: saved-lse softmax backward recomputation
  already in the engine; Dao (2023), FlashAttention-2 key ownership already in
  bk2/bk3. Unverified - lead to check: FlashAttention backward recomputation and
  FlashAttention-2 backward partitioning. No new attention algorithm claimed.
- Gray and Neuhoff (1998), Quantization survey: uniform scalar rounding in
  synthetic fixtures only; not GRAPA's learned quantizer. Burden and Faires,
  Numerical Analysis (2011): symmetric central finite differences in CPU tests.
  Both unverified - lead to check by title/author/year.
- NVIDIA CUDA/WMMA/cuobjdump/CUDA events/Nsight Systems: existing kernel APIs,
  disassembly and kernel timing instrumentation (years unverified - lead to
  check). NIST FIPS 180-4 (2015): SHA-256 manifests. Project-Tensor (2026):
  variant dispatch, diagnostic deletion c and manifest schema. Ours: switch
  integration, serialized fixtures, bounded command staging and raw-receipt
  comparisons. No prior art known to me for the particular test fixtures.

## Residuals / not claimed fixed

Runtime off bytes, on forward bytes, on numerics, refine-all bytes, repeatability
and performance are all unmeasured here. The original atomic base/bk1 APA
backwards are not governed by TC_DETERMINISTIC; those runtime identity and
repeatability risks remain visible to L-G1/L-G5. The seven literal SASS
mismatches are RED static checks; address-only classification is reasoning,
not a replacement for GPU gates. P1 has static support on the default base
instructions, not runtime proof; P2 (1.3-2x) has no timing data. A model/optimizer
step was not timed: the engine has no such entry point and this order supplied
no registered checkpoint/optimizer fixture; GRAPA owns that measurement.
Nothing is promoted, no default is changed, FFN behavior is not claimed fixed.

Registration was written before gates and never edited. Amendments 001-003
separately document restored missing sources, diagnostic c/L19 and pooled-error
interpretation, and full forward-family staged coverage, all before GPU data.
No git, subagents, background waits, file deletions or writes outside the target.
Scratch is retained (vendor/build/original/SASS/mutant/pre-singleton input).

## Every file written

All paths are relative to the authorized target. The following inventory
includes authored/restored sources, staged package files, receipts, serialized
inputs and every generated/vendor/scratch file. FILES_WRITTEN.txt duplicates
this list; DELIVERY_SHA256.json hashes every listed file except itself.

```text
artifacts/apa_ste1/AMENDMENT_001.md
artifacts/apa_ste1/AMENDMENT_002.md
artifacts/apa_ste1/AMENDMENT_003.md
artifacts/apa_ste1/CENSUS.md
artifacts/apa_ste1/DELIVERY_SHA256.json
artifacts/apa_ste1/DIFF_SUMMARY.md
artifacts/apa_ste1/FILES_WRITTEN.txt
artifacts/apa_ste1/FINAL_REPORT.md
artifacts/apa_ste1/GPU_COMMANDS.sh
artifacts/apa_ste1/GPU_PASS_RULES.md
artifacts/apa_ste1/INPUTS.json
artifacts/apa_ste1/LEDGER.md
artifacts/apa_ste1/MUTATION.json
artifacts/apa_ste1/REGISTRATION.md
artifacts/apa_ste1/SELECTED_ONLY_MUTANT.patch
artifacts/apa_ste1/SOURCE_DIFF.patch
artifacts/apa_ste1/build.sh
artifacts/apa_ste1/check_sass.py
artifacts/apa_ste1/engine/ENGINE_MANIFEST.json
artifacts/apa_ste1/engine/tensor_cuda/tensor_cuda/__init__.py
artifacts/apa_ste1/engine/tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/apa_ste1/engine/tensor_cuda/tensor_cuda/apa_sp2.py
artifacts/apa_ste1/engine/tensor_cuda/tensor_cuda/functional.py
artifacts/apa_ste1/engine/tensor_cuda/tensor_cuda/nn.py
artifacts/apa_ste1/engine/tensor_cuda/tensor_cuda/optim.py
artifacts/apa_ste1/engine/tensor_cuda/tensor_cuda/quant.py
artifacts/apa_ste1/engine/tensor_cuda/tensor_cuda/quantization/__init__.py
artifacts/apa_ste1/engine/tensor_cuda/tensor_cuda/quantization/affine.py
artifacts/apa_ste1/gpu_gate.py
artifacts/apa_ste1/inputs/grapa.npz
artifacts/apa_ste1/inputs/small1.npz
artifacts/apa_ste1/inputs/small2.npz
artifacts/apa_ste1/inspect_sass_addresses.py
artifacts/apa_ste1/receipts/BUILD_RECEIPT.json
artifacts/apa_ste1/receipts/LEAD_GATES_STAGED.json
artifacts/apa_ste1/receipts/SASS_ADDRESS_DIFFS.json
artifacts/apa_ste1/receipts/SASS_IDENTITY.json
artifacts/apa_ste1/receipts/SEAT_GATES.json
artifacts/apa_ste1/receipts/SWITCH_IMPORT.json
artifacts/apa_ste1/receipts/build.log
artifacts/apa_ste1/receipts/build_final.log
artifacts/apa_ste1/receipts/cpu.log
artifacts/apa_ste1/receipts/cpu.xml
artifacts/apa_ste1/receipts/cpu_artifacts.log
artifacts/apa_ste1/receipts/cpu_artifacts.xml
artifacts/apa_ste1/receipts/cpu_final.log
artifacts/apa_ste1/receipts/cpu_final.xml
artifacts/apa_ste1/receipts/cpu_release.log
artifacts/apa_ste1/receipts/cpu_release.xml
artifacts/apa_ste1/receipts/cpu_verified.log
artifacts/apa_ste1/receipts/cpu_verified.xml
artifacts/apa_ste1/receipts/import_smoke.log
artifacts/apa_ste1/receipts/prepare.log
artifacts/apa_ste1/receipts/reference_smoke.log
artifacts/apa_ste1/receipts/restored_source.json
artifacts/apa_ste1/reference.py
artifacts/apa_ste1/scratch/INPUTS.pre_singleton.json
artifacts/apa_ste1/scratch/arm_all.sass
artifacts/apa_ste1/scratch/arm_bwd.sass
artifacts/apa_ste1/scratch/base_all.sass
artifacts/apa_ste1/scratch/base_bwd.sass
artifacts/apa_ste1/scratch/build/CMakeCache.txt
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CMakeCUDACompiler.cmake
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CMakeCXXCompiler.cmake
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CMakeDetermineCompilerABI_CUDA.bin
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CMakeDetermineCompilerABI_CXX.bin
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CMakeSystem.cmake
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CompilerIdCUDA/CMakeCUDACompilerId.cu
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CompilerIdCUDA/a.out
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cpp1.ii
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cpp4.ii
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.c
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.cpp
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.gpu
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.stub.c
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.fatbin
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.fatbin.c
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.module_id
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.o
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.ptx
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.sm_52.cubin
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.fatbin
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.fatbin.c
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.o
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.reg.c
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.sm_52.cubin
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CompilerIdCXX/CMakeCXXCompilerId.cpp
artifacts/apa_ste1/scratch/build/CMakeFiles/3.28.3/CompilerIdCXX/a.out
artifacts/apa_ste1/scratch/build/CMakeFiles/CMakeConfigureLog.yaml
artifacts/apa_ste1/scratch/build/CMakeFiles/CMakeDirectoryInformation.cmake
artifacts/apa_ste1/scratch/build/CMakeFiles/Makefile.cmake
artifacts/apa_ste1/scratch/build/CMakeFiles/Makefile2
artifacts/apa_ste1/scratch/build/CMakeFiles/TargetDirectories.txt
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/DependInfo.cmake
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/build.make
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/cmake_clean.cmake
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/cmake_device_link.o
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/compiler_depend.internal
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/compiler_depend.make
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/compiler_depend.ts
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/depend.make
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/deviceLinkLibs.rsp
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/deviceObjects1.rsp
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/dlink.txt
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/flags.make
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/includes_CUDA.rsp
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/link.txt
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/progress.make
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/attention_tf32.cu.o
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/attention_tf32.cu.o.d
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/autograd.cpp.o
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/autograd.cpp.o.d
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/bindings.cpp.o
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/bindings.cpp.o.d
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/conv.cu.o
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/conv.cu.o.d
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/dda.cu.o
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/dda.cu.o.d
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/embed_deterministic.cu.o
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/embed_deterministic.cu.o.d
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/gather_deterministic.cu.o
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/gather_deterministic.cu.o.d
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/gemm_apa.cu.o
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/gemm_apa.cu.o.d
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/kernels.cu.o
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/kernels.cu.o.d
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/matmul.cu.o
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/matmul.cu.o.d
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/ops.cpp.o
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/ops.cpp.o.d
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/paint_bake.cu.o
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/paint_bake.cu.o.d
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/paint_inpaint.cu.o
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/paint_inpaint.cu.o.d
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/paint_raster.cu.o
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/paint_raster.cu.o.d
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/terrain.cu.o
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/terrain.cu.o.d
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/terrain_objects.cu.o
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/terrain_objects.cu.o.d
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/tf32_gemm.cu.o
artifacts/apa_ste1/scratch/build/CMakeFiles/_tensor_cuda.dir/src/tf32_gemm.cu.o.d
artifacts/apa_ste1/scratch/build/CMakeFiles/cmake.check_cache
artifacts/apa_ste1/scratch/build/CMakeFiles/progress.marks
artifacts/apa_ste1/scratch/build/Makefile
artifacts/apa_ste1/scratch/build/_deps/pybind11-build/CMakeFiles/CMakeDirectoryInformation.cmake
artifacts/apa_ste1/scratch/build/_deps/pybind11-build/CMakeFiles/progress.marks
artifacts/apa_ste1/scratch/build/_deps/pybind11-build/Makefile
artifacts/apa_ste1/scratch/build/_deps/pybind11-build/cmake_install.cmake
artifacts/apa_ste1/scratch/build/cmake_install.cmake
artifacts/apa_ste1/scratch/grapa.pre_singleton.npz
artifacts/apa_ste1/scratch/mutant/src/attention_tf32.cu
artifacts/apa_ste1/scratch/mutant/src/kernels.cu
artifacts/apa_ste1/scratch/original/src/attention_tf32.cu
artifacts/apa_ste1/scratch/original/src/bindings.cpp
artifacts/apa_ste1/scratch/original/src/kernels.cu
artifacts/apa_ste1/scratch/original/src/ops.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/.appveyor.yml
artifacts/apa_ste1/scratch/vendor/pybind11/.clang-format
artifacts/apa_ste1/scratch/vendor/pybind11/.clang-tidy
artifacts/apa_ste1/scratch/vendor/pybind11/.cmake-format.yaml
artifacts/apa_ste1/scratch/vendor/pybind11/.codespell-ignore-lines
artifacts/apa_ste1/scratch/vendor/pybind11/.gitattributes
artifacts/apa_ste1/scratch/vendor/pybind11/.github/CODEOWNERS
artifacts/apa_ste1/scratch/vendor/pybind11/.github/CONTRIBUTING.md
artifacts/apa_ste1/scratch/vendor/pybind11/.github/ISSUE_TEMPLATE/bug-report.yml
artifacts/apa_ste1/scratch/vendor/pybind11/.github/ISSUE_TEMPLATE/config.yml
artifacts/apa_ste1/scratch/vendor/pybind11/.github/dependabot.yml
artifacts/apa_ste1/scratch/vendor/pybind11/.github/labeler.yml
artifacts/apa_ste1/scratch/vendor/pybind11/.github/labeler_merged.yml
artifacts/apa_ste1/scratch/vendor/pybind11/.github/matchers/pylint.json
artifacts/apa_ste1/scratch/vendor/pybind11/.github/pull_request_template.md
artifacts/apa_ste1/scratch/vendor/pybind11/.github/workflows/ci.yml
artifacts/apa_ste1/scratch/vendor/pybind11/.github/workflows/configure.yml
artifacts/apa_ste1/scratch/vendor/pybind11/.github/workflows/format.yml
artifacts/apa_ste1/scratch/vendor/pybind11/.github/workflows/labeler.yml
artifacts/apa_ste1/scratch/vendor/pybind11/.github/workflows/pip.yml
artifacts/apa_ste1/scratch/vendor/pybind11/.github/workflows/upstream.yml
artifacts/apa_ste1/scratch/vendor/pybind11/.gitignore
artifacts/apa_ste1/scratch/vendor/pybind11/.pre-commit-config.yaml
artifacts/apa_ste1/scratch/vendor/pybind11/.readthedocs.yml
artifacts/apa_ste1/scratch/vendor/pybind11/CMakeLists.txt
artifacts/apa_ste1/scratch/vendor/pybind11/LICENSE
artifacts/apa_ste1/scratch/vendor/pybind11/MANIFEST.in
artifacts/apa_ste1/scratch/vendor/pybind11/README.rst
artifacts/apa_ste1/scratch/vendor/pybind11/SECURITY.md
artifacts/apa_ste1/scratch/vendor/pybind11/docs/Doxyfile
artifacts/apa_ste1/scratch/vendor/pybind11/docs/Makefile
artifacts/apa_ste1/scratch/vendor/pybind11/docs/_static/css/custom.css
artifacts/apa_ste1/scratch/vendor/pybind11/docs/advanced/cast/chrono.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/advanced/cast/custom.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/advanced/cast/eigen.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/advanced/cast/functional.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/advanced/cast/index.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/advanced/cast/overview.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/advanced/cast/stl.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/advanced/cast/strings.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/advanced/classes.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/advanced/embedding.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/advanced/exceptions.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/advanced/functions.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/advanced/misc.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/advanced/pycpp/index.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/advanced/pycpp/numpy.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/advanced/pycpp/object.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/advanced/pycpp/utilities.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/advanced/smart_ptrs.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/basics.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/benchmark.py
artifacts/apa_ste1/scratch/vendor/pybind11/docs/benchmark.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/changelog.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/classes.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/cmake/index.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/compiling.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/conf.py
artifacts/apa_ste1/scratch/vendor/pybind11/docs/faq.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/index.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/installing.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/limitations.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/pybind11-logo.png
artifacts/apa_ste1/scratch/vendor/pybind11/docs/pybind11_vs_boost_python1.png
artifacts/apa_ste1/scratch/vendor/pybind11/docs/pybind11_vs_boost_python1.svg
artifacts/apa_ste1/scratch/vendor/pybind11/docs/pybind11_vs_boost_python2.png
artifacts/apa_ste1/scratch/vendor/pybind11/docs/pybind11_vs_boost_python2.svg
artifacts/apa_ste1/scratch/vendor/pybind11/docs/reference.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/release.rst
artifacts/apa_ste1/scratch/vendor/pybind11/docs/requirements.txt
artifacts/apa_ste1/scratch/vendor/pybind11/docs/upgrade.rst
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/attr.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/buffer_info.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/cast.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/chrono.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/common.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/complex.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/detail/class.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/detail/common.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/detail/descr.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/detail/init.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/detail/internals.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/detail/type_caster_base.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/detail/typeid.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/eigen.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/eigen/common.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/eigen/matrix.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/eigen/tensor.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/embed.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/eval.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/functional.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/gil.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/gil_safe_call_once.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/iostream.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/numpy.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/operators.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/options.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/pybind11.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/pytypes.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/stl.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/stl/filesystem.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/stl_bind.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/type_caster_pyobject_ptr.h
artifacts/apa_ste1/scratch/vendor/pybind11/include/pybind11/typing.h
artifacts/apa_ste1/scratch/vendor/pybind11/noxfile.py
artifacts/apa_ste1/scratch/vendor/pybind11/pybind11/__init__.py
artifacts/apa_ste1/scratch/vendor/pybind11/pybind11/__main__.py
artifacts/apa_ste1/scratch/vendor/pybind11/pybind11/_version.py
artifacts/apa_ste1/scratch/vendor/pybind11/pybind11/commands.py
artifacts/apa_ste1/scratch/vendor/pybind11/pybind11/py.typed
artifacts/apa_ste1/scratch/vendor/pybind11/pybind11/setup_helpers.py
artifacts/apa_ste1/scratch/vendor/pybind11/pyproject.toml
artifacts/apa_ste1/scratch/vendor/pybind11/setup.cfg
artifacts/apa_ste1/scratch/vendor/pybind11/setup.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/CMakeLists.txt
artifacts/apa_ste1/scratch/vendor/pybind11/tests/conftest.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/constructor_stats.h
artifacts/apa_ste1/scratch/vendor/pybind11/tests/cross_module_gil_utils.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/cross_module_interleaved_error_already_set.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/eigen_tensor_avoid_stl_array.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/env.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/extra_python_package/pytest.ini
artifacts/apa_ste1/scratch/vendor/pybind11/tests/extra_python_package/test_files.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/extra_setuptools/pytest.ini
artifacts/apa_ste1/scratch/vendor/pybind11/tests/extra_setuptools/test_setuphelper.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/local_bindings.h
artifacts/apa_ste1/scratch/vendor/pybind11/tests/object.h
artifacts/apa_ste1/scratch/vendor/pybind11/tests/pybind11_cross_module_tests.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/pybind11_tests.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/pybind11_tests.h
artifacts/apa_ste1/scratch/vendor/pybind11/tests/pytest.ini
artifacts/apa_ste1/scratch/vendor/pybind11/tests/requirements.txt
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_async.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_async.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_buffers.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_buffers.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_builtin_casters.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_builtin_casters.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_call_policies.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_call_policies.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_callbacks.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_callbacks.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_chrono.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_chrono.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_class.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_class.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_cmake_build/CMakeLists.txt
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_cmake_build/embed.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_cmake_build/installed_embed/CMakeLists.txt
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_cmake_build/installed_function/CMakeLists.txt
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_cmake_build/installed_target/CMakeLists.txt
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_cmake_build/main.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_cmake_build/subdirectory_embed/CMakeLists.txt
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_cmake_build/subdirectory_function/CMakeLists.txt
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_cmake_build/subdirectory_target/CMakeLists.txt
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_cmake_build/test.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_const_name.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_const_name.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_constants_and_functions.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_constants_and_functions.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_copy_move.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_copy_move.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_custom_type_casters.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_custom_type_casters.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_custom_type_setup.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_custom_type_setup.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_docstring_options.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_docstring_options.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_eigen_matrix.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_eigen_matrix.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_eigen_tensor.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_eigen_tensor.inl
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_eigen_tensor.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_embed/CMakeLists.txt
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_embed/catch.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_embed/external_module.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_embed/test_interpreter.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_embed/test_interpreter.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_embed/test_trampoline.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_enum.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_enum.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_eval.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_eval.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_eval_call.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_exceptions.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_exceptions.h
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_exceptions.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_factory_constructors.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_factory_constructors.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_gil_scoped.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_gil_scoped.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_iostream.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_iostream.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_kwargs_and_defaults.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_kwargs_and_defaults.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_local_bindings.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_local_bindings.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_methods_and_attributes.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_methods_and_attributes.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_modules.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_modules.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_multiple_inheritance.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_multiple_inheritance.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_numpy_array.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_numpy_array.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_numpy_dtypes.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_numpy_dtypes.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_numpy_vectorize.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_numpy_vectorize.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_opaque_types.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_opaque_types.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_operator_overloading.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_operator_overloading.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_pickling.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_pickling.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_python_multiple_inheritance.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_python_multiple_inheritance.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_pytypes.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_pytypes.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_sequences_and_iterators.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_sequences_and_iterators.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_smart_ptr.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_smart_ptr.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_stl.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_stl.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_stl_binders.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_stl_binders.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_tagbased_polymorphic.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_tagbased_polymorphic.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_thread.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_thread.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_type_caster_pyobject_ptr.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_type_caster_pyobject_ptr.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_union.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_union.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_unnamed_namespace_a.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_unnamed_namespace_a.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_unnamed_namespace_b.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_unnamed_namespace_b.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_vector_unique_ptr_member.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_vector_unique_ptr_member.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_virtual_functions.cpp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/test_virtual_functions.py
artifacts/apa_ste1/scratch/vendor/pybind11/tests/valgrind-numpy-scipy.supp
artifacts/apa_ste1/scratch/vendor/pybind11/tests/valgrind-python.supp
artifacts/apa_ste1/scratch/vendor/pybind11/tools/FindCatch.cmake
artifacts/apa_ste1/scratch/vendor/pybind11/tools/FindEigen3.cmake
artifacts/apa_ste1/scratch/vendor/pybind11/tools/FindPythonLibsNew.cmake
artifacts/apa_ste1/scratch/vendor/pybind11/tools/JoinPaths.cmake
artifacts/apa_ste1/scratch/vendor/pybind11/tools/check-style.sh
artifacts/apa_ste1/scratch/vendor/pybind11/tools/cmake_uninstall.cmake.in
artifacts/apa_ste1/scratch/vendor/pybind11/tools/codespell_ignore_lines_from_errors.py
artifacts/apa_ste1/scratch/vendor/pybind11/tools/libsize.py
artifacts/apa_ste1/scratch/vendor/pybind11/tools/make_changelog.py
artifacts/apa_ste1/scratch/vendor/pybind11/tools/pybind11.pc.in
artifacts/apa_ste1/scratch/vendor/pybind11/tools/pybind11Common.cmake
artifacts/apa_ste1/scratch/vendor/pybind11/tools/pybind11Config.cmake.in
artifacts/apa_ste1/scratch/vendor/pybind11/tools/pybind11NewTools.cmake
artifacts/apa_ste1/scratch/vendor/pybind11/tools/pybind11Tools.cmake
artifacts/apa_ste1/scratch/vendor/pybind11/tools/pyproject.toml
artifacts/apa_ste1/scratch/vendor/pybind11/tools/setup_global.py.in
artifacts/apa_ste1/scratch/vendor/pybind11/tools/setup_main.py.in
artifacts/apa_ste1/stage_engine.py
artifacts/apa_ste1/tests/test_reference.py
artifacts/apa_ste1/tests/test_source.py
artifacts/apa_ste1/tests/test_staged_artifacts.py
artifacts/apa_ste1/tests/test_switch.py
tensor_cuda/include/tc/apa_dk_ste.h
tensor_cuda/include/tc/deterministic_gather.h
tensor_cuda/src/attention_tf32.cu
tensor_cuda/src/bindings.cpp
tensor_cuda/src/embed_deterministic.cu
tensor_cuda/src/gather_deterministic.cu
tensor_cuda/src/kernels.cu
tensor_cuda/src/ops.cpp
tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
```
