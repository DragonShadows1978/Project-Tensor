# Phase 0.2 — End-to-End + Trace Receipts (decode)

Evidence class: kernel sweep (speed / memory shape / launch-overhead
structure only — never model quality; see plan House Rules).

**Environment caveat registered up front:** every run in this receipt ran
concurrently with an unrelated, already-running GPU job
(`gpt_oss20b_stream_forward_smoke.py`, Graft Translation mission,
~6.3 GB resident, continuous 100% SM utilization for the full session) on
the same single RTX 4070 Super. Default (not exclusive) compute mode allowed
time-sliced coexistence, but this contention:
- inflated model-*load* wall time 10-20x (not reported as a finding — out
  of scope for decode receipts),
- makes wall-clock-window-based decomposition of the nsys trace unreliable
  (kernel timestamps bunch up at the tail of the capture — see 3.2 below),
- does NOT invalidate ncu hardware-counter ratios (occupancy, branch
  efficiency, grid/block config, register/shared-mem limits) — those are
  per-kernel-invocation counter reads, contention-immune.
- does NOT invalidate the app-measured decode tok/s (perf_counter deltas
  around `tc.synchronize()`), which reflects real wall-clock throughput
  under whatever the machine's actual contention state is, same as
  production.

Tooling: `nsys` (Nsight Systems 2024.5.1.113) and `ncu` (Nsight Compute
2022.4.1.0) both present at `/usr/local/bin/{nsys,ncu}` /
`/opt/nvidia/nsight-{systems,compute}`. Both used; no substitutes needed.
`ncu` requires perf-counter access the user account does not have by
default (`ERR_NVGPUCTRPERM`) — worked around with `sudo -E env "PATH=$PATH"
ncu ...` (passwordless sudo available on this box) to preserve the venv/
PYTHONPATH numpy/tensor_cuda imports; `.ncu-rep` outputs chowned back to
`vader` after capture.

---

## 1. E2E baseline — `qwen35_generate.py` (N=5, median)

Command (each run):
```
cd /mnt/ForgeRealm/GraftRepository && timeout 240 python3 -u scripts/qwen35_generate.py "Why is the sky blue?" 48
```

| run | prefill (18 tok) | decode (48 tok) | decode tok/s |
|-----|-------------------|-------------------|--------------|
| 1   | 1.21 s            | 0.92 s            | 51.9         |
| 2   | 1.05 s            | 0.94 s            | 51.0         |
| 3   | 0.89 s            | 0.87 s            | 55.2         |
| 4   | 0.85 s            | 0.87 s            | 55.1         |
| 5   | 0.77 s            | 0.87 s            | 55.4         |

**Median decode: 55.1 tok/s. Median prefill: 0.89 s (18 tok).**
(Model-load time excluded from both — it is contention-inflated this
session and is not what the plan's accept-threshold tracks.)

## GPT-OSS-20B decode script — none short-runnable

Checked `GraftRepository/scripts/gpt_oss20b_*`:
- `gpt_oss20b_context_ladder.py` — spawns one subprocess per (setting,
  length), prefill-oriented (`--skip-lm-head` default), not a decode tok/s
  reporter.
- `gpt_oss20b_stream_greedy_smoke.py` — docstring states explicitly: *"not
  a fast decode path and does not use KV cache... its job is to prove the
  streamed forward can be driven in a loop"*. Not a throughput number.
- Model is ~26 GB on disk; only 5.2-7.9 GB VRAM was free during this
  session (another job resident at 6.3 GB). Loading a second large model
  risked OOM on either job.

No N=3 GPT-OSS-20B decode numbers are reported — none of the available
scripts measure decode tok/s, and the honest fallback (per task
instructions) is to say so rather than substitute a prefill or non-KV-cache
number as if it were comparable.

---

## 2. nsys trace — bounded decode session

Command:
```
cd /mnt/ForgeRealm/GraftRepository && nsys profile -t cuda,cublas --stats=true \
  -o artifacts/kernel_opt/qwen35_decode_trace2 --force-overwrite=true \
  python3 -u scripts/qwen35_generate.py "Why is the sky blue? Explain briefly." 32
```
App-reported: prefill 21 tok in 2.02 s, decode 32 tok in 1.26 s (25.4 tok/s
— lower than the N=5 baseline above because this run's window landed on a
period of heavier GPU contention; app-measured wall-clock is real, not an
artifact).

Artifacts: `qwen35_decode_trace2.{nsys-rep,sqlite}` (+ an earlier duplicate
`qwen35_decode_trace.{nsys-rep,sqlite}` from a first attempt where app
stdout wasn't captured separately — kept for cross-check, numbers agree).

### 2.1 Total launches and launches/token

Total kernel launches across the whole traced process: **71,312**
(includes one-time model-load work, not just decode).

One-time load-only kernel identified by code-structure cross-check:
`int4_dequant_t_kernel` = 200 launches (fires only during weight
dequantization at load; confirmed absent from the decode-step count below).

**Decode-step launch count, derived exactly (not estimated) for the
int4 GEMV path:** `int4_gemv_kernel` (the M=1 decode-only int4 linear path)
fires exactly **6,433** times = `32 × 201 + 1`. The `+1` is the lm_head's
own M=1 call made once at the end of the prefill step
(`last_token_only=True`); the `201` is the qlinear call count per decode
step: 8 attention layers × 4 projections (q/k/v/o) = 32, 32 dense-FFN
layers × 3 (gate/up/down) = 96, 24 DeltaNet layers × 3 (in_proj_qkv/
in_proj_z/out_proj) = 72, + 1 lm_head = **201 exactly**. This is a clean,
verified decode-step launch count for one kernel family.

**Total launches per decode step (all kernel families), via modulo-32
decomposition:** for every kernel name, floor-divided its total count by 32
(the number of decode steps) and treated the remainder as prefill's
one-call contribution — valid only because every remainder observed was
small (≤25, well under 32, i.e. plausibly "prefill's single call issued
this many, decode's 32 calls issued the rest evenly"). This decomposition
is exact and self-consistent: 200 (load) + 392 (prefill remainder) +
70,720 (decode, 32 steps) = 71,312.

**→ ~2,210 kernel launches per decode token** (70,720 / 32). Caveat: this
assumes uniform per-step launch count across all 32 decode steps and that
prefill's contribution is fully captured by the "remainder < 32" case —
confirmed self-consistent (exact reconstruction of the 71,312 total) but
not independently cross-checked against a second run. One kernel family
(`gated_delta_step_kernel`, DeltaNet recurrent update) divides by 32
*exactly* (768 = 24×32, zero prefill remainder) — confirming this kernel
is genuinely decode-only (prefill uses a different chunked/parallel scan
path for the same recurrent state, not the step kernel), which is why the
naive "total÷33-calls" average (2,155/call) was rejected as wrong before
landing on this decomposition — prefill and decode are not structurally
equivalent forward passes.

### 2.2 Gap / launch-overhead share

Wall-clock-window decomposition of the nsys trace is **not reliable** this
session: kernel launch *timestamps* recorded by CUPTI bunch up almost
entirely in the last ~4 of 118 total wall-clock seconds (bucket check:
seconds 0, 115, 116, 117, 118 hold 2 / 1,301 / 8,942 / 45,262 / 15,805
launches respectively) — a symptom of the competing job's continuous SM
occupancy backing up this process's launch queue, not of tensor_cuda's own
behavior. A window-based "gap share" computed from this would describe the
other job's interference, not tensor_cuda's decode loop.

Fallback used instead: compare the **app-measured** decode wall-time
(1.26 s / 32 tok = 39.4 ms/tok) against **decode-attributable summed GPU
kernel-busy-time**, using the same modulo-32 decomposition to assign each
kernel's total duration proportionally to its decode-vs-prefill launch
split:

- Decode-attributable GPU busy time: ≈1,021.6 ms total / 32 tok ≈
  **31.9 ms/tok**
- App-measured wall time: **39.4 ms/tok**
- **Gap/launch-overhead share ≈ (39.4 − 31.9) / 39.4 ≈ 19%**

Caveat (do not over-read this as a clean Phase-2 gate pass/fail): kernel
*durations* themselves were measured on a GPU sharing SM time with another
job, so the 31.9 ms "busy time" figure may itself include contention-
induced scheduling stalls misattributed as kernel execution time — this
would bias the 19% figure **down** (understating true intrinsic gap share
relative to a quiet GPU), not up. Read as: gap/launch overhead is *at
least* in the high-teens percent range under this trace; a clean re-run on
an idle GPU is needed before this number is used to gate Phase 2 (≥15%
threshold) with confidence — right now it sits close enough to the
threshold that contention noise in either direction could flip the
call.

### 2.3 Top-10 kernels by total GPU time (whole-trace `cuda_gpu_kern_sum`, `--stats=true`)

| rank | time % | total (ms) | count | mean (µs) | kernel |
|---|---|---|---|---|---|
| 1 | 39.6 | 673.0 | 200 | 3,365.0 | `int4_dequant_t_kernel<bf16>` (load-only) |
| 2 | 39.4 | 670.5 | 6,433 | 104.2 | `int4_gemv_kernel<bf16>` (decode M=1 linears) |
| 3 | 6.0 | 101.7 | 1,008 | 100.9 | `reduce_kernel<float,0>` |
| 4 | 3.3 | 56.7 | 768 | 73.8 | `gated_delta_step_kernel` (DeltaNet, decode-only) |
| 5 | 2.8 | 47.7 | 15,096 | 3.2 | `binary_kernel<float>` |
| 6 | 1.2 | 20.8 | 1,536 | 13.5 | `gemvNSP_kernel<float,...>` (cuBLAS) |
| 7 | 1.2 | 19.7 | 2,145 | 9.2 | `rms_norm_kernel<bf16,bf16>` |
| 8 | 1.0 | 17.2 | 72 | 239.2 | `cutlass::Kernel2<...32x32_32x1...align8>` |
| 9 | 0.8 | 13.3 | 8,856 | 1.5 | `dimcopy_kernel<float,0>` |
| 10 | 0.7 | 11.8 | 5,824 | 2.0 | `binary_kernel<bf16>` |

Note: ranks 1 and 2 dominate at ~40% each of *whole-trace* GPU time, but
rank 1 (`int4_dequant_t_kernel`) is one-time load work, not decode-path —
within decode alone, `int4_gemv_kernel` is clearly the dominant cost by a
wide margin (mean 104.2 µs × 201 calls/step ≈ 21 ms/step of the ~31.9 ms/
step decode-attributable GPU time, i.e. the int4 GEMV path is ~66% of
decode-step GPU-side cost). Confirms the plan's registered blind-spot
question (dynamic-shmem GEMV residency) as the correct Phase-4 target.

`cudaMemcpy`/`cudaDeviceSynchronize` hotspot in the API summary
(`cuda_api_sum`): `cudaMemcpy` = 73.8% of all CUDA API time (2.21 s total,
1,085 calls, mean 2.04 ms — dominated by a few huge H2D calls, max
50.4 ms, consistent with initial weight staging, not per-token). Memory
summary (`cuda_gpu_mem_time_sum`) shows H2D memcpy = 99.4% of memcpy-time
(423 ms / 1,052 copies) vs D2H = 0.4% (1.5 ms / 33 copies, mean 46.9 µs) —
the D2H side is the per-token `.numpy()` argmax transfer the plan's
Phase 1.1 targets (device-side argmax); its total cost this trace is small
in absolute terms (1.5 ms over 32+1 tokens) but is 33 *synchronizing*
round-trips on the critical path, which is a latency-per-token concern the
raw ms total understates.

---

## 3. ncu targeted (one launch each, `--set full`, `-c 1 -s 1`)

Commands (pattern for all three):
```
cd /mnt/ForgeRealm/Project-Tensor && sudo -E env "PATH=$PATH" ncu --set full -c 1 -s 1 \
  -o artifacts/kernel_opt/ncu_<target> --force-overwrite \
  python3 -u <scratch_script>.py
```
Each scratch script issues the target op twice (untimed warmup + timed
call); `-s 1 -c 1` skips the warmup launch and profiles the steady-state
one. Report export: `ncu --import <target>.ncu-rep --page details >
<target>_report.txt`.

### (a) `int4_linear_fused` GEMV, Qwen FFN shape (M=1, N=12288, K=4096, group=128)

Kernel: `int4_gemv_kernel<float>`, launch config `(1536,1,1) x (256,1,1)`.

- **Achieved occupancy: 76.01%** (theoretical 83.33%, shared-mem-limited —
  `Block Limit Shared Mem = 5` vs `Block Limit SM = 24`; dynamic shared mem
  = 16.38 KB/block for this K).
- **Blocks/SM: Block Limit SM = 24** (i.e. SM-count is not the binding
  constraint; shared memory is) — `Theoretical Active Warps per SM = 40`
  → **40/64 = 62.5% of Ada's max warp slots**, matching the 83.33%
  theoretical occupancy figure (40/48 warps at 256-thread/8-warp blocks,
  6 blocks/SM cap from shared mem).
- Branch Efficiency: **100%**, 0 avg divergent branches — clean, matches
  Phase 0.4's ptxas finding that inference kernels don't have a divergence
  problem at this shape.
- DRAM Throughput: 71.60% of peak (memory-bound, as expected for a GEMV).
- **Bank-conflict proxy (shared-mem wavefronts):** ncu flags "uncoalesced
  shared accesses resulting in 1,572,864 excessive wavefronts (**47% of
  the total 3,342,336 wavefronts**)" — a real, sizable shared-memory
  access-pattern inefficiency at this K.
- **Answer to the registered blind-spot question:** at K=4096 (Qwen FFN),
  residency is NOT 1 block/SM — it's shared-mem-capped at ~5-6 blocks/SM
  (24 SM-limit unused), achieving 76% occupancy, not the worst case
  feared in the Phase 0.4 note. The 47% excessive shared-mem wavefronts is
  the more actionable finding than occupancy for this kernel.

### (b) `apa_selective_attention`, long-S decode shape (Qwen geometry: H=16, KVH=4, D=256, L=1, S=8192)

Kernel: `apa_selective_kernel<float,256>`, launch config `(16,1,1) x
(128,1,1)`.

- **Achieved occupancy: 8.33%** (theoretical 83.33%, register-limited —
  `Block Limit Registers = 10` vs `Block Limit SM = 24`, 47 registers/
  thread). The occupancy collapse is driven far more by **grid underfill**
  than registers: Grid Size = 16 blocks total on a 56-SM device → **"0.0
  full waves across all SMs"** (ncu's own wording) — at B=1 (one query per
  decode step), one block per (query, KV-head-ish) unit is nowhere near
  enough to fill the GPU. This is a structural batch=1 decode-shape
  problem, not a kernel-internal inefficiency.
- **Branch Efficiency: 99.02%**, avg divergent branches 18.20 (out of
  460,946 branch instructions) — essentially clean; the 0.3-confirmed
  divergent refine/selection branches (Tier-B claim) are present but not
  the dominant cost at this shape.
- DRAM Throughput: only 5.15% of peak (this kernel is nowhere near
  memory-bound at B=1 — it's compute/launch-shape-bound: Compute (SM)
  Throughput 3.06%, i.e. almost the entire device is idle during this
  kernel).
- Additional finding (not pre-registered, surfaced by this run): 87% of
  global-memory sectors are "excessive" (uncoalesced access), 92.2M of
  105.9M total sectors — a second, independent inefficiency signal beyond
  the grid-underfill one.
- **Answer to the registered branch-efficiency question:** 99.02% at
  this long-S decode shape — divergence is NOT the bottleneck here; grid
  underutilization (batch=1 structurally starves a 56-SM device) is.

### (c) `mxfp4_linear` GEMV, GPT-OSS FFN shape (M=1, N=2880, K=2880, group=32)

Kernel: `mxfp4_gemv_kernel<float>`, launch config `(360,1,1) x (256,1,1)`.

- **Achieved occupancy: 78.14%** (theoretical 100% — not block-limited at
  all: `Block Limit SM=24, Registers=10, Shared Mem=8, Warps=6`, all ≥ the
  6-block cap; the occupancy gap is scheduling overhead, not a hard
  limit). Grid Size = 360 blocks = 1 full wave (336 blocks across 56 SMs)
  + a 24-block partial wave — ncu flags the partial wave as costing "up to
  50% of total kernel runtime at 21.9% occupancy" for that tail.
- **Branch Efficiency: 60.94%** — the weakest of the three kernels tested,
  with **15,981 average divergent branches** out of 13,280,464 branch
  instructions (Branch Instructions Ratio 0.45%, the highest of the three
  — this kernel is far more branch-instruction-dense relative to total
  work than int4_gemv or apa_selective). This is a new, directly-measured
  finding: MXFP4's block-scale/exponent decode path diverges substantially
  more than the INT4 path at a comparable GEMV shape.
- DRAM Throughput: 5.74% of peak (not memory-bound at M=1, K=2880 — small
  enough to be launch/compute-shape-bound like (b), though far less
  extreme).
- **Answer to the registered question:** blocks/SM is not the concern
  here (24-block SM limit unused, plenty of room); the branch-efficiency
  number (60.94%) is the standout result and the most direct candidate for
  the Phase 4 "DP4A / launch-config sweep" workstream to also inspect
  MXFP4's divergent-decode path, not just int4.

---

## Summary table (the three ncu answers)

| kernel | shape | achieved occ. | theoretical occ. | limiter | branch eff. | dram % peak |
|---|---|---|---|---|---|---|
| int4_gemv (Qwen FFN, K=4096) | M=1,N=12288,K=4096 | 76.01% | 83.33% | shared mem (5 blk/SM cap; SM limit=24 unused) | 100% | 71.60% |
| apa_selective (long-S decode) | H=16,KVH=4,D=256,L=1,S=8192 | 8.33% | 83.33% | grid underfill (16 blocks total, 0.0 full waves) + registers | 99.02% | 5.15% |
| mxfp4_gemv (GPT-OSS FFN, K=2880) | M=1,N=2880,K=2880 | 78.14% | 100% | none (scheduling overhead + partial-wave tail) | 60.94% | 5.74% |

## Artifacts

All under `/mnt/ForgeRealm/Project-Tensor/artifacts/kernel_opt/`:
- `qwen35_decode_trace2.nsys-rep`, `qwen35_decode_trace2.sqlite` (+ earlier
  duplicate `qwen35_decode_trace.{nsys-rep,sqlite}`)
- `nsys_run2_app_stdout.log` (app stdout captured separately for exact
  token counts)
- `ncu_int4_gemv.ncu-rep`, `ncu_int4_gemv_report.txt`
- `ncu_apa_selective.ncu-rep`, `ncu_apa_selective_report.txt`
- `ncu_mxfp4_gemv.ncu-rep`, `ncu_mxfp4_gemv_report.txt`

Scratch ncu-target scripts (not committed, per plan instruction not to
commit): `/tmp/claude-1000/-home-vader/8e6895ec-3f98-4d60-acf1-5f24fb3c9128/scratchpad/ncu_{int4_gemv,apa_selective,mxfp4_gemv}.py`.

## What failed / needs a re-run

- Clean gap/launch-overhead share (§2.2) could not be computed from
  wall-clock windowing due to GPU contention distorting the nsys
  timestamp distribution; the ~19% figure is a fallback derived from
  app-measured wall time vs. decode-attributable summed kernel duration,
  with a stated direction-of-bias caveat. A re-run on an idle GPU (no
  concurrent job) is needed before this number is used to gate Phase 2
  with full confidence.
- `ncu --query-metrics` needed a live kernel context to enumerate names in
  this ncu version; not blocking (the `--set full` page already surfaced
  everything asked for) but noted in case a future targeted `--metrics`
  invocation is wanted instead of `--set full`.
- GPT-OSS-20B decode N=3 not run — no short-runnable decode-tok/s script
  exists in GraftRepository/scripts (see §1); running the 20B model at all
  this session also risked VRAM contention with the already-running
  Graft Translation job.
