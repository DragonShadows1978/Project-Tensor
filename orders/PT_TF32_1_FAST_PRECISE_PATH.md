# PT-TF32-1 — a fast PRECISE training path: TF32 tensor-core GEMMs and TF32-input fast attention kernels for fp32 blocks

Registered 2026-09-25 14:00 EDT by the lead (Fable 5.1). Seat: GPT-6 Astra at max reasoning. YOUR WRITABLE TARGET is
this fork worktree `/mnt/ForgeRealm/wt/pt-tf32` (branch `tf32-fast-path` from main d62c170) — edits and builds
AUTHORIZED here. The live engine `/mnt/ForgeRealm/Project-Tensor/tensor_cuda` (loaded by a live training run) must not be
edited, rebuilt or imported by anything you run. You have NO GPU: build + CPU/unit work + a blocked-report; the lead
runs every GPU gate.

## Why (evidence in the GRAPA repo, read-only: `/mnt/ForgeRealm/wt/grapa-cc41/docs/CC39B_LEDGER.md`, `CC41_LEDGER.md`,
`artifacts/cc41/`; the board `/mnt/ForgeRealm/AI_Research_Board.md`, entries 2026-09-25 10:50 → 13:45)
Training GRAPA-MLA (243M, W4096, batch 1, bf16 compute) spikes when a massive-activation delimiter token's forward-pass
error is amplified ≈ 4×/block by the SwiGLU FFNs of blocks 9–13; the bf16 gradient at such a state is chaotic (five
roundings, five answers) and the update destroys layers 0–5. fp64 = fp32 there; the backward in bf16 is harmless. The
validated remedy is fp32 FORWARD in the lower blocks (0–10 passes: cos +1.0000 vs fp64 at the exact onset state; a
sweep of smaller sets is running). Cost today: 5.09 → 13.07 s/step, because the fast attention pair is BF16-ONLY
(`tensor_cuda/src/kernels.cu:3184` "BP-KERNEL-3 requires BF16 and D/VD <= 128"; `:8163` "BP-KERNEL-4: same CUDA BF16
required") so fp32 blocks fall back to the scalar/key-parallel kernels, and fp32 GEMMs are plain SGEMM (TF32 is never
enabled; `kernels.cu:6295` notes the intent). The training run's duty wall is strict (David): a step must stay near
5 s. Target: fp32-block forward+backward at ≤ ~6.5 s/step for an 11-block set (≤ 1.3× the bf16 step).

## Build (all opt-in; every default unchanged; the bf16 paths byte-identical)
1. **TF32 GEMMs for fp32 tensors:** an engine switch (`tc._C.set_tf32_gemm(bool)` + env `TC_TF32_GEMM=1`) that routes
   fp32 matmul/linear (cuBLAS/cublasLt calls used by the training path — find every site) through TF32 tensor-core math
   (`CUBLAS_TF32_TENSOR_OP_MATH` / `CUBLAS_COMPUTE_32F_FAST_TF32`), fp32 accumulate. Gate: vs FP64 reference on the
   model's GEMM shapes (d_model 1024, FFN 4096-ish, MLA latent 256/768, vocab 8192 × W4096) — TF32 error ≲ 1e-3 rel,
   ≥ 5× faster than SGEMM at those shapes (micro-benchmark script for the lead).
2. **TF32-input fast attention kernels:** `h_tf32` (forward, from BP-KERNEL-4 `h`) and `g1_tf32` (backward, from
   BP-KERNEL-3 `g1`): accept fp32 Q/K/V/dO (and the fp32 saved forward), use WMMA `precision::tf32` fragments
   (m16n16k8) with fp32 accumulation for the tiled bulk scores, dO·Vᵀ and the gradient outer products, keep p and dS in
   fp32 (do NOT round them to bf16 — that rounding was g1's choice for the bf16 path), exact fp32 scores for selected
   pairs on the masked scalar path as in g1, same selection rule, same contracts (D/VD ≤ 128, refine fraction, rope key
   width 64 split as in the model), same `bp_kernel_2_set_variant('g1_tf32')` / `bp_kernel_4_set_variant('h_tf32')`
   plumbing. Gates as BP-KERNEL-2/3/4: rel-L2 dK/dQ/dV and forward output vs an FP64 reference at the model's shapes
   (H 16, L = S = 2048 and 4096, D 96/64 split, VD 64), the 2×-spread rule vs the shipped `a` kernel's own distance, plus
   the existing kernel unit tests; micro-benchmark vs `g1`/`h` (bf16) and vs `f`/`a` (fp32). Target ≤ 2× the bf16 kernel
   time.
3. **Optional if time allows:** a forward-fp32/backward-bf16 split at the block boundary (the backward is harmless in
   bf16 per CC39-B) — only if (1)+(2) do not reach the step target; describe the design either way.
4. Build the fork engine (`tensor_cuda/build.sh` / CMake in THIS worktree; its own `.so` — never the live one). Deliver:
   `docs/PT_TF32_1_LEDGER.md` (design, taken vs yours, gates the lead must run with exact commands, expected numbers),
   the gate/bench scripts, unit tests that run on CPU where possible, `## Done` with receipts verbatim (build log lines,
   test counts, file SHAs, the blocked-report for every GPU gate you could not run), and a "GRAPA integration" note:
   what `grapa/fwd_precision.py` must set to use the TF32 variants for fp32 blocks.

## Prior art (annotate at code site + ledger + report)
TF32 (NVIDIA Ampere 2020), WMMA tf32 fragments (CUDA 11), cuBLAS TF32 modes, FlashAttention-2 (Dao 2023) for the tiled
skeleton, the BP-KERNEL-2/3/4 ledgers in `docs/` for what g1/h took. Say what is taken and what is yours.

## Boundaries (binding)
- Writable: `/mnt/ForgeRealm/wt/pt-tf32` only. Never `/mnt/ForgeRealm/Project-Tensor` (the live engine), never `/mnt/ForgeRealm/GRAPA-Native-LLM`,
  `/mnt/ForgeRealm/wt/grapa-*` (read-only evidence), `/mnt/ForgeRealm/grapa_run*/`, `/mnt/ForgeRealm/grapa_side_*`. No GPU
  (`CUDA_VISIBLE_DEVICES=''`; a live run holds the GPU). No git (the lead commits). No subagents. Never kill processes you
  did not start; never touch `/tmp/forge-gpu.lock`. RED honesty: unbuilt/unproven = say so.
