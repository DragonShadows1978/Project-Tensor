# PT-DET-1 — a deterministic embedding backward and a run-to-run reproducibility gate for the training step

Registered 2026-09-26 07:45 EDT by the lead (Fable 5.1). Seat: GPT-6 Astra at max. YOUR WRITABLE TARGET is the fork
worktree `/mnt/ForgeRealm/wt/pt-tf32` (branch tf32-fast-path, HEAD = PT-TF32-4 + orders). The live engine
`/mnt/ForgeRealm/Project-Tensor` is never edited, rebuilt or imported. No GPU here (build + CPU + blocked-report).

## Evidence (read-only: `/mnt/ForgeRealm/wt/grapa-cc46/docs/CC46B_LEDGER.md`, `CC46C_LEDGER.md`, `artifacts/cc46b/`, `artifacts/cc46c/`)
Two runs of the same 30 training steps from the same checkpoint diverge in every configuration (bf16 on the live
engine included). The only gradient that differs at the FIRST step is `emb.weight`. `tensor_cuda/src/kernels.cu
embed_bwd_kernel` does one fp32 `atomicAdd` per (position, feature) into the embedding gradient row of that position's
token — the summation order across repeated tokens is whatever the hardware schedules. With bf16 downstream the
last-bit differences were usually absorbed (the v2 side runs replayed 130 steps bit-for-bit); with the v3 corpus
(many repeated delimiter tokens) and fp32 block-0 inputs they survive and grow to 1e-3 relative within ~50 steps.
`reduce_lastdim_kernel` (`.sum()`) is a fixed tree (fine). A CPU emulation of the kernel's per-token contention is in
`scripts/cc46c_reduction_probe.py` (read-only) with the registered prediction that a one-hot SGEMM embedding backward
(arm "i") is bit-reproducible.

## Build
1. A deterministic `embed_bwd` path, opt-in (`tc._C.set_deterministic_embed_bwd(bool)` + env `TC_DET_EMBED_BWD=1`;
   default OFF = today's kernel byte-for-byte). Options — pick by measurement, document the other: (a) sort positions
   by token id (one CUB/thrust radix sort of ≤ 4,096 keys, or a fixed-order host sort of the ids the trainer already
   has) then a segmented, fixed-order reduction per token row with one writer per (token, feature); (b) a one-hot
   SGEMM (V × L one-hot times dY: 8,192 × 4,096 × 1,024 — ≈ 34 GFLOP, too slow as dense; only if a sparse variant is
   cheap). Gate: bitwise-identical gradient across 5 repeated calls on the model's real token batches (take the batches
   npz from `/mnt/ForgeRealm/wt/grapa-cc46/artifacts/cc46/batches.json` / the trace dir if present, else synthetic
   batches with the v3 repeat statistics: ≥ 30 % repeated tokens), equality to the fp64 reference ≤ 1e-6 rel-L2,
   and cost ≤ 2× the atomic kernel (the embedding backward is a tiny share of the step; report ms).
2. Audit for OTHER nondeterministic sites on the training path (any `atomicAdd` in a backward that survives to the
   gradient: attention bwd variants a/f (known atomics in `a`), norms, the optimizer, `index_add`-style ops); list them
   with file:line and whether the live v3 or v2 path hits them. Fix only the embedding backward in this order; the
   list is for the next.
3. A run-to-run reproducibility GATE for the certification harness: `scripts/pt_det_1.py repro --steps 30` runs the
   same 30 training steps twice under the lock (the CC46 probe's mechanics: same checkpoint, same argv, diff the
   logged loss/gnorm text and the saved weights bitwise) for (i) the live v3 argv with the deterministic embed bwd on,
   (ii) bf16 with it on, (iii) each with it off (controls, expected to differ) — registered predictions before the run.
   Register it as a required lane in the slot runner (any future engine certification must pass it).
4. Rebuild in-tree; `docs/PT_DET_1_LEDGER.md`; CPU tests (the sort/segmented path on CPU vs a reference; the gate's
   diff logic); `## Done` with receipts verbatim, the blocked-report, the lead's slot sequence (≤ 25 min). Prior art
   (deterministic scatter-add via sort + segmented reduce: CUB/NVIDIA; PyTorch's `index_add_` deterministic mode;
   Demmel & Nguyen reproducible summation) at the code site + ledger + report.

## Boundaries (binding)
- Writable: `/mnt/ForgeRealm/wt/pt-tf32` only. Never `/mnt/ForgeRealm/Project-Tensor`, `/mnt/ForgeRealm/GRAPA-Native-LLM`,
  `/mnt/ForgeRealm/wt/grapa-*`, `/mnt/ForgeRealm/grapa_run*/`, `/mnt/ForgeRealm/grapa_run7_trace/`, engines, snapshots.
  No GPU (`CUDA_VISIBLE_DEVICES=''`; run 7 is live). No git. No subagents. Never kill processes you did not start;
  never touch `/tmp/forge-gpu.lock`. RED honesty.
