# BP-KERNEL-1 — micro-census of `apa_selective_bwd_kernel`: do the atomics own it? (2026-09-13)

Seat: Codex Astra (`gpt-6-astra`, reasoning high). Lead: Fable 5.1 (plans, runs the GPU cell, verifies, commits).
David's order (2026-09-13 03:20 EDT): "run the kernel micro-census, thats the obvious next one."
Parent result: `/mnt/Shared/BP_Census_1_Result_2026-09-13.md` — the real GRAPA step is 11.8 s, of which the native APA backward
kernel is 10.0 s (85 %); the whole-model forward is 0.45 s. The successor registered there is this order; its prediction and
falsifier are copied below and are immutable.

## Writable target (explicit grant)
YOUR WRITABLE TARGET is `/mnt/ForgeRealm/wt/pt-bk1` (Project-Tensor worktree, branch `bp-kernel-1`, forked from `main`
754dc1c — NOT from the timing-hook branch). Edits AUTHORIZED: new kernel VARIANTS in `tensor_cuda/src/kernels.cu` +
`ops.cpp` + `bindings.cpp` behind an explicit variant argument (the default path — variant `a` — must stay byte-identical in
source and behaviour), new `scripts/bp_kernel_1.py`, `tests/test_bp_kernel_1.py`, `artifacts/bp_kernel_1/`,
`docs/BP_KERNEL_1_LEDGER.md`. CPU build AUTHORIZED (`cmake -S tensor_cuda -B tensor_cuda/build-bk1 -DCMAKE_BUILD_TYPE=Release
-DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc
-DFETCHCONTENT_SOURCE_DIR_PYBIND11=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src
-DFETCHCONTENT_FULLY_DISCONNECTED=ON && cmake --build tensor_cuda/build-bk1 -j4` — the last full build took 46 s).
Read-only: everything else, including `/mnt/ForgeRealm/wt/grapa-bp1` (census receipts) and `/mnt/ForgeRealm/GRAPA-Native-LLM`.
No git (lead commits). No subagents. **No GPU in this seat** (sandbox has no device; the lead runs the cell). Foreground only,
every Bash call < 10 min. Never kill or signal a process you did not start. No network.

## The kernel
`tensor_cuda/src/kernels.cu:2524` `apa_selective_bwd_kernel<T,DMAX>`: one block per query row; threads stride over keys;
per key: scalar FP32 dot loops over D against global K/KQ/V; two passes (A: row-dot; B: gradients); dK/dV written with
per-element `atomicAdd` in T (BF16) from every query row; dQ reduced in shared memory. Launched at :2696; Python binding
`apa_selective_bwd` (bindings.cpp:835); forward train kernel :2344 (no atomics). Read BP-SCOUT-1 §"What native APA actually
computes" and idea #1 (`/mnt/ForgeRealm/Project-Tensor/artifacts/bp_scout_1/REPORT.md`) before touching anything.

## Variants (each an explicit, separately selectable kernel; never the default)
- **a** — the kernel as is (control; the default path, untouched).
- **b** — dK/dV accumulated into **FP32 scratch buffers** with FP32 `atomicAdd`, cast to T once at the end. Same math,
  different rounding order — a VALID gradient candidate.
- **c** — **atomics removed**: dK/dV not written (dQ still computed). TIMING ONLY — not a gradient. Its purpose is to bound
  what the atomics cost.
- **d** — the **output-dot identity** (scout idea #1: rowdot = dO · O, with O saved or recomputed once) replacing Pass A's
  per-key recomputation, applied on top of **b**. VALID gradient candidate.
- Optional **e** (only if a–d land with budget to spare): key-tiled Pass B with K/V staged through shared memory, on top of d.

## Shapes and inputs (registered)
The census shapes: read `effective_config` in `/mnt/ForgeRealm/wt/grapa-bp1/artifacts/bp_census_1/receipt.json` and the
attention geometry the checkpointed model actually uses (`grapa/model_mla.py` + `attention_mla.py` — H, KVH, D, VD, causal,
refine 0.15 → the `thr` rule); record them in the registration. B=1, L=S=2048, BF16, seeded inputs
(`artifacts/bp_kernel_1/inputs.npz`, sha256 pinned): q, k, kq, v, dO, plus lse/thr produced by the real forward train kernel.

## Gates and measurement (registered; do not re-tune)
1. **Correctness gate first, timing second.** For b and d (and e): dQ, dK, dV vs variant a on the same inputs —
   max-abs and relative error, registered tolerance for BF16 atomics' order nondeterminism: run a TWICE and use its own
   run-to-run spread ×2 as the tolerance floor; a candidate exceeding that on any of dQ/dK/dV is **RED, its timing does not
   count**. Also assert a's own run-to-run spread is finite and reported.
2. Timing: CUDA events around the bare op call (`tc._C.apa_selective_bwd` or the variant binding), device-synchronised,
   3 warm-ups + 10 measured launches per variant, interleaved order a b c d (a b c d …) to cancel drift; report mean/min ms.
3. **Prediction (immutable):** time(a) − time(c) ≥ 50 % of time(a) — the atomics own the kernel.
   **Falsifier:** < 50 % promotes the scalar dot loops (tiling / tensor cores, variant e) as the first target.
   Secondary, registered: d ≤ 0.6 × a if the prediction holds and d is green.
4. Budget: ONE GPU cell ≤ 300 s work, ≤ 590 s lease (`flock /tmp/forge-gpu.lock`, the `gpu_lease` convention in
   `GraftRepository/scripts/grm_cmc1_gpu_arms.py`), single card, foreground, self-terminating. Incomplete → INCONCLUSIVE, no
   shortening of shapes or samples. Create-only receipt `artifacts/bp_kernel_1/receipt.json`; a second cell needs a lead order.

## Deliverables
`scripts/bp_kernel_1.py` (`--dry-run` = CPU path through the same code with tiny shapes, produces the same receipt schema;
`--run` = the cell), `artifacts/bp_kernel_1/registration.json` (pins: kernels.cu/ops.cpp/bindings.cpp before+after sha256,
inputs, shapes, tolerance rule, prediction/falsifier verbatim, budget; written BEFORE the run; harness fails closed on drift),
`tests/test_bp_kernel_1.py` (CPU: registration immutability, schema, dry-run receipt, tolerance rule on fabricated spreads,
verdict logic on fabricated times, create-only, variant `a` source region unchanged vs main — pin the byte range),
`artifacts/bp_kernel_1/lead_commands.txt` (one command per line, each accepting an appended `--dry-run`),
`artifacts/bp_kernel_1/engine_build_receipt.json` (your CPU build: commands, rc, .so sha256), `docs/BP_KERNEL_1_LEDGER.md`.

## Done — paste verbatim in your final message
1. What each variant changes, with code sites (line ranges) and the diff stat; confirmation variant a's kernel body is
   byte-identical to main (the pinned range and its sha256).
2. Last line of `python3 -m pytest -q tests/test_bp_kernel_1.py`, of the build, and of `--dry-run` (ends in a receipt path).
3. Registration sha256; the shapes registered; the tolerance rule as implemented.
4. Prior art (FlashAttention backward structure / Dao 2022; atomics-free dK/dV via key-parallel passes; output-dot identity —
   name what is taken vs ours, or "unverified — lead to check").
5. Model + effort; confirmation: no GPU, no git, no subagents, nothing killed, no network, no edits outside the grants.
