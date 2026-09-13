# BP-KERNEL-3 — tile the atomics-free APA backward and put its dot products on tensor cores (2026-09-13)

Seat: Codex Astra (`gpt-6-astra`, reasoning high). Lead: Fable 5.1 (plans, runs the GPU cells, verifies, commits).
David's order (2026-09-13 ~12:45 EDT): "run BP-KERNEL-3, thats the obvious next one."
Parents: `/mnt/Shared/BP_Kernel_2_Result_2026-09-13.md` (variant f: query-owned dQ kernel + key-owned dK/dV kernel, no atomics,
130.6 ms at the model's shapes, GREEN against the FP64 reference; real step 11.6 → 5.0 s; real train.py at window 12288
328 → 148.5 s/step). What remains inside f is **scalar FP32 dot loops over D reading K/KQ/V from global memory, per (query,key)
pair, in both kernels**. This order registers the tiling step. Predictions below are immutable.

## Writable target (explicit grant)
YOUR WRITABLE TARGET is `/mnt/ForgeRealm/wt/pt-bk3` (Project-Tensor worktree, branch `bp-kernel-3`, forked from `main` 2a0ea28 —
variants a/b/c/d/f, `scripts/bp_kernel_2.py`, `scripts/bp_census_2.py`, tests and receipts are on main). Edits AUTHORIZED: new
variant(s) in `tensor_cuda/src/kernels.cu` + `ops.cpp` + `bindings.cpp` behind the existing explicit variant dispatch (**default
path a stays byte-identical; the BP-KERNEL-1 byte-range pin must still hold**), `scripts/bp_kernel_3.py`, `scripts/bp_census_3.py`,
`tests/test_bp_kernel_3.py`, `artifacts/bp_kernel_3/`, `docs/BP_KERNEL_3_LEDGER.md`. CPU build AUTHORIZED (same recipe, build dir
`tensor_cuda/build-bk3`, arch 89 only — this machine has one card). Copy the gitignored pinned arrays you need from the canonical
checkout: `/mnt/ForgeRealm/Project-Tensor/artifacts/bp_kernel_1/{inputs,forward_state}.npz`,
`/mnt/ForgeRealm/Project-Tensor/artifacts/bp_kernel_2/reference.npz` (sha 43f6d06a…) — same inputs, same reference, same gate.
READ-ONLY: `/mnt/ForgeRealm/GRAPA-Native-LLM` (canonical; the step harness `scripts/bp_census_1.py` is there now — `bp_census_3.py`
must reference THAT path, repo-relative to GRAPA's root, never a `wt/` path: `scripts/bp_census_2.py:30` hard-codes the pruned
`wt/grapa-bp1` and is a known defect — do not repeat it), everything else. No git, no subagents, **no GPU in this seat**, foreground,
every Bash call < 10 min, never kill or signal a process you did not start, no network, no checkpoint writes.

## Variant g (registered)
Both f kernels, re-shaped:
- **Key/query tiling through shared memory.** The query-owned kernel stages tiles of K, KQ, V (e.g. 64 keys × D) into shared memory
  and processes a tile of queries per block; the key-owned kernel stages tiles of Q and dO likewise. No per-pair global reloads.
- **Tensor-core MMA (BF16 in, FP32 accumulate) for the dense products:** bulk scores Q·KQᵀ, dO·Vᵀ, and the dK/dV/dQ outer-product
  accumulations. **The APA exact score Q·Kᵀ is needed only for selected pairs** (≈ 15 % at refine 0.15). Register two sub-routes and
  build both if budget allows, else g1 first:
  - **g1 — selective exact:** compute the exact score only for selected pairs (gathered/compacted rows per tile, or a masked scalar
    path inside the tile) — keeps APA's arithmetic saving.
  - **g2 — dense exact, masked select:** compute the full Q·Kᵀ tile on tensor cores too and select per element. Costs the exact
    product everywhere but stays on MMA. This is the measurement of whether APA's selection still buys anything once the dot
    products are on tensor cores — an honest question, report it either way.
- Same math as f (same softmax VJP, same saved lse/thr, same output-dot row-dot), same causal handling, dK/dV/dQ each written
  once, no atomics. Prior art: FlashAttention-2 (Dao 2023) tiling/backward structure — taken; the APA selective tile is ours.
- Numerics: BF16 MMA inputs with FP32 accumulate is a precision change vs f's FP32 scalar products — the FP64 gate decides.

## Gates and predictions (immutable)
1. **Correctness first:** dQ/dK/dV vs the FP64 reference, tolerance = 2 × |a − reference| per array and metric (as BP-KERNEL-2).
   RED → timing does not count.
2. Timing as before: CUDA events around the bare op, 3 warm-ups + 10 measured, interleaved a f g1 (g2).
   **Prediction:** best green g ≤ **0.35 × f** (≤ ~46 ms at f ≈ 131). **Falsifier:** > 0.35 × f → report which of tiling / MMA /
   selection-gather is the residual (per-kernel timing of the query-owned and key-owned halves is REQUIRED in the receipt).
   Secondary, registered: g1 ≤ g2 (selection still pays on tensor cores); if g2 < g1, say so plainly.
3. **The real step (cell 2):** `scripts/bp_census_3.py` — same protocol as `bp_census_2.py` (two arms in one cell, f then best-green g,
   2 + 5 steps each, same checkpoint/tokens/config), receipts to `artifacts/bp_kernel_3/census/`, registration before the run.
   **Prediction:** whole step ≤ **2.5 s** with g (f-arm reproduces ≈ 5.0 s within 10 %). If no g is green, cell 2 runs f vs g
   anyway labelled TIMING-ONLY.
4. Budget: two lead-run cells, each ≤ 300 s work / ≤ 590 s lease, single card, self-terminating; incomplete → INCONCLUSIVE.

## Deliverables
`scripts/bp_kernel_3.py` (`--dry-run` CPU tiny-shape path, `--run` = cell 1; reuse `bp_kernel_2.py`'s reference/gate code by import,
do not fork it), `scripts/bp_census_3.py` (`--dry-run`, `--run` = cell 2), `artifacts/bp_kernel_3/registration.json` (pins,
predictions verbatim, budget; fail-closed on drift), `tests/test_bp_kernel_3.py` (CPU: tile-boundary correctness of the tiling
logic vs a tiny dense oracle incl. causal + selection edge cases + non-multiple-of-tile shapes; gate/verdict logic on fabricated
inputs; variant a byte-range still pinned; schema; create-only; mark nothing as a receipt), `artifacts/bp_kernel_3/lead_commands.txt`
(each line accepts `--dry-run`), `artifacts/bp_kernel_3/engine_build_receipt.json`, `docs/BP_KERNEL_3_LEDGER.md`.

## Done — paste verbatim in your final message
1. g1/g2: tile shapes, MMA fragments used (wmma/mma.sync, shapes), code sites (line ranges), diff stat; confirmation a's pinned
   range is unchanged (sha).
2. Last lines of `pytest -q tests/test_bp_kernel_3.py`, the build, both `--dry-run` commands (each ends in a receipt path).
3. Registration sha256; the two predictions verbatim; whether g2 was built.
4. Prior art (FlashAttention-2 / Dao 2023; CUDA WMMA/mma.sync; taken vs ours; "unverified — lead to check" where you cannot verify).
5. Model + effort; confirmations: no GPU, no git, no subagents, nothing killed, no network, no edits outside the grants.
