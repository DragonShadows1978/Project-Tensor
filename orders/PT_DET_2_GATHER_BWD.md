# PT-DET-2 — the second unordered site on the training path: a deterministic loss-gather backward

Registered 2026-09-26 08:30 EDT by the lead (Fable 5.1). Seat: GPT-6 Astra at max. YOUR WRITABLE TARGET is the fork
worktree `/mnt/ForgeRealm/wt/pt-tf32` (branch tf32-fast-path, HEAD = PT-DET-1 5430f39). Live engine never touched. No GPU.

## Evidence
Your PT-DET-1 audit (`docs/PT_DET_1_AUDIT.md`): on the registered v2/v3 training routes, after the embedding backward,
the only remaining order-nondeterministic site is the gather / top-k backward (`tensor_cuda/src/kernels.cu:4813` via
`tensor_cuda/src/ops.cpp:960`, `:977`), used by the loss gather — duplicate destination indices are accumulated with
unordered atomics. With PT-DET-1 on, full-step run-to-run reproducibility still depends on this site.

## Build
1. A deterministic gather backward, opt-in under the SAME switch family (`TC_DET_EMBED_BWD` → rename the family to
   `TC_DETERMINISTIC=1` / `set_deterministic(bool)` covering both sites; keep the PT-DET-1 name as an alias so the
   registered PT-DET-1 harness still runs) — sort the destination indices, segmented fixed-order reduction, one writer
   per destination; default OFF byte-identical. Gate: bitwise-identical across 5 calls on the model's loss-gather
   shapes (W4096 rows, vocab 8192, duplicates present), fp64 rel-L2 ≤ 1e-6, cost ≤ 2× the atomic path.
2. Extend the PT-DET-1 repro lane: arms (i) v3 argv with the family ON, (ii) bf16 with it ON, (iii) controls OFF —
   registered predictions: ON arms bitwise-identical over 30 steps (losses, gnorms, saved weights); OFF differ.
3. Rebuild in-tree; `docs/PT_DET_2_LEDGER.md`; CPU tests; `## Done` with receipts, blocked-report, the lead's slot
   sequence (fold into `pt_det_1_slot.py` so ONE ≤ 25-min slot certifies both). Prior art at the code site.

## Boundaries (binding)
- Writable: `/mnt/ForgeRealm/wt/pt-tf32` only. Never Project-Tensor main, GRAPA main, `wt/grapa-*`, run dirs, traces,
  engines, snapshots. No GPU (`CUDA_VISIBLE_DEVICES=''`; run 7 is live). No git. No subagents. Never kill processes
  you did not start; never touch `/tmp/forge-gpu.lock`. RED honesty.
