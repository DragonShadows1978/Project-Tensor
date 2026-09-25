# PT-TF32-2 — certify the fast precise path: dK precision, edge cases, GEMM tensor cores, TF32-appropriate gates

Registered 2026-09-25 16:15 EDT by the lead (Fable 5.1). Seat: GPT-6 Astra at max. YOUR WRITABLE TARGET is the fork
worktree `/mnt/ForgeRealm/wt/pt-tf32` (branch tf32-fast-path; PT-TF32-1 and the lead's slot-9 receipts are committed
under `artifacts/pt_tf32_1/lead_*`). The live engine `/mnt/ForgeRealm/Project-Tensor` is never edited, rebuilt or
imported. No GPU here: build + CPU work + blocked-report; the lead runs the GPU gates.

## Slot-9 receipts (read them; they are the evidence)
- GRAPA exact-state, TF32 blocks 0–10: onset cos +0.9994 vs FP64 (GREEN; bf16 path −0.606); healthy +0.99949 (GREEN);
  control +0.99867 (RED by 3e-4 against a 0.999 bar set without a TF32 noise floor); **step time 5.20 s vs 5.09 bf16
  (ratio 1.02, GREEN)**. The path works and is fast enough. Now certify it.
- Kernel gates: GEMM accuracy PASS (rel-L2 2.9e-4 vs FP64 on all shapes) but speed 1.33× SGEMM vs the ≥ 5× rule
  (0.783 → 0.588 ms at 4096×1024×1024) — TF32 tensor cores are evidently not engaging in `matmul.cu`'s path (a real TF32
  tensor-core GEMM at that shape is ≈ 0.1 ms on this card); attention forward RED on all 4 cases by the "2× the fp32 `a`
  kernel's distance to FP64" rule — `a` is fp32-exact (8.9e-7) so a TF32 kernel (3.2e-4) can never pass it: the
  registration compared unlike precisions (category error; the RED stands as recorded); attention backward: dQ and dV
  PASS, **dK rel-L2 1.6e-2 vs bound 1.3e-3 — 10× beyond TF32 rounding: a defect in g1_tf32's dK path** (isolated and
  downstream both RED); selection flips 2.8e-5 PASS; memcheck 0 errors; **7 of `tests/test_pt_tf32_gpu.py`
  `test_native_padding_grouped_heads_and_fp32_storage` cases fail** (widths (19,17) and (96,64) at small lengths, causal,
  grouped heads H 4 / KVH 2, one element ≈ 2e-5 off at 1e-4 scale, rtol 3e-3 / atol 2e-5).

## Build
1. **dK precision in g1_tf32:** find why dK is 10× worse than dQ/dV (candidates: the dS·Q accumulation path, the
   selected-pair exact path vs the tiled path mixing, p/dS precision, the row-term D_i, tile-edge handling). Fix; unit-
   gate on CPU where possible; the lead re-runs the FP64 gate.
2. **Edge cases:** fix the 7 failing padding/grouped-head cases (or prove they are tolerance artefacts of near-zero
   values under TF32 — then the test's tolerance must be justified from the TF32 rounding model, not loosened to pass).
3. **GEMM tensor cores:** make the TF32 GEMM path actually use tensor cores (cublasLt with `CUBLAS_COMPUTE_32F_FAST_TF32`
   / `cublasGemmEx` TF32 math, or a WMMA tf32 kernel); target ≥ 5× SGEMM at the model shapes; keep accuracy ≤ 1e-3.
4. **Gates registered for TF32 (new registration file, the old one and its RED receipts untouched, rationale written):**
   attention forward/backward vs FP64 with TF32-level bounds (state the rounding model: 10-bit mantissa inputs, fp32
   accumulate → expected rel-L2 ≈ 1e-4–1e-3 at these shapes; bound = 3× the expected, derived, not fitted); healthy/
   control "unchanged" bar = a measured TF32 noise floor (run the no-policy pair twice and the policy pair twice on the
   healthy state; bar = 1 − 3 × the observed spread) — the lead runs the measurement, you register the rule and the
   computation; racecheck/synccheck gates; the exact-state onset gate (cos ≥ 0.99) and the timing gate (≤ 1.3× bf16)
   unchanged.
5. Rebuild the fork engine in-tree; deliver `docs/PT_TF32_2_LEDGER.md` (what changed and why, per receipt), CPU tests,
   the lead's exact GPU sequence (one slot, ≤ 30 min), `## Done` with receipts verbatim and the blocked-report for every
   GPU gate. Prior art at the code site + ledger + report.

## Boundaries (binding)
- Writable: `/mnt/ForgeRealm/wt/pt-tf32` only. Never `/mnt/ForgeRealm/Project-Tensor`, `/mnt/ForgeRealm/GRAPA-Native-LLM`,
  `/mnt/ForgeRealm/wt/grapa-*` (read-only), `/mnt/ForgeRealm/grapa_run*/`, `/mnt/ForgeRealm/grapa_side_*`. No GPU
  (`CUDA_VISIBLE_DEVICES=''`). No git. No subagents. Never kill processes you did not start; never touch
  `/tmp/forge-gpu.lock`. RED honesty.
