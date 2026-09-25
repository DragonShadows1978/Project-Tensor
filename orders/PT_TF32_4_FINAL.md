# PT-TF32-4 — final certification pass: derived tail bounds for dK, a tensor-core gate that measures what it claims, the two edge cases

Registered 2026-09-25 19:40 EDT by the lead (Fable 5.1). Seat: GPT-6 Astra at max. YOUR WRITABLE TARGET is the fork
worktree `/mnt/ForgeRealm/wt/pt-tf32` (branch tf32-fast-path, PT-TF32-3 committed 3c53811; slot-11 receipts under
`artifacts/pt_tf32_3/lead_slot_01/`). Live engine never touched. No GPU here; the lead runs the slot.

## Slot-11 receipts (fork a1bf6ce6; read them)
- MODEL LEVEL, all GREEN: onset, healthy, control (noise-floor bar 0.99843), step time 5.105 → 5.66 s/step (ratio 1.11,
  bar 1.3). Sanitizers memcheck / racecheck / synccheck 0 errors. The engine does the job at the model level.
- KERNEL LEVEL, RED on three registered bars that turn out to be mis-derived or at tiny shapes:
  1. **dK**: the 10× defect is GONE — isolated dK rel-L2 2.96e-4, downstream 4.5e-4, same-state 4.1e-4 against an
     expected 6.9e-4 and bound 2.08e-3 (all GREEN on rel-L2). What trips is `normalized_max_abs`: downstream 2.20e-3 and
     same-state (cases 0, 2, 3) vs the SAME bound 2.08e-3 — a per-element max over millions of TF32-rounded sums compared
     against a bound derived for an L2 average. Derive the max-abs bound properly (extreme-value statistics of the
     rounding model: per-element σ from the rel-L2 expectation, N elements → expected max ≈ σ·√(2 ln N), bound = that
     plus a stated margin) and register it; do NOT fit it to the observed values — show the derivation and the numbers
     it yields BEFORE comparing. If the observed max exceeds the derived bound, that is a real tail defect: find it.
  2. **GEMM**: the diagnostics prove tensor cores ARE engaged (cublasLt algo 21, numerical flags 0x40202 = INPUT_TF32 |
     ACCUMULATOR_32F | HMMA, compute type 77 = 32F_FAST_TF32) and speedups are 1.2–2.4× (dWeight/dInput lowest). The
     ≥ 5× bar was an estimate from peak ratios; SGEMM already runs at ≈ 11 TFLOP/s on these shapes. Register the gate
     as what it was meant to check — "HMMA/TF32 algorithm selected for every shape (flags) AND accuracy ≤ 1e-3" — with
     speedup reported as a diagnostic, and explain in the ledger why 5× was wrong (roofline at M=4096 with these K/N;
     cuBLAS algo selection; the mathmode query returned status 7 — say what that means and fix the query if it is a
     usage error). If any shape is NOT on an HMMA algo (dWeight?), make it so.
  3. **Edge cases**: the same two `test_native_padding_grouped_heads_and_fp32_storage` cases (`widths2-lengths3/4-True`:
     D 96 / VD 64, L,S (17,31) and (33,35), causal, H 4 / KVH 2) — one element ≈ 2e-5 off at 1e-4 scale, atol 2e-5.
     Fix the kernel for those shapes, or derive the tolerance from the rounding model at that value scale and update the
     test with the derivation. Either way the test passes for a stated reason, not a loosened number.
- Every other lane unchanged; every default unchanged; bf16 paths byte-identical; harness stays disk-safe.

## Deliver
Rebuild in-tree; `docs/PT_TF32_4_LEDGER.md` (derivations, the new registration file with rationale — the old ones and
their receipts untouched); CPU tests; `## Done` with receipts verbatim + the blocked-report + the lead's slot sequence
(≤ 20 min: units, gemm, attention ×4, sanitizers; the model-level lanes may be re-run for completeness). Prior art at
the code site + ledger + report (extreme-value bounds: Gumbel / David & Nagaraja; TF32 rounding: NVIDIA; cuBLASLt flags).

## Boundaries (binding)
- Writable: `/mnt/ForgeRealm/wt/pt-tf32` only. Never `/mnt/ForgeRealm/Project-Tensor`, `/mnt/ForgeRealm/GRAPA-Native-LLM`,
  `/mnt/ForgeRealm/wt/grapa-*` (read-only), `/mnt/ForgeRealm/grapa_run*/`, `/mnt/ForgeRealm/grapa_side_*`. No GPU
  (`CUDA_VISIBLE_DEVICES=''`). No git. No subagents. Never kill processes you did not start; never touch
  `/tmp/forge-gpu.lock`. RED honesty.
