# APAMQ-SB1 — "Blend Done Right": cuBLASLt INT8 Bulk + Gather Refine, fp32 Scores

YOUR WRITABLE TARGET is this git worktree (Project-Tensor branch
`apamq-sb`, base a3336aa) — `tensor_cuda/*`, `scripts/`, `docs/`
AUTHORIZED, including building this worktree's own extension. Run
first, report after; a registered order IS the permission. Disposable
branch: bold implementation, honest receipts.

HARD BOUNDARIES: canonical Project-Tensor and other worktrees
READ-ONLY (pybind11 reuse as in FA1's build recipe; you MAY read
/mnt/ForgeRealm/wt/apamq-fa2 read-only to reuse its int8-Q quantizer
design — and DECLARE the round-half convention explicitly this time;
fa2's one RED gate was a kernel-vs-numpy tie-break divergence). No
git, no subagents, no network. Sandbox has NO GPU: one-command GPU
legs for the lead; skip-safe tests. RED honesty.

## Premise (pre-nailed — the measured background)

The fused/selective kernels lose to cuBLAS by 15–55× at prefill
because a one-query-per-block skeleton has no K reuse and starves
tensor cores (APAMQ ledger, 2026-08-14). The OLD blend had the right
architecture (GEMM bulk + GEMM refine) and two proven defects:
(1) it materialized scores in bf16 — measured noise source
(force-all receipt: Δattn 0.125 with identical selection); (2) it
computed the FULL exact rank matrix instead of only selected keys.
SB1 is the blend rebuilt without either defect, with the bulk GEMM on
integer tensor cores via the LIBRARY (no custom tiling):

- **Bulk:** per-row int8 Q codes × per-key int8 K codes →
  cublasLtMatmul INT8→INT32 (exact integer sums), fp32 scale product
  applied on the int32 output. NO bf16 score tensor anywhere; scores
  live fp32 (or int32+scales) until softmax.
- **Threshold/select:** mean+z·std of |bulk| per row over valid keys
  (same semantics as apa_selective/apa_blend_softmax), emit selected
  ratio + a compacted index list per row-block.
- **Refine:** GATHER selected K rows (bf16) into a compact matrix,
  ONE skinny bf16 GEMM (fp32 accumulate) for exact scores of only the
  selected ~10–15%, SCATTER over the bulk scores.
- **Softmax + P·V:** fp32-score softmax (bottom-right causal,
  Lq/row0/window conventions identical to apa_blend_softmax's BOUNDED
  path), then P·V via existing matmul. Probabilities may be bf16 for
  the P·V GEMM; SCORES never are.

Expose as one engine entry (your naming; suggest
`apa_gemm_selective_attention(q, k, v, scale, zthr, is_causal, ...)`)
operating on chunk shapes (B, H/KVH GQA-aware, L up to chunk, S). K
int8 codes+scales may be computed per call this order (persistent
workspace is DF1's lane, not yours) — but structure the code so a
cached codes tensor can be passed in later.

## Registered gates (write them; lead runs GPU)

- G-SB1-a: composed numpy fp32 reference with both quantizers
  mirrored exactly (declare rounding conventions; zero-tolerance on
  integer sums, 1e-6 on scale products, 2e-2 end-to-end).
- G-SB1-b: rect-causal with-cache S>L + MQA kv=1 D=512 + GQA kv∈{4,8}
  D=128 (the 121→11M bug class gets its explicit test).
- G-SB1-c: selection-fraction sanity vs apa_selective_attention at
  matched zthr on identical inputs (different quantizers ⇒ sets may
  differ; report overlap, no tolerance gate — data only).
- G-SB1-d (perf, judged by lead): D=512 kv=1 prefill L=512:
  **target ≤1.5× cuBLAS standard wall at S=16K** (standard ≈11.7 ms →
  ≤17.6 ms); report the transient cost honestly (this path
  materializes chunk×S fp32 scores — it is the SPEED mode for
  wall-less regimes; the fused kernel remains the memory mode).
  Extend scripts/apamq_e1_sweep.py with path `gemm_apa`.
- Quality (later, port level, pre-registered now): an FC ppl arm for
  this mode must be ≤ apa_blend arm +0.25% relative — same G-C bar.

## Done

Final message verbatim: files + line counts, build command + result,
cublasLt algo/layout choices made (and the int8 layout requirements
you had to satisfy), quantizer rounding conventions declared, CPU
check outputs, exact lead commands (gates + sweep), deviations. No
GPU numbers — the lead measures.
