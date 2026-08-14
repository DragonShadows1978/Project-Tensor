# APAMQ F-A2 Tensor-Core Rung — STOP Receipt

Status: **STOPPED; no tensor-core dispatch is registered or presented as a
performance result.** F-A1 remains independently shippable.

## Candidate design inspected

The viable causal/GQA shape is one CTA per `(batch, query_head, query_tile)`:

1. Reuse F-A1's call-local symmetric-7 INT4 codes and fp32 per-key scales.
2. Walk K in 64-key tiles and D in 64-wide slices. Dequantize each packed tile
   into 16-bit shared memory, then use `mma.sync`/WMMA to accumulate a 64x64
   bulk-score tile in fp32. D=512 therefore takes eight K-slices without
   requiring a 64x512 Q and K tile to coexist in shared memory.
3. Apply bottom-right bounds per query row before threshold statistics. Make a
   first full K walk for `sum(abs(score))` and `sum(score^2)`, then a second K
   walk for the same blend and online-softmax merge as F-A1. Exact refinement
   can initially remain the warp-cooperative CUDA-core dot; a later sparse MMA
   compaction is separable from the bulk rung.
4. Keep decode on F-A1's split-K path. Query tiling is a prefill optimization;
   `L=1` cannot fill an MMA query tile and already needs key-range parallelism.

The repository's existing noncausal Q-tile kernel validates the broad tiling
shape, but it is specialized to fp16, noncausal attention, and D<=64. It also
has a separately registered threshold/value-rounding contract. It cannot be
reused for bf16 D=512 causal selective attention by dispatch alone.

## Why this rung stopped

F-A1's required bulk contract dequantizes `code * fp32_scale` in fp32
registers. Ada tensor cores require the matrix operands to be represented as
bf16/fp16 or TF32. Staging the reconstructed K tile in any of those formats
adds an operand-rounding boundary before the dot. That is not merely fp32
reassociation: it can move a score across `mean + z*std`, changing the
selection mask. The order explicitly requires the same selection with only
float-reassociation-class differences.

A staged cuBLAS alternative has the same issue and additionally materializes
bulk scores, exact scores, and weights. It would compute exact QK for every key
to make the blend cheap, abandoning the selective-work property and adding
large `L*S` transients. It is therefore not registered as F-A2 merely because
it would invoke tensor cores.

The only strict-semantics fallback found is to recompute every tensor-core bulk
score with the fp32-dequant CUDA-core dot before forming statistics and the
mask. That preserves F-A1 but also restores the bulk work F-A2 is meant to
remove, so it is not a credible route to the registered <=3x target.

## Evidence and next decision

- Compile evidence: the F-A1 scalar/decode family builds for sm_89 with nvcc
  12.6.85. No tensor-core candidate was landed.
- Runtime evidence: none in this seat; the sandbox has no GPU. No timing,
  refine-mask, or occupancy claim is made.
- Harness: `scripts/apamq_e1_sweep.py` now measures `int4_apa` in the same
  matrix as `standard` and legacy `fused_apa`, including the call-local pack
  in wall time and transient memory. There is deliberately no mislabeled
  `int4_tc_apa` row.

To reopen F-A2, the lead must authorize one semantic change: register bf16 (or
fp16) reconstructed bulk K as the tensor-core reference and add a selection-
mask/output tolerance gate against F-A1. With that contract, the tiled design
above is implementable and the same E1 harness can add `int4_tc_apa` as a
fourth path before any <=3x verdict is made.
