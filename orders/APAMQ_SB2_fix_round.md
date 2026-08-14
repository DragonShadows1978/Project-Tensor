# APAMQ-SB2 — Fix Round: Illegal Access + O(chunks) Syncs + Transient Cap

Same worktree, grants, and boundaries as APAMQ_SB1_int8_gemm_speed_blend.md.
Sandbox has NO GPU; lead runs GPU legs. RED honesty. This is the ONE
fix round before stash-or-merge.

## Measured facts (lead-run, logs/sb1_gates_gpu.log + sb1_perf_gpu.log)

1. **CRASH:** `gemm_apa` D=512 KVH=1 L=512 S=16384 prefill → "illegal
   memory access" (also kills one gate at tensor_cuda/__init__.py:986
   and context-poisons 4 subsequent gates; 6 passed before it).
   S=65536 prefill and S=65536 decode RUN CLEAN — the bug is
   shape-dependent (S=16K path differs: chunking boundary? Spad?
   bounded-causal index arithmetic at S==L·32? audit hard).
2. **Perf:** 64K prefill 1074 ms vs standard ~64.5 (16.6×; target was
   ≤1.5× at 16K). Structure suggests the refine pipeline: ~200
   compact-chunks × (D2H count readback + skinny GEMM launch) =
   serialized sync storm. Decode 64K = 5.0 ms (best APA decode number
   yet measured — preserve this property).
3. **Memory:** 64K prefill pool peak 4.89 GB (unchunked int32+fp32
   score matrices). Speed mode may spend transients, but not that.

## Tasks

1. **Kill the illegal access.** Static audit of index math (Spad
   round-up vs true S in every kernel touching C/D, bounded-causal
   lo/hi, gather index bounds, chunk-tail rows). Add debug asserts
   compiled under a flag. Ship a compute-sanitizer command for the
   lead (`compute-sanitizer --tool memcheck python3 ...` on the
   failing cell) in case your fix misses — but aim to fix it by
   audit, and explain the found bug precisely.
2. **O(1) synchronizations per call.** Replace per-chunk D2H count
   readbacks: device-side compaction (atomic-append or scan) into a
   preallocated index buffer sized from a refine-fraction bound
   (cap ~2× nominal; on overflow, clamp selection to capacity and
   RECORD an overflow flag in stats — never crash, never silently
   change semantics without reporting), then at most ONE D2H sync
   per call before the batched skinny GEMMs.
3. **Cap transients.** Internal L-sub-chunking so int32+fp32 score
   materialization stays ≤ ~384 MiB per call at any S (mirrors the
   port's adaptive blend budget). Report the sub-chunk policy.
4. **Gates:** re-run list = SB1's gate script (all 11 must pass now)
   + the three perf cells + decode regression (keep ≤5.5 ms at 64K
   decode). Registered rails (unchanged + new): ≤1.5× standard at
   D=512 KVH=1 prefill 16K; pool transient ≤384 MiB at 64K prefill;
   zero sanitizer findings on the previously-failing cell.

## Done

Final message verbatim: the found bug (exact line + mechanism), diff
summary, build result, CPU checks, the compaction design chosen,
sub-chunk policy, exact lead commands, deviations. No GPU numbers.
