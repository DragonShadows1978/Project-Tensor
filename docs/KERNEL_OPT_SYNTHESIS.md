# Kernel Optimization Narrative Synthesis

Updated 2026-07-07 evening. The plan and addenda are the fixed intent; the
ledger holds the receipts; this file is what it means.

The program set out to make tensor_cuda faster using a large research drop,
and the research turned out to be the least of it. The inv_f32d181e corpus
was three things at once: genuinely useful extractions, careful analysis of
the wrong codebase, and partial synthesis over real web captures from agents
that died mid-run (usage exhaustion; the second investigation repeated the
pattern via per-agent timeouts). The durable lesson is a verification
protocol, not a reading list: check completion state, trust raw captures
over synthesis, verify reader verdicts against primary material — the
"fabricated" label itself turned out to be wrong.

What actually drove the program was Phase 0's own receipts. They killed the
corpus's headline lever (register pressure — zero spills everywhere),
exposed that the engine had already absorbed the obvious wins (pooled
allocator, fused RMSNorm, online-softmax APA), and located three real
defects nobody had written up: a decode grid that starved 56 SMs with 16
blocks, a divergent nibble decoder on the GPT-OSS expert path, and — via
the second investigation pointing back at our own ncu artifact — key loads
that touched 25 cache sectors where 4 would do.

The scoreboard after one day: five adopted or landed results, four honest
negatives. Adopted: A1 split-K attention decode (+34.6% median at long
context, APA function bit-preserved); A2 branchless mxfp4 decode (+61% on
the expert GEMVs); A5 warp-cooperative key loads (+64.8% median — prefill
3.2–5.7×, with a dispatch that returns GPT-OSS's D=64 decode to the
verbatim old path). Negative, reverted with receipts: A3 bank-conflict
padding and A4 gemm staging (both the same lesson — reducing a theoretical
hazard buys nothing when DRAM or FMA is the binding resource), Phase 1.1
device argmax (a 600KB/token copy is immaterial at 17.5ms/token), and
Phase 2 CUDA graphs (correct, parity-clean, −47% launches — and still only
+2.5% at the gate workload, because the hybrid model's eager half bounds
it; parked, not deleted).

The pattern across all nine: measurement beat theory every single time.
Nothing adopted came from the research corpus's recommendations as written;
everything adopted came from ncu/nsys receipts on the live tree, twice via
the corpus pointing at where to look. And the APA function — bulk-bits
scores, z-score threshold, full precision on the refine percentile — went
through three kernel restructurings without moving: the parity suite, not
good intentions, is what guarantees that.

Open: Phase 3.1 (O(S²) blend-mask elimination, in flight), Phase 3.2/5
(online causal softmax, elementwise fusion — modest expected yield),
Phase 4.1 DP4A (deferred with cause: needs int8 activations, attacks
arithmetic in DRAM-bound kernels). The long-context GPT-OSS workload —
the machine's flagship — is where the adopted set compounds: split-K ×
coalesced keys × branchless expert decode, all on its decode path.
