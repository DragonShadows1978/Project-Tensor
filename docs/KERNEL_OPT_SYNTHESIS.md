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

The program closed the same evening with every avenue executed or
dispositioned. Phase 3.1 joined the adopted set (+43–87% on the composed
blend operation — the O(S²) mask and its two full elementwise adds per
layer are simply gone). The tail died honestly: online causal softmax
wins only ~10% in a narrow underoccupied regime (below the program's
bar), and SwiGLU fusion — +40% at kernel level — moved e2e by 0.7%,
which is precisely the microbench-only outcome the plan's 2% loop gate
was written to reject.

Final count: four adopted (A1 split-K, A2 branchless mxfp4, A5
warp-cooperative key loads, 3.1 blend bounds), six closed negatives with
receipts, four dispositioned with cause. The APA function — bulk-bits
scores, z-score threshold, full precision on the refine percentile —
survived four kernel restructurings unchanged; parity suites are the
proof. The flagship GPT-OSS-20B long-context workload compounds all
four adoptions: decode attention +25–35%, expert GEMVs +61%,
long-context prefill 3.2×, chunked blend prefill +67%.

What the program actually taught, beyond the speedups: measurement
discipline is the product. Three instrument laws (P-state bimodality,
concurrent-context poisoning, alloc-jitter) now govern any future gate
on this machine; the noise floor killed as many plausible wins as it
confirmed real ones; and every negative carries the receipt that stops
it from being re-proposed. Deferred with cause, for future programs:
DP4A behind a model-PPL bridge, CUDA graphs behind a workload whose
attention is capturable, and FP8-on-Ada sitting unread in 155 real web
captures.
