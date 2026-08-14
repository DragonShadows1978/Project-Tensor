# APAMQ-DF1 — Decode Is 300× Off Its Bandwidth Floor: Find It, Build the Fix Variants

YOUR WRITABLE TARGET is this git worktree (Project-Tensor branch
`apamq-df`, base a3336aa = FA1 int4 kernel) — `tensor_cuda/*`,
`scripts/`, `docs/` AUTHORIZED, including building this worktree's own
extension. Run first, report after; a registered order IS the
permission. Disposable branch: bold implementation, honest receipts.

HARD BOUNDARIES: canonical Project-Tensor and other worktrees
READ-ONLY (pybind11 reuse via
-DFETCHCONTENT_SOURCE_DIR_PYBIND11=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src;
you MAY read /mnt/ForgeRealm/wt/apamq-fa2 for the int8 quantizer
tie-break lesson). No git, no subagents, no network. Sandbox has NO
GPU: ship one-command GPU legs (bench + ncu command list) for the
lead; skip-safe tests. RED honesty.

## The measured anomaly (do not re-derive)

E1 sweep (artifacts/apamq_e1/RESULTS.md, this worktree): decode L=1,
kv=1, D=512, S=65536: standard 0.529 ms; fused_apa 9.96 ms; int4_apa
10.83 ms. Physics: decode bulk reads S×D int4 ≈ 16 MB (+16 MB bf16
refine subset ~15%) → ~50–70 µs at this card's bandwidth. The int4
path is ~200–300× off its floor. Decode ms/tok is the number that
matters for serving (G-C measured APA decode +30–50% vs standard at
the port level).

## Pre-registered suspects (test each; add your own)

- **S1 per-call repack:** apa_selective_attention_int4 packs the WHOLE
  K cache every call. At decode that reads 32 MB bf16 + writes 4 MB
  packed per layer per token — and it re-does this every token.
  (~2× floor, cannot alone explain 300×, but pure waste.)
- **S2 stats rescan:** the split-K threshold-stats stage rescans all S
  keys per call — a second full pass. (Another ~2×.)
- **S3 occupancy/launch structure:** at kv=1, L=1, B=1: rows = H = 16
  blocks against 56 SMs unless split-K partitions fill the grid;
  verify what P actually is at these shapes, whether the WCOOP=false
  branch fires, and whether three sequential tiny launches
  (stats/split/merge) serialize with sync gaps. THIS class is the
  only one big enough for 300× — pin it with numbers.

## Tasks

1. **Stage-resolved micro-bench** (`scripts/apamq_df1_bench.py`): time
   pack / stats / split / merge separately (events around each
   launch), decode shapes kv=1 D∈{128,512} S∈{8K,16K,32K,64K}, plus
   an ncu command list per stage (occupancy, DRAM throughput,
   launch counts) the lead can run verbatim.
2. **Fix variants**, each behind an env/flag so the lead A/Bs them:
   - **V1 persistent packed-K workspace:** optional opaque handle the
     caller carries; pack ONLY appended rows per call (incremental,
     mirrors the ring's O(new-rows) contract). Yes, this reintroduces
     persistent derived state — at int4 it is 4 MB per layer at 64K
     (vs the 144 MiB bf16 ring that motivated FA1); the purity-vs-
     bandwidth tradeoff is now a measured decision, not a doctrine.
     Default OFF; existing no-state API unchanged.
   - **V2 fused stats+split:** compute threshold stats and the
     bulk/refine pass in one kernel (or cache stats incrementally in
     the workspace) so S is read once, not twice.
   - **V3 grid fill:** whatever S3 shows — split-K P sized to fill 56
     SMs at rows=16, merged launches, or a persistent-block variant.
3. **Gates:** all 11 FA1 gates stay green (V-off path byte-identical);
   V-on paths match V-off outputs at reassociation tolerance (state
   it); new decode-equivalence test per variant, skip-safe.
4. **Register the target now:** decode int4 D=512 kv=1 S=64K ≤ 1.0 ms
   (≈2× standard's 0.53 ms; floor is ~0.07 ms — headroom stays
   honest). Report best-variant number against it; a miss is a
   result.

## Done

Final message verbatim: per-stage timing table design, the variant
flag matrix, exact lead commands (bench, ncu list, pytest), build
command + result, CPU-check outputs, tolerances registered, deviations.
No GPU numbers — the lead measures.
