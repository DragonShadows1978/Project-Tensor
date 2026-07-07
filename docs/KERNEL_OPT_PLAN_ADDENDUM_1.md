# Kernel Optimization Plan — Addendum 1 (Measured Targets)

Status: immutable at initial commit. Supplements
`KERNEL_OPT_IMPLEMENTATION_PLAN.md`; all of its House Rules, registered
thresholds, parity gates, evidence-class discipline, and power bounds apply
unchanged. Registered because Phase 0 receipts (ledger 2026-07-07 11:15)
identified measured targets not enumerated in the base plan. Authorized by
David 2026-07-07: decisions to orchestrator's discernment; sole restriction
restated below.

## APA Invariant (sharpened, per David 2026-07-07)

APA remains as it is: **two-pass bulk-bits then selective precision** —
pass 1 scores every key with quantized (bulk-bits) dot products and derives
the selection threshold from full-key-range statistics; pass 2 refines the
selected keys at full precision. This algorithmic structure and its
selection semantics are inviolable. Work-distribution, memory-layout, and
instruction-level changes that preserve the exact selection semantics and
pass the existing APA parity tests are permitted.

## Workstreams

A2 **mxfp4_gemv branch divergence** (measured: 60.94% branch efficiency,
    ~16k divergent branches/launch). Replace the 16-way `mxfp4_value`
    switch with a branchless E2M1 decode (arithmetic or register/constant
    LUT) and reuse scale reads where lanes share (n,g). Constraint: decoded
    values must be bit-identical to the current table (exact parity).
    Gate: kernel-level accept at GPT-OSS FFN shapes (M=1, K=2880, plus
    expert variants); existing mxfp4 tests pass.

A3 **int4_gemv shared-memory bank conflicts** (measured: 47% excessive
    shared-mem wavefronts, 71.6% DRAM/peak, 5 blocks/SM). Pad or swizzle
    the dynamic-shmem x staging / access pattern. Gate: kernel-level accept
    at K ∈ {4096, 12288, 15360}; int4 parity tests pass.

A1 **apa_selective decode grid underfill** (measured: 8.33% achieved
    occupancy, 16 blocks on 56 SMs at decode). Split-K / flash-decoding
    grid shape: partition the key range across blocks with per-partition
    online-softmax partials (m, l, acc) and a reduction stage. The APA
    invariant binds the design: the pass-1 threshold derives from
    full-key-range statistics, so the implementation must produce the SAME
    threshold as today (global stats stage before or across partitions),
    then apply pass-2 selective refinement per key exactly as now. Only the
    work distribution changes. Gate: kernel-level accept at decode
    S ∈ {2048, 8192, 32768} on gpt_oss20b and qwen35 geometries; APA
    parity tests pass; existing kernel outputs matched within current test
    tolerances.

A4 **mxfp4_gemm weight-read coalescing + scale staging** (inspected:
    adjacent lanes stride G·16 bytes in packed blocks; per-element global
    scale reads). Restage weight tile loads for coalescing and stage
    scales in shared memory. Gate: kernel-level accept at prefill
    L ∈ {512, 2048} GPT-OSS geometries; mxfp4 tests pass.

Order of execution: A2 → A3 → A1 → A4 (small to large), interleaved with
base-plan Phase 1.1 (device argmax) which needs no addendum.

## Gating Validity Rule

Kernel-level A/B timing gates are valid only on an idle GPU (no concurrent
compute jobs). Implementation and correctness/parity testing may proceed
under contention; every timing gate defers until an idle window and is
ledgered with the GPU state at measurement time. Each gating run bounded
≤10 min.

## Acceptance

Each workstream: parity gate + base-plan kernel-level accept threshold
(≥15% median kernel-time reduction at its listed shapes, no shape
regressing >5%), receipts in the ledger. Workstreams failing their gate are
reverted and ledgered as negative results — a failure is still a result.
