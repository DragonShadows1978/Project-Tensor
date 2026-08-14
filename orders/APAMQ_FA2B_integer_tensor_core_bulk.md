# APAMQ-FA2B — Tensor-Core APA: Integer Bulk Pass + bf16-MMA Refine

YOUR WRITABLE TARGET is the git worktree you are launched in
(Project-Tensor branch `apamq-fa2`, forked off apamq-fa a3336aa which
contains apa_selective_attention_int4) — `tensor_cuda/src/`,
`tensor_cuda/include/`, `tensor_cuda/tensor_cuda/`,
`tensor_cuda/tests/`, `scripts/`, `docs/` AUTHORIZED, including
building this worktree's own extension. Run first, report after; a
registered order IS the permission. This branch is DISPOSABLE by
design (the lead stashes it if results are bad) — be bold in the
implementation, honest in the receipts.

HARD BOUNDARIES: canonical /mnt/ForgeRealm/Project-Tensor and all
other worktrees READ-ONLY (pybind11 sources may be reused read-only
from the canonical build as FA did:
-DFETCHCONTENT_SOURCE_DIR_PYBIND11=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src).
No git, no subagents, no network. Sandbox has NO GPU: build with
nvcc, ship skip-safe tests + one-command GPU legs for the lead. RED
honesty; a perf miss is a result.

## Context (pre-nailed — read these in-worktree first)

- docs/APAMQ_FA2_TENSOR_CORE_STOP.md — YOUR predecessor's STOP: naive
  bf16 staging of dequantized K perturbs threshold statistics /
  selection masks beyond the reassociation-only invariant. This order
  supersedes the STOP with two lanes that DON'T have that problem.
- docs/APA_MQA_FIX_PLAN.md (F-A2) — registered target: ≤3× the cuBLAS
  standard path wall time at D=512, prefill (L=512 chunk), S=16K;
  stretch = parity. Current gap ~12.5× (artifacts/apamq_e1/RESULTS.md
  in this worktree).
- Measured background (lead receipts, 2026-08-13): FBD1 showed
  selection flips at quantization-noise scale have ~zero correlation
  with output error; and the int4 kernel's packed keys give a 4×
  memory-traffic advantage over bf16 K that scalar dequant currently
  squanders (kernel is ALU-bound).

## Lane 1 (exact, drop-in): bf16-MMA refine pass

Q and K are stored bf16. bf16×bf16 products accumulated in fp32 are
EXACT (8+8-bit mantissas fit fp32), so a bf16-MMA-with-fp32-accumulate
refine pass differs from the current scalar path ONLY in accumulation
order — inside the reassociation invariant. Tensor-core the refined
(selected-key) exact dots of apa_selective_attention_int4. Gate:
allclose vs the a3336aa kernel at reassociation tolerance (state it),
selection identical by construction.

## Lane 2 (the main event): integer bulk pass

Add an int8-quantized-Q × int4-K integer bulk pass: per-query-vector
symmetric int8 quantization of Q (one fp32 scale per query row), then
bulk_j = (sum_d qcode_d * kcode_{j,d}) * (qscale * kscale_j) with the
integer sum computed via dp4a (__dp4a) and/or mma.s8 IMMA tiles —
your design choice; justify it with the arithmetic-intensity math in
the report. Integer sums are EXACT — no rounding pathology is
possible; the only semantic change vs a3336aa is the added int8-Q
quantizer, making this a NEW OPERATING POINT, not a drop-in: expose
it as a distinct mode (new entry point or explicit flag), leave the
a3336aa behavior untouched and selectable.

Correctness gates (write them; lead runs GPU): (a) integer-sum
bit-exactness vs a numpy int32 reference on the same codes (zero
tolerance on the integer part; fp32 scale-product tolerance ~1e-6);
(b) end-to-end vs a numpy fp32 composed reference with the int8-Q +
int4-K quantizers mirrored exactly (EXP-APA-2 conventions); (c) the
existing 11 gates of test_apa_selective_int4.py stay green
(regression — the a3336aa path must be unchanged); (d) selection
behavior tests at MQA kv=1 D=512 rect-causal S>L.

Threshold statistics stay in fp32 computed from the integer-derived
bulk scores (exact), same mean+z·std semantics. Non-selected keys'
softmax scores = the integer-derived bulk scores (fp32 after scale) —
NO bf16 materialization anywhere (that disease is documented in the
APAMQ ledger).

## Perf legs (lead-run; extend scripts/apamq_e1_sweep.py)

Add the new path(s) as sweep rows (e.g. `int4_tc_apa`). Cells the
target is judged on: D=512, kv=1, prefill L=512, S ∈ {8K, 16K, 32K,
64K}, plus decode L=1 same S range (the bandwidth-advantage
hypothesis: int-bulk decode may BEAT standard at long S — measure,
don't assume). Include D=128 rows for regression breadth.

## Done

Final message verbatim: files + line counts, exact build command +
result, CPU-check/pytest outputs, the gate list with registered
tolerances, the dp4a-vs-IMMA design rationale actually implemented,
lane 1 status, and any deviation. No GPU numbers — the lead measures.
