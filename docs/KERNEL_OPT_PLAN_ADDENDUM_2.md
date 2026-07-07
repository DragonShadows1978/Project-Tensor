# Kernel Optimization Plan — Addendum 2 (APA Key-Load Coalescing)

Status: immutable at initial commit. Supplements the base plan and
Addendum 1; all House Rules, the APA function invariant (bulk-bits scores →
z-score threshold → full precision on the refine percentile; David's
sharpened wording, ledger 2026-07-07), parity gates, thresholds, and the
Gating Validity Rule apply unchanged.

Origin: inv_0a7a5f94 triage (2026-07-07) — the one genuinely new finding,
grounded in this program's OWN Phase 0.2 ncu receipt
(`artifacts/kernel_opt/ncu_apa_selective_report.txt`, lead-verified):
25.3 sectors/request on global loads (≈4 optimal for these access sizes)
and 88.1% of warp stall cycles on L1TEX scoreboard dependencies in
`apa_selective`. Split-K (A1) fixed grid underfill; the per-block key
gather pattern remains ~5-6× wasteful on memory transactions.

## Workstream

A5 **Warp-cooperative key loads in the apa_selective family.** Current
    pattern: each thread owns key j (strided j = tid, tid+nt, …) and loops
    d = 0..D — at fixed d, lanes read addresses one row-stride apart
    (uncoalesced, ~1 sector per lane). Rework: lanes within a warp
    cooperate on one key at a time — contiguous reads of the key row
    across lanes (coalesced), warp-shuffle reduction for the bulk and
    refine dots, warp-level online-softmax accumulation. The APA function
    is untouched: identical bulk scores, identical threshold statistics,
    identical |bulk| ≥ thr refine decision, identical softmax math — only
    the assignment of lanes to loads changes. Applies to: the split-K
    stats/split kernels (A1's structure), the fused `apa_selective_kernel`
    (+ sink and train-forward variants ONLY if the diff stays surgical —
    the backward kernel is out of scope). Sequencing: BLOCKED until
    Phase 2 (CUDA graphs) lands — same file, avoid concurrent kernels.cu
    surgery.

## Gates (registered now)

- Parity: existing APA tests + the A1 split-K parity suite pass unchanged;
  kernel-vs-kernel diff vs current outputs within existing tolerances
  (expect 0 at f32 — reduction ORDER may change with warp-cooperative
  accumulation; if f32 outputs shift within tolerance but not bit-exact,
  document the reassociation and verify against the NumPy reference, same
  standard as A1).
- Perf: kernel-level accept (≥15% median, no gate shape regressing >5%)
  at decode S ∈ {2048, 8192, 32768} AND prefill L ∈ {512, 2048} across
  gpt_oss20b/qwen35/gemma4 geometries — prefill included because the
  fused kernel's two passes share the same access pattern and prefill is
  where apa_selective costs 200-550 ms/call.
- ncu corroboration: sectors/request and L1TEX stall share must improve
  materially (directional receipt, not a numeric gate).
- Timing gates in a verified-quiet window only.

## Acceptance

Same as Addendum 1: pass → adopt with receipts; fail → revert, ledger as
negative result. The A3/A4 lesson (theoretical-hazard reduction ≠
wall-clock win) is the null hypothesis this workstream must defeat — but
unlike A3/A4, here the ncu receipt shows the memory pattern IS the
dominant stall, not a latency-hidden side issue.
