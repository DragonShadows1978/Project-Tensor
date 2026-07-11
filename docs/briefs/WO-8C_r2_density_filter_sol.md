# WO-8C-r2 — density filter, escalated attempt (engine)

You are an implementation agent on Project-Tensor (repo root = cwd).
This is an ESCALATION of a failed order: read
docs/briefs/WO-8C_density_filter.md first — its intent, gates, and
rails carry over except as amended below. Append scope/results to
docs/briefs/WO-8C_r2_ledger.md.

## Why the first attempt failed (lead-verified receipts)

The prior agent's work is preserved read-only in git stash
(`git stash show -p 'stash@{0}'` — inspect freely, do NOT pop) and
failed three registered gates:
- Parity: kernel vs NumPy reference max|Δ| 24–35 (gate ≤1) at
  filter 1/2, depth rel ~1e-3.
- Perf: 209.4ms total @512×512×192 (target ≤12ms; base smooth mode
  is 2.84ms) — it evaluated the box filter PER SAMPLE in the mixed
  band, uncached.
- Benefit marginal as measured: 45° ramp max deviation 0.586 →
  0.523 → 0.513 across filter 0/1/2.

## Amendments (frozen)

1. DESIGN IS MANDATED, not optional: maintain a CACHED filtered
   density field (fp16 or u8-quantized; document precision choice
   and its parity impact) computed ONCE per (grid revision × filter
   level) — a separate device tensor invalidated by the revision
   counter, never rebuilt per frame. Report its memory cost at
   512×512×192 in the ledger.
2. The NumPy reference and kernel must implement the SAME arithmetic
   (same accumulation order/precision class) so the ≤1 |Δ| parity
   band is achievable — if fp16 caching makes ≤1 impossible, use the
   next cheapest representation that achieves it; parity band does
   not move.
3. Sequence your work so the in-tree .so is rebuilt/installed only
   at the END (source + tests first, build last) — another repo
   imports this artifact live.
4. Also register a BENEFIT gate this time: on the 45° ramp, filter=2
   max deviation must be ≤ 0.35 voxel (prior attempt's 0.513 shows
   the naive box filter barely helps — if a plain box filter cannot
   reach 0.35, a wider/Gaussian-weighted kernel within the same
   cached-field design is in scope; if NOTHING within this design
   reaches it, report the best measured value verbatim as a red —
   that is a valid finding that the whole approach is dead).

All original WO-8C gates otherwise stand (blocky + filter=0 byte-
identical, parity band, perf ≤12ms @512, suite green, A/B/C PPMs).

## Rails

As WO-8C: tensor_cuda source additive, tests additive,
artifacts/wo8c_*.ppm, docs/briefs/WO-8C_r2_ledger.md. APA and all
existing op behavior untouchable; Scorch paths read-only. No
subagents/git-write/network/pip. Report: gate table + timing +
memory cost verbatim.
