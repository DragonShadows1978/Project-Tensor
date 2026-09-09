# APA-SP4G amendment 2 (lead, 2026-09-07) — refine-all ≠ standard on Gemma 4; margin replay not bitwise; 16K rows rail-bound

Same worktree, same rules; your a1 committed as e4008f4, the lead's
receipts as the commit after it (`artifacts/apa_sp4g/jobs_a1/`).

## What the card says (PROTOCOL-G, QAT INT4, bulk 4)

| cell | A standard | B two-pass r=0.15 (realised 0.152) | D refine-all |
|---|---:|---:|---:|
| ppl@2048 ×4 windows (2048 targets ×4) | 165.644 | 167.649 | 169.789 |
| per window | 52.5 / 538.0 / 363.0 / 73.5 | 49.4 / 571.5 / 374.4 / 74.7 | 53.5 / 582.6 / 355.9 / 75.0 |
| ppl@8192 last-512 | 38.864 | 38.561 | 39.382 |
| clean decode ms/token @2048 / @8192 | 31.3 / 73.3 | 38.7 / 87.8 | — |
| ceiling 4096 / 8192 | fit / fit | fit / fit | fit / fit |

**1. `exactness_2048` and `exactness_8192` RED:** D − A = +4.14 ppl
(+2.5%) at 2048 and +0.52 (+1.3%) at 8192. On MiniCPM3 the same D pin
held to 0.0005. A refine-all single pass computes every key exactly,
so this is NOT the SP rule; something differs between what A attends
over and what D attends over. Register hypotheses BEFORE touching
anything, then test each on the card via a registered diagnostic cell
(≤ 285 s each), in this order of my suspicion:
   (a) **key/value source parity**: A reads the standard cache (bf16 K,
   V) while the APA arms read the KVRing's reconstructed/quantized
   K (and INT8 V, `QUANT_V`) — if so, D ≠ A is storage precision, and
   the honest fix is a registered arm A′ = standard attention over the
   SAME reconstructed K/V the APA arms see (the reference the SP3
   report calls "same tensors"), with A′ − A reported as the storage
   cost;
   (b) attention scale (Gemma 4 uses 1.0, not D^-0.5) and any
   attention-logit softcap the standard path applies that the SP
   launcher does not;
   (c) the shared K=V global projection: what V does the SP entry
   receive vs what the standard path uses, per layer;
   (d) p-RoPE / qk-norm order relative to where the arms fork;
   (e) bf16 accumulation differences (your S2) — quantify by an
   fp32-compute run of one window if the engine allows; 4 ppl points
   is far beyond rounding on MiniCPM3's evidence, say so if it isn't.
   Deliver: the hypothesis table with a cell per row, and a registered
   D/A tolerance rule for THIS model only after the cause is named.
   Do NOT widen the 0.005 gate; if A′ is the right reference, the gate
   is D vs A′.

**2. `margin_B_8192_l*` all RED `NATIVE_REPLAY_NOT_BITWISE`
(`apa_sp4g_metrics.py:57`).** Find the nondeterminism (atomics in the
D=512 kernel's reduction? cuBLAS blend path? band replay context?).
If the native path is legitimately non-bitwise-reproducible, register
a tolerance replay (max-abs / relative, stated) with the reason; if it
is a replay-context bug (wrong prefix, wrong RoPE offset, wrong mask
for the band), fix it and keep bitwise. Either way the margin numbers
must come from the SAME selection the ppl arm used.

**3. 16384 rows RED "worker terminated"** = rail (12B prefill at 16K
exceeds 285 s), not OOM: mark `ppl_*_16384` and the 16K/24K/32K
ceilings as RAIL non-fits in the report, do not retry; the lead can
grant a long lease later.

**4. Context, not a task:** ppl ≈ 165 on raw wikitext with the `-it`
model is the known regime (June ledger saw 121 in a refine sweep); the
per-window spread (52 → 538) is the -it model's template-boundness.
State this in RESULTS.md so nobody reads 165 as a port failure.

CPU gates for every change, fingerprint amendment (which existing
receipts stay valid), refreshed lead commands, blocked-report. No git,
no subagents, foreground only, < 10 min per call, never kill anything.
Prior Art Directive applies.

## Done (verbatim)

1. Hypothesis table for D ≠ A with the diagnostic cell ids; A′ arm if
   registered; the tolerance/gate rule you propose (lead decides).
2. The replay finding (bug vs nondeterminism) and what you changed.
3. Rail non-fit bookkeeping; RESULTS.md context note.
4. Fingerprint amendment; CPU gates; lead commands; prior art; RED;
   process safety; model id and effort.
