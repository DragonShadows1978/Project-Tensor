# APA-SP5 amendment 2 (lead, 2026-09-08) — find the OOM wall: resident-mode 16K/24K, and the first streamed-mode rung for the single pass

David, 2026-09-08: "and we're checking OOM levels?" Yes. Your a1
ceiling receipts put standard at OOM by 2,048, both APA arms at FIT
4,096 (chunk 512) and RAIL at 8,192 with 544 MiB still free, and the
cache line (53.6 MiB per 1K tokens) predicts memory exhaustion near
17,800 tokens in resident-expert mode. That prediction has not been
tested. **David authorizes the long lease for this amendment:** up to
1,500 s worker / 1,560 s outer per cell, one cell per lease, flock
`--wait`, foreground, no kills, 30 s cooldown — the same terms as the
Gemma ceiling probe. Do not generalize it.

## Mission

1. **Resident mode, the wall:** `ceiling_long_{B,C}_{16384,24576}`,
   prefill-only from the pinned stream, chunk 512, ascending, stop an
   arm at its first OOM (later rungs become registered non-fits).
   Receipt: fit / OOM / rail, peak and min-free from `cudaMemGetInfo`,
   completed tokens at failure, wall. Register the prediction: 16,384
   FITS both arms (~525 s), 24,576 OOMs both arms at roughly 17.5–18.5K
   completed tokens, single-pass at the same or a slightly later token
   than two-pass. Register yours beside it.
2. **Streamed mode, first same-axis rung:** the H4 ladder's
   two-pass record (`GraftRepository/artifacts/gpt_oss_20b/h4_context_ladder_apa_16k_sampled.json`:
   16,384 tokens, peak 2.0 GB, 1,016 s, streamed forward via
   `scripts/gpt_oss20b_context_ladder.py` / `gpt_oss20b_stream_forward_smoke.py`)
   has no single-pass counterpart. Register `ladder_stream_{B,C}_16384`
   reproducing that ladder's construction exactly (same script path,
   same settings, `--settings apa_r0.15` for B and the SP entry with
   frozen δ=3.16 for C through the same seam, same sampled corpus) so
   the two are comparable to the July/H4 record; peak, wall, fit. 16K
   fits the 1,500 s rail on the H4 wall (1,016 s); 32K+ (37 min, 93 min
   at 64K) does NOT and is registered as long-lease-only cells for a
   separate authorization. If the streamed path cannot host the SP
   entry without editing the READ-ONLY port, say so with the line and
   stop that item; do not patch the port.
3. Fingerprint amendment; refreshed lead commands; ledger; prior art
   (nothing new expected). You may run all of these yourself under the
   lease rules — six cells, roughly 1.5 hours; keep each foreground
   call under 10 minutes by polling the leased job from a fresh call
   rather than sitting inside one, and never use background tasks.

## Done (verbatim)

1. The resident-mode wall per arm (tokens at OOM, peak, min-free) and
   both prediction sets' verdicts.
2. The streamed-mode 16K point for B and C beside the H4 record, or the
   exact reason it could not be built.
3. Fingerprint amendment; lead commands; RED; process safety; model id
   and effort.
