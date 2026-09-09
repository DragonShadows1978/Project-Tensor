# APA-SP4G amendment 6 (lead ruling, 2026-09-07) — exactness is a per-call property on this model; register the noise floor; unblock C and E

Card results for a5 (`artifacts/apa_sp4g/jobs*/`, `captures_a5/`,
`propagation_a5/`):

- Scored cached blocks, identical inputs, fp32: SP vs standard relF
  7e-7 at (L=64, S_all=1087), and at (64, 2047) on layers 5 and 47;
  bf16: SP closer to dense fp32 than standard is, again. With a3 that
  is five call shapes, all agreeing to 3e-5 max-abs.
- Propagation, window 0, bf16, A vs D residual after each global
  layer (relF / max-abs): L5 0.0048/0.25 → L11 0.0074/0.25 → L17
  0.0103/2.45 → L23 0.038/67.5 → L29 0.125/40.7 → L35 0.129/32.1 →
  L41 0.145/24.5 → L47 0.074/3.47 → final norm 0.119/55.5. A ≈ 0.3 %
  rounding-level difference at the first global layer is amplified
  ~30× by layer 29.
- `ppl_a4_D32_2048_w0` = 53.47239 (pins complete), bit-identical to
  D bf16, because the SP kernel accumulates in fp32 internally and
  rounds once on return; A32 = 49.92118 vs A bf16 = 52.48049: the
  standard branch alone moves 2.56 ppl with its own rounding path.

## Ruling (lead; implement as an immutable amendment JSON, registration untouched)

1. **Exactness for this model is a PER-CALL property and it PASSES:**
   the refine-all single pass equals the standard branch to ≤ 3.5e-5
   max-abs / ≤ 7e-7 rel-Frobenius in fp32 on identical inputs across
   the five captured shapes (prefill 512, cached 511, cached 64 at
   S_all 1087 and 2047, layers 5 and 47). Record this as the D/A
   exactness verdict in RESULTS.md with the per-call table.
2. **Model-level perplexity on Gemma 4 QAT INT4 carries a registered
   numerical-path noise floor of ±2.56 ppl on window 0** (= |A32 − A
   bf16|, the standard branch's own rounding sensitivity), and the
   propagation table is the mechanism. Every model-level ppl
   difference between arms on this model is reported against that
   floor; differences inside it are "not resolvable on this model",
   not wins or losses. The 0.005 gate stays as written for the record
   (RED) and is annotated as inapplicable here for the reason above.
3. **One more receipt to make the ruling airtight (cheap):**
   `diag_a6_propagation_A32_vs_A_2048_w0` — the same per-layer
   residual comparison between standard-fp32 and standard-bf16. The
   prediction: the same amplification profile (relF growing to ~0.1 by
   layer 29). If it shows that, the ruling stands on its own evidence;
   if A32 vs A propagates flat, say so and stop — the lead was wrong.
4. **Unblock C and E** under the ruling: calibration (match B's
   realised 0.152 on the global layers), C and E ppl on all four
   windows and 8192 (bf16 production path), C margins at 2048 and
   8192 on the 8 global layers, C clean decode at 2048 and 8192. Table
   columns: A bf16, B, C, D, E, with the ±2.56 floor stated in the
   header. P2 is evaluated as "C − B inside the floor" or not.
5. Fingerprint amendment; CPU gates; refreshed lead commands and
   blocked-report. No git, no subagents, foreground only, < 10 min per
   call, never kill anything. Prior Art Directive applies; the
   chaotic-amplification observation is worth a literature line if you
   know one (error propagation in deep residual nets under low
   precision) — "unverified, lead to check" is acceptable.

## Done (verbatim)

1. The amendment JSON path/sha recording the ruling and the floor.
2. The A32-vs-A propagation cell id and your prediction beside mine.
3. The unblocked C/E cell list with estimates; lead commands.
4. Fingerprint amendment; CPU gates; prior art; RED; process safety;
   model id and effort.
