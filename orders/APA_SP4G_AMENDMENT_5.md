# APA-SP4G amendment 5 (lead, 2026-09-07) — compare the calls that were never compared: the 64-query scored blocks

Same worktree, same rules; a4 committed as e475acb; lead receipts in
`artifacts/apa_sp4g/jobs*/`.

Card: `ppl_a4_D32_2048_w0` = 53.47239 (dtype pins complete: native
inputs fp32, native output fp32, returned bf16 pre-o_proj), A32 =
49.92118, |Δ| = 3.551 → gate RED, and it stopped every successor
including the propagation captures, which are diagnostics and should
never have depended on the gate. Ungate them (item 3).

The contradiction to resolve: a3 shows SP and standard agree to 3e-5 in
fp32 on layer 5's two PREFILL chunks (L=512, S=512 and L=511,
S_all=1023) with identical inputs, yet the full window disagrees by 3.55
ppl even with both in fp32. Layer 5 is the first global layer, so its
inputs are identical in both runs; a 3e-5 output difference cannot
grow into 3.55 ppl across 8 layers. Therefore the calls that differ
are the ones a3 did NOT compare: each window makes 144 native calls =
8 layers × (2 prefill chunks + 16 cached blocks of L=64 over the 1024
scored positions). The scored positions — the only ones perplexity
sees — are attended in L=64 blocks with S_all from 1088 to 2048, a
shape the a3 captures never exercised and the kernel512 pin covered
only synthetically (L=7, S=11; L=1, S=4097).

## Mission

1. **Per-call comparison on scored blocks:** capture and compare, on
   identical inputs, layer 5 cached block 0 (L=64, S_all=1088) and
   block 15 (L=64, S_all=2048), plus layer 47 block 15, exactly as a3
   did (SP vs standard vs dense fp32, in fp32 and bf16; parity of q/k/v;
   the causal bound, `position_offset`, `kq_count` and the launcher's
   dispatch choice — prefill SP kernel vs split-K/decode-shaped SP —
   recorded per call). Cells `diag_a5_call_l05_b00`, `diag_a5_call_l05_b15`,
   `diag_a5_call_l47_b15`. Register the prediction: these DISAGREE
   beyond 1e-3 rel-Frobenius in fp32, and the disagreement is in the
   SP path's handling of L ≪ S_all with a cache (offset / causal bound /
   dispatch), not in standard.
2. **If they disagree:** name the mechanism from the captured
   arguments, fix it in the harness seam if it is a call-argument
   error, or report it as an SP launcher defect for L ≪ S_all if it
   is the kernel path (do NOT patch kernels; the lead decides), then
   re-run D32 window 0. If they agree: report that, and the propagation
   capture (item 3) becomes the only remaining lead.
3. **Ungate** `diag_a4_propagation_{A,D}_2048_w0` and the aggregate
   from the D32 gate (they depend on A32 and D32 having RUN, not
   passed); the lead runs them regardless of item 1's outcome.
4. Fingerprint amendment; CPU gates (a test that the capture shapes
   include an L=64 cached block); refreshed lead commands and
   blocked-report. No git, no subagents, foreground only, < 10 min per
   call, never kill anything. Prior Art Directive applies.

## Done (verbatim)

1. The three cell ids, their captured shapes and dispatch paths, and
   your registered prediction.
2. What you changed (fix or none) and why; whether D32 is re-runnable.
3. The ungated propagation cells.
4. Fingerprint amendment; CPU gates; lead commands; prior art; RED;
   process safety; model id and effort.
