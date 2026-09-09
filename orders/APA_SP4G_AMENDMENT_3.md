# APA-SP4G amendment 3 (lead, 2026-09-07) — the per-call comparison: why refine-all ≠ standard on Gemma 4

Same worktree, same rules; a2 receipts committed by the lead
(`artifacts/apa_sp4g/jobs*/`). Card results for your a2 cells:

- `diag_a2_{source,scale,value,rope}_2048`: all PASS and all reproduce
  A's window-0 ppl 52.48049 exactly → the tensors entering both
  branches are identical (a–d eliminated).
- `diag_a2_fp32_A_2048_w0` = 49.921; `diag_a2_fp32_2048_w0` (D in fp32)
  = 53.472; D32 − A32 = +3.55 → precision is NOT the cause (e
  eliminated; note A itself gains 2.5 ppl in fp32, worth a line in the
  report). D bf16 was 53.47 too: D is insensitive to precision, A is not.
- `diag_a2_replay_B_8192_l05`: context bug confirmed; your corrected
  capture works — all 16 B margins (2048 and 8192) PASS bitwise.
- `kernel512` PASS: the SP kernels match dense on synthetic D=512/MQA
  inputs including L<S.
- C cells: `ppl_capture_C_2048_w0` and `margin_a2_C_*` FileNotFoundError
  (the C calibration never ran because the exactness gate is RED).

So: identical inputs, exact kernel on synthetic shapes, precision
ruled out, and D still returns a different attention output than A
inside the model. What remains is the CALL, not the math:
(i) the arguments the harness hands the SP entry on this
architecture — scale (Gemma: 1.0), sinks (none: is a zero/None sink
folded correctly?), the `is_causal` flag and the bottom-right causal
bound under the adapter's 512-token chunked prefill with cache
(L=512 or 64, S_all growing), the 16→1 head mapping, V (K=V shared);
(ii) an op inside the standard global-attention branch that the SP
path has no counterpart for (post-attention scaling, `layer_scalar`
placement, any masking of the last input token, dtype of the output
handed back); (iii) the fusion boundary: exactly which tensor D's
output replaces and whether anything downstream differs (the
`fast_max_seq=0`/`apa_min_context=0` forcing — do A and D run the same
prefill path length-wise?).

## Mission

1. **Per-call same-input comparison, one global layer, one call:**
   capture q, k, v, scale, offsets, mask/causal parameters at the fork
   for layer 5 during window-0 prefill (first chunk AND a later cached
   chunk with L<S_all), run BOTH the standard branch and the SP
   refine-all entry on those exact tensors in isolation, and report
   max-abs and relative-Frobenius between the two outputs per call, plus
   a dense fp32 NumPy reference for both. Also run the SP kernel with a
   sweep of the suspects (scale ×{1.0, D^-0.5}, sink {none, zero},
   causal {on, off}, offset handling) and report which single change
   makes SP match standard, if any. Register the cell(s) first
   (`diag_a3_call_l05_c0`, `diag_a3_call_l05_c1`, `diag_a3_sweep_l05`).
2. **Audit the SP call site** in your model seam against
   `core/gemma4_tc.py`'s standard branch line by line; list every
   argument and every op between the fork and the merge for both
   paths as a table in the ledger BEFORE the card runs, with your
   prediction of the culprit.
3. **If the culprit is a harness argument:** fix it, re-run D window 0,
   and only then unblock C (calibration, ppl, margins). If the culprit
   is a genuine semantic difference between the standard branch and
   the SP kernel (e.g. an op SP lacks), do NOT patch the kernel; report
   it as the finding with the exact op, and register A′ = the standard
   branch minus that op as the comparator so C/E can still be measured
   against a like-for-like reference. Lead decides the rest.
4. Fingerprint amendment, CPU gates, refreshed lead commands and
   blocked-report. No git, no subagents, foreground only, < 10 min per
   call, never kill anything. Prior Art Directive applies.

## Done (verbatim)

1. The fork/merge audit table with your predicted culprit.
2. Diagnostic cell ids and, once the lead runs them, the answer they
   will give (state what each outcome would mean).
3. Fix or A′ registration path, whichever applies, and what it unblocks.
4. CPU gates; lead commands; prior art; RED; process safety; model id
   and effort.
