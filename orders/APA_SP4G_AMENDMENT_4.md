# APA-SP4G amendment 4 (lead, 2026-09-07) — the kernels agree; the gap is numerical path. Settle the reference and unblock C.

Card results for a3 (receipts under `artifacts/apa_sp4g/jobs*/`, captures under `captures_a3/`):

Layer 5, chunk 0 (L=512, S=512) and cached chunk 1 (L=511, S_all=1023, offset 512), identical q/k/v (parity max-abs 0.0):

| comparison | max-abs | rel-Frobenius |
|---|---:|---:|
| SP vs standard, fp32 | 3.1e-5 / 3.5e-5 | 5e-7 / 7e-7 |
| standard fp32 vs dense fp32 | 1.6e-5 / 1.0e-5 | 3e-7 / 2e-7 |
| SP fp32 vs dense fp32 | 2.7e-5 / 3.2e-5 | 4e-7 / 6e-7 |
| standard bf16 vs dense fp32 | 0.097 / 0.107 | 2.45e-3 / 2.42e-3 |
| SP bf16 vs dense fp32 | 0.062 / 0.062 | 1.62e-3 / 1.63e-3 |
| SP vs standard, bf16 | 0.125 / 0.125 | 2.73e-3 / 2.79e-3 |
| actual D vs A after o_proj (bf16) | — / 0.125 | — / 1.87e-3 |

Your prediction ("no single argument culprit; look at the cast/o_proj
boundary and propagation") is confirmed by the card. Read plainly: the
single-pass kernel and the standard branch compute the same attention
(fp32 agreement to 3e-5), and in the production bf16 path the SP output
is CLOSER to the exact answer than the standard branch's. The 2.5 % ppl
gap at the model level is therefore propagation of two different bf16
rounding paths through a model that is bf16-sensitive: recall
`diag_a2_fp32_A_2048_w0` = 49.92 vs A bf16 52.48 — standard alone moves
2.5 ppl between fp32 and bf16 on window 0.

One a2 receipt is now suspect: `diag_a2_fp32_2048_w0` (D "fp32") = 53.472,
identical to D bf16 53.472. Given a3 shows the SP entry runs fp32 fine
when fed fp32, that cell most likely never put the SP path in fp32.
Verify and say so.

## Mission

1. **Fix and re-run true-fp32 D on window 0** (`ppl_a4_D32_2048_w0`):
   global attention in fp32 on BOTH the SP call and everything the A32
   cell ran in fp32, same tensors, same schedule; assert inside the
   worker that the SP kernel received fp32 (dtype pin in the receipt).
   Register the prediction: D32 within 0.005 ppl of A32 = 49.921.
2. **Propagation capture, window 0, bf16**: record the residual stream
   after each of the 8 global layers (and after the final norm) for A
   and D, report per-layer rel-Frobenius between them, so the report
   can show where the 0.19 % per-call difference becomes a 2.5 % ppl
   difference (`diag_a4_propagation_2048_w0`).
3. **Reference ruling (lead), to implement:** the exactness gate for
   this model is **D32 vs A32 ≤ 0.005** (registered here, a4). The
   bf16 rows stay as measured and are reported as numerical-path
   sensitivity, with the per-call table above. If (1) passes, C/E are
   UNBLOCKED: run calibration, C and E ppl (bf16 production path, all
   four windows and 8192), C margins (2048 and 8192), and C clean decode,
   comparing C against BOTH A bf16 and A32 in the table. If (1) fails,
   stop and report; the lead investigates further.
4. Fingerprint amendment; CPU gates (dtype pin test for the fp32 path);
   refreshed lead commands and blocked-report. No git, no subagents,
   foreground only, < 10 min per call, never kill anything. Prior Art
   Directive applies (nothing new expected).

Disk note for the lead's own bookkeeping: `captures_a2/` 9.4 GB and
`margin_errors_a2/` 34 GB are gitignored; keep them until the report
is written.

## Done (verbatim)

1. The suspect a2 cell: what it actually ran, and the fix.
2. New cell ids (D32, propagation, and the unblocked C/E set), rails,
   estimates; the dtype pin test name.
3. Fingerprint amendment; CPU gates; lead commands; prior art; RED;
   process safety; model id and effort.
