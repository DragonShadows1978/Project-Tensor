# APA-SP3 amendment 3 (lead, 2026-09-06) — capture cell RED on the card: `tc.cat` of selection masks

Lead GPU run so far (PROTOCOL-2, bits 4; receipts committed under
`artifacts/apa_sp3/jobs/`): `kernel96` PASS (D=96 pins, all dtypes,
prefill + split-K); `g0_A_1/2` = 8.655828 (bit-identical across two
fresh processes); `g0_B_1/2` = 8.778802 (likewise); `g0` PASS
(B − A = +0.123, inside the ±0.3 prediction); `ppl_b4_D_1024` =
8.655352 (refine-all SP vs A: −0.0005 → lead P1 HIT);
`ppl_b4_A_8192` RED `cudaMalloc out of memory` (registered non-fit,
amendment 1 — leave it RED, do not retry); `ppl_b4_B_8192` = 10.4036
(512 targets, peak 4,446 MiB, 46 s).

Then **`capture_b4_B_1024` RED** and every remaining cell depends on
it. Stored traceback (receipt `capture_b4_B_1024.json`):

```
scripts/apa_sp3_model.py:227 in blend
    self.capture(q, k, kq, tc.cat(masks, dim=2), causal, 'B_native_blend', bulk_pins)
tensor_cuda/__init__.py:1132 in cat -> _C.cat(list(tensors), dim)
RuntimeError: op supports float32/float16/bfloat16 only
```

The engine's `cat` accepts float dtypes only; your selection masks are
not float. Fix in `scripts/apa_sp3_model.py` only (concatenate on the
host as numpy, or cast the masks to a float dtype before `tc.cat` and
back to uint8 in `capture`, your choice — state which, and make sure
the packed-mask semantics, `selected` counts and the `test_capture_*`
pins are unchanged). Note also what the traceback shows about arm B on
this adapter at S=1024: the two-pass path is `_cublas_blend_attention`
(the adapter's native blend), which you hooked as `B_native_blend`;
confirm in the ledger that C/D/E are hooked at the SAME call site with
the SAME q/k/kq/v tensors, and that the fraction you match for C is the
fraction this blend path actually refines.

Same worktree, same boundaries, same rules as the order and amendments
1–2 (no git, no subagents, foreground only, < 10 min per call, never
kill anything, registration immutable, amendment JSONs separate,
existing kernel bodies byte-identical). Your sandbox has no GPU: fix,
add a CPU test that reproduces the dtype failure against a stub `cat`
that rejects non-float, re-run all CPU gates, refresh the blocked
report and `lead_commands.txt`. The existing PASS receipts must remain
valid: if your change alters any file the receipt fingerprint covers,
say so explicitly and say which receipts become stale by your own
rule (the lead will decide whether to re-run them, never you).

## Done (verbatim)

1. The fix (file/lines) and the repro test name.
2. Which existing receipts your fingerprint rule marks stale, if any.
3. The B call-site confirmation for C/D/E hooking and the fraction
   definition for C.
4. CPU gate results; refreshed blocked-report; prior art unchanged or
   added; RED; process safety; model id and reasoning effort.
