# APA-SP4G amendment 4 — reference ruling and numerical path

RED / GPU handoff: D32 exactness has not passed. C/E remain blocked; no dtype bug is claimed fixed.

Evidence classes: source inspection, historical card measurements, author CPU tests; new GPU cells only where a current PASS receipt exists.

## The suspect A2 cell

`diag_a2_fp32_2048_w0` = 53.472390467473765, exactly the saved D bf16 PPL. The sealed A2 worker explicitly ran `.float()` on Q/K/Kq/V, called native SP on those values, and returned its output cast to bf16. A32 used the same conversions and final cast, with standard QK/softmax/PV. Both receipts contain 144 comparisons; the native binding forwards tensor data and the launcher dispatches on q.dtype. **The proposed missed-cast cause is not supported by source evidence.** A2 omitted observed dtype fields, so its actual runtime dtype cannot be retroactively asserted.

SP already uses float dots, online softmax and accumulation for bf16 inputs, then rounds once on output. Consequently an unchanged D output after fp32 attention and a bf16 return is plausible. A4 uses explicit `astype("float32")` plus assertions immediately before the native call and on its output; every call is receipted. This fixes the missing verification, with no identified arithmetic correction. Extending fp32 through o_proj would change the registered A32 comparator and is not implemented.

Source receipts: `scripts/apa_sp4g_a2_model.py:85-93`, `tensor_cuda/src/bindings.cpp:222,664-680`, `tensor_cuda/src/ops.cpp:1013`, `tensor_cuda/src/kernels.cu:7357-7505`; hashes in `a4_before.json`, original registration/build manifest.

Additional historical-payload measurement: `a4_suspect_cell_audit.json` verifies that on BOTH A3 layer5 calls, saved SP fp32 cast to bf16 equals saved SP bf16 **bitwise**, and their projected outputs also equal **bitwise** (max abs and rel-F both0). This directly supports precision insensitivity on those calls; it does not prove all-layer equality or retrospectively supply A2 runtime dtype pins.

## Registered reference

Lead prediction: D32 within **0.005** PPL of A32 **49.92117893813879** on window0. Gate additionally requires complete fp32 native input/output pins and the same 144-call schedule as A2 A32. Seat prediction: no source-backed missed cast was found; unchanged arithmetic may repeat D32=53.47239 and fail. If the new gate fails, STOP; do not run propagation/calibration/C/E or adjust precision scope. A32 here means global attention in fp32 with bf16 activations/projections elsewhere, not a whole-model fp32 reference.

The lead changes the active comparator in amendment013 only. Historical bf16 exactness RED receipts remain byte-identical; the bf16 differences are reported as numerical-path sensitivity. Local agreement to a dense fp32 calculation is finite numerical evidence, not exact real arithmetic.

## PPL — QAT weights; APA touches only the 8 global layers

| Population | A bf16 | A32 global attention | B bf16 | D bf16 | C bf16 | E bf16 | C minus A | C minus A32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048_w0 | 52.4804871 | 49.9211789 | 49.4104156 | 53.4723905 | UNRUN / blocked | UNRUN / blocked | UNRUN / blocked | UNRUN / blocked |
| 2048_w1 | 537.973255 | UNRUN / blocked | 571.490253 | 582.576312 | UNRUN / blocked | UNRUN / blocked | UNRUN / blocked | UNRUN / blocked |
| 2048_w2 | 362.996187 | UNRUN / blocked | 374.408275 | 355.943254 | UNRUN / blocked | UNRUN / blocked | UNRUN / blocked | UNRUN / blocked |
| 2048_w3 | 73.4590166 | UNRUN / blocked | 74.7195284 | 74.9501517 | UNRUN / blocked | UNRUN / blocked | UNRUN / blocked | UNRUN / blocked |
| 8192 | 38.8638919 | UNRUN / blocked | 38.5613503 | 39.3822883 | UNRUN / blocked | UNRUN / blocked | UNRUN / blocked | UNRUN / blocked |
| 2048 pooled (4096 scored targets) | 165.644274 | UNRUN / blocked | 167.649292 | 169.788756 | UNRUN / blocked | UNRUN / blocked | UNRUN / blocked | UNRUN / blocked |

Short rows score last1024 of each2048 window; long8192 scores last512; fp64 NLL and original adaptive512-prefix/64-scoring feed unchanged. Raw WikiText with the instruction-tuned model has a known large per-window spread; the approximate165 pooled baseline is not by itself evidence of a port failure.

## A3 per-call card evidence (layer5)

| Comparison | c0 max abs / rel-F | cached c1 max abs / rel-F |
|---|---:|---:|
| SP_vs_A_bf16 | 0.125 / 0.00272521 | 0.125 / 0.002789995 |
| standard_bf16_vs_dense_fp32 | 0.09737968 / 0.002454029 | 0.1072559 / 0.002421874 |
| sp_bf16_vs_dense_fp32 | 0.06243706 / 0.001622895 | 0.06216812 / 0.001629367 |
| SP_vs_A_fp32 | 3.051758e-05 / 4.919979e-07 | 3.528595e-05 / 6.597439e-07 |
| standard_fp32_vs_dense_fp32 | 1.621246e-05 / 2.851905e-07 | 9.536743e-06 / 2.044162e-07 |
| sp_fp32_vs_dense_fp32 | 2.670288e-05 / 4.360286e-07 | 3.242493e-05 / 6.415103e-07 |
| standard_bf16_vs_staged_numpy | 0.0625 / 9.004767e-05 | 0.125 / 0.002335501 |
| SP_vs_A_after_fp32_to_bf16 | 0.0078125 / 3.082075e-05 | 0.0078125 / 3.838542e-05 |
| SP_vs_A_after_fp32_cast_o_proj | 0.0625 / 0.0001491583 | 0.0625 / 0.0001501557 |
| actual_D_vs_A_projected | 0.125 / 0.001953675 | 0.125 / 0.001866105 |

Same Q/K/V bitwise; c0 L512/S512, c1 L511/S1023/offset512. Receipts `jobs_a3/diag_a3_call_l05_c{0,1}.json`; local FP32 agreement and bf16 errors are descriptive per-call measurements.

## Residual propagation

Two separate bf16 workers save all2047 executed rows after layers5,11,17,23,29,35,41,47 (complete block including layer_scalar), plus final norm after compute cast and before logit row slicing. Both use the same weights/ids/schedule. The aggregate computes ||D-A||F/||A||F over all rows, with per-chunk rows also retained; it does not average chunk ratios.

UNRUN / blocked by D32. No propagation numbers inferred from the per-call table.

## C/E execution and G2/G3

Calibration uses B actual-PPL window0 selection fraction and up to12 C trials on that same population (queries0..2046), classical bounded bisection and one frozen delta matched within0.01. Completed matching trials carry the existing receipt and execute no model. C window0/8192 PPL workers also capture their exact per-call tensors/output/native mask. The16 C margin cells reuse A2 bitwise full-call replay before tiling statistics; error/mass/selected populations are all actually executed PPL pairs. E reuses SP2 delta=upward(log(100)+2*upward(empirical max eq)); B and C8 layers×2 lengths are calibration evidence only. This is a conditional finite-key bound, not a universal certification. Clean C decode uses the original unwrapped June protocol with32 synced steps.

G2/G3 rows establish nothing about model quality by themselves.

| Cell | Status | Worker estimate (s) | Error |
|---|---|---:|---|
| ppl_a4_D32_2048_w0 | UNRUN | 90–275 |  |
| diag_a4_propagation_A_2048_w0 | UNRUN | 90–275 |  |
| diag_a4_propagation_D_2048_w0 | UNRUN | 90–275 |  |
| diag_a4_propagation_2048_w0 | UNRUN | 1–30 |  |
| trial_a4_00 | UNRUN | 90–275 |  |
| trial_a4_01 | UNRUN | 90–275 |  |
| trial_a4_02 | UNRUN | 90–275 |  |
| trial_a4_03 | UNRUN | 90–275 |  |
| trial_a4_04 | UNRUN | 90–275 |  |
| trial_a4_05 | UNRUN | 90–275 |  |
| trial_a4_06 | UNRUN | 90–275 |  |
| trial_a4_07 | UNRUN | 90–275 |  |
| trial_a4_08 | UNRUN | 90–275 |  |
| trial_a4_09 | UNRUN | 90–275 |  |
| trial_a4_10 | UNRUN | 90–275 |  |
| trial_a4_11 | UNRUN | 90–275 |  |
| freeze_a4 | UNRUN | 1–15 |  |
| ppl_a4_A32_2048_w1 | UNRUN | 90–275 |  |
| ppl_a4_A32_2048_w2 | UNRUN | 90–275 |  |
| ppl_a4_A32_2048_w3 | UNRUN | 90–275 |  |
| ppl_a4_A32_8192 | UNRUN | 90–275 |  |
| ppl_a4_C_2048_w0 | UNRUN | 90–275 |  |
| ppl_a4_C_2048_w1 | UNRUN | 90–275 |  |
| ppl_a4_C_2048_w2 | UNRUN | 90–275 |  |
| ppl_a4_C_2048_w3 | UNRUN | 90–275 |  |
| ppl_a4_C_2048 | UNRUN | 1–30 |  |
| ppl_a4_C_8192 | UNRUN | 90–275 |  |
| margin_a4_C_2048_l05 | UNRUN | 10–90 |  |
| margin_a4_C_2048_l11 | UNRUN | 10–90 |  |
| margin_a4_C_2048_l17 | UNRUN | 10–90 |  |
| margin_a4_C_2048_l23 | UNRUN | 10–90 |  |
| margin_a4_C_2048_l29 | UNRUN | 10–90 |  |
| margin_a4_C_2048_l35 | UNRUN | 10–90 |  |
| margin_a4_C_2048_l41 | UNRUN | 10–90 |  |
| margin_a4_C_2048_l47 | UNRUN | 10–90 |  |
| margin_a4_C_8192_l05 | UNRUN | 60–270 |  |
| margin_a4_C_8192_l11 | UNRUN | 60–270 |  |
| margin_a4_C_8192_l17 | UNRUN | 60–270 |  |
| margin_a4_C_8192_l23 | UNRUN | 60–270 |  |
| margin_a4_C_8192_l29 | UNRUN | 60–270 |  |
| margin_a4_C_8192_l35 | UNRUN | 60–270 |  |
| margin_a4_C_8192_l41 | UNRUN | 60–270 |  |
| margin_a4_C_8192_l47 | UNRUN | 60–270 |  |
| eq_a4 | UNRUN | 1–30 |  |
| ppl_a4_E_2048_w0 | UNRUN | 90–275 |  |
| ppl_a4_E_2048_w1 | UNRUN | 90–275 |  |
| ppl_a4_E_2048_w2 | UNRUN | 90–275 |  |
| ppl_a4_E_2048_w3 | UNRUN | 90–275 |  |
| ppl_a4_E_2048 | UNRUN | 1–30 |  |
| ppl_a4_E_8192 | UNRUN | 90–275 |  |
| decode_a4_C_2048 | UNRUN | 90–275 |  |
| decode_a4_C_8192 | UNRUN | 90–275 |  |

Author CPU gates: **101 passed, 0 failed, 0 skipped**; mutations **8/8** (threshold0.80). Fingerprint amendment016 SHA `c1e024c9d6924ffb834af389c9dfff53e272c54ca187912f4b953cd747e3ea20`.

## Gates, fingerprints, commands and RED

Immutable registration013 SHA `9fce2b05849eb4e69e8cb5cc2a3434e4c05510677128473aa1f49a4c28af2b1a`. Original registration SHA `099a8bd9e1bb94909a81c110521d2d3a9d642d5fac27d49f6820b97cb3fbbd1e`. Fingerprint amendment016 pins the final A4 closure; original execution files and all historical receipts preserved. Amendment015 corrects the new calibration lookup to the real B PPL receipt global_fraction.fraction, with no algorithm/threshold change. The new regression failed with KeyError before correction; initial100-pass CPU gate/fingerprint014 remain as superseded receipts. Existing A2 captures/error arrays retained. `receipt_audit_A4.json` and `DELIVERY_CHECKS_A4_V2.json` contain the delivery inventory.

CPU dtype regression: `test_a4_fp32_native_dispatch_dtype_pin_and_returned_arm`. Full author baseline, negative cases and copied-source mutation results: `CPU_GATES_A4_V2.json`. These are not blind verification; lead-owned blind review UNRUN.

Each command in `lead_commands.txt` is a separate foreground call. Worker285s/hard290s, outer585s/hard588s, lease wait20s, cooldown30s. Model load planning75–130s within cell estimate90–275s; margin10–90s short/60–270s long, CPU aggregates1–30s. New timings unmeasured. Disk floor12GiB. No >=16384 retries: prior16K PPL/16K–32K ceilings remain RAIL/non-fit within lease, not OOM or a hardware capacity conclusion.

GPU visibility: `a4_device_visibility.json`. This sandbox has no CUDA device; no new model loads/card results claimed. Not claimed fixed: D/A model exactness, a missed-cast root cause, complete numerical propagation cause, C/E quality or long-context capacity. A3 local agreement does not itself unblock C. No git, subagents, background work/waits, process kills/signals, services, model/product/kernel/SP3 edits. Seat gpt-6-astra / reasoning xhigh (`logs/apa_sp4g_a4_r1.log`).

## Prior art

June Gemma port/floor and SP3/SP4G A2/A3 (2026): seam, feeding, replay, provenance, capture and clean decode reused; new work is explicit dtype assertions, residual instrumentation, registered-reference routing and tables. [BLASST, Yuan et al. 2025/2026](https://arxiv.org/abs/2512.12087): inherited running-maximum criterion; [ThriftAttention, Sharratt 2026](https://arxiv.org/abs/2605.23081): precision/softmax-weight motivation; [FlashAttention-2, Dao 2023](https://arxiv.org/abs/2307.08691): inherited online softmax; [TurboQuant, Zandieh et al. 2025](https://arxiv.org/abs/2504.19874): inherited reconstructed Kq. Primary arXiv records checked in this turn; no external benchmark reproduced. SP2 (2026) conditional error-bound delta reused. Standard Frobenius norm, FP precision ablation, bisection, SHA256/NIST (2001), Make/Feldman (1979), DeMillo/Lipton/Sayward (1978) mutation testing (unverified — lead to check: Hints on Test Data Selection). No new attention or optimization algorithm; no prior art known to me for a distinct novel method introduced here.
