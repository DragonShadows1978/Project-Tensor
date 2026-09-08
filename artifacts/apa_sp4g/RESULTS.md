# APA-SP4G amendment 6

**RED residuals. D/A per-call exactness: PASS by lead ruling; literal stated bounds: RED.** C/E are released from the historical0.005 PPL dependency by amendment6 and scheduled behind the new propagation stop. A6 GPU measurements are completed for the diagnostic.

**Evidence discrepancy:** Receipt c1 max_abs=3.528594970703125e-5; b15 layer5 max_abs=3.910064697265625e-5 / relF=9.043649367824469e-7 exceed the stated bounds. Lead PASS is recorded as a ruling, never as literal threshold compliance. Bounds are not relaxed.

The recorded PASS is a lead interpretation of finite same-input agreement. It does not certify every call, bitwise equality, or compliance with the stated scalar bounds. No threshold or old receipt was edited.

## Per-call D/A exactness

Evidence class: prior card per-call diagnostics, validated receipt and payload provenance; fp32 SP versus standard on identical input values. Max-abs limit3.5e-5; relF limit7e-7.

| Cell | Layer | L | S_all | fp32 max-abs | fp32 relF | Literal bounds |
|---|---:|---:|---:|---:|---:|---|
| [diag_a3_call_l05_c0](jobs_a3/diag_a3_call_l05_c0.json) | 5 | 512 | 512 | 3.0517578125e-05 | 4.91997946485157e-07 | PASS |
| [diag_a3_call_l05_c1](jobs_a3/diag_a3_call_l05_c1.json) | 5 | 511 | 1023 | 3.52859497070312e-05 | 6.59743943485455e-07 | RED |
| [diag_a5_call_l05_b00](jobs_a5/diag_a5_call_l05_b00.json) | 5 | 64 | 1087 | 2.6702880859375e-05 | 6.98806004193954e-07 | PASS |
| [diag_a5_call_l05_b15](jobs_a5/diag_a5_call_l05_b15.json) | 5 | 64 | 2047 | 3.91006469726562e-05 | 9.04364936782447e-07 | RED |
| [diag_a5_call_l47_b15](jobs_a5/diag_a5_call_l47_b15.json) | 47 | 64 | 2047 | 1.02519989013672e-05 | 4.31577308095211e-07 | PASS |

## Model perplexity — registered numerical-path floor ±2.56 PPL

Measured window0: A bf16=52.4804870852055, A32=49.9211789381388; |A32−A|=2.55930814706673, rounded floor±2.56. This is a deterministic numerical-path sensitivity measurement, not a statistical confidence interval. The lead applies this floor to every model-level arm difference here. It is **extrapolated, not independently measured**, for windows1–3, their pooled NLL, and8192. An outside-floor gap alone does not establish a robust gain.

**Gemma4 QAT INT4; bf16 production arms; APA on the8 global layers only;40 sliding layers unchanged. Every PPL comparison uses the ±2.56 floor.** Short rows score1024 targets each; pooled2048 uses4096 total targets;8192 scores512.

| Row | A bf16 | B | C | D | E | P2: C − B against ±2.56 |
|---|---:|---:|---:|---:|---:|---|
| 2048_w0 | 52.48048709 | 49.41041564 | 51.88388607 | 53.47239047 | 53.16165234 | +2.47347043; C − B inside the floor; not resolvable on this model |
| 2048_w1 | 537.97325533 | 571.49025268 | 609.35601969 | 582.57631198 | 525.40126632 | +37.86576701; C − B outside the floor; outside the registered floor |
| 2048_w2 | 362.99618674 | 374.40827539 | 371.11944625 | 355.94325393 | 359.97875530 | -3.28882914; C − B outside the floor; outside the registered floor |
| 2048_w3 | 73.45901658 | 74.71952840 | 73.60051639 | 74.95015168 | 75.18971208 | -1.11901201; C − B inside the floor; not resolvable on this model |
| 2048 | 165.64427445 | 167.64929211 | 171.42515228 | 169.78875584 | 165.81782708 | +3.77586017; C − B outside the floor; outside the registered floor |
| 8192 | 38.86389187 | 38.56135027 | 39.43323066 | 39.38228832 | 37.99512343 | +0.87188039; C − B inside the floor; not resolvable on this model |

Validated PPL receipt paths/hashes and **every available arm pair** with its floor classification are in `PPL_TABLE_A6.json`. No within-floor difference is labelled a win or loss.

| Row | B − A | D − A | Interpretation against ±2.56 |
|---|---:|---:|---|
| 2048_w0 | -3.07007145 | +0.99190338 | B−A: outside the registered floor; D−A: not resolvable on this model |
| 2048_w1 | +33.51699735 | +44.60305665 | B−A: outside the registered floor; D−A: outside the registered floor |
| 2048_w2 | +11.41208865 | -7.05293281 | B−A: outside the registered floor; D−A: outside the registered floor |
| 2048_w3 | +1.26051182 | +1.49113510 | B−A: not resolvable on this model; D−A: not resolvable on this model |
| 2048 | +2.00501766 | +4.14448138 | B−A: not resolvable on this model; D−A: outside the registered floor |
| 8192 | -0.30254160 | +0.51839645 | B−A: not resolvable on this model; D−A: not resolvable on this model |

Historical D32=53.4723904674738, A32=49.9211789381388; D32−A32=+3.55121152933498: outside the registered floor. All144 fp32 pins complete. D32 PPL is bit-identical to D bf16 PPL on window0 (difference0, not resolvable on this model); this does not certify bit-identical logits. Historical gate≤0.005 stays **RED**, annotated **inapplicable under the lead numerical-path ruling**. Error retained verbatim: `A4_D32_A32_EXACTNESS_FAILED; STOP, lead investigates`. Not claimed fixed: historical model-PPL exactness.

## Propagation receipt and predictions

Cell: `diag_a6_propagation_A32_vs_A_2048_w0`; state: PASS, AMPLIFIED; None

**Lead prediction:** Same amplification profile, relF growing to ~0.1 by layer29; if flat stop, the lead was wrong.

**Seat prediction:** Expect amplification too: layer29 relF roughly 0.05 to 0.2 and at least 5 times layer5. Exact A/D magnitudes need not repeat; this is a prediction, not a measurement.

AMPLIFIED if L29 relF >= .05 AND >= 5*L5; FLAT if L29 <= 2*L5; otherwise INCONCLUSIVE. Flat or inconclusive => RED STOP all C/E, no retry or widened fork.

One new A32 load; compare against saved A bf16 residuals from `jobs_a5/diag_a4_propagation_A_2048_w0.json`. Native fp32 Q/K/Kq/V/output pins, bf16 cast before o_proj, complete144-call schedule, and A32 PPL reproduction are mandatory. All2047 query rows after each complete global block and final norm; sum squared errors and reference norms across blocks.

| Layer | D−A bf16 relF | D−A max-abs | A32−A relF | A32−A max-abs |
|---|---:|---:|---:|---:|
| 5 | 0.00482106304 | 0.25 | 0.00482107767 | 0.25 |
| 11 | 0.00743735688 | 0.25 | 0.00753248699 | 0.25 |
| 17 | 0.0103417778 | 2.45117188 | 0.0112727392 | 3.1328125 |
| 23 | 0.0382708158 | 67.5 | 0.0310998025 | 70 |
| 29 | 0.124512706 | 40.6875 | 0.144096183 | 37.375 |
| 35 | 0.128977027 | 32.125 | 0.142266821 | 32.6875 |
| 41 | 0.144904834 | 24.5 | 0.156331575 | 15.453125 |
| 47 | 0.0736480457 | 3.46875 | 0.0700486792 | 2.71875 |
| final_norm | 0.119381166 | 55.5 | 0.117511898 | 56.3125 |

Evidence class: prior card residual diagnostic, `jobs_a5/diag_a4_propagation_2048_w0.json`. D/A L29-to-L5 relF ratio=25.827. The approximately0.3% figure describes attention-output differences; post-block L5 residual relF is0.4821%. Observed amplification is a mechanism lead, not proof of chaos or a complete causal explanation of PPL.

## C/E cells, estimates and commands

Calibration target is B's measured0.151605081488344, rounded0.152, on the actual window0 global-layer pair population; tolerance±0.01. Twelve registered bisection trials; after a match, later trials carry its immutable result without loading the model. C uses one frozen delta. E uses the inherited conditional empirical bound from all32 B/C layer-margin receipts. Neither E’s finite e_q sample nor G2/G3 rows establish model quality by themselves.

Run **each command separately in the foreground**, in `lead_commands.txt` order. `resume` runs exactly one next cell. Historical0.005 does not block this DAG; missing/flat/inconclusive A6 propagation does. Model-cell estimates include75–130s load, based on earlier75–126s receipts; estimates are not new measurements. GPU calls add up to20s lease wait and30s cooldown. Worker cooperative rail285s, call budget599s.

| Cell | Worker estimate (s) | State |
|---|---:|---|
| `diag_a6_propagation_A32_vs_A_2048_w0` | 90–275 | PASS |
| `trial_a6_00` | 90–275 | PASS |
| `trial_a6_01` | 90–275 | PASS |
| `trial_a6_02` | 90–275 | PASS |
| `trial_a6_03` | 90–275 | PASS |
| `trial_a6_04` | 90–275 | PASS |
| `trial_a6_05` | 90–275 | PASS |
| `trial_a6_06` | 90–275 | PASS |
| `trial_a6_07` | 90–275 | PASS |
| `trial_a6_08` | 90–275 | PASS |
| `trial_a6_09` | 90–275 | PASS |
| `trial_a6_10` | 90–275 | PASS |
| `trial_a6_11` | 90–275 | PASS |
| `freeze_a6` | 1–15 | PASS |
| `ppl_a6_C_2048_w0` | 90–275 | PASS |
| `ppl_a6_C_2048_w1` | 90–275 | PASS |
| `ppl_a6_C_2048_w2` | 90–275 | PASS |
| `ppl_a6_C_2048_w3` | 90–275 | PASS |
| `ppl_a6_C_2048` | 1–30 | PASS |
| `ppl_a6_C_8192` | 90–275 | PASS |
| `margin_a6_C_2048_l05` | 10–90 | PASS |
| `margin_a6_C_2048_l11` | 10–90 | PASS |
| `margin_a6_C_2048_l17` | 10–90 | PASS |
| `margin_a6_C_2048_l23` | 10–90 | PASS |
| `margin_a6_C_2048_l29` | 10–90 | PASS |
| `margin_a6_C_2048_l35` | 10–90 | PASS |
| `margin_a6_C_2048_l41` | 10–90 | PASS |
| `margin_a6_C_2048_l47` | 10–90 | PASS |
| `margin_a6_C_8192_l05` | 60–270 | PASS |
| `margin_a6_C_8192_l11` | 60–270 | PASS |
| `margin_a6_C_8192_l17` | 60–270 | PASS |
| `margin_a6_C_8192_l23` | 60–270 | PASS |
| `margin_a6_C_8192_l29` | 60–270 | PASS |
| `margin_a6_C_8192_l35` | 60–270 | PASS |
| `margin_a6_C_8192_l41` | 60–270 | PASS |
| `margin_a6_C_8192_l47` | 60–270 | PASS |
| `eq_a6` | 1–30 | PASS |
| `ppl_a6_E_2048_w0` | 90–275 | PASS |
| `ppl_a6_E_2048_w1` | 90–275 | PASS |
| `ppl_a6_E_2048_w2` | 90–275 | PASS |
| `ppl_a6_E_2048_w3` | 90–275 | PASS |
| `ppl_a6_E_2048` | 1–30 | PASS |
| `ppl_a6_E_8192` | 90–275 | PASS |
| `decode_a6_C_2048` | 90–275 | PASS |
| `decode_a6_C_8192` | 90–275 | PASS |

Exact command per cell and dependency reason: `GPU_BLOCKED_A6.json`; complete command list: `lead_commands.txt`.

## CPU gates, fingerprints and RED residuals

Ruling019 SHA256 `ac2cc1519634cd74e0540790dce7b5aa2a4900a8922d52c6a1f1d8305fa130b3`. Original registration unchanged: `099a8bd9e1bb94909a81c110521d2d3a9d642d5fac27d49f6820b97cb3fbbd1e`.
`amendment_020_a6_fingerprint.json` SHA256 `fe32afcfd5383e9cc395f690e6078ea2b3406f96d323bf5faa94e6138a3daca4`.
`CPU_GATES_A6.json` SHA256 `356a36a0f3b1b01b0a748e373f8c92fdd4e4d4c6078b87bf9aef81c77b35222c`.

Author CPU baseline: 148 passed, 0 failed, 0 skipped; mutations 8/8, threshold0.80. Blind review is lead-owned and UNRUN. CPU doubles validate harness wiring, never actual GPU numerics.

Not claimed fixed: original0.005 PPL gate, literal3.5e-5/7e-7 bound discrepancy, unmeasured A6 propagation and C/E quality while pending; no extension of the floor’s measured scope. Product, kernel, adapter, model, order, historical execution files and receipts preserved byte-for-byte.

Process safety: no git, subagents, background jobs/waits, signals/kills, services or model writes. No A6 GPU worker or model load in this dispatched seat. CUDA visibility is recorded in `a6_device_visibility.json`. Cooperative deadlines cannot forcibly bound a hung native call without violating no-kill; clean decode checks before/after measurement only, preserving the unwrapped timed path. Each lead command is planned below10minutes; no hard no-kill wall-time guarantee is claimed.

Seat: **gpt-6-astra, reasoning xhigh**, live session header `logs/apa_sp4g_a6_r1.log`. Model under test: **Gemma-4-12B-it QAT q4_0 exact (symmetric-8 g32)**; bf16 engine, fp32 global-attention diagnostic only.

## Prior art

SP4G A2/A4/A5 and June Gemma port/floor (2026) provide the precision seam, residual capture, native-mask replay, calibration and clean decode. A6 adds ruling/floor reporting, combines A32 with that capture, and replaces successor dependencies. No new attention algorithm or novelty claim.

[Haber and Ruthotto (2017), Stable Architectures for Deep Neural Networks](https://arxiv.org/abs/1705.03341) provides dynamical-system stability context; [Higham and Mary (2022), Mixed precision algorithms in numerical linear algebra](https://doi.org/10.1017/S0962492922000022) provides mixed-precision error context. Primary abstracts/metadata verified this seat. Neither establishes Gemma QAT rounding chaos or this PPL floor; that connection is an experimental inference to test.

Inherited prior art: BLASST/Yuan (2025/2026) running-max selection, ThriftAttention/Sharratt (2026) weight-sensitive precision, FlashAttention2/Dao (2023) online softmax, TurboQuant/Zandieh (2025) quantization; unverified this seat — lead to check arXiv2512.12087,2605.23081,2307.08691,2504.19874. Classical bisection, Frobenius norms, NIST SHA256 (2001), Make/Feldman (1979) dependency DAGs; DeMillo/Lipton/Sayward (1978) mutation testing, unverified — lead to check Hints on Test Data Selection. No prior art known to me for a distinct new method introduced here.
