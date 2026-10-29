# APA-SPQ8 report

**G1 RED; G2 RED.** Arm B: NOT ESTABLISHED (prerequisite RED). Arm C: NOT ESTABLISHED (prerequisite RED).
Supplemental G1 with the operating configuration (dense layer0): Qwen3-1.7B PASS, Qwen3-4B PASS. Original registered G1 verdict retained.
Registration SHA-256: `a0d2d6850b92c846ed5332082ce180233a3f4a32dde59f508255f6b71990e26f`. Seat **gpt-6.1-sol, reasoning high**. Evidence: actual packed engine kernels in HF model decoding, exact model prefill; bf16 D=128 TQ4 (4-bit indices + fp32 norm, 4.25 bits/element), row carry, δ=8.1, layer 0 exact unless G1 all-refined.

Teacher-forced distributions are saved as full fp32 logits at every step, paired on the same corpus tokens. All confidence intervals below are 95% percentile bootstrap over six windows (10,000 paired draws, fixed seed); no token-independent CI. Free runs generate exactly 128 tokens, without EOS stopping. First divergence is zero-based, with 128 for no divergence.

## Qwen3-1.7B

G1 (window 0, 256 decoded/scored tokens): `{"G1": "RED", "S0": 7936, "tokens": 256, "mean_kl": 0.001003519111400224, "top1": 0.984375, "delta_nll": 0.0008420462776488058, "full_mask_checks": 7168}`.
G2 (window 0, 256 steps, arm a, δ=8.1 layer 0 exact): RED; maximum per-step fraction discrepancy 1.1648332191721567e-06; maximum per-layer raw deviation ratio 1.3601654098143918.

| arm | windows / decoded tokens | S0 | mean KL (nats), CI | top-1 fraction, CI | ΔNLL (nats/token), CI |
|---|---:|---:|---|---|---|
| a | 6 / 1536 | [7936] | 0.00113148 [0.000770461, 0.00169089] | 0.98763 [0.98112, 0.992839] | -3.00532e-05 [-0.00385397, 0.00386563] |
| b | 6 / 1536 | [7936] | 0.000998552 [0.000733497, 0.00136781] | 0.981771 [0.979167, 0.983724] | -0.00105114 [-0.00538452, 0.00297548] |
| c | 6 / 1536 | [7936] | 0.000962105 [0.000692734, 0.00131535] | 0.986979 [0.981771, 0.992188] | -0.00219352 [-0.00478979, 0.000935679] |

| paired vs a | ΔKL, CI | Δtop-1, CI | ΔΔNLL, CI | relative ΔNLL gap | KL / top1 / NLL / free |
|---|---|---|---|---|---|
| b | -0.000132926 [-0.000317432, -2.67103e-05] | -0.00585938 [-0.0104167, -0.000651042] | -0.00102108 [-0.00275221, 0.000730297] | 33.9759 | False / False / False / True |
| c | -0.000169373 [-0.000391431, -2.83888e-05] | -0.000651042 [-0.0078125, 0.00651042] | -0.00216347 [-0.00434378, 0.000217417] | 71.988 | False / True / False / True |

| arm | prompts / generated tokens | median divergence vs exact | mean token match vs exact | median divergence vs a | mean token match vs a |
|---|---:|---:|---:|---:|---:|
| exact | 8 / 1024 | 128.0 | 1.0 | 61.5 | 0.6181640625 |
| a | 8 / 1024 | 61.5 | 0.6181640625 | 128.0 | 1.0 |
| b | 8 / 1024 | 71.5 | 0.640625 | 46.5 | 0.484375 |
| c | 8 / 1024 | 77.5 | 0.6103515625 | 51.5 | 0.5703125 |

| flip arm vs a | sampled layer steps / decisions | flips | min / median / p90 / p99 / max consequence |
|---|---:|---:|---|
| b | 243 / 31356288 | 28 | 2.09399e-08 / 1.19331e-05 / 8.52412e-05 / 0.000318082 / 0.000332662 |
| c | 243 / 31356288 | 28 | 2.09399e-08 / 2.05731e-05 / 0.000143674 / 0.00102812 / 0.00128535 |

Supplemental dense-layer0 G1 (same w0, S0=7936, 256 scored tokens; layers1..N-1 all-refined): `{"G1": "PASS", "S0": 7936, "tokens": 256, "mean_kl": 0.0008008560453219397, "top1": 0.98828125, "delta_nll": -0.004171854845568951, "full_mask_checks": 6912, "layer0": "exact"}`. This does not replace the original registered result.

| G2 layer (a, w0, 256 steps) | kernel dev | SPQ7 fp32 dev | raw ratio | bf16-ref ratio (diagnostic) | raw pass |
|---|---:|---:|---:|---:|---|
| 1 | 0.00964301 | 0.00950555 | 1.01446 | 1 | True |
| 2 | 0.0104231 | 0.0102873 | 1.0132 | 0.999997 | True |
| 3 | 0.00782688 | 0.00764696 | 1.02353 | 0.999997 | True |
| 4 | 0.00652881 | 0.00630994 | 1.03469 | 0.999966 | True |
| 5 | 0.0200522 | 0.0199833 | 1.00344 | 0.999992 | True |
| 6 | 0.00636573 | 0.0061441 | 1.03607 | 1.00003 | True |
| 7 | 0.00594215 | 0.0057132 | 1.04007 | 1 | True |
| 8 | 0.00389329 | 0.00351813 | 1.10663 | 1 | True |
| 9 | 0.00373749 | 0.00335355 | 1.11449 | 1 | True |
| 10 | 0.00626274 | 0.00603532 | 1.03768 | 1 | True |
| 11 | 0.00282526 | 0.00229517 | 1.23096 | 1 | True |
| 12 | 0.00366068 | 0.00326323 | 1.1218 | 0.999998 | True |
| 13 | 0.00266003 | 0.00209156 | 1.27179 | 0.999974 | False |
| 14 | 0.00464883 | 0.00434313 | 1.07039 | 0.99998 | True |
| 15 | 0.00435604 | 0.00402427 | 1.08244 | 1.00001 | True |
| 16 | 0.00244396 | 0.00179681 | 1.36017 | 1.00001 | False |
| 17 | 0.00360517 | 0.00320114 | 1.12622 | 0.999967 | True |
| 18 | 0.00477825 | 0.00448475 | 1.06544 | 0.999984 | True |
| 19 | 0.00387834 | 0.0035049 | 1.10655 | 0.999989 | True |
| 20 | 0.00323513 | 0.00277251 | 1.16686 | 1.00002 | True |
| 21 | 0.00396035 | 0.003596 | 1.10132 | 0.999956 | True |
| 22 | 0.00472901 | 0.00442269 | 1.06926 | 0.999992 | True |
| 23 | 0.00415169 | 0.00380801 | 1.09025 | 1.00005 | True |
| 24 | 0.00476747 | 0.00447217 | 1.06603 | 1.00003 | True |
| 25 | 0.00608545 | 0.00585598 | 1.03918 | 0.999995 | True |
| 26 | 0.00410339 | 0.00374889 | 1.09456 | 0.999972 | True |
| 27 | 0.00455388 | 0.00424199 | 1.07352 | 1.00005 | True |

## Qwen3-4B

G1 (window 0, 256 decoded/scored tokens): `{"G1": "PASS", "S0": 7936, "tokens": 256, "mean_kl": 0.0008701731710146756, "top1": 0.9921875, "delta_nll": -0.00018201990482757946, "full_mask_checks": 9216}`.
G2 (window 0, 256 steps, arm a, δ=8.1 layer 0 exact): RED; maximum per-step fraction discrepancy 3.228535803340149e-06; maximum per-layer raw deviation ratio 3.1933641760094615.

| arm | windows / decoded tokens | S0 | mean KL (nats), CI | top-1 fraction, CI | ΔNLL (nats/token), CI |
|---|---:|---:|---|---|---|
| a | 6 / 1536 | [7936] | 0.000689259 [0.000565476, 0.000811857] | 0.989583 [0.984375, 0.995443] | -0.00129551 [-0.00342515, 0.00127315] |
| b | 6 / 1536 | [7936] | 0.000762276 [0.000695616, 0.00082211] | 0.986979 [0.981771, 0.992188] | -0.00115123 [-0.0041731, 0.00143308] |
| c | 6 / 1536 | [7936] | 0.00067951 [0.000556848, 0.000779732] | 0.990234 [0.986979, 0.99349] | -0.000576857 [-0.00376756, 0.00219899] |

| paired vs a | ΔKL, CI | Δtop-1, CI | ΔΔNLL, CI | relative ΔNLL gap | KL / top1 / NLL / free |
|---|---|---|---|---|---|
| b | 7.30168e-05 [-1.4279e-05, 0.000195129] | -0.00260417 [-0.0078125, 0.00260417] | 0.000144275 [-0.00119255, 0.00147974] | 0.111366 | True / False / False / False |
| c | -9.74893e-06 [-5.87623e-05, 3.73625e-05] | 0.000651042 [-0.00455729, 0.00651042] | 0.000718653 [-0.00126579, 0.00266901] | 0.554726 | True / True / False / True |

| arm | prompts / generated tokens | median divergence vs exact | mean token match vs exact | median divergence vs a | mean token match vs a |
|---|---:|---:|---:|---:|---:|
| exact | 8 / 1024 | 128.0 | 1.0 | 85.5 | 0.6171875 |
| a | 8 / 1024 | 85.5 | 0.6171875 | 128.0 | 1.0 |
| b | 8 / 1024 | 65.0 | 0.576171875 | 46.5 | 0.5107421875 |
| c | 8 / 1024 | 90.5 | 0.62890625 | 85.5 | 0.6591796875 |

| flip arm vs a | sampled layer steps / decisions | flips | min / median / p90 / p99 / max consequence |
|---|---:|---:|---|
| b | 315 / 81294080 | 56 | 5.76149e-09 / 1.73479e-05 / 0.00036659 / 0.00089366 / 0.00121948 |
| c | 315 / 81294080 | 43 | 2.61083e-08 / 1.19849e-05 / 0.000392266 / 0.000603368 / 0.00062708 |

Supplemental dense-layer0 G1 (same w0, S0=7936, 256 scored tokens; layers1..N-1 all-refined): `{"G1": "PASS", "S0": 7936, "tokens": 256, "mean_kl": 0.0008230890141717733, "top1": 0.9921875, "delta_nll": -0.0033199319592600585, "full_mask_checks": 8960, "layer0": "exact"}`. This does not replace the original registered result.

| G2 layer (a, w0, 256 steps) | kernel dev | SPQ7 fp32 dev | raw ratio | bf16-ref ratio (diagnostic) | raw pass |
|---|---:|---:|---:|---:|---|
| 1 | 0.00206652 | 0.0012449 | 1.65999 | 0.999991 | False |
| 2 | 0.00193107 | 0.00100623 | 1.91911 | 1.00009 | False |
| 3 | 0.0023417 | 0.00165306 | 1.41659 | 1.00002 | False |
| 4 | 0.00199349 | 0.00109946 | 1.81316 | 0.999999 | False |
| 5 | 0.00257038 | 0.00196337 | 1.30917 | 1 | False |
| 6 | 0.00174609 | 0.000546786 | 3.19336 | 1.00002 | False |
| 7 | 0.00347097 | 0.0030482 | 1.13869 | 1.00001 | True |
| 8 | 0.00822106 | 0.00805128 | 1.02109 | 1.00001 | True |
| 9 | 0.00322765 | 0.00276445 | 1.16756 | 0.999923 | True |
| 10 | 0.00450677 | 0.00419132 | 1.07526 | 1 | True |
| 11 | 0.00338274 | 0.0029442 | 1.14895 | 0.999969 | True |
| 12 | 0.00287106 | 0.00234262 | 1.22558 | 0.999972 | True |
| 13 | 0.00240268 | 0.00174668 | 1.37556 | 0.999962 | False |
| 14 | 0.00228388 | 0.00157225 | 1.45263 | 1.00002 | False |
| 15 | 0.00216191 | 0.00138085 | 1.56564 | 0.999992 | False |
| 16 | 0.00224934 | 0.00151842 | 1.48137 | 0.999997 | False |
| 17 | 0.00193884 | 0.00100632 | 1.92666 | 1.00001 | False |
| 18 | 0.00193703 | 0.000996459 | 1.94392 | 0.999984 | False |
| 19 | 0.00211154 | 0.00130796 | 1.61437 | 1 | False |
| 20 | 0.00217302 | 0.00140561 | 1.54596 | 1 | False |
| 21 | 0.00214203 | 0.00135739 | 1.57805 | 0.999977 | False |
| 22 | 0.00223241 | 0.00149412 | 1.49413 | 0.999997 | False |
| 23 | 0.00198513 | 0.00109149 | 1.81873 | 1.00001 | False |
| 24 | 0.00224028 | 0.00151276 | 1.48092 | 0.999998 | False |
| 25 | 0.00287337 | 0.00235465 | 1.2203 | 1.00001 | True |
| 26 | 0.00369964 | 0.00331184 | 1.11709 | 1.00004 | True |
| 27 | 0.00285343 | 0.00232301 | 1.22833 | 0.999974 | True |
| 28 | 0.0027613 | 0.00220311 | 1.25337 | 1.00001 | False |
| 29 | 0.00384038 | 0.00345755 | 1.11072 | 1.00002 | True |
| 30 | 0.00368655 | 0.00329628 | 1.1184 | 0.999989 | True |
| 31 | 0.00565397 | 0.00540684 | 1.04571 | 1.00001 | True |
| 32 | 0.00467325 | 0.00436874 | 1.0697 | 0.999999 | True |
| 33 | 0.00415426 | 0.00380738 | 1.09111 | 1.00003 | True |
| 34 | 0.00274801 | 0.00216699 | 1.26812 | 0.99997 | False |
| 35 | 0.00512641 | 0.00485279 | 1.05638 | 1 | True |

P3 pooled across models/arms: {'count': 155, 'quantiles': {'0': 5.76148808884609e-09, '0.5': 1.3149014908974379e-05, '0.9': 0.00030574290671451024, '0.99': 0.0008995839743962773, '1': 0.001285348719136893}, 'verdict': 'HELD'}. The 1.7B C stratum is also reported above; registration did not state pooling.

## Predictions

- P1: MISSED BY NUMERIC RULE; attribution blocked by prerequisite RED.
- P2: MISSED BY NUMERIC RULE; attribution blocked by prerequisite RED.
- P3_stratified: MISSED.
- P3: pooled HELD; model/arm strata MISSED (aggregation unspecified in registration).

## Interpretation and limitations

Flip percentiles are empirical quantiles from the registered nine-step subset of window 0, not a population-tail bound. No claim is made about unmeasured models, operating points or decode steps.

The comparison measures the complete r2 and tensor-core decode arms. It includes their PV accumulation and bf16 probability/output rounding differences as well as selection flips. The same-input flip subset isolates arithmetic mask differences, but a zero-flip subset cannot establish a per-flip bound. Failed G1/G2 prerequisites prevent using the descriptive model comparison to certify that flips are immaterial. No speed, other models, prefill APA, INT4, or broader tasks were evaluated.

Registered G1/G2 are not claimed fixed. Supplemental dense-layer0 G1, when present, is separately labeled and does not discard the originally registered RED result.

The relative ΔNLL gate uses the literal ΔNLL_a denominator (amendment 001). Its near-zero denominator is intentionally not replaced after seeing results. G2 uses raw SPQ7 fp32 output deviation (before bf16 rounding), as SPQ7 G2; the rounded comparison is diagnostic only.

## Deviations and open items

- Initial G1 registration routed layer0 through all-refined r2 (SPQ7 floor style) instead of retaining the operating dense layer0. Supplemental dense-layer0 check was separately registered in amendment003; both recorded, original G1 RED retained.
- G1 literal +∞ is rejected by the read-only engine. Maximum finite fp32 delta plus all-key mask assertions exercises refine-every-key semantics; recorded before evaluation in amendment 002.
- Both models completed at S0=7936; the registered 4B S0=4096 fallback was not invoked.
- Original registration remains create-only. Amendment 001, before evaluation, removed an invented early stop and corrected the relative ΔNLL denominator. No gate thresholds changed.
- Full-capacity engine K/V are maintained with write_rows, and packed state appended each step. Prefix slices make device copies because the ABI requires contiguous keys/values. New query/key/value and returned attention rows bridge through host memory; correctness was prioritized, speed is out of scope.
- One GPU cell timed out: `END CELL Qwen3-4B_teacher_w12_w16_w20 exit 124 seconds 600.08`. The 600-second limit was enforced; last logged C step224, partial file contained240 valid rows and16 zero suffix rows. Only completed replacement C/w20 enters metrics. Preservation manifests: ['artifacts/apa_spq8/attempts/1791053472768894745_Qwen3-4B_w20_c_timeout_logits.manifest.json']. Partial arrays remain preserved; incomplete runs do not enter metrics.
- No files removed or moved to DELETE_ME. Build objects, caches, full logits and logs remain inside the permitted paths.
- P3 aggregation was not specified in the registration: pooled p99 and model/arm p99 are both reported with the same 1e-3 bound; no favorable aggregation is silently chosen.
- Open: G1 misses top1 on 1.7B by one discrete mismatch at this 256-token gate size; G2 raw output deviation and bf16-rounded deviation disagree. The current gates remain RED; future gate precision/sample choices belong to the lead.
- Open: selection-only model-level causality is not established because r2 and tc have different PV precision schedules. Engine changes or mask-replay controls are outside this harness-only order.
- Open: define P3 aggregation and relative DeltaNLL scale before a successor; the registered DeltaNLL_a denominator is near zero, and its literal rule is retained here.
- Blind verification is owned by the lead; no subagents were launched. Author checks and end-to-end gates are baseline evidence, not blind certification.

## Prior art

- [TurboQuant, Zandieh et al. (2025)](https://arxiv.org/abs/2504.19874): rotation, Lloyd–Max scalar codebook and per-key norms, MSE stage only. Taken unchanged through SPQ3/engine; no QJL residual claimed.
- [BLASST, Yuan, Shinn et al. (2025)](https://arxiv.org/abs/2512.12087): running-max threshold criterion. Taken through APA-SP1/SP2, adapted to per-key precision rather than dropping keys.
- APA, David Perry (2026), local paper sections 2.1–2.2: full denominator with bulk/refine precision allocation. Mechanism unchanged.
- [Qwen3, Qwen Team (2025)](https://huggingface.co/Qwen/Qwen3-1.7B): model architecture and own normalized projections/RoPE, reused through HF.
- [Henry et al. (2019)](https://arxiv.org/abs/1904.06376): split-bf16 arithmetic in the existing FXTC1 kernels; no arithmetic implementation added here.
- Per-flip key-weight/head-output-norm diagnostic: formula mandated by the lead SPQ8 order; no specific prior art known to me for this normalized single-key diagnostic. Taken: the stated measurement; ours: the instrumentation, not a novelty claim.
- SPQ7 (2026): window/token protocol, fp32 rule simulation and KL/top1/NLL metrics. Standard stable softmax (Higham 2002), bootstrap (Efron 1979), and util-linux flock/GNU timeout are reused. New contribution: contained live-kernel model harness and paired single-key interventions, no new attention/statistical algorithm.

## Files and receipts

- `artifacts/apa_spq8/verdict.json`, `summary.json`, `audit.json`, complete teacher/free summaries and source snapshots.
- `artifacts/apa_spq8/registration.json`, `registration.sha256`, `tokens.json`, amendment files.
- `artifacts/apa_spq8/build/module.sha256`; build log `logs/apa_spq8_build.log`.
- `artifacts/apa_spq8/g1/`, `eval/`, `free/`: per-step distributions, window metrics, per-layer G2 and flip receipts.
- `artifacts/apa_spq8/audit.json`, `logs/apa_spq8_cpu_tests.log`: author verification receipts.
- Harness: `scripts/apa_spq8_*` (run, quant, register, build, GPU wrapper, batch, finish, recover, status, audit, report); `tests/test_apa_spq8_metrics.py`; execution ledger `docs/APA_SPQ8_LEDGER.md`.
