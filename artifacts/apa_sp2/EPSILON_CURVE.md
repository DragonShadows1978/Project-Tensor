# APA-SP2 epsilon curve for David

kernel sweep; this establishes nothing about model quality.

Status: **COMPLETE**; 448/448 epsilon rows. No epsilon is selected or recommended.

The table uses independent standard-normal queries and keys, TurboQuant 2/4-bit reconstructed fp32 keys, and the unchanged two-pass kernel at percentile 0.15 (z=1.0364333894937898). Timings exclude quantization and diagnostics for both paths. Each epsilon has a same-invocation interleaved baseline. Dense fp32 deviations cover every output element. Causal decode sees all keys; a causal label does not itself establish peaked attention.

Read each class from larger to smaller epsilon: the derived margin grows, and for fixed inputs and margins the refine set can only grow. Neither output deviation nor measured speed has a monotonicity theorem. Crossings are the first strict improvement/regression in that grid order, separately for relative Frobenius and max abs; missing earlier rows block a crossing claim. Margin exceedances are reported without fitting the table to the holdout. With no GPU receipts there is no empirical reading of the curve yet. The eventual choice belongs to David.

## Margin table

| Bits | D | Count | Max | p99.9 | Mean | Registered e_q |
|---|---|---|---|---|---|---|
| 2 | 64 | 13762608 | 2.3154934128542095 | 1.1768225385248687 | 0.26734052069046704 | 2.315493583679199 |
| 2 | 128 | 13762608 | 1.9718845675295202 | 1.1541615803171423 | 0.2706264128934229 | 1.9718847274780273 |
| 4 | 64 | 13762608 | 1.0106457381177103 | 0.35496299748043386 | 0.0751211928964718 | 1.010645866394043 |
| 4 | 128 | 13762608 | 0.8716113419285496 | 0.342565029066423 | 0.0764183888872646 | 0.8716114163398743 |

Rule: maximum over all registered calibration logits, rounded upward to fp32, pooled by (bits,D). This is finite calibration evidence, not a universal bound. p99.9 is descriptive only.

RED reasoning bound: skipped probability mass <= min(1,N*epsilon*w_star); perturbation L1 <= min(2,2*N*epsilon*w_star*(1-exp(-e_q))). At N=32768 and epsilon=1e-3, N*epsilon=32.768; at epsilon=1e-4 it is 3.2768. Without an observed w_star or tighter information the worst-case mass bound is vacuous even at the smallest grid epsilon. These are arithmetic bounds, not measured errors.

## Predictions

| Prediction | Verdict |
|---|---|
| P1 | HIT |
| P2 | MISS |
| P3 | HIT |
| P4 | HIT |

## prefill_s2048_d64_c0_h4_kv4, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15428275 | 0.1930049 | 0.034407325 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c0_h4_kv4_e0.1788668280500024684.json |
| 0.1 | 0.99962366 | 0.00015887154 | 0.00032256544 | 1.0007 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c0_h4_kv4_e0.1788668280500024684.json |
| 0.03 | 0.99998909 | 1.053442e-05 | 3.2618642e-05 | 0.995628 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c0_h4_kv4_e1.1788668313139517227.json |
| 0.01 | 0.9999997 | 2.2484978e-06 | 1.4863908e-05 | 0.995762 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c0_h4_kv4_e2.1788668345790544865.json |
| 0.003 | 1 | 1.0527639e-06 | 4.4703484e-07 | 0.995745 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c0_h4_kv4_e3.1788668378441186418.json |
| 0.001 | 1 | 1.0527639e-06 | 4.4703484e-07 | 0.997451 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c0_h4_kv4_e4.1788668411091398396.json |
| 0.0003 | 1 | 1.0527639e-06 | 4.4703484e-07 | 0.996922 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c0_h4_kv4_e5.1788668443770990991.json |
| 0.0001 | 1 | 1.0527639e-06 | 4.4703484e-07 | 0.989654 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c0_h4_kv4_e6.1788668501628222162.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: 0.03.

## prefill_s2048_d64_c0_h8_kv2, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15440357 | 0.18306837 | 0.039011192 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c0_h8_kv2_e0.1788668535543948394.json |
| 0.1 | 0.99962097 | 0.00017556425 | 0.00088642538 | 1.0281 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c0_h8_kv2_e0.1788668535543948394.json |
| 0.03 | 0.99998739 | 1.7820599e-05 | 0.00013612304 | 1.01703 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c0_h8_kv2_e1.1788668569490103099.json |
| 0.01 | 0.99999961 | 1.3316184e-06 | 5.9306622e-06 | 1.05249 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c0_h8_kv2_e2.1788668603446753148.json |
| 0.003 | 0.99999994 | 1.0439181e-06 | 1.007691e-06 | 1.00466 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c0_h8_kv2_e3.1788668637448239849.json |
| 0.001 | 1 | 1.0392894e-06 | 5.364418e-07 | 1.02881 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c0_h8_kv2_e4.1788668671391377383.json |
| 0.0003 | 1 | 1.0392894e-06 | 5.364418e-07 | 1.02469 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c0_h8_kv2_e5.1788668705339209965.json |
| 0.0001 | 1 | 1.0392894e-06 | 5.364418e-07 | 1.01441 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c0_h8_kv2_e6.1788668739297503569.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s2048_d64_c1_h4_kv4, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15451742 | 0.19574679 | 0.99516284 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c1_h4_kv4_e0.1788668771764550213.json |
| 0.1 | 0.99971809 | 6.6720505e-05 | 0.00053317845 | 1.02773 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c1_h4_kv4_e0.1788668771764550213.json |
| 0.03 | 0.99999226 | 3.7055939e-06 | 2.9116869e-05 | 1.03194 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c1_h4_kv4_e1.1788668804225638252.json |
| 0.01 | 0.99999976 | 4.8465575e-07 | 2.2985041e-06 | 1.03013 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c1_h4_kv4_e2.1788668836682613057.json |
| 0.003 | 1 | 4.723042e-07 | 7.7486038e-07 | 1.02442 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c1_h4_kv4_e3.1788668869156859487.json |
| 0.001 | 1 | 4.723042e-07 | 7.7486038e-07 | 1.02663 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c1_h4_kv4_e4.1788668901621992197.json |
| 0.0003 | 1 | 4.723042e-07 | 7.7486038e-07 | 1.025 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c1_h4_kv4_e5.1788668934114050758.json |
| 0.0001 | 1 | 4.723042e-07 | 7.7486038e-07 | 1.02735 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c1_h4_kv4_e6.1788668966584771385.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s2048_d64_c1_h8_kv2, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15453101 | 0.19702903 | 0.83061755 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c1_h8_kv2_e0.1788669000337374717.json |
| 0.1 | 0.99973906 | 5.8413845e-05 | 0.00085885823 | 1.08024 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c1_h8_kv2_e0.1788669000337374717.json |
| 0.03 | 0.99999243 | 3.5652395e-06 | 4.4092536e-05 | 1.0845 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c1_h8_kv2_e1.1788669034081380997.json |
| 0.01 | 0.99999988 | 4.8210608e-07 | 2.6151538e-06 | 1.1117 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c1_h8_kv2_e2.1788669067836147669.json |
| 0.003 | 1 | 4.6551761e-07 | 7.7486038e-07 | 1.08238 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c1_h8_kv2_e3.1788669101569037476.json |
| 0.001 | 1 | 4.6551761e-07 | 7.7486038e-07 | 1.08374 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c1_h8_kv2_e4.1788669135374067949.json |
| 0.0003 | 1 | 4.6551761e-07 | 7.7486038e-07 | 1.08319 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c1_h8_kv2_e5.1788669169148072954.json |
| 0.0001 | 1 | 4.6551761e-07 | 7.7486038e-07 | 1.08656 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d64_c1_h8_kv2_e6.1788669202885876936.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s2048_d128_c0_h4_kv4, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15458441 | 0.19211542 | 0.037922564 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c0_h4_kv4_e0.1788669236013836910.json |
| 0.1 | 0.99861139 | 0.00044537891 | 0.00084578991 | 1.02019 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c0_h4_kv4_e0.1788669236013836910.json |
| 0.03 | 0.99996698 | 2.9693782e-05 | 8.8006258e-05 | 1.01267 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c0_h4_kv4_e1.1788669269164032190.json |
| 0.01 | 0.99999917 | 2.1274987e-06 | 1.296401e-05 | 1.01393 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c0_h4_kv4_e2.1788669302291224623.json |
| 0.003 | 1 | 1.109237e-06 | 4.61936e-07 | 1.00994 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c0_h4_kv4_e3.1788669335426491422.json |
| 0.001 | 1 | 1.109237e-06 | 4.61936e-07 | 1.02748 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c0_h4_kv4_e4.1788669368554141344.json |
| 0.0003 | 1 | 1.109237e-06 | 4.61936e-07 | 1.01662 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c0_h4_kv4_e5.1788669401675103791.json |
| 0.0001 | 1 | 1.109237e-06 | 4.61936e-07 | 1.0158 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c0_h4_kv4_e6.1788669434807050846.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s2048_d128_c0_h8_kv2, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15471181 | 0.18917879 | 0.037043661 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c0_h8_kv2_e0.1788669469860124563.json |
| 0.1 | 0.99859181 | 0.00044698305 | 0.0011545643 | 1.02151 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c0_h8_kv2_e0.1788669469860124563.json |
| 0.03 | 0.99996334 | 3.1363288e-05 | 0.00013553351 | 1.00835 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c0_h8_kv2_e1.1788669504976741842.json |
| 0.01 | 0.99999923 | 2.2766075e-06 | 1.2729317e-05 | 1.01189 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c0_h8_kv2_e2.1788669539967037337.json |
| 0.003 | 0.99999997 | 1.1026854e-06 | 1.2144446e-06 | 0.985512 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c0_h8_kv2_e3.1788669574969734724.json |
| 0.001 | 1 | 1.0992534e-06 | 5.2154064e-07 | 1.01645 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c0_h8_kv2_e4.1788669609970961748.json |
| 0.0003 | 1 | 1.0992534e-06 | 5.2154064e-07 | 1.02426 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c0_h8_kv2_e5.1788669645022441371.json |
| 0.0001 | 1 | 1.0992534e-06 | 5.2154064e-07 | 0.998301 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c0_h8_kv2_e6.1788669680015685389.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: 0.003.

## prefill_s2048_d128_c1_h4_kv4, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15481232 | 0.19224515 | 0.59603114 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c1_h4_kv4_e0.1788669713045162495.json |
| 0.1 | 0.99901903 | 0.00018321232 | 0.0011836402 | 1.01786 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c1_h4_kv4_e0.1788669713045162495.json |
| 0.03 | 0.99997486 | 1.2799569e-05 | 0.00017541833 | 1.02162 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c1_h4_kv4_e1.1788669746055227347.json |
| 0.01 | 0.9999994 | 8.996763e-07 | 1.2874603e-05 | 1.01855 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c1_h4_kv4_e2.1788669779060422810.json |
| 0.003 | 1 | 5.2310303e-07 | 1.3113022e-06 | 1.01567 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c1_h4_kv4_e3.1788669812082879711.json |
| 0.001 | 1 | 5.2310303e-07 | 1.3113022e-06 | 1.01607 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c1_h4_kv4_e4.1788669845113654733.json |
| 0.0003 | 1 | 5.2310303e-07 | 1.3113022e-06 | 1.01781 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c1_h4_kv4_e5.1788669878149316360.json |
| 0.0001 | 1 | 5.2310303e-07 | 1.3113022e-06 | 1.01899 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c1_h4_kv4_e6.1788669911211642928.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s2048_d128_c1_h8_kv2, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15473464 | 0.19378293 | 0.75478368 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c1_h8_kv2_e0.1788669945986563630.json |
| 0.1 | 0.99891739 | 0.00016903118 | 0.0011110902 | 1.05212 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c1_h8_kv2_e0.1788669945986563630.json |
| 0.03 | 0.99997837 | 1.1393442e-05 | 0.00023583323 | 1.04251 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c1_h8_kv2_e1.1788669980733511867.json |
| 0.01 | 0.99999958 | 6.1864747e-07 | 7.5977296e-06 | 1.03166 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c1_h8_kv2_e2.1788670015449107646.json |
| 0.003 | 1 | 5.2431629e-07 | 8.3446503e-07 | 1.04652 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c1_h8_kv2_e3.1788670050157222848.json |
| 0.001 | 1 | 5.2431629e-07 | 8.3446503e-07 | 1.05384 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c1_h8_kv2_e4.1788670084921654961.json |
| 0.0003 | 1 | 5.2431629e-07 | 8.3446503e-07 | 1.03825 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c1_h8_kv2_e5.1788670119715842697.json |
| 0.0001 | 1 | 5.2431629e-07 | 8.3446503e-07 | 1.05787 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s2048_d128_c1_h8_kv2_e6.1788670154480354111.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s8192_d64_c0_h4_kv4, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15432397 | 0.18477743 | 0.019984162 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c0_h4_kv4_e0.1788670207300060726.json |
| 0.1 | 0.99900464 | 0.00033605972 | 0.00059143454 | 1.09784 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c0_h4_kv4_e0.1788670207300060726.json |
| 0.03 | 0.99996525 | 2.9033864e-05 | 6.9387257e-05 | 1.07884 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c0_h4_kv4_e1.1788670260276133803.json |
| 0.01 | 0.99999893 | 3.4952152e-06 | 1.1237338e-05 | 1.08808 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c0_h4_kv4_e2.1788670313300168878.json |
| 0.003 | 0.99999998 | 1.9794303e-06 | 4.9546361e-07 | 1.08383 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c0_h4_kv4_e3.1788670366466143361.json |
| 0.001 | 1 | 1.9780436e-06 | 4.4330955e-07 | 1.08591 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c0_h4_kv4_e4.1788670419266614147.json |
| 0.0003 | 1 | 1.9780436e-06 | 4.4330955e-07 | 1.08516 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c0_h4_kv4_e5.1788670472050753242.json |
| 0.0001 | 1 | 1.9780436e-06 | 4.4330955e-07 | 1.08557 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c0_h4_kv4_e6.1788670524588698524.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s8192_d64_c0_h8_kv2, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15432311 | 0.18613837 | 0.019781322 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c0_h8_kv2_e0.1788670598867780451.json |
| 0.1 | 0.99899619 | 0.00034821062 | 0.00080082566 | 1.10612 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c0_h8_kv2_e0.1788670598867780451.json |
| 0.03 | 0.99996477 | 3.1268453e-05 | 9.7431242e-05 | 1.10071 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c0_h8_kv2_e1.1788670673468831607.json |
| 0.01 | 0.99999892 | 3.2891512e-06 | 1.366064e-05 | 1.10052 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c0_h8_kv2_e2.1788670747979032955.json |
| 0.003 | 0.99999998 | 2.0070136e-06 | 1.5087426e-06 | 1.10016 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c0_h8_kv2_e3.1788670845329885724.json |
| 0.001 | 1 | 2.0013502e-06 | 5.1781535e-07 | 1.1002 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c0_h8_kv2_e4.1788670919756807634.json |
| 0.0003 | 1 | 2.0013502e-06 | 5.1781535e-07 | 1.09875 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c0_h8_kv2_e5.1788670994002996001.json |
| 0.0001 | 1 | 2.0013502e-06 | 5.1781535e-07 | 1.09818 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c0_h8_kv2_e6.1788671068567875872.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s8192_d64_c1_h4_kv4, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15433411 | 0.18381387 | 0.63171825 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c1_h4_kv4_e0.1788671119862337902.json |
| 0.1 | 0.99925076 | 0.00012771947 | 0.00082404702 | 1.10607 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c1_h4_kv4_e0.1788671119862337902.json |
| 0.03 | 0.99997494 | 9.9625893e-06 | 6.1988831e-05 | 1.12537 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c1_h4_kv4_e1.1788671170980553308.json |
| 0.01 | 0.99999931 | 9.5042811e-07 | 6.9327652e-06 | 1.12651 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c1_h4_kv4_e2.1788671222330324371.json |
| 0.003 | 0.99999999 | 7.6283825e-07 | 1.013279e-06 | 1.12467 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c1_h4_kv4_e3.1788671273636808856.json |
| 0.001 | 1 | 7.6213725e-07 | 1.013279e-06 | 1.10621 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c1_h4_kv4_e4.1788671324764093520.json |
| 0.0003 | 1 | 7.6213725e-07 | 1.013279e-06 | 1.12734 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c1_h4_kv4_e5.1788671376142311907.json |
| 0.0001 | 1 | 7.6213725e-07 | 1.013279e-06 | 1.12465 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c1_h4_kv4_e6.1788671427158769994.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s8192_d64_c1_h8_kv2, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15438734 | 0.19708462 | 1.2113711 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c1_h8_kv2_e0.1788671498138899016.json |
| 0.1 | 0.99924933 | 0.00014157694 | 0.0012748949 | 1.16478 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c1_h8_kv2_e0.1788671498138899016.json |
| 0.03 | 0.99997228 | 1.4298236e-05 | 0.0002808515 | 1.16227 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c1_h8_kv2_e1.1788671569269943916.json |
| 0.01 | 0.99999879 | 1.5784317e-06 | 2.8658658e-05 | 1.15948 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c1_h8_kv2_e2.1788671640539441362.json |
| 0.003 | 0.99999996 | 7.6500173e-07 | 2.3804605e-06 | 1.16104 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c1_h8_kv2_e3.1788671711813928576.json |
| 0.001 | 0.99999999 | 7.6114252e-07 | 8.6426735e-07 | 1.16392 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c1_h8_kv2_e4.1788671782786605894.json |
| 0.0003 | 1 | 7.6104634e-07 | 8.6426735e-07 | 1.1605 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c1_h8_kv2_e5.1788671853401487226.json |
| 0.0001 | 1 | 7.6104634e-07 | 8.6426735e-07 | 1.16385 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d64_c1_h8_kv2_e6.1788671924935720518.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s8192_d128_c0_h4_kv4, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15457677 | 0.19277302 | 0.01825642 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c0_h4_kv4_e0.1788671985617644876.json |
| 0.1 | 0.99633186 | 0.00095256547 | 0.0021992896 | 1.24801 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c0_h4_kv4_e0.1788671985617644876.json |
| 0.03 | 0.99988308 | 9.6142545e-05 | 0.00041655824 | 1.18879 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c0_h4_kv4_e1.1788672047139041328.json |
| 0.01 | 0.99999686 | 9.6244476e-06 | 5.4968521e-05 | 1.17149 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c0_h4_kv4_e2.1788672108469957338.json |
| 0.003 | 0.99999996 | 2.142468e-06 | 6.2659383e-06 | 1.17939 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c0_h4_kv4_e3.1788672169727740464.json |
| 0.001 | 1 | 2.0339244e-06 | 4.8056245e-07 | 1.16848 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c0_h4_kv4_e4.1788672230864269863.json |
| 0.0003 | 1 | 2.0339244e-06 | 4.8056245e-07 | 1.1846 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c0_h4_kv4_e5.1788672292399507804.json |
| 0.0001 | 1 | 2.0339244e-06 | 4.8056245e-07 | 1.1797 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c0_h4_kv4_e6.1788672353340298926.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s8192_d128_c0_h8_kv2, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15456918 | 0.18912584 | 0.019145463 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c0_h8_kv2_e0.1788672444530429142.json |
| 0.1 | 0.99643586 | 0.00088422852 | 0.0021014959 | 1.38945 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c0_h8_kv2_e0.1788672444530429142.json |
| 0.03 | 0.99989399 | 7.1602524e-05 | 0.00025328249 | 1.3273 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c0_h8_kv2_e1.1788672535989708850.json |
| 0.01 | 0.99999762 | 5.2429122e-06 | 3.2866374e-05 | 1.32395 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c0_h8_kv2_e2.1788672627150889191.json |
| 0.003 | 0.99999999 | 2.0300863e-06 | 2.4326146e-06 | 1.32988 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c0_h8_kv2_e3.1788672719224814595.json |
| 0.001 | 1 | 2.021799e-06 | 5.364418e-07 | 1.32322 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c0_h8_kv2_e4.1788672811033581285.json |
| 0.0003 | 1 | 2.021799e-06 | 5.364418e-07 | 1.32811 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c0_h8_kv2_e5.1788672902686136854.json |
| 0.0001 | 1 | 2.021799e-06 | 5.364418e-07 | 1.32982 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c0_h8_kv2_e6.1788672994380303328.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s8192_d128_c1_h4_kv4, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15460745 | 0.19517047 | 0.81964239 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c1_h4_kv4_e0.1788673053216199983.json |
| 0.1 | 0.99724703 | 0.0003456459 | 0.0022556782 | 1.21414 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c1_h4_kv4_e0.1788673053216199983.json |
| 0.03 | 0.9999202 | 2.3490732e-05 | 0.0001867786 | 1.21397 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c1_h4_kv4_e1.1788673111797810366.json |
| 0.01 | 0.99999829 | 1.8271362e-06 | 1.9624829e-05 | 1.20197 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c1_h4_kv4_e2.1788673170849748168.json |
| 0.003 | 0.99999998 | 8.2421997e-07 | 2.8728973e-06 | 1.17733 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c1_h4_kv4_e3.1788673229734416173.json |
| 0.001 | 1 | 8.1572531e-07 | 8.9406967e-07 | 1.22853 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c1_h4_kv4_e4.1788673288436929717.json |
| 0.0003 | 1 | 8.1572531e-07 | 8.9406967e-07 | 1.20809 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c1_h4_kv4_e5.1788673347010266389.json |
| 0.0001 | 1 | 8.1572531e-07 | 8.9406967e-07 | 1.22232 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c1_h4_kv4_e6.1788673405727039514.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s8192_d128_c1_h8_kv2, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15460196 | 0.19248155 | 1.2291732 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c1_h8_kv2_e0.1788673492742917040.json |
| 0.1 | 0.99723735 | 0.00032764213 | 0.0043151528 | 1.34429 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c1_h8_kv2_e0.1788673492742917040.json |
| 0.03 | 0.99992018 | 2.7886351e-05 | 0.00065891445 | 1.32472 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c1_h8_kv2_e1.1788673579528461517.json |
| 0.01 | 0.99999831 | 2.7207252e-06 | 6.736815e-05 | 1.33328 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c1_h8_kv2_e2.1788673666356160503.json |
| 0.003 | 0.99999999 | 8.0177303e-07 | 1.8943101e-06 | 1.34107 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c1_h8_kv2_e3.1788673753713835875.json |
| 0.001 | 1 | 8.0073939e-07 | 1.1920929e-06 | 1.33506 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c1_h8_kv2_e4.1788673840678009601.json |
| 0.0003 | 1 | 8.0073939e-07 | 1.1920929e-06 | 1.34623 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c1_h8_kv2_e5.1788673927074907124.json |
| 0.0001 | 1 | 8.0073939e-07 | 1.1920929e-06 | 1.32227 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b2_prefill_s8192_d128_c1_h8_kv2_e6.1788674013797377085.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s8192_d64_c0_h4_kv4, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15756226 | 0.20803705 | 0.011794964 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c0_h4_kv4_e0.1788674045133031247.json |
| 0.1 | 0.9994812 | 0.00021158642 | 1.5215948e-05 | 1.92548 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c0_h4_kv4_e0.1788674045133031247.json |
| 0.03 | 0.99996948 | 2.50508e-06 | 1.8253922e-07 | 1.92134 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c0_h4_kv4_e1.1788674076476632344.json |
| 0.01 | 1 | 1.4565974e-06 | 8.9406967e-08 | 1.92945 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c0_h4_kv4_e2.1788674107862043382.json |
| 0.003 | 1 | 1.4565974e-06 | 8.9406967e-08 | 1.92859 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c0_h4_kv4_e3.1788674139201400599.json |
| 0.001 | 1 | 1.4565974e-06 | 8.9406967e-08 | 1.91712 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c0_h4_kv4_e4.1788674170553877882.json |
| 0.0003 | 1 | 1.4565974e-06 | 8.9406967e-08 | 1.92996 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c0_h4_kv4_e5.1788674201917812594.json |
| 0.0001 | 1 | 1.4565974e-06 | 8.9406967e-08 | 1.92732 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c0_h4_kv4_e6.1788674233280123961.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s8192_d64_c0_h8_kv2, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15512085 | 0.17685873 | 0.010123506 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c0_h8_kv2_e0.1788674264615994032.json |
| 0.1 | 0.99919128 | 0.00024428856 | 3.2883137e-05 | 1.93138 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c0_h8_kv2_e0.1788674264615994032.json |
| 0.03 | 0.99995422 | 5.2691235e-05 | 6.8377703e-06 | 1.9224 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c0_h8_kv2_e1.1788674295963240920.json |
| 0.01 | 1 | 1.573714e-06 | 1.5832484e-07 | 1.92267 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c0_h8_kv2_e2.1788674327300493464.json |
| 0.003 | 1 | 1.573714e-06 | 1.5832484e-07 | 1.92254 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c0_h8_kv2_e3.1788674358655468088.json |
| 0.001 | 1 | 1.573714e-06 | 1.5832484e-07 | 1.91608 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c0_h8_kv2_e4.1788674390001693179.json |
| 0.0003 | 1 | 1.573714e-06 | 1.5832484e-07 | 1.93716 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c0_h8_kv2_e5.1788674421353735293.json |
| 0.0001 | 1 | 1.573714e-06 | 1.5832484e-07 | 1.92816 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c0_h8_kv2_e6.1788674452706158312.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s8192_d64_c1_h4_kv4, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15447998 | 0.17802948 | 0.0098750256 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c1_h4_kv4_e0.1788674484071344702.json |
| 0.1 | 1 | 1.5825695e-06 | 1.4901161e-07 | 1.91654 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c1_h4_kv4_e0.1788674484071344702.json |
| 0.03 | 1 | 1.5825695e-06 | 1.4901161e-07 | 1.92126 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c1_h4_kv4_e1.1788674515442484110.json |
| 0.01 | 1 | 1.5825695e-06 | 1.4901161e-07 | 1.91642 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c1_h4_kv4_e2.1788674546821467360.json |
| 0.003 | 1 | 1.5825695e-06 | 1.4901161e-07 | 1.92735 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c1_h4_kv4_e3.1788674578200647289.json |
| 0.001 | 1 | 1.5825695e-06 | 1.4901161e-07 | 1.92791 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c1_h4_kv4_e4.1788674609614008371.json |
| 0.0003 | 1 | 1.5825695e-06 | 1.4901161e-07 | 1.92282 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c1_h4_kv4_e5.1788674641002863162.json |
| 0.0001 | 1 | 1.5825695e-06 | 1.4901161e-07 | 1.92084 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c1_h4_kv4_e6.1788674672392774909.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s8192_d64_c1_h8_kv2, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15391541 | 0.20756631 | 0.011606678 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c1_h8_kv2_e0.1788674703760520700.json |
| 0.1 | 0.99996948 | 6.8561136e-05 | 7.8269513e-06 | 1.92043 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c1_h8_kv2_e0.1788674703760520700.json |
| 0.03 | 1 | 1.6610397e-06 | 1.3038516e-07 | 1.92884 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c1_h8_kv2_e1.1788674735154666018.json |
| 0.01 | 1 | 1.6610397e-06 | 1.3038516e-07 | 1.93662 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c1_h8_kv2_e2.1788674766533563321.json |
| 0.003 | 1 | 1.6610397e-06 | 1.3038516e-07 | 1.92942 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c1_h8_kv2_e3.1788674797911229245.json |
| 0.001 | 1 | 1.6610397e-06 | 1.3038516e-07 | 1.92366 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c1_h8_kv2_e4.1788674829283659748.json |
| 0.0003 | 1 | 1.6610397e-06 | 1.3038516e-07 | 1.93232 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c1_h8_kv2_e5.1788674860669006417.json |
| 0.0001 | 1 | 1.6610397e-06 | 1.3038516e-07 | 1.9276 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d64_c1_h8_kv2_e6.1788674892051199544.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s8192_d128_c0_h4_kv4, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.1534729 | 0.19705313 | 0.011168509 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c0_h4_kv4_e0.1788674923516690884.json |
| 0.1 | 0.99938965 | 0.00018161795 | 1.5836209e-05 | 1.41987 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c0_h4_kv4_e0.1788674923516690884.json |
| 0.03 | 1 | 1.5871164e-06 | 1.1920929e-07 | 1.42061 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c0_h4_kv4_e1.1788674954986001881.json |
| 0.01 | 1 | 1.5871164e-06 | 1.1920929e-07 | 1.41444 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c0_h4_kv4_e2.1788674986465931262.json |
| 0.003 | 1 | 1.5871164e-06 | 1.1920929e-07 | 1.41657 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c0_h4_kv4_e3.1788675017949014237.json |
| 0.001 | 1 | 1.5871164e-06 | 1.1920929e-07 | 1.41513 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c0_h4_kv4_e4.1788675049431688203.json |
| 0.0003 | 1 | 1.5871164e-06 | 1.1920929e-07 | 1.41483 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c0_h4_kv4_e5.1788675080909457590.json |
| 0.0001 | 1 | 1.5871164e-06 | 1.1920929e-07 | 1.4151 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c0_h4_kv4_e6.1788675112395713610.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s8192_d128_c0_h8_kv2, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.154953 | 0.18534449 | 0.011301201 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c0_h8_kv2_e0.1788675143866737582.json |
| 0.1 | 0.99937439 | 0.00018112343 | 1.9824132e-05 | 1.68467 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c0_h8_kv2_e0.1788675143866737582.json |
| 0.03 | 1 | 1.7194956e-06 | 2.682209e-07 | 1.68309 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c0_h8_kv2_e1.1788675175339349218.json |
| 0.01 | 1 | 1.7194956e-06 | 2.682209e-07 | 1.6807 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c0_h8_kv2_e2.1788675206783418593.json |
| 0.003 | 1 | 1.7194956e-06 | 2.682209e-07 | 1.68356 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c0_h8_kv2_e3.1788675238234464995.json |
| 0.001 | 1 | 1.7194956e-06 | 2.682209e-07 | 1.68334 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c0_h8_kv2_e4.1788675269681338385.json |
| 0.0003 | 1 | 1.7194956e-06 | 2.682209e-07 | 1.6841 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c0_h8_kv2_e5.1788675301134403398.json |
| 0.0001 | 1 | 1.7194956e-06 | 2.682209e-07 | 1.67723 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c0_h8_kv2_e6.1788675332582688288.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s8192_d128_c1_h4_kv4, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15570068 | 0.1991646 | 0.01058127 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c1_h4_kv4_e0.1788675364099196610.json |
| 0.1 | 0.99951172 | 0.00018245251 | 1.6899779e-05 | 1.4177 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c1_h4_kv4_e0.1788675364099196610.json |
| 0.03 | 1 | 1.6879153e-06 | 1.5646219e-07 | 1.41135 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c1_h4_kv4_e1.1788675395599786459.json |
| 0.01 | 1 | 1.6879153e-06 | 1.5646219e-07 | 1.41915 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c1_h4_kv4_e2.1788675427114028364.json |
| 0.003 | 1 | 1.6879153e-06 | 1.5646219e-07 | 1.42249 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c1_h4_kv4_e3.1788675458643013227.json |
| 0.001 | 1 | 1.6879153e-06 | 1.5646219e-07 | 1.41959 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c1_h4_kv4_e4.1788675490175631135.json |
| 0.0003 | 1 | 1.6879153e-06 | 1.5646219e-07 | 1.41854 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c1_h4_kv4_e5.1788675521699228874.json |
| 0.0001 | 1 | 1.6879153e-06 | 1.5646219e-07 | 1.42049 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c1_h4_kv4_e6.1788675553219194664.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s8192_d128_c1_h8_kv2, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.1546936 | 0.18334726 | 0.011681937 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c1_h8_kv2_e0.1788675584701194911.json |
| 0.1 | 0.99864197 | 0.00026615751 | 4.4648536e-05 | 1.67739 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c1_h8_kv2_e0.1788675584701194911.json |
| 0.03 | 0.99998474 | 2.0902261e-06 | 2.2351742e-07 | 1.67775 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c1_h8_kv2_e1.1788675616199156613.json |
| 0.01 | 1 | 1.6157862e-06 | 1.6391277e-07 | 1.6813 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c1_h8_kv2_e2.1788675647685259495.json |
| 0.003 | 1 | 1.6157862e-06 | 1.6391277e-07 | 1.67756 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c1_h8_kv2_e3.1788675679194522000.json |
| 0.001 | 1 | 1.6157862e-06 | 1.6391277e-07 | 1.67877 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c1_h8_kv2_e4.1788675710680829637.json |
| 0.0003 | 1 | 1.6157862e-06 | 1.6391277e-07 | 1.67577 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c1_h8_kv2_e5.1788675742170222130.json |
| 0.0001 | 1 | 1.6157862e-06 | 1.6391277e-07 | 1.67684 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s8192_d128_c1_h8_kv2_e6.1788675773664207254.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s32768_d64_c0_h4_kv4, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15464783 | 0.1959233 | 0.0049530978 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c0_h4_kv4_e0.1788675805299899012.json |
| 0.1 | 0.99993134 | 4.2889538e-05 | 1.9851141e-06 | 3.48334 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c0_h4_kv4_e0.1788675805299899012.json |
| 0.03 | 1 | 3.1858423e-06 | 1.4901161e-07 | 3.49156 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c0_h4_kv4_e1.1788675836939083711.json |
| 0.01 | 1 | 3.1858423e-06 | 1.4901161e-07 | 3.47547 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c0_h4_kv4_e2.1788675868592806258.json |
| 0.003 | 1 | 3.1858423e-06 | 1.4901161e-07 | 3.48899 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c0_h4_kv4_e3.1788675900272011728.json |
| 0.001 | 1 | 3.1858423e-06 | 1.4901161e-07 | 3.48763 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c0_h4_kv4_e4.1788675931921938036.json |
| 0.0003 | 1 | 3.1858423e-06 | 1.4901161e-07 | 3.47226 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c0_h4_kv4_e5.1788675963592027459.json |
| 0.0001 | 1 | 3.1858423e-06 | 1.4901161e-07 | 3.48585 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c0_h4_kv4_e6.1788675995263885686.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s32768_d64_c0_h8_kv2, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15437698 | 0.19705049 | 0.0058185337 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c0_h8_kv2_e0.1788676026855653877.json |
| 0.1 | 0.99988556 | 4.3374248e-05 | 1.8961728e-06 | 2.52214 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c0_h8_kv2_e0.1788676026855653877.json |
| 0.03 | 1 | 3.4808532e-06 | 1.527369e-07 | 2.52545 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c0_h8_kv2_e1.1788676058473250072.json |
| 0.01 | 1 | 3.4808532e-06 | 1.527369e-07 | 2.51744 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c0_h8_kv2_e2.1788676090085781846.json |
| 0.003 | 1 | 3.4808532e-06 | 1.527369e-07 | 2.51795 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c0_h8_kv2_e3.1788676121690889437.json |
| 0.001 | 1 | 3.4808532e-06 | 1.527369e-07 | 2.5182 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c0_h8_kv2_e4.1788676153316856920.json |
| 0.0003 | 1 | 3.4808532e-06 | 1.527369e-07 | 2.51955 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c0_h8_kv2_e5.1788676184942227936.json |
| 0.0001 | 1 | 3.4808532e-06 | 1.527369e-07 | 2.5224 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c0_h8_kv2_e6.1788676216568079840.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s32768_d64_c1_h4_kv4, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15395355 | 0.18982679 | 0.0046608816 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c1_h4_kv4_e0.1788676248278642443.json |
| 0.1 | 0.99996185 | 2.6677734e-05 | 1.2461096e-06 | 3.4584 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c1_h4_kv4_e0.1788676248278642443.json |
| 0.03 | 1 | 2.7083847e-06 | 1.0803342e-07 | 3.48648 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c1_h4_kv4_e1.1788676279958293940.json |
| 0.01 | 1 | 2.7083847e-06 | 1.0803342e-07 | 3.48416 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c1_h4_kv4_e2.1788676311648251742.json |
| 0.003 | 1 | 2.7083847e-06 | 1.0803342e-07 | 3.47231 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c1_h4_kv4_e3.1788676343368705379.json |
| 0.001 | 1 | 2.7083847e-06 | 1.0803342e-07 | 3.48233 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c1_h4_kv4_e4.1788676375053649931.json |
| 0.0003 | 1 | 2.7083847e-06 | 1.0803342e-07 | 3.46728 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c1_h4_kv4_e5.1788676406743449589.json |
| 0.0001 | 1 | 2.7083847e-06 | 1.0803342e-07 | 3.47451 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c1_h4_kv4_e6.1788676438486226081.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s32768_d64_c1_h8_kv2, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15432358 | 0.17969173 | 0.0053405925 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c1_h8_kv2_e0.1788676470122719876.json |
| 0.1 | 0.99965668 | 0.00013282073 | 7.2550029e-06 | 2.5202 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c1_h8_kv2_e0.1788676470122719876.json |
| 0.03 | 0.99999619 | 3.2935792e-06 | 1.5646219e-07 | 2.52332 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c1_h8_kv2_e1.1788676501775052673.json |
| 0.01 | 1 | 3.2851367e-06 | 1.5646219e-07 | 2.51637 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c1_h8_kv2_e2.1788676533413527390.json |
| 0.003 | 1 | 3.2851367e-06 | 1.5646219e-07 | 2.52538 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c1_h8_kv2_e3.1788676565064001003.json |
| 0.001 | 1 | 3.2851367e-06 | 1.5646219e-07 | 2.52078 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c1_h8_kv2_e4.1788676596718268977.json |
| 0.0003 | 1 | 3.2851367e-06 | 1.5646219e-07 | 2.52084 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c1_h8_kv2_e5.1788676628374493904.json |
| 0.0001 | 1 | 3.2851367e-06 | 1.5646219e-07 | 2.52151 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d64_c1_h8_kv2_e6.1788676660024890262.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s32768_d128_c0_h4_kv4, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15408325 | 0.19013824 | 0.0050505003 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c0_h4_kv4_e0.1788676692004080917.json |
| 0.1 | 0.99876404 | 0.00046226472 | 2.4985522e-05 | 4.57913 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c0_h4_kv4_e0.1788676692004080917.json |
| 0.03 | 0.99993896 | 7.5072262e-05 | 3.7556747e-06 | 4.78623 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c0_h4_kv4_e1.1788676724036508761.json |
| 0.01 | 1 | 3.2222446e-06 | 1.4528632e-07 | 4.78635 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c0_h4_kv4_e2.1788676756138375302.json |
| 0.003 | 1 | 3.2222446e-06 | 1.4528632e-07 | 4.77124 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c0_h4_kv4_e3.1788676788112502631.json |
| 0.001 | 1 | 3.2222446e-06 | 1.4528632e-07 | 4.78528 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c0_h4_kv4_e4.1788676820128558472.json |
| 0.0003 | 1 | 3.2222446e-06 | 1.4528632e-07 | 4.77088 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c0_h4_kv4_e5.1788676852109731977.json |
| 0.0001 | 1 | 3.2222446e-06 | 1.4528632e-07 | 4.78006 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c0_h4_kv4_e6.1788676884123389463.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s32768_d128_c0_h8_kv2, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15439987 | 0.17873548 | 0.0061601677 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c0_h8_kv2_e0.1788676915957132498.json |
| 0.1 | 0.99961853 | 0.00012130015 | 5.3662807e-06 | 4.26179 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c0_h8_kv2_e0.1788676915957132498.json |
| 0.03 | 1 | 3.3318991e-06 | 1.7229468e-07 | 4.27222 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c0_h8_kv2_e1.1788676947809518698.json |
| 0.01 | 1 | 3.3318991e-06 | 1.7229468e-07 | 4.28044 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c0_h8_kv2_e2.1788676979658133475.json |
| 0.003 | 1 | 3.3318991e-06 | 1.7229468e-07 | 4.27246 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c0_h8_kv2_e3.1788677011503661302.json |
| 0.001 | 1 | 3.3318991e-06 | 1.7229468e-07 | 4.27131 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c0_h8_kv2_e4.1788677043350362211.json |
| 0.0003 | 1 | 3.3318991e-06 | 1.7229468e-07 | 4.2801 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c0_h8_kv2_e5.1788677075199432730.json |
| 0.0001 | 1 | 3.3318991e-06 | 1.7229468e-07 | 4.26771 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c0_h8_kv2_e6.1788677107078561020.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s32768_d128_c1_h4_kv4, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15460205 | 0.19470359 | 0.0059781938 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c1_h4_kv4_e0.1788677139082442256.json |
| 0.1 | 0.99754333 | 0.00056479818 | 2.6269816e-05 | 4.78213 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c1_h4_kv4_e0.1788677139082442256.json |
| 0.03 | 0.99992371 | 3.2819096e-05 | 1.8747523e-06 | 4.78424 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c1_h4_kv4_e1.1788677171090757738.json |
| 0.01 | 1 | 3.159342e-06 | 1.7136335e-07 | 4.78454 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c1_h4_kv4_e2.1788677203102951294.json |
| 0.003 | 1 | 3.159342e-06 | 1.7136335e-07 | 4.78769 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c1_h4_kv4_e3.1788677235119736463.json |
| 0.001 | 1 | 3.159342e-06 | 1.7136335e-07 | 4.78713 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c1_h4_kv4_e4.1788677267131519906.json |
| 0.0003 | 1 | 3.159342e-06 | 1.7136335e-07 | 4.78286 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c1_h4_kv4_e5.1788677299172003139.json |
| 0.0001 | 1 | 3.159342e-06 | 1.7136335e-07 | 4.78085 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c1_h4_kv4_e6.1788677331177388345.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s32768_d128_c1_h8_kv2, 2 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15426636 | 0.17992786 | 0.0062549225 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c1_h8_kv2_e0.1788677363049453295.json |
| 0.1 | 0.99904633 | 0.00024427329 | 1.4239311e-05 | 4.26925 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c1_h8_kv2_e0.1788677363049453295.json |
| 0.03 | 0.99999619 | 3.3614674e-06 | 2.9988587e-07 | 4.27099 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c1_h8_kv2_e1.1788677394930106689.json |
| 0.01 | 1 | 3.2967088e-06 | 2.9988587e-07 | 4.27525 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c1_h8_kv2_e2.1788677426804509548.json |
| 0.003 | 1 | 3.2967088e-06 | 2.9988587e-07 | 4.27421 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c1_h8_kv2_e3.1788677458708527522.json |
| 0.001 | 1 | 3.2967088e-06 | 2.9988587e-07 | 4.28615 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c1_h8_kv2_e4.1788677490587608074.json |
| 0.0003 | 1 | 3.2967088e-06 | 2.9988587e-07 | 4.27343 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c1_h8_kv2_e5.1788677522498899481.json |
| 0.0001 | 1 | 3.2967088e-06 | 2.9988587e-07 | 4.26311 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b2_decode_s32768_d128_c1_h8_kv2_e6.1788677554388490806.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s2048_d64_c0_h4_kv4, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15476632 | 0.051626598 | 0.011574124 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c0_h4_kv4_e0.1788677587448848467.json |
| 0.1 | 0.85484147 | 0.0048344069 | 0.0034894086 | 1.4366 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c0_h4_kv4_e0.1788677587448848467.json |
| 0.03 | 0.97952533 | 0.0008361155 | 0.0011056885 | 1.16945 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c0_h4_kv4_e1.1788677620505599489.json |
| 0.01 | 0.99808323 | 0.000126675 | 0.00027102884 | 1.0067 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c0_h4_kv4_e2.1788677653584488628.json |
| 0.003 | 0.99991775 | 1.1209115e-05 | 4.5716763e-05 | 0.996607 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c0_h4_kv4_e3.1788677686667989520.json |
| 0.001 | 0.9999966 | 1.6026936e-06 | 4.9117953e-06 | 0.996414 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c0_h4_kv4_e4.1788677719754789683.json |
| 0.0003 | 0.99999994 | 1.0528553e-06 | 4.4703484e-07 | 1.03533 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c0_h4_kv4_e5.1788677752861046355.json |
| 0.0001 | 0.99999994 | 1.0528553e-06 | 4.4703484e-07 | 0.999037 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c0_h4_kv4_e6.1788677785966040845.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: 0.003.

## prefill_s2048_d64_c0_h8_kv2, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15482038 | 0.04885712 | 0.010712359 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c0_h8_kv2_e0.1788677820437107907.json |
| 0.1 | 0.85633966 | 0.0045869956 | 0.0049492642 | 1.49379 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c0_h8_kv2_e0.1788677820437107907.json |
| 0.03 | 0.97980037 | 0.00081171104 | 0.0013868734 | 1.12985 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c0_h8_kv2_e1.1788677854906510004.json |
| 0.01 | 0.99809292 | 0.0001295425 | 0.00042644143 | 1.02316 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c0_h8_kv2_e2.1788677889378061970.json |
| 0.003 | 0.99991301 | 1.3594357e-05 | 7.262826e-05 | 1.02895 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c0_h8_kv2_e3.1788677923866389658.json |
| 0.001 | 0.99999651 | 1.6904103e-06 | 1.193583e-05 | 1.02949 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c0_h8_kv2_e4.1788677958350971851.json |
| 0.0003 | 0.99999985 | 1.0458882e-06 | 1.3522804e-06 | 1.03846 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c0_h8_kv2_e5.1788677992817497546.json |
| 0.0001 | 1 | 1.0392894e-06 | 5.364418e-07 | 1.02886 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c0_h8_kv2_e6.1788678027273734709.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s2048_d64_c1_h4_kv4, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15485391 | 0.054160442 | 0.15947193 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c1_h4_kv4_e0.1788678060270217370.json |
| 0.1 | 0.88072271 | 0.0021189919 | 0.0068162084 | 1.15848 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c1_h4_kv4_e0.1788678060270217370.json |
| 0.03 | 0.98452358 | 0.00036294202 | 0.0024578571 | 1.04541 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c1_h4_kv4_e1.1788678093263751482.json |
| 0.01 | 0.99859902 | 6.7185569e-05 | 0.0010961592 | 1.02635 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c1_h4_kv4_e2.1788678126273714844.json |
| 0.003 | 0.99993625 | 1.1725482e-05 | 0.00020766258 | 1.02975 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c1_h4_kv4_e3.1788678159284721850.json |
| 0.001 | 0.99999619 | 1.1919785e-06 | 2.1185726e-05 | 1.02821 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c1_h4_kv4_e4.1788678192308502136.json |
| 0.0003 | 0.99999988 | 4.7251942e-07 | 7.7486038e-07 | 1.0221 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c1_h4_kv4_e5.1788678225341536674.json |
| 0.0001 | 1 | 4.723042e-07 | 7.7486038e-07 | 1.02682 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c1_h4_kv4_e6.1788678258321495391.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s2048_d64_c1_h8_kv2, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15498491 | 0.053595587 | 0.21413423 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c1_h8_kv2_e0.1788678292615010059.json |
| 0.1 | 0.88377024 | 0.0019987133 | 0.0094543695 | 1.22018 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c1_h8_kv2_e0.1788678292615010059.json |
| 0.03 | 0.98523855 | 0.00030695147 | 0.0011937618 | 1.11365 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c1_h8_kv2_e1.1788678326870036240.json |
| 0.01 | 0.99872711 | 4.2797017e-05 | 0.00037424266 | 1.08422 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c1_h8_kv2_e2.1788678361186491501.json |
| 0.003 | 0.9999496 | 3.6544493e-06 | 6.0260296e-05 | 1.08046 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c1_h8_kv2_e3.1788678395469245308.json |
| 0.001 | 0.99999839 | 5.666217e-07 | 5.8114529e-06 | 1.08217 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c1_h8_kv2_e4.1788678429776992373.json |
| 0.0003 | 1 | 4.6551761e-07 | 7.7486038e-07 | 1.07766 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c1_h8_kv2_e5.1788678464056162502.json |
| 0.0001 | 1 | 4.6551761e-07 | 7.7486038e-07 | 1.08096 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d64_c1_h8_kv2_e6.1788678498378810377.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s2048_d128_c0_h4_kv4, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15487731 | 0.051502696 | 0.0098044686 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c0_h4_kv4_e0.1788678532105293533.json |
| 0.1 | 0.79538774 | 0.006536637 | 0.0044241752 | 1.36069 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c0_h4_kv4_e0.1788678532105293533.json |
| 0.03 | 0.96948111 | 0.0011275602 | 0.0014421344 | 1.13243 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c0_h4_kv4_e1.1788678565895796964.json |
| 0.01 | 0.99740207 | 0.00016054707 | 0.00044924021 | 1.01987 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c0_h4_kv4_e2.1788678599869768478.json |
| 0.003 | 0.99991381 | 1.3091802e-05 | 4.8652291e-05 | 1.03813 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c0_h4_kv4_e3.1788678633859814855.json |
| 0.001 | 0.9999975 | 1.8524757e-06 | 9.9195167e-06 | 1.0151 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c0_h4_kv4_e4.1788678667878835853.json |
| 0.0003 | 1 | 1.109237e-06 | 4.61936e-07 | 0.946689 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c0_h4_kv4_e5.1788678701669329234.json |
| 0.0001 | 1 | 1.109237e-06 | 4.61936e-07 | 0.998104 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c0_h4_kv4_e6.1788678735471296443.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: 0.0003.

## prefill_s2048_d128_c0_h8_kv2, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15491793 | 0.05103938 | 0.010057822 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c0_h8_kv2_e0.1788678771134598581.json |
| 0.1 | 0.79751295 | 0.0064070264 | 0.005843997 | 1.38247 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c0_h8_kv2_e0.1788678771134598581.json |
| 0.03 | 0.97005776 | 0.0011237874 | 0.0029889941 | 1.13418 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c0_h8_kv2_e1.1788678806829109257.json |
| 0.01 | 0.99742615 | 0.00017000393 | 0.00081989914 | 1.02789 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c0_h8_kv2_e2.1788678842591080482.json |
| 0.003 | 0.99990898 | 1.7449712e-05 | 0.00015425682 | 1.04701 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c0_h8_kv2_e3.1788678878267359191.json |
| 0.001 | 0.99999708 | 1.9503769e-06 | 1.4125369e-05 | 1.00017 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c0_h8_kv2_e4.1788678914001059424.json |
| 0.0003 | 0.99999988 | 1.1097891e-06 | 1.7508864e-06 | 0.981085 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c0_h8_kv2_e5.1788678949714174511.json |
| 0.0001 | 1 | 1.0992534e-06 | 5.2154064e-07 | 1.04087 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c0_h8_kv2_e6.1788678985397997828.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: 0.0003.

## prefill_s2048_d128_c1_h4_kv4, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15493052 | 0.049488818 | 0.12689772 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c1_h4_kv4_e0.1788679019028825145.json |
| 0.1 | 0.83105493 | 0.0028335765 | 0.0071859062 | 1.15475 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c1_h4_kv4_e0.1788679019028825145.json |
| 0.03 | 0.97719007 | 0.00045095899 | 0.0023455024 | 1.03974 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c1_h4_kv4_e1.1788679052625255109.json |
| 0.01 | 0.99815077 | 6.0471607e-05 | 0.00046985596 | 1.02106 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c1_h4_kv4_e2.1788679086260042981.json |
| 0.003 | 0.99994507 | 4.8571013e-06 | 8.3446503e-05 | 1.01632 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c1_h4_kv4_e3.1788679119849446632.json |
| 0.001 | 0.99999857 | 5.7899849e-07 | 2.6524067e-06 | 1.01707 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c1_h4_kv4_e4.1788679153442920555.json |
| 0.0003 | 1 | 5.2310303e-07 | 1.3113022e-06 | 1.01784 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c1_h4_kv4_e5.1788679187022630942.json |
| 0.0001 | 1 | 5.2310303e-07 | 1.3113022e-06 | 1.01253 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c1_h4_kv4_e6.1788679220625682408.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s2048_d128_c1_h8_kv2, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15493785 | 0.054637708 | 0.38526094 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c1_h8_kv2_e0.1788679255909644117.json |
| 0.1 | 0.82459402 | 0.0029981683 | 0.0083121657 | 1.20667 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c1_h8_kv2_e0.1788679255909644117.json |
| 0.03 | 0.97499805 | 0.00048394561 | 0.0019664727 | 1.06886 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c1_h8_kv2_e1.1788679291306879741.json |
| 0.01 | 0.99788674 | 6.7133464e-05 | 0.00054724514 | 1.05155 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c1_h8_kv2_e2.1788679326683088074.json |
| 0.003 | 0.99993059 | 5.4608614e-06 | 7.840991e-05 | 1.0542 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c1_h8_kv2_e3.1788679362064125444.json |
| 0.001 | 0.99999845 | 7.4862385e-07 | 2.2053719e-05 | 1.04043 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c1_h8_kv2_e4.1788679397383656417.json |
| 0.0003 | 1 | 5.2431629e-07 | 8.3446503e-07 | 1.04503 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c1_h8_kv2_e5.1788679432766154069.json |
| 0.0001 | 1 | 5.2431629e-07 | 8.3446503e-07 | 1.04255 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_prefill_s2048_d128_c1_h8_kv2_e6.1788679468070888094.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s8192_d64_c0_h4_kv4, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15473209 | 0.049678112 | 0.0049500382 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c0_h4_kv4_e0.1788679521385251574.json |
| 0.1 | 0.76840696 | 0.0073920965 | 0.0028506592 | 1.64742 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c0_h4_kv4_e0.1788679521385251574.json |
| 0.03 | 0.95928398 | 0.001418448 | 0.0011260659 | 1.32435 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c0_h4_kv4_e1.1788679574937119230.json |
| 0.01 | 0.99543526 | 0.00023509334 | 0.000342004 | 1.12712 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c0_h4_kv4_e2.1788679628452039503.json |
| 0.003 | 0.99976243 | 2.4407735e-05 | 6.6047534e-05 | 1.08454 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c0_h4_kv4_e3.1788679681840734209.json |
| 0.001 | 0.99998964 | 3.1825648e-06 | 7.7858567e-06 | 1.08483 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c0_h4_kv4_e4.1788679735194782989.json |
| 0.0003 | 0.99999979 | 1.9828835e-06 | 9.1642141e-07 | 1.08104 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c0_h4_kv4_e5.1788679788990326507.json |
| 0.0001 | 0.99999999 | 1.9780389e-06 | 4.4330955e-07 | 1.08378 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c0_h4_kv4_e6.1788679842204213140.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s8192_d64_c0_h8_kv2, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15476261 | 0.04999231 | 0.0050276238 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c0_h8_kv2_e0.1788679917291187114.json |
| 0.1 | 0.76808519 | 0.0074885467 | 0.0029697027 | 1.67276 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c0_h8_kv2_e0.1788679917291187114.json |
| 0.03 | 0.95917583 | 0.001460326 | 0.0014633592 | 1.34592 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c0_h8_kv2_e1.1788679992147917880.json |
| 0.01 | 0.99538633 | 0.0002455222 | 0.00034739077 | 1.13928 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c0_h8_kv2_e2.1788680067409149890.json |
| 0.003 | 0.99975668 | 2.6332694e-05 | 5.2470714e-05 | 1.10079 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c0_h8_kv2_e3.1788680142867116442.json |
| 0.001 | 0.99998925 | 3.2776828e-06 | 8.2582701e-06 | 1.10103 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c0_h8_kv2_e4.1788680217719755488.json |
| 0.0003 | 0.9999998 | 2.0070661e-06 | 6.4820051e-07 | 1.09892 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c0_h8_kv2_e5.1788680292758239268.json |
| 0.0001 | 1 | 2.0014248e-06 | 5.1781535e-07 | 1.10085 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c0_h8_kv2_e6.1788680367910243646.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s8192_d64_c1_h4_kv4, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15476452 | 0.050388729 | 0.14687811 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c1_h4_kv4_e0.1788680419538172757.json |
| 0.1 | 0.80032459 | 0.0031861697 | 0.0050213039 | 1.43875 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c1_h4_kv4_e0.1788680419538172757.json |
| 0.03 | 0.96712263 | 0.00057388287 | 0.0018468648 | 1.17865 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c1_h4_kv4_e1.1788680470978831036.json |
| 0.01 | 0.99645588 | 9.4121552e-05 | 0.00061342306 | 1.11406 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c1_h4_kv4_e2.1788680522707891282.json |
| 0.003 | 0.99981811 | 1.0117286e-05 | 0.00010592141 | 1.11044 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c1_h4_kv4_e3.1788680574654272944.json |
| 0.001 | 0.99999143 | 1.2589437e-06 | 8.739531e-06 | 1.1123 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c1_h4_kv4_e4.1788680626270509688.json |
| 0.0003 | 0.99999982 | 7.650749e-07 | 1.013279e-06 | 1.12627 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c1_h4_kv4_e5.1788680678064737283.json |
| 0.0001 | 1 | 7.6213725e-07 | 1.013279e-06 | 1.12379 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c1_h4_kv4_e6.1788680729932562962.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s8192_d64_c1_h8_kv2, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15480391 | 0.053145173 | 0.32249433 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c1_h8_kv2_e0.1788680801317748311.json |
| 0.1 | 0.80158097 | 0.003213485 | 0.005045874 | 1.54238 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c1_h8_kv2_e0.1788680801317748311.json |
| 0.03 | 0.96746689 | 0.00057466903 | 0.0017884178 | 1.24123 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c1_h8_kv2_e1.1788680873061147845.json |
| 0.01 | 0.99651167 | 9.2876271e-05 | 0.00043879449 | 1.16485 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c1_h8_kv2_e2.1788680944904406668.json |
| 0.003 | 0.99982351 | 9.3318267e-06 | 6.4825639e-05 | 1.15461 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c1_h8_kv2_e3.1788681017141385867.json |
| 0.001 | 0.99999263 | 1.1596955e-06 | 7.4007548e-06 | 1.1589 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c1_h8_kv2_e4.1788681088962815635.json |
| 0.0003 | 0.99999986 | 7.6344711e-07 | 1.1138618e-06 | 1.16574 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c1_h8_kv2_e5.1788681160345908423.json |
| 0.0001 | 1 | 7.6104634e-07 | 8.6426735e-07 | 1.16003 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d64_c1_h8_kv2_e6.1788681231521295678.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s8192_d128_c0_h4_kv4, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15479607 | 0.051741519 | 0.0052246636 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c0_h4_kv4_e0.1788681293585063834.json |
| 0.1 | 0.68717849 | 0.0102492 | 0.0039010993 | 1.62901 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c0_h4_kv4_e0.1788681293585063834.json |
| 0.03 | 0.93992909 | 0.0020164375 | 0.0016207919 | 1.44888 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c0_h4_kv4_e1.1788681355351974575.json |
| 0.01 | 0.99355127 | 0.00033149435 | 0.00076436996 | 1.26807 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c0_h4_kv4_e2.1788681416690109008.json |
| 0.003 | 0.99972507 | 3.4747674e-05 | 0.00011818111 | 1.20018 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c0_h4_kv4_e3.1788681478491676136.json |
| 0.001 | 0.99999074 | 3.8888311e-06 | 1.4971942e-05 | 1.16467 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c0_h4_kv4_e4.1788681540085998054.json |
| 0.0003 | 0.99999985 | 2.0422155e-06 | 1.3168901e-06 | 1.18498 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c0_h4_kv4_e5.1788681602087294140.json |
| 0.0001 | 1 | 2.0339244e-06 | 4.8056245e-07 | 1.18049 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c0_h4_kv4_e6.1788681664007088100.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s8192_d128_c0_h8_kv2, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15480002 | 0.05068883 | 0.0050097202 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c0_h8_kv2_e0.1788681755968039260.json |
| 0.1 | 0.68859768 | 0.0099822194 | 0.0034301146 | 1.84164 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c0_h8_kv2_e0.1788681755968039260.json |
| 0.03 | 0.94053872 | 0.0019416055 | 0.0014769807 | 1.6214 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c0_h8_kv2_e1.1788681847470248037.json |
| 0.01 | 0.99369137 | 0.00030785602 | 0.00045741349 | 1.40927 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c0_h8_kv2_e2.1788681939269068428.json |
| 0.003 | 0.99974112 | 2.9554894e-05 | 8.8773668e-05 | 1.33499 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c0_h8_kv2_e3.1788682031418214710.json |
| 0.001 | 0.99999216 | 3.3335546e-06 | 1.257658e-05 | 1.3243 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c0_h8_kv2_e4.1788682123067990362.json |
| 0.0003 | 0.99999991 | 2.023326e-06 | 5.4575503e-07 | 1.3347 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c0_h8_kv2_e5.1788682214758002238.json |
| 0.0001 | 1 | 2.021799e-06 | 5.364418e-07 | 1.34285 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c0_h8_kv2_e6.1788682306814095656.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s8192_d128_c1_h4_kv4, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15481279 | 0.051360065 | 0.15355158 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c1_h4_kv4_e0.1788682366252556942.json |
| 0.1 | 0.72703367 | 0.0044865416 | 0.01048629 | 1.507 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c1_h4_kv4_e0.1788682366252556942.json |
| 0.03 | 0.951374 | 0.00080577552 | 0.0023919791 | 1.28184 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c1_h4_kv4_e1.1788682425688781984.json |
| 0.01 | 0.99508446 | 0.00012270551 | 0.00053866091 | 1.2281 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c1_h4_kv4_e2.1788682485472529380.json |
| 0.003 | 0.99979888 | 1.1473794e-05 | 6.0536899e-05 | 1.20921 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c1_h4_kv4_e3.1788682544857019444.json |
| 0.001 | 0.99999335 | 1.2777167e-06 | 9.9390745e-06 | 1.20288 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c1_h4_kv4_e4.1788682604085083297.json |
| 0.0003 | 0.99999989 | 8.1644157e-07 | 8.9406967e-07 | 1.20161 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c1_h4_kv4_e5.1788682663296252632.json |
| 0.0001 | 1 | 8.1572531e-07 | 8.9406967e-07 | 1.19491 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c1_h4_kv4_e6.1788682722462918744.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## prefill_s8192_d128_c1_h8_kv2, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15482433 | 0.053477929 | 0.34167004 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c1_h8_kv2_e0.1788682810167515448.json |
| 0.1 | 0.72701566 | 0.0042992554 | 0.0098370314 | 1.69419 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c1_h8_kv2_e0.1788682810167515448.json |
| 0.03 | 0.95135472 | 0.00075554545 | 0.0021122359 | 1.417 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c1_h8_kv2_e1.1788682897205815569.json |
| 0.01 | 0.99510109 | 0.00011001902 | 0.00063484907 | 1.34832 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c1_h8_kv2_e2.1788682984341729854.json |
| 0.003 | 0.9998091 | 1.0553808e-05 | 0.00016528368 | 1.3434 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c1_h8_kv2_e3.1788683070930199397.json |
| 0.001 | 0.99999465 | 1.1065746e-06 | 1.5079975e-05 | 1.33487 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c1_h8_kv2_e4.1788683157856855291.json |
| 0.0003 | 0.99999995 | 8.0108963e-07 | 1.1920929e-06 | 1.3507 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c1_h8_kv2_e5.1788683244928935997.json |
| 0.0001 | 1 | 8.0073939e-07 | 1.1920929e-06 | 1.36008 | RED_BOUND_EXCEEDED | artifacts/apa_sp2/gpu/sweep_b4_prefill_s8192_d128_c1_h8_kv2_e6.1788683332601044062.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s8192_d64_c0_h4_kv4, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15661621 | 0.056465288 | 0.002589343 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c0_h4_kv4_e0.1788683364632167116.json |
| 0.1 | 0.82800293 | 0.0051669493 | 0.00024013268 | 1.98157 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c0_h4_kv4_e0.1788683364632167116.json |
| 0.03 | 0.97677612 | 0.00081479083 | 4.9670693e-05 | 1.93554 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c0_h4_kv4_e1.1788683396623164023.json |
| 0.01 | 0.99804688 | 8.8305865e-05 | 5.7592988e-06 | 1.93284 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c0_h4_kv4_e2.1788683428625198427.json |
| 0.003 | 0.99990845 | 5.955237e-06 | 4.4400804e-07 | 1.92769 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c0_h4_kv4_e3.1788683460636311372.json |
| 0.001 | 1 | 1.4565974e-06 | 8.9406967e-08 | 1.92974 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c0_h4_kv4_e4.1788683492642860881.json |
| 0.0003 | 1 | 1.4565974e-06 | 8.9406967e-08 | 1.92182 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c0_h4_kv4_e5.1788683524643381168.json |
| 0.0001 | 1 | 1.4565974e-06 | 8.9406967e-08 | 1.92461 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c0_h4_kv4_e6.1788683556694611727.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s8192_d64_c0_h8_kv2, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15548706 | 0.047495607 | 0.0026585311 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c0_h8_kv2_e0.1788683588717518786.json |
| 0.1 | 0.87017822 | 0.0043763261 | 0.0004655011 | 1.94048 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c0_h8_kv2_e0.1788683588717518786.json |
| 0.03 | 0.98352051 | 0.00087998808 | 0.0001391042 | 1.93356 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c0_h8_kv2_e1.1788683620736051115.json |
| 0.01 | 0.99853516 | 9.8422652e-05 | 1.3781711e-05 | 1.9272 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c0_h8_kv2_e2.1788683652724480487.json |
| 0.003 | 0.99990845 | 1.0314409e-05 | 1.3113022e-06 | 1.93034 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c0_h8_kv2_e3.1788683684730804669.json |
| 0.001 | 1 | 1.573714e-06 | 1.5832484e-07 | 1.92638 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c0_h8_kv2_e4.1788683716741194070.json |
| 0.0003 | 1 | 1.573714e-06 | 1.5832484e-07 | 1.92873 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c0_h8_kv2_e5.1788683748754541680.json |
| 0.0001 | 1 | 1.573714e-06 | 1.5832484e-07 | 1.92784 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c0_h8_kv2_e6.1788683780776654625.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s8192_d64_c1_h4_kv4, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15408325 | 0.050876642 | 0.0032143928 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c1_h4_kv4_e0.1788683812850934494.json |
| 0.1 | 0.92358398 | 0.0029164189 | 0.00015224144 | 1.92225 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c1_h4_kv4_e0.1788683812850934494.json |
| 0.03 | 0.99441528 | 0.0003431025 | 2.2295862e-05 | 1.93537 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c1_h4_kv4_e1.1788683844902985554.json |
| 0.01 | 0.99984741 | 2.0977331e-05 | 1.9866857e-06 | 1.92818 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c1_h4_kv4_e2.1788683876951251518.json |
| 0.003 | 1 | 1.5825695e-06 | 1.4901161e-07 | 1.91954 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c1_h4_kv4_e3.1788683909041297486.json |
| 0.001 | 1 | 1.5825695e-06 | 1.4901161e-07 | 1.92703 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c1_h4_kv4_e4.1788683941104030256.json |
| 0.0003 | 1 | 1.5825695e-06 | 1.4901161e-07 | 1.92388 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c1_h4_kv4_e5.1788683973162860196.json |
| 0.0001 | 1 | 1.5825695e-06 | 1.4901161e-07 | 1.92878 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c1_h4_kv4_e6.1788684005232045856.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s8192_d64_c1_h8_kv2, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15454102 | 0.05135372 | 0.0029019397 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c1_h8_kv2_e0.1788684037301552163.json |
| 0.1 | 0.90666199 | 0.0037357478 | 0.00025411509 | 1.95042 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c1_h8_kv2_e0.1788684037301552163.json |
| 0.03 | 0.99224854 | 0.00046116122 | 3.4620985e-05 | 1.92216 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c1_h8_kv2_e1.1788684069364122517.json |
| 0.01 | 0.99977112 | 2.2677841e-05 | 2.0079315e-06 | 1.93053 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c1_h8_kv2_e2.1788684101423661940.json |
| 0.003 | 1 | 1.6610397e-06 | 1.3038516e-07 | 1.92827 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c1_h8_kv2_e3.1788684133491788432.json |
| 0.001 | 1 | 1.6610397e-06 | 1.3038516e-07 | 1.93533 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c1_h8_kv2_e4.1788684165580851954.json |
| 0.0003 | 1 | 1.6610397e-06 | 1.3038516e-07 | 1.91887 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c1_h8_kv2_e5.1788684197653413161.json |
| 0.0001 | 1 | 1.6610397e-06 | 1.3038516e-07 | 1.91466 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d64_c1_h8_kv2_e6.1788684229718133729.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s8192_d128_c0_h4_kv4, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15472412 | 0.052459064 | 0.0029588789 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c0_h4_kv4_e0.1788684261872138074.json |
| 0.1 | 0.81118774 | 0.0059069212 | 0.00040250085 | 1.50262 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c0_h4_kv4_e0.1788684261872138074.json |
| 0.03 | 0.97808838 | 0.00077966314 | 5.2317977e-05 | 1.46483 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c0_h4_kv4_e1.1788684294034330540.json |
| 0.01 | 0.99899292 | 6.5877469e-05 | 5.306676e-06 | 1.45873 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c0_h4_kv4_e2.1788684326198884877.json |
| 0.003 | 1 | 1.5871164e-06 | 1.1920929e-07 | 1.46591 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c0_h4_kv4_e3.1788684358375381934.json |
| 0.001 | 1 | 1.5871164e-06 | 1.1920929e-07 | 1.46521 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c0_h4_kv4_e4.1788684390580879954.json |
| 0.0003 | 1 | 1.5871164e-06 | 1.1920929e-07 | 1.47449 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c0_h4_kv4_e5.1788684422758392654.json |
| 0.0001 | 1 | 1.5871164e-06 | 1.1920929e-07 | 1.46756 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c0_h4_kv4_e6.1788684454941507966.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s8192_d128_c0_h8_kv2, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15505981 | 0.049677925 | 0.0030795576 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c0_h8_kv2_e0.1788684487086206565.json |
| 0.1 | 0.8223877 | 0.0057958626 | 0.00044504367 | 1.69343 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c0_h8_kv2_e0.1788684487086206565.json |
| 0.03 | 0.97897339 | 0.0007195314 | 9.5883384e-05 | 1.68707 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c0_h8_kv2_e1.1788684519226797499.json |
| 0.01 | 0.99862671 | 8.3589026e-05 | 1.0183081e-05 | 1.68666 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c0_h8_kv2_e2.1788684551381857983.json |
| 0.003 | 0.99996948 | 4.6972824e-06 | 6.1653554e-07 | 1.683 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c0_h8_kv2_e3.1788684583531231273.json |
| 0.001 | 1 | 1.7194956e-06 | 2.682209e-07 | 1.68393 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c0_h8_kv2_e4.1788684615688234412.json |
| 0.0003 | 1 | 1.7194956e-06 | 2.682209e-07 | 1.68483 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c0_h8_kv2_e5.1788684647848101876.json |
| 0.0001 | 1 | 1.7194956e-06 | 2.682209e-07 | 1.68708 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c0_h8_kv2_e6.1788684680017527744.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s8192_d128_c1_h4_kv4, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15634155 | 0.050989814 | 0.0025068112 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c1_h4_kv4_e0.1788684712226547908.json |
| 0.1 | 0.84558105 | 0.0048150279 | 0.00029209442 | 1.46534 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c1_h4_kv4_e0.1788684712226547908.json |
| 0.03 | 0.98486328 | 0.0006047916 | 3.7308782e-05 | 1.38403 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c1_h4_kv4_e1.1788684744460698526.json |
| 0.01 | 0.99935913 | 5.5766211e-05 | 4.6007335e-06 | 1.38363 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c1_h4_kv4_e2.1788684776678554662.json |
| 0.003 | 1 | 1.6879153e-06 | 1.5646219e-07 | 1.38761 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c1_h4_kv4_e3.1788684808913068279.json |
| 0.001 | 1 | 1.6879153e-06 | 1.5646219e-07 | 1.59054 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c1_h4_kv4_e4.1788684841173647468.json |
| 0.0003 | 1 | 1.6879153e-06 | 1.5646219e-07 | 1.58701 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c1_h4_kv4_e5.1788684873395116750.json |
| 0.0001 | 1 | 1.6879153e-06 | 1.5646219e-07 | 1.58566 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c1_h4_kv4_e6.1788684905628091298.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s8192_d128_c1_h8_kv2, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15499878 | 0.047798492 | 0.0029866456 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c1_h8_kv2_e0.1788684937818285174.json |
| 0.1 | 0.77944946 | 0.0062415226 | 0.00053489115 | 1.69515 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c1_h8_kv2_e0.1788684937818285174.json |
| 0.03 | 0.96917725 | 0.0010106695 | 0.00012652948 | 1.69078 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c1_h8_kv2_e1.1788684970009140618.json |
| 0.01 | 0.99757385 | 0.00013586572 | 1.8741237e-05 | 1.68777 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c1_h8_kv2_e2.1788685002211365572.json |
| 0.003 | 0.99989319 | 9.2817003e-06 | 1.4975667e-06 | 1.68524 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c1_h8_kv2_e3.1788685034420458725.json |
| 0.001 | 1 | 1.6157862e-06 | 1.6391277e-07 | 1.68911 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c1_h8_kv2_e4.1788685066654812077.json |
| 0.0003 | 1 | 1.6157862e-06 | 1.6391277e-07 | 1.6848 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c1_h8_kv2_e5.1788685098865001510.json |
| 0.0001 | 1 | 1.6157862e-06 | 1.6391277e-07 | 1.68327 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s8192_d128_c1_h8_kv2_e6.1788685131077239624.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s32768_d64_c0_h4_kv4, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15457153 | 0.055722317 | 0.0014764884 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c0_h4_kv4_e0.1788685163454820695.json |
| 0.1 | 0.88330841 | 0.0037989239 | 0.00013247412 | 3.54661 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c0_h4_kv4_e0.1788685163454820695.json |
| 0.03 | 0.98905945 | 0.00051650151 | 1.5503727e-05 | 3.50002 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c0_h4_kv4_e1.1788685195824722278.json |
| 0.01 | 0.99951172 | 4.8053093e-05 | 2.003293e-06 | 3.48711 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c0_h4_kv4_e2.1788685228193657853.json |
| 0.003 | 1 | 3.1858423e-06 | 1.4901161e-07 | 3.48638 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c0_h4_kv4_e3.1788685260583207712.json |
| 0.001 | 1 | 3.1858423e-06 | 1.4901161e-07 | 3.48186 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c0_h4_kv4_e4.1788685292967404500.json |
| 0.0003 | 1 | 3.1858423e-06 | 1.4901161e-07 | 3.47828 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c0_h4_kv4_e5.1788685325364877884.json |
| 0.0001 | 1 | 3.1858423e-06 | 1.4901161e-07 | 3.48845 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c0_h4_kv4_e6.1788685357751005377.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s32768_d64_c0_h8_kv2, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15506744 | 0.053175782 | 0.0014491653 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c0_h8_kv2_e0.1788685390113997054.json |
| 0.1 | 0.87561798 | 0.0038489917 | 0.00010866299 | 2.58638 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c0_h8_kv2_e0.1788685390113997054.json |
| 0.03 | 0.9874649 | 0.00051967065 | 1.979433e-05 | 2.53906 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c0_h8_kv2_e1.1788685422455518198.json |
| 0.01 | 0.99935913 | 4.7553306e-05 | 2.1164306e-06 | 2.53168 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c0_h8_kv2_e2.1788685454805255891.json |
| 0.003 | 0.99998856 | 5.6637114e-06 | 3.2503158e-07 | 2.52476 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c0_h8_kv2_e3.1788685487164720948.json |
| 0.001 | 1 | 3.4808532e-06 | 1.527369e-07 | 2.52458 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c0_h8_kv2_e4.1788685519529847209.json |
| 0.0003 | 1 | 3.4808532e-06 | 1.527369e-07 | 2.52423 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c0_h8_kv2_e5.1788685551878992683.json |
| 0.0001 | 1 | 3.4808532e-06 | 1.527369e-07 | 2.5182 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c0_h8_kv2_e6.1788685584233839693.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s32768_d64_c1_h4_kv4, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15428162 | 0.052614155 | 0.0015495094 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c1_h4_kv4_e0.1788685616651567500.json |
| 0.1 | 0.90200806 | 0.0031755208 | 8.5464679e-05 | 3.53478 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c1_h4_kv4_e0.1788685616651567500.json |
| 0.03 | 0.99297333 | 0.00038362433 | 1.4173798e-05 | 3.46423 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c1_h4_kv4_e1.1788685649088634747.json |
| 0.01 | 0.9997406 | 2.3613342e-05 | 7.613562e-07 | 3.46905 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c1_h4_kv4_e2.1788685681516403038.json |
| 0.003 | 1 | 2.7083847e-06 | 1.0803342e-07 | 3.47864 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c1_h4_kv4_e3.1788685713946602990.json |
| 0.001 | 1 | 2.7083847e-06 | 1.0803342e-07 | 3.48199 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c1_h4_kv4_e4.1788685746386066148.json |
| 0.0003 | 1 | 2.7083847e-06 | 1.0803342e-07 | 3.49129 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c1_h4_kv4_e5.1788685778856810607.json |
| 0.0001 | 1 | 2.7083847e-06 | 1.0803342e-07 | 3.49295 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c1_h4_kv4_e6.1788685811297626608.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s32768_d64_c1_h8_kv2, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15436172 | 0.053223893 | 0.0016748263 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c1_h8_kv2_e0.1788685843687809105.json |
| 0.1 | 0.82915878 | 0.0051904003 | 0.00023603346 | 2.59352 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c1_h8_kv2_e0.1788685843687809105.json |
| 0.03 | 0.97631836 | 0.00075747154 | 3.4392346e-05 | 2.54494 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c1_h8_kv2_e1.1788685876069216274.json |
| 0.01 | 0.99822998 | 9.1576444e-05 | 4.1108578e-06 | 2.52799 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c1_h8_kv2_e2.1788685908494949790.json |
| 0.003 | 0.9999733 | 3.9146109e-06 | 1.5646219e-07 | 2.52598 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c1_h8_kv2_e3.1788685940892240834.json |
| 0.001 | 1 | 3.2851367e-06 | 1.5646219e-07 | 2.52876 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c1_h8_kv2_e4.1788685973319470687.json |
| 0.0003 | 1 | 3.2851367e-06 | 1.5646219e-07 | 2.51447 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c1_h8_kv2_e5.1788686005728180024.json |
| 0.0001 | 1 | 3.2851367e-06 | 1.5646219e-07 | 2.52247 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d64_c1_h8_kv2_e6.1788686038141366519.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s32768_d128_c0_h4_kv4, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15431976 | 0.051716618 | 0.0017752144 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c0_h4_kv4_e0.1788686070862145378.json |
| 0.1 | 0.79980469 | 0.0059033153 | 0.00019945577 | 4.81115 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c0_h4_kv4_e0.1788686070862145378.json |
| 0.03 | 0.97249603 | 0.00090182712 | 3.2717362e-05 | 4.78477 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c0_h4_kv4_e1.1788686103584132649.json |
| 0.01 | 0.99784851 | 0.0001190381 | 5.2601099e-06 | 4.75121 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c0_h4_kv4_e2.1788686136309222348.json |
| 0.003 | 0.99995422 | 4.5981657e-06 | 1.9057188e-07 | 4.77891 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c0_h4_kv4_e3.1788686169059756363.json |
| 0.001 | 1 | 3.2222446e-06 | 1.4528632e-07 | 4.77627 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c0_h4_kv4_e4.1788686201782565821.json |
| 0.0003 | 1 | 3.2222446e-06 | 1.4528632e-07 | 4.77516 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c0_h4_kv4_e5.1788686234530078036.json |
| 0.0001 | 1 | 3.2222446e-06 | 1.4528632e-07 | 4.76954 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c0_h4_kv4_e6.1788686267303688529.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s32768_d128_c0_h8_kv2, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15522003 | 0.049920705 | 0.0018328596 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c0_h8_kv2_e0.1788686299912794789.json |
| 0.1 | 0.83892441 | 0.0048069288 | 0.00016412139 | 4.33606 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c0_h8_kv2_e0.1788686299912794789.json |
| 0.03 | 0.98324966 | 0.00064521109 | 2.1938235e-05 | 4.27238 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c0_h8_kv2_e1.1788686332560669412.json |
| 0.01 | 0.99918747 | 5.838573e-05 | 2.6524067e-06 | 4.25853 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c0_h8_kv2_e2.1788686365168886519.json |
| 0.003 | 0.99999237 | 3.6375522e-06 | 1.7229468e-07 | 4.26686 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c0_h8_kv2_e3.1788686397781692803.json |
| 0.001 | 1 | 3.3318991e-06 | 1.7229468e-07 | 4.25698 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c0_h8_kv2_e4.1788686430413956678.json |
| 0.0003 | 1 | 3.3318991e-06 | 1.7229468e-07 | 4.25765 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c0_h8_kv2_e5.1788686463035461334.json |
| 0.0001 | 1 | 3.3318991e-06 | 1.7229468e-07 | 4.27704 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c0_h8_kv2_e6.1788686495646781683.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s32768_d128_c1_h4_kv4, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15505981 | 0.04935112 | 0.0015714577 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c1_h4_kv4_e0.1788686528460788373.json |
| 0.1 | 0.75828552 | 0.0070028047 | 0.00027291477 | 4.812 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c1_h4_kv4_e0.1788686528460788373.json |
| 0.03 | 0.9597702 | 0.0011741388 | 4.6047382e-05 | 4.7915 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c1_h4_kv4_e1.1788686561226237806.json |
| 0.01 | 0.99614716 | 0.00014044688 | 6.1877072e-06 | 4.81925 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c1_h4_kv4_e2.1788686594035133907.json |
| 0.003 | 0.99987793 | 1.0562325e-05 | 4.125759e-07 | 4.79808 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c1_h4_kv4_e3.1788686626829262228.json |
| 0.001 | 1 | 3.159342e-06 | 1.7136335e-07 | 4.78031 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c1_h4_kv4_e4.1788686659620184760.json |
| 0.0003 | 1 | 3.159342e-06 | 1.7136335e-07 | 4.76465 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c1_h4_kv4_e5.1788686692446405387.json |
| 0.0001 | 1 | 3.159342e-06 | 1.7136335e-07 | 4.78039 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c1_h4_kv4_e6.1788686725233984356.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.

## decode_s32768_d128_c1_h8_kv2, 4 bits

| Rule / epsilon | Refine fraction | Relative Frobenius | Max abs | Speed vs two-pass | Guarantee | Receipt |
|---|---|---|---|---|---|---|
| Two-pass p=0.15 | 0.15481186 | 0.048844423 | 0.001457595 | 1 | reference | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c1_h8_kv2_e0.1788686757898638769.json |
| 0.1 | 0.80035019 | 0.0059736204 | 0.00022901688 | 4.34096 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c1_h8_kv2_e0.1788686757898638769.json |
| 0.03 | 0.9732933 | 0.00093952563 | 4.4349581e-05 | 4.26734 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c1_h8_kv2_e1.1788686790557074368.json |
| 0.01 | 0.99793625 | 0.00011403957 | 6.0582533e-06 | 4.2701 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c1_h8_kv2_e2.1788686823216741952.json |
| 0.003 | 0.99995422 | 7.0219657e-06 | 2.9988587e-07 | 4.25901 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c1_h8_kv2_e3.1788686855911319565.json |
| 0.001 | 1 | 3.2967088e-06 | 2.9988587e-07 | 4.27196 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c1_h8_kv2_e4.1788686888591929441.json |
| 0.0003 | 1 | 3.2967088e-06 | 2.9988587e-07 | 4.27137 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c1_h8_kv2_e5.1788686921264443662.json |
| 0.0001 | 1 | 3.2967088e-06 | 2.9988587e-07 | 4.2533 | EMPIRICAL_BOUND_HELD | artifacts/apa_sp2/gpu/sweep_b4_decode_s32768_d128_c1_h8_kv2_e6.1788686953953006069.json |

First relative-Frobenius improvement: 0.1; first max-abs improvement: 0.1; first speed below 1: NOT_OBSERVED_ON_GRID.
