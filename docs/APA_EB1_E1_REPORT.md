Certified variant: KILLED
Empirical variant: NOT SUPPORTED

L1 loose layers: 9/9. L2 median pooled Spearman: 0.969241. L3 hybrid saving cells: 0/9; empirical (b): 0/9. L4 median layer slope: 0.31544007929963064.

Registration SHA-256: `806cb7baa65c7d520f6ab8d190987d7a949da344be1932519f8986ecefa13afb`

G1: PASS; G2: PASS; G3: PASS.

R2 violation count: 0 (complete registered census).

Scope for every numerical table: TQ4, 4.25 bits per element stored, computed in fp64 here; one wikitext-2 sequence, two Qwen3 models, last 64 decode rows, all query heads, causal, offline CPU numerics.

| Prediction | Result | Registered text |
|---|---|---|
| L1 | HIT | at TQ4, the certified CS score-side bound is loose. The median row-level bound/actual exceeds 10× on at least 5 of the 9 non-early layers. |
| L2 | HIT | for TQ4-K + `Vbf16`, the empirical predicted contribution ranks the actual contribution with median Spearman ≥ 0.8 over the non-early layers. |
| L3 | MISS | at the error SP achieves at δ = 8.1, arm (e) reads ≤ 0.75× SP's bits per key element on at least 6 of the 9 non-early layers. The lead expects the saving to come mostly from the values, not the keys. |
| L4 | MISS | the R5 slope of `b_min` against `log2 w_B` lies in [0.7, 1.3], the theory's 1.0. |
| L5 | HIT | R2 finds 0 violations. |
| S1 | HIT | At TQ4, the certified CS variant is KILLED under the registered row-level 10x rule. |
| S2 | HIT | For TQ4-K + Vbf16, the median non-early per-layer pooled empirical Spearman is at least 0.8. |
| S3 | HIT | The empirical variant is NOT SUPPORTED under the registered matched-error saving rule within the prescribed sweep. |

R1 prediction (pooled per-layer Spearman; per-row medians in JSON):

| Model/layer | B | Method | CS | ORACLE | Empirical |
|---|---|---|---|---|---|
| Qwen3-1.7B L11 | 64 | Vbf16 | 0.410275 | 0.735266 | 0.942456 |
| Qwen3-1.7B L11 | 64 | Vint8 | 0.410633 | 0.735863 | 0.9429 |
| Qwen3-1.7B L11 | 64 | Vcent | 0.390319 | 0.747718 | 0.965705 |
| Qwen3-1.7B L16 | 64 | Vbf16 | 0.638628 | 0.873941 | 0.973916 |
| Qwen3-1.7B L16 | 64 | Vint8 | 0.638813 | 0.874299 | 0.974217 |
| Qwen3-1.7B L16 | 64 | Vcent | 0.625121 | 0.884391 | 0.984419 |
| Qwen3-1.7B L22 | 64 | Vbf16 | 0.161678 | 0.812456 | 0.954519 |
| Qwen3-1.7B L22 | 64 | Vint8 | 0.161433 | 0.81291 | 0.954763 |
| Qwen3-1.7B L22 | 64 | Vcent | 0.00960021 | 0.777802 | 0.95599 |
| Qwen3-1.7B L27 | 64 | Vbf16 | 0.30527 | 0.859114 | 0.969241 |
| Qwen3-1.7B L27 | 64 | Vint8 | 0.305326 | 0.859307 | 0.969362 |
| Qwen3-1.7B L27 | 64 | Vcent | 0.225238 | 0.833927 | 0.978128 |
| Qwen3-4B L7 | 64 | Vbf16 | 0.726809 | 0.838836 | 0.956401 |
| Qwen3-4B L7 | 64 | Vint8 | 0.727083 | 0.839131 | 0.956623 |
| Qwen3-4B L7 | 64 | Vcent | 0.687881 | 0.843504 | 0.976747 |
| Qwen3-4B L14 | 64 | Vbf16 | 0.599022 | 0.873522 | 0.974798 |
| Qwen3-4B L14 | 64 | Vint8 | 0.600071 | 0.874375 | 0.97548 |
| Qwen3-4B L14 | 64 | Vcent | 0.569191 | 0.882179 | 0.98321 |
| Qwen3-4B L21 | 64 | Vbf16 | 0.675866 | 0.897281 | 0.976402 |
| Qwen3-4B L21 | 64 | Vint8 | 0.676306 | 0.897878 | 0.976913 |
| Qwen3-4B L21 | 64 | Vcent | 0.653841 | 0.903818 | 0.985557 |
| Qwen3-4B L28 | 64 | Vbf16 | 0.427261 | 0.917649 | 0.97234 |
| Qwen3-4B L28 | 64 | Vint8 | 0.427797 | 0.918202 | 0.972815 |
| Qwen3-4B L28 | 64 | Vcent | 0.362728 | 0.913735 | 0.980932 |
| Qwen3-4B L35 | 64 | Vbf16 | 0.336714 | 0.802712 | 0.966141 |
| Qwen3-4B L35 | 64 | Vint8 | 0.336804 | 0.802944 | 0.966246 |
| Qwen3-4B L35 | 64 | Vcent | 0.272574 | 0.787712 | 0.972387 |

R2 soundness: strict `actual > bound` comparisons, without tolerance. CS and ORACLE cover single-block, simultaneous all-block and certified allocation output; online mass is also checked. Counts by cell are in JSON. A witness is required for every violated kind, with up to ten examples per head/method.

R3 bound/actual, median / p99 (zero-actual blocks counted separately in JSON):

| Model/layer | Method | CS block | CS row | ORACLE block | ORACLE row |
|---|---|---|---|---|---|
| Qwen3-1.7B L11 | Vbf16 | 1.383e+08 / 7.938e+13 | 1.929e+08 / 1.157e+13 | 1828 / 4.586e+06 | 2819 / 4.266e+05 |
| Qwen3-1.7B L11 | Vint8 | 1.381e+08 / 7.926e+13 | 1.901e+08 / 3.418e+12 | 1831 / 4.575e+06 | 2819 / 1.263e+05 |
| Qwen3-1.7B L11 | Vint4g16 | 1.234e+08 / 6.99e+13 | 1.587e+08 / 4.94e+11 | 1684 / 4.172e+06 | 2176 / 4.123e+04 |
| Qwen3-1.7B L11 | Vcent | 3.436e+07 / 3.223e+13 | 2.348e+07 / 8.22e+10 | 561.3 / 2.168e+06 | 357.8 / 7914 |
| Qwen3-1.7B L16 | Vbf16 | 2.921e+07 / 2.025e+10 | 4.361e+07 / 5.511e+09 | 762.6 / 3.04e+04 | 1180 / 8087 |
| Qwen3-1.7B L16 | Vint8 | 2.916e+07 / 2.021e+10 | 4.199e+07 / 5.5e+09 | 765.7 / 3.025e+04 | 1182 / 7040 |
| Qwen3-1.7B L16 | Vint4g16 | 2.591e+07 / 1.862e+10 | 3.511e+07 / 5.499e+09 | 705.9 / 2.751e+04 | 1039 / 6150 |
| Qwen3-1.7B L16 | Vcent | 6.694e+06 / 6.879e+09 | 3.882e+06 / 1.459e+09 | 234.2 / 1.092e+04 | 166.1 / 1458 |
| Qwen3-1.7B L22 | Vbf16 | 8.601e+07 / 1.338e+12 | 2.995e+08 / 1.435e+12 | 837.5 / 1.199e+04 | 1875 / 1.501e+04 |
| Qwen3-1.7B L22 | Vint8 | 8.576e+07 / 1.338e+12 | 2.806e+08 / 1.437e+12 | 839.7 / 1.194e+04 | 1874 / 1.395e+04 |
| Qwen3-1.7B L22 | Vint4g16 | 7.868e+07 / 1.273e+12 | 2.605e+08 / 1.215e+12 | 801.4 / 1.086e+04 | 1607 / 8452 |
| Qwen3-1.7B L22 | Vcent | 2.37e+07 / 5.239e+11 | 3.034e+07 / 2.469e+11 | 325.2 / 5503 | 222.5 / 1455 |
| Qwen3-1.7B L27 | Vbf16 | 1.248e+10 / 2.104e+15 | 4.05e+10 / 5.666e+15 | 2543 / 8.259e+04 | 8504 / 1.73e+05 |
| Qwen3-1.7B L27 | Vint8 | 1.244e+10 / 2.103e+15 | 3.652e+10 / 5.662e+15 | 2548 / 8.021e+04 | 8405 / 1.493e+05 |
| Qwen3-1.7B L27 | Vint4g16 | 1.199e+10 / 2.057e+15 | 2.387e+10 / 5.452e+15 | 2499 / 7.519e+04 | 4890 / 5.067e+04 |
| Qwen3-1.7B L27 | Vcent | 6.477e+09 / 1.394e+15 | 1.411e+09 / 1.414e+15 | 1609 / 5.164e+04 | 396.9 / 8337 |
| Qwen3-4B L7 | Vbf16 | 7.024e+06 / 4.745e+09 | 1.216e+07 / 1.904e+10 | 405.7 / 3.761e+04 | 908.1 / 7.879e+04 |
| Qwen3-4B L7 | Vint8 | 7.021e+06 / 4.696e+09 | 1.167e+07 / 1.019e+10 | 407.1 / 3.684e+04 | 908.7 / 2.447e+04 |
| Qwen3-4B L7 | Vint4g16 | 6.376e+06 / 3.992e+09 | 7.986e+06 / 3.107e+09 | 381.7 / 3.501e+04 | 737.9 / 5805 |
| Qwen3-4B L7 | Vcent | 1.712e+06 / 1.734e+09 | 6.569e+05 / 2.811e+08 | 140.4 / 1.296e+04 | 66.54 / 879.5 |
| Qwen3-4B L14 | Vbf16 | 1.705e+08 / 1.4e+11 | 2.553e+08 / 3.97e+10 | 1382 / 7.626e+04 | 1505 / 1.568e+04 |
| Qwen3-4B L14 | Vint8 | 1.702e+08 / 1.378e+11 | 2.533e+08 / 3.678e+10 | 1386 / 7.543e+04 | 1507 / 1.385e+04 |
| Qwen3-4B L14 | Vint4g16 | 1.564e+08 / 1.223e+11 | 2.204e+08 / 1.875e+10 | 1304 / 6.812e+04 | 1336 / 7520 |
| Qwen3-4B L14 | Vcent | 4.869e+07 / 4.779e+10 | 3.735e+07 / 3.828e+09 | 500.2 / 2.998e+04 | 246.5 / 1829 |
| Qwen3-4B L21 | Vbf16 | 7.028e+07 / 7.989e+10 | 1.105e+08 / 4.526e+10 | 1095 / 3.322e+04 | 1610 / 7181 |
| Qwen3-4B L21 | Vint8 | 7.019e+07 / 7.875e+10 | 1.105e+08 / 4.511e+10 | 1099 / 3.23e+04 | 1618 / 7188 |
| Qwen3-4B L21 | Vint4g16 | 6.29e+07 / 6.921e+10 | 1.012e+08 / 3.752e+10 | 1027 / 2.775e+04 | 1460 / 5818 |
| Qwen3-4B L21 | Vcent | 1.576e+07 / 2.276e+10 | 1.467e+07 / 8.002e+09 | 343.4 / 1.015e+04 | 285.2 / 1124 |
| Qwen3-4B L28 | Vbf16 | 2.554e+08 / 4.733e+11 | 7.975e+08 / 5.251e+11 | 830.3 / 1.639e+04 | 2640 / 8.394e+04 |
| Qwen3-4B L28 | Vint8 | 2.55e+08 / 4.666e+11 | 7.854e+08 / 4.958e+11 | 832.8 / 1.586e+04 | 2640 / 6.349e+04 |
| Qwen3-4B L28 | Vint4g16 | 2.362e+08 / 4.418e+11 | 5.765e+08 / 2.366e+11 | 798.6 / 1.305e+04 | 2290 / 3.703e+04 |
| Qwen3-4B L28 | Vcent | 7.093e+07 / 2.376e+11 | 5.162e+07 / 7.139e+09 | 306.7 / 6559 | 247.3 / 1374 |
| Qwen3-4B L35 | Vbf16 | 2.724e+10 / 3.916e+17 | 7.291e+10 / 1.58e+17 | 3785 / 8.451e+05 | 7936 / 3.035e+05 |
| Qwen3-4B L35 | Vint8 | 2.721e+10 / 3.914e+17 | 6.945e+10 / 1.574e+17 | 3792 / 8.393e+05 | 7862 / 1.307e+05 |
| Qwen3-4B L35 | Vint4g16 | 2.616e+10 / 3.801e+17 | 4.146e+10 / 1.403e+17 | 3732 / 8.146e+05 | 6201 / 4.781e+04 |
| Qwen3-4B L35 | Vcent | 1.478e+10 / 2.166e+17 | 3.941e+09 / 3.321e+16 | 2373 / 5.192e+05 | 795.1 / 9853 |

R4 bits per key element at SP delta 8.1 achieved error (each arm interpolates its own measured curve; unbracketed is not evidence of a saving):

| Model/layer | SP error | SP bits | a bits | b bits | c bits | e bits | e/SP |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B L11 | 0.00261258 | 22.3316 | UNBRACKETED | UNBRACKETED | UNBRACKETED | UNBRACKETED | UNBRACKETED |
| Qwen3-1.7B L16 | 0.00341645 | 21.7388 | UNBRACKETED | UNBRACKETED | UNBRACKETED | 20.2256 | 0.930391 |
| Qwen3-1.7B L22 | 0.00664877 | 21.2379 | UNBRACKETED | UNBRACKETED | UNBRACKETED | 18.6103 | 0.876278 |
| Qwen3-1.7B L27 | 0.00829102 | 20.8476 | UNBRACKETED | UNBRACKETED | UNBRACKETED | UNBRACKETED | UNBRACKETED |
| Qwen3-4B L7 | 0.00708467 | 21.0298 | UNBRACKETED | UNBRACKETED | UNBRACKETED | UNBRACKETED | UNBRACKETED |
| Qwen3-4B L14 | 0.00227471 | 22.0906 | UNBRACKETED | UNBRACKETED | UNBRACKETED | UNBRACKETED | UNBRACKETED |
| Qwen3-4B L21 | 0.00184056 | 22.1468 | UNBRACKETED | UNBRACKETED | UNBRACKETED | 18.311 | 0.826801 |
| Qwen3-4B L28 | 0.00509109 | 20.9394 | UNBRACKETED | UNBRACKETED | UNBRACKETED | UNBRACKETED | UNBRACKETED |
| Qwen3-4B L35 | 0.00891345 | 20.9954 | UNBRACKETED | UNBRACKETED | UNBRACKETED | UNBRACKETED | UNBRACKETED |

R4 achieved-error curves, mean per-row relative norm and mean bits:

| Model/layer | B | Arm | delta or tau | Error | Error/baseline | Bits |
|---|---|---|---|---|---|---|
| Qwen3-1.7B L11 | 64 | a | 0.5 | 1.33587e-14 | 7.30175e-12 | 36.2499 |
| Qwen3-1.7B L11 | 64 | a | 1 | 1.35614e-14 | 7.52651e-12 | 36.2499 |
| Qwen3-1.7B L11 | 64 | a | 2 | 8.77656e-14 | 4.80859e-11 | 36.2498 |
| Qwen3-1.7B L11 | 64 | a | 4 | 1.05639e-13 | 7.03816e-11 | 36.2496 |
| Qwen3-1.7B L11 | 64 | a | 8 | 7.67788e-13 | 4.33769e-10 | 36.2491 |
| Qwen3-1.7B L11 | 64 | b | 0.5 | 1.04332e-06 | 0.000971078 | 35.0099 |
| Qwen3-1.7B L11 | 64 | b | 1 | 2.50374e-06 | 0.00239647 | 34.5016 |
| Qwen3-1.7B L11 | 64 | b | 2 | 6.04499e-06 | 0.00593424 | 33.8509 |
| Qwen3-1.7B L11 | 64 | b | 4 | 1.41104e-05 | 0.0142526 | 33.0618 |
| Qwen3-1.7B L11 | 64 | b | 8 | 3.2805e-05 | 0.0335505 | 32.1304 |
| Qwen3-1.7B L11 | 64 | c | 0.5 | 1.38588e-05 | 0.0162069 | 28.9756 |
| Qwen3-1.7B L11 | 64 | c | 1 | 3.04408e-05 | 0.0365039 | 27.3168 |
| Qwen3-1.7B L11 | 64 | c | 2 | 6.57622e-05 | 0.0809881 | 25.4077 |
| Qwen3-1.7B L11 | 64 | c | 4 | 0.000141183 | 0.177591 | 23.2996 |
| Qwen3-1.7B L11 | 64 | c | 8 | 0.000298155 | 0.38258 | 21.0407 |
| Qwen3-1.7B L11 | 64 | d | 6 | 0.00843093 | 17.5028 | 20.8671 |
| Qwen3-1.7B L11 | 64 | d | 7 | 0.00515189 | 10.1411 | 21.4084 |
| Qwen3-1.7B L11 | 64 | d | 8.1 | 0.00261258 | 4.99736 | 22.3316 |
| Qwen3-1.7B L11 | 64 | d | 9 | 0.00150473 | 2.75681 | 23.3576 |
| Qwen3-1.7B L11 | 64 | d | 10 | 0.000763031 | 1.33625 | 24.7272 |
| Qwen3-1.7B L11 | 64 | e | 0.5 | 0.00261263 | 4.9974 | 21.1146 |
| Qwen3-1.7B L11 | 64 | e | 1 | 0.00261277 | 4.9975 | 20.6233 |
| Qwen3-1.7B L11 | 64 | e | 2 | 0.00261312 | 4.99773 | 19.9996 |
| Qwen3-1.7B L11 | 64 | e | 4 | 0.00261405 | 4.99839 | 19.2536 |
| Qwen3-1.7B L11 | 64 | e | 8 | 0.00261681 | 5.0003 | 18.3888 |
| Qwen3-1.7B L16 | 64 | a | 0.5 | 0 | 0 | 36.25 |
| Qwen3-1.7B L16 | 64 | a | 1 | 0 | 0 | 36.25 |
| Qwen3-1.7B L16 | 64 | a | 2 | 5.72647e-13 | 3.01731e-10 | 36.2499 |
| Qwen3-1.7B L16 | 64 | a | 4 | 1.2221e-12 | 8.83535e-10 | 36.2496 |
| Qwen3-1.7B L16 | 64 | a | 8 | 1.69819e-11 | 1.38483e-08 | 36.2483 |
| Qwen3-1.7B L16 | 64 | b | 0.5 | 9.08432e-07 | 0.000918273 | 34.8265 |
| Qwen3-1.7B L16 | 64 | b | 1 | 2.24642e-06 | 0.0022983 | 34.24 |
| Qwen3-1.7B L16 | 64 | b | 2 | 5.45658e-06 | 0.00572653 | 33.5084 |
| Qwen3-1.7B L16 | 64 | b | 4 | 1.35178e-05 | 0.0141869 | 32.6174 |
| Qwen3-1.7B L16 | 64 | b | 8 | 3.12842e-05 | 0.0335167 | 31.5446 |
| Qwen3-1.7B L16 | 64 | c | 0.5 | 1.42758e-05 | 0.0185529 | 29.138 |
| Qwen3-1.7B L16 | 64 | c | 1 | 3.25911e-05 | 0.0428339 | 27.1551 |
| Qwen3-1.7B L16 | 64 | c | 2 | 7.30135e-05 | 0.0971692 | 24.9455 |
| Qwen3-1.7B L16 | 64 | c | 4 | 0.000159098 | 0.215519 | 22.5154 |
| Qwen3-1.7B L16 | 64 | c | 8 | 0.000342127 | 0.470222 | 19.9845 |
| Qwen3-1.7B L16 | 64 | d | 6 | 0.0098999 | 19.8126 | 20.6821 |
| Qwen3-1.7B L16 | 64 | d | 7 | 0.00617633 | 12.4631 | 21.0609 |
| Qwen3-1.7B L16 | 64 | d | 8.1 | 0.00341645 | 7.02698 | 21.7388 |
| Qwen3-1.7B L16 | 64 | d | 9 | 0.00196874 | 3.98324 | 22.5445 |
| Qwen3-1.7B L16 | 64 | d | 10 | 0.00100213 | 1.98352 | 23.7297 |
| Qwen3-1.7B L16 | 64 | e | 0.5 | 0.00341644 | 7.02698 | 20.3223 |
| Qwen3-1.7B L16 | 64 | e | 1 | 0.00341642 | 7.02695 | 19.7489 |
| Qwen3-1.7B L16 | 64 | e | 2 | 0.00341637 | 7.0269 | 19.0371 |
| Qwen3-1.7B L16 | 64 | e | 4 | 0.00341653 | 7.027 | 18.1893 |
| Qwen3-1.7B L16 | 64 | e | 8 | 0.00341723 | 7.02741 | 17.2045 |
| Qwen3-1.7B L22 | 64 | a | 0.5 | 0 | 0 | 36.25 |
| Qwen3-1.7B L22 | 64 | a | 1 | 0 | 0 | 36.25 |
| Qwen3-1.7B L22 | 64 | a | 2 | 0 | 0 | 36.25 |
| Qwen3-1.7B L22 | 64 | a | 4 | 0 | 0 | 36.25 |
| Qwen3-1.7B L22 | 64 | a | 8 | 0 | 0 | 36.25 |
| Qwen3-1.7B L22 | 64 | b | 0.5 | 8.87081e-07 | 0.000779469 | 35.3696 |
| Qwen3-1.7B L22 | 64 | b | 1 | 2.29191e-06 | 0.00205818 | 34.8515 |
| Qwen3-1.7B L22 | 64 | b | 2 | 5.65159e-06 | 0.00522702 | 34.1813 |
| Qwen3-1.7B L22 | 64 | b | 4 | 1.37758e-05 | 0.0130251 | 33.3264 |
| Qwen3-1.7B L22 | 64 | b | 8 | 3.26487e-05 | 0.0316532 | 32.3295 |
| Qwen3-1.7B L22 | 64 | c | 0.5 | 1.37312e-05 | 0.0137306 | 32.4085 |
| Qwen3-1.7B L22 | 64 | c | 1 | 3.24376e-05 | 0.0333336 | 30.9448 |
| Qwen3-1.7B L22 | 64 | c | 2 | 7.58419e-05 | 0.0794953 | 29.122 |
| Qwen3-1.7B L22 | 64 | c | 4 | 0.000174724 | 0.185515 | 26.9416 |
| Qwen3-1.7B L22 | 64 | c | 8 | 0.000398506 | 0.427882 | 24.4319 |
| Qwen3-1.7B L22 | 64 | d | 6 | 0.0183682 | 30.102 | 20.4518 |
| Qwen3-1.7B L22 | 64 | d | 7 | 0.0116672 | 19.1441 | 20.6963 |
| Qwen3-1.7B L22 | 64 | d | 8.1 | 0.00664877 | 10.8409 | 21.2379 |
| Qwen3-1.7B L22 | 64 | d | 9 | 0.00398596 | 6.41493 | 22.0025 |
| Qwen3-1.7B L22 | 64 | d | 10 | 0.00207191 | 3.23375 | 23.2735 |
| Qwen3-1.7B L22 | 64 | e | 0.5 | 0.00664879 | 10.8409 | 20.3575 |
| Qwen3-1.7B L22 | 64 | e | 1 | 0.00664878 | 10.8409 | 19.8394 |
| Qwen3-1.7B L22 | 64 | e | 2 | 0.00664877 | 10.8409 | 19.1697 |
| Qwen3-1.7B L22 | 64 | e | 4 | 0.00664877 | 10.8409 | 18.3172 |
| Qwen3-1.7B L22 | 64 | e | 8 | 0.00664909 | 10.841 | 17.3274 |
| Qwen3-1.7B L27 | 64 | a | 0.5 | 0 | 0 | 36.25 |
| Qwen3-1.7B L27 | 64 | a | 1 | 0 | 0 | 36.25 |
| Qwen3-1.7B L27 | 64 | a | 2 | 0 | 0 | 36.25 |
| Qwen3-1.7B L27 | 64 | a | 4 | 0 | 0 | 36.25 |
| Qwen3-1.7B L27 | 64 | a | 8 | 0 | 0 | 36.25 |
| Qwen3-1.7B L27 | 64 | b | 0.5 | 1.21843e-06 | 0.0010327 | 33.9496 |
| Qwen3-1.7B L27 | 64 | b | 1 | 2.88125e-06 | 0.00249784 | 33.1567 |
| Qwen3-1.7B L27 | 64 | b | 2 | 6.66544e-06 | 0.00589114 | 32.2001 |
| Qwen3-1.7B L27 | 64 | b | 4 | 1.4763e-05 | 0.0133141 | 31.0974 |
| Qwen3-1.7B L27 | 64 | b | 8 | 3.24565e-05 | 0.0298773 | 29.882 |
| Qwen3-1.7B L27 | 64 | c | 0.5 | 2.46208e-05 | 0.0220049 | 28.3616 |
| Qwen3-1.7B L27 | 64 | c | 1 | 5.6288e-05 | 0.0511625 | 26.2542 |
| Qwen3-1.7B L27 | 64 | c | 2 | 0.000125682 | 0.115929 | 23.9414 |
| Qwen3-1.7B L27 | 64 | c | 4 | 0.000274371 | 0.256413 | 21.4284 |
| Qwen3-1.7B L27 | 64 | c | 8 | 0.000580111 | 0.548493 | 18.8484 |
| Qwen3-1.7B L27 | 64 | d | 6 | 0.018493 | 29.6473 | 20.3476 |
| Qwen3-1.7B L27 | 64 | d | 7 | 0.0130615 | 20.9932 | 20.4943 |
| Qwen3-1.7B L27 | 64 | d | 8.1 | 0.00829102 | 13.3904 | 20.8476 |
| Qwen3-1.7B L27 | 64 | d | 9 | 0.00533747 | 8.754 | 21.3619 |
| Qwen3-1.7B L27 | 64 | d | 10 | 0.00304175 | 5.11437 | 22.2087 |
| Qwen3-1.7B L27 | 64 | e | 0.5 | 0.00829106 | 13.3904 | 18.5476 |
| Qwen3-1.7B L27 | 64 | e | 1 | 0.00829114 | 13.3905 | 17.7567 |
| Qwen3-1.7B L27 | 64 | e | 2 | 0.00829131 | 13.3906 | 16.8103 |
| Qwen3-1.7B L27 | 64 | e | 4 | 0.00829168 | 13.3909 | 15.734 |
| Qwen3-1.7B L27 | 64 | e | 8 | 0.00829267 | 13.3917 | 14.5784 |
| Qwen3-4B L7 | 64 | a | 0.5 | 0 | 0 | 36.25 |
| Qwen3-4B L7 | 64 | a | 1 | 0 | 0 | 36.25 |
| Qwen3-4B L7 | 64 | a | 2 | 5.31337e-14 | 2.0303e-11 | 36.25 |
| Qwen3-4B L7 | 64 | a | 4 | 1.6096e-12 | 9.30126e-10 | 36.2499 |
| Qwen3-4B L7 | 64 | a | 8 | 1.73868e-11 | 1.41895e-08 | 36.2491 |
| Qwen3-4B L7 | 64 | b | 0.5 | 2.56361e-06 | 0.00183646 | 34.4523 |
| Qwen3-4B L7 | 64 | b | 1 | 6.18048e-06 | 0.00448506 | 33.5867 |
| Qwen3-4B L7 | 64 | b | 2 | 1.4636e-05 | 0.0107159 | 32.5337 |
| Qwen3-4B L7 | 64 | b | 4 | 3.36767e-05 | 0.0249645 | 31.3535 |
| Qwen3-4B L7 | 64 | b | 8 | 7.6457e-05 | 0.0571551 | 30.0662 |
| Qwen3-4B L7 | 64 | c | 0.5 | 2.40374e-05 | 0.0189073 | 28.7386 |
| Qwen3-4B L7 | 64 | c | 1 | 5.44711e-05 | 0.0430345 | 26.4232 |
| Qwen3-4B L7 | 64 | c | 2 | 0.000122819 | 0.0975807 | 23.7987 |
| Qwen3-4B L7 | 64 | c | 4 | 0.000270476 | 0.217057 | 20.9653 |
| Qwen3-4B L7 | 64 | c | 8 | 0.00058112 | 0.471458 | 18.0945 |
| Qwen3-4B L7 | 64 | d | 6 | 0.015956 | 18.9879 | 20.439 |
| Qwen3-4B L7 | 64 | d | 7 | 0.0108284 | 13.1699 | 20.633 |
| Qwen3-4B L7 | 64 | d | 8.1 | 0.00708467 | 8.87284 | 21.0298 |
| Qwen3-4B L7 | 64 | d | 9 | 0.00481099 | 6.14327 | 21.5614 |
| Qwen3-4B L7 | 64 | d | 10 | 0.00287247 | 3.75293 | 22.4352 |
| Qwen3-4B L7 | 64 | e | 0.5 | 0.00708467 | 8.87284 | 19.2344 |
| Qwen3-4B L7 | 64 | e | 1 | 0.00708468 | 8.87285 | 18.3737 |
| Qwen3-4B L7 | 64 | e | 2 | 0.00708482 | 8.87293 | 17.3318 |
| Qwen3-4B L7 | 64 | e | 4 | 0.00708551 | 8.87334 | 16.1758 |
| Qwen3-4B L7 | 64 | e | 8 | 0.0070874 | 8.87456 | 14.939 |
| Qwen3-4B L14 | 64 | a | 0.5 | 0 | 0 | 36.25 |
| Qwen3-4B L14 | 64 | a | 1 | 0 | 0 | 36.25 |
| Qwen3-4B L14 | 64 | a | 2 | 0 | 0 | 36.25 |
| Qwen3-4B L14 | 64 | a | 4 | 5.74097e-15 | 3.34972e-12 | 36.25 |
| Qwen3-4B L14 | 64 | a | 8 | 2.75304e-13 | 1.6177e-10 | 36.2497 |
| Qwen3-4B L14 | 64 | b | 0.5 | 1.04458e-06 | 0.000839571 | 34.1917 |
| Qwen3-4B L14 | 64 | b | 1 | 2.42315e-06 | 0.00200108 | 33.5269 |
| Qwen3-4B L14 | 64 | b | 2 | 5.58272e-06 | 0.00470636 | 32.7362 |
| Qwen3-4B L14 | 64 | b | 4 | 1.27947e-05 | 0.0109549 | 31.8234 |
| Qwen3-4B L14 | 64 | b | 8 | 2.99087e-05 | 0.0259868 | 30.7443 |
| Qwen3-4B L14 | 64 | c | 0.5 | 2.16642e-05 | 0.0215176 | 27.508 |
| Qwen3-4B L14 | 64 | c | 1 | 4.71668e-05 | 0.0480135 | 25.5239 |
| Qwen3-4B L14 | 64 | c | 2 | 0.00010056 | 0.105287 | 23.3389 |
| Qwen3-4B L14 | 64 | c | 4 | 0.000208343 | 0.224304 | 21.0669 |
| Qwen3-4B L14 | 64 | c | 8 | 0.000422505 | 0.471247 | 18.7138 |
| Qwen3-4B L14 | 64 | d | 6 | 0.00681864 | 10.3097 | 20.8038 |
| Qwen3-4B L14 | 64 | d | 7 | 0.0040827 | 5.94155 | 21.273 |
| Qwen3-4B L14 | 64 | d | 8.1 | 0.00227471 | 3.15386 | 22.0906 |
| Qwen3-4B L14 | 64 | d | 9 | 0.00135396 | 1.79743 | 23.0183 |
| Qwen3-4B L14 | 64 | d | 10 | 0.000705863 | 0.903218 | 24.294 |
| Qwen3-4B L14 | 64 | e | 0.5 | 0.00227473 | 3.15387 | 20.0576 |
| Qwen3-4B L14 | 64 | e | 1 | 0.00227477 | 3.15389 | 19.4197 |
| Qwen3-4B L14 | 64 | e | 2 | 0.00227492 | 3.15399 | 18.6807 |
| Qwen3-4B L14 | 64 | e | 4 | 0.00227528 | 3.15419 | 17.8544 |
| Qwen3-4B L14 | 64 | e | 8 | 0.00227625 | 3.15478 | 16.9175 |
| Qwen3-4B L21 | 64 | a | 0.5 | 0 | 0 | 36.25 |
| Qwen3-4B L21 | 64 | a | 1 | 0 | 0 | 36.25 |
| Qwen3-4B L21 | 64 | a | 2 | 1.44402e-14 | 1.73182e-11 | 36.2499 |
| Qwen3-4B L21 | 64 | a | 4 | 4.13649e-13 | 1.96439e-10 | 36.2498 |
| Qwen3-4B L21 | 64 | a | 8 | 1.59757e-12 | 1.37271e-09 | 36.2493 |
| Qwen3-4B L21 | 64 | b | 0.5 | 7.91343e-07 | 0.000738065 | 34.4633 |
| Qwen3-4B L21 | 64 | b | 1 | 1.93197e-06 | 0.0018622 | 33.8438 |
| Qwen3-4B L21 | 64 | b | 2 | 4.50513e-06 | 0.00448391 | 33.0773 |
| Qwen3-4B L21 | 64 | b | 4 | 1.05086e-05 | 0.010771 | 32.1455 |
| Qwen3-4B L21 | 64 | b | 8 | 2.51623e-05 | 0.0262218 | 31.0376 |
| Qwen3-4B L21 | 64 | c | 0.5 | 1.62755e-05 | 0.0192018 | 28.1594 |
| Qwen3-4B L21 | 64 | c | 1 | 3.60419e-05 | 0.0431284 | 26.2547 |
| Qwen3-4B L21 | 64 | c | 2 | 7.83235e-05 | 0.0954176 | 24.1506 |
| Qwen3-4B L21 | 64 | c | 4 | 0.000169392 | 0.209818 | 21.8856 |
| Qwen3-4B L21 | 64 | c | 8 | 0.00036241 | 0.456194 | 19.5034 |
| Qwen3-4B L21 | 64 | d | 6 | 0.00610074 | 9.95795 | 20.8472 |
| Qwen3-4B L21 | 64 | d | 7 | 0.00354991 | 5.64859 | 21.3276 |
| Qwen3-4B L21 | 64 | d | 8.1 | 0.00184056 | 2.79153 | 22.1468 |
| Qwen3-4B L21 | 64 | d | 9 | 0.0010086 | 1.4711 | 23.0675 |
| Qwen3-4B L21 | 64 | d | 10 | 0.00049776 | 0.685412 | 24.3316 |
| Qwen3-4B L21 | 64 | e | 0.5 | 0.00184056 | 2.79153 | 20.4062 |
| Qwen3-4B L21 | 64 | e | 1 | 0.00184054 | 2.79151 | 19.834 |
| Qwen3-4B L21 | 64 | e | 2 | 0.00184056 | 2.79147 | 19.1412 |
| Qwen3-4B L21 | 64 | e | 4 | 0.00184056 | 2.79138 | 18.3256 |
| Qwen3-4B L21 | 64 | e | 8 | 0.00184077 | 2.79125 | 17.3711 |
| Qwen3-4B L28 | 64 | a | 0.5 | 0 | 0 | 36.25 |
| Qwen3-4B L28 | 64 | a | 1 | 0 | 0 | 36.25 |
| Qwen3-4B L28 | 64 | a | 2 | 0 | 0 | 36.25 |
| Qwen3-4B L28 | 64 | a | 4 | 0 | 0 | 36.25 |
| Qwen3-4B L28 | 64 | a | 8 | 0 | 0 | 36.25 |
| Qwen3-4B L28 | 64 | b | 0.5 | 1.94044e-06 | 0.00160386 | 33.5753 |
| Qwen3-4B L28 | 64 | b | 1 | 4.5543e-06 | 0.00384422 | 32.7101 |
| Qwen3-4B L28 | 64 | b | 2 | 1.04314e-05 | 0.00894623 | 31.706 |
| Qwen3-4B L28 | 64 | b | 4 | 2.33268e-05 | 0.020266 | 30.5673 |
| Qwen3-4B L28 | 64 | b | 8 | 5.28933e-05 | 0.0462436 | 29.2826 |
| Qwen3-4B L28 | 64 | c | 0.5 | 2.22633e-05 | 0.0201669 | 28.1179 |
| Qwen3-4B L28 | 64 | c | 1 | 4.93428e-05 | 0.0451654 | 26.0804 |
| Qwen3-4B L28 | 64 | c | 2 | 0.000107172 | 0.0994224 | 23.7893 |
| Qwen3-4B L28 | 64 | c | 4 | 0.00023072 | 0.216895 | 21.3165 |
| Qwen3-4B L28 | 64 | c | 8 | 0.000485391 | 0.466151 | 18.763 |
| Qwen3-4B L28 | 64 | d | 6 | 0.0145866 | 20.3344 | 20.398 |
| Qwen3-4B L28 | 64 | d | 7 | 0.00895208 | 12.5792 | 20.5691 |
| Qwen3-4B L28 | 64 | d | 8.1 | 0.00509109 | 7.13619 | 20.9394 |
| Qwen3-4B L28 | 64 | d | 9 | 0.0030756 | 4.26455 | 21.4453 |
| Qwen3-4B L28 | 64 | d | 10 | 0.00160915 | 2.18556 | 22.2748 |
| Qwen3-4B L28 | 64 | e | 0.5 | 0.00509111 | 7.13621 | 18.2898 |
| Qwen3-4B L28 | 64 | e | 1 | 0.00509115 | 7.13623 | 17.4663 |
| Qwen3-4B L28 | 64 | e | 2 | 0.00509131 | 7.13636 | 16.5389 |
| Qwen3-4B L28 | 64 | e | 4 | 0.00509213 | 7.13694 | 15.5274 |
| Qwen3-4B L28 | 64 | e | 8 | 0.00509496 | 7.13903 | 14.4259 |
| Qwen3-4B L35 | 64 | a | 0.5 | 0 | 0 | 36.25 |
| Qwen3-4B L35 | 64 | a | 1 | 0 | 0 | 36.25 |
| Qwen3-4B L35 | 64 | a | 2 | 0 | 0 | 36.25 |
| Qwen3-4B L35 | 64 | a | 4 | 0 | 0 | 36.25 |
| Qwen3-4B L35 | 64 | a | 8 | 0 | 0 | 36.25 |
| Qwen3-4B L35 | 64 | b | 0.5 | 1.17379e-06 | 0.000921774 | 34.1746 |
| Qwen3-4B L35 | 64 | b | 1 | 2.72101e-06 | 0.0021907 | 33.4219 |
| Qwen3-4B L35 | 64 | b | 2 | 6.24119e-06 | 0.00510904 | 32.493 |
| Qwen3-4B L35 | 64 | b | 4 | 1.36837e-05 | 0.0114553 | 31.4316 |
| Qwen3-4B L35 | 64 | b | 8 | 2.95438e-05 | 0.0252383 | 30.2546 |
| Qwen3-4B L35 | 64 | c | 0.5 | 2.48195e-05 | 0.0210696 | 28.0662 |
| Qwen3-4B L35 | 64 | c | 1 | 5.64649e-05 | 0.0487621 | 26.0903 |
| Qwen3-4B L35 | 64 | c | 2 | 0.000125962 | 0.11 | 23.9229 |
| Qwen3-4B L35 | 64 | c | 4 | 0.000277104 | 0.243983 | 21.5797 |
| Qwen3-4B L35 | 64 | c | 8 | 0.00060788 | 0.54208 | 19.0958 |
| Qwen3-4B L35 | 64 | d | 6 | 0.0212751 | 33.3616 | 20.387 |
| Qwen3-4B L35 | 64 | d | 7 | 0.0146353 | 23.3153 | 20.5671 |
| Qwen3-4B L35 | 64 | d | 8.1 | 0.00891345 | 14.7738 | 20.9954 |
| Qwen3-4B L35 | 64 | d | 9 | 0.00549104 | 9.33232 | 21.6283 |
| Qwen3-4B L35 | 64 | d | 10 | 0.00292579 | 5.1017 | 22.6998 |
| Qwen3-4B L35 | 64 | e | 0.5 | 0.00891347 | 14.7738 | 18.9422 |
| Qwen3-4B L35 | 64 | e | 1 | 0.0089135 | 14.7738 | 18.2064 |
| Qwen3-4B L35 | 64 | e | 2 | 0.00891362 | 14.7739 | 17.3061 |
| Qwen3-4B L35 | 64 | e | 4 | 0.00891388 | 14.7741 | 16.2948 |
| Qwen3-4B L35 | 64 | e | 8 | 0.00891493 | 14.7749 | 15.1961 |

R5 bits law (uncensored blocks only; 95% query-head cluster bootstrap interval, 200 resamples; no independence claim across this single sequence):

| Model/layer | B | mass slope [CI] | gap slope [CI] | uncensored | low=2 | high>=8 | no pass at 8 |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B L11 | 64 | 0.29284 [0.24061, 0.3976] | -0.38882 [-0.4835, -0.3355] | 39727 | 5331 | 86014 | 66297 |
| Qwen3-1.7B L16 | 64 | 0.37703 [0.34496, 0.42839] | -0.3684 [-0.46805, -0.31947] | 45154 | 5028 | 80890 | 58353 |
| Qwen3-1.7B L22 | 64 | 0.37098 [0.34505, 0.40418] | -0.35661 [-0.38404, -0.32227] | 26776 | 920 | 103376 | 77078 |
| Qwen3-1.7B L27 | 64 | 0.29876 [0.25674, 0.35865] | -0.30053 [-0.36721, -0.24689] | 46634 | 11646 | 72792 | 44016 |
| Qwen3-4B L7 | 64 | 0.27252 [0.22896, 0.34157] | -0.29291 [-0.35462, -0.25828] | 103228 | 3854 | 155062 | 99111 |
| Qwen3-4B L14 | 64 | 0.3875 [0.36859, 0.41203] | -0.41257 [-0.44035, -0.38831] | 79264 | 27550 | 155330 | 110628 |
| Qwen3-4B L21 | 64 | 0.38236 [0.36404, 0.40423] | -0.41308 [-0.50843, -0.32985] | 82511 | 20122 | 159511 | 114126 |
| Qwen3-4B L28 | 64 | 0.31544 [0.29641, 0.33843] | -0.35143 [-0.38324, -0.33106] | 81121 | 18396 | 162627 | 112024 |
| Qwen3-4B L35 | 64 | 0.29507 [0.26221, 0.33503] | -0.29116 [-0.34661, -0.25926] | 81259 | 26828 | 154057 | 100492 |

Early layers (excluded from every verdict):

| Model/layer | empirical rho bf16 | CS row median | ORACLE row median | SP bits | hybrid/SP | R5 slope | R2 |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B L0 | 0.876 | 3.4398e+41 | 1.8834e+10 | 26.425 | UNBRACKETED | 0.2577118873009797 | 0 |
| Qwen3-1.7B L5 | 0.98019 | 1.5192e+23 | 1.4355e+06 | 20.917 | 0.6944986557054472 | 0.28517116170661544 | 0 |
| Qwen3-4B L0 | 0.93455 | 7.1763e+31 | 5.7652e+08 | 27.167 | UNBRACKETED | 0.22772394865868875 | 0 |

B=16 sensitivity (same pipeline; not a parameter choice):

| Model/layer | early | empirical rho bf16 | CS row median | R5 mass slope | hybrid/SP | R2 |
|---|---|---|---|---|---|---|
| Qwen3-1.7B L0 | True | 0.9039 | 1.2705e+41 | 0.24970349526947866 | None | 0 |
| Qwen3-1.7B L5 | True | 0.98059 | 9.9138e+21 | 0.2806109076338927 | 0.6275470870965922 | 0 |
| Qwen3-1.7B L11 | False | 0.94867 | 1.1459e+08 | 0.3036119081934073 | None | 0 |
| Qwen3-1.7B L16 | False | 0.97399 | 2.5304e+07 | 0.3626884802181558 | 0.8853308558881127 | 0 |
| Qwen3-1.7B L22 | False | 0.95514 | 1.2992e+08 | 0.37597747799980347 | 0.931082674001142 | 0 |
| Qwen3-1.7B L27 | False | 0.96845 | 2.0362e+10 | 0.3010694427762315 | None | 0 |
| Qwen3-4B L0 | True | 0.94369 | 2.4436e+31 | 0.21768811901918775 | None | 0 |
| Qwen3-4B L7 | False | 0.95999 | 5.2125e+06 | 0.25479564978562125 | None | 0 |
| Qwen3-4B L14 | False | 0.9745 | 1.4037e+08 | 0.3852928792835328 | None | 0 |
| Qwen3-4B L21 | False | 0.97784 | 5.6519e+07 | 0.39428740174332033 | None | 0 |
| Qwen3-4B L28 | False | 0.97126 | 3.6381e+08 | 0.31643893747671276 | None | 0 |
| Qwen3-4B L35 | False | 0.96814 | 2.4281e+10 | 0.2909196001417101 | None | 0 |

G1/G2 receipts: `artifacts/apa_eb1/capture_hashes.json`, `g1/result.json`, `g2/result.json`, `g2/unit.log`; every mutant has its source, unified diff, killing test and log. Per-layer arrays and JSON are beside one another under `artifacts/apa_eb1/<model>_L<layer>_B<B>/`.

Arithmetic and decisions: online mass uses inclusive `logcumsumexp` and prefix maximum Delta. Certified absolute per-block contribution is `wbar*(2*Vmax*Delta + eta)`, via integration of the softmax Jacobian and the value triangle inequality; its sum bounds every intermediate set of score/value changes. Both exact-K and TQ4-K variants are in the menu. The empirical score estimate is exactly `scale*norm(q)*RMS norm(K-Khat)`, a random-direction estimate, not a certificate; no additional sqrt(D) divisor is introduced. R1 uses final bulk mass for its descriptive estimate; causal arms (a),(b),(e) use only the prefix through the decision block. Each row uses its own bf16 P-rounding unit; allocation share is tau*baseline/N_visible_blocks. R5 high censoring includes an explicit sentinel 9 when even 8 bits fails. Predictions L4 and S are evaluated on B=64 only. No amendments to thresholds or grids are made.

The cost ledger is an ideal bits-read model: TQ4 4.25 always; exact K +16; Vbf16 16; Vint8 8.25; Vint4g16 6 (4-bit codes plus the specified 32 bits per group); centroid 16/n_visible_keys. Value reconstructions and centroid means are computed in fp64 for this offline study; physical scale/centroid rounding and metadata traffic beyond the order model are not measured. All visible keys stay in each normalizer. This seat does not authorize E2.

## Prior art

Calver, arXiv 2605.20868 (2026), is the closest work and supplies the score/value decomposition and certified per-block promotion. ThriftAttention, arXiv 2605.23081 (2026), supplies weight-proportional error. BLASST, arXiv 2512.12087 (2025), supplies the running-max criterion; APA-SP1/SP2 changes its action to precision refinement. TurboQuant, Zandieh et al., arXiv 2504.19874 (2025), supplies random rotation and the Lloyd-Max codebook; SPQ3 reconstruction is copied with provenance. MuSe, arXiv 2509.10406 (2025), and Fast Multipole Attention, arXiv 2310.11960 (2023), precede centroid/monopole attention. Huang & Schultheiss (1963) precedes continuous log-weight optimal bit allocation; that optimum does not prove this discrete method menu globally optimal. vAttention, arXiv 2510.05688 (2025), is related statistical certification, not used as our deterministic bound. The house SP2 derivation and GHOST_PRECISION (paper section 3.4) are earlier boundary/precision evidence. Standard components include Cauchy-Schwarz, the softmax Jacobian, Spearman (1904), OLS (Legendre 1805/Gauss 1809), online log-sum-exp (Milakov & Gimelshein 2018), and bootstrap (Efron 1979). Taken: these ingredients. Ours: this measurement implementation and registered evidence; no algorithmic novelty claim. Literature above is lead-supplied, not independently fetched. Additional unverified — lead to check search terms: "softmax Jacobian perturbation bound", "symmetric absmax per-token quantization", "cluster bootstrap regression", "mutation testing DeMillo Lipton Sayward 1978".

A1 integrity/orchestration prior art: SHA-256 (NIST FIPS180-4,2015), ordered old/new digest transitions, mutation testing (DeMillo/Lipton/Sayward,1978); standard ingredients, no novelty claimed. Unverified — lead to check: FIPS180-4 2015 SHA-256; mutation testing DeMillo Lipton Sayward1978. The measurement repair retains the registered bound/centroid prior art at its existing code sites.

## Scope

Evidence class: synthetic unit/mutation tests and finite measurements on captured activations. TQ4, 4.25 bits per element stored, computed in fp64 here; one wikitext-2 sequence, two Qwen3 models, last 64 decode rows, all query heads, causal, offline CPU numerics. No kernels, speed, KV residency, peak device memory, perplexity, generations, prefill, other models/lengths/texts, or early-layer policy are established. RSS is a CPU process safety receipt, not a model memory result. Author tests are baseline evidence; blind verification belongs to the lead. Production, engine source, other worktrees and the immutable order are unchanged. No files were deleted.

## Amendments

No registered parameter, threshold, grid, prediction or verdict rule changes.

A1: Harness defect: vm holds three non-centroid methods; vm[:-1] dropped Vint4g16 and produced three metadata slots for a four-method menu. Restore vm in both write-time and partial-causal loops; add amendment-chain integrity and integration/mutation tests.
Manifest: `artifacts/apa_eb1/amendment_A1.json`; SHA-256 `9871b79376bd450c2e68922f2846ba3f349ebda3d4bba446925c1bfae9fcd672`. Diff: `artifacts/apa_eb1/amendment_A1.diff`.

| File | Old SHA-256 | New SHA-256 |
|---|---|---|
| scripts/apa_eb1_run.py | 50158d90ff35540d84cec62efbadc29a6365c6341cf7fe2239a0a512a9a47a88 | eee8da087d7297d394f11c3317b9ccb98dfca946cbbedfcbef35447028a249c4 |
| scripts/apa_eb1_lib.py | cba18d64c8d08f28459ef699e9cbd388cca11fb24147f889b8b7b1fb95d18263 | 9b503edba4766829396ef3adc47e39ae9c5d45b018c610e15e0bd7f905b4c3e9 |
| scripts/apa_eb1_resume.py | ADDED | 180e62db22c77227c69e62ad679fc074e801c3f2903845054f1c692671916da2 |
| tests/test_apa_eb1_integration.py | ADDED | e40ccda7ebc7cd5ab79e2db9899e9811943e3a5bdd64a724292a49d6acc87fad |
| artifacts/apa_eb1/amendment_A1.diff | ADDED | f5897170979f04fd68e675598149575ef93d5ef659f819a8ac98ac52c2f1488c |
| artifacts/apa_eb1/amendment_A1_sources.diff | ADDED | d79f88bbacf04140984413922f6dbf3adaef68c4cd03651686f0b0a1489a0cf1 |

## Attempt 1

Attempt1 stopped before completing its first cell with `IndexError: index 3 is out of bounds for dimension 2 with size 3`. The lead ruled this a harness defect, not a hypothesis result. Its G3 receipts and report are preserved under `artifacts/apa_eb1/g3_attempt1/`; the log copy is `logs/apa_eb1_seat_g3_attempt1.log`. `attempt1_preservation.json` records the moves and copies. Original G1/G2 receipts remain in place.

A1 rerun evidence: `g2_A1/result.json` (registered 11 tests, all six registered mutants, A1 integration and hash-chain tests); `g2_A1/A1_mutant_result.json` and its diff/command/log (attempt1 vm slice killed at B16 and B64); `g1_A1/result.json` (all six cell numerical records exactly reproduce attempt1); `capture_hashes_A1.json` (all 50 real input hashes reverified).


Receipt validation: `artifacts/apa_eb1/receipt_validation.json` verifies all24 cells, all840 arrays, exact R4 curve means and R2 counts recomputed from stored block/row/mass/allocation arrays. Visible log gaps are finite; masked future-block gaps are +inf and excluded by R5. The first checker incorrectly required all gaps finite; its failure is preserved in `receipt_validation_attempt1.json`. Correcting that audit assumption changed no measurement code/number and reran no cell. Peak CPU process RSS was3467156KiB (under16GiB); no speed result is claimed. No A2/A3 was needed.

Interpretation boundary: B64 non-early empirical ranking median0.969241 met L2; saving criterion was not established within the registered grid. Arm(b) is unbracketed at the SP target on all9 layers; hybrid(e) is unbracketed on6, and its three interpolated ratios are0.930391,0.876278,0.826801, all above0.75. NOT SUPPORTED is the registered finite-sweep outcome, not a claim that no future sweep could save bits.

## Spec ledger

1. met — `artifacts/apa_eb1/attempt1_preservation.json` — Attempt1 receipts moved; report and log copied.
2. met — `artifacts/apa_eb1/amendment_A1.diff` — Exact proposed two-line measurement correction applied.
3. met — `artifacts/apa_eb1/g2_A1/unit.log` — Hash-chain accepts true chain; rejects script/digest/old-hash drift and gaps.
4. met — `artifacts/apa_eb1/amendment_A1.json` — Create-only manifest/digest, disposition, diff and source hashes.
5. met — `artifacts/apa_eb1/g2_A1/result.json` — 24 tests PASS; six registered mutants and A1 mutant killed.
6. met — `artifacts/apa_eb1/g1_A1/result.json` — PASS; attempt1 numbers exactly match.
7. met — `artifacts/apa_eb1/g3_result.json` — PASS all24 cells;840 arrays and per-cell JSON.
8. met — `artifacts/apa_eb1/summary.json` — No further G3 crash; A2/A3 unused.
9. met — `artifacts/apa_eb1/receipt_validation.json` — R2 count0; no numeric or R2 retry.
10. met — `docs/APA_EB1_E1_REPORT.md` — Report regenerated; amendments and attempt1 retained; ledger appended.

Complete experiment inventory: `artifacts/apa_eb1/written_files.txt`. A1 written/preserved file list: `artifacts/apa_eb1/written_files_A1.txt`. Final hashes and integrity checks: `artifacts/apa_eb1/final_integrity.json`. No scratch was deleted; the mutant source remains in the /tmp path recorded by `g2_A1/A1_vm_slice_command.json`.
