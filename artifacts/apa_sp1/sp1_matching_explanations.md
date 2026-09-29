# Every SP1 decode fraction mismatch

this establishes nothing about model quality

Fixed calibration residual plus holdout variation, not a failed monotonicity property. Causal/noncausal decode both see S keys but use independent registered seeds; MHA/GQA also use different seeds and 4/8 query rows. No delta retuning.

| Class | delta | Calibration gap | Holdout SP | Holdout z | Gap | Excess over .02 | Verdict | Receipt |
|---|---|---|---|---|---|---|---|---|
| decode_s2048_d128_c0_h4_kv4 | 2.0 | -0.001385 | 0.119751 | 0.152832 | -0.033081 | 0.013081 | INVALID matched-budget comparison | [decode_s2048_d128_c0_h4_kv4.1788659538446487677.json](gpu/decode_s2048_d128_c0_h4_kv4.1788659538446487677.json) |
| decode_s2048_d128_c1_h4_kv4 | 2.05 | -0.002155 | 0.131348 | 0.152588 | -0.021240 | 0.001240 | INVALID matched-budget comparison | [decode_s2048_d128_c1_h4_kv4.1788659600011674911.json](gpu/decode_s2048_d128_c1_h4_kv4.1788659600011674911.json) |
| decode_s2048_d64_c0_h4_kv4 | 1.95 | -0.004971 | 0.134521 | 0.159058 | -0.024536 | 0.004536 | INVALID matched-budget comparison | [decode_s2048_d64_c0_h4_kv4.1788659415156212037.json](gpu/decode_s2048_d64_c0_h4_kv4.1788659415156212037.json) |
| decode_s2048_d64_c1_h4_kv4 | 2.0 | 0.002434 | 0.182983 | 0.152344 | +0.030640 | 0.010640 | INVALID matched-budget comparison | [decode_s2048_d64_c1_h4_kv4.1788659476719702668.json](gpu/decode_s2048_d64_c1_h4_kv4.1788659476719702668.json) |
| decode_s32768_d128_c0_h8_kv2 | 2.75 | -0.000327 | 0.120422 | 0.154789 | -0.034367 | 0.014367 | INVALID matched-budget comparison | [decode_s32768_d128_c0_h8_kv2.1788660070650132533.json](gpu/decode_s32768_d128_c0_h8_kv2.1788660070650132533.json) |
| decode_s32768_d128_c1_h8_kv2 | 2.75 | -0.001728 | 0.176922 | 0.155441 | +0.021481 | 0.001481 | INVALID matched-budget comparison | [decode_s32768_d128_c1_h8_kv2.1788660133929549361.json](gpu/decode_s32768_d128_c1_h8_kv2.1788660133929549361.json) |
| decode_s32768_d64_c0_h8_kv2 | 2.75 | 0.003649 | 0.220078 | 0.155598 | +0.064480 | 0.044480 | INVALID matched-budget comparison | [decode_s32768_d64_c0_h8_kv2.1788659944842680974.json](gpu/decode_s32768_d64_c0_h8_kv2.1788659944842680974.json) |
| decode_s32768_d64_c1_h8_kv2 | 2.8 | 0.002269 | 0.182213 | 0.154499 | +0.027714 | 0.007714 | INVALID matched-budget comparison | [decode_s32768_d64_c1_h8_kv2.1788660007297424630.json](gpu/decode_s32768_d64_c1_h8_kv2.1788660007297424630.json) |
| decode_s8192_d128_c0_h8_kv2 | 2.45 | 0.000072 | 0.184128 | 0.155090 | +0.029037 | 0.009037 | INVALID matched-budget comparison | [decode_s8192_d128_c0_h8_kv2.1788659820499466113.json](gpu/decode_s8192_d128_c0_h8_kv2.1788659820499466113.json) |
| decode_s8192_d128_c1_h4_kv4 | 2.4 | 0.004924 | 0.110229 | 0.156403 | -0.046173 | 0.026173 | INVALID matched-budget comparison | [decode_s8192_d128_c1_h4_kv4.1788659851454274137.json](gpu/decode_s8192_d128_c1_h4_kv4.1788659851454274137.json) |
| decode_s8192_d128_c1_h8_kv2 | 2.4 | 0.004924 | 0.182449 | 0.155228 | +0.027222 | 0.007222 | INVALID matched-budget comparison | [decode_s8192_d128_c1_h8_kv2.1788659882344236395.json](gpu/decode_s8192_d128_c1_h8_kv2.1788659882344236395.json) |
| decode_s8192_d64_c0_h4_kv4 | 2.35 | -0.001862 | 0.107910 | 0.155731 | -0.047821 | 0.027821 | INVALID matched-budget comparison | [decode_s8192_d64_c0_h4_kv4.1788659661741945323.json](gpu/decode_s8192_d64_c0_h4_kv4.1788659661741945323.json) |
| decode_s8192_d64_c0_h8_kv2 | 2.35 | -0.001862 | 0.173904 | 0.153519 | +0.020386 | 0.000386 | INVALID matched-budget comparison | [decode_s8192_d64_c0_h8_kv2.1788659692563049358.json](gpu/decode_s8192_d64_c0_h8_kv2.1788659692563049358.json) |
| decode_s8192_d64_c1_h8_kv2 | 2.4 | 0.005280 | 0.179871 | 0.155777 | +0.024094 | 0.004094 | INVALID matched-budget comparison | [decode_s8192_d64_c1_h8_kv2.1788659758620586857.json](gpu/decode_s8192_d64_c1_h8_kv2.1788659758620586857.json) |

The JSON supplies all 24 rows, including per-head counts, query norms, prefix maxima and CPU-versus-receipt counts. Every failed row has a small calibration residual and a holdout gap beyond 0.02. Frozen delta is never adjusted. These failures do invalidate matched-budget speed/deviation comparisons.
