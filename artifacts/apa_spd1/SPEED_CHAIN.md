# APA-SPD1 speed chain

G2: RED; 48/50 complete cell receipts.

kernel sweep; this establishes nothing about model quality

Primary engine/APA rows are FP32; fused SDPA rows are BF16. Additional BF16 dense, math, two-pass and SP rows provide same-dtype comparisons. Accuracy always uses the full engine FP32 output from the original shared inputs; BF16 errors include input rounding.

Latency: CUDA-event median and IQR, nine calls after three warmups, rotating interleaving, legacy stream 0. Wall median is separate. Peak: max of three call-local allocation high-water deltas, including output and measured GQA expansion; engine CUDA pool and torch native allocator are separate accounting systems. Resident q/k/v/kq, quantizer setup, diagnostics and transfers are excluded. kq remains floating; this does not measure compressed KV-cache residency.

SP1 frozen δ values are reused on TurboQuant4 keys with r=0.15. E1 extras use symmetric INT4 and r=0.10; their transferred δ values are UNCALIBRATED. Fraction matching is a bounded CPU z-score estimate against the corresponding GPU SP mask sample, not exact CUDA baseline selection.

| Cell (B/H/KV/L/S/D/causal in registration) | Contender | dtype | Status | CUDA median ms | IQR ms | Wall ms | Peak MiB | rel Frobenius | max abs | SP / two-pass fraction estimate |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| prefill_s512_d64_c0_h4_kv4 | engine_dense_fp32 | float32 | OK | 0.154624 | 0.00409599 | 0.165422 | 16.0156 | 0 | 0 | — |
| prefill_s512_d64_c0_h4_kv4 | torch_math_fp32 | float32 | OK | 0.183296 | 0.01616 | 0.194366 | 9.50244 | 2.85951e-07 | 1.56462e-07 | — |
| prefill_s512_d64_c0_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 0.55024 | 0.00204802 | 0.559206 | 0.5 | 0.0523218 | 0.0196712 | — |
| prefill_s512_d64_c0_h4_kv4 | apa_sp_fp32 | float32 | OK | 0.36864 | 0.00307202 | 0.378074 | 0.5 | 0.0490482 | 0.0243643 | 0.15807 / 0.15480; estimate matched |
| prefill_s512_d64_c0_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 0.115712 | 0.01024 | 0.126549 | 8.00781 | 0.00627435 | 0.00470078 | — |
| prefill_s512_d64_c0_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 0.218368 | 0.02272 | 0.228722 | 11.0024 | 0.00351387 | 0.00246242 | — |
| prefill_s512_d64_c0_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 0.550976 | 0.00524801 | 0.560587 | 0.25 | 0.0524646 | 0.0171584 | — |
| prefill_s512_d64_c0_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 0.375808 | 0.00716799 | 0.384856 | 0.25 | 0.0491802 | 0.0253337 | 0.15793 / 0.15479; estimate matched |
| prefill_s512_d64_c0_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 0.065536 | 0.006432 | 0.0746506 | 0.25 | 0.00380908 | 0.00246242 | — |
| prefill_s512_d64_c0_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 0.08704 | 0.01184 | 0.0968524 | 1.27441 | 0.0038156 | 0.00246242 | — |
| prefill_s512_d64_c0_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s512_d64_c0_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s512_d64_c0_h8_kv2 | engine_dense_fp32 | float32 | OK | 0.223936 | 0.00496 | 0.233971 | 32.0312 | 0 | 0 | — |
| prefill_s512_d64_c0_h8_kv2 | torch_math_fp32 | float32 | OK | 0.214016 | 0.025568 | 0.224324 | 21.0044 | 2.83898e-07 | 2.68221e-07 | — |
| prefill_s512_d64_c0_h8_kv2 | apa_two_pass_fp32 | float32 | OK | 1.0791 | 0.036608 | 1.08797 | 1 | 0.051425 | 0.0227535 | — |
| prefill_s512_d64_c0_h8_kv2 | apa_sp_fp32 | float32 | OK | 0.696608 | 0.0256 | 0.706873 | 1 | 0.0481707 | 0.0237413 | 0.14853 / 0.15515; estimate matched |
| prefill_s512_d64_c0_h8_kv2 | engine_dense_bf16 | bfloat16 | OK | 0.183296 | 0.00921601 | 0.194337 | 16.0156 | 0.00613009 | 0.0061824 | — |
| prefill_s512_d64_c0_h8_kv2 | torch_math_bf16 | bfloat16 | OK | 0.238592 | 0.012288 | 0.249962 | 22.5044 | 0.00341224 | 0.00237212 | — |
| prefill_s512_d64_c0_h8_kv2 | apa_two_pass_bf16 | bfloat16 | OK | 1.03619 | 0.03856 | 1.04593 | 0.5 | 0.0516256 | 0.0229283 | — |
| prefill_s512_d64_c0_h8_kv2 | apa_sp_bf16 | bfloat16 | OK | 0.698368 | 0.12288 | 0.707916 | 0.5 | 0.0483457 | 0.0243557 | 0.14864 / 0.15517; estimate matched |
| prefill_s512_d64_c0_h8_kv2 | torch_efficient_bf16 | bfloat16 | OK | 0.116736 | 0.017408 | 0.126319 | 1.5 | 0.00370542 | 0.00237212 | — |
| prefill_s512_d64_c0_h8_kv2 | torch_flash_bf16 | bfloat16 | OK | 0.078848 | 0.008992 | 0.0887779 | 0.516602 | 0.00370928 | 0.00237212 | — |
| prefill_s512_d64_c0_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s512_d64_c0_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s512_d64_c1_h4_kv4 | engine_dense_fp32 | float32 | OK | 0.121856 | 0.001024 | 0.131869 | 8.5 | 0 | 0 | — |
| prefill_s512_d64_c1_h4_kv4 | torch_math_fp32 | float32 | OK | 0.223232 | 0.00307201 | 0.234051 | 10.5024 | 1.88258e-07 | 9.53674e-07 | — |
| prefill_s512_d64_c1_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 0.338752 | 0.00374401 | 0.347345 | 0.5 | 0.053053 | 0.220837 | — |
| prefill_s512_d64_c1_h4_kv4 | apa_sp_fp32 | float32 | OK | 0.260288 | 0.00512001 | 0.269398 | 0.5 | 0.0310552 | 0.058838 | 0.16160 / 0.15565; estimate matched |
| prefill_s512_d64_c1_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 0.081536 | 0.00579201 | 0.0916929 | 4.25 | 0.00429558 | 0.017221 | — |
| prefill_s512_d64_c1_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 0.254976 | 0.004096 | 0.266092 | 12.0024 | 0.00311233 | 0.00902724 | — |
| prefill_s512_d64_c1_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 0.335872 | 0.004832 | 0.34481 | 0.25 | 0.0537271 | 0.22803 | — |
| prefill_s512_d64_c1_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 0.258048 | 0.011264 | 0.267545 | 0.25 | 0.0312412 | 0.0575332 | 0.16139 / 0.15602; estimate matched |
| prefill_s512_d64_c1_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 0.065536 | 0.0072 | 0.074992 | 0.25 | 0.00334932 | 0.0119932 | — |
| prefill_s512_d64_c1_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 0.084992 | 0.009312 | 0.0949493 | 1.27441 | 0.00335573 | 0.0119932 | — |
| prefill_s512_d64_c1_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s512_d64_c1_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s512_d64_c1_h8_kv2 | engine_dense_fp32 | float32 | OK | 0.164864 | 0.00127999 | 0.17456 | 17 | 0 | 0 | — |
| prefill_s512_d64_c1_h8_kv2 | torch_math_fp32 | float32 | OK | 0.259072 | 0.00819197 | 0.269348 | 22.0044 | 1.82938e-07 | 4.47035e-07 | — |
| prefill_s512_d64_c1_h8_kv2 | apa_two_pass_fp32 | float32 | OK | 0.577536 | 0.00608003 | 0.586567 | 1 | 0.0556597 | 0.266007 | — |
| prefill_s512_d64_c1_h8_kv2 | apa_sp_fp32 | float32 | OK | 0.374784 | 0.01536 | 0.385156 | 1 | 0.0312418 | 0.070773 | 0.15176 / 0.15564; estimate matched |
| prefill_s512_d64_c1_h8_kv2 | engine_dense_bf16 | bfloat16 | OK | 0.116736 | 0.004736 | 0.127591 | 8.5 | 0.00444049 | 0.0128777 | — |
| prefill_s512_d64_c1_h8_kv2 | torch_math_bf16 | bfloat16 | OK | 0.282624 | 0.00716799 | 0.293333 | 23.5044 | 0.00311647 | 0.0136833 | — |
| prefill_s512_d64_c1_h8_kv2 | apa_two_pass_bf16 | bfloat16 | OK | 0.57472 | 0.00883198 | 0.584352 | 0.5 | 0.0550003 | 0.24492 | — |
| prefill_s512_d64_c1_h8_kv2 | apa_sp_bf16 | bfloat16 | OK | 0.33152 | 0.016384 | 0.339761 | 0.5 | 0.0314342 | 0.0724021 | 0.15185 / 0.15596; estimate matched |
| prefill_s512_d64_c1_h8_kv2 | torch_efficient_bf16 | bfloat16 | OK | 0.108928 | 0.009536 | 0.118363 | 1.5 | 0.00333921 | 0.0136833 | — |
| prefill_s512_d64_c1_h8_kv2 | torch_flash_bf16 | bfloat16 | OK | 0.0768 | 0.007392 | 0.0876053 | 0.516602 | 0.00334232 | 0.0136833 | — |
| prefill_s512_d64_c1_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s512_d64_c1_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s512_d128_c0_h4_kv4 | engine_dense_fp32 | float32 | OK | 0.162816 | 0.00192 | 0.172796 | 16.0156 | 0 | 0 | — |
| prefill_s512_d128_c0_h4_kv4 | torch_math_fp32 | float32 | OK | 0.181248 | 0.004416 | 0.192093 | 10.0024 | 6.21855e-07 | 5.66244e-07 | — |
| prefill_s512_d128_c0_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 0.665344 | 0.00614405 | 0.674312 | 1 | 0.0523242 | 0.0175466 | — |
| prefill_s512_d128_c0_h4_kv4 | apa_sp_fp32 | float32 | OK | 0.492544 | 0.004096 | 0.501707 | 1 | 0.0467733 | 0.0209311 | 0.16621 / 0.15630; estimate matched |
| prefill_s512_d128_c0_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 0.117504 | 0.00512 | 0.127341 | 8.00781 | 0.00611497 | 0.00675616 | — |
| prefill_s512_d128_c0_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 0.211968 | 0.00307201 | 0.2226 | 13.0024 | 0.00339657 | 0.0019798 | — |
| prefill_s512_d128_c0_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 0.652928 | 0.00588799 | 0.661578 | 0.5 | 0.0523106 | 0.0181893 | — |
| prefill_s512_d128_c0_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 0.498688 | 0.00512001 | 0.508148 | 0.5 | 0.046838 | 0.0214632 | 0.16602 / 0.15631; estimate matched |
| prefill_s512_d128_c0_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 0.067584 | 0.006272 | 0.0773161 | 0.5 | 0.00369932 | 0.00218505 | — |
| prefill_s512_d128_c0_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 0.085376 | 0.011072 | 0.095801 | 2.52441 | 0.00367995 | 0.0019798 | — |
| prefill_s512_d128_c0_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s512_d128_c0_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s512_d128_c0_h8_kv2 | engine_dense_fp32 | float32 | OK | 0.24144 | 0.00716801 | 0.251575 | 32.0312 | 0 | 0 | — |
| prefill_s512_d128_c0_h8_kv2 | torch_math_fp32 | float32 | OK | 0.2304 | 0.004096 | 0.240785 | 24.0044 | 6.33592e-07 | 7.45058e-07 | — |
| prefill_s512_d128_c0_h8_kv2 | apa_two_pass_fp32 | float32 | OK | 1.2585 | 0.00716794 | 1.26812 | 2 | 0.0524824 | 0.0189945 | — |
| prefill_s512_d128_c0_h8_kv2 | apa_sp_fp32 | float32 | OK | 0.915552 | 0.2048 | 0.92714 | 2 | 0.0471404 | 0.0259841 | 0.15243 / 0.15562; estimate matched |
| prefill_s512_d128_c0_h8_kv2 | engine_dense_bf16 | bfloat16 | OK | 0.188416 | 0.002048 | 0.199045 | 16.0156 | 0.00621042 | 0.0059648 | — |
| prefill_s512_d128_c0_h8_kv2 | torch_math_bf16 | bfloat16 | OK | 0.25424 | 0.00614402 | 0.265311 | 27.0044 | 0.00341972 | 0.00300783 | — |
| prefill_s512_d128_c0_h8_kv2 | apa_two_pass_bf16 | bfloat16 | OK | 1.23926 | 0.00819206 | 1.24832 | 1 | 0.0525783 | 0.0199267 | — |
| prefill_s512_d128_c0_h8_kv2 | apa_sp_bf16 | bfloat16 | OK | 0.914528 | 0.226112 | 0.924846 | 1 | 0.0473068 | 0.0242034 | 0.15192 / 0.15572; estimate matched |
| prefill_s512_d128_c0_h8_kv2 | torch_efficient_bf16 | bfloat16 | OK | 0.134144 | 0.00409599 | 0.14306 | 3 | 0.00372014 | 0.00307128 | — |
| prefill_s512_d128_c0_h8_kv2 | torch_flash_bf16 | bfloat16 | OK | 0.099328 | 0.00812799 | 0.109316 | 9.0791 | 0.00368648 | 0.00300783 | — |
| prefill_s512_d128_c0_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s512_d128_c0_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s512_d128_c1_h4_kv4 | engine_dense_fp32 | float32 | OK | 0.132032 | 0.001312 | 0.142429 | 9 | 0 | 0 | — |
| prefill_s512_d128_c1_h4_kv4 | torch_math_fp32 | float32 | OK | 0.22528 | 0.011328 | 0.235966 | 11.0024 | 4.38953e-07 | 1.37091e-06 | — |
| prefill_s512_d128_c1_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 0.38912 | 0.00614399 | 0.397871 | 1 | 0.0546112 | 0.234896 | — |
| prefill_s512_d128_c1_h4_kv4 | apa_sp_fp32 | float32 | OK | 0.344064 | 0.00512001 | 0.353908 | 1 | 0.0315 | 0.06798 | 0.17411 / 0.15654; estimate matched |
| prefill_s512_d128_c1_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 0.086016 | 0.00512 | 0.0956398 | 4.5 | 0.0043458 | 0.0113338 | — |
| prefill_s512_d128_c1_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 0.254976 | 0.00102401 | 0.266092 | 14.0024 | 0.00307926 | 0.00930011 | — |
| prefill_s512_d128_c1_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 0.384032 | 0.00512001 | 0.393493 | 0.5 | 0.0549899 | 0.23747 | — |
| prefill_s512_d128_c1_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 0.346112 | 0.00527999 | 0.355221 | 0.5 | 0.0321625 | 0.147195 | 0.17381 / 0.15620; estimate matched |
| prefill_s512_d128_c1_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 0.068608 | 0.006336 | 0.078077 | 0.5 | 0.00330278 | 0.00835729 | — |
| prefill_s512_d128_c1_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 0.083968 | 0.009056 | 0.0943379 | 2.52441 | 0.00329927 | 0.00835729 | — |
| prefill_s512_d128_c1_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s512_d128_c1_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s512_d128_c1_h8_kv2 | engine_dense_fp32 | float32 | OK | 0.182272 | 0.004384 | 0.191842 | 18 | 0 | 0 | — |
| prefill_s512_d128_c1_h8_kv2 | torch_math_fp32 | float32 | OK | 0.278528 | 0.004096 | 0.289286 | 25.0044 | 4.29086e-07 | 1.49012e-06 | — |
| prefill_s512_d128_c1_h8_kv2 | apa_two_pass_fp32 | float32 | OK | 0.692224 | 0.00511998 | 0.702045 | 2 | 0.052413 | 0.24733 | — |
| prefill_s512_d128_c1_h8_kv2 | apa_sp_fp32 | float32 | OK | 0.52224 | 0.063584 | 0.531563 | 2 | 0.0306757 | 0.0806984 | 0.15896 / 0.15452; estimate matched |
| prefill_s512_d128_c1_h8_kv2 | engine_dense_bf16 | bfloat16 | OK | 0.132096 | 0.004096 | 0.142569 | 9 | 0.00435829 | 0.0162823 | — |
| prefill_s512_d128_c1_h8_kv2 | torch_math_bf16 | bfloat16 | OK | 0.303104 | 0.00432 | 0.313361 | 28.0044 | 0.00305175 | 0.0126295 | — |
| prefill_s512_d128_c1_h8_kv2 | apa_two_pass_bf16 | bfloat16 | OK | 0.68064 | 0.00307202 | 0.68912 | 1 | 0.0527303 | 0.254349 | — |
| prefill_s512_d128_c1_h8_kv2 | apa_sp_bf16 | bfloat16 | OK | 0.515072 | 0.099168 | 0.52458 | 1 | 0.0307341 | 0.0797187 | 0.15850 / 0.15446; estimate matched |
| prefill_s512_d128_c1_h8_kv2 | torch_efficient_bf16 | bfloat16 | OK | 0.114688 | 0.003072 | 0.126098 | 3 | 0.0032775 | 0.0126295 | — |
| prefill_s512_d128_c1_h8_kv2 | torch_flash_bf16 | bfloat16 | OK | 0.08704 | 0.00768 | 0.0979742 | 9.0791 | 0.00327075 | 0.0126295 | — |
| prefill_s512_d128_c1_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s512_d128_c1_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s2048_d64_c0_h4_kv4 | engine_dense_fp32 | float32 | OK | 1.84422 | 0.108896 | 1.96105 | 256.062 | 0 | 0 | — |
| prefill_s2048_d64_c0_h4_kv4 | torch_math_fp32 | float32 | OK | 1.38445 | 0.104448 | 1.39687 | 146.008 | 3.71295e-07 | 1.49012e-07 | — |
| prefill_s2048_d64_c0_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 7.93498 | 0.771904 | 7.94699 | 2 | 0.0492446 | 0.00906741 | — |
| prefill_s2048_d64_c0_h4_kv4 | apa_sp_fp32 | float32 | OK | 5.50605 | 1.04691 | 5.51601 | 2 | 0.0438982 | 0.0116422 | 0.16075 / 0.15442; estimate matched |
| prefill_s2048_d64_c0_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 1.08954 | 0.110688 | 1.10056 | 128.031 | 0.00633487 | 0.00323972 | — |
| prefill_s2048_d64_c0_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 1.61997 | 0.19968 | 1.63126 | 152.008 | 0.00337664 | 0.00288376 | — |
| prefill_s2048_d64_c0_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 7.9104 | 0.744224 | 7.92458 | 1 | 0.0494263 | 0.00920975 | — |
| prefill_s2048_d64_c0_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 4.9255 | 0.611296 | 4.94559 | 1 | 0.0440264 | 0.0122288 | 0.16042 / 0.15447; estimate matched |
| prefill_s2048_d64_c0_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 0.172032 | 0.037888 | 0.182274 | 1 | 0.00368389 | 0.00288376 | — |
| prefill_s2048_d64_c0_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 0.177152 | 0.02752 | 0.187745 | 1.03223 | 0.00368725 | 0.00288376 | — |
| prefill_s2048_d64_c0_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s2048_d64_c0_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s2048_d64_c0_h8_kv2 | engine_dense_fp32 | float32 | OK | 3.44576 | 0.042624 | 8.5902 | 512.125 | 0 | 0 | — |
| prefill_s2048_d64_c0_h8_kv2 | torch_math_fp32 | float32 | OK | 2.62144 | 0.16896 | 2.64153 | 300.016 | 3.74905e-07 | 2.08616e-07 | — |
| prefill_s2048_d64_c0_h8_kv2 | apa_two_pass_fp32 | float32 | OK | 14.2283 | 0.180224 | 14.2388 | 4 | 0.0490836 | 0.00932621 | — |
| prefill_s2048_d64_c0_h8_kv2 | apa_sp_fp32 | float32 | OK | 9.3911 | 1.03117 | 9.40236 | 4 | 0.043872 | 0.0131053 | 0.15639 / 0.15468; estimate matched |
| prefill_s2048_d64_c0_h8_kv2 | engine_dense_bf16 | bfloat16 | OK | 2.32038 | 0.031744 | 2.4107 | 256.062 | 0.00636129 | 0.00441924 | — |
| prefill_s2048_d64_c0_h8_kv2 | torch_math_bf16 | bfloat16 | OK | 3.15085 | 0.13312 | 3.16275 | 306.016 | 0.00337445 | 0.00149783 | — |
| prefill_s2048_d64_c0_h8_kv2 | apa_two_pass_bf16 | bfloat16 | OK | 14.2118 | 0.325088 | 14.2267 | 2 | 0.0490879 | 0.00924117 | — |
| prefill_s2048_d64_c0_h8_kv2 | apa_sp_bf16 | bfloat16 | OK | 8.64288 | 0.894977 | 8.65509 | 2 | 0.0439325 | 0.0126386 | 0.15654 / 0.15470; estimate matched |
| prefill_s2048_d64_c0_h8_kv2 | torch_efficient_bf16 | bfloat16 | OK | 0.283648 | 0.04096 | 0.293584 | 6 | 0.00368617 | 0.00149783 | — |
| prefill_s2048_d64_c0_h8_kv2 | torch_flash_bf16 | bfloat16 | OK | 0.226144 | 0.018816 | 0.23856 | 2.06348 | 0.00368458 | 0.00149783 | — |
| prefill_s2048_d64_c0_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s2048_d64_c0_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s2048_d64_c1_h4_kv4 | engine_dense_fp32 | float32 | OK | 0.730112 | 0.00275201 | 0.739997 | 130 | 0 | 0 | — |
| prefill_s2048_d64_c1_h4_kv4 | torch_math_fp32 | float32 | OK | 1.72954 | 0.016384 | 1.74053 | 162.008 | 2.22666e-07 | 3.57628e-07 | — |
| prefill_s2048_d64_c1_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 4.06528 | 0.017664 | 4.07496 | 2 | 0.0510484 | 0.168054 | — |
| prefill_s2048_d64_c1_h4_kv4 | apa_sp_fp32 | float32 | OK | 2.59072 | 0.397312 | 2.60007 | 2 | 0.0297834 | 0.0427321 | 0.13135 / 0.15516; UNMATCHED/UNCALIBRATED |
| prefill_s2048_d64_c1_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 0.334848 | 0.00716799 | 0.345311 | 65 | 0.00456056 | 0.0144337 | — |
| prefill_s2048_d64_c1_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 1.97734 | 0.00921595 | 1.98993 | 168.008 | 0.00315326 | 0.00746286 | — |
| prefill_s2048_d64_c1_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 4.09805 | 0.0211201 | 4.10703 | 1 | 0.0510117 | 0.15338 | — |
| prefill_s2048_d64_c1_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 2.57536 | 0.37136 | 2.58531 | 1 | 0.0299283 | 0.044565 | 0.13147 / 0.15523; UNMATCHED/UNCALIBRATED |
| prefill_s2048_d64_c1_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 0.145408 | 0.009216 | 0.155083 | 1 | 0.00337797 | 0.0103772 | — |
| prefill_s2048_d64_c1_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 0.147456 | 0.019552 | 0.157076 | 1.03223 | 0.00338321 | 0.0103772 | — |
| prefill_s2048_d64_c1_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s2048_d64_c1_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s2048_d64_c1_h8_kv2 | engine_dense_fp32 | float32 | OK | 1.33222 | 0.0237119 | 1.42073 | 260 | 0 | 0 | — |
| prefill_s2048_d64_c1_h8_kv2 | torch_math_fp32 | float32 | OK | 3.52768 | 0.123936 | 3.54052 | 316.016 | 2.2751e-07 | 4.76837e-07 | — |
| prefill_s2048_d64_c1_h8_kv2 | apa_two_pass_fp32 | float32 | OK | 7.5071 | 0.719872 | 7.51685 | 4 | 0.0517266 | 0.210715 | — |
| prefill_s2048_d64_c1_h8_kv2 | apa_sp_fp32 | float32 | OK | 4.70835 | 0.730112 | 4.71898 | 4 | 0.030956 | 0.0869894 | 0.15055 / 0.15520; estimate matched |
| prefill_s2048_d64_c1_h8_kv2 | engine_dense_bf16 | bfloat16 | OK | 0.74752 | 0.045056 | 0.760004 | 130 | 0.00459137 | 0.0147285 | — |
| prefill_s2048_d64_c1_h8_kv2 | torch_math_bf16 | bfloat16 | OK | 3.92909 | 0.193536 | 3.94151 | 322.016 | 0.00319759 | 0.0134921 | — |
| prefill_s2048_d64_c1_h8_kv2 | apa_two_pass_bf16 | bfloat16 | OK | 7.8377 | 0.722976 | 7.85418 | 2 | 0.0520711 | 0.213507 | — |
| prefill_s2048_d64_c1_h8_kv2 | apa_sp_bf16 | bfloat16 | OK | 4.66534 | 0.128 | 4.67512 | 2 | 0.0311813 | 0.0936587 | 0.15082 / 0.15517; estimate matched |
| prefill_s2048_d64_c1_h8_kv2 | torch_efficient_bf16 | bfloat16 | OK | 0.239328 | 0.03008 | 0.248559 | 6 | 0.00342446 | 0.0134921 | — |
| prefill_s2048_d64_c1_h8_kv2 | torch_flash_bf16 | bfloat16 | OK | 0.178176 | 0.012384 | 0.186943 | 2.06348 | 0.00342929 | 0.0134921 | — |
| prefill_s2048_d64_c1_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s2048_d64_c1_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s2048_d128_c0_h4_kv4 | engine_dense_fp32 | float32 | OK | 1.91488 | 0.018528 | 2.00134 | 256.062 | 0 | 0 | — |
| prefill_s2048_d128_c0_h4_kv4 | torch_math_fp32 | float32 | OK | 1.47882 | 0.019456 | 1.49052 | 148.008 | 7.20875e-07 | 5.06639e-07 | — |
| prefill_s2048_d128_c0_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 9.60922 | 0.157663 | 9.61993 | 4 | 0.0515714 | 0.00982175 | — |
| prefill_s2048_d128_c0_h4_kv4 | apa_sp_fp32 | float32 | OK | 6.48912 | 1.65171 | 6.49888 | 4 | 0.0457695 | 0.0118032 | 0.15593 / 0.15419; estimate matched |
| prefill_s2048_d128_c0_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 1.16019 | 0.0194559 | 1.17273 | 128.031 | 0.00644857 | 0.00480741 | — |
| prefill_s2048_d128_c0_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 1.7224 | 0.0276481 | 1.73319 | 160.008 | 0.00339664 | 0.00172061 | — |
| prefill_s2048_d128_c0_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 9.3143 | 0.198656 | 9.32543 | 2 | 0.0517421 | 0.0105868 | — |
| prefill_s2048_d128_c0_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 6.46253 | 1.71827 | 6.4731 | 2 | 0.0459059 | 0.0134741 | 0.15616 / 0.15408; estimate matched |
| prefill_s2048_d128_c0_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 0.306944 | 0.014432 | 0.316687 | 2 | 0.00372564 | 0.00172061 | — |
| prefill_s2048_d128_c0_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 0.298624 | 0.01024 | 0.30754 | 2.03223 | 0.0037252 | 0.00172061 | — |
| prefill_s2048_d128_c0_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s2048_d128_c0_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s2048_d128_c0_h8_kv2 | engine_dense_fp32 | float32 | OK | 3.70467 | 0.085824 | 9.0744 | 512.125 | 0 | 0 | — |
| prefill_s2048_d128_c0_h8_kv2 | torch_math_fp32 | float32 | OK | 2.7945 | 0.11808 | 2.80709 | 312.016 | 6.95803e-07 | 3.35276e-07 | — |
| prefill_s2048_d128_c0_h8_kv2 | apa_two_pass_fp32 | float32 | OK | 18.3673 | 0.164801 | 18.381 | 8 | 0.0501691 | 0.00957212 | — |
| prefill_s2048_d128_c0_h8_kv2 | apa_sp_fp32 | float32 | OK | 12.6208 | 1.2544 | 12.6322 | 8 | 0.0445065 | 0.0139221 | 0.15608 / 0.15486; estimate matched |
| prefill_s2048_d128_c0_h8_kv2 | engine_dense_bf16 | bfloat16 | OK | 2.50368 | 0.11776 | 2.57923 | 256.062 | 0.00629224 | 0.00309502 | — |
| prefill_s2048_d128_c0_h8_kv2 | torch_math_bf16 | bfloat16 | OK | 3.29027 | 0.0245759 | 3.30183 | 324.016 | 0.00333105 | 0.0017654 | — |
| prefill_s2048_d128_c0_h8_kv2 | apa_two_pass_bf16 | bfloat16 | OK | 18.0897 | 0.642273 | 18.0994 | 4 | 0.0502889 | 0.0101829 | — |
| prefill_s2048_d128_c0_h8_kv2 | apa_sp_bf16 | bfloat16 | OK | 12.7724 | 1.52678 | 12.8023 | 4 | 0.0446363 | 0.0144963 | 0.15584 / 0.15496; estimate matched |
| prefill_s2048_d128_c0_h8_kv2 | torch_efficient_bf16 | bfloat16 | OK | 0.536576 | 0.019456 | 0.546591 | 12 | 0.00364627 | 0.00142661 | — |
| prefill_s2048_d128_c0_h8_kv2 | torch_flash_bf16 | bfloat16 | OK | 0.411648 | 0.009152 | 0.422036 | 4.06348 | 0.00364329 | 0.0017654 | — |
| prefill_s2048_d128_c0_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s2048_d128_c0_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s2048_d128_c1_h4_kv4 | engine_dense_fp32 | float32 | OK | 0.813056 | 0.00512004 | 0.823153 | 132 | 0 | 0 | — |
| prefill_s2048_d128_c1_h4_kv4 | torch_math_fp32 | float32 | OK | 1.79712 | 0.0256001 | 1.81215 | 164.008 | 4.995e-07 | 1.72853e-06 | — |
| prefill_s2048_d128_c1_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 5.10941 | 0.134368 | 5.11837 | 4 | 0.0511767 | 0.182609 | — |
| prefill_s2048_d128_c1_h4_kv4 | apa_sp_fp32 | float32 | OK | 3.53075 | 0.591872 | 3.54039 | 4 | 0.0290833 | 0.0479915 | 0.15746 / 0.15543; estimate matched |
| prefill_s2048_d128_c1_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 0.390144 | 0.00307199 | 0.399845 | 66 | 0.0045577 | 0.0120301 | — |
| prefill_s2048_d128_c1_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 2.06848 | 0.011456 | 2.07953 | 176.008 | 0.00311891 | 0.0118616 | — |
| prefill_s2048_d128_c1_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 4.8128 | 0.0256 | 4.82195 | 2 | 0.0512125 | 0.156507 | — |
| prefill_s2048_d128_c1_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 3.54406 | 0.543744 | 3.55451 | 2 | 0.0293068 | 0.0511846 | 0.15726 / 0.15561; estimate matched |
| prefill_s2048_d128_c1_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 0.208896 | 0.011616 | 0.21731 | 2 | 0.00335711 | 0.0118616 | — |
| prefill_s2048_d128_c1_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 0.178176 | 0.018656 | 0.188375 | 2.03223 | 0.00334968 | 0.0118616 | — |
| prefill_s2048_d128_c1_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s2048_d128_c1_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s2048_d128_c1_h8_kv2 | engine_dense_fp32 | float32 | OK | 1.48954 | 0.021504 | 1.56795 | 264 | 0 | 0 | — |
| prefill_s2048_d128_c1_h8_kv2 | torch_math_fp32 | float32 | OK | 3.712 | 0.0431039 | 3.72637 | 328.016 | 4.95356e-07 | 1.43051e-06 | — |
| prefill_s2048_d128_c1_h8_kv2 | apa_two_pass_fp32 | float32 | OK | 9.56202 | 0.0471039 | 9.57104 | 8 | 0.0522474 | 0.189787 | — |
| prefill_s2048_d128_c1_h8_kv2 | apa_sp_fp32 | float32 | OK | 6.29555 | 0.538624 | 6.30599 | 8 | 0.0293188 | 0.0562125 | 0.15421 / 0.15565; estimate matched |
| prefill_s2048_d128_c1_h8_kv2 | engine_dense_bf16 | bfloat16 | OK | 0.816128 | 0.021312 | 0.825958 | 132 | 0.00458919 | 0.0135612 | — |
| prefill_s2048_d128_c1_h8_kv2 | torch_math_bf16 | bfloat16 | OK | 4.14925 | 0.0139523 | 4.16014 | 340.016 | 0.00313382 | 0.011642 | — |
| prefill_s2048_d128_c1_h8_kv2 | apa_two_pass_bf16 | bfloat16 | OK | 8.92314 | 0.208896 | 8.93309 | 4 | 0.0523897 | 0.196457 | — |
| prefill_s2048_d128_c1_h8_kv2 | apa_sp_bf16 | bfloat16 | OK | 6.20954 | 0.53248 | 6.22063 | 4 | 0.0295106 | 0.0597123 | 0.15438 / 0.15568; estimate matched |
| prefill_s2048_d128_c1_h8_kv2 | torch_efficient_bf16 | bfloat16 | OK | 0.342016 | 0.031936 | 0.351283 | 12 | 0.00337633 | 0.011642 | — |
| prefill_s2048_d128_c1_h8_kv2 | torch_flash_bf16 | bfloat16 | OK | 0.241312 | 0.011264 | 0.250613 | 4.06348 | 0.00336955 | 0.011642 | — |
| prefill_s2048_d128_c1_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s2048_d128_c1_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s8192_d64_c0_h4_kv4 | engine_dense_fp32 | float32 | OK | 26.8973 | 0.222208 | 102.375 | 4096.25 | 0 | 0 | — |
| prefill_s8192_d64_c0_h4_kv4 | torch_math_fp32 | float32 | OK | 19.281 | 0.0348167 | 19.3005 | 2312.03 | 5.1662e-07 | 8.56817e-08 | — |
| prefill_s8192_d64_c0_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 116.376 | 0.842979 | 116.421 | 8 | 0.0503426 | 0.00484874 | — |
| prefill_s8192_d64_c0_h4_kv4 | apa_sp_fp32 | float32 | OK | 72.5197 | 5.10361 | 72.8646 | 8 | 0.0447621 | 0.00726085 | 0.15249 / 0.15517; estimate matched |
| prefill_s8192_d64_c0_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 17.6579 | 0.247808 | 52.7707 | 2048.12 | 0.00677164 | 0.00127995 | — |
| prefill_s8192_d64_c0_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 22.9601 | 0.0327682 | 22.9766 | 2336.03 | 0.00344083 | 0.000595011 | — |
| prefill_s8192_d64_c0_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 115.203 | 0.693855 | 115.221 | 4 | 0.050469 | 0.00474331 | — |
| prefill_s8192_d64_c0_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 68.6152 | 4.51571 | 68.6304 | 4 | 0.0449173 | 0.00930477 | 0.15223 / 0.15511; estimate matched |
| prefill_s8192_d64_c0_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 1.41005 | 0.05632 | 1.42042 | 4 | 0.00378396 | 0.000595011 | — |
| prefill_s8192_d64_c0_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 1.14483 | 0.067584 | 1.1547 | 4.12598 | 0.00378548 | 0.000652216 | — |
| prefill_s8192_d64_c0_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s8192_d64_c0_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s8192_d64_c0_h8_kv2 | engine_dense_fp32 | float32 | BLOCKED_AFTER_ERROR | — | — | — | — | 0 | 0 | — |
| prefill_s8192_d64_c0_h8_kv2 | torch_math_fp32 | float32 | BLOCKED_AFTER_ERROR | — | — | — | — | 5.18618e-07 | 1.04308e-07 | — |
| prefill_s8192_d64_c0_h8_kv2 | apa_two_pass_fp32 | float32 | BLOCKED_AFTER_ERROR | — | — | — | — | 0.0508021 | 0.0048778 | — |
| prefill_s8192_d64_c0_h8_kv2 | apa_sp_fp32 | float32 | BLOCKED_AFTER_ERROR | — | — | — | — | 0.045335 | 0.00812977 | 0.16430 / 0.15482; estimate matched |
| prefill_s8192_d64_c0_h8_kv2 | engine_dense_bf16 | bfloat16 | BLOCKED_AFTER_ERROR | — | — | — | — | 0.00683168 | 0.00176698 | — |
| prefill_s8192_d64_c0_h8_kv2 | torch_math_bf16 | bfloat16 | BLOCKED_AFTER_ERROR | — | — | — | — | 0.0034588 | 0.00101078 | — |
| prefill_s8192_d64_c0_h8_kv2 | apa_two_pass_bf16 | bfloat16 | BLOCKED_AFTER_ERROR | — | — | — | — | 0.050897 | 0.00506526 | — |
| prefill_s8192_d64_c0_h8_kv2 | apa_sp_bf16 | bfloat16 | BLOCKED_AFTER_ERROR | — | — | — | — | 0.0454533 | 0.00797497 | 0.16448 / 0.15482; estimate matched |
| prefill_s8192_d64_c0_h8_kv2 | torch_efficient_bf16 | bfloat16 | BLOCKED_AFTER_ERROR | — | — | — | — | 0.00380653 | 0.00101078 | — |
| prefill_s8192_d64_c0_h8_kv2 | torch_flash_bf16 | bfloat16 | BLOCKED_AFTER_ERROR | — | — | — | — | 0.00380723 | 0.00101078 | — |
| prefill_s8192_d64_c0_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s8192_d64_c0_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s8192_d64_c1_h4_kv4 | engine_dense_fp32 | float32 | OK | 10.8503 | 0.732161 | 46.6608 | 2056 | 0 | 0 | — |
| prefill_s8192_d64_c1_h4_kv4 | torch_math_fp32 | float32 | OK | 27.989 | 0.366592 | 28.0632 | 2568.03 | 2.80086e-07 | 3.57628e-07 | — |
| prefill_s8192_d64_c1_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 56.8994 | 1.14176 | 56.9298 | 8 | 0.0514581 | 0.17386 | — |
| prefill_s8192_d64_c1_h4_kv4 | apa_sp_fp32 | float32 | OK | 34.2354 | 1.86675 | 34.276 | 8 | 0.026493 | 0.032003 | 0.15159 / 0.15474; estimate matched |
| prefill_s8192_d64_c1_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 5.32787 | 0.130048 | 20.7994 | 1028 | 0.00483492 | 0.0108098 | — |
| prefill_s8192_d64_c1_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 32.4222 | 1.80054 | 32.5186 | 2592.03 | 0.00327184 | 0.0121114 | — |
| prefill_s8192_d64_c1_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 57.6543 | 1.46329 | 57.7065 | 4 | 0.0516759 | 0.172932 | — |
| prefill_s8192_d64_c1_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 34.0784 | 1.44573 | 34.1094 | 4 | 0.0267068 | 0.0328091 | 0.15170 / 0.15468; estimate matched |
| prefill_s8192_d64_c1_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 0.900096 | 0.206848 | 0.915677 | 4 | 0.00352521 | 0.0121114 | — |
| prefill_s8192_d64_c1_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 0.792576 | 0.029696 | 0.852318 | 4.12598 | 0.00353302 | 0.0121114 | — |
| prefill_s8192_d64_c1_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s8192_d64_c1_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s8192_d64_c1_h8_kv2 | engine_dense_fp32 | float32 | OK | 22.5472 | 1.32576 | 98.4738 | 4112 | 0 | 0 | — |
| prefill_s8192_d64_c1_h8_kv2 | torch_math_fp32 | float32 | OK | 56.1082 | 0.838654 | 56.1585 | 4912.06 | 2.76857e-07 | 3.72529e-07 | — |
| prefill_s8192_d64_c1_h8_kv2 | apa_two_pass_fp32 | float32 | OK | 115.778 | 1.75616 | 115.85 | 16 | 0.050778 | 0.188536 | — |
| prefill_s8192_d64_c1_h8_kv2 | apa_sp_fp32 | float32 | OK | 65.4797 | 2.432 | 65.513 | 16 | 0.0265135 | 0.0336594 | 0.15813 / 0.15456; estimate matched |
| prefill_s8192_d64_c1_h8_kv2 | engine_dense_bf16 | bfloat16 | OK | 10.7377 | 0.596992 | 46.3202 | 2056 | 0.0047725 | 0.0128237 | — |
| prefill_s8192_d64_c1_h8_kv2 | torch_math_bf16 | bfloat16 | OK | 63.8341 | 0.849251 | 63.9447 | 4936.06 | 0.00321975 | 0.0113671 | — |
| prefill_s8192_d64_c1_h8_kv2 | apa_two_pass_bf16 | bfloat16 | OK | 116.006 | 2.92429 | 116.372 | 8 | 0.0508543 | 0.186256 | — |
| prefill_s8192_d64_c1_h8_kv2 | apa_sp_bf16 | bfloat16 | OK | 65.9855 | 3.13139 | 66.1317 | 8 | 0.0267284 | 0.0325051 | 0.15826 / 0.15459; estimate matched |
| prefill_s8192_d64_c1_h8_kv2 | torch_efficient_bf16 | bfloat16 | OK | 1.62099 | 0.264192 | 1.6387 | 24 | 0.00346594 | 0.0113671 | — |
| prefill_s8192_d64_c1_h8_kv2 | torch_flash_bf16 | bfloat16 | OK | 1.27693 | 0.016384 | 1.31141 | 8.25098 | 0.00346835 | 0.0113671 | — |
| prefill_s8192_d64_c1_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s8192_d64_c1_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s8192_d128_c0_h4_kv4 | engine_dense_fp32 | float32 | OK | 29.6182 | 0.994303 | 104.839 | 4096.25 | 0 | 0 | — |
| prefill_s8192_d128_c0_h4_kv4 | torch_math_fp32 | float32 | OK | 21.9433 | 0.609282 | 21.9737 | 2320.03 | 8.84927e-07 | 1.86265e-07 | — |
| prefill_s8192_d128_c0_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 167.011 | 3.28116 | 167.134 | 16 | 0.0514416 | 0.00481728 | — |
| prefill_s8192_d128_c0_h4_kv4 | apa_sp_fp32 | float32 | OK | 105.984 | 6.19827 | 106.068 | 16 | 0.0436264 | 0.00660531 | 0.16643 / 0.15459; estimate matched |
| prefill_s8192_d128_c0_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 19.071 | 0.953344 | 54.6952 | 2048.12 | 0.00676531 | 0.00145043 | — |
| prefill_s8192_d128_c0_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 24.9477 | 0.938688 | 24.9826 | 2368.03 | 0.00339285 | 0.000546798 | — |
| prefill_s8192_d128_c0_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 149.944 | 2.36429 | 150.079 | 8 | 0.0515468 | 0.00494676 | — |
| prefill_s8192_d128_c0_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 102.099 | 5.08419 | 102.161 | 8 | 0.0437611 | 0.00655133 | 0.16659 / 0.15455; estimate matched |
| prefill_s8192_d128_c0_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 3.28605 | 0.2048 | 3.30788 | 8 | 0.00374934 | 0.000546798 | — |
| prefill_s8192_d128_c0_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 2.16576 | 0.07376 | 2.30438 | 8.12598 | 0.00374555 | 0.000546798 | — |
| prefill_s8192_d128_c0_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s8192_d128_c0_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s8192_d128_c0_h8_kv2 | engine_dense_fp32 | float32 | BLOCKED_AFTER_ERROR | — | — | — | — | 0 | 0 | — |
| prefill_s8192_d128_c0_h8_kv2 | torch_math_fp32 | float32 | BLOCKED_AFTER_ERROR | — | — | — | — | 9.02916e-07 | 2.90573e-07 | — |
| prefill_s8192_d128_c0_h8_kv2 | apa_two_pass_fp32 | float32 | BLOCKED_AFTER_ERROR | — | — | — | — | 0.0529828 | 0.00505839 | — |
| prefill_s8192_d128_c0_h8_kv2 | apa_sp_fp32 | float32 | BLOCKED_AFTER_ERROR | — | — | — | — | 0.0449188 | 0.00644073 | 0.16398 / 0.15502; estimate matched |
| prefill_s8192_d128_c0_h8_kv2 | engine_dense_bf16 | bfloat16 | BLOCKED_AFTER_ERROR | — | — | — | — | 0.00692245 | 0.00157999 | — |
| prefill_s8192_d128_c0_h8_kv2 | torch_math_bf16 | bfloat16 | BLOCKED_AFTER_ERROR | — | — | — | — | 0.00345551 | 0.000681624 | — |
| prefill_s8192_d128_c0_h8_kv2 | apa_two_pass_bf16 | bfloat16 | BLOCKED_AFTER_ERROR | — | — | — | — | 0.053086 | 0.00511806 | — |
| prefill_s8192_d128_c0_h8_kv2 | apa_sp_bf16 | bfloat16 | BLOCKED_AFTER_ERROR | — | — | — | — | 0.0450494 | 0.0063263 | 0.16410 / 0.15498; estimate matched |
| prefill_s8192_d128_c0_h8_kv2 | torch_efficient_bf16 | bfloat16 | BLOCKED_AFTER_ERROR | — | — | — | — | 0.00382262 | 0.000681624 | — |
| prefill_s8192_d128_c0_h8_kv2 | torch_flash_bf16 | bfloat16 | BLOCKED_AFTER_ERROR | — | — | — | — | 0.00382341 | 0.00079504 | — |
| prefill_s8192_d128_c0_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s8192_d128_c0_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s8192_d128_c1_h4_kv4 | engine_dense_fp32 | float32 | OK | 12.426 | 0.999168 | 48.3872 | 2064 | 0 | 0 | — |
| prefill_s8192_d128_c1_h4_kv4 | torch_math_fp32 | float32 | OK | 29.7864 | 1.26361 | 29.7999 | 2576.03 | 5.6111e-07 | 1.13249e-06 | — |
| prefill_s8192_d128_c1_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 81.3855 | 2.36563 | 81.4238 | 16 | 0.052089 | 0.222575 | — |
| prefill_s8192_d128_c1_h4_kv4 | apa_sp_fp32 | float32 | OK | 49.6674 | 4.40358 | 49.7207 | 16 | 0.0263128 | 0.0431983 | 0.16031 / 0.15508; estimate matched |
| prefill_s8192_d128_c1_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 5.8409 | 0.114688 | 21.7548 | 1032 | 0.00463578 | 0.0145338 | — |
| prefill_s8192_d128_c1_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 33.4817 | 0.251904 | 33.573 | 2624.03 | 0.00313915 | 0.0113358 | — |
| prefill_s8192_d128_c1_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 71.6841 | 2.65626 | 71.722 | 8 | 0.0521908 | 0.221283 | — |
| prefill_s8192_d128_c1_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 47.0346 | 1.83504 | 47.0485 | 8 | 0.0264802 | 0.0396453 | 0.16134 / 0.15509; estimate matched |
| prefill_s8192_d128_c1_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 1.81539 | 0.167744 | 1.82529 | 8 | 0.00340807 | 0.0113358 | — |
| prefill_s8192_d128_c1_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 1.24138 | 0.048416 | 1.27832 | 8.12598 | 0.00339932 | 0.0113358 | — |
| prefill_s8192_d128_c1_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s8192_d128_c1_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s8192_d128_c1_h8_kv2 | engine_dense_fp32 | float32 | OK | 25.0972 | 1.50221 | 102.76 | 4128 | 0 | 0 | — |
| prefill_s8192_d128_c1_h8_kv2 | torch_math_fp32 | float32 | OK | 59.7115 | 1.44794 | 60.255 | 4960.06 | 5.55823e-07 | 1.90735e-06 | — |
| prefill_s8192_d128_c1_h8_kv2 | apa_two_pass_fp32 | float32 | OK | 170.003 | 1.1395 | 170.139 | 32 | 0.0512581 | 0.29553 | — |
| prefill_s8192_d128_c1_h8_kv2 | apa_sp_fp32 | float32 | OK | 95.3651 | 7.47312 | 95.4463 | 32 | 0.0262117 | 0.0358459 | 0.15493 / 0.15498; estimate matched |
| prefill_s8192_d128_c1_h8_kv2 | engine_dense_bf16 | bfloat16 | OK | 11.4801 | 1.11821 | 46.9703 | 2064 | 0.00467196 | 0.0176528 | — |
| prefill_s8192_d128_c1_h8_kv2 | torch_math_bf16 | bfloat16 | OK | 66.5139 | 1.78688 | 66.5438 | 5008.06 | 0.00316445 | 0.0117559 | — |
| prefill_s8192_d128_c1_h8_kv2 | apa_two_pass_bf16 | bfloat16 | OK | 149.094 | 2.53337 | 149.17 | 16 | 0.0514153 | 0.301593 | — |
| prefill_s8192_d128_c1_h8_kv2 | apa_sp_bf16 | bfloat16 | OK | 93.7216 | 3.78573 | 93.7781 | 16 | 0.0264109 | 0.0403178 | 0.15528 / 0.15501; estimate matched |
| prefill_s8192_d128_c1_h8_kv2 | torch_efficient_bf16 | bfloat16 | OK | 3.31366 | 0.217184 | 3.33288 | 48 | 0.00342071 | 0.0128716 | — |
| prefill_s8192_d128_c1_h8_kv2 | torch_flash_bf16 | bfloat16 | OK | 2.20774 | 0.029696 | 2.28466 | 16.251 | 0.00341507 | 0.0128716 | — |
| prefill_s8192_d128_c1_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| prefill_s8192_d128_c1_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s2048_d64_c0_h4_kv4 | engine_dense_fp32 | float32 | OK | 0.072704 | 0.006144 | 0.0827969 | 0.125031 | 0 | 0 | — |
| decode_s2048_d64_c0_h4_kv4 | torch_math_fp32 | float32 | OK | 0.156032 | 0.006368 | 0.166183 | 2.03223 | 2.57241e-07 | 3.72529e-08 | — |
| decode_s2048_d64_c0_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 0.221184 | 0.00611199 | 0.229483 | 0.000976562 | 0.0530479 | 0.00579897 | — |
| decode_s2048_d64_c0_h4_kv4 | apa_sp_fp32 | float32 | OK | 0.16384 | 0.002048 | 0.172896 | 0.00198364 | 0.0405127 | 0.00449464 | 0.16992 / 0.15552; estimate matched |
| decode_s2048_d64_c0_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 0.07168 | 0.005984 | 0.082586 | 0.0625153 | 0.00678998 | 0.00100416 | — |
| decode_s2048_d64_c0_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 0.192512 | 0.00736 | 0.202372 | 6.0332 | 0.00352539 | 0.00043115 | — |
| decode_s2048_d64_c0_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 0.232416 | 0.00307199 | 0.240784 | 0.000488281 | 0.0542215 | 0.00638697 | — |
| decode_s2048_d64_c0_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 0.161792 | 0.002048 | 0.171664 | 0.00149536 | 0.0410471 | 0.00470764 | 0.16956 / 0.15613; estimate matched |
| decode_s2048_d64_c0_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 0.115712 | 0.01024 | 0.125407 | 0.000488281 | 0.00398966 | 0.00043115 | — |
| decode_s2048_d64_c0_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 0.080896 | 0.006144 | 0.0905404 | 0.0102539 | 0.00373496 | 0.000374228 | — |
| decode_s2048_d64_c0_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s2048_d64_c0_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s2048_d64_c0_h8_kv2 | engine_dense_fp32 | float32 | OK | 0.078816 | 0.00192001 | 0.088918 | 0.250061 | 0 | 0 | — |
| decode_s2048_d64_c0_h8_kv2 | torch_math_fp32 | float32 | OK | 0.175104 | 0.00428799 | 0.18527 | 12.0645 | 3.22922e-07 | 3.72529e-08 | — |
| decode_s2048_d64_c0_h8_kv2 | apa_two_pass_fp32 | float32 | OK | 0.221184 | 0.003392 | 0.230796 | 0.00195312 | 0.0503253 | 0.00619993 | — |
| decode_s2048_d64_c0_h8_kv2 | apa_sp_fp32 | float32 | OK | 0.160768 | 0.00102401 | 0.169941 | 0.00396729 | 0.0451953 | 0.00687225 | 0.17517 / 0.15759; estimate matched |
| decode_s2048_d64_c0_h8_kv2 | engine_dense_bf16 | bfloat16 | OK | 0.074752 | 0.007968 | 0.0854307 | 0.125031 | 0.00615397 | 0.000874251 | — |
| decode_s2048_d64_c0_h8_kv2 | torch_math_bf16 | bfloat16 | OK | 0.210976 | 0.00908801 | 0.221237 | 14.0664 | 0.00330008 | 0.000356667 | — |
| decode_s2048_d64_c0_h8_kv2 | apa_two_pass_bf16 | bfloat16 | OK | 0.233472 | 0.00275199 | 0.241797 | 0.000976562 | 0.0501014 | 0.00608423 | — |
| decode_s2048_d64_c0_h8_kv2 | apa_sp_bf16 | bfloat16 | OK | 0.159744 | 0.00307199 | 0.16989 | 0.00299072 | 0.0448887 | 0.00658999 | 0.17499 / 0.15753; estimate matched |
| decode_s2048_d64_c0_h8_kv2 | torch_efficient_bf16 | bfloat16 | OK | 0.15328 | 0.01248 | 0.162476 | 4.00098 | 0.00364082 | 0.000458479 | — |
| decode_s2048_d64_c0_h8_kv2 | torch_flash_bf16 | bfloat16 | OK | 0.084992 | 0.006688 | 0.0953698 | 0.0185547 | 0.00371133 | 0.000458479 | — |
| decode_s2048_d64_c0_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s2048_d64_c0_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s2048_d64_c1_h4_kv4 | engine_dense_fp32 | float32 | OK | 0.048128 | 0.006144 | 0.0575082 | 0.0634766 | 0 | 0 | — |
| decode_s2048_d64_c1_h4_kv4 | torch_math_fp32 | float32 | OK | 0.149504 | 0.002048 | 0.159281 | 2.03223 | 2.30293e-07 | 3.72529e-08 | — |
| decode_s2048_d64_c1_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 0.221184 | 0.002368 | 0.229373 | 0.000976562 | 0.0415796 | 0.00461177 | — |
| decode_s2048_d64_c1_h4_kv4 | apa_sp_fp32 | float32 | OK | 0.160768 | 0.00102399 | 0.170432 | 0.00198364 | 0.0402668 | 0.00522605 | 0.16345 / 0.15588; estimate matched |
| decode_s2048_d64_c1_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 0.049152 | 0.003776 | 0.0600731 | 0.0317383 | 0.00516987 | 0.000790372 | — |
| decode_s2048_d64_c1_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 0.185344 | 0.00511999 | 0.197182 | 6.0332 | 0.00294556 | 0.000366375 | — |
| decode_s2048_d64_c1_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 0.232448 | 0.002048 | 0.240945 | 0.000488281 | 0.042465 | 0.00468531 | — |
| decode_s2048_d64_c1_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 0.160768 | 0.00307201 | 0.171513 | 0.00149536 | 0.0406837 | 0.00518284 | 0.16528 / 0.15576; estimate matched |
| decode_s2048_d64_c1_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 0.115712 | 0.007168 | 0.125076 | 0.000488281 | 0.00345039 | 0.000478841 | — |
| decode_s2048_d64_c1_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 0.077824 | 0.00511999 | 0.0868542 | 0.0102539 | 0.00334409 | 0.000389032 | — |
| decode_s2048_d64_c1_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s2048_d64_c1_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s2048_d64_c1_h8_kv2 | engine_dense_fp32 | float32 | OK | 0.055296 | 0.001024 | 0.0650417 | 0.126953 | 0 | 0 | — |
| decode_s2048_d64_c1_h8_kv2 | torch_math_fp32 | float32 | OK | 0.17408 | 0.00307199 | 0.184438 | 12.0645 | 3.83086e-07 | 5.96046e-08 | — |
| decode_s2048_d64_c1_h8_kv2 | apa_two_pass_fp32 | float32 | OK | 0.221184 | 0.002048 | 0.229343 | 0.00195312 | 0.0520005 | 0.00641964 | — |
| decode_s2048_d64_c1_h8_kv2 | apa_sp_fp32 | float32 | OK | 0.15872 | 0.00102399 | 0.168428 | 0.00396729 | 0.0488523 | 0.00568136 | 0.13257 / 0.15399; UNMATCHED/UNCALIBRATED |
| decode_s2048_d64_c1_h8_kv2 | engine_dense_bf16 | bfloat16 | OK | 0.052224 | 0.00512 | 0.0618957 | 0.0634766 | 0.005168 | 0.000638857 | — |
| decode_s2048_d64_c1_h8_kv2 | torch_math_bf16 | bfloat16 | OK | 0.208128 | 0.002048 | 0.218663 | 14.0664 | 0.00360134 | 0.000452776 | — |
| decode_s2048_d64_c1_h8_kv2 | apa_two_pass_bf16 | bfloat16 | OK | 0.232448 | 0.004032 | 0.241105 | 0.000976562 | 0.0509463 | 0.00609781 | — |
| decode_s2048_d64_c1_h8_kv2 | apa_sp_bf16 | bfloat16 | OK | 0.159744 | 0.00393601 | 0.169469 | 0.00299072 | 0.0480656 | 0.00549729 | 0.13147 / 0.15344; UNMATCHED/UNCALIBRATED |
| decode_s2048_d64_c1_h8_kv2 | torch_efficient_bf16 | bfloat16 | OK | 0.1536 | 0.00716799 | 0.163188 | 4.00098 | 0.00401466 | 0.000518516 | — |
| decode_s2048_d64_c1_h8_kv2 | torch_flash_bf16 | bfloat16 | OK | 0.084992 | 0.006144 | 0.0943579 | 0.0185547 | 0.00394438 | 0.000452776 | — |
| decode_s2048_d64_c1_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s2048_d64_c1_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s2048_d128_c0_h4_kv4 | engine_dense_fp32 | float32 | OK | 0.086016 | 0.028672 | 0.0983458 | 0.125031 | 0 | 0 | — |
| decode_s2048_d128_c0_h4_kv4 | torch_math_fp32 | float32 | OK | 0.172032 | 0.027648 | 0.183496 | 4.0332 | 2.70898e-07 | 3.35276e-08 | — |
| decode_s2048_d128_c0_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 0.428832 | 0.068352 | 0.438407 | 0.00195312 | 0.0501154 | 0.00719586 | — |
| decode_s2048_d128_c0_h4_kv4 | apa_sp_fp32 | float32 | OK | 0.432128 | 0.016384 | 0.442525 | 0.00393677 | 0.0489393 | 0.0070053 | 0.12219 / 0.15686; UNMATCHED/UNCALIBRATED |
| decode_s2048_d128_c0_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 0.083968 | 0.014336 | 0.0999277 | 0.0625153 | 0.00597555 | 0.000704359 | — |
| decode_s2048_d128_c0_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 0.208896 | 0.0256 | 0.220647 | 12.0352 | 0.00314474 | 0.000422351 | — |
| decode_s2048_d128_c0_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 0.414592 | 0.00502399 | 0.424531 | 0.000976562 | 0.0498823 | 0.00758737 | — |
| decode_s2048_d128_c0_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 0.418816 | 0.00307199 | 0.430132 | 0.00296021 | 0.0499725 | 0.00730509 | 0.11914 / 0.15662; UNMATCHED/UNCALIBRATED |
| decode_s2048_d128_c0_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 0.134144 | 0.01536 | 0.143731 | 0.000976562 | 0.00360349 | 0.00051409 | — |
| decode_s2048_d128_c0_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 0.09344 | 0.01504 | 0.103956 | 0.0341797 | 0.00349804 | 0.000422351 | — |
| decode_s2048_d128_c0_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s2048_d128_c0_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s2048_d128_c0_h8_kv2 | engine_dense_fp32 | float32 | OK | 0.078848 | 0.000319995 | 0.0885772 | 0.250061 | 0 | 0 | — |
| decode_s2048_d128_c0_h8_kv2 | torch_math_fp32 | float32 | OK | 0.176128 | 0.00342399 | 0.187033 | 24.0664 | 3.92739e-07 | 5.58794e-08 | — |
| decode_s2048_d128_c0_h8_kv2 | apa_two_pass_fp32 | float32 | OK | 0.4096 | 0.0256 | 0.418289 | 0.00390625 | 0.0512189 | 0.00660165 | — |
| decode_s2048_d128_c0_h8_kv2 | apa_sp_fp32 | float32 | OK | 0.416768 | 0.011264 | 0.426916 | 0.00787354 | 0.0532599 | 0.00792433 | 0.10724 / 0.15448; UNMATCHED/UNCALIBRATED |
| decode_s2048_d128_c0_h8_kv2 | engine_dense_bf16 | bfloat16 | OK | 0.074752 | 0.00409599 | 0.0851713 | 0.125031 | 0.00647401 | 0.000868037 | — |
| decode_s2048_d128_c0_h8_kv2 | torch_math_bf16 | bfloat16 | OK | 0.211968 | 0.008192 | 0.221157 | 28.0703 | 0.00356838 | 0.000457384 | — |
| decode_s2048_d128_c0_h8_kv2 | apa_two_pass_bf16 | bfloat16 | OK | 0.413312 | 0.00608 | 0.421015 | 0.00195312 | 0.0514458 | 0.00690768 | — |
| decode_s2048_d128_c0_h8_kv2 | apa_sp_bf16 | bfloat16 | OK | 0.41984 | 0.00307199 | 0.429591 | 0.00592041 | 0.0531431 | 0.00757514 | 0.10791 / 0.15442; UNMATCHED/UNCALIBRATED |
| decode_s2048_d128_c0_h8_kv2 | torch_efficient_bf16 | bfloat16 | OK | 0.165888 | 0.00524801 | 0.176163 | 8.00195 | 0.00379964 | 0.000534058 | — |
| decode_s2048_d128_c0_h8_kv2 | torch_flash_bf16 | bfloat16 | OK | 0.08704 | 0.007232 | 0.0963621 | 0.0664062 | 0.00387102 | 0.000565276 | — |
| decode_s2048_d128_c0_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s2048_d128_c0_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s2048_d128_c1_h4_kv4 | engine_dense_fp32 | float32 | OK | 0.0512 | 0.003168 | 0.0611548 | 0.0644531 | 0 | 0 | — |
| decode_s2048_d128_c1_h4_kv4 | torch_math_fp32 | float32 | OK | 0.159744 | 0.011264 | 0.170061 | 4.0332 | 2.59652e-07 | 2.79397e-08 | — |
| decode_s2048_d128_c1_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 0.425984 | 0.017408 | 0.43483 | 0.00195312 | 0.0545525 | 0.00619851 | — |
| decode_s2048_d128_c1_h4_kv4 | apa_sp_fp32 | float32 | OK | 0.472064 | 0.03072 | 0.481499 | 0.00393677 | 0.0397537 | 0.00568551 | 0.24304 / 0.15540; UNMATCHED/UNCALIBRATED |
| decode_s2048_d128_c1_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 0.0512 | 0.00512 | 0.062048 | 0.0322266 | 0.00541279 | 0.000594474 | — |
| decode_s2048_d128_c1_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 0.201728 | 0.009216 | 0.212211 | 12.0352 | 0.00296323 | 0.000436991 | — |
| decode_s2048_d128_c1_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 0.413696 | 0.004096 | 0.422968 | 0.000976562 | 0.0550464 | 0.00596386 | — |
| decode_s2048_d128_c1_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 0.470016 | 0.003968 | 0.481208 | 0.00296021 | 0.0398179 | 0.00524874 | 0.23608 / 0.15576; UNMATCHED/UNCALIBRATED |
| decode_s2048_d128_c1_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 0.128672 | 0.00511999 | 0.138351 | 0.000976562 | 0.00334788 | 0.000436991 | — |
| decode_s2048_d128_c1_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 0.083296 | 0.006848 | 0.0932859 | 0.0341797 | 0.00337648 | 0.000476312 | — |
| decode_s2048_d128_c1_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s2048_d128_c1_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s2048_d128_c1_h8_kv2 | engine_dense_fp32 | float32 | OK | 0.05632 | 0.000992 | 0.065533 | 0.128906 | 0 | 0 | — |
| decode_s2048_d128_c1_h8_kv2 | torch_math_fp32 | float32 | OK | 0.177152 | 0.01024 | 0.188235 | 24.0664 | 3.95518e-07 | 5.96046e-08 | — |
| decode_s2048_d128_c1_h8_kv2 | apa_two_pass_fp32 | float32 | OK | 0.41136 | 0.01024 | 0.420824 | 0.00390625 | 0.0531452 | 0.00654413 | — |
| decode_s2048_d128_c1_h8_kv2 | apa_sp_fp32 | float32 | OK | 0.452608 | 0.00307199 | 0.463494 | 0.00787354 | 0.0446851 | 0.00572252 | 0.18451 / 0.15784; UNMATCHED/UNCALIBRATED |
| decode_s2048_d128_c1_h8_kv2 | engine_dense_bf16 | bfloat16 | OK | 0.055264 | 0.005888 | 0.0651032 | 0.0644531 | 0.00530404 | 0.000731081 | — |
| decode_s2048_d128_c1_h8_kv2 | torch_math_bf16 | bfloat16 | OK | 0.216448 | 0.008192 | 0.226938 | 28.0703 | 0.00346997 | 0.000536386 | — |
| decode_s2048_d128_c1_h8_kv2 | apa_two_pass_bf16 | bfloat16 | OK | 0.433152 | 0.027648 | 0.442195 | 0.00195312 | 0.0523491 | 0.00712503 | — |
| decode_s2048_d128_c1_h8_kv2 | apa_sp_bf16 | bfloat16 | OK | 0.454656 | 0.004096 | 0.465007 | 0.00592041 | 0.0442567 | 0.00590433 | 0.18353 / 0.15747; UNMATCHED/UNCALIBRATED |
| decode_s2048_d128_c1_h8_kv2 | torch_efficient_bf16 | bfloat16 | OK | 0.16896 | 0.00300801 | 0.178867 | 8.00195 | 0.00384797 | 0.000451744 | — |
| decode_s2048_d128_c1_h8_kv2 | torch_flash_bf16 | bfloat16 | OK | 0.086912 | 0.006656 | 0.0959509 | 0.0664062 | 0.00386253 | 0.000505868 | — |
| decode_s2048_d128_c1_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s2048_d128_c1_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s8192_d64_c0_h4_kv4 | engine_dense_fp32 | float32 | OK | 0.08704 | 0.01024 | 0.0963919 | 0.500031 | 0 | 0 | — |
| decode_s8192_d64_c0_h4_kv4 | torch_math_fp32 | float32 | OK | 0.165888 | 0.00307199 | 0.177896 | 8.12598 | 2.41267e-07 | 1.11759e-08 | — |
| decode_s8192_d64_c0_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 0.46048 | 0.0512 | 0.470417 | 0.00502014 | 0.0549777 | 0.00343207 | — |
| decode_s8192_d64_c0_h4_kv4 | apa_sp_fp32 | float32 | OK | 0.177152 | 0.017408 | 0.187383 | 0.00500488 | 0.0332827 | 0.00185169 | 0.27887 / 0.15482; UNMATCHED/UNCALIBRATED |
| decode_s8192_d64_c0_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 0.079872 | 0.00716799 | 0.0908212 | 0.250015 | 0.00658463 | 0.000371398 | — |
| decode_s8192_d64_c0_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 0.206848 | 0.00803201 | 0.218181 | 24.127 | 0.00350896 | 0.000212383 | — |
| decode_s8192_d64_c0_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 0.405408 | 0.0072 | 0.416325 | 0.00453186 | 0.0554503 | 0.00324122 | — |
| decode_s8192_d64_c0_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 0.16896 | 0.010304 | 0.17993 | 0.0045166 | 0.0337244 | 0.00163739 | 0.27969 / 0.15469; UNMATCHED/UNCALIBRATED |
| decode_s8192_d64_c0_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 0.3328 | 0.016192 | 0.343278 | 0.000488281 | 0.00412931 | 0.000212383 | — |
| decode_s8192_d64_c0_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 0.088064 | 0.007904 | 0.101832 | 0.0180664 | 0.00379032 | 0.000212383 | — |
| decode_s8192_d64_c0_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s8192_d64_c0_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s8192_d64_c0_h8_kv2 | engine_dense_fp32 | float32 | OK | 0.164864 | 0.00307201 | 0.176644 | 1.00006 | 0 | 0 | — |
| decode_s8192_d64_c0_h8_kv2 | torch_math_fp32 | float32 | OK | 0.218112 | 0.008448 | 0.228862 | 48.252 | 4.65604e-07 | 3.1665e-08 | — |
| decode_s8192_d64_c0_h8_kv2 | apa_two_pass_fp32 | float32 | OK | 0.459776 | 0.042848 | 0.470076 | 0.0100403 | 0.0441415 | 0.00289567 | — |
| decode_s8192_d64_c0_h8_kv2 | apa_sp_fp32 | float32 | OK | 0.1792 | 0.004096 | 0.188606 | 0.0100098 | 0.0317948 | 0.00280122 | 0.23837 / 0.15401; UNMATCHED/UNCALIBRATED |
| decode_s8192_d64_c0_h8_kv2 | engine_dense_bf16 | bfloat16 | OK | 0.118784 | 0.005984 | 0.130456 | 0.500031 | 0.00659625 | 0.000487726 | — |
| decode_s8192_d64_c0_h8_kv2 | torch_math_bf16 | bfloat16 | OK | 0.230656 | 0.009216 | 0.242317 | 56.2539 | 0.003451 | 0.000231732 | — |
| decode_s8192_d64_c0_h8_kv2 | apa_two_pass_bf16 | bfloat16 | OK | 0.408576 | 0.014336 | 0.419502 | 0.00906372 | 0.0444302 | 0.00261135 | — |
| decode_s8192_d64_c0_h8_kv2 | apa_sp_bf16 | bfloat16 | OK | 0.177152 | 0.00511999 | 0.187594 | 0.0090332 | 0.0314723 | 0.00318813 | 0.23782 / 0.15382; UNMATCHED/UNCALIBRATED |
| decode_s8192_d64_c0_h8_kv2 | torch_efficient_bf16 | bfloat16 | OK | 0.367616 | 0.026624 | 0.377172 | 16.001 | 0.00366644 | 0.000227518 | — |
| decode_s8192_d64_c0_h8_kv2 | torch_flash_bf16 | bfloat16 | OK | 0.093184 | 0.009216 | 0.103866 | 0.065918 | 0.00380255 | 0.000352278 | — |
| decode_s8192_d64_c0_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s8192_d64_c0_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s8192_d64_c1_h4_kv4 | engine_dense_fp32 | float32 | OK | 0.073696 | 0.006144 | 0.0828872 | 0.250977 | 0 | 0 | — |
| decode_s8192_d64_c1_h4_kv4 | torch_math_fp32 | float32 | OK | 0.161792 | 0.00511999 | 0.172105 | 8.12598 | 2.69094e-07 | 1.95578e-08 | — |
| decode_s8192_d64_c1_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 0.457728 | 0.053216 | 0.467211 | 0.00502014 | 0.049545 | 0.0025468 | — |
| decode_s8192_d64_c1_h4_kv4 | apa_sp_fp32 | float32 | OK | 0.171008 | 0.01536 | 0.181292 | 0.00500488 | 0.0418887 | 0.00236616 | 0.17633 / 0.15494; UNMATCHED/UNCALIBRATED |
| decode_s8192_d64_c1_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 0.05728 | 0.00432 | 0.0669667 | 0.125488 | 0.00571221 | 0.000369534 | — |
| decode_s8192_d64_c1_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 0.201728 | 0.006144 | 0.212151 | 24.127 | 0.00394905 | 0.000292663 | — |
| decode_s8192_d64_c1_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 0.405504 | 0.00921601 | 0.415805 | 0.00453186 | 0.0491263 | 0.0026991 | — |
| decode_s8192_d64_c1_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 0.165888 | 0.012064 | 0.17495 | 0.0045166 | 0.0415974 | 0.00252555 | 0.17731 / 0.15469; UNMATCHED/UNCALIBRATED |
| decode_s8192_d64_c1_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 0.329728 | 0.012288 | 0.33958 | 0.000488281 | 0.00404759 | 0.000292663 | — |
| decode_s8192_d64_c1_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 0.08416 | 0.007488 | 0.0945893 | 0.0180664 | 0.00409175 | 0.000292663 | — |
| decode_s8192_d64_c1_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s8192_d64_c1_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s8192_d64_c1_h8_kv2 | engine_dense_fp32 | float32 | OK | 0.146432 | 0.00281601 | 0.155734 | 0.501953 | 0 | 0 | — |
| decode_s8192_d64_c1_h8_kv2 | torch_math_fp32 | float32 | OK | 0.212288 | 0.01024 | 0.22298 | 48.252 | 5.24948e-07 | 2.6077e-08 | — |
| decode_s8192_d64_c1_h8_kv2 | apa_two_pass_fp32 | float32 | OK | 0.458752 | 0.039936 | 0.468655 | 0.0100403 | 0.0532318 | 0.00264335 | — |
| decode_s8192_d64_c1_h8_kv2 | apa_sp_fp32 | float32 | OK | 0.175104 | 0.00511999 | 0.184879 | 0.0100098 | 0.0355219 | 0.00203789 | 0.26540 / 0.15616; UNMATCHED/UNCALIBRATED |
| decode_s8192_d64_c1_h8_kv2 | engine_dense_bf16 | bfloat16 | OK | 0.095136 | 0.004768 | 0.104487 | 0.250977 | 0.00564186 | 0.000353537 | — |
| decode_s8192_d64_c1_h8_kv2 | torch_math_bf16 | bfloat16 | OK | 0.22528 | 0.00512001 | 0.235384 | 56.2539 | 0.00395208 | 0.000278298 | — |
| decode_s8192_d64_c1_h8_kv2 | apa_two_pass_bf16 | bfloat16 | OK | 0.408576 | 0.012288 | 0.418079 | 0.00906372 | 0.053626 | 0.00308783 | — |
| decode_s8192_d64_c1_h8_kv2 | apa_sp_bf16 | bfloat16 | OK | 0.172032 | 0.006016 | 0.182003 | 0.0090332 | 0.0356595 | 0.00223334 | 0.26598 / 0.15593; UNMATCHED/UNCALIBRATED |
| decode_s8192_d64_c1_h8_kv2 | torch_efficient_bf16 | bfloat16 | OK | 0.362496 | 0.013696 | 0.372192 | 16.001 | 0.00422048 | 0.000278298 | — |
| decode_s8192_d64_c1_h8_kv2 | torch_flash_bf16 | bfloat16 | OK | 0.08704 | 0.01104 | 0.0958699 | 0.065918 | 0.00425832 | 0.000278298 | — |
| decode_s8192_d64_c1_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s8192_d64_c1_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s8192_d128_c0_h4_kv4 | engine_dense_fp32 | float32 | OK | 0.141312 | 0.026624 | 0.151486 | 0.500031 | 0 | 0 | — |
| decode_s8192_d128_c0_h4_kv4 | torch_math_fp32 | float32 | OK | 0.17408 | 0.008384 | 0.184007 | 16.127 | 3.87626e-07 | 2.23517e-08 | — |
| decode_s8192_d128_c0_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 1.26976 | 0.37376 | 1.27918 | 0.00990295 | 0.0555598 | 0.00314189 | — |
| decode_s8192_d128_c0_h4_kv4 | apa_sp_fp32 | float32 | OK | 0.57856 | 0.152576 | 0.590233 | 0.0098877 | 0.0324999 | 0.00237498 | 0.28165 / 0.15381; UNMATCHED/UNCALIBRATED |
| decode_s8192_d128_c0_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 0.093184 | 0.002048 | 0.103635 | 0.250015 | 0.00707981 | 0.000472974 | — |
| decode_s8192_d128_c0_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 0.207872 | 0.014336 | 0.218312 | 48.1289 | 0.00339942 | 0.000215519 | — |
| decode_s8192_d128_c0_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 1.2544 | 0.37376 | 1.26461 | 0.00892639 | 0.0558226 | 0.00302262 | — |
| decode_s8192_d128_c0_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 0.539648 | 0.199584 | 0.548655 | 0.00891113 | 0.0329561 | 0.00244828 | 0.28284 / 0.15372; UNMATCHED/UNCALIBRATED |
| decode_s8192_d128_c0_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 0.379904 | 0.022528 | 0.390687 | 0.000976562 | 0.00362777 | 0.000215519 | — |
| decode_s8192_d128_c0_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 0.089088 | 0.022848 | 0.0985661 | 0.0458984 | 0.00365749 | 0.000228833 | — |
| decode_s8192_d128_c0_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s8192_d128_c0_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s8192_d128_c0_h8_kv2 | engine_dense_fp32 | float32 | OK | 0.178176 | 0.006144 | 0.188606 | 1.00006 | 0 | 0 | — |
| decode_s8192_d128_c0_h8_kv2 | torch_math_fp32 | float32 | OK | 0.482304 | 0.051552 | 0.493631 | 96.2539 | 6.69927e-07 | 3.91155e-08 | — |
| decode_s8192_d128_c0_h8_kv2 | apa_two_pass_fp32 | float32 | OK | 1.18886 | 0.308224 | 1.19984 | 0.0198059 | 0.0530095 | 0.00312865 | — |
| decode_s8192_d128_c0_h8_kv2 | apa_sp_fp32 | float32 | OK | 0.509952 | 0.070656 | 0.519289 | 0.0197754 | 0.0334376 | 0.00180414 | 0.27168 / 0.15480; UNMATCHED/UNCALIBRATED |
| decode_s8192_d128_c0_h8_kv2 | engine_dense_bf16 | bfloat16 | OK | 0.119808 | 0.007168 | 0.132611 | 0.500031 | 0.00685945 | 0.000424568 | — |
| decode_s8192_d128_c0_h8_kv2 | torch_math_bf16 | bfloat16 | OK | 0.505856 | 0.00480002 | 0.516985 | 112.258 | 0.00330848 | 0.000290602 | — |
| decode_s8192_d128_c0_h8_kv2 | apa_two_pass_bf16 | bfloat16 | OK | 1.18374 | 0.29696 | 1.19468 | 0.0178528 | 0.0527954 | 0.00278638 | — |
| decode_s8192_d128_c0_h8_kv2 | apa_sp_bf16 | bfloat16 | OK | 0.508928 | 0.054272 | 0.51932 | 0.0178223 | 0.0335812 | 0.00198916 | 0.27226 / 0.15485; UNMATCHED/UNCALIBRATED |
| decode_s8192_d128_c0_h8_kv2 | torch_efficient_bf16 | bfloat16 | OK | 0.446464 | 0.025856 | 0.456562 | 32.002 | 0.00360463 | 0.000189383 | — |
| decode_s8192_d128_c0_h8_kv2 | torch_flash_bf16 | bfloat16 | OK | 0.093248 | 0.019424 | 0.106922 | 0.129395 | 0.00365684 | 0.000220407 | — |
| decode_s8192_d128_c0_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s8192_d128_c0_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s8192_d128_c1_h4_kv4 | engine_dense_fp32 | float32 | OK | 0.134048 | 0.033088 | 0.14303 | 0.251953 | 0 | 0 | — |
| decode_s8192_d128_c1_h4_kv4 | torch_math_fp32 | float32 | OK | 0.171008 | 0.020384 | 0.181182 | 16.127 | 3.48295e-07 | 2.04891e-08 | — |
| decode_s8192_d128_c1_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 1.2665 | 0.375808 | 1.27568 | 0.00990295 | 0.048388 | 0.00304752 | — |
| decode_s8192_d128_c1_h4_kv4 | apa_sp_fp32 | float32 | OK | 0.550912 | 0.176128 | 0.719408 | 0.0098877 | 0.0374347 | 0.00252452 | 0.21637 / 0.15494; UNMATCHED/UNCALIBRATED |
| decode_s8192_d128_c1_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 0.08704 | 0.009568 | 0.0967318 | 0.125977 | 0.00564688 | 0.000381876 | — |
| decode_s8192_d128_c1_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 0.212992 | 0.034496 | 0.223522 | 48.1289 | 0.00338052 | 0.00019921 | — |
| decode_s8192_d128_c1_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 1.25312 | 0.36864 | 1.26343 | 0.00892639 | 0.0496813 | 0.00308999 | — |
| decode_s8192_d128_c1_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 0.512 | 0.200704 | 0.534499 | 0.00891113 | 0.0381959 | 0.00246083 | 0.21487 / 0.15509; UNMATCHED/UNCALIBRATED |
| decode_s8192_d128_c1_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 0.377856 | 0.026624 | 0.386689 | 0.000976562 | 0.00383249 | 0.00022984 | — |
| decode_s8192_d128_c1_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 0.093184 | 0.02048 | 0.110298 | 0.0458984 | 0.00380933 | 0.000228055 | — |
| decode_s8192_d128_c1_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s8192_d128_c1_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s8192_d128_c1_h8_kv2 | engine_dense_fp32 | float32 | OK | 0.16896 | 0.009984 | 0.178447 | 0.503906 | 0 | 0 | — |
| decode_s8192_d128_c1_h8_kv2 | torch_math_fp32 | float32 | OK | 0.46592 | 0.011264 | 0.476779 | 96.2539 | 6.63728e-07 | 4.65661e-08 | — |
| decode_s8192_d128_c1_h8_kv2 | apa_two_pass_fp32 | float32 | OK | 1.19088 | 0.315392 | 1.20087 | 0.0198059 | 0.0550553 | 0.00276696 | — |
| decode_s8192_d128_c1_h8_kv2 | apa_sp_fp32 | float32 | OK | 0.500736 | 0.070656 | 0.510944 | 0.0197754 | 0.0373252 | 0.00201528 | 0.24947 / 0.15434; UNMATCHED/UNCALIBRATED |
| decode_s8192_d128_c1_h8_kv2 | engine_dense_bf16 | bfloat16 | OK | 0.103424 | 0.004096 | 0.113604 | 0.251953 | 0.00550215 | 0.000336748 | — |
| decode_s8192_d128_c1_h8_kv2 | torch_math_bf16 | bfloat16 | OK | 0.499712 | 0.003104 | 0.510613 | 112.258 | 0.00349729 | 0.000247795 | — |
| decode_s8192_d128_c1_h8_kv2 | apa_two_pass_bf16 | bfloat16 | OK | 1.18477 | 0.3072 | 1.1952 | 0.0178528 | 0.0553541 | 0.00287434 | — |
| decode_s8192_d128_c1_h8_kv2 | apa_sp_bf16 | bfloat16 | OK | 0.49664 | 0.055232 | 0.507538 | 0.0178223 | 0.0379927 | 0.00208424 | 0.24892 / 0.15430; UNMATCHED/UNCALIBRATED |
| decode_s8192_d128_c1_h8_kv2 | torch_efficient_bf16 | bfloat16 | OK | 0.433152 | 0.026624 | 0.443306 | 32.002 | 0.0039624 | 0.000247795 | — |
| decode_s8192_d128_c1_h8_kv2 | torch_flash_bf16 | bfloat16 | OK | 0.094208 | 0.010144 | 0.104788 | 0.129395 | 0.00379778 | 0.000247795 | — |
| decode_s8192_d128_c1_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s8192_d128_c1_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s32768_d64_c0_h4_kv4 | engine_dense_fp32 | float32 | OK | 0.22096 | 0.010336 | 0.233511 | 2.00003 | 0 | 0 | — |
| decode_s32768_d64_c0_h4_kv4 | torch_math_fp32 | float32 | OK | 0.27648 | 0.004096 | 0.290588 | 32.501 | 2.9194e-07 | 1.11759e-08 | — |
| decode_s32768_d64_c0_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 1.38752 | 0.063488 | 1.39729 | 0.0171051 | 0.0466869 | 0.00132233 | — |
| decode_s32768_d64_c0_h4_kv4 | apa_sp_fp32 | float32 | OK | 0.311296 | 0.017408 | 0.321367 | 0.0170898 | 0.0268888 | 0.000808512 | 0.34714 / 0.15475; UNMATCHED/UNCALIBRATED |
| decode_s32768_d64_c0_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 0.15136 | 0.013312 | 0.162226 | 1.00002 | 0.00679846 | 0.000219987 | — |
| decode_s32768_d64_c0_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 0.495616 | 0.024576 | 0.505855 | 96.502 | 0.00346973 | 0.000110149 | — |
| decode_s32768_d64_c0_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 1.18477 | 0.029696 | 1.19521 | 0.0166168 | 0.0467299 | 0.00129573 | — |
| decode_s32768_d64_c0_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 0.295936 | 0.00716799 | 0.305707 | 0.0166016 | 0.027388 | 0.000883491 | 0.34769 / 0.15476; UNMATCHED/UNCALIBRATED |
| decode_s32768_d64_c0_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 1.25056 | 0.074752 | 1.26091 | 0.000488281 | 0.0036527 | 0.000110149 | — |
| decode_s32768_d64_c0_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 0.109568 | 0.042912 | 0.124185 | 0.027832 | 0.00363733 | 0.000110149 | — |
| decode_s32768_d64_c0_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s32768_d64_c0_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s32768_d64_c0_h8_kv2 | engine_dense_fp32 | float32 | OK | 0.5168 | 0.009152 | 0.533837 | 4.00006 | 0 | 0 | — |
| decode_s32768_d64_c0_h8_kv2 | torch_math_fp32 | float32 | OK | 0.973824 | 0.00729597 | 0.99107 | 193.002 | 9.08244e-07 | 2.79397e-08 | — |
| decode_s32768_d64_c0_h8_kv2 | apa_two_pass_fp32 | float32 | OK | 1.43155 | 0.0829439 | 1.44436 | 0.0342102 | 0.0543019 | 0.00182862 | — |
| decode_s32768_d64_c0_h8_kv2 | apa_sp_fp32 | float32 | OK | 0.444416 | 0.00204799 | 0.454678 | 0.0341797 | 0.027671 | 0.00117579 | 0.36295 / 0.15452; UNMATCHED/UNCALIBRATED |
| decode_s32768_d64_c0_h8_kv2 | engine_dense_bf16 | bfloat16 | OK | 0.306144 | 0.00614399 | 0.316978 | 2.00003 | 0.00720896 | 0.00021534 | — |
| decode_s32768_d64_c0_h8_kv2 | torch_math_bf16 | bfloat16 | OK | 1.07213 | 0.01024 | 1.08567 | 225.004 | 0.00343544 | 0.00010153 | — |
| decode_s32768_d64_c0_h8_kv2 | apa_two_pass_bf16 | bfloat16 | OK | 1.30048 | 0.016384 | 1.31091 | 0.0332336 | 0.0547679 | 0.00195631 | — |
| decode_s32768_d64_c0_h8_kv2 | apa_sp_bf16 | bfloat16 | OK | 0.442368 | 0.00511998 | 0.452754 | 0.0332031 | 0.0275322 | 0.00121532 | 0.36292 / 0.15461; UNMATCHED/UNCALIBRATED |
| decode_s32768_d64_c0_h8_kv2 | torch_efficient_bf16 | bfloat16 | OK | 1.41238 | 0.033792 | 1.42292 | 64.001 | 0.00370402 | 0.00012745 | — |
| decode_s32768_d64_c0_h8_kv2 | torch_flash_bf16 | bfloat16 | OK | 0.120832 | 0.01824 | 0.13232 | 0.0878906 | 0.00373158 | 0.000114007 | — |
| decode_s32768_d64_c0_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s32768_d64_c0_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s32768_d64_c1_h4_kv4 | engine_dense_fp32 | float32 | OK | 0.23552 | 0.00908801 | 0.247978 | 1.00098 | 0 | 0 | — |
| decode_s32768_d64_c1_h4_kv4 | torch_math_fp32 | float32 | OK | 0.282784 | 0.064448 | 0.307139 | 32.501 | 3.38393e-07 | 1.02445e-08 | — |
| decode_s32768_d64_c1_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 1.39264 | 0.066432 | 1.40509 | 0.0171051 | 0.0462271 | 0.0014078 | — |
| decode_s32768_d64_c1_h4_kv4 | apa_sp_fp32 | float32 | OK | 0.318464 | 0.018432 | 0.341785 | 0.0170898 | 0.0319007 | 0.00134339 | 0.32500 / 0.15415; UNMATCHED/UNCALIBRATED |
| decode_s32768_d64_c1_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 0.161568 | 0.011264 | 0.172045 | 0.500488 | 0.00657582 | 0.0003135 | — |
| decode_s32768_d64_c1_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 0.49152 | 0.00921601 | 0.505624 | 96.502 | 0.00351328 | 0.000122458 | — |
| decode_s32768_d64_c1_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 1.18464 | 0.026592 | 1.19584 | 0.0166168 | 0.0471967 | 0.00156273 | — |
| decode_s32768_d64_c1_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 0.301056 | 0.008192 | 0.313792 | 0.0166016 | 0.0330532 | 0.00126849 | 0.32524 / 0.15431; UNMATCHED/UNCALIBRATED |
| decode_s32768_d64_c1_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 1.25338 | 0.077984 | 1.26348 | 0.000488281 | 0.00385893 | 0.000122458 | — |
| decode_s32768_d64_c1_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 0.106592 | 0.013312 | 0.126038 | 0.027832 | 0.00383845 | 0.000121683 | — |
| decode_s32768_d64_c1_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s32768_d64_c1_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s32768_d64_c1_h8_kv2 | engine_dense_fp32 | float32 | OK | 0.526336 | 0.015392 | 0.536672 | 2.00195 | 0 | 0 | — |
| decode_s32768_d64_c1_h8_kv2 | torch_math_fp32 | float32 | OK | 0.974848 | 0.00921601 | 0.987133 | 193.002 | 9.29128e-07 | 2.79397e-08 | — |
| decode_s32768_d64_c1_h8_kv2 | apa_two_pass_fp32 | float32 | OK | 1.43226 | 0.09216 | 1.44268 | 0.0342102 | 0.0501363 | 0.00148214 | — |
| decode_s32768_d64_c1_h8_kv2 | apa_sp_fp32 | float32 | OK | 0.46592 | 0.00204802 | 0.475347 | 0.0341797 | 0.0245616 | 0.000694385 | 0.41681 / 0.15524; UNMATCHED/UNCALIBRATED |
| decode_s32768_d64_c1_h8_kv2 | engine_dense_bf16 | bfloat16 | OK | 0.311296 | 0.00627199 | 0.321938 | 1.00098 | 0.00566505 | 0.000181435 | — |
| decode_s32768_d64_c1_h8_kv2 | torch_math_bf16 | bfloat16 | OK | 1.08749 | 0.0235519 | 1.09811 | 225.004 | 0.00346763 | 0.000116399 | — |
| decode_s32768_d64_c1_h8_kv2 | apa_two_pass_bf16 | bfloat16 | OK | 1.28922 | 0.0211201 | 1.3187 | 0.0332336 | 0.0503632 | 0.00147697 | — |
| decode_s32768_d64_c1_h8_kv2 | apa_sp_bf16 | bfloat16 | OK | 0.458752 | 0.01024 | 0.475296 | 0.0332031 | 0.0245804 | 0.000755893 | 0.41636 / 0.15525; UNMATCHED/UNCALIBRATED |
| decode_s32768_d64_c1_h8_kv2 | torch_efficient_bf16 | bfloat16 | OK | 1.41926 | 0.0349759 | 1.43011 | 64.001 | 0.00388477 | 0.000116399 | — |
| decode_s32768_d64_c1_h8_kv2 | torch_flash_bf16 | bfloat16 | OK | 0.126048 | 0.00336 | 0.136578 | 0.0878906 | 0.00391305 | 0.000119514 | — |
| decode_s32768_d64_c1_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s32768_d64_c1_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s32768_d128_c0_h4_kv4 | engine_dense_fp32 | float32 | OK | 0.400384 | 0.027648 | 0.410705 | 2.00003 | 0 | 0 | — |
| decode_s32768_d128_c0_h4_kv4 | torch_math_fp32 | float32 | OK | 0.625664 | 0.0245759 | 0.638495 | 64.502 | 4.86996e-07 | 1.67638e-08 | — |
| decode_s32768_d128_c0_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 4.03859 | 0.0051198 | 4.04888 | 0.0337067 | 0.0511164 | 0.00154531 | — |
| decode_s32768_d128_c0_h4_kv4 | apa_sp_fp32 | float32 | OK | 0.745472 | 0.00614399 | 0.759693 | 0.0336914 | 0.0262223 | 0.000997728 | 0.36419 / 0.15482; UNMATCHED/UNCALIBRATED |
| decode_s32768_d128_c0_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 0.233088 | 0.00995199 | 0.24397 | 1.00002 | 0.00736082 | 0.000275623 | — |
| decode_s32768_d128_c0_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 1.09158 | 0.00911999 | 1.10584 | 192.504 | 0.00328578 | 0.000116026 | — |
| decode_s32768_d128_c0_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 3.9697 | 0.656096 | 3.97979 | 0.0327301 | 0.0511291 | 0.0014736 | — |
| decode_s32768_d128_c0_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 0.717824 | 0.00511998 | 0.727493 | 0.0327148 | 0.026744 | 0.000873247 | 0.36237 / 0.15445; UNMATCHED/UNCALIBRATED |
| decode_s32768_d128_c0_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 1.42643 | 0.018432 | 1.43664 | 0.000976562 | 0.00365262 | 0.000141971 | — |
| decode_s32768_d128_c0_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 0.177152 | 0.055296 | 0.186572 | 0.0498047 | 0.00351758 | 0.000116026 | — |
| decode_s32768_d128_c0_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s32768_d128_c0_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s32768_d128_c0_h8_kv2 | engine_dense_fp32 | float32 | OK | 0.559104 | 0.00838405 | 0.570225 | 4.00006 | 0 | 0 | — |
| decode_s32768_d128_c0_h8_kv2 | torch_math_fp32 | float32 | OK | 2.35213 | 1.06803 | 2.36401 | 385.004 | 1.17448e-06 | 4.65661e-08 | — |
| decode_s32768_d128_c0_h8_kv2 | apa_two_pass_fp32 | float32 | OK | 3.85434 | 0.494592 | 3.86631 | 0.0674133 | 0.0498194 | 0.00189156 | — |
| decode_s32768_d128_c0_h8_kv2 | apa_sp_fp32 | float32 | OK | 0.633856 | 0.012288 | 0.64614 | 0.0673828 | 0.0262043 | 0.000847502 | 0.37333 / 0.15427; UNMATCHED/UNCALIBRATED |
| decode_s32768_d128_c0_h8_kv2 | engine_dense_bf16 | bfloat16 | OK | 0.320512 | 0.00921601 | 0.331034 | 2.00003 | 0.00687548 | 0.000234218 | — |
| decode_s32768_d128_c0_h8_kv2 | torch_math_bf16 | bfloat16 | OK | 2.65421 | 0.0266242 | 2.66535 | 449.008 | 0.00330114 | 0.00011682 | — |
| decode_s32768_d128_c0_h8_kv2 | apa_two_pass_bf16 | bfloat16 | OK | 3.78141 | 1.08339 | 3.7914 | 0.0654602 | 0.0489203 | 0.00168599 | — |
| decode_s32768_d128_c0_h8_kv2 | apa_sp_bf16 | bfloat16 | OK | 0.616384 | 0.041984 | 0.628545 | 0.0654297 | 0.0260577 | 0.00102827 | 0.37367 / 0.15439; UNMATCHED/UNCALIBRATED |
| decode_s32768_d128_c0_h8_kv2 | torch_efficient_bf16 | bfloat16 | OK | 1.73978 | 0.015456 | 1.75915 | 128.002 | 0.00365268 | 0.000120031 | — |
| decode_s32768_d128_c0_h8_kv2 | torch_flash_bf16 | bfloat16 | OK | 0.149408 | 0.018432 | 0.165772 | 0.208496 | 0.00368254 | 0.000125632 | — |
| decode_s32768_d128_c0_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s32768_d128_c0_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s32768_d128_c1_h4_kv4 | engine_dense_fp32 | float32 | OK | 0.400384 | 0.02736 | 0.411075 | 1.00195 | 0 | 0 | — |
| decode_s32768_d128_c1_h4_kv4 | torch_math_fp32 | float32 | OK | 0.628736 | 0.050176 | 0.640087 | 64.502 | 4.90344e-07 | 1.67638e-08 | — |
| decode_s32768_d128_c1_h4_kv4 | apa_two_pass_fp32 | float32 | OK | 4.03315 | 0.190464 | 4.04513 | 0.0337067 | 0.0481096 | 0.00138046 | — |
| decode_s32768_d128_c1_h4_kv4 | apa_sp_fp32 | float32 | OK | 0.759808 | 0.00409603 | 0.769883 | 0.0336914 | 0.0244053 | 0.000716048 | 0.36742 / 0.15373; UNMATCHED/UNCALIBRATED |
| decode_s32768_d128_c1_h4_kv4 | engine_dense_bf16 | bfloat16 | OK | 0.236544 | 0.02048 | 0.256474 | 0.500977 | 0.00514849 | 0.0002063 | — |
| decode_s32768_d128_c1_h4_kv4 | torch_math_bf16 | bfloat16 | OK | 1.09373 | 0.0152 | 1.10579 | 192.504 | 0.00325341 | 0.000120573 | — |
| decode_s32768_d128_c1_h4_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 3.96493 | 0.638944 | 3.97585 | 0.0327301 | 0.0491204 | 0.0013992 | — |
| decode_s32768_d128_c1_h4_kv4 | apa_sp_bf16 | bfloat16 | OK | 0.724992 | 0.00511998 | 0.734566 | 0.0327148 | 0.024755 | 0.000721463 | 0.36742 / 0.15381; UNMATCHED/UNCALIBRATED |
| decode_s32768_d128_c1_h4_kv4 | torch_efficient_bf16 | bfloat16 | OK | 1.41962 | 0.0136 | 1.43015 | 0.000976562 | 0.00352407 | 0.000122729 | — |
| decode_s32768_d128_c1_h4_kv4 | torch_flash_bf16 | bfloat16 | OK | 0.167904 | 0.065664 | 0.176914 | 0.0498047 | 0.00343039 | 0.000104137 | — |
| decode_s32768_d128_c1_h4_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s32768_d128_c1_h4_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s32768_d128_c1_h8_kv2 | engine_dense_fp32 | float32 | OK | 0.568128 | 0.023552 | 0.590885 | 2.00391 | 0 | 0 | — |
| decode_s32768_d128_c1_h8_kv2 | torch_math_fp32 | float32 | OK | 2.36544 | 0.188576 | 2.37783 | 385.004 | 1.1362e-06 | 5.21541e-08 | — |
| decode_s32768_d128_c1_h8_kv2 | apa_two_pass_fp32 | float32 | OK | 3.85325 | 0.543744 | 3.86386 | 0.0674133 | 0.0505353 | 0.00160021 | — |
| decode_s32768_d128_c1_h8_kv2 | apa_sp_fp32 | float32 | OK | 0.612352 | 0.00921601 | 0.630119 | 0.0673828 | 0.0282802 | 0.0010743 | 0.31977 / 0.15502; UNMATCHED/UNCALIBRATED |
| decode_s32768_d128_c1_h8_kv2 | engine_dense_bf16 | bfloat16 | OK | 0.321536 | 0.014336 | 0.332848 | 1.00195 | 0.00527848 | 0.000202027 | — |
| decode_s32768_d128_c1_h8_kv2 | torch_math_bf16 | bfloat16 | OK | 2.64819 | 0.0174079 | 2.65971 | 449.008 | 0.00331291 | 0.000113314 | — |
| decode_s32768_d128_c1_h8_kv2 | apa_two_pass_bf16 | bfloat16 | OK | 3.74886 | 1.08749 | 3.75969 | 0.0654602 | 0.0511543 | 0.00156823 | — |
| decode_s32768_d128_c1_h8_kv2 | apa_sp_bf16 | bfloat16 | OK | 0.558944 | 0.059392 | 0.568653 | 0.0654297 | 0.0287009 | 0.00111939 | 0.31916 / 0.15480; UNMATCHED/UNCALIBRATED |
| decode_s32768_d128_c1_h8_kv2 | torch_efficient_bf16 | bfloat16 | OK | 1.74371 | 0.04608 | 1.75645 | 128.002 | 0.00357926 | 0.000112308 | — |
| decode_s32768_d128_c1_h8_kv2 | torch_flash_bf16 | bfloat16 | OK | 0.154624 | 0.013152 | 0.164861 | 0.208496 | 0.00353439 | 0.000107568 | — |
| decode_s32768_d128_c1_h8_kv2 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| decode_s32768_d128_c1_h8_kv2 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| e1_prefill_l512_s8192_d128_c1_h16_kv4 | engine_dense_fp32 | float32 | OK | 3.72605 | 0.134016 | 9.31941 | 516 | 0 | 0 | — |
| e1_prefill_l512_s8192_d128_c1_h16_kv4 | torch_math_fp32 | float32 | OK | 8.64666 | 0.760832 | 8.67404 | 728.008 | 8.56897e-07 | 1.56462e-07 | — |
| e1_prefill_l512_s8192_d128_c1_h16_kv4 | apa_two_pass_fp32 | float32 | OK | 37.7324 | 1.32403 | 37.7888 | 4 | 0.0655104 | 0.00629896 | — |
| e1_prefill_l512_s8192_d128_c1_h16_kv4 | apa_sp_fp32 | float32 | OK | 32.4252 | 1.09568 | 32.4381 | 4 | 0.0533828 | 0.00757822 | 0.15165 / 0.11628; UNMATCHED/UNCALIBRATED |
| e1_prefill_l512_s8192_d128_c1_h16_kv4 | engine_dense_bf16 | bfloat16 | OK | 1.73978 | 0.048128 | 1.86741 | 258 | 0.00519269 | 0.000887983 | — |
| e1_prefill_l512_s8192_d128_c1_h16_kv4 | torch_math_bf16 | bfloat16 | OK | 9.21498 | 1.09466 | 9.2345 | 756.008 | 0.00332431 | 0.000457555 | — |
| e1_prefill_l512_s8192_d128_c1_h16_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 34.7788 | 1.68029 | 34.7907 | 2 | 0.0656006 | 0.00647223 | — |
| e1_prefill_l512_s8192_d128_c1_h16_kv4 | apa_sp_bf16 | bfloat16 | OK | 28.8399 | 6.42819 | 28.9223 | 2 | 0.05349 | 0.00711266 | 0.15130 / 0.11626; UNMATCHED/UNCALIBRATED |
| e1_prefill_l512_s8192_d128_c1_h16_kv4 | torch_efficient_bf16 | bfloat16 | OK | 1.3271 | 0.135104 | 1.36861 | 66 | 0.00366178 | 0.000444822 | — |
| e1_prefill_l512_s8192_d128_c1_h16_kv4 | torch_flash_bf16 | bfloat16 | OK | 0.88576 | 0.121856 | 0.896542 | 2.03223 | 0.00366193 | 0.000457555 | — |
| e1_prefill_l512_s8192_d128_c1_h16_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| e1_prefill_l512_s8192_d128_c1_h16_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |
| e1_prefill_l512_s32768_d128_c1_h16_kv4 | engine_dense_fp32 | float32 | OK | 18.8467 | 0.718176 | 54.7305 | 2052 | 0 | 0 | — |
| e1_prefill_l512_s32768_d128_c1_h16_kv4 | torch_math_fp32 | float32 | OK | 34.5907 | 1.34336 | 34.646 | 2900.01 | 1.14441e-06 | 7.63685e-08 | — |
| e1_prefill_l512_s32768_d128_c1_h16_kv4 | apa_two_pass_fp32 | float32 | OK | 167.526 | 1.77664 | 167.56 | 4 | 0.0672655 | 0.00312635 | — |
| e1_prefill_l512_s32768_d128_c1_h16_kv4 | apa_sp_fp32 | float32 | OK | 181.343 | 6.97456 | 181.395 | 4 | 0.0549117 | 0.00381828 | 0.14776 / 0.11628; UNMATCHED/UNCALIBRATED |
| e1_prefill_l512_s32768_d128_c1_h16_kv4 | engine_dense_bf16 | bfloat16 | OK | 7.29088 | 0.541696 | 22.7782 | 1026 | 0.0053202 | 0.000420569 | — |
| e1_prefill_l512_s32768_d128_c1_h16_kv4 | torch_math_bf16 | bfloat16 | OK | 36.5793 | 1.00659 | 36.6343 | 3000.01 | 0.00339605 | 0.000207296 | — |
| e1_prefill_l512_s32768_d128_c1_h16_kv4 | apa_two_pass_bf16 | bfloat16 | OK | 149.133 | 2.26509 | 149.182 | 2 | 0.067296 | 0.00298757 | — |
| e1_prefill_l512_s32768_d128_c1_h16_kv4 | apa_sp_bf16 | bfloat16 | OK | 141.372 | 7.96774 | 141.446 | 2 | 0.0550298 | 0.00362605 | 0.14742 / 0.11625; UNMATCHED/UNCALIBRATED |
| e1_prefill_l512_s32768_d128_c1_h16_kv4 | torch_efficient_bf16 | bfloat16 | OK | 4.66534 | 0.441312 | 4.70441 | 258 | 0.00375678 | 0.000213504 | — |
| e1_prefill_l512_s32768_d128_c1_h16_kv4 | torch_flash_bf16 | bfloat16 | OK | 2.69722 | 0.118848 | 2.71046 | 2.03223 | 0.00376036 | 0.0002093 | — |
| e1_prefill_l512_s32768_d128_c1_h16_kv4 | flash_attn_bf16 | bfloat16 | UNAVAILABLE | — | — | — | — | — | — | — |
| e1_prefill_l512_s32768_d128_c1_h16_kv4 | apa_sp2_fp32 | float32 | UNAVAILABLE | — | — | — | — | — | — | — |

## Reading the measurements

- engine_dense_fp32, prefill: 0 faster / 0 slower median cells versus same-dtype engine dense; dense/contender speed ratio 1–1× over 24 cells; 24 overlapping IQRs (timing separation uncertain).
- engine_dense_fp32, decode: 0 faster / 0 slower median cells versus same-dtype engine dense; dense/contender speed ratio 1–1× over 24 cells; 24 overlapping IQRs (timing separation uncertain).
- torch_math_fp32, prefill: 8 faster / 16 slower median cells versus same-dtype engine dense; dense/contender speed ratio 0.378–1.4× over 24 cells; 1 overlapping IQRs (timing separation uncertain).
- torch_math_fp32, decode: 0 faster / 24 slower median cells versus same-dtype engine dense; dense/contender speed ratio 0.238–0.833× over 24 cells; 0 overlapping IQRs (timing separation uncertain).
- apa_two_pass_fp32, prefill: 0 faster / 24 slower median cells versus same-dtype engine dense; dense/contender speed ratio 0.0987–0.36× over 24 cells; 0 overlapping IQRs (timing separation uncertain).
- apa_two_pass_fp32, decode: 0 faster / 24 slower median cells versus same-dtype engine dense; dense/contender speed ratio 0.0991–0.367× over 24 cells; 0 overlapping IQRs (timing separation uncertain).
- apa_sp_fp32, prefill: 0 faster / 24 slower median cells versus same-dtype engine dense; dense/contender speed ratio 0.104–0.468× over 24 cells; 0 overlapping IQRs (timing separation uncertain).
- apa_sp_fp32, decode: 2 faster / 22 slower median cells versus same-dtype engine dense; dense/contender speed ratio 0.108–1.16× over 24 cells; 0 overlapping IQRs (timing separation uncertain).
- engine_dense_bf16, prefill: 0 faster / 0 slower median cells versus same-dtype engine dense; dense/contender speed ratio 1–1× over 24 cells; 24 overlapping IQRs (timing separation uncertain).
- engine_dense_bf16, decode: 0 faster / 0 slower median cells versus same-dtype engine dense; dense/contender speed ratio 1–1× over 24 cells; 24 overlapping IQRs (timing separation uncertain).
- torch_math_bf16, prefill: 0 faster / 24 slower median cells versus same-dtype engine dense; dense/contender speed ratio 0.164–0.769× over 24 cells; 0 overlapping IQRs (timing separation uncertain).
- torch_math_bf16, decode: 0 faster / 24 slower median cells versus same-dtype engine dense; dense/contender speed ratio 0.121–0.515× over 24 cells; 0 overlapping IQRs (timing separation uncertain).
- apa_two_pass_bf16, prefill: 0 faster / 24 slower median cells versus same-dtype engine dense; dense/contender speed ratio 0.0489–0.243× over 24 cells; 0 overlapping IQRs (timing separation uncertain).
- apa_two_pass_bf16, decode: 0 faster / 24 slower median cells versus same-dtype engine dense; dense/contender speed ratio 0.0587–0.32× over 24 cells; 0 overlapping IQRs (timing separation uncertain).
- apa_sp_bf16, prefill: 0 faster / 24 slower median cells versus same-dtype engine dense; dense/contender speed ratio 0.0516–0.352× over 24 cells; 0 overlapping IQRs (timing separation uncertain).
- apa_sp_bf16, decode: 0 faster / 24 slower median cells versus same-dtype engine dense; dense/contender speed ratio 0.109–0.692× over 24 cells; 0 overlapping IQRs (timing separation uncertain).
- torch_efficient_bf16, prefill: 24 faster / 0 slower median cells versus same-dtype engine dense; dense/contender speed ratio 1.07–12.5× over 24 cells; 1 overlapping IQRs (timing separation uncertain).
- torch_efficient_bf16, decode: 0 faster / 24 slower median cells versus same-dtype engine dense; dense/contender speed ratio 0.121–0.626× over 24 cells; 0 overlapping IQRs (timing separation uncertain).
- torch_flash_bf16, prefill: 23 faster / 1 slower median cells versus same-dtype engine dense; dense/contender speed ratio 0.959–15.4× over 24 cells; 2 overlapping IQRs (timing separation uncertain).
- torch_flash_bf16, decode: 13 faster / 11 slower median cells versus same-dtype engine dense; dense/contender speed ratio 0.614–2.53× over 24 cells; 4 overlapping IQRs (timing separation uncertain).

## Registered predictions

| Prediction | Verdict | Evaluated / expected | Hits |
|---|---|---:|---:|
| P1_fp32_parity | BLOCKED | 48 / 50 | 48 |
| A1_fp32_parity | BLOCKED | 48 / 50 | 48 |
| P1_torch_flash_bf16_ge2x_vs_dense_fp32_mixed_dtype | BLOCKED | 16 / 18 | 16 |
| P1_torch_efficient_bf16_ge2x_vs_dense_fp32_mixed_dtype | BLOCKED | 16 / 18 | 16 |
| P2_two_pass_slower | BLOCKED | 39 / 41 | 39 |
| P3_half_gap_prefill_s8192_matched_only | BLOCKED | 6 / 9 | 2 |
| P3_decode_s32768_within2x | HIT | 8 / 8 | 8 |
| P3_SP_never_beats_flash_mixed_dtype | BLOCKED | 48 / 50 | 48 |
| A2_flash_faster_prefill_mixed_dtype | BLOCKED | 16 / 18 | 16 |
| P4_apa_two_pass_fp32_same_dtype | BLOCKED | 7 / 9 | 7 |
| A4_apa_two_pass_fp32 | BLOCKED | 6 / 8 | 6 |
| P4_apa_sp_fp32_same_dtype | BLOCKED | 7 / 9 | 7 |
| A4_apa_sp_fp32 | BLOCKED | 6 / 8 | 6 |
| P4_apa_two_pass_bf16_same_dtype | BLOCKED | 7 / 9 | 7 |
| P4_apa_sp_bf16_same_dtype | BLOCKED | 7 / 9 | 7 |
| A3_at_least_one_decode_fraction_mismatch | HIT | 24 / 24 | 21 |
| A5_paper_shape_mismatch | HIT | — / — | — |

P1/P3 flash speed comparisons above are explicitly FP32 engine/APA versus BF16 flash. Same-dtype ratios appear in the reading. P3 half-gap requires a positive two-pass latency gap and estimated matched budget; uncalibrated E1 transfers cannot satisfy that premise. Missing or unsupported rows block universal predictions, never count as wins. Per-cell prediction failures and blocked IDs are in speed_chain.json.

## Reconciliation with earlier receipts

The two E1 extras match B=1, H=16, KV=4, D=128, causal L=512 at S=8192/32768, with BF16 rows and r=0.10 symmetric INT4 bulk reconstruction. E1 RESULTS.md measured one warmup and three synchronized wall calls; SPD1 uses three warmups and nine CUDA-event calls plus separate wall medians. Seeds and rounding preparation differ, so timing differences require those qualifications. SP1’s 48 geometries match, but its bulk K+0.1-noise data and FP32-only timing are not the TurboQuant4 data here; its delta values are retained without assuming budget equality. The paper §4.5 used B=2,H=4,D=64,r=0.15 and an older kernel generation: its 2048-token 27.77 ms SDPA / 13.03 ms APA (2.1×) result has no identical SPD1 shape and cannot be chained as a flash speedup. Existing receipts are context, never filled into missing SPD1 rows.

## Availability, errors, and receipts

- decode_s2048_d128_c0_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/decode_s2048_d128_c0_h4_kv4.1788703540982125295.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- decode_s2048_d128_c0_h8_kv2: COMPLETE; `artifacts/apa_spd1/gpu/decode_s2048_d128_c0_h8_kv2.1788703577047880630.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- decode_s2048_d128_c1_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/decode_s2048_d128_c1_h4_kv4.1788703613029224400.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- decode_s2048_d128_c1_h8_kv2: COMPLETE; `artifacts/apa_spd1/gpu/decode_s2048_d128_c1_h8_kv2.1788703649180003049.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- decode_s2048_d64_c0_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/decode_s2048_d64_c0_h4_kv4.1788703397051029906.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- decode_s2048_d64_c0_h8_kv2: COMPLETE; `artifacts/apa_spd1/gpu/decode_s2048_d64_c0_h8_kv2.1788703433037108268.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- decode_s2048_d64_c1_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/decode_s2048_d64_c1_h4_kv4.1788703469028212822.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- decode_s2048_d64_c1_h8_kv2: COMPLETE; `artifacts/apa_spd1/gpu/decode_s2048_d64_c1_h8_kv2.1788703505002472691.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- decode_s32768_d128_c0_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/decode_s32768_d128_c0_h4_kv4.1788704125646026082.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- decode_s32768_d128_c0_h8_kv2: COMPLETE; `artifacts/apa_spd1/gpu/decode_s32768_d128_c0_h8_kv2.1788704162970351607.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- decode_s32768_d128_c1_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/decode_s32768_d128_c1_h4_kv4.1788704199750750114.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- decode_s32768_d128_c1_h8_kv2: COMPLETE; `artifacts/apa_spd1/gpu/decode_s32768_d128_c1_h8_kv2.1788704243877951941.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- decode_s32768_d64_c0_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/decode_s32768_d64_c0_h4_kv4.1788703979878658597.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- decode_s32768_d64_c0_h8_kv2: COMPLETE; `artifacts/apa_spd1/gpu/decode_s32768_d64_c0_h8_kv2.1788704016431407267.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- decode_s32768_d64_c1_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/decode_s32768_d64_c1_h4_kv4.1788704052742524078.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- decode_s32768_d64_c1_h8_kv2: COMPLETE; `artifacts/apa_spd1/gpu/decode_s32768_d64_c1_h8_kv2.1788704089399027698.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- decode_s8192_d128_c0_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/decode_s8192_d128_c0_h4_kv4.1788703835171025056.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- decode_s8192_d128_c0_h8_kv2: COMPLETE; `artifacts/apa_spd1/gpu/decode_s8192_d128_c0_h8_kv2.1788703871395771839.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- decode_s8192_d128_c1_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/decode_s8192_d128_c1_h4_kv4.1788703907539944205.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- decode_s8192_d128_c1_h8_kv2: COMPLETE; `artifacts/apa_spd1/gpu/decode_s8192_d128_c1_h8_kv2.1788703943715431129.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- decode_s8192_d64_c0_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/decode_s8192_d64_c0_h4_kv4.1788703685163468797.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- decode_s8192_d64_c0_h8_kv2: COMPLETE; `artifacts/apa_spd1/gpu/decode_s8192_d64_c0_h8_kv2.1788703721297628412.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- decode_s8192_d64_c1_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/decode_s8192_d64_c1_h4_kv4.1788703757356417434.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- decode_s8192_d64_c1_h8_kv2: COMPLETE; `artifacts/apa_spd1/gpu/decode_s8192_d64_c1_h8_kv2.1788703799163350341.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- e1_prefill_l512_s32768_d128_c1_h16_kv4: COMPLETE; `artifacts/apa_spd1/gpu/e1_prefill_l512_s32768_d128_c1_h16_kv4.1788704319859182494.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- e1_prefill_l512_s8192_d128_c1_h16_kv4: COMPLETE; `artifacts/apa_spd1/gpu/e1_prefill_l512_s8192_d128_c1_h16_kv4.1788704280707423819.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s2048_d128_c0_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/prefill_s2048_d128_c0_h4_kv4.1788702683379143519.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s2048_d128_c0_h8_kv2: COMPLETE; `artifacts/apa_spd1/gpu/prefill_s2048_d128_c0_h8_kv2.1788702720095072243.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s2048_d128_c1_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/prefill_s2048_d128_c1_h4_kv4.1788702757515054523.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s2048_d128_c1_h8_kv2: COMPLETE; `artifacts/apa_spd1/gpu/prefill_s2048_d128_c1_h8_kv2.1788702793932659983.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s2048_d64_c0_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/prefill_s2048_d64_c0_h4_kv4.1788702452869243125.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s2048_d64_c0_h8_kv2: COMPLETE; `artifacts/apa_spd1/gpu/prefill_s2048_d64_c0_h8_kv2.1788702489356672096.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s2048_d64_c1_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/prefill_s2048_d64_c1_h4_kv4.1788702526426107948.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s2048_d64_c1_h8_kv2: COMPLETE; `artifacts/apa_spd1/gpu/prefill_s2048_d64_c1_h8_kv2.1788702562697090724.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s512_d128_c0_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/prefill_s512_d128_c0_h4_kv4.1788702308917320736.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s512_d128_c0_h8_kv2: COMPLETE; `artifacts/apa_spd1/gpu/prefill_s512_d128_c0_h8_kv2.1788702344875094637.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s512_d128_c1_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/prefill_s512_d128_c1_h4_kv4.1788702380875935830.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s512_d128_c1_h8_kv2: COMPLETE; `artifacts/apa_spd1/gpu/prefill_s512_d128_c1_h8_kv2.1788702416864442387.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s512_d64_c0_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/prefill_s512_d64_c0_h4_kv4.1788702165017889371.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s512_d64_c0_h8_kv2: COMPLETE; `artifacts/apa_spd1/gpu/prefill_s512_d64_c0_h8_kv2.1788702201042199824.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s512_d64_c1_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/prefill_s512_d64_c1_h4_kv4.1788702237015542480.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s512_d64_c1_h8_kv2: COMPLETE; `artifacts/apa_spd1/gpu/prefill_s512_d64_c1_h8_kv2.1788702272970371777.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s8192_d128_c0_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/prefill_s8192_d128_c0_h4_kv4.1788703006394721141.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s8192_d128_c0_h8_kv2: ERROR; `artifacts/apa_spd1/gpu/prefill_s8192_d128_c0_h8_kv2.1788704480532393638.receipt.json`
```text
Traceback (most recent call last):
  File "/mnt/ForgeRealm/Project-Tensor-wt-apa-spd1/scripts/apa_spd1_bench.py", line 305, in measure_cell
    ms, wall = telemetry.timed(funcs[name])
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/ForgeRealm/Project-Tensor-wt-apa-spd1/scripts/apa_spd1_bench.py", line 79, in timed
    out = call()
          ^^^^^^
  File "/mnt/ForgeRealm/Project-Tensor-wt-apa-spd1/scripts/apa_spd1_bench.py", line 223, in <lambda>
    funcs[name] = lambda q=q, k=k, v=v: grouped_dense(tc, q, k, v, shape)
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/ForgeRealm/Project-Tensor-wt-apa-spd1/scripts/apa_spd1_common.py", line 176, in grouped_dense
    weights = tc.causal_softmax(scores) if shape['causal'] else scores.softmax(-1)
                                                                ^^^^^^^^^^^^^^^^^^
RuntimeError: cudaMallocAsync failed: out of memory
```
  engine_dense_fp32: BLOCKED_AFTER_ERROR: RuntimeError: cudaMallocAsync failed: out of memory
  torch_math_fp32: BLOCKED_AFTER_ERROR: RuntimeError: cudaMallocAsync failed: out of memory
  apa_two_pass_fp32: BLOCKED_AFTER_ERROR: RuntimeError: cudaMallocAsync failed: out of memory
  apa_sp_fp32: BLOCKED_AFTER_ERROR: RuntimeError: cudaMallocAsync failed: out of memory
  engine_dense_bf16: BLOCKED_AFTER_ERROR: RuntimeError: cudaMallocAsync failed: out of memory
  torch_math_bf16: BLOCKED_AFTER_ERROR: RuntimeError: cudaMallocAsync failed: out of memory
  apa_two_pass_bf16: BLOCKED_AFTER_ERROR: RuntimeError: cudaMallocAsync failed: out of memory
  apa_sp_bf16: BLOCKED_AFTER_ERROR: RuntimeError: cudaMallocAsync failed: out of memory
  torch_efficient_bf16: BLOCKED_AFTER_ERROR: RuntimeError: cudaMallocAsync failed: out of memory
  torch_flash_bf16: BLOCKED_AFTER_ERROR: RuntimeError: cudaMallocAsync failed: out of memory
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s8192_d128_c1_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/prefill_s8192_d128_c1_h4_kv4.1788703098300436909.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s8192_d128_c1_h8_kv2: COMPLETE; `artifacts/apa_spd1/gpu/prefill_s8192_d128_c1_h8_kv2.1788703346481263265.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s8192_d64_c0_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/prefill_s8192_d64_c0_h4_kv4.1788702830846568103.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s8192_d64_c0_h8_kv2: ERROR; `artifacts/apa_spd1/gpu/prefill_s8192_d64_c0_h8_kv2.1788704439680269366.receipt.json`
```text
Traceback (most recent call last):
  File "/mnt/ForgeRealm/Project-Tensor-wt-apa-spd1/scripts/apa_spd1_bench.py", line 305, in measure_cell
    ms, wall = telemetry.timed(funcs[name])
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/ForgeRealm/Project-Tensor-wt-apa-spd1/scripts/apa_spd1_bench.py", line 79, in timed
    out = call()
          ^^^^^^
  File "/mnt/ForgeRealm/Project-Tensor-wt-apa-spd1/scripts/apa_spd1_bench.py", line 223, in <lambda>
    funcs[name] = lambda q=q, k=k, v=v: grouped_dense(tc, q, k, v, shape)
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/ForgeRealm/Project-Tensor-wt-apa-spd1/scripts/apa_spd1_common.py", line 176, in grouped_dense
    weights = tc.causal_softmax(scores) if shape['causal'] else scores.softmax(-1)
                                                                ^^^^^^^^^^^^^^^^^^
RuntimeError: cudaMallocAsync failed: out of memory
```
  engine_dense_fp32: BLOCKED_AFTER_ERROR: RuntimeError: cudaMallocAsync failed: out of memory
  torch_math_fp32: BLOCKED_AFTER_ERROR: RuntimeError: cudaMallocAsync failed: out of memory
  apa_two_pass_fp32: BLOCKED_AFTER_ERROR: RuntimeError: cudaMallocAsync failed: out of memory
  apa_sp_fp32: BLOCKED_AFTER_ERROR: RuntimeError: cudaMallocAsync failed: out of memory
  engine_dense_bf16: BLOCKED_AFTER_ERROR: RuntimeError: cudaMallocAsync failed: out of memory
  torch_math_bf16: BLOCKED_AFTER_ERROR: RuntimeError: cudaMallocAsync failed: out of memory
  apa_two_pass_bf16: BLOCKED_AFTER_ERROR: RuntimeError: cudaMallocAsync failed: out of memory
  apa_sp_bf16: BLOCKED_AFTER_ERROR: RuntimeError: cudaMallocAsync failed: out of memory
  torch_efficient_bf16: BLOCKED_AFTER_ERROR: RuntimeError: cudaMallocAsync failed: out of memory
  torch_flash_bf16: BLOCKED_AFTER_ERROR: RuntimeError: cudaMallocAsync failed: out of memory
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s8192_d64_c1_h4_kv4: COMPLETE; `artifacts/apa_spd1/gpu/prefill_s8192_d64_c1_h4_kv4.1788702917304935080.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime
- prefill_s8192_d64_c1_h8_kv2: COMPLETE; `artifacts/apa_spd1/gpu/prefill_s8192_d64_c1_h8_kv2.1788702958993364081.receipt.json`
  flash_attn_bf16: UNAVAILABLE: No module named 'flash_attn'
  apa_sp2_fp32: UNAVAILABLE: SP2 frozen e_q table absent at runtime

## Prior art

Reused: Vaswani et al. (2017) dense attention; [PyTorch SDPA](https://docs.pytorch.org/docs/2.11/generated/torch.nn.attention.sdpa_kernel.html) (contributors, 2023–2026); [FlashAttention-2](https://arxiv.org/abs/2307.08691) (Dao, 2023); [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., 2025; repository MSE reconstruction); APA/E1/SP1/SP1.1 (David and Project-Tensor seats, 2026); [BLASST](https://arxiv.org/abs/2512.12087) (Yuan et al., 2025/2026) running-max prior art; [online normalizer](https://arxiv.org/abs/1805.02867) (Milakov and Gimelshein, 2018); Flash-Decoding (Dao et al., 2023). [ThriftAttention](https://arxiv.org/abs/2605.23081) (Sharratt, 2026) is related selective mixed precision, not a contender. No new attention algorithm is introduced. Full attribution and verification limits: PRIOR_ART.md.

## Process exits

- decode_s2048_d128_c0_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/decode_s2048_d128_c0_h4_kv4.1788703540982125295.exit.json`
- decode_s2048_d128_c0_h8_kv2: exit 0; `artifacts/apa_spd1/gpu/decode_s2048_d128_c0_h8_kv2.1788703577047880630.exit.json`
- decode_s2048_d128_c1_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/decode_s2048_d128_c1_h4_kv4.1788703613029224400.exit.json`
- decode_s2048_d128_c1_h8_kv2: exit 0; `artifacts/apa_spd1/gpu/decode_s2048_d128_c1_h8_kv2.1788703649180003049.exit.json`
- decode_s2048_d64_c0_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/decode_s2048_d64_c0_h4_kv4.1788703397051029906.exit.json`
- decode_s2048_d64_c0_h8_kv2: exit 0; `artifacts/apa_spd1/gpu/decode_s2048_d64_c0_h8_kv2.1788703433037108268.exit.json`
- decode_s2048_d64_c1_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/decode_s2048_d64_c1_h4_kv4.1788703469028212822.exit.json`
- decode_s2048_d64_c1_h8_kv2: exit 0; `artifacts/apa_spd1/gpu/decode_s2048_d64_c1_h8_kv2.1788703505002472691.exit.json`
- decode_s32768_d128_c0_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/decode_s32768_d128_c0_h4_kv4.1788704125646026082.exit.json`
- decode_s32768_d128_c0_h8_kv2: exit 0; `artifacts/apa_spd1/gpu/decode_s32768_d128_c0_h8_kv2.1788704162970351607.exit.json`
- decode_s32768_d128_c1_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/decode_s32768_d128_c1_h4_kv4.1788704199750750114.exit.json`
- decode_s32768_d128_c1_h8_kv2: exit 0; `artifacts/apa_spd1/gpu/decode_s32768_d128_c1_h8_kv2.1788704243877951941.exit.json`
- decode_s32768_d64_c0_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/decode_s32768_d64_c0_h4_kv4.1788703979878658597.exit.json`
- decode_s32768_d64_c0_h8_kv2: exit 0; `artifacts/apa_spd1/gpu/decode_s32768_d64_c0_h8_kv2.1788704016431407267.exit.json`
- decode_s32768_d64_c1_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/decode_s32768_d64_c1_h4_kv4.1788704052742524078.exit.json`
- decode_s32768_d64_c1_h8_kv2: exit 0; `artifacts/apa_spd1/gpu/decode_s32768_d64_c1_h8_kv2.1788704089399027698.exit.json`
- decode_s8192_d128_c0_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/decode_s8192_d128_c0_h4_kv4.1788703835171025056.exit.json`
- decode_s8192_d128_c0_h8_kv2: exit 0; `artifacts/apa_spd1/gpu/decode_s8192_d128_c0_h8_kv2.1788703871395771839.exit.json`
- decode_s8192_d128_c1_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/decode_s8192_d128_c1_h4_kv4.1788703907539944205.exit.json`
- decode_s8192_d128_c1_h8_kv2: exit 0; `artifacts/apa_spd1/gpu/decode_s8192_d128_c1_h8_kv2.1788703943715431129.exit.json`
- decode_s8192_d64_c0_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/decode_s8192_d64_c0_h4_kv4.1788703685163468797.exit.json`
- decode_s8192_d64_c0_h8_kv2: exit 0; `artifacts/apa_spd1/gpu/decode_s8192_d64_c0_h8_kv2.1788703721297628412.exit.json`
- decode_s8192_d64_c1_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/decode_s8192_d64_c1_h4_kv4.1788703757356417434.exit.json`
- decode_s8192_d64_c1_h8_kv2: exit 0; `artifacts/apa_spd1/gpu/decode_s8192_d64_c1_h8_kv2.1788703799163350341.exit.json`
- e1_prefill_l512_s32768_d128_c1_h16_kv4: exit 0; `artifacts/apa_spd1/gpu/e1_prefill_l512_s32768_d128_c1_h16_kv4.1788704319859182494.exit.json`
- e1_prefill_l512_s8192_d128_c1_h16_kv4: exit 0; `artifacts/apa_spd1/gpu/e1_prefill_l512_s8192_d128_c1_h16_kv4.1788704280707423819.exit.json`
- prefill_s2048_d128_c0_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/prefill_s2048_d128_c0_h4_kv4.1788702683379143519.exit.json`
- prefill_s2048_d128_c0_h8_kv2: exit 0; `artifacts/apa_spd1/gpu/prefill_s2048_d128_c0_h8_kv2.1788702720095072243.exit.json`
- prefill_s2048_d128_c1_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/prefill_s2048_d128_c1_h4_kv4.1788702757515054523.exit.json`
- prefill_s2048_d128_c1_h8_kv2: exit 0; `artifacts/apa_spd1/gpu/prefill_s2048_d128_c1_h8_kv2.1788702793932659983.exit.json`
- prefill_s2048_d64_c0_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/prefill_s2048_d64_c0_h4_kv4.1788702452869243125.exit.json`
- prefill_s2048_d64_c0_h8_kv2: exit 0; `artifacts/apa_spd1/gpu/prefill_s2048_d64_c0_h8_kv2.1788702489356672096.exit.json`
- prefill_s2048_d64_c1_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/prefill_s2048_d64_c1_h4_kv4.1788702526426107948.exit.json`
- prefill_s2048_d64_c1_h8_kv2: exit 0; `artifacts/apa_spd1/gpu/prefill_s2048_d64_c1_h8_kv2.1788702562697090724.exit.json`
- prefill_s512_d128_c0_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/prefill_s512_d128_c0_h4_kv4.1788702308917320736.exit.json`
- prefill_s512_d128_c0_h8_kv2: exit 0; `artifacts/apa_spd1/gpu/prefill_s512_d128_c0_h8_kv2.1788702344875094637.exit.json`
- prefill_s512_d128_c1_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/prefill_s512_d128_c1_h4_kv4.1788702380875935830.exit.json`
- prefill_s512_d128_c1_h8_kv2: exit 0; `artifacts/apa_spd1/gpu/prefill_s512_d128_c1_h8_kv2.1788702416864442387.exit.json`
- prefill_s512_d64_c0_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/prefill_s512_d64_c0_h4_kv4.1788702165017889371.exit.json`
- prefill_s512_d64_c0_h8_kv2: exit 0; `artifacts/apa_spd1/gpu/prefill_s512_d64_c0_h8_kv2.1788702201042199824.exit.json`
- prefill_s512_d64_c1_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/prefill_s512_d64_c1_h4_kv4.1788702237015542480.exit.json`
- prefill_s512_d64_c1_h8_kv2: exit 0; `artifacts/apa_spd1/gpu/prefill_s512_d64_c1_h8_kv2.1788702272970371777.exit.json`
- prefill_s8192_d128_c0_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/prefill_s8192_d128_c0_h4_kv4.1788703006394721141.exit.json`
- prefill_s8192_d128_c0_h8_kv2: exit 1; `artifacts/apa_spd1/gpu/prefill_s8192_d128_c0_h8_kv2.1788703055236326479.exit.json`
- prefill_s8192_d128_c0_h8_kv2: exit 1; `artifacts/apa_spd1/gpu/prefill_s8192_d128_c0_h8_kv2.1788704480532393638.exit.json`
- prefill_s8192_d128_c1_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/prefill_s8192_d128_c1_h4_kv4.1788703098300436909.exit.json`
- prefill_s8192_d128_c1_h8_kv2: exit 0; `artifacts/apa_spd1/gpu/prefill_s8192_d128_c1_h8_kv2.1788703346481263265.exit.json`
- prefill_s8192_d64_c0_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/prefill_s8192_d64_c0_h4_kv4.1788702830846568103.exit.json`
- prefill_s8192_d64_c0_h8_kv2: exit 1; `artifacts/apa_spd1/gpu/prefill_s8192_d64_c0_h8_kv2.1788702876608658574.exit.json`
- prefill_s8192_d64_c0_h8_kv2: exit 1; `artifacts/apa_spd1/gpu/prefill_s8192_d64_c0_h8_kv2.1788704439680269366.exit.json`
- prefill_s8192_d64_c1_h4_kv4: exit 0; `artifacts/apa_spd1/gpu/prefill_s8192_d64_c1_h4_kv4.1788702917304935080.exit.json`
- prefill_s8192_d64_c1_h8_kv2: exit 0; `artifacts/apa_spd1/gpu/prefill_s8192_d64_c1_h8_kv2.1788702958993364081.exit.json`
