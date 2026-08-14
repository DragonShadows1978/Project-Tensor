# APAMQ-E1 MQA-Geometry Kernel Transient Sweep

Evidence class: **KERNEL SWEEP**. Values are raw measurements; no H-A/T1 verdict is made here.

Each cell is `CUDA-pool peak delta MiB / warm median wall ms` (1 warm-up, 3 timed calls).

## D=128

### PREFILL (L=512, bottom-right causal)

| Path | KV heads | S=4096 | S=8192 | S=16384 | S=32768 | S=65536 |
|---|---:|---:|---:|---:|---:|---:|
| standard | 1 | 130.0 MiB / 1.042 ms | 258.0 MiB / 2.013 ms | 514.0 MiB / 9.051 ms | 1026.0 MiB / 22.990 ms | 2050.0 MiB / 53.382 ms |
| standard | 4 | 130.0 MiB / 1.122 ms | 258.0 MiB / 2.177 ms | 514.0 MiB / 9.306 ms | 1026.0 MiB / 23.415 ms | 2050.0 MiB / 55.198 ms |
| standard | 8 | 130.0 MiB / 1.139 ms | 258.0 MiB / 2.191 ms | 514.0 MiB / 9.262 ms | 1026.0 MiB / 23.444 ms | 2050.0 MiB / 54.608 ms |
| standard | 16 | 130.0 MiB / 1.138 ms | 258.0 MiB / 2.137 ms | 514.0 MiB / 9.237 ms | 1026.0 MiB / 23.328 ms | 2050.0 MiB / 54.847 ms |
| fused_apa | 1 | 2.0 MiB / 17.449 ms | 2.0 MiB / 35.778 ms | 2.0 MiB / 72.921 ms | 2.0 MiB / 148.857 ms | 2.0 MiB / 299.219 ms |
| fused_apa | 4 | 2.0 MiB / 17.475 ms | 2.0 MiB / 35.516 ms | 2.0 MiB / 74.261 ms | 2.0 MiB / 148.984 ms | 2.0 MiB / 302.359 ms |
| fused_apa | 8 | 2.0 MiB / 17.472 ms | 2.0 MiB / 35.832 ms | 2.0 MiB / 73.550 ms | 2.0 MiB / 149.711 ms | 2.0 MiB / 300.590 ms |
| fused_apa | 16 | 2.0 MiB / 17.507 ms | 2.0 MiB / 35.854 ms | 2.0 MiB / 72.893 ms | 2.0 MiB / 149.131 ms | 2.0 MiB / 298.090 ms |
| int4_apa | 1 | 2.3 MiB / 16.965 ms | 2.5 MiB / 34.782 ms | 3.1 MiB / 71.037 ms | 4.1 MiB / 144.277 ms | 6.2 MiB / 290.989 ms |
| int4_apa | 4 | 3.1 MiB / 16.975 ms | 4.1 MiB / 34.844 ms | 6.2 MiB / 71.231 ms | 10.5 MiB / 144.550 ms | 19.0 MiB / 294.873 ms |
| int4_apa | 8 | 4.1 MiB / 16.998 ms | 6.2 MiB / 34.872 ms | 10.5 MiB / 71.313 ms | 19.0 MiB / 146.081 ms | 36.0 MiB / 294.036 ms |
| int4_apa | 16 | 6.2 MiB / 16.974 ms | 10.5 MiB / 35.370 ms | 19.0 MiB / 71.848 ms | 36.0 MiB / 146.511 ms | 70.0 MiB / 295.145 ms |
| int4_bf16_mma_apa | 1 | 2.3 MiB / 29.230 ms | 2.6 MiB / 60.017 ms | 3.1 MiB / 121.279 ms | 4.2 MiB / 244.191 ms | 6.3 MiB / 488.031 ms |
| int4_bf16_mma_apa | 4 | 3.1 MiB / 29.186 ms | 4.2 MiB / 59.954 ms | 6.3 MiB / 121.325 ms | 10.5 MiB / 244.415 ms | 19.0 MiB / 492.371 ms |
| int4_bf16_mma_apa | 8 | 4.2 MiB / 29.328 ms | 6.3 MiB / 60.017 ms | 10.5 MiB / 121.383 ms | 19.0 MiB / 245.772 ms | 36.0 MiB / 493.867 ms |
| int4_bf16_mma_apa | 16 | 6.3 MiB / 29.168 ms | 10.5 MiB / 59.973 ms | 19.0 MiB / 122.501 ms | 36.0 MiB / 247.510 ms | 70.0 MiB / 496.273 ms |
| int8q_int4_apa | 1 | 3.3 MiB / 41.700 ms | 3.6 MiB / 85.354 ms | 4.1 MiB / 173.119 ms | 5.2 MiB / 340.219 ms | 7.3 MiB / 672.480 ms |
| int8q_int4_apa | 4 | 4.1 MiB / 41.588 ms | 5.2 MiB / 85.401 ms | 7.3 MiB / 173.107 ms | 11.6 MiB / 339.506 ms | 20.1 MiB / 674.552 ms |
| int8q_int4_apa | 8 | 5.2 MiB / 41.662 ms | 7.3 MiB / 85.540 ms | 11.6 MiB / 173.142 ms | 20.1 MiB / 338.566 ms | 37.1 MiB / 670.494 ms |
| int8q_int4_apa | 16 | 7.3 MiB / 41.688 ms | 11.6 MiB / 85.571 ms | 20.1 MiB / 173.384 ms | 37.1 MiB / 343.419 ms | 71.1 MiB / 669.382 ms |

### DECODE (L=1, bottom-right causal)

| Path | KV heads | S=4096 | S=8192 | S=16384 | S=32768 | S=65536 |
|---|---:|---:|---:|---:|---:|---:|
| standard | 1 | 0.3 MiB / 0.163 ms | 0.5 MiB / 0.158 ms | 1.0 MiB / 0.167 ms | 2.0 MiB / 0.191 ms | 4.0 MiB / 0.261 ms |
| standard | 4 | 0.3 MiB / 0.147 ms | 0.5 MiB / 0.226 ms | 1.0 MiB / 0.231 ms | 2.0 MiB / 0.423 ms | 4.0 MiB / 0.802 ms |
| standard | 8 | 0.3 MiB / 0.146 ms | 0.5 MiB / 0.168 ms | 1.0 MiB / 0.292 ms | 2.0 MiB / 0.518 ms | 4.0 MiB / 0.954 ms |
| standard | 16 | 0.3 MiB / 0.216 ms | 0.5 MiB / 0.255 ms | 1.0 MiB / 0.422 ms | 2.0 MiB / 0.799 ms | 4.0 MiB / 1.591 ms |
| fused_apa | 1 | 0.0 MiB / 0.620 ms | 0.0 MiB / 0.929 ms | 0.1 MiB / 1.535 ms | 0.1 MiB / 2.731 ms | 0.3 MiB / 5.288 ms |
| fused_apa | 4 | 0.0 MiB / 0.631 ms | 0.0 MiB / 0.929 ms | 0.1 MiB / 1.536 ms | 0.1 MiB / 3.259 ms | 0.3 MiB / 6.993 ms |
| fused_apa | 8 | 0.0 MiB / 0.621 ms | 0.0 MiB / 0.923 ms | 0.1 MiB / 1.878 ms | 0.1 MiB / 3.786 ms | 0.3 MiB / 7.536 ms |
| fused_apa | 16 | 0.0 MiB / 0.625 ms | 0.0 MiB / 1.202 ms | 0.1 MiB / 2.197 ms | 0.1 MiB / 4.141 ms | 0.3 MiB / 8.206 ms |
| int4_apa | 1 | 0.3 MiB / 0.773 ms | 0.6 MiB / 1.172 ms | 1.1 MiB / 1.978 ms | 2.3 MiB / 3.526 ms | 4.5 MiB / 6.886 ms |
| int4_apa | 4 | 1.1 MiB / 0.773 ms | 2.2 MiB / 1.177 ms | 4.3 MiB / 2.000 ms | 8.6 MiB / 3.647 ms | 17.3 MiB / 11.001 ms |
| int4_apa | 8 | 2.1 MiB / 0.774 ms | 4.3 MiB / 1.192 ms | 8.6 MiB / 2.100 ms | 17.1 MiB / 5.993 ms | 34.3 MiB / 11.776 ms |
| int4_apa | 16 | 4.3 MiB / 0.792 ms | 8.5 MiB / 1.318 ms | 17.1 MiB / 3.445 ms | 34.1 MiB / 6.672 ms | 68.3 MiB / 12.726 ms |
| int4_bf16_mma_apa | 1 | 0.3 MiB / 0.975 ms | 0.6 MiB / 1.388 ms | 1.1 MiB / 2.204 ms | 2.3 MiB / 3.867 ms | 4.5 MiB / 7.244 ms |
| int4_bf16_mma_apa | 4 | 1.1 MiB / 0.991 ms | 2.2 MiB / 1.393 ms | 4.3 MiB / 2.186 ms | 8.6 MiB / 3.901 ms | 17.3 MiB / 12.328 ms |
| int4_bf16_mma_apa | 8 | 2.1 MiB / 0.989 ms | 4.3 MiB / 1.423 ms | 8.6 MiB / 2.474 ms | 17.1 MiB / 6.455 ms | 34.3 MiB / 13.005 ms |
| int4_bf16_mma_apa | 16 | 4.3 MiB / 1.008 ms | 8.5 MiB / 1.719 ms | 17.1 MiB / 3.907 ms | 34.1 MiB / 7.434 ms | 68.3 MiB / 15.119 ms |
| int8q_int4_apa | 1 | 0.3 MiB / 0.561 ms | 0.6 MiB / 0.733 ms | 1.1 MiB / 1.080 ms | 2.3 MiB / 1.728 ms | 4.5 MiB / 3.268 ms |
| int8q_int4_apa | 4 | 1.1 MiB / 0.566 ms | 2.2 MiB / 0.748 ms | 4.3 MiB / 1.103 ms | 8.6 MiB / 1.836 ms | 17.3 MiB / 4.202 ms |
| int8q_int4_apa | 8 | 2.1 MiB / 0.569 ms | 4.3 MiB / 0.769 ms | 8.6 MiB / 1.303 ms | 17.1 MiB / 2.352 ms | 34.3 MiB / 5.051 ms |
| int8q_int4_apa | 16 | 4.3 MiB / 0.587 ms | 8.5 MiB / 0.992 ms | 17.1 MiB / 1.675 ms | 34.1 MiB / 3.176 ms | 68.3 MiB / 6.664 ms |

## D=512

### PREFILL (L=512, bottom-right causal)

| Path | KV heads | S=4096 | S=8192 | S=16384 | S=32768 | S=65536 |
|---|---:|---:|---:|---:|---:|---:|
| standard | 1 | 136.0 MiB / 1.723 ms | 264.0 MiB / 3.373 ms | 520.0 MiB / 11.734 ms | 1032.0 MiB / 28.464 ms | 2056.0 MiB / 64.493 ms |
| standard | 4 | 136.0 MiB / 1.757 ms | 264.0 MiB / 3.441 ms | 520.0 MiB / 11.947 ms | 1032.0 MiB / 28.753 ms | 2056.0 MiB / 68.132 ms |
| standard | 8 | 136.0 MiB / 1.768 ms | 264.0 MiB / 3.449 ms | 520.0 MiB / 11.944 ms | 1032.0 MiB / 28.853 ms | 2056.0 MiB / 68.660 ms |
| standard | 16 | 136.0 MiB / 1.770 ms | 264.0 MiB / 3.425 ms | 520.0 MiB / 11.967 ms | 1032.0 MiB / 29.002 ms | 2056.0 MiB / 67.945 ms |
| fused_apa | 1 | 8.0 MiB / 46.096 ms | 8.0 MiB / 98.483 ms | 8.0 MiB / 201.919 ms | 8.0 MiB / 479.629 ms | 8.0 MiB / 1035.853 ms |
| fused_apa | 4 | 8.0 MiB / 46.620 ms | 8.0 MiB / 99.068 ms | 8.0 MiB / 206.458 ms | 8.0 MiB / 486.752 ms | 8.0 MiB / 1041.338 ms |
| fused_apa | 8 | 8.0 MiB / 47.291 ms | 8.0 MiB / 97.602 ms | 8.0 MiB / 206.407 ms | 8.0 MiB / 496.289 ms | 8.0 MiB / 1044.203 ms |
| fused_apa | 16 | 8.0 MiB / 46.837 ms | 8.0 MiB / 97.763 ms | 8.0 MiB / 223.543 ms | 8.0 MiB / 512.426 ms | 8.0 MiB / 1058.393 ms |
| int4_apa | 1 | 9.0 MiB / 39.034 ms | 10.0 MiB / 85.823 ms | 12.1 MiB / 182.148 ms | 16.1 MiB / 361.034 ms | 24.2 MiB / 746.580 ms |
| int4_apa | 4 | 12.1 MiB / 40.231 ms | 16.1 MiB / 85.460 ms | 24.2 MiB / 180.191 ms | 40.5 MiB / 361.485 ms | 73.0 MiB / 751.071 ms |
| int4_apa | 8 | 16.1 MiB / 39.257 ms | 24.2 MiB / 86.264 ms | 40.5 MiB / 178.074 ms | 73.0 MiB / 362.805 ms | 138.0 MiB / 760.385 ms |
| int4_apa | 16 | 24.2 MiB / 39.616 ms | 40.5 MiB / 85.689 ms | 73.0 MiB / 177.613 ms | 138.0 MiB / 369.154 ms | 268.0 MiB / 765.939 ms |
| int4_bf16_mma_apa | 1 | 9.0 MiB / 83.486 ms | 10.1 MiB / 170.965 ms | 12.1 MiB / 345.834 ms | 16.2 MiB / 712.329 ms | 24.3 MiB / 1450.472 ms |
| int4_bf16_mma_apa | 4 | 12.1 MiB / 83.279 ms | 16.2 MiB / 171.487 ms | 24.3 MiB / 345.766 ms | 40.5 MiB / 714.629 ms | 73.0 MiB / 1441.644 ms |
| int4_bf16_mma_apa | 8 | 16.2 MiB / 83.515 ms | 24.3 MiB / 171.957 ms | 40.5 MiB / 346.432 ms | 73.0 MiB / 711.267 ms | 138.0 MiB / 1459.094 ms |
| int4_bf16_mma_apa | 16 | 24.3 MiB / 83.777 ms | 40.5 MiB / 172.019 ms | 73.0 MiB / 346.873 ms | 138.0 MiB / 710.574 ms | 268.0 MiB / 1453.140 ms |
| int8q_int4_apa | 1 | 13.1 MiB / 162.680 ms | 14.1 MiB / 328.082 ms | 16.1 MiB / 649.310 ms | 20.2 MiB / 1307.973 ms | 28.3 MiB / 2656.865 ms |
| int8q_int4_apa | 4 | 16.1 MiB / 162.771 ms | 20.2 MiB / 328.090 ms | 28.3 MiB / 652.393 ms | 44.6 MiB / 1309.622 ms | 77.1 MiB / 2659.111 ms |
| int8q_int4_apa | 8 | 20.2 MiB / 163.036 ms | 28.3 MiB / 328.097 ms | 44.6 MiB / 650.881 ms | 77.1 MiB / 1309.475 ms | 142.1 MiB / 2656.867 ms |
| int8q_int4_apa | 16 | 28.3 MiB / 163.057 ms | 44.6 MiB / 327.440 ms | 77.1 MiB / 653.763 ms | 142.1 MiB / 1309.725 ms | 272.1 MiB / 2657.352 ms |

### DECODE (L=1, bottom-right causal)

| Path | KV heads | S=4096 | S=8192 | S=16384 | S=32768 | S=65536 |
|---|---:|---:|---:|---:|---:|---:|
| standard | 1 | 0.3 MiB / 0.157 ms | 0.5 MiB / 0.161 ms | 1.0 MiB / 0.170 ms | 2.0 MiB / 0.307 ms | 4.0 MiB / 0.529 ms |
| standard | 4 | 0.3 MiB / 0.216 ms | 0.5 MiB / 0.260 ms | 1.0 MiB / 0.419 ms | 2.0 MiB / 0.756 ms | 4.0 MiB / 1.423 ms |
| standard | 8 | 0.3 MiB / 0.280 ms | 0.5 MiB / 0.448 ms | 1.0 MiB / 0.789 ms | 2.0 MiB / 1.498 ms | 4.0 MiB / 2.903 ms |
| standard | 16 | 0.3 MiB / 0.423 ms | 0.5 MiB / 0.744 ms | 1.0 MiB / 1.382 ms | 2.0 MiB / 2.634 ms | 4.0 MiB / 5.447 ms |
| fused_apa | 1 | 0.1 MiB / 1.053 ms | 0.1 MiB / 1.405 ms | 0.3 MiB / 2.090 ms | 0.5 MiB / 4.196 ms | 1.0 MiB / 9.959 ms |
| fused_apa | 4 | 0.1 MiB / 1.039 ms | 0.1 MiB / 1.879 ms | 0.3 MiB / 3.024 ms | 0.5 MiB / 4.803 ms | 1.0 MiB / 9.735 ms |
| fused_apa | 8 | 0.1 MiB / 1.474 ms | 0.1 MiB / 2.160 ms | 0.3 MiB / 3.167 ms | 0.5 MiB / 5.229 ms | 1.0 MiB / 11.145 ms |
| fused_apa | 16 | 0.1 MiB / 1.779 ms | 0.1 MiB / 2.313 ms | 0.3 MiB / 3.503 ms | 0.5 MiB / 6.548 ms | 1.0 MiB / 13.525 ms |
| int4_apa | 1 | 1.1 MiB / 1.036 ms | 2.2 MiB / 1.493 ms | 4.3 MiB / 2.411 ms | 8.7 MiB / 4.371 ms | 17.3 MiB / 10.833 ms |
| int4_apa | 4 | 4.2 MiB / 1.040 ms | 8.3 MiB / 1.728 ms | 16.5 MiB / 3.853 ms | 33.0 MiB / 7.019 ms | 66.0 MiB / 14.003 ms |
| int4_apa | 8 | 8.2 MiB / 1.292 ms | 16.4 MiB / 2.485 ms | 32.8 MiB / 4.278 ms | 65.5 MiB / 7.706 ms | 131.0 MiB / 15.602 ms |
| int4_apa | 16 | 16.3 MiB / 1.834 ms | 32.7 MiB / 2.971 ms | 65.3 MiB / 4.926 ms | 130.5 MiB / 9.244 ms | 261.0 MiB / 18.803 ms |
| int4_bf16_mma_apa | 1 | 1.1 MiB / 1.557 ms | 2.2 MiB / 2.005 ms | 4.3 MiB / 2.888 ms | 8.6 MiB / 5.106 ms | 17.3 MiB / 11.970 ms |
| int4_bf16_mma_apa | 4 | 4.1 MiB / 1.546 ms | 8.3 MiB / 2.400 ms | 16.5 MiB / 4.644 ms | 33.0 MiB / 8.001 ms | 66.0 MiB / 15.885 ms |
| int4_bf16_mma_apa | 8 | 8.2 MiB / 2.091 ms | 16.4 MiB / 3.410 ms | 32.8 MiB / 5.416 ms | 65.5 MiB / 9.604 ms | 131.0 MiB / 19.158 ms |
| int4_bf16_mma_apa | 16 | 16.3 MiB / 2.762 ms | 32.6 MiB / 4.016 ms | 65.3 MiB / 6.555 ms | 130.5 MiB / 12.475 ms | 261.0 MiB / 24.758 ms |
| int8q_int4_apa | 1 | 1.1 MiB / 1.612 ms | 2.2 MiB / 2.236 ms | 4.3 MiB / 3.456 ms | 8.7 MiB / 6.085 ms | 17.3 MiB / 13.334 ms |
| int8q_int4_apa | 4 | 4.1 MiB / 1.628 ms | 8.3 MiB / 2.476 ms | 16.5 MiB / 4.268 ms | 33.0 MiB / 7.099 ms | 66.0 MiB / 14.176 ms |
| int8q_int4_apa | 8 | 8.2 MiB / 2.041 ms | 16.4 MiB / 3.004 ms | 32.8 MiB / 4.843 ms | 65.5 MiB / 9.050 ms | 131.0 MiB / 17.906 ms |
| int8q_int4_apa | 16 | 16.3 MiB / 2.455 ms | 32.6 MiB / 3.676 ms | 65.3 MiB / 6.214 ms | 130.5 MiB / 12.354 ms | 261.0 MiB / 24.170 ms |

## Factual notes

- Inputs were configured as BF16. Fused APA was configured with a signed per-key-vector 4-bit bulk quantize/dequantize and `refine_percentile=0.10` (`z = NormalPPF(0.90)`). Bulk-key preparation was outside the measured attention call.
- STANDARD is configured to invoke `tensor_cuda.matmul(q_grouped, k, trans_b=True)`, `tensor_cuda.causal_softmax(scores)`, then `tensor_cuda.matmul(weights_grouped, v)`. Q and weights are grouped as `(B, kv_heads, (q_heads/kv_heads)*L, ...)`; K/V are not expanded.
- FUSED APA is configured to invoke `tensor_cuda.apa_selective_attention(q, k, kq, v, scale, zthr, True)` with native `(q_heads, kv_heads)` geometry.
- INT4 APA is configured to invoke `tensor_cuda.apa_selective_attention_int4(q, k, v, scale, zthr, True)` with native `(q_heads, kv_heads)` geometry. Its call-local pack workspace and pack launch are included in both wall and pool measurements; it has no persistent kq operand.
- INT4 BF16-MMA APA invokes `tensor_cuda.apa_selective_attention_int4_bf16_mma(...)`: F-A1 bulk/statistics/selection with selected exact dots reassociated through BF16 WMMA/fp32 accumulation.
- INT8Q INT4 APA invokes `tensor_cuda.apa_selective_attention_int8q_int4(...)`: per-query symmetric INT8 Q and per-key symmetric-7 INT4 K are packed inside the measured call; exact dp4a int32 sums feed fp32 threshold statistics and non-selected softmax scores.
- Pool values are call-local high-water deltas. NVML before/during/after absolute samples are retained per timed repetition in `results.json`.
- No cells OOMed, errored, or were skipped.
