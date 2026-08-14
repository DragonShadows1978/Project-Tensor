# APAMQ-E1 MQA-Geometry Kernel Transient Sweep

Evidence class: **KERNEL SWEEP**. Values are raw measurements; no H-A/T1 verdict is made here.

Each cell is `CUDA-pool peak delta MiB / warm median wall ms` (1 warm-up, 3 timed calls).

## D=128

### PREFILL (L=512, bottom-right causal)

| Path | KV heads | S=4096 | S=8192 | S=16384 | S=32768 | S=65536 |
|---|---:|---:|---:|---:|---:|---:|
| standard | 1 | 130.0 MiB / 1.036 ms | 258.0 MiB / 2.064 ms | 514.0 MiB / 8.992 ms | 1026.0 MiB / 22.879 ms | 2050.0 MiB / 53.665 ms |
| standard | 4 | 130.0 MiB / 1.094 ms | 258.0 MiB / 2.172 ms | 514.0 MiB / 10.124 ms | 1026.0 MiB / 23.534 ms | 2050.0 MiB / 52.585 ms |
| standard | 8 | 130.0 MiB / 1.157 ms | 258.0 MiB / 2.242 ms | 514.0 MiB / 9.649 ms | 1026.0 MiB / 23.919 ms | 2050.0 MiB / 55.056 ms |
| standard | 16 | 130.0 MiB / 1.203 ms | 258.0 MiB / 2.149 ms | 514.0 MiB / 9.270 ms | 1026.0 MiB / 23.396 ms | 2050.0 MiB / 55.115 ms |
| fused_apa | 1 | 2.0 MiB / 18.434 ms | 2.0 MiB / 35.648 ms | 2.0 MiB / 71.865 ms | 2.0 MiB / 147.157 ms | 2.0 MiB / 313.743 ms |
| fused_apa | 4 | 2.0 MiB / 17.141 ms | 2.0 MiB / 36.940 ms | 2.0 MiB / 73.610 ms | 2.0 MiB / 153.015 ms | 2.0 MiB / 305.451 ms |
| fused_apa | 8 | 2.0 MiB / 16.374 ms | 2.0 MiB / 36.397 ms | 2.0 MiB / 72.784 ms | 2.0 MiB / 149.087 ms | 2.0 MiB / 301.916 ms |
| fused_apa | 16 | 2.0 MiB / 17.190 ms | 2.0 MiB / 36.914 ms | 2.0 MiB / 70.148 ms | 2.0 MiB / 147.845 ms | 2.0 MiB / 297.197 ms |
| int4_apa | 1 | 2.3 MiB / 15.802 ms | 2.5 MiB / 35.257 ms | 3.1 MiB / 69.899 ms | 4.1 MiB / 143.057 ms | 6.2 MiB / 289.765 ms |
| int4_apa | 4 | 3.1 MiB / 16.021 ms | 4.1 MiB / 33.852 ms | 6.2 MiB / 71.022 ms | 10.5 MiB / 142.933 ms | 19.0 MiB / 293.039 ms |
| int4_apa | 8 | 4.1 MiB / 15.847 ms | 6.2 MiB / 34.778 ms | 10.5 MiB / 70.495 ms | 19.0 MiB / 142.884 ms | 36.0 MiB / 290.579 ms |
| int4_apa | 16 | 6.2 MiB / 15.836 ms | 10.5 MiB / 34.872 ms | 19.0 MiB / 70.140 ms | 36.0 MiB / 144.069 ms | 70.0 MiB / 293.011 ms |

### DECODE (L=1, bottom-right causal)

| Path | KV heads | S=4096 | S=8192 | S=16384 | S=32768 | S=65536 |
|---|---:|---:|---:|---:|---:|---:|
| standard | 1 | 0.3 MiB / 0.176 ms | 0.5 MiB / 0.161 ms | 1.0 MiB / 0.171 ms | 2.0 MiB / 0.186 ms | 4.0 MiB / 0.246 ms |
| standard | 4 | 0.3 MiB / 0.142 ms | 0.5 MiB / 0.170 ms | 1.0 MiB / 0.234 ms | 2.0 MiB / 0.423 ms | 4.0 MiB / 0.746 ms |
| standard | 8 | 0.3 MiB / 0.149 ms | 0.5 MiB / 0.175 ms | 1.0 MiB / 0.293 ms | 2.0 MiB / 0.523 ms | 4.0 MiB / 0.916 ms |
| standard | 16 | 0.3 MiB / 0.158 ms | 0.5 MiB / 0.344 ms | 1.0 MiB / 0.496 ms | 2.0 MiB / 0.801 ms | 4.0 MiB / 1.596 ms |
| fused_apa | 1 | 0.0 MiB / 0.568 ms | 0.0 MiB / 0.854 ms | 0.1 MiB / 1.416 ms | 0.1 MiB / 2.518 ms | 0.3 MiB / 6.951 ms |
| fused_apa | 4 | 0.0 MiB / 0.804 ms | 0.0 MiB / 0.922 ms | 0.1 MiB / 1.402 ms | 0.1 MiB / 3.723 ms | 0.3 MiB / 6.674 ms |
| fused_apa | 8 | 0.0 MiB / 0.626 ms | 0.0 MiB / 0.845 ms | 0.1 MiB / 1.841 ms | 0.1 MiB / 3.583 ms | 0.3 MiB / 7.230 ms |
| fused_apa | 16 | 0.0 MiB / 0.624 ms | 0.0 MiB / 1.142 ms | 0.1 MiB / 2.073 ms | 0.1 MiB / 4.004 ms | 0.3 MiB / 7.961 ms |
| int4_apa | 1 | 0.3 MiB / 0.703 ms | 0.6 MiB / 1.069 ms | 1.1 MiB / 1.810 ms | 2.3 MiB / 3.287 ms | 4.5 MiB / 6.406 ms |
| int4_apa | 4 | 1.1 MiB / 0.706 ms | 2.2 MiB / 1.115 ms | 4.3 MiB / 1.821 ms | 8.6 MiB / 3.428 ms | 17.3 MiB / 10.580 ms |
| int4_apa | 8 | 2.1 MiB / 0.708 ms | 4.3 MiB / 1.192 ms | 8.6 MiB / 2.023 ms | 17.1 MiB / 5.736 ms | 34.3 MiB / 11.189 ms |
| int4_apa | 16 | 4.3 MiB / 0.761 ms | 8.5 MiB / 1.209 ms | 17.1 MiB / 3.264 ms | 34.1 MiB / 6.336 ms | 68.3 MiB / 12.247 ms |

## D=512

### PREFILL (L=512, bottom-right causal)

| Path | KV heads | S=4096 | S=8192 | S=16384 | S=32768 | S=65536 |
|---|---:|---:|---:|---:|---:|---:|
| standard | 1 | 136.0 MiB / 1.605 ms | 264.0 MiB / 3.364 ms | 520.0 MiB / 11.798 ms | 1032.0 MiB / 29.544 ms | 2056.0 MiB / 59.305 ms |
| standard | 4 | 136.0 MiB / 1.830 ms | 264.0 MiB / 3.495 ms | 520.0 MiB / 11.989 ms | 1032.0 MiB / 30.672 ms | 2056.0 MiB / 64.546 ms |
| standard | 8 | 136.0 MiB / 1.756 ms | 264.0 MiB / 3.504 ms | 520.0 MiB / 11.917 ms | 1032.0 MiB / 28.925 ms | 2056.0 MiB / 67.581 ms |
| standard | 16 | 136.0 MiB / 1.823 ms | 264.0 MiB / 3.523 ms | 520.0 MiB / 12.022 ms | 1032.0 MiB / 29.020 ms | 2056.0 MiB / 64.520 ms |
| fused_apa | 1 | 8.0 MiB / 49.216 ms | 8.0 MiB / 98.708 ms | 8.0 MiB / 210.143 ms | 8.0 MiB / 506.332 ms | 8.0 MiB / 1089.319 ms |
| fused_apa | 4 | 8.0 MiB / 47.241 ms | 8.0 MiB / 99.804 ms | 8.0 MiB / 209.223 ms | 8.0 MiB / 498.751 ms | 8.0 MiB / 1059.778 ms |
| fused_apa | 8 | 8.0 MiB / 46.887 ms | 8.0 MiB / 98.277 ms | 8.0 MiB / 210.055 ms | 8.0 MiB / 503.177 ms | 8.0 MiB / 1052.506 ms |
| fused_apa | 16 | 8.0 MiB / 45.439 ms | 8.0 MiB / 96.883 ms | 8.0 MiB / 220.532 ms | 8.0 MiB / 502.760 ms | 8.0 MiB / 1038.317 ms |
| int4_apa | 1 | 9.0 MiB / 37.933 ms | 10.0 MiB / 84.338 ms | 12.1 MiB / 179.162 ms | 16.1 MiB / 359.869 ms | 24.2 MiB / 744.914 ms |
| int4_apa | 4 | 12.1 MiB / 37.944 ms | 16.1 MiB / 86.194 ms | 24.2 MiB / 176.281 ms | 40.5 MiB / 360.020 ms | 73.0 MiB / 748.406 ms |
| int4_apa | 8 | 16.1 MiB / 40.206 ms | 24.2 MiB / 87.539 ms | 40.5 MiB / 178.384 ms | 73.0 MiB / 362.542 ms | 138.0 MiB / 758.637 ms |
| int4_apa | 16 | 24.2 MiB / 40.233 ms | 40.5 MiB / 84.213 ms | 73.0 MiB / 177.914 ms | 138.0 MiB / 369.061 ms | 268.0 MiB / 760.217 ms |

### DECODE (L=1, bottom-right causal)

| Path | KV heads | S=4096 | S=8192 | S=16384 | S=32768 | S=65536 |
|---|---:|---:|---:|---:|---:|---:|
| standard | 1 | 0.3 MiB / 0.168 ms | 0.5 MiB / 0.160 ms | 1.0 MiB / 0.189 ms | 2.0 MiB / 0.299 ms | 4.0 MiB / 0.517 ms |
| standard | 4 | 0.3 MiB / 0.169 ms | 0.5 MiB / 0.257 ms | 1.0 MiB / 0.418 ms | 2.0 MiB / 0.750 ms | 4.0 MiB / 1.400 ms |
| standard | 8 | 0.3 MiB / 0.280 ms | 0.5 MiB / 0.448 ms | 1.0 MiB / 0.794 ms | 2.0 MiB / 1.476 ms | 4.0 MiB / 2.850 ms |
| standard | 16 | 0.3 MiB / 0.425 ms | 0.5 MiB / 0.744 ms | 1.0 MiB / 1.476 ms | 2.0 MiB / 2.682 ms | 4.0 MiB / 5.344 ms |
| fused_apa | 1 | 0.1 MiB / 1.279 ms | 0.1 MiB / 1.330 ms | 0.3 MiB / 2.287 ms | 0.5 MiB / 4.937 ms | 1.0 MiB / 9.706 ms |
| fused_apa | 4 | 0.1 MiB / 0.964 ms | 0.1 MiB / 1.956 ms | 0.3 MiB / 2.963 ms | 0.5 MiB / 4.905 ms | 1.0 MiB / 9.752 ms |
| fused_apa | 8 | 0.1 MiB / 1.507 ms | 0.1 MiB / 2.037 ms | 0.3 MiB / 3.022 ms | 0.5 MiB / 5.201 ms | 1.0 MiB / 10.775 ms |
| fused_apa | 16 | 0.1 MiB / 1.677 ms | 0.1 MiB / 2.192 ms | 0.3 MiB / 3.337 ms | 0.5 MiB / 6.342 ms | 1.0 MiB / 13.032 ms |
| int4_apa | 1 | 1.1 MiB / 0.951 ms | 2.2 MiB / 1.391 ms | 4.3 MiB / 2.319 ms | 8.7 MiB / 4.085 ms | 17.3 MiB / 10.070 ms |
| int4_apa | 4 | 4.2 MiB / 0.993 ms | 8.3 MiB / 1.595 ms | 16.5 MiB / 3.652 ms | 33.0 MiB / 6.789 ms | 66.0 MiB / 13.445 ms |
| int4_apa | 8 | 8.2 MiB / 1.220 ms | 16.4 MiB / 2.352 ms | 32.8 MiB / 4.172 ms | 65.5 MiB / 7.490 ms | 131.0 MiB / 15.025 ms |
| int4_apa | 16 | 16.3 MiB / 1.735 ms | 32.7 MiB / 2.842 ms | 65.3 MiB / 4.776 ms | 130.5 MiB / 9.013 ms | 261.0 MiB / 18.228 ms |

## Factual notes

- Inputs were configured as BF16. Fused APA was configured with a signed per-key-vector 4-bit bulk quantize/dequantize and `refine_percentile=0.10` (`z = NormalPPF(0.90)`). Bulk-key preparation was outside the measured attention call.
- STANDARD is configured to invoke `tensor_cuda.matmul(q_grouped, k, trans_b=True)`, `tensor_cuda.causal_softmax(scores)`, then `tensor_cuda.matmul(weights_grouped, v)`. Q and weights are grouped as `(B, kv_heads, (q_heads/kv_heads)*L, ...)`; K/V are not expanded.
- FUSED APA is configured to invoke `tensor_cuda.apa_selective_attention(q, k, kq, v, scale, zthr, True)` with native `(q_heads, kv_heads)` geometry.
- INT4 APA is configured to invoke `tensor_cuda.apa_selective_attention_int4(q, k, v, scale, zthr, True)` with native `(q_heads, kv_heads)` geometry. Its call-local pack workspace and pack launch are included in both wall and pool measurements; it has no persistent kq operand.
- Pool values are call-local high-water deltas. NVML before/during/after absolute samples are retained per timed repetition in `results.json`.
- No cells OOMed, errored, or were skipped.
