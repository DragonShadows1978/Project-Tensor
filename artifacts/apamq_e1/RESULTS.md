# APAMQ-E1 MQA-Geometry Kernel Transient Sweep

Evidence class: **KERNEL SWEEP**. Values are raw measurements; no H-A/T1 verdict is made here.

Each cell is `CUDA-pool peak delta MiB / warm median wall ms` (1 warm-up, 3 timed calls).

## D=128

### PREFILL (L=512, bottom-right causal)

| Path | KV heads | S=4096 | S=8192 | S=16384 | S=32768 | S=65536 |
|---|---:|---:|---:|---:|---:|---:|
| standard | 1 | 130.0 MiB / 1.116 ms | 258.0 MiB / 2.009 ms | 514.0 MiB / 8.953 ms | 1026.0 MiB / 23.372 ms | 2050.0 MiB / 51.863 ms |
| standard | 4 | 130.0 MiB / 1.134 ms | 258.0 MiB / 2.214 ms | 514.0 MiB / 9.393 ms | 1026.0 MiB / 23.618 ms | 2050.0 MiB / 52.451 ms |
| standard | 8 | 130.0 MiB / 1.109 ms | 258.0 MiB / 2.207 ms | 514.0 MiB / 9.315 ms | 1026.0 MiB / 23.538 ms | 2050.0 MiB / 54.498 ms |
| standard | 16 | 130.0 MiB / 1.122 ms | 258.0 MiB / 2.137 ms | 514.0 MiB / 9.203 ms | 1026.0 MiB / 23.297 ms | 2050.0 MiB / 51.718 ms |
| fused_apa | 1 | 2.0 MiB / 16.319 ms | 2.0 MiB / 33.771 ms | 2.0 MiB / 72.957 ms | 2.0 MiB / 146.912 ms | 2.0 MiB / 302.022 ms |
| fused_apa | 4 | 2.0 MiB / 16.469 ms | 2.0 MiB / 35.875 ms | 2.0 MiB / 72.911 ms | 2.0 MiB / 148.864 ms | 2.0 MiB / 302.237 ms |
| fused_apa | 8 | 2.0 MiB / 16.346 ms | 2.0 MiB / 35.633 ms | 2.0 MiB / 71.855 ms | 2.0 MiB / 146.978 ms | 2.0 MiB / 298.900 ms |
| fused_apa | 16 | 2.0 MiB / 16.369 ms | 2.0 MiB / 35.556 ms | 2.0 MiB / 71.599 ms | 2.0 MiB / 146.844 ms | 2.0 MiB / 297.009 ms |

### DECODE (L=1, bottom-right causal)

| Path | KV heads | S=4096 | S=8192 | S=16384 | S=32768 | S=65536 |
|---|---:|---:|---:|---:|---:|---:|
| standard | 1 | 0.3 MiB / 0.168 ms | 0.5 MiB / 0.188 ms | 1.0 MiB / 0.205 ms | 2.0 MiB / 0.244 ms | 4.0 MiB / 0.300 ms |
| standard | 4 | 0.3 MiB / 0.147 ms | 0.5 MiB / 0.167 ms | 1.0 MiB / 0.332 ms | 2.0 MiB / 0.493 ms | 4.0 MiB / 0.829 ms |
| standard | 8 | 0.3 MiB / 0.143 ms | 0.5 MiB / 0.166 ms | 1.0 MiB / 0.295 ms | 2.0 MiB / 0.523 ms | 4.0 MiB / 0.918 ms |
| standard | 16 | 0.3 MiB / 0.155 ms | 0.5 MiB / 0.258 ms | 1.0 MiB / 0.418 ms | 2.0 MiB / 0.787 ms | 4.0 MiB / 1.485 ms |
| fused_apa | 1 | 0.0 MiB / 0.570 ms | 0.0 MiB / 0.910 ms | 0.1 MiB / 1.874 ms | 0.1 MiB / 2.531 ms | 0.3 MiB / 4.862 ms |
| fused_apa | 4 | 0.0 MiB / 0.573 ms | 0.0 MiB / 0.848 ms | 0.1 MiB / 1.403 ms | 0.1 MiB / 3.053 ms | 0.3 MiB / 6.696 ms |
| fused_apa | 8 | 0.0 MiB / 0.567 ms | 0.0 MiB / 0.853 ms | 0.1 MiB / 1.736 ms | 0.1 MiB / 3.589 ms | 0.3 MiB / 7.262 ms |
| fused_apa | 16 | 0.0 MiB / 0.570 ms | 0.0 MiB / 1.113 ms | 0.1 MiB / 2.065 ms | 0.1 MiB / 3.961 ms | 0.3 MiB / 7.926 ms |

## D=512

### PREFILL (L=512, bottom-right causal)

| Path | KV heads | S=4096 | S=8192 | S=16384 | S=32768 | S=65536 |
|---|---:|---:|---:|---:|---:|---:|
| standard | 1 | 136.0 MiB / 1.654 ms | 264.0 MiB / 3.446 ms | 520.0 MiB / 11.752 ms | 1032.0 MiB / 28.872 ms | 2056.0 MiB / 62.102 ms |
| standard | 4 | 136.0 MiB / 1.690 ms | 264.0 MiB / 3.436 ms | 520.0 MiB / 12.192 ms | 1032.0 MiB / 28.728 ms | 2056.0 MiB / 64.182 ms |
| standard | 8 | 136.0 MiB / 1.771 ms | 264.0 MiB / 3.438 ms | 520.0 MiB / 11.921 ms | 1032.0 MiB / 28.898 ms | 2056.0 MiB / 68.348 ms |
| standard | 16 | 136.0 MiB / 1.749 ms | 264.0 MiB / 3.429 ms | 520.0 MiB / 11.965 ms | 1032.0 MiB / 28.961 ms | 2056.0 MiB / 64.121 ms |
| fused_apa | 1 | 8.0 MiB / 45.384 ms | 8.0 MiB / 98.198 ms | 8.0 MiB / 207.053 ms | 8.0 MiB / 476.092 ms | 8.0 MiB / 1030.704 ms |
| fused_apa | 4 | 8.0 MiB / 46.748 ms | 8.0 MiB / 96.809 ms | 8.0 MiB / 204.804 ms | 8.0 MiB / 480.128 ms | 8.0 MiB / 1031.412 ms |
| fused_apa | 8 | 8.0 MiB / 45.153 ms | 8.0 MiB / 97.310 ms | 8.0 MiB / 204.645 ms | 8.0 MiB / 487.968 ms | 8.0 MiB / 1031.798 ms |
| fused_apa | 16 | 8.0 MiB / 46.844 ms | 8.0 MiB / 96.744 ms | 8.0 MiB / 220.206 ms | 8.0 MiB / 502.288 ms | 8.0 MiB / 1037.532 ms |

### DECODE (L=1, bottom-right causal)

| Path | KV heads | S=4096 | S=8192 | S=16384 | S=32768 | S=65536 |
|---|---:|---:|---:|---:|---:|---:|
| standard | 1 | 0.3 MiB / 0.161 ms | 0.5 MiB / 0.201 ms | 1.0 MiB / 0.174 ms | 2.0 MiB / 0.381 ms | 4.0 MiB / 0.582 ms |
| standard | 4 | 0.3 MiB / 0.164 ms | 0.5 MiB / 0.265 ms | 1.0 MiB / 0.417 ms | 2.0 MiB / 0.748 ms | 4.0 MiB / 1.402 ms |
| standard | 8 | 0.3 MiB / 0.266 ms | 0.5 MiB / 0.443 ms | 1.0 MiB / 0.790 ms | 2.0 MiB / 1.488 ms | 4.0 MiB / 2.858 ms |
| standard | 16 | 0.3 MiB / 0.422 ms | 0.5 MiB / 0.741 ms | 1.0 MiB / 1.377 ms | 2.0 MiB / 2.665 ms | 4.0 MiB / 5.380 ms |
| fused_apa | 1 | 0.1 MiB / 0.987 ms | 0.1 MiB / 1.310 ms | 0.3 MiB / 1.950 ms | 0.5 MiB / 4.198 ms | 1.0 MiB / 9.340 ms |
| fused_apa | 4 | 0.1 MiB / 0.959 ms | 0.1 MiB / 1.786 ms | 0.3 MiB / 2.912 ms | 0.5 MiB / 4.808 ms | 1.0 MiB / 9.744 ms |
| fused_apa | 8 | 0.1 MiB / 1.395 ms | 0.1 MiB / 2.047 ms | 0.3 MiB / 3.019 ms | 0.5 MiB / 5.206 ms | 1.0 MiB / 10.791 ms |
| fused_apa | 16 | 0.1 MiB / 1.677 ms | 0.1 MiB / 2.197 ms | 0.3 MiB / 3.342 ms | 0.5 MiB / 6.331 ms | 1.0 MiB / 13.018 ms |

## Factual notes

- Inputs were configured as BF16. Fused APA was configured with a signed per-key-vector 4-bit bulk quantize/dequantize and `refine_percentile=0.10` (`z = NormalPPF(0.90)`). Bulk-key preparation was outside the measured attention call.
- STANDARD is configured to invoke `tensor_cuda.matmul(q_grouped, k, trans_b=True)`, `tensor_cuda.causal_softmax(scores)`, then `tensor_cuda.matmul(weights_grouped, v)`. Q and weights are grouped as `(B, kv_heads, (q_heads/kv_heads)*L, ...)`; K/V are not expanded.
- FUSED APA is configured to invoke `tensor_cuda.apa_selective_attention(q, k, kq, v, scale, zthr, True)` with native `(q_heads, kv_heads)` geometry.
- Pool values are call-local high-water deltas. NVML before/during/after absolute samples are retained per timed repetition in `results.json`.
- No cells OOMed, errored, or were skipped.
