## A6 clean decode — authoritative decode/P5 column

Supersedes the historical raw/pool decode P5 columns. GPU timings are absent until the lead runs each cell. 32 synchronized token steps, scalar device argmax copied to host; forward-only times also retained. One discarded warmup preserves prefill cache and timed contexts. Pool enabled before weights; pool reserved/used high water is not a whole-device resident peak. A uses absorbed MLA; B/C retain expanded selective attention with fast projections/norms. No per-layer diagnostic hook.

| Cell | Status/outcome | ms/token | Forward ms/token | tokens/s | Pool peak MiB |
|---|---|---:|---:|---:|---:|
| decode_repro_b4_A_2048 | GPU_BLOCKED | — | — | — | — |
| decode_bisect_00_reference | GPU_BLOCKED | — | — | — | — |
| decode_bisect_01_wrapper | GPU_BLOCKED | — | — | — | — |
| decode_bisect_02_host_logits | GPU_BLOCKED | — | — | — | — |
| decode_bisect_03_last_token_only | GPU_BLOCKED | — | — | — | — |
| decode_bisect_04_cache_recompute | GPU_BLOCKED | — | — | — | — |
| decode_bisect_05_pool_after | GPU_BLOCKED | — | — | — | — |
| decode_bisect_06_interposer | GPU_BLOCKED | — | — | — | — |
| decode_bisect_07_int4_eager | GPU_BLOCKED | — | — | — | — |
| decode_bisect_08_norm_eager | GPU_BLOCKED | — | — | — | — |
| decode_bisect_09_expanded_mla | GPU_BLOCKED | — | — | — | — |
| decode_bisect_10_softmax_eager | GPU_BLOCKED | — | — | — | — |
| decode_bisect_11_legacy_stack | GPU_BLOCKED | — | — | — | — |
| decode_clean_b4_A_2048 | GPU_BLOCKED | — | — | — | — |
| decode_clean_b4_B_2048 | GPU_BLOCKED | — | — | — | — |
| decode_clean_b4_C_2048 | GPU_BLOCKED | — | — | — | — |
| decode_clean_b4_A_8192 | NON_FIT_REGISTERED_DENSE (registered, no worker) | — | — | — | — |
| decode_clean_b4_B_8192 | GPU_BLOCKED | — | — | — | — |
| decode_clean_b4_C_8192 | GPU_BLOCKED | — | — | — | — |
| decode_clean_b4_A_32768 | NON_FIT_REGISTERED_DENSE (registered, no worker) | — | — | — | — |
| decode_clean_b4_B_32768 | GPU_BLOCKED | — | — | — | — |
| decode_clean_b4_C_32768 | GPU_BLOCKED | — | — | — | — |
| decode_clean_b8_A_2048 | GPU_BLOCKED | — | — | — | — |
| decode_clean_b8_B_2048 | GPU_BLOCKED | — | — | — | — |
| decode_clean_b8_C_2048 | GPU_BLOCKED | — | — | — | — |
| decode_clean_b8_A_8192 | NON_FIT_REGISTERED_DENSE (registered, no worker) | — | — | — | — |
| decode_clean_b8_B_8192 | GPU_BLOCKED | — | — | — | — |
| decode_clean_b8_C_8192 | GPU_BLOCKED | — | — | — | — |
| decode_clean_b8_A_32768 | NON_FIT_REGISTERED_DENSE (registered, no worker) | — | — | — | — |
| decode_clean_b8_B_32768 | GPU_BLOCKED | — | — | — | — |
| decode_clean_b8_C_32768 | GPU_BLOCKED | — | — | — | — |

Reproduction: `{"comparison": "UNASSESSED", "ms_token": null, "status": "BLOCKED"}`.
P5: `{"S": 32768, "bits": 4, "ratio": null, "ratio_8192": null, "reason": "32K clean measurement absent/non-fit/invalid; 8192 ratio does not evaluate P5", "source_kind": "decode_clean", "status": "UNASSESSABLE", "threshold": 2.0}`.
Rungs 01–10 each compare with 00 and change exactly one configuration field. Rung 11 is a combined legacy-flag endpoint, not a single-factor attribution. The old A5 loop also had in-process PPL controls and lacked argmax/warmup, so 11 is not a byte-for-byte A5 replay.
32K B/C plans require same-bit/arm clean8192 receipt: setup + 16×prefill + 4×(warmup+decode work) + 15 seconds. At or above 290s, record non-fit before lease. Dense A8192/32768 remains registered non-fit. Missing measurements block planning.

