# A6 registered cells, rails and estimates

Immutable registration: amendment_011_decode_clean.json. Each executed cell receives one independent foreground lease; worker TERM 290s (+5s grace), outer 585s (+3s grace), command wrapper 590s (+2s). No automatic RED retry. Estimates are unmeasured planning values.

| Cell | Worker estimate seconds | Dependency additions | Fit / execution state |
|---|---|---|---|
| decode_repro_b4_A_2048 | [20, 280] | g0, kernel96 | GPU_BLOCKED |
| decode_bisect_00_reference | [20, 280] | g0, kernel96 | GPU_BLOCKED |
| decode_bisect_01_wrapper | [20, 280] | g0, kernel96 | GPU_BLOCKED |
| decode_bisect_02_host_logits | [20, 280] | g0, kernel96 | GPU_BLOCKED |
| decode_bisect_03_last_token_only | [20, 280] | g0, kernel96 | GPU_BLOCKED |
| decode_bisect_04_cache_recompute | [20, 280] | g0, kernel96 | GPU_BLOCKED |
| decode_bisect_05_pool_after | [20, 280] | g0, kernel96 | GPU_BLOCKED |
| decode_bisect_06_interposer | [20, 280] | g0, kernel96 | GPU_BLOCKED |
| decode_bisect_07_int4_eager | [20, 280] | g0, kernel96 | GPU_BLOCKED |
| decode_bisect_08_norm_eager | [20, 280] | g0, kernel96 | GPU_BLOCKED |
| decode_bisect_09_expanded_mla | [20, 280] | g0, kernel96 | GPU_BLOCKED |
| decode_bisect_10_softmax_eager | [20, 280] | g0, kernel96 | GPU_BLOCKED |
| decode_bisect_11_legacy_stack | [20, 280] | g0, kernel96 | GPU_BLOCKED |
| decode_clean_b4_A_2048 | [20, 280] | g0, kernel96 | GPU_BLOCKED |
| decode_clean_b4_B_2048 | [20, 280] | g0, kernel96, ppl_b4_D_1024, freeze_b4 | GPU_BLOCKED |
| decode_clean_b4_C_2048 | [20, 280] | g0, kernel96, ppl_b4_D_1024, freeze_b4 | GPU_BLOCKED |
| decode_clean_b4_A_8192 | None | g0, kernel96 | NON_FIT_REGISTERED_DENSE |
| decode_clean_b4_B_8192 | [20, 280] | g0, kernel96, ppl_b4_D_1024, freeze_b4 | GPU_BLOCKED |
| decode_clean_b4_C_8192 | [20, 280] | g0, kernel96, ppl_b4_D_1024, freeze_b4 | GPU_BLOCKED |
| decode_clean_b4_A_32768 | None | g0, kernel96 | NON_FIT_REGISTERED_DENSE |
| decode_clean_b4_B_32768 | None | g0, kernel96, ppl_b4_D_1024, freeze_b4, decode_clean_b4_B_8192 | GPU_BLOCKED |
| decode_clean_b4_C_32768 | None | g0, kernel96, ppl_b4_D_1024, freeze_b4, decode_clean_b4_C_8192 | GPU_BLOCKED |
| decode_clean_b8_A_2048 | [20, 280] | g0, kernel96 | GPU_BLOCKED |
| decode_clean_b8_B_2048 | [20, 280] | g0, kernel96, ppl_b4_D_1024, freeze_b8 | GPU_BLOCKED |
| decode_clean_b8_C_2048 | [20, 280] | g0, kernel96, ppl_b4_D_1024, freeze_b8 | GPU_BLOCKED |
| decode_clean_b8_A_8192 | None | g0, kernel96 | NON_FIT_REGISTERED_DENSE |
| decode_clean_b8_B_8192 | [20, 280] | g0, kernel96, ppl_b4_D_1024, freeze_b8 | GPU_BLOCKED |
| decode_clean_b8_C_8192 | [20, 280] | g0, kernel96, ppl_b4_D_1024, freeze_b8 | GPU_BLOCKED |
| decode_clean_b8_A_32768 | None | g0, kernel96 | NON_FIT_REGISTERED_DENSE |
| decode_clean_b8_B_32768 | None | g0, kernel96, ppl_b4_D_1024, freeze_b8, decode_clean_b8_B_8192 | GPU_BLOCKED |
| decode_clean_b8_C_32768 | None | g0, kernel96, ppl_b4_D_1024, freeze_b8, decode_clean_b8_C_8192 | GPU_BLOCKED |

No-hook pin: `test_decode_clean_no_attention_hook_installed` (parametrized A/B/C).
Clean 32K B/C: setup + 16*prefill + 4*(warmup+decode_work) + 15 seconds from same-arm/bit clean8192. Missing/invalid source blocks; estimate >=290 is terminal non-fit before lease. A8192/32768 remains registered dense non-fit.
The 31 cells are one greedy reproduction, twelve teacher-forced bisect jobs (including reference and combined endpoint), and eighteen clean cells (bulk bits 4 and optional 8; model weights always INT4).
