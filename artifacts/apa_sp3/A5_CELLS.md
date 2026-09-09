# A5 cells and planning estimates

Evidence class: registered planning, not measured speed. New kind `decode_pool`, 32 steps, pool ON.
All existing cells remain registered unchanged. Default selection excludes 18 raw decode and 126 32K capture cells.

| ID | Dependencies | Estimate seconds | Worker TERM |
|---|---|---|---:|
| decode_pool_b4_A_2048 | g0, ppl_b4_D_1024, freeze_b4 | 60–280 (unmeasured) | 290s |
| decode_pool_b4_B_2048 | g0, ppl_b4_D_1024, freeze_b4 | 60–280 (unmeasured) | 290s |
| decode_pool_b4_C_2048 | g0, ppl_b4_D_1024, freeze_b4 | 60–280 (unmeasured) | 290s |
| decode_pool_b4_A_8192 | g0, ppl_b4_D_1024, freeze_b4 | registered dense non-fit; no model execution | 290s |
| decode_pool_b4_B_8192 | g0, ppl_b4_D_1024, freeze_b4 | 60–280 (unmeasured) | 290s |
| decode_pool_b4_C_8192 | g0, ppl_b4_D_1024, freeze_b4 | 60–280 (unmeasured) | 290s |
| decode_pool_b8_A_2048 | g0, ppl_b4_D_1024, freeze_b8 | 60–280 (unmeasured) | 290s |
| decode_pool_b8_B_2048 | g0, ppl_b4_D_1024, freeze_b8 | 60–280 (unmeasured) | 290s |
| decode_pool_b8_C_2048 | g0, ppl_b4_D_1024, freeze_b8 | 60–280 (unmeasured) | 290s |
| decode_pool_b8_A_8192 | g0, ppl_b4_D_1024, freeze_b8 | registered dense non-fit; no model execution | 290s |
| decode_pool_b8_B_8192 | g0, ppl_b4_D_1024, freeze_b8 | 60–280 (unmeasured) | 290s |
| decode_pool_b8_C_8192 | g0, ppl_b4_D_1024, freeze_b8 | 60–280 (unmeasured) | 290s |
| decode_pool_b4_A_32768 | g0, ppl_b4_D_1024, freeze_b4, decode_pool_b4_A_8192 | registered dense non-fit; no model execution | 290s |
| decode_pool_b4_B_32768 | g0, ppl_b4_D_1024, freeze_b4, decode_pool_b4_B_8192 | pending same-arm/bit pool8192: setup+guard+16*prefill+4*decode_work+15 | 290s |
| decode_pool_b4_C_32768 | g0, ppl_b4_D_1024, freeze_b4, decode_pool_b4_C_8192 | pending same-arm/bit pool8192: setup+guard+16*prefill+4*decode_work+15 | 290s |
| decode_pool_b8_A_32768 | g0, ppl_b4_D_1024, freeze_b8, decode_pool_b8_A_8192 | registered dense non-fit; no model execution | 290s |
| decode_pool_b8_B_32768 | g0, ppl_b4_D_1024, freeze_b8, decode_pool_b8_B_8192 | pending same-arm/bit pool8192: setup+guard+16*prefill+4*decode_work+15 | 290s |
| decode_pool_b8_C_32768 | g0, ppl_b4_D_1024, freeze_b8, decode_pool_b8_C_8192 | pending same-arm/bit pool8192: setup+guard+16*prefill+4*decode_work+15 | 290s |

32K: estimate >=290s is terminal non-fit, with an immutable plan and fit=false receipt before GPU lease. Missing 8192 measurement blocks. No retry.
Plans are create-only `artifacts/apa_sp3/plans_a5/<cell>.json`; actual 32K estimates do not exist until the lead measures pool8192.
Explicit 32K capture list: `lead_commands_32k_captures.txt`; retained a4 disk rail still applies.
