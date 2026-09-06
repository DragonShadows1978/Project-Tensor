# A4 registered new/changed cells

| ID | Kind | Dependencies | Worker estimate s | Worker rail s |
|---|---|---|---|---:|
| ppl_T_1024 | torch_reference | g0 | [30, 480] | 480 |
| ppl_T_8192 | torch_reference | g0 | [30, 480] | 480 |
| ppl_T_32768 | torch_reference | g0 | [30, 480] | 480 |
| ceiling_b4_A_4096 | ceiling | g0 | [30, 280] | 290 |
| ceiling_b4_A_8192 | ceiling | g0 | [30, 280] | 290 |
| ceiling_b4_A_16384 | ceiling | g0 | [30, 280] | 290 |
| ceiling_b4_A_24576 | ceiling | g0 | [30, 280] | 290 |
| ceiling_b4_A_32768 | ceiling | g0 | [30, 280] | 290 |
| ceiling_b4_B_4096 | ceiling | g0 | [30, 280] | 290 |
| ceiling_b4_B_8192 | ceiling | g0 | [30, 280] | 290 |
| ceiling_b4_B_16384 | ceiling | g0 | [30, 280] | 290 |
| ceiling_b4_B_24576 | ceiling | g0 | [30, 280] | 290 |
| ceiling_b4_B_32768 | ceiling | g0 | [30, 280] | 290 |
| ppl_b4_B_32768 | ppl_long | g0, kernel96, ppl_b4_D_1024, ceiling_b4_B_32768 | [60, 480] | 480 |
| ppl_b4_D_32768 | ppl_long | g0, kernel96, ppl_b4_D_1024, ppl_T_32768 | [60, 480] | 480 |
| capture_b4_B_32768_r00_01 | capture_range | ppl_b4_B_32768 | [30, 188] | 290 |
| capture_b4_B_32768_r01_02 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r00_01 | [30, 188] | 290 |
| capture_b4_B_32768_r02_03 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r01_02 | [30, 188] | 290 |
| capture_b4_B_32768_r03_04 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r02_03 | [30, 188] | 290 |
| capture_b4_B_32768_r04_05 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r03_04 | [30, 188] | 290 |
| capture_b4_B_32768_r05_06 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r04_05 | [30, 188] | 290 |
| capture_b4_B_32768_r06_07 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r05_06 | [30, 188] | 290 |
| capture_b4_B_32768_r07_08 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r06_07 | [30, 188] | 290 |
| capture_b4_B_32768_r08_09 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r07_08 | [30, 188] | 290 |
| capture_b4_B_32768_r09_10 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r08_09 | [30, 188] | 290 |
| ceiling_b4_C_4096 | ceiling | g0, ppl_b4_D_1024, freeze_b4 | [30, 280] | 290 |
| ceiling_b4_C_8192 | ceiling | g0, ppl_b4_D_1024, freeze_b4 | [30, 280] | 290 |
| ceiling_b4_C_16384 | ceiling | g0, ppl_b4_D_1024, freeze_b4 | [30, 280] | 290 |
| ceiling_b4_C_24576 | ceiling | g0, ppl_b4_D_1024, freeze_b4 | [30, 280] | 290 |
| ceiling_b4_C_32768 | ceiling | g0, ppl_b4_D_1024, freeze_b4 | [30, 280] | 290 |
| capture_b4_B_8192_r00_16 | capture_range | g0, ppl_b4_D_1024, freeze_b4 | [30, 188] | 290 |
| capture_b4_C_8192_r00_16 | capture_range | g0, ppl_b4_D_1024, freeze_b4 | [30, 188] | 290 |
| capture_b8_B_8192_r00_16 | capture_range | g0, ppl_b4_D_1024, freeze_b8 | [30, 188] | 290 |
| capture_b8_C_8192_r00_16 | capture_range | g0, ppl_b4_D_1024, freeze_b8 | [30, 188] | 290 |
| capture_b4_B_32768_r10_11 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r09_10 | [30, 188] | 290 |
| ppl_b4_C_32768 | ppl_long | g0, kernel96, ppl_b4_D_1024, ceiling_b4_C_32768, freeze_b4 | [60, 480] | 480 |
| capture_b4_B_8192_r16_32 | capture_range | g0, ppl_b4_D_1024, freeze_b4, capture_b4_B_8192_r00_16 | [30, 188] | 290 |
| capture_b4_C_8192_r16_32 | capture_range | g0, ppl_b4_D_1024, freeze_b4, capture_b4_C_8192_r00_16 | [30, 188] | 290 |
| capture_b8_B_8192_r16_32 | capture_range | g0, ppl_b4_D_1024, freeze_b8, capture_b8_B_8192_r00_16 | [30, 188] | 290 |
| capture_b8_C_8192_r16_32 | capture_range | g0, ppl_b4_D_1024, freeze_b8, capture_b8_C_8192_r00_16 | [30, 188] | 290 |
| capture_b4_B_32768_r11_12 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r10_11 | [30, 188] | 290 |
| capture_b4_B_8192_r32_48 | capture_range | g0, ppl_b4_D_1024, freeze_b4, capture_b4_B_8192_r16_32 | [30, 188] | 290 |
| capture_b4_C_8192_r32_48 | capture_range | g0, ppl_b4_D_1024, freeze_b4, capture_b4_C_8192_r16_32 | [30, 188] | 290 |
| capture_b8_B_8192_r32_48 | capture_range | g0, ppl_b4_D_1024, freeze_b8, capture_b8_B_8192_r16_32 | [30, 188] | 290 |
| capture_b8_C_8192_r32_48 | capture_range | g0, ppl_b4_D_1024, freeze_b8, capture_b8_C_8192_r16_32 | [30, 188] | 290 |
| capture_b4_B_32768_r12_13 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r11_12 | [30, 188] | 290 |
| capture_b4_C_32768_r00_01 | capture_range | ppl_b4_C_32768 | [30, 188] | 290 |
| capture_b4_B_8192_r48_62 | capture_range | g0, ppl_b4_D_1024, freeze_b4, capture_b4_B_8192_r32_48 | [30, 172] | 290 |
| capture_b4_C_8192_r48_62 | capture_range | g0, ppl_b4_D_1024, freeze_b4, capture_b4_C_8192_r32_48 | [30, 172] | 290 |
| capture_b8_B_8192_r48_62 | capture_range | g0, ppl_b4_D_1024, freeze_b8, capture_b8_B_8192_r32_48 | [30, 172] | 290 |
| capture_b8_C_8192_r48_62 | capture_range | g0, ppl_b4_D_1024, freeze_b8, capture_b8_C_8192_r32_48 | [30, 172] | 290 |
| capture_b4_B_32768_r13_14 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r12_13 | [30, 188] | 290 |
| capture_b4_C_32768_r01_02 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r00_01 | [30, 188] | 290 |
| capture_b4_B_8192 | capture_aggregate | capture_b4_B_8192_r00_16, capture_b4_B_8192_r16_32, capture_b4_B_8192_r32_48, capture_b4_B_8192_r48_62 | [5, 280] | 480 |
| capture_b4_C_8192 | capture_aggregate | capture_b4_C_8192_r00_16, capture_b4_C_8192_r16_32, capture_b4_C_8192_r32_48, capture_b4_C_8192_r48_62 | [5, 280] | 480 |
| capture_b8_B_8192 | capture_aggregate | capture_b8_B_8192_r00_16, capture_b8_B_8192_r16_32, capture_b8_B_8192_r32_48, capture_b8_B_8192_r48_62 | [5, 280] | 480 |
| capture_b8_C_8192 | capture_aggregate | capture_b8_C_8192_r00_16, capture_b8_C_8192_r16_32, capture_b8_C_8192_r32_48, capture_b8_C_8192_r48_62 | [5, 280] | 480 |
| capture_b4_B_32768_r14_15 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r13_14 | [30, 188] | 290 |
| capture_b4_C_32768_r02_03 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r01_02 | [30, 188] | 290 |
| capture_b4_B_32768_r15_16 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r14_15 | [30, 188] | 290 |
| capture_b4_C_32768_r03_04 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r02_03 | [30, 188] | 290 |
| capture_b4_B_32768_r16_17 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r15_16 | [30, 188] | 290 |
| capture_b4_C_32768_r04_05 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r03_04 | [30, 188] | 290 |
| capture_b4_B_32768_r17_18 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r16_17 | [30, 188] | 290 |
| capture_b4_C_32768_r05_06 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r04_05 | [30, 188] | 290 |
| capture_b4_E_8192_r00_16 | capture_range | ppl_b4_E_8192, eq_b4 | [30, 188] | 290 |
| capture_b4_B_32768_r18_19 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r17_18 | [30, 188] | 290 |
| capture_b4_C_32768_r06_07 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r05_06 | [30, 188] | 290 |
| capture_b4_E_8192_r16_32 | capture_range | ppl_b4_E_8192, eq_b4, capture_b4_E_8192_r00_16 | [30, 188] | 290 |
| capture_b4_B_32768_r19_20 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r18_19 | [30, 188] | 290 |
| capture_b4_C_32768_r07_08 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r06_07 | [30, 188] | 290 |
| capture_b4_E_8192_r32_48 | capture_range | ppl_b4_E_8192, eq_b4, capture_b4_E_8192_r16_32 | [30, 188] | 290 |
| capture_b4_B_32768_r20_21 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r19_20 | [30, 188] | 290 |
| capture_b4_C_32768_r08_09 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r07_08 | [30, 188] | 290 |
| capture_b4_E_8192_r48_62 | capture_range | ppl_b4_E_8192, eq_b4, capture_b4_E_8192_r32_48 | [30, 172] | 290 |
| capture_b4_B_32768_r21_22 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r20_21 | [30, 188] | 290 |
| capture_b4_C_32768_r09_10 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r08_09 | [30, 188] | 290 |
| capture_b4_E_8192 | capture_aggregate | capture_b4_E_8192_r00_16, capture_b4_E_8192_r16_32, capture_b4_E_8192_r32_48, capture_b4_E_8192_r48_62 | [5, 280] | 480 |
| capture_b4_B_32768_r22_23 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r21_22 | [30, 188] | 290 |
| capture_b4_C_32768_r10_11 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r09_10 | [30, 188] | 290 |
| capture_b4_B_32768_r23_24 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r22_23 | [30, 188] | 290 |
| capture_b4_C_32768_r11_12 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r10_11 | [30, 188] | 290 |
| capture_b4_B_32768_r24_25 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r23_24 | [30, 188] | 290 |
| capture_b4_C_32768_r12_13 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r11_12 | [30, 188] | 290 |
| capture_b4_B_32768_r25_26 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r24_25 | [30, 188] | 290 |
| capture_b4_C_32768_r13_14 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r12_13 | [30, 188] | 290 |
| capture_b4_B_32768_r26_27 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r25_26 | [30, 188] | 290 |
| capture_b4_C_32768_r14_15 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r13_14 | [30, 188] | 290 |
| capture_b4_B_32768_r27_28 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r26_27 | [30, 188] | 290 |
| capture_b4_C_32768_r15_16 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r14_15 | [30, 188] | 290 |
| capture_b4_B_32768_r28_29 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r27_28 | [30, 188] | 290 |
| capture_b4_C_32768_r16_17 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r15_16 | [30, 188] | 290 |
| capture_b4_B_32768_r29_30 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r28_29 | [30, 188] | 290 |
| capture_b4_C_32768_r17_18 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r16_17 | [30, 188] | 290 |
| capture_b4_B_32768_r30_31 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r29_30 | [30, 188] | 290 |
| capture_b4_C_32768_r18_19 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r17_18 | [30, 188] | 290 |
| capture_b4_B_32768_r31_32 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r30_31 | [30, 188] | 290 |
| capture_b4_C_32768_r19_20 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r18_19 | [30, 188] | 290 |
| capture_b4_B_32768_r32_33 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r31_32 | [30, 188] | 290 |
| capture_b4_C_32768_r20_21 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r19_20 | [30, 188] | 290 |
| capture_b4_B_32768_r33_34 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r32_33 | [30, 188] | 290 |
| capture_b4_C_32768_r21_22 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r20_21 | [30, 188] | 290 |
| capture_b4_B_32768_r34_35 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r33_34 | [30, 188] | 290 |
| capture_b4_C_32768_r22_23 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r21_22 | [30, 188] | 290 |
| capture_b4_B_32768_r35_36 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r34_35 | [30, 188] | 290 |
| capture_b4_C_32768_r23_24 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r22_23 | [30, 188] | 290 |
| capture_b4_B_32768_r36_37 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r35_36 | [30, 188] | 290 |
| capture_b4_C_32768_r24_25 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r23_24 | [30, 188] | 290 |
| capture_b4_B_32768_r37_38 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r36_37 | [30, 188] | 290 |
| capture_b4_C_32768_r25_26 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r24_25 | [30, 188] | 290 |
| capture_b4_B_32768_r38_39 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r37_38 | [30, 188] | 290 |
| capture_b4_C_32768_r26_27 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r25_26 | [30, 188] | 290 |
| capture_b4_B_32768_r39_40 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r38_39 | [30, 188] | 290 |
| capture_b4_C_32768_r27_28 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r26_27 | [30, 188] | 290 |
| capture_b4_B_32768_r40_41 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r39_40 | [30, 188] | 290 |
| capture_b4_C_32768_r28_29 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r27_28 | [30, 188] | 290 |
| capture_b4_B_32768_r41_42 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r40_41 | [30, 188] | 290 |
| capture_b4_C_32768_r29_30 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r28_29 | [30, 188] | 290 |
| capture_b4_B_32768_r42_43 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r41_42 | [30, 188] | 290 |
| capture_b4_C_32768_r30_31 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r29_30 | [30, 188] | 290 |
| capture_b4_B_32768_r43_44 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r42_43 | [30, 188] | 290 |
| capture_b4_C_32768_r31_32 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r30_31 | [30, 188] | 290 |
| capture_b4_B_32768_r44_45 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r43_44 | [30, 188] | 290 |
| capture_b4_C_32768_r32_33 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r31_32 | [30, 188] | 290 |
| capture_b4_B_32768_r45_46 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r44_45 | [30, 188] | 290 |
| capture_b4_C_32768_r33_34 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r32_33 | [30, 188] | 290 |
| capture_b4_B_32768_r46_47 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r45_46 | [30, 188] | 290 |
| capture_b4_C_32768_r34_35 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r33_34 | [30, 188] | 290 |
| capture_b4_B_32768_r47_48 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r46_47 | [30, 188] | 290 |
| capture_b4_C_32768_r35_36 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r34_35 | [30, 188] | 290 |
| capture_b4_B_32768_r48_49 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r47_48 | [30, 188] | 290 |
| capture_b4_C_32768_r36_37 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r35_36 | [30, 188] | 290 |
| capture_b4_B_32768_r49_50 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r48_49 | [30, 188] | 290 |
| capture_b4_C_32768_r37_38 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r36_37 | [30, 188] | 290 |
| capture_b4_B_32768_r50_51 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r49_50 | [30, 188] | 290 |
| capture_b4_C_32768_r38_39 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r37_38 | [30, 188] | 290 |
| capture_b4_B_32768_r51_52 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r50_51 | [30, 188] | 290 |
| capture_b4_C_32768_r39_40 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r38_39 | [30, 188] | 290 |
| capture_b4_B_32768_r52_53 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r51_52 | [30, 188] | 290 |
| capture_b4_C_32768_r40_41 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r39_40 | [30, 188] | 290 |
| capture_b4_B_32768_r53_54 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r52_53 | [30, 188] | 290 |
| capture_b4_C_32768_r41_42 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r40_41 | [30, 188] | 290 |
| capture_b4_B_32768_r54_55 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r53_54 | [30, 188] | 290 |
| capture_b4_C_32768_r42_43 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r41_42 | [30, 188] | 290 |
| capture_b4_B_32768_r55_56 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r54_55 | [30, 188] | 290 |
| capture_b4_C_32768_r43_44 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r42_43 | [30, 188] | 290 |
| capture_b4_B_32768_r56_57 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r55_56 | [30, 188] | 290 |
| capture_b4_C_32768_r44_45 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r43_44 | [30, 188] | 290 |
| capture_b4_B_32768_r57_58 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r56_57 | [30, 188] | 290 |
| capture_b4_C_32768_r45_46 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r44_45 | [30, 188] | 290 |
| capture_b4_B_32768_r58_59 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r57_58 | [30, 188] | 290 |
| capture_b4_C_32768_r46_47 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r45_46 | [30, 188] | 290 |
| capture_b4_B_32768_r59_60 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r58_59 | [30, 188] | 290 |
| capture_b4_C_32768_r47_48 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r46_47 | [30, 188] | 290 |
| capture_b4_B_32768_r60_61 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r59_60 | [30, 188] | 290 |
| capture_b4_C_32768_r48_49 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r47_48 | [30, 188] | 290 |
| capture_b4_B_32768_r61_62 | capture_range | ppl_b4_B_32768, capture_b4_B_32768_r60_61 | [30, 188] | 290 |
| capture_b4_C_32768_r49_50 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r48_49 | [30, 188] | 290 |
| capture_b4_B_32768 | capture_aggregate | capture_b4_B_32768_r00_01, capture_b4_B_32768_r01_02, capture_b4_B_32768_r02_03, capture_b4_B_32768_r03_04, capture_b4_B_32768_r04_05, capture_b4_B_32768_r05_06, capture_b4_B_32768_r06_07, capture_b4_B_32768_r07_08, capture_b4_B_32768_r08_09, capture_b4_B_32768_r09_10, capture_b4_B_32768_r10_11, capture_b4_B_32768_r11_12, capture_b4_B_32768_r12_13, capture_b4_B_32768_r13_14, capture_b4_B_32768_r14_15, capture_b4_B_32768_r15_16, capture_b4_B_32768_r16_17, capture_b4_B_32768_r17_18, capture_b4_B_32768_r18_19, capture_b4_B_32768_r19_20, capture_b4_B_32768_r20_21, capture_b4_B_32768_r21_22, capture_b4_B_32768_r22_23, capture_b4_B_32768_r23_24, capture_b4_B_32768_r24_25, capture_b4_B_32768_r25_26, capture_b4_B_32768_r26_27, capture_b4_B_32768_r27_28, capture_b4_B_32768_r28_29, capture_b4_B_32768_r29_30, capture_b4_B_32768_r30_31, capture_b4_B_32768_r31_32, capture_b4_B_32768_r32_33, capture_b4_B_32768_r33_34, capture_b4_B_32768_r34_35, capture_b4_B_32768_r35_36, capture_b4_B_32768_r36_37, capture_b4_B_32768_r37_38, capture_b4_B_32768_r38_39, capture_b4_B_32768_r39_40, capture_b4_B_32768_r40_41, capture_b4_B_32768_r41_42, capture_b4_B_32768_r42_43, capture_b4_B_32768_r43_44, capture_b4_B_32768_r44_45, capture_b4_B_32768_r45_46, capture_b4_B_32768_r46_47, capture_b4_B_32768_r47_48, capture_b4_B_32768_r48_49, capture_b4_B_32768_r49_50, capture_b4_B_32768_r50_51, capture_b4_B_32768_r51_52, capture_b4_B_32768_r52_53, capture_b4_B_32768_r53_54, capture_b4_B_32768_r54_55, capture_b4_B_32768_r55_56, capture_b4_B_32768_r56_57, capture_b4_B_32768_r57_58, capture_b4_B_32768_r58_59, capture_b4_B_32768_r59_60, capture_b4_B_32768_r60_61, capture_b4_B_32768_r61_62 | [5, 280] | 480 |
| capture_b4_C_32768_r50_51 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r49_50 | [30, 188] | 290 |
| capture_b4_C_32768_r51_52 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r50_51 | [30, 188] | 290 |
| capture_b4_C_32768_r52_53 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r51_52 | [30, 188] | 290 |
| capture_b4_C_32768_r53_54 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r52_53 | [30, 188] | 290 |
| capture_b4_C_32768_r54_55 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r53_54 | [30, 188] | 290 |
| capture_b4_C_32768_r55_56 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r54_55 | [30, 188] | 290 |
| capture_b4_C_32768_r56_57 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r55_56 | [30, 188] | 290 |
| capture_b4_C_32768_r57_58 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r56_57 | [30, 188] | 290 |
| capture_b4_C_32768_r58_59 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r57_58 | [30, 188] | 290 |
| capture_b4_C_32768_r59_60 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r58_59 | [30, 188] | 290 |
| capture_b4_C_32768_r60_61 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r59_60 | [30, 188] | 290 |
| capture_b4_C_32768_r61_62 | capture_range | ppl_b4_C_32768, capture_b4_C_32768_r60_61 | [30, 188] | 290 |
| capture_b4_C_32768 | capture_aggregate | capture_b4_C_32768_r00_01, capture_b4_C_32768_r01_02, capture_b4_C_32768_r02_03, capture_b4_C_32768_r03_04, capture_b4_C_32768_r04_05, capture_b4_C_32768_r05_06, capture_b4_C_32768_r06_07, capture_b4_C_32768_r07_08, capture_b4_C_32768_r08_09, capture_b4_C_32768_r09_10, capture_b4_C_32768_r10_11, capture_b4_C_32768_r11_12, capture_b4_C_32768_r12_13, capture_b4_C_32768_r13_14, capture_b4_C_32768_r14_15, capture_b4_C_32768_r15_16, capture_b4_C_32768_r16_17, capture_b4_C_32768_r17_18, capture_b4_C_32768_r18_19, capture_b4_C_32768_r19_20, capture_b4_C_32768_r20_21, capture_b4_C_32768_r21_22, capture_b4_C_32768_r22_23, capture_b4_C_32768_r23_24, capture_b4_C_32768_r24_25, capture_b4_C_32768_r25_26, capture_b4_C_32768_r26_27, capture_b4_C_32768_r27_28, capture_b4_C_32768_r28_29, capture_b4_C_32768_r29_30, capture_b4_C_32768_r30_31, capture_b4_C_32768_r31_32, capture_b4_C_32768_r32_33, capture_b4_C_32768_r33_34, capture_b4_C_32768_r34_35, capture_b4_C_32768_r35_36, capture_b4_C_32768_r36_37, capture_b4_C_32768_r37_38, capture_b4_C_32768_r38_39, capture_b4_C_32768_r39_40, capture_b4_C_32768_r40_41, capture_b4_C_32768_r41_42, capture_b4_C_32768_r42_43, capture_b4_C_32768_r43_44, capture_b4_C_32768_r44_45, capture_b4_C_32768_r45_46, capture_b4_C_32768_r46_47, capture_b4_C_32768_r47_48, capture_b4_C_32768_r48_49, capture_b4_C_32768_r49_50, capture_b4_C_32768_r50_51, capture_b4_C_32768_r51_52, capture_b4_C_32768_r52_53, capture_b4_C_32768_r53_54, capture_b4_C_32768_r54_55, capture_b4_C_32768_r55_56, capture_b4_C_32768_r56_57, capture_b4_C_32768_r57_58, capture_b4_C_32768_r58_59, capture_b4_C_32768_r59_60, capture_b4_C_32768_r60_61, capture_b4_C_32768_r61_62 | [5, 280] | 480 |
