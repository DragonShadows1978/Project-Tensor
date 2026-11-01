# A4 CPU mutant results

54 single-site mutants reach their changed expression during call phase (M14 by recorded call-phase default binding). 51 CPU kills and three passing equivalences. These are not GPU kills. Negations D5_g_npbool and D5_c_npbool remain reach checks. Both new deletion arms additionally pass all 405 real GN tests: E_npbool_g_full_cpu.json and E_npbool_c_full_cpu.json.

Prior art: DeMillo/Lipton/Sayward (1978); Offutt/Voas RIP and equivalent mutation analysis (year unverified - lead to check); PEP 357 (2006).

| Mutant | CPU result | Reached lines | Equivalence | Trace |
|---|---|---|---|---|
| D5_bias_dtype | KILLED | [554] |  | mutants/D5_bias_dtype/trace.json |
| D5_bias_length | KILLED | [564] |  | mutants/D5_bias_length/trace.json |
| D5_bias_rank | KILLED | [562] |  | mutants/D5_bias_rank/trace.json |
| D5_c_bool | KILLED | [526] |  | mutants/D5_c_bool/trace.json |
| D5_c_index | KILLED | [529] |  | mutants/D5_c_index/trace.json |
| D5_c_int | KILLED | [529] |  | mutants/D5_c_int/trace.json |
| D5_c_npbool | KILLED | [526] |  | mutants/D5_c_npbool/trace.json |
| D5_c_positive | KILLED | [532] |  | mutants/D5_c_positive/trace.json |
| D5_c_store | KILLED | [529] |  | mutants/D5_c_store/trace.json |
| D5_channels | KILLED | [548] |  | mutants/D5_channels/trace.json |
| D5_g_bool | KILLED | [518] |  | mutants/D5_g_bool/trace.json |
| D5_g_index | KILLED | [521] |  | mutants/D5_g_index/trace.json |
| D5_g_int | KILLED | [521] |  | mutants/D5_g_int/trace.json |
| D5_g_npbool | KILLED | [518] |  | mutants/D5_g_npbool/trace.json |
| D5_g_positive | KILLED | [524] |  | mutants/D5_g_positive/trace.json |
| D5_g_store | KILLED | [521] |  | mutants/D5_g_store/trace.json |
| D5_input_dtype | KILLED | [550] |  | mutants/D5_input_dtype/trace.json |
| D5_rank | KILLED | [546] |  | mutants/D5_rank/trace.json |
| D5_weight_dtype | KILLED | [552] |  | mutants/D5_weight_dtype/trace.json |
| D5_weight_length | KILLED | [560] |  | mutants/D5_weight_length/trace.json |
| D5_weight_rank | KILLED | [558] |  | mutants/D5_weight_rank/trace.json |
| E_npbool_c | PASS equivalence | [526] | EQUIVALENCES.md#numpy-bool-c | mutants/E_npbool_c/trace.json |
| E_npbool_g | PASS equivalence | [518] | EQUIVALENCES.md#numpy-bool-g | mutants/E_npbool_g/trace.json |
| M01 | KILLED | [579] |  | mutants/M01/trace.json |
| M02 | KILLED | [580] |  | mutants/M02/trace.json |
| M03 | KILLED | [580] |  | mutants/M03/trace.json |
| M04m | KILLED | [579] |  | mutants/M04m/trace.json |
| M04v | KILLED | [579] |  | mutants/M04v/trace.json |
| M05 | KILLED | [576] |  | mutants/M05/trace.json |
| M06in | KILLED | [576] |  | mutants/M06in/trace.json |
| M06out | KILLED | [581] |  | mutants/M06out/trace.json |
| M07cast | KILLED | [588] |  | mutants/M07cast/trace.json |
| M07fast | KILLED | [585] |  | mutants/M07fast/trace.json |
| M08cast | KILLED | [588] |  | mutants/M08cast/trace.json |
| M08fast | KILLED | [585] |  | mutants/M08fast/trace.json |
| M09cast | KILLED | [588] |  | mutants/M09cast/trace.json |
| M09fast | KILLED | [585] |  | mutants/M09fast/trace.json |
| M10 | KILLED | [534] |  | mutants/M10/trace.json |
| M11 | KILLED | [582] |  | mutants/M11/trace.json |
| M12 | KILLED | [580] |  | mutants/M12/trace.json |
| M13 | KILLED | [580] |  | mutants/M13/trace.json |
| M14 | KILLED | call default binding |  | mutants/M14/trace.json |
| M15b | KILLED | [587] |  | mutants/M15b/trace.json |
| M15w | KILLED | [586] |  | mutants/M15w/trace.json |
| M15x | KILLED | [571] |  | mutants/M15x/trace.json |
| M16 | KILLED | [589] |  | mutants/M16/trace.json |
| M17b | KILLED | [587] |  | mutants/M17b/trace.json |
| M17w | KILLED | [586] |  | mutants/M17w/trace.json |
| M18b | KILLED | [587] |  | mutants/M18b/trace.json |
| M18w | KILLED | [586] |  | mutants/M18w/trace.json |
| V1 | KILLED | [584] |  | mutants/V1/trace.json |
| V2 | KILLED | [583] |  | mutants/V2/trace.json |
| V3 | PASS equivalence | [571] | EQUIVALENCES.md#v3 | mutants/V3/trace.json |
| V4 | KILLED | [583] |  | mutants/V4/trace.json |
