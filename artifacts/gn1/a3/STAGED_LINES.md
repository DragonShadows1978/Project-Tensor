# Staged lines

Run from A3’s own directory. STAGED_GPU_COMMANDS.sh runs each line in the foreground through the generic installed runner with --max and inner timeout. The script adds a final shell bound. This round ran only --audit and redirected CPU children, never these GPU lines.

| Label | Expected child exit | Expected parent exit | Inner seconds | Runner --max seconds | Parent bound seconds |
| --- | --- | --- | --- | --- | --- |
| baseline_focus | 1 | 0 | 240 | 300 | 330 |
| candidate_focus | 0 | 0 | 240 | 300 | 330 |
| baseline_regression | 1 | 0 | 840 | 900 | 930 |
| candidate_regression | 0,1 | 0 | 840 | 900 | 930 |
| D5_bias_dtype | 1 | 0 | 180 | 240 | 270 |
| D5_bias_length | 1 | 0 | 180 | 240 | 270 |
| D5_bias_rank | 1 | 0 | 180 | 240 | 270 |
| D5_c_bool | 1 | 0 | 180 | 240 | 270 |
| D5_c_index | 1 | 0 | 180 | 240 | 270 |
| D5_c_int | 1 | 0 | 180 | 240 | 270 |
| D5_c_npbool | 1 | 0 | 180 | 240 | 270 |
| D5_c_positive | 1 | 0 | 180 | 240 | 270 |
| D5_c_store | 1 | 0 | 180 | 240 | 270 |
| D5_channels | 1 | 0 | 180 | 240 | 270 |
| D5_g_bool | 1 | 0 | 180 | 240 | 270 |
| D5_g_index | 1 | 0 | 180 | 240 | 270 |
| D5_g_int | 1 | 0 | 180 | 240 | 270 |
| D5_g_npbool | 1 | 0 | 180 | 240 | 270 |
| D5_g_positive | 1 | 0 | 180 | 240 | 270 |
| D5_g_store | 1 | 0 | 180 | 240 | 270 |
| D5_input_dtype | 1 | 0 | 180 | 240 | 270 |
| D5_rank | 1 | 0 | 180 | 240 | 270 |
| D5_weight_dtype | 1 | 0 | 180 | 240 | 270 |
| D5_weight_length | 1 | 0 | 180 | 240 | 270 |
| D5_weight_rank | 1 | 0 | 180 | 240 | 270 |
| M01 | 1 | 0 | 180 | 240 | 270 |
| M02 | 1 | 0 | 180 | 240 | 270 |
| M03 | 1 | 0 | 180 | 240 | 270 |
| M04m | 1 | 0 | 180 | 240 | 270 |
| M04v | 1 | 0 | 180 | 240 | 270 |
| M05 | 1 | 0 | 180 | 240 | 270 |
| M06in | 1 | 0 | 180 | 240 | 270 |
| M06out | 1 | 0 | 180 | 240 | 270 |
| M07cast | 1 | 0 | 180 | 240 | 270 |
| M07fast | 1 | 0 | 180 | 240 | 270 |
| M08cast | 1 | 0 | 180 | 240 | 270 |
| M08fast | 1 | 0 | 180 | 240 | 270 |
| M09cast | 1 | 0 | 180 | 240 | 270 |
| M09fast | 1 | 0 | 180 | 240 | 270 |
| M10 | 1 | 0 | 180 | 240 | 270 |
| M11 | 1 | 0 | 180 | 240 | 270 |
| M12 | 1 | 0 | 180 | 240 | 270 |
| M13 | 1 | 0 | 180 | 240 | 270 |
| M14 | 1 | 0 | 180 | 240 | 270 |
| M15b | 1 | 0 | 180 | 240 | 270 |
| M15w | 1 | 0 | 180 | 240 | 270 |
| M15x | 1 | 0 | 180 | 240 | 270 |
| M16 | 1 | 0 | 180 | 240 | 270 |
| M17b | 1 | 0 | 180 | 240 | 270 |
| M17w | 1 | 0 | 180 | 240 | 270 |
| M18b | 1 | 0 | 180 | 240 | 270 |
| M18w | 1 | 0 | 180 | 240 | 270 |
| V1 | 1 | 0 | 180 | 240 | 270 |
| V2 | 1 | 0 | 180 | 240 | 270 |
| V3 | 0 | 0 | 180 | 240 | 270 |
| V4 | 1 | 0 | 180 | 240 | 270 |

Audit: `RUNS=<absolute a3 run root> TMPDIR=/tmp/gn1_a3_staged timeout 400s bash STAGED_GPU_COMMANDS.sh --audit`, expected parent 0; 29 reference calls. A full scorer must refuse this audit root. Final scorer after all real labels: `RUNS=<same root> timeout 120s python3 score_gpu.py`, expected 0.
