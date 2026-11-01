# G4 controls

48/48 PASS through the real sealed parent/scorer. CPU forge stand-in only. See HOSTILE_ADAPTATIONS.md for each REV1 adaptation.

| Control | Target | Observed result | Receipt |
|---|---|---|---|
| H1 | START.json | rejected; PASS | G4_hostile.json controls[0] |
| H1b | run-root guard | rejected; PASS | G4_hostile.json controls[1] |
| H2_REV1_exact | device-pattern failure | rejected; PASS | G4_hostile.json controls[2] |
| H2c_REV1_exact | device-pattern failure | rejected; PASS | G4_hostile.json controls[3] |
| H3_foreign_arm | arm/suite binding | rejected; PASS | G4_hostile.json controls[4] |
| H3b_suite_swap | expected exits | rejected; PASS | G4_hostile.json controls[5] |
| H4_junit_foreign_ID | unregistered JUnit ID | rejected; PASS | G4_hostile.json controls[6] |
| Junit_one_ID_disagreement | JUnit/summary outcomes disagree | rejected; PASS | G4_hostile.json controls[7] |
| Junit_duplicate_pass | duplicate JUnit ID | rejected; PASS | G4_hostile.json controls[8] |
| H2_pattern_00 | device-pattern failure | rejected; PASS | G4_hostile.json controls[9] |
| H2_pattern_01 | device-pattern failure | rejected; PASS | G4_hostile.json controls[10] |
| H2_pattern_02 | device-pattern failure | rejected; PASS | G4_hostile.json controls[11] |
| H2_pattern_03 | device-pattern failure | rejected; PASS | G4_hostile.json controls[12] |
| H2_pattern_04 | device-pattern failure | rejected; PASS | G4_hostile.json controls[13] |
| H2_pattern_05 | device-pattern failure | rejected; PASS | G4_hostile.json controls[14] |
| H2_pattern_06 | device-pattern failure | rejected; PASS | G4_hostile.json controls[15] |
| H2_pattern_07 | device-pattern failure | rejected; PASS | G4_hostile.json controls[16] |
| H2_pattern_08 | device-pattern failure | rejected; PASS | G4_hostile.json controls[17] |
| H2_pattern_09 | device-pattern failure | rejected; PASS | G4_hostile.json controls[18] |
| H2_pattern_10 | device-pattern failure | rejected; PASS | G4_hostile.json controls[19] |
| H2_pattern_11 | device-pattern failure | rejected; PASS | G4_hostile.json controls[20] |
| Reference_only_kill | no real non-reference | rejected; PASS | G4_hostile.json controls[21] |
| Generic_RuntimeError_not_kill | no real non-reference | rejected; PASS | G4_hostile.json controls[22] |
| Unregistered_TypeError_not_kill | no real non-reference | rejected; PASS | G4_hostile.json controls[23] |
| Wrong_nn_hash | nn.py binding | rejected; PASS | G4_hostile.json controls[24] |
| Wrong_engine_hash | engine binding | rejected; PASS | G4_hostile.json controls[25] |
| Setup_failure | setup/teardown failures | rejected; PASS | G4_hostile.json controls[26] |
| Collection_error | collection errors | rejected; PASS | G4_hostile.json controls[27] |
| H6_hang_after_summary | expected exits | rejected; PASS | G4_hostile.json controls[28] |
| D10_timeout | forge outcome: timeout | rejected; PASS | G4_hostile.json controls[29] |
| D10_window_closed | forge outcome: window-closed | rejected; PASS | G4_hostile.json controls[30] |
| H5_retry_75 | not admitted then admitted | refusal rejected; retry admitted; PASS | G4_hostile.json controls[31] |
| H5_retry_76 | not admitted then admitted | refusal rejected; retry admitted; PASS | G4_hostile.json controls[32] |
| One_byte_engine_copy | engine copy differs from pin | rejected; PASS | G4_hostile.json controls[33] |
| Registered_refusal_positive | GroupNorm-origin TypeError admitted as kill | admitted; PASS | G4_hostile.json controls[34] |
| Full_valid_synthetic | positive recomputation | admitted; PASS | G4_hostile.json controls[35] |
| Stored_parent_tamper | stored parent receipt disagrees | rejected; PASS | G4_hostile.json controls[36] |
| Expected_exit_tamper | gpu_lines expected exits | rejected; PASS | G4_hostile.json controls[37] |
| START_seal_tamper | START seal | rejected; PASS | G4_hostile.json controls[38] |
| D6_normal | legacy failure same class accepted | admitted; PASS | G4_hostile.json controls[39] |
| D6_regression_new_failure | baseline-passing | rejected; PASS | G4_hostile.json controls[40] |
| D6_regression_changed_class | different exception class | rejected; PASS | G4_hostile.json controls[41] |
| D6_regression_missing_pass | baseline-passing | rejected; PASS | G4_hostile.json controls[42] |
| D6_regression_skip_pass | baseline-passing | rejected; PASS | G4_hostile.json controls[43] |
| D6_regression_legacy_error | legacy setup error same class accepted | admitted; PASS | G4_hostile.json controls[44] |
| D6_regression_new_setup_error | baseline-passing | rejected; PASS | G4_hostile.json controls[45] |
| D6_regression_error_changed_class | different exception class | rejected; PASS | G4_hostile.json controls[46] |
| D6_dual_failure | legacy call failure plus teardown error accepted | admitted; PASS | G4_hostile.json controls[47] |
