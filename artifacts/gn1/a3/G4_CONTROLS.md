# G4 hostile controls

107/107 final controls meet their registered targets. These are CPU instrument/supervision checks. See HOSTILE_ADAPTATIONS.md for all 48 A2 adaptations.

| Control | Result | Deciding rule / target | Receipt |
| --- | --- | --- | --- |
| H1 | PASS | START.json | G4_hostile.json / controls[0] |
| H1b | PASS | run-root guard | G4_hostile.json / controls[1] |
| H2_REV1_exact | PASS | device-pattern failure | G4_hostile.json / controls[2] |
| H2c_REV1_exact | PASS | device-pattern failure | G4_hostile.json / controls[3] |
| H3_foreign_arm | PASS | arm/suite binding | G4_hostile.json / controls[4] |
| H3b_suite_swap | PASS | expected exits | G4_hostile.json / controls[5] |
| H4_junit_foreign_ID | PASS | unregistered JUnit ID | G4_hostile.json / controls[6] |
| Junit_one_ID_disagreement | PASS | JUnit/summary outcomes disagree | G4_hostile.json / controls[7] |
| Junit_duplicate_pass | PASS | duplicate JUnit ID | G4_hostile.json / controls[8] |
| H2_pattern_00 | PASS | device-pattern failure | G4_hostile.json / controls[9] |
| H2_pattern_01 | PASS | device-pattern failure | G4_hostile.json / controls[10] |
| H2_pattern_02 | PASS | device-pattern failure | G4_hostile.json / controls[11] |
| H2_pattern_03 | PASS | device-pattern failure | G4_hostile.json / controls[12] |
| H2_pattern_04 | PASS | device-pattern failure | G4_hostile.json / controls[13] |
| H2_pattern_05 | PASS | device-pattern failure | G4_hostile.json / controls[14] |
| H2_pattern_06 | PASS | device-pattern failure | G4_hostile.json / controls[15] |
| H2_pattern_07 | PASS | device-pattern failure | G4_hostile.json / controls[16] |
| H2_pattern_08 | PASS | device-pattern failure | G4_hostile.json / controls[17] |
| H2_pattern_09 | PASS | device-pattern failure | G4_hostile.json / controls[18] |
| H2_pattern_10 | PASS | device-pattern failure | G4_hostile.json / controls[19] |
| H2_pattern_11 | PASS | device-pattern failure | G4_hostile.json / controls[20] |
| Reference_only_kill | PASS | no real non-reference | G4_hostile.json / controls[21] |
| Real_engine_device_veto | PASS | device-pattern failure | G4_hostile.json / controls[22] |
| Real_engine_generic_kill | PASS | same-root candidate pass / GroupNorm call / no device pattern | G4_hostile.json / controls[23] |
| P2_callee_origin | PASS | no real non-reference | G4_hostile.json / controls[24] |
| Real_engine_candidate_not_passed | PASS | same-root candidate focus must pass input | G4_hostile.json / controls[25] |
| Wrong_nn_hash | PASS | nn.py binding | G4_hostile.json / controls[26] |
| Wrong_engine_hash | PASS | engine binding | G4_hostile.json / controls[27] |
| Setup_failure | PASS | setup/teardown failures | G4_hostile.json / controls[28] |
| Collection_error | PASS | collection errors | G4_hostile.json / controls[29] |
| H6_hang_after_summary | PASS | expected exits | G4_hostile.json / controls[30] |
| D10_timeout | PASS | forge outcome: timeout | G4_hostile.json / controls[31] |
| D10_window_closed | PASS | forge outcome: window-closed | G4_hostile.json / controls[32] |
| H5_retry_75 | PASS | not admitted then admitted | G4_hostile.json / controls[33] |
| H5_retry_76 | PASS | not admitted then admitted | G4_hostile.json / controls[34] |
| One_byte_engine_copy | PASS | engine copy differs from pin | G4_hostile.json / controls[35] |
| Registered_refusal_positive | PASS | GroupNorm-origin TypeError admitted as kill | G4_hostile.json / controls[36] |
| Full_valid_synthetic | PASS | positive recomputation | G4_hostile.json / controls[37] |
| Stored_parent_tamper | PASS | stored parent receipt disagrees | G4_hostile.json / controls[38] |
| Expected_exit_tamper | PASS | gpu_lines expected exits | G4_hostile.json / controls[39] |
| START_seal_tamper | PASS | START seal | G4_hostile.json / controls[40] |
| D6_normal | PASS | legacy failure same class accepted | G4_hostile.json / controls[41] |
| D6_regression_new_failure | PASS | baseline-passing | G4_hostile.json / controls[42] |
| D6_regression_changed_class | PASS | different exception class | G4_hostile.json / controls[43] |
| D6_regression_missing_pass | PASS | baseline-passing | G4_hostile.json / controls[44] |
| D6_regression_skip_pass | PASS | baseline-passing | G4_hostile.json / controls[45] |
| D6_regression_legacy_error | PASS | legacy setup error same class accepted | G4_hostile.json / controls[46] |
| D6_regression_new_setup_error | PASS | baseline-passing | G4_hostile.json / controls[47] |
| D6_regression_error_changed_class | PASS | different exception class | G4_hostile.json / controls[48] |
| D6_dual_failure | PASS | legacy call failure plus teardown error accepted | G4_hostile.json / controls[49] |
| Real_baseline_D5_registration | PASS | real GN test outcomes and registered baseline D5 exception classes | G4_hostile.json / controls[50] |
| Resolved_TMPDIR_escape | PASS | TMPDIR must resolve to /tmp/gn1_a3_* | G4_hostile.json / controls[51] |
| Runner_no_window75_H5 | PASS | no-window; None; not_admitted | G4_runner.json / controls[0] |
| Runner_no_window64_usage | PASS | no-window; None; admitted | G4_runner.json / controls[1] |
| Runner_no_lock76 | PASS | no-lock; None; not_admitted | G4_runner.json / controls[2] |
| B1_rechecked_out_retry | PASS | rechecked-out; None; not_admitted | G4_runner.json / controls[3] |
| B2_runner_deadline | PASS | timeout; runner-deadline; admitted | G4_runner.json / controls[4] |
| B3_window_closed | PASS | window-closed; window-closed; admitted | G4_runner.json / controls[5] |
| B4_outer_timeout | PASS | log-exit-mismatch; outer-timeout; admitted | G4_runner.json / controls[6] |
| B5_inner_timeout124 | PASS | ran; inner-timeout; admitted | G4_runner.json / controls[7] |
| B5_inner_timeout137 | PASS | ran; inner-timeout; admitted | G4_runner.json / controls[8] |
| Runner_ran_candidate_focus | PASS | valid completed CPU stand-in; ran | G4_runner.json / controls[9] |
| Runner_ran_baseline_focus | PASS | valid completed CPU stand-in; ran | G4_runner.json / controls[10] |
| Runner_ran_cannot_start126 | PASS | ran; None; admitted | G4_runner.json / controls[11] |
| Runner_ran_cannot_start127 | PASS | ran; None; admitted | G4_runner.json / controls[12] |
| A_H4_flat | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / results[0] |
| A_H2_flat | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / results[1] |
| A_H3_flat | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / results[2] |
| A_H3b_flat | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / results[3] |
| A_H2c_flat | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / results[4] |
| A_H6_flat | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / results[5] |
| A_H5_first_flat | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / results[6] |
| A_H5_retry_flat | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / results[7] |
| A_scorer_flat | PASS | G5 not passed | hostile_rev2_acd.json / results[8] |
| A_H4_attempt | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / results[9] |
| A_H2_attempt | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / results[10] |
| A_H3_attempt | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / results[11] |
| A_H3b_attempt | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / results[12] |
| A_H2c_attempt | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / results[13] |
| A_H6_attempt | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / results[14] |
| A_H5_first_attempt | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / results[15] |
| A_H5_retry_attempt | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / results[16] |
| A_scorer_attempt | PASS | G5 not passed | hostile_rev2_acd.json / results[17] |
| A_H1 | PASS | scorer exit 1, G5 not passed | hostile_rev2_acd.json / results[18] |
| A_H1b | PASS | scorer refuses root outside a2; writes nothing there | hostile_rev2_acd.json / results[19] |
| C1_equivalent_V3_fails | PASS | rejected by: equivalent mutant did not pass | hostile_rev2_acd.json / results[20] |
| C2_forge_log_exit_mismatch | PASS | rejected by: log-exit-mismatch | hostile_rev2_acd.json / results[21] |
| C3_census_missing_id | PASS | rejected by: test-ID census mismatch | hostile_rev2_acd.json / results[22] |
| C4_unregistered_skip | PASS | rejected by: unregistered skip | hostile_rev2_acd.json / results[23] |
| C5_baseline_extra_failure | PASS | rejected by: baseline failures differ | hostile_rev2_acd.json / results[24] |
| C6_exit0_with_failure_regression | PASS | rejected by: pytest exit inconsistent | hostile_rev2_acd.json / results[25] |
| C7_child_present_normal | PASS | evaluate incomplete child; no retry past a child result | hostile_rev2_acd.json / results[26] |
| C7_child_present_hang_failure | PASS | evaluate incomplete child; no retry past a child result | hostile_rev2_acd.json / results[27] |
| C7_child_never_started | PASS | incomplete not admitted; retry allowed | hostile_rev2_acd.json / results[28] |
| C8_full_positive | PASS | every label admitted and G5 passed | hostile_rev2_acd.json / results[29] |
| D_H2_bound | PASS | rejected by: device-pattern failure | hostile_rev2_acd.json / results[30] |
| D_H2c_bound | PASS | rejected by: device-pattern failure | hostile_rev2_acd.json / results[31] |
| D_H3_bound | PASS | rejected by: arm/suite binding | hostile_rev2_acd.json / results[32] |
| D_H3b_bound | PASS | rejected by: expected exits | hostile_rev2_acd.json / results[33] |
| D_kill_plus_teardown_device | PASS | rejected by: device-pattern failure | hostile_rev2_acd.json / results[34] |
| D_honest_kill_text_mentions_pattern | PASS | rejected by: device-pattern failure | hostile_rev2_acd.json / results[35] |
| D_H4_bound | PASS | rejected by: unregistered JUnit ID | hostile_rev2_acd.json / results[36] |
| D_H5_real_forge_no_window_then_retry | PASS | not admitted, retry admitted and selected | hostile_rev2_acd.json / results[37] |
| Stored_timeout_cause_tamper | PASS | stored timeout cause differs from log | G4_extra.json / controls[0] |
| Import_helper_pin_parent | PASS | import module seal mismatch: scripts/apa_om135_core.py | G4_extra.json / controls[1] |
| Import_helper_pin_child | PASS | import module seal mismatch: scripts/apa_om135_core.py | G4_extra.json / controls[2] |
| Import_helper_pin_scorer | PASS | import module seal mismatch: scripts/apa_om135_core.py | G4_extra.json / controls[3] |
