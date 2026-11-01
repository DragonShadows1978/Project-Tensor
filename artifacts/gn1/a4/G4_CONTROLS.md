# G4 final controls

131/131 targets PASS: 107 inherited and 24 A4. Each named receipt records the deciding raw parent/scorer results. Negative controls pass only when the requested refusal/admission/cause occurs. Malformed-row controls are explicitly synthetic; real runner interruption cases use the installed runner with scratch resources.

| Control | Result | Deciding rule | Receipt / record |
|---|---|---|---|
| H1 | PASS | START.json | G4_hostile.json / 0 |
| H1b | PASS | run-root guard | G4_hostile.json / 1 |
| H2_REV1_exact | PASS | device-pattern failure | G4_hostile.json / 2 |
| H2c_REV1_exact | PASS | device-pattern failure | G4_hostile.json / 3 |
| H3_foreign_arm | PASS | arm/suite binding | G4_hostile.json / 4 |
| H3b_suite_swap | PASS | expected exits | G4_hostile.json / 5 |
| H4_junit_foreign_ID | PASS | unregistered JUnit ID | G4_hostile.json / 6 |
| Junit_one_ID_disagreement | PASS | JUnit/summary outcomes disagree | G4_hostile.json / 7 |
| Junit_duplicate_pass | PASS | duplicate JUnit ID | G4_hostile.json / 8 |
| H2_pattern_00 | PASS | device-pattern failure | G4_hostile.json / 9 |
| H2_pattern_01 | PASS | device-pattern failure | G4_hostile.json / 10 |
| H2_pattern_02 | PASS | device-pattern failure | G4_hostile.json / 11 |
| H2_pattern_03 | PASS | device-pattern failure | G4_hostile.json / 12 |
| H2_pattern_04 | PASS | device-pattern failure | G4_hostile.json / 13 |
| H2_pattern_05 | PASS | device-pattern failure | G4_hostile.json / 14 |
| H2_pattern_06 | PASS | device-pattern failure | G4_hostile.json / 15 |
| H2_pattern_07 | PASS | device-pattern failure | G4_hostile.json / 16 |
| H2_pattern_08 | PASS | device-pattern failure | G4_hostile.json / 17 |
| H2_pattern_09 | PASS | device-pattern failure | G4_hostile.json / 18 |
| H2_pattern_10 | PASS | device-pattern failure | G4_hostile.json / 19 |
| H2_pattern_11 | PASS | device-pattern failure | G4_hostile.json / 20 |
| Reference_only_kill | PASS | no real non-reference | G4_hostile.json / 21 |
| Real_engine_device_veto | PASS | device-pattern failure | G4_hostile.json / 22 |
| Real_engine_generic_kill | PASS | same-root candidate pass / GroupNorm call / no device pattern | G4_hostile.json / 23 |
| P2_callee_origin | PASS | no real non-reference | G4_hostile.json / 24 |
| Real_engine_candidate_not_passed | PASS | same-root candidate focus must pass input | G4_hostile.json / 25 |
| Wrong_nn_hash | PASS | nn.py binding | G4_hostile.json / 26 |
| Wrong_engine_hash | PASS | engine binding | G4_hostile.json / 27 |
| Setup_failure | PASS | setup/teardown failures | G4_hostile.json / 28 |
| Collection_error | PASS | collection errors | G4_hostile.json / 29 |
| H6_hang_after_summary | PASS | expected exits | G4_hostile.json / 30 |
| D10_timeout | PASS | forge outcome: timeout | G4_hostile.json / 31 |
| D10_window_closed | PASS | forge outcome: window-closed | G4_hostile.json / 32 |
| H5_retry_75 | PASS | not admitted then admitted | G4_hostile.json / 33 |
| H5_retry_76 | PASS | not admitted then admitted | G4_hostile.json / 34 |
| One_byte_engine_copy | PASS | engine copy differs from pin | G4_hostile.json / 35 |
| Registered_refusal_positive | PASS | GroupNorm-origin TypeError admitted as kill | G4_hostile.json / 36 |
| Full_valid_synthetic | PASS | positive recomputation | G4_hostile.json / 37 |
| Stored_parent_tamper | PASS | stored parent receipt disagrees | G4_hostile.json / 38 |
| Expected_exit_tamper | PASS | gpu_lines expected exits | G4_hostile.json / 39 |
| START_seal_tamper | PASS | START seal | G4_hostile.json / 40 |
| D6_normal | PASS | legacy failure same class accepted | G4_hostile.json / 41 |
| D6_regression_new_failure | PASS | baseline-passing | G4_hostile.json / 42 |
| D6_regression_changed_class | PASS | different exception class | G4_hostile.json / 43 |
| D6_regression_missing_pass | PASS | baseline-passing | G4_hostile.json / 44 |
| D6_regression_skip_pass | PASS | baseline-passing | G4_hostile.json / 45 |
| D6_regression_legacy_error | PASS | legacy setup error same class accepted | G4_hostile.json / 46 |
| D6_regression_new_setup_error | PASS | baseline-passing | G4_hostile.json / 47 |
| D6_regression_error_changed_class | PASS | different exception class | G4_hostile.json / 48 |
| D6_dual_failure | PASS | legacy call failure plus teardown error accepted | G4_hostile.json / 49 |
| Real_baseline_D5_registration | PASS | real GN test outcomes and registered baseline D5 exception classes | G4_hostile.json / 50 |
| Resolved_TMPDIR_escape | PASS | TMPDIR must resolve to /tmp/gn1_a4_* | G4_hostile.json / 51 |
| Runner_no_window75_H5 | PASS | no-window; None; not_admitted | G4_runner.json / 0 |
| Runner_no_window64_usage | PASS | no-window; None; not_admitted | G4_runner.json / 1 |
| Runner_no_lock76 | PASS | no-lock; None; not_admitted | G4_runner.json / 2 |
| B1_rechecked_out_retry | PASS | rechecked-out; None; not_admitted | G4_runner.json / 3 |
| B2_runner_deadline | PASS | timeout; runner-deadline; admitted | G4_runner.json / 4 |
| B3_window_closed | PASS | window-closed; window-closed; admitted | G4_runner.json / 5 |
| B4_outer_timeout | PASS | log-exit-mismatch; outer-timeout; admitted | G4_runner.json / 6 |
| B5_inner_timeout124 | PASS | ran; inner-timeout; admitted | G4_runner.json / 7 |
| B5_inner_timeout137 | PASS | ran; inner-timeout; admitted | G4_runner.json / 8 |
| Runner_ran_candidate_focus | PASS | valid completed CPU stand-in; ran | G4_runner.json / 9 |
| Runner_ran_baseline_focus | PASS | valid completed CPU stand-in; ran | G4_runner.json / 10 |
| Runner_ran_cannot_start126 | PASS | ran; None; admitted | G4_runner.json / 11 |
| Runner_ran_cannot_start127 | PASS | ran; None; admitted | G4_runner.json / 12 |
| A_H4_flat | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / 0 |
| A_H2_flat | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / 1 |
| A_H3_flat | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / 2 |
| A_H3b_flat | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / 3 |
| A_H2c_flat | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / 4 |
| A_H6_flat | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / 5 |
| A_H5_first_flat | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / 6 |
| A_H5_retry_flat | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / 7 |
| A_scorer_flat | PASS | G5 not passed | hostile_rev2_acd.json / 8 |
| A_H4_attempt | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / 9 |
| A_H2_attempt | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / 10 |
| A_H3_attempt | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / 11 |
| A_H3b_attempt | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / 12 |
| A_H2c_attempt | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / 13 |
| A_H6_attempt | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / 14 |
| A_H5_first_attempt | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / 15 |
| A_H5_retry_attempt | PASS | rejected (no pass, no kill) | hostile_rev2_acd.json / 16 |
| A_scorer_attempt | PASS | G5 not passed | hostile_rev2_acd.json / 17 |
| A_H1 | PASS | scorer exit 1, G5 not passed | hostile_rev2_acd.json / 18 |
| A_H1b | PASS | scorer refuses root outside a2; writes nothing there | hostile_rev2_acd.json / 19 |
| C1_equivalent_V3_fails | PASS | rejected by: equivalent mutant did not pass | hostile_rev2_acd.json / 20 |
| C2_forge_log_exit_mismatch | PASS | rejected by: log-exit-mismatch | hostile_rev2_acd.json / 21 |
| C3_census_missing_id | PASS | rejected by: test-ID census mismatch | hostile_rev2_acd.json / 22 |
| C4_unregistered_skip | PASS | rejected by: unregistered skip | hostile_rev2_acd.json / 23 |
| C5_baseline_extra_failure | PASS | rejected by: baseline failures differ | hostile_rev2_acd.json / 24 |
| C6_exit0_with_failure_regression | PASS | rejected by: pytest exit inconsistent | hostile_rev2_acd.json / 25 |
| C7_child_present_normal | PASS | evaluate incomplete child; no retry past a child result | hostile_rev2_acd.json / 26 |
| C7_child_present_hang_failure | PASS | evaluate incomplete child; no retry past a child result | hostile_rev2_acd.json / 27 |
| C7_child_never_started | PASS | incomplete not admitted; retry allowed | hostile_rev2_acd.json / 28 |
| C8_full_positive | PASS | every label admitted and G5 passed | hostile_rev2_acd.json / 29 |
| D_H2_bound | PASS | rejected by: device-pattern failure | hostile_rev2_acd.json / 30 |
| D_H2c_bound | PASS | rejected by: device-pattern failure | hostile_rev2_acd.json / 31 |
| D_H3_bound | PASS | rejected by: arm/suite binding | hostile_rev2_acd.json / 32 |
| D_H3b_bound | PASS | rejected by: expected exits | hostile_rev2_acd.json / 33 |
| D_kill_plus_teardown_device | PASS | rejected by: device-pattern failure | hostile_rev2_acd.json / 34 |
| D_honest_kill_text_mentions_pattern | PASS | rejected by: device-pattern failure | hostile_rev2_acd.json / 35 |
| D_H4_bound | PASS | rejected by: unregistered JUnit ID | hostile_rev2_acd.json / 36 |
| D_H5_real_forge_no_window_then_retry | PASS | not admitted, retry admitted and selected | hostile_rev2_acd.json / 37 |
| Stored_timeout_cause_tamper | PASS | stored timeout cause differs from log | G4_extra.json / 0 |
| Import_helper_pin_parent | PASS | import module seal mismatch: scripts/apa_om135_core.py | G4_extra.json / 1 |
| Import_helper_pin_child | PASS | import module seal mismatch: scripts/apa_om135_core.py | G4_extra.json / 2 |
| Import_helper_pin_scorer | PASS | import module seal mismatch: scripts/apa_om135_core.py | G4_extra.json / 3 |
| B6_lock_outer_retry | PASS | D2 no start operands -> not admitted; retry admitted | G4_a4.json / 0 |
| B7_grace_no_result | PASS | D3 outer fires in runner grace -> outer-timeout; D2 no retry | G4_a4.json / 1 |
| B7b_grace_result | PASS | D3 outer fires in runner grace -> outer-timeout; D2 no retry | G4_a4.json / 2 |
| B8_window_grace | PASS | D3 window-closed/143 with exit124 -> outer-timeout | G4_a4.json / 3 |
| C7b_started_no_result | PASS | D2/D4 started incomplete child admitted and failed; no retry | G4_a4.json / 4 |
| E5_truncated_incomplete | PASS | D4 JSON parse problem recorded; clean retry refusal | G4_a4.json / 5 |
| E5_truncated_completed | PASS | D4 evaluate never raises; JSON reason in parent and scorer | G4_a4.json / 6 |
| D5_sealed_tamper | PASS | D5 sealed runtime digest mismatch veto | G4_a4.json / 7 |
| D5_unsealed | PASS | D5 unknown origins listed without veto | G4_a4.json / 8 |
| D6_live_E_npbool_g | PASS | D6 arm interpreter NumPy 2.4.2 rejects numpy.bool_ through operator.index | G4_a4.json / 9 |
| D6_live_E_npbool_c | PASS | D6 arm interpreter NumPy 2.4.2 rejects numpy.bool_ through operator.index | G4_a4.json / 10 |
| D6_acceptance_invalidates | PASS | D6 live acceptance invalidates equivalence | G4_a4.json / 11 |
| D7_outer | PASS | D7 exact outer binding to gpu_lines.json | G4_a4.json / 12 |
| D7_maximum | PASS | D7 exact maximum binding to gpu_lines.json | G4_a4.json / 13 |
| D7_minimum | PASS | D7 exact minimum binding to gpu_lines.json | G4_a4.json / 14 |
| D7_inner | PASS | D7 exact inner binding to gpu_lines.json | G4_a4.json / 15 |
| D7_expected_exits | PASS | D7 exact expected_exits binding to gpu_lines.json | G4_a4.json / 16 |
| D3_evaluate_never_raises | PASS | D3 attribution exception becomes a parent reason; scorer rejects | G4_a4.json / 17 |
| D2_start_marker | PASS | D2 positive start evidence independently admits incomplete failure | G4_a4.json / 18 |
| D2_start_invocation | PASS | D2 positive start evidence independently admits incomplete failure | G4_a4.json / 19 |
| D2_start_ran | PASS | D2 positive start evidence independently admits incomplete failure | G4_a4.json / 20 |
| D2_start_timeout | PASS | D2 positive start evidence independently admits incomplete failure | G4_a4.json / 21 |
| D2_start_window-closed | PASS | D2 positive start evidence independently admits incomplete failure | G4_a4.json / 22 |
| D2_start_junit | PASS | D2 positive start evidence independently admits incomplete failure | G4_a4.json / 23 |
