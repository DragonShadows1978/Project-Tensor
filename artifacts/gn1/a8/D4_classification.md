# D4 classification under C

Names, targets and deciding rules in A6 JSON receipts; A7 added controls have no result JSON, so registration is the explicit census source.

| Source | Control | Class | Target | Deciding rule |
|---|---|---|---|---|
| a6/G4_hostile.json | H1 | ii | START.json | START.json |
| a6/G4_hostile.json | H1b | i | run-root guard | run-root guard |
| a6/G4_hostile.json | H2_REV1_exact | i | device-pattern failure | device-pattern failure |
| a6/G4_hostile.json | H2c_REV1_exact | i | device-pattern failure | device-pattern failure |
| a6/G4_hostile.json | H3_foreign_arm | i | arm/suite binding | arm/suite binding |
| a6/G4_hostile.json | H3b_suite_swap | i | expected exits | expected exits |
| a6/G4_hostile.json | H4_junit_foreign_ID | i | unregistered JUnit ID | unregistered JUnit ID |
| a6/G4_hostile.json | Junit_one_ID_disagreement | i | JUnit/summary outcomes disagree | JUnit/summary outcomes disagree |
| a6/G4_hostile.json | Junit_duplicate_pass | i | duplicate JUnit ID | duplicate JUnit ID |
| a6/G4_hostile.json | H2_pattern_00 | i | device-pattern failure | device-pattern failure |
| a6/G4_hostile.json | H2_pattern_01 | i | device-pattern failure | device-pattern failure |
| a6/G4_hostile.json | H2_pattern_02 | i | device-pattern failure | device-pattern failure |
| a6/G4_hostile.json | H2_pattern_03 | i | device-pattern failure | device-pattern failure |
| a6/G4_hostile.json | H2_pattern_04 | i | device-pattern failure | device-pattern failure |
| a6/G4_hostile.json | H2_pattern_05 | i | device-pattern failure | device-pattern failure |
| a6/G4_hostile.json | H2_pattern_06 | i | device-pattern failure | device-pattern failure |
| a6/G4_hostile.json | H2_pattern_07 | i | device-pattern failure | device-pattern failure |
| a6/G4_hostile.json | H2_pattern_08 | i | device-pattern failure | device-pattern failure |
| a6/G4_hostile.json | H2_pattern_09 | i | device-pattern failure | device-pattern failure |
| a6/G4_hostile.json | H2_pattern_10 | i | device-pattern failure | device-pattern failure |
| a6/G4_hostile.json | H2_pattern_11 | i | device-pattern failure | device-pattern failure |
| a6/G4_hostile.json | Reference_only_kill | i | no real non-reference | no real non-reference |
| a6/G4_hostile.json | Real_engine_device_veto | i | device-pattern failure | device-pattern failure |
| a6/G4_hostile.json | Real_engine_generic_kill | i | same-root candidate pass / GroupNorm call / no device pattern | same-root candidate pass / GroupNorm call / no device pattern |
| a6/G4_hostile.json | P2_callee_origin | i | no real non-reference | no real non-reference |
| a6/G4_hostile.json | Real_engine_candidate_not_passed | i | same-root candidate focus must pass input | same-root candidate focus must pass input |
| a6/G4_hostile.json | Wrong_nn_hash | i | nn.py binding | nn.py binding |
| a6/G4_hostile.json | Wrong_engine_hash | i | engine binding | engine binding |
| a6/G4_hostile.json | Setup_failure | i | setup/teardown failures | setup/teardown failures |
| a6/G4_hostile.json | Collection_error | i | collection errors | collection errors |
| a6/G4_hostile.json | H6_hang_after_summary | i | expected exits | expected exits |
| a6/G4_hostile.json | D10_timeout | i | forge outcome: timeout | forge outcome: timeout |
| a6/G4_hostile.json | D10_window_closed | i | forge outcome: window-closed | forge outcome: window-closed |
| a6/G4_hostile.json | H5_retry_75 | i | not admitted then admitted | not admitted then admitted |
| a6/G4_hostile.json | H5_retry_76 | i | not admitted then admitted | not admitted then admitted |
| a6/G4_hostile.json | One_byte_engine_copy | i | engine copy differs from pin | engine copy differs from pin |
| a6/G4_hostile.json | Registered_refusal_positive | i | GroupNorm-origin TypeError admitted as kill | GroupNorm-origin TypeError admitted as kill |
| a6/G4_hostile.json | Full_valid_synthetic | i | positive recomputation | D1 positive line sensitivity: all 58 lines pass, control root cannot establish GPU acceptance |
| a6/G4_hostile.json | Stored_parent_tamper | iii | stored parent receipt disagrees | stored parent receipt disagrees |
| a6/G4_hostile.json | Expected_exit_tamper | iii | gpu_lines expected exits | gpu_lines expected exits |
| a6/G4_hostile.json | START_seal_tamper | iii | START seal | START seal |
| a6/G4_hostile.json | D6_normal | i | legacy failure same class accepted | legacy failure same class accepted |
| a6/G4_hostile.json | D6_regression_new_failure | i | baseline-passing | baseline-passing |
| a6/G4_hostile.json | D6_regression_changed_class | i | different exception class | different exception class |
| a6/G4_hostile.json | D6_regression_missing_pass | i | baseline-passing | baseline-passing |
| a6/G4_hostile.json | D6_regression_skip_pass | i | baseline-passing | baseline-passing |
| a6/G4_hostile.json | D6_regression_legacy_error | i | legacy setup error same class accepted | legacy setup error same class accepted |
| a6/G4_hostile.json | D6_regression_new_setup_error | i | baseline-passing | baseline-passing |
| a6/G4_hostile.json | D6_regression_error_changed_class | i | different exception class | different exception class |
| a6/G4_hostile.json | D6_dual_failure | i | legacy call failure plus teardown error accepted | legacy call failure plus teardown error accepted |
| a6/G4_hostile.json | Real_baseline_D5_registration | i | real GN test outcomes and registered baseline D5 exception classes | real GN test outcomes and registered baseline D5 exception classes |
| a6/G4_hostile.json | Resolved_TMPDIR_escape | i | TMPDIR must resolve to /tmp/gn1_a6_* | TMPDIR must resolve to /tmp/gn1_a6_* |
| a6/G4_runner.json | Runner_no_window75_H5 | i | no-window; None; not_admitted | no-window; None; not_admitted |
| a6/G4_runner.json | Runner_no_window64_usage | i | no-window; None; not_admitted | no-window; None; not_admitted |
| a6/G4_runner.json | Runner_no_lock76 | i | no-lock; None; not_admitted | no-lock; None; not_admitted |
| a6/G4_runner.json | B1_rechecked_out_retry | i | rechecked-out; None; not_admitted | rechecked-out; None; not_admitted |
| a6/G4_runner.json | B2_runner_deadline | i | timeout; runner-deadline; admitted | timeout; runner-deadline; admitted |
| a6/G4_runner.json | B3_window_closed | i | window-closed; window-closed; admitted | window-closed; window-closed; admitted |
| a6/G4_runner.json | B4_outer_timeout | i | log-exit-mismatch; outer-timeout; admitted | log-exit-mismatch; outer-timeout; admitted |
| a6/G4_runner.json | B5_inner_timeout124 | i | ran; inner-timeout; admitted | ran; inner-timeout; admitted |
| a6/G4_runner.json | B5_inner_timeout137 | i | ran; inner-timeout; admitted | ran; inner-timeout; admitted |
| a6/G4_runner.json | Runner_ran_candidate_focus | i | valid completed CPU stand-in; ran | valid completed CPU stand-in; ran |
| a6/G4_runner.json | Runner_ran_baseline_focus | i | valid completed CPU stand-in; ran | valid completed CPU stand-in; ran |
| a6/G4_runner.json | Runner_ran_cannot_start126 | i | ran; None; admitted | ran; None; admitted |
| a6/G4_runner.json | Runner_ran_cannot_start127 | i | ran; None; admitted | ran; None; admitted |
| a6/hostile_rev2_acd.json | A_H4_flat | i | rejected (no pass, no kill) | rejected (no pass, no kill) |
| a6/hostile_rev2_acd.json | A_H2_flat | i | rejected (no pass, no kill) | rejected (no pass, no kill) |
| a6/hostile_rev2_acd.json | A_H3_flat | i | rejected (no pass, no kill) | rejected (no pass, no kill) |
| a6/hostile_rev2_acd.json | A_H3b_flat | i | rejected (no pass, no kill) | rejected (no pass, no kill) |
| a6/hostile_rev2_acd.json | A_H2c_flat | i | rejected (no pass, no kill) | rejected (no pass, no kill) |
| a6/hostile_rev2_acd.json | A_H6_flat | i | rejected (no pass, no kill) | rejected (no pass, no kill) |
| a6/hostile_rev2_acd.json | A_H5_first_flat | i | rejected (no pass, no kill) | rejected (no pass, no kill) |
| a6/hostile_rev2_acd.json | A_H5_retry_flat | i | rejected (no pass, no kill) | rejected (no pass, no kill) |
| a6/hostile_rev2_acd.json | A_scorer_flat | i | G5 not passed | G5 not passed |
| a6/hostile_rev2_acd.json | A_H4_attempt | i | rejected (no pass, no kill) | rejected (no pass, no kill) |
| a6/hostile_rev2_acd.json | A_H2_attempt | i | rejected (no pass, no kill) | rejected (no pass, no kill) |
| a6/hostile_rev2_acd.json | A_H3_attempt | i | rejected (no pass, no kill) | rejected (no pass, no kill) |
| a6/hostile_rev2_acd.json | A_H3b_attempt | i | rejected (no pass, no kill) | rejected (no pass, no kill) |
| a6/hostile_rev2_acd.json | A_H2c_attempt | i | rejected (no pass, no kill) | rejected (no pass, no kill) |
| a6/hostile_rev2_acd.json | A_H6_attempt | i | rejected (no pass, no kill) | rejected (no pass, no kill) |
| a6/hostile_rev2_acd.json | A_H5_first_attempt | i | rejected (no pass, no kill) | rejected (no pass, no kill) |
| a6/hostile_rev2_acd.json | A_H5_retry_attempt | i | rejected (no pass, no kill) | rejected (no pass, no kill) |
| a6/hostile_rev2_acd.json | A_scorer_attempt | i | G5 not passed | G5 not passed |
| a6/hostile_rev2_acd.json | A_H1 | ii | scorer exit 1, G5 not passed | scorer exit 1, G5 not passed |
| a6/hostile_rev2_acd.json | A_H1b | i | scorer refuses root outside a2; writes nothing there | scorer refuses root outside a2; writes nothing there |
| a6/hostile_rev2_acd.json | C1_equivalent_V3_fails | i | rejected by: equivalent mutant did not pass | rejected by: equivalent mutant did not pass |
| a6/hostile_rev2_acd.json | C2_forge_log_exit_mismatch | i | rejected by: log-exit-mismatch | rejected by: log-exit-mismatch |
| a6/hostile_rev2_acd.json | C3_census_missing_id | i | rejected by: test-ID census mismatch | rejected by: test-ID census mismatch |
| a6/hostile_rev2_acd.json | C4_unregistered_skip | i | rejected by: unregistered skip | rejected by: unregistered skip |
| a6/hostile_rev2_acd.json | C5_baseline_extra_failure | i | rejected by: baseline failures differ | rejected by: baseline failures differ |
| a6/hostile_rev2_acd.json | C6_exit0_with_failure_regression | i | rejected by: pytest exit inconsistent | rejected by: pytest exit inconsistent |
| a6/hostile_rev2_acd.json | C7_child_present_normal | ii | evaluate incomplete child; no retry past a child result | evaluate incomplete child; no retry past a child result |
| a6/hostile_rev2_acd.json | C7_child_present_hang_failure | ii | evaluate incomplete child; no retry past a child result | evaluate incomplete child; no retry past a child result |
| a6/hostile_rev2_acd.json | C7_child_never_started | ii | incomplete not admitted; retry allowed | incomplete not admitted; retry allowed |
| a6/hostile_rev2_acd.json | C8_full_positive | i | every label admitted and G5 passed | D1 positive line sensitivity: all 58 lines pass, control root cannot establish GPU acceptance |
| a6/hostile_rev2_acd.json | D_H2_bound | i | rejected by: device-pattern failure | rejected by: device-pattern failure |
| a6/hostile_rev2_acd.json | D_H2c_bound | i | rejected by: device-pattern failure | rejected by: device-pattern failure |
| a6/hostile_rev2_acd.json | D_H3_bound | i | rejected by: arm/suite binding | rejected by: arm/suite binding |
| a6/hostile_rev2_acd.json | D_H3b_bound | i | rejected by: expected exits | rejected by: expected exits |
| a6/hostile_rev2_acd.json | D_kill_plus_teardown_device | i | rejected by: device-pattern failure | rejected by: device-pattern failure |
| a6/hostile_rev2_acd.json | D_honest_kill_text_mentions_pattern | i | rejected by: device-pattern failure | rejected by: device-pattern failure |
| a6/hostile_rev2_acd.json | D_H4_bound | i | rejected by: unregistered JUnit ID | rejected by: unregistered JUnit ID |
| a6/hostile_rev2_acd.json | D_H5_real_forge_no_window_then_retry | i | not admitted, retry admitted and selected | not admitted, retry admitted and selected |
| a6/G4_extra.json | Stored_timeout_cause_tamper | iii | stored timeout cause differs from log | stored timeout cause differs from log |
| a6/G4_extra.json | Import_helper_pin_parent | iii | import module seal mismatch: scripts/apa_om135_core.py | import module seal mismatch: scripts/apa_om135_core.py |
| a6/G4_extra.json | Import_helper_pin_child | iii | import module seal mismatch: scripts/apa_om135_core.py | import module seal mismatch: scripts/apa_om135_core.py |
| a6/G4_extra.json | Import_helper_pin_scorer | iii | import module seal mismatch: scripts/apa_om135_core.py | import module seal mismatch: scripts/apa_om135_core.py |
| a6/G4_a4.json | B6_lock_outer_retry | i | D2 no start operands -> not admitted; retry admitted | D2 no start operands -> not admitted; retry admitted |
| a6/G4_a4.json | B7_grace_no_result | i | D3 outer fires in runner grace -> outer-timeout; D2 no retry | D3 outer fires in runner grace -> outer-timeout; D2 no retry |
| a6/G4_a4.json | B7b_grace_result | i | D3 outer fires in runner grace -> outer-timeout; D2 no retry | D3 outer fires in runner grace -> outer-timeout; D2 no retry |
| a6/G4_a4.json | B8_window_grace | i | D3 window-closed/143 with exit124 -> outer-timeout | D3 window-closed/143 with exit124 -> outer-timeout |
| a6/G4_a4.json | C7b_started_no_result | ii | D2/D4 started incomplete child admitted and failed; no retry | D2/D4 started incomplete child admitted and failed; no retry |
| a6/G4_a4.json | E5_truncated_incomplete | ii | D4 JSON parse problem recorded; clean retry refusal | D4 JSON parse problem recorded; clean retry refusal |
| a6/G4_a4.json | E5_truncated_completed | ii | D4 evaluate never raises; JSON reason in parent and scorer | D4 evaluate never raises; JSON reason in parent and scorer |
| a6/G4_a4.json | D5_sealed_tamper | i | D5 sealed runtime digest mismatch veto | D5 sealed runtime digest mismatch veto |
| a6/G4_a4.json | D5_unsealed | i | D5 unknown origins listed without veto | D5 unknown origins listed without veto |
| a6/G4_a4.json | D6_live_E_npbool_g | i | D6 arm interpreter NumPy 2.4.2 rejects numpy.bool_ through operator.index | D6 arm interpreter NumPy 2.4.2 rejects numpy.bool_ through operator.index |
| a6/G4_a4.json | D6_live_E_npbool_c | i | D6 arm interpreter NumPy 2.4.2 rejects numpy.bool_ through operator.index | D6 arm interpreter NumPy 2.4.2 rejects numpy.bool_ through operator.index |
| a6/G4_a4.json | D6_acceptance_invalidates | i | D6 live acceptance invalidates equivalence | D6 live acceptance invalidates equivalence |
| a6/G4_a4.json | D7_outer | iii | D7 exact outer binding to gpu_lines.json | D7 exact outer binding to gpu_lines.json |
| a6/G4_a4.json | D7_maximum | iii | D7 exact maximum binding to gpu_lines.json | D7 exact maximum binding to gpu_lines.json |
| a6/G4_a4.json | D7_minimum | iii | D7 exact minimum binding to gpu_lines.json | D7 exact minimum binding to gpu_lines.json |
| a6/G4_a4.json | D7_inner | iii | D7 exact inner binding to gpu_lines.json | D7 exact inner binding to gpu_lines.json |
| a6/G4_a4.json | D7_expected_exits | iii | D7 exact expected_exits binding to gpu_lines.json | D7 exact expected_exits binding to gpu_lines.json |
| a6/G4_a4.json | D3_evaluate_never_raises | i | D4 total attribution: non-start row plus124 becomes outer-timeout; scorer rejects | D4 total attribution: non-start row plus124 becomes outer-timeout; scorer rejects |
| a6/G4_a4.json | D2_start_marker | i | D2 positive start evidence independently admits incomplete failure | D2 positive start evidence independently admits incomplete failure |
| a6/G4_a4.json | D2_start_invocation | i | D2 positive start evidence independently admits incomplete failure | D2 positive start evidence independently admits incomplete failure |
| a6/G4_a4.json | D2_start_ran | i | D2 positive start evidence independently admits incomplete failure | D2 positive start evidence independently admits incomplete failure |
| a6/G4_a4.json | D2_start_timeout | i | D2 positive start evidence independently admits incomplete failure | D2 positive start evidence independently admits incomplete failure |
| a6/G4_a4.json | D2_start_window-closed | i | D2 positive start evidence independently admits incomplete failure | D2 positive start evidence independently admits incomplete failure |
| a6/G4_a4.json | D2_start_junit | i | D2 positive start evidence independently admits incomplete failure | D2 positive start evidence independently admits incomplete failure |
| a6/G4_a5.json | REV4_max_equals | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | REV4_min_abbrev | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | REV4_min_equals | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | REV4_inner_short_k | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | REV4_inner_kill_after_space | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | N2b_unset_grace | i | D3 normalised exit / outer-timeout, D2 short-bound refusal, no surviving child | D3 normalised exit / outer-timeout, D2 short-bound refusal, no surviving child |
| a6/G4_a5.json | N2c_unset_grace | i | D3 normalised exit / outer-timeout, D2 short-bound refusal, no surviving child | D3 normalised exit / outer-timeout, D2 short-bound refusal, no surviving child |
| a6/G4_a5.json | N2b_D3 | i | D3 normalised exit / outer-timeout, D2 short-bound refusal, no surviving child | D3 normalised exit / outer-timeout, D2 short-bound refusal, no surviving child |
| a6/G4_a5.json | N2c_D3 | i | D3 normalised exit / outer-timeout, D2 short-bound refusal, no surviving child | D3 normalised exit / outer-timeout, D2 short-bound refusal, no surviving child |
| a6/G4_a5.json | Raw_SIGKILL_normalisation | i | D3 raw negative return becomes137 before D4 outer-timeout attribution | D3 raw negative return becomes137 before D4 outer-timeout attribution |
| a6/G4_a5.json | N5 | i | D4 non-start row/124 returns outer-timeout, not-admitted receipt | D4 non-start row/124 returns outer-timeout, not-admitted receipt |
| a6/G4_a5.json | N4 | ii | D5 corrupt prior receipt refuses with JSON error, no traceback | D5 corrupt prior receipt refuses with JSON error, no traceback |
| a6/G4_a5.json | Atomic_publication | i | D5 complete linked receipt, no-overwrite publication | D5 complete linked receipt, no-overwrite publication |
| a6/G4_a5.json | Option_runner_000 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_001 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_002 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_003 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_004 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_005 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_006 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_007 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_008 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_009 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_010 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_011 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_012 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_013 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_014 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_015 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_016 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_017 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_018 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_019 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_020 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_021 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_022 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_023 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_024 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_025 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_026 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_027 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_028 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_029 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_030 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_031 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_032 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_033 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_034 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_035 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_036 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_037 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_038 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_039 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_040 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_041 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_042 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_043 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_runner_044 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_000 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_001 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_002 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_003 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_004 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_005 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_006 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_007 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_008 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_009 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_010 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_011 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_012 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_013 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_014 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_015 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_016 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_017 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_018 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_019 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_020 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_021 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_022 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_023 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_024 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_025 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_026 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_027 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_028 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_029 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_030 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_031 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_032 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_033 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_034 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_035 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_036 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_037 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_038 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_039 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_040 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_041 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_042 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_043 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_044 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_045 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_046 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_047 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_048 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_049 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_050 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_051 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_052 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_053 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_054 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_055 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_056 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_057 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_058 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_059 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_060 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_061 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_062 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_063 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_064 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_065 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_066 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_067 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_068 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_069 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_070 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_071 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_072 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_073 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_074 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Option_timeout_075 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Grammar_runner_equals_min | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Grammar_runner_equals_max | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Grammar_runner_equals_label | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Grammar_runner_reordered | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Grammar_runner_duplicate | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Grammar_runner_float_min | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Grammar_runner_plus_max | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Grammar_runner_zero_padded | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Grammar_duration_0 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Grammar_duration_1 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Grammar_duration_2 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Grammar_duration_3 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Grammar_duration_4 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Grammar_duration_5 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Grammar_duration_6 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_a5.json | Grammar_duration_7 | i | D2 exact runner/inner argv refusal | D2 exact runner/inner argv refusal |
| a6/G4_p116.json | N2b_D3_PID | i | P116/D3 observe actual child PID from start marker immediately after parent return | P116/D3 observe actual child PID from start marker immediately after parent return |
| a6/G4_p116.json | N2c_D3_PID | i | P116/D3 observe actual child PID from start marker immediately after parent return | P116/D3 observe actual child PID from start marker immediately after parent return |
| a6/G4_a6.json | CPU_control_root | i | D1 all 58 lines recompute; control root never establishes acceptance | D1 all 58 lines recompute; control root never establishes acceptance |
| a6/G4_a6.json | NEW_parent_result_deep_nesting | ii | D2 Exception catches RecursionError; summary written and affected line/root refuses | D2 Exception catches RecursionError; summary written and affected line/root refuses |
| a6/G4_a6.json | NEW_start_json_deep_nesting | ii | D2 Exception catches RecursionError; summary written and affected line/root refuses | D2 Exception catches RecursionError; summary written and affected line/root refuses |
| a6/G4_a6.json | NEW_regression_deep_child_result | ii | D2 Exception catches RecursionError; summary written and affected line/root refuses | D2 Exception catches RecursionError; summary written and affected line/root refuses |
| a6/G4_a6.json | raw_exit_minus9_exit0 | iii | D3 integer raw_exit and independently recomputed normalisation | D3 integer raw_exit and independently recomputed normalisation |
| a6/G4_a6.json | raw_exit_removed | iii | D3 integer raw_exit and independently recomputed normalisation | D3 integer raw_exit and independently recomputed normalisation |
| a6/G4_a6.json | raw_exit_bool | iii | D3 integer raw_exit and independently recomputed normalisation | D3 integer raw_exit and independently recomputed normalisation |
| a6/G4_a6.json | raw_exit_string | iii | D3 integer raw_exit and independently recomputed normalisation | D3 integer raw_exit and independently recomputed normalisation |
| a6/G4_a6.json | Q2_wrapped_runner_command0 | i | D1 pinned argv/program/environment refusal | D1 pinned argv/program/environment refusal |
| a6/G4_a6.json | D0c_runner_substitute | i | D1 pinned argv/program/environment refusal | D1 pinned argv/program/environment refusal |
| a6/G4_a6.json | D0c_timeout_wrapper | i | D1 pinned argv/program/environment refusal | D1 pinned argv/program/environment refusal |
| a6/G4_a6.json | D0c_interpreter_wrapper | i | D1 pinned argv/program/environment refusal | D1 pinned argv/program/environment refusal |
| a6/G4_a6.json | Q2_path_shadow_inner_timeout | i | D1 pinned argv/program/environment refusal | D1 pinned argv/program/environment refusal |
| a6/G4_a6.json | D0c_PATH_shadow | i | D1 pinned argv/program/environment refusal | D1 pinned argv/program/environment refusal |
| a6/G4_a6.json | Recorded_pin_runner | iii | program pin mismatch: runner | program pin mismatch: runner |
| a6/G4_a6.json | Recorded_pin_timeout | iii | program pin mismatch: timeout | program pin mismatch: timeout |
| a6/G4_a6.json | Recorded_pin_interpreter | iii | program pin mismatch: interpreter | program pin mismatch: interpreter |
| a6/G4_a6.json | Recorded_pin_runner_interpreter | iii | program pin mismatch: runner_interpreter | program pin mismatch: runner_interpreter |
| a6/G4_a6.json | Recorded_pin_parent_interpreter | iii | program pin mismatch: parent_interpreter | program pin mismatch: parent_interpreter |
| a6/G4_a6.json | Recorded_pin_outer_timeout | iii | program pin mismatch: outer_timeout | program pin mismatch: outer_timeout |
| a6/G4_a6.json | Recorded_pin_shebang_env | iii | program pin mismatch: shebang_env | program pin mismatch: shebang_env |
| a6/G4_a6.json | Recorded_env_PATH | iii | child environment differs from staged environment | child environment differs from staged environment |
| a6/G4_a6.json | Recorded_env_FORGE_GPU_SUMMARY | iii | child environment differs from staged environment | child environment differs from staged environment |
| a6/G4_a6.json | Recorded_env_FORGE_GPU_WINDOW_FILE | iii | child environment differs from staged environment | child environment differs from staged environment |
| a6/G4_a6.json | Recorded_env_FORGE_GPU_LOCK | iii | child environment differs from staged environment | child environment differs from staged environment |
| a6/G4_a6.json | Recorded_env_FORGE_GPU_KILL_GRACE | iii | child environment differs from staged environment | child environment differs from staged environment |
| a6/G4_a6.json | Recorded_env_FORGE_GPU_POLL | iii | child environment differs from staged environment | child environment differs from staged environment |
| a6/G4_a6.json | Recorded_env_FORGE_GPU_RUN_LOG | iii | child environment differs from staged environment | child environment differs from staged environment |
| a6/G4_a6.json | Recorded_env_FORGE_GPU_UNKNOWN | iii | child environment differs from staged environment | child environment differs from staged environment |
| a6/G4_a6.json | Actual_env_FORGE_GPU_SUMMARY | i | D1 child-received environment recorded; control root rejected | D1 child-received environment recorded; control root rejected |
| a6/G4_a6.json | Actual_env_FORGE_GPU_WINDOW_FILE | i | D1 child-received environment recorded; control root rejected | D1 child-received environment recorded; control root rejected |
| a6/G4_a6.json | Actual_env_FORGE_GPU_LOCK | i | D1 child-received environment recorded; control root rejected | D1 child-received environment recorded; control root rejected |
| a6/G4_a6.json | Actual_env_FORGE_GPU_KILL_GRACE | i | D1 child-received environment recorded; control root rejected | D1 child-received environment recorded; control root rejected |
| a6/G4_a6.json | Actual_env_FORGE_GPU_POLL | i | D1 child-received environment recorded; control root rejected | D1 child-received environment recorded; control root rejected |
| a6/G4_a6.json | Actual_env_FORGE_GPU_RUN_LOG | i | D1 child-received environment recorded; control root rejected | D1 child-received environment recorded; control root rejected |
| a6/G4_a6.json | REV5_postexit_seal_failure | i | D4 protected seal/retry/minimal receipt; double failure incomplete fallback | D4 protected seal/retry/minimal receipt; double failure incomplete fallback |
| a6/G4_a6.json | D4_write_retry | i | D4 protected seal/retry/minimal receipt; double failure incomplete fallback | D4 protected seal/retry/minimal receipt; double failure incomplete fallback |
| a6/G4_a6.json | D4_double_write_incomplete | ii | D4 protected seal/retry/minimal receipt; double failure incomplete fallback | D4 protected seal/retry/minimal receipt; double failure incomplete fallback |
| a6/D0_MODE_PROMOTION.json | D0_control_receipt_promotion | iii | D0(a)/D1 CPU control root must never establish acceptance after receipt-only forgery | D0(a)/D1 CPU control root must never establish acceptance after receipt-only forgery |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | exact_59_file_promotion | iii | mode/evidence receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | root_only | iii | mode/evidence receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | all_parents_only | iii | mode/evidence receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | supplemental_D0a_honest_CPU_evidence_promotion | iii | mode/evidence receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_baseline_focus_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_baseline_focus_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_baseline_focus_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_baseline_focus_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_candidate_focus_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_candidate_focus_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_candidate_focus_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_candidate_focus_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_baseline_regression_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_baseline_regression_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_baseline_regression_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_baseline_regression_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_candidate_regression_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_candidate_regression_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_candidate_regression_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_candidate_regression_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_bias_dtype_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_bias_dtype_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_bias_dtype_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_bias_dtype_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_bias_length_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_bias_length_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_bias_length_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_bias_length_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_bias_rank_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_bias_rank_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_bias_rank_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_bias_rank_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_c_bool_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_c_bool_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_c_bool_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_c_bool_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_c_index_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_c_index_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_c_index_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_c_index_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_c_int_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_c_int_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_c_int_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_c_int_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_c_npbool_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_c_npbool_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_c_npbool_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_c_npbool_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_c_positive_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_c_positive_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_c_positive_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_c_positive_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_c_store_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_c_store_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_c_store_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_c_store_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_channels_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_channels_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_channels_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_channels_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_g_bool_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_g_bool_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_g_bool_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_g_bool_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_g_index_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_g_index_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_g_index_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_g_index_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_g_int_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_g_int_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_g_int_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_g_int_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_g_npbool_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_g_npbool_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_g_npbool_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_g_npbool_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_g_positive_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_g_positive_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_g_positive_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_g_positive_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_g_store_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_g_store_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_g_store_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_g_store_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_input_dtype_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_input_dtype_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_input_dtype_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_input_dtype_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_rank_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_rank_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_rank_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_rank_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_weight_dtype_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_weight_dtype_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_weight_dtype_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_weight_dtype_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_weight_length_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_weight_length_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_weight_length_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_weight_length_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_weight_rank_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_D5_weight_rank_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_weight_rank_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_D5_weight_rank_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M01_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M01_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M01_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M01_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M02_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M02_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M02_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M02_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M03_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M03_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M03_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M03_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M04m_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M04m_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M04m_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M04m_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M04v_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M04v_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M04v_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M04v_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M05_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M05_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M05_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M05_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M06in_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M06in_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M06in_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M06in_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M06out_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M06out_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M06out_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M06out_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M07cast_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M07cast_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M07cast_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M07cast_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M07fast_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M07fast_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M07fast_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M07fast_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M08cast_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M08cast_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M08cast_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M08cast_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M08fast_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M08fast_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M08fast_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M08fast_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M09cast_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M09cast_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M09cast_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M09cast_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M09fast_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M09fast_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M09fast_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M09fast_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M10_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M10_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M10_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M10_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M11_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M11_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M11_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M11_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M12_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M12_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M12_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M12_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M13_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M13_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M13_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M13_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M14_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M14_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M14_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M14_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M15b_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M15b_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M15b_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M15b_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M15w_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M15w_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M15w_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M15w_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M15x_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M15x_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M15x_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M15x_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M16_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M16_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M16_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M16_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M17b_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M17b_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M17b_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M17b_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M17w_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M17w_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M17w_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M17w_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M18b_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M18b_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M18b_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M18b_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M18w_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_M18w_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M18w_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_M18w_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_V1_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_V1_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_V1_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_V1_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_V2_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_V2_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_V2_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_V2_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_V3_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_V3_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_V3_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_V3_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_V4_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_V4_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_V4_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_V4_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_E_npbool_g_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_E_npbool_g_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_E_npbool_g_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_E_npbool_g_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_E_npbool_c_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_singleton_E_npbool_c_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_E_npbool_c_root_False | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | partial_complement_E_npbool_c_root_True | iii | partial parent mode receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | evidence_cube_0_0_0 | iii | root/parent/child well-formed receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | evidence_cube_0_0_1 | iii | root/parent/child well-formed receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | evidence_cube_0_1_0 | iii | root/parent/child well-formed receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | evidence_cube_0_1_1 | iii | root/parent/child well-formed receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | evidence_cube_1_0_0 | iii | root/parent/child well-formed receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | evidence_cube_1_0_1 | iii | root/parent/child well-formed receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | evidence_cube_1_1_0 | iii | root/parent/child well-formed receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | evidence_cube_1_1_1 | iii | root/parent/child well-formed receipt edits | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | omit_evidence_program | iii | omitted field in edited receipt | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | omit_evidence_CUDA_VISIBLE_DEVICES | iii | omitted field in edited receipt | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | omit_evidence_device_name | iii | omitted field in edited receipt | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | omit_evidence_cuda_runtime_version | iii | omitted field in edited receipt | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | omit_evidence_cuda_driver_version | iii | omitted field in edited receipt | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | omit_evidence_device_ordinal | iii | omitted field in edited receipt | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | omit_evidence_native_status | iii | omitted field in edited receipt | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | omit_evidence_engine_sha256 | iii | omitted field in edited receipt | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | omit_evidence_provider | iii | omitted field in edited receipt | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | omit_evidence_query_error | iii | omitted field in edited receipt | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | omit_evidence_captured_at | iii | omitted field in edited receipt | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | edit_type_cuda_runtime_version | iii | version/ordinal field type edit | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | edit_type_cuda_driver_version | iii | version/ordinal field type edit | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | edit_type_device_ordinal | iii | version/ordinal field type edit | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | native_status_failure_cudaGetDevice | iii | native-status field edit | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | native_status_failure_cudaGetDeviceProperties_v2 | iii | native-status field edit | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | native_status_failure_cudaRuntimeGetVersion | iii | native-status field edit | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | native_status_failure_cudaDriverGetVersion | iii | native-status field edit | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | standin_writer_candidate_focus | iii | hand-edited child evidence with stand-in writer | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | standin_writer_baseline_regression | iii | hand-edited child evidence with stand-in writer | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | standin_writer_M01 | iii | hand-edited child evidence with stand-in writer | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | standin_digest_real_path | iii | well-formed device/mode receipt edit | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | empty_CUDA_VISIBLE_DEVICES | iii | well-formed device/mode receipt edit | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | root_parent_child_modes | iii | well-formed device/mode receipt edit | history under C |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | actual_standin_substitution_at_sealed_child_path | i | actual wrong child program under real child path | child program differs from sealed run_gpu_tests.py |
| a7/ADDITIONAL_CONTROL_REGISTRATION.md | GPU_shaped_schema_accept | iii | constructed GPU-shaped input retained as history; A8 uses direct unit test | history under C |
