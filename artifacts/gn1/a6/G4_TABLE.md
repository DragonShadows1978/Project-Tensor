# G4 hostile targets

**G4 FAIL: 321 registered targets met; one supplemental in-scope target failed (321/322).** Each runs through the real A6 parent and scorer. D1 control roots cannot establish GPU acceptance. The two inherited full-root positive target names retain their old wording in raw JSON; their registered D1 predicate and deciding rule below require all lines to pass and the root to refuse.

| Control | Result | Deciding rule | Receipt/index |
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
| Full_valid_synthetic | PASS | D1 positive line sensitivity: all 58 lines pass, control root cannot establish GPU acceptance | G4_hostile.json / 37 |
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
| Resolved_TMPDIR_escape | PASS | TMPDIR must resolve to /tmp/gn1_a6_* | G4_hostile.json / 51 |
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
| C8_full_positive | PASS | D1 positive line sensitivity: all 58 lines pass, control root cannot establish GPU acceptance | hostile_rev2_acd.json / 29 |
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
| D3_evaluate_never_raises | PASS | D4 total attribution: non-start row plus124 becomes outer-timeout; scorer rejects | G4_a4.json / 17 |
| D2_start_marker | PASS | D2 positive start evidence independently admits incomplete failure | G4_a4.json / 18 |
| D2_start_invocation | PASS | D2 positive start evidence independently admits incomplete failure | G4_a4.json / 19 |
| D2_start_ran | PASS | D2 positive start evidence independently admits incomplete failure | G4_a4.json / 20 |
| D2_start_timeout | PASS | D2 positive start evidence independently admits incomplete failure | G4_a4.json / 21 |
| D2_start_window-closed | PASS | D2 positive start evidence independently admits incomplete failure | G4_a4.json / 22 |
| D2_start_junit | PASS | D2 positive start evidence independently admits incomplete failure | G4_a4.json / 23 |
| REV4_max_equals | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 0 |
| REV4_min_abbrev | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 1 |
| REV4_min_equals | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 2 |
| REV4_inner_short_k | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 3 |
| REV4_inner_kill_after_space | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 4 |
| N2b_unset_grace | PASS | D3 normalised exit / outer-timeout, D2 short-bound refusal, no surviving child | G4_a5.json / 5 |
| N2c_unset_grace | PASS | D3 normalised exit / outer-timeout, D2 short-bound refusal, no surviving child | G4_a5.json / 6 |
| N2b_D3 | PASS | D3 normalised exit / outer-timeout, D2 short-bound refusal, no surviving child | G4_a5.json / 7 |
| N2c_D3 | PASS | D3 normalised exit / outer-timeout, D2 short-bound refusal, no surviving child | G4_a5.json / 8 |
| Raw_SIGKILL_normalisation | PASS | D3 raw negative return becomes137 before D4 outer-timeout attribution | G4_a5.json / 9 |
| N5 | PASS | D4 non-start row/124 returns outer-timeout, not-admitted receipt | G4_a5.json / 10 |
| N4 | PASS | D5 corrupt prior receipt refuses with JSON error, no traceback | G4_a5.json / 11 |
| Atomic_publication | PASS | D5 complete linked receipt, no-overwrite publication | G4_a5.json / 12 |
| Option_runner_000 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 13 |
| Option_runner_001 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 14 |
| Option_runner_002 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 15 |
| Option_runner_003 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 16 |
| Option_runner_004 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 17 |
| Option_runner_005 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 18 |
| Option_runner_006 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 19 |
| Option_runner_007 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 20 |
| Option_runner_008 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 21 |
| Option_runner_009 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 22 |
| Option_runner_010 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 23 |
| Option_runner_011 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 24 |
| Option_runner_012 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 25 |
| Option_runner_013 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 26 |
| Option_runner_014 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 27 |
| Option_runner_015 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 28 |
| Option_runner_016 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 29 |
| Option_runner_017 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 30 |
| Option_runner_018 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 31 |
| Option_runner_019 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 32 |
| Option_runner_020 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 33 |
| Option_runner_021 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 34 |
| Option_runner_022 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 35 |
| Option_runner_023 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 36 |
| Option_runner_024 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 37 |
| Option_runner_025 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 38 |
| Option_runner_026 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 39 |
| Option_runner_027 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 40 |
| Option_runner_028 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 41 |
| Option_runner_029 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 42 |
| Option_runner_030 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 43 |
| Option_runner_031 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 44 |
| Option_runner_032 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 45 |
| Option_runner_033 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 46 |
| Option_runner_034 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 47 |
| Option_runner_035 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 48 |
| Option_runner_036 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 49 |
| Option_runner_037 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 50 |
| Option_runner_038 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 51 |
| Option_runner_039 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 52 |
| Option_runner_040 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 53 |
| Option_runner_041 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 54 |
| Option_runner_042 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 55 |
| Option_runner_043 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 56 |
| Option_runner_044 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 57 |
| Option_timeout_000 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 58 |
| Option_timeout_001 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 59 |
| Option_timeout_002 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 60 |
| Option_timeout_003 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 61 |
| Option_timeout_004 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 62 |
| Option_timeout_005 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 63 |
| Option_timeout_006 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 64 |
| Option_timeout_007 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 65 |
| Option_timeout_008 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 66 |
| Option_timeout_009 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 67 |
| Option_timeout_010 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 68 |
| Option_timeout_011 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 69 |
| Option_timeout_012 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 70 |
| Option_timeout_013 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 71 |
| Option_timeout_014 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 72 |
| Option_timeout_015 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 73 |
| Option_timeout_016 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 74 |
| Option_timeout_017 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 75 |
| Option_timeout_018 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 76 |
| Option_timeout_019 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 77 |
| Option_timeout_020 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 78 |
| Option_timeout_021 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 79 |
| Option_timeout_022 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 80 |
| Option_timeout_023 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 81 |
| Option_timeout_024 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 82 |
| Option_timeout_025 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 83 |
| Option_timeout_026 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 84 |
| Option_timeout_027 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 85 |
| Option_timeout_028 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 86 |
| Option_timeout_029 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 87 |
| Option_timeout_030 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 88 |
| Option_timeout_031 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 89 |
| Option_timeout_032 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 90 |
| Option_timeout_033 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 91 |
| Option_timeout_034 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 92 |
| Option_timeout_035 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 93 |
| Option_timeout_036 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 94 |
| Option_timeout_037 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 95 |
| Option_timeout_038 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 96 |
| Option_timeout_039 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 97 |
| Option_timeout_040 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 98 |
| Option_timeout_041 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 99 |
| Option_timeout_042 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 100 |
| Option_timeout_043 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 101 |
| Option_timeout_044 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 102 |
| Option_timeout_045 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 103 |
| Option_timeout_046 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 104 |
| Option_timeout_047 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 105 |
| Option_timeout_048 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 106 |
| Option_timeout_049 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 107 |
| Option_timeout_050 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 108 |
| Option_timeout_051 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 109 |
| Option_timeout_052 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 110 |
| Option_timeout_053 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 111 |
| Option_timeout_054 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 112 |
| Option_timeout_055 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 113 |
| Option_timeout_056 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 114 |
| Option_timeout_057 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 115 |
| Option_timeout_058 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 116 |
| Option_timeout_059 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 117 |
| Option_timeout_060 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 118 |
| Option_timeout_061 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 119 |
| Option_timeout_062 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 120 |
| Option_timeout_063 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 121 |
| Option_timeout_064 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 122 |
| Option_timeout_065 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 123 |
| Option_timeout_066 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 124 |
| Option_timeout_067 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 125 |
| Option_timeout_068 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 126 |
| Option_timeout_069 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 127 |
| Option_timeout_070 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 128 |
| Option_timeout_071 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 129 |
| Option_timeout_072 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 130 |
| Option_timeout_073 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 131 |
| Option_timeout_074 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 132 |
| Option_timeout_075 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 133 |
| Grammar_runner_equals_min | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 134 |
| Grammar_runner_equals_max | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 135 |
| Grammar_runner_equals_label | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 136 |
| Grammar_runner_reordered | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 137 |
| Grammar_runner_duplicate | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 138 |
| Grammar_runner_float_min | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 139 |
| Grammar_runner_plus_max | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 140 |
| Grammar_runner_zero_padded | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 141 |
| Grammar_duration_0 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 142 |
| Grammar_duration_1 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 143 |
| Grammar_duration_2 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 144 |
| Grammar_duration_3 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 145 |
| Grammar_duration_4 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 146 |
| Grammar_duration_5 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 147 |
| Grammar_duration_6 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 148 |
| Grammar_duration_7 | PASS | D2 exact runner/inner argv refusal | G4_a5.json / 149 |
| N2b_D3_PID | PASS | P116/D3 observe actual child PID from start marker immediately after parent return | G4_p116.json / 0 |
| N2c_D3_PID | PASS | P116/D3 observe actual child PID from start marker immediately after parent return | G4_p116.json / 1 |
| CPU_control_root | PASS | D1 all 58 lines recompute; control root never establishes acceptance | G4_a6.json / 0 |
| NEW_parent_result_deep_nesting | PASS | D2 Exception catches RecursionError; summary written and affected line/root refuses | G4_a6.json / 1 |
| NEW_start_json_deep_nesting | PASS | D2 Exception catches RecursionError; summary written and affected line/root refuses | G4_a6.json / 2 |
| NEW_regression_deep_child_result | PASS | D2 Exception catches RecursionError; summary written and affected line/root refuses | G4_a6.json / 3 |
| raw_exit_minus9_exit0 | PASS | D3 integer raw_exit and independently recomputed normalisation | G4_a6.json / 4 |
| raw_exit_removed | PASS | D3 integer raw_exit and independently recomputed normalisation | G4_a6.json / 5 |
| raw_exit_bool | PASS | D3 integer raw_exit and independently recomputed normalisation | G4_a6.json / 6 |
| raw_exit_string | PASS | D3 integer raw_exit and independently recomputed normalisation | G4_a6.json / 7 |
| Q2_wrapped_runner_command0 | PASS | D1 pinned argv/program/environment refusal | G4_a6.json / 8 |
| D0c_runner_substitute | PASS | D1 pinned argv/program/environment refusal | G4_a6.json / 9 |
| D0c_timeout_wrapper | PASS | D1 pinned argv/program/environment refusal | G4_a6.json / 10 |
| D0c_interpreter_wrapper | PASS | D1 pinned argv/program/environment refusal | G4_a6.json / 11 |
| Q2_path_shadow_inner_timeout | PASS | D1 pinned argv/program/environment refusal | G4_a6.json / 12 |
| D0c_PATH_shadow | PASS | D1 pinned argv/program/environment refusal | G4_a6.json / 13 |
| Recorded_pin_runner | PASS | program pin mismatch: runner | G4_a6.json / 14 |
| Recorded_pin_timeout | PASS | program pin mismatch: timeout | G4_a6.json / 15 |
| Recorded_pin_interpreter | PASS | program pin mismatch: interpreter | G4_a6.json / 16 |
| Recorded_pin_runner_interpreter | PASS | program pin mismatch: runner_interpreter | G4_a6.json / 17 |
| Recorded_pin_parent_interpreter | PASS | program pin mismatch: parent_interpreter | G4_a6.json / 18 |
| Recorded_pin_outer_timeout | PASS | program pin mismatch: outer_timeout | G4_a6.json / 19 |
| Recorded_pin_shebang_env | PASS | program pin mismatch: shebang_env | G4_a6.json / 20 |
| Recorded_env_PATH | PASS | child environment differs from staged environment | G4_a6.json / 21 |
| Recorded_env_FORGE_GPU_SUMMARY | PASS | child environment differs from staged environment | G4_a6.json / 22 |
| Recorded_env_FORGE_GPU_WINDOW_FILE | PASS | child environment differs from staged environment | G4_a6.json / 23 |
| Recorded_env_FORGE_GPU_LOCK | PASS | child environment differs from staged environment | G4_a6.json / 24 |
| Recorded_env_FORGE_GPU_KILL_GRACE | PASS | child environment differs from staged environment | G4_a6.json / 25 |
| Recorded_env_FORGE_GPU_POLL | PASS | child environment differs from staged environment | G4_a6.json / 26 |
| Recorded_env_FORGE_GPU_RUN_LOG | PASS | child environment differs from staged environment | G4_a6.json / 27 |
| Recorded_env_FORGE_GPU_UNKNOWN | PASS | child environment differs from staged environment | G4_a6.json / 28 |
| Actual_env_FORGE_GPU_SUMMARY | PASS | D1 child-received environment recorded; control root rejected | G4_a6.json / 29 |
| Actual_env_FORGE_GPU_WINDOW_FILE | PASS | D1 child-received environment recorded; control root rejected | G4_a6.json / 30 |
| Actual_env_FORGE_GPU_LOCK | PASS | D1 child-received environment recorded; control root rejected | G4_a6.json / 31 |
| Actual_env_FORGE_GPU_KILL_GRACE | PASS | D1 child-received environment recorded; control root rejected | G4_a6.json / 32 |
| Actual_env_FORGE_GPU_POLL | PASS | D1 child-received environment recorded; control root rejected | G4_a6.json / 33 |
| Actual_env_FORGE_GPU_RUN_LOG | PASS | D1 child-received environment recorded; control root rejected | G4_a6.json / 34 |
| REV5_postexit_seal_failure | PASS | D4 protected seal/retry/minimal receipt; double failure incomplete fallback | G4_a6.json / 35 |
| D4_write_retry | PASS | D4 protected seal/retry/minimal receipt; double failure incomplete fallback | G4_a6.json / 36 |
| D4_double_write_incomplete | PASS | D4 protected seal/retry/minimal receipt; double failure incomplete fallback | G4_a6.json / 37 |
| D0_control_receipt_promotion | FAIL | D0(a)/D1 CPU control root must never establish acceptance after receipt-only forgery | D0_MODE_PROMOTION.json |
