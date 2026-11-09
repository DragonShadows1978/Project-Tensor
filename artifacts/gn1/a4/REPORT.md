# GN1 A4

**ACCEPT (CPU gates G1 to G6).** R3-F01 to R3-F06 and D7 are fixed in A4. The product and test file remain at cf922891 bytes; frozen earlier receipts and the pinned module/source map are unchanged. Ready for lead-dispatched GN1-REV4; no GPU execution or performance claim.

Branch `gn1`; immutable amendment commit `3810fe3a4c8e493c5fc7ca5fd9fe2975959eaadc`. Final seal SHA-256: `647175e302683ba6be090a7e8d677f3fe1eb06c1af422aa757a66e9c5c575683`. The seal records the amendment commit and exact tooling bytes; no git writes or rebuilds occurred.

Evidence class: real CPU collection/reference/guard tests, real Python GN tests on the explicitly labeled NumPy engine stand-in, raw installed-runner CPU controls with scratch ownership resources, synthetic hostile receipts, exact SHA-256 comparisons, and finite reasoning for equivalences. Acceptance does not establish a GPU pass.

## G1 to G6

| Gate | Result | Evidence | Receipts |
|---|---|---|---|
| G1 | PASS | Product/test byte-identical to cf922891; frozen a3/rev1/rev2/rev3 maps equal; every tooling hunk mapped. | G1_scope.json; *_frozen_{before,after}.json; changes_vs_a3.patch; TOOLING_HUNK_MAP.json |
| G2 | PASS | 405/405 GN CPU stand-in tests; native44 reference/domain IDs and57 metadata guards; both arms import/collect; all five ID files equal A3; 324-case tolerance derivation unchanged. | G2_shim.json; G2_driver.json; cpu_candidate_mutation_references.json; native_validation.json; tolerance_bounds.json; pytorch_behavior.json; ids_*.txt |
| G3 | PASS | 56 module copies match original pin; 72 source-map files unchanged; zero rebuilds and zero arm symlinks. | G3_pin.json; engine_sources_verified.json; arm_maps.json |
| G4 | PASS | 131/131 targets:107 inherited and24 A4, all against the final seal. | G4_all.json; G4_CONTROLS.md; G4_hostile.json; G4_runner.json; hostile_rev2_acd.json; G4_extra.json; G4_a4.json |
| G5 | PASS | 54 single-site mutants load/differ/reach in call phase;51 CPU kills and3 equivalent passes. Both new deletions also pass all405 real GN tests. | G5_mutants.json; mutants/*/trace.json; E_npbool_g_full_cpu.json; E_npbool_c_full_cpu.json |
| G6 | PASS | 58 staged labels with exact bounds/exits; two complete honest stand-in roots score0; seal-verified arm regeneration; native audit29 reference calls. | G6_staging.json; staging_exact_bounds.json; regeneration_verified.json; audit_gate.json; STAGED_GPU_COMMANDS.sh; delivery_verify.json |

## Per-finding fix table

| Finding | Status | Fix | Receipts |
|---|---|---|---|
| R3-F01 | FIXED | Atomic early child marker; admission requires absence of every start operand. | G4_a4.json: B6_lock_outer_retry; D2_start_* |
| R3-F02 | FIXED | Kill-grace rows timeout/window-closed with 143/137 and recorded124/137 map to outer-timeout; evaluate records exceptions. | G4_a4.json: B7_grace_no_result, B7b_grace_result, B8_window_grace, D3_evaluate_never_raises |
| R3-F03 | FIXED | Incomplete started attempts are admitted, evaluated and failed; a retry cannot erase them. | G4_a4.json: C7b_started_no_result; six D2_start_* controls |
| R3-F04 | FIXED | child_result.json uses temporary-file publication with os.replace; JSON reading is inside evaluate try. | G4_a4.json: E5_truncated_incomplete and E5_truncated_completed; audit_cpu/ |
| R3-F05 | FIXED | Runtime sys.modules origins/digests augment collection pins; known tamper fails, unseen origins are unsealed lead-review records. | RUNTIME_SEAL_CENSUS.json; runtime_trace_*.json; G4_a4.json: D5_sealed_tamper and D5_unsealed |
| R3-F06 | FIXED | Both numpy.bool_ clause deletions are sealed passing equivalences with version and live operator.index proof; negations retained. | EQUIVALENCES.md; mutants.json; E_npbool_{g,c}_full_cpu.json; G4_a4.json: D6_* |
| D7 | FIXED | Scorer binds selected admitted line outer, runner max/min, inner timeout and expected exits to gpu_lines.json. Never-admitted attempts are not counted as lines. | G4_a4.json: D7_outer, D7_maximum, D7_minimum, D7_inner, D7_expected_exits; staging_exact_bounds.json |

## Tooling diff summary

32 changed source/input files and 117 mapped hunks versus A3; 31 copied inputs remain byte-identical. See changes_vs_a3.patch and TOOLING_HUNK_MAP.{json,md}. Mechanical D1 path/order changes are identified separately. Functional changes are D2-D7 and their required controls, scope checks and staging. No product, test, tolerance, engine or source-map edits.

Core changes: evidence.py handles positive start operands, incomplete attempts, fail-closed evaluation, runtime origins, live equivalences and exact bounds. runtime_evidence.py supplies atomic publication, module origins and the live index check. run_gpu_tests.py publishes the early marker and post-pytest census/result. seal_staging.py unions collection/runtime imports. build_mutants.py and gpu_lines.json register the two equivalent deletions and explicit runner minima. The staged shell preserves the A3 bounds and adds the equivalence lines.

## G4 table


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

## Runtime seal census


Collection: 534 unique origins. Final local/external import maps: 17 local and 537 external origins, excluding arm maps. The runtime union adds 29 origins, including 11 numpy.random and 5 numpy.testing origins. All sealed digests match at final verification. No unsealed modules occur in either honest full stand-in root or in the native audit line. The explicit unsealed hostile control imports a new scratch module and reports it without a veto.

| CPU stand-in suite | Module entries | IDs | Calls | Pytest exit | Setup errors |
|---|---|---|---|---|---|
| focus | 548 | 412 | 412 | 1 | 0 |
| mutation | 545 | 405 | 405 | 0 | 0 |
| regression | 562 | 468 | 453 | 1 | 15 |

The stand-in does not implement unrelated engine operations. Focus keeps 7 failures; regression keeps 33 call failures, 6 skips and 15 setup errors. These are import-census outcomes, not GPU regressions or a passing engine suite. NumPy testing calibration is explicit in each trace; unseen real-device paths remain unsealed and must be reviewed by the lead.

Added origins:

- `$HOME/.local/lib/python3.12/site-packages/numpy/ma/__init__.py`
- `$HOME/.local/lib/python3.12/site-packages/numpy/ma/core.py`
- `$HOME/.local/lib/python3.12/site-packages/numpy/ma/extras.py`
- `$HOME/.local/lib/python3.12/site-packages/numpy/random/__init__.py`
- `$HOME/.local/lib/python3.12/site-packages/numpy/random/_bounded_integers.cpython-312-x86_64-linux-gnu.so`
- `$HOME/.local/lib/python3.12/site-packages/numpy/random/_common.cpython-312-x86_64-linux-gnu.so`
- `$HOME/.local/lib/python3.12/site-packages/numpy/random/_generator.cpython-312-x86_64-linux-gnu.so`
- `$HOME/.local/lib/python3.12/site-packages/numpy/random/_mt19937.cpython-312-x86_64-linux-gnu.so`
- `$HOME/.local/lib/python3.12/site-packages/numpy/random/_pcg64.cpython-312-x86_64-linux-gnu.so`
- `$HOME/.local/lib/python3.12/site-packages/numpy/random/_philox.cpython-312-x86_64-linux-gnu.so`
- `$HOME/.local/lib/python3.12/site-packages/numpy/random/_pickle.py`
- `$HOME/.local/lib/python3.12/site-packages/numpy/random/_sfc64.cpython-312-x86_64-linux-gnu.so`
- `$HOME/.local/lib/python3.12/site-packages/numpy/random/bit_generator.cpython-312-x86_64-linux-gnu.so`
- `$HOME/.local/lib/python3.12/site-packages/numpy/random/mtrand.cpython-312-x86_64-linux-gnu.so`
- `$HOME/.local/lib/python3.12/site-packages/numpy/testing/__init__.py`
- `$HOME/.local/lib/python3.12/site-packages/numpy/testing/_private/__init__.py`
- `$HOME/.local/lib/python3.12/site-packages/numpy/testing/_private/extbuild.py`
- `$HOME/.local/lib/python3.12/site-packages/numpy/testing/_private/utils.py`
- `$HOME/.local/lib/python3.12/site-packages/numpy/testing/overrides.py`
- `/usr/lib/python3.12/_sysconfigdata__x86_64-linux-gnu.py`
- `/usr/lib/python3.12/getpass.py`
- `/usr/lib/python3.12/lib-dynload/termios.cpython-312-x86_64-linux-gnu.so`
- `/usr/lib/python3.12/secrets.py`
- `/usr/lib/python3.12/sysconfig.py`
- `artifacts/gn1/a4/cpu_shim.py`
- `artifacts/gn1/a4/evidence.py`
- `artifacts/gn1/a4/run_gpu_tests.py`
- `artifacts/gn1/a4/runtime_evidence.py`
- `artifacts/gn1/a4/trace_runtime.py`

## Equivalence records

| Equivalence | Site | Proof obligation | Real CPU suite | Expected staged exit |
|---|---|---|---|---|
| E_npbool_g | nn.py:518 | NumPy2.4.2; operator.index(numpy.bool_(True)) raises TypeError | 405/405 passed | 0 |
| E_npbool_c | nn.py:526 | NumPy2.4.2; operator.index(numpy.bool_(True)) raises TypeError | 405/405 passed | 0 |
| V3 | low-precision upcast guard | Existing sealed dtype-domain proof | Registered witness passed | 0 |

For each deletion, the built-in bool guard stays. NumPy bool reaches the following operator.index call and is rejected with TypeError under NumPy2.4.2; non-bool operands follow the unchanged path. The proof depends on NumPy behavior. Each real child line records its interpreter version and live index result. A version mismatch or acceptance fails the equivalence line. G4 also verifies live acceptance invalidates the proof. D5_g_npbool and D5_c_npbool remain negated-clause reach controls. See EQUIVALENCES.md, staging_seal.json equivalences, and the two full_cpu receipts.

## Predictions

- **P108 HELD**: final replay meets all131 targets, including every REV3 reproduction and all107 inherited targets. Initial instrument failures below remain negative receipts; they were not reclassified as passes.
- **P109 HELD**: runtime census adds numpy.random and numpy.testing origins and29 origins in total beyond collection. The calibration that touches the lazy testing namespace is explicit. Final verify_seal and arm checks find no sealed-byte drift.

## Final staging and binding

STAGED_GPU_COMMANDS.sh runs from A4's own directory. All58 lines carry an outer timeout, runner --max/--min, an inner timeout and registered integer exits. staging_exact_bounds.json parses each exact staged command against gpu_lines.json. The selected admitted line is bound to these fields in the scorer. A refused attempt with no start evidence does not count as a line; its retry can count only with the registered bounds. Parent completion receipts, child markers, raw runner rows, child summaries and JUnit ID-by-ID comparisons remain required.

The two current-seal honest full roots are hostile_runs/Full_valid_synthetic and rev2_runs_133140/C8_full_positive. Each contains58 labels and scores0. The native audit line has29 passing reference IDs, child0, parent0 and no unsealed modules. Both full stand-in roots also have no unsealed modules. Arm regeneration verifies all56 arms against the seal.

## Retained failed launches and limits

- runtime_regression_gate.json.initial: initial author-only census completeness assertion raised `AssertionError` because453 call reports were compared with468 collected IDs;15 unsupported stand-in setup errors account for the difference. Raw pytest outcomes remain33 failures,414 passes,6 skips and15 setup errors. The census check now requires setup coverage for every ID; it does not change any test or acceptance threshold. See AMENDMENTS.md.
- rev2_gate.log.initial: `FileNotFoundError: [Errno 2] No such file or directory: '/tmp/gn1_a4_boot/rev2_forge/summary.log'`. Created the required scratch directory and reran unchanged sealed controls.
- a4_controls_gate.log.initial_margin and a4_runs.initial_margin: the runner reported `forge-gpu-run: --margin must cover the kill grace`. Those launches did not reach the intended mechanism. Control margins now cover each grace, matching the installed runner's option rule.
- g4_final_gate.json.seal4:131-target replay initially had130 expected targets and one missed defensive malformed-row target. The real runner correctly wrote ran/124, so a malformed no-lock/124 path was not reached. The corrected defensive control uses the inherited synthetic log transport through the unchanged real parent/scorer. All real runner interruption controls retain the installed runner.

Every failed/provisional directory and old seal is retained. Only final-seal receipts establish acceptance. The runtime census is finite and CPU-bound; real-device-only imports remain unsealed lead-review items per D5. GN1-REV4 and the lead's first GPU run are outstanding by design.

## Prior art

- Atomic publication: POSIX rename(2), IEEE Std1003.1; the tool takes atomic name replacement and adds GN1 marker/result fields. Year unverified - lead to check: POSIX rename atomic replacement.
- Runtime origin census: Python sys.modules and __file__, Python Software Foundation; the tool hashes origins after pytest and binds them to the GN1 seal. Year unverified - lead to check: Python sys.modules module origins.
- SHA-256: FIPS180-4 (2015), reused without modification.
- operator.index: PEP357 (2006), reused for the NumPy-version-dependent equivalence proof.
- Mutation analysis: DeMillo, Lipton and Sayward (1978); equivalent mutants and RIP model: Offutt/Voas, year unverified - lead to check. The GN1 mutant registry and live obligations are the application here.
- Fault injection: Meszaros (2007); pytest hooks: Krekel et al. (2004+, year unverified). REV3's reproduction scripts (2026) supply the lock/grace/incomplete-attempt mechanisms.
- GNU timeout/POSIX signal exits and BSD flock supply the supervision/lock mechanisms; history unverified - lead to check GNU timeout kill-after and BSD flock history. The unchanged numerical derivation retains Higham (1996/2002), with its existing conditional powf assumption; CPU stand-in retains its A3 prior-art annotations.

No novel algorithm claimed. Code sites retain prior-art comments; LEDGER.md records the same provenance. External literature was not fetched in this CPU/source-only round.

## Done

All writes are under artifacts/gn1/a4/ and /tmp/gn1_a4_boot; no logs/ files were needed. files_written.txt and files_written.json enumerate every written workspace file. scratch_files_written.txt enumerates every retained scratch file. sha256_maps.json hashes the delivered files, with its own self-hash deliberately excluded. Frozen before/after maps and G1_scope.json supply the immutable baseline comparison. LEFT_FOR_LEAD_TO_BIN.md lists every failed/provisional top-level artifact and the entire scratch tree; nothing was deleted or moved outside A4.

Registration started 2026-11-01 18:00:39 UTC. Synthesis prepared at 2026-11-01 18:41:50 UTC within the2h ceiling. All gate commands used explicit foreground timeouts below20min, GPU hidden and threads1.
