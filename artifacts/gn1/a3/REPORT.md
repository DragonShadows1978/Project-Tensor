# GN1 A3 report

**ACCEPT at the registered CPU G1–G6 gates. Ready for independent GN1-REV3; GPU execution remains unperformed.**

Evidence classes: real Python tests on a NumPy engine stand-in; native module import/collection and metadata-only guards; real installed-runner supervision of scratch CPU children; exact digest/diff checks. The 56-label positive runs use faithful bound stand-in receipts. These results do not establish CUDA numerics, GPU mutant kills, performance, other layers, float16 pow backward, or reshape(-1).

Order commit: `86e3ec8fd8874adf97307e40cc1928f6890bcdf7`; order: `orders/GN1_A3_REV2_FIXES.md`. Final seal SHA-256: `c09a218192fd61ade4f4da1c150f562b975ec61cb7a6ea364b0f2d570dcd9fbc`. Module SHA-256: `c9b7dc610421010a8b11e8e3af74d1a9ad99928dabb98835aa6c265754daf8ce`.

## G1 to G6

| Gate | Result | Evidence scope | Receipts |
| --- | --- | --- | --- |
| G1 | PASS | GroupNorm-only nn.py change; D5-only vs bd30498a; test file; frozen maps equal | G1_scope.json; changes_vs_base.patch; changes_vs_a2.patch; nn_vs_bd30498a.diff |
| G2 | PASS | 405/405 real GN tests on CPU stand-in; 44 native reference/domain IDs; 57 native metadata guards; both arms collect | G2_shim.json; cpu_candidate_mutation_references.json; native_validation.json; ids_*.txt |
| G3 | PASS | 54 module copies match original pin/source map; zero builds or arm symlinks | G3_pin.json; engine_sources_verified.json; arm_maps.json |
| G4 | PASS | 107/107 final hostile targets: 52 A2-derived, 13 real-runner, 38 REV2, 4 supplementary | G4_hostile.json; G4_runner.json; hostile_rev2_acd.json; G4_extra.json |
| G5 | PASS | 52 load/differ/reach; 51 CPU kills; V3 passes as registered equivalence; M14 actual default binding | G5_mutants.json; mutants/*/trace.json |
| G6 | PASS | 56 staged labels; both full stand-in roots score 0; arm regeneration matches seal; audit 29 IDs passes | G6_staging.json; regeneration_verified.json; delivery_verify.json; STAGED_GPU_COMMANDS.sh |

Frozen maps: a2 4237 files, rev1 22 files, rev2 36 files. Before/after map digests match; see G1_scope.json. The baseline is 0e1dea3’s package with a byte copy of the pinned module. No builds, arm symlinks, git writes or production/GPU window/lock access.

## Per-finding fix table

| Finding | Status | Fix | Receipt |
| --- | --- | --- | --- |
| R2-F01 | FIXED | Recognizes rechecked-out; exact 75/76 no-child lease refusals retry | G4_runner.json: B1_rechecked_out_retry, Runner_no_lock76, Runner_no_window75_H5 |
| R2-F02 | FIXED | Lead option (a); preserved original class/message; real test/plugin controls; same-root candidate input pass | G4_hostile.json: Real_engine_device_veto, Real_engine_generic_kill, Real_engine_candidate_not_passed |
| R2-F03 | FIXED | Raising frame only; callee constructor error is not a GroupNorm refusal | G4_hostile.json: P2_callee_origin; run_gpu_tests.py |
| R2-F04 | FIXED | Four timeout causes from raw runner rows; scorer recomputes and rejects cause tampering | G4_runner.json: B2–B5; G4_extra.json: Stored_timeout_cause_tamper |
| R2-F05 | FIXED | PEP 357 index protocol, native int storage, Python/NumPy boolean refusal | G2_shim.json: test_index_counts, test_numpy_bool_refusal; native_validation.json; pytorch_behavior.json |
| R2-F06 | FIXED | Weight/bias rank and length ValueError before tensor ops; 12 wrong shapes tested | G2_shim.json: test_parameter_shape_refusal; native_validation.json; G5_mutants.json |
| R2-F07 | FIXED | Incomplete child evaluated and cannot be retried; no-child interruption permits retry | hostile_rev2_acd.json: C7_child_present_normal, C7_child_present_hang_failure, C7_child_never_started |
| R2-F08 | FIXED | Call-phase tracing only; M14 records actual eps=0.001 default binding | mutants/M14/trace.json; G5_mutants.json |
| R2-F09 | FIXED | V4 negates input fast-path decision at nn.py:583; registered/staged/killed | mutants/V4/nn.diff; mutants/V4/trace.json; gpu_lines.json |
| R2-F10 | FIXED | Parent and staged shell use resolved TMPDIR guard | G4_hostile.json: Resolved_TMPDIR_escape; evidence.py:scratch_guard |
| R2-F11 | FIXED | Import-traced pins include scripts/apa_om135_core.py; child verifies them | import_trace.json; staging_seal.json; G4_extra.json: Import_helper_pin_child |

## nn.py diff against bd30498a

```diff
diff --git a/tensor_cuda/tensor_cuda/nn.py b/tensor_cuda/tensor_cuda/nn.py
index 7e78267b..52e17bd2 100644
--- a/tensor_cuda/tensor_cuda/nn.py
+++ b/tensor_cuda/tensor_cuda/nn.py
@@ -512,13 +512,24 @@ class GroupNorm(Module):
         super().__init__()
         # Prior art: Wu and He (2018), Group Normalization; PyTorch GroupNorm
         # divisibility refusal (docs unverified — lead to check: "GroupNorm").
-        # A2 domain checks, inspired by PyTorch GroupNorm argument validation;
-        # exact positive-int/bool policy is the GN1 A2 requirement.
-        if (isinstance(num_groups, bool) or not isinstance(num_groups, int)
-                or num_groups <= 0):
+        # Prior art: Python operator.index, PEP 357 (2006). A3 takes the
+        # integer-index protocol; excluding both boolean types is D5 policy.
+        import operator
+        if isinstance(num_groups, bool) or isinstance(num_groups, np.bool_):
             raise ValueError(f"num_groups ({num_groups!r}) must be a positive integer (bool excluded)")
-        if (isinstance(num_channels, bool) or not isinstance(num_channels, int)
-                or num_channels <= 0):
+        try:
+            num_groups = int(operator.index(num_groups))
+        except TypeError as exc:
+            raise ValueError(f"num_groups ({num_groups!r}) must be a positive integer (bool excluded)") from exc
+        if num_groups <= 0:
+            raise ValueError(f"num_groups ({num_groups!r}) must be a positive integer (bool excluded)")
+        if isinstance(num_channels, bool) or isinstance(num_channels, np.bool_):
+            raise ValueError(f"num_channels ({num_channels!r}) must be a positive integer (bool excluded)")
+        try:
+            num_channels = int(operator.index(num_channels))
+        except TypeError as exc:
+            raise ValueError(f"num_channels ({num_channels!r}) must be a positive integer (bool excluded)") from exc
+        if num_channels <= 0:
             raise ValueError(f"num_channels ({num_channels!r}) must be a positive integer (bool excluded)")
         if num_channels % num_groups != 0:
             raise ValueError(
@@ -542,6 +553,16 @@ class GroupNorm(Module):
             raise TypeError(f"GroupNorm weight dtype {self.weight.dtype!r} is unsupported")
         if self.bias.dtype not in ("float32", "float16", "bfloat16"):
             raise TypeError(f"GroupNorm bias dtype {self.bias.dtype!r} is unsupported")
+        # Prior art: PyTorch GroupNorm parameter shape validation (observed
+        # separately on CPU). D5 requires explicit refusal before tensor ops.
+        if self.weight.ndim != 1:
+            raise ValueError("GroupNorm weight must have rank 1")
+        if self.weight.shape[0] != self.C:
+            raise ValueError(f"GroupNorm weight length must equal num_channels ({self.C})")
+        if self.bias.ndim != 1:
+            raise ValueError("GroupNorm bias must have rank 1")
+        if self.bias.shape[0] != self.C:
+            raise ValueError(f"GroupNorm bias length must equal num_channels ({self.C})")
         # Prior art: PyTorch half normalization uses float32 accumulation;
         # Micikevicius et al. (2018), Mixed Precision Training (both unverified
         # — lead to check: "PyTorch group_norm acc_type half"). Round 1's
```

## D2 engine refusal rule

The lead chose option (a). An engine refusal of a registered valid input is a kill only when the same run root’s candidate focus passed that input, no device pattern appears in the original exception or failure evidence, and the refusal occurs during the GroupNorm call phase. The test re-raises the refusal as AssertionError with its original class and message; the actual Reports plugin preserves both and the evaluator reads both. Unwrapped GroupNorm TypeError/ValueError refusals use only the raising frame, GroupNorm.__init__ or GroupNorm.forward. A callee-raised factory TypeError is refused as a kill.

The three D2 controls execute the real selected test and Reports plugin on the CPU engine; remaining census IDs are binding scaffolding. The additional baseline control executes all 405 real GN IDs and phases. These scopes are explicit in real_test_evidence.json under the corresponding attempts.

## G2 IDs and observed PyTorch

| Arm/suite | IDs | Exact ID file |
| --- | --- | --- |
| baseline_focus | 412 | ids_baseline_focus.txt |
| candidate_focus | 412 | ids_candidate_focus.txt |
| candidate_mutation | 405 | ids_candidate_mutation.txt |
| baseline_regression | 468 | ids_baseline_regression.txt |
| candidate_regression | 468 | ids_candidate_regression.txt |
| audit | 29 | ids_audit.txt |

405 candidate stand-in tests pass; 44 native reference/domain IDs pass; 57 native metadata-only guard cases pass. All earlier float64 reference checks remain present. The 324-case tolerance derivation is byte-identical to A2’s (SHA-256 `5051b795a569472ad747662d24d01931999bdcb9eb7dfb2fbc036f7eea734279`). PyTorch 2.11.0+cu130 accepts the six NumPy-integer operand/type combinations in constructor/forward and refuses all 12 wrong parameter shapes on CPU; see PYTORCH_BEHAVIOR.md and pytorch_behavior.json.

## G4 table

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

Every one of A2’s 48 control adaptations is listed in HOSTILE_ADAPTATIONS.md. The two impossible synthetic exception summaries were replaced as D2 requires.

## Runner-outcome list

| Exact token | Exit | Installed source lines | G4_runner.json control |
| --- | --- | --- | --- |
| ran | Child exit including 0, 1, 126, 127, signal-derived exits | 179, 185, 189, 218; emitted at 302 | Runner_ran_candidate_focus, Runner_ran_baseline_focus, Runner_ran_cannot_start126/127; B5 |
| timeout | 124 | 184–185, 218; emitted at 302 | B2_runner_deadline |
| window-closed | 124 | 185, 198–200, 218; emitted at 302 | B3_window_closed |
| no-window | 64 (usage), 75 (not admitted) | 230; 246–249 and 302 | Runner_no_window64_usage, Runner_no_window75_H5 |
| no-lock | 76 | 266, 275; emitted at 302 | Runner_no_lock76 |
| rechecked-out | 75 | 281, 285; emitted at 302 | B1_rechecked_out_retry |

| Cause | Recomputed rule | Receipt |
| --- | --- | --- |
| outer-timeout | Exit 124/137 and missing row, or ran row with differing logged 143/137 | B4_outer_timeout; H6_hang_after_summary |
| runner-deadline | timeout row with matching 124/137 | B2_runner_deadline |
| window-closed | window-closed row with matching 124/137 | B3_window_closed |
| inner-timeout | ran row with matching 124/137 | B5_inner_timeout124; B5_inner_timeout137 |

Installed source line numbers refer to runner_source.txt, re-read live, especially 281 and 285. The scorer checks timeout_cause against the raw log. Exact no-child lease refusals are retryable; incomplete child-bearing attempts are evaluated, fail closed without L45 completion evidence, and cannot be retried past their result.

## Mutant table

| Mutant | Edit line | Registered witness | Call reach | CPU result | G5 |
| --- | --- | --- | --- | --- | --- |
| D5_bias_dtype | 554 | test_dtype_refusal[int64-bias] | 554 | CPU kill: AssertionError | PASS |
| D5_bias_length | 564 | test_parameter_shape_refusal[short-bias] | 564 | CPU kill: AssertionError | PASS |
| D5_bias_rank | 562 | test_parameter_shape_refusal[matrix61-bias] | 562 | CPU kill: AssertionError | PASS |
| D5_c_bool | 526 | test_positive_integer_refusal[true-num_channels] | 526 | CPU kill: AssertionError | PASS |
| D5_c_index | 529 | test_index_counts[np_int64-num_channels] | 529 | CPU kill: ValueError | PASS |
| D5_c_int | 529 | test_positive_integer_refusal[float-num_channels] | 529 | CPU kill: AssertionError | PASS |
| D5_c_npbool | 526 | test_index_counts[np_int64-num_channels] | 526 | CPU kill: ValueError | PASS |
| D5_c_positive | 532 | test_positive_integer_refusal[negative-num_channels] | 532 | CPU kill: AssertionError | PASS |
| D5_c_store | 529 | test_index_counts[np_int64-num_channels] | 529 | CPU kill: AssertionError | PASS |
| D5_channels | 548 | test_shape_refusal[channels] | 548 | CPU kill: AssertionError | PASS |
| D5_g_bool | 518 | test_positive_integer_refusal[true-num_groups] | 518 | CPU kill: AssertionError | PASS |
| D5_g_index | 521 | test_index_counts[np_int64-num_groups] | 521 | CPU kill: ValueError | PASS |
| D5_g_int | 521 | test_positive_integer_refusal[float-num_groups] | 521 | CPU kill: AssertionError | PASS |
| D5_g_npbool | 518 | test_index_counts[np_int64-num_groups] | 518 | CPU kill: ValueError | PASS |
| D5_g_positive | 524 | test_positive_integer_refusal[negative-num_groups] | 524 | CPU kill: AssertionError | PASS |
| D5_g_store | 521 | test_index_counts[np_int64-num_groups] | 521 | CPU kill: AssertionError | PASS |
| D5_input_dtype | 550 | test_dtype_refusal[int64-input] | 550 | CPU kill: AssertionError | PASS |
| D5_rank | 546 | test_shape_refusal[rank1] | 546 | CPU kill: AssertionError | PASS |
| D5_weight_dtype | 552 | test_dtype_refusal[int64-weight] | 552 | CPU kill: AssertionError | PASS |
| D5_weight_length | 560 | test_parameter_shape_refusal[short-weight] | 560 | CPU kill: AssertionError | PASS |
| D5_weight_rank | 558 | test_parameter_shape_refusal[matrix61-weight] | 558 | CPU kill: AssertionError | PASS |
| M01 | 579 | test_forward_values[float32-r2-n1-g2-default] | 579 | CPU kill: AssertionError | PASS |
| M02 | 580 | test_near_epsilon[forward-float32] | 580 | CPU kill: AssertionError | PASS |
| M03 | 580 | test_near_epsilon[forward-float32] | 580 | CPU kill: AssertionError | PASS |
| M04m | 579 | test_forward_values[float32-r3-n2-g2-default] | 579 | CPU kill: AssertionError | PASS |
| M04v | 579 | test_forward_values[float32-r3-n2-g2-default] | 579 | CPU kill: AssertionError | PASS |
| M05 | 576 | test_forward_values[float32-r2-n1-g2-default] | 576 | CPU kill: AssertionError | PASS |
| M06in | 576 | test_forward_values[float32-r3-n2-g2-default] | 576 | CPU kill: AssertionError | PASS |
| M06out | 581 | test_forward_values[float32-r3-n2-g2-default] | 581 | CPU kill: AssertionError | PASS |
| M07cast | 588 | test_forward_values[float16-r3-n2-g2-default] | 588 | CPU kill: AssertionError | PASS |
| M07fast | 585 | test_forward_values[float32-r3-n2-g2-default] | 585 | CPU kill: AssertionError | PASS |
| M08cast | 588 | test_forward_values[float16-r3-n2-g2-default] | 588 | CPU kill: AssertionError | PASS |
| M08fast | 585 | test_forward_values[float32-r3-n2-g2-default] | 585 | CPU kill: AssertionError | PASS |
| M09cast | 588 | test_forward_values[float16-r3-n2-g2-default] | 588 | CPU kill: AssertionError | PASS |
| M09fast | 585 | test_forward_values[float32-r3-n2-g2-default] | 585 | CPU kill: AssertionError | PASS |
| M10 | 534 | test_nondivisible_refusal | 534 | CPU kill: AssertionError | PASS |
| M11 | 582 | test_forward_values[float32-r3-n2-g2-default] | 582 | CPU kill: AssertionError | PASS |
| M12 | 580 | test_forward_values[float32-r3-n2-g2-default] | 580 | CPU kill: AssertionError | PASS |
| M13 | 580 | test_forward_values[float32-r3-n2-g2-default] | 580 | CPU kill: AssertionError | PASS |
| M14 | 511 | test_constructor_defaults | default eps=0.001 bound in actual call | CPU kill: AssertionError | PASS |
| M15b | 587 | test_forward_values[float16-r3-n2-g2-default] | 587 | CPU kill: AssertionError | PASS |
| M15w | 586 | test_forward_values[float16-r3-n2-g2-default] | 586 | CPU kill: AssertionError | PASS |
| M15x | 571 | test_forward_values[float16-r3-n2-g2-default] | 571 | CPU kill: AssertionError | PASS |
| M16 | 589 | test_mixed_parameters[forward-default-float16] | 589 | CPU kill: AssertionError | PASS |
| M17b | 587 | test_distinct_parameter_dtypes[forward-w32-b16] | 587 | CPU kill: AssertionError | PASS |
| M17w | 586 | test_distinct_parameter_dtypes[forward-w16-b32] | 586 | CPU kill: AssertionError | PASS |
| M18b | 587 | test_mixed_parameters[forward-half-float16] | 587 | CPU kill: AssertionError | PASS |
| M18w | 586 | test_mixed_parameters[forward-half-float16] | 586 | CPU kill: AssertionError | PASS |
| V1 | 584 | test_distinct_parameter_dtypes[forward-w32-b16] | 584 | CPU kill: AssertionError | PASS |
| V2 | 583 | test_distinct_parameter_dtypes[forward-w16-b32] | 583 | CPU kill: AssertionError | PASS |
| V3 | 571 | test_forward_values[float16-r3-n2-g2-default] | 571 | equivalent; passes | PASS |
| V4 | 583 | test_mixed_parameters[forward-default-float16] | 583 | CPU kill: AssertionError | PASS |

52 loads/differences/reaches; 51 CPU kills plus V3’s unchanged equivalence. M14 binds mutated default eps=0.001 during its actual call; its edited definition line is correctly absent from the call trace. V4’s staged line is present. Full diffs and traces live under mutants/<ID>/.

## Seal import trace summary

Collected 468 regression IDs; traced/censused 534 module origins (21 local, 513 external). Nine local arm modules are covered by each arm map; the other 12 local modules and all 513 external origins are pinned directly. The trace uses module-execution profiling plus the loaded-module origin census, and the seal pins the trace itself. Child verify_start calls verify_seal before importing its arm.

| Imported local module | SHA-256 | Child pin |
| --- | --- | --- |
| artifacts/gn1/a3/arms/candidate/tensor_cuda/__init__.py | f801ff5bb4b8487b45e79a2c98334f14cc94a80fdb096a578e9441f5a2bf7bf6 | arm map |
| artifacts/gn1/a3/arms/candidate/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so | c9b7dc610421010a8b11e8e3af74d1a9ad99928dabb98835aa6c265754daf8ce | arm map |
| artifacts/gn1/a3/arms/candidate/tensor_cuda/apa_sp2.py | 018a26b6bf7acfd85bc23e5084d0367952e1b2c81898da8585c8ecda302a9f07 | arm map |
| artifacts/gn1/a3/arms/candidate/tensor_cuda/functional.py | ef8d1fa2aa2b4f7e818d01c17377ab026d763f690de23fb8831de67f0b131c6e | arm map |
| artifacts/gn1/a3/arms/candidate/tensor_cuda/nn.py | fea952f3ebeb0694044245d6e2fc98acadbdb04ccad9ddd5790fae89be761ca5 | arm map |
| artifacts/gn1/a3/arms/candidate/tensor_cuda/optim.py | 41693c324d5ea2f081ec7777ce8165fbc65d04a7b557fce27ff6d9b9a24457ee | arm map |
| artifacts/gn1/a3/arms/candidate/tensor_cuda/quant.py | 9bb2093df1fa751022400791579afebfe904e7c616d3cd090e0746266829ecd7 | arm map |
| artifacts/gn1/a3/arms/candidate/tensor_cuda/quantization/__init__.py | d29a048409906800e68bbed774978764045451b4451eb2d31ec9d5343293658f | arm map |
| artifacts/gn1/a3/arms/candidate/tensor_cuda/quantization/affine.py | 8bf31258f8635fa4948d08805fb4909ea6a36189d40f0ee2cc589b796c5e3943 | arm map |
| artifacts/gn1/a3/trace_imports.py | 09f38f92907d8ceef6d609621d995b77713271347a8d9525e85da7b3849290ed | import_modules |
| scripts/apa_om135_core.py | cb2f5dd76c9ab7552bc5b8aca7ee0ff4462acc3f05c97156616bdf661336f849 | import_modules |
| tensor_cuda/tests/test_apa_om135_imm.py | 8ac67998194ce0cca0dd5ad507bcb0811e2fb4f4c220dc5ccc1a3419623fd4b3 | import_modules |
| tensor_cuda/tests/test_attention_phase5.py | 23e533cf0b5742cdef46f1a731cbd96c8535e41f2a05da6c6f71a9fdc28961a7 | import_modules |
| tensor_cuda/tests/test_conv_phase3b.py | 2e225ab94c2ded7b559a3c8d8f4d751be91ca848587e41574b17a3f24d8430a8 | import_modules |
| tensor_cuda/tests/test_ext_phase7.py | 8ec82d0438a28057036fca1721c990e19ced5828b5cb57918762326fd7e47455 | import_modules |
| tensor_cuda/tests/test_groupnorm_gn1.py | 0933bed97114a776832790b4045a06640b92097731f114a5913042c1419f49e7 | import_modules |
| tensor_cuda/tests/test_hy3d_engine_ops.py | 55b30170c63711b44f5efc41f4250e3c69b6e969d58378f619576227931efe9d | import_modules |
| tensor_cuda/tests/test_integration.py | ffc947c62c2028890d47030817a2611e6a936c80fffe57f5af8a58115d1bdf1c | import_modules |
| tensor_cuda/tests/test_nn_phase3.py | 2c0a66f3bff5ec2a920d4939fe1d986fc203036d5bb767dbb12693ef0478b1f5 | import_modules |
| tensor_cuda/tests/test_phase9.py | db0de76dc3d1d5d76cbcfcf6d816c77700836b1e143ae5b718a2904318f37f95 | import_modules |
| tensor_cuda/tests/test_rnn_phase5b.py | 0a87495014abe357185eaf5a589666d9e49722cf3bd21895eb170e394ad54f29 | import_modules |

Import trace: import_trace.json. The child verifies all input module pins via verify_start/verify_seal, then verifies its full arm. G4_extra.json proves the parent, child and scorer reject a bad helper pin without changing the read-only helper.

## Staged script and expected exits

[STAGED_GPU_COMMANDS.sh](STAGED_GPU_COMMANDS.sh) is the complete foreground stage; [STAGED_LINES.md](STAGED_LINES.md) lists all 56 child exits and parent exit 0. Every GPU line has --max and an inner timeout. The 29-ID audit line was executed through A3’s real child/plugin and audit stand-in. The complete scorer correctly rejects audit roots as GPU acceptance. Both valid 56-label stand-in runs score exit 0 after independent recomputation; delivery_verify.json records them. Arm regeneration --verify-seal matches all 54 maps exactly.

## Earlier results and amendments

Initial artifacts remain on disk. They are not final passing receipts and are not silently replaced:

- `initial_mutation/`: six initial G5 rows failed: matrix23 rank witnesses were masked by length checks, NumPy boolean-clause deletion was equivalent in NumPy 2.4.2, and the CPU shortcut did not count valid raising-frame ValueErrors. `AMENDMENTS.md` records isolated witnesses, predicate-negation registration and the D2/D3 counter correction.
- `hostile_runs.initial` / `hostile_gate.log.initial`: real-test passing setup/call/teardown reports exposed duplicate JUnit cases in the stand-in writer. The writer now emits calls and errors while preserving all real phases.
- `rev2_gate.log.initial`: a missing scratch runner directory stopped the final H5 control; it was prepared and the unchanged instrument rerun.
- `hostile_runs.bound20/Real_baseline_D5_registration`: the additional full baseline CPU control exceeded the inherited 20 s fixture bound, correctly produced outer-timeout and was rejected. Its separately registered 50 s CPU bound does not change staged bounds or any value/ID/exit predicate.
- `.prebaseline` and `.prelabel` trees retain earlier successful seals/runs. Final counters below refer to the final seal only; earlier runtime and witness failures remain disclosed.

The baseline registration adds exactly the 24 new D5 failures/classes observed from the real test file on the original package, preserving all 352 legacy IDs and their original numerical refusal rule. A real baseline test/plugin control proves this registration is executable; only the ext_phase7 scaffold remains synthetic in that control.

## Predictions

- **P94 HELD** for the final registered acceptance run: all 107 hostile targets meet their deciding rules, including every required REV2 reproduction and all installed-runner outcome tokens. Initial instrument failures and the bound correction are disclosed above.
- **P95 HELD** for the final amended mutant registration: V4 and all 21 D5 mutants are killed by registered CPU witnesses. This is not a claim that the initial redundant deletion mutants or masked witnesses passed; their failed receipts remain in initial_mutation/. No GPU mutant kill is claimed.

## Prior art

- Python `operator.index`, PEP 357 (2006): takes the integer-index protocol and native `int` normalization; explicit Python/NumPy boolean exclusion is the order’s policy.
- Wu and He (2018), Group Normalization: existing grouping/population-statistics formula; A3 changes domain guards only. PyTorch 2.11.0 behaviour is a live CPU observation, recorded separately.
- DeMillo, Lipton and Sayward (1978), mutation analysis; Offutt/Voas RIP model (year unverified — lead to check): single-site mutants and call-phase reach. A3 adds V4, D5 decision mutants and explicit default-binding reach. Search leads: mutation analysis, reachability infection propagation.
- Meszaros (2007), test doubles/fault injection; A2 and REV2 (2026) are the direct prior systems for these controls. A3 uses the actual installed runner for runner outcomes, and actual test/plugin reports for engine refusals.
- Python pytest report/profile hooks, GNU timeout/POSIX signal semantics, BSD flock, and SHA-256/FIPS 180-4: receipt binding, module tracing, supervision and digest maps. Exact GN1 decision conjunctions come from the immutable order and REV2, with no novelty claim.
- Higham (1996/2002), rounding analysis, and mixed precision references already present in GN1: tolerances and the prior 324-case derivation remain byte-identical. Literature citations beyond observed source/runtime facts are unverified — lead to check named works.

## Done

Only GroupNorm, its test file, A3 artifacts and approved /tmp scratch were written. Every persistent written file is listed in files_written.txt and files_written.json; sha256_maps.json maps all deliverable bytes except its own digest. DONE_VERBATIM.md repeats this full report and the literal file inventory.

No files were deleted. Optional lead cleanup after review: initial_mutation/, archived .initial/.prebaseline/.prelabel/.bound20 receipt trees and superseded staging_seal.previous.*.json files, all retained as evidence. /tmp/gn1_a3_work is left as exempt scratch, with its complete inventory under scratch_files_written.txt. No bin move was made.
