# GN1 A2 report

**ACCEPT for the registered CPU gates: G1–G6 PASS.** All 11 REV1 findings have scoped fixes and receipts. P86 HELD; P87 HELD. Independent GN1-REV2 and the lead’s GPU run remain pending. No GPU execution, engine rebuild, C++/CUDA change, git write, subagent or deletion occurred. CPU stand-in assertions are not device kills or GPU numerical evidence.

Order: `orders/GN1_A2_REV1_FIXES.md`, immutable commit `13b8f77b8c3363593b256f9e60b7df197a5f8b19`. Registration: REGISTRATION.md; execution amendments: AMENDMENT_EXECUTION_NOTES.md; modifications and preparation corrections: LEDGER.md. The final content seal is staging_seal.json; previous seals and CPU attempts are retained as historical receipts. Every command used an explicit timeout below 20 minutes and TMPDIR under /tmp/gn1_a2_*, with CUDA hidden for CPU execution and thread counts at one.

## Gates

| Gate | Verdict | Evidence and scope |
|---|---|---|
| G1 | PASS | G1_scope.json; changes_vs_base.patch; changes_vs_a1.patch; nn_vs_0c6a4244.diff. Only GroupNorm and the test file differ in product scope. 7,753 tracked hashes: only the two authorized product/test files changed. 702 frozen GN1/A1/REV1 files match before/after; maps have identical SHA-256. |
| G2 | PASS | G2_cpu.json; native_cpu_runs.json; cpu_candidate_mutation_references.json: 29 references + 14 constructor refusals + divisibility refusal pass natively (44 calls). native_validation.json: 35 actual-class metadata-only guard cases. G2_shim.json: all 381 real Python test IDs pass with explicit NumPy stand-in, including new refusals and both mixed forward/backward cases. Both native arms import and collect (388 focus, 444 regression per arm), CUDA hidden. Full IDs: COLLECTED_IDS.md and ids_*.txt. |
| G3 | PASS | G3_pin.json; engine_sources_verified.json; arm_maps.json; regeneration_verified.json. Original pin/source map unchanged, all 72 source hashes match; all 43 A2 engine byte copies match pin; no symlinks/builds. |
| G4 | PASS | G4_hostile.json; G4_CONTROLS.md; hostile_runs/ and hostile_run.log: 48 controls through real sealed parent/scorer. HOSTILE_ADAPTATIONS.md lists each REV1 adaptation. |
| G5 | PASS | G5_mutants.json; mutants.json; mutants/*/trace.json; MUTANTS.md. All 41 single-site mutants load, differ and reach. 40 CPU assertion failures; V3 equivalence proof in EQUIVALENCES.md. Actual device kills remain unclaimed. |
| G6 | PASS | G6_staging.json; STAGED_GPU_COMMANDS.sh; gpu_lines.json; staging_seal.json; audit_cpu/. 45 staged labels, expected exits registered; real audit child 29 references passed, exit 0; parent passed after child exit, forge outcome ran. |

Frozen-map SHA-256: `29d5b007d3ad8243367b8ae85e1166b3a4b1e4fa62d0d4b82daca4c473e072ee` before and after. Engine module SHA-256: `c9b7dc610421010a8b11e8e3af74d1a9ad99928dabb98835aa6c265754daf8ce`. Source-map SHA-256: `98ef59cd48200fb4af38caf9dd017022f0ad032b7f42082dfd1574506b470a5c`.

## Per-finding fix table

| Finding | Fix | Receipt |
|---|---|---|
| R1-F01 | Case-insensitive session-wide device veto; real assertion/refusal kill required. | evidence.py; G4_hostile.json H2_REV1_exact/H2c_REV1_exact/H2_pattern_00–11 |
| R1-F02 | Bind child arm/suite, nn/engine/seal/root and JUnit outcomes ID by ID; exclude test_reference_* and unregistered exception classes. | G4_hostile.json H3/H3b/H4, Junit_one_ID_disagreement, Reference_only_kill, Wrong_nn_hash, Wrong_engine_hash, One_byte_engine_copy |
| R1-F03 | Scorer re-verifies START/seal/root, child/JUnit, expected exits, forge log and stored verdicts; recomputes dependencies. | G4_hostile.json H1/H1b, Stored_parent_tamper, Expected_exit_tamper, START_seal_tamper; evidence.py |
| R1-F04 | Split all sites, including both affine branches and both mean/variance expressions. One runnable expression per mutant. | build_mutants.py; G5_mutants.json; mutants/*/nn.diff and trace.json |
| R1-F05 | Both distinct parameter dtype configurations get forward/backward cases; V1/V2 assert-fail; V3 has a post-D5 equivalence proof. | G2_shim.json; mutants/V1/trace.json; mutants/V2/trace.json; mutants/V3/trace.json; EQUIVALENCES.md |
| R1-F06 | Positive integer/bool checks before divisibility; rank/channel and input/weight/bias dtype refusals before tensor ops. | nn_vs_0c6a4244.diff; native_validation.json (35 cases); G2_shim.json; pytorch_behavior.json |
| R1-F07 | Candidate regression permits exits 0/1; old passes must still pass and each candidate failure must match a baseline failure class. | gpu_lines.json; G4_hostile.json D6_normal, D6_regression_legacy_error and negative D6 controls |
| R1-F08 | Numbered not-admitted 75/76 attempts do not consume a label; scorer lists attempts and selects last admitted. | G4_hostile.json H5_retry_75/H5_retry_76; hostile_runs/H5_retry_*/ |
| R1-F09 | Replace multiplier prose with per-op first-order bound and final-rounding composition; preserve every number. | TOLERANCES.md; derive_tolerances.py; tolerance_bounds.json (324 IDs) |
| R1-F10 | 43 engine byte copies, built and child-start hash checked; deterministic arm regeneration from order commit + D5/mutant recipes + pin. | regenerate_arms.py; regeneration_verified.json; arm_maps.json; G3_pin.json |
| R1-F11 | Read runner’s final nine-field log; record ran/timeout/window-closed/no-window/no-lock plus exit. Missing log stays missing. | G4_hostile.json D10_timeout/D10_window_closed/H6_hang_after_summary; parent_result.json forge_outcome |

## nn.py diff against 0c6a4244

```diff
diff --git a/tensor_cuda/tensor_cuda/nn.py b/tensor_cuda/tensor_cuda/nn.py
index 21342154..7e78267b 100644
--- a/tensor_cuda/tensor_cuda/nn.py
+++ b/tensor_cuda/tensor_cuda/nn.py
@@ -512,6 +512,14 @@ class GroupNorm(Module):
         super().__init__()
         # Prior art: Wu and He (2018), Group Normalization; PyTorch GroupNorm
         # divisibility refusal (docs unverified — lead to check: "GroupNorm").
+        # A2 domain checks, inspired by PyTorch GroupNorm argument validation;
+        # exact positive-int/bool policy is the GN1 A2 requirement.
+        if (isinstance(num_groups, bool) or not isinstance(num_groups, int)
+                or num_groups <= 0):
+            raise ValueError(f"num_groups ({num_groups!r}) must be a positive integer (bool excluded)")
+        if (isinstance(num_channels, bool) or not isinstance(num_channels, int)
+                or num_channels <= 0):
+            raise ValueError(f"num_channels ({num_channels!r}) must be a positive integer (bool excluded)")
         if num_channels % num_groups != 0:
             raise ValueError(
                 f"num_channels ({num_channels}) must be divisible by "
@@ -522,6 +530,18 @@ class GroupNorm(Module):
         self.bias = parameter(np.zeros(num_channels))
 
     def forward(self, x):
+        # Prior art: PyTorch GroupNorm rank/channel checks (version unverified
+        # — lead to check); A2 explicitly limits this engine's dtype domain.
+        if x.ndim < 2:
+            raise ValueError(f"GroupNorm input rank ({x.ndim}) must be at least 2")
+        if x.shape[1] != self.C:
+            raise ValueError(f"GroupNorm input channels ({x.shape[1]}) must equal num_channels ({self.C})")
+        if x.dtype not in ("float32", "float16", "bfloat16"):
+            raise TypeError(f"GroupNorm input dtype {x.dtype!r} is unsupported")
+        if self.weight.dtype not in ("float32", "float16", "bfloat16"):
+            raise TypeError(f"GroupNorm weight dtype {self.weight.dtype!r} is unsupported")
+        if self.bias.dtype not in ("float32", "float16", "bfloat16"):
+            raise TypeError(f"GroupNorm bias dtype {self.bias.dtype!r} is unsupported")
         # Prior art: PyTorch half normalization uses float32 accumulation;
         # Micikevicius et al. (2018), Mixed Precision Training (both unverified
         # — lead to check: "PyTorch group_norm acc_type half"). Round 1's
```

## G4 control table

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

## Mutant table with witness traces

Witness prefix is `tensor_cuda/tests/test_groupnorm_gn1.py::`; traces are real Python line events with explicit CPU stand-in. M14’s definition line is traced during import and its test calls the constructor without eps. Every other mutant reaches its expression during its test. Every mutant is exactly one replacement expression; no multi-line exception is needed.

| Mutant | Witness (prefix below) | Edited → reached lines | CPU outcome | Trace |
|---|---|---|---|---|
| D5_bias_dtype | `test_dtype_refusal[int64-bias]` | [543] → [543] | AssertionError | [trace](mutants/D5_bias_dtype/trace.json) |
| D5_c_bool | `test_positive_integer_refusal[true-num_channels]` | [520] → [520] | AssertionError | [trace](mutants/D5_c_bool/trace.json) |
| D5_c_int | `test_positive_integer_refusal[float-num_channels]` | [520] → [520] | AssertionError | [trace](mutants/D5_c_int/trace.json) |
| D5_c_positive | `test_positive_integer_refusal[negative-num_channels]` | [521] → [521] | AssertionError | [trace](mutants/D5_c_positive/trace.json) |
| D5_channels | `test_shape_refusal[channels]` | [537] → [537] | AssertionError | [trace](mutants/D5_channels/trace.json) |
| D5_g_bool | `test_positive_integer_refusal[true-num_groups]` | [517] → [517] | AssertionError | [trace](mutants/D5_g_bool/trace.json) |
| D5_g_int | `test_positive_integer_refusal[float-num_groups]` | [517] → [517] | AssertionError | [trace](mutants/D5_g_int/trace.json) |
| D5_g_positive | `test_positive_integer_refusal[negative-num_groups]` | [518] → [518] | AssertionError | [trace](mutants/D5_g_positive/trace.json) |
| D5_input_dtype | `test_dtype_refusal[int64-input]` | [539] → [539] | AssertionError | [trace](mutants/D5_input_dtype/trace.json) |
| D5_rank | `test_shape_refusal[rank1]` | [535] → [535] | AssertionError | [trace](mutants/D5_rank/trace.json) |
| D5_weight_dtype | `test_dtype_refusal[int64-weight]` | [541] → [541] | AssertionError | [trace](mutants/D5_weight_dtype/trace.json) |
| M01 | `test_forward_values[float32-r2-n1-g2-default]` | [558] → [558] | AssertionError | [trace](mutants/M01/trace.json) |
| M02 | `test_near_epsilon[forward-float32]` | [559] → [559] | AssertionError | [trace](mutants/M02/trace.json) |
| M03 | `test_near_epsilon[forward-float32]` | [559] → [559] | AssertionError | [trace](mutants/M03/trace.json) |
| M04m | `test_forward_values[float32-r3-n2-g2-default]` | [558] → [558] | AssertionError | [trace](mutants/M04m/trace.json) |
| M04v | `test_forward_values[float32-r3-n2-g2-default]` | [558] → [558] | AssertionError | [trace](mutants/M04v/trace.json) |
| M05 | `test_forward_values[float32-r2-n1-g2-default]` | [555] → [555] | AssertionError | [trace](mutants/M05/trace.json) |
| M06in | `test_forward_values[float32-r3-n2-g2-default]` | [555] → [555] | AssertionError | [trace](mutants/M06in/trace.json) |
| M06out | `test_forward_values[float32-r3-n2-g2-default]` | [560] → [560] | AssertionError | [trace](mutants/M06out/trace.json) |
| M07cast | `test_forward_values[float16-r3-n2-g2-default]` | [567] → [567] | AssertionError | [trace](mutants/M07cast/trace.json) |
| M07fast | `test_forward_values[float32-r3-n2-g2-default]` | [564] → [564] | AssertionError | [trace](mutants/M07fast/trace.json) |
| M08cast | `test_forward_values[float16-r3-n2-g2-default]` | [567] → [567] | AssertionError | [trace](mutants/M08cast/trace.json) |
| M08fast | `test_forward_values[float32-r3-n2-g2-default]` | [564] → [564] | AssertionError | [trace](mutants/M08fast/trace.json) |
| M09cast | `test_forward_values[float16-r3-n2-g2-default]` | [567] → [567] | AssertionError | [trace](mutants/M09cast/trace.json) |
| M09fast | `test_forward_values[float32-r3-n2-g2-default]` | [564] → [564] | AssertionError | [trace](mutants/M09fast/trace.json) |
| M10 | `test_nondivisible_refusal` | [523] → [523] | AssertionError | [trace](mutants/M10/trace.json) |
| M11 | `test_forward_values[float32-r3-n2-g2-default]` | [561] → [561] | AssertionError | [trace](mutants/M11/trace.json) |
| M12 | `test_forward_values[float32-r3-n2-g2-default]` | [559] → [559] | AssertionError | [trace](mutants/M12/trace.json) |
| M13 | `test_forward_values[float32-r3-n2-g2-default]` | [559] → [559] | AssertionError | [trace](mutants/M13/trace.json) |
| M14 | `test_constructor_defaults` | [511] → [511] | AssertionError | [trace](mutants/M14/trace.json) |
| M15b | `test_forward_values[float16-r3-n2-g2-default]` | [566] → [566] | AssertionError | [trace](mutants/M15b/trace.json) |
| M15w | `test_forward_values[float16-r3-n2-g2-default]` | [565] → [565] | AssertionError | [trace](mutants/M15w/trace.json) |
| M15x | `test_forward_values[float16-r3-n2-g2-default]` | [550] → [550] | AssertionError | [trace](mutants/M15x/trace.json) |
| M16 | `test_mixed_parameters[forward-default-float16]` | [568] → [568] | AssertionError | [trace](mutants/M16/trace.json) |
| M17b | `test_distinct_parameter_dtypes[forward-w32-b16]` | [566] → [566] | AssertionError | [trace](mutants/M17b/trace.json) |
| M17w | `test_distinct_parameter_dtypes[forward-w16-b32]` | [565] → [565] | AssertionError | [trace](mutants/M17w/trace.json) |
| M18b | `test_mixed_parameters[forward-half-float16]` | [566] → [566] | AssertionError | [trace](mutants/M18b/trace.json) |
| M18w | `test_mixed_parameters[forward-half-float16]` | [565] → [565] | AssertionError | [trace](mutants/M18w/trace.json) |
| V1 | `test_distinct_parameter_dtypes[forward-w32-b16]` | [563] → [563] | AssertionError | [trace](mutants/V1/trace.json) |
| V2 | `test_distinct_parameter_dtypes[forward-w16-b32]` | [562] → [562] | AssertionError | [trace](mutants/V2/trace.json) |
| V3 | `test_forward_values[float16-r3-n2-g2-default]` | [550] → [550] | equivalent (V3 proof) | [trace](mutants/V3/trace.json) |

## D8 bounds and margins


Evidence class: finite-census, first-order numerical error analysis, not a
CUDA run or a universal accuracy proof. Every A1 tolerance number is retained
exactly (`tolerance_bounds.json` compares the registration). `derive_tolerances.py`
reuses REV1 `gn_emulate.py::running_bound` without editing it and adds the four
new mixed-parameter value/gradient cases: 324 value IDs in total. Inputs are
represented source values, so their initial error is zero. No output of the
candidate enters the reference or the bound.

Prior art: Wilkinson running error analysis (1960s), Higham (1996/2002),
Accuracy and Stability of Numerical Algorithms, ch. 3, for error propagation
and gamma bounds; IEEE 754 rounding (1985/2019); REV1 graph transcription.
Literature unverified — lead to check named sources.

## Operation model and powf assumption

Let u=2^-24, gamma_k=ku/(1-ku), E(v) the absolute error bound, and |.| act
componentwise. For represented operands, E(input)=0. For a sum of m terms,
E(sum v_i) <= sum E(v_i)+gamma_(m-1) sum |v_i|. This covers any sequential
or shallower reduction tree; mean includes rounded reciprocal 1/m with
E(1/m)<=u/m and the final multiply. Broadcast/reshape/permutation are exact.
For a product, E(ab)<=|b|E(a)+|a|E(b)+u|ab|; for a sum/difference,
E(a+-b)<=E(a)+E(b)+u|a+-b|. Products of errors are discarded: these are
explicitly first-order bounds, not interval certificates. All finite cases
avoid overflow in FP32; underflow of final low precision destinations is
covered below. Singleton groups have exact mean, zero centering and dx=0;
the cancellation is used, rather than bounding two independent copies of x.

Assumption: CUDA Math API single-precision `powf` maximum error **4 ulp**,
including powers -1/2 and -3/2. **Unverified — lead to check:** CUDA 12.6 Math
API Standard Functions powf maximum ulp error (and compiler fast-math mode).
One ulp/|z| <= 2u, so 4 ulp gives relative error <=8u. This is an assumption
about both engine pow kernels, not an observed accuracy claim. For z>0,
E(z^p)<=|p|z^(p-1)E(z)+8u|z^p|. E(eps)<=u|eps| covers scalar storage.

## Forward and backward recurrences

For each group of M values:
mu=mean(x); d=x-mu; v=mean(d*d); a=v+eps; s=a^(-1/2);
h=d*s; t=h*w; y=t+b. Apply the sum/product rules at each node. In particular,
E(s)<=0.5*a^(-3/2)E(a)+8u*s. Affine multiplication and addition each have
their own rounding term; parameter upcasts are exact for represented FP16/BF16.

For the engine VJP graph (p is represented projection), q=p*w,
c=q*s, z=sum(q*d), k=-0.5*a^(-3/2), r=z*k/M,
f=2*r*d, j=mean(-f), l=mean(-c), dx=((f+j)+c)+l.
E(k) uses 1.5*a^(-5/2)E(a)+8u*a^(-3/2) before its -0.5 multiply.
Every reduction and every addition in the displayed accumulation gets the
operation rule above. dw=sum(p*h), db=sum(p) over batch/spatial axes, also with
gamma_(m-1). The executable recurrence records the worst element, operand,
ID, reference, error and exact budget, rather than introducing a fitted factor.

The FP32 compute-budget maximum is 0.907336823472800 of the backward budget, on
test_low_variance[backward-var1e-4-half-bfloat16] (dx). Remaining compute margin: 9.266317653%.
The largest forward compute ratio is 0.485682114204027. This reproduces REV1’s 0.907
worst-conditioned result and includes the new mixed configurations.

## Destination rounding and unchanged budgets

If FP32 compute has |error|<=A+R|ref|, one final destination rounding with
unit roundoff u_d and underflow half-step s_d gives
E_final <= (1+u_d)E_compute+u_d|ref|+s_d, hence the registered pair
A_d=(1+u_d)A+s_d, R_d=(1+u_d)R+u_d. FP16: u_d=2^-11, s_d=2^-25.
BF16: u_d=2^-8, s_d=2^-134. For an FP32 destination u_d=s_d=0 here because
its operation rounding is already in E_compute. Each parameter gradient uses
its own leaf dtype; it need not equal the input/output dtype.

| Destination | Phase | atol | rtol | Worst bound / budget | Margin | Worst case / operand |
|---|---|---|---|---|---|---|
| float32 | forward | 3e-05 | 3e-05 | 0.485682114204027 | 51.431788580% | `test_near_epsilon[forward-float32]` / y |
| float32 | backward | 0.0003 | 0.0003 | 0.803250678008349 | 19.674932199% | `test_near_epsilon[backward-float32]` / dx |
| float16 | forward | 3.0044450759887697e-05 | 0.0005182958984375 | 0.943650728613587 | 5.634927139% | `test_forward_values[float16-r5-n2-g2-default]` / y |
| float16 | backward | 0.00030017628669738765 | 0.000788427734375 | 0.937804853431982 | 6.219514657% | `test_low_variance[backward-var1e-4-half-float16]` / dx |
| bfloat16 | forward | 3.01171875e-05 | 0.0039363671875 | 0.992432055320783 | 0.756794468% | `test_forward_values[bfloat16-r5-n2-g2-default]` / y |
| bfloat16 | backward | 0.00030117187499999995 | 0.004207421875 | 0.981971213924931 | 1.802878608% | `test_low_variance[backward-var1e-4-half-bfloat16]` / dx |

Margins are first-order bound margins, not measured GPU errors. Even the
smallest final BF16 margin remains positive in this finite census. The bound
shows no unsound budget; none was widened. P87 HELD within these stated
assumptions. Network literature verification belongs to the lead.

The reference-only central difference checks retain their separate 2e-7
assertions: original h=2^-18 at eps=0.125 and low-variance h=2^-25. They validate
the analytic float64 oracle through independent finite computation (29 IDs,
G2 native receipt); they are not dtype error budgets or device error bounds.
The classical truncation term is h^2 sup|third derivative of F|/6 and evaluation
term is (E(F(x+h))+E(F(x-h)))/(2h). We do not promote the passing finite checks
to a uniform float64 derivative certificate. The former unexplained 7/70/20
factors are not used anywhere in the device-budget derivation.

## PyTorch comparison


CPU observation of PyTorch 2.11.0+cu130; A2 requirements govern the product even where PyTorch differs. No GPU queried or allocated.

| Case | PyTorch result |
|---|---|
| nondivisible | ValueError |
| num_groups/zero | ZeroDivisionError |
| num_groups/negative | accepted constructor |
| num_groups/float | accepted constructor |
| num_groups/true | accepted constructor |
| num_groups/false | ZeroDivisionError |
| num_groups/string | TypeError |
| num_groups/none | TypeError |
| num_channels/zero | accepted constructor |
| num_channels/negative | RuntimeError |
| num_channels/float | TypeError |
| num_channels/true | ValueError |
| num_channels/false | TypeError |
| num_channels/string | TypeError |
| num_channels/none | TypeError |
| rank0 | RuntimeError |
| rank1 | RuntimeError |
| channels | RuntimeError |
| torch.int64/input | RuntimeError |
| torch.int64/weight | RuntimeError |
| torch.int64/bias | RuntimeError |
| torch.bool/input | RuntimeError |
| torch.bool/weight | RuntimeError |
| torch.bool/bias | RuntimeError |
| torch.uint8/input | RuntimeError |
| torch.uint8/weight | RuntimeError |
| torch.uint8/bias | RuntimeError |
| mixed/torch.float32/torch.float16 | RuntimeError |
| mixed/torch.float16/torch.float32 | RuntimeError |

Exact exception messages and operands: pytorch_behavior.json. Both mixed configurations refuse during forward; there is no backward outcome to claim.

## Staged lines and expected exits

| Label | Arm/suite | Expected child exits | Expected parent | Inner / max / outer (s) |
|---|---|---|---|---|
| baseline_focus | baseline/focus | [1] | 0 | 240 / 300 / 330 |
| candidate_focus | candidate/focus | [0] | 0 | 240 / 300 / 330 |
| baseline_regression | baseline/regression | [1] | 0 | 840 / 900 / 930 |
| candidate_regression | candidate/regression | [0, 1] | 0 | 840 / 900 / 930 |
| D5_bias_dtype | D5_bias_dtype/mutation | [1] | 0 | 180 / 240 / 270 |
| D5_c_bool | D5_c_bool/mutation | [1] | 0 | 180 / 240 / 270 |
| D5_c_int | D5_c_int/mutation | [1] | 0 | 180 / 240 / 270 |
| D5_c_positive | D5_c_positive/mutation | [1] | 0 | 180 / 240 / 270 |
| D5_channels | D5_channels/mutation | [1] | 0 | 180 / 240 / 270 |
| D5_g_bool | D5_g_bool/mutation | [1] | 0 | 180 / 240 / 270 |
| D5_g_int | D5_g_int/mutation | [1] | 0 | 180 / 240 / 270 |
| D5_g_positive | D5_g_positive/mutation | [1] | 0 | 180 / 240 / 270 |
| D5_input_dtype | D5_input_dtype/mutation | [1] | 0 | 180 / 240 / 270 |
| D5_rank | D5_rank/mutation | [1] | 0 | 180 / 240 / 270 |
| D5_weight_dtype | D5_weight_dtype/mutation | [1] | 0 | 180 / 240 / 270 |
| M01 | M01/mutation | [1] | 0 | 180 / 240 / 270 |
| M02 | M02/mutation | [1] | 0 | 180 / 240 / 270 |
| M03 | M03/mutation | [1] | 0 | 180 / 240 / 270 |
| M04m | M04m/mutation | [1] | 0 | 180 / 240 / 270 |
| M04v | M04v/mutation | [1] | 0 | 180 / 240 / 270 |
| M05 | M05/mutation | [1] | 0 | 180 / 240 / 270 |
| M06in | M06in/mutation | [1] | 0 | 180 / 240 / 270 |
| M06out | M06out/mutation | [1] | 0 | 180 / 240 / 270 |
| M07cast | M07cast/mutation | [1] | 0 | 180 / 240 / 270 |
| M07fast | M07fast/mutation | [1] | 0 | 180 / 240 / 270 |
| M08cast | M08cast/mutation | [1] | 0 | 180 / 240 / 270 |
| M08fast | M08fast/mutation | [1] | 0 | 180 / 240 / 270 |
| M09cast | M09cast/mutation | [1] | 0 | 180 / 240 / 270 |
| M09fast | M09fast/mutation | [1] | 0 | 180 / 240 / 270 |
| M10 | M10/mutation | [1] | 0 | 180 / 240 / 270 |
| M11 | M11/mutation | [1] | 0 | 180 / 240 / 270 |
| M12 | M12/mutation | [1] | 0 | 180 / 240 / 270 |
| M13 | M13/mutation | [1] | 0 | 180 / 240 / 270 |
| M14 | M14/mutation | [1] | 0 | 180 / 240 / 270 |
| M15b | M15b/mutation | [1] | 0 | 180 / 240 / 270 |
| M15w | M15w/mutation | [1] | 0 | 180 / 240 / 270 |
| M15x | M15x/mutation | [1] | 0 | 180 / 240 / 270 |
| M16 | M16/mutation | [1] | 0 | 180 / 240 / 270 |
| M17b | M17b/mutation | [1] | 0 | 180 / 240 / 270 |
| M17w | M17w/mutation | [1] | 0 | 180 / 240 / 270 |
| M18b | M18b/mutation | [1] | 0 | 180 / 240 / 270 |
| M18w | M18w/mutation | [1] | 0 | 180 / 240 / 270 |
| V1 | V1/mutation | [1] | 0 | 180 / 240 / 270 |
| V2 | V2/mutation | [1] | 0 | 180 / 240 / 270 |
| V3 | V3/mutation | [0] | 0 | 180 / 240 / 270 |

Exact staged script (no GPU line executed by this seat):

```bash
#!/usr/bin/env bash
# Prior art: GNU timeout/POSIX shell; A1 stage layout. A2 D6/D7 expected exits.
# One label per foreground command. Every parent expected 0; no default RUNS.
set -euo pipefail
: "${RUNS:?RUNS required; reuse the same root for all 45 labels}"
[[ "$RUNS" == /* ]] || { echo "RUNS must be absolute" >&2; exit 64; }
TOOL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$TOOL"
export RUNS
export TMPDIR="${TMPDIR:-/tmp/gn1_a2_staged}" PYTHONDONTWRITEBYTECODE=1
[[ "$TMPDIR" == /tmp/gn1_a2_* ]] || { echo "TMPDIR must be /tmp/gn1_a2_*" >&2; exit 64; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
GPU_RUN=forge-gpu-run
label="${1:?select a registered label, or --audit}"
if [[ "$label" == --audit ]]; then
    export GN1_AUDIT=1 CUDA_VISIBLE_DEVICES=
    GPU_RUN="$TOOL/forge_gpu_run_standin.py"
    label=candidate_focus
else
    unset GN1_AUDIT
fi
mkdir -p -- "$TMPDIR"
timeout 20s python3 start_gpu_run.py
case "$label" in
  baseline_focus) # expected child 1, parent 0; inner 240s, --max 300s, outer 330s
    timeout 350s python3 run_gpu_line.py baseline_focus 1 330 "$GPU_RUN" --min 250 --max 300 --label gn1-a2-baseline_focus -- timeout --kill-after=10s 240s python3 run_gpu_tests.py baseline focus
    ;;
  candidate_focus) # expected child 0, parent 0; inner 240s, --max 300s, outer 330s
    timeout 350s python3 run_gpu_line.py candidate_focus 0 330 "$GPU_RUN" --min 250 --max 300 --label gn1-a2-candidate_focus -- timeout --kill-after=10s 240s python3 run_gpu_tests.py candidate focus
    ;;
  baseline_regression) # expected child 1, parent 0; inner 840s, --max 900s, outer 930s
    timeout 950s python3 run_gpu_line.py baseline_regression 1 930 "$GPU_RUN" --min 850 --max 900 --label gn1-a2-baseline_regression -- timeout --kill-after=10s 840s python3 run_gpu_tests.py baseline regression
    ;;
  candidate_regression) # expected child 0,1, parent 0; inner 840s, --max 900s, outer 930s
    timeout 950s python3 run_gpu_line.py candidate_regression 0,1 930 "$GPU_RUN" --min 850 --max 900 --label gn1-a2-candidate_regression -- timeout --kill-after=10s 840s python3 run_gpu_tests.py candidate regression
    ;;
  D5_bias_dtype) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py D5_bias_dtype 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-D5_bias_dtype -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_bias_dtype mutation
    ;;
  D5_c_bool) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py D5_c_bool 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-D5_c_bool -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_c_bool mutation
    ;;
  D5_c_int) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py D5_c_int 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-D5_c_int -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_c_int mutation
    ;;
  D5_c_positive) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py D5_c_positive 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-D5_c_positive -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_c_positive mutation
    ;;
  D5_channels) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py D5_channels 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-D5_channels -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_channels mutation
    ;;
  D5_g_bool) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py D5_g_bool 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-D5_g_bool -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_g_bool mutation
    ;;
  D5_g_int) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py D5_g_int 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-D5_g_int -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_g_int mutation
    ;;
  D5_g_positive) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py D5_g_positive 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-D5_g_positive -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_g_positive mutation
    ;;
  D5_input_dtype) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py D5_input_dtype 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-D5_input_dtype -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_input_dtype mutation
    ;;
  D5_rank) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py D5_rank 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-D5_rank -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_rank mutation
    ;;
  D5_weight_dtype) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py D5_weight_dtype 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-D5_weight_dtype -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_weight_dtype mutation
    ;;
  M01) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M01 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M01 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M01 mutation
    ;;
  M02) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M02 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M02 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M02 mutation
    ;;
  M03) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M03 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M03 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M03 mutation
    ;;
  M04m) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M04m 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M04m -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M04m mutation
    ;;
  M04v) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M04v 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M04v -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M04v mutation
    ;;
  M05) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M05 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M05 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M05 mutation
    ;;
  M06in) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M06in 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M06in -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M06in mutation
    ;;
  M06out) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M06out 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M06out -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M06out mutation
    ;;
  M07cast) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M07cast 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M07cast -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M07cast mutation
    ;;
  M07fast) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M07fast 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M07fast -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M07fast mutation
    ;;
  M08cast) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M08cast 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M08cast -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M08cast mutation
    ;;
  M08fast) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M08fast 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M08fast -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M08fast mutation
    ;;
  M09cast) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M09cast 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M09cast -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M09cast mutation
    ;;
  M09fast) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M09fast 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M09fast -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M09fast mutation
    ;;
  M10) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M10 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M10 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M10 mutation
    ;;
  M11) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M11 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M11 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M11 mutation
    ;;
  M12) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M12 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M12 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M12 mutation
    ;;
  M13) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M13 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M13 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M13 mutation
    ;;
  M14) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M14 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M14 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M14 mutation
    ;;
  M15b) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M15b 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M15b -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M15b mutation
    ;;
  M15w) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M15w 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M15w -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M15w mutation
    ;;
  M15x) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M15x 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M15x -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M15x mutation
    ;;
  M16) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M16 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M16 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M16 mutation
    ;;
  M17b) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M17b 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M17b -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M17b mutation
    ;;
  M17w) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M17w 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M17w -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M17w mutation
    ;;
  M18b) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M18b 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M18b -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M18b mutation
    ;;
  M18w) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M18w 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M18w -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M18w mutation
    ;;
  V1) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py V1 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-V1 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py V1 mutation
    ;;
  V2) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py V2 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-V2 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py V2 mutation
    ;;
  V3) # expected child 0, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py V3 0 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-V3 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py V3 mutation
    ;;
  *) echo "unknown GN1 A2 label: $label" >&2; exit 64 ;;
esac
# After every admitted label: RUNS=/same/root timeout 30s python3 score_gpu.py (expected 0).
```

## Regeneration

`regenerate_arms.py --verify-seal` computes every package byte from the immutable
order commit, the registered D5 insertions and single-site mutant recipe, and
the module pin. A1 mutant definitions are read from that commit using read-only
git. Baseline package bytes come from 0e1dea3. It copies the pinned module,
checks SHA-256 while building, refuses existing different bytes/symlinks, and
compares the regenerated maps with the seal. The child repeats arm/module
checks before importing and asserts the engine is its own copy. Two receipts (initial
and final) cover deterministic agreement (no rebuild).
The script and recipes are the lead’s commit record; arms are generated local
artifacts and must not be committed. This seat performs no git writes.

Reproduction from the worktree (after obtaining the pinned module):

```bash
TMPDIR=/tmp/gn1_a2_reproduce CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 90s python3 artifacts/gn1/a2/regenerate_arms.py --verify-seal
```

## Predictions

- P86 HELD — all 48 final controls meet their targets; H2c, H3b and H6 stay rejected. H5 refuses each lease-denied attempt and admits its valid retry; it does not reject the retry.
- P87 HELD — every A1 tolerance number is exactly retained. FP32 compute bound peaks at 0.9073368234728002 of budget. Final destination rounding leaves the smaller, explicitly recorded BF16 margin; no budget was widened.

## Prior art

Wu and He (2018), Group Normalization, for the existing population-statistic
formula (untouched); PyTorch GroupNorm for validation context, with concrete
CPU behavior verified at version 2.11.0+cu130 (A2’s positive-int/refusal policy
and mixed dtype acceptance are order requirements, not claims of PyTorch
parity). DeMillo, Lipton and Sayward (1978), mutation adequacy; Offutt/Voas RIP
model, for single-site mutants and witness reach. Wilkinson (1960s) and Higham
(1996/2002), for first-order running error recurrences. IEEE 754 (1985/2019),
for rounding/underflow; NIST SHA-256 (2001), for seals; pytest/JUnit and GNU
timeout/Python subprocess, for reports and foreground supervision. Meszaros
(2007), fault-injection test stubs; Karpathy micrograd (2020), expression-graph
reverse mode used only in the CPU stand-in; classical central differences,
Fornberg (1988), for retained reference checks. Bibliographic details and
CUDA powf 4-ulp accuracy are unverified — lead to check named titles/authors
and CUDA 12.6 Math API powf maximum ulp error. V3’s elementary case analysis:
no specific prior art known to me. A2 contributes the order-specific guards,
receipt bindings, attempt records, single-site recut and finite-census bound
wiring. Prior art is also recorded at code sites and in LEDGER.md.

## Failures retained and limits

The first stand-in import failed with `AttributeError: module 'tensor_cuda' has no attribute 'tensor'` because pytest replaced a module lacking
package metadata; the exact quoted exception is in G2_shim.json.attempt1 and cpu_candidate_shim.log.attempt1. The corrected shim
uses package metadata and the real suite passes. Initial tolerance JSON
serialization failed on NumPy int64 indices; indices now serialize as Python
ints. Two unmatched apply_patch contexts and one inline documentation quoting
`SyntaxError: invalid decimal literal` changed no files. The first hostile run was RED only because its
missing-ID control expected a later reason string; the exact reasons were `JUnit/summary outcomes disagree ID by ID` and `baseline-passing IDs no longer pass`; its control had expected `missing test calls`. That
expectation was corrected to the registered D6 rule. Earlier hostile
runs and seals are retained; the final 48 controls all passed under the final
seal. A final scorer rerun after restoring deliberately tampered receipts
accepts the full synthetic run (`full_synthetic_restored.log`); that acceptance
is CPU harness sensitivity, never GPU acceptance.

Not claimed fixed or established: GPU values/gradients, real device mutant
kills, the real regression outcome, FP16-POW1, reshape(-1), other layers,
performance, or independent review. The lead runs GN1-REV2 before any GPU run.

## Done

Every workspace file written is listed exactly in files_written.txt/json;
the two product/test paths and every A2 arm/module/mutant/receipt/tool are
included. DONE_VERBATIM.md embeds the report and that exhaustive list. All
files under A2 are intentional deliverables or retained provenance; no other
workspace path was edited. No deletion or DELETE_ME move occurred.

Leave-and-list for the lead: `/tmp/gn1_a2_boot` is scratch (including the saved
one-byte-control engine copy and outside-guard directory). Generated A2 arms
are retained and are not for commit; the lead may bin them after review, then
regenerate from the script. The .attempt* trees and previous seals
remain as historical receipts, not active runs. Nothing was silently removed.

Wall-clock since A2 start snapshot: 2617.5 s (budget 7200 s); every single command bound <1200 s.
