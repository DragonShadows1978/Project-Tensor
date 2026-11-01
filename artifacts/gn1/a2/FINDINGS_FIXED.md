# Per-finding fixes

All R1-F01 to R1-F11 fixed for the registered CPU acceptance; independent REV2 pending.

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
