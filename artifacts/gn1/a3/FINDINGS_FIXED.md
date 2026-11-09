# Per-finding fix table

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
