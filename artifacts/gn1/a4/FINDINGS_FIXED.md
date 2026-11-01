# Findings fixed

| Finding | Status | Fix | Receipts |
|---|---|---|---|
| R3-F01 | FIXED | Atomic early child marker; admission requires absence of every start operand. | G4_a4.json: B6_lock_outer_retry; D2_start_* |
| R3-F02 | FIXED | Kill-grace rows timeout/window-closed with 143/137 and recorded124/137 map to outer-timeout; evaluate records exceptions. | G4_a4.json: B7_grace_no_result, B7b_grace_result, B8_window_grace, D3_evaluate_never_raises |
| R3-F03 | FIXED | Incomplete started attempts are admitted, evaluated and failed; a retry cannot erase them. | G4_a4.json: C7b_started_no_result; six D2_start_* controls |
| R3-F04 | FIXED | child_result.json uses temporary-file publication with os.replace; JSON reading is inside evaluate try. | G4_a4.json: E5_truncated_incomplete and E5_truncated_completed; audit_cpu/ |
| R3-F05 | FIXED | Runtime sys.modules origins/digests augment collection pins; known tamper fails, unseen origins are unsealed lead-review records. | RUNTIME_SEAL_CENSUS.json; runtime_trace_*.json; G4_a4.json: D5_sealed_tamper and D5_unsealed |
| R3-F06 | FIXED | Both numpy.bool_ clause deletions are sealed passing equivalences with version and live operator.index proof; negations retained. | EQUIVALENCES.md; mutants.json; E_npbool_{g,c}_full_cpu.json; G4_a4.json: D6_* |
| D7 | FIXED | Scorer binds selected admitted line outer, runner max/min, inner timeout and expected exits to gpu_lines.json. Never-admitted attempts are not counted as lines. | G4_a4.json: D7_outer, D7_maximum, D7_minimum, D7_inner, D7_expected_exits; staging_exact_bounds.json |
