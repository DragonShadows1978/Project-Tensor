# Installed runner outcome census

Source re-read live, numbered copy in runner_source.txt. Every logged token is listed; line 243 status-only return emits no row. Source SHA-256: `ad6c0a22edc3baf8fd513f1123e6d460266a7ede77b91c496b3ea38e8724e107`. The installed runner is named generically; public paths use `$HOME/`.

| Exact token | Exit | Installed source lines | G4_runner.json control |
| --- | --- | --- | --- |
| ran | Child exit including 0, 1, 126, 127, signal-derived exits | 179, 185, 189, 218; emitted at 302 | Runner_ran_candidate_focus, Runner_ran_baseline_focus, Runner_ran_cannot_start126/127; B5 |
| timeout | 124 | 184–185, 218; emitted at 302 | B2_runner_deadline |
| window-closed | 124 | 185, 198–200, 218; emitted at 302 | B3_window_closed |
| no-window | 64 (usage), 75 (not admitted) | 230; 246–249 and 302 | Runner_no_window64_usage, Runner_no_window75_H5 |
| no-lock | 76 | 266, 275; emitted at 302 | Runner_no_lock76 |
| rechecked-out | 75 | 281, 285; emitted at 302 | B1_rechecked_out_retry |

| Recorded cause | Recomputed rule | Receipt control |
| --- | --- | --- |
| outer-timeout | Exit 124/137 and missing row, or ran row with differing logged 143/137 | B4_outer_timeout; H6_hang_after_summary |
| runner-deadline | timeout row with matching 124/137 | B2_runner_deadline |
| window-closed | window-closed row with matching 124/137 | B3_window_closed |
| inner-timeout | ran row with matching 124/137 | B5_inner_timeout124; B5_inner_timeout137 |

No-child admission rule is exactly (75, no-window/rechecked-out) or (76, no-lock). Usage exit 64 is an admitted failure. Malformed/mismatching logs fail closed. Incomplete parent attempts with a child are evaluated and never retried; L45 still prevents them establishing acceptance without a parent completion record. No-child incomplete attempts are inventoried as not admitted and may retry.
