# Per-finding fix table

| Finding | Implemented fix | Deciding receipts |
|---|---|---|
| R4-F01 MINOR | D2 compares the entire runner argument list and entire inner argv to gpu_lines.json; every alternative spelling fails. | G4_a5.json: REV4_max_equals, REV4_min_abbrev, REV4_min_equals, REV4_inner_short_k, REV4_inner_kill_after_space; OPTION_CENSUS.json; 121 option-spelling controls and 16 additional grammar controls. |
| R4-F02 MINOR | D3 explicitly stages grace10/poll5, outer kill-after20, script margin40; parent normalizes negative exits and records raw_exit. | G4_a5.json: N2b/N2c unset_grace and D3; Raw_SIGKILL_normalisation (-9 ->137); G4_p116.json: both actual child PIDs absent immediately after return; D3_ARITHMETIC.md. |
| R4-F03 NIT | D4 timeout_cause is total, maps non-start124/137 to outer-timeout and unknown attribution to unattributed; all parent evaluation paths are protected and prelaunch failures are JSON refusals. | G4_a5.json: N5; inherited G4_a4.json: D3_evaluate_never_raises, E5 and start-evidence controls. |
| R4-F04 NIT | D5 writes/fsyncs a temporary receipt, then os.link publishes without overwrite; corrupt prior receipts return a JSON error. The temporary is retained. | G4_a5.json: N4 and Atomic_publication (complete matching content, shared inode, link count2). |
