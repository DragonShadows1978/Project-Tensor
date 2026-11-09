# GN1 A5 report

**G1–G6 PASS — technical acceptance on CPU. All four GN1-REV4 findings fixed; 283/283 hostile targets meet their registered targets. GN1-REV5 and the lead's first GPU run remain outstanding.**

Evidence class: CPU test/suite execution, installed-runner interruption controls, exact byte comparisons and source reasoning. No device execution, product changes, rebuild, performance result or model-quality claim. Branch gn1, order commit 2ed9586eb478371f810e164e92754810768692cd. Delivery seal 9980e0e6eba15e9b714c711cc8bf5e5bce5d034eec06ec91bcfc41976af700eb. The immutable order is orders/GN1_A5_REV4_FIXES.md; registration precedes every gate.

## Gates

| Gate | Result | Evidence / scope |
|---|---|---|
| G1 | PASS | G1_scope.json; frozen_before.json equals frozen_after.json,20,288 files; product and test match cf922891; no earlier-round/review bytes changed; tooling_a4_to_a5.diff and TOOLING_DIFF_MAP.md. |
| G2 | PASS | G2_checks.json; NumPy2.4.2, pytest9.0.2; both native arms import/collect with CUDA hidden; all five ID files match A4 (focus412, mutation405, regression468);44 native reference/guard calls,57 metadata guard cases;405 real GroupNorm tests pass under the explicit CPU NumPy stand-in. |
| G3 | PASS | G3_pin.json; original module pin and72-source map unchanged;56 arm module copies match c9b7dc610421010a8b11e8e3af74d1a9ad99928dabb98835aa6c265754daf8ce; no arm symlinks; no build. |
| G4 | PASS | G4_summary.json, G4_TABLE.md and seven raw batches;283/283 targets; each names its deciding rule and parent/scorer receipts. |
| G5 | PASS | G5_checks.json and G5_mutants.json;54 unchanged A4 mutants/witnesses reached,51 CPU kills and3 registered equivalences; delivery-tooling recheck passes. CPU kills establish this finite stand-in lane only. |
| G6 | PASS | G6_staging.json, staging_exact_bounds.json, regeneration_verified.json, honest_labels_verified.json;58 exact staged lines,56 arms regenerate against seal, two honest58-label CPU stand-in roots recompute as passing with0 unsealed modules; staged audit29/29 reference calls, child0/parent0. |

## Per-finding fixes

| Finding | Implemented fix | Deciding receipts |
|---|---|---|
| R4-F01 MINOR | D2 compares the entire runner argument list and entire inner argv to gpu_lines.json; every alternative spelling fails. | G4_a5.json: REV4_max_equals, REV4_min_abbrev, REV4_min_equals, REV4_inner_short_k, REV4_inner_kill_after_space; OPTION_CENSUS.json; 121 option-spelling controls and 16 additional grammar controls. |
| R4-F02 MINOR | D3 explicitly stages grace10/poll5, outer kill-after20, script margin40; parent normalizes negative exits and records raw_exit. | G4_a5.json: N2b/N2c unset_grace and D3; Raw_SIGKILL_normalisation (-9 ->137); G4_p116.json: both actual child PIDs absent immediately after return; D3_ARITHMETIC.md. |
| R4-F03 NIT | D4 timeout_cause is total, maps non-start124/137 to outer-timeout and unknown attribution to unattributed; all parent evaluation paths are protected and prelaunch failures are JSON refusals. | G4_a5.json: N5; inherited G4_a4.json: D3_evaluate_never_raises, E5 and start-evidence controls. |
| R4-F04 NIT | D5 writes/fsyncs a temporary receipt, then os.link publishes without overwrite; corrupt prior receipts return a JSON error. The temporary is retained. | G4_a5.json: N4 and Atomic_publication (complete matching content, shared inode, link count2). |

## Tooling diff summary

The copy contains36 inherited Python/shell sources and four support sources: a5_controls.py, control_transport.py, cpu_driver_a5.py, rev4_forge.py. The raw source diff has32 hunks in13 files. tooling_hunks.json maps every hunk; tooling_functional.diff removes the mechanical namespace/order/scratch relocation before showing behavior changes.

Functional changes are D2 exact argv equality and CPU-control scaffolding to present that grammar, D3 exit normalization/grace staging, D4 total attribution/protected evaluation, and D5 atomic no-overwrite publication/prior-read refusals. seal_staging.py and verify_scope.py have mechanical A5 relocations only. The consolidation driver used for G1–G6 is retained at /tmp/gn1_a5_seat/final_verify.py and named in its launch receipts. Existing inherited target predicates remain intact except the former attribution-exception target now expects D4's required outer-timeout result. No numerical or mutation behavior changed.

Synthetic fixture runner executables generate deliberately malformed raw operands that the installed runner cannot be instructed to produce. Installed-runner cases execute its actual source, with summary/window/lock redirected to scratch. Their private PATH dispatcher routes only the registered python3 run_gpu_tests.py child into a CPU fixture; the installed runner really sees the exact registered inner argv. Margin overrides in timing controls are recorded separately; min/max and inner bounds remain visible. Lead staging uses the installed runner directly. Public supplementary transport logs name it generically; parent command operands and sealed inputs remain byte-for-byte intact.

## G4 table

| Batch / class | Targets | Result | Receipt |
|---|---:|---|---|
| Inherited hostile payloads |52| PASS | G4_hostile.json |
| Installed runner outcomes/interruption |13| PASS | G4_runner.json |
| REV2 payload replay |38| PASS | hostile_rev2_acd.json |
| Supplementary seal/cause controls |4| PASS | G4_extra.json |
| A4/REV3 reproductions and obligations |24| PASS | G4_a4.json |
| REV4 five bypasses |5| PASS | G4_a5.json |
| N2b/N2c, grace unset |2| PASS | G4_a5.json |
| N2b/N2c, D3 staging |2| PASS | G4_a5.json |
| Raw SIGKILL normalization |1| PASS | G4_a5.json |
| N5, N4, atomic publication |3| PASS | G4_a5.json |
| Runner accepted option-name spellings |45| PASS | OPTION_CENSUS.json; G4_a5.json |
| GNU timeout accepted option-name spellings |76| PASS | OPTION_CENSUS.json; G4_a5.json |
| Additional argv/value grammar classes |16| PASS | G4_a5.json |
| Actual-PID observations for both D3 children |2| PASS | G4_p116.json |
| Total |283| PASS | G4_summary.json |

G4_TABLE.md lists all283 individual targets, each deciding rule and raw JSON index. The option census derives the runner's option set from its live parser and timeout's set from live help/probes. It covers every unambiguous accepted long-option-name prefix with separated/equals argument forms, short attached/separated forms and representative short clusters. Numeric values, option repetition and arbitrary short clusters have infinite spellings; their classes are sampled, while exact argv equality rejects all noncanonical lists by construction. The canonical staged spelling is positively tested by both full honest roots and the audit line. No special case for REV4's five strings is used.

## D3 arithmetic

Source inspection / reasoning: the installed runner's line115 defines default grace10; line132 defines default poll5; lines205–217 send TERM, wait the full grace and send process-group KILL; line204's child check sleeps at most0.02s. The staged script explicitly sets grace10 and poll5. Thus10 +5 =15 <20s outer kill-after; adding the child check conservatively gives15.02 <20. Script timeout is the registered outer bound +40s, which leaves20s after the parent's outer KILL delay for receipt publication and shell margin. D3_ARITHMETIC.md includes the numbered live source excerpt and its SHA-256. This is a configured ordering argument, not a hard bound on arbitrary scheduler/filesystem stalls.

Both N2b and N2c under D3 return exit124 with outer-timeout. G4_p116.json observes each actual child PID from child_started.json immediately after return: both absent. The original obedient-child helper lacked a separate PID file; the additional two controls close that observation gap. The deliberately invalid grace25 control produces raw_exit=-9, exit=137, cause=outer-timeout, with no surviving child. The grace-unset controls also attribute correctly under A5's raised outer kill delay; their shortened outer bounds fail D2 as expected.

## Predictions

- **P115 HELD:** final delivery-seal replay meets all283 targets, including all131 inherited targets, REV4's reproductions and the enumerated spelling controls. The failed inspection launch below remains a failed launch; it is not counted as a passing control.
- **P116 HELD:** the finite D3 obedient and TERM-ignoring interruptions leave no live child; both actual PIDs are absent immediately after parent return. No general OS scheduling or resource-stall guarantee is claimed.

## Retained failures and procedural exception

- **FAIL — inspection harness, corrected and replayed:** pre_helper_fix/a5_controls_gate.log records `json.decoder.JSONDecodeError: Unterminated string starting at: line 242 column 5 (char22312)`. N4 had deliberately truncated a prior parent receipt. A5's parent already returned the required JSON refusal; the control helper then tried to parse that old receipt as a fresh one. The helper now records the read error. All283 final targets and both full honest roots use the delivery seal after correction. No target predicate or threshold was weakened.
- **FAIL — initial setup environment, corrected before implementation/gates:** the first three read-only house-rule/memory/pwd inspections used TMPDIR=/tmp before /tmp/gn1_a5_seat was created, contrary to the requested command environment. They ran no tests, created no temp files and changed no workspace bytes. Every subsequent implementation command and all CPU gates use /tmp/gn1_a5_seat. This is disclosed as a procedural exception; the technical gate receipts establish only G1–G6.
- Initial seals and superseded receipt batches remain under initial_seal/ and pre_helper_fix/. Earlier CPU receipts are retained in cpu_initial/. AMENDMENTS.md and LEDGER.md record each correction. Nothing was deleted.

## Prior art

- Python argparse allow_abbrev: Python Software Foundation, introduced in Python3.5 (2015). Taken: the live parser's prefix/equals behavior as hostile operands. A5 adds the order's exact-list acceptance rule. Historical attribution unverified — lead to check: argparse allow_abbrev Python3.5 history.
- GNU coreutils timeout(1), installed9.4: taken short/long option parsing, TERM/KILL delays and128+signal exits; existing runner supplies its TERM/grace/group-KILL sequence. A5 configures the registered ordering and normalizes Python's negative exits. Historical year unverified — lead to check: coreutils timeout kill-after history.
- POSIX link(2), IEEE Std1003.1: taken atomic name publication with EEXIST/no overwrite; A5 writes/fsyncs a complete temporary receipt before linking and retains the temporary. Historical year unverified — lead to check: POSIX link atomic no replace.
- SHA-256, NIST FIPS180-4 (2015): reused content manifests, pins and seals unchanged.
- Fault injection: Meszaros (2007); REV4's independently authored reproductions (2026) and earlier GN1 controls supply the payloads. Linux procfs supplies direct process-state observation; history unverified — lead to check: Linux procfs process stat history.
- Unchanged numerical/mutation prior art remains in the copied code: Higham (1996/2002), PEP357 (2006), DeMillo/Lipton/Sayward (1978), Offutt/Voas equivalence/reach model (year unverified — lead to check), pytest APIs (Krekel et al.,2004+, historical attribution unverified). No novel algorithm claimed. Code-site comments and LEDGER.md carry the same provenance; no external literature was fetched.

## Done

Every workspace file written is enumerated in files_written.txt and files_written.json. Every retained task scratch file is enumerated in scratch_files_written.txt and scratch_files_written.json. All writes are under artifacts/gn1/a5/ and /tmp/gn1_a5_seat; logs/ was unused. sha256_maps.json hashes every workspace deliverable except its own self-hash; scratch_sha256_maps.json hashes the retained scratch files. LEFT_FOR_LEAD_TO_BIN.md and left_for_lead_to_bin.json list superseded batches, previous seals, linked temporary parent receipts and the whole scratch tree. No files were deleted, no git writes occurred, no other worktree or product/test bytes changed, no module was rebuilt, no GPU lock/window was touched, no GPU executed, and no subagent was used.

The report was created 2026-11-01T19:09:37Z, registration was written19:10:41Z, and synthesis was prepared 2026-11-01T19:50:09.244924+00:00. This remains within the 2 h ceiling, including initial inspection; every gate command is explicitly foreground-bounded below 20 minutes. Long-running tool returns were resumed with the maximum 300000 ms wait and no intervening status messages.
