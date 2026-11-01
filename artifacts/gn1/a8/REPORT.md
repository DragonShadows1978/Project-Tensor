# GN1 A8 report

ACCEPT: G1 to G6 PASS under contract C. P125 HELD. P126 HELD.

These are CPU controls, real Python/native CPU checks, and one data-only positive device-record unit check. Device end-to-end acceptance remains the lead's run after a clean GN1-REV6. No GPU execution or performance result is claimed.

| Gate | Verdict | Receipt |
|---|---|---|
| G1 | PASS | [G1_scope.json](G1_scope.json) |
| G2 | PASS | [G2_summary.json](G2_summary.json) |
| G3 | PASS | [G3_pin.json](G3_pin.json) |
| G4 | PASS | [G4_summary.json](G4_summary.json) |
| G5 | PASS | [G5_mutants.json](G5_mutants.json) |
| G6 | PASS | [G6_staging.json](G6_staging.json) |

The order is unchanged at commit `6eb2f564cc8a774f1baaa6625f79a6b5b92bf47d`; branch `gn1`. The delivery seal is `74ab5d82e7a289a324099141c72d841287a9a397b8ecf7dbf4847c449b6e8cce`. Elapsed to final synthesis: 2453.2 seconds of the 5,400-second ceiling. Every foreground command was bounded below 20 minutes. CUDA_VISIBLE_DEVICES was empty on CPU children; the positive unit record was data only. No git writes, rebuild, subagents, deletes, real ownership-file access, or production execution.

## G1 scope and tooling diff

Unit/instrument evidence: [G1_scope.json](G1_scope.json), [tooling_hunks.json](tooling_hunks.json), [tooling_a6_to_a8.diff](tooling_a6_to_a8.diff), [tooling_functional.diff](tooling_functional.diff).

The product and test match cf922891 byte for byte. Baseline git blobs are `52e17bd25bcfb7163bdd292f9235a2e3579eb3b9` and `14815dcc3e79ffd149a34421f7b9fa6653858804`. All 44,761 earlier file hashes match before/after; 134 earlier symlinks have unchanged targets with lstat ctime/mtime predating registration ([earlier_symlinks_verified.json](earlier_symlinks_verified.json)). Existing untracked arms/build files were preserved, not treated as seat changes.

The active functional comparison has **9 files / 17 hunks**, mapped only to D2/D3:

| Design | Sites | Change |
|---|---|---|
| D2 | device_evidence.py; run_gpu_tests.py | Unchanged A7 query/schema; real sealed child owns capture before tests or CPU fixture. |
| D2 | hostile_forge.py; rev2_forge.py; rev4_forge.py | Fixtures write their own evidence; sealed child supplies its own capture to its fixture; fix A7's misplaced insertion. |
| D2 | STAGED_GPU_COMMANDS.sh; a6_controls.py | Require a nonempty device selection; expect the CPU root's device-evidence refusal. |
| D2/D3 | evidence.py; score_gpu.py | Child-derived mode and identity; invocation/time consistency; exact parent declarations; root declarations only demote. |

D1 namespace/order/scratch substitutions are retained in the raw peer diff and removed, without normalizing expressions, from the functional diff. The unchanged class-(iii) `extra_controls.py` is verified by relocated-byte equality and was not executed. The copied A7 `a7_controls.py` is an opaque historical D2/D3 control source, not executed or semantically inspected; its source/copy digests are in the hunk receipt. It is explicitly outside the active functional diff, not silently discarded. Verification sources live in [verification/](verification/); no new interruption or post-run receipt-editing payload was introduced.

## G2 and G5 CPU evidence

[G2_summary.json](G2_summary.json), [cpu_driver_results.json](cpu_driver_results.json), [G2_driver.json](G2_driver.json), [G2_shim.json](G2_shim.json): all A6 CPU checks and exact ID manifests pass. NumPy is 2.4.2 and pytest is 9.0.2. Installed runner and pinned module are present; no module build ran.

[G5_mutants.json](G5_mutants.json): all 54 unchanged mutants pass their registered CPU checks: 51 CPU kills and 3 registered equivalents. Edit-site reach and real witness IDs are retained. CPU results do not establish device mutant kills.

## G3 module

[G3_pin.json](G3_pin.json), [engine_sources_verified.json](engine_sources_verified.json): pin JSON is unchanged, all 72 source hashes match, and all 56 arm module copies equal `c9b7dc610421010a8b11e8e3af74d1a9ad99928dabb98835aa6c265754daf8ce`. Regenerated mutant inputs equal A6. Zero builds.

## Device evidence and mode census

The source is A7's unchanged [device_evidence.py](../a7/device_evidence.py). API declarations and header hashes are in [DEVICE_API_SOURCES_COMPLETE.json](DEVICE_API_SOURCES_COMPLETE.json), supplementing A7's list with cudaGetDevice at cuda_runtime_api.h:2303. cudaGetDeviceProperties maps to the v2 ABI at header line172; its declaration is line1816; cudaDriverGetVersion is line9958; cudaRuntimeGetVersion is line9987; the device-name field is driver_types.h:2284-2286.

| Field | Requirement | A7 source lines |
|---|---|---|
| program.resolved_path / program.sha256 | resolved sealed run_gpu_tests.py and exact SHA-256 | a7/device_evidence.py:24-26,37; validation:92-93 |
| native_status.cudaGetDevice / device_ordinal | status "0"; nonnegative exact decimal ordinal string | a7/device_evidence.py:48-56,62-63; validation:97-100 |
| device_name / native_status.cudaGetDeviceProperties_v2 | nonempty stripped name; status "0" | a7/device_evidence.py:65-72; validation:96,100 |
| cuda_runtime_version / cuda_driver_version | positive exact decimal strings; query statuses "0" | a7/device_evidence.py:48-61; validation:97-100 |
| CUDA_VISIBLE_DEVICES | nonempty exact string equal to child environment copy | a7/device_evidence.py:37; validation:94-95 |
| engine_sha256 | exact pinned engine digest | a7/device_evidence.py:46-47; validation:101 |
| provider / query_error | "pinned-engine:libcudart" / null | a7/device_evidence.py:40,73-74; validation:102 |
| captured_at | ISO timestamp; invocation and result equal; within parent child interval | a7/device_evidence.py:41,103; a8/evidence.py:445-449 |

[DEVICE_FIELDS.json](DEVICE_FIELDS.json) and [MODE_CENSUS.json](MODE_CENSUS.json) enumerate every device field, START audit/control, parent audit/control, child invocation/result audit, derived line mode, and controlling environment inputs. Status/version/ordinal fields use A7's exact decimal strings; receipt modes use exact booleans. Programs bind by resolved path and SHA-256; argv binds by exact lists.

[DEVICE_HIDDEN_PROBE.json](DEVICE_HIDDEN_PROBE.json) records runtime `12060`, driver `13020`, cudaGetDevice status `100`, null device name/ordinal, and empty CUDA_VISIBLE_DEVICES. This is a hidden-device query, not device execution. Normal child capture is [run_gpu_tests.py:125-146](run_gpu_tests.py); CPU fixture ownership is lines81-89. The shared D2/D3 line check is [evidence.py:459-481](evidence.py); root demotion is [score_gpu.py:18-38](score_gpu.py).

## D4 classification table

Full per-target table: [D4_classification.md](D4_classification.md); machine-readable targets, names, rules and bases: [D4_classification.json](D4_classification.json).

| Census | Class (i) run condition | Class (ii) abnormal artifact | Class (iii) history |
|---|---:|---:|---:|
| A6 JSON targets | 277 | 13 | 32 |
| A7 registration operand rows | 1 | 0 | 269 |

A7 produced **no added-control outcome JSON** before its provider stop. Its registration-only families are separately labelled and expanded to 270 operand rows; these are not invented prior observations or a claim to reconstruct unseen script control names. The A7 actual sealed-child substitution is covered by the new live substitution control. The 269 edited/constructed historical operands are not run. Current mode/device operands are separately enumerated by the field censuses and direct unit cases.

The 32 A6 historical names, not run: `Stored_parent_tamper`, `Expected_exit_tamper`, `START_seal_tamper`, `Stored_timeout_cause_tamper`, `Import_helper_pin_parent`, `Import_helper_pin_child`, `Import_helper_pin_scorer`, `D7_outer`, `D7_maximum`, `D7_minimum`, `D7_inner`, `D7_expected_exits`, `raw_exit_minus9_exit0`, `raw_exit_removed`, `raw_exit_bool`, `raw_exit_string`, `Recorded_pin_runner`, `Recorded_pin_timeout`, `Recorded_pin_interpreter`, `Recorded_pin_runner_interpreter`, `Recorded_pin_parent_interpreter`, `Recorded_pin_outer_timeout`, `Recorded_pin_shebang_env`, `Recorded_env_PATH`, `Recorded_env_FORGE_GPU_SUMMARY`, `Recorded_env_FORGE_GPU_WINDOW_FILE`, `Recorded_env_FORGE_GPU_LOCK`, `Recorded_env_FORGE_GPU_KILL_GRACE`, `Recorded_env_FORGE_GPU_POLL`, `Recorded_env_FORGE_GPU_RUN_LOG`, `Recorded_env_FORGE_GPU_UNKNOWN`, `D0_control_receipt_promotion`.

Classification was made from JSON names, targets and deciding rules before execution. Before G4, the import-seal rows were refined to class (iii), while runtime/equivalence operands were class (i); the initial table is retained as D4_classification.initial.json. Mixed inherited drivers are selected by AST statement spans in [control_selection.json](control_selection.json), with all historical bodies excluded. No old class-(iii) target was replayed.

## G4 table

Full per-control targets, deciding rules and receipt/index links: [G4_table.md](G4_table.md). Aggregate: [G4_summary.json](G4_summary.json).

| Receipt | Targets | Result |
|---|---:|---|
| [G4_hostile.json](G4_hostile.json) | 49 | PASS |
| [G4_runner.json](G4_runner.json) | 13 | PASS |
| [hostile_rev2_acd.json](hostile_rev2_acd.json) | 38 | PASS |
| [G4_a4.json](G4_a4.json) | 19 | PASS |
| [G4_a5.json](G4_a5.json) | 150 | PASS |
| [G4_a6.json](G4_a6.json) | 19 | PASS |
| [G4_p116.json](G4_p116.json) | 2 | PASS |
| [G4_live_controls.json](G4_live_controls.json) | 4 | PASS |
| [G4_device_units.json](G4_device_units.json) | 25 | PASS (24 refusal; 1 accept as unit data) |
| [G4_root_refusals.json](G4_root_refusals.json) | 270 roots | All REFUSED |

All 290 inherited class-(i)/(ii) targets meet their local targets, with their original deciding rules retained. Positive CPU sensitivity targets still cannot establish root acceptance; the 270 resulting roots were rescored with the delivered scorer and all refused. The new live checks cover staged CUDA hidden, actual sealed-child substitution (seal mismatch), an actual wrong child writer (identity mismatch), and a GPU-mode parent with real no-device child evidence (mode binding). [D4_live_controls_registration.json](D4_live_controls_registration.json) names those rules before execution.

The positive unit directly calls `check_line_device` with A7-probe-shaped data, successful native statuses, a name, nonempty device selection and the sealed child identity. It executes no programs from the record. Missing fields, exact-type errors, failed statuses, wrong child identities, empty device selection and parent demotions are refused as data-only unit cases. Device end-to-end acceptance remains the lead gate.

## G6 staging and predictions

[STAGED_GPU_COMMANDS.sh](STAGED_GPU_COMMANDS.sh), [gpu_lines.json](gpu_lines.json), [G6_staging.json](G6_staging.json), [regeneration_verified.json](regeneration_verified.json), [G6_audit.json](G6_audit.json).

All 58 line expectations and actual CPU fixture argv match the stage. Regeneration verifies all arms against the delivery seal. The staged `--audit` command passes 29 real reference calls. The honest 58-line CPU fixture uses the installed pinned runner, sealed child and private ownership files; every child records its real identity and hidden-device evidence. Its root passes all 58 line checks and is refused with exactly:

`not a device run: every line requires positive child device evidence`

Receipt: [a6_runs/honest_all_labels/G5_SUMMARY.json](a6_runs/honest_all_labels/G5_SUMMARY.json). This is CPU fixture sensitivity, not execution of all native tests on a device.

- **P125 HELD:** 290 inherited run/abnormal targets plus the registered live and direct negative device/mode checks refuse by their named rules; 270 resulting control roots all refuse.
- **P126 HELD:** all 58 honest CPU line checks pass; the sole root error is lack of positive device evidence.

## Retained failures and limits

- Initial A5 replay: `IndentationError: unexpected indent`, rev4_forge.py line64. A7 inserted a D2 capture into the incomplete-child branch before `data` existed. A8 corrects capture placement; delivery replay passes. Initial seal and replay roots/logs remain under [pre_rev4_fix/](pre_rev4_fix/).
- Runner launcher retry: `FileExistsError: [Errno 17] File exists: '/tmp/gn1_a8_scratch/runner_controls'`. It executed no targets; fresh scratch fixes the launch. Receipt is pre_rev4_fix/runner_scratch_reuse/.
- Initial G1 instrument: `AssertionError` from treating all pre-existing untracked files as new writes; the next inventory check also needed older directory symlinks. [G1_preexisting_untracked.json](G1_preexisting_untracked.json), [final_verification.pre_symlink.log](final_verification.pre_symlink.log), and [verification/verify_final.initial.py](verification/verify_final.initial.py) retain the issue. Corrected verification uses frozen maps and pre-registration lstat times, without changing the scope bar. [final_verification.log](final_verification.log) records G1/G3/G6 PASS.

No device end-to-end acceptance, performance, protection against deliberate post-run well-formed receipt edits, or lead-side run-root manifest is claimed. Those remain the contract's stated lead work/out-of-scope boundaries. The active functional diff and opaque historical copies are explicitly distinguished for GN1-REV6.

## Prior art

NVIDIA CUDA Runtime API 12.6 (2024): existing device/version queries and cudaDeviceProp name field, checked against local headers. Python ctypes (PSF, Python 2.5, 2006) supplies ABI calls; runpy supplies init_globals for the existing CPU fixture. SHA-256: NIST FIPS180-4 (2015), inherited byte/program seals. Meszaros (2007), xUnit Test Patterns, supplies the inherited test-stub/fault-injection approach. Pytest report hooks and the A6 foreground timeout/receipt pattern are inherited. Python difflib uses its existing comparison machinery; Ratcliff/Metzener (1988) is an unverified attribution lead.

D2/D3's conjunction and demote-only rule are the binding lead design, not a new numerical algorithm. No prior art known to me for this exact order-specific conjunction beyond inherited GN1. Historical attributions not verified through a network are **unverified — lead to check**: search terms `CUDA Runtime API 12.6 device management version management`, `Python ctypes 2.5 history`, `Python runpy init_globals`, `Meszaros xUnit Test Patterns 2007`, `difflib Ratcliff Metzener 1988`. Code comments and ledger carry the corresponding attribution.

## Done

Every file written: [files_written.txt](files_written.txt) and [files_written.json](files_written.json), 7588 files under artifacts/gn1/a8; 80 scratch files in [scratch_files_written.json](scratch_files_written.json). [sha256_maps.json](sha256_maps.json) records final artifact hashes except its own bytes. No logs/ files were written.

Anything left for the lead to bin is explicitly expanded in [left_for_lead_to_bin.json](left_for_lead_to_bin.json): retained initial/old-seal replays, the initial verifier and failed inventory log, prior seal, saved sealed-child source and atomic-publication temporaries; scratch remains under /tmp/gn1_a8_scratch and /tmp/gn1_a8_delivery. All items stay in place. No deletes were performed. Retain failure/publication evidence through review.

Next lead step: clean GN1-REV6 under contract C, then the lead device run and lead-managed post-run SHA-256 manifest. This seat does not perform either step.
