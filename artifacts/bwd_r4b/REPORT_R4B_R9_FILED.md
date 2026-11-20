<!-- Filed by the lead from the seat's final message (the session hook refuses seat report files). -->

# BWD-R4b r9: both blocking findings and H01 to H03 are fixed, G6 passes, and 8 lines plus the 4 ref-at-wave commands are restaged

- **F01:** the generic `verify-receipt` command is gone, and no script references it.
- **F02:** the END marker is now written after the census. It records the census sha256, and root verification recomputes that digest.
- **H01 to H03:** one fix each, each with its planted edit.
- **Planted edits:** r4bm30 to r4bm35 each pass under r8 and are refused under r9.
- **Positive cases:** these all still pass:
  - the real bundles (ref-at-wave-r8, -r7, -r6, -r4);
  - the real r8 G5 trace (1530 of 1530 samples);
  - the lead's r8 G-API receipt, through the H03-checked `verify_launch`;
  - the lead's r8 G6 records.
- **Real ref-at-wave-r8 root:** all four censuses read strictly and clean, and the merge census covers every Voice source the manifest names. Its END markers predate the census digest, so the END-to-census binding cannot be checked on that root.
- **Not touched:** no test, product or kernel edits, no git command of my own, no deletes, no GPU.
- **Field audit:** retired as ruled; no r9 audit.
- **Ledger:** the first new line is "## r9 start 2026-11-20T00:23Z". The round ended at 01:05Z, inside the 2-hour wall.
- **Report:** as in earlier rounds, the report is this message rather than a file.

## Diff summary

**Edited:**
- `scripts/bwd_r4b_stage_r8.py`: the `verify_receipt` function, its CLI branch and its docstring lines are removed (F01); every r8 label is now refused and names the r9 launcher.
- `scripts/bwd_r4b_refwave_r8.py`: refuses superseded use and names r9.
- `scripts/bwd_r4b_gpu_r8.py`:
  - H01: `parse_junit` counts every `testsuite` element under either root shape.
  - H02: `child_error`, `junit_error` and `census_error` must be present before they are tested for null, including before the reparse join in `verify_launch`.
  - H03: `verify_launch` joins the declared result names to a listing of the launch's output directory, and requires the consumer's names among them.
  - New round constants `LEASE_PREFIX` and `ROOT_RULE`.
- `scripts/bwd_r4b_verify.py`: `census_clean`, `verify_root_r9` and `marker_names_r9`; `verify_root_r8` is kept, now through a shared `_verify_root`.

**New:**

| File | Purpose |
|---|---|
| `bwd_r4b_refwave_r9.py` | The F02 fix: START, the mode, the census, then END last; failures give a failing END plus a RED marker and exit 1 |
| `bwd_r4b_gpu_r9.py` | Sets the r9 round constants on the r8 line module |
| `bwd_r4b_stage_r9.py` | The r9 launcher, with no receipt command |
| `bwd_r4b_suite_r9.py` | G6 runner; adds the H03 listing of `G6_<label>_R4B_R9.*` |
| `bwd_r4b_selftest_r9.py` | The CPU self-test, in two parts |

## Findings map

`FINDINGS_MAP_R4B_R9.json`, sha256 `99743759…`.

| Item | Fix | Check |
|---|---|---|
| F01 | Command removed. Provenance is now required of every machine consumer of another launch's results: the mutants' clean baseline and G6 | No references; the CLI answers "invalid choice"; all eight lead r8 receipts hold JSON results without `launch_id` (the r8 cause) |
| F02 | END after the census, bound to its digest; `verify_root_r9` in the preflight and worker of `g1-mfcc` and `r4bm03` | r4bm30 to r4bm32, plus a positive real merge |
| H01 | Count every `testsuite` element | r4bm33 |
| H02 | Required keys present before the null tests | r4bm34 |
| H03 | Declared results joined to the directory listing (G-API and G6) | r4bm35 |

## Planted-edit table (CPU)

| Edit | Variants | r8 | r9 |
|---|---|---|---|
| r4bm30 (L02 smoke) | Census writer under a 128-byte file-size limit, via the real merge CLI path in a child process | END `passed: true`, then `OSError 27`; the 128-byte partial census was accepted by both the root rule and preflight | exit 1, END `passed: false`, one RED marker, root refused |
| r4bm31 | Census truncated to 128 bytes | accepted | refused (strict read) |
| r4bm32 | A clean census over one path fewer; a census recording a change | accepted | refused (digest; change) |
| r4bm33 (H01) | Nested suite under a `testsuite` root | accepted | refused (2 suites) |
| r4bm34 (H02) | Each of the three keys removed from the real record | 3/3 accepted | 3/3 refused |
| r4bm35 (H03) | A receipt omitting the launch's `.txt` file; an undeclared `G6_*.json.partial` | both accepted | both refused |

## Self-test

| Part | Checks | Receipt | sha256 |
|---|---:|---|---|
| producer | 9/9 | `selftest-r9-producer-r3_R4B_R9/SELFTEST_PRODUCER_R4B_R9.json` | `7927125a…` |
| harness, rerun on the final bytes | 11/11 | `selftest-r9-harness-final_R4B_R9/SELFTEST_HARNESS_R4B_R9.json` | `c57ec5eb…` |

- **Positive r9 merge:** the merge child passed in 55.9 s. The manifest equals the sealed `634bf770…`, END records the census digest, and the root and the consumer both accept it.
- **Pytest path on CPU:**

| Package | Passed | Skipped | Failed |
|---|---:|---:|---:|
| candidate | 37 | 66 | 0 |
| r4bm01 | 37 | 66 | 0 |
| r4bm05 | 28 | 66 | 9 (expected) |

- **Trackability grep:** no match.
- **R4a census:** identical at the end.
- **Voice:** 0 changed, 0 writes; `tail_vjp.py` is still `aed3e3b9…`.

## G6

Receipt `G6_PER_ID_R4B_R9.json`, sha256 `a6c6a1fb…`. Provenance, including the H03 listing check, found 0 problems on all three arms.

| Arm | Passed | Skipped | Failed | Errors | Collection errors |
|---|---:|---:|---:|---:|---:|
| R4a `cd7d7d4d` | 1347 | 626 | 2171 | 15 | 1 |
| production `3138cf54` | 1222 | 339 | 2171 | 15 | 1 |
| candidate | 1383 | 693 | 2171 | 15 | 1 |

- **Against R4a:** PASS. Only the 103 `test_bwd_r4b_*` IDs change (36 passed, 67 skipped).
- **Against production:** PASS. R4a's 412 changes plus the same 103. 0 unexpected IDs on either side.

## Pin comparison

The new pins are `STAGED_PINS_R4B_R9.json`: 72 pins, sha256 `ec97a1b6…`, sealed.

- **r8 pins changed:** `bwd_r4b_gpu_r8.py`, `bwd_r4b_refwave_r8.py`, `bwd_r4b_stage_r8.py`, `bwd_r4b_verify.py`.
- **r4 to r7 pins:** as in r8, plus `bwd_r4b_verify.py`.
- **Unaffected:** none of the files the 10 unaffected r4 lines execute changed, and `packages_r4` is unchanged.

## Restaged lines

All 8 dry runs gave exit 78 with `"would_start": true`, no lease and no worker. L79 holds: every `--min` is at most 510 under `--max 540`. Host peaks are unchanged.

| Line | Min/max s | Expected peak host bytes |
|---|---:|---:|
| g1-linear | 240/540 | 5.5e9 |
| g1-mfcc | 340/540 | 8.0e9 |
| r4bm03 | 280/540 | 7.0e9 |
| g5-checkpoint | 440/540 | 5.0e9 |
| g5-plain | 360/540 | 5.0e9 |
| g-api | 480/540 | 3.5e9 |
| r4bm01 | 480/540 | 3.5e9 |
| r4bm05 | 480/540 | 3.5e9 |

- **ref-at-wave:** four commands through `bwd_r4b_refwave_r9.py`, root `ref-at-wave-r9`. Expected seconds: reference 30, each fd part 540, merge 90; 6.0e9 peak each.
- **If any ref-at-wave command fails:** rerun all four on `ref-at-wave-r9-a2`, then pass `--ref-root ref-at-wave-r9-a2` to `g1-mfcc` and `r4bm03`.
- **Run order:**
  1. The four ref-at-wave commands.
  2. `g1-mfcc` and `r4bm03`.
  3. `g-api` before `r4bm01` and `r4bm05`.

## Deviations

1. **Repair 1 of 3:** the merge children inherited the self-test's hard address-space limit, so the producer's own `limit_host(10)` failed. The fix keeps the self-test's hard limit for the producer part. The first producer run left two staged roots that were not used.
2. **Repair 2 of 3:** r4bm32's first planted census was not actually different. The lead's four censuses are byte-identical (same reads, same file stats). The planted census is now a clean one over one fewer path.
3. **Similarity hook:** the hook refused the first write of `bwd_r4b_refwave_r9.py`; a retry of the same content succeeded. The self-test file was written with a heredoc.
4. **G6 arms** ran detached, and I waited for each in the foreground with `tail --pid`.

## Prior art

- **Commit record written last:** Gray (1978); ARIES (Mohan et al., 1992); "marker file last" publication.
- **Launch provenance:** in-toto (Torres-Arias et al., 2019), taken in its simplest form.
- **Fault injection:** POSIX `setrlimit(RLIMIT_FSIZE)`.
- **Planted edits:** mutation testing (DeMillo, Lipton and Sayward, 1978).
- **Unverified, lead to check:** each name above.

## Files written in r9

All paths are under `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_bwd/WorkTree/tensor_fork_bwd_wt_r4b/`.

**Code:** the edited and new files in the diff summary.

**Receipts in `artifacts/bwd_r4b/`:**
- Findings and staging: `FINDINGS_MAP_R4B_R9.json`, `G_API_CENSUS_R4B_R9.json`, `STAGED_PINS_R4B_R9.{json,sha256}`, `STAGED_LINES_R4B_R9.md`, `PIN_COMPARISON_R4B_R9.json`, eight `dry_*_R4B_R9.json`.
- G6: `G6_{r4a,prod,candidate}_{R4B_R9.json,R4B_R9.xml,RUN_R4B_R9.json,LAUNCH_R4B_R9.json}`, `G6_PER_ID_R4B_R9.json`.
- Census and ledger: `CENSUS_END_R4B_R9.sha256`, `LEDGER_R4B.md`.
- Self-tests: `selftest-r9-{harness,harness-final,producer,producer-r2,producer-r3}_R4B_R9/`.
- Merge roots: `st-r9-producer{,-r2,-r3}-{pos,fail,failr8}_R4B_R9/`, the bundle copies the merge children ran on.
- Self-test worker outputs: `worker_selftest-*_R4B_R9/`.

**Logs (`logs/`):**
- `bwd_r4b_r9_selftest_*.log`
- `bwd_r4b_r9_stage_prepare.log`
- `bwd_r4b_r9_dry_runs.log`, eight `bwd_r4b_r9_dry_*_R4B_R9.log`
- `bwd_r4b_r9_g6_*.log`
- `bwd_r4b_r9_trackability.log`

**Scratch left in /tmp (nothing deleted):**
- `/tmp/bwd_r4b_r9_before`, `_dev`, `_census_*`, `_g6`, `_selftest_*`;
- `/tmp/bwd_r4b_r9_lease_selftest_*`;
- eight `/tmp/bwd_r4b_r9_lease_*` (dry runs).
