<!-- Filed by the lead from the seat's final message (the session hook refuses seat report files). -->

# BWD-R4b r8: confirm 2's five findings are fixed by class, every decision field is audited, G6 passes, and 8 lines plus the 4 ref-at-wave commands are restaged

All five findings (R7-F01 to F05) are closed as shared rules, which also closes the seven partly closed items. Each planted edit r4bm20 to r4bm29 passes under the r7 code and is refused under r8. The real bundles (ref-at-wave-r7, -r6 and -r4), the real r7 G5 trace and the real r7 G-API record with its JUnit report all still pass.

- **Field audit:** `FIELD_AUDIT_R4B_R8.json` has 87 rows over 21 receipts. Each of the 77 decision fields names its writer, its check and the line that checks it; 10 fields are marked informational. Each of the 30 new checks names a planted edit that passed.
- **Not touched:** no GPU, no git command of my own, no deletes, no Voice writes, no test, product, native or kernel edits, no bar change.
- **Report:** the hook refused the report file in every earlier round and the session bars report files, so I did not create it. This message is the report.
- **Ledger:** the first new line is "## r8 start 2026-11-19T22:56Z". The round ended at 23:46Z.

## Diff summary

**Edited:**
- `scripts/bwd_r4b_verify.py`:
  - one shared comparison, `same`/`join`/`canonical`, that every join now calls;
  - the strict parser refuses `NaN` and `Infinity`;
  - `close()` requires a finite float;
  - new `verify_root_r8` (one attempt per root);
  - two checks the audit found: manifest `probe_rows` must equal 40, and the record's `wave_c.path` must equal the live capture's path.
- `scripts/bwd_r4b_stage_r7.py` and `scripts/bwd_r4b_refwave_r7.py`: both refuse superseded use and name the r8 script.

**New:**

| File | Purpose |
|---|---|
| `bwd_r4b_refwave_r8.py` | Start and end markers per mode; RETRY markers; checks that earlier modes passed; red verdict on any refusal; producer verdict joined with `same` |
| `bwd_r4b_gpu_r8.py` | Sampler header values; explicit JUnit totals; `verify_launch` provenance; the JUnit copy kept in the worker output |
| `bwd_r4b_stage_r8.py` | Launch identity, TMPDIR and result digests in every receipt; `--ref-root`; `verify-receipt` for any line |
| `bwd_r4b_suite_r8.py` | G6 RUN and worker LAUNCH records; argv joins |
| `bwd_r4b_selftest_r8.py` | The CPU self-test, in three parts |
| `bwd_r4b_audit_r8.py` | Builds the field audit |

## Findings map

`FINDINGS_MAP_R4B_R8.json`, sha256 `ed68bb33…`.

| Item | Class fix | Planted edits |
|---|---|---|
| R7-F01 (closes R6-F01, R5-F02) | Exact typed joins everywhere: bool, int and float are distinct; non-finite values refused | r4bm20 (L02 smoke), r4bm29, `same_rule_unit` |
| R7-F02 (closes R6-F02, R5-F08) | One attempt per root. Preflight and worker accept only the exact entries of one complete passing attempt. Retries use `ref-at-wave-r8-aN`. | r4bm21 (two checks) |
| R7-F03 (closes R6-F03, R5-F05) | Sampler header `pid`, `clock` (`CLOCK_MONOTONIC`), `period` (finite, positive, equal to the worker's) and `mode` joined to the worker's values; `errors` must be a list of strings | r4bm22, r4bm23 |
| R7-F04 (closes the R6-F05 totals part) | Exactly one suite named `pytest`; all four totals present as non-negative integers; every case inside that suite | r4bm24 |
| R7-F05 (closes the R6-F05 baseline part) | Per-launch provenance for every line: current seal, exit 0, label, result digests, launch identity, and the JUnit source under the launch's own TMPDIR | r4bm25 (G-API), r4bm26 (G6) |

## Field audit

`FIELD_AUDIT_R4B_R8.json`, sha256 `eed07906…`.

- **Receipts covered (21):**
  - merge verdict, manifest, FD summary, reference record, operand record, FD parts, oracle rows;
  - attempt markers, and the retry and red markers;
  - sampler record, JUnit XML, G-API worker JSON, launcher receipt, G6 records, census;
  - wave-capture receipt and worker, the inventory, round-1 sealed files, staged pins, module records, and TOLERANCES.
- **Writers:** each row names the file and line that writes the field, located by a pattern in the current source so a moved line cannot be cited.
  - Our own writers include `bwd_r4b_refwave.py:230` (verdict), `bwd_r4b_nvml_sampler_r6.py:118` (sampler), `bwd_r4b_stage_r8.py` (receipt) and `bwd_r4b_suite.py:177` (G6 arm).
  - The JUnit fields are written by pytest's junitxml plugin, at the `--junitxml` path set at `bwd_r4b_gpu_r7.py:291`.
  - Where nothing of ours writes a file in r8 (round-1 files, TOLERANCES), the row says so.
- **Informational fields (10):** for example `driver`, `relative_l2`, the A1-2 and A2-2 report-only fields, and the JUnit `time`.
- **Built-in cross-check:** the generator refuses to write unless every new check names a planted edit that passed in the final self-test receipts.

## Planted-edit table (CPU)

| Edit | Variants | r7 | r8 |
|---|---|---|---|
| r4bm20 (L02 smoke) | summary `probes` 40.0; `clean_passes` 40.0; a nested `rows` value as float | 3/3 accepted (preflight and worker) | 3/3 refused |
| r4bm21 red retry on a reused root | merge retry, then reference retry, on a passing root | r7 preflight still accepted the root | both refused; 2 RETRY markers; the root is refused |
| r4bm21 root rule | end marker missing; RED marker; red sidecar in `merge/`; failed fd attempt; launch mismatch; `exit` 0.0 | no root rule existed | 6/6 refused; merge after a failed fd attempt leaves a red verdict and no manifest |
| r4bm22 | `CLOCK_REALTIME` | valid | refused |
| r4bm23 | period −0.002; period 0.004; period as a string; `fake` mode on the GPU line; non-string `errors` | valid on the first three | 5/5 refused |
| r4bm24 | `failures` and `errors` absent; `tests` absent; `skipped` 1.0; suite renamed; second suite | accepted the absent totals, the renamed suite and the second suite | 5/5 refused; the end-to-end line goes red |
| r4bm25 G-API | foreign worker result (the confirm's case); wrong seal; exit 1; another label; another TMPDIR; edited JUnit copy | r7 `newest_g_api` accepted the foreign result | 6/6 refused |
| r4bm26 G6 | foreign arm result; launch mismatch; result changed after it was recorded | the r7 compare's acceptance expression accepted the foreign result | 3/3 refused |
| r4bm27 | manifest `probe_rows` 41 (sealed pin withheld) | accepted | refused |
| r4bm28 | record `wave_c.path` changed (sealed pin withheld) | accepted | refused |
| r4bm29 | stored derived value as an int or a bool | accepted | refused |

## Self-test

All three parts passed on the final bytes. Receipts are under `artifacts/bwd_r4b/`.

| Part | Checks | Receipt | sha256 |
|---|---:|---|---|
| verify | 8/8 | `selftest-r8-verify-final2_R4B_R8/SELFTEST_VERIFY_R4B_R8.json` | `a1ca1d07…` |
| producer | 3/3 | `selftest-r8-producer-final2_R4B_R8/SELFTEST_PRODUCER_R4B_R8.json` | `da8bd974…` |
| harness | 13/13 | `selftest-r8-harness-final2_R4B_R8/SELFTEST_HARNESS_R4B_R8.json` | `d1d5adf6…` |

- **Producer positive case:** the r8 wrapper's merge on a marker-built root reproduced the sealed manifest `634bf770…` in 41.7 s with a 2.76 GB peak, and the consumer accepted it.
- **Harness:** the CPU pytest path ran each package with complete censuses and explicit totals:

| Package | Passed | Skipped | Failed |
|---|---:|---:|---:|
| candidate | 37 | 66 | 0 |
| r4bm01 | 37 | 66 | 0 |
| r4bm05 | 28 | 66 | 9 (expected) |

- **Trackability grep:** no match.
- **R4a census:** identical at the end.
- **Voice:** 0 changed, 0 writes; `tail_vjp.py` is still `aed3e3b9…`.

## G6

Receipt `G6_PER_ID_R4B_R8.json`, sha256 `9c64b8eb…`. Provenance found 0 problems on all three arms.

| Arm | Passed | Skipped | Failed | Errors | Collection errors |
|---|---:|---:|---:|---:|---:|
| R4a `cd7d7d4d` | 1347 | 626 | 2171 | 15 | 1 |
| production `3138cf54` | 1222 | 339 | 2171 | 15 | 1 |
| candidate | 1383 | 693 | 2171 | 15 | 1 |

- **Against R4a:** PASS. Only the 103 `test_bwd_r4b_*` IDs change (36 passed, 67 skipped).
- **Against production:** PASS. R4a's 412 changes plus the same 103. 0 unexpected IDs on either side.

## Pin comparison

From `PIN_COMPARISON_R4B_R8.json`. The new pins are `STAGED_PINS_R4B_R8.json`: 66 pins, sha256 `992b1d55…`, sealed.

- **r7 pins changed:** `bwd_r4b_refwave_r7.py`, `bwd_r4b_stage_r7.py`, `bwd_r4b_verify.py`.
- **r4 to r6 pins changed:** the same files as in r7, plus `bwd_r4b_verify.py` in the r5 and r6 sets.
- **Unaffected:** none of the files the 10 unaffected r4 lines execute changed, and `packages_r4` is unchanged.

## Restaged lines

All 8 dry runs gave exit 78 with `"would_start": true`, no lease and no worker, and each recorded its launch identity. L79 holds: every `--min` is at most 510 under `--max 540`. Host peaks are unchanged from r7.

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

- **ref-at-wave:** four commands through `bwd_r4b_refwave_r8.py`, root `ref-at-wave-r8`, 6.0e9 peak each. Expected seconds: reference 30, each fd part 540, merge 90.
- **If any ref-at-wave command fails:** rerun all four on `ref-at-wave-r8-a2`, then pass `--ref-root ref-at-wave-r8-a2` to `g1-mfcc` and `r4bm03`.
- **Checking a receipt:** `bwd_r4b_stage_r8.py verify-receipt <receipt>` checks any line's receipt.
- **Run order:**
  1. The four ref-at-wave commands.
  2. `g1-mfcc` and `r4bm03`.
  3. `g-api` before `r4bm01` and `r4bm05`.

## Deviations

1. **Repair 1 of 3:** prepare failed because `gpu_r8` did not expose `collect_census`. It stopped before writing anything (`logs/bwd_r4b_r8_stage_prepare_failed.log`). I added one alias line, then reran all three self-test parts on the final bytes.
2. **Field audit renamed:** the audit built from the pre-repair receipts was renamed to `FIELD_AUDIT_PRE_REPAIR1_R4B_R8.json` (a rename, not a delete), and the audit was regenerated.
3. **Earlier self-test runs:** the first self-test runs (with no suffix, and with `-final`) passed, but their bytes are superseded. Separately, adding r4bm29 for the `close()` change forced one rerun.
4. **Memory rail:** one producer start stopped at the rail before running (`*_memrail.log`).
5. **G6 arms** ran detached, and I waited for each in the foreground with `tail --pid` (7 to 8 minutes per arm).
6. **Self-test worker directory names** use local time but end in "Z". This is cosmetic; the names are self-test only.

## Prior art

- **Typed JSON comparison:** RFC 8259 (Bray, 2017) for the value kinds, contrasted with RFC 8785 canonical JSON (Rundgren, Jordan and Erdtman, 2020), whose number serialization would merge 40 and 40.0.
- **Launch provenance:** in-toto (Torres-Arias et al., 2019) and SLSA provenance, taken in their simplest form, with no signatures.
- **Write-once output roots:** Nix (Dolstra, 2006) and per-attempt workflow output directories.
- **Check-then-commit:** Gray (1978); ARIES (Mohan et al., 1992).
- **Clock and report formats:** POSIX `CLOCK_MONOTONIC`; pytest's junitxml (Ant/JUnit XML format).
- **Audit form:** related to data lineage, W3C PROV (2013).
- **Planted edits:** mutation testing (DeMillo, Lipton and Sayward, 1978); test doubles (Meszaros, 2007).
- **Unverified, lead to check:** each name above.

## Files written in r8

All paths are under `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_bwd/WorkTree/tensor_fork_bwd_wt_r4b/`.

**Code:** the edited and new files in the diff summary.

**Receipts in `artifacts/bwd_r4b/`:**
- Findings and audit: `FINDINGS_MAP_R4B_R8.json`, `FIELD_AUDIT_R4B_R8.json`, `FIELD_AUDIT_PRE_REPAIR1_R4B_R8.json`.
- Staging: `G_API_CENSUS_R4B_R8.json`, `STAGED_PINS_R4B_R8.{json,sha256}`, `STAGED_LINES_R4B_R8.md`, `PIN_COMPARISON_R4B_R8.json`, eight `dry_*_R4B_R8.json`.
- G6: `G6_{r4a,prod,candidate}_{R4B_R8.json,R4B_R8.xml,RUN_R4B_R8.json,LAUNCH_R4B_R8.json}`, `G6_PER_ID_R4B_R8.json`.
- Census and ledger: `CENSUS_END_R4B_R8.sha256`, `LEDGER_R4B.md`.
- Self-tests: `selftest-r8-{verify,producer,harness}{,-final,-final2}_R4B_R8/`.
- Producer copies: `selftest-r8-producer{,-final,-final2}-{root,fdfail}_R4B_R8/`. The three root copies are 413 MB each.
- Self-test worker outputs: `worker_selftest-*_R4B_R8/`.

**Logs (`logs/`):**
- `bwd_r4b_r8_selftest_*.log`
- `bwd_r4b_r8_field_audit{,_pre_repair1}.log`
- `bwd_r4b_r8_stage_prepare{,_failed}.log`
- `bwd_r4b_r8_dry_runs.log`, eight `bwd_r4b_r8_dry_*_R4B_R8.log`
- `bwd_r4b_r8_g6_*.log`
- `bwd_r4b_r8_trackability.log`

**Scratch left in /tmp (nothing deleted):** 31 `/tmp/bwd_r4b_r8_*` directories:
- `/tmp/bwd_r4b_r8_before`, `_dev`, `_census_*`, `_g6`;
- the self-test scratch directories;
- the self-test lease directories;
- eight dry-run lease directories.
