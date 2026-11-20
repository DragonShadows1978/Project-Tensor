# BWD-R4b r6: all nine re-check 2 findings are fixed, G6 passes, and 8 lines plus the 4 ref-at-wave commands are restaged

Each of re-check 2's nine findings is fixed as a general rule under your threat model. Planted edits r4bm10 to r4bm14 are all refused on CPU, and the positive case on your real bundle still passes. Each planted refusal holds two ways: by the sealed manifest digest, and by the rule alone with that pin withheld.

- **G6:** passes against both baselines.
- **Restaged:** 8 lines and the four ref-at-wave commands, under `_R4B_R6` pins. All 8 dry runs gave exit 78, `"would_start": true`, no lease and no worker.
- **Unaffected:** 10 r4 lines keep their receipts.
- **Not touched:** no native, tensor_cuda or kernel file, no bar, no sealed file. No GPU use and nothing deleted.
- The hook refused `REPORT_R4B_R6.md` again, so this message is the report. The ledger starts at "r6 start"; the round ended at 20:15Z.

## Diff summary

**Edited:**
- `scripts/bwd_r4b_verify.py`:
  - F01: every crossing record must be complete; FD parts must carry the right identity and stitch into the merged oracle.
  - F02: sealed manifest digest, binding to the live capture, and the cotangent digest.
  - F03: dtype, shape and finiteness are checked before any subtraction, and bounds compare NaN-safely.
  - F04: duplicate-free JSON with explicit schemas; registered crop and the full operand record at preflight.
- `tc_signal/tc_signal/_conv_backward.py` (F06, Python only): alignment is checked from `data_ptr` before anything is allocated.
- `tc_signal/tests/test_bwd_r4b_public_conv_backward.py`: the F07 test corrected, and 3 host tests appended for F06.
- `scripts/bwd_r4b_stage.py` and `scripts/bwd_r4b_stage_r5.py` (F08): the old launchers refuse superseded labels.

**New:**
- `bwd_r4b_gpu_r6.py`: the restaged line bodies, plus F05 coverage and F09 timeout handling.
- `bwd_r4b_nvml_sampler_r6.py`: the sampler, now recording when each query runs.
- `bwd_r4b_build_r6.py`, `bwd_r4b_stage_r6.py`, `bwd_r4b_refwave_r6.py`, `bwd_r4b_suite_r6.py`, `bwd_r4b_selftest_r6.py`.

**Packages:** `packages_r6` is `packages_r4` with only `tc_signal/_conv_backward.py` replaced, and the two mutant packages re-derived from it. `MODULES_R4B_R6.json` sha256 is `b747ee95…`. The native modules are unchanged.

**The sealed pin:** the sealed inventory's `derived_from` manifest digest (`634bf770…`) is byte-identical to both your r4 and r5 manifests. The producer is deterministic for this capture, so a fresh ref-at-wave run must reproduce that digest.

## Findings map

`FINDINGS_MAP_R4B_R6.json`, sha256 `494a8698…`.

| Finding | Fix | Check (in `SELFTEST_R4B_R6.json` unless noted) |
|---|---|---|
| R5-F01 | Each evaluated step factor carries its crossing map, and the final halving's map equals T(1)'s. Each FD part's identity is checked, and the merged oracle must equal the stitch of the checked parts. | r4bm10, and r4bm11's unstitched case |
| R5-F02 | The manifest must equal the sealed digest. Parts and the reference record are bound to the live capture, wave and crop, and the cotangent digest to the live cotangent. The wave-capture receipt must have run under the r4 pins and packages, and its worker's operand record must match. | r4bm11 |
| R5-F03 | Every stored tensor and direction must be float64, the exact shape and finite before any subtraction; bound checks are NaN-safe. | r4bm12 |
| R5-F04 | Every JSON read rejects duplicate keys and follows an explicit schema. The launcher checks the crop values and ordered source bases, not just their count. | r4bm13 |
| R5-F05 | Coverage uses only samples where NVML listed this process; the 5 ms median rule is back; phase marks, the gap series and query timings are kept in the receipt; G5 is now called a sampled estimate. | r4bm14; sampler process check |
| R5-F06 | Element-0 address must be non-NULL and 4-byte aligned before allocation; that memory is really on the labelled device is checked by the native run entry, as now documented. | 3 appended host tests |
| R5-F07 | The mixed-device test now creates the operand with `tc.empty`, asserts its label, checks the device switch's status, and restores the prior device. | GPU g-api line; skips here with the registered device-count reason |
| R5-F08 | The r4 and r5 launchers refuse superseded labels and name the current launcher. | 4 of 4 refusals |
| R5-F09 | The after-census runs in a `finally` path, so a timeout keeps a red receipt with the child's output. | timeout receipt check |

**The six partly-fixed originals:**
- **F02, F03, F04, F08, F09, F10:** each closed by the fix of the same number above.
- **F10 also:** closed by F07's corrected test.
- **F03 also:** a deliberately and consistently forged oracle is out of scope under your ruling, except where the sealed digest binds it.

## Planted-edit table (CPU)

| Edit | Variants | Result |
|---|---|---|
| r4bm10 omitted crossings (the r6 smoke) | maps emptied; one factor dropped; final halving's map emptied | 3/3 refused |
| r4bm11 foreign part | part from another capture with no rows; merged oracle not equal to the stitched parts; record from another run (cotangent, wave and crop) | 3/3 refused, each by the sealed pin and by the rule alone |
| r4bm12 NaN direction | NaN direction; float32 reference tensor; NaN direction fed to the G2 check | 3/3 refused |
| r4bm13 duplicate key | duplicate manifest key; launcher crop value changed; strict read | 3/3 refused |
| r4bm14 missing-PID interval | a 130 ms stretch with no sample for this process; a 6 ms median gap | both refused; the r5 rule accepted the first |
| Positive cases | your real bundle (sealed digest and launcher preflight); both G2 probe sets | accepted |

## Self-test

`selftest-r6-verify-r3`, rc 0, log `logs/bwd_r4b_r6_selftest_r3.log`. Every check passed:

- the positive case, all 5 planted classes, and the r6 sampler process (median gap 2.0 ms, max 2.1 ms);
- F08 and F09;
- the pytest path on `packages_r6`:

| Package | Host tests passed | GPU tests skipped | Host failures |
|---|---:|---:|---:|
| candidate | 37 | 66 | 0 |
| r4bm01 | 37 | 66 | 0 |
| r4bm05 | 28 | 66 | 9 (expected) |

## G6

Receipt `G6_PER_ID_R4B_R6.json`. Each arm ran under RLIMIT_AS 12 GiB; the candidate arm imported from `packages_r6`.

| Arm | Passed | Skipped | Failed | Errors | Collection errors |
|---|---:|---:|---:|---:|---:|
| R4a `cd7d7d4d` | 1347 | 626 | 2171 | 15 | 1 |
| production `3138cf54` | 1222 | 339 | 2171 | 15 | 1 |
| candidate | 1383 | 693 | 2171 | 15 | 1 |

- **Against R4a:** PASS. Only the 103 `test_bwd_r4b_*` IDs change (36 passed, 67 skipped).
- **Against production:** PASS. R4a's 412 changes with identical outcomes, plus the same 103. 0 unexpected IDs on either side.

## Pin comparison

From `PIN_COMPARISON_R4B_R6.json`:

| Pin set | Files that changed |
|---|---|
| r4 (35 pins) | `bwd_r4b_stage.py` (F08); the test file (F07, F06) |
| r5 (43 pins) | those two, plus `bwd_r4b_stage_r5.py` and `bwd_r4b_verify.py` |

None of the 12 files the unaffected lines execute changed, and `packages_r4` is unchanged.

**Unaffected, keeping their receipts:** `g0-crop`, `g0-full`, `wave-capture`, `g3`, `g4-session-linear`, `g4-session-mfcc`, `g4-fresh-linear`, `g4-fresh-mfcc`, `r4bm02`, `r4bm04`.

## Restaged lines

`STAGED_LINES_R4B_R6.md`; `STAGED_PINS_R4B_R6.json` holds 52 pins, sha256 `f667476c…`. Host peaks come from your measured scope peaks, with headroom.

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

- **ref-at-wave:** four commands through `bwd_r4b_refwave_r6.py`, root `ref-at-wave-r6`, 6.0e9 peak each. Expected 30 s, 430 s, 430 s and 40 s.
- **Wrapper:** it refuses a wrong root after its pin check, and a stand-in run through it records the same sources as the sealed manifest.
- **Run order:**
  1. The four ref-at-wave commands.
  2. `g1-mfcc` and `r4bm03`.
  3. `g-api` before `r4bm01` and `r4bm05`.

## Deviations

- **Repair 1 of 3: cotangent digest.** It bound only at the producer's BLAS thread count. Measured: g_c is reproduced bit for bit at 3 threads in both Pythons, but not at 4. The reference is bitwise only in `/usr/bin/python3` at 3 threads. The consumer's recomputation now runs under `threadpool_limits(3)`. The reference keeps its 1e-12 × largest-tensor tolerance.
- **Repair 2 of 3:** a keyword clash in the self-test.
- **Packages split:** `packages_r6` is used only by the restaged lines. The unaffected lines' receipts were made on `packages_r4`, which differs from it only in `_conv_backward.py`, a module no chain line calls.
- **Old launchers can't relaunch:** the r4 and r5 launchers changed for F08, so their pin checks will no longer pass for a relaunch. Their earlier receipts stand as produced.
- **G6 timing:** the arms ran under host load 12–15 and took up to 7.7 min each, close to the 10-minute tool limit. All completed.

## Prior art

- **Content-addressed verification against an independent sealed pin:** Merkle (1979); Nix (Dolstra, 2006).
- **Duplicate-key-refusing JSON parsing:** standard practice (e.g. Python's `object_pairs_hook`); no specific source known to me.
- **Fixed-rate, process-scoped NVML sampling:** no specific source known to me.
- **Mutation testing:** DeMillo, Lipton and Sayward (1978).
- **Ours:** the stitch-equals-merged rule, the capture-record binding, the thread-pinned recomputation, and PID-valid coverage.
- **Unverified, lead to check:** "Merkle 1979", "Dolstra Nix 2006", "DeMillo Lipton Sayward 1978".

## Files written in r6

**Code:** the edited and new files in the diff summary.

**Receipts in `artifacts/bwd_r4b/`:**
- Build: `packages_r6/` (3 packages), `MODULES_R4B_R6.json`, two `*_R4B_R6.diff` files.
- Self-tests: `selftest-r6-verify/` (failed, before repair 1), `selftest-r6-verify-r2/` (failed, before repair 2), `selftest-r6-verify-r3/` (`SELFTEST_R4B_R6.json`, passing), `selftest-r6-refwave-wrapper/`.
- G6: `G6_{r4a,prod,candidate}_R4B_R6.{json,xml}`, `G6_*_RUN_R4B_R6.json`, `G6_PER_ID_R4B_R6.json`.
- Staging: `FINDINGS_MAP_R4B_R6.json`, `STAGED_PINS_R4B_R6.{json,sha256}`, `STAGED_LINES_R4B_R6.md`, `PIN_COMPARISON_R4B_R6.json`, eight `dry_*_R4B_R6.json`.
- Census and ledger: `CENSUS_END_R4B_R6.sha256`, `LEDGER_R4B.md`.

**Logs:** `logs/bwd_r4b_r6_{freeze, selftest, selftest_r2, selftest_r3, g6_run_*, g6_*, g6_compare, trackability, stage_prepare, dry_runs, wrapper_check}.log`, and eight `logs/bwd_r4b_r6_dry_*_R4B_R6.log`.

**Scratch left in /tmp (nothing deleted):** `/tmp/bwd_r4b_r6_g6/`, `/tmp/bwd_r4b_r6_probe/`, `/tmp/bwd_r4b_r6_selftest{1,2,3}/`, eight `/tmp/bwd_r4b_r6_lease_*`.

_Filed by the lead from the seat's final message; the seat's own report write was refused by a hook._
