# BWD-R4b r5: the three consistency gaps are closed, and only 3 lines plus the ref-at-wave commands need re-running

The G2 check and the reference-bundle verifier no longer trust any value carried by the data they check. Values come from the registration, digests are recomputed from the actual bytes, and the manifest must match a registered inventory exactly.

- **Planted edits:** r4bm07, r4bm08 and r4bm09 are each refused on CPU, every variant included. The positive case on your real r4 bundle still passes. All of the earlier F02 and F03 refusal cases still refuse.
- **G6:** passes against both baselines.
- **Restaged:** `g1-linear`, `g1-mfcc`, `r4bm03` and the four ref-at-wave commands, under `_R4B_R5` pins. All 3 dry runs gave exit 78, `"would_start": true`, no lease and no worker.
- **Unchanged:** all 35 r4 pins are byte-identical, so the other 15 r4 lines keep their r4 receipts.

The hook refused `REPORT_R4B_R5.md` again, so this message is the report. The round ran from about 18:08Z to 18:44Z.

## Diff summary

To keep every r4-pinned file byte-identical, all r5 code is in new files. The only tracked file I modified is `LEDGER_R4B.md`.

| New file | What it does |
|---|---|
| `scripts/bwd_r4b_verify.py` | The fixes themselves (the three rules below). |
| `scripts/bwd_r4b_gpu_r5.py` | The bodies of the three restaged lines: the unchanged r4 bodies, using the new verifiers. The bundle is read from `ref-at-wave-r5`. |
| `scripts/bwd_r4b_refwave_r5.py` | Runs the unchanged producer at root `ref-at-wave-r5`, after checking every r5 pin. |
| `scripts/bwd_r4b_stage_r5.py` | Launcher and pin comparison for the restage. |
| `scripts/bwd_r4b_selftest_r5.py` | The CPU checks for this round. |
| `scripts/bwd_r4b_suite_r5.py` | The unchanged G6 suite, writing `_R4B_R5` receipts. |
| `artifacts/bwd_r4b/REF_AT_WAVE_INVENTORY_R4B_R5.json` | The registered inventory (sha256 `54e2d584…`, pinned in `bwd_r4b_verify.py`). |

**What `bwd_r4b_verify.py` enforces:**

1. **Registered values only (`g2_registered`).** It takes ρ, the fp64 bound, the strength floor, the step triples and the crossing rule from the sealed registration, and recomputes them from the reference gradient. It refuses a row when:
   - its stored ρ, bound or projection differs from the registered or recomputed value;
   - its chosen triple, or T(1), records any crossing;
   - its stored plateau or error claim disagrees with what its own slopes give;
   - any recomputed value breaks a bound.

   Each direction's sha256 is also recomputed and compared with its registered pin.
2. **Recomputed digests (`verify_bundle_r5`).** Every file's sha256 is recomputed from its bytes, and so is every digest the bundle stores about itself: the reference archive, the direction archive, and each direction. Beyond that, the consumer recomputes the A1-1 reference and the A2-1 probes and directions itself from `wave_c`, and compares them with the bundle's.
3. **Exact inventory.** The registered inventory lists every entry with counts: 7 outputs, 2 parts, 7 scripts, 78 Voice files, 13 operand keys, 315 weights and 2 bases. A manifest that omits, adds or renames any entry is refused.

## Planted-edit results (CPU)

Self-test `selftest-r5-verify-r3`, rc 0, log `logs/bwd_r4b_r5_selftest_r3.log`.

| Edit or case | Variants | Result |
|---|---|---|
| Positive: your real r4 bundle, plus both G2 probe sets | — | **accepted**; the A1-1 recomputation agrees within tolerance |
| **r4bm07** oracle declares its own bounds (the r5 smoke) | reviewer's full edit; ρ alone; bound alone; declared crossing alone; plateau claims alone; direction scaled alone; slopes over the registered bound | **7/7 refused** |
| **r4bm08** bundle digest mismatch | stored digest disagrees with the bytes; fully restamped so every stored digest agrees (caught by the consumer's recomputation) | **2/2 refused** |
| **r4bm09** manifest inventory omitted | omitted entries plus a substituted reference; complete inventory with a re-sealed substitution (caught by recomputation, 2.3e-3 off); renamed part entry | **3/3 refused** |
| Earlier F02 cases | target, pairs, empty oracle, stale capture, live target change | 5/5 refused |
| Earlier F03 cases | empty, 39 rows, duplicate, missing direction, no slopes, over-bound claim | 6/6 refused |

**Other checks on the final bytes:**
- The ref-at-wave wrapper refuses a wrong root after its pin check, and runs reference mode on the stand-in (16.8 s, 2.38 GB). Both the r5 pin set (`2d6fbe43…`) and the r4 pin set (`021a8f23…`) verify (`logs/bwd_r4b_r5_wrapper_check.log`).
- The grep for home paths and the owner's name over `scripts/bwd_r4b_*.py` finds nothing (`logs/bwd_r4b_r5_trackability.log`).
- I did not rerun the r4 worker self-test: every file it covers is byte-identical to r4, so its 21/21 result still applies.

## G6

Receipt `G6_PER_ID_R4B_R5.json`; each arm ran under RLIMIT_AS 12 GiB.

| Arm | Passed | Skipped | Failed | Errors | Collection errors |
|---|---:|---:|---:|---:|---:|
| R4a `cd7d7d4d` | 1347 | 626 | 2171 | 15 | 1 |
| production `3138cf54` | 1222 | 339 | 2171 | 15 | 1 |
| candidate | 1380 | 693 | 2171 | 15 | 1 |

- **Against R4a:** PASS. Only the 100 `test_bwd_r4b_*` IDs change (33 passed, 67 skipped).
- **Against production:** PASS. The changes are R4a's 412 r5d changes, with identical outcomes, plus the same 100.
- 0 unexpected IDs on either side.

## Pin comparison

From `PIN_COMPARISON_R4B_R5.json`:
- **r4 pins:** 35 of 35 byte-identical, 0 changed.
- **New r5 files:** the six scripts and the inventory listed above, plus `STAGED_PINS_R4B_R4.json`, now pinned.
- **Unaffected r4 lines, keeping their r4 receipts:** `g-api`, `g0-crop`, `g0-full`, `g3`, `g4-fresh-linear`, `g4-fresh-mfcc`, `g4-session-linear`, `g4-session-mfcc`, `g5-checkpoint`, `g5-plain`, `r4bm01`, `r4bm02`, `r4bm04`, `r4bm05`, `wave-capture`.

## Restaged lines

`STAGED_LINES_R4B_R5.md`; `STAGED_PINS_R4B_R5.json` holds 43 pins, sha256 `2d6fbe43…`.

| Line | Min/max s | Peak host bytes | MemoryMax | Basis for the peak |
|---|---:|---:|---:|---|
| g1-linear | 240/540 | 5.5e9 | 6.5e9 | r4 measured 4.53 GB |
| g1-mfcc | 340/540 | 8.0e9 | 9.0e9 | r4 measured 5.26 GB, plus the consumer's own recomputation |
| r4bm03 | 280/540 | 7.0e9 | 8.0e9 | r4 measured 4.02 GB, plus the recomputation |

**ref-at-wave:** four lead CPU commands through `bwd_r4b_refwave_r5.py`, root `ref-at-wave-r5`, 6.0e9 bytes expected peak each.

| Command | Expected | Timeout |
|---|---:|---:|
| reference | 30 s | 330 s |
| fd part 0 / part 1 (can run in parallel) | 430 s each | 730 s each |
| merge | 40 s | 340 s |

**Run order:**
1. The four ref-at-wave commands. They use the existing r4 wave-capture, whose line is unaffected.
2. `g1-mfcc` and `r4bm03`.

`g1-linear` can run at any point.

## Deviations

- **Repair 1 of 3: bitwise comparison failed on the real bundle.** The consumer's recomputation is bitwise equal to the producer's only at the same BLAS thread count:
  - with 3 threads, 316 of 316 tensors are bitwise equal;
  - with 4 threads, 0 of 316 are, with a worst difference of 1.63e-13 absolute L2 (6.9e-15 of the largest tensor).

  The GPU worker runs 4 threads, so the check now allows 1e-12 × the largest tensor per tensor. A substituted reference is about 1e-4 of the largest tensor off.
- **Repair 2 of 3:** the verifier now refuses stored plateau and error claims that disagree with their recomputation, and the self-test's bundle clone keeps a renamed part's digest.
- **Producer kept unchanged on purpose:** `bwd_r4b_refwave.py` is untouched, so your real r4 bundle can serve as the positive case. Its old, weaker `verify_bundle` is still in that file but no line calls it any more.
- **r4 self-test not rerun:** justified by the byte-identical pins above.

## Prior art

- **Content-addressed verification:** recompute a digest from the bytes and compare it with an independent pin, as in Merkle (1979), Git object stores and Nix (Dolstra, 2006). Taken.
- **Recomputing a derived artefact from its inputs instead of trusting a stored copy:** reproducible-build practice. Taken.
- **Mutation testing:** DeMillo, Lipton and Sayward (1978).
- **Ours:** the registered inventory, the per-row registered-value rules, the consumer's recomputation at `wave_c`, and the fp64 roundoff bound, which I measured.
- **Unverified, lead to check:** "Merkle 1979", "Dolstra Nix 2006", "DeMillo Lipton Sayward 1978".

## Files written in r5

**Scripts and inventory:** the six new scripts and `REF_AT_WAVE_INVENTORY_R4B_R5.json` from the diff summary.

**Receipts in `artifacts/bwd_r4b/`:**
- Self-tests: `selftest-r5-verify/` (failed run), `selftest-r5-verify-r2/` (failed run), `selftest-r5-verify-r3/` (`SELFTEST_R4B_R5.json`, passing), `selftest-r5-refwave-wrapper/`.
- G6: `G6_{r4a,prod,candidate}_R4B_R5.{json,xml}`, `G6_{r4a,prod,candidate}_RUN_R4B_R5.json`, `G6_PER_ID_R4B_R5.json`.
- Staging: `STAGED_PINS_R4B_R5.{json,sha256}`, `STAGED_LINES_R4B_R5.md`, `PIN_COMPARISON_R4B_R5.json`, three `dry_*_R4B_R5.json`.
- Census and ledger: `CENSUS_END_R4B_R5.sha256`, `LEDGER_R4B.md`.

**Logs:** `logs/bwd_r4b_r5_{inventory, selftest, selftest_r2, selftest_r3, g6_run_*, g6_*, g6_compare, trackability, stage_prepare, dry_runs, wrapper_check}.log`, and three `logs/bwd_r4b_r5_dry_*_R4B_R5.log`.

**Scratch left in place in /tmp (nothing deleted):** `/tmp/bwd_r4b_r5_g6/`, `/tmp/bwd_r4b_r5_selftest{1,2,3}/` (bundle copies included), three `/tmp/bwd_r4b_r5_lease_*`.

_Filed by the lead from the seat's final message; the seat's own report write was refused by a hook._
