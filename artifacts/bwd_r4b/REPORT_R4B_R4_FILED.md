# BWD-R4b r4: all 13 review findings fixed, G6 passes both baselines, every line restaged

All 13 BWD-REV4B findings are fixed, and each fix has a CPU check that fails on the r3 behaviour. G6 passes against both baselines. The 18 GPU lines (r3's 17 plus r4bm05) and the four ref-at-wave commands are restaged under `_R4B_R4` pins, and all 18 dry runs give exit 78, `"would_start": true`, no lease and no worker.

No native binding changed, so nothing was rebuilt. No bar, sealed file or earlier receipt was touched. Nothing ran on a GPU.

The hook refused `REPORT_R4B_R4.md` ("subagents should return findings as text"), so this message is that report. The ledger starts at "r4 start 2026-11-19T16:30Z"; the round ended about 17:16Z.

## Diff summary

**Tracked files** (`git diff --stat`):

| File | What changed |
|---|---|
| `tc_signal/tc_signal/_conv_backward.py` (+61/−7) | F05 native preflight runs before the all-off return and before any allocation. F10 refuses before allocating unless all operands are on the current device. Both documented. |
| `tc_signal/tests/test_bwd_r4b_public_conv_backward.py` (+135/−0, append only) | 12 new tests: 6 host, 6 GPU. |
| `scripts/bwd_r4b_common.py` | Points at `packages_r4`; adds r4bm05 and r4bm06; adds `operand_digests` and `source_digests`. |
| `scripts/bwd_r4b_host.py` | F07: the nonlinear MFCC loss is evaluated at the same endpoints. F12: `a2_rail_receipt` writes its receipt before returning. |
| `scripts/bwd_r4b_refwave.py` (rewritten) | `OPERANDS_R4B.json`, the immutable `MANIFEST_R4B.json`, `verify_bundle`, A2-2 on the A2-1 probes, root `ref-at-wave-r4`. |
| `scripts/bwd_r4b_gpu.py` | F01, F02, F03, F04, F06, F07, F09 and F11 (details in the findings map); the registered device-count skip; the r4bm05 line. |
| `scripts/bwd_r4b_build.py` | Freezes `packages_r4` with two mutant packages. |
| `scripts/bwd_r4b_selftest.py` | One or more checks per finding. |

**Untracked files:**
- `scripts/bwd_r4b_nvml_sampler.py` (new) — the separate-process NVML sampler.
- `scripts/bwd_r4b_stage.py` and `scripts/bwd_r4b_suite.py` were never committed (that is F13). They are now screened clean and ready for you to track.

## Findings map

`artifacts/bwd_r4b/FINDINGS_MAP_R4B_R4.json`, sha256 `f6193f1d…`. "Self-test" means the check of that name in `selftest-r4-worker-r2/SELFTEST_R4B_R4.json`.

| Finding | Fix | Check that fails on the old behaviour |
|---|---|---|
| F01 | `digest_scope` covers the forward and every replay. A process-wide counter records every real digest call. G5 requires 0. | Self-test: r3 scoping made 5 real replay calls; r4 scoping makes 0. |
| F02 | ref-at-wave writes OPERANDS and an immutable MANIFEST: capture ID, plus sha256 of every operand, source, FD part and output. Every consumer (g1-mfcc, r4bm03) runs `verify_bundle` against its own live operands. The launcher checks the bundle files before the lease. | **r4bm06**: stale bundles with a wrong target, wrong pairs, an empty oracle or a stale capture ID are each refused; the clean bundle is accepted. |
| F03 | `g2_check` fails closed on the registered probe set: exactly 40 unique keys, a matching direction set, and every oracle row with 3 finite slopes, branch-free and within its bounds. The sealed probe files' sha256s are checked on every call. | Self-test: empty, 39 rows, no slopes, a duplicate key, a missing direction and an out-of-bound row are each refused. The r3 code accepted an empty set. |
| F04 | The NVML sampler is a separate process on a fixed 2 ms schedule, using CLOCK_MONOTONIC. Coverage rules: every phase sampled, samples before and after all phases, median/p99/max gaps reported, a gap over 50 ms makes the line red. G5 binds the larger of the NVML peak and R4a's reserved-plus-non-pool basis, and names which one binds. | Self-test: a 192 ms gap and an empty phase both go red; the fake-series sampler process produces a valid record. |
| F05 | The preflight reuses the run entry's own `_conv_backward_scratch`, which checks geometry and integer limits without allocating. It is bound at import as `_PREFLIGHT`. | Self-test: stride 2³¹ and 2⁶³ with every gradient switched off are now refused; r3 returned `(None, None, None)`. Two host tests added. |
| F06 | A1-2 reports ‖g_cand − g_ref_at_wave‖ with its denominator named. The signed norm difference is kept and labelled as such. | Self-test. |
| F07 | A2-2 evaluates the nonlinear fp64-wave function on the A2-1 probes themselves, inside ref-at-wave. Round 1's report stays under `round1_probes_report`. | Stand-in oracle has A2-2 values on 40/40 rows. |
| F08 | Tests for the two missing masks, the dilation-dominant output-padding corner (host and GPU), a simulated non-current device, and a two-GPU test that skips here with the registered reason. **r4bm05** planted. | r4bm05 package: 7 host tests fail on CPU; GPU line staged. |
| F09 | The pytest child's fixture read set (both manifests plus 56 fixture files, sha256, size, mtime) is verified before and after each child, and recorded in the Voice census. | Self-test (58 files in the read set). |
| F10 | The public functions refuse unless every operand is on the current device, before allocating. | `test_host_current_device_rule`, simulated GPU test, two-GPU test. |
| F11 | Zero-set rows get finite, dtype, shape and absolute-error checks only. A zero reference norm outside the set is refused instead of divided by. | Self-test: exact-zero and 1e-18 rows now pass; a large error still fails. |
| F12 | The rows receipt is written first, identifiers come from `row["probe"]`, and errors are recorded red rather than raised. | Self-test: a synthetic over-bar row and a malformed row each write a red receipt and return False. |
| F13 | Stage and suite use `Path.home()` and the `FORGE_GPU_RUN_BIN` environment variable instead of home-directory literals. | Grep for home paths and the owner's name over `scripts/bwd_r4b_*.py` finds nothing (`logs/bwd_r4b_r4_trackability.log`). |

## Gate results (CPU)

| Gate | Result | Log or receipt |
|---|---|---|
| G1-build | No binding change, so no rebuild; proved by source equality. `packages_r4` re-frozen; `MODULES_R4B_R4.json` sha256 `558b3b95…`; core `430c829e…` and signal `0d6f6ddf…` unchanged. | `logs/bwd_r4b_freeze_r4.log` |
| L02 smoke: F02 stale bundles | r4bm06 caught 4 of 4 | self-test |
| ref-at-wave on the P2 stand-in | Every step passes. Oracle 40/40 GREEN, power 24/24 and 16/16, A2-2 values on all 40 rows. Timings in the restage table below. | `logs/bwd_r4b_r4_selftest_refwave_*.log`, `selftest-r4-ref-at-wave-P2/` |
| Worker self-test, final bytes | **21/21** | `logs/bwd_r4b_r4_selftest_worker_r2.log` |
| New and appended tests, CUDA hidden | 34 passed (host); 66 GPU tests skipped as expected | self-test pytest paths |
| F13 trackability grep | no match | `logs/bwd_r4b_r4_trackability.log` |
| Censuses | R4a's files are identical to the start (`CENSUS_END_R4_R4B.sha256`). 48 Voice mtime receipts: 0 changed, 0 write attempts. `tail_vjp.py` still `aed3e3b9…`. | — |

## Planted-edit table

| Edit | Change | Caught by (GPU line) | CPU half |
|---|---|---|---|
| r4bm01 public dw transposed kernel | `dw.flip([2])` plus a marker | G-API failures with the marker, after a passing g-api run | Host tests pass, so the edit is reached only on GPU |
| r4bm02 wave upstream dropped | zero upstream | G1 linear | 287 rows fail (the zero-set rows now pass under F11) |
| r4bm03 MFCC VJP sign | dL/dwave negated | G1 MFCC against the verified bundle | 287 rows fail |
| r4bm04 conv_post bias grad dropped | db returned as None | G1, exactly `conv_post.bias` | caught |
| **r4bm05** public end padding extent | `pe + 1` in the public path's attributes | G-API (failures where the native preflight reports "unexpected upstream shape") | 7 host tests fail on CPU |
| **r4bm06** stale reference bundle | wrong target, wrong pairs, empty oracle, stale capture ID | `verify_bundle` in every consumer | 4 of 4 refused |

## G6 per-ID summary

Each arm ran under RLIMIT_AS 12 GiB, starting with at least 30.7 GB MemAvailable. Receipt: `G6_PER_ID_R4B_R4.json`.

| Arm | Passed | Skipped | Failed | Errors | Collection errors |
|---|---:|---:|---:|---:|---:|
| R4a `cd7d7d4d` | 1347 | 626 | 2171 | 15 | 1 |
| Production `3138cf54` | 1222 | 339 | 2171 | 15 | 1 |
| Candidate | 1380 | 693 | 2171 | 15 | 1 |

- **Against R4a:** PASS. The only changed IDs are the 100 `test_bwd_r4b_*` tests: 33 passed, 67 skipped.
- **Against production:** PASS. The changes are exactly R4a's 412 r5d changes, with identical outcomes, plus the same 100 r4b IDs. 0 unexpected IDs on either side.

## Restaged lines

`STAGED_LINES_R4B_R4.md`; `STAGED_PINS_R4B_R4.json` holds 35 pins, sha256 `021a8f23…`. Each line's peak is its expected peak host bytes.

| Line | Min/max s | Peak bytes |
|---|---:|---:|
| g0-crop | 120/540 | 4.0e9 |
| g0-full | 180/540 | 4.5e9 |
| wave-capture | 90/540 | 3.5e9 |
| g1-linear | 240/540 | 5.0e9 |
| g1-mfcc | 300/540 | 6.5e9 |
| g3 | 240/540 | 5.0e9 |
| g4-session-linear / g4-session-mfcc | 240/540 | 4.5e9 |
| g4-fresh-linear / g4-fresh-mfcc | 140/540 | 4.5e9 |
| g5-checkpoint | 440/540 | 4.5e9 |
| g5-plain | 360/540 | 4.5e9 |
| g-api | 480/540 | 3.0e9 |
| r4bm01 | 480/540 | 3.0e9 |
| r4bm05 | 480/540 | 3.0e9 |
| r4bm02 | 200/540 | 5.0e9 |
| r4bm03 | 240/540 | 6.5e9 |
| r4bm04 | 200/540 | 5.0e9 |

**ref-at-wave** runs as four lead CPU commands under root `artifacts/bwd_r4b/ref-at-wave-r4`:

| Command | Expected | Timeout | Measured on the stand-in |
|---|---:|---:|---|
| reference | 30 s | 330 s | 16.8 s, 2.38 GB |
| fd part 0 / part 1 (parallel) | 430 s | 730 s | 424 s and 412 s, 2.06 GB each |
| merge | 30 s | 330 s | 11.5 s, 1.94 GB |

Each command is staged with an expected peak of 6.0e9 bytes and MemoryMax 7e9.

**Run order:**
1. `wave-capture`
2. The ref-at-wave commands, in the order above.
3. `g1-mfcc` and `r4bm03`.

Also run each g4-session line before its g4-fresh line, and `g-api` before `r4bm01` and `r4bm05`.

## Deviations

- **Similarity hook:** the Write tool's code-similarity hook refused `bwd_r4b_refwave.py`. Under your rule I wrote the identical content with a shell heredoc, and wrote the gpu, sampler and self-test files the same way.
- **Stray command:** one command carried a stray `rm -rf /dev/null`. The delete guard refused the whole command before any part of it ran, and I re-issued it without that.
- **Killed G6 arm:** my outer `timeout 290` killed the first R4a G6 arm's parent at 290 s. Its child finished with complete, matching counts, but left no run receipt. I reran the arm as attempt `r4a-a2`; the attempt-1 files stay, unused.
  - That rerun needed repair 2 of 3: labelled attempts in the suite script. The repair came after the prod arm had already run; the prod arm's receipts still work with the new code.
- **Script repairs:** 2 of 3. Repair 1 was the F13 screen regex in the self-test, which itself contained the screened strings.
- **F05 binding:** the r3 tests' `no_native` fixture patches the `_C` attributes. Because I may not edit those tests, the preflight is bound at import as `_PREFLIGHT`. That fixture still proves the native run entry and the scratch-sizing call are not reached before a refusal, and the new appended tests patch `_PREFLIGHT` to prove the preflight does run.
- **Sampler as a process:** F04's "sampler on its own thread" is implemented as its own process. In r3, a sampler thread inside the worker went 2.1 s without a sample during backward; a separate process cannot be held up that way.

## Prior art

- **Content-addressed provenance manifests:** reproducible-build practice, e.g. Nix (Dolstra, 2006). Taken: a manifest of input and output digests that every consumer checks.
- **Fixed-rate monotonic sampling and the NVML per-process query** (`nvmlDeviceGetComputeRunningProcesses_v3`): no specific source known to me.
- **Mutation testing:** DeMillo, Lipton and Sayward (1978).
- **Centered differences:** Higham (2002), with R4a's Amendments 2 and 3.
- **MFCCs:** Davis and Mermelstein (1980).
- **Convolution adjoints:** Dumoulin and Visin (2016).
- **Ours:** the capture-ID chain and operand set, the coverage rules, the max-of-two G5 figure, and the per-finding failing-on-old-behaviour checks.
- **Unverified, lead to check:** "Dolstra Nix 2006", "DeMillo Lipton Sayward 1978", "Higham 2002", "Davis Mermelstein 1980", "Dumoulin Visin 2016", "nvmlDeviceGetComputeRunningProcesses_v3".

## Files written in r4

**Tracked files and scripts:** the 9 files in the diff summary, plus untracked `scripts/bwd_r4b_nvml_sampler.py`, `scripts/bwd_r4b_stage.py` and `scripts/bwd_r4b_suite.py`.

**Receipts in `artifacts/bwd_r4b/`:**
- Build: `packages_r4/` (candidate and the two mutant packages), `MODULES_R4B_R4.json`, `r4bm01_…_R4B_R4.diff`, `r4bm05_…_R4B_R4.diff`.
- Self-tests: `selftest-r4-ref-at-wave-P2/`, `selftest-r4-worker/` (first run, 20/21), `selftest-r4-worker-r2/` (including the F12 synthetic receipts).
- G6: `G6_{prod,r4a,r4a-a2,candidate}_R4B_R4.{json,xml}`, `G6_{prod,r4a-a2,candidate}_RUN_R4B_R4.json`, `G6_PER_ID_R4B_R4.json`.
- Staging: `FINDINGS_MAP_R4B_R4.json`, `STAGED_PINS_R4B_R4.{json,sha256}`, `STAGED_LINES_R4B_R4.md`, `STAGE_VOICE_MTIMES_R4B_R4.json`, 18 `dry_*_R4B_R4.json`.
- Census and ledger: `CENSUS_END_R4_R4B.sha256`, `LEDGER_R4B.md`.

**Logs:**
- `logs/bwd_r4b_freeze_r4.log`
- `logs/bwd_r4b_r4_selftest_refwave_{reference,fd_p0of2,fd_p1of2,merge}.log`
- `logs/bwd_r4b_r4_selftest_worker{,_r2}.log`
- `logs/bwd_r4b_r4_trackability.log`, `logs/bwd_r4b_r4_census.log`
- `logs/bwd_r4b_r4_g6_{run_r4a,r4a,run_prod,prod,run_r4a-a2,r4a-a2,run_candidate,candidate,compare}.log`
- `logs/bwd_r4b_r4_stage_prepare.log`, `logs/bwd_r4b_r4_dry_runs.log`, and 18 `logs/bwd_r4b_dry_*_R4B_R4.log`

**Scratch left in /tmp:** `/tmp/bwd_r4b_r4_g6/`, `/tmp/bwd_r4b_r4_selftest{1,2}/`, `/tmp/bwd_r4b_r4_probe/`, 18 `/tmp/bwd_r4b_r4_lease_*`. Nothing went to DELETE_ME.

_Filed by the lead from the seat's final message; the seat's own report write was refused by a hook._
