<!-- Filed by the lead from the seat's final message (the session hook refuses seat report files). -->

# BWD-R4b r10: the START and END markers' pid is now joined in both consumers, and the lines are restaged

The one hardening fix (confirm 4 H01) is in. An END marker whose pid differs from its START marker is now refused by both the root check and the prior-mode check; r9 accepted it in both. The planted edit r4bm36 shows this, and the real `ref-at-wave-r9` root and a fresh merge still pass.

- **Self-test:** passed 4/4 on the first run, on the final bytes. No script repairs were needed.
- **Restage:** 8 lines plus the 4 ref-at-wave commands, under new r10 pins. All 8 dry runs gave exit 78 with `"would_start": true`.
- **G6:** not rerun, per the order. `G6_PER_ID_R4B_R9.json` stands.
- **Not touched:** no test, product, package, native or kernel edits, no git, no deletes, no GPU, no Voice writes.
- **Ledger:** the first new line is "## r10 start 2026-11-20T01:50Z". The round ended at 01:54Z.

## Diff summary

**Edited:**
- `scripts/bwd_r4b_verify.py`: `_verify_root` joins END's `pid` to START's with the shared typed join, so `verify_root_r9` now refuses a mismatch.
- `scripts/bwd_r4b_refwave_r9.py`: `prior_passed` makes the same join. Its CLI now refuses superseded use and names r10.
- `scripts/bwd_r4b_stage_r9.py`: every r9 label is refused and names the r10 launcher.

**New:**

| File | Purpose |
|---|---|
| `bwd_r4b_refwave_r10.py` | The r9 wrapper code on root `ref-at-wave-r10`, under the r10 pins |
| `bwd_r4b_gpu_r10.py` | The r10 round constants |
| `bwd_r4b_stage_r10.py` | The r10 launcher |
| `bwd_r4b_selftest_r10.py` | The CPU self-test |

## r4bm36 in both consumers

| Variant | Root check, r9 | Root check, r10 | Prior-mode check, r9 | Prior-mode check, r10 |
|---|---|---|---|---|
| Clean copy | pass | pass | pass | pass |
| `merge` END pid +1 | accepted | refused | (merge END is not an input here) | (merge END is not an input here) |
| `reference` END pid −1 | accepted | refused | accepted | refused |

The refusal message is "end marker names another process than its start marker".

## Positive cases

- **Real `ref-at-wave-r9` root:** the r10 root check, the prior-mode check and the full bundle verification with fresh products all pass.
- **Fresh merge on a copy:** the reference and fd outputs were copied in through the r9 marker writers. The merge then ran in a child process through the wrapper's CLI path, in 36.7 s. It reproduced the sealed manifest `634bf770…`, and the root check and full bundle check accepted it.
- **Superseded entry points:** the r9 launcher and the r9 wrapper both refuse and name r10.

## Self-test

`artifacts/bwd_r4b/selftest-r10_R4B_R10/SELFTEST_R4B_R10.json`, sha256 `64ec288f…`, 4/4 PASS.
- **Trackability grep:** no match.
- **R4a census:** identical at the end.
- **Voice:** 0 changed, 0 write attempts; `tail_vjp.py` is still `aed3e3b9…`.

## Pin comparison

The new pins are `STAGED_PINS_R4B_R10.json`: 77 pins, sha256 `7bf60c20…`, sealed.
- **r9 pins changed:** `bwd_r4b_refwave_r9.py`, `bwd_r4b_stage_r9.py`, `bwd_r4b_verify.py`.
- **Unaffected:** none of the files the 10 unaffected r4 lines execute changed, and `packages_r4` is unchanged.

## Restaged lines

All 8 dry runs gave exit 78 with `"would_start": true`, no lease and no worker. Each `--min` is at most 510 under `--max 540`.

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

- **ref-at-wave:** four commands through `bwd_r4b_refwave_r10.py`, root `ref-at-wave-r10`. Expected seconds: reference 30, each fd part 540, merge 90; 6.0e9 peak each.
- **If any ref-at-wave command fails:** rerun all four on `ref-at-wave-r10-a2`, then pass `--ref-root ref-at-wave-r10-a2` to `g1-mfcc` and `r4bm03`.
- **Run order:**
  1. The four ref-at-wave commands.
  2. `g1-mfcc` and `r4bm03`.
  3. `g-api` before `r4bm01` and `r4bm05`.

## Deviations

None. No script needed a repair, and there were no tool slips.

## Prior art

- **Joining both markers of one commit record:** commit-record discipline (Gray, 1978; ARIES, Mohan et al., 1992), as in r9.
- **Planted edit:** mutation testing (DeMillo, Lipton and Sayward, 1978).
- **Unverified, lead to check:** "Gray 1978", "ARIES Mohan 1992", "DeMillo Lipton Sayward 1978".

## Files written in r10

All paths are under `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_bwd/WorkTree/tensor_fork_bwd_wt_r4b/`.

**Code:** the edited and new files in the diff summary.

**Receipts in `artifacts/bwd_r4b/`:**
- Self-test: `selftest-r10_R4B_R10/` (`SELFTEST_R4B_R10.json`, `child_merge.log`, `VOICE_MTIMES_R4B_R10.json`).
- Merge copy: `st-r10-merge_R4B_R10/`, the bundle copy the merge child ran on.
- Staging: `STAGED_PINS_R4B_R10.{json,sha256}`, `STAGED_LINES_R4B_R10.md`, `PIN_COMPARISON_R4B_R10.json`, `G_API_CENSUS_R4B_R10.json`, eight `dry_*_R4B_R10.json`.
- Census and ledger: `CENSUS_END_R4B_R10.sha256`, `LEDGER_R4B.md`.

**Logs (`logs/`):**
- `bwd_r4b_r10_selftest.log`
- `bwd_r4b_r10_stage_prepare.log`
- `bwd_r4b_r10_dry_runs.log`, eight `bwd_r4b_r10_dry_*_R4B_R10.log`
- `bwd_r4b_r10_trackability.log`

**Scratch left in /tmp (nothing deleted):** `/tmp/bwd_r4b_r10_before`, `/tmp/bwd_r4b_r10_selftest`, `/tmp/bwd_r4b_r10_census_*`, and eight `/tmp/bwd_r4b_r10_lease_*`.
