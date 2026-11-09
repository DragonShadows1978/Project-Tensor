# GN1-REV4 review report

- **Reviewer:** Claude Opus 5.5 (Agent-tool seat). This is a cross-model re-review of Codex's A4 (`a14c0aad`).
- **Order:** `orders/GN1_REV4_REVIEW.md` (b6805dcf).
- **Run:** 2026-11-01, 13:46 to 14:08 EST, CPU only.

**Note from the lead.** The seat's tool rules forbid report files, so the lead filed its final message here, re-laid
out in sections with its wording kept. `findings.json` is the seat's own record.

The seat made one disclosed adaptation, the same as REV3's: `parent_wrap4.py` runs A4's real `run_gpu_line.py` with
the TMPDIR check widened to `/tmp/gn1_rev4_*`. Everything else ran byte for byte from a seal-verified mirror, with
independently regenerated arms. The installed runner's summary, window and lock were always redirected.

**Verdict: REJECT.** 4 findings: 0 BLOCKER, 0 MAJOR, 2 MINOR, 2 NIT.
- R3-F01 to R3-F06 are all fixed against their own reproductions. Three of those fixes have a new adjacent defect
  (R4-F02, F03, F04).
- D7 is not fixed (R4-F01).
- Q2, Q4 and Q6 are SOUND. Q3 and Q5 (bound binding) are not.

## Findings

**R4-F01, MINOR. The D7 bound check accepts commands whose real bounds differ from `gpu_lines.json`.**
- **Where:** `a4/evidence.py:323-324` checks `--max`/`--min` by counting tokens. `:328-329` takes the inner duration as
  the first token after `timeout` that does not start with `-`.
- **Reproduction:** `hostile_rev4_cd.json` part d, line `candidate_focus`. Each of these passes the scorer:
  1. `--min 250 --max 300 --max=900`;
  2. `--mi 1`;
  3. `--min=1`;
  4. inner `timeout -k 240 900s`;
  5. inner `timeout --kill-after 240 900s`.
- **What actually runs:** the runner's own argparse gives a max of 900 for case 1 and a min of 1 for cases 2 and 3.
  GNU `timeout` reads the value after `-k` as the kill delay, so in cases 4 and 5 the real inner bound is 900 s.
- **Fix:**
  - Require the exact form: the runner arguments equal `['--min', str(min), '--max', str(max), '--label', label]`,
    or are parsed with the runner's argparse spec;
  - the inner command equals `['timeout', '--kill-after=10s', f'{inner}s', 'python3', 'run_gpu_tests.py', arm,
    suite]`;
  - add the five cases above as G4 controls.

**R4-F02, MINOR. In the staged setup, an outer timeout during a run is recorded as exit -9, with no runner row and no
cause.**
- **Where:**
  - `a4/run_gpu_line.py:56-58`: the outer `timeout --kill-after=10s`, recording a raw return code;
  - `evidence.py:185-187`: attribution only for exits 124 and 137;
  - `STAGED_GPU_COMMANDS.sh`: does not set `FORGE_GPU_KILL_GRACE`. The runner's default grace is 10 s (runner line
    115), equal to the outer kill delay, and the runner waits its full grace (lines 209-215).
- **Reproduction:** `hostile_rev4_bn.json`, cases N2b and N2c. With the real runner and the default grace, the result is
  exit -9, `forge_outcome` "missing", and `timeout_cause` null. A4's controls all set the grace explicitly.
- **Fix:**
  - normalise negative exits (`rc < 0` becomes `128 - rc`) before attribution;
  - make the runner's grace finish before the outer KILL: set `FORGE_GPU_KILL_GRACE` below the outer kill delay, or
    raise the outer `--kill-after` above grace plus poll, and keep the script-level timeout's margin;
  - add a G4 control with the grace left unset.

**R4-F03, NIT. The parent can still crash without a receipt in the not-admitted branch.**
- **Where:** `a4/run_gpu_line.py:62` calls `timeout_cause` outside any `try`. `evidence.py:201` raises for
  `no-lock`, `rechecked-out`, `no-window` or unrecognised rows when the exit is 124 or 137.
- **Reproduction:** N5, a `no-lock 76` row with exit 124. The parent exits 1 with a `ValueError` and writes no
  receipt.
- **Fix:** `timeout_cause` never raises. It returns `outer-timeout` for any non-start row with exit 124 or 137, and
  `unattributed` otherwise. Alternatively, wrap the call. Add N5 as a G4 control.

**R4-F04, NIT. A truncated `parent_result.json` crashes every later parent for that label.**
- **Where:** `a4/run_gpu_line.py:90-91` writes it non-atomically. `:42` reads it via `evidence.py:386` outside any
  `try`.
- **Reproduction:** N4. The next parent exits 1 with a `JSONDecodeError`. This fails closed.
- **Fix:** publish atomically: write a temporary file, then `os.link` it to the final name, which keeps no-overwrite.
  In the prior-attempt check, catch read errors and refuse with a JSON error line. Add N4 as a G4 control.

## Per question

- **Q1:** R3-F01 to R3-F06 are FIXED against their reproductions; D7 is NOT FIXED.
  - B6 and N1: lock-wait kills are not admitted, and the retry is selected.
  - B7, B7b and B8: the cause is `outer-timeout`.
  - C7b and N3: 10 classes of start evidence behave as specified.
  - E5 and E5b: a truncated child result becomes a recorded reason.
  - REV3's runtime-import probe: 0 of 23 files missing, with 537 pins.
  - `npbool_deletion_probe`: both deletions pass all 405 tests, and the negations fail 367 and 358 tests.
- **Q2, admission: SOUND.**
- **Q3, timeout causes and robustness: UNSOUND** (F02, F03, F04).
- **Q4, the runtime seal: SOUND.**
  - A real staged `candidate_focus` child recorded 540 modules: 0 tampered, 0 unsealed.
  - The census-instrument change is neutral: the path sets for regression (557), focus (543) and mutation (540) are
    identical before and after it.
  - P109 holds.
- **Q5:** the equivalence records are SOUND. Bound binding is UNSOUND (F01).
- **Q6, arms and regeneration: SOUND.** All 56 arms regenerate byte-identically, and an honest run of all 58 labels
  passes the scorer.

The seat also disclosed corrections to its own harness. Superseded hostile files were caused by its own argument
errors, which D7 correctly refused.

## Prior art

- Meszaros (2007).
- DeMillo, Lipton and Sayward (1978).
- Offutt and Voas: unverified.
- PEP 357.
- FIPS 180-4.
- argparse `allow_abbrev` and GNU `timeout -k`, both observed live.

## Files

`artifacts/gn1/rev4/`:
- `findings.json`;
- the mirror and regeneration scripts;
- `parent_wrap4.py`, `rev4_forge.py`;
- `hostile_rev4*`;
- `rev4_probes.py` and its outputs;
- `real_child_census.*`.

Scratch is left under `/tmp/gn1_rev4_*`.
