# GN1-REV3 review report

- **Reviewer:** Claude Opus 5.5 (Agent-tool seat). This is a cross-model re-review of Codex's A3 (`cf922891`).
- **Order:** `orders/GN1_REV3_REVIEW.md` (7c4e6483).
- **Run:** 2026-11-01, 12:33 to about 12:58 EST, CPU only.

**Note from the lead.** The seat's tool rules forbid it from writing report files. The lead filed its final message
here, re-laid out in sections with its wording kept. `findings.json` is the seat's own machine-readable record. The
seat disclosed one adaptation. `rev3/parent_wrap.py` runs A3's real `run_gpu_line.py` with a single change: its
TMPDIR check uses `/tmp/gn1_rev3_*`, the reviewer's allowed prefix. Everything else is A3's code, byte for byte, in a
seal-verified scratch mirror.

**Verdict: REJECT.** 6 findings: 0 BLOCKER, 0 MAJOR, 3 MINOR, 3 NIT.

- **REV2's findings:** 10 FIXED, 1 NOT FIXED. R2-F07 is incomplete and becomes R3-F03.
- **Adjacent defects:** the fixes for R2-F01, R2-F04 and R2-F11 each have a new defect beside them (R3-F01, R3-F02,
  R3-F05).
- **Sound:** the kill rule (Q2) and D5 validation (Q4). Arm regeneration is byte-identical, and an honest run of all
  56 labels passes the scorer.

## Findings

**R3-F01, MINOR. An outer timeout during the runner's lock wait counts as an admitted failure, and its label can
never be retried.**
- **Where:** `a3/evidence.py:199-201`, `a3/run_gpu_line.py:54-61`. The runner's lock wait (lines 255-278) has no signal
  handler, so a SIGTERM kills it before its log row is written.
- **Reproduction:** B6, using the real runner with the scratch lock held. There is no runner row and no
  `invocation.json`, yet the parent records exit 124, `admitted`, cause `outer-timeout`. The retry is refused.
- **Fix:** treat exit 124 or 137 with no runner row and no child start marker as not admitted. Have
  `run_gpu_tests.py` write a start marker as its first action. Add a G4 control that holds the lock past the outer
  bound.

**R3-F02, MINOR. The timeout-cause function raises on real runner rows, and the parent dies without a receipt.**
- **Where:** `a3/evidence.py:184-196`, called at `:288` outside `evaluate()`'s `try`.
- **Defect:** a SIGTERM during the runner's kill grace logs `timeout 143` or `window-closed 143`.
- **Reproduction:** B7, B7b and B8. Each parent exits 1 with `ValueError: cannot attribute timeout from runner log`,
  and in B7 and B8 the attempt is then retried as not admitted.
- **Fix:** map those rows to `outer-timeout`. Never raise from `evaluate()`; record the problem as a reason instead.
  Add B7, B7b and B8 as G4 controls.

**R3-F03, MINOR. "The child never started" is tested as "no `child_result.json`". This is why R2-F07 is not fixed.**
- **Where:** `a3/evidence.py:327-331`. The code ignores `invocation.json` (`run_gpu_tests.py:104`) and the runner's
  row.
- **Reproduction:** C7b. A child that started and died is retried, and the retry passes.
- **Fix:** an attempt is not admitted only when there is no start evidence at all: no marker or `invocation.json`, no
  `ran`, `timeout` or `window-closed` row, and no `junit.xml`. Otherwise it is admitted, evaluated and failed. Add
  C7b as a G4 control.

**R3-F04, NIT. A truncated `child_result.json` crashes every later parent for that label.**
- **Where:** `a3/evidence.py:334`, and the non-atomic write at `run_gpu_tests.py:116`.
- **Reproduction:** E5. The parent dies with a `JSONDecodeError`. This fails closed.
- **Fix:** read the file inside `evaluate()`'s `try`, write it atomically, and add E5 as a G4 control.

**R3-F05, NIT. The import-traced seal sees only collection.**
- **Where:** `a3/trace_imports.py:27`, `seal_staging.py:17-20`, `evidence.py:57-58`.
- **Reproduction:** `seal_runtime_imports.py`. Runtime imports such as `np.random` (`numpy/random/mtrand`) and
  `np.testing` load 23 files, and 19 of them are not among the 513 pins. Local modules are fully covered.
- **Fix:** take the census during execution. The child records the origins in `sys.modules` with sha256, and
  `evaluate()` requires each to be sealed. At minimum, state the gap.

**R3-F06, NIT. The two `numpy.bool_`-clause deletion mutants were dropped instead of being recorded as equivalent.**
- **Where:** `a3/AMENDMENTS.md:3`, `a3/MUTANTS.md:60`, `a3/EQUIVALENCES.md`. A2's bar
  (`orders/GN1_A2_REV1_FIXES.md:65`) requires every mutant to be killed or to carry a proof.
- **Reproduction:** `npbool_deletion_probe.py`. Under NumPy 2.4.2, `operator.index(numpy.bool_)` raises, and each
  deletion passes all 405 GN tests.
- **Fix:** record both as sealed equivalences, with the proof and its NumPy-version dependency, and stage them with
  an expected pass, as V3 is. Keep the negations as reach checks.

**Outside the order's scope, not counted.** The scorer does not tie the parent's recorded outer bound, or the runner's
`--max` and inner timeout, to `gpu_lines.json`. It checks only the command's last three arguments. A2 and A3 both
had this. The staged script matches `gpu_lines.json` on all 56 lines. The lead decides.

## Per question

- **Q1, REV2 findings: 10 FIXED, 1 NOT FIXED (R2-F07).**
  - Every REV2 reproduction was re-run against A3.
  - REV1's H1 to H6: 20 of 20 rejected.
  - C1 to C6 were rejected by their rules, and D's 7 of 7 pass.
  - C8: all 56 labels pass the scorer.
- **Q2, the kill rule: SOUND.**
  - P1 and P5, generic engine errors: kills.
  - P3 and P6, device faults: vetoed.
  - P4, an error raised only in backward: not counted.
  - P7: a kill.
  - P2 and P8, a callee raising inside `__init__`: not counted.
  - A3's real-test controls behave the same when re-run independently.
- **Q3, runner outcomes: UNSOUND** (F01, F02, F03). Every logged outcome is recognised. Not covered: the unlogged
  lock-wait kill, rows with 143 inside the grace, and a started child that left no result.
- **Q4, D5 validation: SOUND.**
  - Counts: 53 of 54 cases behave as expected. The one miss is the probe's own wrong expectation: `2**70` is refused
    by NumPy's allocation.
  - Shapes: 36 of 36 are refused before any tensor op.
  - All 324 A2-accepted configurations give bit-identical results.
- **Q5, the amendments:** amendment 1 is UNSOUND at NIT level (F06); amendment 2 is SOUND.
  - The deletion is really equivalent under NumPy 2.4.2.
  - The matrix61 witness is a fair isolation.
  - The 20 s to 50 s bound appears only in `hostile_scorer.py:178`, and no acceptance predicate reads it.
- **Q6, mutants and seal: UNSOUND at NIT level** (F05).
  - All 52 mutants are single-site.
  - 51 reach their edit during the call; M14 is reached through default-argument binding.
  - 51 witnesses fail on their mutant, and V3 passes as registered.
  - Regeneration matches all 54 arms, and the TMPDIR guard resolves paths.
  - Scope: `nn.py` and the test file only.

## Prior art

- Fault injection: Meszaros (2007).
- Mutation analysis: DeMillo, Lipton and Sayward (1978). Equivalent mutants and the reachability-infection-
  propagation model: Offutt and Voas. Unverified; lead to check.
- `operator.index`: PEP 357 (2006).
- Boundary-value testing: Myers (1979). Unverified; lead to check.
- SHA-256: FIPS 180-4.
- PyTorch 2.11.0's behaviour was observed live on CPU.

## Files

In `artifacts/gn1/rev3/`:
- `findings.json`;
- the mirror and regeneration scripts and outputs;
- `parent_wrap.py`, `rev3_forge.py`;
- `hostile_rev3*` scripts and outputs;
- `q3_probe_rev3*`, `d5_probe_rev3*`, `q5_*`, `v4_rev2_nodes*`, `npbool_deletion_probe*`, `seal_runtime_imports*`,
  `seal_mirror_collect*`, `tmpdir_guard_probe*`.

The runner's absolute path and the home directory are redacted in the `hostile_rev3_{ad,b,c}.json` files. Scratch is
left under `/tmp/gn1_rev3_*`.
