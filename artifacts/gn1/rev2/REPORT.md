# GN1-REV2 review report

- **Reviewer:** Claude Opus 5.5 (Agent-tool seat). This is a cross-model re-review of Codex's A2 (`bd30498a`).
- **Order:** `orders/GN1_REV2_REVIEW.md` (02be3cb0).
- **Run:** 2026-11-01, 11:07 to 11:36 EST, CPU only.

**Note from the lead.** The seat's tool refused to let it write this file. The lead filed the seat's final message
here, re-laid out in sections with its wording kept. `findings.json` is the seat's own machine-readable record.

**Verdict: REJECT.** 11 findings: 0 BLOCKER, 0 MAJOR, 4 MINOR, 7 NIT. Of REV1's findings, 9 are FIXED and 2 are NOT
FIXED:
- R1-F08: the installed runner's `rechecked-out` refusal is not recognized (R2-F01).
- R1-F11: the recorded outer-timeout cause is wrong (R2-F04).

Both of REV1's MAJORs (device-fault kills, unbound child summaries) are closed. Arm regeneration is byte-identical,
the D8 bounds reproduce byte-identically (P87), and V3's equivalence proof is correct.

## Findings

**R2-F01, MINOR. R1-F08 is not fixed (D7).**
- **Where:** `a2/evidence.py:157`, `:295`, `a2/run_gpu_line.py:57`, and the installed forge-gpu-run lines 281 and 285.
- **Defect:** the runner logs `rechecked-out` with exit 75 when the window changes while it waits for the lock, or when
  the budget falls below `--min` after the wait. A2 maps that to `unrecognized`, records it as admitted and failed,
  and refuses every retry. All 45 lines would then need a new root.
- **Reproduction:** `hostile_rev2.py` part b, case B1, with the real installed runner redirected to scratch.
- **Fix:** recognise `rechecked-out`. With no `child_result.json`, treat 75 with no-window or rechecked-out, and 76
  with no-lock, as not admitted. Add a G4 control for every outcome the runner can log.

**R2-F02, MINOR. The test wrapper widens D2's kill rule.**
- **Where:** `tests/test_groupnorm_gn1.py:139-142`, `a2/hostile_forge.py:76-79`, and the G4 rows
  `Generic_RuntimeError_not_kill` and `Unregistered_TypeError_not_kill`.
- **Defect:** `_check_values` re-raises any RuntimeError, ValueError or TypeError from `layer(x)` as an
  AssertionError. A generic engine RuntimeError that is not a device fault therefore counts as a kill. The two G4
  controls inject exceptions into a stand-in summary in a shape the sealed test file cannot produce.
- **Reproduction:** `q3_probe.py` P1.
- **Fix, for the lead to choose:** (a) amend D2 so that an engine refusal of a registered valid input is a kill
  unless it carries a device pattern, and replace the controls with real-test-file controls; or (b) narrow the wrapper
  to the registered engine refusal texts.

**R2-F03, NIT. A GroupNorm refusal is detected by stack presence, not by where the exception was raised.**
- **Where:** `a2/run_gpu_tests.py:37-39`, against the comment at `evidence.py:170-171`.
- **Defect:** any GroupNorm frame anywhere in the traceback marks the exception as a refusal.
- **Reproduction:** `q3_probe.py` P2. A factory TypeError raised inside `__init__` counts as a kill.
- **Fix:** classify by the raising frame (`excinfo.traceback[-1]`), and add a control in which a callee raises.

**R2-F04, NIT. R1-F11 is not fixed (D10).**
- **Where:** `a2/run_gpu_line.py:83`.
- **Defect:** `outer_timeout_exit` is true for every exit 124 or 137. That is wrong for three cases:
  - B2, the runner's own deadline;
  - B3, window-closed;
  - B5, the child's inner timeout.

  The scorer never checks the field.
- **Fix:** record the cause `outer-timeout` only when the exit is 124 or 137 and the runner's row is missing, or reads
  `ran` with a logged 143 or 137 that differs. Have the scorer recompute it.

**R2-F05, MINOR. D5 refuses NumPy integer counts.**
- **Where:** `nn.py:517-522`.
- **Defect:** `not isinstance(v, int)` refuses `np.int64`, `np.int32` and `np.uint8`. A1 and PyTorch 2.11 both
  accept them.
- **Reproduction:** `d5_probe.py`.
- **Fix:** accept any value `operator.index` accepts, except `bool` and `numpy.bool_`, and store it as `int`. Add
  tests.

**R2-F06, MINOR. Parameter shape is not validated.**
- **Where:** `nn.py:532-545`, `564` and `567`.
- **Defect:** a weight of shape (2,3), (6,1) or (1,6,1) is silently reshaped and used, and a wrong-size weight is
  refused only incidentally (`kernels.cu:759`). PyTorch refuses all three shapes.
- **Fix:** raise ValueError when `ndim != 1` or `shape[0] != C`. Add tests and mutants.

**R2-F07, NIT. A killed parent bricks its label.**
- **Where:** `a2/run_gpu_line.py:40-43`, `evidence.py:283-284`.
- **Defect:** an attempt directory with no `parent_result.json` makes every later parent for that label crash. This
  fails closed.
- **Reproduction:** case C7.
- **Fix:** if the child's result exists, evaluate that attempt rather than retrying. If the child never started, mark
  it not admitted and allow a retry.

**R2-F08, NIT. M14's trace evidence comes from import time.**
- **Where:** `a2/MUTANTS.md:36`, `mutants/M14/trace.json`.
- **Defect:** the default-argument edit at `nn.py:511` runs when the class is defined, so the trace would hold for any
  test. The kill itself is real.
- **Reproduction:** `q5_mutant_trace.py`.
- **Fix:** trace only the call phase, and record M14 as reached through default-argument binding.

**R2-F09, NIT. The fast-path condition `dtype == "float32"` at `nn.py:562` has no mutant.**
- **Reproduction:** `q5_uncovered_probe.py`. A one-site mutant is killed by
  `test_mixed_parameters[forward-default-float16|bfloat16]`.
- **Fix:** register it as V4 and stage its line.

**R2-F10, NIT. The TMPDIR guard is a string-prefix check.**
- **Where:** `a2/run_gpu_line.py:48`, `a2/STAGED_GPU_COMMANDS.sh:11`.
- **Defect:** `/tmp/gn1_a2_boot/../gn1_rev2_tmp` passes the guard.
- **Fix:** compare `Path(TMPDIR).resolve()`.

**R2-F11, NIT. The seal misses a module a sealed regression test imports.**
- **Where:** `a2/seal_staging.py:17-21`. `tests/test_apa_om135_imm.py:25-26` imports `scripts/apa_om135_core.py`.
- **Reproduction:** a mirror built from the seal alone fails to collect the regression suite. With the module copied
  (sha256 `cb2f5dd7…`), it collects 444 IDs.
- **Fix:** seal the module, and have the child verify it.

## Per question

**Q1, REV1 findings: 9 FIXED, 2 NOT FIXED (F08, F11).**
- REV1's own H1 to H6, run from REV1's stand-in text, gave 20 runs, all rejected.
- Re-run with A2's bindings added, H2, H2c, H3, H3b and H4 were each rejected by the rule they target.
- H5 through the real runner: the no-window refusal was not admitted, and the retry was admitted and selected.
- New cases, all rejected:
  - a device fault in teardown, or inside `forward`;
  - a dropped census ID;
  - an unregistered skip;
  - a failing V3;
  - a log/exit mismatch;
  - an extra baseline failure;
  - exit 0 together with a failure.
- A valid run of all 45 labels passes the scorer.

**Q2, D5 validation: UNSOUND** (F05, F06).
- Every refusal REV1 listed works.
- All 75 configurations A1 accepted are still accepted, with bit-identical outputs and gradients.

**Q3, the kill rule and the device veto: UNSOUND** (F02, F03).
- The device veto held on every case.
- The engine census shows every CUDA error message on this path carries a veto pattern.
- For the 444 regression IDs, JUnit and the summary agree on every ID.

**Q4, the scorer: UNSOUND** (F01, F04, F07).
- These were all refused:
  - forged parent-only receipts;
  - a run root outside `a2/`;
  - tampering with START, the seal, a stored receipt or an expected exit;
  - a census mismatch;
  - a log/exit mismatch.
- A fully self-consistent forgery cannot be refused by any unsigned scheme. That is a limit, not a finding.

**Q5, mutants: UNSOUND, NIT level** (F08, F09).
- All 41 mutants are single-site, and 40 of 41 witnesses reach their edit during the call.
- Every witness fails on its mutant and passes on the candidate, as stand-in evidence.
- V3 matches the candidate on all 432 engine dtype combinations.

**Q6, arms and regeneration: SOUND.** All 43 regenerated arms equal A2's on disk, the seal and `arm_maps.json`. There
are no symlinks, and every engine copy matches the pin.

**Q7: REJECT.**

## Prior art

- Fault injection and test stubs: Meszaros (2007).
- Mutation analysis: DeMillo, Lipton and Sayward (1978); the reachability-infection-propagation (RIP) model of Offutt
  and Voas. Unverified; lead to check.
- Boundary-value testing: Myers (1979).
- SHA-256: FIPS 180-4.
- PyTorch 2.11's GroupNorm behaviour was observed live.

## Files

In `artifacts/gn1/rev2/`:
- `findings.json` and `REDACTION.md`;
- `regen_compare.*`, `regen_maps_*.json`;
- `mirror_setup.*`;
- `hostile_rev2.py`, `rev2_forge.py`, `hostile_rev2_{a,b,c,d}.{json,log}`;
- `d5_probe.*`, `q3_probe.*`, `q5_mutant_trace.*`, `q5_uncovered_probe.*`, `junit_real_probe.*`.

Scratch is left under `/tmp/gn1_rev2_*`.
