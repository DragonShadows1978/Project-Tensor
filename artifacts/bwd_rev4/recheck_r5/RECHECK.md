# BWD-REV4 re-check of BWD-R4a r5b and r5c

**Verdict: 1 new MAJOR, 1 MINOR, 8 NIT.**

- **The MAJOR is in the F03 fix.** The checkpoint guard records input revisions *after* `fn` returns. A checkpointed function that writes its own input in place therefore yields a silently wrong gradient, and the replay writes the caller's tensor a second time. I demonstrated this on CPU on the shipped `packages_r5` candidate.
- **Every other original finding has a correct fix, test or note.** Some carry the NITs below.

**recheck.json** sha256 `180279ecda02647681fee03f367ec809ce727423baba12ebdd69a02689f6123d`. Date line from the same command: `Thu Nov 19 07:40:06 AM EST 2026` (in `recheck.seal`). The write of `RECHECK.md` was not attempted, because this session's tools refused my earlier report-file write; its content is this message.

Correction of my own earlier report: my BWD-REV4 final message said 4 MAJOR / 6 MINOR / 4 NIT. `findings.json` final (`08d56016…`) has 3 / 6 / 5, as the lead noted. The JSON is right.

## What I read and ran

**Read:**
- `git diff 541faa2d a515439d -- tensor_cuda tc_signal scripts`, all of it.
- The three orders.
- REPORT_R5, REPORT_R5C, both findings maps and DESIGN_NOTES_R4A_R5.
- AMENDMENT2_R5 and AMENDMENT3_R5C. Their sha256s match their seals: `4f4c68ac…` and `b2605789…`.
- MUTANT_R5M01_R5.json.
- In core: `core.h:124-127` and `bindings.cpp:2725-2732`.
- The existing checkpoint tests.

**Ran:** CPU only, CUDA hidden, `nice -n 19`, `timeout` of 300 s or less.
- **`checks/r5_checkpoint_guard_probe.py`**, on the candidate and on the r5m01 package. Exit code 0 for both.
- **`checks/r5_affine_test_power.py`**, which uses the new test module's own operands, reference and bound. Exit code 0.
  - After its JSON, the output has one tc_signal exit message saying no CUDA device was found. That is expected on a process with no GPU.

## New findings (BWDR5R)

- **F01, MAJOR.** `tensor_cuda/src/bindings.cpp:148-159` (comment at 151); docstring at `tensor_cuda/__init__.py:1462-1469`.
  - **What happens:** the revisions are recorded after `fn` returns. In probe case B, `fn` reads a=[1,2,3], then sets a[0]+=1 with `mark_modified`.
  - **Result:** no error is raised, `b.grad` is [2,2,3] where the true value is [1,2,3], and `a` ends as [3,2,3] because the replay repeats the write.
  - **The comment is wrong:** it says "fn's own in-place use of an input is part of the forward it saved". Checkpoint saves only the inputs, not their values before `fn` ran, so the replay cannot reproduce that forward.
  - **Nothing tests this:** a variant that records before `fn` would pass all six tests, and r5m01 does not cover it.
  - **Exposure:** the R4a and R4b harnesses cannot reach this path (their segment inputs are fresh op outputs, and F07's digest would flag it). It is still a silent-corruption path in the engine API.
  - **Fix:** take the revision snapshot before `fn`. If any revision changed by the time `fn` returns, raise at once, or refuse at backward. Correct the comment and docstring, add case B as a host test, and add a mutant that records after `fn`.
- **F02, MINOR.** Tensors captured by `fn`'s closure are outside the guard. In probe case C, `w` is captured and then changed after the forward. No error is raised, and `a.grad` is [40,-5,6] where the true value is [4,-5,6].
  - The R4a harness passes every parameter explicitly, and F12's census now refuses containers that could hide a tensor, so the harness is not exposed.
  - Fix: document that every tensor `fn` reads must be passed as an input.
- **F03, NIT.** The checkpoint docstring's list of undetected writes is narrower than attach_vjp's (`bindings.cpp:2729-2732`).
  - It omits writes through a DLPack re-import and APA `packed_append`.
  - It also omits tc_signal's own untracked `out=` paths, which write without `mark_modified` (`_norms.py:111-115`, the conv paths).
- **F04, NIT.** The zero/sign-flip power check (`bwd_r4_r5.py:275-309, 349-375`) cannot fail once the clean value passes.
  - With a bar of ρ·|fd| and ρ ≤ 1e-3, any fd ≠ 0 makes zeroing and flipping fail by algebra.
  - The bar's real power is ρ itself. Measured against r4's candidate errors (at most 1.06e-4 relative), that leaves about a 9× margin.
- **F05, NIT.** A3-1's floor (`bwd_r4_r5c.py:57-63`) never binds on the probes chosen. They are the three largest-|g| entries, so |fd| sits 6.8 to 10× above the floor on all 18 coordinate rows.
- **F06, NIT.** A3-2's triple choice (`bwd_r4_r5c.py:131-136, 152-158`) uses agreement with the fp64 VJP as one of its criteria, so on that row the finite difference is not fully independent of the VJP it checks.
  - It is sound here. The plateaus for T(2), T(1) and T(1/2) are 2.78e-4, 6.95e-5 and 1.74e-5: they fall 4× per halving, the signature of h² truncation.
  - The chosen finite-difference error, 1.85e-5, is 300× below the candidate's bar of ρ·|fd| = 5.6e-3.
- **F07, NIT.** The replay-identity probe (`bwd_r4_gpu.py:124-202`) digests segment outputs only. An input change that rounding absorbs before the output would pass.
  - Fix: also digest each segment's replay inputs against its forward inputs.
- **F08, NIT.** The suite-gpu rule that no GPU test may skip (`bwd_r4_r5.py:494`) matches only IDs containing `::test_gpu`.
  - Four `@gpu` tests in `test_bwd_r1_core.py` are named `test_core_*`, so a skip of any of them would not fail the line.
- **F09, NIT.** `g1_annotated` (`bwd_r4_r5.py:397-406`) still emits the old R4_G1 `worst_parameter` line, which picks a floor row (relative L2 7.9e8). The correct worst is in the new `G1_GROUPS` file.
- **F10, NIT.** The F08 design note calls G5's figure an upper bound on this process's memory use. It is an upper bound only at the sampled phases.

## Q1: each original finding

| Finding | Status | Where |
|---|---|---|
| F01 | Fixed. NITs F04-F06. | `bwd_r4_r5.py:145-309, 349-395`; `bwd_r4_r5c.py:57-173, 230-248`; both amendments |
| F02 | Fixed (staged). | `bwd_r4_gpu.py:337-404, 475-504`; `bwd_r4_r5.py:397-419`; `bwd_r4_stage_r5.py:49-73, 262, 307` (fine G4 has its own baseline) |
| F03 | Partly fixed. A changed input is refused: case A refuses on the candidate and passes silently on r5m01. Gaps: new F01 and F02. | `bindings.cpp:148-170`; `test_bwd_r4_checkpoint_revision.py` |
| F04 | Flagged per Voice's decision. Correct. | `bwd_r4_r5.py:47-142`; REPORT_R5 item 7 |
| F05 | Fixed (details below). | `test_bwd_r4_norm_affine.py:40-141` |
| F06 | Fixed and meaningful. | `test_bwd_r2_paths.py:70-81` |
| F07 | Fixed (staged). NIT F07. | `bwd_r4_gpu.py:124-202, 328-335`; `bwd_r4_r5.py:421-458` |
| F08 | Noted for R4b. NIT F10. | DESIGN_NOTES F08 |
| F09 | Fixed in the new tables. NIT F09. | `bwd_r4_r5.py:117-142` |
| F10 | Noted, correct. | DESIGN_NOTES |
| F11 | Ruled, correct. | DESIGN_NOTES |
| F12 | Fixed. | `bwd_r4_gpu.py:94-121` |
| F13 | Noted, not credited. | DESIGN_NOTES |
| F14 | Staged. NIT F08. | `bwd_r4_r5.py:460-505` |

**F05 detail:**
- The new test rejects all seven variants: dscale zero, dscale from batch 0 only, r4m03, dbias from batch 0 only, r2m03, dX ignoring the scale, and dX zero.
- The shipped rule passes with a worst ratio of 4.7e-4 of the bar.
- Residual: the scale-only and bias-only cases are still tested only at T=1.

**F06 detail:** the tracked `out=` ValueError is raised only by `_recording`, so it proves the tracked route. The `no_grad` `out=` call reaching the planner proves the untracked route.

## Q2: the F03 checkpoint revision check

**Is it correct and complete?** It is correct for writes after the forward that bump the revision. It is not complete: new F01 and F02.

**Is recording after `fn` right?** No (new F01).

**Is any other path missed?** Yes:
- closure-captured tensors (F02);
- the documented raw-write class, which includes tc_signal's own `out=` (F03).

`numpy()` is not a write path: it returns a copy (probe D).

**Does r5m01 test it?** r5m01 is caught by the three host refusal tests. Nothing tests when the revisions are recorded.

## Q3: the F01 probes

- **Is the ρ·|fd| bar non-vacuous?** Yes; the power checks themselves are tautological (F04).
- **Is A3-1's floor sound?** Yes, and it does not bind on the probes chosen (F05). It hides nothing:
  - it is a validity rule, not an acceptance bar;
  - the probes, seed and ρ are unchanged;
  - the fp64 check is still required;
  - no row was dropped;
  - registering it after the fp64-only data is a disclosed deviation with no candidate data involved.
- **Is A3-2's fp64-only step choice sound?** Yes (F06).

## Q4: replay identity

Per-segment output identity, with deterministic kernels and the input guard, is strong evidence of exact recomputation, but not proof (F07).

r5m02 is a valid mutant:
- x + (−0.0) is bitwise x for every x, so only the one edited element changes;
- the edit counts exactly one changed element;
- the line requires the mismatch to land on the edited segment, so an edit that does not propagate shows as a surviving mutant, never as a false kill.

## Q5: tests

- **Do new tests weaken old ones?** No. The renamed test keeps its original assertion and adds two. The affine and checkpoint tests are new or appended.
- **No existing checkpoint test mutates an input.**
- **Is the rename and `out=` witness meaningful?** Yes.

## Q6: introduced by the fix rounds

New F01 and F02 are in the core guard. F03 to F10 are NITs in the harness and notes.

## Files written

All under `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_bwd/WorkTree/tensor_fork_bwd_wt_rev4/artifacts/bwd_rev4/recheck_r5/`:
- recheck.json
- recheck.seal
- checks/r5_checkpoint_guard_probe.py
- checks/r5_checkpoint_guard_probe_candidate.out
- checks/r5_checkpoint_guard_probe_r5m01.out
- checks/r5_affine_test_power.py
- checks/r5_affine_test_power.out

Scratch: `/tmp/bwd_rev4_r5/`.

## Prior art

- **The checkpoint guard:** PyTorch saved-tensor version counters (Paszke et al., 2019). PyTorch's reentrant `torch.utils.checkpoint` also saves its inputs after the function runs, and has the same gap; I take the comparison from there. The case-B and case-C probes are mine.
- **Recomputation:** Chen et al. (2016).
- **Step choice:** Ridders (1982); Richardson extrapolation (1911); Higham (2002).
- **Mutation analysis:** DeMillo, Lipton and Sayward (1978).
- **The affine rule:** Ioffe and Szegedy (2015).
- **No prior art known to me** for the variant table used against the new affine test.

All unverified, lead to check: "torch.utils.checkpoint in-place inputs version counter", "Ridders 1982 numerical derivative", "DeMillo Lipton Sayward 1978".

_Filed by the lead from the reviewer's final message (its own write was refused)._
