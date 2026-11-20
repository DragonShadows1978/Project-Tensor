# BWD-REV4 confirm of BWD-R4a r5d (d7207e56)

r5d fixes or notes all ten re-check findings. The MAJOR (BWDR5R-F01) is fixed: on the rebuilt candidate, a checkpointed function that writes its own input is now refused in the forward.

The changes introduce no product defect. There is one harness defect: the staged suite-gpu line will fail by construction on the TF32 canary test (BWDR5D-F01, MINOR). There are also 3 NITs.

**confirm.json** sha256 `aea8aa479323eba6bd903980c654a98609f5bda8977dfdf9d5c2dae58e808974`. Date line from the same command: `Thu Nov 19 08:11:54 AM EST 2026` (in `confirm.seal`). It was re-sealed once, 8 s after the first seal, to correct two line citations; the first seal is superseded.

## What I read and ran

**Read:**
- the whole diff `a515439d..d7207e56` for `tensor_cuda`, `tc_signal` and `scripts`;
- the r5d order;
- REPORT_R5D, FINDINGS_MAP_R5D, DESIGN_NOTES_R4A_R5D, MUTANTS_CPU_R5D, MODULES_R5D;
- the case-B smoke log and the suite-gpu dry receipt;
- the fixture readers in the test support modules.

**Verified:**
- The candidate core module's sha256 is `430c829e…`.
- `MODULES_R5D` records `bindings.cpp` as `51e9c702…`, which equals `git show d7207e56:tensor_cuda/src/bindings.cpp`.

**Ran** (CPU only, CUDA hidden, `nice -n 19`, timeout 300 s or less):

1. **`checks/r5_checkpoint_guard_probe.py` (cases A–D):**
   - On the **candidate**:
     - A: refused after the forward.
     - B: refused in the forward. `fn` was called once and `a` = [2,2,3].
     - C: not refused; `a.grad` = [40,-5,6]. This is the documented limit.
     - D: `numpy()` returns a copy.
   - On **r5m03**: A is refused. B is accepted; `fn` was called twice, `b.grad` = [2,2,3] where the true value is [1,2,3], and `a` = [3,2,3]. So r5m03 reproduces r5b's defect, and the case-B test kills it.
2. **`checks/r5d_refusal_hygiene.py`** (one repair: case G's host backward needs a CUDA add, so it is recorded as not testable on CPU):
   - After a forward refusal: grad mode is restored, and the returned tensor's refcount is unchanged after 200 refusals.
   - No gradients are left behind, and a clean checkpoint afterwards gives exact gradients.
   - A `mark_modified()` with no value change is refused.
   - On the no-grad or no-active-input early return, an input write is not refused.
   - An exception raised by `fn` itself passes through unchanged, with grad mode restored.
3. **The F08 plugin under pytest** on `test_gpu_strided_canary` (`NVIDIA_TF32_OVERRIDE=0`, CUDA hidden): the canary is GPU-marked, and its skip is counted in `gpu_skips`.

## Rulings on BWDR5R-F01 to F10

| Finding | Ruling | Where / evidence |
|---|---|---|
| F01 (MAJOR) | **Fixed** | `bindings.cpp:153-182`: pre-`fn` snapshot, forward refusal at 169, backward check at 181. Docstring at `__init__.py:1455-1488`. New tests at `test_bwd_r4_checkpoint_revision.py:147-219`. Probes A, B and the hygiene checks above. |
| F02 | **Noted** | The contract is in the docstring, pinned by a test, and explained in the design note. Case C behaves as documented. |
| F03 | **Fixed** | The docstring's list of undetected writes (`__init__.py:1480-1486`) now matches attach_vjp's, plus tc_signal's `out=` paths. |
| F04, F05, F06, F10 | **Noted** | REPORT_R5D Notes; accurate. |
| F07 | **Fixed** | Input digests: `bwd_r4_gpu.py:129-132, 177, 197-199, 341-345`. They are now required by the replay-identity and r5m02 checks in `bwd_r4_r5.py`. See D-F02 and D-F03. |
| F08 | **Fixed in mechanism** | The fixture-root env names match their readers, the readers only read, and the marker record fails closed if missing. But see D-F01. |
| F09 | **Fixed** | `bwd_r4_r5.py:397-411` now reports the worst over the non-floor rows. |

## Your four focus points

- **Forward refusal path:** no defect.
  - The throw happens after the NoGradGuard scope has closed and before `obj.cast`, so the output is released.
  - No graph exists, because `fn` ran under no_grad, and no leak showed over 200 refusals.
  - The input keeps `fn`'s single write, which the new test asserts.
- **No-grad early return (`bindings.cpp:135-138`):** no defect. An input write is not refused there, but no replay exists on that path, so no wrong gradient is possible.
- **F07 input digests:** correct. They hash the inputs actually consumed, after any mutant edit. Two NITs: D-F02 and D-F03.
- **F08 marker plugin and fixture roots:** the roots are right, but the plugin produces D-F01.

## New findings (BWDR5D)

- **D-F01, MINOR.** The suite-gpu line will fail by construction on `test_gpu_strided_canary` (`test_bwd_r1_matmul.py:99-102`).
  - The test has the `@gpu` marker plus a second skip condition that fires whenever `NVIDIA_TF32_OVERRIDE` is set.
  - The staged line sets `NVIDIA_TF32_OVERRIDE=0`, and the pytest child inherits it.
  - The plugin (`bwd_r4_pytest_gpu_r5d.py:22-27, 40-47`) treats a test as GPU-gated if it carries any CUDA skip marker, regardless of which condition actually fired. So the TF32 skip counts as a GPU skip, and the line returns False.
  - BWD-R1 ran this canary on its own line with the override removed (`scripts/bwd_r1_stage_r5.py:147`). r5b's name-based rule had the same outcome.
  - **Fix:** classify a skip by the reason that actually fired. Either run the canary without the override, or declare it an expected skip on this line.
- **D-F02, NIT.** The per-segment digests run inside G5's timed window.
  - Each digest copies inputs and outputs to the host and hashes them, in the forward and in each replay.
  - G5's seconds are therefore not comparable with r4's, and any device copy taken for `.numpy()` lands inside the peak window. The G5 bar is report-only.
  - **Fix:** add a digest-off switch for G5, or disclose it.
- **D-F03, NIT.** r5m02 only counts as caught (`bwd_r4_r5.py:456-458`) if the output digest also mismatches.
  - If rounding absorbs the one-ulp edit before the output, the mutant would be reported as surviving even though the input check caught it.
  - **Fix:** score the kill on the input mismatch, and report the output mismatch as information.
- **D-F04, NIT.** DESIGN_NOTES_R4A_R5D says the segment operations read no tensor outside their arguments.
  - `resblk1d` reads `self.C(p['div'])`, a cached constant tensor, at Voice `model.py:771`.
  - That is harmless, since the constant is never written, but the sentence is inaccurate and should say so.

## Files written

All under `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_bwd/WorkTree/tensor_fork_bwd_wt_rev4/artifacts/bwd_rev4/confirm_r5d/`:
- `confirm.json`, `confirm.seal`
- `checks/r5_checkpoint_guard_probe.py`
- `checks/guard_probe_candidate.out`
- `checks/guard_probe_r5m03_checkpoint_revision_recorded_after_fn.out`
- `checks/r5d_refusal_hygiene.py`, `checks/r5d_refusal_hygiene.out`
- `checks/f08_plugin_tf32_canary.out`, `checks/f08_plugin_tf32_canary_markers.json`

Scratch is in `/tmp/bwd_rev4_r5d/`.

## Prior art

- **Revision guard:** PyTorch saved-tensor version counters (Paszke et al., 2019). PyTorch's reentrant `torch.utils.checkpoint` has the same input-only contract.
- **Gradient checkpointing:** Chen et al. (2016).
- **Leak witness:** CPython reference counting, used as the witness in probe E.
- **F08 plugin:** the pytest hook API.
- **Mutants:** mutation analysis (DeMillo, Lipton & Sayward, 1978).
- **Mine:** probes E–I and the TF32 classification argument. No prior art known to me for these.
- **Unverified, lead to check:**
  - "torch.utils.checkpoint reentrant inputs";
  - "pytest skipif multiple markers reason".

_Filed by the lead from the reviewer's final message._
