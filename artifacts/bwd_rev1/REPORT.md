# BWD-REV1: blind cross-model review of BWD-R1 (Claude Opus 5.5; author Codex)

**Verdict: 3 MAJOR, 8 MINOR, 5 NIT, 0 BLOCKER.** In the cases I exercised, the hook's runtime behaviour matches the registered design:
- validation runs before accumulation in every slot;
- the callback is released on backward, on abandonment and on an exception;
- grad mode is off inside the callback;
- re-entrancy is refused.

The strict matmul backward is coded as specified. The MAJORs concern a test file that breaks plain collection (F01), a no-TF32 gate that cannot see a TF32 compute type at the call site (F02), and stale non-leaf gradients after a failed traversal, which the design's recovery rule does not clear (F03).

- **Diff reviewed:** `git diff d5c55d6c 4d844a05 -- tensor_cuda`: 8 files, +772/-3.
- **Candidate module:** sha256 `78c2b99be540e6a925e708141090ab724bd1056f27fd469d175d216d1e0d0613`, recomputed from a real copy at `/tmp/bwd_rev1_pkg2`.
- **CPU only:** `CUDA_VISIBLE_DEVICES` was empty for every command.

## Seals

| File | sha256 | Date line (same command) |
|---|---|---|
| `findings.v1.json` (blind) | `70f0e7ac371af4ce94862ecc98c13ca053d8f8af9d00796763a090d5fb69f2dc` | `2026-11-18T16:37:58-05:00` |
| `findings.json` (final) | `a8a267d29d48beaa5f85fd9320c010ab4a65d042b405b04cbcc85d3b4554a696` | `2026-11-18T16:42:44-05:00` |

v1 was written before I opened any BWD-R1 report or receipt. In either worktree, I had only listed file names to locate the candidate `.so`. The final version adds receipt evidence to F01, F02, F05, F07 and F08, and adds F14-F16. Nothing was withdrawn or downgraded.

## Findings

Full text, arguments and fixes are in `findings.json`.

| ID | Sev | Where | Finding |
|---|---|---|---|
| F01 | MAJOR | test_bwd_r1_core.py:9, test_bwd_r1_matmul.py:9 | They import `bwd_r1_oracle`, which exists only in `scripts/`. A plain pytest run stops at collection (exit 2). |
| F02 | MAJOR | ops.cpp:707-725; test_bwd_r1_matmul.py:34 | The compute and algo fields of the configuration check are constants compared with themselves. The canary runs under a mandatory `NVIDIA_TF32_OVERRIDE=0`. A TF32 compute type written at the `cublasGemmEx` call survives. r1m04 was killed only by its edited constant. |
| F03 | MAJOR | autograd.cpp:156-198 | A failed traversal leaves partial grads on non-leaf nodes. The next backward through them adds them (C1/C1b/C1c). |
| F04 | MINOR | autograd.cpp:121-123 | Re-backward through a consumed hook fails mid-walk, after leaf accumulation (C2). |
| F05 | MINOR | test_bwd_r1_hook.py:85-116,143-161 | Faults are injected only in slot 1. No test mutates a saved tensor during the callback, and r1m03 removes both revision checks together. |
| F06 | MINOR | autograd.cpp:14-34 | The single lane is not enforced. A second thread sees grad mode off and silently records no graph, and its backward is refused as "re-entrant" (C6). |
| F07 | MINOR | test_bwd_r1_matmul.py:59-70 | The determinism witness has K=7. The split-K-prone Kokoro cell (K=1028, M=1, N=128) is never rerun. |
| F08 | MINOR | DLPack import; tc_signal `out` | Engine writes through a DLPack re-import bump a different revision counter. tc_signal does not bump `out`. The write-and-mark-before-attach contract is undocumented. |
| F09 | MINOR | bindings.cpp:2639-2643 | A callback that captures the result forms a cycle Python's GC cannot see: a permanent leak (C5). |
| F10 | NIT | autograd.cpp:104-106 | A partial byte overlap between distinct Storages is not refused. |
| F11 | NIT | ops.cpp:685-716,774 | Round-named env var read per GEMM, per-batch getters, an unlocked static map, and teardown at exit. |
| F12 | NIT | ops.cpp:785-788 | The per-batch heuristic pick can differ with alignment; a strided-batched pedantic call would be cleaner. |
| F13 | NIT | ops.cpp:744-772 | The fp16/bf16 fallback, the batch-mismatch branch, the current-device branch and the K==0 branch are untested. |
| F14 | MINOR | r1 packages/*.so | 7 symlinks, against the standing law. The PROD pin resolves into mainchk2's tree. |
| F15 | NIT | ops.cpp:757-762 | The forward's silent stride-0 for rank mismatch is mirrored rather than refused. |
| F16 | MINOR | test_bwd_r1_matmul.py:34 | `os.environ["NVIDIA_TF32_OVERRIDE"]` raises KeyError on GPU runs without it. |

## Q1: the hook's correctness and safety

Read: `autograd.cpp` 1-202 and `autograd.h` at HEAD, `bindings.cpp:120-183,729-770,1808-1884,2632-2670`, `core.h:50-135`, `kernels.cu:1364` (`reduce_to` clones) and the in-place writers (`kernels.cu:882,1630-1692,5395-5429,8053-8088`). Ran: `checks/rev1_host_checks.py` (log `checks/checks_run3_final.log`).

- **Validation.** Shape, dtype, device and count are checked for every slot, and the frozen-slot rule is enforced, all before any `accumulate_grad` (autograd.cpp:134-147). The binding rejects a non-tuple return and non-Tensor items. C3 shows slot 0 and slot 2 faults are caught, with no grad defined anywhere. Clone-on-accumulate means a returned gradient never aliases `dy`, an input or another leaf. Gap: the tests (F05).
- **Revision checks.**
  - **Caught:** engine in-place ops (`fill_`, optimizer steps, `scale_`, `axpy_` and `write_rows` all call `mark_modified`); `mark_modified()`; mutation during the callback (C4).
  - **Not caught:** writes through a separate Storage aliasing the same memory, such as `tc.from_dlpack(x)` of an engine tensor or an unreported external write; APA `packed_append` (F08).
  - Voice's write-then-mark pattern is covered, as long as it happens before attach.
- **Lifetime.**
  - **On backward:** `std::exchange` moves the callback into the invocation, so it is released at the end of the call or on unwinding.
  - **On graph destruction:** state is destroyed, and the deleter takes the GIL.
  - **On an exception:** released during unwinding.
  - Tested for all three.
  - **Exception:** a result-capturing callback leaks permanently (C5, F09).
  - **GIL and locks:** Python objects are created and destroyed under `gil_scoped_acquire`. The only engine mutex (the DLPack import lifecycle) is never held across Python. The backward binding holds the GIL throughout.
- **Concurrency.**
  - **Re-entrancy:** refused.
  - **Recursion:** `attach_vjp` inside a callback returns `value` under no-grad.
  - **Autograd inside a callback:** with grad re-enabled it builds a throwaway graph; NoGradGuard restores the mode.
  - **Deadlocks:** none found.
  - **State left behind on a failed traversal:**
    - stale non-leaf grads (F03, C1);
    - a partial leaf update when re-backward reaches a consumed hook (F04, C2).
  - **Other threads:** the process-global mode and flag leak to other threads (F06, C6).
- **Storage reuse.**
  - **Safe for:**
    - fresh allocator outputs;
    - DLPack-imported values (`test_gpu_external_value_storage_lifetime`);
    - Voice's write-then-mark pattern, provided it happens before attach.
  - **Unsafe for:** a reused caller-provided `out=`, because tc_signal does not mark `out`. R2 must refuse it, as DESIGN §4 says (F08).

## Q2: the strict matmul backward

Read: `ops.cpp:651-805`, `matmul.cu`, `kernels.cu:1409`, `eb3.cuh:66-70`. GPU logs, sha recomputed: G4-strict `0fd3254c…`, G5-m04 `6a931cd9…`, G5-m05 `1d01535e…`, G4-forward R2 `92e1d94c…`.

- **Handle per device, thread safety.**
  - Keyed by `cudaGetDevice`, created lazily, retried after a failed init.
  - Operands must be on the current device.
  - No lock: safe only because backward holds the GIL on one lane (F11).
- **Workspace.**
  - 32 MiB from `cudaMalloc`, installed once, after `cublasSetStream(cudaStreamLegacy)`.
  - Never re-set; static lifetime.
  - Legacy stream, consistent with the engine's stream-0 kernels; per-thread-stream builds are refused elsewhere.
- **TF32 or non-deterministic fallback.**
  - As coded, no: `COMPUTE_32F_PEDANTIC`, `PEDANTIC_MATH`, atomics off, a fixed workspace and one stream. cuBLAS's reproducibility conditions are unverified; lead to check.
  - The gate cannot detect a TF32 regression at the call (F02).
  - The determinism witness is too small to exercise split-K (F07).
- **Forward byte-identical.**
  - The diff leaves `matmul.cu` and the forward line unchanged.
  - G4-forward recounted: 72 cases per arm, 72/72 equal sha256, candidate `78c2b99b` against PROD `a2f6a326`.
  - PROD's 46 pinned source sha256s equal `git show 3138cf54` (46/46). The candidate's equal `4d844a05`.
- **Every cuBLAS error checked.** Yes. The destructor ignores errors by design. The forward's unchecked `handle()` and `SgemmStridedBatched` are pre-existing and outside the diff.
- **Semantics.** The dA/dB forms and the leading dimensions are correct (`ldb = k` with `OP_T`, `n` with `OP_N`). K==0 gives zeros. Rank mismatch is mirrored (F15).

## Q3: tests and gates

Plain collection: exit 2. Collection with `scripts/` on the path: 19 passed, 58 skipped. Recounted: 77 IDs in 25 functions (core 49, matmul 2, hook 26 = 19 host + 7 GPU).

**Per-test reach census:**

| Test (IDs) | Reaches |
|---|---|
| hook `test_host_single_inverse_reshape` (1) | binding; `attach_vjp` 90-115; grad_fn 116-148 (guard, NoGrad, exchange, both revision checks, validation, accumulate); asserts grad off. r1m01 killer |
| `test_host_ordered_active_slots` (1) | two active slots, accumulate order. r1m02 killer |
| `test_host_frozen_saved_slot_and_structural_zero` (1) | `active=false`; None for an active slot |
| `test_host_bypass_identity_and_no_retain` (2) | early return at 111; binding's `std::move(value)`; release |
| `test_host_validation_is_before_accumulation` (6) | count 134-135; frozen/shape/dtype in **slot 1 only**; binding tuple and element errors |
| `test_host_value_must_be_detached_and_separate` (1) | 94-96; 104 (same Storage). Not the data_ptr branch |
| `test_host_callback_exception_restores_mode` (1) | Python exception; NoGradGuard restore |
| `test_host_revision_detects_alias_mutation` (2) | first `check_revisions`, frozen slot 1 and value. r1m03 killer |
| `test_host_callable_lifetime` (3) | exchange; "already consumed"; deleter on backward, abandonment and exception |
| `test_host_callback_reentrancy_refused` (1) | `refuse_nested_backward`; flag reset |
| `test_gpu_seed_fanin_and_repeated_slots` (1) | repeated slot, fan-in add at :57 |
| `test_gpu_wrong_device_is_atomic` (1) | device mismatch (slot 1); input device check 101-103 |
| `test_gpu_actual_write_rejected` (2) | active slot 0 revision through `fill_`, and DLPack plus a manual `mark_modified` |
| `test_gpu_external_value_storage_lifetime` (2) | external Storage reuse and release |
| `test_gpu_checkpoint_replay_through_hook` (1) | checkpoint bypass; re-attach on replay; nested backward |
| matmul `test_gpu_canary_and_configuration` (1) | `matmul_backward` 2D, both directions; `initialize`/`run`; config log. m04/m05 killer |
| matmul `test_gpu_matmul_repeat_five` (1) | the same 2 GEMMs ×5 (5×7×8) |
| core: 8 matmul IDs | `matmul_backward`: plain, Kokoro trans_b, trans, batch_shared(±trans), batched(±trans), exact taps |
| core: 41 other IDs | pre-existing VJPs only |

- **A leaking callback?** Killed: weakref checks on all three exit paths, plus "already consumed".
- **Skipped validation for one slot?** Slot 1 is caught; a slot-0 or slot-≥2 skip survives (F05).
- **Grad recording on?** Killed: the callbacks assert `not tc.is_grad_enabled()`.
- **Mutant coverage.**
  - **Covered:** m01-m05.
  - **Missed:**
    - TF32 at the call site (F02);
    - a one-slot validation skip, or the post-callback revision check alone (F05);
    - DESIGN §6's "write a frozen leaf" and "leak grad mode on exception" (the tests would kill them, but they never ran as mutants);
    - §6's matmul rows (transB flip, dW swap, dropped shared-B batch reduction), not registered by the R1 order.
- **Recounted (L44):**
  - G6: 3,941 IDs; 3,864 equal; 77 new (19 passed, 58 skipped); `unexpected` = {}.
  - G5 CPU: 3/3 killed.
  - GPU: G2 7 passed; G4-strict 2 passed with 5 identical sha pairs; m04 and m05 failed on the configuration mismatch.

## Q4: anything else

- F14: the R1 seat's symlinks.
- F16: the canary test's KeyError.
- F11: a round-named debug env var in product code.

## Checks run

```
L01: git --version -> 2.43.0; python3 --version -> 3.12.3; pytest 9.0.2
cp -rL <r1>/artifacts/bwd_r1/packages/candidate/tensor_cuda /tmp/bwd_rev1_pkg2/   (sha 78c2b99b..., 0 symlinks)
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=/tmp/bwd_rev1_pkg2 timeout 300 nice -n 19 python3 -B -P checks/rev1_host_checks.py -> checks/checks_run3_final.log
CUDA_VISIBLE_DEVICES= PYTHONPATH=/tmp/bwd_rev1_pkg2 timeout 600 nice -n 19 python3 -m pytest -p no:cacheprovider -q <3 files> -> checks/pytest_plain.log (exit 2)
  + PYTHONPATH+=<wt>/scripts -> checks/pytest_with_scripts.log (19 passed, 58 skipped, exit 0)
```

checks_run3_final.log:
```
CHECK C1a OBSERVED first traversal raised 'k fails'; intermediate h.grad after failure = [1.0, 1.0]; x.grad=None; w.grad=None
CHECK C1b OBSERVED second traversal reached the stale-grad ADD path (autograd.cpp:57 ew_binary): CUDA error at ew_binary: no CUDA-capable device is detected
CHECK C1c CONTROL fresh h: x.grad=[1.0, 1.0] (no ADD path, no CUDA call)
CHECK C2 OBSERVED raised 'attach_vjp(P): callback already consumed'; w.grad=[1.0, 1.0] x.grad=None; p.grad (op node) = [1.0, 1.0]
CHECK C3.slot0 CAUGHT attach_vjp(custom): gradient slot 0: shape mismatch; grads=[None, None, None]
CHECK C3.slot2 CAUGHT attach_vjp(custom): gradient slot 2: shape mismatch; grads=[None, None, None]
CHECK C4 CAUGHT attach_vjp(custom): saved input slot 1 was modified after attachment; x.grad=None
CHECK C5 LEAK callback captured the result (docstring forbids it); after del + gc.collect() weakref alive=True
CHECK C6 OBSERVED second thread during a callback: {'grad_enabled': False, 'fresh_requires_grad': False, 'other_backward': 'attach_vjp: re-entrant backward is not supported'}
CHECK C7 N/A host DLPack import unavailable (BufferError: from_dlpack: only CUDA device memory (kDLCUDA = 2) can be imported; the producer reports device (1, 0) (CPU)); reasoning only
DONE
```

## Deviations

- **Symlink in scratch:** my first scratch `cp -r` created a symlink at `/tmp/bwd_rev1_pkg/tensor_cuda/_tensor_cuda…so`. The delete guard refused the `rm`, so it is left for the OS. All results were re-run from the real copy, with identical output.
- **Short reads without a timeout:** a few listing and grep commands ran without `timeout`.
- **Check script extended once:** I added C1c; 0 repairs.

## Prior art

- **The hook as reviewed:** PyTorch `autograd.Function`, `save_for_backward` and the saved-tensor version counter (Paszke et al., 2019), and JAX `custom_vjp` (Bradbury et al., 2018). F08's aliasing gap and F09's output-capture cycle are the known PyTorch classes.
- **F03/F04 fix:** strong exception safety (Abrahams, 1998). The engine cleanup is mine.
- **F02:** NVIDIA `NVIDIA_TF32_OVERRIDE` (2020). Unverified for an explicit `FAST_TF32` compute type; search "NVIDIA_TF32_OVERRIDE cublasGemmEx CUBLAS_COMPUTE_32F_FAST_TF32".
- **F07:** cuBLAS 12.x reproducibility, split-K. Unverified; search "cuBLAS results reproducibility split-K workspace".
- **F06:** PyTorch thread-local `GradMode`, suggested only.
- **Mutation-gap analysis:** DeMillo, Lipton and Sayward (1978).
- **Reach census and checks C1-C7:** no prior art known to me.

All of these are unverified (no network in this seat); lead to check.

## Files written

- artifacts/bwd_rev1/: findings.v1.json, findings.v1.sha256, findings.json, findings.sha256, REPORT.md (this text; the tool refused the write)
- artifacts/bwd_rev1/checks/: rev1_host_checks.py, checks_run1.log, checks_run2_realcopy.log, checks_run3_final.log, pytest_plain.log, pytest_with_scripts.log
- Scratch: /tmp/bwd_rev1_scratch/, /tmp/bwd_rev1_pkg/ (stray symlink), /tmp/bwd_rev1_pkg2/
