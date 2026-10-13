# P1T-REV: blind cross-model review of TensorCUDA DLPack interop (core) and the `tc_signal` add-on

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev`.
Edits there are AUTHORIZED, but only under `artifacts/p1t_rev/`, `logs/` and `docs/P1T_REV_*.md`. You may also write
scratch under `/tmp/p1t_rev_*`. This is a REVIEW: do not modify any source, test or build file anywhere.

## Why

Project-Voice needs six generic ops from TensorCUDA to port Kokoro-82M.

David, 2026-10-12, verbatim: "I don't want to build it directly INTO the core of TensorCUDA - It needs to be separate I
guess. I don't want to bloat the kernels if we don't have too."

So the work was split:
- **Core TensorCUDA** gained only DLPack interop: export, import, `data_ptr`, an in-place `fill_` using the existing
  fill kernel, `mark_modified`, and a device-side `empty`. It has no new kernels.
- **A separate add-on package, `tc_signal/`**, has its own `.so`, consumes operands through DLPack, and does not link
  core. It holds conv1d, conv_transpose1d, an ONNX LSTM, instance/layer/group norm, nearest upsample, reflection pad
  and rint.

Both are infra code headed for Project-Tensor production main. House rule §10 requires a cross-model review receipt
before promotion. You are that review. Be adversarial: you are looking for reasons NOT to promote.

## Mechanism & scope

- **Measures:** whether the diff `9df5bef..00590b7`, restricted to `tensor_cuda/` and `tc_signal/`, is correct, memory-safe, free of regressions to existing
  core behaviour, faithful to its spec, and safe to promote into production.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/docs/P1T_TENSOR_ALT_PLAN.md`, the approved
  design. Also:
  - the spec `/mnt/ForgeRealm/Project-Voice/research/kernels/TENSOR_IMPL_PLAN_P1T.md` §4–§5;
  - the contract `/mnt/ForgeRealm/Project-Voice/engine/tensor_requests/p1t_tests/adapter_api.py`;
  - the four seat orders `orders/P1T_M1_DLPACK.md`, `orders/P1T_M1_AMENDMENT_1.md`, `orders/P1T_M2_TC_SIGNAL_CONV.md`,
    `orders/P1T_M3_TC_SIGNAL_LSTM.md`, `orders/P1T_M4_TC_SIGNAL_NORMS_SMALL.md` (in this worktree).
- **Exercises:** reading the code under review, which is the 42 files in
  `git diff 9df5bef --stat -- tensor_cuda tc_signal`:
  - core: `tensor_cuda/include/tc/core.h`, `tensor_cuda/src/kernels.cu`, `tensor_cuda/src/bindings.cpp`,
    `tensor_cuda/tensor_cuda/__init__.py`, `tensor_cuda/include/dlpack/dlpack.h`, `tensor_cuda/tests/test_dlpack.py`;
  - add-on: everything under `tc_signal/`.

  You may also run CPU-only checks: compile, `nm`/`ldd`/`objdump`, small host-side scripts, and fp64 reasoning.
- **Does not cover:**
  - re-deriving Voice's tests or tolerances (they were blind-reviewed separately);
  - performance tuning;
  - the seats' receipts beyond spot checks;
  - GPU execution. Your sandbox has no GPU. When a claim needs a GPU, write a probe script under
    `artifacts/p1t_rev/probes/`; the lead will run it.
- **Required evidence:**
  - every finding with file:line, severity (BLOCKING / MAJOR / MINOR / NIT), a concrete failure scenario, and a
    suggested fix;
  - for each question R1–R6 below, a verdict with the lines you read;
  - GPU probe scripts for any claim you cannot settle on the CPU.
- **Operand census:** the edge cases to reason about:
  - zero-size and 0-d tensors;
  - non-zero `byte_offset`;
  - size-1 dimensions with odd strides;
  - read-only and versioned capsules;
  - capsules consumed twice;
  - producers on non-legacy streams;
  - interpreter shutdown with live imported tensors;
  - exceptions thrown half-way through import;
  - shapes whose element counts exceed 2^31 (int32 casts into cuBLAS m/n/k and lda);
  - huge T for LSTM;
  - L=1 norms;
  - NaN/Inf inputs;
  - groups not dividing C;
  - output aliasing an input.
- **Baseline isolation:**
  - The baseline is production main `9df5bef`.
  - Core's existing test suite was run by the lead: the FAILED/ERROR set equals the 52 known baseline ids, with no
    new failures.
  - Voice's runner gives PASS 236/236 on T1–T6.
  - Treat these as claims to probe, not as facts.
- **Forbidden configuration:**
  - editing anything outside `artifacts/p1t_rev/`, `logs/` and `docs/P1T_REV_*.md`;
  - running GPU code;
  - network access beyond what your sandbox allows;
  - git writes. Read-only git (`git diff`, `git log`, `git show`) is ALLOWED for this review.

## Questions

- **R1, core regression risk.** Does any changed core line alter existing behaviour?
  - `Storage` gains fields and a constructor; its destructor gains a branch.
  - New bindings were added.
  - `__init__.py` was edited; one line was deleted.

  Check ABI and layout assumptions, the pooled versus raw free paths, `revision` semantics, and import-time side
  effects. Verdict: SAFE / RISK.
- **R2, DLPack correctness and memory safety.** Check:
  - export capsule lifetime;
  - deleter rules: an unconsumed capsule calls the deleter, and a consumed one is renamed `used_*`;
  - versioned (≥ 1.0) handling;
  - the read-only flag;
  - the stream contract (event record plus wait; `-1`; `None`/`1`/`2`; `0` refused);
  - import keepalive;
  - drain-before-deleter, and GIL acquisition in the deleter, including from non-Python threads and during
    interpreter finalization;
  - double-free and use-after-free paths, and leaks on exception paths.

  Verdict: SOUND / UNSOUND.
- **R3, the `tc_signal` operand contract.** Check:
  - validation completeness (device, dtype, contiguity, `byte_offset`, shape, overlap/aliasing);
  - capsule consumption and release after launch;
  - the cross-stream lifetime caveat;
  - allocator behaviour;
  - "C++ allocates nothing";
  - the cuBLAS handle configuration: default math, atomics not allowed, the stream set to legacy. M3 has its own
    handle; is that safe?
  - two handles in one process.

  Verdict: SOUND / UNSOUND.
- **R4, kernel and numerics correctness.** For conv1d (im2col and the SGEMM mapping), conv_transpose1d (gather), the
  LSTM (gate order, the reverse walk, state in place, the per-step GEMM-then-cell ordering), the norms (shifted
  two-pass, the block tree, L=1, eps) and the small ops (reflect index math, floor upsample, `rintf`), check:
  - index math;
  - int32/int64 overflow;
  - out-of-bounds reads and writes;
  - races (in-place state, shared memory);
  - determinism;
  - the absence of atomics, fast-math and intrinsic transcendentals;
  - agreement with the ONNX semantics.

  Verdict: CORRECT / DEFECTS.
- **R5, decoupling and build.**
  - `tc_signal` must not link, include or depend on core.
  - It must share the CUDA runtime: `ldd`, RUNPATH, no static cudart.
  - The build must be reproducible offline (`PYBIND11_SOURCE_DIR`).
  - The `dlpack.h` copies must be legitimate: core's is upstream v1.3, while `tc_signal`'s is torch's variant. Is the
    difference harmful?

  Verdict: PASS / FAIL.
- **R6, overall.** One of APPROVE / APPROVE-WITH-NITS / REJECT, for promotion into production main, with the blocking
  findings listed.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO subagents. No background waits. Run everything synchronously to completion.
- **No deletes, anywhere.** Agents never delete files; you are not covered by Claude's hooks, so this is on you. Never
  run `rm`, `git clean`, or anything like them.
- Never kill a process you did not start.
- Everything outside your writable paths is READ-ONLY. That includes production `/mnt/ForgeRealm/Project-Tensor`
  (outside `Fork/`), `/mnt/ForgeRealm/Project-Voice`, and every other fork or worktree.
- Be blind to the authors' conclusions. Read the code first and form your own view, and only then read the seats'
  reports (`docs/P1T_M{1..4}_REPORT.md`) to cross-check.
- RED honesty: if you cannot settle something, say so and write the probe. Never guess a verdict.
- Prior Art Directive (AGENTS.md): when a finding rests on a known pitfall or standard, name the source: the DLPack
  spec, the array API standard, CUDA stream semantics, the cuBLAS docs, the ONNX operator specs.

## Done

- Write `docs/P1T_REV_REPORT.md`: R1–R6 verdicts, the findings table (id, severity, file:line, scenario, fix) and the
  probe list, plus `artifacts/p1t_rev/findings.json`, a machine-readable copy of the findings.
- Your final message must contain, verbatim:
  - the R1–R6 verdict table;
  - every BLOCKING and MAJOR finding in full;
  - the count of MINOR and NIT findings;
  - the probe script paths, with what each one decides.
