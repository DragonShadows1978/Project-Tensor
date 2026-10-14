# P1T-FIX2: close the round-2 review's robustness findings in core DLPack and `tc_signal`

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix2`
(branch `voice-fix2`: `voice-fix1` d116ac0 plus the round-2 receipts). Edits, core and `tc_signal` builds, and GPU
runs inside it are AUTHORIZED. You may also write scratch and mutant builds under `/tmp/p1t_fix2_*`.

## Why

David, 2026-10-12, verbatim: "If the Blind Review passes, I approve the merger."

**Round 1** REJECTED promotion. FIX1 (`docs/P1T_FIX1_REPORT.md`) closed its findings, and the lead verified that.

**Round 2** of the blind cross-model review (Codex gpt-6.1-sol high) did not finish. The provider's content filter
stopped the run before it wrote a verdict; see `logs/p1t_rev2_r1.log`, `docs/P1T_REV2_LEDGER.md` and
`artifacts/p1t_rev2/`. Before the stop, its CPU probes (`artifacts/p1t_rev2/probes/`, `host_checks.py`,
`core_cpu.cpp`, `cpu_native.cpp`) found four defects:

- **N1:** `tc_signal.synchronize()` returns success when `cudaGetDevice` fails. Log line ~9666: case `empty_sync`,
  "cudaGetDevice= 100 device= -1 / synchronize returned success".
- **N2:** zero-extent core shapes reach signed integer overflow. The checked element count is 0, but stride and byte
  products overflow. Log ~9668–9675: UBSan reports `4 * 4611686018427387904` for the `prefix` and `suffix` cases.
- **N3:** a host allocation failure (`std::bad_alloc`) inside the new release/drain path calls `std::terminate`. Log
  ~9676: case `oom_sync`, rc −6.
- **N4:** a producer deleter that throws a C++ exception during release calls `std::terminate`. Log ~9679: case
  `throw_deleter`, rc −6.

The reviewer also noted, before any classification:
- **N5:** "the finalization check still precedes GIL acquisition without lifecycle coordination". This is FIX1's own
  residual: a check-then-act window in which a foreign thread passes the finalization check just before finalization
  begins.

No verdict means no pass, so there is no merge. Fix N1–N5 and prove each fix. A round-3 review follows.

## Mechanism & scope

- **Measures:**
  - whether N1–N5 are closed, shown by the round-2 CPU probes and host checks reproducing BEFORE (on d116ac0) and
    passing AFTER, plus a new regression test per item that fails on its mutant;
  - that nothing regresses: the round-1 probes, Voice's runner T1–T6, the `tc_signal` tests, `test_dlpack.py`, and
    core's suite (52 baseline ids), run together in one pytest session.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix2/docs/P1T_REV2_LEDGER.md`
  (the reviewer's notes). Also:
  - `artifacts/p1t_rev2/REGISTRATION.md` and `artifacts/p1t_rev2/probes/*` (the probe sources and how they were
    built: see the log around their compile commands);
  - `logs/p1t_rev2_r1.log` (receipts);
  - the round-1 report `docs/P1T_REV_REPORT.md` and FIX1's report `docs/P1T_FIX1_REPORT.md`.
- **Exercises:**
  - core: `tensor_cuda/src/bindings.cpp`, `tensor_cuda/src/kernels.cu` (the Storage external path and its strides
    helper only, where needed), `tensor_cuda/include/tc/core.h`, `tensor_cuda/tensor_cuda/__init__.py`,
    `tensor_cuda/tests/test_dlpack.py`;
  - add-on: `tc_signal/src/*`, `tc_signal/tc_signal/*`, `tc_signal/tests/*`, `tc_signal/README.md`;
  - the round-1 and round-2 probes, run read-only.
- **Does not cover:**
  - new features;
  - performance work;
  - core code outside the DLPack/empty/fill_/mark_modified surface, the Storage external path and the shape/stride
    helpers those use;
  - Voice's tests (READ-ONLY);
  - multi-GPU execution.
- **Required evidence:**
  - a per-item table for N1–N5 (fix with file:line, proving test, probe before and after);
  - the full regression logs;
  - the mutant table;
  - the diff stat against d116ac0;
  - both `.so` sha256s;
  - the N5 design note.
- **Operand census:**
  - **N2:** shapes with a 0 in the first, middle or last position next to extents up to 2^62 and 2^63−1, for core
    `empty`, export, import and every `tc_signal` plan.
  - **N3:** allocation failure at each allocation site on the release path: event pool, pending deque, owner vector,
    error-message building.
  - **N4:** throwing deleters on the immediate, deferred, bound-drain, fallback-sync, `synchronize()` and atexit paths.
  - **N5:** a foreign thread releasing during and after the atexit phase, under a bounded wait.
- **Baseline isolation:**
  - "Before" means the d116ac0 build. Prebuilt copies are in
    `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix1`; use them read-only.
  - "After" means your rebuilt core and `tc_signal`, in fresh processes, under the GPU lock.
  - Core's suite is compared with the 52-id baseline log named below.
  - Mutants are built outside the worktree.
- **Forbidden configuration:**
  - editing the reviewer's probes, Voice's tests or the round-1 probes;
  - calling deleters from CUDA host callbacks;
  - device-wide syncs on the per-call path;
  - fast-math, TF32 or atomics;
  - new core `__global__` kernels;
  - `std::terminate` / `abort` as an error path;
  - writing outside the target or `/tmp`.

## Required fixes

1. **N1.** CUDA errors on the `synchronize()` and release paths are never reported as success.
   - `tc_signal.synchronize()` raises `RuntimeError` naming the CUDA error when the device query or a sync fails.
   - The atexit hook catches and logs instead of raising.
   - Document the semantics in the README.
2. **N2.** Every extent, stride and byte product is checked, including when the element count is 0. One acceptable
   rule: each extent × element size and each stride product must fit in int64 regardless of the other extents; refuse
   otherwise.
   - Apply it in core `empty`, export, import and the strides helpers those use.
   - Apply it in `tc_signal`'s `validate_tensor` and every plan.
   - The UBSan probe cases must show no runtime error.
3. **N3.** No allocation failure on the release path terminates the process.
   - Make `noexcept` boundaries catch, then leak the entry (counted) or fall back to a synchronous release.
   - Pre-reserve where possible.
4. **N4.** A throwing producer deleter is contained.
   - Wrap every deleter invocation and catch everything.
   - Count it and keep going with the remaining managers.
   - Never terminate, and never call a deleter twice.
5. **N5.** Lifecycle coordination for deleters that need the GIL:
   - Register an early-finalization hook. A Python `atexit` callback runs before CPython marks the runtime finalizing;
     verify that ordering against CPython 3.12 source or docs and cite it. The hook sets a "closing" flag under a
     mutex and waits, with the GIL released and a bounded timeout, for in-flight GIL-acquiring releases to finish.
   - A releasing thread takes the mutex, checks the flag, and increments an in-flight counter before calling
     `PyGILState_Ensure`. After the flag is set, releases leak (counted) instead.
   - Keep the existing `Py_IsFinalizing` / `Py_IsInitialized` checks as a second line of defence.
   - Prove it with a foreign-thread test (the `core_shutdown.py` pattern) that releases during the atexit phase.

## How to run

- **GPU:** every GPU command runs as `flock -w 900 /tmp/forge-gpu.lock timeout …`. Use the native driver only. You
  are the only seat running.
- **Builds:**
  - core: `cd <target>/tensor_cuda && ./build.sh 89`;
  - `tc_signal`: `cd <target>/tc_signal && PYBIND11_SOURCE_DIR=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src ./build.sh 89`.
- **Package path:** `PYTHONPATH=<target>/tensor_cuda/tensor_cuda:<target>/tensor_cuda:<target>/tc_signal`, plus
  `PYTHONDONTWRITEBYTECODE=1`.
- **Round-2 CPU probes:** rebuild them from `artifacts/p1t_rev2/probes/` with the compile commands the reviewer used,
  including UBSan where it used it, into `/tmp/p1t_fix2_probes/`. Run `host_checks.py` (round-2 version) on both
  builds.
- **Round-1 probes:** run every one in `artifacts/p1t_rev/probes/` as its README says. All must still PASS, and
  `addon_lifetime` must pass 3 runs with 0 incorrect.
- **Voice's runner, T1–T6.** Run it from `/mnt/ForgeRealm/Project-Voice`, which is READ-ONLY to you:
  ```
  cd /mnt/ForgeRealm/Project-Voice && flock -w 900 /tmp/forge-gpu.lock timeout 1200 env PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=<pp> python3 engine/tensor_requests/p1t_tests/run_p1t.py --adapter <target>/artifacts/p1t_final/tensorcuda_adapter.py \
    --items T1,T2,T3,T4,T5,T6 --fixtures /mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures \
    --tolerances engine/tensor_requests/p1t_tests/tolerances_r2.json --out <target>/artifacts/p1t_fix2/p1t_T1toT6.json
  ```
- **One combined pytest session:**
  `cd <target> && flock -w 900 /tmp/forge-gpu.lock timeout 1500 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<pp> python3 -m pytest tensor_cuda/tests tc_signal/tests -q -p no:cacheprovider --ignore=tensor_cuda/tests/test_selector_accuracy.py`
  Compare it with
  `python3 /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/scripts/apa_mf2_2_g1_compare.py /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/logs/apa_mf1_g1_baseline_unchanged.log <log>`.
  The only failures allowed are the 52 baseline ids.

## Gates

- **G1 (items closed):**
  - each of N1–N4 reproduces on d116ac0 and passes on your build, using the reviewer's probe (or an exact
    re-implementation, if a probe cannot run unmodified; say so) plus your new test;
  - N5's test passes;
  - no UBSan runtime errors in the N2 cases.
- **G2 (no regression):**
  - every round-1 probe passes (`addon_lifetime` 3/3 runs, 0 incorrect);
  - Voice's runner on T1–T6 exits 0 with PASS 236/236;
  - the combined session shows exactly the 52 baseline ids.
- **G3 (teeth):** build these mutants outside the worktree. Each must fail its new test:
  - (a) N1 error swallowed;
  - (b) N2 check removed for zero-extent shapes;
  - (c) N3 catch removed, which must terminate in the test's subprocess;
  - (d) N4 catch removed;
  - (e) N5 in-flight wait removed. Report what the test observes.
- **G4 (scope):**
  - the diff against d116ac0 touches only the Exercises paths, plus `docs/P1T_FIX2_*.md`, `artifacts/p1t_fix2/**` and
    `logs/**`;
  - no new core `__global__`;
  - the probe files are byte-identical.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. Your order file is
  immutable.
- NO git (the lead commits; read-only git is fine). NO subagents. No background waits: run every gate synchronously to
  completion.
- No deletes, anywhere. Never kill a process you did not start.
- These are READ-ONLY: production `/mnt/ForgeRealm/Project-Tensor` (outside `Fork/`), `/mnt/ForgeRealm/Project-Voice`,
  and every other fork or worktree.
- RED honesty: a failed gate is a result. Report it with receipts; never weaken a gate, probe, test or tolerance.
- Prior Art Directive (AGENTS.md): annotate at the code site, in the ledger and in the report. Known prior art:
  - the CPython 3.12 `Py_FinalizeEx` ordering (atexit before the finalizing flag) and `PyGILState_Ensure` rules;
  - pybind11's `gil_scoped_acquire` notes;
  - C++ `noexcept` and `std::terminate` semantics;
  - checked arithmetic (GCC `__builtin_*_overflow`, C23 `ckd_*`, CERT INT32-C);
  - UBSan;
  - PyTorch's and CuPy's shutdown handling of CUDA resources.
- If the AFTERIMAGE memory hook denies a Write, read what it shows you and retry. If a report Write is refused
  outright, put the full report in your final message.

## Done

- Write `docs/P1T_FIX2_REPORT.md` (the per-item table, gate table, N5 design, mutant table and prior art) and
  `docs/P1T_FIX2_LEDGER.md`.
- Your final message must contain, verbatim:
  - the per-item table;
  - the gate table;
  - the probe before/after lines;
  - the G2 verdict line;
  - the compare output;
  - the G3 mutant table;
  - the diff stat;
  - both `.so` sha256s;
  - every deviation, with its reason.
