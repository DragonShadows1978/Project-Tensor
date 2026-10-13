# P1T-FIX4 ledger (order `orders/P1T_FIX4_DEVICE_AND_ERRORS.md`, fork commit 3c7dc22)

Seat: Opus 5.5 (high), Claude Agent-tool seat. Writable target: this worktree (`voice-fix4`) and `/tmp/p1t_fix4_*`.
Append-only. Times are local (EDT), 2026-10-13.

## 02:19 Registration (before any source change or gate run; immutable from here on)

Identity at start:
- HEAD 3c7dc22 (= voice-fix3 62c8d9f + round-4 receipts + this order). `git diff --stat 62c8d9f..HEAD -- tensor_cuda
  tc_signal` is empty: the code under review is exactly 62c8d9f's.
- Order sha256 `fee7bb87952724ea6b03ef37af3a318f2eeb722eafee61b0b7e42fe058439c74`.
- Reviewer probe files (rounds 1-4, `artifacts/p1t_rev{,2,3,4}/probes/*`), aggregate sha256 of the `sha256sum` lines
  (bash glob order): `7d6392b4e4fd1b3a011f5f083b5bad19837bd0e3f511c9b4ff75e66a79e8cb92`.
- Frozen sources (62c8d9f content) sha256:
  - `tensor_cuda/src/bindings.cpp` `40455f83eca65af4725aec32ff5b0ad3f90ca479e88a99f86f34fd8a2b31ff20`
  - `tensor_cuda/src/kernels.cu` `f6f5366954291e56ceed698381c297aae4295354072264cbd27fc8afbf8f90fd`
  - `tensor_cuda/tensor_cuda/__init__.py` `ad8eaacda9eccdc1a5c210f557bc6ab4da59d9e36807c851ccdfce99e882771b`
  - `tensor_cuda/tests/test_dlpack.py` `9573e3952f0d20ecce0b657a2c4e6ca04dfda150281b84765fadea5d140634f7`
  - `tc_signal/src/dlpack_operand.cpp` `e453e74ee0e093c6b17ec11787a662038eab52b13198cf47dcbb8b4caf747b29`
  - `tc_signal/src/dlpack_operand.h` `11766f8ca4676d5a9f50900468d1ef697cd6ae960caf12b4aabee4489d6e37f1`
  - `tc_signal/src/conv.cu` `867c05a91b596859e40b6486582a97685db0c7b68d930f4851414f9789198e02`
  - `tc_signal/src/lstm.cu` `fb9f16d9ad42616436248380803075816a4d728ef2960dde7a6a944289bd7485`
  - `tc_signal/src/norms.cu` `c81cf4a875c6d59b5420007bf21e1ea7b16a004afb5df406e52b1b95ee6406d7`
  - `tc_signal/src/small_ops.cu` `4f9b0b6c487984ff31fc640e2bf64944790ff2b969558f75fc09a17f80452848`
  - `tc_signal/README.md` `0f92cb5b7f921dc5d52a2ffcf2b4bdaf4967dedd2d3a6f3c6ce94dd9914afee4`
  - `tc_signal/tc_signal/__init__.py` `c4e84c108bd2a4f3c65c84011ac3d382c531cdcb8b09c4a8ae1a54ef0a668b8d`
- "Before" binaries (wt_fix3, 62c8d9f, read-only; equal to `artifacts/p1t_fix3/build_sha256.txt`):
  core `35b8261680760851e5d150f851d58ad8f0fbec617deb225dd6d7565010487a23`,
  tc_signal `222861219e29e8486d7a6f92ee874f5b4342c6f26d11a3440bf9bb99787e535a`.
- Host: RTX 4070 SUPER (one GPU), native driver 595.99.02, CUDA 12.6, CPython 3.12.3, g++ 13.3.0, torch 2.11.0+cu130
  (torch's own libcudart.so.13 is a SEPARATE runtime instance from the libcudart.so.12 that core and tc_signal share).

Mechanism check (HOUSE_RULES §3): the mechanism doc `docs/P1T_REV4_REPORT.md` places V4-01 in
`tensor_cuda/src/bindings.cpp` `dlpack_fence_consumer` / `tensor_dlpack`, V4-03 in `tensor_from_dlpack_capsule`,
V4-04 in `tc_signal/src/dlpack_operand.cpp` `synchronize_and_release_all` / `sync_and_release_all_impl`, V4-02 in
`tensor_cuda/tensor_cuda/__init__.py` `from_dlpack`. The reviewer probes `core_export_device.py` (2 GPUs),
`core_import_query_error.py` (full rebuilt core, no visible device) and `sync_existing_error.py` (one GPU, real
runtime) run the real built modules. The order's Exercises match. No scope mismatch.

Gates (copied from the order; thresholds fixed now):
- G1 (closed): `core_import_query_error.py` and `sync_existing_error.py` fail on 62c8d9f and pass on the after build;
  `core_export_device.py` is INCONCLUSIVE (exit 3, one GPU) and the new shim test proves V4-01's guard switches and
  restores, and fails on mutant (a); the V4-02 docstring is updated.
- G2 (sweep): the sweep table lists every CUDA runtime/cuBLAS call site in `git diff 9df5bef..<final> -- tensor_cuda
  tc_signal`, each with its device, how that device is established, how its return is handled, and a verdict OK or
  FIXED; every FIXED site has a test or a stated reason why none is feasible.
- G3 (no regression): every round-1..4 probe passes or is INCONCLUSIVE only for lack of 2 GPUs; round-1
  `addon_lifetime` 3/3 runs exit 0 with 0 incorrect; round-3 `core_exit_overrun` during and bounded rc 0; Voice runner
  T1-T6 PASS 236/236; combined session FAILED/ERROR ids == the 52 baseline ids (compare prints `G1 PASS`).
- G4 (teeth): mutants built outside the worktree under `/tmp/p1t_fix4_mut_{a,b,c}`: (a) V4-01 guard removed, (b) V4-03
  check removed, (c) V4-04 capture removed; each must fail its new test.
- G5 (scope): diff vs 62c8d9f touches only the Exercises paths plus `docs/P1T_FIX4_*.md`, `artifacts/p1t_fix4/**`,
  `logs/**` (the lead's round-4 receipt and order commits already on HEAD are listed separately); probe files
  byte-identical (aggregate hash above); no new core `__global__`.

Empirical CUDA 12.6 semantics measured before the design (receipt `/tmp/p1t_fix4_scratch/semantics.py`,
`sticky_attrs.py`; output copied to `artifacts/p1t_fix4/cuda_semantics.txt` below):
- slot: a refused launch (`cudaLaunchKernel(NULL, grid 0)`) leaves 98 in the slot; `cudaPeekAtLastError` keeps it; a
  SUCCESSFUL call does not reset it; a FAILING call overwrites it (98 -> 101 after `cudaSetDevice(999)`);
  `cudaGetLastError` returns and resets it.
- no visible device: `cudaGetLastError` returns 100 as the first call and does NOT reset (100 again, peek 100).
- sticky (torch device-side assert, same process): our instance's slot stays 0 until a call touches the context;
  `cudaGetDevice`, `cudaSetDevice(0)` and `cudaPointerGetAttributes` still SUCCEED; `cudaStreamQuery`,
  `cudaStreamSynchronize`, `cudaEventCreateWithFlags` return 710; `cudaGetLastError` then returns 710 and the slot reads
  0 again, but the context reproduces 710 on the next context call.
- `cudaPointerGetAttributes`: torch default allocator and expandable segments -> type 2 (device), device 0; core raw and
  pooled `empty` -> type 2, device 0; host / null / bogus pointer -> type 0 (unregistered), device -1, return 0.

Design decisions registered before coding:
- V4-01: `tensor_dlpack` establishes the tensor's device from its label and PROVES it against the allocation
  (`cudaPointerGetAttributes` of element 0, nonempty tensors), refusing a mismatch with BufferError (the engine labels
  its own allocations cuda:0 while allocating on the current device, so the label alone is an implicit current-device
  assumption). The fence then runs under a checked guard on that device: get, switch, create, record on its legacy
  stream, wait, destroy, restore; every return is checked; errors are raised after the caller's device is restored. The
  consumer wait is issued inside the guard (stream=2 then names the tensor device's per-thread stream; a handle may be
  on any device: "event may be from a different device than stream", CUDA 12.6 `cuda_runtime_api.h`
  cudaStreamWaitEvent).
- V4-03: `cudaGetDevice` checked in the import before Storage construction and capsule renaming; RuntimeError naming
  cudaGetDevice; the slot entry of OUR failed call is consumed.
- V4-04: `synchronize()` CONSUMES (cudaGetLastError) the calling thread's existing error FIRST, drains as before (release
  what completed, leak what cannot be proven), then raises a RuntimeError naming the pre-existing error with any drain
  failure. Clear, not peek (justified in the report from the receipts above).
- Error-slot policy for the sweep: whoever reports a CUDA error consumes it from the slot (raises after a failed call
  consume it); launch paths check the slot right before their first launch so a pre-existing error is named as such
  and nothing is launched; no-throw paths count what they cannot raise (tc_signal: raised by the next synchronize()).
- Device restores: a restore failure on a success path raises; in a destructor it is counted (never thrown).
- No core header change (core.h is outside the Exercises list): cross-TU helpers are declared locally.

## 02:20-02:23 Semantics receipts and documentation sources

- `artifacts/p1t_fix4/cuda_semantics.txt` (re-run of the registration measurements, saved verbatim).
- `artifacts/p1t_fix4/cuda_doc_quotes.txt`: CUDA 12.6 Programming Guide 3.2.9.3 (Stream and Event Behavior) and 3.2.12
  (Error Checking), fetched through the AtlasForge web proxy (cache file named in the receipt). Key lines: "cudaEventRecord()
  will fail if the input event and input stream are associated to different devices"; "cudaStreamWaitEvent() will succeed
  even if the input stream and input event are associated to different devices"; "cudaEventSynchronize() and cudaEventQuery()
  will succeed even if the input event is associated to a device that is different from the current device"; the error
  variable is "overwritten by the error code every time an error occurs"; "calling cudaGetLastError() just before the kernel
  launch".
- Local prior-art reads (installed torch 2.11.0+cu130): `torch/_tensor.py` Tensor.__dlpack__ refuses a current-device
  mismatch with BufferError (no guard); `torch/utils/dlpack.py` from_dlpack passes the current stream of the tensor's device;
  `c10/cuda/impl/CUDAGuardImpl.h` uncheckedSetDevice -> C10_CUDA_CHECK_WARN (consume + warn); `c10/cuda/CUDAEvent.h`
  destroys an event under a CUDAGuard on its creating device. `nm -D libcublas.so.12`: no undefined cuda* symbol (cuBLAS
  does not share libcudart.so.12).
- Finding about the order's prior-art list: it names "PyTorch CUDAGuard / OptionalCUDAGuard around event record in its
  DLPack path". In the installed torch 2.11 the Python DLPack export does NOT guard: it refuses the mismatch. The guard
  prior art is c10's CUDAGuard in general (and CUDAEvent destroy). Recorded as read; the order's guard design is kept.

## 02:22-02:33 Implementation (files)

- core `tensor_cuda/src/kernels.cu`: drain restore checked + counted; `raise_cuda_failure`; `CheckedDeviceGuard`;
  `dlpack_device_restore_failures()`; `cuda_storage_device_mismatch()` (label vs cudaPointerGetAttributes); `fill_`
  (label proof, guard, pre-launch slot take, raising restore).
- core `tensor_cuda/src/bindings.cpp`: local declarations (no core.h change); `dlpack_fence_consumer(stream, device)` (V4-01:
  guard, every return checked, restore before raise, wait inside the guard); `tensor_dlpack` label proof (BufferError);
  import `cudaGetDevice` checked (V4-03); `tensor_empty` labels the current device, refuses another explicit index,
  consumes a failed allocation's slot entry; stats key `device_restore_failures`; docstrings.
- core `tensor_cuda/tensor_cuda/__init__.py`: V4-02 docstring (unbounded wait, knob = first diagnostic delay); V4-03 and
  empty device notes.
- tc_signal: new `src/cuda_guard.h` (one checked DeviceGuard / QuietDeviceGuard, rules 1-2); conv.cu, lstm.cu, norms.cu,
  small_ops.cu use it, `check_cuda` consumes, success paths call `guard.restore()`; conv.cu / lstm.cu destroy a new
  cuBLAS handle whose setting fails; `dlpack_operand.cpp`: V4-04 capture in `synchronize_and_release_all`, pre-launch
  slot take in the RetainUntilComplete constructor (after acquire_event), drain consumes a non-NotReady query error,
  events destroyed on their own device, unraisable-failure record (restore, destroy) raised once by synchronize(), new
  pending_stats keys; `dlpack_operand.h` and `README.md` docs.
- Builds r1: core rc 0 (1m30s; only the pre-existing kernels.cu:3285 warning; FetchContent cloned pybind11 v2.12.0, as
  in FIX3 D17), `logs/p1t_fix4_core_build_r1.log`; tc_signal rc 0, `logs/p1t_fix4_tc_signal_build_r1.log`.
- Smoke on the after build: core_import_query_error.py PASS rc 0; sync_existing_error.py PASS rc 0;
  core_export_device.py INCONCLUSIVE rc 3.

## 02:33-02:41 New tests

- `tensor_cuda/tests/test_dlpack.py` FIX4 block (15 tests, LD_PRELOAD shim `_FIX4_SHIM_C`, torch-free subprocesses
  except the sticky case). First run: the shim's function-pointer macro did not compile (own test error, fixed:
  `REAL(R, n, A)`); then 15 passed.
- `tc_signal/tests/test_fix4_findings.py` (14 tests) + `fix4_cudart_shim.c` + `fix4_stage_kernel.cu` (nvcc, a real
  4096-thread launch-configuration refusal). One own-test fix before the first run: allocations made before the
  simulated switch (tc.empty labels the current device, which the shim fakes). 14 passed.
- Before-witness (the new tests on the 62c8d9f binaries, read-only): core 13 failed / 2 passed (the same-device export
  and the sticky-import census pass on both: they pin behavior that was already right); tc_signal 11 failed / 3 passed
  (the three device-placement tests pass on both: tc_signal's launch paths already switched to the operand device; they
  pin it). Logs and per-test first failure line: `artifacts/p1t_fix4/before_witness/`. Notable witnesses: export logs
  `create@1 record@1 wait@1 destroy@1` (V4-01); import `used_dltensor` with no device (V4-03); sync returned success
  (V4-04); fill_ `launch@1`; a failed restore / destroy left in the slot and reported later by core as `CUDA error at
  to_host: invalid device ordinal` (class b).

## 02:41-02:49 G4 mutants (first pass)

- `artifacts/p1t_fix4/g4/make_mutants.py` (one asserted edit each, trees `/tmp/p1t_fix4_mut_{a,b,c}`), `run_mutant.sh`.
- (a) guard removed: 6 V4-01 tests failed (log `create@1 record@1 wait@1 destroy@1`); probe core_export_device exit 3.
  (b) check removed: test_fix4_v403_import_refuses_* failed; probe core_import_query_error rc 1 ("core import succeeded
  after failed device query; name=b'used_dltensor'"). (c) capture removed: the 3 V4-04 tests failed; probe
  sync_existing_error rc 1 ("synchronize returned success ... slot=98"). All killed. (Re-run on the final sources below.)

## 02:49-02:59 G3 probe battery (round 4 on both arms; rounds 1-3 on the after arm)

- Tooling: `make_probe_copies.py` (path-only copies, asserted substitutions, diffs written next to each copy),
  `regen_core_interop_import.py` (import half of core_interop_cpu regenerated from a tree; self-check: regenerating from
  62c8d9f reproduces the probe byte for byte = True), `run_rev4_probes.sh <arm> <root> <out> <cpu|gpu|rf>` (split so
  each part runs synchronously inside one foreground call), `run_rounds123_probes.sh`.
- lifecycle.cpp lines 11-165 are one verbatim block of BOTH 62c8d9f's and this tree's bindings.cpp (receipt
  `probe_runs/lifecycle_slice_check_{before,after}.txt`): arm-independent.
- Before (62c8d9f): core_import_query_error rc 1 (FAIL), sync_existing_error rc 1 (FAIL), core_export_device /
  core_device_drain / cuda_errors_multidevice exit 3, everything else rc 0 (release_failures copy: 66 passed);
  sync_slot_cpu rc 0 = defect trace reproduced; core_interop SPY/import rc 0 = defect traces reproduced.
- After: core_import_query_error PASS, sync_existing_error PASS, the three two-GPU probes exit 3, all others rc 0;
  sync_slot_cpu rc 1 (raises naming the staged cudaErrorInvalidConfiguration); core_interop import regenerated rc 1
  (IMPORT REFUSED, cudaGetDevice / cudaErrorNoDevice); core_interop SPY and unregenerated import rc 0 on both arms (they
  embed 62c8d9f slices: arm-independent by construction).
- host_checks: 40/40 case-condition pairs keep their rc; texts identical except, in the nogpu condition, the new V4-04
  clause in the atexit synchronize line (the runtime keeps cudaErrorNoDevice in the slot permanently, so synchronize now
  also names it as pre-existing); `probe_runs/rev4_host_checks_compare.txt`.
- Rounds 1-3 (after): addon_lifetime 3/3 runs 0/12 incorrect, core_edges 0, core_shutdown normal 0 / finalizing 0,
  numerics_and_streams 0, round-1 host_checks final lines identical to FIX3's receipt, core_device_drain 3, round-2
  host_checks nogpu 20/20 rc 0 and gpu as FIX3 (oom_sync -6: FIX2 D4, the probe's own teardown), round-3
  core_exit_overrun during 0 / bounded 0.

## 02:59-03:07 Voice T1-T6 and combined session (first pass)

- Voice: PASS 236/236 (identical f1 figures to FIX3). Combined: 52 = 52 ids, G1 PASS, 2718 passed.

## 03:07-03:22 G5 and sweep; late findings in my own work

- G5 checker `g5_scope.py` (FIX3's adapted): PASS.
- Sweep extractor `sweep/extract_sites.py` + brace-matching attribution `sweep/funcs.py` (first version was quadratic
  and was auto-backgrounded by the tool after 120 s; I stopped my own task and rewrote it with caching) +
  `sweep/annotate.py` (every site must match one rule; 0 unannotated).
- While deciding which FIXED sites lacked a test I found an ordering issue in my own fill_: a failed success-path restore
  raised BEFORE `mark_modified()`, so a queued write could go without a revision bump. Fixed (bump, then restore) ->
  core rebuild r2 (`logs/p1t_fix4_core_build_r2.log`, rc 0). Added tests: export create/wait failure, label-proof query
  failure, fill_ guard failures (switch, restore, launch+unwinding restore), empty's failed allocation, no-device empty;
  tc_signal: restore failure while unwinding, check_cuda consumption (lstm memset), LSTM device placement, LSTM cuBLAS
  handle setting. Own test errors fixed before acceptance: a misplaced parametrize decorator; the conv prealloc after the
  allocator became pop-based.
- One test-code member of class (b) found by the sweep: `fix2_native_harness.cpp` ignored its pre-drain
  `cudaStreamSynchronize` result. Now checked (test_fix2_findings 19 passed).

## 03:22-04:06 Final re-runs (final sources and binaries)

- New tests on 62c8d9f: core 20 failed / 2 passed; tc_signal 14 failed / 4 passed (`before_witness/`). The passes pin
  behavior that was already right (same-device export, sticky-import census, tc_signal device placement x4).
- Mutants re-made from the final sources and re-run: (a) 8 V4-01 tests failed (killed), probe exit 3; (b) 1 failed
  (killed), probe rc 1; (c) 3 failed (killed), probe rc 1. Test files cmp-identical to the worktree's.
- After-arm round-4 battery, rounds 1-3, Voice and the combined session re-run on the final binaries:
  core `1da017ca2ec917019acc1ae9220fca87a74594257c053050056ed55eb2cd9b48`,
  tc_signal `909af0c5e0c76cfd16c4016abb72b35bdde5a01cc96505334c8cc36876426352`. Results as in the first pass; Voice PASS
  236/236; combined (03:58-04:05) 2729 passed, 52 = 52 ids, G1 PASS. G5 PASS. Sweep: 148 product sites, 0 unannotated,
  OK 90 / FIXED 58 (54 with a test, 4 with a stated reason).
- Host load average 26-33 throughout (other users' processes); wall times are not performance measurements.

Prior art (also at the code sites and in the report): CUDA 12.6 runtime current-device semantics (cudaSetDevice /
cudaGetDevice header docs); cudaEventRecord same-context rule and cudaStreamWaitEvent cross-device rule (header + PG
3.2.9.3); cudaGetLastError / cudaPeekAtLastError and the clear-before-launch rule (PG 3.2.12); PyTorch c10 CUDAGuard /
C10_CUDA_CHECK_WARN / CUDAEvent destroy-on-own-device (torch 2.11 headers); PyTorch 2.11 Tensor.__dlpack__ mismatch
refusal and torch.utils.dlpack consumer stream choice (installed sources); CuPy cupy.cuda.Device context (not re-read
here: unverified, lead to check "cupy.cuda.Device __exit__ setDevice"); LD_PRELOAD interposition (glibc); DeMillo, Lipton
& Sayward 1978 mutation testing. Ours: the allocation-vs-label proof at export / fill_, restore-before-raise for the
fence, the pre-launch take at the shared RetainUntilComplete constructor, routing unraisable failures to the next
synchronize(), and the genuine-slot shim.

## 04:06-04:10 Report and final scope check

- `docs/P1T_FIX4_REPORT.md` written (per-finding table, sweep table embedded from `artifacts/p1t_fix4/sweep/sweep_table.md`,
  probe before/after table, gate table, Voice line, compare output, mutant table, census, prior art, deviations D1-D17,
  residuals, diff stat, changed files, scratch list).
- Final `g5_scope.py`: 179 paths OK, 0 OUT, probe aggregate == registered, G5 PASS. GPU idle afterwards (218 MiB, P8).
- No git writes, no deletes, no subagents; all scratch under /tmp/p1t_fix4_* (listed in the report).
