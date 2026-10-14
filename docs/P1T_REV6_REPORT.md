# P1T-REV6 independent review

IN PROGRESS: no merge verdict yet. CPU-only review; no GPU execution.

Registration: artifacts/p1t_rev6/REGISTRATION.md. Reviewed snapshot: acd42fa; full and FIX5 diffs saved.

## New finding (confirmed)

V6-01 MAJOR: synchronize deadlocks with a finite producer deleter waiting for a caller-held lock. Thread A holds the producer allocator lock and calls synchronize. Thread B is releasing a refused read-only operand, and its ordinary finite deleter needs that same lock. synchronize now waits for B before any CUDA drain; B waits for A. Releasing the GIL does not release the application lock. A rescue after 1 second is required to complete the CPU witness. No pending GPU work is needed.

Location: tc_signal/src/dlpack_operand.cpp:1284-1289; 291-305; 249-264. Fix: Keep mandatory close-and-wait in the actual exit hook; remove the unconditional unrelated-thread callback wait from ordinary synchronize. If callback completion is desired as a separate API, give it an explicit no-locks-held contract and a pre-call generation boundary. Preserve admission and safe leakage during exit.

Evidence: CPU full freshly compiled reviewed extension; logs/p1t_rev6_sync_lock.log; probes/sync_lock_inversion_cpu.py. BLOCKED_BEFORE_CUDA=1; elapsed=1.010 rescue=True release_finished=True.

## Prior art

Coffman et al. 1971 circular wait; CPython3.12 locks/GIL PSF2023; DLPack ownership DMLC2017. Taken: lock-order analysis; contribution: refused-call finite callback scenario.

## V6-02 MAJOR

Core exit hook does not wait for Python invoked by export argument conversion. A daemon thread calls Tensor.__dlpack__(copy=Flag()) or max_version=(Index(),3). The accepted __bool__/__index__ sleeps for 0.6 seconds and releases the GIL. Core import-release in_flight remains zero; the real core shutdown hook returns, finalization starts, and the callback attempts to retake the GIL while pybind11 objects are still on the native stack. Fresh exact-source host slices exit -11 in both cases; joined controls finish export and exit0. A zero-size CUDA tensor on the lead exercises the full public binding without GPU work.

Location: tensor_cuda/src/bindings.cpp:483-515; 799-870. Fix: Either admit the entire export argument-conversion section and close/wait for those callbacks in the core exit lifecycle, or strictly refuse arguments that require arbitrary Python conversion before invoking them. Cover sequence length/item methods as well as __index__/__bool__; preserve explicit GIL restore and self-exclusion.

Evidence: logs/p1t_rev6_casts_exit_bool.log and casts_exit_index.log: rc -11; corresponding --control logs: rc0. Exact current-source slices pinned in source_slices.json; real-core single-GPU probe pending.

Prior art: CPython3.12 lifecycle/GIL PSF2023; C++17 object cleanup; pybind11v2.12 object decref and dispatcher forced-unwind contracts (2015/2024). Taken: lifecycle and ownership contracts; contribution: finite user-cast exit witnesses.

## V6-03 MINOR

Transient synchronize failures disappear if diagnostic allocation fails. A one-shot cudaGetDevice, cudaStreamSynchronize, or completion-event query failure is consumed into a local SyncReport. Building its diagnostic throws bad_alloc (registered alloc:error_message hook). The caller sees only MemoryError; the local report is discarded. A retry whose CUDA calls succeed returns success with no original error or persistent counter. Query variant leaks its operand safely but loses the CUDA reason; empty-list sync/device variants leave no counter at all.

Location: tc_signal/src/dlpack_operand.cpp:986-1025; 1298-1304; 1314-1355. Fix: Persist all current-drain SyncReport failures before any message allocation, using allocation-free bookkeeping; clear them only after a diagnostic is successfully raised. Preserve existing prior/deferred error persistence and safe release/leak rules.

Evidence: Three actual-source CPU runtime-replacement traces, logs/p1t_rev6_diagnostic_oom_{query,sync,device}.log: FIRST MemoryError; FAIL second synchronize returned success: transient CUDA diagnostic disappeared; rc1 each. Prior-slot/OOM control passes.

Prior art: CUDA12.6 runtime error-slot and stream/event return-code contracts NVIDIA2024; C++17 allocation/exception safety; SQLite OOM fault injection. Taken: consume only after durable accounting; contribution: current-drain plus message-OOM combinations.
