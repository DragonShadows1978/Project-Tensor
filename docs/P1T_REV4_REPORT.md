# P1T-REV4 independent review

**REJECT merge of 9df5bef..62c8d9f. Zero BLOCKING, one MAJOR, two MINOR and one NIT finding.** Earlier findings are CLOSED in their original scope. FIX3 closes V3-01/V3-02, but the export fence still records on the wrong device after a current-device switch (V4-01). David's conditional merge approval is therefore not met. No defect is claimed fixed by this review.

| Question | Verdict |
|---|---|
| R1 | RISK |
| R2 | UNSOUND |
| R3 | UNSOUND |
| R4 | CORRECT |
| R5 | PASS |
| R6 | REJECT |

## Scope and evidence boundaries

Evidence classes: source reasoning, CPU native reproduction/UBSan, CPU compilation/link inspection, live real-data metadata and historical receipt audit. Reviewed all changed production implementation, all plan functions, kernel equations, package wrappers/builds, DLPack ABI fields, and relevant tests/probes. The independent implementation assessment preceded the fix reports. The mechanism is the approved core-DLPack/add-on split; no scope mismatch.

HEAD is 678bc76eae007033d58eab26d96ee851f8211032. All 48 submitted files byte-match 62c8d9f (`source_inventory.json`); `working_vs_target.diff` is empty. Full submitted diff and FIX3 delta are `full.diff` and `fix3.diff`. Registration preceded gates and is immutable. Existing core kernel bodies, ordinary pooled/raw/CPU allocation branches, caches and round are preserved except the checked numel/stride helper repair and new fill binding that reuses an existing kernel. No new core kernel was added.

CLOSED below means the earlier specific defect is closed by the stated source mechanism and bounded evidence. It does not certify every device, numerical input, failure or interpreter configuration. GPU-dependent witnesses remain explicit lead probes. Full core binaries were not built/executed by this seat; current hash-pinned verbatim source slices exercise CPU lifecycle/import paths. Export/event and last-error CPU traces use CUDA-call replacements and prove call placement only; they do not establish GPU behavior. The device/error consequences are source reasoning from the primary contracts, with GPU probes supplied.

## Closure table

| Item | Status | Lines read and evidence |
|---|---|---|
| F01 | CLOSED | tc_signal/src/dlpack_operand.cpp:773-855; bindings.cpp:284-286,326-327,343-344; lstm_bindings.cpp:239-242; m4_bindings.cpp:239-241,294-295,315-316,334-335 — Event-fenced managers on success/partial launch; caught push fallback and leak without completion proof. Source reasoning; release GPU probe pending. |
| F02 | CLOSED | tc_signal/src/dlpack_operand.cpp:196-201,314-323; m4_bindings.cpp:65-104; bindings.cpp:239-260; lstm_bindings.cpp:178-210 — Required None guards and safe absent metadata. CPU None cases raise TypeError without launch. |
| F03 | CLOSED | tensor_cuda/src/bindings.cpp:307-331,832-883,910-924; tc_signal/src/checked_math.h:31-89; dlpack_operand.cpp:126-193,760-777; all plan functions — Rank/count/bytes/address/geometry checked; FIX3 closes aggregate sums. CPU refusal census, UBSan and entry-point checks pass. |
| F04 | CLOSED | tensor_cuda/src/bindings.cpp:641-691,711-764; tensor_cuda/tensor_cuda/__init__.py:188-223 — Finalizing/closing releases leak without GIL acquisition; admitted releases finish before ordinary finalization. Current CPU lifecycle slice passes. |
| F05 | CLOSED | tensor_cuda/src/kernels.cu:221-264 — External destructor selects Storage device, syncs it and restores caller; failed drain leaks. Source closure; two-device GPU check pending. V4-01 is a separate export path. |
| F06 | CLOSED | tensor_cuda/src/bindings.cpp:448-487 — unique_ptr owns export context through metadata/capsule construction; ownership transfers once. C++ RAII reasoning. |
| F07 | CLOSED | tensor_cuda/include/tc/core.h:80-88; docs/P1T_FIX1_M1_ADDENDUM.md:20-33 — Coordinated native-client rebuild is documented. Closed as rollout prerequisite, not binary ABI compatibility. |
| N1 | CLOSED | tc_signal/src/dlpack_operand.cpp:688-727,818-836,858-897; tc_signal/tc_signal/__init__.py:139-155 — Original failed-query/empty-list silent success fixed; CPU cudaErrorNoDevice is raised and logged. V4-04 separately covers an existing synchronous error slot. |
| N2 | CLOSED | tensor_cuda/src/bindings.cpp:307-331,438-444; kernels.cu:80-98; tc_signal/src/checked_math.h:63-89; dlpack_operand.cpp:158-168 — Zero-neighbor extent/byte bound checked before empty treatment; prefix/suffix refusals and 18 UBSan census cases pass. |
| N3 | CLOSED | tc_signal/src/dlpack_operand.cpp:469-472,519-572,586-600,688-727,773-837; bindings.cpp:408-410 — State initialized at import; fixed sync bookkeeping, reserved owners/recycle slots and caught pending push. CPU OOM safe; full release fault sweep remains GPU probe. |
| N4 | CLOSED | tc_signal/src/dlpack_operand.cpp:238-261; tensor_cuda/src/bindings.cpp:672-680 — Each ordinary C++ deleter exception is caught and counted; owner cleared first. CPU throwing ManagedOwner survives. Full core foreign-thread check is pending. |
| N5 | CLOSED | tensor_cuda/src/bindings.cpp:627-691,711-764; tensor_cuda/tensor_cuda/__init__.py:188-223 — Admission/closing/in-flight coordination with unbounded quiescence; no GIL wait under mutex. Finite overrun exits normally. |
| V3-01 | CLOSED | tensor_cuda/src/bindings.cpp:711-764; tensor_cuda/tensor_cuda/__init__.py:174-223; tensor_cuda/tests/test_dlpack.py:1232-1286 — 5.5 s admitted deleter crosses 5 s diagnostic delay, finishes before finalization; CPU rc0, still_in_flight=0. No early-return timeout remains. |
| V3-02 | CLOSED | tc_signal/src/dlpack_operand.cpp:623-656,760-809,833-836; tc_signal/tests/fix3_native_harness.cpp:99-191 — Pre-launch aggregate refusal, nonthrowing sticky saturation and exact below-limit bound comparisons. 17 UBSan checks and two real entry-point refusals pass. |

## New findings

| ID | Severity | File:line | Defect | Fix |
|---|---|---|---|---|
| V4-01 | MAJOR | tensor_cuda/src/bindings.cpp:376-400,446 | Export fence records on the current device rather than the tensor device | Guard event creation/record with the tensor device and restore the caller device, or explicitly refuse a current-device mismatch before exporting. |
| V4-02 | NIT | tensor_cuda/tensor_cuda/__init__.py:107-110 | from_dlpack still documents the removed bounded shutdown wait | Update the public docstring to describe the diagnostic delay and unbounded wait. |
| V4-03 | MINOR | tensor_cuda/src/bindings.cpp:817-823,888-901 | Core import ignores cudaGetDevice failure and consumes the capsule | Check cudaGetDevice return; raise a named RuntimeError before Storage construction or capsule renaming. |
| V4-04 | MINOR | tc_signal/src/dlpack_operand.cpp:510-516,688-704,858-861 | synchronize omits the already-present synchronous CUDA error slot | Capture the calling thread existing CUDA error on entry, drain safely, and raise a named error even if every stream synchronization succeeds. |

### V4-01 — MAJOR: export fence uses the caller's current device, not the tensor's device

Locations: `tensor_cuda/src/bindings.cpp:376-400` (`dlpack_fence_consumer`), `:446` (call without tensor device), `:459-463` (capsule reports the actual tensor device).

Concrete scenario: import a real CUDA tensor on device 0 while device 0 is current. Enqueue a long legacy-stream operation followed by a write to that tensor on device 0. Keep the tensor/manager alive. Switch the host thread's current CUDA device to 1, then call `t.__dlpack__(stream=device0_nonblocking_consumer_handle)`. The exporter creates and records its event on device 1's legacy stream: it never receives `a.device.index`, selects it, or refuses a mismatch. `cudaStreamWaitEvent` permits a stream on device 0 to wait on an event from device 1, so the handoff can return successfully while protecting only unrelated device-1 work. The consumer can complete its read before the pending device-0 write, producing stale or racing values. This does not require freeing the producer or running a core operation after the device switch. The source also does not guard special stream=2 against this mismatch.

Fix: use checked scoped selection of the tensor device for event creation/record and restore the caller's current device; validate the requested consumer stream as appropriate. If the engine intentionally supports only the current device, explicitly refuse the mismatch before exporting a capsule. A correct external-Storage destructor does not supply the missing producer-to-consumer dependency.

Evidence: unchanged core export source compiled into `probes/core_interop_cpu.cpp`, with CPU call tracing, prints `tensor_device=0 recorded_device=1 caller_device=1` (`logs/p1t_rev4_core_interop.log`). This establishes the device omission and call placement, not real GPU stale-data reproduction. The race follows by reasoning from verified CUDA per-device event/stream semantics and the DLPack producer's ordering obligation. `probes/core_export_device.py` supplies the real two-device witness and fixed-path check; pending, not guessed PASS. Severity MAJOR reflects silently incorrect values under a legal stream handoff rather than a missing diagnostic alone. This was present in the initial DLPack implementation; it is not introduced by FIX3. F05's destructor fix remains CLOSED on its own path.

Prior art: DLPack (dmlc contributors, 2017+) / Python Array API (2021+) stream handoff; NVIDIA CUDA 12.6 device selection, event recording and cross-device waits. Taken: those ordering constraints and the standard scoped device guard. Ours: the device-switch review scenario. [DLPack Python specification](https://dmlc.github.io/dlpack/latest/python_spec.html), [CUDA event recording](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-runtime-api/group__CUDART__EVENT.html), [CUDA multi-device streams/events](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#stream-and-event-behavior).

### V4-02 — NIT: public from_dlpack help still promises a bounded exit wait

Location: `tensor_cuda/tensor_cuda/__init__.py:107-110`. It says the hook waits at most `_DLPACK_SHUTDOWN_WAIT_S` seconds. FIX3's implementation waits without a bound; the knob is only the first-diagnostic delay. A caller inspecting this API's help can infer a false upper bound on process exit. Fix: update this docstring to match the hook documentation at `:188-202`. Evidence: source comparison, no runtime uncertainty. Prior art: CPython lifecycle/atexit API documentation (3.12, 2023); no new algorithm or optimization.

### V4-03 — MINOR: core import ignores a failed CUDA current-device query

Location: `tensor_cuda/src/bindings.cpp:817-823`, followed by ownership transfer at `:888-901`. `cur` starts at zero; `cudaGetDevice(&cur)` is unchecked. On failure, the importer compares the producer's ordinal with an unvalidated zero, proceeds to construct Storage and consumes the capsule. This hides the real CUDA error and can cause subsequent cleanup to leak on the failed drain. It also misreports failures as a device mismatch for other descriptor ordinals.

Concrete CPU reproduction: no usable CUDA device; a zero-size fp32 CUDA-device-0 descriptor with null data, shape `[0]`, and a null deleter. Current verbatim import/Storage code against the real CUDA 12.6 runtime reports:

```
cudaGetDevice_error=100 device=-1
IMPORT SUCCEEDED name=used_dltensor
```

Receipt: `logs/p1t_rev4_core_interop.log`. No CUDA allocation, kernel or fake-address access is performed. This is a synthetic descriptor/error-path reproduction, not evidence for a resident CUDA array on a GPU-less host. A failed query caused by an earlier CUDA error has the same unchecked branch.

Fix: check `cudaGetDevice`'s return and raise a named RuntimeError before Storage construction or capsule renaming. Refusal should leave ownership with the capsule. `probes/core_import_query_error.py` checks the full rebuilt binding without a visible GPU. Prior art: CUDA 12.6 checked return-code contract and DLPack ownership handoff. This is a new core-path finding; N1's add-on query fix is intact. [CUDA device management](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-runtime-api/group__CUDART__DEVICE.html).

### V4-04 — MINOR: synchronize does not inspect a preexisting synchronous CUDA error

Locations: `tc_signal/src/dlpack_operand.cpp:510-516,688-704,858-861`. A synchronous launch refusal can leave the calling host thread's last-error slot nonzero while queuing no work. A later successful stream sync proves completion of queued work; it does not by itself check that stored launch error. When device queries and stream sync return success, the empty-list path immediately returns success without ever querying the slot. This contradicts the package's broad error-reporting promise and the explicitly requested preexisting-error census. Deferred release can still be safe here; the defect is a false successful diagnostic, not a demonstrated premature release.

Fix: capture the caller's existing error before operations that may clear/replace it, drain pending work safely, and report that captured error even if all stream syncs succeed. Preserve the existing once-only reporting policy as appropriate. Do not skip draining because a synchronous launch was refused.

Evidence class: source reasoning from the distinction between CUDA's per-thread synchronous error slot and stream completion. Actual-source CPU call tracing with explicitly successful query/sync returns prints `synchronize returned success; previous_error_slot=9` (`logs/p1t_rev4_sync_slot.log`). Those replacements are a call-path witness only; no real CUDA runtime reproduction is claimed. `probes/sync_existing_error.py` is a short one-GPU real-runtime check, separated from the old multi-device probe so one-GPU hardware can exercise it; pending. The earlier round-3 two-device probe already registered this expected behavior, but the lead's one-GPU runs exited INCONCLUSIVE before staging it. N1's original failed-query/no-device bug is CLOSED; this is a distinct synchronous-error variant.

Prior art: NVIDIA CUDA 12.6 `cudaPeekAtLastError` / `cudaGetLastError` and `cudaStreamSynchronize`; taken: their separate contracts. Ours: independent staging and a one-device witness. [CUDA error handling](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-runtime-api/group__CUDART__ERROR.html), [CUDA stream synchronization](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-runtime-api/group__CUDART__STREAM.html).

## R1–R6 reasoning

R1 RISK: representable ordinary core paths retain their baseline semantics by source comparison; no newly failing baseline test is claimed. Native clients need the coordinated rebuild documented for F07. The new interop surface's unguarded export device dependency prevents a SAFE integration conclusion. The historical 52-id comparison cannot cover a two-device handoff it never exercised.

R2 UNSOUND: V4-01 loses producer ordering; V4-03 hides import query errors. Other reviewed ownership paths preserve offsets, exact dtype mapping, writable-only core imports, version checks, once-only used-capsule naming and RAII. Import teardown selects the allocation device and refuses unsafe release after failed drains or closing/finalizing. Admission stays closed while FIX3 waits for all other-thread admitted releases to leave after GIL release. Main-thread self-exclusion avoids waiting for its own stack; callbacks must return before that thread can finalize. The unsupported free-threaded/subinterpreter/cancelled-producer cases are not newly certified.

R3 UNSOUND at the preexisting synchronous-error reporting boundary (V4-04). The deferred ownership mechanism itself is sound by source review: retain all consumed managers through success/partial launches; reserve owner and event-recycle capacity before launch; contain queue allocation failure with checked synchronous fallback; leak without completion proof. Remove entries before producer callbacks so reentrant drains do not release an entry twice. Per-device fixed-capacity synchronization/error bookkeeping allocates no host containers on release. Ordinary native producer exceptions are contained per manager, cleared before calling, and counted. All nine operations construct a retainer before their first launch. GPU failure-path behavior remains finite lead verification.

R4 CORRECT for reviewed equations and checked host arithmetic; no universal numerical/GPU certification. Conv im2col and grouped GEMM use the proper row-major/column-major transpose mapping; cuBLAS dimensions are checked before narrowing. ConvTranspose uses the ONNX input-channel-major weight layout and deterministic gather tap/divisibility rule. LSTM has ONNX i/o/f/c gates, W/R bias halves, state initialization, direction/time write mapping, and one stream ordering recurrent GEMM before state updates. Norms map per-row/per-element/per-channel affine indices correctly; shifted two-pass biased variance and reduction barriers are consistent. Exact integer nearest/reflection maps and rintf half-even semantics match the narrow contract. Validated fp32 byte bounds constrain the remaining kernel index/loop arithmetic; near-limit output/scratch quantities are checked in every plan. Empty add-on operands are deliberately refused; scalar rint is valid. Extreme fp32 statistic overflow and cross-toolkit reproducibility are not excluded by a finite fixture pass.

R5 PASS for package separation and the CPU build/link scope: fresh unchanged scratch add-on builds with CUDA 12.6, GCC 13 and CPython 3.12/pybind11 v2.12.0; core bindings pass syntax compilation. The add-on has no core header/source/symbol dependency, links shared cudart/cuBLAS at `/usr/local/cuda-12.6/lib64`, and keeps the approved lazy default allocator import. Final deployed core binary/link identity still requires the lead's clean rebuild receipt. DLPack headers differ in comments/examples and an inert Torch stable guard, not the consumed ABI layouts. Standard Release flags have no fast math or per-thread default-stream switch. Two lazy cuBLAS handles are intentional/documented prior scope; both use the same ordered legacy stream and required math/atomics modes.

R6 REJECT: the open MAJOR V4-01 prevents meeting David's condition. FIX3 itself has no newly demonstrated ownership/arithmetic defect. Correct the named export path, verify the two-device handoff and error probes against rebuilt binaries, and rerun the unchanged relevant regression/acceptance gates before seeking a new approval.

## CPU receipts and historical isolation

- `logs/p1t_rev4_build.log`: add-on build and core syntax PASS. `build_identity.json` pins scratch module SHA256 and identical copied inputs. `ldd.txt`, `objdump.txt`, `undefined_symbols.txt`: shared CUDA links and no core symbol dependency. Python files/probes AST-parse.
- `host_checks.json`: 20 recorded subprocesses, each rc0 with expected shape/refusal/error outcome; the collector's own rc is not used alone as a pass. `operand_census.json`: 17 checks and seven once-only consumed managers.
- `logs/p1t_rev4_ck_census.log`: 18 checked-math cases, zero failures, no UBSan diagnostic. `logs/p1t_rev4_accounting.log`: 17 saturation/aggregate/bound cases, zero failures, no UBSan diagnostic. `aggregate_entrypoints.json`: actual rint/divide entry points refuse before event acquisition/launch; two/three distinct producer deleters run once, nothing queued.
- `logs/p1t_rev4_lifecycle_{default,short}.log`: current verbatim admission/GIL/wait slice permits ordinary finalization only after the finite callback finished. Default 5.5 s callback crosses the 5 s diagnostic delay, then reports still_in_flight0, rc0. Short case reports 0.1/0.2/0.4 s diagnostics, completes at approximately0.6 s, rc0. These CPU slice runs are not full rebuilt-core GPU-Storage tests.
- `real_data_census.json`: live Voice manifest139 cases,61 weight archives read by NPZ headers. `real_metadata_plans.json`:129 recorded Conv/ConvTranspose/LSTM/norm/Round shapes match native plan output; ten Resize/Pad cases require data-valued attributes and remain GPU numerical probe coverage. No real Voice numerical acceptance run here.
- `lead_receipt_audit.json`: supplied lead JSON independently counts236 PASS/236 entries. Lead/FIX3 combined logs and the recorded baseline each have the exact same52 FAILED/ERROR IDs. This verifies the historical receipt contents, not this source's GPU behavior. Lead exit-overrun logs show normal completion after5.5 s; addon.log records12 correct finite trials. Neither checks V4-01 or the separate preexisting-error case. No baseline source defect outside the submitted diff is reclassified as a new regression.

## Operand census

| Condition | Evidence / explicit residual |
|---|---|
| Zero-size and 0-d | Core CPU empty helper accepts valid empty shapes; add-on zero refuses, scalar rint plans (). Full CUDA roundtrip pending core_edges. |
| Near int64 limits, zero first/middle/last | Checked extent/byte bound rejects overflow before empty treatment; 18 UBSan cases and aggregate entry-point refusals pass. |
| Nonzero byte_offset | Host accepts offset4 and rejects offset2 for fp32; core offset/range checks source-reviewed. GPU data addressing pending core_edges/numerics. |
| Extent1 unusual strides | Host accepts INT64_MIN singleton stride while compact stepped axes are enforced. |
| Read-only/versioned/copied | Host readers accept supported flags; writers refuse read-only/copy and deleters run once. Core read-only refusal source-reviewed. |
| Capsule twice / same operand twice | Same bare capsule twice is refused on second consumption; first manager deleted once; later used_* refused. Fresh read exports may alias; written ranges cannot. |
| Nondefault producer stream | Protocol requests stream1; retained ownership survives launch by source. Actual forward ordering/reclamation pending addon_lifetime/numerics; switched-device core export fails source ordering (V4-01). |
| Producer deleter raises | Actual addon native throwing reset survives; core per-call catch reviewed. Full real-Storage foreign release remains lead check. Forced cancellation/pthread_exit are not ordinary producer exceptions. |
| Host allocation failure on release | CPU OOM safely propagates; no allocating take-all deque, reserved owners/recycle and caught queue push. GPU fault sweep pending release_failures. |
| Exit pending or foreign releases in progress | Current CPU finite-overrun ordinary exit passes; addon atexit sync/log and core closing leaks source-reviewed. Full rebuilt paths pending core_exit_overrun/release_failures. A producer that never returns intentionally blocks exit. |
| Existing CUDA error | Returned query/sync failures are recorded/raised; no-device path witnessed. Synchronous last-error slot omission is V4-04; real witness pending sync_existing_error. |
| Multi-device pending entries | Per-device stream guards and FIFO query/pop reviewed; failed devices leak safely. Actual two-device check pending cuda_errors_multidevice. Export fence is separately V4-01. |

## Probe list

All GPU probes are **PENDING LEAD**, and must use rebuilt reviewed binaries. Exit3 is INCONCLUSIVE, not PASS. Run sequentially under the lead's existing GPU-lock policy. The CPU helpers were built/run here as described above; only the CPU modes of native harnesses were run.

| Path under artifacts/p1t_rev4/probes/ | Decides |
|---|---|
| core_export_device.py | Two GPUs: decides V4-01 wrong-device export dependency; explicit mismatch refusal or correct producer completion and values must pass. timeout60; exit3 needs two GPUs. |
| core_import_query_error.py | Full rebuilt core, CUDA_VISIBLE_DEVICES empty: decides V4-03 refusal before ownership transfer. timeout30. |
| sync_existing_error.py | One GPU: decides V4-04 preexisting synchronous launch-error reporting, with no device-memory fault. timeout30; exit3 if staging is inconclusive. |
| addon_lifetime.py | F01 side-stream capsule-only manager retention and finite reclamation/reuse stress. timeout60; all 12 outputs correct and entries retained. |
| core_edges.py | Core scalar/zero/offset/singleton/used-capsule/stream and shape-refusal cases. timeout60; every assertion passes. |
| core_device_drain.py | Two GPUs: F05 last-use completion before deleter and caller-device restoration. timeout60; exit3 needs two GPUs. |
| core_exit_overrun.py | Full imported Storage foreign release at ordinary exit: run during and bounded, optionally nested/after. timeout25 per process; completed deleter before finalizing, rc0. |
| numerics_and_streams.py | Finite ONNX fp64 parity, repeatability, dual handles, L=1/eps=0, special rounding and nondefault producer ordering. timeout90. |
| release_failures.py | Actual FIX1/FIX2 OOM, producer exceptions, fallback/bound errors, partial launch and pending-at-exit tests. timeout650; rc0 and no skips. |
| cuda_errors_multidevice.py | Multi-device FIFO retention, exact output, restore caller device plus existing-error census. timeout90; exit3 needs two GPUs. |
| host_checks.py / operand_census.py / real_metadata_plans.py | CPU rebuilt add-on: 20 subprocess cases, 17 ownership/metadata cases and 129 live fixture geometries. timeout120/40/40; inspect each result, not collector rc alone. |
| lifecycle.cpp / lifecycle_timeout.py | CPU current core admission/GIL/wait slice: default and short ordinary-finalization cases. timeout25/15; finite deleter completes first, still_in_flight=0, rc0. |
| core_interop_cpu.cpp / build_core_interop_cpu.sh | CPU actual core import with real runtime (V4-03 witness); export call tracing (V4-01 mechanism only). timeout400; trace reproduces wrong-device record and failed-query consumption. |
| sync_slot_cpu.cpp | CPU actual add-on sync with explicit successful API returns and a staged slot; decides whether source checks the slot (it does not). Trace only, not real CUDA validation. timeout15. |
| core_cpu.cpp / build_cpu.sh | CPU builds, core zero-extent source helpers and core syntax. Includes actual FIX2 ck_census and FIX3 accounting harness compilation; run only CPU modes, never dtor/sweep here. timeout1100 build. |

The copied earlier GPU probes are byte-identical to `artifacts/p1t_rev3/probes/`. New probes have one-line pass/fail rules and prior-art comments. CPU rebuild: from the review root run `timeout 1100 bash artifacts/p1t_rev4/probes/build_cpu.sh`; execute the host scripts with `PYTHONDONTWRITEBYTECODE=1` and `PYTHONPATH=/tmp/p1t_rev4_build/tc_signal`, lifecycle scripts with `PYTHONPATH=/tmp/p1t_rev4_build`. Run `fix2_harness ck_census` and `fix3_harness accounting`; their GPU modes remain prohibited in this sandbox. `build_core_interop_cpu.sh` builds/runs both CPU variants. CPU tracing gates expect the recorded defect as a witness; a rc0 trace is not an approval of the product.

## Prior art

No product algorithm was implemented. Findings/fix recommendations apply existing standards: DLPack (dmlc,2017+) / Python Array API (2021+) ownership and stream handoff; NVIDIA CUDA12.6 device/event/error contracts; CPython3.12 (2023) GIL/atexit/finalization; C++17 (2017) RAII/noexcept/signed arithmetic and GCC5 (2015) checked builtins; ONNX Conv, ConvTranspose, LSTM, normalization, Resize, Pad and Round operator semantics. Probe comments and ledger name what was taken and the review's scenarios. The scoped device guard/error checks are standard practice, not a new optimization. [CPython lifecycle docs](https://docs.python.org/3.12/c-api/init.html), [ONNX LSTM](https://onnx.ai/onnx/operators/onnx__LSTM.html). Variance/reduction historical authorship remains as attributed in reviewed code; exact additional history was not re-audited here (unverified — lead to check: Chan Golub LeVeque1983 shifted variance; Harris2007 CUDA reduction; RCU McKenney Slingwine1998). No new prior-art claim is attached to a product invention.

## Hygiene and procedural notes

No source/test/build file edits, git writes, subagents, GPU execution, deletions, or unrelated process kills. Writes are confined to the authorized review artifacts/docs/logs and `/tmp/p1t_rev4_build`. Nothing was moved to DELETE_ME. Scratch tree/builds/helpers remain at `/tmp/p1t_rev4_build`.

One probe build first missed CPython linker flags because the driver read only the first python3-config output line; corrected probe driver only, failed log retained (`logs/p1t_rev4_export_spy_build.log`). One orchestration wait after a yielded maximum foreground PTY wait used1000ms instead of the required maximum; the following wait used300000ms and the build completed normally. No status was sent between those waits. Both deviations are recorded in the ledger and do not change a code finding or gate threshold.
