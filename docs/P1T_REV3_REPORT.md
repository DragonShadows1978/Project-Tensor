# P1T-REV3 independent code review

**REJECT merge of 9df5bef..5f9f955. One MAJOR, one MINOR, zero BLOCKING and zero NIT findings.** F04 and N5 remain PARTLY closed because normal interpreter finalization can abort after the bounded release wait. F03 remains PARTLY closed at aggregate deferred-release bookkeeping. Neither finding is claimed fixed. David's conditional approval has not been met.

| Question | Verdict |
|---|---|
| R1 | RISK |
| R2 | UNSOUND |
| R3 | UNSOUND |
| R4 | DEFECTS |
| R5 | PASS |
| R6 | REJECT |

## Scope, identity and evidence

Reviewed the full changed production implementation under tensor_cuda/ and tc_signal/, all plan functions, package wrappers/build files and the relevant tests/oracles; saved the full submitted diff and FIX2 delta in `artifacts/p1t_rev3/{full,fix2}.diff`. The independent code view preceded FIX1/FIX2 report reading. The approved core-DLPack/add-on mechanism matches this review; no scope mismatch.

The worktree HEAD is 10b6838e4158798f2554798db4d2b979acf5f217. All 46 submitted files byte-match5f9f955 (`source_inventory.json`); later HEAD metadata does not change the reviewed code. All 38 scratch compilation inputs match the working inputs (`build_identity.json`). Registration preceded gates and was not edited. No source/test/build file changed, no git write, no GPU execution, no subagent, no background job, no deletion or unrelated process kill.

Evidence classes: source reasoning, CPU native reproduction/UBSan, CPU compilation/link inspection, real fixture metadata comparison, read-only historical receipt audit. The lifecycle reproduction compiles an unchanged hash-pinned bindings.cpp:571-683 slice; it exercises the actual admission/GIL/wait mechanism without CUDA. The core helper probe regenerates current verbatim slices instead of recompiling old embedded FIX1 code. The aggregate probe calls actual Operand validation and note_push; its per-entry arithmetic statement is copied verbatim from the destructor. Neither probe is claimed as full rebuilt-core GPU execution.

## Closure table

CLOSED means the original defect is closed by the named source mechanism and finite evidence; it does not certify every GPU/runtime configuration. Deployment probes remain separately listed.

| Item | Status | Lines read and evidence |
|---|---|---|
| F01 | CLOSED | tc_signal/src/dlpack_operand.cpp:617-683,716-796; bindings.cpp:284-344; lstm_bindings.cpp:239-242; m4_bindings.cpp:239-355 — Event-fenced ownership on success/partial launch, synchronous fallback and deliberate leak without proof; source reasoning. Lead addon logs 12/12 correct per run; GPU probe pending. |
| F02 | CLOSED | tc_signal/src/dlpack_operand.cpp:196-201,314-323; m4_bindings.cpp:65-104; bindings.cpp:239-259; lstm_bindings.cpp:178-210 — Required guards and safe absent metadata; host_checks None cases clean TypeError, no allocator/launch. |
| F03 | PARTLY closed | tensor_cuda/src/bindings.cpp:306-330,750-820,828-843; tc_signal/src/checked_math.h:31-89; dlpack_operand.cpp:126-193,602,731 — Original rank/count/byte/address and plan overflow cases fixed. Aggregate retained bytes still overflow (V3-02 MINOR); metadata-only UBSan evidence. |
| F04 | PARTLY closed | tensor_cuda/src/bindings.cpp:596-643,653-682; tensor_cuda/tensor_cuda/__init__.py:178-201 — Finalizing/closing refusals and admitted-release drain work; finite timeout overrun still aborts (V3-01 MAJOR). |
| F05 | CLOSED | tensor_cuda/src/kernels.cu:221-264 — Drain switches to Storage device and restores caller; failures leak. Source closure; actual two-device probe remains pending. |
| F06 | CLOSED | tensor_cuda/src/bindings.cpp:451-486 — unique_ptr owns export context through vector/capsule construction, then transfers once. RAII source reasoning; no live bad_alloc injection of full core. |
| F07 | CLOSED | tensor_cuda/include/tc/core.h:80-88; docs/P1T_FIX1_M1_ADDENDUM.md — Documented coordinated rebuild prerequisite; closed as rollout documentation, not binary ABI compatibility. |
| N1 | CLOSED | tc_signal/src/dlpack_operand.cpp:471-518,573-582,644-683,768-837; tc_signal/tc_signal/__init__.py:139-155 — Original failed cudaGetDevice/empty-list silent success fixed: live CPU no-device sync raises named RuntimeError; destructor failures retained for next sync. Pending last-error-slot behavior remains lead probe. |
| N2 | CLOSED | tensor_cuda/src/bindings.cpp:306-330,440-443; kernels.cu:80-98; tc_signal/src/checked_math.h:63-89; dlpack_operand.cpp:158-168 — Zero-neighbor bound/bytes checked. CPU prefix/suffix refuse cleanly; native ck_census 18/18 with no UBSan diagnostic. |
| N3 | CLOSED | tc_signal/src/dlpack_operand.cpp:459-462,584-599,644-709,716-778,799-837; bindings.cpp:410-412 — No allocating take-all container; reserved owners/event recycle slots; caught queue push; import-time state. CPU oom_sync raises safely; deferred/push fault GPU gate pending. |
| N4 | CLOSED | tc_signal/src/dlpack_operand.cpp:237-261; tensor_cuda/src/bindings.cpp:627-634 — Catch per producer call, clear ownership before call; CPU throw_deleter survives. Core catch source-reviewed; full foreign-thread GPU path pending. Forced unwind is the N5 residual, not an ordinary C++ producer exception. |
| N5 | PARTLY closed | tensor_cuda/src/bindings.cpp:582-643,653-682; tensor_cuda/tests/test_dlpack.py:1089-1090,1192-1200 — Admission mutex/in-flight protocol closes ordinary race, but timeout returns before quiescence. Existing bounded regression exits via os._exit; V3-01 ordinary exit aborts. |

## New findings

| ID | Severity | File:line | Scenario | Fix |
|---|---|---|---|---|
| V3-01 | MAJOR | tensor_cuda/src/bindings.cpp:627-634,653-682; tensor_cuda/tensor_cuda/__init__.py:181-199 | An admitted finite 5.5 s deleter exceeds the default 5 s exit wait; normal finalization starts before it finishes and the process aborts. | Keep finalization behind admitted-release quiescence. Timeout diagnostics must not return to ordinary finalization with Python-dependent releases running; test normal exit without os._exit. |
| V3-02 | MINOR | tc_signal/src/dlpack_operand.cpp:602,731,774 | Individually validated near-limit operand byte totals overflow when added per entry or to the queue, potentially bypassing the bound drain. Synthetic metadata boundary, not resident-allocation/GPU reproduction. | Check aggregate bytes before launch; make queue arithmetic/bound comparisons overflow-safe without throwing in the destructor. |

### V3-01 — MAJOR: shutdown timeout resumes finalization with admitted releases still using Python

Locations: `tensor_cuda/src/bindings.cpp:627-634` (GIL acquisition, deleter, catch-all, GIL release), `:653-682` (bounded wait), `tensor_cuda/tensor_cuda/__init__.py:181-199` (logs timeout and returns). The existing bounded test uses `os._exit(0)` at `tensor_cuda/tests/test_dlpack.py:1089-1090`.

Scenario: a foreign-thread producer deleter enters under the actual admission protocol, then sleeps5.5 s while releasing the GIL. The default 5 s atexit wait returns with still_in_flight=1. A module finalizer permits normal interpreter teardown to proceed while yielding the GIL. When the finite deleter wakes, CPython 3.12 finalization terminates the foreign thread during GIL reacquisition. The forced unwind meets the catch-all/noexcept cleanup boundary and aborts the process. A warning or leaked manager cannot stop a callback already running.

CPU reproduction: `lifecycle_results.json` / `logs/p1t_rev3_lifecycle.log`, on CPython 3.12.3. Control delay 0.6 s/wait 1 s exits 0 and reports still_in_flight0. Delay0.6s/wait0.1 s and delay5.5 s/wait 5 s both exit -6. Default case verbatim:

```
HOOK {'in_flight_at_close': 1, 'still_in_flight': 1, 'timed_out': True, 'waited_us': 5000064}
FINALIZING True
FATAL: exception not rethrown
```

This is a finite callback, not an assertion about a producer that never returns. FIX2's bounded regression proves timing/reporting before immediate process exit; it bypasses normal finalization. The FIX2 report acknowledges a timeout residual, but that residual keeps F04/N5 open under this review's merge standard. The full real-Storage probe is `probes/core_exit_overrun.py`, pending for the lead; the CPU slice already settles the review finding.

Fix: stop admitting releases before finalization and do not let ordinary finalization start until every admitted Python-dependent release has left, including its GIL release. A timeout may report stalled releases but must not certify quiescence. If bounded process exit is a requirement, define an explicit process termination policy separately; it must not silently resume CPython teardown with a running callback. Add a finite-overrun regression that allows normal exit. Simply rethrowing a forced unwind still crosses a noexcept destructor; enlarging the timeout only moves the failing boundary.

Origin: incomplete FIX2 N5 closure, not an independently proved ordinary-operation regression. Prior art: CPython 3.12 GIL/finalization semantics and the standard admission-counter and quiescence rule; C++17 cleanup boundaries. Taken: those lifecycle constraints. This review contributes the finite-overrun normal-exit scenario. [CPython 3.12 lifecycle docs](https://docs.python.org/3.12/c-api/init.html), [pinned3.12.3 GIL source](https://raw.githubusercontent.com/python/cpython/v3.12.3/Python/ceval_gil.c).

### V3-02 — MINOR: aggregate retained-byte accounting is unchecked

Locations: `tc_signal/src/dlpack_operand.cpp:731` adds operand bytes to one entry; `:602` adds entry bytes to queue bytes; `:774` compares the result with the byte bound only afterward. Validation proves each operand byte count, not either aggregate addition.

Scenario: synthetic disjoint fp32 descriptors have shape `(2**61-1,)`, adjusted addresses 4 and 2**63, and 9223372036854775804 bytes each. Actual Operand validation and expect_disjoint accept them. The verbatim destructor expression then overflows when adding the second operand. Independently, actual note_push overflows when the queue contains 4 bytes and one entry adds9223372036854775804 bytes. UBSan exits 1 for each mode:

```
signed integer overflow: 9223372036854775804 + 9223372036854775804 cannot be represented in type 'long int'
signed integer overflow: 4 + 9223372036854775804 cannot be represented in type 'long int'
```

Receipt: `pending_bytes_results.json`, `probes/pending_bytes.cpp`. This establishes host arithmetic UB for the accepted metadata, not the existence of exabyte GPU arrays or a full kernel-launch reproduction. MINOR reflects that practical hardware reachability limit. Under wrapping execution, a negative accounting value can suppress the256 MiB bound check; ordinary representable aggregates work.

Fix: validate the aggregate operand byte count before launch, where refusal can raise ValueError safely. Handle queue headroom before adding and keep the bound comparison/accounting correct without allocating or throwing from the destructor. Do not merely insert a throwing ck_add inside noexcept cleanup. Origin: FIX1 deferred queue, retained by FIX2; not a new FIX2 delta regression. Prior art: C++17 signed overflow rules, GCC5(2015) checked arithmetic and UBSan diagnostics(2013+). Taken: checked addition/diagnostics; review contribution: aggregate boundary cases. [C++ expression rules](https://eel.is/c++draft/expr.pre).

## R1–R6 reasoning

R1 RISK: ordinary core allocation/free branches and existing kernel bodies are preserved; external fields default null, and existing revision/offset behavior is unchanged. FIX2 numel/stride helpers agree with baseline on representable supported shapes and replace overflowing products with exceptions. Nevertheless the new imported-Storage lifecycle has V3-01 and the native Storage ABI requires coordinated rebuilds. No claim of a newly failing baseline suite case.

R2 UNSOUND: manager ownership, used_* names, RAII export context, byte_offset, contiguous/singleton checks, dtype/flags, event fence and device-correct external teardown are structurally sound in their reviewed ordinary paths. Bare capsules carry the caller's stream-ordering obligation. Admission closes the ordinary shutdown race, but V3-01 prevents lifecycle soundness. GPU fence/offset/multi-device behavior is not executed here.

R3 UNSOUND at the required near-limit robustness boundary: consumed managers survive success and partial launches via events, fallback sync proves completion before release, and failed completion deliberately leaks. Entries are removed before reentrant deleters, with no queue reference carried across callbacks. Allocation failure is caught on queue push, recycle capacity is preallocated, and state is initialized at import. CUDA failures returned by query/sync are surfaced, and ordinary throwing deleters are contained per manager. V3-02 leaves aggregate arithmetic outside the checked contract. No ordinary-sized premature-release defect found; GPU OOM/pending/multi-device/last-error-slot cases remain probes, not guessed passes.

R4 DEFECTS for V3-02 aggregate arithmetic. All plan functions and kernel equations were read. Positive validated fp32 byte bounds also bound kernel pointer/index intermediates; convolution and LSTM cuBLAS casts are checked. Conv grouped SGEMM dimensions/layout, ConvTranspose gather tap condition, LSTM ONNX i/o/f/c gates/bias halves/reverse indexing, norm affine indices/centered variance/reduction barriers, exact integer nearest/reflect maps and rintf rounding match their contracts by source reasoning. Zero dimensions are intentionally refused by add-on; scalar rint is accepted. Extreme fp32 norm-statistic overflow is not broader finite-data certification. No new core __global__ kernel or modification to core round was introduced. Numerical/determinism probes remain finite GPU gates.

R5 PASS for separation and CPU build/link scope: fresh offline scratch add-on build succeeds with CUDA 12.6, GCC 13, CPython 3.12 and pybind11 v2.12.0. Core bindings pass C++ syntax compilation. The add-on's ldd/objdump links shared libcudart.so.12 and libcublas.so.12 at /usr/local/cuda-12.6/lib64; undefined symbols include no core dependency. Its default allocator imports core lazily, as approved; callback allocation removes that Python dependency. No core headers/sources compile into it. The two vendored DLPack headers differ in comments/examples/torch stable guard, not the consumed ABI layouts. Final core binary was not rebuilt here; promotion still requires clean native-client rebuild and final binary identity/link checks. [DLPack Python ownership](https://dmlc.github.io/dlpack/latest/python_spec.html), [array API stream contract](https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html).

R6 REJECT: V3-01 is an open MAJOR. Passing finite fixtures, historical regression IDs, or the source closure of other items cannot turn the unexercised timeout finalization into an approval. Correct it, verify normal full-path exit, rerun the unchanged relevant gates against rebuilt binaries, and review again.

## CPU receipts and baseline isolation

- `logs/p1t_rev3_build.log`: offline unchanged scratch add-on build PASS. Module SHA256 e865901c8296250735c9d53f72368d7496de2bc44158d68a56886cfa5191412d (`build_identity.json`). Core syntax log has no diagnostic. Link receipts: `logs/p1t_rev3_ldd.log`, `undefined_symbols.txt`, `objdump.txt`, build flags/link text.
- `host_checks.json`:20 subprocess checks each exit 0 with their expected acceptance/refusal; receipt collector exit 0 alone is not the gate. Normal empty accepts, prefix/suffix refuse, no-device sync raises, host OOM propagates and throwing native deleter is contained. `logs/p1t_rev3_ck_census.log`:18 cases,0 failures, no UBSan diagnostic.
- `operand_census.json`:17 host checks pass,7 consumed managers each deleted once. Unconsumed synthetic capsules intentionally have no destructor and their descriptor keepalives last to process exit; not an allocator/lifetime proof.
- `real_data_census.json`: live manifest139 cases,61 weight archives inspected via NPZ headers. `real_metadata_plans.json`:129 recorded fixture geometries match native plans. Ten Resize/Pad cases need data valued scale/pads and are covered by pending numerical probes, not counted as host geometry passes. No real numerical or Voice acceptance test executed.
- `lead_receipt_audit.json`: independently parsed historical lead JSON 236 PASS/236 cases and compared the lead combined log with the recorded baseline:52/52 exact FAILED/ERROR IDs,0 new,0 missing. These are receipts from the lead's other worktree; not this seat's live suite rerun or evidence for timeout-overrun safety. Lead addon logs each contain12 correct finite trials; ordinary/finalizing release logs pass their narrower scenarios.
- First real-metadata driver used wrong GroupNorm argument order; corrected probe only, failed log preserved. A census helper initially used a removed private NumPy header reader; corrected to public versioned header readers. No reviewed-code defect inferred from either harness mistake.

## Operand census

| Operand / operating condition | Evidence and remaining limit |
|---|---|
| Zero-size and0-d | Core helper normal empty pass; add-on zero refused and scalar rint planned (). Core CUDA scalar/empty roundtrip pending core_edges. |
| Near int64 limits, zero first/middle/last | Core prefix/suffix clean overflow refusal;18 native checked-math cases;17 operand census includes zero-neighbor cases. Aggregate bytes fail V3-02. |
| Non-zero byte_offset | Host planning accepts aligned offset4 and rejects offset2; actual data addressing remains core_edges/numerics probe. |
| Singleton unusual strides | Host accepts INT64_MIN stride on extent1 with compact stepped axis; source never steps singleton axes. |
| Read-only/versioned/copied | Versioned flags1/2/3 accepted for readers, refused for writers before launch; both consumed owners delete once on refusal. Core read-only refusal source-reviewed. |
| Capsule twice | Same-call duplicate and later used_* capsule refused; first consumed manager deletes once. Fresh exports of one producer may alias for reads; written overlap refused. |
| Non-default producer stream | Consumer requests stream1; source event fence correct, actual ordering/lifetime pending addon_lifetime and numerics. |
| Producer deleter raises | Actual native throwing ManagedOwner reset survives; core per-call catch reviewed; GPU6-path regression gate pending. Cancellation/forced unwind is not an ordinary C++ exception and remains tied to V3-01. |
| Host release allocation failure | Actual cpu_native oom_sync safely errors; caught pending push, reserved recycle/owner capacity, fixed sync set reviewed. Real queue OOM sweep pending release_failures. |
| Pending exit | Add-on atexit drains with GIL before finalization; failing sync logs and leaks. Existing lead gate narrow; pending-exit release suite listed. |
| In-progress foreign-thread exit | Actual lifecycle slice control completes; finite wait overrun aborts V3-01. Full imported Storage probe pending. |
| CUDA errors present at synchronize | No-device query error reproduced and named. Returned sync/fallback/bound failures source-reviewed and regression probe listed. Preexisting launch last-error-slot behavior unresolved; cuda_errors_multidevice explicitly decides it. |
| Multiple pending devices | Fixed-capacity per-device sync report, event queries and device guards read; no physical multi-device execution. Two-device probes pending; single device is inconclusive. |

## Probe list

Every GPU-marked script is unexecuted by this seat. Run sequentially under the lead's GPU ownership/lock policy with bounded foreground timeout; inspect actual assertions, not just collected JSON. No approval depends on assuming these pass.

| Path under artifacts/p1t_rev3/ | State | Pass/fail decision |
|---|---|---|
| probes/host_checks.py | CPU executed | Clean None/overflow/refusal/scalar checks, duplicate-capsule refusal, no-device sync error; dispatches core_cpu and cpu_native.20 cases rc0. |
| probes/core_cpu.cpp | CPU executed | Current verbatim core helpers/empty: normal empty succeeds; prefix/suffix overflow refuses without UBSan.3 cases rc0. |
| probes/cpu_native.cpp | CPU executed | Real release-path host allocation failure and throwing deleter do not abort.2 cases rc0 (included in host20). |
| probes/lifecycle_timeout.py | CPU executed | Finite admitted deleter must complete before normal finalization, exit 0.Uses lifecycle_slice.cpp. Control rc0;0.1 s and default 5 s overrun rc-6. |
| probes/pending_bytes.cpp | CPU executed | Per-entry and queue-byte accumulation must avoid signed overflow for individually validated descriptors. Both modes rc1 under UBSan; entry statement copied verbatim, note_push actual source. |
| probes/operand_census.py | CPU executed | Offsets/singleton strides/0-d/read-only-copied flags/duplicate consumption/zero-neighbor overflow obey contract;17 cases pass,7 consumed deleters called once. |
| probes/real_metadata_plans.py | CPU executed | Recorded real fixture output geometry matches reviewed plans;129 cases pass. Metadata only, placeholders never launched. |
| probes/core_exit_overrun.py | LEAD GPU ONLY, pending | Full real imported Storage + foreign pthread, during and bounded modes: ordinary exit 0, no abort/hang, finite 5.5 s deleter across default 5 s wait. No os._exit escape. |
| probes/addon_lifetime.py | LEAD GPU ONLY, pending | 12 finite non-default-stream last-owner trials: all outputs correct and every launch increments retained entries. No universal race-freedom claim. |
| probes/core_device_drain.py | LEAD GPU ONLY, pending | Two devices: producer deleter observes device0 work complete after drop on current device1; caller device restored. Fewer than2 devices => exit3 inconclusive. |
| probes/core_edges.py | LEAD GPU ONLY, pending | Reject rank/count/byte wraps; scalar/empty/offset/singleton-stride interop and stream0 refusal obey contract. |
| probes/numerics_and_streams.py | LEAD GPU ONLY, pending | Finite conv/LSTM determinism and parity, handle settings, L=1 norm, special rint, nonlegacy producer ordering; all assertions pass. |
| probes/cuda_errors_multidevice.py | LEAD GPU ONLY, pending | Two-device FIFO releases once, outputs correct, caller device restored; synchronize raises for staged preexisting launch error. Exact runtime last-error behavior unresolved until run; fewer than2 devices => exit3. |
| probes/release_failures.py | LEAD GPU ONLY, pending | All existing FIX1/FIX2 add-on regressions execute without skips, including partial launch, queue OOM, event recycle, fallback/bound failures, throwing deleters and pending exit. Exit0 plus nonempty passing suite required. |

Support files: `probes/build_cpu.sh` rebuilds the unchanged scratch add-on and host drivers under /tmp/p1t_rev3_build with foreground timeouts; `probes/lifecycle_slice.cpp` is the pinned unchanged lifecycle body plus CPU dispatcher; `probes/release_worker.cpp` is the optional round1 native consumer helper (not needed by core_exit_overrun's pthread staging). CPU commands: `PYTHONPATH=/tmp/p1t_rev3_build/tc_signal CUDA_VISIBLE_DEVICES= timeout 90 python probes/host_checks.py`; `timeout 30 /tmp/p1t_rev3_build/native_harness ck_census`; `PYTHONPATH=/tmp/p1t_rev3_build timeout 15 python probes/lifecycle_timeout.py default`; `timeout 15 /tmp/p1t_rev3_build/pending_bytes [queue]`. Prefix paths with artifacts/p1t_rev3/. The per-entry aggregate statement is pinned to this reviewed revision; adapt the driver when reviewing a changed byte-accounting implementation.

## Prior art

Review constraints and suggested repairs use established standards/patterns: dmlc DLPack(2017+, vendoredv1.3), Python array API(2021–2023 stream negotiation), NVIDIA CUDA 12.6 stream/event last-use semantics, CPython 3.12.3 finalization/GIL behavior, C++17 signed arithmetic/noexcept/RAII and GCC5(2015) checked additions. The original kernels use the cited ONNX Conv/ConvTranspose/LSTM/normalization/Resize/Pad/Round equations; this review proposes no new kernel algorithm. Test tooling uses GCC/Clang UBSan and the earlier round's native metadata/lifecycle harnesses. The review's contributions are adversarial finite timeout-overrun and aggregate-byte scenarios plus real metadata comparison, not novel algorithms.

Primary references checked: [CUDA 12.6 stream API](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-runtime-api/group__CUDART__STREAM.html), [CUDA 12.6 event API](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-runtime-api/group__CUDART__EVENT.html), [CPython 3.12.3 finalization source](https://raw.githubusercontent.com/python/cpython/v3.12.3/Python/pylifecycle.c), [ONNX LSTM](https://onnx.ai/onnx/operators/onnx__LSTM.html), [ONNX GroupNormalization21](https://onnx.ai/onnx/operators/onnx__GroupNormalization.html). No product code comment added because source edits are forbidden; prior-art annotations appear at the authored probe sites, ledger, each finding and this report.

## Residuals and hygiene

No broad GPU correctness/performance/Voice tolerance claim. Free-threaded/subinterpreter certification remains outside the submitted GIL-based contract. The core Storage rebuild prerequisite is documented, not waived. Late deliberate leaks preserve safety only for releases not yet admitted or without GPU completion proof; they cannot cure V3-01's running callback. No unrelated baseline 52 failure is attributed to this change.

Scratch retained, nothing deleted: /tmp/p1t_rev3_build/ (unchanged add-on copy, build and host binaries), /tmp/p1t_rev3_write_report.py. Deliverables are only artifacts/p1t_rev3/**, logs/p1t_rev3_* and docs/P1T_REV3_*.md. Final source inventory recheck required before handoff.
