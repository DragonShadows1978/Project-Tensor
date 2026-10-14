# P1T-REV7 independent review

**REJECT `9df5bef..429446d`: 0 BLOCKING, 1 MAJOR, 1 MINOR, 0 NIT open.** V7-01 aborts ordinary interpreter exit when conversion of a documented diagnostic-delay knob raises OverflowError. FIX6 exposes this add-on regression by removing synchronize's extra wait; the core hook has the same bypass. V7-02 misattributes a completion-event query failure to a successful stream sync. David's conditional merge approval is not met. No defect is waived or repaired by this review.

| Question | Verdict |
|---|---|
| R1 | SAFE |
| R2 | UNSOUND |
| R3 | UNSOUND |
| R4 | CORRECT |
| R5 | PASS |
| R6 | REJECT |

## Scope, evidence and baseline isolation

Evidence classes: source reasoning; fresh CPU full-add-on reproduction/build; exact-source host traces/slices; finite UBSan arithmetic/ownership tests; historical lead-receipt audit; primary API documentation. No GPU execution, Voice acceptance rerun, performance measurement, or general certification. Checkout HEAD is6a9be0d87edabec3f849260263398a0d7d66ce42; product diff versus429446d is empty. Saved full.diff and fix6.diff cover the requested ranges. Product source was read before sweep/U/N verdicts; initial mechanism-summary reading verified scope.

R1 SAFE concerns existing9df5bef engine behavior, not the new shutdown/interop defect. Checked shape helpers preserve representable values; overflow refusal replaces UB. No existing binding was rebound in FIX6; the public synchronize addition reports only new P1-T deferred failures. The original lead combined-session log independently matches the52 baseline failed ids. Its Voice JSON contains249 PASS cases; actual mutant logs show6failed,30failed/17passed,7failed;184 fixed-build tests passed. These are historical audited receipts (lead_receipt_audit.json), not our own GPU runs, and do not cover V7-01/02.

R4 CORRECT is a source/finite-host arithmetic judgment within the stated contract. Conv/GEMM layouts, transposed gather tap/weight indexing, ONNX LSTM i/o/f/c direction/state indexing, norm affine/row/tree mappings, word-copy upsample/pad and rintf were read. Every grid-stride total is byte-bounded well below INT64_MAX, leaving room for ceiling/increment arithmetic; im2col cuBLAS dimensions and LSTM sizes checked before narrowing. fill_ uses checked wide blocks and refuses excess before revision/write. GPU numerics/ordering are lead probe work. Existing core operations unchanged by P1-T remain baseline.

R5 PASS: fresh unchanged add-on copy built under/tmp/p1t_rev7_build, core bindings syntax-checked. CMake/native includes and nm/ldd/objdump show no TensorCUDA linkage/headers/symbol dependency; shared libcudart.so.12 and cuBLAS only. Actual host RUNPATH is/usr/lib/x86_64-linux-gnu, the detected toolkit layout here. Default Python allocator imports core at call time; exchange is DLPack-only. Core native ABI clients still require a coordinated clean rebuild (F07); no old-ABI compatibility claim.

## Closure table

Aliases: C=`tensor_cuda/src/bindings.cpp`; K=`tensor_cuda/src/kernels.cu`; O=`tc_signal/src/dlpack_operand.cpp`; A=`tc_signal/src/bindings.cpp`; L=`tc_signal/src/lstm_bindings.cpp`; M=`tc_signal/src/m4_bindings.cpp`; CM=`tc_signal/src/checked_math.h`; CI=`tensor_cuda/tensor_cuda/__init__.py`; SI=`tc_signal/tc_signal/__init__.py`.

CLOSED means the earlier concrete mechanism remains closed in its original scope. PARTLY closed preserves the working native/default-path fix while naming the new scenario that defeats its guarantee.22 CLOSED,3 PARTLY closed,0 wholly OPEN. F04's late-entry finalizing check remains correct; N5/V3-01/V5-01's broader before-finalizing guarantee is not complete under V7-01. V6-01's original ordinary-sync deadlock is closed, despite the new hook-error regression exposed by its removal.

| Item | Status | Lines read and evidence |
|---|---|---|
| F01 | CLOSED | O:1160-1257; A:280-287; L:232-244; M:236-244,289-355. Prelaunch reserves, event retention, checked fallback and leak without completion proof intact. |
| F02 | CLOSED | O:484-488,618-627; A:241-264; L:181-211; M:66-105. Required None refusals and absent-metadata guard intact; CPU census/argument refusals pass. |
| F03 | CLOSED | C:322-346,1060-1111,1138-1188; O:414-480; CM:31-89; A:100-123,174-187. Checked rank/count/byte/address and geometry paths intact; UBSan census passes. |
| F04 | CLOSED | C:829-879; O:193-227. Original finalizing-entry GIL refusal/leak remains. Admitted-before-finalizing coordination is separately PARTLY under N5/V5-01. |
| F05 | CLOSED | K:286-308,447-473. Imported Storage drains its own device, checks restore, leaks on failed completion proof, records failures. FIX6 adds reporting without altering release order. |
| F06 | CLOSED | C:635-674. Export context stays under unique_ptr through metadata/capsule allocation, then transfers ownership once. |
| F07 | CLOSED | tensor_cuda/include/tc/core.h:80-92; docs/P1T_FIX1_M1_ADDENDUM.md:17-29. Coordinated native rebuild remains required; add-on includes/links no core ABI. |
| N1 | CLOSED | O:1026-1069,1354-1433. Query/sync/prior/deferred failures raise, leak only without proof; 8 fresh host trace cases pass. V7-02 is distinct incorrect attribution. |
| N2 | CLOSED | C:322-346,619-621; K:80-97; CM:63-89; O:440-480. Zero-neighbor extent/byte bounds checked; 18-case UBSan harness and operand census pass. |
| N3 | CLOSED | O:924-938,1072-1095,1160-1257,1354-1433. Pre-reserved owners/event slots, caught push failure, nonallocating drains and persistent OOM report records intact; CPU OOM propagation passes. |
| N4 | CLOSED | O:533-565; C:861-868. Ordinary C++ deleter exceptions contained, owner cleared before callback, never retried; fresh native throwing-deleter test passes. |
| N5 | PARTLY closed | C:840-970; CI:217-240. Native admission/unbounded wait correct, default control passes; diagnostic conversion can skip the hook entirely (V7-01). |
| V3-01 | PARTLY closed | C:899-954; CI:217-222. Native wait has no bound and handles own-thread exclusion; configured diagnostic OverflowError bypasses that wait (V7-01). |
| V3-02 | CLOSED | O:961-995,1147-1169,1253-1257. Checked aggregate and sticky saturated total preserve bounds; 17-case accounting UBSan harness passes. |
| V4-01 | CLOSED | C:426-485,624-633; K:431-444. Allocation-label proof and tensor-device fence/destroy/restore preserved. Physical two-device ordering remains lead-only. |
| V4-02 | CLOSED | CI:98-116,201-213. Import documentation correctly distinguishes unbounded wait from diagnostic-delay knob. V7-01 concerns exception handling, not the old bounded-wait text. |
| V4-03 | CLOSED | C:1039-1051,1116-1129. Device query checked before Storage ownership/rename; no FIX6 semantic regression at these sites. |
| V4-04 | CLOSED | O:1357-1367,1378-1383,1421-1433. Prior slot captured before drain and retained until exception exists; fresh prior_oom passes. |
| V5-01 | PARTLY closed | O:217-248,324-354,533-547,1119-1137; SI:186-209. All native callback kinds admitted and default hooks wait; FIX6 exposes diagnostic-conversion bypass (full add-on rc134; V7-01). |
| V5-02 | CLOSED | O:1260-1298,1409-1433; tc_signal/tests/test_fix5_findings.py (query regression); fresh query_once PASS. Non-NotReady drain failure recorded before clearing and raised once. |
| V5-03 | CLOSED | tc_signal/src/conv.cu:84-128; tc_signal/src/lstm.cu:107-146; tc_signal/tests/test_fix5_findings.py:894-919. Scoped handle ownership survives cache-insertion OOM; regenerated CPU conv/LSTM creates=destroys=1. |
| V5-04 | CLOSED | K:580-617; tensor_cuda/tests/test_dlpack.py:1935-1969. Wide grid calculation/refusal before launch/revision; regenerated CPU huge-grid witness passes. |
| V6-01 | CLOSED | O:304-319,1119-1123,1354-1367; tc_signal/tests/test_fix6_findings.py:154-164,362-373. Ordinary sync no other-thread wait; fresh full-addon lock witness and trace late-release pass. |
| V6-02 | CLOSED | C:542-598,1256-1354,1515-1521; tensor_cuda/tests/test_dlpack.py:2184-2209,2296-2303. C-level reads/refusal replace user conversion; actual-dispatch host census passes, no methods called. |
| V6-03 | CLOSED | O:734-765,830-831,1363-1367,1415-1433; tc_signal/tests/test_fix6_findings.py:404-487. Drain failures persisted allocation-free before message, cleared after captured exception; all3 fresh diagnostic_oom probes pass. |

## New findings

| ID | Severity | File:line | Title |
|---|---|---|---|
| V7-01 | MAJOR | tc_signal/tc_signal/__init__.py:194-209; tensor_cuda/tensor_cuda/__init__.py:217-240 | Diagnostic-delay conversion failure bypasses mandatory shutdown quiescence |
| V7-02 | MINOR | tc_signal/src/dlpack_operand.cpp:1048-1058; 1389-1392; SyncReport:734-759 | Completion-event query failure is reported as a failed stream synchronization |

### V7-01 — MAJOR OPEN: Diagnostic-delay conversion failure bypasses mandatory shutdown quiescence

Location: `tc_signal/tc_signal/__init__.py:194-209; tensor_cuda/tensor_cuda/__init__.py:217-240`.

Scenario: Set the documented tc_signal._RELEASE_EXIT_DIAG_S to 1 << 4096. A daemon thread is inside a finite producer deleter (read-only output refusal before any launch); at ordinary exit float(knob) raises OverflowError. Only TypeError/ValueError are handled by the fallback, so the outer handler logs and returns without _exit_close_admission. Finalization overtakes the deleter and CPython 3.12 forced unwind aborts through the noexcept/catch-all release frame. Core has the same bypass for _DLPACK_SHUTDOWN_WAIT_S. This does not unregister or replace the shutdown hook.

Evidence: Fresh unchanged full add-on built in /tmp/p1t_rev7_build; logs/p1t_rev7_knob_exit.log: OverflowError: int too large to convert to float; FINALIZING True; FATAL: exception not rethrown; rc134. Default-knob control logs/p1t_rev7_knob_exit_control.log rc0. Core counterpart source-confirmed; additional probe in progress. Core actual-source lifecycle + actual Python hook also rc134, logs/p1t_rev7_core_knob_exit.log; default control rc0, logs/p1t_rev7_core_knob_exit_control.log. Full core lead probe pending; this is host slice evidence, not a full-core CUDA run.

Fix: Make close-and-wait unconditional with respect to diagnostic configuration: validate/clamp the knob before shutdown or handle conversion failures with an already available safe default and still call the raw close function. At minimum cover OverflowError; cover other ordinary conversion exceptions too. Add finite-callback shutdown regressions for both hooks.

Prior art: CPython 3.12 (PSF2023) float overflow, atexit-before-finalizing and GIL/forced-unwind lifecycle; DLPack (DMLC2017) producer lifetime; C++17 noexcept. Taken: lifecycle invariant and REV5 finite-callback staging; contribution: diagnostic-only configuration failure bypass.

### V7-02 — MINOR OPEN: Completion-event query failure is reported as a failed stream synchronization

Location: `tc_signal/src/dlpack_operand.cpp:1048-1058; 1389-1392; SyncReport:734-759`.

Scenario: The phase1 legacy-stream synchronization succeeds. A one-shot cudaEventQuery in phase2 fails; sync_and_release_all_impl puts its error into the same untyped SyncReport.dev/err arrays as stream errors. synchronize then says the legacy stream could not be synchronized. The failing API is absent and the named sync actually succeeded, misleading diagnosis. The operand is conservatively leaked and the error retained/raised, so this is a diagnostic defect, not a lifetime failure.

Evidence: Actual-source CPU trace; logs/p1t_rev7_query_diagnostic_rc.log: successful sync devices [0], then message says legacy stream of device 0 could not be synchronized; probe exits1. No GPU execution.

Fix: Carry an allocation-free failure-kind tag (stream sync vs event query) in SyncReport and preserve it through merge; build the diagnostic from that tag, naming cudaEventQuery and the affected device for query failures. Add a test distinguishing the two sites.

Prior art: NVIDIA CUDA12.6 (2024) cudaEventQuery and cudaStreamSynchronize distinct return-code contracts; ordinary site-specific error reporting, no specific prior art known to me. Taken: API attribution; contribution: successful sync followed by one-shot query failure witness.

V7-01 baseline distinction: full add-on before binary SHA3324bb4a…2552 matches the lead's acd42fa receipt; identical overflow scenario exits0 there because the old synchronize waits for the finite callback first (logs/p1t_rev7_knob_exit_before_fix6.log). Fresh FIX6 addon aborts134, default control0. This is not removal/replacement of a hook, an infinite producer, cancellation, GPU execution, or a caller-held-lock contract violation. The knob controls logging delay only; even unusable configuration is explicitly intended to fall back and still wait. Other ordinary conversion exceptions can take the same outer bypass; only OverflowError is needed for the demonstrated defect. Core's bypass predates FIX6 within P1-T; its exact host source/Python-hook witness also aborts134/control0. Full core probe is pending lead.

V7-02 is an existing P1-T diagnostic defect, not a new FIX6 arithmetic/lifetime regression. It retains/raises the error and leaks conservatively, so severity is MINOR. No catastrophic outcome is inferred from the inaccurate message.

## Sweep-table check (g)/(h)

All27(g) and14(h) rows checked. No omitted native P1-T binding or wait primitive found. Missing analysis: diagnostic conversion error exits at both Python hooks (g13/g26,h2/h9), V7-01. h8 rationale corrected for native DLPack consumer threads; g17 count corrected to18 plan/run bindings. V7-02 is an error-attribution defect outside classes(g)/(h).

| Row | Verdict | Sites / reasoning |
|---|---|---|
| g1 | CONFIRMED | C:542-598,1515-1521; export C-level argument reads; core CPU census. |
| g2 | CONFIRMED | C:1526-1531,1298-1301; self type check before native cast. |
| g3 | CONFIRMED | C:1532-1539; data_ptr self/arity C-level. |
| g4 | CONFIRMED | C:1540-1546,1312-1320; fill real read; typed native kernel unchanged. |
| g5 | CONFIRMED | C:1562-1569; mark_modified self checked. |
| g6 | CONFIRMED | C:1570-1575; property getter wraps args/kwargs explicitly. |
| g7 | CONFIRMED | C:974-1130,2328-2333; capsule-only native import; no producer method called. |
| g8 | CONFIRMED | C:1334-1354,2334-2343; shape/container/string reads bypass subclasses; CPU census. |
| g9 | CONFIRMED | C:2345-2388; zero-arg stats reject extra args before body. |
| g10 | CONFIRMED | C:1383-1457,2404-2405; raw parser; no Python object owned across native wait. |
| g11 | CONFIRMED | C:2391-2401; new raise binding uses zero-param CoreCallArgs; exact source registration tested. |
| g12 | CONFIRMED | CI:137-151; producer calls in Python frame; native capsule import C-only. |
| g13 | QUALIFY: V7-01 | CI:217-222: Python-only conversion is correctly classified, but its OverflowError path bypasses mandatory wait. Original OK cannot establish hook safety. |
| g14 | CONFIRMED NATIVE MECHANISM | C:840-879; admission brackets GIL acquire/deleter/release. Native mechanism remains right; Python hook reachability fails under V7-01. |
| g15 | CONFIRMED | C:351-384; own export context C++ cleanup; imported last Storage delegates admitted g14. |
| g16 | CONFIRMED | C:1290,1515-1575; borrower/caller retains args; no unadmitted last foreign-object drop found. |
| g17 | CONFIRMED; COUNT CORRECTED | A:392-447; L:251-276; M:360-461. There are18 plan/run bindings (9 ops times2), not20; with g18 probes the count is20. All actual bindings accounted for. |
| g18 | CONFIRMED | A:448-461; M:463-472; probe/test body operands routed through to_capsule; no-match repr removed. |
| g19 | CONFIRMED | A:462-477; L:277-292; device C-level int32 read; positional/keyword census. |
| g20 | CONFIRMED | A:478-503,504-519,528-558; M:473-488; stats/test-name/zero-arg checks C-level; all native functions enumerated. |
| g21 | CONFIRMED | A:352-379,522-524; py_args.h:203-248; raw exit argument reader. |
| g22 | CONFIRMED | O:356-399; CallbackSection constructed before attribute lookups, casts and locals; locals die before section leaves. |
| g23 | CONFIRMED | O:533-565; manager cleared first, admitted producer deleter, ordinary exceptions contained. |
| g24 | CONFIRMED | O:324-354; every last capsule drop including used capsules admitted/kept. No native site missed. |
| g25 | CONFIRMED | A:198-206,250-252; L:194-196; M:215-223; CapsuleRef owns the other capsule ref through tuple construction/throws. |
| g26 | QUALIFY: V7-01 | SI:194-209: conversion is in Python, but unchecked ordinary conversion failure skips the raw close. Other wrappers stay Python-only as claimed. |
| g27 | CONFIRMED | All29 add-on bindings + P1-T core registration use args/kwargs or raw waits. Native conversion/error matching census1067 cases, zero Python conversion calls. |

| Row | Verdict | Sites / reasoning |
|---|---|---|
| h1 | CONFIRMED FIXED | O:1354-1367. Cross-thread release wait absent from ordinary sync; lock and late-release probes PASS. |
| h2 | PARTLY: V7-01 | O:1119-1137,268-319 native exit wait correct; SI:194-209 can skip it. No-lock contract justified but reachability was not checked. |
| h3 | CONFIRMED | O:294-298 polling fallback waits only at exit, atomics predicate, no callbacks under mutex. |
| h4 | CONFIRMED | O:315-317; mutex gone before restore; raw exit caller owns no Python reference across wait. |
| h5 | CONFIRMED | O:205-214; leave critical section empty; waiter drops mutex during wait/diagnostics; no GIL/callback dependency under lock. |
| h6 | CONFIRMED | O:845-852; actual stream completion required; no host function launched by P1-T. N5 app-lock caveat correct. |
| h7 | CONFIRMED | C:845-855,873-879; no GIL acquisition/callback under lifecycle mutex. |
| h8 | CONCLUSION OK; REASON CORRECTED | C:359-364,840-868; K:447-473. A native DLPack consumer may drop an exported context without the GIL and release imported Storage, so PyGILState_Ensure is not always reentrant. No P1-T mutex held there; DLPack explicitly requires GIL acquisition for Python APIs. App-finalizer lock issues fall under corrected N6. |
| h9 | PARTLY: V7-01 | C:923-954 native cv/GIL ordering correct; CI:217-222 diagnostic OverflowError skips close/wait. |
| h10 | CONFIRMED | C:2357-2361; lifecycle stats mutex cannot be held by a thread waiting for GIL. |
| h11 | CONFIRMED | K:255-275,379-417; record mutex encloses no callback/GIL wait. New report/ack paths keep snapshot arithmetic serialized. |
| h12 | CONFIRMED | K:286-308; own-device Storage stream drain is necessary; external app callbacks subject to N5. |
| h13 | CONFIRMED | C:465-473; stream event dependency device-side; no host wait for another producer callback. |
| h14 | CONFIRMED WITH LEAD N6 CORRECTION | O:997-1004; C:861-868. Callback runs on release caller, possibly different from original submitting thread. Arbitrary-finalizer app-lock issue, not FIX5 unrelated-thread wait. |

Search census: all P1-T registrations, C-API conversions, producer attribute calls, cast/len paths, capsule/reference drops, explicit SaveThread/RestoreThread, lifecycle/record mutex users, cv/poll waits, stream synchronizations and event fences were enumerated. Unmodified baseline checkpoint/getitem/terrain and other engine binding callbacks were excluded; no new P1-T routing to their conversion code was found. h8's statement that every drop already holds the GIL is too strong: native DLPack consumers can call exported deleters without it. The mechanism still acquires with no P1-T mutex held; app-finalizer locks follow N6. This is a reasoning correction, not a new code defect.

## User-visible changes U1–U9

| Item | Verdict | Reason / evidence |
|---|---|---|
| U1 | ACCEPTABLE | Ordinary sync no longer waits on another thread; release stays on that thread. Correctly removes caller-lock inversion. O:1354-1367; fresh lock/late probes. |
| U2 | ACCEPTABLE | DLPack typed protocol permits refusing conversion-only objects/numpy bool/int scalars; exact values/subclasses read safely. C:542-598; CPU census. |
| U3 | ACCEPTABLE | All new native P1-T parameters use C-level matching/read/refusal. Public wrappers convert in Python. No existing baseline binding rebinding; 1067 CPU checks. |
| U4 | ACCEPTABLE | Retain drain error through reporting OOM and raise on retry; O:1363-1433; three trace probes pass. |
| U5 | ACCEPTABLE | Core deferred errors now have a public reporting path and exit logging; C:1466-1493; K:391-417; CI:1371-1383. Semantics intentional; V7-01 is separate hook reachability. |
| U6 | ACCEPTABLE | Close before waiting admits no new callbacks; later releases deliberately leak at process exit. Native mechanism correct; open V7-01 must be fixed for Python hook to reach it. |
| U7 | ACCEPTABLE | Consumed-capsule destructor remains producer code; admit last reference or keep after close. O:324-354. No manager double-release. |
| U8 | ACCEPTABLE | Raw CPython wait functions retain no dispatcher-owned Python objects across GIL release. C:1383-1457; A:352-379; py_args.h:203-248. |
| U9 | ACCEPTABLE | Two bounded one-shot test hooks permit real-source regression testing; defaults unarmed, allocation-free flag consumption. O:1031-1033,1048-1049,1263-1264. |

## FIX6 Not-a-defect items N1–N9 (including lead corrections)

These are FIX6 table labels, distinct from round-2 N1–N5 closure IDs above. All nine dispositions are correct within the lead's corrected scope; none waives V7-01/02.

| Item | Verdict | Reason / evidence |
|---|---|---|
| N1 | CORRECTLY NOT A DEFECT | Producer-initiated pthread_exit/cancellation violates returning C deleter contract; independent finite callback is V7-01, not cancellation. O:533-547; C:861-868; vendored dlpack.h:397. |
| N2 | CORRECTLY NOT A DEFECT | A nonreturning producer prolongs documented exit-only quiescence. Unbounded waiting is necessary; V7-01 is a different accidental skipped wait. |
| N3 | CORRECTLY NOT A DEFECT | Ordinary count wait removed. Native exit closes admission before counting, so later releases back out. O:217-227,1119-1123. |
| N4 | CORRECTLY NOT A DEFECT | Explicit unregister deliberately disables the documented hook; finalizing-entry checks remain. Repro V7-01 leaves hooks registered. |
| N5 | CORRECTLY NOT A DEFECT | No P1-T host function enqueued. App callback/app lock order remains app responsibility; lead CUDA host-function citation checked; stream waits are required for memory safety. |
| N6 | CORRECTLY NOT A DEFECT | Accept lead correction: global FIFO lets another calling thread run a deleter; finalizers may run under arbitrary app locks. O:837-839,997-1004; README:157-160; Python finalizer docs. |
| N7 | CORRECTLY NOT A DEFECT | Environment/performance limits are evidence boundaries, not implementation defects. Lead historical mutant/249-case/baseline claims independently audited; no new GPU run claimed. |
| N8 | CORRECTLY NOT A DEFECT | With lead correction: undeclared free-thread support reenables GIL; own-GIL interpreters reject single-phase modules; legacy shared-GIL mixing with PyGILState is already unsupported in baseline core. No free-thread/subinterpreter certification. |
| N9 | CORRECTLY NOT A DEFECT | Former R7 was a defect and is fixed by deferred raise/ack; K:255-275,391-417; C:1466-1493; CI:1371-1383. Not a waiver. |

## Operand census and limits

| Operand / condition | Evidence / boundary |
|---|---|
| Zero-size and0-d | Core valid scalar/empty arithmetic; add-on empty refused, scalar rint accepted;17-case actual-addon metadata census. Core GPU roundtrip still lead-only. |
| int64 extremes / zero neighbors |18-case extent/byte UBSan census and17-case queue accounting pass; partial/kernel pointer products reasoned from validated byte bounds. |
| Nonzero byte_offset / singleton strides | Full-addon plan offset4 accepted/offset2 refused; INT64_MIN singleton stride accepted; core byte-offset/total/address checks read. Data-content GPU checks pending. |
| Read-only/versioned/copied capsules | Supported readers accept flags; writers refuse read-only/copied; core read-only refused before ownership; census once-only ownership. |
| Twice-consumed capsule / same operand twice | Same bare capsule twice refuses; first consumed manager once. Fresh producer capsules may share reader memory; all written/scratch overlaps rejected. |
| Nondefault producer streams | stream1 acquisition and tensor-device export event/wait ordering read; real forward/lifetime/stream assertions pending lead. |
| Raising deleters | Ordinary C++ exception contained once (fresh native probe). Producer-initiated thread exit/cancellation contract excluded; finite ordinary producer exit failure is V7-01. |
| Host allocation failure | Actual-source owner/event pre-reserve and nonallocating drain reviewed; native OOM propagation,3 reporting-OOM traces,handle-cache insertion OOM pass. No GPU OOM sweep run here. |
| Unusual argument types / both call styles |1067 native checks: all29 addon bindings, all typed positions positional/keyword; int/float/tuple/list/str subclasses, NumPy scalar types, arbitrary conversion/representation methods. Core actual dispatch slice covers export/empty/raw shutdown/report; fill/self/property sites source-checked, full GPU regression probe supplied. |
| Exit with callbacks/conversions on other threads | Default admitted callback controls pass; native argument methods refused. Documented delay conversion exception skips close:V7-01 rc134. |
| Sync while other-thread release running | Full-addon no-GPU lock inversion and actual-source later release tests pass; no ordinary other-thread wait. |
| Prior/one-shot CUDA errors at sync/op/fill/export | Prior_oom/query_once/diagnostic_oom host traces pass; source checked prelaunch capture/device guards/export raises. Error-attribution defectV7-02. Real CUDA/shim/fill/export checks pending lead. |
| Multi-device pending entries / different current device | Actual-source trace records [1,0,2,0] device syncs/restores. Own-device fences/guards/drains read. Physical2-GPU probe exits3 on lead1-GPU; one-GPU interposed checks in fix56. |

## CPU receipts and probes

Fresh add-on compile and core syntax PASS (logs/p1t_rev7_build.log,p1t_rev7_cpu_compile.log);14 checks in cpu_results.json all rc0; argument_census.json1067/zero calls; operand_census.json17 cases/7 once-only consumed managers. Regenerated handle/fill probes PASS. V7-01/02 are negative results with separate receipts, not counted as passing gates. Controls and before-arm inputs use the same scenario; binaries/source slices hashed. No skipped test is a pass.

| Probe path / lane | Decides / rule / state |
|---|---|
| artifacts/p1t_rev7/probes/knob_exit_cpu.py | Actual full add-on diagnostic-overflow shutdown: pass iff child exits0 and finite deleter finishes before finalization; --control default. FIX6 rc134, control0, pre-FIX6 same input0. CPU RUN. |
| artifacts/p1t_rev7/probes/core_knob_exit_cpu.py | Actual-source core lifecycle + actual Python hook diagnostic overflow: same normal-exit rule; current rc134/control0. Host slice, not full core. CPU RUN. |
| artifacts/p1t_rev7/probes/core_knob_exit_gpu.py | Full rebuilt core V7-01: finite imported-manager deleter before finalization, child0; one GPU, unavailable3. LEAD PENDING. |
| artifacts/p1t_rev7/probes/query_diagnostic_cpu.py | V7-02 actual-source phase2 query attribution after successful phase1 sync: must name cudaEventQuery, no false sync failure. Current rc1. CPU RUN. |
| artifacts/p1t_rev7/probes/query_diagnostic_gpu.py | V7-02 real module/pending entry with one-shot query test hook: same attribution rule; one GPU/unavailable3. LEAD PENDING. |
| artifacts/p1t_rev7/probes/argument_census_cpu.py | Every add-on native typed position both call styles plus core actual dispatch slices: zero Python conversions/refusal callbacks;1067 checks PASS. CPU RUN. |
| artifacts/p1t_rev7/probes/run_cpu_checks.py | 14 finite host trace/arithmetic/lock/exception checks, all must exit0; PASS. CPU RUN. |
| artifacts/p1t_rev7/probes/build_cpu.sh | Build current-source add-on trace, UBSan native harnesses, syntax-check core binding: compile/link success; PASS. CPU RUN. |
| artifacts/p1t_rev7/probes/build_core_host.py | Regenerate/hash exact source spans and actual core dispatch/raw wait/error registrations; compile with UBSan; PASS. CPU RUN. |
| artifacts/p1t_rev7/probes/gpu_regressions.py fix56 | Rebuilt core/add-on FIX5/FIX6 tests: zero failures/errors/skips; absence/skips3; singleGPU. LEAD PENDING. |
| artifacts/p1t_rev7/probes/gpu_regressions.py semantics | Finite add-on Conv/ConvTranspose/LSTM/norm/small-op/stream contract tests: zero failures/errors/skips, absence/skips3; singleGPU. LEAD PENDING. |
| artifacts/p1t_rev7/probes/gpu_regressions.py multidevice | Physical multi-device DLPack/device checks: requires2 GPUs, exits3 on lead singleGPU; zero failures/errors/skips. One-GPU device/error interposition is in fix56. LEAD PENDING. |

Run GPU probes only after cleanly rebuilding the reviewed sources into actual package paths, with those paths on PYTHONPATH; sequence them in the foreground under timeout. core_knob_exit_gpu manages its package path in its parent; query_diagnostic_gpu requires PYTHONPATH=<root>/tc_signal. GPU lanes enforce zero failures/errors/skips; missing prerequisites/2-GPU absence is3 INCONCLUSIVE. The current rejection is already supported by CPU execution; a GPU pass cannot waive an open real defect.

## Prior art

No product algorithm or fix implemented. Finding/probe comments and LEDGER.md attribute the borrowed contracts and this review's scenarios. V7-01 uses CPython3.12 (PSF2023) float OverflowError, atexit/finalization/GIL and C++17 noexcept (ISO2017), plus DLPack ownership (DMLC2017); recommendation keeps the existing admission/quiescence mechanism unconditional on diagnostic parsing. V7-02 uses distinct CUDA12.6 EventQuery/StreamSynchronize return contracts (NVIDIA2024); a fixed-capacity failure-kind tag is an ordinary discriminated record, no specific prior art known to me. Other reused checks: GCC/Clang UBSan/overflow builtins (2013/2015), prior REV5/6 actual-source traces, scope-guard RAII (Alexandrescu/Marginean2000) in existing code, SQLite-style OOM fault injection. Probe-specific contribution is stated in each header. No new optimization or literature novelty claimed.

Primary sources checked live: [Python float conversion](https://docs.python.org/3.12/library/functions.html#float), [CPython lifecycle/GIL](https://docs.python.org/3.12/c-api/init.html), [DLPack Python ownership/GIL](https://dmlc.github.io/dlpack/latest/python_spec.html), [CUDA12.6 event API](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-runtime-api/group__CUDART__EVENT.html), [Python finalizer lock/thread warning](https://docs.python.org/3.12/reference/datamodel.html#object.__del__). Existing operator arithmetic uses ONNX Conv/ConvTranspose/LSTM/norm/Resize/Pad/Round specifications. Historical algorithm-title/year claims beyond these contracts were not re-audited; unverified — lead to check: Coffman Elphick Shoshani1971 circular wait; Chan Golub LeVeque1983 shifted variance; Harris2007 CUDA reduction; RCU McKenney Slingwine1998. Those algorithms were neither modified nor newly proposed here.

## Hygiene and deviations

No source,test or build file edits; no git writes,subagents,GPU execution,deletions,services or unrelated process kills. Outputs restricted to review artifacts/logs/docs and/tmp scratch, except one unintended automatic Python cache write from importing fix5_regen; newly generated cache moved without deletion to artifacts/p1t_rev7/scratch/generated_python_cache and recorded in scratch/MANIFEST.md/LEDGER.md. No file moved to DELETE_ME. Scratch retained at/tmp/p1t_rev7_build. Initial argument-harness literal_eval/typed harness-registration errors were corrected in review code only, receipts kept, no criteria weakened. Native product assertions were not disabled to turn a red result green. Long commands ran under foreground timeout; every yielded session resumed at300000ms with no intervening status. Product diff versus429446d remains empty.
