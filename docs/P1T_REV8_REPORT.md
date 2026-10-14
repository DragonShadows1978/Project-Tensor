# P1T-REV8 independent code review

**REJECT. Two MINOR defects remain open; zero BLOCKING, MAJOR or NIT findings.** V7-01 and V7-02 are CLOSED. The three previously PARTLY closed lifetime items (N5, V3-01, V5-01) are CLOSED. Earlier N1 is PARTLY closed again because new diagnostic-event failures are erased. No known defect is waived or accepted as a residual; the no-waiver instruction is why this is REJECT despite having no open MAJOR finding.

Scope: `9df5bef..6991939`, restricted to `tensor_cuda/` and `tc_signal/`, with FIX7 delta `429446d..6991939`. Worktree HEAD is `551079e4a54469edc51f898dabee7dffd8163992`; read-only diff confirms its product files equal `6991939`. Full scope and FIX7 diffs are saved here. No source, test or build file changed. No GPU code executed, no subagents, no deletion, no git writes. Builds and generated source-linked harnesses live under `/tmp/p1t_rev8_build`.

Evidence classes: code reasoning, actual-source finite CPU traces/validation, compilation, ELF inspection, and historical lead receipts checked against their raw artifacts. CUDA function replacements exercise product control flow and allowed return values; they do not prove physical GPU behavior. The full add-on's refusal and callback paths were exercised without launching GPU work. GPU numerical, physical stream/fence and two-device conclusions remain lead-only probes. Voice's acceptance tolerances and performance remain out of scope.

## R1–R6

| Question | Verdict | Evidence / scope |
|---|---|---|
| R1 | SAFE | Ordinary core semantics preserved by source review; checked helpers remove overflow UB. Lead baseline-ID comparison independently matches 52/52; this is not a new GPU suite run. |
| R2 | UNSOUND | Lifetime/device/argument fixes hold; native exit error reporting fails under OOM (V8-02). |
| R3 | UNSOUND | Deferred-release/validation fixes hold; new diagnostic errors vanish and final exit reports can vanish (V8-01/V8-02). |
| R4 | CORRECT | Source arithmetic/indexing and geometry checked; host boundary/UBSan checks and CUDA compilation pass. GPU numerics not executed here. |
| R5 | PASS | Fresh full add-on build and core bindings/kernels compilation pass; ELF needs only shared CUDA/cuBLAS and standard libraries, no core ABI. |
| R6 | REJECT | Two real MINOR defects remain open. No BLOCKING/MAJOR found, but no-waiver rule prevents calling these harmless nits or accepted residuals. |

## Closure of the 27 earlier items

Source legend: C = `tensor_cuda/src/bindings.cpp`; K = `tensor_cuda/src/kernels.cu`; CI = `tensor_cuda/tensor_cuda/__init__.py`; O = `tc_signal/src/dlpack_operand.cpp`; A = `tc_signal/src/bindings.cpp`; L = `tc_signal/src/lstm_bindings.cpp`; M = `tc_signal/src/m4_bindings.cpp`; CM = `tc_signal/src/checked_math.h`; PA = `tc_signal/src/py_args.h`; SI = `tc_signal/tc_signal/__init__.py`. Line ranges refer to this reviewed source. CLOSED means the original defect/site is closed under its documented contract, not universal certification of its broader defect class.

| Item | Status | Lines read and evidence |
|---|---|---|
| F01 | CLOSED | O:1500-1603; A:262-288; L:209-244; M:229-242,287-336. Prelaunch reservation, completion retention and checked fallback intact; host release checks pass. |
| F02 | CLOSED | O:511-516,645-649; A:258-261; L:181-184,202-208; M:66-105. Required-None and absent-metadata refusals intact; argument census passes. |
| F03 | CLOSED | C:324-348,1162-1212,1241-1250; O:455-507; CM:31-89; A:101-124,175-188. Checked shape/byte/address/geometry bounds intact; UBSan and descriptor checks pass. |
| F04 | CLOSED | C:850-900; O:194-234. Finalizing-entry refusal and admission handshake retained; callback quiescence separately covered by N5/V5-01. |
| F05 | CLOSED | K:289-320,463-488. Imported Storage drains its own device and leaks without completion proof; restore and error records checked. Host multi-device trace passes; physical test lead-only. |
| F06 | CLOSED | C:637-676. unique_ptr owns export context until PyCapsule_New succeeds; all metadata allocation failures release it once. |
| F07 | CLOSED | tensor_cuda/include/tc/core.h:80-92; docs/P1T_FIX1_M1_ADDENDUM.md:17-31. Native ABI rebuild requirement explicit; fresh add-on includes/links no core ABI. |
| N1 | PARTLY closed | O:1287-1336,1773-1807 retain ordinary query/sync/prior errors; O:1049-1094 newly erases diagnostic errors (V8-01); O:1759-1770 drops exit reports on OOM (V8-02). Original sites work, global no-silent-error rule reopened. |
| N2 | CLOSED | C:324-348,621-624; K:80-97; CM:63-89; O:469-498. Zero-neighbor extent bound and bytes checked; zero/huge descriptor checks and UBSan pass. |
| N3 | CLOSED | O:1185-1199,1339-1370,1500-1603,1783-1807. Reserved release bookkeeping, caught push failure and persistent records survive OOM without terminate; CPU OOM checks pass. Last-reader logging is V8-02. |
| N4 | CLOSED | O:559-591; C:861-900. Manager pointer cleared before contained ordinary C++ deleter exception; fresh throwing-deleter CPU harness passes. |
| N5 | CLOSED | C:861-900,939-1068,1495-1562,1640-1675; CI:246-294. Close/wait cannot be bypassed by unreadable diagnostic values or pre-hook exceptions; overflow-knob and both lock-retry CPU stagings pass. |
| V3-01 | CLOSED | C:939-1042; CI:248-294. No timeout return; own-thread exclusion retained, runner covers interrupted hook; finite core callback completes before finalization. |
| V3-02 | CLOSED | O:1222-1255,1487-1509. Checked aggregate and sticky saturated queue bytes retained; 17-case inherited accounting UBSan check passes. |
| V4-01 | CLOSED | C:428-488,626-635; K:443-460. Label proof and tensor-device fence/destroy/restore intact; host multi-device trace passes; physical two-device fence remains lead-only. |
| V4-02 | CLOSED | CI:97-117,207-244. Documentation states unbounded wait and diagnostic delay; no reintroduced bounded-exit claim. |
| V4-03 | CLOSED | C:1141-1154,1217-1223. Checked cudaGetDevice before Storage ownership/rename; failed import keeps capsule ownership. |
| V4-04 | CLOSED | O:1776-1807,1708-1713. Prior slot persisted before drain and cleared only after captured RuntimeError; prior/OOM trace passes. |
| V5-01 | CLOSED | O:218-249,331-345,351-380,559-591,1422-1454; SI:214-231,266. Every deleter/capsule-drop/producer callback admitted; hook finally and runner always close. 22 knob classes and exception/interrupt CPU stagings pass. |
| V5-02 | CLOSED | O:1606-1643. Original next-call query failure recorded before clearing, raised once; query_once CPU check passes. New diagnostic-event site is V8-01, not a regression at this site. |
| V5-03 | CLOSED | tc_signal/src/conv.cu:90-135; tc_signal/src/lstm.cu:107-154. New-handle RAII survives map insertion OOM; FIX7 only changes status text/check helpers, not ownership. |
| V5-04 | CLOSED | K:596-647. Wide ceil/grid refusal before launch/revision intact; exact-source fill_blocks huge-grid CPU refusal passes; kernels.cu compiles. |
| V6-01 | CLOSED | O:331-345,1422-1427,1773-1782. Only exit close waits for other threads; ordinary synchronize still has no such wait. Fresh lock-inversion and late-release CPU checks pass. |
| V6-02 | CLOSED | C:544-599,1357-1454,1708-1768; PA:78-195. C-level value reads and raw shutdown entry intact. 1067 argument checks show zero conversion calls; both exit-entry censuses pass. |
| V6-03 | CLOSED | O:907-941,1000-1008,1783-1807. Tagged drain reports persist before message and clear after captured exception. All three inherited drain/OOM cases and 22 FIX7 attribution assertions pass; exit final-reader gap is V8-02. |
| V7-01 | CLOSED | C:1495-1562,1640-1675; PA:222-279; A:356-453; CI:246-294; SI:214-266. No Python knob conversion, C-level default, finally close, native runner; CPU native-entry, 22-class add-on and core overflow/lock controls pass. |
| V7-02 | CLOSED | O:780-941,1303-1325,1703-1751. FailedCall/Purpose/device/error travel together; query after successful sync names cudaEventQuery. Fresh query probe and plain/OOM attribution assertions pass. |

## New findings

### V8-01 — MINOR: L1 diagnostic event failures are consumed without recording or reporting

Location: `tc_signal/src/dlpack_operand.cpp:1052`; 1052-1059,1066-1069,1087-1094.

Scenario: Inside the exit runner window, cudaEventCreateWithFlags, cudaEventRecord, cudaEventQuery or cudaEventDestroy fails once with a non-sticky error; the later cudaStreamSynchronize succeeds. The diagnostic helper clears the error with cudaGetLastError and stores no FailureRecord, so synchronize returns success and the exit log contains no CUDA failure. Create/record failure also falls straight through to a potentially unbounded silent stream wait, contradicting L1 a3. After a query error following a diagnostic it can print that the work finished although completion was not proved by the query.

Fix: Record every failed diagnostic event call before consuming its slot, with call/purpose/device tags, and route it through the existing raise-once/exit logger. Provide a diagnosed fallback when the marker cannot be created/recorded; print completion only on success.

Evidence: CPU actual-source host trace: logs/p1t_rev8_diagnostic_errors.log, four one-shot call failures erased (error=null, slot=0); no GPU run.

Prior art: NVIDIA CUDA 12.6 Runtime API event/error-slot contracts (2024); FIX5 V5-02 record-before-consume. Standard contracts and existing error accounting reused; reviewer contribution is the exit-window failure census.

### V8-02 — MINOR: Native exit loggers silently abandon recorded CUDA failures on reporting allocation failure

Location: `tc_signal/src/dlpack_operand.cpp:1768`; tc_signal/src/dlpack_operand.cpp:1759-1770, tensor_cuda/src/bindings.cpp:1660-1672.

Scenario: The Python hook is interrupted, so its native runner is the last reporter. A CUDA failure is already recorded (for example the interrupted add-on runner synchronize drained with an error and could not allocate its message). report_message / unraisable_message or the core record snapshot then throws bad_alloc. The exit logger catches it and writes nothing. Keeping the record cannot recover it after interpreter exit: there is no next synchronize.

Fix: Provide an allocation-free stderr fallback that emits the existing tagged records, or at least the count and first failed call/device/error, when rich-message construction fails. Keep quiescence first and preserve the hook exception.

Evidence: Actual add-on source included and C++ allocation failures injected only inside exit_log_unreported: logs/p1t_rev8_exit_log_oom.log contains record_still_pending=True and no CUDA stderr; control logs/p1t_rev8_exit_log_control.log names cudaEventQuery and consumes record. Core identical silent catch confirmed by source reasoning; no GPU required. Core exact-source OOM/control now independently reproduced: logs/p1t_rev8_core_exit_oom_retry.log has no CUDA stderr, logs/p1t_rev8_core_exit_control_retry.log logs the recorded failure. Initial child had an invalid stats-key assertion; corrected before accepting evidence.

Prior art: SQLite OOM fault-injection testing; C++17 bad_alloc; existing allocation-free FailureRecord/static core site string. Taken: fail allocations and report using pre-existing POD data; reviewer contribution is the last-reader-at-exit scenario.

Severity rationale: both findings affect exceptional-path error visibility and promised exit diagnostics. The recorded scenarios do not establish early release, incorrect tensor data, or interpreter abort, so they are MINOR rather than MAJOR. For V8-01, a successful subsequent stream sync still supplies completion proof; it does not retrospectively report an earlier non-sticky event error. For V8-02, keeping a record is useful during the run but cannot rescue it when the final reader is exiting. A finite CPU falsification is enough to identify both code defects; GPU reproduction is supplementary.

## Sweep-table check: (i), (j), L1

All 26 i rows, 35 j rows and 12 L1 rows inspected. i24/i25/j24/a3 PARTLY; missing diagnostic event failure sites and final-logger OOM case. Core close/relock and add-on polling diagnostic fixes independently pass. a9 rationale corrected for std::string allocation.

| Row | Source sites | Verdict | Check |
|---|---|---|---|
| i1 | A:417-453; SI:214 | CONFIRMED | Fresh first-bytecode interrupt/deleted-_C children close and wait through runner. |
| i2 | A:417-453; SI:214 | CONFIRMED | Fresh first-bytecode interrupt/deleted-_C children close and wait through runner. |
| i3 | SI:215-217 | CONFIRMED | Constant local stores run no callback or conversion. |
| i4 | SI:218-231 | CONFIRMED | Exception/BaseException from synchronize cannot skip finally close; five exception-kind children pass. |
| i5 | SI:218-231 | CONFIRMED | Exception/BaseException from synchronize cannot skip finally close; five exception-kind children pass. |
| i6 | SI:224-230; PA:222-279 | CONFIRMED | Deleted knob or unreadable value selects default; 22 knob-class CPU children pass, zero method calls. |
| i7 | SI:224-230; PA:222-279 | CONFIRMED | Deleted knob or unreadable value selects default; 22 knob-class CPU children pass, zero method calls. |
| i8 | PA:239-279; A:356-371 | CONFIRMED | Malformed call refuses before close; every value uses C read/default. Native-entry CPU census passes. |
| i9 | PA:239-279; A:356-371 | CONFIRMED | Malformed call refuses before close; every value uses C read/default. Native-entry CPU census passes. |
| i10 | O:1422-1454 | CONFIRMED | Atomic close precedes unbounded wait; result allocation follows. close_raises child still reaches quiescence. |
| i11 | O:1422-1454 | CONFIRMED | Atomic close precedes unbounded wait; result allocation follows. close_raises child still reaches quiescence. |
| i12 | SI:232-260 | CONFIRMED | stderr work follows close; three broken-stderr children complete. |
| i13 | A:417-453 | CONFIRMED FOR QUIESCENCE | Native runner always reaches close after hook raises; final-report allocation gap separately i24. |
| i14 | C:1640-1675; CI:246 | CONFIRMED | Raw runner closes after pre-hook exception; source and raw-runner CPU control checked. |
| i15 | C:1640-1675; CI:246 | CONFIRMED | Raw runner closes after pre-hook exception; source and raw-runner CPU control checked. |
| i16 | CI:248-255; C:1495-1562 | CONFIRMED | No diagnostic conversion callbacks; huge-int real hook plus both raw entry call styles pass. |
| i17 | CI:248-255; C:1495-1562 | CONFIRMED | No diagnostic conversion callbacks; huge-int real hook plus both raw entry call styles pass. |
| i18 | CI:248-255; C:1495-1562 | CONFIRMED | No diagnostic conversion callbacks; huge-int real hook plus both raw entry call styles pass. |
| i19 | C:939-1042 | CONFIRMED | Initial/re-lock failures retried with diagnostics; 15-failure CPU close and relock witnesses pass. |
| i20 | C:1015-1042 | CONFIRMED | Current libstdc++ timed wait path retains explicit GIL restore; no early timeout return; scoped implementation reasoning. |
| i21 | CI:257-290; C:1608-1627 | CONFIRMED AT LISTED SITES | Post-close result failure cannot skip raising records; runtime record consumption waits for captured RuntimeError. Rich final-logger failure is i25. |
| i22 | CI:257-290; C:1608-1627 | CONFIRMED AT LISTED SITES | Post-close result failure cannot skip raising records; runtime record consumption waits for captured RuntimeError. Rich final-logger failure is i25. |
| i23 | A:424-435; O:1287-1336 | CONFIRMED | Interrupted runner drains before closing when hook did not close; existing source-linked release/interrupt probes pass. |
| i24 | SI:233-238; A:424-451; O:1759-1770 | PARTLY | Transient message OOM retry works, but last native logger swallows bad_alloc: V8-02. Fresh OOM/control pair disproves unconditional no-loss claim. |
| i25 | CI:275-283; C:1660-1672 | PARTLY | Normal interrupted-hook report exists; core runner swallows reporting bad_alloc with no output: V8-02, independently reproduced. |
| i26 | A:444-449; C:1646-1653 | CONFIRMED | Successful rich result reports runner wait; OOM of final failure logger separately covered by i24/i25. |
| j1 | O:1292-1299,825-829 | CONFIRMED | Current-device query names cudaGetDevice and does not invent a current device; plain/OOM trace assertions pass. |
| j2 | O:1098-1137,1305-1308,830-843 | CONFIRMED | Guard tags and spill device retained; guard/set/get and 70-device spill trace assertions pass. |
| j3 | O:1098-1137,1305-1308,830-843 | CONFIRMED | Guard tags and spill device retained; guard/set/get and 70-device spill trace assertions pass. |
| j4 | O:915-939,1324,1720-1732 | CONFIRMED | Query tag preserved through merge/OOM; capped record overflow counts unnamed failures; fresh trace assertions pass. |
| j5 | O:915-939,1324,1720-1732 | CONFIRMED | Query tag preserved through merge/OOM; capped record overflow counts unnamed failures; fresh trace assertions pass. |
| j6 | O:915-939,1324,1720-1732 | CONFIRMED | Query tag preserved through merge/OOM; capped record overflow counts unnamed failures; fresh trace assertions pass. |
| j7 | O:936-939,1548-1602 | CONFIRMED | First() pairs one call/device/error; record guard and fallback differentiated; bound/fallback trace assertions pass. |
| j8 | O:936-939,1548-1602 | CONFIRMED | First() pairs one call/device/error; record guard and fallback differentiated; bound/fallback trace assertions pass. |
| j9 | O:936-939,1548-1602 | CONFIRMED | First() pairs one call/device/error; record guard and fallback differentiated; bound/fallback trace assertions pass. |
| j10 | O:1170-1182,1481-1484,1606-1643 | CONFIRMED | Destroy guard, next-call query and caller restore name actual API/target. Fresh destroy/query/restore host assertions pass. |
| j11 | O:1170-1182,1481-1484,1606-1643 | CONFIRMED | Destroy guard, next-call query and caller restore name actual API/target. Fresh destroy/query/restore host assertions pass. |
| j12 | O:1170-1182,1481-1484,1606-1643 | CONFIRMED | Destroy guard, next-call query and caller restore name actual API/target. Fresh destroy/query/restore host assertions pass. |
| j13 | O:1524-1534,1708-1713 | CONFIRMED | Pre-existing per-thread slot explicitly has unknown producing call/device; prior and prelaunch trace assertions pass. |
| j14 | O:1524-1534,1708-1713 | CONFIRMED | Pre-existing per-thread slot explicitly has unknown producing call/device; prior and prelaunch trace assertions pass. |
| j15 | O:1339-1369 | CONFIRMED | Creation vs failed guard call distinguished; creation-guard CPU assertion passes. |
| j16 | tc_signal/src/cuda_guard.h:72-112; conv.cu:216-280; lstm.cu:223-270; norms.cu:159-170; small_ops.cu:90-118 | CONFIRMED BY SOURCE/BUILD | Every call site uses correct helper; checks distinguish asynchronous launch slot from successful cuBLAS status. Fresh .cu compilation passes, physical launch/slot tests lead-only. |
| j17 | tc_signal/src/cuda_guard.h:72-112; conv.cu:216-280; lstm.cu:223-270; norms.cu:159-170; small_ops.cu:90-118 | CONFIRMED BY SOURCE/BUILD | Every call site uses correct helper; checks distinguish asynchronous launch slot from successful cuBLAS status. Fresh .cu compilation passes, physical launch/slot tests lead-only. |
| j18 | tc_signal/src/cuda_guard.h:72-112; conv.cu:216-280; lstm.cu:223-270; norms.cu:159-170; small_ops.cu:90-118 | CONFIRMED BY SOURCE/BUILD | Every call site uses correct helper; checks distinguish asynchronous launch slot from successful cuBLAS status. Fresh .cu compilation passes, physical launch/slot tests lead-only. |
| j19 | tc_signal/src/cuda_guard.h:72-112; conv.cu:216-280; lstm.cu:223-270; norms.cu:159-170; small_ops.cu:90-118 | CONFIRMED BY SOURCE/BUILD | Every call site uses correct helper; checks distinguish asynchronous launch slot from successful cuBLAS status. Fresh .cu compilation passes, physical launch/slot tests lead-only. |
| j20 | tc_signal/src/conv.cu:283-321; cuda_guard.h:120-189 | CONFIRMED | No-device property query and scoped guard target/restore messages match call purpose. |
| j21 | tc_signal/src/conv.cu:283-321; cuda_guard.h:120-189 | CONFIRMED | No-device property query and scoped guard target/restore messages match call purpose. |
| j22 | SI:232-260; O:276-326 | CONFIRMED | Hook exception/result text follows close; timed/polling wait text describes actual wait; CPU staging passes. |
| j23 | SI:232-260; O:276-326 | CONFIRMED | Hook exception/result text follows close; timed/polling wait text describes actual wait; CPU staging passes. |
| j24 | A:428-451; O:1759-1770 | PARTLY | Existing emitted text is correct, but OOM silently emits no recorded CUDA failure: V8-02. Diagnostic event calls added by L1 are missing from j sweep: V8-01. |
| k1 | K:289-320,339-371 | CONFIRMED BY SOURCE/BUILD | Each imported-drain branch names one API; restore names its target; fill scoped guard remains checked. kernels.cu compiles. |
| k2 | K:289-320,339-371 | CONFIRMED BY SOURCE/BUILD | Each imported-drain branch names one API; restore names its target; fill scoped guard remains checked. kernels.cu compiles. |
| k3 | K:289-320,339-371 | CONFIRMED BY SOURCE/BUILD | Each imported-drain branch names one API; restore names its target; fill scoped guard remains checked. kernels.cu compiles. |
| k4 | K:449-451,622-644 | CONFIRMED BY SOURCE/BUILD | Pending-slot source unknown; launch-check ambiguity stated; failed pointer query says allocation device unknown. kernels.cu compiles; real driver checks lead-only. |
| k5 | K:449-451,622-644 | CONFIRMED BY SOURCE/BUILD | Pending-slot source unknown; launch-check ambiguity stated; failed pointer query says allocation device unknown. kernels.cu compiles; real driver checks lead-only. |
| k6 | K:449-451,622-644 | CONFIRMED BY SOURCE/BUILD | Pending-slot source unknown; launch-check ambiguity stated; failed pointer query says allocation device unknown. kernels.cu compiles; real driver checks lead-only. |
| k7 | C:428-488,1141-1154,1261-1286 | CONFIRMED | Fence errors name failing step/tensor device and restore target; import/empty current query has no invented current device. |
| k8 | C:428-488,1141-1154,1261-1286 | CONFIRMED | Fence errors name failing step/tensor device and restore target; import/empty current query has no invented current device. |
| k9 | C:1596-1627,1646-1675; CI:257-290 | CONFIRMED FOR EMITTED TEXT | Reporter paths and first call/device correctly described. Missing final text on OOM is independently accounted in i25/V8-02. |
| k10 | C:1596-1627,1646-1675; CI:257-290 | CONFIRMED FOR EMITTED TEXT | Reporter paths and first call/device correctly described. Missing final text on OOM is independently accounted in i25/V8-02. |
| k11 | C:1596-1627,1646-1675; CI:257-290 | CONFIRMED FOR EMITTED TEXT | Reporter paths and first call/device correctly described. Missing final text on OOM is independently accounted in i25/V8-02. |
| a1 | O:276-345 | CONFIRMED | Condition wait and polling fallback diagnose; fresh interposed mutex failure writes polling lines and callback finishes before finalization. |
| a2 | O:276-345 | CONFIRMED | Condition wait and polling fallback diagnose; fresh interposed mutex failure writes polling lines and callback finishes before finalization. |
| a3 | O:1049-1105; A:399-418 | PARTLY | Healthy marker polling diagnoses. Failed create/record returns into silent cudaStreamSynchronize; nonsticky event errors cleared without record (V8-01). |
| a4 | O:1260-1265,559-574 | CONFIRMED | Executing producer code on exiting thread is not a new P1-T wait; a nonreturning deleter violates producer contract. |
| a5 | C:939-1042 | CONFIRMED | Both lock retry sites diagnosed; fresh 15-failure CPU stagings each count diagnostic lines and quiesce. |
| a6 | C:939-1042 | CONFIRMED | Both lock retry sites diagnosed; fresh 15-failure CPU stagings each count diagnostic lines and quiesce. |
| a7 | C:1017-1057,1640-1658 | CONFIRMED | Timed wait and runner share unbounded diagnosed close; source and finite callback/lock witnesses pass. |
| a8 | C:1017-1057,1640-1658 | CONFIRMED | Timed wait and runner share unbounded diagnosed close; source and finite callback/lock witnesses pass. |
| a9 | K:391-428 | CONFIRMED WITH REASON CORRECTION | No producer/Python callback under registry mutex; snapshot does allocate a std::string, so the literal no-call rationale is too broad. Reporting OOM handled as V8-02; no new unbounded application-callback wait found. |
| a10 | K:463-488 | CONFIRMED | Storage drain is unchanged per-drop lifetime fence, not a wait introduced/called by exit hook. Physical GPU execution excluded. |
| a11 | O:206-215,289-296,309-324; C:1021-1033 | CONFIRMED IN STATED SCOPE | Lifecycle mutex released before diagnostics; leave uses short critical section; stderr channel backpressure itself is outside the promised diagnostic delivery guarantee. |
| a12 | O:206-215,289-296,309-324; C:1021-1033 | CONFIRMED IN STATED SCOPE | Lifecycle mutex released before diagnostics; leave uses short critical section; stderr channel backpressure itself is outside the promised diagnostic delivery guarantee. |
| missing-j-L1-event-calls | O:1052-1059,1066-1069,1094 | MISSING / DEFECT (MINOR) | Diagnostic marker create/record/query/destroy have no failure tags/records; four-case CPU census RED (V8-01). Query-error branch may also emit an unproved finished line. |
| missing-i-final-logger-OOM | O:1768-1770; C:1671-1672 | MISSING FAILURE CASE / DEFECT (MINOR) | Final logger message/snapshot allocation can fail after rich synchronize failure; empty catches keep a record with no later reader (V8-02). |

FIX6 corrections: h8's corrected explanation is right: native consumers can drop an exported context without holding the GIL; C:861-900 admits before PyGILState_Ensure, and the exit wait sees those releases. The old g17 raw shutdown-dispatch assumption is correctly replaced with the native METH_VARARGS/METH_KEYWORDS entry (C:1546-1583); unsupported diagnostic values now close using the default, not refuse. Add-on g10/g21 likewise use PA:239-279 and A:356-371. h2/h9/g14's quiescence reachability corrections hold for the code reviewed, with reporting failure separated as V8-02. A colliding custom key inserted in hook globals can invoke equality during lookup, including the hook's LOAD_GLOBAL; that crafted module-dictionary behavior was checked as a contract-boundary case, not an ordinary knob conversion or a new finding. Arbitrary replacement hooks and deliberate unregister are likewise outside the registered actual-hook guarantee.

## FIX7 user-visible changes

The actual FIX7 report contains U10 in its L1 addendum as well as the requested U1–U9; it is judged explicitly because it introduced V8-01.

| Change | Verdict | Evidence |
|---|---|---|
| U1 | ACCEPTABLE | Both raw entries read any diagnostic value without Python methods; malformed shape still refuses. C:1495-1562; PA:222-279; fresh native-entry census. |
| U2 | ACCEPTABLE | Knob passed unchanged; unsupported values default. CI:248-255; SI:224-230; 22-class CPU exit census. |
| U3 | ACCEPTABLE | Native atexit runners cover pre-frame exceptions; unregister name documented. CI:294; SI:266; fresh interrupted-hook/runner CPU checks. |
| U4 | ACCEPTABLE | BaseException cannot skip the finally close; runner preserves original exception. SI:218-231; C:1640-1675; exception CPU census. |
| U5 | DEFECT (MINOR, V8-02) | Close/wait backstop works, but the promised last CUDA-failure report vanishes if rich-message allocation fails. O:1759-1770; C:1660-1672; both CPU OOM/control pairs. |
| U6 | ACCEPTABLE | One post-close retry reports a transient synchronize-message OOM. SI:233-238; CPU MemoryError staging. Persistent final-logger OOM is separately V8-02. |
| U7 | ACCEPTABLE | Existing changed messages name failed calls/purpose/device correctly; 22 fresh attribution assertions pass. O:821-939; cuda_guard.h:72-112; K:289-320,639-644. |
| U8 | ACCEPTABLE | Stats/attributes additive, counters are exposed and observed. O:1810 onward; C:2538-2582; raw-entry/lock census. |
| U9 | ACCEPTABLE | Both core lock attempts retry with GIL released and diagnosed delay; source C:939-1042 and fresh close/relock interposer checks pass. |
| U10 | DEFECT (MINOR, V8-01) | Additional change present in the actual report beyond the order U1-U9: marker-call failures erase errors and bypass the polled diagnostic wait. O:1049-1105; four-case CPU census RED. |

## FIX7 Not-a-defect judgments

These N1–N4 labels belong to FIX7's justification table and are separate from the historical closure-table N1–N5.

| Item | Verdict | Evidence |
|---|---|---|
| N1 | CORRECTLY NOT A DEFECT | Actual hooks close before any normal return; an arbitrary replacement hook returning without close is outside that guarantee. C:1640-1642; A:417-421; native-runner control. |
| N2 | CORRECTLY NOT A DEFECT | Explicit unregister/_clear/os._exit deliberately removes documented protection. CI:292-294; SI:264-266; CPython atexit docs. |
| N3 | CORRECTLY NOT A DEFECT | Unreadable diagnostic value selects documented default and never runs conversions. PA:222-279; C:1528-1543; raw-entry census. |
| N4 | CORRECTLY NOT A DEFECT | Preserving KeyboardInterrupt for atexit to print is documented behavior after quiescence. C:1674; A:452; interrupted-hook CPU staging. |

## Operand and operating-condition census

- Scalar/zero-size: core allows scalar and valid empty shapes; add-on rint accepts scalar, other op rank/empty checks reject unsupported cases. Actual plan/validation checks exercised 0-d NULL shape, zeros, and zero beside huge extents (17-case receipt). No tensor data dereference in CPU plan checks.
- Near-int64 extents, offset/address wrapping, singleton strides: checked in source and finite descriptors; core fill geometry refuses overflow before launch/revision. UBSan arithmetic/accounting harnesses pass. Core kernels compile.
- Read-only/versioned/copied capsules, repeat consumption and same bare capsule twice: actual add-on refusal paths exercised. Same producer used twice is legal when it supplies two independently owned capsules; output/scratch overlap checks remain mandatory.
- Producer stream ordering: O:412-422 asks stream=1, output copy=False; C:428-488 exports a tensor-device event dependency. Physical producer/consumer nondefault-stream behavior is lead-only `semantics` and device lanes.
- Throwing deleters and host allocation failures: contained C++ deleter harness, pending allocation/accounting/error OOM checks pass; final logger OOM is RED. Tests exercise actual included product lifecycle, not a copied release algorithm.
- Native arguments: C-level readers ignore Python overrides of int/float/tuple/list/str subclasses, refuse numpy scalar types that are not accepted subclasses, and refuse conversion-only objects without invoking them. 1067-case census covers all 29 add-on bindings' typed positions in positional/keyword forms, plus selected core dispatch; remaining core data_ptr/fill_/mark_modified/revision/entry sites were read at C:1708-1768,2538-2600. Both exit entries have independent positional/keyword value-and-shape censuses. Public wrappers intentionally perform conversion in Python frames.
- Exit with callbacks in flight: actual full add-on 22-class settings census, deleted setting/_C, Exception/BaseException, first-bytecode interruption and broken stderr all pass; core actual-source finite callback/huge knob and raw entry/runner controls pass. Both core lock retry sites and add-on polling fallback have fresh interposed-lock witnesses.
- Concurrent synchronize: lock-inversion and late-release witnesses pass; only exit close waits for another thread. Producer __dlpack__ and capsule destructor admission sites O:351-426 were read. Deliberately removing hooks is not covered.
- CUDA errors before/once during synchronize/op/fill/export: ordinary source-linked drain/prior-slot, guard, spill and persistence assertions pass; new exit marker-call errors are RED. Physical launch/fill/export error probes remain lead-only; no baseline unrelated binding defect reclassified as P1-T.
- Multi-device pending entries/current-device mismatch: fresh host traces cover device switch/restore and 70-device spill bookkeeping; real two-device execution remains INCONCLUSIVE on the lead's one GPU and must exit3. No two-GPU receipt is treated as a pass.

## Compilation and finite CPU receipts

Fresh full add-on build PASS (`logs/p1t_rev8_addon_build.log`), core binding syntax/host slices PASS, actual `tensor_cuda/src/kernels.cu` sm89 object compilation PASS (`logs/p1t_rev8_core_kernels_build.log`). Add-on ELF inspection records shared libcudart.so.12, libcublas.so.12 and RUNPATH /usr/local/cuda-12.6/lib64; no core include/link/symbol dependency. This is build/separation evidence, not a full core runtime suite run.

14 inherited CPU checks PASS; 1067 argument checks PASS; 17 descriptor checks PASS; 36 FIX7 no-GPU children PASS; 22 current-source attribution assertions PASS; 3 lifecycle lock/poll stagings PASS. V8-01's four diagnostic failure cases RED; V8-02 add-on/core OOM arms produce no recorded CUDA failure line, while both controls do. All malformed/failed harness attempts are kept in logs: attribution's first AST-filter attempt failed NameError, and the first core OOM child queried a nonexistent stats key. Neither is used as defect evidence or a pass; corrected reruns supply the receipts.

## Lead receipts and baseline isolation

`artifacts/p1t_rev8/lead_receipt_crosscheck.json` independently counts the 249 per-case PASS verdicts in the Voice JSON, and parses both the named baseline log and the lead combined-session log: 52 failure/error IDs in each, no new or removed IDs. The raw combined summary is 37 failed, 3001 passed, 47 skipped, 15 errors. Those 52 baseline failures are not findings. The in-tree binary hashes match the lead's final core 750112413b… and add-on ca5770c152… hashes. These are verified historical receipts, not GPU runs by this reviewer. Round-7 before/after and mutant kill summaries were read and checked against the current tests' asserted behavior; they miss the new diagnostic-event and final-logger OOM paths demonstrated here. Their success is therefore finite evidence, not closure of all failure cases.

## Probes

CPU build prerequisites: create `/tmp/p1t_rev8_build`; run `probes/build_cpu.sh`, `probes/build_core_host.py`; build a byte-identical scratch copy of tc_signal with its existing build.sh and offline pybind11. Use PYTHONDONTWRITEBYTECODE=1, CUDA_VISIBLE_DEVICES empty, PYTHONPATH=/tmp/p1t_rev8_build:/tmp/p1t_rev8_build/tc_signal for CPU drivers. Supplemental C++ trace modules include current product source through -Itc_signal/src, never frozen copies. Core host module has a historical `_p1t_rev7_core` name, but its seven exact spans are freshly extracted and hashed in core_source_slices.json. OOM variants are rebuilt against current sources.

| Path under artifacts/p1t_rev8/ | Decides / result |
|---|---|
| probes/build_cpu.sh | Build source-linked release/UBSan harnesses and syntax-check core bindings. PASS in logs/p1t_rev8_cpu_build.log. |
| probes/build_core_host.py | Generate/hash actual core DLPack/lifecycle/dispatch spans and compile host module. PASS in logs/p1t_rev8_core_build.log. |
| probes/run_cpu_checks.py | 14 inherited release, OOM, accounting and concurrent-sync CPU checks. PASS; cpu_results.json. |
| probes/argument_census_cpu.py | 1067 argument reads/refusals, 29 add-on bindings plus selected core dispatch; zero user conversion calls. PASS. |
| probes/operand_census_cpu.py | 17 actual-source scalar/empty/huge/offset/read-only/copied/consumed/stride/grid descriptor checks without launch. PASS. |
| probes/knob_exit_cpu.py | Full add-on finite deleter completes before finalization despite huge diagnostic knob. PASS; --control retained. |
| probes/core_knob_exit_cpu.py | Exact-source core hook/lifecycle huge diagnostic knob; callback completes before finalization. PASS; --control retained. |
| probes/query_diagnostic_cpu.py | Query failure after successful stream sync names cudaEventQuery. PASS. |
| probes/fix7_cpu_census.py | 36 no-GPU children: 22 add-on knob classes, exceptions/stderr/pre-frame interrupt/deleted module, both raw exit entries/runners. PASS. |
| probes/attribution_cpu.py | 22 actual FIX7 host trace assertions including guard/spill/record/query/destroy and message OOM. PASS after correcting AST skip-guard filter; first harness attempt failed NameError. |
| probes/lock_checks_cpu.py | Core initial/relock retries and add-on polling fallback all diagnose and wait for finite callback. Three CPU stagings PASS. |
| probes/diagnostic_errors_cpu.cpp + probes/diagnostic_errors_cpu.py | V8-01: include current product source; one-shot diagnostic create/record/query/destroy errors must be reported. RED: 4/4 erased. |
| probes/exit_log_oom_cpu.cpp + probes/verify_exit_log_cpu.py | V8-02 add-on: final logger must name the saved query failure even under C++ allocation failure. OOM arm RED, normal control PASS; exit_log_oom_cpu.py is child. |
| probes/build_core_host_oom.py + probes/core_exit_log_oom_cpu.py | V8-02 core: actual native runner must emit its saved failure under message/snapshot allocation failure. OOM child has no CUDA stderr, control does; source-hashed generated module. Both child rc0 after stats-key correction; verify_core_exit_log_cpu.py is the parent stderr gate (OOM RED, control PASS). |
| probes/diagnostic_event_shim.cpp + probes/diagnostic_events_gpu.py | LEAD ONLY, single GPU: real built add-on must report four one-shot marker errors; failed setup must still diagnose finite stream wait. Shim compiles; GPU probe NOT RUN; exit3 without GPU. |
| probes/gpu_regressions.py fix7 / fix56 / semantics / multidevice | LEAD ONLY: reviewed binaries run original FIX7/FIX5-6 or finite arithmetic/stream tests, zero-skips gate. Two-device lane exits3 on one GPU; fix7 and semantics are single-GPU forms. NOT RUN here. |

## Prior art

The review uses existing contracts, not a new tensor/release algorithm. DLPack ownership, capsule names and stream dependencies: [DMLC DLPack Python specification (2017 onward)](https://dmlc.github.io/dlpack/latest/python_spec.html). Exit exception ordering and raw C APIs: [CPython 3.12 atexit (PSF, 2023)](https://docs.python.org/3.12/library/atexit.html) and [dictionary C API](https://docs.python.org/3.12/c-api/dict.html). Event return values and completion evidence: [NVIDIA CUDA 12.6 Runtime API (2024)](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-runtime-api/group__CUDART__EVENT.html). OOM failure-path testing: [SQLite developers' testing documentation](https://www.sqlite.org/testing.html), including transient and persistent allocation failures; publication year is not stated there. C++17 bad_alloc/RAII/checked lifetime reasoning follows the C++ standard and the existing source annotations; ONNX LSTM/Conv/ConvTranspose/normalization/Resize/Pad/Round semantics and the product's cited IEEE rounding/reduction algorithms were used for source checks, without proposing new arithmetic. The new contributions are the exit-window event-failure census and the final-reader OOM counterexample. Suggested fixes reuse existing POD/static error records, record-before-consume and no-allocation stderr formatting; producer lifetimes/stream ordering need no redesign. Prior-art comments are at the probe sites and both findings, and recorded in the ledger; product source comments cannot be edited under this review order.

## Execution deviations and hygiene

The required mechanism report's opening was read at startup and included its headline verdict claims before the source analysis; those claims were not used as evidence. The copied CPU runner initially retained a round-7 output directory and transiently overwrote artifacts/p1t_rev7/cpu_results.json, violating the output scope. The new receipt was preserved under round8, the prior tracked round7 bytes restored, and the copied runner corrected. Final read-only status has only artifacts/p1t_rev8/ and docs/P1T_REV8_REPORT.md untracked, with logs ignored. No source/test/build edits, no deletion and no background job. Scratch remains under /tmp/p1t_rev8_build; no DELETE_ME moves were needed. No ongoing process remains from this review.
