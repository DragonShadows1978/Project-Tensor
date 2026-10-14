# P1T-REV9 independent code review

**REJECT. Two open MINOR defects; BLOCKING 0, MAJOR 0, NIT 0.** The original V8-01/V8-02 sites are fixed. N1 remains PARTLY closed because the global error-visibility rule still fails in the core fence and core exit summary. No real defect is accepted as a residual. Under the owner's no-waiver condition, these require correction before merge.

Evidence class: source reasoning, finite CPU host traces/UBSan/native-hook checks, compilation and ELF inspection; historical lead receipts separately checked. No GPU code executed, no source/test/build file edited, no subagents, no deletions, no git writes. Scratch is under /tmp/p1t_rev9_*.

Scope: 9df5bef..e4cf1d6 restricted to tensor_cuda/ and tc_signal/; FIX8 delta 6991939..e4cf1d6. Observed checkout HEAD 80e9662; read-only diff of product trees against e4cf1d6 is empty. Full diffs and source hashes: artifacts/p1t_rev9/full_scope.diff, fix8_product.diff, reviewed_source_hashes.json. Registration is immutable: artifacts/p1t_rev9/registration.md; execution receipts/deviations: docs/P1T_REV9_LEDGER.md. JSON: artifacts/p1t_rev9/findings.json.

## R1-R6

| Question | Verdict |
|---|---|
| R1 | SAFE |
| R2 | UNSOUND |
| R3 | SOUND |
| R4 | CORRECT |
| R5 | PASS |
| R6 | REJECT |

- **R1:** Reasoning plus historical baseline comparison: ordinary valid core arithmetic unchanged, external_storage branch separate, no new baseline failure IDs (52 vs52). ABI clients require rebuild as already documented. No GPU regression suite rerun.
- **R2:** V9-01 compound fence/report OOM drops a CUDA error; V9-02 inconsistent core exit snapshot loses known attribution. Both independently reproduced CPU-only against regenerated current sources.
- **R3:** Source reasoning, current-source release/argument/descriptor/OOM/exit checks, actual 12.6 native hook acknowledgement cases and L2 host interposer pass. No add-on defect established; dismissed test-only recycle hypothesis is explicitly separated.
- **R4:** Source arithmetic/indexing/geometry reasoning, checked-arithmetic UBSan gates, fresh nvcc12.6 core compilation and complete add-on builds pass. Device numeric/ordering execution not performed; lead probes supplied.
- **R5:** Full unchanged add-on compiled/linked in /tmp with CUDA12.6 and shared libcudart RUNPATH. No core headers, ABI symbols or core binary dependency. Core bindings syntax, relevant kernel compilation and source-linked core host module link pass; whole core production build not rerun.
- **R6:** Two open real MINOR engineering defects; owner forbids treating real defects as waived residuals. REJECT follows round8 precedent; no open BLOCKING or MAJOR, but no APPROVE-WITH-NITS disguise for these defects.

## Closure of the 29 earlier items

C = tensor_cuda/src/bindings.cpp; K = tensor_cuda/src/kernels.cu; H = tensor_cuda/include/tc/core.h; O = tc_signal/src/dlpack_operand.cpp; A = tc_signal/src/bindings.cpp; L = tc_signal/src/lstm_bindings.cpp; M = tc_signal/src/m4_bindings.cpp; G = tc_signal/src/cuda_guard.h; PA = tc_signal/src/py_args.h; CM = tc_signal/src/checked_math.h; CI = tensor_cuda/tensor_cuda/__init__.py; SI = tc_signal/tc_signal/__init__.py. All line numbers name the reviewed current sources. CLOSED at original sites does not mean every member of the broader failure class is absent. The 26 previously CLOSED items were checked for FIX8 regression; N1, V8-01 and V8-02 additionally exercised against current-source CPU checks. Total: 28 CLOSED, 1 PARTLY closed, 0 OPEN among these 29.

| Item | Status | Lines read / evidence |
|---|---|---|
| F01 | CLOSED | O:1756-1861; A:256-288; L:209-244; M:226-242,284-336; Prelaunch reservations, deferred ownership and checked fallback intact; release CPU gates pass. |
| F02 | CLOSED | O:511-516,645-649; A:258-261; L:202-208; M:66-105; Required-None and absent metadata refused before launch; descriptor/argument checks pass. |
| F03 | CLOSED | C:328-352,1161-1212,1234-1250; O:455-507; CM:31-89; A:101-124,175-188; Checked extents, bytes, addresses and geometry intact; UBSan/descriptor/build gates pass. |
| F04 | CLOSED | C:852-900; O:194-234; Finalizing refusal and admission handshake unchanged; finite exit callbacks finish before finalization. |
| F05 | CLOSED | K:289-320,493-519; Imported storage drains its own device, restores and leaks without completion proof; host multi-device trace passes. |
| F06 | CLOSED | C:357-390,639-678; Export context stays in unique_ptr until capsule creation succeeds; allocation failure unwinds ownership. |
| F07 | CLOSED | H:80-92; FIX1 addendum:17-31; add-on CMake:50-63; Native ABI rebuild requirement remains explicit; add-on includes/links no core ABI. |
| N1 | PARTLY closed | O:1531-1580,2104-2185; C:465-489,1632-1653; Original and FIX8 error paths work; V9-01 still loses an error and V9-02 loses its known attribution. |
| N2 | CLOSED | C:328-352,624-625; K:80-98; CM:63-89; O:469-498; Zero-neighbor extent bounds preserved; empty/scalar/huge metadata and UBSan checks pass. |
| N3 | CLOSED | O:1583-1626,1756-1861,2104-2185; Reserved release bookkeeping and caught push OOM intact; reporting records persist through allocation failure. |
| N4 | CLOSED | O:559-591; C:862-903; Ordinary throwing deleters contained and counted once; native throw-deleter gate passes. |
| N5 | CLOSED | C:862-903,986-1069,1544-1559,1732-1758; CI:253-308; Unconditional close/wait and native runner retained; knob/interrupt/lock CPU stagings pass. |
| V3-01 | CLOSED | C:940-1044; CI:207-308; Unbounded diagnosed wait and own-thread exclusion retained; finite callback exit gate passes. |
| V3-02 | CLOSED | O:1446-1499,1743-1760; Checked aggregate and sticky saturated queue bytes intact; accounting UBSan gate passes. |
| V4-01 | CLOSED | C:432-489,633-637; K:473-489; Own-device fence, destroy and restore intact; label proof retained. Physical ordering probe remains lead-only. |
| V4-02 | CLOSED | CI:107-117,207-249; SI:171-187; Public docs still state an unbounded wait and a diagnostic delay. |
| V4-03 | CLOSED | C:1142-1152,1217-1223; Checked device query precedes ownership/rename; failed import preserves capsule ownership. |
| V4-04 | CLOSED | O:2153-2185; Prior slot error recorded before consumption and drain; prior/OOM host gate passes. |
| V5-01 | CLOSED | O:218-249,331-380,559-591; SI:226-239; A:458-503; Deleter/capsule/producer admission retained; ordinary hooks and interrupted runners close/wait. |
| V5-02 | CLOSED | O:1864-1894; Next-call query failure recorded before consumption; one-shot host gate passes. |
| V5-03 | CLOSED | conv.cu:91-136; lstm.cu:93-138; New handle ownership survives cache-insertion OOM; unchanged by FIX8. |
| V5-04 | CLOSED | K:628-677; Wide ceil/grid refusal precedes fill launch and revision bump; huge-grid host refusal and nvcc compile pass. |
| V6-01 | CLOSED | O:331-345,2150-2160; Ordinary synchronize never waits for another thread; lock inversion and late-release gates pass. |
| V6-02 | CLOSED | C:546-600,1357-1452,1493-1559; PA:65-193,222-276; C-level argument reads/raw exit entries intact; 1067 argument checks and exit-entry census pass. |
| V6-03 | CLOSED | O:962-996,2153-2185; Drain failures persist before allocation and clear after captured RuntimeError; three drain/OOM gates pass. |
| V7-01 | CLOSED | C:1493-1559,1732-1758; A:356-407,458-503; CI:253-308; SI:226-278; Unreadable/deleted knobs and hook exceptions cannot skip quiescence; 36 exit children and lock checks pass. |
| V7-02 | CLOSED | O:780-951,1531-1580,1961-2009; G:89-117; Existing failed-call/purpose/device attribution retained; 22 host assertions pass. New core summary race is V9-02. |
| V8-01 | CLOSED | O:1169-1222,1248-1350; Marker failures recorded; fallback diagnosed; completion has proof. Four marker cases, 15 FIX8 host tests and 5 watchdog stagings pass. |
| V8-02 | CLOSED | O:2029-2185; A:428-503; C:1674-1758; K:426-439; Original final-report OOM silence fixed; both native OOM/control gates pass. New inconsistent-snapshot defect is V9-02. |

## New findings

| Finding | Severity | Location | Defect |
|---|---|---|---|
| V9-01 | MINOR | tensor_cuda/src/bindings.cpp:477 | Export fence loses an earlier CUDA failure when cleanup also fails and report allocation fails |
| V9-02 | MINOR | tensor_cuda/src/bindings.cpp:1632 | Core exit fallback retains a different snapshot from the report it acknowledges |

### V9-01 — MINOR: Export fence loses an earlier CUDA failure when cleanup also fails and report allocation fails

**OPEN.** Locations: tensor_cuda/src/bindings.cpp:465-489, tensor_cuda/src/kernels.cu:331-334.

Scenario: cudaEventRecord returns cudaErrorInvalidValue; cudaEventDestroy returns cudaErrorInvalidResourceHandle and overwrites the same thread last-error slot. Building the combined message/exception throws bad_alloc. The earlier record error exists only in local e, not in persistent core records; after unwinding only error400 remains in the slot, and unreported count is0. A later check cannot report the original error1. Single-failure OOM preserves error1; plain compound failure reports both, so only compound+OOM fails.

Evidence: Actual current-source core spans compiled by build_core_fence_cpu.py; corrected four-arm probe core_fence_cpu.py exit1. logs/p1t_rev9_core_fence_corrected.log: compound/OOM [bad_alloc, empty message, slot400, unreported0], while both controls pass. Initial child SyntaxError corrected with unchanged assertions; old log retained.

Fix: Persist each failed fence/cleanup call in allocation-free tagged records before another CUDA call can overwrite the slot, retaining them through report-construction failure; acknowledge each only once the combined report exists.

Prior art: NVIDIA CUDA12.6 Programming Guide3.2.12 (2024), one overwritable per-host-thread runtime error variable; FIX5 record-before-consume and SQLite OOM fault injection. Taken: persist errors before destructive side effects; reviewer contribution: compound fence-cleanup/report-allocation census.

### V9-02 — MINOR: Core exit fallback retains a different snapshot from the report it acknowledges

**OPEN.** Locations: tensor_cuda/src/bindings.cpp:1632-1653, tensor_cuda/src/kernels.cu:411-438, tensor_cuda/src/bindings.cpp:1715-1721.

Scenario: A foreign native thread records the first CUDA failure after dlpack_raise_unraisable reads its allocation-free raw snapshot but before it reads the rich snapshot. The rich RuntimeError names that failure and n=1 is acknowledged, but the retained exit summary uses raw_what=nullptr/raw_device=-1/raw_error=0 from the earlier empty snapshot. If the actual Python hook cannot write (sys.stderr=None), the runner prints only an unnamed failure with a false concurrent-report explanation, and consumes the summary; the actual call, device and error disappear from delivered reporting. No lock or allocation failure is needed.

Evidence: Source-linked build_core_race_cpu.py, POSIX interposer schedules actual note_core_unraisable on a native writer thread after the raw snapshot unlock; actual Python _dlpack_shutdown_hook extracted by AST unchanged. core_snapshot_cpu.py exit1; written control names cudaStreamSynchronize/device0/cudaErrorInvalidValue, unwritten arm reports only first not named although first_unraisable stats retain the correct named record. logs/p1t_rev9_core_snapshot_after_unlock.log.

Fix: Take count, static site pointer, device and error once under the record mutex; construct the rich message from that same snapshot and retain that exact snapshot in CoreExitReport. Keep records if construction fails.

Prior art: C++17 mutex synchronization/consistent snapshots (ISO2017), POSIX pthreads2008; FIX8 acknowledged exit reports. Taken: snapshot once under a lock; reviewer contribution: first-record arrival between FIX8 raw/rich snapshots.

The error identity/count findings are MINOR: they concern diagnostic retention, not demonstrated early memory release, incorrect computation or a crash. Both have concrete actual-source CPU reproductions and intact passing controls. Neither is the accepted pybind11-to-CPython translation limitation. No BLOCKING or MAJOR finding is open.

The discarded add-on recycle hypothesis is not a finding: event-pool capacity is reserved before launch and acquire removes/creates an event before it is returned, so production recycling cannot enter the forced allocation fallback in that constructor. The exploratory compound_cleanup_cpu.py and initial log stay on disk, labelled test-only; watchdog_cpu.py uses the same source-linked module to exercise an actually reachable watchdog failure path.

## Sweep (k), (l), and L2

All 34 k rows, all 7 l readers plus13 catches, and L2 checked against fresh site inventory. Lexical site coverage confirmed; compound cleanup/report OOM missing from k29 verdict (V9-01), concurrent raw/rich snapshot missing from l6 verdict (V9-02). L2 confirmed. No additional product site omission found.

| Rows | Current sites | Verdict | Independent check |
|---|---|---|---|
| k1 | O:1169-1222 | CONFIRMED FIXED | All four marker failures record before consume; independent four-case and plain/OOM tests pass. |
| k2-k4 | O:1215,1562,1889 | CONFIRMED NOT A FAILURE | NotReady is incomplete work, not an error; only defensively consumed when slot equals NotReady. |
| k5-k8 | O:1537-1569,1864-1894,2154-2158 | CONFIRMED OK | Query/current/prior errors copied into plain records before consumption; prior/query/OOM checks pass. |
| k9-k12 | O:1328-1350,1365-1375; G:152-158,176-199 | CONFIRMED OK | Noexcept return/value storage cannot allocate between consumption and retaining error; all QuietGuard users record/raise it. Restore errors record first. |
| k13-k14 | G:57-97 | CONFIRMED FIXED | Message and runtime_error exist before consumption; four helper OOM/control tests pass. |
| k15 | O:1595-1622 | CONFIRMED FIXED | Failed event creation/guard recorded in bad_alloc catch; actual source acquire/OOM assertion passes. |
| k16-k18 | G:99-117; conv.cu:52-59 | CONFIRMED FIXED | Launch/cuBLAS-after checks peek and share report-first helper; host helper assertions pass. |
| k19 | G:119-122 | CONFIRMED OK | cuBLAS status reporting does not consume CUDA slot; scope distinguished. |
| k20 | O:1756-1792 | CONFIRMED FIXED IN PRODUCTION | Prior peek retained on allocation failure; recycled event has reserved capacity. Test-only forced recycle fallback is not production reachability. |
| k21-k22 | O:1807-1838,1414-1425 | CONFIRMED OK | Record/destroy failures noted before consume; guard-call attribution kept. |
| k23-k24 | K:289-320,371-379 | CONFIRMED OK | Imported drain and guard restore failures recorded before consumption. |
| k25-k28 | K:331-337,421,652-674; C:450-461 | CONFIRMED FIXED FOR SINGLE ERRORS | Report-first helpers retain slot through bad_alloc; no consuming call in these direct reporting steps. |
| k29 | C:465-489 | PARTLY — DEFECT (V9-01) | Failed fence and failed cleanup can overwrite each other before reporting; compound/OOM failure scenario omitted. |
| k30-k32 | C:1142-1147,1260-1287 | CONFIRMED FIXED / OK | Query raises share report-first helper; allocation catch rethrows already-built engine runtime_error without new reporting allocation. |
| k33-k34 | K:255-275; O:1640-1647; conv.cu:108-127; lstm.cu:110-129 | CONFIRMED OK | Atomic unnamed count survives record-lock/OOM failure; new-handle cleanup status counted, not CUDA-slot consumption. |

| Rows | Current sites | Verdict | Independent check |
|---|---|---|---|
| l1 | O:2104-2122 | CONFIRMED FIXED | Full report else plain summary; record consumed; both original OOM/control cases pass. |
| l2 | A:428-503 | CONFIRMED FIXED | ASCII existing message avoids error_already_set::what; final readers reached on both normal/raised paths. |
| l3-l4 | O:2083-2147; SI:240-278 | CONFIRMED FIXED | Raised summary retained until lines acknowledged; actual add-on normal/None/raising/interrupt/closed-fd cases pass. |
| l5 | C:1692-1758; K:426-439 | CONFIRMED FIXED | Raw record snapshot and allocation-free fallback reached after hook on both paths; original core OOM/control probe passes. |
| l6 | C:1632-1653,1715-1721; CI:282-300 | PARTLY — DEFECT (V9-02) | Acknowledged rich report and retained summary use different snapshots; foreign native writer scenario omitted. |
| l7 | A:466-503; C:1739-1758 | CONFIRMED OK | First-bytecode exception preserved, admission closed before final reads; interrupted-hook CPU gates pass. |
| c1-c6 | O:2115-2122; A:471-493; C:1701-1713; SI:264-278 | CONFIRMED | Typed catches retain/report records and do not bypass final reader. Failed stderr acknowledgement tested. |
| c7-c13 | C:1751-1758; K:255-275,426-458; O:569-574,2183-2185 | CONFIRMED WITH l6 CAVEAT | Non-CUDA cleanup exceptions counted/reported or leave CUDA records for last reader. pybind11 discard_as_unraisable uses restore/WriteUnraisable, not allocating what(). Snapshot consistency still fails l6. |

| L2 item | Current sites | Verdict | Independent check |
|---|---|---|---|
| last-access flag | O:1293-1315,1322 | CONFIRMED | All schedule/mutex/CV access, including unlock destruction, ends before release-store exited_. |
| stop protocol | O:1267-1290 | CONFIRMED | Repeated notify until acquire sees exited_; lock failures counted. Host lock1/3 with missed notify and held-writer join failure pass. |
| start failure | O:1250-1261 | CONFIRMED | Host thread allocation failure prints immediate diagnostic and finite sync still finishes. |
| join fallback | O:1280-1287 | CONFIRMED IN VALID-THREAD SCOPE | Detach after final state access is safe; no watchdog operation after detach in host interposer receipt. Stated stderr backpressure limitation retained. |

No additional unlisted production CUDA-call or last-reader site was established. The inventory is artifacts/p1t_rev9/cuda_calls.json and cuda_site_census.txt (raw inventory includes baseline core calls, which are excluded from findings unless newly reachable). All 14 add-on CUDA API names and enclosing guards/report/release users were traced. Completeness of a lexical site list is not completeness of its failure scenarios: k29 and l6 are the two specific missed cases.

The remaining k rows are correct within the stated capacity/valid-object contracts: no-throw value retention is sufficient where a return must immediately enter a record; NotReady is not a failure; cuBLAS statuses are distinguished from the CUDA runtime slot. The watchdog's final release/acquire handshake publishes schedule writes and completes all mutex/CV member use before any detach. Stderr backpressure remains the explicitly documented transport limit. Application allocation-free summaries use pre-existing POD/static texts and C stdio; they do not format a pybind11 lazy what().

Corrected FIX7 rows: i24: CONFIRMED FIXED for add-on; original marker and OOM gaps closed.; i25: PARTLY in new core snapshot race V9-02; original OOM gap closed.; j24: CONFIRMED FIXED; no lazy allocating error_already_set::what.; a3: CONFIRMED FIXED; source and host diagnosed fallback/L2 checks.; a9: CONFIRMED corrected rationale: rich string copy can allocate under record mutex; raw snapshot copies only POD/static pointer.

## FIX8 user-visible changes and not-a-defect claims

| Item | Verdict |
|---|---|
| U1 | ACCEPTABLE |
| U2 | ACCEPTABLE |
| U3 | ACCEPTABLE |
| U4 | ACCEPTABLE |
| U5 | ACCEPTABLE |
| U6 | DEFECT — MINOR (V9-02) |
| U7 | DEFECT — MINOR (V9-01) |
| U8 | ACCEPTABLE |

- **U1:** O:1169-1222,2150-2185: marker failures recorded and raised once; fresh 4-case probe and 8 plain/OOM cases pass.
- **U2:** O:1248-1350: watchdog touches no CUDA/Python, retains schedule and L2 handshake; control/lock1/lock3/join/start-OOM host cases pass.
- **U3:** O:1208-1221,1345-1346: finished requires successful query or sync; failed-query+sync case never prints finished.
- **U4:** A:458-503; C:1732-1758: final readers called after actual hook on normal and exceptional outcomes; original exit OOM controls pass.
- **U5:** O:2029-2122; C:1674-1713; K:426-439: application allocation-free count/first summaries replace lost reports; both native OOM controls pass.
- **U6:** Core C:1632-1653 retains an earlier raw snapshot than the acknowledged rich report. A first native failure between snapshots loses its known call/device/error in fallback. Add-on acknowledgement cases pass.
- **U7:** Core C:465-489 cleanup overwrites an earlier failure before the combined report can allocate; reordered throw helper preserves only the last slot value. Single-error OOM paths pass.
- **U8:** O:2253-2258; C:1763-1794, stats bindings; additive diagnostics/ack hooks exposed; raw noargs functions reject malformed calls through CPython.

| Item | Verdict |
|---|---|
| N1 | CORRECTLY NOT A DEFECT |
| N2 | CORRECTLY NOT A DEFECT |
| N3 | CORRECTLY NOT A DEFECT |

- **N1:** CPython/pybind11 translation OOM after a C++ report exists still raises MemoryError; P1-T cannot guarantee interpreter allocation. V9-01 occurs earlier, before the C++ report exists, and is not excused by this item. Explicit report-record APIs preserve their records until a Python exception exists.
- **N2:** Direct hook calls/unregister intentionally bypass documented runner protection. Both actual packages register native runners (CI:308; SI:283). Our V9-02 reproducer runs the actual hook through its actual runner.
- **N3:** Closed fd2 leaves no delivery channel; attempted C stdio reporting consumes records without crash. A Python stderr=None while fd2 stays open is different and must retain named fallback, as V9-02 shows.


**D2 / D12: SOUND.** Independent source comparison confirms exactly one stream-context line changed to torch.cuda.default_stream(). Recorded torch2.11 ExternalStream(0) has flags1/nonblocking and legacy sync returns without waiting; CUDA legacy/nonblocking semantics justify no waiting diagnostic. Historical lead before/after legacy-copy logs show all four before children fail and all four after children pass. No physical rerun here; not generalized to all torch versions.

Receipts: legacy_probe_diff.txt is exactly the single stream-context change; artifacts/p1t_fix8/extstream_receipt.txt records nonblocking flags1, a 0.0000s legacy wait and stream handle different from torch's legacy/default stream0. The lead's before/after logs retain the as-written RED and show the corrected scenario's treatment effect. No product-side fake wait diagnostic is warranted. This is a sound host/version-specific ruling, not a universal claim about every ExternalStream implementation.

## Operand and execution census

| Area | What was exercised or checked | Boundary |
|---|---|---|
| Scalar/empty/int64 metadata | 17 live add-on descriptor/core host checks plus fix2 ck_census and fix3 accounting UBSan: 0-d rint; empty add-on refusal; valid empty core; overflow beside zero; offset alignment/wrap; singleton stride; read-only/copied/versioned/consumed capsule; fill grid refusal. | No device allocation or kernel launch for synthetic descriptors. |
| Required operands and aliases | Required-None checks, same capsule passed twice/refused and reused capsule refusal; all written-vs-input/output range checks before RetainUntilComplete. Producer input used twice can export two independently owned capsules; raw capsule can be consumed once. | Device data contents not executed. |
| Arguments | 1067 checks across29 add-on bindings, typed positions and keywords; int/float/tuple/list/str subclasses; NumPy scalars; Python conversion-method objects. Both raw exit entries/runners exercised; public wrapper conversions occur in Python frames. | Finite type census; no assertion over all conceivable Python objects. |
| Release/concurrency | 14 release checks, native throwing-deleter and OOM harnesses; synchronize under another thread release and caller-held lock; 36 exit children; three lifecycle lock-failure stagings; five watchdog failure stagings. | Host real pthread/GIL behavior; CUDA completion/failure scheduling is simulated in source-linked trace modules. |
| Exit reporting | Original add-on/core logger full/OOM controls;15 FIX8 host assertions; five actual CUDA12.6 add-on native-hook cases, including stderr None/raising/KeyboardInterrupt/closed fd. V9-02 runs actual core hook/native writer with intact fd2. | No-GPU CUDA runtime itself reports prior NoDevice in actual add-on hook staging: summary counts2 and correctly names that first prior failure. |
| Streams/devices | Own-device guards, consumer fence, producer stream1 handoff and checked import device; host multi-device/spill/restore traces; CUDA-error slot failures before sync/ops/fill/export read from source and report helpers. | Physical producer-stream ordering and two-device correctness require lead GPU lanes; two-device lane exits3 on one GPU. |
| Arithmetic/kernels | Conv forward im2col/row-major SGEMM, grouped/transposed gather indices; ONNX LSTM iofc and reverse indexing, state/scratch sizes; norm affine/index/tree synchronization; nearest/reflection bit copies; rintf tie-to-even. All planned counts/bytes bound device indices and grid-stride increments; core fill wide ceil refuses grid overflow. | Source correctness and compilation, not an independent device numeric/tolerance proof. Voice tolerances excluded. |
| Package separation | Fresh complete add-on CUDA12.6 build and ELF; core bindings syntax and kernels.cu nvcc12.6 compile; core host-span modules link. | Initial direct CMake default selected distro12.0; corrected fresh126 build supplies separation evidence. Entire production core build not rerun. |


## Validation receipts and baseline isolation

CPU gate summary: artifacts/p1t_rev9/cpu_gates.json, cpu_results.json, argument_census.json, operand_census.json, attribution_cpu.json, fix7_cpu_census.json, lock_checks.json, fix8_host.xml. All original host gates pass, with15 FIX8 host tests (8 marker plain/OOM, healthy/proof controls, acquisition OOM and4 guard-helper cases). Additional watcher and native acknowledgement logs pass; new defect probes intentionally return1. Source-linked core spans have file/start/end/SHA256 receipts. Build result files: cpu_build.exit=0, core_addon_build.exit=0, core_kernels_compile.exit=0, addon126_build.exit=0.

Package ELF: CUDA12.6 scratch add-on RUNPATH /usr/local/cuda-12.6/lib64 and shared libcudart.so.12/cublas.so.12; no core ABI undefined symbols/dependency. The checked-in core resolves libcudart12.6 too. The initial12.0 direct CMake receipt is retained as extra compilation evidence, not used for the shared126 claim. Hashes and symbol checks are retained; F07 native rebuild requirement still applies to any old out-of-tree Storage client.

Historical lead evidence independently checked, not relabelled as reviewer execution:
- Parsed baseline and combined-session failure/error ID sets:52 and52, new0, disappeared0 (baseline_compare.json). Combined log:37 failed,3040 passed,47 skipped,15 errors; these are historical baseline failures, not new findings here.
- Voice runner log/counts:249 PASS,0 FAIL/SKIP/ERROR. Voice acceptance is outside this review; no new execution.
- FIX8 mutants e/f/g/h summaries show nonzero assertion failures after successful rebuilds; source reasoning agrees with what they remove. Historical receipts do not establish coverage of V9-01/V9-02.
- Lead legacy-stream before/after probe supports the D2/D12 ruling; no GPU rerun. Physical two-device lane remains INCONCLUSIVE on the lead's one GPU.

Deviations/harness issues are retained in the ledger and older logs: mandated mechanism opening exposed author verdicts before source review; no author verdict was accepted without independent source/evidence checks. Initial CPU-build tool result lost its session identifier, so a bounded foreground rerun supplied the credited result. Copied FIX7 census framing corrected only RESULT-line interleaving. Core-fence child had an unmatched parenthesis before correction; acknowledgement probe had an initial namespace import path and an incorrect first-record expectation before correction to the actual no-device census. The add-on recycle hypothesis was refuted by production reachability. No failed harness run is credited as a pass.

## Probe list

| Path(s) | Result | One-line decision |
|---|---|---|
| artifacts/p1t_rev9/probes/build_cpu.sh | PASS | Current-source release/UBSan harnesses and core bindings syntax. |
| artifacts/p1t_rev9/probes/build_core_host.py + build_core_host_oom.py | PASS | Current-source core spans compile/link; exact span hashes retained. |
| artifacts/p1t_rev9/probes/run_cpu_checks.py | PASS 14/14 | Release, OOM, byte accounting, concurrent synchronize and throwing-deleter CPU cases. |
| artifacts/p1t_rev9/probes/argument_census_cpu.py | PASS 1067 | Every add-on typed argument position/keyword style plus core dispatch; no Python conversion method entered. |
| artifacts/p1t_rev9/probes/operand_census_cpu.py | PASS 17 | Scalars/empty/huge extents/offset/flags/consumed-twice/stride/grid metadata without launches. |
| artifacts/p1t_rev9/probes/attribution_cpu.py | PASS 22 | Current-source failed-call/purpose/device attribution, plain/OOM. |
| artifacts/p1t_rev9/probes/fix7_cpu_census.py | PASS 36 children | Knob classes, deleted configuration, hook exceptions, both raw close entries and native runners; robust RESULT framing only. |
| artifacts/p1t_rev9/probes/lock_checks_cpu.py | PASS 3 | Core close/relock retry and add-on lifecycle polling quiescence/diagnostics. |
| artifacts/p1t_rev9/probes/knob_exit_cpu.py + core_knob_exit_cpu.py + query_diagnostic_cpu.py | PASS | Finite callback exit despite huge knob; query failure names query after successful sync. |
| artifacts/p1t_rev9/probes/diagnostic_errors_cpu.cpp + diagnostic_errors_cpu.py | PASS 4 | Original V8-01 marker error retention against current source. |
| artifacts/p1t_rev9/probes/exit_log_oom_cpu.cpp + verify_exit_log_cpu.py + verify_core_exit_log_cpu.py | PASS | Original V8-02 final logger control/allocation-OOM, both packages. |
| artifacts/p1t_rev9/probes/compound_cleanup_cpu.cpp + watchdog_shim.c + watchdog_cpu.py | PASS 5 watchdog stagings | Current-source L2 stop/start protocol: control, lock1/3 lost notify, join failure and thread-start allocation failure. |
| artifacts/p1t_rev9/probes/acknowledgement_cpu.py | PASS 5 | Actual CUDA12.6 native add-on and actual hook: acknowledgement under normal/None/raising/interrupt/closed stderr, no launch. |
| artifacts/p1t_rev9/probes/build_core_fence_cpu.py + core_fence_cpu.py | RED (V9-01) | Four-arm export-fence error retention: compound CUDA cleanup failure plus report OOM loses prior error; other three controls pass. |
| artifacts/p1t_rev9/probes/build_core_race_cpu.py + core_snapshot_shim.c + core_snapshot_cpu.py | RED (V9-02) | Native first-record arrival after raw snapshot unlock must still deliver its known site/device/error when hook cannot write; written control passes. |
| artifacts/p1t_rev9/probes/diagnostic_event_shim.cpp + diagnostic_events_gpu_legacy.py | LEAD ONLY — NOT RUN | Single-GPU marker errors and real finite legacy-stream wait; exit3 with no GPU. Tests accepted D2/D12 scenario. |
| artifacts/p1t_rev9/probes/gpu_regressions.py fix8 / fix7 / fix56 / semantics / multidevice | LEAD ONLY — NOT RUN | Current reviewed binaries run original regressions, arithmetic and producer-stream tests, zero skips; multidevice exits3 on one GPU. Other lanes single-GPU. |


The FIX8 host-test extraction is /tmp/p1t_rev9_build/test_fix8_host.py; its15 tests are AST-selected original CPU tests without the irrelevant module-level GPU skip guard. No GPU test body was selected. Builds and probe execution stayed foreground under timeout; yielded runs were resumed with maximum300000ms waits. No process not started by this seat was killed. Scratch listed for reproducibility: /tmp/p1t_rev9_build, /tmp/p1t_rev9_pytest_host, /tmp/p1t_rev9_finalize.py, /tmp/p1t_rev9_house_rules_read.txt. No DELETE_ME moves were needed.

## Prior art

- **V9-01:** NVIDIA CUDA12.6 (2024), Programming Guide §3.2.12: per-thread error state is overwritten on later failures. FIX5 record-before-consume is reused; proposed fix persists each value before cleanup can overwrite it. SQLite (D. Richard Hipp/project, ongoing; exact OOM-testing introduction year unverified — lead to check "SQLite malloc failure testing history") informs fault injection. Reviewer contribution is the compound fence-cleanup/report-OOM scenario, not a new error-accounting algorithm.
- **V9-02:** ISO C++17 (2017) mutex serialization and POSIX pthreads (2008): each individual locked read is consistent, but two different reads can observe different generations. Proposed fix uses the existing allocation-free raw snapshot once for message, acknowledgement and fallback. Reviewer contribution is scheduling a native record immediately after raw-snapshot unlock. Probe-site comments and ledger carry these annotations; product comments were not edited because this is read-only code review.
- **Existing contracts:** DLPack (DMLC,2017+), Python array API (Consortium,2021-2023), CPython3.12 (PSF,2023), pybind11v2.12 (Jakob et al.,2024), ONNX operators (ONNX project,2017+), CUDA12.6 (NVIDIA,2024). L2 uses the existing release/acquire last-access handshake and repeated notification; Mesa lost-wakeup literature (Lampson/Redell,1980) is a named lead already annotated in code. No new product algorithm was implemented.

Primary sources checked live: [CUDA12.6 error handling](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#error-handling), [DLPack Python specification](https://dmlc.github.io/dlpack/latest/python_spec.html), [CPython3.12 atexit](https://docs.python.org/3.12/library/atexit.html), [CPython3.12.3 errors.c](https://github.com/python/cpython/blob/v3.12.3/Python/errors.c), [pybind11v2.12 translator](https://github.com/pybind/pybind11/blob/v2.12.0/include/pybind11/detail/internals.h), [C++ thread member contracts](https://eel.is/c++draft/thread.thread.member). These support standard/lifecycle boundaries; CPU trace receipts establish the two particular implementation failures.
