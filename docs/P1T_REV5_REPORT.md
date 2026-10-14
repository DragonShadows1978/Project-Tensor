# P1T-REV5 independent review

**REJECT merge of `9df5bef..1475fae`. Zero BLOCKING, one MAJOR, three MINOR, zero NIT findings.** Earlier findings are CLOSED in their original specific scope. The add-on can abort at ordinary interpreter exit with a finite producer deleter running on another thread (V5-01). FIX4 also loses a recoverable event-query diagnostic (V5-02). David's conditional merge approval is not met. No defect is claimed fixed by this review.

| Question | Verdict |
|---|---|
| R1 | SAFE |
| R2 | SOUND |
| R3 | UNSOUND |
| R4 | DEFECTS |
| R5 | PASS |
| R6 | REJECT |

## Scope and evidence

Evidence classes: source reasoning, CPU full-extension reproduction, CPU actual-source call traces, checked-arithmetic/UBSan checks, CPU build/link inspection, live fixture metadata and historical receipt audit. No GPU execution. All52 submitted files byte-match1475fae (source_inventory.json). Both full and FIX4 diffs are saved (full.diff, fix4.diff); production core/add-on implementations and wrappers/builds read before fix reports. Registration precedes gates and is immutable. The mechanism matches the approved core-DLPack/add-on split.

CLOSED means the earlier concrete defect is closed by the stated code mechanism, not universal certification of every failure/thread/device/input. N5/V3-01 concern core import releases; V5-01 is the independently uncoordinated add-on counterpart. F03's original rank/count/byte/address/geometry fixes are intact; the new fill grid issue is V5-04. No baseline full/other engine operation is reclassified as newly defective. Free-threaded Python, subinterpreters and producer-initiated cancellation remain outside certification; V5-01 uses an ordinary finite callback on CPython3.12.3, not producer cancellation.

## Closure table

| Item | Status | Lines read and evidence |
|---|---|---|
| F01 | CLOSED | tc_signal/src/dlpack_operand.cpp:787-875; bindings.cpp:284-286; lstm_bindings.cpp:239-242; m4_bindings.cpp:239-241,294-295,315-316,334-335 — Event-retained managers cover success/partial launch; checked fallback leaks without completion proof. V5-01 is a separate shutdown gap. |
| F02 | CLOSED | tc_signal/src/dlpack_operand.cpp:197-201,315-324; m4_bindings.cpp:65-104; bindings.cpp:240-260; lstm_bindings.cpp:178-210 — Required None refusal; absent metadata raises. Fresh CPU refusal checks pass. |
| F03 | CLOSED | tensor_cuda/src/bindings.cpp:312-336,937-988,1017-1026; tc_signal/src/checked_math.h:31-89; dlpack_operand.cpp:127-194 — Original rank/count/byte/address/geometry overflow paths checked. V5-04 separately finds fill grid narrowing. |
| F04 | CLOSED | tensor_cuda/src/bindings.cpp:710-775,794-847; tensor_cuda/tensor_cuda/__init__.py:201-236 — Core admission closes before finalization; late releases leak without GIL. Add-on callback gap is V5-01. |
| F05 | CLOSED | tensor_cuda/src/kernels.cu:232-248,338-364 — External Storage drains its own device; failed drain leaks; caller restore checked/counted. |
| F06 | CLOSED | tensor_cuda/src/bindings.cpp:531-570 — unique_ptr retains export context until capsule transfer; metadata exceptions unwind ownership. |
| F07 | CLOSED | tensor_cuda/include/tc/core.h:80-88; docs/P1T_FIX1_M1_ADDENDUM.md:20-33 — Coordinated native rebuild documented; closure is rollout prerequisite, not old ABI compatibility. |
| N1 | CLOSED | tc_signal/src/dlpack_operand.cpp:695-734,924-988; tc_signal/tc_signal/__init__.py:139-155 — Original query/sync failures raise; CPU no-device check passes. V5-02 is a distinct query-error accounting gap. |
| N2 | CLOSED | tensor_cuda/src/bindings.cpp:312-336,515-517; kernels.cu:80-98; tc_signal/src/checked_math.h:63-89 — Zero-neighbor extent/byte products checked; fresh 18-case UBSan census passes. |
| N3 | CLOSED | tc_signal/src/dlpack_operand.cpp:480-484,593-607,695-734,737-760,787-875 — Pre-reserved owners/recycle slots; nonallocating drain; pending push caught. CPU OOM propagation passes. |
| N4 | CLOSED | tc_signal/src/dlpack_operand.cpp:238-260; tensor_cuda/src/bindings.cpp:757-763 — Ordinary C++ deleter exceptions contained once; native CPU throwing-owner check passes. |
| N5 | CLOSED | tensor_cuda/src/bindings.cpp:734-775,794-847; tensor_cuda/tensor_cuda/__init__.py:201-236 — Original core admission/GIL race coordinated. Add-on has its own uncoordinated releases (V5-01). |
| V3-01 | CLOSED | tensor_cuda/src/bindings.cpp:794-847; tensor_cuda/tensor_cuda/__init__.py:201-236 — No bounded early return; fresh core 5.5s lifecycle slice exits0, in-flight0, diagnostic1. |
| V3-02 | CLOSED | tc_signal/src/dlpack_operand.cpp:630-663,774-795,827,871-875 — Checked aggregate before launch and sticky saturated queue total; 17 fresh UBSan accounting checks pass. |
| V4-01 | CLOSED | tensor_cuda/src/bindings.cpp:414-477,525-529; kernels.cu:322-335 — Fence selects tensor device; label proof; checked destroy/restore. Real two-device witness still pending. |
| V4-02 | CLOSED | tensor_cuda/tensor_cuda/__init__.py:107-116 — Import docstring describes unbounded wait and diagnostic-delay knob. |
| V4-03 | CLOSED | tensor_cuda/src/bindings.cpp:916-928,993-999 — Device query checked before Storage construction/capsule rename. Historical full-core refusal receipt audited. |
| V4-04 | CLOSED | tc_signal/src/dlpack_operand.cpp:924-988 — Prior error taken before drain, kept through message allocation and raised. Source and historical one-GPU receipt agree. |

## New findings

| ID | Severity | File:line | Defect | Fix |
|---|---|---|---|---|
| V5-01 | MAJOR | tc_signal/src/dlpack_operand.cpp:238-260,668-673,878-900; tc_signal/tc_signal/__init__.py:139-155 | Interpreter exit overtakes a producer deleter already running on another thread | Track admitted add-on releases independently of the pending queue; close admission in atexit and wait with the GIL released for all other-thread admitted callbacks to leave before finalization. Leak later releases without entering Python, use self-exclusion, and contain ordinary C++ exceptions only. Apply this to immediate, deferred and capsule callback release paths. |
| V5-02 | MINOR | tc_signal/src/dlpack_operand.cpp:878-894 | A recoverable event-query error is consumed without being reported or remembered | Persist non-NotReady query failures before consuming the slot, and report them once at synchronize while continuing safe completion/release logic. |
| V5-03 | MINOR | tc_signal/src/conv.cu:89-105; tc_signal/src/lstm.cu:112-128 | A successfully created cuBLAS handle leaks when the host cache insertion fails | Keep a newly created handle in scoped ownership through both settings and map insertion; destroy it on every exception before insertion commits, checking cleanup status without throwing during unwinding. |
| V5-04 | MINOR | tensor_cuda/src/kernels.cu:469 (inherited nblk/fill_kernel at398-403) | The new fill_ narrows a supported int64 element count to a wrapped grid and can silently fill only a prefix | Before narrowing in the new fill_ path, calculate the block count in int64 and explicitly refuse counts exceeding the supported grid.x bound. Alternatively use a reviewed capped grid-stride fill path, without silently narrowing; preserving the no-new-core-kernel plan favors the refusal. |

### V5-01 — MAJOR: interpreter exit overtakes an add-on producer deleter already running on another thread

Location: `tc_signal/src/dlpack_operand.cpp:238-260,668-673,878-900`; `tc_signal/tc_signal/__init__.py:139-155`.

Scenario: a daemon thread calls the add-on with a versioned read-only output. The Operand constructor consumes the capsule, refuses the write and calls its producer deleter during unwinding. The finite deleter sleeps, releasing the GIL. The main thread runs the add-on atexit hook, which synchronizes a queue with zero entries and returns. CPython starts finalizing before the callback finishes. When the callback attempts to reacquire the GIL, CPython 3.12.3 exits that thread; the forced unwind through the catch-all/noexcept cleanup boundary aborts the process. Deferred release has the same gap: the entry is popped before callbacks run, so a running callback is invisible to the queue. The core's separate import-release counter does not cover arbitrary add-on producers.

Evidence: the freshly built, unchanged full add-on aborts under `timeout 15` with rc134 and `FATAL: exception not rethrown` (`logs/p1t_rev5_addon_exit_full_cpu.log`). This refusal happens before event acquisition or any launch; no GPU execution is needed. The joined-worker control exits0 and prints `DELETER finished` before `FINALIZING True`. An independent actual-source deferred-release trace also aborts (child rc-6) after `CALLBACK IN PROGRESS entries=0` and `HOOK returned entries=0`; its CUDA calls are replacements, so that trace establishes host lifecycle only. The full deferred GPU probe remains pending.

Fix: give add-on releases an admission/closing/in-flight protocol independent of the pending queue. At atexit, close admission and wait with the GIL released until every other-thread admitted release has left; use self-exclusion for a hook invoked within its own callback. Late releases must leak without entering Python. Cover immediate refusals and deferred/fallback/bound drains. Keep ordinary C++ deleter exception containment; allowing finalization to manufacture a forced unwind through that boundary is unsafe. No fix is implemented by this review.

Prior art: CPython 3.12 lifecycle/GIL rules (Python Software Foundation, 2023), C++17 noexcept, and the standard admission/quiescence pattern already used by core `ImportReleaseLifecycle`. Taken: those contracts and the existing coordination pattern. This review contributes the add-on callback/ordinary-exit scenarios. [CPython lifecycle](https://docs.python.org/3.12/c-api/init.html), [pinned CPython 3.12.3 GIL source](https://raw.githubusercontent.com/python/cpython/v3.12.3/Python/ceval_gil.c).

### V5-02 — MINOR: A recoverable event-query error is consumed without being reported or remembered

Location: `tc_signal/src/dlpack_operand.cpp:878-894`.

Scenario: cudaEventQuery in drain_completed returns a one-time error, then later stream synchronization and event query succeed. drain_completed clears the slot but records no error; the next synchronize silently succeeds.

Evidence: CPU actual-source trace query_error_cpu.py: slot 0; release_errors and unreported_release_errors 0; next synchronize success, manager deleted once. Interposer GPU probe pending.

Fix: Persist non-NotReady query failures before consuming the slot, and report them once at synchronize while continuing safe completion/release logic.

Origin: FIX4 consumes the query error without preserving it.

Prior art: NVIDIA CUDA 12.6 event-query return-code and last-error contracts (2024). Taken: checked returns and consume-after-accounting; ours: one-shot query-failure scenario.

### V5-03 — MINOR: A successfully created cuBLAS handle leaks when the host cache insertion fails

Location: `tc_signal/src/conv.cu:89-105; tc_signal/src/lstm.cu:112-128`.

Scenario: All handle settings succeed, then handles.emplace(device,h) throws std::bad_alloc. The raw handle has no owner/catch on this path, never reaches the cache, and is never destroyed. Repeated attempts can leak a handle and its device resources per failed host insertion.

Evidence: CPU unchanged-function slices with successful cuBLAS-call replacements and one real operator new failure after the last setting: conv creates=1 destroys=0; lstm creates=1 destroys=0; probe rc1. No CUDA execution.

Fix: Keep a newly created handle in scoped ownership through both settings and map insertion; destroy it on every exception before insertion commits, checking cleanup status without throwing during unwinding.

Origin: Initial P1-T cache creation; FIX4 D7 closes setting failures only.

Prior art: ISO C++17 RAII/exception safety (2017), NVIDIA cuBLAS12.6 create/destroy resource contract (2024). Taken: scoped ownership until transfer; ours: cache-insertion OOM witness.

### V5-04 — MINOR: The new fill_ narrows a supported int64 element count to a wrapped grid and can silently fill only a prefix

Location: `tensor_cuda/src/kernels.cu:469 (inherited nblk/fill_kernel at398-403)`.

Scenario: For n=2**40+1 fp32 elements, the shape/byte bounds fit int64. fill_ calls the baseline nblk helper, which narrows ceil(n/256)=2**32+1 to int1 on the reviewed ABI. The non-grid-stride fill kernel then writes only256 elements, its launch succeeds, and the revision is bumped. This is a huge descriptor boundary (4TiB+4bytes), not a resident array reproduction on the lead GPU.

Evidence: CPU exact-helper arithmetic witness: n=1099511627777 grid=1 covered=256; rc1, no UBSan error because signed narrowing is implementation-defined. The defect is the new fill_ call site reusing an insufficient helper; no baseline full/other engine op is classified as newly broken.

Fix: Before narrowing in the new fill_ path, calculate the block count in int64 and explicitly refuse counts exceeding the supported grid.x bound. Alternatively use a reviewed capped grid-stride fill path, without silently narrowing; preserving the no-new-core-kernel plan favors the refusal.

Origin: Initial new fill_ binding uses a pre-existing helper without a new size rail.

Prior art: ISO C++17 integral conversions and NVIDIA CUDA12.6 grid/dim3 semantics. Taken: checked narrowing/launch bounds; ours: int64 fill boundary witness.

V5-02 applies the distinction between observing an error and erasing its thread slot: a queue entry does not retain the observed error merely by remaining queued. A later success is not a record of that earlier failure. Its one-shot interposer is an explicit failure-path witness; no claim is made that every healthy GPU naturally produces this return sequence. [CUDA event-query contract](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-runtime-api/group__CUDART__EVENT.html), [CUDA error-slot contract](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-runtime-api/group__CUDART__ERROR.html).

V5-04 is a very large metadata boundary; no4TiB resident array was allocated. CUDA's valid1-block launch and the reviewed non-grid-stride kernel cover256 elements, as source reasoning. The one-GPU probe traps/refuses the launch on synthetic metadata without touching the small backing memory; it validates call placement, not a4TiB numerical run.

## Sweep-table check

Independent added-line lexical scanning (comments/string literals excluded) finds **148 product sites:114 CUDA runtime/cuBLAS/launch sites and34 error-string lookups,0 missing locations**, matching111 table rows. Every changed production file was scanned, including the new guard. `call_inventory.json` records each site. Scope is direct sites in the submitted diff; baseline allocator/`cuda_check_last` calls were separately followed. No additional direct CUDA/cuBLAS call site is missing.

The all-OK/FIXED verdict claim is false: rows47/50 lose the query diagnostic (V5-02). D7 handles failed cuBLAS settings correctly but does not cover host cache insertion (V5-03). Row111 correctly checks device/error handling yet leaves grid narrowing unchecked (V5-04). Host callback quiescence (V5-01) is also absent; a CUDA-call inventory does not establish it. Specific row spot checks follow; agreement is source reasoning plus the bounded receipts named elsewhere, not a fresh GPU pass.

| FIX4 row(s) | Review verdict | Evidence / scope |
|---|---|---|
| 4,5,6,7,8,9,55,56,57,58,59,60 | AGREE for CUDA/cuBLAS returns; LIMITED cleanup | Creation/settings/destruction statuses checked on operand device; insertion OOM not covered (V5-03). |
| 23,24,25,26,27,28,29,30,31,32 | AGREE | Shared guards check query/switch/restore; success restores raise, unwinding/quiet restores counted and deferred. |
| 33,35,39,41,45 | AGREE | Sync/event calls guarded or legal cross-device query; release/fallback/error records handled conservatively. |
| 44 | AGREE at this site | Retainer takes an existing slot before launches; no claim this can recover an error already discarded by the earlier drain. |
| 47,50 | DISAGREE | Query failure cleared without persistent record; later success cannot re-report it. CPU source trace settles V5-02. |
| 48,49 | AGREE | NotReady is normal incompletion, defensive clearing conditional on matching slot. |
| 51 | AGREE | synchronize captures entry error before CUDA queries/deleters and reports after safe drain. |
| 61,62,63,64,65,72,73,76,77,78,79,80,81 | AGREE | LSTM memcpy/memset/GEMM and all norm/small launches use operand DeviceGuard and check errors; arithmetic reviewed separately. |
| 83,84,85,86,87,88,89,90 | AGREE | Export switch/event create-record-wait/destroy/restore all checked, no allocating C++ step between switch and restore. |
| 91,92,93,94,95 | AGREE | Import query precedes ownership; empty labels current device/refuses explicit mismatch and consumes failed allocation return. |
| 96,97,98,99,100,101 | AGREE | External drain checks query/switch/sync and counts failed restore while preserving safe release after completed drain. |
| 104,105,106,107,108,109,110,111 | AGREE for device/error checks; LIMITED geometry | Label proof and fill guard/restore correct; row111 does not check grid narrowing (V5-04). |

## FIX4 deviations D2–D7

- D2 label proof: export/fill query element0 and refuse label mismatch before fence/write; zero arrays skip memory proof. Read bindings525-529 and kernels322-335. tc_signal trusts producer descriptors as the DLPack contract; deliberate mislabelled foreign capsules are producer violations. `__dlpack_device__` advertises the label, then actual export refuses an incorrect allocation label.
- D3 empty labeling: checked current-device query, explicit cuda:N must equal current; cpu allocation and zero/rank/byte rules reviewed. Read bindings1015-1065. Pre-existing Device::from_string permissiveness remains baseline.
- D4 pre-launch take: fill reads the slot after label proof/guard, retainer after event acquisition. Source location is correct for detecting a slot that still exists. V5-02 demonstrates that earlier drain can already erase a diagnostic without accounting.
- D5 synchronize: capture first, still drain, keep prior diagnostic through message OOM, consume once on reporting. Read operand924-988. Ordinary returned query/sync errors and sticky faults still cause safe leak/raise; real sticky context behavior remains lead validation.
- D6 restores: success paths raise, operands retained through retainer destruction after partial work; fill revision increments before restoring so a failed restore cannot hide the write. Unwinding/quiet paths count and defer add-on errors; core teardown counts. Event destruction checked on its creating device. Read guard58-125, kernels265-302, fill453-474.
- D7 cuBLAS: setting failure destroys the just-created handle and checks destroy status. The unowned interval extends through map insertion, which is still unsafe on host OOM (V5-03). No new core/add-on source was changed here.

## R1–R6 reasoning

R1 SAFE for effects on **existing** representable core behavior, with the documented coordinated native rebuild prerequisite. Baseline ordinary Storage pooled/raw/CPU branches and existing kernel bodies/caches/round are preserved. numel/stride helpers retain representable values and replace overflow with refusal. New fill defects are R4; add-on shutdown defects are R3. No fresh full-core GPU regression suite was run; source reasoning and the independently checked historical52-id comparison support the bounded SAFE statement.

R2 SOUND for reviewed core DLPack ownership, stream ordering, device/error handling: export manager RAII and offsets, once-only capsule naming, exact dtype/version/flags/contiguity checks, rank/byte/address bounds, correct tensor-device fence, checked restore, checked import query before ownership transfer, and Storage-device drain/leak and coordinated finalization. Real two-GPU fence/destructor behavior remains explicitly unexecuted by this seat and INCONCLUSIVE in the supplied lead receipts. Source correctness of that guard is not confused with empirical two-GPU certification.

R3 UNSOUND: V5-01 can abort on an ordinary finite add-on callback at exit; V5-02 clears a release-query error without durable diagnostic accounting. The reviewed deferred ownership protects asynchronous memory use on success/partial launch and fails conservatively without completion proof. Operand offset/alignment/shape/alias/version/read-only/copy checks otherwise agree with the package contract. Queue removal before callbacks prevents double-draining but is insufficient for host shutdown quiescence.

R4 DEFECTS: V5-04 silently narrows the new fill grid for an accepted very large int64 extent. Add-on Conv/grouped BLAS transpose mapping, deterministic ConvTranspose gather, ONNX LSTM i/o/f/c gates and bias/direction/state ordering, shifted two-pass biased variance/affine maps, integer nearest/reflect indexing and rintf are correct by source review within their checked input/geometry rails. Add-on fp32 byte bounds leave enough headroom for loop/grid increments. Fresh checked-math18 and aggregate/accounting17 checks pass, without UBSan diagnostics. No universal GPU numerical/bitwise claim is made.

R5 PASS for separation and CPU build/link scope: fresh unchanged scratch build, core-binding syntax compilation, shared cudart/cuBLAS linkage and RUNPATH/usr/local/cuda-12.6/lib64; no undefined core symbol dependency. All21 checked copied production/build inputs match. Default allocator's lazy core import is approved by the design. Vendored DLPack headers' consumed layouts agree; their differences are comments/examples and an inert Torch stable guard. Final deployed core binary still requires lead clean-build identity.

R6 REJECT: one open MAJOR prevents David's condition. Fix V5-01, settle the other named defects, run relevant lead probes/rebuilt regression gates, and obtain a new independent merge verdict. No GPU run was made or requested by this seat.

## CPU checks and historical receipts

- Fresh add-on build and core syntax PASS: logs/p1t_rev5_build.log; build_identity.json, ldd.txt, objdump.txt, undefined_symbols.txt. Full core was not rebuilt by this seat.
- host_checks.json:20 individual child subprocesses, each rc0 with expected refusal/shape/OOM/contained-deleter outcome. Collector rc alone was not treated as PASS. operand_census.json:17 metadata/ownership cases,7 once-only consumed managers.
- Fresh actual FIX2 ck_census:18 cases/0 failures; actual FIX3 accounting:17/0; UBSan clean. Core lifecycle slice5.5s: rc0, still_in_flight0, diagnostic1; slice content matches current core code.
- V5-01 full-addon reproduction rc134, joined control rc0; deferred actual-source trace child rc-6. V5-02 trace rc1; V5-03 host-allocation fault rc1; V5-04 arithmetic witness rc1. These are RED defect receipts, not passing product gates. Test interposers compile; no GPU mode executed.
- Live Voice metadata census:139 manifest cases,61 NPZ header archives;129 applicable Conv/ConvTranspose/LSTM/norm/Round plans match recorded output geometries. Remaining10 Resize/Pad cases need data-valued attributes and remain numerical/stream lead coverage. No Voice numerical acceptance test was run.
- Lead receipt audit reads the original wt_fix4 files read-only: JSON has249 entries, all249 verdictPASS (all12 entries carrying an ok field are True) including T1R/T4S; combined log's52 FAILED/ERROR ids exactly equal the stated baseline52. FIX4 tests40 passed; export mutant8 failed/14 passed; two-device export log isINCONCLUSIVE. Before/after import and sync probes show the claimed failure/pass. These are verified historical receipt contents, not new GPU runs and not coverage of V5-01–V5-04. Details: lead_receipt_audit.json.

## Operand census

| Condition | Evidence / remaining lead check |
|---|---|
| Zero-size /0-d | Core checks preserve valid empty/scalar shapes; add-on refuses empty, permits scalar rint. Fresh CPU scalar/zero checks; core_edges for GPU roundtrip. |
| int64 limits incl zero neighbor | Extent and byte bound checked before empty treatment;18 UBSan cases pass. V5-04 finds separate nonempty fill grid boundary. |
| byte_offset | Core aligned element offset, total/range checks; add-on uses adjusted pointer/alignment; fresh host offset4 accept/offset2 refusal. core_edges/data probe pending. |
| Size1 unusual strides | Ignored only where never stepped; fresh INT64_MIN singleton stride accepted. |
| Read-only/versioned/copied | Core read-only refused before consume; add-on readers accept supported flags, writers refuse read-only/copied; host once-only checks pass. |
| Capsule twice / same input twice | Same bare capsule reused refused after first consumption; first manager once. Fresh exports may alias as read inputs; output/scratch alias refusal. |
| Nondefault producer streams | stream1 request and tensor-device export fence reviewed; addon_lifetime/numerics GPU forward ordering pending. |
| Producer deleter raises | Ordinary C++ exceptions caught/counted, pointer cleared first; CPU native throw check passes. Consumer-created finalization unwind is V5-01. |
| Host allocation failure on release | Owner/event reserve before launch, caught deque push and synchronous fallback, fixed-capacity drains. CPU OOM propagation passes; full fault sweep pending. V5-03 is creation/cache OOM. |
| Exit pending / releases active on other threads | Core5.5s coordinated lifecycle passes; add-on pending queue drain does not cover active immediate/popped callbacks: V5-01 rc134. |
| Prior CUDA errors at sync/op/fill (sticky/nonsticky) | Entry capture and pre-launch take reviewed; safe sticky drain leaks; historical receipts checked, real probes pending. V5-02 accounting loss named separately. Empty fill returns without launch/error-slot query. |
| Multi-device pending entries | Quiet guard per-device sync, cross-device event query legal, FIFO pop and failed-completion leaks reviewed; two-device probe exits3 on one GPU. |
| Current device differs at export/import/fill/all ops | Export/fill select tensor device and restore; import explicitly refuses mismatch before ownership transfer; all add-on launch paths use shared DeviceGuard. One-GPU interposer probe covers actual checks; two-device empirical cases pending. |
| Tensor label differs from memory | Core export/fill refuse label mismatch; foreign DLPack descriptor accuracy remains producer obligation. |
| empty cuda/cuda:N/cpu/zero | cuda labels checked current, explicitN must match, cpu uses existing CPU allocation, nbytes0 performs no allocation; shape/rank/byte refusals before allocation. GPU/no-device full-core probes pending. |
| Restore fails on success/unwinding | Success raise; fill revision counted first; add-on managers retained; unwind count/deferred error without throw; core destructor counted. Source and historical shim receipts; lead repeat probe pending. |

## Probe list

All paths are under this review. GPU probes are PENDING LEAD, sequential foreground timeout, with rebuilt reviewed binaries. Exit3 is INCONCLUSIVE. The three two-device probes explicitly exit3 on the lead's one-GPU host; device_error_sweep_single_gpu provides a one-GPU interposer form for export/restore/device/error call placement, not a physical two-device race proof. No suite skip is a PASS.

| Path under artifacts/p1t_rev5/probes/ | Decides / pass-fail rule | State |
|---|---|---|
| addon_exit_full_cpu.py | V5-01 actual full add-on immediate-refusal exit; PASS finite callback completes before finalization, exit0; current rc134; joined control rc0 | CPU RUN |
| addon_trace.cpp + addon_exit_cpu.py | V5-01 deferred-release mechanism with CUDA-call replacements; same rule; current child rc-6 | CPU RUN |
| addon_exit_gpu.py | V5-01 full add-on completed-event release at ordinary exit; one GPU; finite callback completes before finalization, exit0; timeout20 | PENDING LEAD |
| query_error_cpu.py | V5-02 one-shot failed event query must remain reportable at next sync; current rc1 | CPU RUN |
| query_error_single_gpu.py + query_once_shim.c | V5-02 genuine invalid-event error interposed once; one GPU, reported exactly once then safe release and clean next sync; timeout30 | PENDING LEAD |
| handle_insert_oom_cpu.cpp | V5-03 actual cache-creation slices: created handles destroyed when real host insertion throws; current rc1 | CPU RUN |
| fill_grid_cpu.cpp | V5-04 exact nblk arithmetic must cover accepted extent or refuse; current rc1, grid1 covers256 for2^40+1 | CPU RUN |
| fill_grid_single_gpu.py + fill_grid_shim.c | V5-04 full fill launch-geometry capture on synthetic metadata; one GPU; require size refusal before trapped launch, no huge allocation/data access; timeout30 | PENDING LEAD |
| device_error_sweep_single_gpu.py | Rebuilt core/add-on unchanged FIX4 interposer tests all pass/no skips; covers one-GPU export/fill/op guards, labels, errors and failed restores; timeout300 | PENDING LEAD |
| core_export_device.py | V4-01 real producer/consumer ordering; timeout60; requires2 GPUs, exits3 on one | PENDING LEAD |
| core_import_query_error.py | V4-03 rebuilt full-core device-query refusal preserves ownership; no visible GPU; timeout30 | PENDING LEAD |
| sync_existing_error.py | V4-04 real pre-existing launch error reported; one GPU; timeout30 | PENDING LEAD |
| addon_lifetime.py | F01 side-stream last-owner retention and correct outputs; timeout60 | PENDING LEAD |
| core_edges.py | Scalar/zero/offset/singleton/versioned/used capsule roundtrips; timeout60 | PENDING LEAD |
| core_device_drain.py | F05 correct Storage-device drain/deleter/restore; timeout60; requires2 GPUs, exits3 on one | PENDING LEAD |
| core_exit_overrun.py | F04/N5/V3-01 full-core in-flight exit quiescence; timeout25 per process | PENDING LEAD |
| numerics_and_streams.py | Finite ONNX parity/determinism, nondefault-stream producer ordering; timeout90 | PENDING LEAD |
| release_failures.py | Actual OOM/fallback/bound/partial-launch/deleter/exit tests all pass/no skips; timeout650 | PENDING LEAD |
| cuda_errors_multidevice.py | Per-device pending FIFO/errors/restore; timeout90; requires2 GPUs, exits3 on one | PENDING LEAD |
| exit_gate.py | Parent enforces child rc0 and DELETER finished before FINALIZING True; full/trace/GPU modes, timeout30; no marker/abort/hang fails | CPU RUN(full/control); GPU PENDING LEAD |
| build_cpu.sh | Fresh add-on build, core syntax and prior CPU harnesses; timeout1100; PASS | CPU RUN |

CPU build driver also copies the earlier host_checks/operand_census/real_metadata_plans/lifecycle drivers to /tmp/p1t_rev5_build with path-only changes; their receipts are above. Review helper build commands are in ledger. The two new single-GPU interposers compiled here to /tmp/p1t_rev5_query_once.so and /tmp/p1t_rev5_fill_grid.so; lead uses their respective LD_PRELOAD values. The fill-size probe explicitly expects the recommended refusal fix; a grid-stride alternative needs a separately registered amendment, not an adjusted pass rule after a result.

## Prior art

No product algorithm implemented. Findings/fix recommendations use DLPack(dmlc,2017+)/Array API(Consortium,2021+) ownership/stream handoff; CPython3.12(PSF,2023) atexit/finalization/GIL, C++17(ISO,2017) RAII/noexcept/integral conversion and GCC checked builtins(2015); NVIDIA CUDA/cuBLAS12.6(2024) event/device/error/launch/resource contracts; ONNX Conv/ConvTranspose/LSTM/norm/Resize/Pad/Round semantics. Probe comments and ledger state what was taken and which scenario is this review's contribution. Admission/quiescence and checked-narrowing/RAII recommendations are standard patterns, not new optimizations. [DLPack specification](https://dmlc.github.io/dlpack/latest/python_spec.html), [ONNX LSTM](https://onnx.ai/onnx/operators/onnx__LSTM.html). Historical origins of existing variance/reduction algorithms were not re-audited (unverified — lead to check: Chan Golub LeVeque1983 shifted variance; Harris2007 CUDA reduction; RCU McKenney Slingwine1998).

## Hygiene and procedural notes

No product source/test/build edits, git writes, subagents, GPU execution, deletions, service changes or unrelated process kills. Writes limited to artifacts/p1t_rev5, logs, docs/P1T_REV5_*.md and /tmp/p1t_rev5* (the trace module now lives under /tmp/p1t_rev5_trace; see corrected scratch naming deviation in ledger). Nothing moved to DELETE_ME. Scratch remains at /tmp/p1t_rev5_build and /tmp/p1t_rev5_*; repeat trace runs use PYTHONPATH=/tmp/p1t_rev5_trace.

Initial real-metadata driver failed because its input census had not yet been generated; first census attempt used a unavailable private NumPy API and failed. Fixed probe setup only using public header readers; original failures retained, final129-case run passes. No thresholds changed. Foreground build runs used timeout; yielded commands were resumed with yield_time_ms300000 and no intervening status messages.
