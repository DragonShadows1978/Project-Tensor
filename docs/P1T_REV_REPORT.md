# P1T-REV report

**REJECT promotion of 9df5bef..00590b7.** One BLOCKING and four MAJOR findings. Two MINOR, zero NIT. No source/test/build files were modified, no GPU work was executed, and nothing was deleted. Findings F01-F05 are not claimed fixed.

| Question | Verdict |
|---|---|
| R1 | RISK |
| R2 | UNSOUND |
| R3 | UNSOUND |
| R4 | DEFECTS |
| R5 | PASS |
| R6 | REJECT |

## Evidence and limits

Source reasoning covers the submitted 42-file diff, pinned by source_inventory.json. The working source matches 00590b7 (read-only git comparison). The mechanism is the approved core-DLPack/add-on split, not a stand-in engine. Registration was written before gates. Tests/census/oracles were inspected for coverage and parsed, not executed as GPU gates.

CPU receipts: offline build of an unchanged scratch copy of tc_signal passed with CUDA 12.6.85, gcc 13.3.0, Python 3.12.3 and local pybind11 v2.12.0. logs/p1t_rev_build.log records the command output; artifacts/p1t_rev/build_sha256.txt pins its module. Core bindings passed C++ syntax compilation only (logs/p1t_rev_core_syntax.txt); the reviewed core was not fully rebuilt or executed. The release-worker probe helper compiled. Host metadata probes ran in isolated processes with core dumps disabled. All Python files and probes passed AST syntax parsing. None of these checks establishes GPU correctness or the claimed acceptance suites.

CPU native reproduction: None upsample/pad/rint processes each returned -11. Reflect planning returned a negative output extent after overflow. Scalar rint planning returned (). Zero-size rint was clearly refused. Conv T_out=2147483648 and LSTM T*N=2147483648 were clearly refused before any CUDA call. Synthetic descriptors are validation-boundary evidence only; their huge shapes do not describe real resident GPU arrays.

Real-data census: read Voice's fixtures/manifest.json and all six real T4 weight archives, recording array shape/dtype in real_lstm_weight_census.json. This confirms the actual weight layouts; it is not a GPU parity run.

## R1 — core regression risk: RISK

Read core.h:54-110; kernels.cu:158-227,278-291; bindings.cpp:619-666,1405-1414; __init__.py:1-20,73-142,1307-1313, and the exact baseline diff. New external fields default to null; ordinary pooled/raw/CPU allocation and freeing retain their branches. Revision is still shared within a Storage and existing offsets are unchanged. fill_ bumps it only after a successful nonempty launch; external writers still need mark_modified. Imported aliases can have separate wrapper revisions, so callers must respect the documented external-write rule. No new kernel, existing round edit, eager CUDA initialization or new eager core import was found. The one deleted Python line is replaced by an expanded __all__ line, preserving the old exports.

The native Storage size is ABI-incompatible with old clients (F07). Existing core consumers of newly created/imported tensors inherit unchecked rank/count paths (F03) and unsafe imported teardown (F04/F05). SAFE would overstate production promotion readiness. Ordinary fully rebuilt pre-existing paths appear unchanged by source review; the 52-id no-new-failure claim remains historical suite evidence, not this seat's live validation.

## R2 — DLPack: UNSOUND

Read bindings.cpp:221-615; kernels.cu:181-207; __init__.py:73-126; vendored header:45-65,214-349; test_dlpack.py lifecycle/offset/refusal cases. Export manager keeps shared Storage ownership; static capsule names, unconsumed destructors, used_* renaming, legacy/versioned major checks, writable-only import and exact dtype mapping are structurally correct. Minor version differences are ABI-compatible. Offsets survive import and re-export; singleton strides are ignored only where never stepped. Zero-size and 0-d imports are intended to work; ordinary metadata handling does not establish device execution.

None/1 and -1 bypass the fence; 0 is refused; 2 maps to cudaStreamPerThread; handles record an event on the producer legacy stream and wait on the consumer stream. Event destruction after queued wait is valid. A bare import capsule correctly carries a caller-ordering obligation. GIL acquisition covers ordinary non-Python threads, but not interpreter finalization (F04). The drain uses the wrong device after a current-device switch (F05). Count/offset arithmetic is unchecked (F03). Export metadata-allocation exceptions leak (F06). Before rename, validation refusals leave capsule ownership intact; rename failure switches the Storage release to a no-op; successful rename followed by Tensor allocation failure unwinds the owning Storage, avoiding a double-free in those paths. No demonstrated double-delete was found in those paths.

## R3 — tc_signal operand contract: UNSOUND

Read dlpack_operand.cpp:61-155,174-213,234-285 and .h:35-89; bindings.cpp:58-291; lstm_bindings.cpp:63-108,164-218; m4_bindings.cpp:42-315; Python allocator/wrappers in full. Inputs/outputs require CUDA fp32, common ordinal, compact strides apart from singleton axes, aligned adjusted data pointer and exact shapes. Written operands reject read-only and copied versioned tensors. Readers may accept read-only or copied inputs. Overlapping output/scratch/input ranges are refused on ordinary representable sizes. Empty dimensions are refused by design; scalar rint is accepted, while norms require rank >=1. Read-only legacy metadata has no flag to inspect.

ManagedOwner as a constructed member prevents leaks when Operand construction throws after capsule consumption. Renaming and once-only release are structurally correct, but launch completion is not lifetime completion (F01). The README's nonlegacy caveat requires a caller obligation that cannot retain hidden copied-export temporaries. Missing mandatory inputs crash (F02), and unchecked arithmetic can invalidate overlap ranges (F03).

Outputs and scratch use the default TensorCUDA allocator or an explicitly supplied callback; native kernels allocate no operand buffers. “C++ allocates nothing” cannot be literal: host vectors/maps, cuBLAS handles and internal workspaces allocate. This exception is documented. Conv and LSTM each lazily create their own per-device cuBLAS handle (conv.cu:96-110; lstm.cu:118-135). Both explicitly set legacy stream, default math, atomics not allowed and host pointer mode. Independent handles on the same ordered stream are safe in the conventional GIL-serialized entry path; source gives no evidence that two handles alone cause a race. This is not free-threaded/subinterpreter certification.

## R4 — kernels and numerics: DEFECTS

Read conv.cu:119-176,199-253; lstm.cu:138-265; norms.cu:97-176; small_ops.cu:74-135 and their geometry headers/bindings. Ordinary representable indexing matches the requested equations:

- Conv im2col row (c*K+k) and column t fetch t*stride-pb+k*dilation. Group blocks are contiguous. Column-major SGEMM computes Y^T=Col^T W^T with m=T_out, n=Og, k=CgK and checked int32 leading dimensions. Batch scratch reuse is ordered on the same stream. The implementation uses grouped GEMM rather than a separate depthwise kernel; the approved notes allow that correctness path.
- ConvTranspose maps each output to its group, walks k and c in fixed order, checks nonnegative divisible source coordinates, bounds q<T, and uses ONNX [C_in, O/group, K] layout. No atomic scatter.
- LSTM is i,o,f,c; bias halves and reverse write index match ONNX. Input projections precede recurrent GEMMs, which precede cell updates on one stream. Each cell thread owns one h/c element, and the GEMM finishes reading prior h before the next cell writes it. Ordinary m/n/k/lda casts are checked; T*N beyond INT_MAX refuses. T within that rail can still enqueue billions of steps, a resource limitation rather than a proved memory fault. No extra sequence-length behavior is silently implemented.
- Norm affine maps are correct for per-row/per-element/per-channel layouts. Shifted mean and centered biased variance avoid the common large-offset cancellation case. Shared-memory reduction has barriers before reuse; grid-stride row reuse does not introduce a same-row race with disjoint x/y. L=1 and positive eps give zero normalized value; eps=0 gives IEEE NaN on zero variance. Epsilon is cast to fp32. Arbitrary extreme finite magnitudes can overflow fp32 statistics; broad all-finite numerical guarantees are not made. NaN/Inf flow through arithmetic; no masking was found.
- Integer upsample uses floor by exact nonnegative division; reflect pads <T need one reflection, and rint uses rintf. Word gathers preserve NaN payloads and signed zero. rint's NaN equality does not require payload preservation.

No executable atomicAdd, intrinsic exp/tanh or fast-math compile flag was found. CUDA transcendental math is expf/tanhf; fixed reduction order and cuBLAS configuration support determinism within one toolkit/device configuration. GPU numerical/bitwise claims remain pending. The geometry overflow and unsupported ranks in F03 prevent a universal CORRECT verdict even though ordinary-shape formulas appear correct. F01 can also invalidate otherwise-correct kernels' operands.

## R5 — decoupling/build: PASS (CPU build/link scope)

Read tc_signal/CMakeLists.txt:1-80, build.sh:1-30, README allocator/build notes, all includes and both dlpack.h copies. Scratch offline build succeeds with PYBIND11_SOURCE_DIR. No core headers, core symbols or _tensor_cuda dependency occur in the add-on link. Python's default allocator lazily importing core is the explicit approved allocation mechanism; supplying an allocator removes that runtime need.

ldd and objdump receipts show NEEDED libcudart.so.12/libcublas.so.12 and RUNPATH /usr/local/cuda-12.6/lib64; no static cudart. The production baseline core resolves the same shared runtime file (logs/p1t_rev_baseline_core_ldd.txt); the submitted core .so is absent in this review checkout, so its final deployment binary still needs a lead receipt. Source flags/link lines are preserved under artifacts/p1t_rev/. This PASS does not certify an uninspected final deployment binary.

Core's Apache-2.0 DLPack v1.3 header and add-on's torch variant have matching relevant layouts: DLTensor=48, DLManagedTensor=64, versioned=80, versioned DLTensor offset=32 on this ABI. Diff shows comments/example edits and an outer TORCH_STABLE_ONLY/TORCH_TARGET_VERSION guard, not a managed-tensor ABI divergence. That guard can intentionally reject a build defining those torch stable macros, but neither package defines them. It does not introduce a PyTorch link/runtime dependency. Historical upstream byte identity was spot-checked against local diffs/license, not independently re-downloaded or certified as provenance.

## R6 — overall: REJECT

F01-F05 prevent production promotion. Correct the reported issues, run the lead probes against rebuilt reviewed binaries, then rerun the unchanged core baseline comparison and Voice runner. Finite passing fixtures cannot override the demonstrated native crashes or ownership defects.

## Findings table

| ID | Severity | File:line | Concrete scenario | Suggested fix |
|---|---|---|---|---|
| F01 | BLOCKING | tc_signal/src/dlpack_operand.cpp:174-185; tc_signal/src/bindings.cpp:255-257; tc_signal/src/lstm_bindings.cpp:215-218; tc_signal/src/m4_bindings.cpp:229-231 | A bare capsule is the last owner of a tensor allocated on a nonlegacy stream. The add-on queues a legacy-stream read and immediately calls its deleter; the producer allocator can reclaim and overwrite that block on its own stream before the read runs. Keeping an already-consumed capsule alive does not keep its deleted manager alive. Inputs may also be producer-created temporary copies (copy is unspecified for readers), whose storage the caller cannot retain. The documented caveat therefore does not make the advertised arbitrary-producer/capsule contract memory-safe. | Retain all consumed managers through GPU completion, including partial-launch exception paths. The simplest correct repair is a checked synchronization of the operand device legacy stream before scope-exit deleters; an asynchronous repair needs event-backed deferred ownership and safe host-side release, rather than a CUDA host callback that calls CUDA. |
| F02 | MAJOR | tc_signal/src/m4_bindings.cpp:59-61,90-101,235-241,253-254,270-288,307-309; tc_signal/src/dlpack_operand.h:66 | upsample_nearest1d(None, 2) and reflection_pad1d(None, (0, 0)) treat x as an absent optional operand, then dereference an empty shape via back(). rint(None, out=<valid scalar capsule>) reaches shape4 on an absent Operand and dereferences its null tensor metadata. All three isolated CPU processes exited -11 (SIGSEGV), without CUDA execution. | Distinguish required x from optional scale/bias in peek4 and consume4; reject None before planning or shape4. Make absent Operand metadata access fail safely as a second line of defense. |
| F03 | MAJOR | tensor_cuda/src/bindings.cpp:527-540,555-568,573-579; tensor_cuda/src/kernels.cu:69-79,219-227; tc_signal/src/m4_bindings.cpp:111-114,253-267; tc_signal/src/dlpack_operand.cpp:149,234-237 | The new empty accepts rank 9 although existing core op metadata arrays hold only 8 axes. empty((1<<62,), float32) has a representable element count but wraps numel*4 to zero on the reviewed 64-bit ABI, yielding null/zero storage for a nonempty tensor; empty((1<<32,1<<32)) also overflows signed element counting. New import uses the same unchecked products and unchecked offset-plus-bytes arithmetic. CPU planning for reflect shape (1,1,1<<62) and pads ((1<<62)-1,(1<<62)-1) returned (1,1,-4611686018427387906). Synthetic large descriptors test validation only, not real allocation validity. Normal >2^31 cuBLAS m/n/k/leading-dimension cases are guarded; these guards do not check all byte totals or signed products. | Before wrapping or allocating, enforce the core rank limit and checked nonnegative element multiplication, element-size multiplication, offset/address addition and all output/scratch geometry. Reject unsupported shapes clearly before narrowing casts or launches; do not treat a wrapped result as empty. |
| F04 | MAJOR | tensor_cuda/src/bindings.cpp:435-447 | The last exported reference to an imported Storage can be released by a non-Python consumer thread while CPython is finalizing. Py_IsInitialized may still be true; PyGILState_Ensure is then unsafe. CPython 3.12 documents termination of that thread during finalization, interrupting the deleter; newer runtimes may hang it. The existing guard handles an already-finalized interpreter, not finalization in progress. Normal-thread GIL acquisition and main-thread ordinary cleanup are separate cases and are not asserted broken. | Define and implement a lifecycle-safe release policy: finish ordinary releases before interpreter teardown, avoid GIL acquisition once finalization starts, and deliberately retain/leak late Python-owned managers when safe release is impossible. Use version-appropriate finalization detection plus lifecycle coordination rather than Py_IsInitialized alone. |
| F05 | MAJOR | tensor_cuda/src/kernels.cu:205-206 | Import on device 0, enqueue a long legacy-stream operation touching the imported memory, switch the current CUDA device to 1, then drop the last imported reference. cudaStreamSynchronize(0) drains device 1, after which the producer deleter may reclaim device-0 memory with its last use still pending. The import-time same-current-device check does not pin the device for the rest of the object lifetime. This is conditional on a multi-device process; it is not a claim of reproduction on the single-GPU target. | Guard the teardown with the Storage device ordinal, drain that device legacy stream and handle errors without unsafe release, then restore the caller device. Alternatively track and synchronize an allocation-device last-use event. |
| F06 | MINOR | tensor_cuda/src/bindings.cpp:385-389 | After raw new DLPackExportCtx, shape/stride vector allocation can throw bad_alloc. The context and its shared Storage reference leak because the cleanup only handles PyCapsule_New failure. | Own the context with a unique_ptr until all metadata and the capsule are constructed successfully, then transfer it to the capsule. |
| F07 | MINOR | tensor_cuda/include/tc/core.h:63-78 | CPU layout measurement gives sizeof(Storage)=40 at 9df5bef and 56 now; revision remains at offset 32. An old native client allocating the old Storage layout and invoking the new destructor would expose out-of-object external_release reads. No such client was found in tc_signal, which does not use core headers; a fully rebuilt core does not have this mixed-layout problem. | Require a clean/coordinated rebuild of all native core clients at promotion, or version/hide the native Storage ABI. Document this as a rollout prerequisite. |

## Full findings and evidence

### F01 — BLOCKING: Operand ownership ends before asynchronous last use

Location: `tc_signal/src/dlpack_operand.cpp:174-185; tc_signal/src/bindings.cpp:255-257; tc_signal/src/lstm_bindings.cpp:215-218; tc_signal/src/m4_bindings.cpp:229-231`.

A bare capsule is the last owner of a tensor allocated on a nonlegacy stream. The add-on queues a legacy-stream read and immediately calls its deleter; the producer allocator can reclaim and overwrite that block on its own stream before the read runs. Keeping an already-consumed capsule alive does not keep its deleted manager alive. Inputs may also be producer-created temporary copies (copy is unspecified for readers), whose storage the caller cannot retain. The documented caveat therefore does not make the advertised arbitrary-producer/capsule contract memory-safe.

Fix: Retain all consumed managers through GPU completion, including partial-launch exception paths. The simplest correct repair is a checked synchronization of the operand device legacy stream before scope-exit deleters; an asynchronous repair needs event-backed deferred ownership and safe host-side release, rather than a CUDA host callback that calls CUDA.

Evidence: Source reasoning; no GPU race executed. README.md:119-126 and the later-read M2 report explicitly disclose immediate release. Probe addon_lifetime.py is pending.

Prior art: DLPack consumer ownership and CUDA last-use/free ordering (dmlc 2017+; NVIDIA CUDA 12.6).

### F02 — MAJOR: None mandatory inputs crash the interpreter

Location: `tc_signal/src/m4_bindings.cpp:59-61,90-101,235-241,253-254,270-288,307-309; tc_signal/src/dlpack_operand.h:66`.

upsample_nearest1d(None, 2) and reflection_pad1d(None, (0, 0)) treat x as an absent optional operand, then dereference an empty shape via back(). rint(None, out=<valid scalar capsule>) reaches shape4 on an absent Operand and dereferences its null tensor metadata. All three isolated CPU processes exited -11 (SIGSEGV), without CUDA execution.

Fix: Distinguish required x from optional scale/bias in peek4 and consume4; reject None before planning or shape4. Make absent Operand metadata access fail safely as a second line of defense.

Evidence: CPU native-extension reproduction: artifacts/p1t_rev/host_checks.json (none_up, none_pad, none_rint).

Prior art: DLPack producer contract and C++ vector/non-null object access preconditions.

### F03 — MAJOR: Unchecked rank, element, byte and geometry arithmetic

Location: `tensor_cuda/src/bindings.cpp:527-540,555-568,573-579; tensor_cuda/src/kernels.cu:69-79,219-227; tc_signal/src/m4_bindings.cpp:111-114,253-267; tc_signal/src/dlpack_operand.cpp:149,234-237`.

The new empty accepts rank 9 although existing core op metadata arrays hold only 8 axes. empty((1<<62,), float32) has a representable element count but wraps numel*4 to zero on the reviewed 64-bit ABI, yielding null/zero storage for a nonempty tensor; empty((1<<32,1<<32)) also overflows signed element counting. New import uses the same unchecked products and unchecked offset-plus-bytes arithmetic. CPU planning for reflect shape (1,1,1<<62) and pads ((1<<62)-1,(1<<62)-1) returned (1,1,-4611686018427387906). Synthetic large descriptors test validation only, not real allocation validity. Normal >2^31 cuBLAS m/n/k/leading-dimension cases are guarded; these guards do not check all byte totals or signed products.

Fix: Before wrapping or allocating, enforce the core rank limit and checked nonnegative element multiplication, element-size multiplication, offset/address addition and all output/scratch geometry. Reject unsupported shapes clearly before narrowing casts or launches; do not treat a wrapped result as empty.

Evidence: CPU reflect metadata reproduction in host_checks.json; core overflow/rank consequences are source reasoning using the existing helper, newly exposed through empty/import. core_edges.py is pending for the reviewed core runtime.

Prior art: DLPack int64 shape/uint64 offset ABI; C++ signed-overflow and size_t arithmetic; cuBLAS 12.6 integer dimension API.

### F04 — MAJOR: Initialization check is insufficient during interpreter finalization

Location: `tensor_cuda/src/bindings.cpp:435-447`.

The last exported reference to an imported Storage can be released by a non-Python consumer thread while CPython is finalizing. Py_IsInitialized may still be true; PyGILState_Ensure is then unsafe. CPython 3.12 documents termination of that thread during finalization, interrupting the deleter; newer runtimes may hang it. The existing guard handles an already-finalized interpreter, not finalization in progress. Normal-thread GIL acquisition and main-thread ordinary cleanup are separate cases and are not asserted broken.

Fix: Define and implement a lifecycle-safe release policy: finish ordinary releases before interpreter teardown, avoid GIL acquisition once finalization starts, and deliberately retain/leak late Python-owned managers when safe release is impossible. Use version-appropriate finalization detection plus lifecycle coordination rather than Py_IsInitialized alone.

Evidence: Source reasoning plus official CPython API documentation; actual core shutdown/foreign-thread behavior is pending core_shutdown.py and release_worker.cpp.

Prior art: CPython 3.12 Initialization, Finalization, and Threads; DLPack non-Python consumer deleters.

### F05 — MAJOR: Imported Storage drains the current device instead of its recorded device

Location: `tensor_cuda/src/kernels.cu:205-206`.

Import on device 0, enqueue a long legacy-stream operation touching the imported memory, switch the current CUDA device to 1, then drop the last imported reference. cudaStreamSynchronize(0) drains device 1, after which the producer deleter may reclaim device-0 memory with its last use still pending. The import-time same-current-device check does not pin the device for the rest of the object lifetime. This is conditional on a multi-device process; it is not a claim of reproduction on the single-GPU target.

Fix: Guard the teardown with the Storage device ordinal, drain that device legacy stream and handle errors without unsafe release, then restore the caller device. Alternatively track and synchronize an allocation-device last-use event.

Evidence: Source reasoning from CUDA current-device stream semantics; core_device_drain.py requires two GPUs and is pending.

Prior art: NVIDIA CUDA 12.6 current-device and default-stream semantics.

### F06 — MINOR: Export context is not exception-owned before capsule construction

Location: `tensor_cuda/src/bindings.cpp:385-389`.

After raw new DLPackExportCtx, shape/stride vector allocation can throw bad_alloc. The context and its shared Storage reference leak because the cleanup only handles PyCapsule_New failure.

Fix: Own the context with a unique_ptr until all metadata and the capsule are constructed successfully, then transfer it to the capsule.

Evidence: C++ exception-path reasoning; import rename/no-op ownership handling and post-consumption Tensor construction have proper local-owner unwinding.

Prior art: C++ RAII/exception ownership (Stroustrup, 1980s onward).

### F07 — MINOR: Native Storage ABI size changes require a coordinated rebuild

Location: `tensor_cuda/include/tc/core.h:63-78`.

CPU layout measurement gives sizeof(Storage)=40 at 9df5bef and 56 now; revision remains at offset 32. An old native client allocating the old Storage layout and invoking the new destructor would expose out-of-object external_release reads. No such client was found in tc_signal, which does not use core headers; a fully rebuilt core does not have this mixed-layout problem.

Fix: Require a clean/coordinated rebuild of all native core clients at promotion, or version/hide the native Storage ABI. Document this as a rollout prerequisite.

Evidence: artifacts/p1t_rev/layout_baseline.txt and layout_current.txt; core include/reference scan.

Prior art: C++ native object-layout ABI compatibility.

## Probe list

All are pending lead execution. See artifacts/p1t_rev/probes/README.md for timeouts, compilation and interpretation.

| Path under artifacts/p1t_rev/probes/ | Decides |
|---|---|
| addon_lifetime.py | F01 side-stream capsule-only reclamation/reuse; a clean finite run is inconclusive |
| core_edges.py | F03 shape refusal and scalar/empty/offset/singleton/used-capsule/stream protocol edges |
| core_device_drain.py | F05 device-0 last use versus current-device-1 deleter; needs two GPUs |
| core_shutdown.py + release_worker.cpp | F04 real imported Storage release on a foreign thread normally and during finalization |
| numerics_and_streams.py | Residual finite LSTM fp64 parity, repeatability, both cuBLAS handles, L=1/eps=0, special rint and nonlegacy producer ordering |

## Author-receipt cross-check (after independent code assessment)

M1 says the late/already-finalized interpreter guard leaks instead of crashing, but does not test finalization in progress. Its published no-drain mutant survived Voice G1, while a corrected own test caught it; this narrows what G1 proves. M2 explicitly discloses unenforced nonlegacy lifetime requirements and multi-GPU untested status. M3 confirms the second handle is a scope deviation and cuBLAS workspace allocation exception; neither fact alone establishes a kernel race. M3/M4 also report an own-suite allocator-test failure rather than a completely clean suite. These observations do not change independently derived F01-F05.

logs/p1t_m1_g2_compare_r2.txt contains “baseline ids 52 now ids 52” and “new failing ids: 0 []”, referencing an older APA baseline and an M1 run; the underlying complete id sets were not rerun here. The claimed final 236/236 Voice report was not present at the supplied artifacts/p1t_int location; it remains a lead claim, not a live verified fact. No independent full core or Voice GPU gate is claimed by this review.

## Prior art

Taken: standard ownership, stream, API/ABI and operator invariants; no new product algorithm is implemented or proposed. The repair recommendations apply these existing invariants. Probe comments and ledger carry attribution. References:

- DLPack (dmlc contributors, 2017+), [Python ownership/GIL/capsule specification](https://dmlc.github.io/dlpack/latest/python_spec.html); Python Array API Consortium data interchange (2021+).
- [CPython 3.12 finalization and GIL API](https://docs.python.org/3.12/c-api/init.html#c.PyGILState_Ensure), explaining F04.
- NVIDIA [CUDA 12.6 stream API](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-runtime-api/group__CUDART__STREAM.html), [CUDA programming guide](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html) and [cuBLAS 12.6 reproducibility/API](https://docs.nvidia.com/cuda/archive/12.6.0/cublas/index.html).
- ONNX [LSTM](https://onnx.ai/onnx/operators/onnx__LSTM.html), [GroupNormalization](https://onnx.ai/onnx/operators/onnx__GroupNormalization.html), Conv, ConvTranspose, InstanceNormalization, LayerNormalization, Resize, Pad and Round semantics from the supplied contract/source.
- Shifted two-pass variance: Chan, Golub and LeVeque (1983); fixed CUDA reduction: Harris (2007); C++ RAII: Stroustrup (1980s onward). Exact historical attribution for these last sources is unverified — lead to check: “Chan Golub LeVeque 1983 shifted variance”, “Harris 2007 CUDA reduction”, “Stroustrup RAII history”. Our work is the bounded adversarial review/probe scenarios, not an invention of those methods.

## Left artifacts / hygiene

Scratch remains at /tmp/p1t_rev_addon, /tmp/p1t_rev_baseline, /tmp/p1t_rev_layout_current, /tmp/p1t_rev_layout_baseline, /tmp/p1t_rev_layout_addon and /tmp/p1t_rev_release_worker.so. No files were moved to DELETE_ME because none required removal. Changes in the worktree are confined to artifacts/p1t_rev/, logs/p1t_rev_* and docs/P1T_REV_*.md.
