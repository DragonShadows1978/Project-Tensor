G1 PASS. G4 PASS. G5 FAIL. RED / acceptance not met.

The literal G5 audit found the dispatcher live transcript, logs/eng3f1_r1.log, growing outside the declared writable set. Its SHIM-RUN header and birth time precede the initial inventory. No agent command targeted it for writing. No exemption was inferred. Product edits pass the named-site audit, no files are missing, and no git command, deletion, subagent or GPU work was performed. Gates stopped at this finding; subsequent work only packaged receipts and the report.

The five findings are addressed in the scoped diff. G2/G3 remain staged, not claimed passed. The MAJOR finding requires the narrow follow-up review specified by the order.

G1: final Build F, CUDA_VISIBLE_DEVICES empty; 81 passed, 46 deselected, zero errors/failures/skips. Evidence: cpu/G1_final/RESULT.json, junit.xml and run.log. Byte-overflow, public/private routes, rank/negative extents, strict strings, and valid CPU/zero-element CUDA paths passed. The RLIMIT_AS child raised MemoryError for valid [2**25], and for the largest valid byte multiple per dtype, then exited 0. Baseline overflow children exited 0; no baseline parent died.

The inherited interposer counted host_calls=0, control_calls=4. Its scope is calls through the module global libcudart imports after import. It cannot see direct-handle, driver-API or other-library calls. An independent process passed the actual host paths while CUDA runtime initialization failed with error 100. cudaGetLastError returned 100 before and after: it is unqueryable as a pending-error check here, not clean. The test requires a clean post-query where the pre-query is clean and rejects host exceptions/new errors. No observed clean last-error query is claimed on this no-device runtime.

R5: 145 executed informative full-width unsigned guards and 45 independently built NumPy constant-reference checks, in cpu/G1_final/l19.jsonl and constant_reference.jsonl. Identity compares dtype/shape, then guards both operands. Bool constant/scalar cases and backward seeds join real outputs into mixed corpora. Full explicitly verifies its independent NumPy constant reference bits. L55 retains offset 0 inside mixed corpora; its actual device comparisons await G2.

Diff summary (source.diff): NDArray allocating constructor checks the int64 byte bound before multiplying, including nonzero extents beside zero; Storage allocating constructor throws bad_alloc on nonzero host malloc failure. All four private hooks enforce TC_MAX_DIMS=8 on creation shapes or clone/to source shapes. Device-taking hooks strictly refuse malformed/noncurrent strings before allocation. Device::from_string is unchanged: it drops an explicit CUDA index and maps non-cpu strings to CUDA index 0. No numerical/device-kernel statement changed; ENG3 NaN changes are comments only. The inherited test file adds R4 and R5 checks and copied-package selection for isolated runs.

NaN claim boundary: inherited measurements used sm_89 builds and the probe census of all 2,046 half NaNs and 1,099 distinct float NaN words. Host/device NaN bits agree on that census and architecture. No claim is made for other architectures. Default CMake also builds sm_90. This round made no GPU NaN measurement.

| Dtype | Overflow shape | Largest legal count | Largest legal byte multiple |
|---|---|---:|---:|
| float32 | [2305843009213693952] | 2305843009213693951 | 9223372036854775804 |
| float16 / bfloat16 | [4611686018427387904] | 4611686018427387903 | 9223372036854775806 |
| int64 | [1152921504606846976] | 1152921504606846975 | 9223372036854775800 |
| bool / uint8 | [4611686018427387904, 4] | 9223372036854775807 | 9223372036854775807 |

Each [0, largest_legal_count] constructs without overflow. The bound is INT64_MAX; wider dtypes use its largest multiple of the element width. BYTE_BOUNDARIES.json records these cases. BOUNDARY.json pairs 79 E/F cases, using bounded children or review receipts 011/028 for known E crashes. The saved lead GPU receipt supplies E impossible-index behavior. Existing E refusals remain visible rather than being claimed new.

G4: MUTANTS.json was written before any mutant run. Final MUTANT_RESULTS.json records 3/3 kills; each named test executed its mutated line and failed as expected. All builds succeeded.

| Mutant | Named test | Observed failure | Marker / pytest exit |
|---|---|---|---|
| f1m01_bytes_unchecked | test_f1_cpu_overflow | child got MemoryError instead of OverflowError; child exit 1 | executed / 1 |
| f1m02_malloc_unchecked | test_f1_cpu_memoryerror | malloc-null child SIGSEGV, exit -11 | executed at null return / 1 |
| f1m03_device_lenient | test_f1_cpu_device_strings | malformed empty device reached cudaMalloc; RuntimeError instead of ValueError | executed / 1 |

Builds used the engine CMake unchanged: Release, sm_89, CUDA 12.6 as E, nice 19, at most -j4, foreground timeout, /usr/bin/time -v and continuous memory/disk monitoring. Neither resource stop rail was reached. Initial mutant builds copied F objects/build trees but dependency regeneration recompiled more than the changed TU; final hook updates rebuilt incrementally. Every source hash map and build command is in build/BUILD_*.json. Final mutant maps match final F outside their registered mutated translation unit; exact diffs are saved in build/.

Final F module: tensor_cuda/build-eng3f1/source_F/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so. The product package module was not overwritten. Superseded r1 receipts are historical; incremental rebuilds replaced their module bytes. E remains read-only, with a pinned package copy for staged comparison.

Every module SHA-256 follows.

| Build | Module SHA-256 |
|---|---|
| E | `9f070b6f4427f2b704f09ec96ef2097791a61e0c02b65e0f384169eac1d4986d` |
| F | `642d5ab7b57d4c6796cd59f701ad4dcdddde24c0a01f478d3c6630b513f6040e` |
| F_r1 | `9899f66aa14a532700f02b1c203c50a21525d800dd6c76bac53a75095e69e190` |
| f1m01_bytes_unchecked | `70fcd64bf2fc991871e633b8a18c049a773d203d147d5eeb17919e4a09777aad` |
| f1m01_bytes_unchecked_r1 | `68ba7a4012fc5ccb7b4d902fc662e152f63d6c6b24364bf1643b2c026738a270` |
| f1m02_malloc_unchecked | `264493c91b61bf03c8027082a62dc1cb1d86fa8c5381ee7605b391f6a257879b` |
| f1m02_malloc_unchecked_r1 | `984d7ba1155924cdab78c0928b3bff6f0f4a7fe274db91234d761bb37b1026ab` |
| f1m03_device_lenient | `8ba79c894b2d3f671a752d9c4f7d831590f051236947c4b3634ace05487b7aaa` |
| f1m03_device_lenient_r1 | `5e9ce59e9245dbe2266e95938c4dd1e7d4ae4de211d1db7a9221756d0eb7275f` |

Build receipt observations: peak compiler RSS 781064 kB; minimum MemAvailable 21544704 kB; minimum disk free 51944103936 bytes. F refusal classes across the 79 boundary cases: {'OverflowError': 28, 'ValueError': 49, 'MemoryError': 2}.

Staged G2/G3: run from the worktree root using each launch script below. Each launcher verifies the module SHA against its build receipt before leasing, runs its CPU prefix with CUDA hidden, captures the line log, and records forge exit, inner exit and log SHA. The exact forge command is shown for each launcher. All CPU prefixes passed; all dry calls returned 78 before leasing. Expected real forge exit=0 and inner exit=0 for every line. No real G2/G3 result exists yet.

G2 runs the inherited cast/L55/G1b rows and R2 impossible/noncurrent-index refusal plus valid-current-index checks. G3-cpu covers the inherited suite with CUDA hidden. G3-g1/g2 use the unchanged inherited file split and same options in separate isolated E/F processes. G3 compares per-ID outcomes and collection-error file IDs: no intended differences are registered. G3-device compares all inherited E2 members, including every full value, with actual-width unsigned views after dtype/shape checks, real mixed corpora and per-output attribution. No binary identity gate is registered.

The prefix receipt covers the common worker CPU path through import, host construction, clone/to and unsigned-bit checks, immediately before its first device allocation. For G3-cpu it stops before suite execution. It is not a completed G3 suite or a GPU result.

Run `bash artifacts/eng3f1/staged/launch-g2.sh` (expected 0). Its forge line:

```bash
forge-gpu-run --min 30 --max 600 --label eng3f1-g2 -- timeout 570s nice -n 19 python3 -B -P artifacts/eng3f1/tools/eng3f1_stage.py run g2
```

Run `bash artifacts/eng3f1/staged/launch-g3-cpu.sh` (expected 0). Its forge line:

```bash
forge-gpu-run --min 30 --max 1800 --label eng3f1-g3-cpu -- timeout 1750s nice -n 19 python3 -B -P artifacts/eng3f1/tools/eng3f1_stage.py run g3-cpu
```

Run `bash artifacts/eng3f1/staged/launch-g3-g1.sh` (expected 0). Its forge line:

```bash
forge-gpu-run --min 30 --max 1800 --label eng3f1-g3-g1 -- timeout 1750s nice -n 19 python3 -B -P artifacts/eng3f1/tools/eng3f1_stage.py run g3-g1
```

Run `bash artifacts/eng3f1/staged/launch-g3-g2.sh` (expected 0). Its forge line:

```bash
forge-gpu-run --min 30 --max 1800 --label eng3f1-g3-g2 -- timeout 1750s nice -n 19 python3 -B -P artifacts/eng3f1/tools/eng3f1_stage.py run g3-g2
```

Run `bash artifacts/eng3f1/staged/launch-g3-device.sh` (expected 0). Its forge line:

```bash
forge-gpu-run --min 30 --max 1800 --label eng3f1-g3-device -- timeout 1750s nice -n 19 python3 -B -P artifacts/eng3f1/tools/eng3f1_stage.py run g3-device
```

Deviations:

- D01: initial TMPDIR=/tmp and a broad sibling filename inventory; no unauthorized sibling contents read. Subsequent reads were scoped.
- D02: forge-gpu-run rejected --version; dry invocation returned 78. Default nvcc was 12.0; E cache selected CUDA 12.6, then explicitly used and versioned.
- D03: one early tool-layer wait used 1000 ms; later waits used 300000 ms. Builds remained foreground with no intervening status messages.
- D04: initial pytest selection also matched parent name tensor_cuda and selected only four new tests. Its receipt is not full G1. Corrected selection reran all 81 CPU rows under a fresh label.
- D05: boundary enumerator TypeError from reusing a tuple shape; fixed its operand variable. Script exception repair 1.
- D06: report generator KeyError arm on inherited E receipt; changed receipt display labels to filenames. Script exception repair 2. No gate changed.
- D07: a shell quoting slip truncated the initial report prefix and attempted prose as commands (command-not-found/syntax error). Rewrote the report with a quoted Python heredoc; no engine or gate changes.
- Literal G5 failure: externally growing dispatcher transcript outside writable roots. No exemption inferred; no further gates ran after detection.

## Prior art

ISO C++17 [expr]/4 (2017), division-before-multiplication, CWE-190/CWE-252 (MITRE, year unverified): arithmetic preconditions and allocation-return checking; ours: constructor placement and scoped operands. Existing tensor_empty/P1T-FIX4 (2026), TC_MAX_DIMS=8: rank/current-device rule; ours: strict private parsing. DeMillo, Lipton and Sayward (1978): mutation testing; ours: these three registered defects and assertions. NumPy unsigned views and the review mixed-corpus protocol (2026): informative same-width comparison; ours: corrected sites and explicit independent constants. SHA-256 (NIST, 2001), CMake (Kitware, 2000), pytest report hooks (2004 onward): pins/builds/reporting. External citations/years unverified -- lead to check: ISO C++17 expr overflow, CWE-190, CWE-252, mutation testing 1978. Existing finite conversion algorithms are unchanged. No prior art known to me for the exact report/inventory assembly. The code sites and ledger carry these annotations.

Every file written is enumerated, one exact worktree-relative path per line, in FILES_WRITTEN.txt: modified product files, artifacts, copied sources/packages, generated build files and modules. FILES.sha256 records their hashes, excluding its own self-hash. The externally written live transcript is separately identified in G5_SCOPE.json. Scratch retained: /tmp/eng3f1_tmp; tensor_cuda/build-eng3f1 contains four build trees, four source copies and reference_E. Nothing was deleted or moved to DELETE_ME. New Python helpers all use eng3f1-prefixed names and run with python3 -B -P; no standard-library name collisions.
