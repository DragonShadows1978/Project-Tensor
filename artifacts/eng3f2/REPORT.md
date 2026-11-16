# ENG3-F2 report

G1 PASS. G3 PASS. G4 PASS. G2 CPU PASS.
G2 GPU STAGED, NOT RUN. Registered acceptance remains pending the lead's GPU lines; no further diagnostic review is requested by this order.

## Changes and unit-test evidence

- ENG3R2-F01 fixed at tensor_cuda/src/kernels.cu:551: the existing byte-bound throw now names the entire shape and canonical dtype. The comparison, limit and allocation path are unchanged.
- ENG3R2-F02 fixed in core_device's shared refusal helper at tensor_cuda/src/bindings.cpp:2661: every byte outside printable ASCII 0x20..0x7e becomes lowercase `\xNN` before exception translation. NUL and following bytes remain visible. The three device-taking hooks use this helper; _core_clone has no string operand.
- ENG3R2-F03 remains closed with no change; no device-index parser or numeric behavior edits.
- Diff: three existing files, 84 inserted lines, 3 replaced lines; test_eng3_host.py only receives appended tests. See source.diff and before/ copies. All engine source hashes match Build F except the two authorized message files (build/BUILD_G.json versus the immutable F receipt).

G1 unit-test/suite evidence: host/G1_first/RESULT.json has 48 passing overflow cases; host/G1/RESULT.json has 254 passing CPU cases (77 inherited, 48 overflow, 129 device-message cases). Overflow cases cover all six dtypes, both creation hooks, bare cpu/cuda descriptors and nonempty/zero-extent shapes without device allocation. Device cases cover all three string-taking hooks, embedded NUL and suffixes, every ASCII control byte and DEL, UTF-8 high bytes and printable malformed inputs. CUDA_VISIBLE_DEVICES was empty in every CPU run. G1.log, G1_first.log and per-case JSONL preserve the actual messages.

G2 CPU regression evidence: staged/g3-cpu/COMPARE.json compares exactly 3,368 test IDs and the collection-error file sets against Build F, with zero differences and zero test writes. Each inherited suite exits 1: 815 passed, 258 skipped, 1,946 failed, 349 error outcomes. These are identical inherited outcomes, not a clean-engine claim. Collection-error details remain in g3/g3-cpu_F/outcomes.json and g3/g3-cpu_G/outcomes.json; no log/timing equality was used. G2_HOST_COMPARE.json separately records exact identity for all 77 inherited host CPU test IDs. Added F2 tests are validated against the order, not expected to pass unchanged Build F.

G3 mutant evidence: f2m01_message_unchanged was fully built from an isolated copy of G with only the F01 throw text reverted to F1's. The first G1 test executes that throw and exits 1 as expected, with 48 failures. Every captured assertion contains the exact old message, proving mutated-line execution. Example failure, verbatim:

```
E       AssertionError: NDArray: byte size overflows int64 (nonzero extents count next to zero)
E       assert '[2305843009213693952]' in 'NDArray: byte size overflows int64 (nonzero extents count next to zero)'
```

See MUTANT_RESULTS.json, host/G3/RESULT.json and G3.log. G remains unchanged.

## Builds and module sha256s

Both full CMake builds configure and compile successfully, sm_89 matching Build F, disconnected pybind11, at most -j4 under nice -n 19 and /usr/bin/time -v, bounded by timeout. Build receipts contain command arrays and SHA256 maps of every copied engine source plus CMakeLists.txt. The monitor enforces MemAvailable >=12 GiB and disk free >=40 GiB throughout. See build/BUILD_G.json, build/BUILD_f2m01_message_unchanged.json and their configure/build logs (including maximum resident set size). No build/GPU performance claim.

- F: `642d5ab7b57d4c6796cd59f701ad4dcdddde24c0a01f478d3c6630b513f6040e` (`tensor_cuda/build-eng3f1/source_F/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`).
- G: `5fcd179b98b0749af0d75414d35ae0fc8519bf200364c30c743904a90fc1a711` (`tensor_cuda/build-eng3f2/source_G/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`).
- f2m01_message_unchanged: `2ac1d7bba241ab9df18778006fef9c698b069e79b3445fdbcc058db3506d50c6` (`tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`).

## Staged GPU lines and expected exits

From `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng3`, the lead runs exactly:

```bash
timeout 1850s env FORGE_GPU_DRY_RUN=0 bash artifacts/eng3f2/staged/launch-g3-g1.sh
timeout 1850s env FORGE_GPU_DRY_RUN=0 bash artifacts/eng3f2/staged/launch-g3-g2.sh
```

Each launcher sets its package, thread, temp and bytecode environment explicitly and sets cwd, then checks BOTH module SHA256 pins and executes both packages' real host prefix before requesting the lease. Its GPU worker starts with the same prefix and stops there in prefix mode, immediately before tc.tensor(src, device=cuda). Each runs the unchanged F1 split-file census/options against F and G. The launcher bodies contain exactly these forge forms:

```bash
forge-gpu-run --min 30 --max 1800 --label eng3f2-g3-g1 -- timeout 1750s nice -n 19 python3 -B -P scripts/eng3f2_stage.py run g3-g1 > artifacts/eng3f2/staged/g3-g1/GPU.log 2>&1
forge-gpu-run --min 30 --max 1800 --label eng3f2-g3-g2 -- timeout 1750s nice -n 19 python3 -B -P scripts/eng3f2_stage.py run g3-g2 > artifacts/eng3f2/staged/g3-g2/GPU.log 2>&1
```

Expected for each real launcher: forge exit 0, inner comparison exit 0, launcher exit 0; individual pytest exits/outcomes are retained and compared per ID. The launcher's finish step records forge_exit, inner_exit and GPU.log SHA256 in staged/<line>/RESULT.json. Lead retains those receipts (L45). A module mismatch refuses before the lease. Scope excludes promotion.

This seat ran only `FORGE_GPU_DRY_RUN=1` launchers: both F/G CPU prefixes passed; both forge/launcher exits were 78, inner_exit null, gpu_run false. The dry-run branch is before lock/lease acquisition. See STAGED_LINES.json, staged/<line>/CPU_PREFIX.json and DRY_RESULT.json, dry_<line>.log and the hashed GPU.log dry-run output. GPU parity is pending, not inferred from these CPU prefixes.

## Scope and deviations

G4 PASS (metadata census plus source/command review): only the authorized files/new roots were written; no files deleted; no git commands/writes, no subagents and no GPU runs. Existing test content is byte-for-byte retained at its prefix. Earlier artifacts/builds, other worktrees and production were read-only. G4_SCOPE.json and SCOPE_BEFORE/AFTER.json hold the workspace evidence; source.diff limits product changes to the two message sites. No claims about arbitrary external filesystem activity are inferred from the workspace census.

Deviations/tool slips (no gate or threshold changes):

- Initial read-only inspection commands did not all set TMPDIR/CUDA_VISIBLE_DEVICES; no gates, builds or Python data processing ran in those commands.
- One ls included absent scripts/eng3f1* and tensor_cuda/tests/test_eng3f1* (exit 2); actual helpers are artifacts/eng3f1/tools. A later read-only cat probe found no conftest.py or artifacts/eng3/tools/suite.py (exit 1).
- Oversized inspection output was truncated; relevant files were reread in bounded batches.
- Own-script repair 1/3: pre-execution review restored an omitted out = BUILD/arm assignment while adapting the inherited build helper; no script exception or gate run preceded it. Own-script repair 2/3: final hash verification detected the external dispatcher appending logs/eng3f2_r1.log; the sealing helper now snapshots and labels live logs rather than claiming their hashes are stable. No gate/source/threshold changed.
- Expected mutant exit 1 and dry-run exits 78 are registered results, not deviations.

No deletes or moves to DELETE_ME. Retained scratch: /tmp/eng3f2_scratch, both build trees and all receipts. SCRATCH_WRITTEN.txt enumerates every retained scratch file/symlink; removed internal pytest temporaries under /tmp are exempt from the no-delete rule.

## Prior art

Existing P1T contextual shape/dtype diagnostics (2026); Python repr and C++ hexadecimal escape conventions (ISO C++17, 2017). Taken: operand context and printable byte escaping. Ours: placement at these two shared diagnostic sites and the registered test census. Bibliography unverified -- lead to check: Python repr nonprintable byte escapes, C++17 hexadecimal escape sequences. No new numeric algorithm, selection rule, data structure or optimization.

Harness prior art: inherited ENG3-F1 suite/split/pin/prefix protocol (2026); pytest hooks/parametrization (2004 onward); NIST SHA-256 (2001); DeMillo, Lipton and Sayward mutation testing (1978); CMake out-of-tree builds (Kitware, 2000); /proc/meminfo and time(1); Python difflib (PSF, year unverified -- lead to check). Taken: per-ID phase folding, inherited corpus/options, hashes, controlled mutation, isolated builds and resource rails. Ours: F2 receipt/report assembly; no prior art known to me for the exact report layout. Code-site comments and LEDGER.md carry these attributions.

## Every file written

FILES_WRITTEN.txt is the complete worktree inventory (307 files, including this report and itself); FILES.sha256 hashes every listed file except the hash manifest itself and the live dispatcher logs identified by VOLATILE_FILES.json. The immutable runtime_logs snapshots are hashed. It covers the three edited files, all scripts/eng3f2_*.py helpers, every generated file in tensor_cuda/build-eng3f2 and every artifacts/eng3f2 receipt. SCRATCH_WRITTEN.txt separately enumerates retained /tmp files. The dispatcher appended logs/eng3f2_r1.log during this session; it is retained in the inventory, with a captured snapshot and its observation limit in VOLATILE_FILES.json. Agent commands created no logs/eng3f2_* file.

The complete worktree file list follows, verbatim:

```text
artifacts/eng3f2/AMENDMENT_01.md
artifacts/eng3f2/FILES.sha256
artifacts/eng3f2/FILES_WRITTEN.txt
artifacts/eng3f2/G1.log
artifacts/eng3f2/G1_first.log
artifacts/eng3f2/G2_HOST_COMPARE.json
artifacts/eng3f2/G2_cpu.log
artifacts/eng3f2/G2_host_F.log
artifacts/eng3f2/G3.log
artifacts/eng3f2/G4_SCOPE.json
artifacts/eng3f2/GATES.json
artifacts/eng3f2/INSTRUMENTS.log
artifacts/eng3f2/LEDGER.md
artifacts/eng3f2/MODULES.json
artifacts/eng3f2/MUTANT_RESULTS.json
artifacts/eng3f2/REGISTRATION.md
artifacts/eng3f2/REGISTRATION.sha256
artifacts/eng3f2/REPORT.md
artifacts/eng3f2/REPRODUCTION_F.json
artifacts/eng3f2/SCOPE_AFTER.json
artifacts/eng3f2/SCOPE_BEFORE.json
artifacts/eng3f2/SCRATCH_WRITTEN.txt
artifacts/eng3f2/STAGED_LINES.json
artifacts/eng3f2/VOLATILE_FILES.json
artifacts/eng3f2/before/bindings.cpp
artifacts/eng3f2/before/kernels.cu
artifacts/eng3f2/before/test_eng3_host.py
artifacts/eng3f2/build/BUILD_G.json
artifacts/eng3f2/build/BUILD_f2m01_message_unchanged.json
artifacts/eng3f2/build/build_G.log
artifacts/eng3f2/build/build_f2m01_message_unchanged.log
artifacts/eng3f2/build/configure_G.log
artifacts/eng3f2/build/configure_f2m01_message_unchanged.log
artifacts/eng3f2/dry_g3-g1.log
artifacts/eng3f2/dry_g3-g2.log
artifacts/eng3f2/g3/g3-cpu_F/junit.xml
artifacts/eng3f2/g3/g3-cpu_F/outcomes.json
artifacts/eng3f2/g3/g3-cpu_G/junit.xml
artifacts/eng3f2/g3/g3-cpu_G/outcomes.json
artifacts/eng3f2/host/G1/RESULT.json
artifacts/eng3f2/host/G1/constant_reference.jsonl
artifacts/eng3f2/host/G1/f2_device.jsonl
artifacts/eng3f2/host/G1/f2_overflow.jsonl
artifacts/eng3f2/host/G1/g1.jsonl
artifacts/eng3f2/host/G1/g1b.jsonl
artifacts/eng3f2/host/G1/junit.xml
artifacts/eng3f2/host/G1/l19.jsonl
artifacts/eng3f2/host/G1/runtime_unavailable.jsonl
artifacts/eng3f2/host/G1_first/RESULT.json
artifacts/eng3f2/host/G1_first/f2_overflow.jsonl
artifacts/eng3f2/host/G1_first/junit.xml
artifacts/eng3f2/host/G2_host_F/RESULT.json
artifacts/eng3f2/host/G2_host_F/constant_reference.jsonl
artifacts/eng3f2/host/G2_host_F/g1.jsonl
artifacts/eng3f2/host/G2_host_F/g1b.jsonl
artifacts/eng3f2/host/G2_host_F/junit.xml
artifacts/eng3f2/host/G2_host_F/l19.jsonl
artifacts/eng3f2/host/G2_host_F/runtime_unavailable.jsonl
artifacts/eng3f2/host/G3/RESULT.json
artifacts/eng3f2/host/G3/junit.xml
artifacts/eng3f2/runtime_logs/eng3f2_r1.log
artifacts/eng3f2/source.diff
artifacts/eng3f2/staged/g3-cpu/COMPARE.json
artifacts/eng3f2/staged/g3-cpu/INNER_RESULT.json
artifacts/eng3f2/staged/g3-cpu/suite_F.log
artifacts/eng3f2/staged/g3-cpu/suite_G.log
artifacts/eng3f2/staged/g3-g1/CPU_PREFIX.json
artifacts/eng3f2/staged/g3-g1/DRY_RESULT.json
artifacts/eng3f2/staged/g3-g1/GPU.log
artifacts/eng3f2/staged/g3-g1/prefix_F.log
artifacts/eng3f2/staged/g3-g1/prefix_G.log
artifacts/eng3f2/staged/g3-g2/CPU_PREFIX.json
artifacts/eng3f2/staged/g3-g2/DRY_RESULT.json
artifacts/eng3f2/staged/g3-g2/GPU.log
artifacts/eng3f2/staged/g3-g2/prefix_F.log
artifacts/eng3f2/staged/g3-g2/prefix_G.log
artifacts/eng3f2/staged/launch-g3-g1.sh
artifacts/eng3f2/staged/launch-g3-g2.sh
logs/eng3f2_r1.log
scripts/eng3f2_build.py
scripts/eng3f2_close.py
scripts/eng3f2_host.py
scripts/eng3f2_stage.py
scripts/eng3f2_suite.py
tensor_cuda/build-eng3f2/G/CMakeCache.txt
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CMakeCUDACompiler.cmake
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CMakeCXXCompiler.cmake
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CMakeDetermineCompilerABI_CUDA.bin
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CMakeDetermineCompilerABI_CXX.bin
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CMakeSystem.cmake
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CompilerIdCUDA/CMakeCUDACompilerId.cu
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CompilerIdCUDA/a.out
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cpp1.ii
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cpp4.ii
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.c
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.cpp
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.gpu
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.stub.c
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.fatbin
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.fatbin.c
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.module_id
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.o
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.ptx
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.sm_52.cubin
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.fatbin
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.fatbin.c
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.o
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.reg.c
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.sm_52.cubin
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CompilerIdCXX/CMakeCXXCompilerId.cpp
tensor_cuda/build-eng3f2/G/CMakeFiles/3.28.3/CompilerIdCXX/a.out
tensor_cuda/build-eng3f2/G/CMakeFiles/CMakeConfigureLog.yaml
tensor_cuda/build-eng3f2/G/CMakeFiles/CMakeDirectoryInformation.cmake
tensor_cuda/build-eng3f2/G/CMakeFiles/Makefile.cmake
tensor_cuda/build-eng3f2/G/CMakeFiles/Makefile2
tensor_cuda/build-eng3f2/G/CMakeFiles/TargetDirectories.txt
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/DependInfo.cmake
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/build.make
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/cmake_clean.cmake
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/cmake_device_link.o
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/compiler_depend.make
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/compiler_depend.ts
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/depend.make
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/deviceLinkLibs.rsp
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/deviceObjects1.rsp
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/dlink.txt
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/flags.make
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/includes_CUDA.rsp
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/link.txt
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/progress.make
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/autograd.cpp.o
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/autograd.cpp.o.d
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/bindings.cpp.o
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/bindings.cpp.o.d
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/conv.cu.o
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/conv.cu.o.d
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/dda.cu.o
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/dda.cu.o.d
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/gemm_apa.cu.o
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/gemm_apa.cu.o.d
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/kernels.cu.o
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/kernels.cu.o.d
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/matmul.cu.o
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/matmul.cu.o.d
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/ops.cpp.o
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/ops.cpp.o.d
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/paint_bake.cu.o
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/paint_bake.cu.o.d
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/paint_inpaint.cu.o
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/paint_inpaint.cu.o.d
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/paint_raster.cu.o
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/paint_raster.cu.o.d
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/terrain.cu.o
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/terrain.cu.o.d
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/terrain_objects.cu.o
tensor_cuda/build-eng3f2/G/CMakeFiles/_tensor_cuda.dir/src/terrain_objects.cu.o.d
tensor_cuda/build-eng3f2/G/CMakeFiles/cmake.check_cache
tensor_cuda/build-eng3f2/G/CMakeFiles/progress.marks
tensor_cuda/build-eng3f2/G/Makefile
tensor_cuda/build-eng3f2/G/_deps/pybind11-build/CMakeFiles/CMakeDirectoryInformation.cmake
tensor_cuda/build-eng3f2/G/_deps/pybind11-build/CMakeFiles/progress.marks
tensor_cuda/build-eng3f2/G/_deps/pybind11-build/Makefile
tensor_cuda/build-eng3f2/G/_deps/pybind11-build/cmake_install.cmake
tensor_cuda/build-eng3f2/G/cmake_install.cmake
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeCache.txt
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CMakeCUDACompiler.cmake
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CMakeCXXCompiler.cmake
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CMakeDetermineCompilerABI_CUDA.bin
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CMakeDetermineCompilerABI_CXX.bin
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CMakeSystem.cmake
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CompilerIdCUDA/CMakeCUDACompilerId.cu
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CompilerIdCUDA/a.out
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cpp1.ii
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cpp4.ii
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.c
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.cpp
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.gpu
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.stub.c
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.fatbin
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.fatbin.c
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.module_id
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.o
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.ptx
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.sm_52.cubin
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.fatbin
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.fatbin.c
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.o
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.reg.c
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.sm_52.cubin
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CompilerIdCXX/CMakeCXXCompilerId.cpp
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/3.28.3/CompilerIdCXX/a.out
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/CMakeConfigureLog.yaml
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/CMakeDirectoryInformation.cmake
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/Makefile.cmake
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/Makefile2
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/TargetDirectories.txt
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/DependInfo.cmake
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/build.make
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/cmake_clean.cmake
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/cmake_device_link.o
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/compiler_depend.make
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/compiler_depend.ts
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/depend.make
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/deviceLinkLibs.rsp
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/deviceObjects1.rsp
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/dlink.txt
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/flags.make
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/includes_CUDA.rsp
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/link.txt
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/progress.make
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/autograd.cpp.o
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/autograd.cpp.o.d
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/bindings.cpp.o
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/bindings.cpp.o.d
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/conv.cu.o
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/conv.cu.o.d
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/dda.cu.o
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/dda.cu.o.d
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/gemm_apa.cu.o
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/gemm_apa.cu.o.d
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/kernels.cu.o
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/kernels.cu.o.d
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/matmul.cu.o
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/matmul.cu.o.d
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/ops.cpp.o
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/ops.cpp.o.d
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/paint_bake.cu.o
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/paint_bake.cu.o.d
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/paint_inpaint.cu.o
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/paint_inpaint.cu.o.d
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/paint_raster.cu.o
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/paint_raster.cu.o.d
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/terrain.cu.o
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/terrain.cu.o.d
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/terrain_objects.cu.o
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/_tensor_cuda.dir/src/terrain_objects.cu.o.d
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/cmake.check_cache
tensor_cuda/build-eng3f2/f2m01_message_unchanged/CMakeFiles/progress.marks
tensor_cuda/build-eng3f2/f2m01_message_unchanged/Makefile
tensor_cuda/build-eng3f2/f2m01_message_unchanged/_deps/pybind11-build/CMakeFiles/CMakeDirectoryInformation.cmake
tensor_cuda/build-eng3f2/f2m01_message_unchanged/_deps/pybind11-build/CMakeFiles/progress.marks
tensor_cuda/build-eng3f2/f2m01_message_unchanged/_deps/pybind11-build/Makefile
tensor_cuda/build-eng3f2/f2m01_message_unchanged/_deps/pybind11-build/cmake_install.cmake
tensor_cuda/build-eng3f2/f2m01_message_unchanged/cmake_install.cmake
tensor_cuda/build-eng3f2/source_G/CMakeLists.txt
tensor_cuda/build-eng3f2/source_G/include/dlpack/dlpack.h
tensor_cuda/build-eng3f2/source_G/include/tc/autograd.h
tensor_cuda/build-eng3f2/source_G/include/tc/bp_op_timing.h
tensor_cuda/build-eng3f2/source_G/include/tc/core.h
tensor_cuda/build-eng3f2/source_G/include/tc/ops.h
tensor_cuda/build-eng3f2/source_G/src/apa_sp1_1.cuh
tensor_cuda/build-eng3f2/source_G/src/apa_sp2.cuh
tensor_cuda/build-eng3f2/source_G/src/autograd.cpp
tensor_cuda/build-eng3f2/source_G/src/bindings.cpp
tensor_cuda/build-eng3f2/source_G/src/conv.cu
tensor_cuda/build-eng3f2/source_G/src/dda.cu
tensor_cuda/build-eng3f2/source_G/src/gemm_apa.cu
tensor_cuda/build-eng3f2/source_G/src/kernels.cu
tensor_cuda/build-eng3f2/source_G/src/matmul.cu
tensor_cuda/build-eng3f2/source_G/src/ops.cpp
tensor_cuda/build-eng3f2/source_G/src/paint_bake.cu
tensor_cuda/build-eng3f2/source_G/src/paint_inpaint.cu
tensor_cuda/build-eng3f2/source_G/src/paint_raster.cu
tensor_cuda/build-eng3f2/source_G/src/terrain.cu
tensor_cuda/build-eng3f2/source_G/src/terrain_objects.cu
tensor_cuda/build-eng3f2/source_G/tensor_cuda/__init__.py
tensor_cuda/build-eng3f2/source_G/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
tensor_cuda/build-eng3f2/source_G/tensor_cuda/apa_sp2.py
tensor_cuda/build-eng3f2/source_G/tensor_cuda/functional.py
tensor_cuda/build-eng3f2/source_G/tensor_cuda/nn.py
tensor_cuda/build-eng3f2/source_G/tensor_cuda/optim.py
tensor_cuda/build-eng3f2/source_G/tensor_cuda/quant.py
tensor_cuda/build-eng3f2/source_G/tensor_cuda/quantization/__init__.py
tensor_cuda/build-eng3f2/source_G/tensor_cuda/quantization/affine.py
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/CMakeLists.txt
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/include/dlpack/dlpack.h
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/include/tc/autograd.h
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/include/tc/bp_op_timing.h
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/include/tc/core.h
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/include/tc/ops.h
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/src/apa_sp1_1.cuh
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/src/apa_sp2.cuh
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/src/autograd.cpp
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/src/bindings.cpp
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/src/conv.cu
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/src/dda.cu
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/src/gemm_apa.cu
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/src/kernels.cu
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/src/matmul.cu
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/src/ops.cpp
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/src/paint_bake.cu
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/src/paint_inpaint.cu
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/src/paint_raster.cu
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/src/terrain.cu
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/src/terrain_objects.cu
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/tensor_cuda/__init__.py
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/tensor_cuda/apa_sp2.py
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/tensor_cuda/functional.py
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/tensor_cuda/nn.py
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/tensor_cuda/optim.py
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/tensor_cuda/quant.py
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/tensor_cuda/quantization/__init__.py
tensor_cuda/build-eng3f2/source_f2m01_message_unchanged/tensor_cuda/quantization/affine.py
tensor_cuda/src/bindings.cpp
tensor_cuda/src/kernels.cu
tensor_cuda/tests/test_eng3_host.py
```
