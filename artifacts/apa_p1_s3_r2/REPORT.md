# APA-P1-S3 r2b report

**ACCEPT for the registered CPU/staging round.** No GPU run, default switch, blind red team or promotion was performed. Those remain the lead's next steps. Evidence: `GATES.json`.

- G1 PASS — per-device merge/choose ownership, guarded allocation/launch, per-device L2, host ownership unit, staged native two-thread comparison.
- G2 PASS — compiled CPU structural check passes both correct predicates and rejects EB3 mutant 3 / EB4's early-counter analogue; real-input GPU cases staged.
- G3 PASS — unchanged lead exact-tie fixture integrated; CPU fixture checks pass; correct/reversed-tie GPU cases staged.
- G4 PASS — default-switch design and complete source/argument census written.
- G5 PASS — six staged lines, each dry-run as written to 78.
- G6 PASS — 2,347 existing test IDs have zero phase/outcome changes; all 744 SASS kernels outside the select families are identical.
- G7 PASS — authorized writes only; no GPU, git command/write, subagent, seat deletion or retirement/move. Order/registration preserved; 591 initial source/config/doc hashes audited.

## Diff and build

`ALL_CHANGES.diff`: **636 added / 36 removed lines**, across three existing source files and three new tests. Existing edits: `eb3.cuh` +161/-20, `eb4.cuh` +45/-16, `bindings.cpp` +21/-0. `eb4_tuning.cuh`, all public Python and other engine files remain unchanged.

Tickets now use a device-keyed map containing separate merge and choose slots plus L2 metadata. A recursive host mutex covers lookup, growth and launch submission, preventing growth from freeing a pointer before another thread submits it. The existing checked device guard sets/restores the operands' device. Growth synchronizes that device only. Contract: one CUDA legacy default stream per device; other streams are out of contract. Preparation helpers outside this patch retain their existing caller-owned device contract.

The handshake and GIL-releasing native test binding compile only with `TC_APA_P1_TEST_HOOKS=1`; its default is 0. Chunk 1 waits before all partial stores, with explicit NaN poison. Chunk 0 signals after its non-merging exit or all merge/output stores. Residency is checked and device polling is bounded; timeout is a harness failure, never a mutant kill. Selection math, reductions and shipped public APIs/defaults are unchanged.

One isolated Release/sm89 build tree used the A1 script's path adapter, with the test flag enabled. It had an initial compile error and two incremental resumes; all receipts remain. Final module SHA-256: `5fba7c20db7e3c40674d2f396cbff77cbba947d0af99b8eac7106d97ac5f4d8f` (`BUILD_COMPLETED.json`).

`SASS_DIFF.json` compares 778 kernels with exact bodies, instruction encodings and control words. Only bijective generated-symbol names and the trailing test-hook argument are mapped. **744 unchanged; 34 changed**, all in the test-enabled select families. The complete changed-kernel list is the following set expansion (G/D/T range independently within each row); full signatures and both symbols are in `CHANGED_KERNELS.txt` / `SASS_DIFF.json`:

| Changed specialization | Ranges | Count |
|---|---|---:|
| `eb3::select_pv<128,128,G,false,1,true,false,false,T,false>` | G=2,4; T=false,true | 4 |
| `eb3::select_pv<128,128,G,false,5,false,false,false,T,false>` | G=2,4; T=false,true | 4 |
| `eb3::select_pv<128,128,G,false,5,false,true,false,false,false>` | G=2,4 | 2 |
| `eb3::select_pv<128,128,G,D,5,false,true,true,false,false>` | G=2,3,4,5,6,7,8; D=false,true | 14 |
| `eb3::select_pv<128,128,G,false,5,false,true,true,false,true>` | G=2,4 | 2 |
| `eb3::select_pv<128,128,G,false,5,false,true,true,true,false>` | G=2,4 | 2 |
| `eb4::select_mixed<128,128,G,D,true>` | G=2,4,8; D=false,true | 6 |

## Tests and limits

The three new files are `test_apa_p1_tickets.py`, `test_apa_p1_early_counter.py`, and `test_apa_p1_exact_tie.py`. CPU result: **5 passed, 4 GPU tests skipped** (`CPU_TESTS_verified.json/.log`). The ownership test compiles actual registry/choose-wrapper/checked-guard source against a CPU CUDA-API recorder. The early test compiles the actual handshake and predicates with CTA-leader instruments: EB3 correct 0, early 1; EB4 correct 0, early 1, each mutant reporting `deterministic poisoned-partial output mismatch`. These are structural/unit results, not GPU proof. Generated C++ and results are retained in `cpu_compiled/`.

The tie fixture is unchanged by AST comparison to the copied lead source: H16/KV8, D128, S2048, blocks 5/20, fraction 1/32. GPU assertions require equal fp32 cutoff scores, one high block per KV head, block 5, and bitwise EB2b/EB4 flags/output. The reverse mutant must first demonstrate block 20. Saved references use full-width integer views and explicit nonzero/nonconstant guards.

All 74 existing test files were run before/after with empty GPU visibility, no pytest cache provider, and /tmp basetemps. Per-ID totals on **both** arms: 475 passed, 1537 failed, 320 skipped, 15 setup errors; the same collection-error file is retained. There are **zero ID/phase/outcome differences** (`OUTCOMES_before.json`, `OUTCOMES_after.json`, `OUTCOMES_DIFF.json`). These are parity results, not an all-passing suite. The collection error is `test_selector_accuracy.py`: `ValueError: invalid literal for int() with base 10: 'before'` / `'after'`, because that script consumes the harness argv. External file opens are recorded in each outcome receipt.

Failed attempts retained: build `error: identifier "test_hook" is undefined` from an unintended bulk-exit insertion; CPU extraction `expected ‘)’ before ‘compiled’` / `expected ‘)’ before ‘compiles’`; initial SASS comparison had 212 unrelated changes from a static shared timeout flag. The bulk exit and marker extraction were corrected; a barrier-reduction broadcast removed the extra shared object, after which all 744 unrelated kernels matched exactly. No gate or tolerance was weakened.

## Default-switch design and caller census

[DEFAULT_SWITCH_DESIGN.md](/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/artifacts/apa_p1_s3_r2/DEFAULT_SWITCH_DESIGN.md) preserves the epsilon contract: derive upward-rounded `-log(epsilon)+2*e_q` from a backend-matching registered margin. A direct packed delta supplies only its computed-score running-prefix rule without that error assumption. The note covers every argument, a proposed typed packed-key quantizer result, production/raw caller migration, shipped packed build/default storage, dense opt-out/fallback, and required memory/calibration/model-quality measurements. It distinguishes 580 B/key/KV-head bf16 SP from 712 tk8, 768 dense bf16 and the proposed 1092 packed fp32 layout. AB-1's RED/boundary results remain labeled; neither AB-1 nor EB4 certifies the new governor.

Census: 560 literal source lines here, 11 in GRAPA's archived engine patch, 0 in the GraftRepositoryMemory/GraftRepository checkout, 0 in Project-Voice. These include definitions/snapshots; they are not caller counts. The argument inventory has 281 Python calls: 80 current source/test/harness sites and 201 artifact/reference sites. Full grep commands/output and all arguments are pasted in the design note and retained in `CALLER_CENSUS_FINAL.json`, `CALLER_ARGUMENTS.md` and `CALLER_CENSUS_FINAL_COMMANDS.md`.

The active quantizer consumers are `scripts/apa_pidv.py:269` (fp32 K, 2 bits, downloaded instrument Kq) and `scripts/apa_sp2_gpu.py:53` (fp32 K, variable bits). Public model/harness calls are `apa_pidv.py:303` (epsilon=.05, bits=2, synthetic instrument margins) and `apa_sp2_gpu.py:163,166` (epsilon, bits, registered margins, causal/diagnostic controls). Direct-delta harness callers and aliases are individually listed with q/k/kq/v, scale/delta, causal/sink/diagnostic expressions in the note. GRAPA has archived native forwarding at `artifacts/bp_census_1/engine_patch/src/bindings.cpp:675,696` and diagnostics at 685,707,713; no active matching application call was found. GraftRepositoryMemory is the workspace service name for the GraftRepository checkout; no separate checkout of that name was found at the checked explicit paths. This is a bounded source census, not runtime reachability evidence.

## Staged lines and dry runs

Run individually from the stated worktree, by the lead with GPU visibility and a live lease. `EXPECTED.json` preceded staging; the documented source-pin amendment preserves the initial records. Correct lines expect 0; named mutants expect 1 at their registered assertion. Unexpected failures/timeouts return 2 and do not count as kills. Every line below was dry-run verbatim with `CUDA_VISIBLE_DEVICES=` and `FORGE_GPU_DRY_RUN=1`: **78,78,78,78,78,78** (`DRYRUN.json`). First unmet live-run need: scheduler state halted; no child was executed.

```sh
forge-gpu-run --min 30 --max 240 -- timeout 210s python -B artifacts/apa_p1_s3_r2/gpu_line.py --case threads
forge-gpu-run --min 30 --max 240 -- timeout 210s python -B artifacts/apa_p1_s3_r2/gpu_line.py --case early_correct
forge-gpu-run --min 30 --max 590 -- timeout 560s python -B artifacts/apa_p1_s3_r2/gpu_line.py --case eb3_early
forge-gpu-run --min 30 --max 590 -- timeout 560s python -B artifacts/apa_p1_s3_r2/gpu_line.py --case eb4_early
forge-gpu-run --min 30 --max 240 -- timeout 210s python -B artifacts/apa_p1_s3_r2/gpu_line.py --case tie_correct
forge-gpu-run --min 30 --max 590 -- timeout 560s python -B artifacts/apa_p1_s3_r2/gpu_line.py --case eb4_reverse_tie
```

In that order: correct two-thread EB3/EB4, both correct early cases, EB3 `APA_EB3_MUTANT=3`, EB4 `NC-1 -> NC-2` analogue (not EB4's unrelated V-scale mutant 3), correct tie, EB4 `APA_EB4_MUTANT=7`. Mutant binaries are isolated derivatives of this round's source/build and compile inside their lead lines with GPU visibility empty for compilation. Actual GPU results are not claimed.

## Prior art

NVIDIA CUDA Samples `threadFenceReduction` (year unverified), CUDA stream-ordered lifetimes (NVIDIA,2021) and ISO C++11 mutex/RAII(2011): inherited tickets/order, new per-device ownership and enqueue protection. DeMillo, Lipton, Sayward(1978): mutation testing; APA-2P survey S3(2026): the two-CTA poisoned-partial design. Batcher(1968), through EB2b and the EB4 lead fixture(2026): inherited stable tie order and unchanged fixture. NVIDIA CUDA barrier reduction supplies the timeout broadcast. S4 inherits TurboQuant(2025), BLASST(2025), Flash-Decoding(2023), KIVI(2024), ThriftAttention(2026), and Project-Tensor SP2/workspaces(2026); this round proposes an adapter/migration, not a new selector. Census/receipts use ripgrep(2016), Python ast(2.6/2008, year unverified), pytest, SHA-256/NIST(2015), and P1's exact comparison approach(2026). No prior art known to me for this exact test/receipt assembly beyond the cited local designs. Literature metadata is **unverified — lead to check**; search terms and attribution at code sites, ledger and design identify the leads.

## Every file written

[FILES_WRITTEN.md](/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/artifacts/apa_p1_s3_r2/FILES_WRITTEN.md) and `FILES_WRITTEN.json` enumerate **193 workspace file entries**, including generated files, copies and symlinks, plus **6,490 individual /tmp scratch entries**. `MANIFEST_SHA256.txt` seals workspace file contents (excluding itself). Nothing was retired, moved or deleted by the seat.

Existing edited/new test files:

```text
tensor_cuda/src/apa_spo/eb3.cuh
tensor_cuda/src/apa_spo/eb4.cuh
tensor_cuda/src/bindings.cpp
tensor_cuda/tests/test_apa_p1_early_counter.py
tensor_cuda/tests/test_apa_p1_exact_tie.py
tensor_cuda/tests/test_apa_p1_tickets.py
```

All remaining written files, relative to `artifacts/apa_p1_s3_r2/`:

```text
ALL_CHANGES.diff
BUILD.json
BUILD.py
BUILD_COMPLETED.json
BUILD_COMPLETED_initial.json
BUILD_PATH_ADAPTER.diff
CALLER_ARGUMENTS.md
CALLER_CENSUS.json
CALLER_CENSUS_COMMANDS.md
CALLER_CENSUS_FINAL.json
CALLER_CENSUS_FINAL_COMMANDS.md
CHANGED_KERNELS.txt
COMMANDS.md
CPU_COMPILED_CENSUS.json
CPU_TESTS.json
CPU_TESTS.log
CPU_TESTS_final.json
CPU_TESTS_final.log
CPU_TESTS_verified.json
CPU_TESTS_verified.log
DEFAULT_SWITCH_DESIGN.head.md
DEFAULT_SWITCH_DESIGN.md
DIFF_SUMMARY.json
DRYRUN.json
ENGINE_CHANGE.diff
EXPECTED.json
FILES_WRITTEN.json
FILES_WRITTEN.md
FIXTURE_PROVENANCE.json
GATES.json
INSTRUMENTS.json
INSTRUMENTS_FINAL.json
LEDGER.md
MANIFEST_SHA256.txt
OUTCOMES_DIFF.json
OUTCOMES_after.json
OUTCOMES_before.json
REGISTRATION.json
REGRESSION_FILES.json
REPORT.md
SASS_BASELINE_PROVENANCE.json
SASS_DIFF.json
SCOPE_AFTER.json
SCOPE_BEFORE.json
STAGED_GPU_LINES.sh
STAGING_AMENDMENT_1.json
TICKET_BOUNDARY.json
TICKET_CENSUS.txt
add_hooks.py
after.sass
after_broadcast.sass
bindings.cpp.diff
build/CMakeCache.txt
build/CMakeFiles/3.28.3/CMakeCUDACompiler.cmake
build/CMakeFiles/3.28.3/CMakeCXXCompiler.cmake
build/CMakeFiles/3.28.3/CMakeDetermineCompilerABI_CUDA.bin
build/CMakeFiles/3.28.3/CMakeDetermineCompilerABI_CXX.bin
build/CMakeFiles/3.28.3/CMakeSystem.cmake
build/CMakeFiles/3.28.3/CompilerIdCUDA/CMakeCUDACompilerId.cu
build/CMakeFiles/3.28.3/CompilerIdCUDA/a.out
build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cpp1.ii
build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cpp4.ii
build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.c
build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.cpp
build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.gpu
build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.stub.c
build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.fatbin
build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.fatbin.c
build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.module_id
build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.o
build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.ptx
build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.sm_52.cubin
build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.fatbin
build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.fatbin.c
build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.o
build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.reg.c
build/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.sm_52.cubin
build/CMakeFiles/3.28.3/CompilerIdCXX/CMakeCXXCompilerId.cpp
build/CMakeFiles/3.28.3/CompilerIdCXX/a.out
build/CMakeFiles/CMakeConfigureLog.yaml
build/CMakeFiles/CMakeDirectoryInformation.cmake
build/CMakeFiles/Makefile.cmake
build/CMakeFiles/Makefile2
build/CMakeFiles/TargetDirectories.txt
build/CMakeFiles/_tensor_cuda.dir/DependInfo.cmake
build/CMakeFiles/_tensor_cuda.dir/build.make
build/CMakeFiles/_tensor_cuda.dir/cmake_clean.cmake
build/CMakeFiles/_tensor_cuda.dir/cmake_device_link.o
build/CMakeFiles/_tensor_cuda.dir/compiler_depend.internal
build/CMakeFiles/_tensor_cuda.dir/compiler_depend.make
build/CMakeFiles/_tensor_cuda.dir/compiler_depend.ts
build/CMakeFiles/_tensor_cuda.dir/depend.make
build/CMakeFiles/_tensor_cuda.dir/deviceLinkLibs.rsp
build/CMakeFiles/_tensor_cuda.dir/deviceObjects1.rsp
build/CMakeFiles/_tensor_cuda.dir/dlink.txt
build/CMakeFiles/_tensor_cuda.dir/flags.make
build/CMakeFiles/_tensor_cuda.dir/includes_CUDA.rsp
build/CMakeFiles/_tensor_cuda.dir/link.txt
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/autograd.cpp.o
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/autograd.cpp.o.d
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/bindings.cpp.o
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/bindings.cpp.o.d
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/conv.cu.o
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/conv.cu.o.d
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/dda.cu.o
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/dda.cu.o.d
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/gemm_apa.cu.o
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/gemm_apa.cu.o.d
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/kernels.cu.o
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/kernels.cu.o.d
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/matmul.cu.o
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/matmul.cu.o.d
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/ops.cpp.o
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/ops.cpp.o.d
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/paint_bake.cu.o
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/paint_bake.cu.o.d
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/paint_inpaint.cu.o
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/paint_inpaint.cu.o.d
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/paint_raster.cu.o
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/paint_raster.cu.o.d
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/terrain.cu.o
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/terrain.cu.o.d
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/terrain_objects.cu.o
build/CMakeFiles/_tensor_cuda.dir/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/tensor_cuda/src/terrain_objects.cu.o.d
build/CMakeFiles/_tensor_cuda.dir/progress.make
build/CMakeFiles/cmake.check_cache
build/CMakeFiles/progress.marks
build/Makefile
build/_deps/pybind11-build/CMakeFiles/CMakeDirectoryInformation.cmake
build/_deps/pybind11-build/CMakeFiles/progress.marks
build/_deps/pybind11-build/Makefile
build/_deps/pybind11-build/cmake_install.cmake
build/cmake_install.cmake
build/compile_commands.json
build/lib/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
build.log
build_hook_broadcast.log
build_resume.log
census.py
cmake_source/CMakeLists.txt
configure.log
cpu_compiled/eb3_correct.cpp
cpu_compiled/eb3_early.cpp
cpu_compiled/eb4_correct.cpp
cpu_compiled/eb4_early.cpp
cpu_compiled/structural_results.txt
cpu_compiled/ticket_ownership.cpp
eb3.cuh.diff
eb4.cuh.diff
gpu_line.py
implement.py
initial_CHANGED_KERNELS.txt
initial_DRYRUN.json
initial_EXPECTED.json
initial_SASS_DIFF.json
instrument_0.log
instrument_1.log
instrument_2.log
instrument_3.log
lead_tie_fixture.py.txt
refs/after/tensor_cuda/__init__.py
refs/after/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
refs/after/tensor_cuda/apa_sp2.py
refs/after/tensor_cuda/functional.py
refs/after/tensor_cuda/nn.py
refs/after/tensor_cuda/optim.py
refs/after/tensor_cuda/quant.py
refs/after/tensor_cuda/quantization/__init__.py
refs/after/tensor_cuda/quantization/affine.py
refs/before/tensor_cuda/__init__.py
refs/before/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
refs/before/tensor_cuda/apa_sp2.py
refs/before/tensor_cuda/functional.py
refs/before/tensor_cuda/nn.py
refs/before/tensor_cuda/optim.py
refs/before/tensor_cuda/quant.py
refs/before/tensor_cuda/quantization/__init__.py
refs/before/tensor_cuda/quantization/affine.py
regression.py
regression_after.log
regression_before.log
sass_compare.py
source_before/tensor_cuda/src/apa_spo/eb3.cuh
source_before/tensor_cuda/src/apa_spo/eb4.cuh
source_before/tensor_cuda/src/apa_spo/eb4_tuning.cuh
source_before/tensor_cuda/src/bindings.cpp
verify.py
```

Scratch retained (every individual entry is in `FILES_WRITTEN.json`):

- `/tmp/apap1s3r2_build`: 0 files
- `/tmp/apap1s3r2_cpu_tests`: 30 files
- `/tmp/apap1s3r2_instruments`: 0 files
- `/tmp/apap1s3r2_regression_after`: 3230 files
- `/tmp/apap1s3r2_regression_before`: 3230 files
