# APA-PF2-F1 report

CPU/build and staging complete; ACCEPT remains pending the lead's GPU gates.
No GPU was used, no git command was invoked, and nothing was explicitly deleted.

**Diff summary**

- F01: use `denom != 0.f`; retain reciprocal then multiply.
- F02: host/device int64 index helper handles both tail additions; no new refusal.
- F05: both stored-norm reads convert every non-finite value to quiet NaN before the existing multiply.
- F04: bind the same exact K, change only visible rows, assert the visible-rows message and zero spied calls, launches and allocations; retain the separate exact-K row.
- Add GPU-skipping non-finite tests, isolated build/mutant receipts and staged harnesses. Existing product edits are confined to packed_prefill.cuh; the only existing harness edit is the F04 row. CHANGES.diff contains the exact diff.

**Module SHA256s**

| Build | Module SHA256 |
|---|---|
| PF, read-only baseline | `c3c27f8651d82390aac89d62ef704e20dade16d38e9ac8f6cf0e3fb456f4f831` |
| F1 | `641a3139181cdee3e800f62afad08c0881b3018c26e57d953fa6c13413f7cbe7` |
| pf1m01_sink_guard_positive | `a5e000602e2c82bc2a98e05d39d99a091327cae9afb45f98bdfc23dd9b2e51cb` |
| pf1m02_norm_finite_only | `24ac67e15f421432a74ae87440c4508ac30c23fe662885294d5cedb5d47e450a` |
| pf1m03_index_int32 | `faedb4e24d0852794aa369a9ece78353cd10387778a30d171d7a759fc69a8147` |

F1 and all three mutant builds passed. Each mutant incrementally rebuilt only kernels.cu.o. Receipts: build_*/BUILD.json; source hashes, command exits, resource minima and log hashes are recorded there. Build PF was read-only.

**CPU results**

- G3 PASS: 441 actual-helper checks, including 64 boundary lanes and 128 additions at INT_MAX and INT_MAX-31, under UBSan; INDEX_F1.json.
- G5 CPU PASS: pf1m03_index_int32 killed, exit 1; F1 control clean. UBSan: `signed integer overflow: 2147483647 + 32 cannot be represented in type 'int'`.
- G7 PASS: 21,965 files outside the writable set unchanged; zero added/missing files there; no git commands, explicit deletes or real GPU leases. Evidence is a worktree stat census plus source/module hashes and command receipts, not an audit of unrelated worktrees.
- CPU pytest: 1 passed, 32 skipped with CUDA_VISIBLE_DEVICES empty; PYTEST_CPU.json.
- G6 SASS CPU PASS: 813 kernels; 781 equal, 32 changed, all spq_pf::prefill; zero added/removed. PF2 mappings only; SASS_COMPARISON.json.
- Final launcher dry runs: 58/58 exit 78 before the lease; DRY_RUNS.json. Every final line is pinned and carries an explicit minimum and --max 540.

Builds used PF flags, CUDA 12.6, -j3 and nice 19. Observed minima: MemAvailable 31.49 GB and disk free 185.79 GB; no 12 GB / 40 GB rail hit.

**COVERAGE.json summary**

28 reachable combinations registered and staged; 0 executed, 0 passed by this seat. Of 32 compiled combinations, the 4 CAP64/SEQ=false combinations are unreachable. Cases include both dtypes, both diagnostics values and every reachable CAP/SEQ pair. No-mask outputs are also compared with a paired diagnostic call; diagnostic masks must equal packed decode. Inherited per-dtype tau: fp32 0.00015640330057766595; bf16 0.013514695825288718. GPU completion writes COVERAGE_LEAD.json.

**Mutants**

| Mutant | Incremental build | Named killing test | Result |
|---|---|---|---|
| pf1m01_sink_guard_positive | PASS | G2 sink, 16 cases; clean F1 first | STAGED |
| pf1m02_norm_finite_only | PASS | G2 norm, 16 cases; clean F1 first | STAGED |
| pf1m03_index_int32 | PASS | G3 actual helper at INT_MAX boundaries | KILLED on CPU |

**Residuals and claim limits**

G1 finite bit identity, G2 propagation, G4 execution coverage, GPU mutant kills and G6 refusal/memory behavior are pending. They are not claimed passed. The unchanged raw-Inf decode may itself hide a norm (source reasoning); REV1's norm reproduction did not compare decode. G2 retains the literal raw-Inf decode row-set assertion and will fail with raw_inf_decode_row_sets if that discrepancy is observed. The NaN-norm decode control is also checked. Unchanged decode is outside scope and is not claimed fixed. No finite-arithmetic, margin, route, Python surface, speed, model-quality or promotion claim is broadened. A separate Claude review still follows.

**STAGED_LINES.md in full**

# STAGED_LINES.md

LEAD ONLY. Run from the worktree root below. This seat ran dry runs only.
There are 58 lines: G1_000..G1_045 cover all 4,848 finite cases once;
G2 has 16 sink and 16 norm cases; G4 covers 28 reachable combinations;
G5 tests both GPU mutants after clean F1 controls; G6 has 26 refusal and
24 memory rows. Exact indices, arguments and module pins: LINES.json.

Every launcher checks module, source, script, registration and shim hashes
before the lease. It sets PATH, LANG/LC_ALL, CUDA 12.6 LD_LIBRARY_PATH,
PYTHONPATH empty, TMPDIR=/tmp/pf2f1_gpu, PYTHONDONTWRITEBYTECODE=1,
TC_APA_SP=1, APA_PF2F1_LEASE=1 and all six BLAS/OpenMP thread variables to 1.
Only HOME, USER, LOGNAME, XDG_RUNTIME_DIR and DBUS_SESSION_BUS_ADDRESS are
inherited when present. CPU prechecks/dry runs have CUDA_VISIBLE_DEVICES
empty; real payloads explicitly use CUDA_VISIBLE_DEVICES=0. Torch intra-op
and inter-op thread counts are 1 where torch is used. Refusal/memory lines
also set LD_PRELOAD=/tmp/pf2f1_shim/libpf2count.so, checked against its pin.
Full per-line environments: gpu/*_dry.start.json and the launcher source.

Every GPU invocation uses forge-gpu-run --min <n> --max 540, inner timeout
500 seconds, nice -n 19 and python3 -B -P. G1 workers have a 460-second limit.
The foreground 600-second pauses below preserve the cooldown between lines.
Each launcher records the forge exit and log SHA in gpu/*_real.exit.json.

```bash
set -eu
cd /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1pf2
export CUDA_VISIBLE_DEVICES= TMPDIR=/tmp/pf2f1_lead PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
timeout 10s mkdir -p "$TMPDIR"
run() {
  timeout 660s env FORGE_GPU_DRY_RUN=0 python3 -B -P scripts/apa_pf2f1_line.py "$1" --lead --min "$2" --max 540
  timeout 610s sleep 600
}
for n in 000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045; do
  run "G1_$n" 500
done
run G2_sink 30
run G2_norm 30
run G4_coverage 120
run G5_sink_mutant 60
run G5_norm_mutant 60
run G6_refusals 60
run G6_memory_qwen3_1.7b_2048_2048 500
run G6_memory_qwen3_1.7b_8192_8192 500
run G6_memory_qwen3_1.7b_32768_2048 500
run G6_memory_qwen3_4b_2048_2048 500
run G6_memory_qwen3_4b_8192_8192 500
run G6_memory_qwen3_4b_32768_2048 500
timeout 120s python3 -B -P scripts/apa_pf2f1_lead_summary.py
```

Exit 75/76 is no experiment; 124 is timeout; none passes a gate. Stop on any
nonzero exit. Outputs use exclusive creation: retain prior attempt files by
moving them within artifacts/apa_pf2f1 before an authorized rerun; never delete
or overwrite a measurement. A measured timeout may require another registered
split; this seat measured no GPU timeout. No gate is narrowed.

All 58 final dry runs returned 78 before the lease (DRY_RUNS.json). The CPU
index mutant is already killed and SASS is checked; neither needs a GPU line.
GPU coverage remains 0/28 until G4_coverage runs; its result is COVERAGE_LEAD.json.
The CPU summary writes ACCEPTANCE_LEAD.json only after all receipts pass.
A separate Claude review still follows.

**Deviations**

- D01: initial build failed at both norm sites: `error: identifier "CUDART_NAN_F" is undefined`. Replaced it with the existing packed quantizer's quiet-NaN intrinsic. Retained failed receipt/log; added build resume entry (script repair 1/3).
- D02: retained the 491-line draft; execution amendment 1 replaced repeated per-case imports with persistent isolated sequential workers and revised splits (script repair 2/3). All cases, shapes and gates remain fixed.
- D03: apply_patch rejected a duplicate-file operation and later an invalid hunk line while authoring the report generator; neither call changed files. Reissued valid patches (tool slips).
- D04: retained first-pass min-30 dry receipts; execution amendment 2 raised minima to 500/120/60/30 seconds by payload. Repeated all 58 final dry runs (script repair 3/3; allowance exhausted).
- D05: F1 resume inherited the failed attempt's finished_utc field. BUILD_F1_COMPLETION.json separately records successful-log and final-receipt timestamps/hashes. No source or gate changed.

## Prior art

IEEE 754-2008 propagation and Project-Tensor P1-F1/F2 (2026) zero-only guard and P1-F3 (2026) NaN-norm sentinel are reused. The index helper uses ordinary C++17 widening / CERT INT32-C; unverified - lead to check those terms. F04 uses isolated xUnit preconditions (Beck 2002; unverified - lead to check). Builds reuse PF2/F2 plumbing, CMake (Kitware 2000) and make (Feldman 1979). Harnesses reuse PF2's grid, oracle, spies, memory counters, SASS mappings and fixed-width bit comparisons. Persistent arm isolation and pipe request-response are ordinary IPC. No prior art known to me for the exact conservative batching work estimate; unverified - lead to check "bounded batching kernel work budget". Packed scoring/refinement/softmax remain inherited: TurboQuant (Zandieh et al. 2025), BLASST (Yuan et al. 2025), Milakov/Gimelshein (2018), FlashAttention (Dao et al. 2022); literature references are unverified leads from existing source comments. Ours is the scoped guard/accessor/index integration and receipt plumbing; no new numerical algorithm.

**Every file written**

The complete retained-file enumeration, including copied packages, mutant source copies, generated build files, logs, prior drafts and /tmp scratch, is FILES_WRITTEN.txt. FILES_WRITTEN.json records their sizes and SHA256s (its own hash is excluded to avoid self-reference). No files were deleted or moved to DELETE_ME. Dispatcher capture logs/apa_pf2f1_r1.log is externally managed and remains live; this seat did not edit it.

Edited existing files:

- tensor_cuda/src/apa_spo/packed_prefill.cuh
- scripts/apa_p1pf2_g45.py (F04 row only)

New test:

- tensor_cuda/tests/test_apa_pf2_nonfinite.py

New scripts:

- scripts/apa_pf2f1_audit.py
- scripts/apa_pf2f1_build.py
- scripts/apa_pf2f1_common.py
- scripts/apa_pf2f1_g45.py
- scripts/apa_pf2f1_gpu.py
- scripts/apa_pf2f1_index.py
- scripts/apa_pf2f1_lead_summary.py
- scripts/apa_pf2f1_line.py
- scripts/apa_pf2f1_materialize.py
- scripts/apa_pf2f1_nonfinite.py
- scripts/apa_pf2f1_prepare.py
- scripts/apa_pf2f1_reference.py
- scripts/apa_pf2f1_sass.py
- scripts/apa_pf2f1_sass_reference.py
- scripts/apa_pf2f1_stage.py

All remaining written files are under artifacts/apa_pf2f1/ and /tmp/pf2f1_* and are individually enumerated in FILES_WRITTEN.txt. They include the failed build, all three incremental builds, all staged packages, both prior staging drafts, both sets of 58 dry-run logs, the exact diff, registration, ledger, census, scope and test receipts, and this report.
