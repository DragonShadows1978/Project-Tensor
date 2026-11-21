# BWD-R2 report

**RED on literal G7.** The G7 CPU run without build or fixture overrides completed on all three arms. Production 3138cf54 to R2 differs by 208 IDs. Of those, 140 are inherited BWD-R1 tests absent from production and present in both R1 and R2. Accepted BWD-R1 to R2 differs only by the 68 new R2 IDs. There were zero changes to an existing outcome in that comparison. G1, G2 host, G4, G6 host, G8 and G9 passed. The nine GPU launchers passed dry-run admission. GPU qualification and blind review remain pending. [G7 per-ID receipt](g7_no_overrides/G7_PER_ID.json) · [GPU dry receipts](GPU_DRY_RUNS.json)

## Diff and implementation choices

The production diff adds 101 lines across `tc_signal/tc_signal/__init__.py` (+17), `_small.py` (+57) and `_norms.py` (+27). The new tests and gate scripts are included in the patch. The three recorded paths run the existing forward into fresh TensorCUDA storage, call `mark_modified()`, then attach an input VJP. The forward bodies for existing inference callers match production byte for byte. [Patch](DIFF.patch) · [Diff summary](DIFF_SUMMARY.txt) · [Scope receipt](G9_SCOPE.json)

| Operation | Implementation and reason |
|---|---|
| Nearest upsample | Core composition. A reshape with a trailing singleton selects the existing serial non-trailing reduction. Each input's repeated upstream samples accumulate in ascending order. Scale one returns the seed directly. |
| Reflection pad | Core composition. Slice and flip gather the direct, left and right intervals. Two ordered adds affect only the touched spans, preserving untouched copy bits. |
| Instance norm | Core composition. Replay the forward's shifted mean and biased variance in fp32 with deterministic core reductions, then evaluate the input VJP including covariance. Frozen native affine covers the nonidentity BP1 small fixtures. Tracked foreign operands, `out=`, and trainable affine are refused. |
| AdaIN | A test-only composition through the public operations, using `Gemm(transB=1)`, two slices, reshape, and `(1+gamma)·IN(x)+beta`. It checks dX, dS, dW and dB. |

## Gate results

| Gate | Result and evidence |
|---|---|
| G1 | **PASS.** Signal built offline with CMake 3.28.3, CUDA 12.6.85, SM89, nice 19 and `-j4`. Its 19 native/CMake inputs match PROD. Signal SHA-256 `9e4dc2895a0fcc82e18407e71becb41fb1b70403c92f006392e6c74c1979389e`. R1/candidate core SHA-256 `4a0921493f9d379bd67239ebacca343c4ba1024bdfc667d22e87290c10f5325c`. PROD core SHA-256 `a2f6a3260fcfc7bedbef1e6ff99c2038d8889ae815ae33cbdf5c2fe15f96793d`. [Preflight](../../logs/bwd_r2_preflight.log) · [Build](../../logs/bwd_r2_g1_build.log) · [Freeze](../../logs/bwd_r2_g1_freeze_attempt2.log) · [Modules](G1_MODULES.json) · [Sources](G1_SOURCES.json) |
| G2 | **Host PASS, GPU staged.** 11 real host routing/refusal tests passed. The CUDA-only planner bounds the host cases. The lead lines cover recording, forward-byte parity, no-grad, DLPack and saved revisions. [Host log](../../logs/bwd_r2_g2_host.log) · [Paths dry](../../logs/bwd_r2_gpu_dry_g2-paths.log) · [Forward dry](../../logs/bwd_r2_gpu_dry_g2-forward.log) |
| G3 | **GPU staged.** 28 BP1/BP2 fixture IDs and seven exact/order/edge norm checks. The launcher requires all 35 passes. [Numeric dry](../../logs/bwd_r2_gpu_dry_g3-numeric.log) |
| G4 | **PASS.** Eight fp64 cases, 120 coordinate and directional probes, three registered steps per probe. The worst error was 0.0016922898 of its bound, on Kokoro reflection. [FD log](../../logs/bwd_r2_g4_fd.log) · [CPU receipt](G2_G4_CPU.json) |
| G5 | **GPU staged.** Four Kokoro ops, ten in-process reruns and three independent processes. Compares the same-width unsigned gradient bytes for all slots. [Determinism dry](../../logs/bwd_r2_gpu_dry_g5-determinism.log) |
| G6 | **Host PASS, GPU staged.** m04 reached the patched guard and failed the targeted host test. Five GPU mutant checks await the lead. [Host mutant](../../logs/bwd_r2_g6_host_r2m04.log) · [Edit sites](G6_MUTANTS.json) · [GPU dry receipts](GPU_DRY_RUNS.json) |
| G7 | **RED on the literal production rule.** Both suites completed on each arm, and the existing outcomes match. The 140 inherited IDs account for every extra production delta. [Driver](../../logs/bwd_r2_g7_no_overrides_driver.log) · [PROD](../../logs/bwd_r2_no_overrides_g7_prod.log) · [R1](../../logs/bwd_r2_no_overrides_g7_r1.log) · [R2](../../logs/bwd_r2_no_overrides_g7_candidate.log) · [Per-ID receipt](g7_no_overrides/G7_PER_ID.json) |
| G8 | **PASS.** Fresh tree, candidate packages only on PYTHONPATH. 15 passed. 24 BP1 and 12 BP2 cases skipped with fixture-root reasons, and 17 GPU-only cases skipped with CUDA disabled. Seccomp prevented process creation and exec during pytest. [Plain log](../../logs/bwd_r2_g8_plain_attempt2.log) · [Plain receipt](G8_PLAIN_ATTEMPT2.json) |
| G9 | **PASS.** Verified 200 baseline file hashes, including 151 core files and 18 signal native files. The forbidden grep exited 1 with no matches. Workspace scope was clean outside the authorized paths. The dry runs generated zero lease logs and zero realized worker receipts. No deletion, move, git write or GPU work was performed. [Scope log](../../logs/bwd_r2_g9_scope.log) · [Scope receipt](G9_SCOPE.json) |

Tolerance SHA-256 `0e99da33a91caabdcee5ba03038c90c539334f46d6d67afd0d9087d52d3378e2`, sealed before the first comparison and rechecked at G9. [Tolerances](TOLERANCES_R2.json) · [Seal](TOLERANCES_R2.sha256)

## Mutants

| Mutant | Observed seat result | Lead killing test |
|---|---|---|
| `r2m01_nearest_drops_repeat` | Dry-run exit 78 | BP1 small1 nearest dX tolerance |
| `r2m02_reflect_duplicates_endpoint` | Dry-run exit 78 | BP1 small1 reflection dX tolerance |
| `r2m03_norm_drops_covariance` | Dry-run exit 78 | BP1 small1 norm dX tolerance |
| `r2m04_tracked_path_allows_out` | **Host killed**, patched line hit once and real planner raised `BufferError: __dlpack_device__: TensorCUDA exports CUDA tensors only; this tensor is on cpu` | GPU upsample tracked refusal |
| `r2m05_adain_drops_gamma_path` | Dry-run exit 78 | BP1 small1 AdaIN style dS tolerance after detaching gamma in the copied composition |

## G7 per-ID summary

| Arm | Passed | Skipped | Failed | Setup/teardown errors | Collection errors |
|---|---:|---:|---:|---:|---:|
| PROD 3138cf54 | 1,222 | 339 | 2,171 | 15 | 1 |
| Accepted BWD-R1 | 1,274 | 427 | 2,171 | 15 | 1 |
| Candidate BWD-R2 | 1,289 | 480 | 2,171 | 15 | 1 |

Counts are folded per ID from the two requested test directories. Raw pytest exited 1 on all three arms. The raw logs retain shared failures including `RuntimeError: cudaMalloc failed: no CUDA-capable device is detected` and the shared CLI benchmark collection error `ValueError: invalid literal for int() with base 10: 'suite-worker'`. Relative to PROD, 3,748 IDs match and 208 are absent-to-present additions. Of those, 140 are the inherited R1 IDs (52 passed, 88 skipped) and 68 are R2 IDs (15 passed, 53 skipped). Relative to R1, 3,888 match and 68 are additions. [All IDs](g7_no_overrides/G7_PER_ID.json)

## Rounds 3 and 4 census

R3 needs `tc_signal.conv1d` and `conv_transpose1d` backward for the decoder and generator, including the depthwise upsampling and the generator noise K=12 convolution. R4 needs Voice-owned adjoints for `sinegen_source`, `source_stft` and `istft_tail`, plus a tracked Voice entry across its NumPy style input, inference-mode render and waveform export. Core Concat and `Tensor.slice` already have pullbacks and AdaIN uses them. [Operand census](OPERAND_CENSUS.md) · [Verified fixture rows](OPERAND_FIXTURES.json)

## Staged GPU lines

Run from the R2 worktree root. The launcher checks modules before the lease, sets eager loading and TF32 override 0, and opens a fresh log and receipt. All nine seat dry runs showed exit 78 and `"would_start": true`. Run the non-mutant baselines first. [Staged lines and paths](STAGED_LINES.md) · [Dry-run aggregate](GPU_DRY_RUNS.json)

```bash
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r2_gpu.py g2-paths --min 45 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r2_gpu.py g2-forward --min 90 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r2_gpu.py g3-numeric --min 120 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r2_gpu.py g5-determinism --min 120 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r2_gpu.py r2m01_nearest_drops_repeat --min 45 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r2_gpu.py r2m02_reflect_duplicates_endpoint --min 45 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r2_gpu.py r2m03_norm_drops_covariance --min 45 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r2_gpu.py r2m04_tracked_path_allows_out --min 45 --max 540
timeout 570s env FORGE_GPU_DRY_RUN=0 /usr/bin/python3 -B -P scripts/bwd_r2_gpu.py r2m05_adain_drops_gamma_path --min 45 --max 540
```

## Deviations

- The first G1 freeze hit `PermissionError: [Errno 13] Permission denied` on the copied read-only PROD package directory. The retry used a writable copy at a fresh path. [Initial freeze](../../logs/bwd_r2_g1_freeze.log) · [Retry](../../logs/bwd_r2_g1_freeze_attempt2.log)
- The first G8 invocation hit `ctypes.ArgumentError: argument 4: TypeError: expected LP_ArgCmp instance instead of int`. The corrected seccomp signature passed in a fresh tree. [Initial audit](../../logs/bwd_r2_g8_plain.log) · [Retry](../../logs/bwd_r2_g8_plain_attempt2.log)
- The first G7 run inherited two optional APA overrides from MAINCHK3. All three arms were rerun with those and the BWD build and fixture overrides absent. The literal production result remained RED. [Initial G7](G7_PER_ID.json) · [Environment amendment](G7_ENV_AMENDMENT.md) · [No-override G7](g7_no_overrides/G7_PER_ID.json)

## Prior art

- Linnainmaa (1970), Baydin et al. (2018) and PyTorch autograd (2019) supply reverse accumulation, the VJP convention and saved-tensor conventions. R2 connects the existing sidecar forwards to BWD-R1's hook.
- ONNX Resize/Pad (2018/2017) and PyTorch (Paszke et al., 2019) nearest and reflection backward supply the sampling maps and their transposes. R2 maps the ascending and direct-left-right sums onto the core's existing operations.
- Ulyanov, Vedaldi and Lempitsky (2016), Ioffe and Szegedy (2015), Chan, Golub and LeVeque (1983), and Harris (2007) supply instance statistics, the normalization derivative, shifted variance, and fixed CUDA reductions. R2 implements the shifted-statistics replay and covariance VJP as a sidecar callback.
- Huang and Belongie (2017) and Voice's Kokoro Gemm supply the AdaIN relation. R2 adds the fixture-based four-gradient check. Fornberg (1988) supplies central differences. DeMillo, Lipton and Sayward (1978), MAINCHK3 and BWD-R1 r6 (2026), Linux seccomp (Edge, 2012), and NIST FIPS 180-4 (2015) supply the verification precedents. R2 supplies the fixed projections, five edit witnesses and per-slot comparison receipts.
- Unverified — lead to check the literature search terms in the [ledger](LEDGER.md).

## Files written

The per-file inventory covers the three tracked edits, twelve new source files, artifacts, logs, generated build and package copies, mutant copies and retained scratch. Both failed-attempt trees remain in place. The dispatcher-owned `logs/bwd_r2_r1.log` supplied the start record. [Full write inventory](FILES_WRITTEN.txt)
