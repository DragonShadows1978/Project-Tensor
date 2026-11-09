# ENG1 report: G1–G4 PASS (CPU round; 14 mutants pre-built; GPU lines staged for G5)

The lead copied this from the seat's final message (Claude Opus 5.5 Agent seat, 12:18–14:15 EST, 11-09); the
seat's own Write of this file was refused by its harness. The only change is that the full source diff is replaced by
a pointer: it is `artifacts/eng1/source.diff`, 280 lines. The lead checked that this file matches
`git diff 1680764d -- tensor_cuda/src tensor_cuda/tests/test_terrain_render.py` byte for byte. The ledger,
`artifacts/eng1/LEDGER.md`, holds every receipt.

**The lead also checked on disk (11-09):**
- G1's scope receipt lists 4 changed tracked files, which match the diff stat;
- `test_eng1.py` has sha256 `d3c9f579…` and 150 collected IDs;
- the 43 CPU IDs pass on the candidate with CUDA hidden: `pytest -k cpu`, 43 passed, 10.8 s.

**Run facts:**
- Order `orders/ENG1_ENGINE_DEFECTS.md`, sha256 c918656e…5ca0c.
- Dispatch commit `007137ded4b115919afc3f001626762c8f83cadb`; diff base `1680764d` (L80).
- CUDA hidden for every command, and no CUDA context was created. Builds ran at nice 19 with -j2.

## Premise corrections and seat decisions (all ledgered)

- **F1, topk (E4).** This branch had no refusal for k > last extent (`kernels.cu:5337-5352` at `1680764d`). On the
  baseline, a (2,0) input with k=1 returns a garbage (2,1) result and no error. The seat added PyTorch's single
  check, `k < 0 || k > S`, before any allocation. Mutant M10 covers it.
- **F2, float64 (E2).** The engine has no float64 dtype (`core.h:30`). `tc.tensor(dtype="float64")` silently builds
  float32 (`__init__.py:49-52`).
  - The "float64" test rows request float64, assert that the result is float32, and compare the uint32 views.
  - L09's uint64 comparison cannot exist on this engine.
- **F3, pooling (E5).** C++ `/` truncates toward zero, so a kernel of 4 on an input of 3 with stride 2 gave OH = 1.
  - `make_dims` now uses floor division. That equals truncation for every geometry that passes, so accepted outputs
    are unchanged.
  - Under floor division, "OH ≤ 0" means exactly "the kernel is larger than the padded input".
  - Mutant M13 restores truncation.
- **F4, terrain receipts (E6/L46).** All 17 `artifacts/wo*.ppm` receipts are tracked, and rewriting identical bytes
  leaves `git status` clean.
  - So the E6 record adds a `-uall` listing and a stat map: size, mtime_ns, and sha256 for each .ppm.
  - Confirmed on CPU: M14 wrote a .ppm byte-identical to the tracked one (sha256 618e9a71…).
- **F5, the terrain edit (E6).** Using `tmp_path` means changing each writer test's `def` line to take `tmp_path`.
  The edit is therefore 6 `def` lines plus 8 write lines. The `repo_root = …` lines stay in place, now unused.
- **F6, GN1 tooling.** GN1's committed A8 tooling lacks `program_binding.py` and A8's own staged script.
  - The adaptation keeps the L45 property (the parent receipt is written after the child exits) and drops GN1's
    contract-C conjunctions.
  - The originals are kept byte-identical in `tool/_gn1_a8_originals/` with SHA256SUMS.
  - `tool/TOOLING_CHANGES.diff` holds the 27 hunks of change.
- **D1, causal_softmax.** The existing `S < L` refusal still runs before the zero guard. So (B, L>0, 0) stays
  refused, and S = 0 gives an empty result only when L = 0.
- **D2, the sink variant.** `apa_blend_softmax_sink` has no host division by S (rows = B·H·L), so it is unchanged.
- **D3, C++ mutant lines and the C_COPY/C_UBSAN controls** deselect only `test_e6_terrain_writes_nothing`, which runs
  no C++ and costs about 5.5 minutes of CPU reference rendering per line. M14 runs the whole file.
- **D4, line (c).** The APA harness tests read repo files through `parents[2]`, so they cannot run from arm copies.
  - Line (c) instead runs the worktree's test files with each arm's package preloaded (GN1's method), plus each
    arm's own copy of the terrain file.
  - **Stated limitation:** tests that read repo files by path, or start subprocesses that import by path, see the
    candidate in both arms.
  - `test_selector_accuracy.py` is excluded: it runs `int(sys.argv[1])` at import, so pytest can't collect it.
- **M03, amended before any run of it.** The registered edit didn't compile (`known` is const). It was replaced by a
  single edit with the same effect: `known = numel_of(others) == 0 ? 1 : numel_of(others)`.
  - The original list is kept as `mutants/mutants.registered_v1.json`.
  - The failed attempt is kept as `mutants/M03_attempt1_compile_error/`.
- **A seat bug, fixed.** The killer-ID matcher treated `[…]` in test IDs as fnmatch character classes, so a real
  kill went uncounted. Brackets are now literal, and 8 constructed cases pass.

## Gates

**G1, scope: PASS.** Receipts: `G1_scope.json`, `source.diff`, `tracked_before.json`, `tracked_after.json`.
- Exactly 4 tracked files changed: `conv.cu`, `kernels.cu`, `ops.cpp` and `test_terrain_render.py`.
- Every hunk lies inside `make_dims`, `NDArray::reshape`, `apa_blend_softmax`, `topk_nd`, `causal_softmax`,
  `rms_norm`, `pow_scalar`, or the E6 `def` and write lines. There were 0 violations.
- 10,569 of 10,573 tracked files are byte-identical. The only new path outside the authorized outputs is
  `tensor_cuda/tests/test_eng1.py`.

**G2, CPU checks: PASS.**
- **References against central differences (`cpu_checks.json`):**
  - E2's reference against float64 central differences: all 5 exponents, relative error below 1e-8.
  - E3's VJP: all 10 class and case pairs, worst 4.2e-9 relative to scale.
- **Hand values** for E1, E2, E3 and E5 pass.
- **The L55 witness:** with x = 1.3671875, g = fl32(0.7) and p = 3, the result in order is 0x407b3880; reordered it
  is 0x407b387f.
- **Collection:** `test_eng1.py` collects 150 IDs on both arms, with no errors (`tool/ids_eng1.txt`).
- **Import with CUDA hidden:** both arms import. The baseline module is 73ededa0…9523e and the candidate module
  3e6d5805…c0ad53 (`g2_import_both_arms.log`).
- **The CPU subset (43 IDs):** all pass on the candidate. On the baseline, exactly the registered 29 fail
  (`g2_baseline_cpu_subset.xml`).
- **E6 on the real tree:** passes on CPU and writes nothing to `artifacts/` (335 s).

**G3, builds: PASS.**
- CUDA 12.6.85 was used for both arms, and configure and build exit 0 (`build/*_base.json`, `build/*_candidate.json`).
- The source maps are identical before and after each build: baseline 5a44463d…, candidate 56b7d0a4….
- The modules are pinned in `tensor_cuda/build-eng1-base/module/` and `tensor_cuda/build-eng1/module/`.
- The modules are stripped, so `nm -D` shows only `PyInit__tensor_cuda`. All 13 function symbols are defined in both
  arms' object files. The 7 new refusal strings appear only in the candidate module (`G3_symbols.json`).

**G4, staging: PASS.**
- The staged script `tool/STAGED_GPU_COMMANDS.sh` (sha256 eee24bd6…) has 22 lines with bounds; `gpu_lines.json`
  lists them, and `mutants/mutants.json` lists the mutants.
- The seal `tool/staging_seal.json` (sha256 d1d55f41…) covers 18 arms, with none pending.
- **The audit-mode line** (`tool/audit_final/`, stand-in runner, CUDA hidden):
  - the script and child both exit 0, with no timeout, and 43 IDs pass;
  - the runner row says `ran 0`;
  - the parent receipt was written after the child exited (19:14:17.885, then .949);
  - the scorer refuses an audit root as GPU acceptance, as designed.
- **The L46 record** parses on the baseline's collection only (`l46_baseline_collect_record.json`).

## Source diff

`artifacts/eng1/source.diff` (280 lines; command and check above). In summary:
- **E1:** `NDArray::reshape` infers a single `-1` and refuses with distinct messages: two or more `-1`, another
  negative extent, a `-1` with a zero product, or a non-integer extent.
- **E2:** `pow_scalar`'s backward computes in float32 for float16 and bfloat16 inputs, rounding once. The float32
  path is the unchanged line.
- **E4:**
  - `apa_blend_softmax`, `causal_softmax` and `rms_norm` return an empty result before the division, with no launch;
  - `topk_nd` refuses k outside [0, S] and returns empty values and indices for S = 0, k = 0 or no rows.
- **E5:** `make_dims` uses floor division and refuses OH ≤ 0 or OW ≤ 0, with a message naming every extent.
- **E6:** the 8 `.ppm` writes move to `tmp_path`.

**New file:** `tensor_cuda/tests/test_eng1.py`, 685 lines, sha256
`d3c9f579dbf51e86af7022c0c8447bcc63426df4170237f047a336ac97ed5f38`.

## Test IDs, tolerances and derivations

All 150 IDs are in `tool/ids_eng1.txt`.
- IDs containing "cpu" use only zero-byte CPU tensors and run with CUDA hidden. Every other ID needs the GPU.
- Unit roundoff u: float16 2⁻¹¹, bfloat16 2⁻⁸, float32 2⁻²⁴.

**E1 (27 IDs): exact equality.**
- Reshape only changes metadata, and integers below 2²⁴ are exact in float32.
- Refusals are checked by their exact message.
- `test_e1_view_shares_storage` writes through a `-1` view and checks that `data_ptr` is equal.

**E2 (31 IDs).**
- `test_e2_pow_grad[{float16,bfloat16,float32,float64}-p{-1.5,-0.5,+0.5,+2,+3}]` (20 IDs).
  - The upstream gradient w is a power of two, sized so that the true gradient fits float16.
  - float16 drops only the points where no weight ≥ 2⁻²⁴ can make the true gradient representable: p = -1.5 at
    x = 1e-5 and at x = 3e-5.
- `test_e2_pow_grad_bits[{float32,float64}-p…]` (10 IDs): exact uint32 equality against the same engine ops
  recomposed, with upstream values that are not powers of two and include the L55 witness.
- **The tolerance:**
  - CUDA `powf` is accurate to 4 ulp (2⁻²¹), and two more float32 roundings add 2⁻²³. That gives float32 rtol 2⁻²⁰,
    with atol 2⁻¹⁵⁰.
  - The half dtypes have rtol = u_d + 2⁻²⁰.
  - Their atol is half the subnormal spacing: 2⁻²⁵ for float16 and 2⁻¹³⁴ for bfloat16.

**E3 census (40 IDs; measurement only).**
- IDs: `{LayerNorm,RMSNorm,BatchNorm1D,InstanceNorm2D,GroupNorm}-{float16,bfloat16}-{regular,near_eps}-{forward,backward}`.
- `near_eps` makes the variance equal the class's eps. For RMSNorm, the mean square equals eps.
- **The tolerance:** tol = (n_d·u_d + n_32·2⁻²⁴)·max|ref| + half the subnormal spacing. Here n counts the roundings
  on the longest path, and the backward counts two per forward op.
  - LN, BN and IN: forward (11, 0), backward (22, 0).
  - RMSNorm: forward (7, 0), backward (14, 0).
  - GroupNorm: forward (1, 11), backward (1, 22), because it upcasts.

**E4 (23 IDs).**
- Shapes, dtypes and messages are checked exactly.
- Each case runs in a child process, so a crash counts as a failure.
- `test_e4_topk_values` checks values and indices exactly against NumPy's argsort.

**E5 (28 IDs).**
- `test_e5_refuse_cpu[{im2col,col2im,avg_pool2d,max_pool2d}-{both_negative,oh_zero,ow_only,trunc_gap,padded}]`
  checks the exact messages.
- `test_e5_valid_pool[k2s2p0,k3s1p0,k3s2p1]`:
  - the max-pool forward is exact;
  - the average-pool forward and backward and the max-pool backward use rtol 2⁻²⁰ (16·2⁻²⁴).
- `test_e5_valid_im2col[...]` is exact.

**E6 (1 ID):** `test_e6_terrain_writes_nothing` compares the git status records and the stat map exactly.

**The line (c) census:** 2,133 regression IDs from 49 files, identical on both arms (`tool/regression_files.json`,
`ids_regression.txt`), and 53 terrain IDs (`ids_terrain.txt`).

## Mutants (all pre-built; none to build at G5)

| ID | Decision | Module | Killers | CPU result |
|---|---|---|---|---|
| M01 | E1: two -1 accepted | 77d33c94… | refuse[two_neg1], [three_neg1], _empty_cpu[two_neg1] | killed (DID NOT RAISE) |
| M02 | E1: inferred extent off by one | 1c5de898… | infer[*], _empty_cpu[*], view_shares_storage | killed by all 4 `_empty_cpu` IDs |
| M03 | E1: -1 with a zero product accepted (amended) | 11d2e058… | refuse[zero_product], _empty_cpu[zero_product] | killed |
| M04 | E1: another negative extent accepted | 1dfda7b6… | [other_negative], [negative_with_neg1], _empty_cpu[other_negative] | killed |
| M05 | E2: half path in the input dtype | 794172a4… | pow_grad[float16-p-1.5], [float16-p-0.5] | GPU only; emulation predicts 5 IDs fail |
| M06 | E2: exponent p instead of p - 1 | 846b2c95… | pow_grad[float16-*], [bfloat16-*] | GPU only; emulation predicts 10 fail |
| M07 | E2: cast before the multiply | 24f3a7fc… | [float16-p-1.5], [float16-p-0.5] | GPU only; emulation predicts both fail |
| M08 | E4: empty result with the wrong shape | 93010ff2… | e4_empty[rms_norm-…] (4) | killed: (0,) returned where (2,0) and (3,1,0) are expected |
| M09 | E4: guard after the division (UBSan) | c1f60cd9… | e4_empty[apa_blend_softmax-last0-cpu/cuda] | killed: "kernels.cu:3861:29: runtime error: division by zero"; C_UBSAN control passes |
| M10 | E4 (F1): topk accepts k > S | e993b679… | e4_refuse[topk-k1-last0-*], [topk-k4-last3-cuda] | killed |
| M11 | E5: >= 0 instead of > 0 | 30f90bb9… | e5_refuse_cpu[*-oh_zero], [*-trunc_gap] | killed (8 IDs) |
| M12 | E5: the check on OH only | 423ce3f6… | [*-ow_only] | killed (4 IDs) |
| M13 | E5 (F3): truncation restored | bb73241b… | [*-trunc_gap] | killed (4 IDs) |
| M14 | E6: one write left in the source tree | candidate module, terrain copy edited | e6_terrain_writes_nothing | killed: `-uall` gained the .ppm |

- **The controls:** C_COPY is the unmutated scratch build (b64323fd…), and C_UBSAN is the candidate built with UBSan
  (3ce824c4…). Both pass the CPU subset.
- **Mutant builds** are incremental, in a scratch tree, with the arms' exact CMake arguments. Their bytes differ from
  the worktree candidate's because the source path is embedded, and C_COPY controls for that.

## Staged lines

`artifacts/eng1/tool/STAGED_GPU_COMMANDS.sh`. Every line's expected parent exit is 0.

```
RUNS=<abs path under artifacts/eng1/tool/gpu_runs/...> TMPDIR=/tmp/eng1_g5 CUDA_VISIBLE_DEVICES=0 timeout 1190s bash STAGED_GPU_COMMANDS.sh LABEL
```

Run the labels in `gpu_lines.json` order, all with the same RUNS. Then run `score_gpu.py` with the same RUNS and CUDA
hidden; it is expected to exit 0.

| Label | Expected child exit | Bounds (inner / `--max` / outer) |
|---|---|---|
| baseline_eng1 | 1 | 900 s / 960 s / 990 s |
| candidate_eng1 | 0 | 900 / 960 / 990 |
| baseline_regression | 0 or 1 | 1050 / 1110 / 1140 |
| candidate_regression | 0 or 1 | 1050 / 1110 / 1140 |
| baseline_terrain | 0 or 1 | 1050 / 1110 / 1140 |
| candidate_terrain | 0 or 1 | 1050 / 1110 / 1140 |
| C_COPY | 0 | 420 / 480 / 510 |
| C_UBSAN | 0 | 900 / 960 / 990 |
| M01..M13, each its own label | 1 | 420 / 480 / 510 |
| M14 | 1 | 900 / 960 / 990 |

- **Line (a):** the registered failures are 54 specific IDs (`tool/baseline_expected_failures.json`): E1 24, E2 5,
  E4 4 topk refusals, E5 20, E6 1.
- **Line (c):** passes when no ID that passes on the baseline fails on the candidate. The terrain lines show writes
  on the baseline arm and none on the candidate.

## Predictions

- **P201: HELD.** No Python check intervenes:
  - `tc.Tensor` is `_C.Tensor` (`__init__.py:24`), and the binding (`bindings.cpp:1879`) passes `-1` straight to C++.
  - **L82:** all 30 engine-side reshape callers pass explicit shapes, so their path is unchanged. The other 9 grep
    hits are NumPy reshapes.
- **P202: HELD on CPU emulation and by source construction.** The device confirmation is G5 lines (a) and (b).
  - The emulated baseline in float16 gives inf for p = -0.5 at every x ≤ 1e-4.
  - The candidate is finite and within tolerance on all 20 IDs.
  - Float32 bit identity holds by construction.
- **P203:** the lead's, at G5.

## Prior art

All unverified; the lead checks them. The same notes are at each code site and in the ledger.
- **E1:** PyTorch's reshape and view `-1` inference (ATen `InferSize.h`) and NumPy's single unknown dimension.
- **E2:** Micikevicius et al. (2018), "Mixed Precision Training"; PyTorch's opmath upcast in the pow backward.
- **E4:** PyTorch's topk range check and its empty results; softmax and the norms on empty input.
- **E5:** PyTorch's `pooling_output_shape` (div_rtn) and its "Output size is too small".
- **The tests:**
  - Higham (2002);
  - IEEE 754;
  - the CUDA `powf` 4-ulp bound;
  - the normalization gradients (Ioffe and Szegedy 2015; Ba, Kiros and Hinton 2016; Zhang and Sennrich 2019; Wu and
    He 2018);
  - central differences (Fornberg 1988).
- **Mutation:** DeMillo, Lipton and Sayward (1978), plus UBSan's `-fsanitize=integer-divide-by-zero`.
- **The tooling:** GN1 A8, GNU timeout, POSIX link(2), SHA-256 and git porcelain.

## Files and items to bin

- **Tracked edits:** `kernels.cu`, `ops.cpp`, `conv.cu`, `test_terrain_render.py`.
- **New:** `test_eng1.py`.
- **Candidate package module** (gitignored): `tensor_cuda/tensor_cuda/_tensor_cuda…so`, 3e6d5805….
- **`artifacts/eng1/`:** 420 files, 315 MB, all listed with sha256 in `artifacts/eng1/files_written.json`. The
  `.so` files are gitignored: 173 MB for the arms and 144 MB for the mutants.
- **The build directories,** 27 MB each: `tensor_cuda/build-eng1/` and `tensor_cuda/build-eng1-base/`.
- **To bin (the lead):**
  - `tool/audit_prelim/`;
  - `tool/census_step.out` and `census_step2.out`;
  - `mutants/M03_attempt1_compile_error/`;
  - `tool/arms/M14/artifacts/wo8a_blocky.ppm`;
  - after G5, the .ppm files that the baseline_eng1, baseline_terrain and M14 lines write into
    `tool/arms/<arm>/artifacts/`, by design.
  - /tmp scratch is exempt.
