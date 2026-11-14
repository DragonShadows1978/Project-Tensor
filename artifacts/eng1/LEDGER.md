# ENG1 ledger (seat: Claude Opus 5.5, Agent-tool seat, CPU only)

Order: `orders/ENG1_ENGINE_DEFECTS.md`, dispatch commit `007137de` on branch `eng1`, diff base `1680764d`.
All commands: `CUDA_VISIBLE_DEVICES=` (empty), `TMPDIR=/tmp/eng1_tmp`, `nice -n 19`, explicit `timeout`.

## Entries

- 2026-11-09 12:18 EST: seat start. Read the order (332 lines), HOUSE_RULES.md, AGENTS.md. Created this ledger and
  REPORT.md.
- 12:20: `tracked_before.json` written (10573 tracked files, sha256 per path; HEAD 007137de).
- 12:23-12:29: baseline arm configured and built from the unedited gn1 sources (CUDA 12.6.85, nvcc
  /usr/local/cuda-12.6) into `tensor_cuda/build-eng1-base/`: configure exit 0, build exit 0 (`build/*_base.json`).
  Source map before and after the build identical (29 inputs, map digest 5a44463d...). The module dropped into the
  package dir was copied to `tensor_cuda/build-eng1-base/module/` (sha256 73ededa0...9523e) before any engine edit.
- 12:30: baseline CPU probe with CUDA hidden (`baseline_cpu_probe.py/.log`): import OK; any CPU tensor with nonzero
  bytes fails in `from_host` ("no CUDA-capable device"), so only zero-byte CPU tensors are usable on CPU; every
  zero-extent case of the four E4 ops returns an empty result (the UB division is folded by this compiler, as REV1 Q5
  said); `topk` k=1 on a (2,0) input returns a (2,1) result with no refusal.
- FINDING F1 (order premise, E4): `topk_nd` has no "existing refusal" for k > last extent on this branch
  (kernels.cu:5337-5352 at 1680764d; the kernel writes sentinel values and index -1 for p >= S). Treatment chosen to
  match L90's stated treatment ("the existing refusal for k > extent"): add PyTorch's single check `k < 0 || k > S`
  in topk_nd, before allocation and division. Not silent: listed in the report as a premise correction.
- FINDING F2 (order premise, E2/L09/L55): the engine has no float64 dtype (core.h:30 DType enum).
  `tc.tensor(..., dtype="float64")` silently builds float32 (`__init__.py:49-52`, `_NP_DTYPE.get(dtype, np.float32)`).
  E2's float64 rows therefore request "float64", assert the resolved dtype is float32, and compare uint32 views. No
  uint64 comparison is possible.
- FINDING F3 (E5): C++ `/` truncates toward zero, so `(H + 2p - k) / s + 1` is 1 when the kernel exceeds the padded
  input by less than the stride (H=3, k=4, s=2). `make_dims` therefore uses a floor quotient (equal to truncation for
  every non-negative numerator, i.e. for every geometry that passes) and refuses OH <= 0 or OW <= 0; with the floor
  quotient that is exactly "kernel larger than the padded input". Extra mutant registered: truncation restored.
- FINDING F4 (E6/L46): all 17 `artifacts/wo*.ppm` receipts are tracked. A rewrite with identical bytes leaves `git
  status --porcelain --ignored` clean, so git status alone cannot "show the .ppm writes" on the baseline. The E6
  record adds a stat map (size, mtime_ns, sha256) of the write target, and `-uall` so untracked files inside an
  untracked directory are listed individually.
- FINDING F5 (E6 scope): fixture injection needs the six writer tests' `def` lines to take `tmp_path`; "no other
  line changes" cannot hold literally. Diff: 6 def lines + 8 write lines, nothing else (the `repo_root = ...` lines
  stay, now unused).
- DECISION D1 (E4, causal_softmax): the existing `S < L` refusal still runs before the zero guard, so (B, L>0, 0)
  keeps its refusal; S = 0 is empty when L = 0, rows = 0 is empty. Rationale: E4 is about the division; the S < L
  check is operand validation that predates it, and L90 says no operand outside the list changes.
- DECISION D2 (E4, sink variant): `apa_blend_softmax_sink` computes rows = B*H*L with no host division by S
  (kernels.cu:3990 at 1680764d); its device code divides only floats by cnt and by L inside the kernel, which no
  launch reaches when rows = 0. Unchanged.
- 12:34-: engine edits (kernels.cu reshape, apa_blend_softmax, topk_nd, causal_softmax, rms_norm; ops.cpp pow_scalar
  backward; conv.cu make_dims), candidate configure exit 0; candidate build started (background, foreground-checked
  before use). E6 terrain edit applied (14 lines).
- 12:48: candidate build exit 0 (`build/build_candidate.json`, 17:35:27-17:48:24 UTC). Module sha256 3e6d5805...c0ad53,
  copied to `tensor_cuda/build-eng1/module/`. Candidate source map 56b7d0a4... identical before and after the build.
- 12:44: `cpu_checks.py` (G2): E2 reference vs float64 central differences, 5/5 exponents rel err < 1e-8; E3 VJP vs
  central differences, all 10 class/case pairs, worst 4.2e-9 relative to scale; E3 hand value; L55 witness x =
  1.3671875, g = fl32(0.7), p = 3: ordered 0x407b3880 vs reordered 0x407b387f; E2 emulation of both arms with the
  test's tolerances: candidate passes all 20, baseline predicted to fail float16 p=-1.5, -0.5, +3 and bfloat16 p=-1.5,
  +3 (no knife-edge case within powf's 4-ulp band). Prior art: central differences (classical); Higham 2002 gamma_n;
  bf16 round-to-nearest-even by integer bias (TensorFlow-style; unverified -- lead to check).
- 12:49: candidate CPU G2: import OK with CUDA hidden; test_eng1.py collects 150 IDs; the 43 CPU-capable IDs pass.
- 12:50-12:56: E6 on CPU against the real tree: `test_e6_terrain_writes_nothing` PASS (5 writers ran, wo9br2 skipped:
  CUDA-only; 335 s under load ~57; 9162 files in the stat map; nothing written into artifacts/).
- FINDING F6 (tooling): GN1's committed A8 tooling is incomplete: `artifacts/gn1/a8/program_binding.py` (imported by
  run_gpu_line.py and evidence.py) and A8's own STAGED_GPU_COMMANDS.sh are absent in this tree and in the gn1
  worktree. The adaptation (`artifacts/eng1/tool/`, originals kept byte-identical in `tool/_gn1_a8_originals/` with
  SHA256SUMS) keeps: START binding, parent receipt written after the child exits with exit + timeout flag (L45),
  link(2) publication, runner (forge-gpu-run) outcome attribution, admission/attempt rules, ID census, JUnit vs
  summary agreement, device-evidence classification, recomputing scorer. Dropped: GN1 contract-C control/hostile
  conjunctions and the program-binding check (source absent). Changes are listed in each file's docstring.
- DECISION D3 (time): C++ mutant lines and the C_UBSAN control deselect only `test_e6_terrain_writes_nothing` (it runs
  no C++; ~5.5 min of CPU reference renders per line); M14 (the E6 mutant) runs the whole file.
- DECISION D4 (regression line c): the APA harness tests read repo files via `parents[2]` (scripts/, artifacts/,
  src text), so they cannot run from arm copies; line (c) runs the worktree's 50 test files with the arm package
  preloaded (GN1's method) plus the arm's own copy of the terrain file (its writes target `arms/<arm>/artifacts/`,
  which is outside the sealed arm map). Limitation, stated in the report: tests that read repo files by path, or that
  spawn subprocesses importing the package by path, see the worktree (candidate) in both arms.
- 12:57: baseline arm, CPU subset (`-k cpu`, 43 IDs): 29 failed, 14 passed, exactly the registered rule set for those
  IDs (E1 empty-tensor inference 4 and refusals 3, E4 topk refusals 2, E5 refusals 20). JUnit:
  `g2_baseline_cpu_subset.xml`.
- 13:01: the harness refused a Write of the full REPORT.md from this seat ("Subagents should return findings as text"). Not worked around: the report is delivered in the seat's final message; REPORT.md stays the early skeleton.
- 13:00: G3 symbols (`G3_symbols.json`): modules are stripped (pybind11 Release); `nm -D` shows PyInit__tensor_cuda
  in both. All 13 function symbols (reshape, topk_nd, rms_norm, causal_softmax, apa_blend_softmax(_sink), pow_scalar,
  im2col, col2im, avgpool2d(_bwd), maxpool2d(_bwd)) are defined in both arms' object files; the 7 new refusal strings
  are in the candidate module only; "reshape: element count mismatch" is in both.
- 13:01: mutant scratch tree `/tmp/eng1_mut_src` configured and fully built (exit 0). Its module (b64323fd...) differs
  in bytes from the worktree candidate (3e6d5805...): the source path is embedded. Added control arm C_COPY (that
  module, must pass every non-E6 ID) so the scratch-build route itself is shown to change no behaviour.
- 13:02: G1 `verify_scope.py`: PASS. Changed tracked files: conv.cu, kernels.cu, ops.cpp, test_terrain_render.py;
  every hunk inside make_dims, NDArray::reshape, apa_blend_softmax, topk_nd, causal_softmax, rms_norm, pow_scalar, or
  the E6 def/write lines; 10569 other tracked files byte-identical (tracked_before/after.json); one untracked file
  outside the authorized outputs: tensor_cuda/tests/test_eng1.py.
- 13:03: line (c) split into regression (the 50 worktree files) and terrain (the arm's terrain copy, with the L46
  record) because the terrain file's CPU references alone take minutes; the regression suite uses pytest's prepend
  import mode (four APA files import sibling modules and failed to collect under importlib).
- 13:03-13:08: M14 (E6 mutant) killed on CPU: `test_e6_terrain_writes_nothing` failed on `uall` gaining
  `?? artifacts/eng1/tool/arms/M14/artifacts/wo8a_blocky.ppm` (317 s). The written file is byte-identical to the
  tracked artifacts/wo8a_blocky.ppm (sha256 618e9a71...): written at the real tracked path, git status would have
  been clean (F4); the stat map catches it. The file stays in the arm's write target (outside the sealed map), listed
  for the lead to bin.
- 13:06: both arms import with CUDA hidden (`g2_import_both_arms.log`).
- 13:07: tensor_cuda/tests/test_selector_accuracy.py excluded from line (c): it runs `BITS = int(sys.argv[1]) if len(sys.argv) > 1 else 4` at import (line 40), so any pytest invocation with arguments fails to collect it (a script, not a collectable test file). regression_files.json: 49 files.
- 13:08: census (`tool/census_step3.out`): eng1 150 IDs (both arms identical), eng1_noe6 149, audit 43, regression
  2133 (49 files, both arms, 0 collection errors), terrain 53 (both arms). Terrain file on CPU (candidate): 14 passed,
  39 CUDA-only skipped, 236 s (`terrain_cpu_timing.log`).
- 13:10: lines/script/seal (preliminary, mutant arms pending). 22 lines. 54 registered baseline failures of 150 IDs
  (E1 24, E2 5 from the emulation, E4 4 topk refusals, E5 20, E6 1). Preliminary audit-mode run of the staged script
  (`tool/audit_prelim/`, stand-in runner, CUDA hidden): START written, child exit 0, timeout false, 43 IDs passed,
  parent receipt written after child exit (child_exited 18:11:51.574 < recorded 18:11:51.758), parent passed; the
  scorer refuses the audit root as GPU acceptance ("audit run cannot establish GPU acceptance", "not a device run").
- 13:14: BUG (mine, fixed): the killer-ID matcher (tool/evidence.py `matches` and cpu_mutant_kills.py) used fnmatch,
  where `[two_neg1]` is a character class, so a real kill was not counted. Fixed: '[' and ']' are literal, '*' is
  the only wildcard (8 constructed matcher cases pass). M01 then counted killed on CPU by
  test_e1_refuse_empty_cpu[two_neg1] ("DID NOT RAISE": the mutated line accepted [-1, -1]). C_COPY, candidate: 0 of 43
  CPU IDs failed; baseline: the registered 29.
- 13:19: M03 as registered failed to compile (kernels.cu(780): 'expression must be a modifiable lvalue'; known is const). Amended before any run of M03 (mutants.json, v1 kept as mutants.registered_v1.json): old line `const int64_t known = numel_of(others);` -> `const int64_t known = numel_of(others) == 0 ? 1 : numel_of(others);` (same decision: -1 accepted with a zero product). The failed attempt's receipts moved to mutants/M03_attempt1_compile_error/ (kept).
- PRIOR ART (code sites carry the same notes; all citations unverified -- lead to check):
  - E1 reshape: PyTorch reshape/view -1 inference and refusals (ATen InferSize.h `infer_size`; messages "only one
    dimension can be inferred", "invalid shape dimension", "shape '[...]' is invalid for input of size N", and the
    ambiguous zero-element case); NumPy reshape's single unknown dimension. Taken: the rule and the refusal set. Ours:
    the messages and their order. Search: "ATen InferSize.h infer_size_impl", "numpy reshape unknown dimension".
  - E2 pow backward: Micikevicius et al. (2018), "Mixed Precision Training" (ICLR), float32 math for half tensors;
    PyTorch's opmath upcast in pow backward kernels. Taken: compute wide, round once. Ours: the composition from the
    engine's astype/ew_pow/nmuls/nmul. Search: "pow_tensor_scalar_backward opmath_type".
  - E4: PyTorch topk "selected index k out of range" (k >= 0 && k <= size) and empty results for k = 0; softmax and
    norm layers returning empty tensors for empty input. Search: "torch.topk k out of range", "softmax empty tensor".
  - E5: PyTorch pooling_output_shape (div_rtn, floor quotient) and "Output size is too small". Search: "ATen Pool.h
    pooling_output_shape div_rtn".
  - Tests: Higham (2002) gamma_n bounds; IEEE 754 unit roundoff; CUDA powf 4-ulp bound (CUDA C++ Programming Guide,
    math function accuracy table); normalization VJPs (Ioffe and Szegedy 2015; Ba, Kiros and Hinton 2016; Zhang and
    Sennrich 2019; Wu and He 2018), re-derived and checked against central differences; central differences
    (classical; Fornberg 1988).
  - Mutation: DeMillo, Lipton and Sayward (1978). M09's instrument: UBSan -fsanitize=integer-divide-by-zero (GCC
    4.9, 2014). Ours: the scratch-tree incremental mutant builds and the C_COPY / C_UBSAN controls.
  - Tooling: GN1 A8 (L45 parent-after-child receipts, START seal, runner attribution) adapted; GNU timeout, POSIX
    link(2), SHA-256 (FIPS 180-4), git status porcelain v1, POSIX stat mtime. No new algorithm claimed.
- 13:16: UBSan tree built (C_UBSAN, flags -fsanitize=integer-divide-by-zero -fno-sanitize-recover=... on host C++
  and on nvcc host code via -Xcompiler; module links libubsan.so.1; sha256 3ce824c4...). C_UBSAN CPU subset: 0 of 43
  failed (the candidate's E4 guards divide by nothing).
- 13:19-13:27: built M01 (77d33c94...), M02 (1c5de898...), M04 (1dfda7b6...), M09 (UBSan, c1f60cd9...); each receipt
  shows the mutated file's sha256 and the pristine file restored (sha256 equal to the candidate's).
- CPU kills (cpu_mutant_kills/summary.json): M02 by test_e1_infer_empty_cpu[*] (4 IDs); M04 by
  test_e1_refuse_empty_cpu[other_negative] ("DID NOT RAISE"); M09 by test_e4_empty[apa_blend_softmax-last0-cpu]: the
  child aborted with "kernels.cu:3861:29: runtime error: division by zero", the mutated line (the division moved
  ahead of the guard), while C_UBSAN passed the same ID.
- 13:36-13:46: built M03 (amended; 11d2e058...) and M08 (93010ff2...). CPU kills: M03 by
  test_e1_refuse_empty_cpu[zero_product] ("DID NOT RAISE"); M08 by test_e4_empty[rms_norm-last0-cpu] and
  [rms_norm-last0-f16-cpu] (returned shape (0,) where (2, 0) and (3, 1, 0) are expected).
- 13:31: `mutant_emulation_e2.py` (prediction evidence only; M05-M07 have GPU-only killers): NumPy emulation scored
  with test_eng1.py's references and tolerances predicts M05 fails 5 IDs (incl. float16 p-1.5, p-0.5: non-finite),
  M06 fails all 10 half IDs, M07 fails float16 p-1.5 and p-0.5 (p*x^(p-1) overflows float16 before the multiply).
  Each set contains the mutant's registered killers.
- 13:47-14:09: built M10 (e993b679...), M05 (794172a4...), M06 (846b2c95...), M07 (24f3a7fc...), M11 (30f90bb9...),
  M12 (423ce3f6...), M13 (bb73241b...); every receipt restores the pristine file. CPU kills: M10 by
  test_e4_refuse[topk-k1-last0-cpu] (no refusal); M11 by the oh_zero and trunc_gap E5 IDs; M12 by the four
  ow_only IDs; M13 by the four trunc_gap IDs. M05-M07 pass the CPU subset (their killers are GPU IDs).
- 14:12: final staging. 18 arms sealed, none pending (staging_seal.json d1d55f41...; STAGED_GPU_COMMANDS.sh
  eee24bd6...; 22 lines; TOOLING_CHANGES.diff: 27 hunks against the GN1 A8 originals).
- 14:14: G4 final audit (`tool/audit_final/`, fresh root, stand-in runner, CUDA hidden): staged script rc 0; START
  bound to the final seal; child exit 0, 43 IDs passed, timeout false; runner row "ran 0"; parent receipt recorded
  after the child exited (19:14:17.885 -> 19:14:17.949 UTC); parent passed. score_gpu.py on the audit root exits 1
  with "audit run cannot establish GPU acceptance" (by design).
- 14:15: G1 final: PASS (unchanged from 13:02; tracked_after.json 10573 paths, 10569 identical).
- VERDICTS: G1 PASS, G2 PASS, G3 PASS, G4 PASS. P201 HELD (no Python check: Tensor.reshape binds straight to
  ops::reshape, bindings.cpp:1879; the baseline's own C++ message answers [-1, 4] on CPU). P202 HELD on CPU emulation
  and source construction; its device confirmation is G5 lines (a)/(b). P203 is the lead's (G5 census).
