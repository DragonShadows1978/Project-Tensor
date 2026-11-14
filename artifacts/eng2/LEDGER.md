# ENG2 ledger (seat: Claude Opus 5.5, Agent tool; order orders/ENG2_DTYPE_REFUSAL_AND_HALF_NORMS.md)

Times are EST (UTC-5) on 2026-11-14 unless marked UTC. No git was run; refs were read from files
(`.git/worktrees/tensor_fork_apa_wt_eng2/HEAD` -> refs/heads/eng2 = dfdbc974429e942281a55f2ece64e69fa1235e16;
packed-refs: refs/heads/eng1 = bc02d3a1b150572af5521725d60b1d0d54305681).

## Setup
- 09:08 read the order, HOUSE_RULES.md, AGENTS.md, ENG1 order, ENG1-A3 order/ruling/report, the ENG1 tool.
- 09:13 L01 instruments (`preflight_L01.txt`): Python 3.12.3, numpy 2.4.2, pytest 9.0.2, cmake 3.28.3, nvcc 12.6.85,
  `forge-gpu-run --help` rc 0. /mnt/ForgeRealm free 84,547,784,704 B (stop rule: no write over 1 MB below 60 GB).
- 09:14 G1 pre-edit map `worktree_before.json` (12,319 files, sha256 02b10c88...7b2e; `snapshot.py`: whole worktree, no
  symlink following, skips only __pycache__/.pytest_cache and the .git pointer).
- 09:14 baseline package snapshot `base_pkg/tensor_cuda/` (8 .py files) = the pre-edit map; all 132 pre-edit
  tensor_cuda/ files also equal ENG1-A3's `a3/worktree_after.json` (0 mismatches).

## G3 builds
- 09:15 configure_base exit 0 (`build/configure_base.json`), cmake -S tensor_cuda -B tensor_cuda/build-eng2/base,
  Release, arch 89, production pybind11 sources, FETCHCONTENT_FULLY_DISCONNECTED, CUDACXX cuda-12.6.
- build_base: attempt 1 (nice 19, -j4) and attempt 2 hit the 580 s bound (exit 124) with only kernels.cu left (load
  24-32 on 12 cores); attempt 3 at default niceness, exit 0 in 135 s. Harness note (L56(b) G3): the bound, not the
  build, failed; the repair is dropping `nice -n 19` (the ENG2 order does not require it). Repairs used for G3: 1 of 3.
  Module 477e81cb21eb13eb2e7a639515edf34a61face790dd5106d62da2b8f01113a54 copied to build-eng2/base/module/.
- 09:46 product edits (F1, N1; see `source.diff`), then configure_cand / build_cand exit 0 (2 min 9 s); module
  f72af46b50923800875f2f1237fe279efb04c72404d844569a684bf4167609a7 (build-eng2/cand/module/).
- The two modules differ (2,502,052 bytes). Explanation, with receipts:
  - every g++ object (autograd, bindings, ops) is byte-identical between the builds; every nvcc object differs, and 9 of
    11 become byte-identical after replacing nvcc's PID-derived temp-file tag `tmpxft_<8 hex>_` and per-compilation
    module-ID suffix `_<8 hex>_<digits>` (`build/G3_obj_normalize.json`); the other two (gemm_apa.cu.o and the device
    link object) change length (the tag sits inside a compressed fatbin);
  - device code: `cuobjdump -sass` of base, cand and a third build `cand_repro` (same sources as cand, built again)
    is identical after normalizing the module IDs (sha256 7b298011..., 1,005,817 lines each; `build/G3_sass_host.json`);
  - host code: `objdump -d .text` differs in exactly 20 `lea` instructions that address the __nv_module_id strings;
  - cand vs cand_repro (no source change at all) differ the same way (`build/G3_so_normalize_cand_vs_repro.json`,
    `build/G3_obj_normalize_cand_vs_repro.json`), so the difference is nvcc build nondeterminism, not the source.
  - The worktree package's module is the pinned candidate (f72af46b...) again after the repro build.

## Product edits (`source.diff`)
- F1 `tensor_cuda/tensor_cuda/__init__.py`: new `_np_dtype` (KeyError/TypeError -> TypeError naming the supported
  dtypes); `tensor()` line 48 and `_factory` line 60 use it; in `_factory` the dtype is resolved before `np_fn` runs.
  Prior art (code comment): NumPy's dtype TypeError, PyTorch's unknown-dtype refusal (unverified -- lead to check);
  TX3-F1's strict lookup in this repo (ValueError, accepts C++ aliases and NumPy dtype objects) is NOT adopted: the
  order's rule refuses anything outside the two maps. Merge note for the lead: TX3-F1 and ENG2 differ in exception
  type, aliases ("half", "f16", "bf16", ...) and NumPy dtype objects.
- N1 `nn.py`: LayerNorm/RMSNorm/BatchNorm1D/InstanceNorm2D `forward`: `half = dtype in ("float16", "bfloat16")`;
  upcast; the float32 path keeps bc02d3a1's statements in the same order (a `not half` return, or an `out` variable
  returned unchanged); half weights/biases converted to float32; one final `.astype(dtype)`. Prior art (code comment):
  PyTorch acc_type float32 accumulation for half norms; Micikevicius et al. (2018) (unverified -- lead to check).
- BatchNorm1D running statistics: stay float32 tensors (`tc.tensor(...)` of a float32 NumPy array); rule unchanged
  `(1 - m) * running + m * batch`; for half inputs they now receive the float32 batch statistics (before: a float16
  `mean.numpy()` made `m * mn` a float16 product under NumPy 2's NEP 50; bfloat16 read back widened to float32).
- Not changed (census, `L60_dtype_census_*.json`): `tc.empty`, `Tensor.astype`, `int4_dequant`/`intn_dequant`/
  `kv_int4_unpack(out_dtype)` already refuse through C++ `dtype_from_string` (RuntimeError "unknown dtype: ...");
  `functional._causal_mask` receives only Tensor.dtype values (internal; float32 mask for every non-half dtype).

## Census and predictions
- L60 `dtype_census.py` on both arms with CUDA hidden (`L60_dtype_census_baseline.json`, `_candidate.json`).
- P303 `caller_census.py` (AST, multi-line calls) over this worktree, Project-Voice/engine and GRAPA-Native-LLM's
  grapa/, scripts/, tests/ (the order's "GRAPA src/" does not exist; GRAPA's code is grapa/): `P303_caller_census.json`.
- N1 emulation `n1_emulation.py` -> `N1_EMULATION.json`; reuses E3's tape unchanged; baseline rows equal all 40
  ENG1 G5 census rows exactly (0 mismatches).
- 10:11 binned `artifacts/eng2/tool/e6_record.py` (my own byte copy of eng1's; ENG2 has no terrain suite) with
  delete-me, id 20261114-101104_ENG2-seat_e6_record.py.

## Tool copy (L40, `COPY_DEPS.json`, `tool/TOOLING_CHANGES.diff`)
- Copied from artifacts/eng1/tool (as A3 left it; evidence.py sha c7201f7d = A3's): arm_origin.py, device_evidence.py,
  evidence.py, forge_gpu_run_standin.py, run_gpu_line.py, run_gpu_tests.py, runtime_evidence.py, score_gpu.py,
  start_gpu_run.py, stage.py, STAGED_GPU_COMMANDS.sh (regenerated by stage.py). Not copied: ENG1's data files (ids_*,
  collect_*, seals, gpu_lines.json, arms/, gpu_runs/, _gn1_a8_originals/), which ENG2 regenerates or does not use.

## Tests (`tensor_cuda/tests/test_eng2.py`, 262 IDs; 121 named *cpu*)
- E3 helpers IMPORTED from test_eng1.py (importlib, same directory), not copied: e3_inputs, e3_reference,
  e3_tolerance, e3_distance, plus _e3_layer/_e3_pattern/_leaf/_const/_np/E3_* (the pieces e3_measure uses).
  Tolerances: E3_ROUNDINGS/e3_tolerance unchanged. Prior art in the module docstring.
- Dev note: running test files directly with pytest from the worktree root in importlib mode imports the outer
  `tensor_cuda/` directory as a namespace package (tc.__file__ None); every run here preloads the arm package first, as
  the staged child does.
- Added beyond the order's list (ledgered): test_n1_half_rule_bits (40 IDs; the registered half rule recomposed with
  the same engine ops, bitwise) and test_n1_batchnorm_running_stats (6 IDs; float32 dtype, rule, derived float32
  tolerance in the test's comment). Both exist to kill MN2 for RMSNorm/BatchNorm1D, whose bfloat16 baseline rows sit
  inside E3's tolerance, and to make each mutant's kill structural as well as numerical.

## Mutants (`mutants/mutants.json`, written once by stage.py registry before any mutant ran; diffs in MUTANT_DIFFS.txt)
- 15 one-hunk mutants: MF1, MF2; MN1/MN2/MN3 per class (LN, RMS, BN, IN); MN4 on LayerNorm (float16 round trip of
  xc * inv on the float32 path). The order lists MN1 and MN2 once; each class has its own forward, so one edit can
  only reach one class: registered per class (deviation, ledgered). An f32->f32 extra cast is provably equivalent
  (a same-dtype cast is a copy whose VJP is the identity) and was not registered as MN4.
- CPU: MF1 kills exactly its 12 registered IDs, MF2 its 48, every killer traced through the hunk line
  (`cpu_mutant_kills.json`, g2/MF*_cpu.json). N1 mutants: 0 CPU failures (norm classes need a device); predicted
  killers from `N1_EMULATION.json`; their L41 evidence is the child's mutant_trace on the GPU.

## G2
- 10:2x arms built (`tool/arms`, 17 arms, 160 MB); census collected (`tool/ids_*.txt`, collect_*.json, 0 errors):
  eng2 262 (both arms equal), audit 121, regnorm 578, regdtype 1779 (both arms equal).
- CPU subset on every arm with CUDA hidden (`g2/cpu_runs.py`): baseline 60 failed / 61 passed (all F1 refusals),
  candidate 121/121 passed, MF1 12 failed, MF2 48 failed, N1 mutants 0 failed.
- Both arms import with CUDA hidden (`g2/import_both_arms.txt`).
- L38 expected change shown on CPU: test_eng1.py's 10 float64 IDs fail on the candidate with the registered TypeError
  (`g2/l38_float64_cpu.txt`).

## L38 (`L38_PINS.json`, `l38_pins.py`)
- 215 dtype-string call sites in the 39 line-(c) files traced to their values; 2 change (test_eng1.py:70/:76, the
  helpers that build E2's 'float64' rows). 10 IDs are expected to change, passed -> failed with
  "TypeError: unsupported dtype 'float64'; ...": test_e2_pow_grad[float64-*] and test_e2_pow_grad_bits[float64-*].
- Norm-output reads: test_attention_phase5 (RMSNorm, LayerNorm via TransformerEncoderLayer), test_nn_phase3
  (LayerNorm), test_ext_phase7 (BatchNorm1D, GroupNorm, InstanceNorm2D), test_groupnorm_gn1 (GroupNorm), all float32
  inputs except GN1's half GroupNorm (unchanged class); test_eng1's E3 census rows change in value, not outcome.
- Tool change for L38: evidence.regression_reasons skips pinned IDs and requires their registered outcomes and text;
  unit-checked on constructed receipts (`g4/l38_regression_check.stdout`, 5 cases, ok).

## Staging and G4
- regression_files.json: regnorm = the 5 files that build a norm class; regdtype = the 34 other test files that call
  tc.tensor/a factory with a dtype other than the literal "float32" (from P303_caller_census.json). Split in two lines
  for time (ENG1 measured 357 s for its 49-file line); bounds inner 600 s (eng2, mutants, regnorm), 1050 s (regdtype).
- 21 lines (gpu_lines.json); 106 registered baseline failures of 262 (baseline_expected_failures.json: 60 CPU F1 IDs
  from the baseline arm's CPU run + 46 N1 IDs from the emulation).
- Seal `tool/staging_seal.json` sha256 8de89b78f7080af2467c21a50a7f7e18fb5cda7140dc24f3a32ebf28aac2a72d (22 tool files,
  44 registration files, 17 arms, engine pins base 477e81cb..., candidate and mutants f72af46b...).
- `g4/cuda_preflight.py`: pre-governor imports + seal + 17 arm checks map no CUDA/engine library, load no tensor_cuda.
- `g4/g4_dry.py dry_1` (root tool/gpu_runs/dry_1): 21/21 ok. Roots (baseline_eng2, candidate_eng2, baseline_regnorm,
  baseline_regdtype): seal and arms verified, governor DRY RUN, exit 78, not admitted, no child. Dependents (2
  candidate regressions, 15 mutants): seal and arms verified, refusal at the dependency check, no attempt; their runner
  argv to forge-gpu-run with FORGE_GPU_DRY_RUN=1: 78, no run-log line (all 21).
- Audit route (`STAGED_GPU_COMMANDS.sh --audit`, root tool/gpu_runs/audit_1): staged exit 0; 10 post-device steps ok
  (`g4/G4_AUDIT_RECEIPT.json`); score_gpu.py exit 1 with "audit run cannot establish GPU acceptance" (designed).

## G1
- `g1_scope.py` after-map (`worktree_after.json`, 12,905 files): changed only __init__.py and nn.py; added outside
  artifacts/eng2 and build-eng2 only test_eng2.py and the installed module; removed none; 12,317 files identical.
  `G1_HUNKS.json`: every changed line sits in nn.py's four forward methods or __init__.py's _np_dtype/tensor/make.

## Prior art (seat's list; all unverified -- lead to check)
- float32 statistics for half norms: PyTorch acc_type in layer_norm/batch_norm/group_norm kernels; Micikevicius et al.
  (2018) "Mixed Precision Training" (ICLR). Taken: compute half statistics in float32, cast once. Ours: the engine op
  sequence. Search terms: "PyTorch layer_norm acc_type half", "Mixed Precision Training Micikevicius 2018".
- dtype refusal: NumPy np.dtype TypeError ("data type not understood"); TX3-F1 strict lookup (this repo). Ours: the
  TypeError message and the map-keys rule.
- Mutation testing: DeMillo, Lipton and Sayward (1978). Tape emulation: Wengert (1964), E3 (ENG1).
- Build normalization: reproducible-builds/diffoscope practice; nvcc tmpxft temp names. sys.settrace (PSF).
