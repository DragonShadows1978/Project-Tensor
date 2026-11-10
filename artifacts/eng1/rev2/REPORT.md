# ENG1-REV2 report: REJECT (1 MINOR, 1 NIT)

The lead copied this report from the seat's final message, because the seat's own tool refused its full write.
`findings.json` in this directory is the seat's own complete file.

**Summary.** A1 fixes R1-F01, R1-F03, R1-F04 and R1-F05 correctly. Its subprocess binding and origin preflight
(R1-F02) work. The defect is its list of 16 excluded source-reading tests, which misses 6 tests (7 IDs). The second
finding is a NIT in the E3 census row. Outside `test_eng1.py` nothing changed, and the CPU subset passes 47 of 47 on
the candidate. The worktree is byte-identical before and after the review's runs.

## Findings

### R2-F01 (MINOR): the 16-test list of source-path readers is incomplete

- **Where:** `artifacts/eng1/tool/source_reading_tests.json`, written from the hand list in
  `artifacts/eng1/a1/classify_sources.py:21-34`. The table is `a1/SOURCE_READING_TESTS.md`.
- **The gap:** seven IDs read worktree source files by path at run time, but they are not in the list. So they stay
  in the 2117-ID baseline/candidate comparison (`tool/ids_a1_regression.txt`).

| Test | What it reads |
|---|---|
| `test_apa_sp3.py::test_compiled_d96_flag_device_and_scalar_guards[7-11]` and `[1-4097]` | `src/kernels.cu` and `src/paint_bake.cu`, through `common.load_runtime()` → `verify_sources()` (`scripts/apa_sp3_common.py:181-182`). This is the same route as the listed `test_every_preexisting_source_and_kernel_body_pinned`. Both arms refuse with `Red: pre-existing source changed: tensor_cuda/src/kernels.cu`. |
| `test_apa_sp4g_a5.py::test_a5_dispatch_L64_is_prefill_even_with_cache` | `src/kernels.cu`, which ENG1 changed: `scripts/apa_sp4g_a5_model.py:40` runs `sha(R/'tensor_cuda/src/kernels.cu')` |
| `test_apa_sp4g_a5.py::test_a5_full_cached_fork_capture_saves_metadata_and_merge` | `src/kernels.cu`, the same way |
| `test_apa_sp4g_a5.py::test_a5_native_pin_and_mask_disagreement_are_receipted` | `src/kernels.cu`, the same way |
| `test_apa_sp4g_a4.py::test_a4_calibration_uses_actual_ppl_fraction_receipt` | 14 `scripts/apa_sp4g*` `.py`/`.sh` files, through `scripts/apa_sp4g_common.py` |
| `test_apa_sp4g_a5.py::test_a5_completed_RED_accepted_without_waiving_gate` | 5 `scripts/apa_sp4g*` files, through `scripts/apa_sp4g_common.py` |

- **The method:**
  - all 49 regression files (2133 IDs) were run on CPU with CUDA hidden, with the candidate arm preloaded and first on
    `PYTHONPATH`, as the staged child does;
  - a Python audit hook recorded every worktree file each test read. It also refused every write outside scratch and
    every subprocess launch: 36 refusals in all (35 subprocess launches and 1 Hugging Face cache copy), none of them
    a write into the worktree;
  - after dropping imports, traceback rendering, temp-tree walks, `artifacts/` receipts and `orders/`, 19 tests read
    source files by path. 13 of them are in A1's 16, and the 6 above are not;
  - 3 of A1's 16 were not observed: two fail on CPU before their read, and the third reads a file outside the
    worktree. Their inclusion is not disputed;
  - an AST scan of all 49 files found no other source reader.
- **Why it matters:** A1's binding fix requires every test that reads candidate source files by path to be listed and
  excluded from the comparison, and A1's report and gate G3 say the 16 are that complete list. Both arms read the same
  worktree files, so these outcomes cannot tell the arms apart. A source pin that the candidate broke would fail on
  both arms and pass the no-new-regression rule.
- **The effect on G5 for these 7 IDs: none observed.**
  - `test_compiled_d96` already failed before ENG1: its pinned `kernels.cu` hash `6c416ece…` matches neither the
    pre-ENG1 file (`d90124e0…`) nor the candidate (`7d52c1a5…`).
  - The a5 tests only record the hash.
  - The two script readers read files ENG1 did not change.
  - Not observed: reads after a first CUDA call, on GPU-only paths.
- **The fix:**
  1. Add the 6 selectors, with their paths, to the JSON list and the `.md` table.
  2. Rerun the `stage.py` census. The comparison becomes 2110 IDs.
  3. Seal into a new seal file.
  4. Prefer building the list from a runtime read trace over a hand list.
- **Receipts:** `q2_source_readers.py`, `q2_readers_{A,B,C}.json`, `q2_classify_reads.{py,json}`,
  `q2_static_scan.{py,json}`, `probe_test_sources.py`, `q2_probe_sources.stdout`.

### R2-F02 (NIT): the engine-to-emulation distance is empty in the 4 rows that need it

- **Where:** `tensor_cuda/tests/test_eng1.py:534-538` (`e3_distance`) and `:584-586`.
- **The gap:** `engine_distance_from_emulation` is `None` when either vector has a non-finite value. In
  `a1/e3_emulation.json` the emulation is non-finite in every element of four rows, all float16 near_eps backward:
  LayerNorm, RMSNorm and BatchNorm1D (64 of 64 each) and InstanceNorm2D (96 of 96).
- **Why it matters:** equal counts do not prove equal positions, signs, or inf against NaN. This is measurement only,
  and A1 disclosed it.
- **The fix:**
  - also record `nonfinite_pattern_equal`, as
    `np.array_equal(np.where(np.isfinite(got),0,got), np.where(np.isfinite(emu),0,emu), equal_nan=True)`;
  - record the maximum distance over the elements that are finite in both.

  Under the 11-03 rule, this is fixed once and closed.

## Questions 1–6

**Q1 (R1-F01, E3): sound, except R2-F02.**
- **The engine's arithmetic,** read from `nn.py` against `ops.cpp`, `autograd.cpp` and `kernels.cu`:
  - elementwise ops compute in fp32 and store once, rounding to nearest, with scalars cast to float;
  - reductions follow `kernels.cu`: 256 serial lanes, then pairwise halving, while other reductions are serial;
  - `mean` is a sum followed by a scalar multiply, and `var` has its own mean node;
  - the pow backward computes in fp32 and casts once, and the cast backward rounds once;
  - gradients accumulate in reverse post-order over each node's parents, and the test's tape builds the same
    structure;
  - GroupNorm upcasts exactly to fp32, then casts the output back once.
- **An own straight-line emulation** (`q1_emulation_check.py`) of LayerNorm and GroupNorm:
  - matches the test's `e3_emulation` bit for bit on all 16 rows;
  - its float64 reference agrees within 5.7e-14, and its errors within 1e-12 of `a1/e3_emulation.json`;
  - it reproduces REV1's two numbers, RMSNorm float16 near_eps 0.015658268723244273 and LayerNorm bfloat16 near_eps
    0.20484508346333996, each within 2.8e-17.
- **`e3_round` against PyTorch's conversions** (`q1_rounding_instrument.py`): 0 mismatches over 2^20 random float32
  patterns, 1152 bfloat16 ties, 4096 float16 ties and 16 specials.
- **The labels:** "accuracy targets (not error bounds)" is at `test_eng1.py:14`, `:285`, `:307` and `:581`.
- **No E3 ID is a gate:**
  - `test_e3_census` asserts only shape, dtype and a valid reference;
  - no E3 ID is among the 59 expected baseline failures;
  - `score_gpu.py` only reports the E3 rows.

**Q2 (R1-F02): the binding is sound; the list is incomplete (R2-F01).**
- **The binding:** all 24 staged lines run `run_gpu_tests.py`, which:
  - puts `arms/<arm>/tensor_cuda` first on `PYTHONPATH` (line 84);
  - runs the origin preflight (line 85) before `import pytest` (line 91);
  - exits 2 on a refusal (line 139), unlike the mutants' expected exit of 1, and writes no `child_result.json` then.

  `evidence.evaluate` checks the origin record again. A1's audit receipts are ordered origin, junit, child result,
  parent result.
- **REV1's probe, rerun per arm** (`q2_probe.py`):

  | Arm | Without binding | With binding | Wrong pin | Decoy cwd package |
  |---|---|---|---|---|
  | Baseline | exit 1, `ModuleNotFoundError` | preflight exit 0; scenario exit 0, importing pin `73ededa0…` | refused before pytest import | refused |
  | Candidate | exit 1, `ModuleNotFoundError` | preflight exit 0; scenario exit 0, importing pin `3e6d5805…` | refused before pytest import | refused |

- **Residual risk, not a finding:** a decoy package in another subprocess's cwd would win over `PYTHONPATH`. No
  staged test starts Python from a directory that holds a real `tensor_cuda` package.

**Q3 (R1-F03, the bfloat16 witness and M15): sound.**
- Re-derived in exact rational arithmetic with an independent round-to-nearest-even (`q3_m15_arith.py`).
- **The bfloat16 witness** (x = 0.0002994537353515625, g = 0.30078125, p = 3):
  - the exact reference is 5693919·2^-46, and the tolerance is 3.1615e-10;
  - the late cast gives 174·2^-31, with error 1.0961e-10, which is 0.347 of the tolerance;
  - the early cast (M15) gives 173·2^-31, with error 3.5605e-10, which is 1.126 of the tolerance.
- **The float16 witness** (x = 0.1492919921875): 0.045 of the tolerance against 1.50.
- **Robustness:** both hold for any `powf` error from -4 to +4 ulp.
- **The baseline:** it passes the bfloat16 witness and fails the float16 one, as A1's expected-failure list says.
- **The call path:** `x.pow(3.0)` → `pow_scalar` backward → the `DType::BFloat16` branch, with exactly g arriving.

**Q4 (R1-F04, M16): sound.**
- `make_dims` (`conv.cu:141-143`) uses a floor quotient. Input (0,1,6,2) with a 3×3 kernel, stride 1 and no
  padding gives OH = 4 and OW = 0 for all four callers.
- The refusal message prints `(4, 0)`.
- M16 changes only `ow <= 0` to `ow < 0`.
- Rerun on the M16 arm (pin `c597a30c…`): exactly the 4 `ow_zero` IDs fail, and the other 43 pass.

**Q5 (R1-F05, the census): sound.**
- **10 entries checked against the code**, matching on call text, receiver and reason:
  - `__init__.py:553:35`;
  - `functional.py:189:53` and `:190:12`;
  - `nn.py:304:15`, `:498:12`, `:498:45` and `:576:12`;
  - `quant.py:124:9` and `:186:11`;
  - `quantization/affine.py:156:14`.
- **An independent count** finds 46 `.reshape(` calls: 37 engine calls on 32 lines, plus 9 NumPy calls.

**Q6 (nothing new broken): sound.**
- **The scope diff** `5031c765..7cf46bd5` touches only `test_eng1.py` (+204/-7), byte-identical to
  `a1/test_eng1.diff`.
- **The tooling:**
  - the parent-after-child property holds: `run_gpu_line.py` changed only the seal name, and still waits at 66,
    records at 70 and evaluates at 116;
  - the seal is `5d326628…` and the staged script `0d0a4635…`;
  - all 20 arms match their pins, and each arm's `test_eng1.py` equals the worktree's.
- **The CPU runs, with CUDA hidden:**
  - the candidate passes 47, deselects 109, exits 0, with pin `3e6d5805…`, and its ID set matches
    `tool/ids_a1_audit.txt`;
  - C_COPY and C_UBSAN each pass 47 of 47;
  - the baseline fails 33, all of them registered expected failures.

## L46

A map of 11455 files was taken before and after the review: both hash to `577ae473…`, with no changed paths
(`l46_compare.json`). There were no git writes, no GPU use and no deletes.

## Prior art (unverified; the lead to check)

- IEEE 754-2019 round-to-nearest-even;
- NumPy float16, and PyTorch `c10::Half` and `c10::BFloat16`;
- Higham (2002);
- Wengert (1964) and Linnainmaa (1970), reverse accumulation;
- Ba, Kiros and Hinton (2016), and Wu and He (2018);
- DeMillo, Lipton and Sayward (1978);
- the CUDA `powf` 4-ulp bound;
- PEP 578 audit hooks (2019);
- REV1's `d4_child_probe.py`, credited;
- Python `ast` and `tokenize`.

## Files

Everything is under `artifacts/eng1/rev2/`:
- `findings.json`, this report, and `l46_before.sha256`, `l46_after.sha256`, `l46_compare.json`;
- the scripts `run.py`, `cpu_subset.py`, `q1_emulation_check.py`, `q1_rounding_instrument.py`, `q2_probe.py`,
  `q2_source_readers.py`, `q2_classify_reads.py`, `q2_static_scan.py`, `probe_test_sources.py`, `q3_m15_arith.py` and
  `q5_census_spot.py`;
- their outputs;
- `q4_cpu_M16_*` and `q6_cpu_*`.

Scratch is under `/tmp/eng1_rev2_work`.
