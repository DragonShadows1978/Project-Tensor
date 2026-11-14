# ENG2 promotion record

**Change:** Python only, in `tensor_cuda/tensor_cuda/`.
- **F1, dtype refusal.** `tc.tensor`, the `_factory` makers (`zeros`, `ones`, `randn`, `rand`), `nn.parameter` and
  `functional._causal_mask` now raise `TypeError` for an unsupported dtype string (for example `"float64"` or
  `"double"`) or a non-string dtype. They check before any allocation or random draw. Before, those requests silently
  built float32.
- **N1, half-precision norms.** LayerNorm, RMSNorm, BatchNorm1D and InstanceNorm2D compute their statistics in float32
  for float16 and bfloat16 inputs, and cast back once, as GroupNorm already did. The float32 path is bit-identical.
- **Tests:**
  - new `tensor_cuda/tests/test_eng2.py`;
  - `tests/test_eng1.py` drops ten float64 rows. Those rows always ran as float32, because the engine has no float64;
    `test_eng2.py` tests the refusal instead.

**Receipts:** `artifacts/eng2/`: round 1, amendments A1–A3, reviews REV1–REV3, and the lead's G5 and G5b GPU runs.

**Scope.**
- **Covers:** the refusal entries above, the four norm forwards, and their tests.
- **Does not cover:**
  - the C++ dtype defects ENG2 recorded: `numpy_dtype` maps unlisted NumPy dtypes to float32 bytes, bool readback goes
    through float32, and non-"cpu" device strings map to cuda:0;
  - `Tensor.astype`, `tc.empty` and the quantization helpers, which are unchanged;
  - performance.

**Consumers.** Two independent censuses checked the tracked Python files of the three consumer projects (Voice, GRAPA,
GRM): A3's, and REV3's over all 2,452 files.
- They found 498 explicit-dtype calls into the changed entries, and none that ENG2 would refuse.
- 49 of those calls pass a dtype through a variable. Each was read: it passes a tensor's own `.dtype` or a
  module-level compute dtype assigned only supported names. That is a reading of the code, not a mechanical trace.

**Review:** cross-model. The implementation (round 1, A1–A3) is Claude Opus 5.5. The reviews are Codex:
- REV1 (gpt-6-astra) found 2 BLOCKER, 2 MAJOR and 1 MINOR, all in the GPU scoring tool and the census, and all caught
  before any GPU time.
- REV2 (gpt-6-astra) found 2 MINOR residuals.
- REV3 (gpt-6.1-sol) reviewed A3 and found 1 MINOR in census completeness.
- All of these are fixed or closed. See `review_receipt.json`.

**GPU evidence:**
- G5: 21 of 21 lines in device mode, 15 of 15 mutants killed (`e2f166b8`).
- G5b, rerun under A3's seal: 21 of 21 lines, 15 of 15 mutants killed, and no pinned failures (`de547743`).
- In G5b, one test, `test_e6_terrain_writes_nothing`, fails identically in both arms of the norm-regression line. It
  inspects `tests/test_terrain_render.py`, which that line's partial arm copy omits. In a full tree it passes (ENG1 G5).

**Approval:** the owner's infrastructure approval is recorded in the local promotion ticket, 2026-11-14.
