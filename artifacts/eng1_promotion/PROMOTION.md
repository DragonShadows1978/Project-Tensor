# ENG1 promotion record

**Change:** five engine defects and one test defect, fixed in `tensor_cuda/src/` with a new value-checking test file.
- **E1, reshape `-1`.** `NDArray::reshape` now infers one `-1` extent, and refuses two or more `-1`, any other
  negative extent, `-1` with a zero product of the other extents, and a non-integer inferred extent. Before, every
  `-1` was refused as a count mismatch.
- **E2, half-precision `pow` backward.** For float16 and bfloat16 inputs, `pow_scalar`'s backward computes in float32
  and casts once. Before, `x^(p-1)` overflowed float16 for small x and negative p. float32 is bit-identical.
- **E3, norm half census.** The half-precision behaviour of LayerNorm-style code, RMSNorm, BatchNorm1D and
  InstanceNorm2D is measured (the fix itself is ENG2).
- **E4, zero last extent.** `rms_norm`, `causal_softmax`, `apa_blend_softmax` and `topk_nd` no longer divide by a zero
  last extent (undefined behaviour in the source).
- **E5, pooling dims.** `make_dims` refuses non-positive output extents instead of returning a wrongly shaped result.
- **E6, terrain test writes.** `tests/test_terrain_render.py` no longer writes `.ppm` files into the source tree.
- **New test file:** `tensor_cuda/tests/test_eng1.py`.

**Receipts:** `artifacts/eng1/`: round 1, amendments A1–A3, reviews REV1–REV2, and the lead's G5 GPU run.

**Scope.**
- **Covers:** the six fixes above and their tests.
- **Does not cover:**
  - the norm-layer half-precision fix and the dtype refusal (ENG2, promoted separately);
  - installing a rebuilt engine module. This merge changes C++ sources only. The installed `_tensor_cuda` binary is
    unchanged until a separate rebuild and install;
  - performance.

**Known limitation, fixed by ENG2:** ten `test_eng1.py` rows labelled float64 ran as float32, because the package then
converted unsupported dtypes to float32 silently. They pass and check float32 values. ENG2 makes float64 refuse, and
relabels those rows before it is promoted.

**Review:** cross-model.
- Round 1 (the six engine fixes): Claude Opus 5.5. REV1: Codex gpt-6.1-sol, 5 findings in the tests and tooling (the
  C++ fixes themselves were not faulted).
- A1 (Codex) fixed them. REV2 (Claude) confirmed the fixes and raised 1 MINOR and 1 NIT. A2 (Codex) fixed both; the
  lead closed them by rerunning REV2's trace (the 11-03 rule: NITs get one fix and close).
- See `review_receipt.json`.

**GPU evidence:** G5, 24 of 24 lines passed in device mode (receipts `bc02d3a1`, `artifacts/eng1/lead/g5_1114/`):
- baseline and candidate on `test_eng1.py`, on a 49-file regression selection (2,133 IDs) and on the terrain tests;
- compute-sanitizer memcheck and UBSan lines;
- 16 of 16 pre-built mutants killed.

**Approval:** the owner's infrastructure approval is recorded in the local promotion ticket, 2026-11-14.
