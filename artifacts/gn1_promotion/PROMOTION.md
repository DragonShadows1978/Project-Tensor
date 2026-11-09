# GN1 promotion record

**Change:** `nn.GroupNorm` (`tensor_cuda/tensor_cuda/nn.py`, GroupNorm class only).
- **Before:** every call failed, because `reshape([N, G, -1])` reached `NDArray::reshape`, which did not infer `-1`.
- **After:**
  - an explicit group extent;
  - refusal of `num_channels % num_groups != 0` and of invalid arguments;
  - biased variance per group (Wu and He, 2018);
  - float32 compute for float16 and bfloat16 inputs and parameters, cast back.
- **New test file:** `tensor_cuda/tests/test_groupnorm_gn1.py`, with value checks against a float64 NumPy reference
  (forward and backward, ranks 2–5, each supported dtype) and refusal checks.

**Receipts:** `artifacts/gn1/`, covering round 1, amendments A1–A8 and reviews REV1–REV6.

**Scope.**
- **Covers:** the GroupNorm class and its tests.
- **Does not cover:**
  - `-1` inference in reshape, which is separate engine work (ENG1, staged in the lab);
  - other normalization layers;
  - performance.

The engine module needs no rebuild, because the change is Python only.

**Review:** cross-model. The implementation is Codex gpt-6.1-sol, and the reviews are Claude Opus 5.5. REV6 accepted
with no findings at `e6843a9e`. See `review_receipt.json`.

**GPU evidence:** G5, 58 of 58 lines passed in device mode (`artifacts/gn1/LEAD_RULING_G5.md`, receipts `1680764d`).

**Approval:** the owner's infrastructure approval is recorded in the local promotion ticket, 2026-11-09.
