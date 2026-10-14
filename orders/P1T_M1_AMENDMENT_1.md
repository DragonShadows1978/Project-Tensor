# P1T-M1 Amendment 1: device-side `tensor_cuda.empty`

This adds to `orders/P1T_M1_DLPACK.md` (bc812f4). Every rule, gate and scope limit there still applies. Where they
conflict, this file wins only on the item below.

## Why

The add-on package (`tc_signal`, M2–M4) allocates its outputs through TensorCUDA, so there is one allocator and one
pool. TensorCUDA's Python factories (`zeros`, `ones`, `full`; `__init__.py:51` `_factory`) build the array in host
numpy and then upload it. A 14 MB Kokoro activation would then cost a host allocation plus an H2D copy per call. The
C++ already has `NDArray::empty` (`include/tc/core.h:84`), which allocates on the device with no copy.

## Mechanism & scope

- **Measures:** whether `tensor_cuda.empty` returns an uninitialized device tensor from the engine's allocator, with
  no host round-trip.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/docs/P1T_TENSOR_ALT_PLAN.md` §4
  ("Outputs are tensor_cuda.Tensors allocated through tensor_cuda itself").
- **Exercises:** `tensor_cuda/src/bindings.cpp`, `tensor_cuda/tensor_cuda/__init__.py`, `tensor_cuda/tests/test_dlpack.py`.
- **Does not cover:** changes to `zeros`, `ones` or `full`; `requires_grad` plumbing beyond the default False; and
  CPU-device `empty` beyond whatever `NDArray::empty` already does.
- **Required evidence:**
  - the test that `empty` returns the requested shape and dtype, and that its `data_ptr` round-trips through DLPack;
  - a timing line comparing `empty` with `zeros` for [1, 512, 7160] fp32 (median of 20).
- **Operand census:** fp32 and fp16; shapes (), [7], [1, 512, 7160].
- **Baseline isolation:** same as the base order (G2 baseline ids, G3 diff scope). `__init__.py` and `bindings.cpp`
  are already in scope.
- **Forbidden configuration:** a new kernel; zero-filling inside `empty`; any change to existing factories.

## Required

1. Bind `NDArray::empty` as `tensor_cuda.empty(*shape, dtype="float32", device="cuda")`, with the same shape
   conventions as `zeros`. It returns a `Tensor` with `requires_grad=False`. Its contents are uninitialized; say so in
   the docstring.
2. Add it to `__all__` next to `from_dlpack`.
3. Test it in `test_dlpack.py`, then report the evidence above in the M1 report under "Amendment 1".

## Done

Add an "Amendment 1" section to `docs/P1T_M1_REPORT.md` with the test result and the empty-vs-zeros timing line.
Include both, verbatim, in your final message alongside the base order's Done items.
