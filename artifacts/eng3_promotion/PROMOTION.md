# ENG3 promotion record

**Change.** Host-side (CPU) tensors in the NDArray core no longer need a GPU, and their conversions are correct. The
diff from its merge base `2478a4d` is 4 files in `tensor_cuda/` (+1,204/−9), plus receipts under `artifacts/` and the
orders.
- **CUDA-free core host paths.** `from_host`, `to_host`, `zeros`, `full`, `clone`, `to` and `astype` call no CUDA for
  CPU tensors. Host casts are bit-exact to the device kernels.
- **Bool export.** `numpy()` on a bool tensor returns correct values. It used to read memory it shouldn't.
- **Allocation safety (F1).**
  - The byte size is checked for int64 overflow before any allocation; the error names the shape and dtype (F2).
  - A failed allocation raises MemoryError and leaks nothing.
- **Private test hooks** (`_core_*`).
  - Strict device strings; a refusal names the full string, escaped (F2).
  - Bare `cuda` labels the current device (F3).
  - A failed CUDA query raises RuntimeError naming the call (F3).
- **Tests:** `tensor_cuda/tests/test_eng3_host.py` and `test_eng3_f3.py`.
  - These include F1's killing tests (moved in, F3).
  - They pin the zero-extent refusal, which matches NumPy (F3, ruled intended).
  - The no-device tests run in a child that hides the device (F4).

**Scope.**
- **Covers:** the host paths and checks above, and their tests.
- **Does not cover:**
  - HOST-COMPUTE-HMM, the CPU path of 172 compute operations, which is still open;
  - `Device::from_string`;
  - rank-9 public tensors;
  - performance;
  - installing a rebuilt engine binary.

**Review: cross-model on every authored part.**
- **ENG3** (`9c6a8f5e`) was written by Claude Opus 5.5. Codex reviewed it blind (REV1, `cecee864`): 1 MAJOR and 4
  MINOR, all fixed in F1.
- **F1 and F2** (`3597cf84`, `60e28eb7`) were written by Codex.
  - Codex reviewed F1 (REV2, `16db930c`): 3 MINOR, 2 fixed in F2 and 1 closed.
  - Claude reviewed F1 and F2 cross-model (REV3, `30ff8775`): 3 MINOR and 3 NIT, all fixed in F3, or closed by
    ruling (F01).
- **F3** (`c42d0880`) was written by Codex. Claude reviewed it (REV4, `b096452f`): the product code is correct, with 1
  MINOR and 3 NIT in tests, all fixed in F4.
- **F4** (`e40e1a25`) was written by Codex. It is a test-only change, verified by the lead's GPU lines.

**Gate evidence:**
- Mutants killed:
  - ENG3: 5 of 5;
  - F1: 3 of 3;
  - F2: 1 of 1;
  - F3: f3m01, f3m02, and f1m02 by the moved test.
- **CPU:** 267 tests passed on Build H.
- **The lead's GPU lines on Build H, against Build G:**
  - 1,685 and 1,683 test IDs identical;
  - 0 differences;
  - 0 writes;
  - the 24 bare-cuda overflow rows on the device branch.
- **REV3's GPU reproductions:** both CONFIRMING, as expected.
- **Build H module:** `ea7233854f0d67c60652a230c7ec8e2e7712c5164144a1259b6be231ba7a4e43`.

**Approval.** The owner approved this merge via Ember on 2026-11-17, replying A to the decision request
`20261117-015200_Project-Tensor_eng3-promote.md` ("merge ENG3 into main now"). The approval was conditional on all the
testing and mutation testing passing. The verbatim text is in the local promotion ticket and the lead's register.
