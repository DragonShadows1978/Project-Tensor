# ENG4 promotion record

**Change.** TensorCUDA's matmul computes only the operand shapes it handles correctly and refuses the rest before any
device work. `numpy_dtype` refuses dtypes it does not list. The diff from production main `f7282b1d` is 6 files in
`tensor_cuda/` (+797/−36), plus receipts under `artifacts/eng4/`, the orders and the round's scripts.
- **Operand contract.** The forward `tc::matmul` and `matmul_backward` share one predicate (`matmul_operand_plan` in
  `matmul.cu`). It accepts:
  - (a) B of rank 2, broadcast over A's batch;
  - (b) equal ranks with equal batch shapes;
  - (c) unequal ranks where every leading B extent is 1.
  Everything else raises `RuntimeError` naming both shapes, before any CUDA call. So do operands on different devices,
  `M`, `N`, `K` or a batch count above `INT_MAX`, and unsupported dtypes.
- **Production defects this fixes,** confirmed on the GPU against production's build unless marked:
  - an equal-rank pair with a smaller B batch read past B's allocation (ENG-MM1; `compute-sanitizer` memcheck: 24
    invalid reads on production, 0 on ENG4);
  - other mismatched batch shapes silently used the wrong B matrices (from the code: production's `matmul.cu:42`
    sets B's batch stride from the rank alone);
  - a `K = 0` product returned unwritten memory (cuBLAS refused the call and its status was ignored). It now returns
    zeros, through a fill that checks its own `cudaMemset` status (F3);
  - the backward of an empty A batch with a shared B raised "reshape: element count mismatch". It now gives B a zero
    gradient (F2);
  - `numpy_dtype` read an unlisted dtype's buffer as float32 (a host over-read, measured for uint16). It now refuses,
    naming the dtype, and refuses non-native byte order (F1).
- **Zero-size and overflow rules.** An empty output returns without a GEMM. The batch product is checked dimension by
  dimension before it can overflow (F2). matmul's cuBLAS GEMM status is checked.
- **Tests:** the new `tensor_cuda/tests/test_eng4_matmul.py`. In `test_bwd_r1_matmul.py`, the test that pinned the old
  first-matrix forward now asserts the refusal (listed in `artifacts/eng4/INTENDED_CHANGES.json`).

**Scope.**
- **Covers:** the matmul forward and backward operand checks, the zero-size results, the GEMM status checks,
  `numpy_dtype`, and their tests.
- **Does not cover:**
  - new broadcasting (NumPy's full rule);
  - host-tensor compute (HOST-COMPUTE-HMM, now in the ENG5 round);
  - the forward's current-device check (single GPU here; recorded as a gap);
  - the unchecked copy status in `NDArray::clone` and related core copies (CLONE-MEMCPY-UNCHECKED, found by REV3,
    inherited, now in ENG5);
  - performance;
  - installing a rebuilt engine binary.

**Review: cross-model.** Claude Opus 5.5 seats wrote every round. Codex gpt-6.2-sol reviewed them.
- **ENG4** (`5174da719`). REV1 was blind (`2c81f8d67`). The provider's content filter stopped it after it wrote its
  findings: 2 MAJOR and 1 MINOR, plus one check on tests reading files. All were fixed in F1 (`e9b0ffb89`).
- **F1.** REV2 (`e78169be2`) continued the sealed first pass and was stopped the same way after its findings: 2 MAJOR
  and 1 MINOR, all fixed in F2 (`9b7eb1796`).
- **F1 and F2.** REV3 (`31a1a7cab`) was a read-only confirm and completed. It found no blocking finding and 2 MINOR:
  - F01: the new zero fills checked the last CUDA error, not their own status. Fixed in F3 (`adb9feb83`), with GPU
    evidence;
  - F02: the inherited core copy status (CLONE-MEMCPY-UNCHECKED). Filed as its own item and taken into ENG5.
- **Where the receipts are:** each review's findings and report are on its review branch in the fork. Their logs are
  held back, listed with sha256 in each branch's `LEAD_HELD_BACK.tsv`.

**Gate evidence:**
- **Mutants:**
  - ENG4: 5 of 5 killed;
  - F1: 8 of 8 killed (ENG4's five re-run on F1, plus three new);
  - F2: 2 of 2 killed (one of `m09`'s witnesses cannot be killed by a test, and is recorded so);
  - F3: `m11` is registered as not killable by a test. A failing `cudaMemset` cannot be forced through the public API.
- **GPU suite on F2, against production's build, per test ID:**
  - 4,038 IDs on each side: 3,931 passed, 40 failed, 15 errors, 52 skipped;
  - 0 differences. The failures and errors are the same tests on both builds.
- **CPU suite:** 4,038 IDs on production against F2, 0 differences. F2 against F3 identical.
- **Accepted results on the GPU (G3):** 84 nonzero cells are bitwise equal to production. The 39 zero-size cells give
  NumPy's result.
- **F3 GPU lines:**
  - the `K = 0` cells are zeros;
  - with a stale out-of-memory error pending, F2 mislabels it as the fill's failure, and F3 returns zeros;
  - ENG4's own tests pass on F3.
- **Build F3 module:** `f005d765e3aed86b537c9660bea4a2bfffc86a121d724c6d7086ec399baeae96`.

**Before push:** a merged-tree check (MAINCHK5) builds this tree from its own commit. It compares it with production
per test ID, on CPU and on GPU. Pending.

**Approval:** pending.
