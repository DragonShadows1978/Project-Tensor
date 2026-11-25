# BWD-F4 report

Filed by the lead (L127: a Claude seat's final message is its report). Seat Claude Sonnet 5.5; order
`orders/BWDF4_REV3_HARDENING.md` at 6ea595b81. The seat's full final message is summarized below in its own terms; the
receipts named here are the evidence.

## Gates (seat)
- **G1:** 24 rows, all as registered (`BOUNDARY_F4.json`; `g1/G1_F4_pytest.txt`: 24 passed). The H1 overflow rows (4)
  and H2 rows (5) fail on Build F3 and pass on Build F4; every refusal makes 0 CUDA calls. Controls identical on F3 and
  F4 (rank-8 table accepted; mean over ranks 1-4 reaches the same device step; products that fit, 2^62).
  Two clauses could not hold as written (RED, lead order faults; see lead notes):
  1. "a valid mean over each rank 1 to 4 gives Build F3's value": the engine has no CPU reduction and the order forbade
     the GPU, so the value is unobservable. Count equivalence rests on a replica of the counting rule
     (`COUNT_REPLICA.log`): 65,224 (shape, axes) points; in the 51,755 where the exact product fits, F3's wrapped count
     = F4's count = the exact count, 0 unequal; in the 13,469 where it overflows, F4 never accepts.
  2. "the reviewer's [2^32, 2^32, 0, 0] case fails on Build F3": it does not. Signed wrap-around gives 0 there, the same
     value as emptiness-first; the defect is undefined behaviour visible only under UBSan (`COUNT_REPLICA.log`:
     "signed integer overflow: 4294967296 * 4294967296"). The discriminating rows are the non-empty-axis overflow rows,
     which on F3 refuse in the reducer with the reducer's text.
- **G2: PASS.** On Build F4 and Build F3, identical counts: `test_bwdf1_zero_extent.py` 1432, `test_bwdf1_copy_ops.py`
  67, `test_bwdf2_fixes.py` 46, `test_bwdf3_fixes.py` 57, `tc_signal/tests/test_bwdf1_hardening.py` 23.
- **G3: PASS.** `h01_mean_unchecked_product` killed 4/4 (refuses with `reduce:` instead of `mean:`, showing the mutated
  count line ran); `h02_table_rank_unbounded` killed 5/5 (rank-9 table accepted, the reading row reaches
  `CUDA error at rope_apply`). `MUTANTS_F4.json` written before any mutant tree existed.
- **G4: PASS.** Writes only in the writable set; no GPU use; no git writes; nothing deleted.

## Closure table (seat)

| Finding | Fix file:line | Test | Result |
|---|---|---|---|
| F01 MAJOR (`mean` signed product) | `tensor_cuda/src/ops.cpp:1063-1092` | `h1-overflow-*` (4), `h1u-*` (2, UB-only), `h1c-*` (7) | Fixed: emptiness first, then a checked product; refuses before any allocation with 0 CUDA calls |
| F02 MINOR (table rank above 8) | `tensor_cuda/src/kernels.cu:8801-8806` | `h2-*` (5), `h2c-*` (5) | Fixed: rank 9 refused in rope_apply, export_rope_rows and swap; rank 8 still accepted |
| F03 NIT (comment) | `tensor_cuda/src/kernels.cu:8596-8605` | `h3-export_rows-empty-view-fires-bound` | Comment corrected |
| F04 NIT (comment) | `tensor_cuda/tests/test_bwdf1_copy_ops.py:268-272` | `test_bwdf1_copy_ops.py` (67 pass) | Comment corrected (verified against Build B's `export_rope_rows`) |

## Modules
- Build F3 `tc`: `36d023b3ce2b474907f1fd33c2e1b0906e2048314c7e59abdf4711ff351a04fb`
- Build F4 `tc`: `8061f980133350fde37941c786444ce186097d05a26151da4d7dfa85681470e1` (source map `58b0d32a...`)
- `h01`: `7e2afefaf4a11e1c3997261da66bdcfbf240092c7a09c095dab0f7c684d12e8c`
- `h02`: `8b993e0dd1c5416de993d3c3d474fd93634cbda42685c54a16a93ad5225a0b01`
- `tc_signal` unchanged (Build F3's `b67d8e04...`, used for G2 only)

## Deviations (seat)
1. Mean's value not observable on CPU (above); the replica grid is the evidence.
2. The reviewer's case is UB-only (above).
3. The tc_signal G2 test used Build F3's frozen module through `BWDF1_TS_PARENT`.
4. Three other "cannot fire" comments (`kernels.cu:8629`, `8727`, `8915`) are inexact for the same empty-view reason
   as F03, outside the order's edit set.
5. No tool slips, no script repairs.

## Lead notes (2026-11-24)
- The two RED clauses are lead order faults: the order asked for a value comparison the CPU cannot make while
  forbidding the GPU, and it assumed the reviewer's case would fail on F3. Neither is a product defect.
- Lead verification of the hardening (diff read): `ops::mean` decides emptiness first, then uses
  `__builtin_mul_overflow`, refusing an overflowing count before `sum` runs. The count, and so the result, is
  unchanged wherever the old product did not overflow. `rope_table_layout_ok` adds `t.ndim() > TC_MAX_DIMS`. The two
  comments now describe the code. Accepted; no re-review (hardening rule).
- The GPU value of `mean` on the final tree is covered by the merged-tree check's GPU suite before promotion.
- Deviation 4's three comments are handed to the next engine round.
