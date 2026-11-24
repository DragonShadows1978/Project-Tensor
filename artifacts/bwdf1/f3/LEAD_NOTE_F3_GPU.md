# BWD-F3 GPU lines: lead reading of the post verdicts (2026-11-24)

The lead's runner (`bwdf3_lead_lines.sh`) ran all 66 staged lines. The staged post code ran at its end, at 03:56.
Its output is `logs/bwdf3_lead_post.log` (held back), and its verdict files are under `artifacts/bwdf1/f3/gpu/`.

## Passed as registered

- **G2 audit witnesses on F3:** `g2_f01_F3`, `g2_f03_F3` and `g2_direct_F3` all PASS. Build B's lines are recorded
  with no bar.
- **G3:** `g3_F3` PASS. The mutant `m03_reduce_gt1` is KILLED.
- **K2, BWD-F2's lines on F3:**
  - `k2_f01`, `k2_f04`, `k2_f07`, `k2_vals` and `k2_count` all PASS on F3. Builds B and E show each defect.
  - The mutants `n03`, `n04` and `n05` are KILLED.
- **P2 count:** `p2_count_F3` PASS, and the mutant `p04_splice_bound_after_work` is KILLED.
- **G4, the whole suite per test ID** on the GPU, Build B against F3 (`P3_GPU_COMPARE.json`): PASS.
  - 5,424 IDs on each side.
  - Exactly 2 differences, and both are intended: `test_bwd_r3_paths.py::test_host_real_planner_and_no_grad[conv1d]`
    and `[conv_transpose1d]` go from failed to passed.
  - 0 unintended differences.

## Not passed, and why

- **`p2_e1_F3`, `p2_e2_F3` and `p2_e3_F3` FAIL: harness, superseded.**
  - Every child died under `--target-processes all` (BWD-F5's `DIAGNOSIS.md`).
  - BWD-F5 repaired the harness and re-ran E1 to E3 on Build F4, which is F3 plus the hardening and is the
    promoted tree. All three PASS there (branch `bwdf4`, commit `41ad223d2`).
- **`copy_F3` FAIL: a harness case, not the product.**
  - 11 of the 12 rows refuse with the named message, and memcheck finds 0 errors.
  - The remaining row, `export_rows-rank0`, never built a rank-0 tensor. It calls `tc.tensor(np.float32(1.0))`, and
    `tc.tensor` passes its data through `np.ascontiguousarray`, which returns an array of at least one dimension
    (NumPy 2.4.2 gives shape `(1,)`).
  - So the call was `export_rows` of a `[1]` tensor over the span 0 to 1, which is valid, and F3 correctly returned
    `[1]`.
  - The rank-0 refusal itself is at `kernels.cu` `export_rows_check` ("rank must be nonzero"). This line did not
    exercise it.
- **`controls` FAIL (7 of 198 rows compared): mostly unanswerable on Build B.**
  - All 7 rows that compared are PASS: the k2 controls, bitwise equal between B and F3.
  - 180 `g2_direct` rows have no Build B side. In the `g2_direct_B` process, the first row (zero-extent `add`,
    production's broadcast defect) left a sticky "unspecified launch failure". Every later row in that process,
    controls included, then failed in `cudaMalloc`, so `g2_direct_B/controls.npz` is empty.
  - F3's 180 control arrays exist and have nonzero results. The comparison against B is unanswered, not failed.
  - The other 11 missing rows are the P2 controls, covered by BWD-F5's re-run: 9 of 11 are bitwise equal. Of the
    other two, one has no Build B side because Build B refuses the call; the other is a 1-element result, bitwise
    equal but marked vacuous.

## What remains for promotion

- **The `g2_direct` accepted-result controls, Build B against the promoted build, bitwise.** Each row needs its own
  process, so that B's defect rows cannot poison its controls.
- **A real rank-0 `export_rows` case** on the promoted build.
- **BWD-F3's ordering-count line on F4** (BWD-F5 deviation 1).

These go to one small staging round. The merged-tree check against production `75c85de6` follows it.
