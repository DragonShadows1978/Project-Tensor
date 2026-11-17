# ENG3-F3 r2: resume with Amendment 1 (the 24 bare-cuda overflow rows; the F01 comment's place)

Follows: ENG3_F3_REV3_FIXES.md

**Dispatched 2026-11-17.** r1 (run 20261117T074308Z-3636119) left Build H (`ea723385…`), the three mutants killed, and
`artifacts/eng3f3/`. Keep every r1 file. Write r2's files with an `_r2` suffix. The writable target, rules, gates and
checklist are the F3 order's, unchanged except as below.

## Amendment 1 (lead; registered before any r2 gate runs)

r1's G1 failed on 24 inherited rows. They are `test_f2_cpu_overflow_message[nonempty-cuda-…]`, listed in
`artifacts/eng3f3/G2_HOST_COMPARE.json` under `inherited_unapproved_differences`. These rows pass bare `"cuda"` on a
host with no CUDA device and expect the OverflowError. F02, as ordered, now resolves the bare spelling by querying
`cudaGetDevice` first, so on a no-device host they get F04's RuntimeError.

The conflict was the lead's: L38 froze those rows while F02 changed their path. The lead did not census which existing
tests pass bare `"cuda"`. Two more lead faults: the F01 comment was asked for in `kernels.cu`, which the same order made
read-only.

- **A1.1:** edit exactly those 24 rows (one parametrized test) so that:
  - with no CUDA device, a bare-`"cuda"` row expects the RuntimeError naming `cudaGetDevice`;
  - with a device, it expects the OverflowError naming the shape and dtype, as before.

  The `"cpu"` rows are unchanged. No other existing test is edited.
- **A1.2:** the F01 ruling's citation goes in the docstring of the new pinning test. `kernels.cu` stays read-only and
  unchanged.
- **A1.3:** rebuild only if a product file changes (none should). Rerun G1 and G2 on CPU against Build G. The 24 IDs are
  listed exceptions, with both outcomes recorded, and every other inherited ID must be identical. Re-stage `g3-g1` and
  `g3-g2` if the test tree changed. On the GPU, the 24 rows take the device branch and must be identical to Build G.

## Mechanism & scope

As the F3 order. **Exercises** adds the 24-row edit. **Forbidden configuration** adds any other edit to an existing test.

## Registration checklist (order-lint)

- Every answer is the F3 order's, unchanged, except as below.
- **L35:** A1.1 to A1.3 are the lead's amendments, registered here.
- **L38:** the exception is A1.1's 24 rows, and only them.

## Done

As the F3 order's "Done", for r1 and r2 combined.
