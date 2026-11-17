# ENG3-F3 r2: resume with Amendment 1 (the 24 bare-cuda overflow rows; the F01 comment's place)

Follows: ENG3_F3_REV3_FIXES.md
ID reuse: ENG3-F3 r2 resumes the third fix round of the same ENG3 change

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

- **Measures:** whether REV3's findings are fixed or closed as ruled, with the 24 bare-cuda rows updated to F02's path
  and nothing else changed.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng3/orders/ENG3_F3_REV3_FIXES.md`,
  with this amendment.
- **Exercises:** A1.1's edit to the 24 rows; A1.2's docstring; the G1 and G2 reruns on CPU; the re-staged GPU lines.
- **Does not cover:** as the F3 order (the constructor's byte rule, HOST-COMPUTE-HMM, `Device::from_string`, rank-9
  public tensors, promotion).
- **Operand census:** the 24 IDs in `G2_HOST_COMPARE.json` (`inherited_unapproved_differences`), the new pinning test,
  and the staged launchers.
- **Baseline isolation:** Build G is the reference. The 24 IDs are the only listed exceptions on CPU, and they must be
  identical on GPU.
- **Required evidence:**
  - `G1_r2` and `G2_r2` results, with the 24 rows' outcomes on both builds;
  - the re-staged lines with expected exits;
  - `REPORT_r2.md`.
- **Forbidden configuration:**
  - any edit to an existing test other than the 24 rows;
  - any product file change;
  - GPU use;
  - git writes;
  - deletes.

## Registration checklist (order-lint)

- Every answer is the F3 order's, unchanged, except as below.
- **L01:** pytest and python3, each run with `--version` first, and Build H's module checked against its sha.
- **L09:** G2 compares outcomes per test ID. Logs and timings are never compared.
- **L12:** inherited: F2's G2 per-ID identity, with the 24 listed exceptions on CPU only.
- **L16:** the cheapest pass would mark the 24 rows skipped. Excluded: each row must assert its exception type and
  message on both branches.
- **L32:** n/a.
- **L35:** A1.1 to A1.3 are the lead's amendments, registered here.
- **L38:** the exception is A1.1's 24 rows, and only them.
- **L41:** n/a. No new mutant; r1's three kills stand, and their module and test receipts are unchanged.
- **L47:** the expected failures are as r1 recorded them. The 24 rows' no-device outcome is the RuntimeError that
  r1's `test_f3_cpu_no_device_query` rows already observe on Build H.
- **L103:** the modules are Build G (`5fcd179b…`, built from `60e28eb7`) and Build H (`ea723385…`, built from r1's
  source tree, which `artifacts/eng3f3/` records against this branch). Each staged line refuses a module whose sha256
  differs, before the lease.
- **L56:** (a) tool slips go under Deviations; (b) at most 3 repairs of your own scripts, counted fresh for r2.
- **L59:** the amendment is committed before launch. r2's start record is the `SHIM-RUN` line of `logs/eng3f3_r2.log`.

## Done

As the F3 order's "Done", for r1 and r2 combined.
