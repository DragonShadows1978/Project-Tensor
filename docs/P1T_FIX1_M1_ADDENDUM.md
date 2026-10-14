# P1T-M1 report addendum (P1T-FIX1): native Storage ABI change is a rollout prerequisite (review finding F07)

This addendum belongs with `docs/P1T_M1_REPORT.md`. It lives in its own file because the P1T-FIX1 order's scope gate (G5)
allows new docs only under `docs/P1T_FIX1_*.md`. The M1 report is unchanged.

## What changed in the native layout

P1T-M1 added two members to `tc::Storage` (`tensor_cuda/include/tc/core.h`). They are needed for DLPack import:
`ExternalRelease external_release` and `void* external_ctx`.
- The reviewer measured the CPU layout on x86-64 (`artifacts/p1t_rev/layout_baseline.txt`, `layout_current.txt`):
  `sizeof(Storage)` is **40 bytes at 9df5bef and 56 bytes now**. `revision` is still at offset 32.
- P1T-FIX1 adds no member and does not change the layout again. It adds the free function
  `storage_external_drain_leaks()` and a comment.

## Rollout prerequisite (the promotion note)

Every native client compiled against `tensor_cuda/include/tc/core.h` must be rebuilt together with the promoted core.
- An old client that allocates the 40-byte `Storage` and then reaches the new destructor would read `external_release`
  past the end of its object. That is undefined behaviour.
- **No such client is known.**
  - `tc_signal` does not include core headers, does not link `_tensor_cuda`, and talks to core only through DLPack
    (reviewer: R5 PASS).
  - The core Python package and `_tensor_cuda` are built from the same tree.
- Any out-of-tree C++/CUDA code that includes `tc/core.h` (none found by the reviewer's scan) needs a clean rebuild at
  promotion. Do not mix it with a pre-M1 object file.
- A full clean core rebuild (`cd tensor_cuda && ./build.sh <arch>`) has no mixed-layout risk.
- Alternative not taken: version or hide the native Storage ABI. That would be a core API change outside this order's
  scope.

Prior art: C++ object-layout ABI compatibility (the One Definition Rule; the Itanium C++ ABI class layout rules). The
rebuild requirement is the standard remedy. Nothing new.
