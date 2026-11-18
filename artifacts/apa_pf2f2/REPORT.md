# APA-PF2-F2 r2, Amendment 1

**Build and CPU gates PASS. GPU acceptance pending.** Both reachable
packed-decode loads now convert raw non-finite norms to a quiet NaN.
F2 and both revert mutants built and loaded. SASS changed in exactly the
12 expected decode instantiations. The 62 lead GPU lines are staged and
dry-verified. GPU behavior is not claimed fixed until the registered lead
gates pass. No CUDA kernel ran in this seat.

## Dispatch and change

Source inspection and the full 13-hit operand census are in
DISPATCH_TRACE.json and OPERAND_CENSUS.txt. The public FP32 TQ4 partition
route reaches bulk_shared at G=4 or G=8 and bulk at every other valid G.
There are four public grouped D/G instantiations and four public bulk
CAP/W instantiations. Native r2 BF16 and INT4 share the load sites. INT4
adds four bulk_shared TQ=false instantiations. The inherited raw +Inf
failure reproduced on 16 lead F1 GPU cases, eight for each load. Its
verified log SHA-256 is
`30e54788ecbd485eac2bb38827301ac1d82671c20226f932eaf48ed8851ca1e5`.

CHANGES.diff is the complete unified product and test diff. The shared
guard follows the lane broadcast and the bulk guard precedes the shared
array write. The finite leg returns the loaded float. The regression file
adds 44 partition-carry cases and defaults to the F2 staged arm when it is
available. The F1 report and staging hold are preserved as REPORT_R1.md
and STAGED_LINES_R1.md.

## Module SHA-256s

| Build | SHA-256 |
|---|---|
| F1 | `641a3139181cdee3e800f62afad08c0881b3018c26e57d953fa6c13413f7cbe7` |
| F2 | `aefe2513a4d33d7310421f65e6619f0706de2889410836bda42193053f8ec97a` |
| pf2m01_bulk_norm_finite_only | `fd0c033ec1d924048d764bfb70c3a8087d78f62b7f433af76a4e3969830123a6` |
| pf2m02_shared_norm_finite_only | `8a2fe35fbafc1613770f3fa599486e1b60b4003a55147bb681b13021a2608e5b` |

## CPU results

| Evidence | Result |
|---|---|
| F2 build | Exit 0 under CMake 3.28.3, CUDA 12.6.85, sm_89, -j3, nice 19 |
| Incremental revert builds | Both exit 0. One rebuilt CUDA object each, kernels.cu.o, followed by device-link and link |
| Device-hidden module imports | F1, F2 and both mutants matched their pins |
| Extracted host load expressions | 90/90 expected bit patterns in F2 and both mutants |
| Pytest | 1 passed, 76 skipped in 0.75s |
| F1 → F2 SASS | 801 unchanged, 12 changed, 0 added, 0 removed |
| F2 → bulk revert SASS | 809 unchanged, 4 changed, 0 added, 0 removed |
| F2 → grouped revert SASS | 805 unchanged, 8 changed, 0 added, 0 removed |
| Final GPU admission dry runs | 62/62 exit 78, would_start true, zero lease attempts |
| G5 scope | Zero unauthorized paths and zero missing worktree files |

CPU_RESULTS.json indexes the host, SASS, build and scope receipts.
The GPU portions of G1–G4 require the lead. The compiled guard and SASS
prove the edit footprint. GPU bitwise equality and row sets remain
unmeasured on F2.

## Mutants

| Mutant | Reverted load | CPU | Staged killing test |
|---|---|---|---|
| pf2m01_bulk_norm_finite_only | packed.cuh:203 | PASS. Only the four bulk instantiations change | G3_bulk. Eight G=1 REV1 cases must fail only raw_inf_decode_row_sets. G=4 is the alternate control |
| pf2m02_shared_norm_finite_only | packed.cuh:149 | PASS. Only the eight bulk_shared instantiations change | G3_bulk_shared. Eight G=4 REV1 cases must fail only raw_inf_decode_row_sets. G=1 is the alternate control |

## GPU launchers

STAGED_LINES.md contains the ordered lead commands. LINES.json pins every
case split and arm. DRY_RUNS.json records exit 78, would_start true and log
SHA-256 for each final pre-lease check.

## Deviations

- One own-script repair of three. The G4 FLT_MAX r02 adapter needed an
  explicit TensorCUDA-to-NumPy conversion. The first dry runs predated that
  correction. Retained old pins and receipts, re-pinned the adapter and
  repeated every dry run. See AMENDMENT_EXECUTION_001.md.
- An optional device-hidden host tensor construction probe failed with
  `RuntimeError: CUDA error at from_host: no CUDA-capable device is detected`.
  The adapter was checked in source. Its execution remains part of lead G4.
- A torch metadata probe failed with
  `SyntaxError: unexpected character after line continuation character`.
  The corrected metadata read confirms torch 2.11.0.
- The launcher appended to logs/apa_pf2f2_r2.log concurrently. The seat
  read that log, and its only direct log write was the final dry-run transcript.

## Prior art

IEEE 754-2008 quiet-NaN propagation and Project-Tensor PF2-F1 (2026).
This change reuses PF2-F1's finite check and 0x7fc00000 sentinel at the two
decode loads found by the dispatch census. External clause lookup
unverified — lead to check `IEEE 754-2008 6.2 quiet NaN multiplication`.
The tests carry forward P1-F3 and PF2-REV1 operands and PF2-F1's normalized
SASS comparison. The isolated site reverts follow DeMillo, Lipton and
Sayward (1978) mutation testing.

## Every file written

FILES_WRITTEN.txt lists the product and test changes, new helpers, generated
build and arm files, SASS dumps, both generations of dry receipts,
registration, report and scratch. FILES_WRITTEN.json includes per-file
SHA-256s for the principal deliverables and classifies the launcher-owned
log separately. All listed scratch was retained.
