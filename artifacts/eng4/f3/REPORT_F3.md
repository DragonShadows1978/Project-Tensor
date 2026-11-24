<!-- Filed by the lead from the seat's final message (the harness refused the seat's own write). Content unchanged
except where marked LEAD; rows and bullets merged for length. Receipts: LEDGER_F3.md. -->

# ENG4-F3 report — both new zero fills check cudaMemset's own status (REV3-F01)

Seat: Claude Opus 5.5 (Agent tool). Order `orders/ENG4_F3_REV3_FILL_STATUS.md` at 251588034 (sha256 f5404764…).
Worktree `tensor_fork_apa_wt_eng4f3`. Nothing was written in `tensor_fork_apa_wt_eng4`; the F2 module was read from it by
path and checked by sha256.

> **LEAD verification (2026-11-23 10:5x EST):** diff read (helper `zeros_checked` checks `cudaMemset`'s return value,
> makes no CUDA call for zero bytes, never reads the last-error slot; both sites use it; `NDArray::zeros` unchanged);
> F3 module `f005d765` installed in this worktree; CPU re-run from `tensor_cuda/` (`test_eng4_matmul.py` plus the BWD
> matmul and native tests): 160 passed, 98 skipped (GPU); K3 receipt 4,038 IDs identical F2 vs F3.

## K1 to K7

- **K1 PASS:** CPU, CUDA hidden, on F3: 157 passed, 83 GPU skipped (`f3/k1/K1_F3.xml`); F2 gives the same 157/83, and
  every F2 pass passes on F3 (host K = 0, host empty-batch zero gradient, new `test_k1_empty_b_no_fill`).
- **K2 PASS:** clean checkout (`git archive` of 251588034 minus `scripts/eng4_*` and `artifacts/eng4`, plus F3's four
  files and module): 157 passed, 83 skipped; outside the package it opened only `pyproject.toml` and the test file.
- **K3 PASS:** CPU suite per ID, F2 vs F3: 4038 IDs each, 0 differences (`f3/K3_CPU_COMPARE.json`).
- **K4:** a witness exists; STAGED as `pending_F2_F3`.
- **K5 (as registered):** `m11_f3_status_ignored` UNKILLABLE BY TEST — built and run; survived (157 passed).
- **K6 STAGED:** 3 lines, dry run 78 each.
- **K7 PASS:** writes only in the writable set; no GPU use of any kind (no line, no step function, no compute-sanitizer,
  nothing that removes `CUDA_VISIBLE_DEVICES`); no git writes; nothing deleted.

## Diff (`f3/F3_source.diff` vs 9b7eb1796, sha256 dfa931c2…; 7 new scripts as new-file hunks)

- `matmul.cu` Z1: `zeros_checked(shape, dtype, device, where)` beside the forward — allocates with the same
  `NDArray(shape, dtype, device)` as `NDArray::zeros`; zero bytes: no fill, no CUDA call; host: `std::memset`; CUDA:
  `cudaMemset`, raising on a non-success return status as "<where>: cudaMemset failed: <name>: <string>"; never reads
  or clears the last-error slot. `core.h`: its declaration.
- `matmul.cu` Z2: K = 0 path is `return zeros_checked(plan.out_shape, a.dtype, a.device, "matmul zero fill");`.
- `ops.cpp` Z2: the empty-batch branch uses `zeros_checked(..., "matmul backward zero gradient")`. Both F2
  `cuda_check_last` calls are gone; an empty B makes no CUDA call.
- Tests: `test_k1_empty_b_no_fill`, `test_gpu_k4_pending_error_not_raised_by_fill`.
- L55: both sites pass the same shape, dtype and device as in F2; `zeros_checked` has exactly these two callers
  (`f3/l55/L55_call_sites.txt`). `NDArray::zeros` and all other core members unchanged.

## Module sha256s

| Build | sha256 |
|---|---|
| F3 | f005d765e3aed86b537c9660bea4a2bfffc86a121d724c6d7086ec399baeae96 (installed in eng4f3) |
| m11_f3 | 6624bc64cae5d350d4a61eb6edd44917fa558ca0b191283ad8ec65de91e8e604 |
| F2 (copied, checked) | c913e8d9e1eff88e6cbc0f68f6f2d03076baa022cc6addd92843215dea7fbd32 |

## Mutant (`f3/MUTANTS_F3.json`, sha256 c3c5821a…, written before the m11 tree and build)

`m11_f3_status_ignored` (`zeros_checked` ignores `cudaMemset`'s status) is registered UNKILLABLE BY TEST: it differs
from F3 only when `cudaMemset` fails on an allocation the helper just made. A non-sticky earlier error is reported
through the slot, not by a later successful `cudaMemset`, and neither F3 nor m11 reads the slot, so K4's witness
returns zeros on both. A fill on a fresh valid allocation fails only under a sticky context error (a device fault or
lost device), which no public call produces without a memory-unsafe operation; forcing one on the shared GPU is outside
the order. Run (`f3/MUTANT_RESULTS_F3.json`): 157 passed — survived as registered (its line is in the CUDA branch).

## K4 finding

Yes, through the public API, by reading the code: `tensor_cuda.empty` consumes a failed allocation's slot entry
(`tensor_empty`, P1T-FIX4). The forward's own output allocation does not: `Storage::Storage` retries a failed
`cudaMalloc` once, clearing the slot before the retry, then throws "cudaMalloc failed" without consuming the retry's
entry. So a K = 0 matmul of zero-element CUDA operands (2^19, 0) × (0, 2^19), needing a 1 TiB output, fails at once,
takes no memory, and leaves `cudaErrorMemoryAllocation` (non-sticky) pending; building zero-element operands makes no
CUDA call. Line `pending_F2_F3` (fresh child processes per build and variant):
- forward (gated): with that error pending, a small K = 0 matmul raises "CUDA error at matmul zero fill: out of memory"
  on F2 and returns zeros on F3; on F3 the stale error surfaces at the next general check, `tc.synchronize()`;
- gradient (recorded, same rule): F2 raises under "matmul backward zero gradient"; F3 gives B zeros.
`own_F3` also carries a GPU test of the F3 side.

## Staged lines (`artifacts/eng4/f3/GPU_LINES_F3.json`, sha256 400db7dd…)

From the eng4f3 root: `python3 -B -P scripts/eng4_f3_gpu_line.py run <line>` (inner
`forge-gpu-run --min <s> --max 540 --label eng4f3-<line> -- timeout 480s python3 -B -P scripts/eng4_f3_gpu_line.py step <line>`);
receipts in `artifacts/eng4/f3/gpu/<line>/`. Lines: `own_F3`, `g3z_F3` (39 zero-size G3 cells + 24 empty-batch gradients
vs NumPy and zeros), `pending_F2_F3` (K4); each `--min` 120, peak host 3 GiB, dry run 78. Each step refuses (exit 7)
unless forge-gpu-run is an ancestor. CPU afterwards:
`python3 -B -P scripts/eng4_f3_g3z.py verdict artifacts/eng4/f3/gpu/g3z_F3/g3z` and
`python3 -B -P scripts/eng4_f3_pending.py verdict artifacts/eng4/f3/gpu/pending_F2_F3/pending`.
L48: `f3/POST_DEVICE_STEPS.json` 23 steps, all at their expected labels.

## L58 census (`grep -n CUDA_VISIBLE_DEVICES scripts/eng4_f3_*.py`)

`eng4_f3_gpu_line.py:199` `step_env()` pops it — the only removal site, reached only after `lease_guard()`;
`eng4_f3_gpu_line.py:142` (`run`) and `stage` pass the environment through. Comments or reads only:
`eng4_f3_gpu_line.py:17/125/195`, `eng4_f3_pending.py:8/88`, `eng4_f3_postdevice.py:4/6/155`, `eng4_f3_suite.py:177`,
`eng4_f3_g3z.py:10`. No site: `eng4_f3_make_arms.py`, `eng4_f3_g4_compare.py`; scripts run unchanged (`eng4_build.py`
reads at 85; `eng4_g3.py`, `eng4_f2_bwd0.py`) have no removal site. No line runs compute-sanitizer.

## Deviations

1. REPORT_F3.md refused by the harness. 2. `scripts/eng4_build.py` (unchanged) gives its scope `TMPDIR=/tmp/eng4_tmp`, so
compiler scratch went there. 3. Memory-hook holds on Write: four scripts retried, content unchanged. 4. K3 shows 375
errors where ENG4's runs showed 15: this worktree lacks the eng4 worktree's gitignored test fixtures; outcomes are
identical per ID across F2 and F3.

## Prior art

CUDA Runtime API: per-call return status and the separate per-thread last-error slot that `cudaGetLastError` resets
(CUDA 12.6 Runtime API). This repo: `ops.cpp` `check_backward_cuda`; ENG4-F2's lease guard and post-device harness.
NumPy `matmul` on zero-size operands. Mutation testing (DeMillo, Lipton and Sayward, 1978). Ours: the helper, its two
call sites, the public-API route to a pending error.
