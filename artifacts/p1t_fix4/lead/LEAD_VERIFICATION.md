# P1T-FIX4 lead verification (Project-Tensor lead, 2026-10-13 04:14–04:40 EDT)

All runs are fresh processes under `flock -w 900 /tmp/forge-gpu.lock`, on native driver 595.99.02.
- "Before" = the 62c8d9f build in `wt_fix3`.
- "After" = this worktree's build:
  - core `_tensor_cuda` sha256 `1da017ca2ec917019acc1ae9220fca87a74594257c053050056ed55eb2cd9b48`;
  - `_tc_signal` sha256 `909af0c5e0c76cfd16c4016abb72b35bdde5a01cc96505334c8cc36876426352`.

  Both hashes match the seat's report.

| Check | Result | Receipt |
|---|---|---|
| Scope | Every changed path is inside the order's Exercises, plus `docs/P1T_FIX4_*`, `artifacts/p1t_fix4/**` and `logs/**`. | `git status` |
| Probe files | `artifacts/p1t_rev{,2,3,4}` are byte-identical to HEAD (`git diff --quiet`). | — |
| New core `__global__` | 0 (vs 9df5bef and vs 62c8d9f). | — |
| `core_import_query_error.py` (`CUDA_VISIBLE_DEVICES=`) | before rc 1 (`used_dltensor`); after rc 0 PASS | `probe_runs/{before,after}_import_query_error.log` |
| `sync_existing_error.py` | before rc 1 (slot 98); after rc 0 PASS | `probe_runs/{before,after}_sync_existing_error.log` |
| `core_export_device.py` | rc 3 INCONCLUSIVE (one GPU) | `probe_runs/after_export_device.log` |
| `core_exit_overrun.py` during / bounded | rc 0 / rc 0. Bounded: the wait was 5.50 s, the deleter completed, in_flight 0, one diagnostic. | `probe_runs/after_exit_overrun_*.log` |
| `addon_lifetime.py` | rc 0, 12 trials correct | `probe_runs/after_addon_lifetime.log` |
| `core_edges.py`, `numerics_and_streams.py` | rc 0, rc 0 | `probe_runs/after_core_edges.log`, `after_numerics.log` |
| FIX4 tests (`-k fix4`, core plus tc_signal) | 40 passed | `probe_runs/after_fix4_tests.log` |
| Mutant (a), V4-01 guard removed (`/tmp/p1t_fix4_mut_a`, sha `bfc62122…`) | 8 failed / 14 passed. The trace `create@1 record@1 wait@1 destroy@1` shows the fence on the wrong device, so the mutant is KILLED. | `probe_runs/mut_a_fix4_tests.log` |
| Voice runner T1–T6 plus T1R, T4S | PASS 249/249 (T1 11, T2 144, T3 28, T4 12, T5 28, T6 13, T1R 1, T4S 12), 74.4 s | `voice_runner.log`, `p1t_T1toT6_T1R_T4S.json` |
| Combined core plus tc_signal session | 37 failed, 2729 passed, 47 skipped, 15 errors. The compare finds 52 = 52 baseline ids, 0 new and 0 no longer failing: **G1 PASS**. | `combined_session.log`, `combined_compare.txt` |

## Code read by the lead

- **`cuda_storage_device_mismatch`:** zero-size tensors and null pointers pass, so a zero-size export is not refused.
- **`tensor_empty`:**
  - the device is a checked `cudaGetDevice`;
  - `cuda:N` must name the current device;
  - the error-slot entry of a failed allocation is consumed.

## Verdict

FIX4 is verified, so the round-5 independent review can start on this commit.
