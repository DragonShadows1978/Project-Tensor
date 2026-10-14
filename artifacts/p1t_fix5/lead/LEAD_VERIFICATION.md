# P1T-FIX5 lead verification (Project-Tensor lead, 2026-10-13, about 07:00 EDT)

All runs were fresh processes under `flock -w 900 /tmp/forge-gpu.lock`, on native driver 595.99.02.

- **Before:** the 1475fae build, in `wt_fix4`.
- **After:** this worktree's build. Both hashes match the seat's report:
  - core `9af40433f69e4d5582a4a83bb65165ee4e9f72cdcf35c28a7ad4f4cceca15510`;
  - `_tc_signal` `3324bb4a44dde6a45e62fcd91114d60ed1ef3268a2b35f7f1aebbe08c67a2552`.
- **Shims:** `/tmp/p1t_rev5_{query_once,fill_grid}.so`, built by the lead from the reviewer's sources with
  `gcc -shared -fPIC -O1 … -ldl`.

| Check | Result | Receipt |
|---|---|---|
| Scope | Every changed path is in the order's Exercises, plus `docs/P1T_FIX5_*`, `artifacts/p1t_fix5/**` and `logs/**`. | `git status` |
| Probe files | `artifacts/p1t_rev{,2,3,4,5}` are byte-identical to HEAD. | — |
| New core `__global__` | 0. The one grep hit is a comment in `kernels.cu` explaining why `fill_` refuses rather than adding a grid-stride kernel. | — |
| `exit_gate.py full`, no GPU | before rc 1 (child -6); after rc 0 | `probe_runs/{before,after}_exit_full_nogpu.log` |
| `exit_gate.py full`, GPU | before rc 1 (child -6); after rc 0 | `probe_runs/*_exit_full_gpu.log` |
| `exit_gate.py gpu` (deferred) | before rc 1 (child -6); after rc 0 | `probe_runs/*_exit_gpu.log` |
| `exit_gate.py gpu --control` | rc 0 on both | `probe_runs/*_exit_gpu_control.log` |
| `query_error_single_gpu.py` | before rc 1 (error lost); after rc 0 (reported once) | `probe_runs/*_query_error_single_gpu.log` |
| `fill_grid_single_gpu.py` | before rc 1 (not refused); after rc 0 (refused before launch) | `probe_runs/*_fill_grid_single_gpu.log` |
| Round-4 regressions, after | `core_exit_overrun` during / bounded rc 0 / rc 0; `addon_lifetime` rc 0 (12 trials); `core_edges` rc 0; `numerics_and_streams` rc 0; `core_import_query_error` (no GPU) rc 0; `sync_existing_error` rc 0 | `probe_runs/after_*.log` |
| FIX4 and FIX5 tests (`-k 'fix5 or fix4'`) | 76 passed | `probe_runs/after_fix45_tests.log` |
| Mutant (a), V5-01 wait removed (`/tmp/p1t_fix5_mut_a`, sha `099a3543…`) | `-k v501`: 11 failed / 5 passed, so KILLED. The seat reported 10; the 0.1 s case depends on timing. | `probe_runs/mut_a_v501_tests.log` |
| Voice runner, T1–T6 plus T1R and T4S | PASS 249/249 (T1 11, T2 144, T3 28, T4 12, T5 28, T6 13, T1R 1, T4S 12); adapter sha 600f38b4, unchanged | `voice_runner.log`, `p1t_all_lead.json` |
| Combined core plus tc_signal session | 37 failed, 2765 passed, 47 skipped, 15 errors (573 s). The compare finds 52 = 52 baseline ids, 0 new and 0 no longer failing: **G1 PASS**. | `combined_session.log`, `combined_compare.txt` |

## Code read by the lead

- **D7, core `dlpack_begin_shutdown`:** the explicit `PyEval_SaveThread` / `PyEval_RestoreThread` pair is in an
  ordinary frame. The catch-all re-takes the GIL and rethrows, and no Python runs inside its try block.
- **D1, the add-on `wait_for_other_threads`:** count-based, with the GIL released and doubling diagnostics.
  `synchronize()` now waits for releases running on other threads. This is accepted for review; the round-6 order
  asks the reviewer to judge it explicitly, together with residuals R2, R3 and R4.
- **V5-04:** `fill_` docstring documents the refusal above 549755813632 elements (2^31-1 blocks × 256).

## Verdict

FIX5 is verified, so the round-6 independent review can start on this commit.
