# APA-P1 GPU run 1 (2026-11-03, GRAPA's APA window): lead ruling

**Registered verdict: RED** (the staged script's `GPU verdict exit=1`). One of 51 lines failed, and it failed in the
scorer, not in the port:
- **P142 (bitwise, port against lab):** HELD, 6 of 6.
- **P143 (no production test outcome changes):** HELD on all 43 test files that define tests.
- **The failing line:** `reg_apa_tiled`. The file defines no tests, and the GPU driver scores pytest's "no tests
  collected" exit as a process failure.

## The run

- **Window:** GRAPA's APA window, go-file seen at 01:22:19, END 13:22:11. The lane wrapper is
  `tensor_fork_apa_wt_apa2p/lead/window_apa.sh` (99b6ca0f). DS1 ran first and stopped after 35 s (its own ruling,
  `tensor_fork_apa_wt_apa2p/lead/LEAD_RULING_DS1_GPU1.md`).
- **P1 lane:** `artifacts/apa_p1/STAGED_GPU_LINES.sh` as staged at d41cf833, from 01:22:54 to 01:41:40, exit 1. Run
  root `artifacts/apa_p1/gpu_20261103T062254Z/`.
- **Post-run seal:** 6,673 files plus 62 symlinks (`lead/gpu1/P1_MANIFEST_gpu_20261103T062254Z.sha256`, sha256
  `cde97d72…dcb4`, and its `.symlinks` list). The lead re-checked it against disk before this commit with
  `sha256sum -c --strict`: rc 0.
- **Handback:** the wrapper wrote `TENSOR_DONE_1103_APA` at 01:41:43, as designed, after both lanes.

## Lines (`lead/gpu1/P1_lines.out`)

- **Bitwise, port against lab (G4.1):** `bitwise_17b_{2048,8192,32768}` and `bitwise_4b_{2048,8192,32768}` each exit 0,
  with `comparison.json` `status: PASS`. Every recorded field is equal: append codes and norms at every history point,
  and the SP (EB3) and SP-tk8 (EB4) outputs, masks and flags.
- **Bytes (G4.3):** `bytes` exit 0.
- **Production regression on GPU (G4.2):** 44 test files. 43 give `status: PASS` with `differences: 0`.
  `reg_apa_tiled` gives `status: FAIL_PYTEST_PROCESS`.

## The failing line

- **What it ran:** `tensor_cuda/tests/test_apa_tiled.py` under production's module first. pytest exited 5 ("no tests
  ran"), in 0.41 s.
- **Why the line failed:** `artifacts/apa_p1/gpu_driver.py:56` treats any pytest exit other than 0 or 1 as a process
  failure, so the port arm never ran.
- **What the file is:** a superseded stub. Its docstring points to `test_apa_selective.py`. It carries a module-level
  `pytestmark = pytest.mark.skip(...)` and defines no test function or class (`grep -c '^\s*(def test|class Test)'`
  gives 0). It imports only `pytest`.
- **The port's version is the same file:** `git diff --stat 0e1dea3 HEAD -- tensor_cuda/tests/test_apa_tiled.py` is
  empty. Neither arm can collect a test ID from it, whatever module is loaded.
- **So:** G4.2's registered comparison ("the same per-ID outcomes") holds for this file with the empty ID set on both
  sides. The exit-1 verdict comes from the driver's scoring rule, not from any outcome of the port.

## Ruling

- **Registered verdict: RED, reported as registered.** The cause is a scorer defect: an honest run of a pinned test file
  can collect zero tests, and the driver must score that outcome rather than fail on it (contract C, lesson L73). It is
  not a product defect.
- **Predictions:**
  - **P142:** HELD (6 of 6 bitwise).
  - **P143:** HELD on 43 files with zero differences. On `test_apa_tiled.py` it holds vacuously, by the static argument
    above.
- **Next, amendment A1 (seat, CPU):**
  - the driver scores a zero-collection file by comparing the collected ID sets of both arms, with an empty set on
    both sides counted as equal;
  - that file's port arm is run once (no test executes, so no kernel launches);
  - the verdict is re-scored from this run's receipts, from the tool's own directory with `--out` (L08). No GPU rerun is
    needed: the other 50 lines are complete and sealed.
- **After A1:** the promotion items S3 (per-device and stream tickets, the early-counter test and the exact-tie test),
  a blind cross-model red team, making packed storage the shipped default, a caller check, and then the owner's
  approval.

## Receipts (this commit)

- `lead/gpu1/`: the lane output, the window log, and the post-run manifest with its symlink list.
- From the run root, the verdict-bearing files only: every `*.exit.json`, the line logs, and each line's
  `comparison.json` and per-arm `production.json` / `port.json` (240 files, 4.4 MB).
- The per-arm pytest logs (103 files), except two that are held back because a library warning in them prints a
  local home path: `reg_apa_sp3_a4/port.log` (1,188 bytes, sha256 `ba8fbdff…6b6f`) and
  `reg_apa_sp3_a4/production.log` (1,188 bytes, sha256 `5f72476a…b872`). Both are listed in the manifest.
- The rest of the run root (2.0 GB: the lab and port arrays, and pytest temporary trees) is held back. The manifest is
  its inventory.

## Prior art

No new algorithm. pytest's exit-code convention (5 = no tests collected); sha256 manifests.
