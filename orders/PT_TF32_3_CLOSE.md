# PT-TF32-3 — close the certification: downstream dK, the two edge cases, GEMM tensor-core dispatch, and a harness that does not fill the disk

Registered 2026-09-25 18:00 EDT by the lead (Fable 5.1). Seat: GPT-6 Astra at max. YOUR WRITABLE TARGET is the fork
worktree `/mnt/ForgeRealm/wt/pt-tf32` (branch tf32-fast-path). The live engine `/mnt/ForgeRealm/Project-Tensor` is never
edited, rebuilt or imported. No GPU: build + CPU + blocked-report; the lead runs the GPU gates.

## Slot-10 receipts (`artifacts/pt_tf32_2/lead_slot_01/`, read them)
The NVMe filled to 0 B during the slot (the harness wrote ≈ 7 GB of gradient/checkpoint dumps by default), so the
`healthy`, `control` and `step_time` lanes died with ENOSPC — invalid, not RED. Valid lanes:
- `gpu_units`: 2 failed / 54 passed — `test_native_padding_grouped_heads_and_fp32_storage[widths2-lengths3-True]` and
  `[widths2-lengths4-True]` (widths (96,64), lengths (17,31) and (33,35), causal, H 4 / KVH 2). Down from 7.
- attention cases 0 and 3: GREEN on every sub-gate under the TF32-appropriate registration; cases 1 and 2: GREEN on
  forward, isolated backward, same-native-state backward and selection, **RED on "downstream dK" only** (dK computed
  from the h_tf32-produced saved state) → the dK path is fixed in isolation but not when the forward state comes from
  h_tf32: look at what h_tf32 saves (lse / threshold / selection mask precision or layout) that g1_tf32 then consumes.
- GEMM: accuracy PASS, speed 1.18–1.33× SGEMM on every shape (RED vs ≥ 5×). The cuBLASLt TF32 dispatch is not reaching
  tensor cores. Verify with the real API: `cublasLtMatmulAlgoGetHeuristic` + `CUBLASLT_MATMUL_DESC_COMPUTE_TYPE =
  CUBLAS_COMPUTE_32F_FAST_TF32` (or `cublasSetMathMode(CUBLAS_TF32_TENSOR_OP_MATH)` on the legacy handle) and check the
  chosen algo's `CUBLASLT_ALGO_CAP_MATHMODE_IMPL`; also check alignment (16-byte) and transposition flags — a fallback to
  the non-tensor-core algo is silent. Provide a CPU-side self-check that prints the selected algo/math mode (the lead runs
  it in the slot before the timing).
- memcheck / racecheck / synccheck: 0 errors each (lane verdict RED only because the 2 failing unit cases exit non-zero).
- noise floor GREEN (bar 0.99843); onset GREEN.

## Build
1. Fix the downstream-dK path; fix or justify (from the TF32 rounding model) the two edge cases; make the TF32 GEMM
   dispatch reach tensor cores (≥ 5× at the model shapes); keep every default unchanged and the bf16 paths byte-identical.
2. Harness: gradient/checkpoint dumps OFF by default (`--keep-grads` opt-in); every lane checks free space ≥ 8 GB
   before starting and writes a BLOCKED receipt instead of running if not; the sanitizer lanes report their own error
   counts as the verdict (0 errors = GREEN) independent of the unit-test exit code; the slot runner's total dump size
   printed in `--print-sequence`.
3. Rebuild the fork engine in-tree; `docs/PT_TF32_3_LEDGER.md`, CPU tests, `## Done` with receipts verbatim + the
   blocked-report + the lead's slot sequence (≤ 30 min, incl. the re-run of healthy/control/step_time). Prior art at the
   code site + ledger + report.

## Boundaries (binding)
- Writable: `/mnt/ForgeRealm/wt/pt-tf32` only. Never `/mnt/ForgeRealm/Project-Tensor`, `/mnt/ForgeRealm/GRAPA-Native-LLM`,
  `/mnt/ForgeRealm/wt/grapa-*` (read-only), `/mnt/ForgeRealm/grapa_run*/`, `/mnt/ForgeRealm/grapa_side_*`. No GPU
  (`CUDA_VISIBLE_DEVICES=''`). No git. No subagents. Never kill processes you did not start; never touch
  `/tmp/forge-gpu.lock`. RED honesty.
