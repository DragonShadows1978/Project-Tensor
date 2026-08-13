# APAMQ Ledger — APA × MQA Root-Cause Investigation

Append-only. Commands, file changes, results, failures, follow-up
decisions, as they happen. Plan: `docs/APA_MQA_ROOTCAUSE_PLAN.md`
(immutable).

## 2026-08-13 — Setup (lead)

- Recon: GPU idle (4070S, 257 MiB used). `kernels.cu:1053`
  TC_APA_MAXD = 512 ("bumped 256->512 for Gemma 4 global") — H-D
  registered from this.
- Plan + ledger + orders APAMQ_E1 (Project-Tensor) / APAMQ_E3
  (GraftRepository) authored and committed before dispatch.
- Dispatch: two Sol-max seats via codex-shim, parallel, flock GPU
  serialization. Sentinel waiters armed at dispatch.
- 21:15Z dispatched: E1 = SHIM-RUN 20260813T211541Z-791086
  (logs/apamq_e1_r1.log, Project-Tensor), E3 = SHIM-RUN
  20260813T211542Z-791603 (logs/apamq_e3_r1.log, GraftRepository).
  Both `-m gpt-5.6-sol`, SHIM_TIMEOUT_SECS=21600. Plan committed
  45ae082 (Project-Tensor) / order 4379e17 (GraftRepository) BEFORE
  dispatch.

## 2026-08-13 — E1 result (lead-run)

- E1 Sol seat: honest RED — codex sandbox exposes no GPU device
  (nvidia-smi driver fail, no /dev/nvidia*). Harness authored clean,
  0 fabricated cells (159 SKIPPED). Routing-ledger note appended.
- Lead ran the seat's harness under flock (logs/apamq_e1_leadrun.log):
  **160/160 cells OK, 0 OOM, 0 error.** Artifacts:
  artifacts/apamq_e1/{results.json,RESULTS.md}.
- **T1 / H-A: CONFIRMED.** Standard-path pool transient is IDENTICAL
  across kv ∈ {1,4,8,16} (D=128: 130→2050 MiB linear in S; D=512:
  136→2056 MiB), and fused-APA transient is IDENTICAL across kv and
  FLAT in S (2 MiB at D=128, 8 MiB at D=512, all S to 64K). Ratio
  deviation across kv ≈ 0%, far inside the ±20% threshold. The fused
  path's transient elimination (up to ~1025× at 64K prefill) is fully
  available at kv=1. KV-head count does not enter the memory
  economics. Evidence class: kernel sweep.
- SPEED (same sweep, secondary): fused is uniformly SLOWER than the
  cuBLAS standard path at these shapes — D=128 prefill ~5.8× at 64K
  (302 vs 52 ms), D=512 prefill ~16.6× (1031 vs 62 ms), decode
  D=512 kv=1 ~16× (9.3 vs 0.58 ms). Mechanism: the fused kernel
  computes dots on CUDA cores (warp-shuffle reductions); the standard
  path rides cuBLAS tensor cores. The O(L)-memory property is bought
  by giving up tensor cores. kv=1 is the FASTEST geometry for both
  paths at decode (smallest KV stream) — consistent with "MQA is the
  kernel's best case," refuting any kv=1 speed penalty.
