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
