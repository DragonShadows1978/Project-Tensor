# Execution amendment 1

Before any staged GPU dry run, replace the overly conservative 491-line draft with persistent isolated PF/F1 workers and up to 128 cases / 4e12 estimated scalar operations per G1 line. This changes process setup and line splits only: seeds, thread counts, full call shapes, output and mask comparisons, gates and thresholds are unchanged. The original draft files remain as *.superseded_v1. Each worker releases all per-case CUDA allocations and acknowledges before the other worker computes. Workers are supervised foreground subprocesses inside one lease; no background lease or detached wait.

Prior art: ordinary persistent worker / pipe request-response IPC and PF2 separate-arm imports (Project-Tensor 2026), reused. No new numerical algorithm. Bounded greedy batching is ordinary scheduling; no prior art known to me for this exact work estimate; unverified - lead to check bounded batching kernel work budget.
