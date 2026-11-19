# Execution amendment 2

Raise admission minima to the payload class: G1 and memory 500 s, coverage 120 s, mutants/refusals 60 s, non-finite 30 s. All max values remain 540 s; payload timeout 500 s; G1 workers 460 s. The draft fixed --min 30 could admit a long line in a short window. No gate, case, shape or acceptance threshold changes. Retain earlier pins and dry runs as *.superseded_min30 and gpu_superseded_min30. Re-run every final staged line in dry-run mode.

Prior art: forge-gpu-run bounded admission (Project-Tensor 2026), reused; no new algorithm.
