# P1T-REV2 ledger

- Read mirror and share house rules, project/worktree AGENTS and immutable order. No subagents; review paths only.
- Read mechanism round-1 report, contract and approved alternative plan; no scope mismatch.
- Read working HEAD 5368e547 and byte-compared 44 changed files with d116ac0; all match: True. Saved full and FIX1 patches and hashes.
- Registration written before any gate. Preliminary reasoning: finalization check still precedes GIL acquisition without lifecycle coordination; synchronization ignores cudaGetDevice failure with an empty list; inspect before classification.
- Prior art: CPython 3.12 runtime finalization and PyGILState_Ensure; CUDA 12.6 API error propagation; DLPack dmlc 2017+ lifetime.
- CPU core helper first compile refused missing iostream in probe driver; corrected probe only, preserved log. No reviewed source changed.
