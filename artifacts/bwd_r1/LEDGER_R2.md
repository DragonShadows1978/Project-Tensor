# BWD-R1 r2 implementation ledger

- 2026-11-18T21:20:01Z. Recorded Python 3.12.3, saved the three harness sources and protected r1 receipt hashes, and sealed REGISTRATION_R2.md with the smoke script before running checks.
- Diagnosis from the lead log and the path census. Under Python -P, the gate script directory is omitted. PROD PYTHONPATH supplies its frozen tree scripts, where the forward helper is absent.
- Prior art. CPython -P (3.11, 2022) defines safe script path handling. CPython sys.setprofile (3.12 documentation, 2023) supplies pre-body call events. The BWD-R1 changes are explicit sibling resolution after engine pinning, a stop at the first tensor call, and versioned output paths using the existing launcher pattern. Project-Tensor L79 and System-Claude GW3 (2026) supply the existing dry-run contract. Unverified — lead to check official Python -P option and sys.setprofile call event documentation, and System-Claude GPU_WINDOW.md GW3.
- 2026-11-18T21:20:01Z. Preflight returned 78 with would_start true. See logs/bwd_r1_r2_preflight.log.
- 2026-11-18T21:20:02Z. Pre-edit smoke loaded the pinned PROD engine and reproduced the exact missing-helper traceback at the gate import, exit 1. See logs/bwd_r1_r2_import_before.log.
- 2026-11-18T21:20:51Z. Patched the forward helper search and isolated the three restaged launchers’ log, receipt and scratch paths. Captured HARNESS_R2.diff and sealed STAGED_LINES_R2.md with its scripts, pins, tolerances and strict prerequisite.
- 2026-11-18T21:21:01Z. Post-edit CPU import check returned 0. See logs/bwd_r1_r2_import.log.
- 2026-11-18T21:21:13Z. G4-forward dry launcher exit 78. See gpu_dry_G4-forward_R2.json and logs/bwd_r1_r2_gpu_dry_G4-forward.log.
- 2026-11-18T21:21:13Z. G5-m04 dry launcher exit 78. See gpu_dry_G5-m04_R2.json and logs/bwd_r1_r2_gpu_dry_G5-m04.log.
- 2026-11-18T21:21:13Z. G5-m05 dry launcher exit 78. See gpu_dry_G5-m05_R2.json and logs/bwd_r1_r2_gpu_dry_G5-m05.log.
- 2026-11-18T21:22:17Z. Verified all three dry-run payloads against their log bytes, r1 argv and module hashes. Only the round-specific run labels and scratch paths changed in the staged environments. Registration and stage seals passed, and all 32 protected r1 file hashes matched the entry snapshot.
- Deviations. Initial short discovery reads, including the Python version probe, lacked the inherited timeout wrapper. The preflight, CPU checks and staged invocations were all wrapped and run in the foreground. The first read-only receipt validation used the wrong argv offset and raised AssertionError. The corrected validator used the -- delimiter and passed on the same receipts. Zero repairs to the staged or smoke scripts.
