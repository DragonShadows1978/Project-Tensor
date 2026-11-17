# APA-P1-S3 r2b GPU lines, lead ruling (2026-11-14): all six as registered

The lead ran the six staged lines of `artifacts/apa_p1_s3_r2/STAGED_GPU_LINES.sh`, individually and verbatim:
- the first four, with `lead/run_p1_s3_gpu.sh` (05:48 to 05:56);
- the two tie lines again, with `lead/run_p1_s3_gpu_ties.sh` (06:27 to 06:31). Their first attempt got
  `forge-gpu-run` exit 76: the lock was not acquired while GRAPA held the GPU, so nothing ran. They were re-run
  unchanged.

The staged file's sha256 was checked before each run: `976504d6…db7f`.

| Line | Expected | Exit | Evidence (`lead_gpu/<case>/`) |
|---|---|---|---|
| threads | 0 | 0 | 1 passed |
| early_correct | 0 | 0 | 2 passed (eb3, eb4) |
| eb3_early | 1 | 1 | AssertionError: deterministic poisoned-partial output mismatch |
| eb4_early | 1 | 1 | AssertionError: deterministic poisoned-partial output mismatch |
| tie_correct | 0 | 0 | 1 passed |
| eb4_reverse_tie | 1 | 1 | AssertionError: exact-tie lower-index winner mismatch |

The per-line exits come from the wrappers' logs (held back as logs; sha256 in `lead/S3_GPU_1114_HELD_BACK.txt`), and
match each case's `RESULT.json` (`test_exit` equals `expected_exit`).

**Still open for P1, from the r2 ruling:**
- the default build's SASS identity, with hooks off;
- the default-switch round;
- one blind red team;
- the promotion ticket, with the owner's approval.
