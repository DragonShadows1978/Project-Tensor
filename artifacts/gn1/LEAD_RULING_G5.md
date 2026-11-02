# Lead ruling on GN1 G5, the GPU run (2026-11-01, window R4GN1): PASS

**Verdict: PASS.** The scorer accepts the root: `passed: true`, no errors. All 58 staged lines passed in device mode,
each with child-written device evidence.

## The run

- **Window:** GRAPA's R4GN1, after TX3-F2g run 4 and the TX3-F1M S049 re-run. The window opened at 22:40:39, with END
  23:40:33.
- **Wrapper:** `lead/gn1_g5.sh`. It ran the 58 labels in `gpu_lines.json` order, from 23:04:57 to 23:11:03, under one
  run root, `a8/gpu_runs/g5_1101/`.
  - Every label was invoked through `a8/STAGED_GPU_COMMANDS.sh` with `CUDA_VISIBLE_DEVICES=0`.
  - The audit, control and fixture selectors and `PYTHONOPTIMIZE` were unset, and the real governor ran.
  - Every invocation returned 0.
- **Manifest:** `post_run_manifest.sh` sealed the run root, 467 files (`lead/g5/gn1_g5_MANIFEST.sha256`).
- **Scoring:** `score_gpu.py` on CPU gave rc 0.
- **Environment record:** PATH, PYTHON\*, LD_\*, CUDA_\* and FORGE_GPU_\* (GN1-REV6 R6-O4), in
  `lead/g5/gn1_g5_env.redacted.txt`.

## Results (`lead/g5/G5_SUMMARY_compact.json`; full output held back, `lead/g5/LEAD_HELD_BACK_G5.txt`)

- **Root:** `passed: true`, `errors: []`.
- **Lines:** 58 of 58 passed, all with `mode: device` and the forge outcome "ran".
- **Device evidence,** recorded by the child (example: `candidate_focus/attempt-0001/child_result.json`, equal to its
  `invocation.json`): NVIDIA GeForce RTX 4070 SUPER, ordinal 0, CUDA runtime 12060, driver 13020,
  `CUDA_VISIBLE_DEVICES=0`.
- **Exits, as registered:**
  - 0 on the 5 lines registered 0: `candidate_focus`, `candidate_regression`, and the three equivalent mutants
    `E_npbool_c`, `E_npbool_g` and `V3`;
  - 1 on the 53 lines registered 1: `baseline_focus`, `baseline_regression`, and 51 killed mutants.
- **Agreement with CPU:** this matches A8 and GN1-REV6 exactly (54 mutants: 51 killed, 3 equivalent).

## Observation

- **A leftover temporary file:** `candidate_focus/attempt-0001/` holds a leftover `parent_result.*.tmp` beside the
  published `parent_result.json`. The published receipt is the one the scorer read and the manifest covers. The
  leftover is a publication temporary. It does not affect the verdict, and it is recorded for the tooling's next
  revision.

## What follows

- **GN1's evidence is complete:** the product fix (`cf922891`), mutation adequacy on CPU and GPU, the reviews (GN1-REV6
  ACCEPT under contract C), and this GPU run.
- **Promotion:** GN1 is infrastructure (Project-Tensor). Its promotion ticket needs the owner's approval in his own
  words. None is on record yet.
