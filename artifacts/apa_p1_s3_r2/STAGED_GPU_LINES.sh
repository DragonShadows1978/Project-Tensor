#!/usr/bin/env bash
# Lead GPU lines, execute individually from: /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod
# EXPECTED.json was written first. All CPU dry runs require FORGE_GPU_DRY_RUN=1.
# Correct lines exit 0; registered mutants exit 1 only for their named assertion. Harness errors exit 2.
# threads: expected 0 (correct build)
forge-gpu-run --min 30 --max 240 -- timeout 210s python -B artifacts/apa_p1_s3_r2/gpu_line.py --case threads
# early_correct: expected 0 (correct build)
forge-gpu-run --min 30 --max 240 -- timeout 210s python -B artifacts/apa_p1_s3_r2/gpu_line.py --case early_correct
# eb3_early: expected 1; deterministic poisoned-partial output mismatch
forge-gpu-run --min 30 --max 590 -- timeout 560s python -B artifacts/apa_p1_s3_r2/gpu_line.py --case eb3_early
# eb4_early: expected 1; deterministic poisoned-partial output mismatch
forge-gpu-run --min 30 --max 590 -- timeout 560s python -B artifacts/apa_p1_s3_r2/gpu_line.py --case eb4_early
# tie_correct: expected 0 (correct build)
forge-gpu-run --min 30 --max 240 -- timeout 210s python -B artifacts/apa_p1_s3_r2/gpu_line.py --case tie_correct
# eb4_reverse_tie: expected 1; exact-tie lower-index winner mismatch
forge-gpu-run --min 30 --max 590 -- timeout 560s python -B artifacts/apa_p1_s3_r2/gpu_line.py --case eb4_reverse_tie
