#!/usr/bin/env bash
# Prior art: GNU timeout, POSIX shell; GN1 L45 completed-child supervision.
# Expected parent exit for EVERY run_line is 0. Child exits/bounds below.
# Usage: RUNS=/explicit/fresh/run/root timeout 1180s bash STAGED_GPU_COMMANDS.sh LABEL
# Repeat with the SAME RUNS for each label (each command stays below 20 minutes).
# CPU plumbing: RUNS=/explicit/audit/root timeout 120s bash STAGED_GPU_COMMANDS.sh --audit
# No RUNS default. Full execution is the lead's G5 after cross-model review.
set -euo pipefail
: "${RUNS:?RUNS is required; use a fresh run root and reuse it across the 18 labels}"
[[ "$RUNS" == /* ]] || { echo "RUNS must be absolute" >&2; exit 64; }
TOOL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$TOOL"
export RUNS
export TMPDIR="$RUNS/tmp" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
GPU_RUN=forge-gpu-run
label="${1:?select one registered label, or --audit}"
if [[ "$label" == --audit ]]; then
    export GN1_AUDIT=1 CUDA_VISIBLE_DEVICES=
    GPU_RUN="$TOOL/forge_gpu_run_standin.py"
    label=candidate_focus
else
    unset GN1_AUDIT
fi
mkdir -p -- "$TMPDIR"
timeout 20s python3 start_gpu_run.py
case "$label" in
  baseline_focus) # expected child 1, parent 0; inner 240s, --max 300s, outer 330s
    timeout 350s python3 run_gpu_line.py baseline_focus 1 330 "$GPU_RUN" --min 250 --max 300 --label gn1-baseline_focus -- timeout --kill-after=10s 240s python3 run_gpu_tests.py baseline focus
    ;;
  candidate_focus) # expected child 0, parent 0; inner 240s, --max 300s, outer 330s
    timeout 350s python3 run_gpu_line.py candidate_focus 0 330 "$GPU_RUN" --min 250 --max 300 --label gn1-candidate_focus -- timeout --kill-after=10s 240s python3 run_gpu_tests.py candidate focus
    ;;
  baseline_regression) # expected child 1, parent 0; inner 840s, --max 900s, outer 930s
    timeout 950s python3 run_gpu_line.py baseline_regression 1 930 "$GPU_RUN" --min 850 --max 900 --label gn1-baseline_regression -- timeout --kill-after=10s 840s python3 run_gpu_tests.py baseline regression
    ;;
  candidate_regression) # expected child 0, parent 0; inner 840s, --max 900s, outer 930s
    timeout 950s python3 run_gpu_line.py candidate_regression 0 930 "$GPU_RUN" --min 850 --max 900 --label gn1-candidate_regression -- timeout --kill-after=10s 840s python3 run_gpu_tests.py candidate regression
    ;;
  M01) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M01 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M01 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M01 mutation
    ;;
  M02) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M02 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M02 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M02 mutation
    ;;
  M03) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M03 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M03 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M03 mutation
    ;;
  M04) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M04 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M04 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M04 mutation
    ;;
  M05) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M05 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M05 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M05 mutation
    ;;
  M06) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M06 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M06 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M06 mutation
    ;;
  M07) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M07 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M07 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M07 mutation
    ;;
  M08) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M08 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M08 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M08 mutation
    ;;
  M09) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M09 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M09 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M09 mutation
    ;;
  M10) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M10 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M10 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M10 mutation
    ;;
  M11) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M11 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M11 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M11 mutation
    ;;
  M12) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M12 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M12 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M12 mutation
    ;;
  M13) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M13 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M13 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M13 mutation
    ;;
  M14) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M14 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M14 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M14 mutation
    ;;
  *) echo "unknown GN1 label: $label" >&2; exit 64 ;;
esac
# After all 18 labels: RUNS=/same/root timeout 20s python3 score_gpu.py (expected 0).
