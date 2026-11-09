#!/usr/bin/env bash
# ENG1 staged GPU lines (adapted from GN1 A8; the lead runs them; the seat never did on a device).
# Prior art: GNU timeout, POSIX shell; GN1 L45 completed-child supervision (parent receipt after child exit).
# Expected parent exit for EVERY line is 0 (the parent passes when the child exit matches the registration
# and the receipts check out). Child exits and bounds are listed per label below and in gpu_lines.json.
# Usage: RUNS=/abs/path/under/artifacts/eng1/tool/gpu_runs/<root> TMPDIR=/tmp/eng1_g5 \
#        CUDA_VISIBLE_DEVICES=0 timeout 1190s bash STAGED_GPU_COMMANDS.sh LABEL
# Run the labels in gpu_lines.json order with the SAME RUNS (mutants need a passing candidate_eng1;
# candidate_regression needs baseline_regression, candidate_terrain needs baseline_terrain).
# Afterwards: RUNS=<same> CUDA_VISIBLE_DEVICES= \
#        timeout 300s python3 score_gpu.py   (expected exit 0).
# CPU plumbing audit: RUNS=/abs/audit/root TMPDIR=/tmp/eng1_audit timeout 1190s bash STAGED_GPU_COMMANDS.sh --audit
set -euo pipefail
: "${RUNS:?RUNS is required; use a fresh run root and reuse it across the labels}"
[[ "$RUNS" == /* ]] || { echo "RUNS must be absolute" >&2; exit 64; }
TOOL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$TOOL"
export RUNS PYTHONDONTWRITEBYTECODE=1
export TMPDIR="${TMPDIR:?TMPDIR must be /tmp/eng1_*}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
GPU_RUN=forge-gpu-run
label="${1:?select one registered label, or --audit}"
if [[ "$label" == --audit ]]; then
    export ENG1_AUDIT=1 CUDA_VISIBLE_DEVICES=
    GPU_RUN="$TOOL/forge_gpu_run_standin.py"
    label=candidate_eng1
else
    unset ENG1_AUDIT
fi
mkdir -p -- "$TMPDIR"
timeout 60s python3 start_gpu_run.py
case "$label" in
  baseline_eng1) # expected child 1, parent 0; inner 900s, --max 960s, outer 990s
    timeout 1010s python3 run_gpu_line.py baseline_eng1 1 990 "$GPU_RUN" --min 910 --max 960 --label eng1-baseline_eng1 -- timeout --kill-after=10s 900s python3 run_gpu_tests.py baseline eng1
    ;;
  candidate_eng1) # expected child 0, parent 0; inner 900s, --max 960s, outer 990s
    timeout 1010s python3 run_gpu_line.py candidate_eng1 0 990 "$GPU_RUN" --min 910 --max 960 --label eng1-candidate_eng1 -- timeout --kill-after=10s 900s python3 run_gpu_tests.py candidate eng1
    ;;
  baseline_regression) # expected child 0,1, parent 0; inner 1050s, --max 1110s, outer 1140s
    timeout 1160s python3 run_gpu_line.py baseline_regression 0,1 1140 "$GPU_RUN" --min 1060 --max 1110 --label eng1-baseline_regression -- timeout --kill-after=10s 1050s python3 run_gpu_tests.py baseline regression
    ;;
  candidate_regression) # expected child 0,1, parent 0; inner 1050s, --max 1110s, outer 1140s
    timeout 1160s python3 run_gpu_line.py candidate_regression 0,1 1140 "$GPU_RUN" --min 1060 --max 1110 --label eng1-candidate_regression -- timeout --kill-after=10s 1050s python3 run_gpu_tests.py candidate regression
    ;;
  baseline_terrain) # expected child 0,1, parent 0; inner 1050s, --max 1110s, outer 1140s
    timeout 1160s python3 run_gpu_line.py baseline_terrain 0,1 1140 "$GPU_RUN" --min 1060 --max 1110 --label eng1-baseline_terrain -- timeout --kill-after=10s 1050s python3 run_gpu_tests.py baseline terrain
    ;;
  candidate_terrain) # expected child 0,1, parent 0; inner 1050s, --max 1110s, outer 1140s
    timeout 1160s python3 run_gpu_line.py candidate_terrain 0,1 1140 "$GPU_RUN" --min 1060 --max 1110 --label eng1-candidate_terrain -- timeout --kill-after=10s 1050s python3 run_gpu_tests.py candidate terrain
    ;;
  C_COPY) # expected child 0, parent 0; inner 420s, --max 480s, outer 510s
    timeout 530s python3 run_gpu_line.py C_COPY 0 510 "$GPU_RUN" --min 430 --max 480 --label eng1-C_COPY -- timeout --kill-after=10s 420s python3 run_gpu_tests.py C_COPY control
    ;;
  C_UBSAN) # expected child 0, parent 0; inner 900s, --max 960s, outer 990s
    timeout 1010s python3 run_gpu_line.py C_UBSAN 0 990 "$GPU_RUN" --min 910 --max 960 --label eng1-C_UBSAN -- timeout --kill-after=10s 900s python3 run_gpu_tests.py C_UBSAN control
    ;;
  M01) # expected child 1, parent 0; inner 420s, --max 480s, outer 510s
    timeout 530s python3 run_gpu_line.py M01 1 510 "$GPU_RUN" --min 430 --max 480 --label eng1-M01 -- timeout --kill-after=10s 420s python3 run_gpu_tests.py M01 mutation
    ;;
  M02) # expected child 1, parent 0; inner 420s, --max 480s, outer 510s
    timeout 530s python3 run_gpu_line.py M02 1 510 "$GPU_RUN" --min 430 --max 480 --label eng1-M02 -- timeout --kill-after=10s 420s python3 run_gpu_tests.py M02 mutation
    ;;
  M03) # expected child 1, parent 0; inner 420s, --max 480s, outer 510s
    timeout 530s python3 run_gpu_line.py M03 1 510 "$GPU_RUN" --min 430 --max 480 --label eng1-M03 -- timeout --kill-after=10s 420s python3 run_gpu_tests.py M03 mutation
    ;;
  M04) # expected child 1, parent 0; inner 420s, --max 480s, outer 510s
    timeout 530s python3 run_gpu_line.py M04 1 510 "$GPU_RUN" --min 430 --max 480 --label eng1-M04 -- timeout --kill-after=10s 420s python3 run_gpu_tests.py M04 mutation
    ;;
  M05) # expected child 1, parent 0; inner 420s, --max 480s, outer 510s
    timeout 530s python3 run_gpu_line.py M05 1 510 "$GPU_RUN" --min 430 --max 480 --label eng1-M05 -- timeout --kill-after=10s 420s python3 run_gpu_tests.py M05 mutation
    ;;
  M06) # expected child 1, parent 0; inner 420s, --max 480s, outer 510s
    timeout 530s python3 run_gpu_line.py M06 1 510 "$GPU_RUN" --min 430 --max 480 --label eng1-M06 -- timeout --kill-after=10s 420s python3 run_gpu_tests.py M06 mutation
    ;;
  M07) # expected child 1, parent 0; inner 420s, --max 480s, outer 510s
    timeout 530s python3 run_gpu_line.py M07 1 510 "$GPU_RUN" --min 430 --max 480 --label eng1-M07 -- timeout --kill-after=10s 420s python3 run_gpu_tests.py M07 mutation
    ;;
  M08) # expected child 1, parent 0; inner 420s, --max 480s, outer 510s
    timeout 530s python3 run_gpu_line.py M08 1 510 "$GPU_RUN" --min 430 --max 480 --label eng1-M08 -- timeout --kill-after=10s 420s python3 run_gpu_tests.py M08 mutation
    ;;
  M09) # expected child 1, parent 0; inner 420s, --max 480s, outer 510s
    timeout 530s python3 run_gpu_line.py M09 1 510 "$GPU_RUN" --min 430 --max 480 --label eng1-M09 -- timeout --kill-after=10s 420s python3 run_gpu_tests.py M09 mutation
    ;;
  M10) # expected child 1, parent 0; inner 420s, --max 480s, outer 510s
    timeout 530s python3 run_gpu_line.py M10 1 510 "$GPU_RUN" --min 430 --max 480 --label eng1-M10 -- timeout --kill-after=10s 420s python3 run_gpu_tests.py M10 mutation
    ;;
  M11) # expected child 1, parent 0; inner 420s, --max 480s, outer 510s
    timeout 530s python3 run_gpu_line.py M11 1 510 "$GPU_RUN" --min 430 --max 480 --label eng1-M11 -- timeout --kill-after=10s 420s python3 run_gpu_tests.py M11 mutation
    ;;
  M12) # expected child 1, parent 0; inner 420s, --max 480s, outer 510s
    timeout 530s python3 run_gpu_line.py M12 1 510 "$GPU_RUN" --min 430 --max 480 --label eng1-M12 -- timeout --kill-after=10s 420s python3 run_gpu_tests.py M12 mutation
    ;;
  M13) # expected child 1, parent 0; inner 420s, --max 480s, outer 510s
    timeout 530s python3 run_gpu_line.py M13 1 510 "$GPU_RUN" --min 430 --max 480 --label eng1-M13 -- timeout --kill-after=10s 420s python3 run_gpu_tests.py M13 mutation
    ;;
  M14) # expected child 1, parent 0; inner 900s, --max 960s, outer 990s
    timeout 1010s python3 run_gpu_line.py M14 1 990 "$GPU_RUN" --min 910 --max 960 --label eng1-M14 -- timeout --kill-after=10s 900s python3 run_gpu_tests.py M14 mutation_e6
    ;;
  *) echo "unknown ENG1 label: $label" >&2; exit 64 ;;
esac
