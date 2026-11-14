#!/usr/bin/env bash
# ENG2 staged GPU lines (copied from ENG1 A3 / GN1 A8 tooling; the lead runs them; the seat never did on a device).
# Prior art: GNU timeout, POSIX shell; GN1 L45 completed-child supervision (parent receipt after child exit).
# Expected parent exit for EVERY line is 0 (the parent passes when the child exit matches the registration
# and the receipts check out). Child exits and bounds are listed per label below and in gpu_lines.json.
# Usage: RUNS=/abs/path/under/artifacts/eng2/tool/gpu_runs/<root> TMPDIR=/tmp/eng2_g5 \
#        CUDA_VISIBLE_DEVICES=0 timeout 1190s bash STAGED_GPU_COMMANDS.sh LABEL
# Run the labels in gpu_lines.json order with the SAME RUNS (mutants need a passing candidate_eng2;
# candidate_regnorm needs baseline_regnorm, candidate_regdtype needs baseline_regdtype).
# Afterwards: RUNS=<same> CUDA_VISIBLE_DEVICES= \
#        timeout 300s python3 score_gpu.py   (expected exit 0).
# CPU plumbing audit: RUNS=/abs/audit/root TMPDIR=/tmp/eng2_audit timeout 1190s bash STAGED_GPU_COMMANDS.sh --audit
set -euo pipefail
: "${RUNS:?RUNS is required; use a fresh run root and reuse it across the labels}"
[[ "$RUNS" == /* ]] || { echo "RUNS must be absolute" >&2; exit 64; }
TOOL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$TOOL"
export RUNS PYTHONDONTWRITEBYTECODE=1
export TMPDIR="${TMPDIR:?TMPDIR must be /tmp/eng2_*}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
GPU_RUN=forge-gpu-run
label="${1:?select one registered label, or --audit}"
if [[ "$label" == --audit ]]; then
    export ENG2_AUDIT=1 CUDA_VISIBLE_DEVICES=
    GPU_RUN="$TOOL/forge_gpu_run_standin.py"
    label=candidate_eng2
else
    unset ENG2_AUDIT
fi
mkdir -p -- "$TMPDIR"
timeout 60s python3 start_gpu_run.py
case "$label" in
  baseline_eng2) # expected child 1, parent 0; inner 600s, --max 660s, outer 690s
    timeout 710s python3 run_gpu_line.py baseline_eng2 1 690 "$GPU_RUN" --min 610 --max 660 --label eng2-baseline_eng2 -- timeout --kill-after=10s 600s python3 run_gpu_tests.py baseline eng2
    ;;
  candidate_eng2) # expected child 0, parent 0; inner 600s, --max 660s, outer 690s
    timeout 710s python3 run_gpu_line.py candidate_eng2 0 690 "$GPU_RUN" --min 610 --max 660 --label eng2-candidate_eng2 -- timeout --kill-after=10s 600s python3 run_gpu_tests.py candidate eng2
    ;;
  baseline_regnorm) # expected child 0,1, parent 0; inner 600s, --max 660s, outer 690s
    timeout 710s python3 run_gpu_line.py baseline_regnorm 0,1 690 "$GPU_RUN" --min 610 --max 660 --label eng2-baseline_regnorm -- timeout --kill-after=10s 600s python3 run_gpu_tests.py baseline regnorm
    ;;
  candidate_regnorm) # expected child 0,1, parent 0; inner 600s, --max 660s, outer 690s
    timeout 710s python3 run_gpu_line.py candidate_regnorm 0,1 690 "$GPU_RUN" --min 610 --max 660 --label eng2-candidate_regnorm -- timeout --kill-after=10s 600s python3 run_gpu_tests.py candidate regnorm
    ;;
  baseline_regdtype) # expected child 0,1, parent 0; inner 1050s, --max 1110s, outer 1140s
    timeout 1160s python3 run_gpu_line.py baseline_regdtype 0,1 1140 "$GPU_RUN" --min 1060 --max 1110 --label eng2-baseline_regdtype -- timeout --kill-after=10s 1050s python3 run_gpu_tests.py baseline regdtype
    ;;
  candidate_regdtype) # expected child 0,1, parent 0; inner 1050s, --max 1110s, outer 1140s
    timeout 1160s python3 run_gpu_line.py candidate_regdtype 0,1 1140 "$GPU_RUN" --min 1060 --max 1110 --label eng2-candidate_regdtype -- timeout --kill-after=10s 1050s python3 run_gpu_tests.py candidate regdtype
    ;;
  MF1) # expected child 1, parent 0; inner 600s, --max 660s, outer 690s
    timeout 710s python3 run_gpu_line.py MF1 1 690 "$GPU_RUN" --min 610 --max 660 --label eng2-MF1 -- timeout --kill-after=10s 600s python3 run_gpu_tests.py MF1 mutation
    ;;
  MF2) # expected child 1, parent 0; inner 600s, --max 660s, outer 690s
    timeout 710s python3 run_gpu_line.py MF2 1 690 "$GPU_RUN" --min 610 --max 660 --label eng2-MF2 -- timeout --kill-after=10s 600s python3 run_gpu_tests.py MF2 mutation
    ;;
  MN1_LN) # expected child 1, parent 0; inner 600s, --max 660s, outer 690s
    timeout 710s python3 run_gpu_line.py MN1_LN 1 690 "$GPU_RUN" --min 610 --max 660 --label eng2-MN1_LN -- timeout --kill-after=10s 600s python3 run_gpu_tests.py MN1_LN mutation
    ;;
  MN1_RMS) # expected child 1, parent 0; inner 600s, --max 660s, outer 690s
    timeout 710s python3 run_gpu_line.py MN1_RMS 1 690 "$GPU_RUN" --min 610 --max 660 --label eng2-MN1_RMS -- timeout --kill-after=10s 600s python3 run_gpu_tests.py MN1_RMS mutation
    ;;
  MN1_BN) # expected child 1, parent 0; inner 600s, --max 660s, outer 690s
    timeout 710s python3 run_gpu_line.py MN1_BN 1 690 "$GPU_RUN" --min 610 --max 660 --label eng2-MN1_BN -- timeout --kill-after=10s 600s python3 run_gpu_tests.py MN1_BN mutation
    ;;
  MN1_IN) # expected child 1, parent 0; inner 600s, --max 660s, outer 690s
    timeout 710s python3 run_gpu_line.py MN1_IN 1 690 "$GPU_RUN" --min 610 --max 660 --label eng2-MN1_IN -- timeout --kill-after=10s 600s python3 run_gpu_tests.py MN1_IN mutation
    ;;
  MN2_LN) # expected child 1, parent 0; inner 600s, --max 660s, outer 690s
    timeout 710s python3 run_gpu_line.py MN2_LN 1 690 "$GPU_RUN" --min 610 --max 660 --label eng2-MN2_LN -- timeout --kill-after=10s 600s python3 run_gpu_tests.py MN2_LN mutation
    ;;
  MN2_RMS) # expected child 1, parent 0; inner 600s, --max 660s, outer 690s
    timeout 710s python3 run_gpu_line.py MN2_RMS 1 690 "$GPU_RUN" --min 610 --max 660 --label eng2-MN2_RMS -- timeout --kill-after=10s 600s python3 run_gpu_tests.py MN2_RMS mutation
    ;;
  MN2_BN) # expected child 1, parent 0; inner 600s, --max 660s, outer 690s
    timeout 710s python3 run_gpu_line.py MN2_BN 1 690 "$GPU_RUN" --min 610 --max 660 --label eng2-MN2_BN -- timeout --kill-after=10s 600s python3 run_gpu_tests.py MN2_BN mutation
    ;;
  MN2_IN) # expected child 1, parent 0; inner 600s, --max 660s, outer 690s
    timeout 710s python3 run_gpu_line.py MN2_IN 1 690 "$GPU_RUN" --min 610 --max 660 --label eng2-MN2_IN -- timeout --kill-after=10s 600s python3 run_gpu_tests.py MN2_IN mutation
    ;;
  MN3_LN) # expected child 1, parent 0; inner 600s, --max 660s, outer 690s
    timeout 710s python3 run_gpu_line.py MN3_LN 1 690 "$GPU_RUN" --min 610 --max 660 --label eng2-MN3_LN -- timeout --kill-after=10s 600s python3 run_gpu_tests.py MN3_LN mutation
    ;;
  MN3_RMS) # expected child 1, parent 0; inner 600s, --max 660s, outer 690s
    timeout 710s python3 run_gpu_line.py MN3_RMS 1 690 "$GPU_RUN" --min 610 --max 660 --label eng2-MN3_RMS -- timeout --kill-after=10s 600s python3 run_gpu_tests.py MN3_RMS mutation
    ;;
  MN3_BN) # expected child 1, parent 0; inner 600s, --max 660s, outer 690s
    timeout 710s python3 run_gpu_line.py MN3_BN 1 690 "$GPU_RUN" --min 610 --max 660 --label eng2-MN3_BN -- timeout --kill-after=10s 600s python3 run_gpu_tests.py MN3_BN mutation
    ;;
  MN3_IN) # expected child 1, parent 0; inner 600s, --max 660s, outer 690s
    timeout 710s python3 run_gpu_line.py MN3_IN 1 690 "$GPU_RUN" --min 610 --max 660 --label eng2-MN3_IN -- timeout --kill-after=10s 600s python3 run_gpu_tests.py MN3_IN mutation
    ;;
  MN4_LN) # expected child 1, parent 0; inner 600s, --max 660s, outer 690s
    timeout 710s python3 run_gpu_line.py MN4_LN 1 690 "$GPU_RUN" --min 610 --max 660 --label eng2-MN4_LN -- timeout --kill-after=10s 600s python3 run_gpu_tests.py MN4_LN mutation
    ;;
  *) echo "unknown ENG2 label: $label" >&2; exit 64 ;;
esac
