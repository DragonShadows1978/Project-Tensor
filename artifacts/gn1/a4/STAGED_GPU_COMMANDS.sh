#!/usr/bin/env bash
# Prior art: A2 stage; GNU timeout/POSIX shell. D4 exact outcome attribution.
set -euo pipefail
: "${RUNS:?RUNS required; reuse the same root for every label}"
[[ "$RUNS" == /* ]] || exit 64
TOOL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$TOOL"
export RUNS
export TMPDIR="${TMPDIR:-/tmp/gn1_a4_staged}" PYTHONDONTWRITEBYTECODE=1
# D7 compares resolved paths through the shared Python guard.
timeout 5s python3 -c 'import os; from evidence import scratch_guard; scratch_guard(os.environ["TMPDIR"])'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
GPU_RUN=forge-gpu-run
label="${1:?select a registered label, or --audit}"
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
  baseline_focus) # expected child 1; parent 0
    timeout 350s python3 run_gpu_line.py baseline_focus 1 330 "$GPU_RUN" --min 250 --max 300 --label gn1-a4-baseline_focus -- timeout --kill-after=10s 240s python3 run_gpu_tests.py baseline focus
    ;;
  candidate_focus) # expected child 0; parent 0
    timeout 350s python3 run_gpu_line.py candidate_focus 0 330 "$GPU_RUN" --min 250 --max 300 --label gn1-a4-candidate_focus -- timeout --kill-after=10s 240s python3 run_gpu_tests.py candidate focus
    ;;
  baseline_regression) # expected child 1; parent 0
    timeout 950s python3 run_gpu_line.py baseline_regression 1 930 "$GPU_RUN" --min 850 --max 900 --label gn1-a4-baseline_regression -- timeout --kill-after=10s 840s python3 run_gpu_tests.py baseline regression
    ;;
  candidate_regression) # expected child 0,1; parent 0
    timeout 950s python3 run_gpu_line.py candidate_regression 0,1 930 "$GPU_RUN" --min 850 --max 900 --label gn1-a4-candidate_regression -- timeout --kill-after=10s 840s python3 run_gpu_tests.py candidate regression
    ;;
  D5_bias_dtype) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py D5_bias_dtype 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-D5_bias_dtype -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_bias_dtype mutation
    ;;
  D5_bias_length) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py D5_bias_length 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-D5_bias_length -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_bias_length mutation
    ;;
  D5_bias_rank) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py D5_bias_rank 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-D5_bias_rank -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_bias_rank mutation
    ;;
  D5_c_bool) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py D5_c_bool 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-D5_c_bool -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_c_bool mutation
    ;;
  D5_c_index) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py D5_c_index 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-D5_c_index -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_c_index mutation
    ;;
  D5_c_int) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py D5_c_int 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-D5_c_int -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_c_int mutation
    ;;
  D5_c_npbool) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py D5_c_npbool 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-D5_c_npbool -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_c_npbool mutation
    ;;
  D5_c_positive) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py D5_c_positive 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-D5_c_positive -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_c_positive mutation
    ;;
  D5_c_store) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py D5_c_store 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-D5_c_store -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_c_store mutation
    ;;
  D5_channels) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py D5_channels 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-D5_channels -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_channels mutation
    ;;
  D5_g_bool) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py D5_g_bool 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-D5_g_bool -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_g_bool mutation
    ;;
  D5_g_index) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py D5_g_index 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-D5_g_index -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_g_index mutation
    ;;
  D5_g_int) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py D5_g_int 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-D5_g_int -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_g_int mutation
    ;;
  D5_g_npbool) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py D5_g_npbool 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-D5_g_npbool -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_g_npbool mutation
    ;;
  D5_g_positive) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py D5_g_positive 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-D5_g_positive -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_g_positive mutation
    ;;
  D5_g_store) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py D5_g_store 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-D5_g_store -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_g_store mutation
    ;;
  D5_input_dtype) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py D5_input_dtype 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-D5_input_dtype -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_input_dtype mutation
    ;;
  D5_rank) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py D5_rank 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-D5_rank -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_rank mutation
    ;;
  D5_weight_dtype) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py D5_weight_dtype 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-D5_weight_dtype -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_weight_dtype mutation
    ;;
  D5_weight_length) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py D5_weight_length 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-D5_weight_length -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_weight_length mutation
    ;;
  D5_weight_rank) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py D5_weight_rank 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-D5_weight_rank -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_weight_rank mutation
    ;;
  M01) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M01 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M01 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M01 mutation
    ;;
  M02) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M02 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M02 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M02 mutation
    ;;
  M03) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M03 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M03 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M03 mutation
    ;;
  M04m) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M04m 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M04m -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M04m mutation
    ;;
  M04v) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M04v 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M04v -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M04v mutation
    ;;
  M05) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M05 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M05 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M05 mutation
    ;;
  M06in) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M06in 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M06in -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M06in mutation
    ;;
  M06out) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M06out 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M06out -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M06out mutation
    ;;
  M07cast) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M07cast 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M07cast -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M07cast mutation
    ;;
  M07fast) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M07fast 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M07fast -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M07fast mutation
    ;;
  M08cast) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M08cast 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M08cast -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M08cast mutation
    ;;
  M08fast) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M08fast 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M08fast -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M08fast mutation
    ;;
  M09cast) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M09cast 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M09cast -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M09cast mutation
    ;;
  M09fast) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M09fast 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M09fast -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M09fast mutation
    ;;
  M10) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M10 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M10 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M10 mutation
    ;;
  M11) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M11 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M11 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M11 mutation
    ;;
  M12) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M12 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M12 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M12 mutation
    ;;
  M13) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M13 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M13 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M13 mutation
    ;;
  M14) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M14 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M14 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M14 mutation
    ;;
  M15b) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M15b 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M15b -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M15b mutation
    ;;
  M15w) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M15w 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M15w -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M15w mutation
    ;;
  M15x) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M15x 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M15x -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M15x mutation
    ;;
  M16) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M16 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M16 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M16 mutation
    ;;
  M17b) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M17b 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M17b -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M17b mutation
    ;;
  M17w) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M17w 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M17w -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M17w mutation
    ;;
  M18b) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M18b 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M18b -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M18b mutation
    ;;
  M18w) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py M18w 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-M18w -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M18w mutation
    ;;
  V1) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py V1 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-V1 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py V1 mutation
    ;;
  V2) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py V2 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-V2 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py V2 mutation
    ;;
  V3) # expected child 0; parent 0
    timeout 290s python3 run_gpu_line.py V3 0 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-V3 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py V3 mutation
    ;;
  V4) # expected child 1; parent 0
    timeout 290s python3 run_gpu_line.py V4 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-V4 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py V4 mutation
    ;;
  E_npbool_g) # expected child 0; parent 0
    timeout 290s python3 run_gpu_line.py E_npbool_g 0 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-E_npbool_g -- timeout --kill-after=10s 180s python3 run_gpu_tests.py E_npbool_g mutation
    ;;
  E_npbool_c) # expected child 0; parent 0
    timeout 290s python3 run_gpu_line.py E_npbool_c 0 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a4-E_npbool_c -- timeout --kill-after=10s 180s python3 run_gpu_tests.py E_npbool_c mutation
    ;;
  *) exit 64 ;;
esac
