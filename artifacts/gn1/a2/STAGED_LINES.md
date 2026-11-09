# Staged lines and expected exits

Every line runs from A2’s own directory through forge-gpu-run and foreground timeout. This seat ran only --audit. Lease refusals are not admitted lines and may be retried.

| Label | Arm/suite | Expected child exits | Expected parent | Inner / max / outer (s) |
|---|---|---|---|---|
| baseline_focus | baseline/focus | [1] | 0 | 240 / 300 / 330 |
| candidate_focus | candidate/focus | [0] | 0 | 240 / 300 / 330 |
| baseline_regression | baseline/regression | [1] | 0 | 840 / 900 / 930 |
| candidate_regression | candidate/regression | [0, 1] | 0 | 840 / 900 / 930 |
| D5_bias_dtype | D5_bias_dtype/mutation | [1] | 0 | 180 / 240 / 270 |
| D5_c_bool | D5_c_bool/mutation | [1] | 0 | 180 / 240 / 270 |
| D5_c_int | D5_c_int/mutation | [1] | 0 | 180 / 240 / 270 |
| D5_c_positive | D5_c_positive/mutation | [1] | 0 | 180 / 240 / 270 |
| D5_channels | D5_channels/mutation | [1] | 0 | 180 / 240 / 270 |
| D5_g_bool | D5_g_bool/mutation | [1] | 0 | 180 / 240 / 270 |
| D5_g_int | D5_g_int/mutation | [1] | 0 | 180 / 240 / 270 |
| D5_g_positive | D5_g_positive/mutation | [1] | 0 | 180 / 240 / 270 |
| D5_input_dtype | D5_input_dtype/mutation | [1] | 0 | 180 / 240 / 270 |
| D5_rank | D5_rank/mutation | [1] | 0 | 180 / 240 / 270 |
| D5_weight_dtype | D5_weight_dtype/mutation | [1] | 0 | 180 / 240 / 270 |
| M01 | M01/mutation | [1] | 0 | 180 / 240 / 270 |
| M02 | M02/mutation | [1] | 0 | 180 / 240 / 270 |
| M03 | M03/mutation | [1] | 0 | 180 / 240 / 270 |
| M04m | M04m/mutation | [1] | 0 | 180 / 240 / 270 |
| M04v | M04v/mutation | [1] | 0 | 180 / 240 / 270 |
| M05 | M05/mutation | [1] | 0 | 180 / 240 / 270 |
| M06in | M06in/mutation | [1] | 0 | 180 / 240 / 270 |
| M06out | M06out/mutation | [1] | 0 | 180 / 240 / 270 |
| M07cast | M07cast/mutation | [1] | 0 | 180 / 240 / 270 |
| M07fast | M07fast/mutation | [1] | 0 | 180 / 240 / 270 |
| M08cast | M08cast/mutation | [1] | 0 | 180 / 240 / 270 |
| M08fast | M08fast/mutation | [1] | 0 | 180 / 240 / 270 |
| M09cast | M09cast/mutation | [1] | 0 | 180 / 240 / 270 |
| M09fast | M09fast/mutation | [1] | 0 | 180 / 240 / 270 |
| M10 | M10/mutation | [1] | 0 | 180 / 240 / 270 |
| M11 | M11/mutation | [1] | 0 | 180 / 240 / 270 |
| M12 | M12/mutation | [1] | 0 | 180 / 240 / 270 |
| M13 | M13/mutation | [1] | 0 | 180 / 240 / 270 |
| M14 | M14/mutation | [1] | 0 | 180 / 240 / 270 |
| M15b | M15b/mutation | [1] | 0 | 180 / 240 / 270 |
| M15w | M15w/mutation | [1] | 0 | 180 / 240 / 270 |
| M15x | M15x/mutation | [1] | 0 | 180 / 240 / 270 |
| M16 | M16/mutation | [1] | 0 | 180 / 240 / 270 |
| M17b | M17b/mutation | [1] | 0 | 180 / 240 / 270 |
| M17w | M17w/mutation | [1] | 0 | 180 / 240 / 270 |
| M18b | M18b/mutation | [1] | 0 | 180 / 240 / 270 |
| M18w | M18w/mutation | [1] | 0 | 180 / 240 / 270 |
| V1 | V1/mutation | [1] | 0 | 180 / 240 / 270 |
| V2 | V2/mutation | [1] | 0 | 180 / 240 / 270 |
| V3 | V3/mutation | [0] | 0 | 180 / 240 / 270 |

```bash
#!/usr/bin/env bash
# Prior art: GNU timeout/POSIX shell; A1 stage layout. A2 D6/D7 expected exits.
# One label per foreground command. Every parent expected 0; no default RUNS.
set -euo pipefail
: "${RUNS:?RUNS required; reuse the same root for all 45 labels}"
[[ "$RUNS" == /* ]] || { echo "RUNS must be absolute" >&2; exit 64; }
TOOL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$TOOL"
export RUNS
export TMPDIR="${TMPDIR:-/tmp/gn1_a2_staged}" PYTHONDONTWRITEBYTECODE=1
[[ "$TMPDIR" == /tmp/gn1_a2_* ]] || { echo "TMPDIR must be /tmp/gn1_a2_*" >&2; exit 64; }
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
  baseline_focus) # expected child 1, parent 0; inner 240s, --max 300s, outer 330s
    timeout 350s python3 run_gpu_line.py baseline_focus 1 330 "$GPU_RUN" --min 250 --max 300 --label gn1-a2-baseline_focus -- timeout --kill-after=10s 240s python3 run_gpu_tests.py baseline focus
    ;;
  candidate_focus) # expected child 0, parent 0; inner 240s, --max 300s, outer 330s
    timeout 350s python3 run_gpu_line.py candidate_focus 0 330 "$GPU_RUN" --min 250 --max 300 --label gn1-a2-candidate_focus -- timeout --kill-after=10s 240s python3 run_gpu_tests.py candidate focus
    ;;
  baseline_regression) # expected child 1, parent 0; inner 840s, --max 900s, outer 930s
    timeout 950s python3 run_gpu_line.py baseline_regression 1 930 "$GPU_RUN" --min 850 --max 900 --label gn1-a2-baseline_regression -- timeout --kill-after=10s 840s python3 run_gpu_tests.py baseline regression
    ;;
  candidate_regression) # expected child 0,1, parent 0; inner 840s, --max 900s, outer 930s
    timeout 950s python3 run_gpu_line.py candidate_regression 0,1 930 "$GPU_RUN" --min 850 --max 900 --label gn1-a2-candidate_regression -- timeout --kill-after=10s 840s python3 run_gpu_tests.py candidate regression
    ;;
  D5_bias_dtype) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py D5_bias_dtype 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-D5_bias_dtype -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_bias_dtype mutation
    ;;
  D5_c_bool) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py D5_c_bool 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-D5_c_bool -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_c_bool mutation
    ;;
  D5_c_int) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py D5_c_int 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-D5_c_int -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_c_int mutation
    ;;
  D5_c_positive) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py D5_c_positive 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-D5_c_positive -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_c_positive mutation
    ;;
  D5_channels) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py D5_channels 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-D5_channels -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_channels mutation
    ;;
  D5_g_bool) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py D5_g_bool 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-D5_g_bool -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_g_bool mutation
    ;;
  D5_g_int) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py D5_g_int 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-D5_g_int -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_g_int mutation
    ;;
  D5_g_positive) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py D5_g_positive 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-D5_g_positive -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_g_positive mutation
    ;;
  D5_input_dtype) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py D5_input_dtype 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-D5_input_dtype -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_input_dtype mutation
    ;;
  D5_rank) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py D5_rank 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-D5_rank -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_rank mutation
    ;;
  D5_weight_dtype) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py D5_weight_dtype 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-D5_weight_dtype -- timeout --kill-after=10s 180s python3 run_gpu_tests.py D5_weight_dtype mutation
    ;;
  M01) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M01 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M01 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M01 mutation
    ;;
  M02) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M02 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M02 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M02 mutation
    ;;
  M03) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M03 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M03 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M03 mutation
    ;;
  M04m) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M04m 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M04m -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M04m mutation
    ;;
  M04v) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M04v 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M04v -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M04v mutation
    ;;
  M05) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M05 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M05 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M05 mutation
    ;;
  M06in) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M06in 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M06in -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M06in mutation
    ;;
  M06out) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M06out 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M06out -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M06out mutation
    ;;
  M07cast) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M07cast 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M07cast -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M07cast mutation
    ;;
  M07fast) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M07fast 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M07fast -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M07fast mutation
    ;;
  M08cast) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M08cast 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M08cast -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M08cast mutation
    ;;
  M08fast) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M08fast 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M08fast -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M08fast mutation
    ;;
  M09cast) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M09cast 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M09cast -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M09cast mutation
    ;;
  M09fast) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M09fast 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M09fast -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M09fast mutation
    ;;
  M10) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M10 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M10 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M10 mutation
    ;;
  M11) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M11 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M11 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M11 mutation
    ;;
  M12) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M12 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M12 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M12 mutation
    ;;
  M13) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M13 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M13 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M13 mutation
    ;;
  M14) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M14 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M14 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M14 mutation
    ;;
  M15b) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M15b 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M15b -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M15b mutation
    ;;
  M15w) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M15w 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M15w -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M15w mutation
    ;;
  M15x) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M15x 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M15x -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M15x mutation
    ;;
  M16) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M16 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M16 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M16 mutation
    ;;
  M17b) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M17b 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M17b -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M17b mutation
    ;;
  M17w) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M17w 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M17w -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M17w mutation
    ;;
  M18b) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M18b 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M18b -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M18b mutation
    ;;
  M18w) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M18w 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-M18w -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M18w mutation
    ;;
  V1) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py V1 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-V1 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py V1 mutation
    ;;
  V2) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py V2 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-V2 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py V2 mutation
    ;;
  V3) # expected child 0, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py V3 0 270 "$GPU_RUN" --min 190 --max 240 --label gn1-a2-V3 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py V3 mutation
    ;;
  *) echo "unknown GN1 A2 label: $label" >&2; exit 64 ;;
esac
# After every admitted label: RUNS=/same/root timeout 30s python3 score_gpu.py (expected 0).
```
