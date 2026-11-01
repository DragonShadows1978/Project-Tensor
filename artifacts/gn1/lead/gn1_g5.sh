#!/usr/bin/env bash
# Lead lane: GN1 G5, the 58 staged GPU lines of a8/ (contract C; A8 ACCEPT, GN1-REV6 ACCEPT), inside an open window.
# Usage: gn1_g5.sh <END iso> <LOGDIR>     (called by TX3-F2g's window_run4_gn1.sh after run 4)
# - One shell invocation per label, in gpu_lines.json order, exactly as the staged script's usage requires: a single
#   absolute RUNS under a8/, CUDA_VISIBLE_DEVICES=0, TMPDIR under /tmp/gn1_a8_*, the real governor. GN1_AUDIT,
#   GN1_CONTROL, GN1_CPU_FIXTURE and PYTHONOPTIMIZE are unset.
# - Before each label: the line's registered maximum + 30 s must remain before END, and the NVMe / root must keep
#   25 / 20 GB free; otherwise stop (the remaining labels resume in a later window under the same RUNS).
# - Records the PATH, PYTHON*, LD_*, CUDA_* and FORGE_GPU_* environment (GN1-REV6 R6-O4) before the first line.
# - Afterwards: post_run_manifest.sh seals RUNS, then score_gpu.py recomputes the verdict on CPU. Every rc is logged.
set -u
END=$1; L=$2
GN=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_gn1/artifacts/gn1
T=$GN/a8
RUNS=$T/gpu_runs/g5_1101
TMPD=/tmp/gn1_a8_g5
LOG=$L/gn1_g5.log
mkdir -p "$L" "$TMPD"
log() { echo "$(date -Is) $*" >> "$LOG"; }
E0=$(date -d "$END" +%s)
log "GN1 G5 start pid=$$ end=$END runs=$RUNS"
if [ -e "$RUNS/START.json" ]; then log "RUNS has a START already: resuming labels without a passing parent receipt"; fi
env | grep -E '^(PATH|PYTHON[A-Z_]*|LD_[A-Z_]*|CUDA_[A-Z_]*|FORGE_GPU_[A-Z_]*)=' | sort > "$L/gn1_g5_env.txt"
ran=0; stopped=""
while IFS=$'\t' read -r label maximum; do
  if ls "$RUNS/$label"/attempt-*/parent_result.json >/dev/null 2>&1; then log "$label: already attempted, skipped"; continue; fi
  left=$(( E0 - $(date +%s) ))
  [ "$left" -lt $(( maximum + 30 )) ] && { log "stop before $label: $left s left, max $maximum"; stopped="time"; break; }
  fnv=$(df -BG --output=avail /mnt/ForgeRealm | tail -1 | tr -dc '0-9'); frt=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  { [ "$fnv" -lt 25 ] || [ "$frt" -lt 20 ]; } && { log "stop before $label: free ${fnv} GB NVMe, ${frt} GB root"; stopped="space"; break; }
  log "$label start (left $left s, max $maximum)"
  timeout --kill-after=30 $(( maximum + 120 )) env -u PYTHONOPTIMIZE -u GN1_AUDIT -u GN1_CONTROL -u GN1_CPU_FIXTURE \
    RUNS="$RUNS" CUDA_VISIBLE_DEVICES=0 TMPDIR="$TMPD" bash "$T/STAGED_GPU_COMMANDS.sh" "$label" < /dev/null >> "$L/gn1_g5_lines.out" 2>&1
  log "$label rc=$?"; ran=$((ran+1))
done < <(python3 -B -c 'import json,sys; [print(r["label"], r["maximum"], sep="\t") for r in json.load(open(sys.argv[1]))]' "$T/gpu_lines.json")
log "labels run: $ran, stopped=${stopped:-no}"
if [ -d "$RUNS" ]; then
  bash "$GN/lead/post_run_manifest.sh" "$RUNS" "$L/gn1_g5_MANIFEST.sha256" >> "$LOG" 2>&1; log "post_run_manifest rc=$?"
  ( cd "$T" && timeout 900 env -u PYTHONOPTIMIZE CUDA_VISIBLE_DEVICES= RUNS="$RUNS" TMPDIR="$TMPD" python3 -B score_gpu.py ) > "$L/gn1_g5_score.out" 2>&1
  log "score_gpu rc=$?"
else
  log "no RUNS directory was created"
fi
log "GN1 G5 done"
