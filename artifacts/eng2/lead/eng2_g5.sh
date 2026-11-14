#!/usr/bin/env bash
# Lead lane: ENG2 G5, the 21 staged GPU lines of tool/ (A1 and A2 ACCEPT; REV1 and REV2 closed), adapted from GN1's gn1_g5.sh.
# Usage: eng2_g5.sh <END iso> <LOGDIR>
# - One shell invocation per label, in gpu_lines.json order, exactly as STAGED_GPU_COMMANDS.sh's usage requires: a
#   single absolute RUNS under tool/gpu_runs/, CUDA_VISIBLE_DEVICES=0, TMPDIR under /tmp/eng2_*, the real governor.
#   ENG2_AUDIT and PYTHONOPTIMIZE are unset.
# - Before each label: the line's registered maximum + 30 s must remain before END, and the NVMe / root must keep
#   25 / 20 GB free; otherwise stop (the remaining labels resume later under the same RUNS).
# - Records the PATH, PYTHON*, LD_*, CUDA_* and FORGE_GPU_* environment before the first line.
# - Afterwards: post_run_manifest.sh seals RUNS, then score_gpu.py recomputes the verdict on CPU. Every rc is logged.
# Prior art: GNU timeout and coreutils; GN1's lead wrapper; no new algorithm.
set -u
END=$1; L=$2
EN=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng2/artifacts/eng2
T=$EN/tool
RUNS=$T/gpu_runs/g5_1114
TMPD=/tmp/eng2_g5
LOG=$L/eng2_g5.log
mkdir -p "$L" "$TMPD"
log() { echo "$(date -Is) $*" >> "$LOG"; }
E0=$(date -d "$END" +%s)
log "ENG2 G5 start pid=$$ end=$END runs=$RUNS"
if [ -e "$RUNS/START.json" ]; then log "RUNS has a START already: resuming labels without a parent receipt"; fi
env | grep -E '^(PATH|PYTHON[A-Z_]*|LD_[A-Z_]*|CUDA_[A-Z_]*|FORGE_GPU_[A-Z_]*)=' | sort > "$L/eng2_g5_env.txt"
ran=0; stopped=""
while IFS=$'\t' read -r label maximum; do
  if ls "$RUNS/$label"/attempt-*/parent_result.json >/dev/null 2>&1; then log "$label: already attempted, skipped"; continue; fi
  left=$(( E0 - $(date +%s) ))
  [ "$left" -lt $(( maximum + 30 )) ] && { log "stop before $label: $left s left, max $maximum"; stopped="time"; break; }
  fnv=$(df -BG --output=avail /mnt/ForgeRealm | tail -1 | tr -dc '0-9'); frt=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  { [ "$fnv" -lt 25 ] || [ "$frt" -lt 20 ]; } && { log "stop before $label: free ${fnv} GB NVMe, ${frt} GB root"; stopped="space"; break; }
  log "$label start (left $left s, max $maximum)"
  timeout --kill-after=30 $(( maximum + 120 )) env -u PYTHONOPTIMIZE -u ENG2_AUDIT \
    RUNS="$RUNS" CUDA_VISIBLE_DEVICES=0 TMPDIR="$TMPD" bash "$T/STAGED_GPU_COMMANDS.sh" "$label" < /dev/null >> "$L/eng2_g5_lines.out" 2>&1
  log "$label rc=$?"; ran=$((ran+1))
done < <(python3 -B -c 'import json,sys; [print(r["label"], r["maximum"], sep="\t") for r in json.load(open(sys.argv[1]))]' "$T/gpu_lines.json")
log "labels run: $ran, stopped=${stopped:-no}"
if [ -d "$RUNS" ]; then
  bash "$EN/lead/post_run_manifest.sh" "$RUNS" "$L/eng2_g5_MANIFEST.sha256" >> "$LOG" 2>&1; log "post_run_manifest rc=$?"
  ( cd "$T" && timeout 900 env -u PYTHONOPTIMIZE CUDA_VISIBLE_DEVICES= RUNS="$RUNS" TMPDIR="$TMPD" python3 -B score_gpu.py ) > "$L/eng2_g5_score.out" 2>&1
  log "score_gpu rc=$?"
else
  log "no RUNS directory was created"
fi
log "ENG2 G5 done"
