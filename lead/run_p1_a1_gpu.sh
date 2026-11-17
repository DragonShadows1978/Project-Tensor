#!/usr/bin/env bash
# Lead lane: APA-P1 A1 GPU run in GRAPA's standing window (until 2026-11-06 02:00).
# Usage: run_p1_a1_gpu.sh <LOGDIR>
# Runs artifacts/apa_p1_a1/STAGED_GPU_LINES.sh exactly as its header says (CUDA_VISIBLE_DEVICES=0, timeout 86400 s),
# after checking its sha256, the window (go-file == GPU_WINDOW, >= 1,800 s left) and the disk stops (>= 25 GB NVMe,
# >= 20 GB /). The staged script itself runs 29 duty-cycle legs (each <= 1,240 s summed --max) with 600 s cooldowns,
# every line through forge-gpu-run --max. Its rc is logged unmasked. The run root is then sealed with
# post_run_manifest.sh. No DONE marker (standing window).
set -u
G=$1
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod
S=$W/artifacts/apa_p1_a1/STAGED_GPU_LINES.sh
GO=/mnt/ForgeRealm/grapa_run7/TENSOR_GO_STANDING_1103
F=/mnt/ForgeRealm/grapa_run7/GPU_WINDOW
MANIFEST_TOOL=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_tx3f2g/artifacts/tx3_f2g/lead/post_run_manifest.sh
LOG=$G/p1_a1_gpu.log
mkdir -p "$G"
log() { echo "$(date -Is) $*" >> "$LOG"; }
log "P1-A1 GPU start pid=$$"
[ "$(sha256sum "$S" | cut -d' ' -f1)" = 6b0e8bf10cc11e45ee8969f8c555dd57ceb3ca9c3efe17e921960853b77d81f7 ] || { log "staged hash differs"; exit 3; }
g=$(cat "$GO" 2>/dev/null); w=$(cat "$F" 2>/dev/null)
case "$g" in "open-until "*) ;; *) log "go-file not open: [$g]"; exit 2;; esac
[ "$w" = "$g" ] || { log "GPU_WINDOW [$w] differs from go-file [$g]"; exit 2; }
left=$(( $(date -d "${g#open-until }" +%s) - $(date +%s) )); [ "$left" -ge 1800 ] || { log "only $left s left"; exit 2; }
fnv=$(df -BG --output=avail /mnt/ForgeRealm | tail -1 | tr -dc '0-9'); frt=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
{ [ "$fnv" -ge 25 ] && [ "$frt" -ge 20 ]; } || { log "disk stop: ${fnv}/${frt} GB"; exit 2; }
log "window ok: $left s left, free ${fnv}/${frt} GB"
RUN=$W/artifacts/apa_p1_a1/gpu_$(date -u +%Y%m%dT%H%M%SZ)
log "run root $RUN"
( cd "$W" && APA_P1_A1_RUN_ROOT="$RUN" CUDA_VISIBLE_DEVICES=0 timeout 86400s bash artifacts/apa_p1_a1/STAGED_GPU_LINES.sh ) < /dev/null > "$G/p1_a1_lines.out" 2>&1
log "staged script rc=$?"
bash "$MANIFEST_TOOL" "$RUN" "$G/P1A1_MANIFEST_$(basename "$RUN").sha256" >> "$LOG" 2>&1; log "manifest rc=$?"
log "P1-A1 GPU done"
