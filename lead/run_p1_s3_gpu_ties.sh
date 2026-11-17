#!/usr/bin/env bash
# Lead lane: APA-P1-S3 r2b's two tie lines (lines 5-6 of the six staged GPU lines; the first run's attempt got exit 76, lock not acquired, nothing ran) (artifacts/apa_p1_s3_r2/STAGED_GPU_LINES.sh), run individually and
# verbatim, in order, from the worktree root. Each line's rc is logged unmasked beside its registered expectation
# (EXPECTED.json, written before staging). Disk stops: >= 25 GB NVMe, >= 20 GB root before each line.
# Prior art: GNU timeout; the earlier P1 lead wrappers; no new algorithm.
# Usage: run_p1_s3_gpu_ties.sh <LOGDIR>
set -u
G=$1
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod
S=$W/artifacts/apa_p1_s3_r2/STAGED_GPU_LINES.sh
LOG=$G/p1_s3_gpu.log
mkdir -p "$G"; cd "$W"
log() { echo "$(date -Is) $*" >> "$LOG"; }
log "P1-S3 ties start pid=$$"
[ "$(sha256sum "$S" | cut -d' ' -f1)" = 976504d6b63c7556f611beac826c4f37815214585da997f3d3fe6d5e3ebddb7f ] || { log "staged hash differs"; exit 3; }
disk() { fnv=$(df -BG --output=avail /mnt/ForgeRealm | tail -1 | tr -dc '0-9'); frt=$(df -BG --output=avail / | tail -1 | tr -dc '0-9'); [ "$fnv" -ge 25 ] && [ "$frt" -ge 20 ] || { log "disk stop: $fnv GB NVMe, $frt GB root"; exit 4; }; }
disk; log "tie_correct start (expected 0)"
forge-gpu-run --min 30 --max 240 -- timeout 210s python -B artifacts/apa_p1_s3_r2/gpu_line.py --case tie_correct >> "$G/tie_correct.out" 2>&1; log "tie_correct rc=$?"
disk; log "eb4_reverse_tie start (expected 1)"
forge-gpu-run --min 30 --max 590 -- timeout 560s python -B artifacts/apa_p1_s3_r2/gpu_line.py --case eb4_reverse_tie >> "$G/eb4_reverse_tie.out" 2>&1; log "eb4_reverse_tie rc=$?"
log "P1-S3 ties done"
