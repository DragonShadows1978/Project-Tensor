#!/usr/bin/env bash
# APA-SP5 leased GPU runner. Prior art: SP3/SP4G lead_gpu.sh (this repo,
# 2026); Unix flock(2) advisory lease. Foreground only -- no background
# wait, no signal games, and the 30 s cooldown is INSIDE the lease exit
# path so a caller can never skip it. Never kills a process it did not
# start; if the lease is busy it reports BLOCKED and exits 75.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
[[ "$ROOT" == /mnt/ForgeRealm/Project-Tensor-wt-apa-sp5 ]] || exit 64
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0 TC_APA_SP=1
unset LD_PRELOAD
WORKER_RAIL=${APA_SP5_WORKER_RAIL:-285}
[[ $# -ge 1 ]] || { echo 'usage: apa_sp5_lead_gpu.sh <script.py> [args...]' >&2; exit 64; }
exec 9>>/tmp/forge-gpu.lock
flock --exclusive --wait 30 9 || { echo 'BLOCKED: GPU lease busy'; exit 75; }
stamp=$(date -u +%Y%m%dT%H%M%S)
base=$(basename "$1" .py)
log="logs/apa_sp5_${base}_${stamp}.log"
mkdir -p logs
timeout --signal=TERM --kill-after=10s "$WORKER_RAIL" python3 "$@" 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
sleep 30      # registered cooldown, held under the lease
echo "SP5-CELL rc=$rc log=$log rail=${WORKER_RAIL}s"
exit "$rc"
