#!/usr/bin/env bash
# Prior art: SP3 leased runner (2026), Unix flock/GNU timeout. Foreground
# bounded owned children only; no foreign PID signals, background waits or git.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
[[ "$ROOT" == /mnt/ForgeRealm/Project-Tensor-wt-apa-sp4g ]] || exit 64
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
unset LD_PRELOAD
action=${1:-summary}
case "$action" in
 list) exec timeout 20s python3 scripts/apa_sp4g_a2_gpu.py list ;;
 summary) exec timeout 60s python3 scripts/apa_sp4g_a2_report.py ;;
 resume)
   [[ $# == 1 ]] || exit 64
   if [[ "${APA_SP4G_A2_OUTER:-0}" != 1 ]]; then
     exec timeout --signal=TERM --kill-after=3s 585s env APA_SP4G_A2_OUTER=1 bash "$0" resume
   fi
   next=$(timeout 60s python3 scripts/apa_sp4g_a2_gpu.py next)
   [[ -n "$next" ]] || exit 0
   exec bash "$0" run "$next" ;;
 run)
   [[ $# == 2 ]] || exit 64
   job=$2
   if [[ "${APA_SP4G_A2_OUTER:-0}" != 1 ]]; then
     exec timeout --signal=TERM --kill-after=3s 585s env APA_SP4G_A2_OUTER=1 bash "$0" run "$job"
   fi
   decision=$(timeout 60s python3 scripts/apa_sp4g_a2_gpu.py preflight "$job")
   [[ "$decision" != DONE ]] || exit 0
   if [[ "$decision" == GPU ]]; then
     exec 9>>/tmp/forge-gpu.lock
     flock --exclusive --wait 20 9 || { echo 'BLOCKED: GPU lease busy'; exit 75; }
     timeout 20s python3 scripts/apa_sp4g_a2_gpu.py idle
     export APA_SP4G_LEASE=1
   fi
   stamp=$(date -u +%Y%m%dT%H%M%S%N)
   log="logs/apa_sp4g_a2_${job}_${stamp}.log"
   set +e
   timeout --signal=TERM --kill-after=5s 285s python3 scripts/apa_sp4g_a2_gpu.py worker "$job" >"$log" 2>&1
   rc=$?
   timeout 30s python3 scripts/apa_sp4g_a2_gpu.py finish "$job" "$rc" "$log"
   finish_rc=$?
   if [[ "$decision" == GPU ]]; then sleep 30; fi
   set -e
   [[ "$finish_rc" == 0 ]] || exit "$finish_rc"
   exit "$rc" ;;
 *) echo 'usage: apa_sp4g_lead_gpu.sh list|run CELL|resume|summary' >&2; exit 64 ;;
esac
