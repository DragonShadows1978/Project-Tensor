#!/usr/bin/env bash
# Prior art: SP3/SP4G (2026), Unix flock foreground lease. A6 uses
# cooperative worker checks; no signal-based timeout and no background wait.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
[[ "$ROOT" == /mnt/ForgeRealm/Project-Tensor-wt-apa-sp4g ]] || exit 64
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
unset LD_PRELOAD
case "${1:-summary}" in
  list) exec python3 scripts/apa_sp4g_a6_gpu.py list ;;
  summary) exec python3 scripts/apa_sp4g_a6_report.py ;;
  resume)
    next=$(python3 scripts/apa_sp4g_a6_gpu.py next)
    [[ -n "$next" ]] || exit 0
    exec bash "$0" run "$next" ;;
  run)
    [[ $# == 2 ]] || exit 64
    job=$2
    decision=$(python3 scripts/apa_sp4g_a6_gpu.py preflight "$job")
    [[ "$decision" != DONE ]] || exit 0
    if [[ "$decision" == GPU ]]; then
      exec 9>>/tmp/forge-gpu.lock
      flock --exclusive --wait 20 9 || { echo 'BLOCKED: GPU lease busy'; exit 75; }
      python3 scripts/apa_sp4g_a6_gpu.py idle
      export APA_SP4G_LEASE=1
    fi
    stamp=$(date -u +%Y%m%dT%H%M%S%N)
    log="logs/apa_sp4g_a6_${job}_${stamp}.log"
    set +e
    python3 scripts/apa_sp4g_a6_gpu.py worker "$job" >"$log" 2>&1
    rc=$?
    python3 scripts/apa_sp4g_a6_gpu.py finish "$job" "$rc" "$log"
    finish_rc=$?
    if [[ "$decision" == GPU ]]; then sleep 30; fi
    set -e
    [[ "$finish_rc" == 0 ]] || exit "$finish_rc"
    exit "$rc" ;;
  *) echo 'usage: apa_sp4g_a6_lead_gpu.sh list|run CELL|resume|summary' >&2; exit 64 ;;
esac
