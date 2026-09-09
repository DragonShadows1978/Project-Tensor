#!/usr/bin/env bash
# Prior art: flock (Unix advisory leases) and GNU timeout; existing SP1/SP2
# runner discipline. SP3 adapts to one full-model or one captured-layer job.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
[[ "$ROOT" == /mnt/ForgeRealm/Project-Tensor-wt-apa-sp3 || "$ROOT" == /mnt/ForgeRealm/Project-Tensor-wt-apa-sp3-a4 || "$ROOT" == /mnt/ForgeRealm/Project-Tensor-wt-apa-sp3-a6 ]] || exit 64
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_MODULES_CACHE="$ROOT/artifacts/apa_sp3/hf_modules_a4"
export CUDA_VISIBLE_DEVICES=0
action=${1:-summary}
case "$action" in
  list) exec timeout 20s python3 scripts/apa_sp3_gpu.py --dry-run ;;
  summary) exec timeout 30s python3 scripts/apa_sp3_report.py ;;
  resume)
    [[ $# -le 2 ]] || exit 64
    bits=${2:-4}
    [[ "$bits" == 4 || "$bits" == 8 ]] || exit 64
    next=$(timeout 20s python3 scripts/apa_sp3_gpu.py --next --bits "$bits")
    [[ -n "$next" ]] || exit 0
    exec bash "$0" run "$next"
    ;;
  run)
    [[ $# == 2 ]] || exit 64
    job=$2
    if [[ "${APA_SP3_OUTER:-0}" != 1 ]]; then
      exec timeout --signal=TERM --kill-after=3s 585s env APA_SP3_OUTER=1 bash "$0" run "$job"
    fi
    # Validate registry and dependencies before acquiring lease or probing GPU.
    decision=$(timeout 15s python3 scripts/apa_sp3_control.py preflight "$job")
    if [[ "$decision" == NON_FIT ]]; then
      echo 'Registered/planned NON_FIT receipt recorded before GPU lease; no retry.'
      exit 0
    fi
    exec bash "$0" _leased "$job"
    ;;
  _leased)
    [[ $# == 2 ]] || exit 64
    [[ "${APA_SP3_OUTER:-0}" == 1 ]] || exit 64
    job=$2
    exec 9>>/tmp/forge-gpu.lock
    flock --exclusive --wait 20 9 || { echo 'BLOCKED: GPU lease busy; operator retains right of way'; exit 75; }
    # Fail closed on missing device or any existing compute process. No signals.
    timeout 10s python3 scripts/apa_sp3_control.py idle
    export APA_SP3_LEASE=1
    kind=$(timeout 10s python3 scripts/apa_sp3_control.py kind "$job")
    observer=$(timeout 10s python3 scripts/apa_sp3_control.py interposer "$job")
    if [[ "$observer" == 1 ]]; then
      export LD_PRELOAD="$ROOT/artifacts/apa_sp3/build/libapa_sp3_peak.so"
    else
      unset LD_PRELOAD
    fi
    mkdir -p logs artifacts/apa_sp3/jobs artifacts/apa_sp3/scratch
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    rail=$(timeout 10s python3 scripts/apa_sp3_control.py timeout "$job")
    set +e
    timeout --signal=TERM --kill-after=5s "${rail}s" python3 scripts/apa_sp3_gpu.py --worker "$job" >"logs/apa_sp3_${job}_${stamp}.log" 2>&1
    rc=$?
    set -e
    unset LD_PRELOAD
    timeout 10s python3 scripts/apa_sp3_control.py finish "$job" "$rc" "logs/apa_sp3_${job}_${stamp}.log"
    # Foreground cooldown while retaining lease; no daemon or background wait.
    timeout 32s sleep 30
    exit "$rc"
    ;;
  *) echo 'usage: apa_sp3_lead_gpu.sh list|run CELL|resume [4|8]|summary' >&2; exit 64 ;;
esac
