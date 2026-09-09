#!/usr/bin/env bash
# One registered measurement class or epsilon row per invocation; no loops/waits in background.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
[[ "$ROOT" == /mnt/ForgeRealm/Project-Tensor-wt-apa-sp2 ]] || exit 64
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mode=${1:-resume}
if [[ "$mode" == list || "$mode" == summary ]]; then
  exec timeout 60s python3 scripts/apa_sp2_gpu.py "$mode"
fi
if [[ "$mode" == resume ]]; then
  target=$(timeout 60s python3 scripts/apa_sp2_gpu.py next)
  [[ "$target" != DONE ]] || { echo 'Complete; run summary.'; exit 0; }
  [[ "$target" != BLOCKED_* ]] || { echo "$target; inspect failure, no automatic retry."; exit 75; }
elif [[ "$mode" == run || "$mode" == _lease ]]; then
  target=${2:?run requires a target from list}
else
  echo 'usage: apa_sp2_lead_gpu.sh list|run TARGET|resume|summary' >&2; exit 64
fi
if [[ "$target" == freeze ]]; then
  exec timeout 60s python3 scripts/apa_sp2_gpu.py freeze
fi
timeout 10s python3 scripts/apa_sp2_gpu.py validate "$target"
if [[ "$mode" != _lease ]]; then
  exec timeout --signal=TERM --kill-after=5s 590s bash "$0" _lease "$target"
fi
exec 9>>/tmp/forge-gpu.lock
flock -w 5 9 || { echo 'BLOCKED: GPU lease busy; no workload launched.' >&2; exit 75; }
export CUDA_VISIBLE_DEVICES=${APA_SP2_GPU:-0}
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]] || { echo 'Exactly one numeric GPU required.' >&2; exit 64; }
command -v nvidia-smi >/dev/null || { echo 'BLOCKED: no nvidia-smi'; exit 75; }
busy=$(timeout 10s nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)
[[ -z "$busy" ]] || { echo "BLOCKED: existing GPU compute PIDs: $busy" >&2; exit 75; }
trap 'timeout 35s python3 -c "import time;time.sleep(30)"' EXIT
export APA_SP2_LEASED=1 TC_APA_SP=1 TC_APA_SELECTIVE_PATH=0
unset TC_APAMQ_DF_V2 TC_APAMQ_DF_V3 TC_APAMQ_DF_V4 TC_APA_STATS_PART_KEYS TC_APA_FRAC
mkdir -p artifacts/apa_sp2/gpu
stamp=$(timeout 5s python3 -c 'import time;print(time.time_ns())')
log="artifacts/apa_sp2/gpu/${target}.${stamp}.log"
set +e
timeout --signal=TERM --kill-after=5s 510s python3 -u scripts/apa_sp2_gpu.py worker "$target" >"$log" 2>&1
rc=$?
set -e
echo "Worker exit=$rc; transcript=$log"
# Persist timeout/process-exit evidence even if Python could not write its receipt.
timeout 10s python3 - "$target" "$log" "$rc" <<'PY'
import sys,time
sys.path.insert(0,'scripts')
from apa_sp2_common import ART,write_new,fingerprint
write_new(ART/'gpu'/f'{sys.argv[1]}.{time.time_ns()}.exit.json',
          dict(target=sys.argv[1],status='WORKER_EXIT',returncode=int(sys.argv[3]),log=sys.argv[2],fingerprint=fingerprint()))
PY
exit "$rc"
