#!/usr/bin/env bash
# One shape class / invocation. Every GPU invocation is leased and <10 minutes.
# resume runs ONE next incomplete class, then returns; no unbounded loop.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
[[ "$ROOT" == /mnt/ForgeRealm/Project-Tensor-wt-apa-sp1 ]] || exit 64
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mode=${1:-resume}
if [[ "$mode" == list || "$mode" == summary ]]; then
  exec timeout 30s python3 scripts/apa_sp1_gpu.py "$mode"
fi
if [[ "$mode" == resume ]]; then
  target=$(timeout 30s python3 scripts/apa_sp1_gpu.py next)
  if [[ "$target" == DONE ]]; then
    echo 'All registered classes have current completed receipts. Run summary.'
    exit 0
  fi
elif [[ "$mode" == run ]]; then
  target=${2:?run requires a listed shape id or boundary/legacy/selector/probe}
elif [[ "$mode" == _lease ]]; then
  target=${2:?internal lease target}
else
  echo "usage: $0 list|summary|resume|run SHAPE_OR_BOUNDARY_LEGACY_SELECTOR_PROBE" >&2
  exit 64
fi
case "$target" in
  boundary|legacy|selector|probe) ;;
  *) [[ "$target" =~ ^(prefill|decode)_s(512|2048|8192|32768)_d(64|128)_c[01]_h(4_kv4|8_kv2)$ ]] || {
       echo 'Invalid shape id; use list' >&2; exit 64;
     } ;;
esac
if [[ "$mode" != _lease ]]; then
  # 5s lease + 10s probe + 5s stamp + 510s worker + teardown/cooling <590s.
  exec timeout --signal=TERM --kill-after=5s 590s bash "$0" _lease "$target"
fi
exec 9>>/tmp/forge-gpu.lock
flock -w 5 9 || { echo 'BLOCKED: /tmp/forge-gpu.lock busy; no GPU work launched' >&2; exit 75; }
export CUDA_VISIBLE_DEVICES=${APA_SP1_GPU:-0}
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]] || { echo 'Exactly one numeric GPU index required' >&2; exit 64; }
# Do not fight non-cooperating clients; never terminate another process.
if command -v nvidia-smi >/dev/null; then
  busy=$(timeout 10s nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)
  if [[ -n "$busy" ]]; then
    echo "BLOCKED: existing compute process(es) on leased GPU: $busy" >&2
    exit 75
  fi
fi
# Hold lease throughout cooldown, including a worker failure/timeout.
trap 'timeout 35s python3 -c "import time;time.sleep(30)"' EXIT
export APA_SP1_LEASED=1 TC_APA_SP=1 TC_APA_SELECTIVE_PATH=0
unset TC_APAMQ_DF_V2 TC_APAMQ_DF_V3 TC_APAMQ_DF_V4 TC_APA_STATS_PART_KEYS TC_APA_FRAC
mkdir -p artifacts/apa_sp1/gpu
case "$target" in
  boundary|legacy|selector|probe) args=("$target") ;;
  *) args=(shape "$target") ;;
esac
# A unique transcript records timeouts even when Python cannot save JSON.
stamp=$(timeout 5s python3 -c 'import time;print(time.time_ns())')
log="artifacts/apa_sp1/gpu/${target}.${stamp}.log"
set +e
timeout --signal=TERM --kill-after=5s 510s python3 -u scripts/apa_sp1_gpu.py "${args[@]}" >"$log" 2>&1
rc=$?
set -e
echo "Worker exit=$rc; transcript=$log"
if [[ $rc -ne 0 ]]; then
  echo "BLOCKED/FAIL: inspect transcript; no automatic retries, threshold changes or process kills."
fi
exit "$rc"
