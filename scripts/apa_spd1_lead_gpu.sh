#!/usr/bin/env bash
# Prior art: SP1 leased runner (Project-Tensor seats, 2026); util-linux flock
# advisory locking and GNU coreutils timeout. Reuse safety pattern; no novel
# scheduling algorithm. One foreground cell, no automatic retries or waits.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
[[ "$ROOT" == /mnt/ForgeRealm/Project-Tensor-wt-apa-spd1 ]] || exit 64
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mode=${1:-list}
case "$mode" in
  list|summary|inventory|--dry-run)
    exec timeout --signal=TERM --kill-after=2s 50s python3 -B scripts/apa_spd1_bench.py "$mode" ;;
  run) target=${2:?run requires one listed cell id} ;;
  resume) target=NEXT ;;
  _lease)
    [[ "${APA_SPD1_BOUNDED:-0}" == 1 ]] || exit 64
    target=${2:?missing target} ;;
  *) echo 'usage: apa_spd1_lead_gpu.sh list|run CELL|resume|summary|inventory|--dry-run' >&2; exit 64 ;;
esac
if [[ "$mode" != _lease ]]; then
  exec timeout --signal=TERM --kill-after=5s 580s env APA_SPD1_BOUNDED=1 bash "$0" _lease "$target"
fi
# Inherited descriptor is checked against the lock inode by the worker.
exec 9>>/tmp/forge-gpu.lock
flock -E 75 -w 5 9 || { echo 'BLOCKED: GPU lease busy; no work launched' >&2; exit 75; }
if [[ "$target" == NEXT ]]; then
  target=$(timeout 5s python3 -B scripts/apa_spd1_bench.py next)
  if [[ "$target" == DONE ]]; then
    echo 'All cells attempted. Run summary; failures require an explicit run CELL.'
    exit 0
  fi
fi
# Exact membership validation is performed by the worker too.
[[ "$target" =~ ^((prefill|decode)_s(512|2048|8192|32768)_d(64|128)_c[01]_h(4_kv4|8_kv2)|e1_prefill_l512_s(8192|32768)_d128_c1_h16_kv4)$ ]] || exit 64
export CUDA_VISIBLE_DEVICES=${APA_SPD1_GPU:-0}
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]] || exit 64
command -v nvidia-smi >/dev/null || { echo 'BLOCKED: nvidia-smi unavailable; no GPU work launched' >&2; exit 75; }
if ! busy=$(timeout 5s nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader); then
  echo 'BLOCKED: GPU process probe failed; no work launched' >&2
  exit 75
fi
[[ -z "$busy" ]] || { echo "BLOCKED: existing GPU compute process(es): $busy" >&2; exit 75; }
export APA_SPD1_LEASED=1 TC_APA_SP=1 TC_APA_SELECTIVE_PATH=0 NVIDIA_TF32_OVERRIDE=0
export PYTORCH_ALLOC_CONF=backend:native
unset PYTORCH_CUDA_ALLOC_CONF TC_APAMQ_DF_V2 TC_APAMQ_DF_V3 TC_APAMQ_DF_V4 TC_APA_STATS_PART_KEYS TC_APA_FRAC TC_ATTN_QTILE
mkdir -p artifacts/apa_spd1/gpu
stamp=$(timeout 3s python3 -B -c 'import time;print(time.time_ns())')
log="artifacts/apa_spd1/gpu/${target}.${stamp}.log"
# Hold the lease through cooldown on success, error, and worker timeout.
trap 'timeout 35s python3 -B -c "import time;time.sleep(30)"' EXIT
set +e
timeout --signal=TERM --kill-after=5s 510s python3 -B -u scripts/apa_spd1_bench.py _worker "$target" "$stamp" >"$log" 2>&1
rc=$?
set -e
timeout 5s python3 -B scripts/apa_spd1_bench.py _finish "$target" "$stamp" "$rc"
echo "Worker exit=$rc; transcript=$log"
exit "$rc"
