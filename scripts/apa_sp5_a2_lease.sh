#!/usr/bin/env bash
# APA-SP5 amendment 2 LONG-LEASE runner (David-authorized for THIS amendment
# only: 1,500 s worker / 1,560 s outer, one cell per lease).
#
# Prior art: SP4G a7 long-lease ceiling probe (this repo, 2026); SP5 a1
# lead_gpu.sh; Unix flock(2) / setsid(1) / timeout(1).
#
# WHY THIS SHAPE: the order requires foreground calls under 10 minutes, but a
# cell may run 25 minutes. So `start` detaches ONE leased job with setsid and
# returns immediately; the caller then polls with `status` from fresh short
# calls. This is NOT a Claude Code background task -- it is a detached OS
# process holding its own flock, exactly as the lead's own detached loop does.
#
# The lease, the 1,500 s timeout(1) leash and the 30 s cooldown all live INSIDE
# the detached job, so they cannot be skipped by a caller who stops polling.
# Nothing is ever killed: if the lease is busy the job waits on flock --wait.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
[[ "$ROOT" == /mnt/ForgeRealm/Project-Tensor-wt-apa-sp5 ]] || exit 64
cd "$ROOT"
RUN=artifacts/apa_sp5/a2_runs
mkdir -p "$RUN" logs

case "${1:?usage: start <tag> <cmd...> | status <tag> | list}" in
  start)
    tag=${2:?tag}; shift 2
    [[ ! -e "$RUN/$tag.state" ]] || { echo "EXISTS $tag"; exit 65; }
    log="logs/apa_sp5_a2_${tag}.log"
    printf 'RUNNING\n' > "$RUN/$tag.state"
    # shellcheck disable=SC2016
    setsid nohup bash -c '
      ROOT=$1; tag=$2; log=$3; RUN=$4; shift 4
      cd "$ROOT"
      export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
             MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
             TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 TC_APA_SP=1
      unset LD_PRELOAD
      exec 9>>/tmp/forge-gpu.lock
      flock --exclusive 9
      timeout --signal=TERM --kill-after=20s 1500 "$@" >"$log" 2>&1
      rc=$?
      sleep 30                       # registered cooldown, held under the lease
      printf "DONE rc=%s\n" "$rc" > "$RUN/$tag.state"
    ' _ "$ROOT" "$tag" "$log" "$RUN" "$@" >/dev/null 2>&1 &
    disown
    echo "STARTED $tag log=$log"
    ;;
  status)
    tag=${2:?tag}
    s=$(cat "$RUN/$tag.state" 2>/dev/null || echo MISSING)
    echo "STATE $tag: $s"
    [[ -f "logs/apa_sp5_a2_${tag}.log" ]] && tail -c 1200 "logs/apa_sp5_a2_${tag}.log"
    ;;
  list) grep -H . "$RUN"/*.state 2>/dev/null || echo 'no runs' ;;
  *) exit 64 ;;
esac
