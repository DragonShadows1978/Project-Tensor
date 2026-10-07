#!/usr/bin/env bash
# Prior art: flock(1), util-linux advisory locking; GNU timeout bounded calls.
# Mine: sequential pinned-revision launcher. No prior art known to me for this
# review-specific orchestration. Never set a driver override; exit codes preserved.
set -euo pipefail
APA_REVIEW_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
APA_MF2_SEAT=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2
CELL=${1:?usage: apa_mf21rev_gpu.sh CELL}
case "$CELL" in
  sentinel|reductions|fused_overflow|definition|short|masks|training|identity|identity_mf2|extension) ;;
  *) echo "unknown CELL: $CELL" >&2; exit 2 ;;
esac
case ":${LD_LIBRARY_PATH:-}:${PATH}:${LD_PRELOAD:-}:" in
  *595.84*) echo "retired 595.84 override detected; refusing" >&2; exit 2 ;;
esac
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 TC_ATTN_QTILE=0
# Native driver; no LD_LIBRARY_PATH alteration.
export APA_MF21REV_AFTER_ENGINE=${APA_MF21REV_AFTER_ENGINE:-$APA_MF2_SEAT/artifacts/apa_mf2_1/build}
export APA_MF21REV_BEFORE_ENGINE=${APA_MF21REV_BEFORE_ENGINE:-$APA_MF2_SEAT/artifacts/apa_mf2/build}
export APA_MF21REV_PRODUCTION_ENGINE=${APA_MF21REV_PRODUCTION_ENGINE:-$APA_MF21REV_BEFORE_ENGINE}
export APA_MF21REV_REV=c5c468d APA_MF21REV_ENGINE=$APA_MF21REV_AFTER_ENGINE
export APA_MF21REV_PKG=$APA_REVIEW_ROOT/artifacts/apa_mf21rev/baselines/c5c468d/pkg
export PYTHONPATH=$APA_MF21REV_ENGINE:$APA_MF21REV_PKG
STAMP=$(date -u +%Y%m%dT%H%M%SZ)_$$
export APA_MF21REV_RUN=$STAMP
LOG=$APA_REVIEW_ROOT/logs/apa_mf21rev_gpu_${CELL}_${STAMP}.log
# Lock is nonblocking: shared-resource busy is RED, no background wait.
# The timeout applies to THIS launcher's children only. No pipeline masks status.
set +e
flock -n /tmp/forge-gpu.lock timeout 600 python3 "$APA_REVIEW_ROOT/tests/apa_mf21rev/gpu_probe.py" "$CELL" > "$LOG" 2>&1
RC=$?
set -e
cat "$LOG"
echo "receipt=$LOG exit_code=$RC"
exit "$RC"
