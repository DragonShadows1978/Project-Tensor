#!/usr/bin/env bash
# APA-MF2.2a item 4 driver: one process per engine under the GPU lock; then compare.
# usage: ARMS="fd32b6d mf2 mf2_1 mf2_2 [after]" STAMP=... bash apa_mf2_2a_census.sh
set -euo pipefail
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2
B=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf21rev/artifacts/apa_mf21rev/baselines
OUT=$W/artifacts/apa_mf2_2a/census; mkdir -p "$OUT"
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
case ":${LD_LIBRARY_PATH:-}:${PATH}:${LD_PRELOAD:-}:" in *595.84*) echo "595.84 override present; refusing" >&2; exit 2;; esac
declare -A ENG=([fd32b6d]=$W/artifacts/apa_mf2_2/build_fd32b6d [mf2]=$W/artifacts/apa_mf2/build [mf2_1]=$W/artifacts/apa_mf2_1/build [mf2_2]=$W/artifacts/apa_mf2_2/build [after]=$W/artifacts/apa_mf2_2a/build [mf2_2r]=$W/artifacts/apa_mf2_2/build)
declare -A PKG=([fd32b6d]=$B/fd32b6d/pkg [mf2]=$W/artifacts/apa_mf2_1/before/pkg [mf2_1]=$B/c5c468d/pkg [mf2_2]=$W/tensor_cuda [after]=$W/tensor_cuda [mf2_2r]=$W/tensor_cuda)
for a in ${ARMS:-fd32b6d mf2 mf2_1 mf2_2}; do
  [[ -f $OUT/arm_${a}_$STAMP.json ]] && continue
  ( export APA_MF2_2A_ENGINE=${ENG[$a]} PYTHONPATH=${ENG[$a]}:${PKG[$a]} PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 TC_ATTN_QTILE=0
    flock -w 900 /tmp/forge-gpu.lock timeout 600 python3 $W/scripts/apa_mf2_2a_census.py arm $a $OUT/arm_${a}_$STAMP.json )
done
echo "stamp=$STAMP"
