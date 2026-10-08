#!/usr/bin/env bash
# APA-MF2.2 G2 driver: one process per engine, each under the GPU lock, then compare.
set -euo pipefail
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2
R=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf21rev/artifacts/apa_mf21rev/baselines
OUT=$W/artifacts/apa_mf2_2/identity; mkdir -p "$OUT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
declare -A ENG=([fd32b6d]=$W/artifacts/apa_mf2_2/build_fd32b6d [mf2]=$W/artifacts/apa_mf2/build [mf2_1]=$W/artifacts/apa_mf2_1/build [mf2_2]=$W/artifacts/apa_mf2_2/build)
declare -A PKG=([fd32b6d]=$R/fd32b6d/pkg [mf2]=$W/artifacts/apa_mf2_1/before/pkg [mf2_1]=$R/c5c468d/pkg [mf2_2]=$W/tensor_cuda)
for a in fd32b6d mf2 mf2_1 mf2_2; do
  ( export APA_MF2_2_ENGINE=${ENG[$a]} APA_MF2_2_PKG=${PKG[$a]}; source $W/scripts/apa_mf2_2_env.sh
    export PYTHONPATH=${ENG[$a]}:${PKG[$a]}:$W/scripts
    flock -w 900 /tmp/forge-gpu.lock timeout 600 python3 $W/scripts/apa_mf2_2_identity.py arm $a $OUT/arm_${a}_$STAMP.json )
done
python3 $W/scripts/apa_mf2_2_identity.py compare $OUT $STAMP $OUT/compare_$STAMP.json
