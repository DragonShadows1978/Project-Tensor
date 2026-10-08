#!/usr/bin/env bash
# APA-MF2.2 G3 driver: MF2.1 (before) and MF2.2 (after), one process per engine and cell kind.
set -euo pipefail
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2
R=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf21rev/artifacts/apa_mf21rev/baselines
OUT=$W/artifacts/apa_mf2_2/n2; mkdir -p "$OUT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
declare -A ENG=([mf2_1]=$W/artifacts/apa_mf2_1/build [mf2_2]=$W/artifacts/apa_mf2_2/build)
declare -A PKG=([mf2_1]=$R/c5c468d/pkg [mf2_2]=$W/tensor_cuda)
KINDS=${KINDS:-"sweep real"}
for a in ${ARMS:-mf2_1 mf2_2}; do
  for kind in $KINDS; do
    ( export APA_MF2_2_ENGINE=${ENG[$a]} APA_MF2_2_PKG=${PKG[$a]}; source $W/scripts/apa_mf2_2_env.sh
      export PYTHONPATH=${ENG[$a]}:${PKG[$a]}:$W/scripts
      if [[ $kind == sweep ]]; then
        echo "== $a sweep"; flock -w 900 /tmp/forge-gpu.lock timeout 600 python3 $W/scripts/apa_mf2_2_n2.py sweep $OUT/sweep_${a}_$STAMP.json
      else
        echo "== $a real 1.7B"; flock -w 900 /tmp/forge-gpu.lock timeout 600 python3 $W/scripts/apa_mf2_2_n2.py real Qwen3-1.7B 0,16,27 $OUT/real_Qwen3-1.7B_${a}_$STAMP.json
        echo "== $a real 4B"; flock -w 900 /tmp/forge-gpu.lock timeout 600 python3 $W/scripts/apa_mf2_2_n2.py real Qwen3-4B 0,21,35 $OUT/real_Qwen3-4B_${a}_$STAMP.json
      fi )
  done
done
echo "stamp=$STAMP"
