#!/usr/bin/env bash
# APA-MF2.2b: census arms (Amendment-1 harness scripts/apa_mf2_2a_census.py, unchanged), one process
# per arm under the GPU lock. after1f = Amendment-1 engine + the pre-Amendment-2 package snapshot;
# after2 = the same engine + this tree's package. NO build. Output: artifacts/apa_mf2_2b/census/.
set -euo pipefail
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2
OUT=$W/artifacts/apa_mf2_2b/census; mkdir -p "$OUT"
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
case ":${LD_LIBRARY_PATH:-}:${PATH}:${LD_PRELOAD:-}:" in *595.84*) echo "595.84 override present; refusing" >&2; exit 2;; esac
ENGINE=$W/artifacts/apa_mf2_2a/build
declare -A PKG=([after1f]=$W/artifacts/apa_mf2_2b/before/pkg [after2]=$W/tensor_cuda)
for a in ${ARMS:-after1f after2}; do
  ( export APA_MF2_2A_ENGINE=$ENGINE PYTHONPATH=$ENGINE:${PKG[$a]} PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 TC_ATTN_QTILE=0
    flock -w 900 /tmp/forge-gpu.lock timeout 600 python3 $W/scripts/apa_mf2_2a_census.py arm $a $OUT/arm_${a}_$STAMP.json )
done
echo "stamp=$STAMP"
