#!/usr/bin/env bash
# APA-MF2.1 GRM arm launcher (one arm per process). usage: apa_mf2_1_grm.sh {before|after}
set -euo pipefail
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2
case "$1" in
  before) export APA_MF2_1_ENGINE=$W/artifacts/apa_mf2/build APA_MF2_1_PKG=$W/artifacts/apa_mf2_1/before/pkg ;;
  after)  export APA_MF2_1_ENGINE=$W/artifacts/apa_mf2_1/build APA_MF2_1_PKG=$W/tensor_cuda ;;
  *) exit 2 ;;
esac
source $W/scripts/apa_mf2_1_env.sh
cd /tmp
exec python3 $W/scripts/apa_mf2_1_grm.py "$1"
