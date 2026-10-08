#!/usr/bin/env bash
# APA-MF2.1 G3(b) arm launcher: one arm per process. usage: apa_mf2_1_identity.sh {before|after|engine_iso}
set -euo pipefail
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2
case "$1" in
  before)     ENG=$W/artifacts/apa_mf2/build;   PKG=$W/artifacts/apa_mf2_1/before/pkg ;;
  after)      ENG=$W/artifacts/apa_mf2_1/build; PKG=$W/tensor_cuda ;;
  engine_iso) ENG=$W/artifacts/apa_mf2/build;   PKG=$W/tensor_cuda ;;
  *) echo "usage: $0 {before|after|engine_iso}" >&2; exit 2 ;;
esac
export APA_MF2_1_ENGINE=$ENG APA_MF2_1_PKG=$PKG
source $W/scripts/apa_mf2_1_env.sh
cd /tmp
exec python3 $W/scripts/apa_mf2_1_identity.py "$1" "$W/artifacts/apa_mf2_1/identity/$1.json"
