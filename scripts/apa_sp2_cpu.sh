#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
[[ "$ROOT" == /mnt/ForgeRealm/Project-Tensor-wt-apa-sp2 ]] || exit 64
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec timeout --signal=TERM --kill-after=5s 180s python3 scripts/apa_sp2_cpu.py
