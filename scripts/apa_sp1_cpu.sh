#!/usr/bin/env bash
# One bounded foreground invocation: unit | calibrate | shard CLASS INDEX | summarize.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
[[ "$ROOT" == /mnt/ForgeRealm/Project-Tensor-wt-apa-sp1 ]] || exit 64
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
case "${1:-unit}" in
  unit) exec timeout 120s python3 -m pytest -q -p no:cacheprovider tensor_cuda/tests/test_apa_sp1_reference.py tensor_cuda/tests/test_apa_sp1_host.py tensor_cuda/tests/test_apa_sp1_harness.py ;;
  calibrate) exec timeout 240s python3 scripts/apa_sp1_cpu.py calibrate ;;
  shard) exec timeout 540s python3 scripts/apa_sp1_cpu.py shard --class "$2" --index "$3" ;;
  summarize) exec timeout 60s python3 scripts/apa_sp1_cpu.py summarize ;;
  *) echo "usage: $0 unit|calibrate|shard CLASS INDEX|summarize" >&2; exit 64 ;;
esac
