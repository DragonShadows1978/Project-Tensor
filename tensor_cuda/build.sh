#!/usr/bin/env bash
# Build the CUDA engine and drop _tensor_cuda.*.so into tensor_cuda/.
# Usage:  ./build.sh [sm_arch]    e.g. ./build.sh 86   (RTX 3070)
set -euo pipefail
ARCH="${1:-86}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cmake -S "$HERE" -B "$HERE/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCH"
cmake --build "$HERE/build" -j
echo "Built. Run: PYTHONPATH=$HERE python -m pytest $HERE/tests"
