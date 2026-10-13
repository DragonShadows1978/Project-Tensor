#!/usr/bin/env bash
# Build the tc_signal add-on and drop _tc_signal.*.so into tc_signal/tc_signal/.
# Usage:  ./build.sh [sm_arch]    e.g. ./build.sh 89   (RTX 4070 SUPER, Ada; the default)
# Toolkit: the newest /usr/local/cuda-* (CUDA 12.6 here), as tensor_cuda/build.sh; override with CUDA_HOME=...
# Offline pybind11: PYBIND11_SOURCE_DIR=<pybind11 v2.12.0 source tree> is passed as
#   -DFETCHCONTENT_SOURCE_DIR_PYBIND11 (for example core's tensor_cuda/build/_deps/pybind11-src, used read-only).
set -euo pipefail
ARCH="${1:-89}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [[ -z "${CUDA_HOME:-}" ]]; then
  newest="$(ls -d /usr/local/cuda-*/ 2>/dev/null | sort -V | tail -1)"
  [[ -n "$newest" ]] && CUDA_HOME="${newest%/}"
fi
if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export CUDACXX="$CUDA_HOME/bin/nvcc"
  echo "Using CUDA toolkit: $CUDA_HOME -> $(readlink -f "$CUDA_HOME") ($($CUDACXX --version | sed -n 's/.*release //p'))"
fi

EXTRA=()
if [[ -n "${PYBIND11_SOURCE_DIR:-}" ]]; then
  echo "Using pybind11 source: $PYBIND11_SOURCE_DIR (offline)"
  EXTRA+=("-DFETCHCONTENT_SOURCE_DIR_PYBIND11=$PYBIND11_SOURCE_DIR")
fi

cmake -S "$HERE" -B "$HERE/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCH" "${EXTRA[@]}"
cmake --build "$HERE/build" -j
echo "Built: $(ls "$HERE"/tc_signal/_tc_signal*.so)"
echo "Use:   PYTHONPATH=$HERE python3 -c 'import tc_signal'"
