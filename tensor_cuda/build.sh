#!/usr/bin/env bash
# Build the CUDA engine and drop _tensor_cuda.*.so into tensor_cuda/.
# Usage:  ./build.sh [sm_arch]    e.g. ./build.sh 89   (RTX 4070 SUPER, Ada)
#                                       ./build.sh 86   (RTX 3070, Ampere)
# Toolkit: prefers the newest /usr/local/cuda-* (override with CUDA_HOME=...).
set -euo pipefail
ARCH="${1:-89}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# Use the newest installed CUDA toolkit unless CUDA_HOME is set, so the build
# uses the up-to-date nvcc (12.6) rather than whatever apt put at
# /usr/bin/nvcc (the distro 12.0). The apt 12.0 stays as a fallback.
if [[ -z "${CUDA_HOME:-}" ]]; then
  newest="$(ls -d /usr/local/cuda-*/ 2>/dev/null | sort -V | tail -1)"
  [[ -n "$newest" ]] && CUDA_HOME="${newest%/}"
fi
if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export CUDACXX="$CUDA_HOME/bin/nvcc"
  echo "Using CUDA toolkit: $($CUDACXX --version | sed -n 's/.*release //p')"
fi

cmake -S "$HERE" -B "$HERE/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCH"
cmake --build "$HERE/build" -j
echo "Built. Run: PYTHONPATH=$HERE python -m pytest $HERE/tests"
