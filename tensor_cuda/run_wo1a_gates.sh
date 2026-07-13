#!/usr/bin/env bash
# WO-1A Release build + HY3D kernel evidence gates. No git mutation occurs.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ARCH="${TC_CUDA_ARCH:-89}"

if [[ "${TC_WO1A_SKIP_BUILD:-0}" != "1" ]]; then
  "$HERE/build.sh" "$ARCH"
fi

export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
# K4 keeps composed as production default; the K1 fused gates opt in here.
export TC_FUSED_SDPA_NONCAUSAL=1

if ! python3 -c 'import numpy as np; import tensor_cuda as tc; tc.tensor(np.zeros(1, dtype=np.float32)); tc.synchronize()' 2>/dev/null; then
  echo "WO1A gates NOT RUN: this process has no usable CUDA device."
  exit 0
fi

export TC_WO1A_FULL_SHAPES=1
python3 -m pytest "$HERE/tests/test_hy3d_engine_ops.py" -q -s
python3 "$HERE/tests/bench_wo1a_sdpa.py" --dtype float16

# K5 is intentionally run without duplicating the expensive K1 full-shape
# references; the normal suite retains its small always-on WO-1A coverage.
unset TC_FUSED_SDPA_NONCAUSAL
export TC_WO1A_FULL_SHAPES=0
python3 -m pytest "$HERE/tests" -q
