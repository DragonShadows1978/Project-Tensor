#!/usr/bin/env bash
# Foreground host build only; no CUDA context and no network/git FetchContent.
set -euo pipefail
if [[ "${APA_SP4G_BUILD_BOUNDED:-0}" != 1 ]]; then
  exec timeout --signal=TERM --kill-after=5s 570s env APA_SP4G_BUILD_BOUNDED=1 bash "$0"
fi
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
[[ "$ROOT" == /mnt/ForgeRealm/Project-Tensor-wt-apa-sp4g ]] || exit 64
cd "$ROOT"
BUILD="$ROOT/artifacts/apa_sp4g/build"
[[ ! -e "$BUILD/manifest.json" ]] || exit 65
mkdir -p "$BUILD"
NVCC=/usr/local/cuda-12.6/bin/nvcc
CUDA=/usr/local/cuda-12.6
PYINC=$(timeout 10s python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')
PYBIND=$(timeout 10s python3 -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("torch").origin).parent/"include")')
EXT=$(timeout 10s python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
INCLUDES=(-I"$ROOT/tensor_cuda/include" -I"$CUDA/include" -I"$PYINC" -I"$PYBIND")
objects=()
for name in kernels paint_raster paint_bake paint_inpaint dda terrain terrain_objects matmul gemm_apa conv; do
  obj="$BUILD/$name.o"
  echo "Compiling $name.cu"
  timeout 180s "$NVCC" -std=c++17 -O3 -arch=sm_89 -Xcompiler=-fPIC -Xptxas=-v \
    "${INCLUDES[@]}" -dc "tensor_cuda/src/$name.cu" -o "$obj"
  objects+=("$obj")
done
for name in autograd ops bindings; do
  obj="$BUILD/$name.o"
  echo "Compiling $name.cpp"
  timeout 120s g++ -std=c++17 -O3 -fPIC "${INCLUDES[@]}" \
    -c "tensor_cuda/src/$name.cpp" -o "$obj"
  objects+=("$obj")
done
timeout 60s "$NVCC" -arch=sm_89 -Xcompiler=-fPIC -dlink "${objects[@]}" -o "$BUILD/device_link.o"
timeout 60s g++ -shared "${objects[@]}" "$BUILD/device_link.o" -L"$CUDA/lib64" \
  -Wl,-rpath,"$CUDA/lib64" -lcudart -lcublas -lcublasLt -o "$BUILD/_tensor_cuda$EXT"
timeout 20s env PYTHONDONTWRITEBYTECODE=1 python3 scripts/apa_sp4g_make_diag.py
timeout 120s "$NVCC" -std=c++17 -O3 -arch=sm_89 -Xcompiler=-fPIC "${INCLUDES[@]}" -c "$BUILD/diagnostics.cu" -o "$BUILD/diagnostics.o"
timeout 60s g++ -std=c++17 -O3 -fPIC "${INCLUDES[@]}" -c "$BUILD/diag_bindings.cpp" -o "$BUILD/diag_bindings.o"
timeout 60s g++ -shared "$BUILD/diagnostics.o" "$BUILD/diag_bindings.o" "$BUILD/_tensor_cuda$EXT" -L"$CUDA/lib64" -Wl,-rpath,"$CUDA/lib64" -Wl,-rpath,"$BUILD" -lcudart -o "$BUILD/_apa_sp4g_diag$EXT"
PYTHONDONTWRITEBYTECODE=1 python3 - <<'SEAL'
import sys
sys.path.insert(0,'scripts')
from apa_sp4g_common import *
r=verify_sources()
files=[R/p for p in r['source_sha256']]+list(BUILD.glob('*.so'))+[BUILD/'diagnostics.cu',BUILD/'diag_bindings.cpp',R/'scripts/apa_sp4g_make_diag.py',R/'scripts/apa_sp4g_build.sh']
publish(BUILD/'manifest.json',dict(registration_sha256=REG_SHA,evidence_class='host CUDA compile/link; no device execution',files={str(p.relative_to(R)):sha(p) for p in files}))
SEAL
