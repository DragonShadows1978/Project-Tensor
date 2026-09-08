#!/usr/bin/env bash
# APA-SP5 host build. Prior art: SP4G build script (2026, this repo), NVIDIA
# nvcc separate compilation + device link (CUDA docs). Foreground host compile
# only; no CUDA context is created, so no GPU lease is taken.
set -euo pipefail
if [[ "${APA_SP5_BUILD_BOUNDED:-0}" != 1 ]]; then
  exec timeout --signal=TERM --kill-after=5s 570s env APA_SP5_BUILD_BOUNDED=1 bash "$0"
fi
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
[[ "$ROOT" == /mnt/ForgeRealm/Project-Tensor-wt-apa-sp5 ]] || exit 64
cd "$ROOT"
BUILD="$ROOT/artifacts/apa_sp5/build"
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
  timeout 300s "$NVCC" -std=c++17 -O3 -arch=sm_89 -Xcompiler=-fPIC \
    "${INCLUDES[@]}" -dc "tensor_cuda/src/$name.cu" -o "$obj"
  objects+=("$obj")
done
for name in autograd ops bindings; do
  obj="$BUILD/$name.o"
  echo "Compiling $name.cpp"
  timeout 180s g++ -std=c++17 -O3 -fPIC "${INCLUDES[@]}" \
    -c "tensor_cuda/src/$name.cpp" -o "$obj"
  objects+=("$obj")
done
timeout 90s "$NVCC" -arch=sm_89 -Xcompiler=-fPIC -dlink "${objects[@]}" -o "$BUILD/device_link.o"
timeout 90s g++ -shared "${objects[@]}" "$BUILD/device_link.o" -L"$CUDA/lib64" \
  -Wl,-rpath,"$CUDA/lib64" -lcudart -lcublas -lcublasLt -o "$BUILD/_tensor_cuda$EXT"
echo "BUILD_OK $BUILD/_tensor_cuda$EXT"
