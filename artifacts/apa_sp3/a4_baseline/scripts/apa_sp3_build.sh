#!/usr/bin/env bash
# Foreground host build only; no CUDA context and no network/git FetchContent.
set -euo pipefail
if [[ "${APA_SP3_BUILD_BOUNDED:-0}" != 1 ]]; then
  exec timeout --signal=TERM --kill-after=5s 570s env APA_SP3_BUILD_BOUNDED=1 bash "$0"
fi
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
[[ "$ROOT" == /mnt/ForgeRealm/Project-Tensor-wt-apa-sp3 ]] || exit 64
cd "$ROOT"
BUILD="$ROOT/artifacts/apa_sp3/build"
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
timeout 10s python3 - <<'PY'
from pathlib import Path
import hashlib,json
root=Path.cwd();build=root/'artifacts/apa_sp3/build'
files=list((root/'tensor_cuda/src').glob('*.cuh'))+list((root/'tensor_cuda/src').glob('*.cu'))+list((root/'tensor_cuda/src').glob('*.cpp'))+list((root/'tensor_cuda/include/tc').glob('*.h'))
result={'evidence_class':'host CUDA compile/link (not GPU execution)',
        'sources':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'modules':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in build.glob('_tensor_cuda*.so')}}
(build/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
PY
timeout 60s env PYTHONDONTWRITEBYTECODE=1 python3 scripts/apa_sp3_make_diag.py
# Separate diagnostic module: instrumented copies, never production source edits.
timeout 120s "$NVCC" -std=c++17 -O3 -arch=sm_89 -Xcompiler=-fPIC "${INCLUDES[@]}" \
  -c "$BUILD/diagnostics.cu" -o "$BUILD/diagnostics.o"
timeout 90s g++ -std=c++17 -O3 -fPIC "${INCLUDES[@]}" -c scripts/apa_sp3_diag_bindings.cpp -o "$BUILD/diag_bindings.o"
timeout 60s g++ -shared "$BUILD/diagnostics.o" "$BUILD/diag_bindings.o" "$BUILD/_tensor_cuda$EXT" \
  -L"$CUDA/lib64" -Wl,-rpath,"$CUDA/lib64" -Wl,-rpath,"$BUILD" -lcudart -o "$BUILD/_apa_sp3_diag$EXT"
timeout 30s g++ -std=c++17 -O2 -fPIC -shared -I"$CUDA/include" scripts/apa_sp3_peak.cpp -ldl -pthread -o "$BUILD/libapa_sp3_peak.so"
timeout 15s env PYTHONDONTWRITEBYTECODE=1 python3 scripts/apa_sp3_common.py seal-build
echo "Build complete in artifacts/apa_sp3/build"
