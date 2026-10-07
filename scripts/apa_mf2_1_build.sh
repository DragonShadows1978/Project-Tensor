#!/usr/bin/env bash
# APA-MF2.1 engine build (copied from scripts/apa_mf2_build.sh; same nvcc/g++ flags).
# Differences: output dir and source root are parameters so the registered kernel
# mutants (M6/M7) build from a scratch COPY of tensor_cuda/{src,include} -- the tree's
# sources are never edited for a mutant. Foreground host build only; no GPU context.
#   APA_MF2_1_BUILD_DIR  (default artifacts/apa_mf2_1/build)
#   APA_MF2_1_SRC_ROOT   (default this worktree; must contain tensor_cuda/src and tensor_cuda/include)
set -euo pipefail
if [[ "${APA_MF2_1_BUILD_BOUNDED:-0}" != 1 ]]; then
  exec timeout --signal=TERM --kill-after=5s 590s env APA_MF2_1_BUILD_BOUNDED=1 bash "$0"
fi
ROOT=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2
SRCROOT="${APA_MF2_1_SRC_ROOT:-$ROOT}"
BUILD="${APA_MF2_1_BUILD_DIR:-$ROOT/artifacts/apa_mf2_1/build}"
cd "$SRCROOT"
mkdir -p "$BUILD"
NVCC=/usr/local/cuda-12.6/bin/nvcc
CUDA=/usr/local/cuda-12.6
PYINC=$(timeout 10s python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')
PYBIND=$(timeout 10s python3 -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("torch").origin).parent/"include")')
EXT=$(timeout 10s python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
INCLUDES=(-I"$SRCROOT/tensor_cuda/include" -I"$CUDA/include" -I"$PYINC" -I"$PYBIND")
objects=()
pids=()
for name in kernels paint_raster paint_bake paint_inpaint dda terrain terrain_objects matmul gemm_apa conv; do
  obj="$BUILD/$name.o"
  echo "Compiling $name.cu"
  timeout 450s "$NVCC" -std=c++17 -O3 -arch=sm_89 -Xcompiler=-fPIC \
    "${INCLUDES[@]}" -dc "tensor_cuda/src/$name.cu" -o "$obj" > "$BUILD/$name.log" 2>&1 &
  pids+=($!)
  objects+=("$obj")
done
for name in autograd ops bindings; do
  obj="$BUILD/$name.o"
  echo "Compiling $name.cpp"
  timeout 300s g++ -std=c++17 -O3 -fPIC "${INCLUDES[@]}" \
    -c "tensor_cuda/src/$name.cpp" -o "$obj" > "$BUILD/$name.log" 2>&1 &
  pids+=($!)
  objects+=("$obj")
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
if [[ $rc != 0 ]]; then echo "COMPILE FAILED"; tail -n 30 "$BUILD"/*.log; exit 1; fi
timeout 60s "$NVCC" -arch=sm_89 -Xcompiler=-fPIC -dlink "${objects[@]}" -o "$BUILD/device_link.o"
timeout 60s g++ -shared "${objects[@]}" "$BUILD/device_link.o" -L"$CUDA/lib64" \
  -Wl,-rpath,"$CUDA/lib64" -lcudart -lcublas -lcublasLt -o "$BUILD/_tensor_cuda$EXT"
APA_BUILD="$BUILD" timeout 10s python3 - <<'PY'
from pathlib import Path
import hashlib,json,os
root=Path.cwd();build=Path(os.environ['APA_BUILD'])
files=sorted(list((root/'tensor_cuda/src').glob('*.cuh'))+list((root/'tensor_cuda/src').glob('*.cu'))+list((root/'tensor_cuda/src').glob('*.cpp'))+list((root/'tensor_cuda/include/tc').glob('*.h')))
result={'evidence_class':'host CUDA compile/link (not GPU execution)','source_root':str(root),
        'sources':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'modules':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in build.glob('_tensor_cuda*.so')}}
(build/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['modules']))
PY
echo "Build complete in $BUILD"
