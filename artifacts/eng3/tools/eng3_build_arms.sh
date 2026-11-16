#!/bin/bash
# ENG3: sequential builds (one CPU-busy job at a time), each configure/build in its own systemd scope (12G, nice 19).
# E rebuilds in place (its module lands in the worktree package and is copied to build-eng3/E/module); each other arm
# builds from /tmp/eng3_src_<arm> into tensor_cuda/build-eng3/<arm>, its module copied to build-eng3/<arm>/module.
# Prior art: none beyond CMake out-of-tree builds.
set -u
cd "$(dirname "$0")/.."
R=artifacts/eng3/build/build_receipts.json
SCOPE="systemd-run --user --scope -q -p MemoryMax=12G -p OOMPolicy=continue -p CPUWeight=50 --setenv=TMPDIR=/tmp/eng3_tmp nice -n 19"
PYB=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src
SO=_tensor_cuda.cpython-312-x86_64-linux-gnu.so
for arm in "$@"; do
  avail=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  if [ "$avail" -lt 12582912 ]; then echo "STOP: MemAvailable $avail kB < 12 GB before $arm"; exit 5; fi
  if [ "$arm" = E ]; then
    python3 -B -P scripts/eng3_run.py $R build_E_final 3000 -- $SCOPE cmake --build tensor_cuda/build-eng3/E -j6 || exit 1
    mkdir -p tensor_cuda/build-eng3/E/module_final && cp tensor_cuda/tensor_cuda/$SO tensor_cuda/build-eng3/E/module_final/
    python3 -B -P scripts/eng3_shamap.py artifacts/eng3/build/BUILD_E.json tensor_cuda/build-eng3/E/module_final/$SO build_E
  else
    python3 -B -P scripts/eng3_run.py $R configure_$arm 600 -- $SCOPE cmake -S /tmp/eng3_src_$arm -B tensor_cuda/build-eng3/$arm -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DFETCHCONTENT_SOURCE_DIR_PYBIND11=$PYB -DFETCHCONTENT_FULLY_DISCONNECTED=ON || exit 1
    python3 -B -P scripts/eng3_run.py $R build_$arm 3000 -- $SCOPE cmake --build tensor_cuda/build-eng3/$arm -j6 || exit 1
    mkdir -p tensor_cuda/build-eng3/$arm/module && cp /tmp/eng3_src_$arm/tensor_cuda/$SO tensor_cuda/build-eng3/$arm/module/
    sha256sum tensor_cuda/build-eng3/$arm/module/$SO
  fi
done
echo ALL-BUILT
