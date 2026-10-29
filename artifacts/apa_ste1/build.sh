#!/usr/bin/env bash
# Prior art: CMake/Project-Tensor build (2026), unchanged toolchain; staging ours.
set -euo pipefail
cd /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_ste1
export CUDA_VISIBLE_DEVICES=
export TMPDIR="$PWD/artifacts/apa_ste1/scratch"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDACXX=/usr/local/cuda-12.6/bin/nvcc
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
timeout --foreground 120s cmake -S tensor_cuda -B artifacts/apa_ste1/scratch/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc -DCMAKE_CXX_COMPILER=/usr/bin/c++ -DFETCHCONTENT_SOURCE_DIR_PYBIND11="$PWD/artifacts/apa_ste1/scratch/vendor/pybind11" -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_BUILD_RPATH=/usr/local/cuda-12.6/targets/x86_64-linux/lib
timeout --foreground 900s cmake --build artifacts/apa_ste1/scratch/build -j4
