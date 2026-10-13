#!/usr/bin/env bash
# Prior art: CPython embed/GCC compilation; standard flags, no new algorithm.
set -euo pipefail
read -r -a includes <<< "$(python3-config --includes)"
read -r -a links <<< "$(python3-config --embed --ldflags)"
for mode in spy import; do
 extra=()
 if [[ "$mode" == spy ]]; then extra+=(-DSPY); fi
 timeout 180 g++ -std=c++17 -O1 artifacts/p1t_rev4/probes/core_interop_cpu.cpp -Itensor_cuda/include -I/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src/include -I/usr/local/cuda-12.6/include "${includes[@]}" "${links[@]}" -L/usr/local/cuda-12.6/lib64 -Wl,-rpath,/usr/local/cuda-12.6/lib64 -lcudart "${extra[@]}" -o "/tmp/p1t_rev4_build/core_$mode"
 timeout 15 "/tmp/p1t_rev4_build/core_$mode"
done
