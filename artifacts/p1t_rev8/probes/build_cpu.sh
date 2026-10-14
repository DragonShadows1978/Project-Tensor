#!/usr/bin/env bash
# PASS iff unchanged sources compile/link; no GPU execution. Prior art: GCC/pybind11/UBSan (2013/2015).
set -euo pipefail
b=/tmp/p1t_rev8_build
pb=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src/include
cu=/usr/local/cuda-12.6
read -r -a pi <<< "$(python3-config --includes)"
read -r -a pl <<< "$(python3-config --embed --ldflags)"
timeout 180 g++ -std=c++17 -O1 -DNDEBUG -shared -fPIC artifacts/p1t_rev6/probes/release_trace.cpp -Itc_signal/src -Itc_signal/include -I"$pb" -I"$cu/include" "${pi[@]}" -pthread -o "$b/_p1t_rev6_trace$(python3-config --extension-suffix)"
for pair in 'tc_signal/tests/fix2_native_harness.cpp fix2_harness' 'tc_signal/tests/fix3_native_harness.cpp fix3_harness' 'artifacts/p1t_rev3/probes/cpu_native.cpp cpu_native'; do
 read -r src name <<< "$pair"
 timeout 180 g++ -std=c++17 -O1 -DNDEBUG -fsanitize=undefined -fno-sanitize-recover=undefined "$src" -Itc_signal/src -Itc_signal/include -I"$pb" -I"$cu/include" "${pi[@]}" "${pl[@]}" -L"$cu/lib64" -Wl,-rpath,"$cu/lib64" -lcudart -pthread -o "$b/$name"
done
timeout 180 g++ -std=c++17 -fsyntax-only tensor_cuda/src/bindings.cpp -Itensor_cuda/include -I"$pb" -I"$cu/include" "${pi[@]}"
