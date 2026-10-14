#!/usr/bin/env bash
# Prior art: GCC/CMake/pybind11 build workflows, UBSan (LLVM 2012+); no novel algorithm.
set -euo pipefail
review_root=$(pwd)
bin_dir=/tmp/p1t_rev5_build
pybind_include=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src/include
cuda_root=/usr/local/cuda-12.6
mkdir -p "$bin_dir"
python - <<'INIT'
from pathlib import Path
import shutil
shutil.copytree('tc_signal','/tmp/p1t_rev5_build/tc_signal',dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('build','*.so','__pycache__'))
for name in ('core_cpu.cpp','lifecycle.cpp','lifecycle_timeout.py','host_checks.py','operand_census.py','real_metadata_plans.py'):
    s=Path('artifacts/p1t_rev4/probes',name).read_text()
    s=s.replace('tensor_fork_voice_wt_p1trev4','tensor_fork_voice_wt_p1trev5').replace('/tmp/p1t_rev4_build','/tmp/p1t_rev5_build').replace('artifacts/p1t_rev4','artifacts/p1t_rev5')
    Path('/tmp/p1t_rev5_build',name).write_text(s)
INIT
CUDACXX="$cuda_root/bin/nvcc" timeout 180 cmake -S "$bin_dir/tc_signal" -B "$bin_dir/tc_signal/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DFETCHCONTENT_SOURCE_DIR_PYBIND11="${pybind_include%/include}"
timeout 240 cmake --build "$bin_dir/tc_signal/build" -j2
read -r -a pyincs <<< "$(python3-config --includes)"
read -r -a pylink <<< "$(python3-config --embed --ldflags)"
common=(-std=c++17 -O1 -g -I"$pybind_include" -I"$cuda_root/include" "${pyincs[@]}" "${pylink[@]}" -L"$cuda_root/lib64" -Wl,-rpath,"$cuda_root/lib64" -lcudart)
ubsan=(-fsanitize=undefined -fno-sanitize-recover=undefined)
timeout 180 g++ "$bin_dir/core_cpu.cpp" -Itensor_cuda/include "${common[@]}" "${ubsan[@]}" -o "$bin_dir/core_cpu"
timeout 180 g++ artifacts/p1t_rev3/probes/cpu_native.cpp -Itc_signal/src -Itc_signal/include "${common[@]}" -o "$bin_dir/cpu_native"
timeout 180 g++ tc_signal/tests/fix2_native_harness.cpp -Itc_signal/src -Itc_signal/include "${common[@]}" "${ubsan[@]}" -o "$bin_dir/fix2_harness"
timeout 180 g++ tc_signal/tests/fix3_native_harness.cpp -Itc_signal/src -Itc_signal/include "${common[@]}" "${ubsan[@]}" -o "$bin_dir/fix3_harness"
timeout 180 g++ -std=c++17 -shared -fPIC -O1 "$bin_dir/lifecycle.cpp" -Itensor_cuda/include -I"$pybind_include" "${pyincs[@]}" -pthread -o "$bin_dir/_p1t_rev4_lifecycle$(python3-config --extension-suffix)"
timeout 180 g++ -std=c++17 -fsyntax-only tensor_cuda/src/bindings.cpp -Itensor_cuda/include -I"$pybind_include" -I"$cuda_root/include" "${pyincs[@]}"
