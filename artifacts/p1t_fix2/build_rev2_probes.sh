#!/usr/bin/env bash
# P1T-FIX2: build the round-2 reviewer's CPU probes with the reviewer's exact compile commands (logs/p1t_rev2_r1.log
# ~9107 cpu_native, ~9574 core_cpu r2), for one source tree, into /tmp/p1t_fix2_probes/<tag>/.
# Usage: build_rev2_probes.sh <tag> <tree_root> <core_cpu.cpp to compile>
#   cpu_native.cpp is compiled UNMODIFIED from the worktree's artifacts/p1t_rev2/probes/; its #include "dlpack_operand.cpp"
#   resolves through -I<tree_root>/tc_signal/src (the reviewer's -Itc_signal/src, made absolute).
#   core_cpu: for "before" the committed probe itself; for "after" the regenerated probe (regen_core_cpu_probe.py).
set -u
TAG=$1; ROOT=$2; CORE_SRC=$3
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix2
O=/tmp/p1t_fix2_probes/$TAG
mkdir -p "$O"
PYB=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src/include
LINK="$(python3-config --includes) $(python3-config --embed --ldflags) -L/usr/local/cuda-12.6/lib64 -Wl,-rpath,/usr/local/cuda-12.6/lib64 -lcudart"
timeout 180 bash -c "g++ -std=c++17 -O1 -g -fsanitize=undefined -fno-sanitize-recover=undefined $CORE_SRC -I$ROOT/tensor_cuda/include -I$PYB -I/usr/local/cuda-12.6/include $LINK -o $O/core_cpu" > "$O/core_cpu_build.log" 2>&1
echo "core_cpu build exit=$? src=$CORE_SRC sha256=$(sha256sum "$CORE_SRC" | cut -c1-16)"
timeout 180 bash -c "g++ -std=c++17 -O1 -g $W/artifacts/p1t_rev2/probes/cpu_native.cpp -I$ROOT/tc_signal/src -I$ROOT/tc_signal/include -I$PYB -I/usr/local/cuda-12.6/include $LINK -o $O/cpu_native" > "$O/cpu_native_build.log" 2>&1
echo "cpu_native build exit=$? tc_signal/src=$ROOT/tc_signal/src"
