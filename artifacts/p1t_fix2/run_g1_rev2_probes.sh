#!/usr/bin/env bash
# P1T-FIX2 G1: the round-2 reviewer's CPU probes, BEFORE (d116ac0: wt_fix1 sources and prebuilt .so) and AFTER (this
# worktree's final sources and .so), each in fresh processes under the GPU lock, in two conditions:
#   nogpu: CUDA_VISIBLE_DEVICES= (the reviewer's sandbox had no usable device: its empty_sync case needs that)
#   gpu:   the device visible (our host)
# plus the supplementary disarm variant of cpu_native oom_sync (make_disarm_variant.py) and gdb backtraces.
# Usage: run_g1_rev2_probes.sh <out_dir>
set -u
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix2
W1=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix1
O=${1:-$W/artifacts/p1t_fix2/probe_runs}
mkdir -p "$O"
L="flock -w 900 /tmp/forge-gpu.lock"
P=$W/artifacts/p1t_rev2/probes
sha256sum $P/* $W/artifacts/p1t_rev2/source_slices.json > "$O/rev2_probe_files_sha256.txt"
# AFTER probes from the final sources
python3 -I $W/artifacts/p1t_fix2/regen_core_cpu_probe.py $P/core_cpu.cpp $W1 $W /tmp/p1t_fix2_probes/after/core_cpu_regen_fix2.cpp
$W/artifacts/p1t_fix2/build_rev2_probes.sh after $W /tmp/p1t_fix2_probes/after/core_cpu_regen_fix2.cpp
$W/artifacts/p1t_fix2/build_rev2_probes.sh before $W1 $P/core_cpu.cpp
python3 -I $W/artifacts/p1t_fix2/make_host_checks_copy.py $P/host_checks.py $W1 $W1/tc_signal /tmp/p1t_fix2_probes/before /tmp/p1t_fix2_probes/before/host_checks_paths.py > "$O/host_checks_paths_before.diff"
python3 -I $W/artifacts/p1t_fix2/make_host_checks_copy.py $P/host_checks.py $W $W/tc_signal /tmp/p1t_fix2_probes/after /tmp/p1t_fix2_probes/after/host_checks_paths.py > "$O/host_checks_paths_after.diff"
cp /tmp/p1t_fix2_probes/after/core_cpu_regen_fix2.cpp "$O/core_cpu_regen_fix2.cpp"
diff -u $P/core_cpu.cpp /tmp/p1t_fix2_probes/after/core_cpu_regen_fix2.cpp > "$O/core_cpu_regen_fix2.diff"
sha256sum $W/tensor_cuda/tensor_cuda/_tensor_cuda*.so $W/tc_signal/tc_signal/_tc_signal*.so $W1/tensor_cuda/tensor_cuda/_tensor_cuda*.so $W1/tc_signal/tc_signal/_tc_signal*.so /tmp/p1t_fix2_probes/*/core_cpu /tmp/p1t_fix2_probes/*/cpu_native > "$O/rev2_inputs_sha256.txt"
for t in before after; do
  for c in nogpu gpu; do
    E="env PYTHONDONTWRITEBYTECODE=1"; [[ $c == nogpu ]] && E="$E CUDA_VISIBLE_DEVICES="
    (cd /tmp/p1t_fix2_probes/$t && $L timeout 300 $E python3 host_checks_paths.py > "$O/rev2_host_checks_${t}_${c}.log" 2>&1; echo "host_checks $t $c exit=$?"; cp host_checks.json "$O/rev2_host_checks_${t}_${c}.json")
  done
  (cd /tmp/p1t_fix2_probes/$t && $L timeout 120 gdb -batch -ex run -ex bt --args ./cpu_native oom_sync 2>&1 | grep -v "^\[New Thread\|^\[Thread\|libthread_db\|Using host\|^warning: 44" > "$O/rev2_oom_sync_gpu_backtrace_$t.txt")
  python3 -I $W/artifacts/p1t_fix2/make_disarm_variant.py $P/cpu_native.cpp /tmp/p1t_fix2_probes/$t/cpu_native_disarm.cpp > /dev/null
  ROOT=$([[ $t == before ]] && echo $W1 || echo $W)
  timeout 180 bash -c "g++ -std=c++17 -O1 -g /tmp/p1t_fix2_probes/$t/cpu_native_disarm.cpp -I$ROOT/tc_signal/src -I$ROOT/tc_signal/include -I/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src/include -I/usr/local/cuda-12.6/include $(python3-config --includes) $(python3-config --embed --ldflags) -L/usr/local/cuda-12.6/lib64 -Wl,-rpath,/usr/local/cuda-12.6/lib64 -lcudart -o /tmp/p1t_fix2_probes/$t/cpu_native_disarm" > "$O/cpu_native_disarm_build_$t.log" 2>&1
  for c in nogpu gpu; do
    E="env"; [[ $c == nogpu ]] && E="env CUDA_VISIBLE_DEVICES="
    $L timeout 60 $E /tmp/p1t_fix2_probes/$t/cpu_native_disarm oom_sync > "$O/rev2_oom_sync_disarm_${t}_${c}.log" 2>&1
    echo "oom_sync disarm-variant $t $c exit=$? :: $(tr '\n' ' ' < "$O/rev2_oom_sync_disarm_${t}_${c}.log" | cut -c1-160)"
  done
done
diff -u $W/tc_signal/tests/fix2_native_harness.cpp $W/tc_signal/tests/fix2_native_harness.cpp > /dev/null
echo "probe files unchanged vs commit:"; git -C $W diff --stat HEAD -- artifacts/p1t_rev2 artifacts/p1t_rev | tail -1; echo "(empty line above = unchanged)"
