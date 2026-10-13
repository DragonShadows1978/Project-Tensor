#!/usr/bin/env bash
# P1T-FIX8 lead send-back L2: the round-8 G1 probes (and their controls), after arm only, on the L2 build, plus the
# legacy-stream scenario copy of diagnostic_events_gpu.py (D2). The same commands as run_rev8_probes.sh, limited to
# what G1 needs: the two V8 trace modules and the event shim built from this tree with the reviewer's command lines,
# the core OOM host module through the path-only copy of build_core_host_oom.py, a byte-identical copy of the built
# add-on package in <bin>/tc_signal. Every run under `flock -w 900 /tmp/forge-gpu.lock timeout <bound>`.
# Usage: run_rev8_g1_l2.sh <out_dir> (absolute). No prior art claimed (run plumbing).
set -u
O=$1
case $O in /*) ;; *) echo "out_dir must be absolute"; exit 2;; esac
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix8; ROOT=$W; ARM=after_l2
P=$ROOT/artifacts/p1t_rev8/probes
B=/tmp/p1t_fix8_rev8/after_l2
L="flock -w 900 /tmp/forge-gpu.lock timeout"
pb=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src/include
cu=/usr/local/cuda-12.6
EXT=$(python3-config --extension-suffix)
mkdir -p "$O" "$B"
s() { local tag=$1 rc=$2 log=$3; echo "$ARM $tag exit=$rc :: $(grep -v '^\s*$' "$log" | tr '\n' ' ' | cut -c1-300)"; }
python3 -I $W/artifacts/p1t_fix8/make_rev8_copies.py "$ROOT" "$B" "$O" > "$O/copies_$ARM.txt"
mkdir -p "$B/tc_signal"; cp -r "$ROOT/tc_signal/tc_signal" "$B/tc_signal/"
sha256sum $ROOT/tensor_cuda/tensor_cuda/_tensor_cuda*.so $ROOT/tc_signal/tc_signal/_tc_signal*.so "$B"/tc_signal/tc_signal/_tc_signal*.so > "$O/binaries_$ARM.txt"
(cd "$ROOT" && $L 900 python3 -I "$B/build_core_host_oom_paths.py") > "$O/build_core_host_oom_$ARM.log" 2>&1; echo "$ARM build_core_host_oom exit=$?"
(cd "$ROOT" && $L 300 g++ -std=c++17 -O1 -shared -fPIC artifacts/p1t_rev8/probes/diagnostic_errors_cpu.cpp -Itc_signal/src \
  -Itc_signal/include -I$pb -I$cu/include $(python3-config --includes) -pthread -o $B/_p1t_rev8_diag$EXT) \
  > "$O/build_diag_$ARM.log" 2>&1; echo "$ARM build_diag exit=$?"
(cd "$ROOT" && $L 300 g++ -std=c++17 -O1 -shared -fPIC -Wl,-Bsymbolic artifacts/p1t_rev8/probes/exit_log_oom_cpu.cpp \
  -Itc_signal/src -Itc_signal/include -I$pb -I$cu/include $(python3-config --includes) -pthread -o $B/_p1t_rev8_oom$EXT) \
  > "$O/build_oom_$ARM.log" 2>&1; echo "$ARM build_oom exit=$?"
$L 300 g++ -shared -fPIC -std=c++17 -I$cu/include $P/diagnostic_event_shim.cpp -ldl -o $B/p1t_rev8_event_shim.so \
  > "$O/build_shim_$ARM.log" 2>&1; echo "$ARM build_shim exit=$?"
sha256sum "$B"/_p1t_rev*.so "$B"/p1t_rev8_event_shim.so >> "$O/binaries_$ARM.txt" 2>/dev/null
CPU="env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$B:$B/tc_signal"
$L 60 $CPU python3 $P/diagnostic_errors_cpu.py > "$O/diagnostic_errors_cpu.log" 2>&1; s diagnostic_errors_cpu $? "$O/diagnostic_errors_cpu.log"
$L 60 $CPU python3 $P/verify_exit_log_cpu.py > "$O/verify_exit_log_cpu.log" 2>&1; s verify_exit_log_cpu $? "$O/verify_exit_log_cpu.log"
for arm in --control --oom; do
  $L 30 $CPU python3 $P/exit_log_oom_cpu.py $arm > "$O/exit_log_oom_cpu${arm}.log" 2>&1; s "exit_log_oom_cpu $arm" $? "$O/exit_log_oom_cpu${arm}.log"
done
$L 60 $CPU python3 $P/verify_core_exit_log_cpu.py > "$O/verify_core_exit_log_cpu.log" 2>&1; s verify_core_exit_log_cpu $? "$O/verify_core_exit_log_cpu.log"
for arm in --control --oom; do
  $L 30 $CPU python3 $P/core_exit_log_oom_cpu.py $arm > "$O/core_exit_log_oom_cpu${arm}.log" 2>&1; s "core_exit_log_oom_cpu $arm" $? "$O/core_exit_log_oom_cpu${arm}.log"
done
$L 180 env LD_PRELOAD=$B/p1t_rev8_event_shim.so PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$ROOT/tc_signal python3 $P/diagnostic_events_gpu.py \
  > "$O/diagnostic_events_gpu.log" 2>&1; s "diagnostic_events_gpu (as written)" $? "$O/diagnostic_events_gpu.log"
$L 180 env LD_PRELOAD=$B/p1t_rev8_event_shim.so PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$ROOT/tc_signal python3 \
  $W/artifacts/p1t_fix8/probe_copies/diagnostic_events_gpu_legacy.py > "$O/diagnostic_events_gpu_legacy.log" 2>&1
s "diagnostic_events_gpu_legacy (copy)" $? "$O/diagnostic_events_gpu_legacy.log"
echo "$ARM done"
