#!/usr/bin/env bash
# P1T-FIX8: every round-8 probe (artifacts/p1t_rev8/probes/) on one arm, each in a fresh process, every run under
# `flock -w 900 /tmp/forge-gpu.lock timeout <bound>` (GPU or not). One summary line per run; logs in <out_dir>
# (ABSOLUTE path required: the builds cd into the arm's root).
#   before = /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev8 (read-only;
#            6991939's sources, tests and binaries, plus the round-8 probes);
#   after  = this worktree and its build.
# Builds go to /tmp/p1t_fix8_rev8/<arm>/ (never /tmp/p1t_rev8_build, the reviewer's retained scratch). Unmodified probe
# sources run from the arm's tree, except the path-only copies of make_rev8_copies.py (build dir, arm root and receipt
# paths only; diffs in <out_dir>). <bin>/tc_signal is a byte-identical copy of the arm's built add-on package. The two
# round-8 trace modules (diagnostic_errors_cpu.cpp, exit_log_oom_cpu.cpp) and the GPU event shim are compiled from the
# arm's own sources with the reviewer's command lines (logs/p1t_rev8_r2.log; the probes' headers).
# Usage: run_rev8_probes.sh <arm> <arm_root> <out_dir>. No prior art claimed (run plumbing).
set -u
ARM=$1; ROOT=$2; O=$3
case $O in /*) ;; *) echo "out_dir must be absolute"; exit 2;; esac
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix8
P=$ROOT/artifacts/p1t_rev8/probes
B=/tmp/p1t_fix8_rev8/$ARM
L="flock -w 900 /tmp/forge-gpu.lock timeout"
pb=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src/include
cu=/usr/local/cuda-12.6
EXT=$(python3-config --extension-suffix)
mkdir -p "$O" "$B"
s() { local tag=$1 rc=$2 log=$3; echo "$ARM $tag exit=$rc :: $(grep -v '^\s*$' "$log" | tr '\n' ' ' | cut -c1-300)"; }
sha256sum $ROOT/tensor_cuda/tensor_cuda/_tensor_cuda*.so $ROOT/tc_signal/tc_signal/_tc_signal*.so > "$O/binaries_$ARM.txt"
python3 -I $W/artifacts/p1t_fix8/make_rev8_copies.py "$ROOT" "$B" "$O" > "$O/copies_$ARM.txt"
mkdir -p "$B/tc_signal"; cp -r "$ROOT/tc_signal/tc_signal" "$B/tc_signal/"
sha256sum "$B"/tc_signal/tc_signal/_tc_signal*.so >> "$O/binaries_$ARM.txt"
# ---- builds (path-only copies of build_cpu.sh / build_core_host.py / build_core_host_oom.py; the reviewer's command
# lines for the two V8 trace modules and the shim), from the arm's sources
(cd "$ROOT" && $L 900 bash "$B/build_cpu_paths.sh") > "$O/build_cpu_$ARM.log" 2>&1; echo "$ARM build_cpu exit=$?"
(cd "$ROOT" && $L 900 python3 -I "$B/build_core_host_paths.py") > "$O/build_core_host_$ARM.log" 2>&1; echo "$ARM build_core_host exit=$?"
(cd "$ROOT" && $L 900 python3 -I "$B/build_core_host_oom_paths.py") > "$O/build_core_host_oom_$ARM.log" 2>&1; echo "$ARM build_core_host_oom exit=$?"
(cd "$ROOT" && $L 300 g++ -std=c++17 -O1 -shared -fPIC artifacts/p1t_rev8/probes/diagnostic_errors_cpu.cpp -Itc_signal/src \
  -Itc_signal/include -I$pb -I$cu/include $(python3-config --includes) -pthread -o $B/_p1t_rev8_diag$EXT) \
  > "$O/build_diag_$ARM.log" 2>&1; echo "$ARM build_diag exit=$?"
(cd "$ROOT" && $L 300 g++ -std=c++17 -O1 -shared -fPIC -Wl,-Bsymbolic artifacts/p1t_rev8/probes/exit_log_oom_cpu.cpp \
  -Itc_signal/src -Itc_signal/include -I$pb -I$cu/include $(python3-config --includes) -pthread -o $B/_p1t_rev8_oom$EXT) \
  > "$O/build_oom_$ARM.log" 2>&1; echo "$ARM build_oom exit=$?"
$L 300 g++ -shared -fPIC -std=c++17 -I$cu/include $P/diagnostic_event_shim.cpp -ldl -o $B/p1t_rev8_event_shim.so \
  > "$O/build_shim_$ARM.log" 2>&1; echo "$ARM build_shim exit=$?"
sha256sum "$B"/_p1t_rev*.so "$B"/fix2_harness "$B"/fix3_harness "$B"/cpu_native "$B"/p1t_rev8_event_shim.so >> "$O/binaries_$ARM.txt" 2>/dev/null
CPU="env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$B:$B/tc_signal"
# ---- V8-01 (CPU trace) and V8-02 (add-on and core OOM arms + controls): the G1 probes
$L 60 $CPU python3 $P/diagnostic_errors_cpu.py > "$O/diagnostic_errors_cpu.log" 2>&1; s diagnostic_errors_cpu $? "$O/diagnostic_errors_cpu.log"
$L 60 $CPU python3 $P/verify_exit_log_cpu.py > "$O/verify_exit_log_cpu.log" 2>&1; s verify_exit_log_cpu $? "$O/verify_exit_log_cpu.log"
for arm in --control --oom; do
  $L 30 $CPU python3 $P/exit_log_oom_cpu.py $arm > "$O/exit_log_oom_cpu${arm}.log" 2>&1; s "exit_log_oom_cpu $arm" $? "$O/exit_log_oom_cpu${arm}.log"
done
$L 60 $CPU python3 $P/verify_core_exit_log_cpu.py > "$O/verify_core_exit_log_cpu.log" 2>&1; s verify_core_exit_log_cpu $? "$O/verify_core_exit_log_cpu.log"
for arm in --control --oom; do
  $L 30 $CPU python3 $P/core_exit_log_oom_cpu.py $arm > "$O/core_exit_log_oom_cpu${arm}.log" 2>&1; s "core_exit_log_oom_cpu $arm" $? "$O/core_exit_log_oom_cpu${arm}.log"
done
# ---- V8-01 on the GPU (real built add-on, LD_PRELOAD shim on the event calls)
$L 180 env LD_PRELOAD=$B/p1t_rev8_event_shim.so PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$ROOT/tc_signal python3 $P/diagnostic_events_gpu.py \
  > "$O/diagnostic_events_gpu.log" 2>&1; s diagnostic_events_gpu $? "$O/diagnostic_events_gpu.log"
# ---- inherited CPU receipts (path-only copies where they write receipts)
$L 900 python3 -I "$B/run_cpu_checks_paths.py" > "$O/run_cpu_checks_$ARM.log" 2>&1; echo "$ARM run_cpu_checks exit=$?"
sed 's/^/  /' "$O/run_cpu_checks_$ARM.log" | cut -c1-200 | head -20
$L 300 $CPU python3 "$B/argument_census_cpu_paths.py" > "$O/argument_census_cpu.log" 2>&1; s argument_census_cpu $? "$O/argument_census_cpu.log"
$L 300 $CPU python3 "$B/operand_census_cpu_paths.py" > "$O/operand_census_cpu.log" 2>&1; s operand_census_cpu $? "$O/operand_census_cpu.log"
$L 600 $CPU python3 "$B/attribution_cpu_paths.py" > "$O/attribution_cpu.log" 2>&1; s attribution_cpu $? "$O/attribution_cpu.log"
$L 900 $CPU python3 "$B/fix7_cpu_census_paths.py" > "$O/fix7_cpu_census.log" 2>&1; s fix7_cpu_census $? "$O/fix7_cpu_census.log"
$L 300 $CPU python3 "$B/lock_checks_cpu_paths.py" > "$O/lock_checks_cpu.log" 2>&1; s lock_checks_cpu $? "$O/lock_checks_cpu.log"
for ctl in "" "--control"; do
  t=knob_exit_cpu${ctl:+_control}
  $L 60 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$ROOT/tc_signal python3 $P/knob_exit_cpu.py $ctl > "$O/$t.log" 2>&1; s "$t" $? "$O/$t.log"
  t=core_knob_exit_cpu${ctl:+_control}
  $L 60 $CPU python3 $P/core_knob_exit_cpu.py $ctl > "$O/$t.log" 2>&1; s "$t" $? "$O/$t.log"
done
$L 60 $CPU python3 $P/query_diagnostic_cpu.py > "$O/query_diagnostic_cpu.log" 2>&1; s query_diagnostic_cpu $? "$O/query_diagnostic_cpu.log"
# ---- gpu_regressions.py fix7 / fix56 / semantics / multidevice (path-only copy: junit receipt -> out_dir)
for lane in fix7 fix56 semantics multidevice; do
  (cd "$ROOT" && $L 1500 env PYTHONDONTWRITEBYTECODE=1 python3 "$B/gpu_regressions_paths.py" $lane) > "$O/gpu_regressions_$lane.log" 2>&1
  rc=$?; echo "$ARM gpu_regressions $lane exit=$rc :: $(tail -2 "$O/gpu_regressions_$lane.log" | tr '\n' ' ' | cut -c1-300)"
done
echo "$ARM done"
