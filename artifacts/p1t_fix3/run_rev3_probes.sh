#!/usr/bin/env bash
# P1T-FIX3: every probe in artifacts/p1t_rev3/probes/README.md on one arm, sequentially, each in a fresh process under
# `flock -w 900 /tmp/forge-gpu.lock timeout ...` (the README's timeouts), logs into <out_dir>.
#   before = 5f9f955: wt_fix2's prebuilt core + tc_signal (read-only) and CPU probes built from wt_fix2's sources;
#   after  = this worktree's build and CPU probes built from its sources (build_rev3_probes.sh).
# CPU probes run in two conditions: nogpu (CUDA_VISIBLE_DEVICES=, the reviewer's sandbox) and gpu (device visible).
# GPU ("LEAD GPU ONLY") probes run with the device visible. A summary line per run goes to stdout.
# Usage: run_rev3_probes.sh <arm> <root> <out_dir>
# No prior art claimed (run plumbing).
set -u
ARM=$1; ROOT=$2; O=$3
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix3
P=$W/artifacts/p1t_rev3/probes
B=/tmp/p1t_fix3_probes/$ARM
L="flock -w 900 /tmp/forge-gpu.lock timeout"
PP=$ROOT/tensor_cuda/tensor_cuda:$ROOT/tensor_cuda:$ROOT/tc_signal
mkdir -p "$O"
sha256sum $ROOT/tensor_cuda/tensor_cuda/_tensor_cuda*.so $ROOT/tc_signal/tc_signal/_tc_signal*.so > "$O/rev3_binaries_$ARM.txt"
s() { local tag=$1 rc=$2 log=$3; echo "$ARM $tag exit=$rc :: $(grep -v '^\s*$' "$log" | tr '\n' ' ' | cut -c1-230)"; }
cd /tmp
for c in nogpu gpu; do
  CV=(); [[ $c == nogpu ]] && CV=(CUDA_VISIBLE_DEVICES=)
  E=(env PYTHONDONTWRITEBYTECODE=1 "${CV[@]}")
  # host_checks (dispatches core_cpu normal/prefix/suffix and cpu_native oom_sync/throw_deleter); path-only copy
  (cd $B && $L 300 "${E[@]}" python3 host_checks_paths.py > "$O/rev3_host_checks_${ARM}_$c.log" 2>&1; echo "$ARM host_checks $c collector exit=$?"; cp host_checks.json "$O/rev3_host_checks_${ARM}_$c.json")
  $L 30 "${E[@]}" $B/native_harness ck_census > "$O/rev3_ck_census_${ARM}_$c.log" 2>&1; r=$?; echo "$ARM ck_census $c exit=$r :: $(tail -1 "$O/rev3_ck_census_${ARM}_$c.log" | cut -c1-120) ubsan_lines=$(grep -c 'runtime error' "$O/rev3_ck_census_${ARM}_$c.log")"
  for m in control short default; do
    $L 15 "${E[@]}" PYTHONPATH=$B python3 $P/lifecycle_timeout.py $m > "$O/rev3_lifecycle_${m}_${ARM}_$c.log" 2>&1; s "lifecycle_timeout $m $c" $? "$O/rev3_lifecycle_${m}_${ARM}_$c.log"
  done
  $L 15 "${E[@]}" $B/pending_bytes > "$O/rev3_pending_bytes_entry_${ARM}_$c.log" 2>&1; s "pending_bytes entry (unmodified) $c" $? "$O/rev3_pending_bytes_entry_${ARM}_$c.log"
  $L 15 "${E[@]}" $B/pending_bytes queue > "$O/rev3_pending_bytes_queue_${ARM}_$c.log" 2>&1; s "pending_bytes queue (unmodified) $c" $? "$O/rev3_pending_bytes_queue_${ARM}_$c.log"
  $L 15 "${E[@]}" $B/pending_bytes_adapt > "$O/rev3_pending_bytes_adapt_entry_${ARM}_$c.log" 2>&1; s "pending_bytes entry (adapted driver) $c" $? "$O/rev3_pending_bytes_adapt_entry_${ARM}_$c.log"
  $L 60 "${E[@]}" PYTHONPATH=$ROOT/tc_signal python3 $B/operand_census_paths.py > "$O/rev3_operand_census_${ARM}_$c.log" 2>&1; s "operand_census $c" $? "$O/rev3_operand_census_${ARM}_$c.log"
  $L 60 "${E[@]}" PYTHONPATH=$ROOT/tc_signal python3 $B/real_metadata_plans_paths.py > "$O/rev3_real_metadata_plans_${ARM}_$c.log" 2>&1; s "real_metadata_plans $c" $? "$O/rev3_real_metadata_plans_${ARM}_$c.log"
done
E=(env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP)
for m in during bounded; do
  T0=$(date +%s.%N)
  $L 20 "${E[@]}" python3 $P/core_exit_overrun.py $m > "$O/rev3_core_exit_overrun_${m}_$ARM.log" 2>&1; r=$?
  echo "$ARM core_exit_overrun $m exit=$r wall_s=$(python3 -c "import time; print(round(time.time() - $T0, 1))") :: $(grep -v '^RESULT' "$O/rev3_core_exit_overrun_${m}_$ARM.log" | tr '\n' ' ' | cut -c1-260)"
done
$L 60 "${E[@]}" python3 $P/addon_lifetime.py > "$O/rev3_addon_lifetime_$ARM.log" 2>&1; r=$?
echo "$ARM addon_lifetime exit=$r incorrect=$(grep -c '"correct": false' "$O/rev3_addon_lifetime_$ARM.log") correct=$(grep -c '"correct": true' "$O/rev3_addon_lifetime_$ARM.log") reused=$(grep -c '"reused": true' "$O/rev3_addon_lifetime_$ARM.log") :: $(tail -1 "$O/rev3_addon_lifetime_$ARM.log")"
$L 60 "${E[@]}" python3 $P/core_device_drain.py > "$O/rev3_core_device_drain_$ARM.log" 2>&1; s "core_device_drain (3 = INCONCLUSIVE, one GPU)" $? "$O/rev3_core_device_drain_$ARM.log"
$L 60 "${E[@]}" python3 $P/core_edges.py > "$O/rev3_core_edges_$ARM.log" 2>&1; s "core_edges" $? "$O/rev3_core_edges_$ARM.log"
$L 90 "${E[@]}" python3 $P/numerics_and_streams.py > "$O/rev3_numerics_and_streams_$ARM.log" 2>&1; s "numerics_and_streams" $? "$O/rev3_numerics_and_streams_$ARM.log"
$L 60 "${E[@]}" python3 $P/cuda_errors_multidevice.py > "$O/rev3_cuda_errors_multidevice_$ARM.log" 2>&1; s "cuda_errors_multidevice (3 = INCONCLUSIVE, one GPU)" $? "$O/rev3_cuda_errors_multidevice_$ARM.log"
RF=$P/release_failures.py; [[ $ARM == before ]] && RF=$B/release_failures_paths.py
$L 660 "${E[@]}" python3 $RF > "$O/rev3_release_failures_$ARM.log" 2>&1; r=$?
echo "$ARM release_failures exit=$r :: $(grep -E 'passed|failed|skipped|error' "$O/rev3_release_failures_$ARM.log" | tail -2 | tr '\n' ' ')"
