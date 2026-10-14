#!/usr/bin/env bash
# P1T-FIX2 G2: every round-1 reviewer probe (artifacts/p1t_rev/probes/, read-only, unedited) on this worktree's final
# core + tc_signal, fresh processes, GPU lock, timeouts per the probes' README. release_worker.so is the existing
# /tmp/p1t_rev_native build (README command; core_shutdown.py hard-codes that path), used read-only, sha recorded.
# Usage: run_g2_round1_probes.sh <out_dir>
set -u
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix2
O=${1:-$W/artifacts/p1t_fix2/probe_runs}
mkdir -p "$O"
PP=$W/tensor_cuda/tensor_cuda:$W/tensor_cuda:$W/tc_signal
E="env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP"
L="flock -w 900 /tmp/forge-gpu.lock"
cd "$W"
sha256sum tensor_cuda/tensor_cuda/_tensor_cuda*.so tc_signal/tc_signal/_tc_signal*.so artifacts/p1t_rev/probes/* artifacts/p1t_rev/host_checks.py /tmp/p1t_rev_native/p1t_rev_release_worker.so > "$O/round1_inputs_sha256.txt"
for i in 1 2 3; do
  $L timeout 60 $E python3 artifacts/p1t_rev/probes/addon_lifetime.py > "$O/round1_addon_lifetime_run$i.log" 2>&1
  echo "addon_lifetime run$i exit=$? incorrect_trials=$(grep -c '"correct": false' "$O/round1_addon_lifetime_run$i.log") reused_trials=$(grep -c '"reused": true' "$O/round1_addon_lifetime_run$i.log") correct_trials=$(grep -c '"correct": true' "$O/round1_addon_lifetime_run$i.log")"
done
$L timeout 60 $E python3 artifacts/p1t_rev/probes/core_edges.py > "$O/round1_core_edges.log" 2>&1; echo "core_edges exit=$?"
$L timeout 30 $E python3 artifacts/p1t_rev/probes/core_shutdown.py --normal > "$O/round1_core_shutdown_normal.log" 2>&1; echo "core_shutdown --normal exit=$?"
$L timeout 30 $E python3 artifacts/p1t_rev/probes/core_shutdown.py > "$O/round1_core_shutdown_final.log" 2>&1; echo "core_shutdown (finalizing) exit=$?"
$L timeout 90 $E python3 artifacts/p1t_rev/probes/numerics_and_streams.py > "$O/round1_numerics_and_streams.log" 2>&1; echo "numerics_and_streams exit=$?"
$L timeout 120 $E python3 artifacts/p1t_rev/host_checks.py > "$O/round1_host_checks.json" 2> "$O/round1_host_checks.stderr"; echo "round-1 host_checks exit=$?"
$L timeout 60 $E python3 artifacts/p1t_rev/probes/core_device_drain.py > "$O/round1_core_device_drain.log" 2>&1; echo "core_device_drain exit=$? (3 = INCONCLUSIVE: one GPU)"
