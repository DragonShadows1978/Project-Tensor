#!/usr/bin/env bash
# P1T-FIX1 G1: the reviewer's probes (read-only, unedited) on this worktree's rebuilt core + tc_signal, fresh processes,
# GPU lock, native driver. Usage: run_g1_probes.sh <out_dir>
set -u
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix1
O=${1:-$W/artifacts/p1t_fix1/probe_runs}
mkdir -p "$O"
PP=$W/tensor_cuda/tensor_cuda:$W/tensor_cuda:$W/tc_signal
E="env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP"
L="flock -w 900 /tmp/forge-gpu.lock"
cd "$W"
sha256sum tensor_cuda/tensor_cuda/_tensor_cuda*.so tc_signal/tc_signal/_tc_signal*.so artifacts/p1t_rev/probes/* artifacts/p1t_rev/host_checks.py > "$O/inputs_sha256.txt"
for i in 1 2 3; do
  $L timeout 60 $E python3 artifacts/p1t_rev/probes/addon_lifetime.py > "$O/addon_lifetime_run$i.log" 2>&1
  echo "addon_lifetime run$i exit=$? incorrect_trials=$(grep -c '"correct": false' "$O/addon_lifetime_run$i.log") reused_trials=$(grep -c '"reused": true' "$O/addon_lifetime_run$i.log")"
done
$L timeout 60 $E python3 artifacts/p1t_rev/probes/core_edges.py > "$O/core_edges.log" 2>&1; echo "core_edges exit=$?"
mkdir -p /tmp/p1t_rev_native
timeout 60 g++ -shared -fPIC -std=c++17 -pthread -I/usr/include/python3.12 -I/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src/include -Itensor_cuda/include artifacts/p1t_rev/probes/release_worker.cpp -o /tmp/p1t_rev_native/p1t_rev_release_worker.so > "$O/release_worker_build.log" 2>&1
echo "release_worker build exit=$? sha256=$(sha256sum /tmp/p1t_rev_native/p1t_rev_release_worker.so | cut -c1-16)"
$L timeout 30 $E python3 artifacts/p1t_rev/probes/core_shutdown.py --normal > "$O/core_shutdown_normal.log" 2>&1; echo "core_shutdown --normal exit=$?"
$L timeout 30 $E python3 artifacts/p1t_rev/probes/core_shutdown.py > "$O/core_shutdown_final.log" 2>&1; echo "core_shutdown (finalizing) exit=$?"
$L timeout 90 $E python3 artifacts/p1t_rev/probes/numerics_and_streams.py > "$O/numerics_and_streams.log" 2>&1; echo "numerics_and_streams exit=$?"
$L timeout 120 $E python3 artifacts/p1t_rev/host_checks.py > "$O/host_checks_after.json" 2> "$O/host_checks_after.stderr"; echo "host_checks exit=$?"
$L timeout 60 $E python3 artifacts/p1t_rev/probes/core_device_drain.py > "$O/core_device_drain.log" 2>&1; echo "core_device_drain exit=$? (3 = INCONCLUSIVE: one GPU)"
