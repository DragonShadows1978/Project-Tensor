#!/usr/bin/env bash
# P1T-FIX3 G2: round-1 (artifacts/p1t_rev/probes/, artifacts/p1t_rev/host_checks.py) and round-2
# (artifacts/p1t_rev2/probes/) reviewer probes on THIS worktree's final build, fresh processes, GPU lock, the probes'
# README timeouts. Probe files are read-only and unedited.
#   * round 1: as FIX2's run_g2_round1_probes.sh (addon_lifetime x3, core_edges, core_shutdown --normal and finalizing,
#     numerics_and_streams, host_checks, core_device_drain). release_worker.so is the existing /tmp/p1t_rev_native
#     build that core_shutdown.py hard-codes, used read-only (sha recorded).
#   * round 2: host_checks through FIX2's committed path-only copier (artifacts/p1t_fix2/make_host_checks_copy.py) and
#     core_cpu regenerated from this tree by FIX2's committed regen_core_cpu_probe.py (base d116ac0 = wt_fix1, read-only;
#     self-check printed), cpu_native compiled unmodified against this tree; into /tmp/p1t_fix3_probes/rev2_after/.
# Usage: run_rounds12_probes.sh <out_dir>
# No prior art claimed (run plumbing).
set -u
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix3
W1=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix1
O=$1
mkdir -p "$O"
PP=$W/tensor_cuda/tensor_cuda:$W/tensor_cuda:$W/tc_signal
E="env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP"
L="flock -w 900 /tmp/forge-gpu.lock"
cd "$W"
sha256sum tensor_cuda/tensor_cuda/_tensor_cuda*.so tc_signal/tc_signal/_tc_signal*.so artifacts/p1t_rev/probes/* artifacts/p1t_rev/host_checks.py artifacts/p1t_rev2/probes/* /tmp/p1t_rev_native/p1t_rev_release_worker.so > "$O/round12_inputs_sha256.txt"
for i in 1 2 3; do
  $L timeout 60 $E python3 artifacts/p1t_rev/probes/addon_lifetime.py > "$O/round1_addon_lifetime_run$i.log" 2>&1
  echo "round1 addon_lifetime run$i exit=$? incorrect_trials=$(grep -c '"correct": false' "$O/round1_addon_lifetime_run$i.log") reused_trials=$(grep -c '"reused": true' "$O/round1_addon_lifetime_run$i.log") correct_trials=$(grep -c '"correct": true' "$O/round1_addon_lifetime_run$i.log")"
done
$L timeout 60 $E python3 artifacts/p1t_rev/probes/core_edges.py > "$O/round1_core_edges.log" 2>&1; echo "round1 core_edges exit=$?"
$L timeout 30 $E python3 artifacts/p1t_rev/probes/core_shutdown.py --normal > "$O/round1_core_shutdown_normal.log" 2>&1; echo "round1 core_shutdown --normal exit=$? :: $(tr '\n' ' ' < "$O/round1_core_shutdown_normal.log" | cut -c1-160)"
$L timeout 30 $E python3 artifacts/p1t_rev/probes/core_shutdown.py > "$O/round1_core_shutdown_final.log" 2>&1; echo "round1 core_shutdown (finalizing) exit=$? :: $(tr '\n' ' ' < "$O/round1_core_shutdown_final.log" | cut -c1-160)"
$L timeout 90 $E python3 artifacts/p1t_rev/probes/numerics_and_streams.py > "$O/round1_numerics_and_streams.log" 2>&1; echo "round1 numerics_and_streams exit=$? :: $(tail -1 "$O/round1_numerics_and_streams.log")"
$L timeout 120 $E python3 artifacts/p1t_rev/host_checks.py > "$O/round1_host_checks.json" 2> "$O/round1_host_checks.stderr"; echo "round1 host_checks exit=$?"
$L timeout 60 $E python3 artifacts/p1t_rev/probes/core_device_drain.py > "$O/round1_core_device_drain.log" 2>&1; echo "round1 core_device_drain exit=$? (3 = INCONCLUSIVE: one GPU)"
# round 2 (CPU probes; both conditions)
R2=/tmp/p1t_fix3_probes/rev2_after
mkdir -p $R2
P2=$W/artifacts/p1t_rev2/probes
python3 -I $W/artifacts/p1t_fix2/regen_core_cpu_probe.py $P2/core_cpu.cpp $W1 $W $R2/core_cpu_regen_fix3.cpp > "$O/round2_core_cpu_regen.txt"
diff -u $P2/core_cpu.cpp $R2/core_cpu_regen_fix3.cpp > "$O/round2_core_cpu_regen_fix3.diff"
python3 -I $W/artifacts/p1t_fix2/make_host_checks_copy.py $P2/host_checks.py $W $W/tc_signal $R2 $R2/host_checks_paths.py > "$O/round2_host_checks_paths.diff"
PYB=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src/include
LINK="$(python3-config --includes) $(python3-config --embed --ldflags) -L/usr/local/cuda-12.6/lib64 -Wl,-rpath,/usr/local/cuda-12.6/lib64 -lcudart"
timeout 180 bash -c "g++ -std=c++17 -O1 -g -fsanitize=undefined -fno-sanitize-recover=undefined $R2/core_cpu_regen_fix3.cpp -I$W/tensor_cuda/include -I$PYB -I/usr/local/cuda-12.6/include $LINK -o $R2/core_cpu" > "$O/round2_core_cpu_build.log" 2>&1; echo "round2 core_cpu build exit=$?"
timeout 180 bash -c "g++ -std=c++17 -O1 -g $P2/cpu_native.cpp -I$W/tc_signal/src -I$W/tc_signal/include -I$PYB -I/usr/local/cuda-12.6/include $LINK -o $R2/cpu_native" > "$O/round2_cpu_native_build.log" 2>&1; echo "round2 cpu_native build exit=$?"
for c in nogpu gpu; do
  CE="env PYTHONDONTWRITEBYTECODE=1"; [[ $c == nogpu ]] && CE="$CE CUDA_VISIBLE_DEVICES="
  (cd $R2 && $L timeout 300 $CE python3 host_checks_paths.py > "$O/round2_host_checks_after_$c.log" 2>&1; echo "round2 host_checks $c collector exit=$?"; cp host_checks.json "$O/round2_host_checks_after_$c.json")
done
