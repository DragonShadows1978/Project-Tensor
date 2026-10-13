#!/usr/bin/env bash
# Lead runs of the round-9 RED CPU probes and the lead-only GPU probes on the reviewed build (e4cf1d6).
set -u; T=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev9; P=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev9/artifacts/p1t_rev9/probes; LD=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev9/artifacts/p1t_rev9/lead; LK="flock -w 900 /tmp/forge-gpu.lock timeout"
cd $T
for s in build_core_fence_cpu.py core_fence_cpu.py build_core_race_cpu.py core_snapshot_cpu.py; do $LK 600 env PYTHONDONTWRITEBYTECODE=1 python3 $P/$s > $LD/${s%.py}.log 2>&1; echo "$s rc=$?"; done
g++ -shared -fPIC -std=c++17 -I/usr/local/cuda-12.6/include $P/diagnostic_event_shim.cpp -ldl -o /tmp/p1t_rev9_lead_event_shim.so; echo "shim rc=$?"
(cd /tmp && $LK 150 env LD_PRELOAD=/tmp/p1t_rev9_lead_event_shim.so PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$T/tc_signal python3 $P/diagnostic_events_gpu_legacy.py > $LD/diagnostic_events_gpu_legacy.log 2>&1; echo "diagnostic_events_gpu_legacy rc=$?")
for lane in fix8 fix7 fix56 semantics multidevice; do $LK 1000 env PYTHONDONTWRITEBYTECODE=1 python3 $P/gpu_regressions.py $lane > $LD/gpu_regressions_$lane.log 2>&1; echo "gpu_regressions $lane rc=$? :: $(tail -1 $LD/gpu_regressions_$lane.log | cut -c1-120)"; done
