#!/usr/bin/env bash
set -u
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix8; PR=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix8/artifacts/p1t_fix8/probe_runs; B=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_p1trev8
date > $PR/started.txt
bash $W/artifacts/p1t_fix8/run_rev8_probes.sh before $B $PR/rev8_before > $PR/rev8_before.summary 2>&1
bash $W/artifacts/p1t_fix8/run_rev8_probes.sh after $W $PR/rev8_after > $PR/rev8_after.summary 2>&1
bash $W/artifacts/p1t_fix8/runners/run_rev7_probes.sh after $W $PR/rev7_after > $PR/rev7_after.summary 2>&1
bash $W/artifacts/p1t_fix8/runners/run_rev6_probes.sh after $W $PR/rev6_after > $PR/rev6_after.summary 2>&1
for part in cpu buildcpu gpu rf; do bash $W/artifacts/p1t_fix8/runners/run_rev5_probes.sh after $W $PR/rev5_after $part > $PR/rev5_after_$part.summary 2>&1; done
bash $W/artifacts/p1t_fix8/runners/run_rev4_cpu_probes.sh after $W $PR/rev4_after > $PR/rev4_after.summary 2>&1
bash $W/artifacts/p1t_fix8/runners/run_rounds123_probes.sh $PR/rounds123_after > $PR/rounds123_after.summary 2>&1
date > $PR/all.done
