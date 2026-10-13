#!/usr/bin/env bash
# P1T-FIX7 lead verification chain (lead, 2026-10-13). Final build: core 75011241..., tc_signal ca5770c1...
# Absolute output dirs; every GPU step under flock (inside the seat runners, or explicitly here).
set -u
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix7; L=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix7/artifacts/p1t_fix7/lead; PP=$W/tensor_cuda/tensor_cuda:$W/tensor_cuda:$W/tc_signal
LK="flock -w 900 /tmp/forge-gpu.lock timeout"
step() { echo "=== $(date +%H:%M:%S) $*"; }
step sha; sha256sum $W/tensor_cuda/tensor_cuda/_tensor_cuda*.so $W/tc_signal/tc_signal/_tc_signal*.so
step rev7_after; bash $W/artifacts/p1t_fix7/run_rev7_probes.sh lead_after $W $L/rev7_after > $L/rev7_after_summary.txt 2>&1; echo rc=$?
step rev6_after; bash $W/artifacts/p1t_fix7/runners/run_rev6_probes.sh lead_after $W $L/rev6_after > $L/rev6_after_summary.txt 2>&1; echo rc=$?
for part in cpu buildcpu gpu rf; do step rev5_after $part; bash $W/artifacts/p1t_fix7/runners/run_rev5_probes.sh lead_after $W $L/rev5_after $part >> $L/rev5_after_summary.txt 2>&1; echo rc=$?; done
step rounds123; bash $W/artifacts/p1t_fix7/runners/run_rounds123_probes.sh $L/rounds123_after > $L/rounds123_after_summary.txt 2>&1; echo rc=$?
for m in a b c d; do step mutant $m; bash $W/artifacts/p1t_fix7/g4/run_mutant.sh $m $L/mut_$m /tmp/p1t_fix7_mutL1_$m > $L/mut_${m}_summary.txt 2>&1; echo rc=$?; done
step voice; (cd /mnt/ForgeRealm/Project-Voice && git status --short > $L/voice_git_status_before.txt; $LK 1200 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 engine/tensor_requests/p1t_tests/run_p1t.py --adapter $W/artifacts/p1t_final/tensorcuda_adapter.py --items T1,T2,T3,T4,T5,T6,T1R,T4S --fixtures /mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures --tolerances engine/tensor_requests/p1t_tests/tolerances_r2.json --out $L/p1t_all_lead.json > $L/voice_runner.log 2>&1; echo rc=$?; git status --short > $L/voice_git_status_after.txt); sha256sum $W/artifacts/p1t_final/tensorcuda_adapter.py
step combined; (cd $W && $LK 3000 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 -m pytest tensor_cuda/tests tc_signal/tests -q -p no:cacheprovider --ignore=tensor_cuda/tests/test_selector_accuracy.py > $L/combined_session.log 2>&1; echo rc=$?)
python3 /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/scripts/apa_mf2_2_g1_compare.py /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2/logs/apa_mf1_g1_baseline_unchanged.log $L/combined_session.log > $L/combined_compare.txt 2>&1; echo compare_rc=$?
step done
