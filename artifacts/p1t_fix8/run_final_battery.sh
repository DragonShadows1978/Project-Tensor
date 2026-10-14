#!/usr/bin/env bash
# P1T-FIX8: the end-of-round battery, once, on the final build (order "Budget"): the G4 mutants (e, f, g), Voice's
# runner, the combined session (the order's exact command) and the real-torch interop tests; each GPU step under the
# flock; the completion marker logs/p1t_fix8_final_battery.done. No prior art claimed (run plumbing).
set -u
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix8; T=$W; G=$W/artifacts/p1t_fix8/g4
for m in e f g; do bash $G/run_mutant.sh $m $G/run_$m > $G/summary_$m.txt 2>&1; echo "exit=$?" >> $G/summary_$m.txt; done
git -C /mnt/ForgeRealm/Project-Voice status --short | wc -l > $W/logs/p1t_fix8_voice_git_status_before.txt
sha256sum $W/artifacts/p1t_final/tensorcuda_adapter.py >> $W/logs/p1t_fix8_voice_git_status_before.txt
(cd /mnt/ForgeRealm/Project-Voice && flock -w 900 /tmp/forge-gpu.lock timeout 1200 env PYTHONDONTWRITEBYTECODE=1   PYTHONPATH=$T/tensor_cuda/tensor_cuda:$T/tensor_cuda:$T/tc_signal python3 engine/tensor_requests/p1t_tests/run_p1t.py   --adapter $T/artifacts/p1t_final/tensorcuda_adapter.py --items T1,T2,T3,T4,T5,T6,T1R,T4S   --fixtures /mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures --tolerances engine/tensor_requests/p1t_tests/tolerances_r2.json   --out $T/artifacts/p1t_fix8/p1t_all.json) > $W/logs/p1t_fix8_g3_voice.log 2>&1; echo "exit=$?" >> $W/logs/p1t_fix8_g3_voice.log
git -C /mnt/ForgeRealm/Project-Voice status --short | wc -l > $W/logs/p1t_fix8_voice_git_status_after.txt
sha256sum $T/tensor_cuda/tensor_cuda/_tensor_cuda*.so $T/tc_signal/tc_signal/_tc_signal*.so > $W/artifacts/p1t_fix8/build_sha256_at_combined.txt
sha256sum $T/tc_signal/tests/*.py $T/tensor_cuda/tests/test_dlpack.py > $W/artifacts/p1t_fix8/test_files_sha256_at_combined.txt
(cd $T && date && flock -w 900 /tmp/forge-gpu.lock timeout 3000 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$T/tensor_cuda/tensor_cuda:$T/tensor_cuda:$T/tc_signal python3 -m pytest tensor_cuda/tests tc_signal/tests -q -p no:cacheprovider --ignore=tensor_cuda/tests/test_selector_accuracy.py; echo "pytest_exit=$?"; date) > $W/logs/p1t_fix8_g3_combined_session.log 2>&1
K="pointer_identity_and_write_through or capsule_protocol_names or asks_producer_for_legacy_stream or unconsumed_capsule_releases or lifetime_after_producer_dropped or deleter_called_once_when_tensor_dies or stream_ordering_torch_side_stream or release_drains_engine_stream_before_deleter or real_torch_consumers"
(cd $T && flock -w 900 /tmp/forge-gpu.lock timeout 900 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$T/tensor_cuda/tensor_cuda:$T/tensor_cuda:$T/tc_signal python3 -m pytest tensor_cuda/tests/test_dlpack.py -q -p no:cacheprovider -rA -k "$K"; echo "exit=$?"; python3 -c "import torch;print('torch', torch.__version__, torch.version.cuda)") > $W/logs/p1t_fix8_g3_torch_interop.log 2>&1
date > $W/logs/p1t_fix8_final_battery.done
