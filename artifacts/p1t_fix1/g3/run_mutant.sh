#!/usr/bin/env bash
# P1T-FIX1 G3: run one mutant's new tests (the mutant tree's byte-identical copy) and, for (a), the reviewer's
# addon_lifetime.py probe (read-only worktree file) on the mutant build. Usage: run_mutant.sh <a|b|c|d>
m=$1
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix1
R=/tmp/p1t_fix1_mut_$m
O=$W/artifacts/p1t_fix1/g3
case $m in
  a|b) PP=$W/tensor_cuda/tensor_cuda:$W/tensor_cuda:$R/tc_signal; T=tc_signal/tests/test_fix1_findings.py;;
  c|d|e) PP=$R/tensor_cuda/tensor_cuda:$R/tensor_cuda:$W/tc_signal; T=tensor_cuda/tests/test_dlpack.py;;
esac
cmp $W/$T $R/$T && echo "test file identical to worktree: $T"
cd $R
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 -c "import tensor_cuda as tc, tc_signal as ts; print('core', tc._C.__file__); print('tc_signal', ts._C.__file__)"
K=""; [[ $m == c || $m == d || $m == e ]] && K="-k fix1"
flock -w 900 /tmp/forge-gpu.lock timeout 900 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 -m pytest $R/$T $K -q -p no:cacheprovider -rfE > $O/mut_${m}_tests.log 2>&1
echo "tests exit=$?  $(tail -1 $O/mut_${m}_tests.log)"
grep -E "^(FAILED|ERROR)" $O/mut_${m}_tests.log | sed 's/ - .*//'
if [[ $m == a ]]; then
  for i in 1 2 3; do
    flock -w 900 /tmp/forge-gpu.lock timeout 60 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 $W/artifacts/p1t_rev/probes/addon_lifetime.py > $O/mut_a_addon_lifetime_run$i.log 2>&1
    echo "addon_lifetime run$i exit=$? incorrect_trials=$(grep -c '"correct": false' $O/mut_a_addon_lifetime_run$i.log)/12 reused=$(grep -c '"reused": true' $O/mut_a_addon_lifetime_run$i.log)/12 $(tail -1 $O/mut_a_addon_lifetime_run$i.log)"
  done
fi
if [[ $m == c ]]; then
  flock -w 900 /tmp/forge-gpu.lock timeout 60 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 $W/artifacts/p1t_rev/probes/core_edges.py > $O/mut_c_core_edges.log 2>&1
  echo "core_edges exit=$? $(tail -1 $O/mut_c_core_edges.log)"
fi
