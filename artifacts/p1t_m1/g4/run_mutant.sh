#!/usr/bin/env bash
# P1T-M1 G4: run test_dlpack.py (the mutant tree's byte-identical copy) and Voice's G1 command on one mutant build.
m=$1
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_m1
R=/tmp/p1t_m1_mut_$m
PP=$R/tensor_cuda/tensor_cuda:$R/tensor_cuda
cmp $W/tensor_cuda/tests/test_dlpack.py $R/tensor_cuda/tests/test_dlpack.py && echo "test file identical to worktree"
cmp $W/artifacts/p1t_m1/tensorcuda_adapter.py $W/artifacts/p1t_m1/tensorcuda_adapter.py
cd $R
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 -c "import tensor_cuda as tc; print('module', tc.__file__, tc._C.__file__)"
sha256sum $R/tensor_cuda/tensor_cuda/_tensor_cuda*.so
flock -w 900 /tmp/forge-gpu.lock timeout 600 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 -m pytest $R/tensor_cuda/tests/test_dlpack.py -q -p no:cacheprovider -rfE > $R/test_dlpack.log 2>&1
echo "test_dlpack exit=$?"
tail -1 $R/test_dlpack.log
grep -E "^(FAILED|ERROR)" $R/test_dlpack.log | sed 's/ - .*//' 
cd /mnt/ForgeRealm/Project-Voice
flock -w 900 /tmp/forge-gpu.lock timeout 600 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 engine/tensor_requests/p1t_tests/run_p1t.py --adapter $W/artifacts/p1t_m1/tensorcuda_adapter.py --items T1 --fixtures /mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures --tolerances engine/tensor_requests/p1t_tests/tolerances_r2.json --out $R/p1t_T1_report.json > $R/g1.log 2>&1
echo "G1 exit=$?"
python3 - "$R/p1t_T1_report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
s = r['summary']
print('G1 verdict', s['verdict'], 'counts', s['counts'], 'failed', s['failed_cases'])
for c in r['cases']:
    if 'trials' in c:
        print('  ', c['id'], c['verdict'], f"{c['failed_trials']}/{c['trials']} failed trials")
    elif c['verdict'] != 'PASS':
        print('  ', c['id'], c['verdict'], c.get('adapter_exception', '')[:100], {k: c[k] for k in c if k.endswith('_ok') or k.startswith('pointer') or 'visible' in k})
PY
