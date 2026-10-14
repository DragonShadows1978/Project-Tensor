#!/usr/bin/env bash
# P1T-FIX4 G4: build one mutant tree (made by make_mutants.py) and run the new FIX4 tests (the tree's own copies,
# cmp-identical to the worktree's) plus the matching reviewer probe against the mutant build, in fresh processes, under
# the GPU lock. Usage: run_mutant.sh <a|b|c>
#   (a), (b) core mutants: tensor_cuda/tests/test_dlpack.py -k fix4; probe core_export_device.py (a) /
#            core_import_query_error.py (b, CUDA_VISIBLE_DEVICES="");
#   (c) tc_signal mutant: tc_signal/tests/test_fix4_findings.py; probe sync_existing_error.py.
# No prior art claimed (run plumbing).
set -u
m=$1
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix4
R=/tmp/p1t_fix4_mut_$m
O=$W/artifacts/p1t_fix4/g4
P=$W/artifacts/p1t_rev4/probes
PYB=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src
case $m in
  c) PP=$W/tensor_cuda/tensor_cuda:$W/tensor_cuda:$R/tc_signal; T=tc_signal/tests/test_fix4_findings.py; K=()
     (cd $R/tc_signal && PYBIND11_SOURCE_DIR=$PYB ./build.sh 89) > $O/mut_${m}_build.log 2>&1; echo "build exit=$?";;
  a|b) PP=$R/tensor_cuda/tensor_cuda:$R/tensor_cuda:$W/tc_signal; T=tensor_cuda/tests/test_dlpack.py; K=(-k fix4)
     (cd $R/tensor_cuda && export CUDACXX=/usr/local/cuda-12.6/bin/nvcc PATH=/usr/local/cuda-12.6/bin:$PATH && \
      cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DFETCHCONTENT_SOURCE_DIR_PYBIND11=$PYB && \
      cmake --build build -j) > $O/mut_${m}_build.log 2>&1; echo "build exit=$?";;
esac
cmp $W/$T $R/$T && echo "test file identical to worktree: $T"
[[ $m == c ]] && cmp $W/tc_signal/tests/fix4_cudart_shim.c $R/tc_signal/tests/fix4_cudart_shim.c && echo "shim identical"
sha256sum $R/tensor_cuda/tensor_cuda/_tensor_cuda*.so $R/tc_signal/tc_signal/_tc_signal*.so 2>/dev/null | sed "s|$R/||" | tee $O/mut_${m}_so_sha256.txt
cd $R
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 -c "import tensor_cuda as tc, tc_signal as ts; print('core', tc._C.__file__); print('tc_signal', ts._C.__file__)"
flock -w 900 /tmp/forge-gpu.lock timeout 1500 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 -m pytest $R/$T "${K[@]}" -q -p no:cacheprovider -rfE > $O/mut_${m}_tests.log 2>&1
echo "tests exit=$?  $(tail -1 $O/mut_${m}_tests.log)"
grep -E "^(FAILED|ERROR)" $O/mut_${m}_tests.log | sed 's/ - .*//'
case $m in
  a) flock -w 900 /tmp/forge-gpu.lock timeout 60 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 $P/core_export_device.py > $O/mut_${m}_probe.log 2>&1;;
  b) flock -w 900 /tmp/forge-gpu.lock timeout 30 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 $P/core_import_query_error.py > $O/mut_${m}_probe.log 2>&1;;
  c) flock -w 900 /tmp/forge-gpu.lock timeout 30 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 $P/sync_existing_error.py > $O/mut_${m}_probe.log 2>&1;;
esac
echo "probe exit=$?  $(tail -1 $O/mut_${m}_probe.log | cut -c1-200)"
