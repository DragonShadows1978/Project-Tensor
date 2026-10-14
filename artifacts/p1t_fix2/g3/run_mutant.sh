#!/usr/bin/env bash
# P1T-FIX2 G3: build one mutant tree (made by make_mutants.py) and run its new tests (the tree's own copies, cmp-identical
# to the worktree's) against the mutant build, in fresh processes, under the GPU lock. Usage: run_mutant.sh <a|b|c|d1|d2|e>
set -u
m=$1
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix2
R=/tmp/p1t_fix2_mut_$m
O=$W/artifacts/p1t_fix2/g3
PYB=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src
case $m in
  a|c|d1) PP=$W/tensor_cuda/tensor_cuda:$W/tensor_cuda:$R/tc_signal; T=tc_signal/tests/test_fix2_findings.py; K=""
          (cd $R/tc_signal && PYBIND11_SOURCE_DIR=$PYB ./build.sh 89) > $O/mut_${m}_build.log 2>&1; echo "build exit=$?";;
  b|d2|e) PP=$R/tensor_cuda/tensor_cuda:$R/tensor_cuda:$W/tc_signal; T=tensor_cuda/tests/test_dlpack.py; K="-k fix2"
          (cd $R/tensor_cuda && export CUDACXX=/usr/local/cuda-12.6/bin/nvcc PATH=/usr/local/cuda-12.6/bin:$PATH && \
           cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DFETCHCONTENT_SOURCE_DIR_PYBIND11=$PYB && \
           cmake --build build -j) > $O/mut_${m}_build.log 2>&1; echo "build exit=$?";;
esac
cmp $W/$T $R/$T && echo "test file identical to worktree: $T"
[[ $m == a || $m == c || $m == d1 ]] && cmp $W/tc_signal/tests/fix2_native_harness.cpp $R/tc_signal/tests/fix2_native_harness.cpp && echo "native harness identical"
sha256sum $R/tensor_cuda/tensor_cuda/_tensor_cuda*.so $R/tc_signal/tc_signal/_tc_signal*.so 2>/dev/null | sed "s|$R/||" | tee $O/mut_${m}_so_sha256.txt
cd $R
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 -c "import tensor_cuda as tc, tc_signal as ts; print('core', tc._C.__file__); print('tc_signal', ts._C.__file__)"
flock -w 900 /tmp/forge-gpu.lock timeout 1500 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 -m pytest $R/$T $K -q -p no:cacheprovider -rfE > $O/mut_${m}_tests.log 2>&1
echo "tests exit=$?  $(tail -1 $O/mut_${m}_tests.log)"
grep -E "^(FAILED|ERROR)" $O/mut_${m}_tests.log | sed 's/ - .*//'
