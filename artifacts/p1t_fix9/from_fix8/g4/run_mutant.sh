#!/usr/bin/env bash
# P1T-FIX8 G4: build and run one mutant tree /tmp/p1t_fix8_mut_<m> made by make_mutants.py. Every run under
# `flock -w 900 /tmp/forge-gpu.lock timeout ...`. The mutant's test files are cmp-checked against this worktree's first
# (the tests that must fail are this round's, unmodified); its product sources must differ only by the mutation. The
# round-8 probes run from the mutant tree (byte-identical copies: they find the mutant's package through parents[3]);
# their trace / OOM host modules are built from the mutant's sources with the reviewer's command lines (and the
# path-only build_core_host_oom copy of make_rev8_copies.py). Mutant (g)'s core is built offline: build.sh's cmake
# commands with FETCHCONTENT_SOURCE_DIR_PYBIND11 = production's pybind11 source (read-only).
# Usage: run_mutant.sh <e|f|g|h> <out_dir> [tag] (tag: the make_mutants.py tag of the tree; L2's runs use _l2). Mutant
# (h) (lead send-back L2) has no round-8 probe: it is killed by the L2 tests (test_fix8_findings.py -k l2).
# No prior art claimed (run plumbing).
set -u
M=$1; O=$2; TAG=${3:-}
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix8
R=/tmp/p1t_fix8_mut_$M$TAG
B=/tmp/p1t_fix8_mut_bin_$M$TAG
P=$R/artifacts/p1t_rev8/probes
L="flock -w 900 /tmp/forge-gpu.lock timeout"
PP=$R/tensor_cuda/tensor_cuda:$R/tensor_cuda:$R/tc_signal
pb=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src/include
EXT=$(python3-config --extension-suffix)
mkdir -p "$O" "$B"
s() { local tag=$1 rc=$2 log=$3; echo "mut_$M $tag exit=$rc :: $(grep -v '^\s*$' "$log" | tr '\n' ' ' | cut -c1-260)"; }
for f in tc_signal/tests/test_fix8_findings.py tc_signal/tests/test_fix7_findings.py tensor_cuda/tests/test_dlpack.py; do
  cmp -s $W/$f $R/$f && echo "mut_$M $f identical to the worktree's: yes" || echo "mut_$M $f DIFFERS"
done
for d in tc_signal/src tc_signal/include tc_signal/tc_signal tensor_cuda/src tensor_cuda/include tensor_cuda/tensor_cuda; do
  diff -rq -x '__pycache__' -x '*.so' $W/$d $R/$d
done
if [[ $M == e || $M == f || $M == h ]]; then
  (cd $R/tc_signal && PYBIND11_SOURCE_DIR=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src ./build.sh 89) > "$O/build_$M.log" 2>&1
  echo "mut_$M build tc_signal exit=$?"
else
  cu="$(ls -d /usr/local/cuda-*/ 2>/dev/null | sort -V | tail -1)"; cu="${cu%/}"
  (cd $R/tensor_cuda && export PATH="$cu/bin:$PATH" CUDACXX="$cu/bin/nvcc" && \
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
     -DFETCHCONTENT_SOURCE_DIR_PYBIND11=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src \
     -DFETCHCONTENT_FULLY_DISCONNECTED=ON && cmake --build build -j) > "$O/build_$M.log" 2>&1
  echo "mut_$M build core exit=$?"
fi
sha256sum $R/tensor_cuda/tensor_cuda/_tensor_cuda*.so $R/tc_signal/tc_signal/_tc_signal*.so > "$O/binaries_$M.txt"
CPU="env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$B"
case $M in
  e)
    (cd $R && $L 1200 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 -m pytest tc_signal/tests/test_fix8_findings.py -q \
      -p no:cacheprovider -k "v801") > "$O/tests_$M.log" 2>&1; echo "mut_$M tests (-k v801) exit=$? :: $(tail -1 $O/tests_$M.log)"
    (cd $R && $L 300 g++ -std=c++17 -O1 -shared -fPIC artifacts/p1t_rev8/probes/diagnostic_errors_cpu.cpp -Itc_signal/src \
      -Itc_signal/include -I$pb -I/usr/local/cuda-12.6/include $(python3-config --includes) -pthread -o $B/_p1t_rev8_diag$EXT) \
      > "$O/build_diag.log" 2>&1; echo "mut_$M build_diag exit=$?"
    $L 60 $CPU python3 $P/diagnostic_errors_cpu.py > "$O/diagnostic_errors_cpu.log" 2>&1; s diagnostic_errors_cpu $? "$O/diagnostic_errors_cpu.log"
    $L 300 g++ -shared -fPIC -std=c++17 -I/usr/local/cuda-12.6/include $P/diagnostic_event_shim.cpp -ldl -o $B/p1t_rev8_event_shim.so \
      > "$O/build_shim.log" 2>&1
    $L 180 env LD_PRELOAD=$B/p1t_rev8_event_shim.so PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$R/tc_signal python3 $P/diagnostic_events_gpu.py \
      > "$O/diagnostic_events_gpu.log" 2>&1; s diagnostic_events_gpu $? "$O/diagnostic_events_gpu.log"
    ;;
  f)
    (cd $R && $L 1200 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 -m pytest tc_signal/tests/test_fix8_findings.py -q \
      -p no:cacheprovider -k "v802") > "$O/tests_$M.log" 2>&1; echo "mut_$M tests (-k v802) exit=$? :: $(tail -1 $O/tests_$M.log)"
    (cd $R && $L 300 g++ -std=c++17 -O1 -shared -fPIC -Wl,-Bsymbolic artifacts/p1t_rev8/probes/exit_log_oom_cpu.cpp \
      -Itc_signal/src -Itc_signal/include -I$pb -I/usr/local/cuda-12.6/include $(python3-config --includes) -pthread \
      -o $B/_p1t_rev8_oom$EXT) > "$O/build_oom.log" 2>&1; echo "mut_$M build_oom exit=$?"
    $L 60 $CPU python3 $P/verify_exit_log_cpu.py > "$O/verify_exit_log_cpu.log" 2>&1; s verify_exit_log_cpu $? "$O/verify_exit_log_cpu.log"
    ;;
  g)
    (cd $R && $L 1200 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 -m pytest tensor_cuda/tests/test_dlpack.py -q \
      -p no:cacheprovider -k "fix8_v802") > "$O/tests_$M.log" 2>&1; echo "mut_$M tests (-k fix8_v802) exit=$? :: $(tail -1 $O/tests_$M.log)"
    python3 -I $W/artifacts/p1t_fix8/make_rev8_copies.py "$R" "$B" "$O/copies" > "$O/copies_$M.txt"
    (cd $R && $L 900 python3 -I "$B/build_core_host_oom_paths.py") > "$O/build_core_host_oom.log" 2>&1; echo "mut_$M build_core_host_oom exit=$?"
    $L 60 $CPU python3 $P/verify_core_exit_log_cpu.py > "$O/verify_core_exit_log_cpu.log" 2>&1; s verify_core_exit_log_cpu $? "$O/verify_core_exit_log_cpu.log"
    ;;
  h)
    (cd $R && $L 1200 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 -m pytest tc_signal/tests/test_fix8_findings.py -q \
      -p no:cacheprovider -k "l2") > "$O/tests_$M.log" 2>&1; echo "mut_$M tests (-k l2) exit=$? :: $(tail -1 $O/tests_$M.log)"
    ;;
esac
echo "mut_$M done"
