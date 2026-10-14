#!/usr/bin/env bash
# P1T-FIX7 G4: build (c only) and run one mutant tree /tmp/p1t_fix7_mut_<m> made by make_mutants.py. Every run under
# `flock -w 900 /tmp/forge-gpu.lock timeout ...`. The mutant's test files are cmp-checked against this worktree's first
# (the tests that must fail are this round's, unmodified); the round-7 probes run from the mutant tree (byte-identical
# copies: they find the mutant's package through parents[3]); the host modules the CPU probes import are built from the
# mutant's sources (path-only copies of the reviewer's build scripts, make_rev7_copies.py, root = the mutant tree).
# Usage: run_mutant.sh <a|b|c|d> <out_dir> [tree]. tree defaults to /tmp/p1t_fix7_mut_<m> (L1: the refreshed trees
# /tmp/p1t_fix7_mutL1_<m>; their probe bin dir is <tree>_bin). Mutant (d) (lead send-back L1) rebuilds the mutant's core
# offline: build.sh's cmake commands with FETCHCONTENT_SOURCE_DIR_PYBIND11 = production's pybind11 source (read-only)
# and FETCHCONTENT_FULLY_DISCONNECTED, then runs the core lock-retry tests. No prior art claimed (run plumbing).
set -u
M=$1; O=$2
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix7
R=${3:-/tmp/p1t_fix7_mut_$M}
if [[ $# -ge 3 ]]; then B=${R}_bin; else B=/tmp/p1t_fix7_mut_bin_$M; fi
P=$R/artifacts/p1t_rev7/probes
L="flock -w 900 /tmp/forge-gpu.lock timeout"
PP=$R/tensor_cuda/tensor_cuda:$R/tensor_cuda:$R/tc_signal
mkdir -p "$O" "$B/tc_signal"
s() { local tag=$1 rc=$2 log=$3; echo "mut_$M $tag exit=$rc :: $(grep -v '^\s*$' "$log" | tr '\n' ' ' | cut -c1-240)"; }
for f in tc_signal/tests/test_fix7_findings.py tc_signal/tests/test_fix6_findings.py tensor_cuda/tests/test_dlpack.py; do
  cmp -s $W/$f $R/$f && echo "mut_$M $f identical to the worktree's: yes" || echo "mut_$M $f DIFFERS"
done
diff -rq $W/tc_signal/src $R/tc_signal/src; diff -rq $W/tensor_cuda/src $R/tensor_cuda/src
diff -q $W/tc_signal/tc_signal/__init__.py $R/tc_signal/tc_signal/__init__.py
diff -q $W/tensor_cuda/tensor_cuda/__init__.py $R/tensor_cuda/tensor_cuda/__init__.py
if [[ $M == c ]]; then
  (cd $R/tc_signal && PYBIND11_SOURCE_DIR=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src ./build.sh 89) > "$O/build_$M.log" 2>&1
  echo "mut_$M build tc_signal exit=$?"
fi
if [[ $M == d ]]; then
  cu="$(ls -d /usr/local/cuda-*/ 2>/dev/null | sort -V | tail -1)"; cu="${cu%/}"
  (cd $R/tensor_cuda && export PATH="$cu/bin:$PATH" CUDACXX="$cu/bin/nvcc" && \
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
     -DFETCHCONTENT_SOURCE_DIR_PYBIND11=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src \
     -DFETCHCONTENT_FULLY_DISCONNECTED=ON && cmake --build build -j) > "$O/build_$M.log" 2>&1
  echo "mut_$M build core exit=$?"
fi
sha256sum $R/tensor_cuda/tensor_cuda/_tensor_cuda*.so $R/tc_signal/tc_signal/_tc_signal*.so > "$O/binaries_$M.txt"
python3 -I $W/artifacts/p1t_fix7/make_rev7_copies.py "$R" "$B" "$O/copies" > "$O/copies_$M.txt"
cp -r $R/tc_signal/tc_signal "$B/tc_signal/"
case $M in
  a)
    (cd $R && $L 900 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 -m pytest tc_signal/tests/test_fix7_findings.py -q \
      -p no:cacheprovider -k "v701") > "$O/tests_$M.log" 2>&1; echo "mut_$M tests (-k v701) exit=$? :: $(tail -1 $O/tests_$M.log)"
    for c in nogpu gpu; do
      if [[ $c == nogpu ]]; then cv="CUDA_VISIBLE_DEVICES="; else cv="P1T_FIX7_GPU=1"; fi
      $L 60 env $cv PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$R/tc_signal python3 $P/knob_exit_cpu.py > "$O/knob_exit_cpu_$c.log" 2>&1
      s "knob_exit_cpu $c" $? "$O/knob_exit_cpu_$c.log"
    done
    ;;
  b)
    (cd $R && $L 1200 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 -m pytest tensor_cuda/tests/test_dlpack.py -q \
      -p no:cacheprovider -k "fix7_v701 or fix7_core_knob") > "$O/tests_$M.log" 2>&1; echo "mut_$M tests (-k fix7_v701) exit=$? :: $(tail -1 $O/tests_$M.log)"
    (cd $R && $L 900 python3 -I "$B/build_core_host_paths.py") > "$O/build_core_host_$M.log" 2>&1; echo "mut_$M build_core_host exit=$?"
    $L 60 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$B python3 $P/core_knob_exit_cpu.py > "$O/core_knob_exit_cpu.log" 2>&1
    s "core_knob_exit_cpu" $? "$O/core_knob_exit_cpu.log"
    $L 120 env PYTHONDONTWRITEBYTECODE=1 python3 $P/core_knob_exit_gpu.py > "$O/core_knob_exit_gpu.log" 2>&1
    s "core_knob_exit_gpu" $? "$O/core_knob_exit_gpu.log"
    ;;
  d)
    (cd $R && $L 1200 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 -m pytest tensor_cuda/tests/test_dlpack.py -q \
      -p no:cacheprovider -rA -k "fix7_l1_core or sweep_i_core_exit_close_retries") > "$O/tests_$M.log" 2>&1
    echo "mut_$M tests (-k 'fix7_l1_core or sweep_i_core_exit_close_retries') exit=$? :: $(tail -1 $O/tests_$M.log)"
    grep -E '^(PASSED|FAILED)' "$O/tests_$M.log" | sed "s/^/mut_$M   /"
    ;;
  c)
    (cd $R && $L 1200 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 -m pytest tc_signal/tests/test_fix7_findings.py -q \
      -p no:cacheprovider -k "v702 or sweep_j") > "$O/tests_$M.log" 2>&1; echo "mut_$M tests (-k 'v702 or sweep_j') exit=$? :: $(tail -1 $O/tests_$M.log)"
    (cd $R && $L 900 bash "$B/build_cpu_paths.sh") > "$O/build_cpu_$M.log" 2>&1; echo "mut_$M build_cpu (trace module) exit=$?"
    $L 60 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$B python3 $P/query_diagnostic_cpu.py > "$O/query_diagnostic_cpu.log" 2>&1
    s "query_diagnostic_cpu" $? "$O/query_diagnostic_cpu.log"
    $L 120 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$R/tc_signal python3 $P/query_diagnostic_gpu.py > "$O/query_diagnostic_gpu.log" 2>&1
    s "query_diagnostic_gpu" $? "$O/query_diagnostic_gpu.log"
    ;;
esac
echo "mut_$M done"
