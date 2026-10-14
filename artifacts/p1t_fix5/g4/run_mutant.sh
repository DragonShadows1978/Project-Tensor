#!/usr/bin/env bash
# P1T-FIX5 G4: run one mutant's new tests (and its round-5 probe) from the MUTANT tree, under the GPU lock.
#   (a) tc_signal test_fix5_findings -k v501 + exit_gate full / trace / gpu (trace built from the mutant's source)
#   (b) tc_signal test_fix5_findings -k v502 + query_error_single_gpu.py + query_error_cpu.py (mutant trace)
#   (c) tc_signal test_fix5_findings -k v503 (GPU shim test + the regenerated reviewer CPU probe)
#   (d) core test_dlpack -k fix5_v504 + fill_grid_single_gpu.py
# The mutant's test files are cmp-checked against the worktree's first. PASS of G4 = the new tests FAIL here.
# Usage: run_mutant.sh <m> <out_dir>. No prior art claimed (run plumbing).
set -u
M=$1; O=$2
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix5
T=/tmp/p1t_fix5_mut_$M
L="flock -w 900 /tmp/forge-gpu.lock timeout"
P=$W/artifacts/p1t_rev5/probes
mkdir -p "$O"
if [[ $M == d ]]; then
  PP=$T/tensor_cuda/tensor_cuda:$T/tensor_cuda:$W/tc_signal
  cmp $T/tensor_cuda/tests/test_dlpack.py $W/tensor_cuda/tests/test_dlpack.py && echo "mut $M test file identical to the worktree's"
else
  PP=$W/tensor_cuda/tensor_cuda:$W/tensor_cuda:$T/tc_signal
  for f in test_fix5_findings.py fix5_regen.py tcs_producer.py fix4_cudart_shim.c; do cmp $T/tc_signal/tests/$f $W/tc_signal/tests/$f || echo "DIFFERS $f"; done; echo "mut $M test files compared"
fi
sha256sum $T/tc_signal/tc_signal/_tc_signal*.so $T/tensor_cuda/tensor_cuda/_tensor_cuda*.so 2>/dev/null > "$O/mut_${M}_sha256.txt"
E=(env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP)
s() { echo "mut $M $1 exit=$2 :: $(grep -v '^\s*$' "$3" | tail -3 | tr '\n' ' ' | cut -c1-400)"; }
case $M in
  a)
    (cd $T && $L 900 "${E[@]}" python3 -m pytest tc_signal/tests/test_fix5_findings.py -q -p no:cacheprovider --basetemp=$T/pt -k v501 > "$O/mut_a_tests.log" 2>&1; s "tests -k v501" $? "$O/mut_a_tests.log")
    pyincs=$(python3-config --includes)
    g++ -std=c++17 -O1 -shared -fPIC $P/addon_trace.cpp -I$T/tc_signal/src -I$T/tc_signal/include -I/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src/include -I/usr/local/cuda-12.6/include $pyincs -o $T/_p1t_rev5_trace$(python3-config --extension-suffix) > "$O/mut_a_trace_build.log" 2>&1
    $L 30 "${E[@]}" CUDA_VISIBLE_DEVICES= python3 $P/exit_gate.py full > "$O/mut_a_exit_gate_full_nogpu.log" 2>&1; s "exit_gate full nogpu" $? "$O/mut_a_exit_gate_full_nogpu.log"
    $L 30 "${E[@]}" python3 $P/exit_gate.py full > "$O/mut_a_exit_gate_full_gpu.log" 2>&1; s "exit_gate full gpu" $? "$O/mut_a_exit_gate_full_gpu.log"
    $L 30 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$T python3 $P/exit_gate.py trace > "$O/mut_a_exit_gate_trace.log" 2>&1; s "exit_gate trace" $? "$O/mut_a_exit_gate_trace.log"
    $L 30 "${E[@]}" python3 $P/exit_gate.py gpu > "$O/mut_a_exit_gate_gpu.log" 2>&1; s "exit_gate gpu" $? "$O/mut_a_exit_gate_gpu.log";;
  b)
    (cd $T && $L 900 "${E[@]}" python3 -m pytest tc_signal/tests/test_fix5_findings.py -q -p no:cacheprovider --basetemp=$T/pt -k v502 > "$O/mut_b_tests.log" 2>&1; s "tests -k v502" $? "$O/mut_b_tests.log")
    pyincs=$(python3-config --includes)
    g++ -std=c++17 -O1 -shared -fPIC $P/addon_trace.cpp -I$T/tc_signal/src -I$T/tc_signal/include -I/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src/include -I/usr/local/cuda-12.6/include $pyincs -o $T/_p1t_rev5_trace$(python3-config --extension-suffix) > "$O/mut_b_trace_build.log" 2>&1
    $L 30 "${E[@]}" LD_PRELOAD=/tmp/p1t_rev5_query_once.so python3 $P/query_error_single_gpu.py > "$O/mut_b_query_error_single_gpu.log" 2>&1; s "query_error_single_gpu" $? "$O/mut_b_query_error_single_gpu.log"
    $L 15 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$T python3 $P/query_error_cpu.py > "$O/mut_b_query_error_cpu.log" 2>&1; s "query_error_cpu" $? "$O/mut_b_query_error_cpu.log";;
  c)
    (cd $T && $L 900 "${E[@]}" python3 -m pytest tc_signal/tests/test_fix5_findings.py -q -p no:cacheprovider --basetemp=$T/pt -k v503 > "$O/mut_c_tests.log" 2>&1; s "tests -k v503" $? "$O/mut_c_tests.log");;
  d)
    (cd $T && $L 900 "${E[@]}" python3 -m pytest tensor_cuda/tests/test_dlpack.py -q -p no:cacheprovider --basetemp=$T/pt -k fix5_v504 > "$O/mut_d_tests.log" 2>&1; s "tests -k fix5_v504" $? "$O/mut_d_tests.log")
    $L 30 "${E[@]}" LD_PRELOAD=/tmp/p1t_rev5_fill_grid.so python3 $P/fill_grid_single_gpu.py > "$O/mut_d_fill_grid_single_gpu.log" 2>&1; s "fill_grid_single_gpu" $? "$O/mut_d_fill_grid_single_gpu.log";;
esac
