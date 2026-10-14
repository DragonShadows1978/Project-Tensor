#!/usr/bin/env bash
# P1T-FIX6 G4: run one mutant's new tests and its finding's round-6 probes. The mutant tree /tmp/p1t_fix6_mut_<m>/ was
# made by make_mutants.py (one edit) and rebuilt; its test files must be byte-identical to this worktree's (checked).
# Probe modules that embed product source (the release trace module, the core host slices) are built from the MUTANT's
# sources. Every GPU-capable run is under `flock -w 900 /tmp/forge-gpu.lock timeout ...`.
# Usage: run_mutant.sh <a|b|c> <out_dir>. No prior art claimed (run plumbing).
set -u
M=$1; O=$2
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix6
R=/tmp/p1t_fix6_mut_$M
P=$W/artifacts/p1t_rev6/probes
B=$R/probe_build
L="flock -w 900 /tmp/forge-gpu.lock timeout"
PP=$R/tensor_cuda/tensor_cuda:$R/tensor_cuda:$R/tc_signal
pb=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src/include
cu=/usr/local/cuda-12.6
read -r -a pyincs <<< "$(python3-config --includes)"
mkdir -p "$O" "$B"
s() { local tag=$1 rc=$2 log=$3; echo "mut_$M $tag exit=$rc :: $(grep -v '^\s*$' "$log" | tr '\n' ' ' | cut -c1-220)"; }
for f in tc_signal/tests/test_fix6_findings.py tc_signal/tests/test_fix5_findings.py tc_signal/tests/test_fix2_findings.py \
         tensor_cuda/tests/test_dlpack.py; do
  cmp -s "$W/$f" "$R/$f" && echo "mut_$M tests identical: $f" || echo "mut_$M TESTS DIFFER: $f"
done
cd /tmp
if [[ $M == a ]]; then
  $L 900 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 -m pytest $R/tc_signal/tests/test_fix6_findings.py \
    $R/tc_signal/tests/test_fix5_findings.py -q -p no:cacheprovider -k "v601 or exit_hook_still_waits or synchronize_does_not_wait" \
    > "$O/mut_a_tests.log" 2>&1; echo "mut_a tests exit=$? :: $(tail -1 "$O/mut_a_tests.log")"
  for c in nogpu gpu; do
    if [[ $c == nogpu ]]; then cv="CUDA_VISIBLE_DEVICES="; else cv="P1T_FIX6_GPU=1"; fi
    $L 15 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP $cv python3 $P/sync_lock_inversion_cpu.py > "$O/mut_a_sync_lock_$c.log" 2>&1
    s "sync_lock_inversion $c" $? "$O/mut_a_sync_lock_$c.log"
  done
fi
if [[ $M == a || $M == c ]]; then
  timeout 180 g++ -std=c++17 -O1 -DNDEBUG -shared -fPIC $P/release_trace.cpp -I$R/tc_signal/src -I$R/tc_signal/include \
    -I$pb -I$cu/include "${pyincs[@]}" -pthread -o "$B/_p1t_rev6_trace$(python3-config --extension-suffix)" \
    > "$O/mut_${M}_trace_build.log" 2>&1; echo "mut_$M build trace (from the mutant's dlpack_operand.cpp) exit=$?"
fi
if [[ $M == a ]]; then
  $L 15 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$B python3 $P/sync_late_release_cpu.py > "$O/mut_a_sync_late_release.log" 2>&1
  s "sync_late_release" $? "$O/mut_a_sync_late_release.log"
fi
if [[ $M == b ]]; then
  $L 900 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 -m pytest $R/tensor_cuda/tests/test_dlpack.py -q \
    -p no:cacheprovider -k "fix6_v602" > "$O/mut_b_tests.log" 2>&1; echo "mut_b tests exit=$? :: $(tail -1 "$O/mut_b_tests.log")"
  for m in bool index; do
    $L 15 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 $P/core_cast_exit_gpu.py $m > "$O/mut_b_core_cast_exit_$m.log" 2>&1
    s "core_cast_exit_gpu $m" $? "$O/mut_b_core_cast_exit_$m.log"
  done
  python3 -I $W/artifacts/p1t_fix6/make_rev6_copies.py "$R" "$B" "$O/mut_b_copies" > "$O/mut_b_copies.txt"
  (cd "$R" && python3 -I "$B/build_core_host_paths.py") > "$O/mut_b_core_host_regen.log" 2>&1; echo "mut_b regenerate core host exit=$?"
  timeout 300 g++ -std=c++17 -O1 -shared -fPIC -fsanitize=undefined "$B/core_host.cpp" -I"$R/tensor_cuda/include" \
    -I"$pb" -I"$cu/include" "${pyincs[@]}" -L"$cu/lib64" -Wl,-rpath,"$cu/lib64" -lcudart -pthread \
    -o "$B/_p1t_rev6_core$(python3-config --extension-suffix)" > "$O/mut_b_core_host_compile.log" 2>&1
  echo "mut_b compile core host exit=$?"
  for m in bool index; do
    $L 15 env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=$B python3 $P/core_host_cpu.py casts_exit $m \
      > "$O/mut_b_casts_exit_$m.log" 2>&1
    s "core_host_cpu casts_exit $m (regenerated from the mutant)" $? "$O/mut_b_casts_exit_$m.log"
  done
fi
if [[ $M == c ]]; then
  $L 900 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 -m pytest $R/tc_signal/tests/test_fix6_findings.py \
    $R/tc_signal/tests/test_fix2_findings.py -q -p no:cacheprovider -k "v603 or (n3_allocation and error_message)" \
    > "$O/mut_c_tests.log" 2>&1; echo "mut_c tests exit=$? :: $(tail -1 "$O/mut_c_tests.log")"
  for m in query sync device; do
    $L 15 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$B python3 $P/release_trace_cpu.py diagnostic_oom_$m > "$O/mut_c_diagnostic_oom_$m.log" 2>&1
    s "release_trace_cpu diagnostic_oom_$m" $? "$O/mut_c_diagnostic_oom_$m.log"
  done
fi
