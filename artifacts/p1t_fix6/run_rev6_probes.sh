#!/usr/bin/env bash
# P1T-FIX6: every round-6 probe (artifacts/p1t_rev6/probes/) on one arm, each run in a fresh process, every run under
# `flock -w 900 /tmp/forge-gpu.lock timeout <bound>` (GPU or not). One summary line per run; logs in <out_dir>.
#   before = acd42fa: wt_fix5's prebuilt core + tc_signal (read-only); CPU probe modules built from wt_fix5's sources;
#   after  = this worktree's build; CPU probe modules built from its sources.
# Builds go to /tmp/p1t_fix6_rev6/<arm>/ (never /tmp/p1t_rev6_build, the reviewer's). Unmodified probe sources, except
# (all reported as deviations):
#   * path-only copies of build_host_probes.sh, build_core_host.py and run_cpu_checks.py (make_rev6_copies.py; diffs
#     next to the copies);
#   * the core host module: the reviewer recorded no compile command for core_host.cpp; reconstructed from the
#     reviewer's binary (/tmp/p1t_rev6_build/_p1t_rev6_core*.so links libcudart and libubsan): g++ -std=c++17 -O1
#     -shared -fPIC -fsanitize=undefined against the arm's tensor_cuda/include;
#   * run_cpu_checks.py puts <bin>/tc_signal on PYTHONPATH (the reviewer's CMake rebuild of the reviewed tree); here that
#     directory holds a byte-identical COPY of the arm's own built package (python files + _tc_signal*.so; sha256
#     recorded), not a rebuild.
# Regeneration self-check: the slices that build_core_host_paths.py takes from the before arm (acd42fa) must have the
# hashes the reviewer pinned in artifacts/p1t_rev6/source_slices.json.
# Usage: run_rev6_probes.sh <arm> <arm_root> <out_dir>
# No prior art claimed (run plumbing).
set -u
ARM=$1; ROOT=$2; O=$3
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix6
P=$W/artifacts/p1t_rev6/probes
B=/tmp/p1t_fix6_rev6/$ARM
L="flock -w 900 /tmp/forge-gpu.lock timeout"
PP=$ROOT/tensor_cuda/tensor_cuda:$ROOT/tensor_cuda:$ROOT/tc_signal
pb=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src/include
cu=/usr/local/cuda-12.6
read -r -a pyincs <<< "$(python3-config --includes)"
mkdir -p "$O" "$B/tc_signal"
sha256sum $ROOT/tensor_cuda/tensor_cuda/_tensor_cuda*.so $ROOT/tc_signal/tc_signal/_tc_signal*.so > "$O/binaries_$ARM.txt"
s() { local tag=$1 rc=$2 log=$3; echo "$ARM $tag exit=$rc :: $(grep -v '^\s*$' "$log" | tr '\n' ' ' | cut -c1-260)"; }

# ---- copies and builds
python3 -I $W/artifacts/p1t_fix6/make_rev6_copies.py "$ROOT" "$B" "$O" > "$O/copies_$ARM.txt"
cp -r "$ROOT/tc_signal/tc_signal" "$B/tc_signal/"
sha256sum "$B"/tc_signal/tc_signal/_tc_signal*.so >> "$O/binaries_$ARM.txt"
(cd "$ROOT" && bash "$B/build_host_probes_paths.sh") > "$O/build_host_probes_$ARM.log" 2>&1; echo "$ARM build_host_probes exit=$?"
(cd "$ROOT" && python3 -I "$B/build_core_host_paths.py") > "$O/build_core_host_$ARM.log" 2>&1; echo "$ARM build_core_host (regenerate) exit=$?"
python3 -I - "$W/artifacts/p1t_rev6/source_slices.json" "$O/source_slices.json" > "$O/slices_selfcheck_$ARM.txt" <<'PY'
import json, sys
ref = json.load(open(sys.argv[1])); got = json.load(open(sys.argv[2]))
for r, g in zip(ref, got):
    print(g['path'], g['start_line'], g['end_line'], 'identical_to_reviewer_pin=', r['sha256'] == g['sha256'])
print('all_identical=', all(r['sha256'] == g['sha256'] for r, g in zip(ref, got)) and len(ref) == len(got))
PY
timeout 300 g++ -std=c++17 -O1 -shared -fPIC -fsanitize=undefined "$B/core_host.cpp" -I"$ROOT/tensor_cuda/include" \
  -I"$pb" -I"$cu/include" "${pyincs[@]}" -L"$cu/lib64" -Wl,-rpath,"$cu/lib64" -lcudart -pthread \
  -o "$B/_p1t_rev6_core$(python3-config --extension-suffix)" > "$O/core_host_compile_$ARM.log" 2>&1
echo "$ARM compile core_host exit=$?"
sha256sum "$B"/_p1t_rev6_*.so "$B"/fix2_harness "$B"/fix3_harness "$B"/cpu_native "$B"/core_host.cpp >> "$O/binaries_$ARM.txt" 2>/dev/null

# ---- the reviewer's CPU checks (no visible GPU; their own per-job timeouts)
$L 900 python3 -I "$B/run_cpu_checks_paths.py" > "$O/run_cpu_checks_$ARM.log" 2>&1; echo "$ARM run_cpu_checks exit=$?"
sed 's/^/  /' "$O/run_cpu_checks_$ARM.log" | cut -c1-240

# ---- sync_lock_inversion_cpu.py on the arm's real package, both conditions
for c in nogpu gpu; do
  if [[ $c == nogpu ]]; then cv="CUDA_VISIBLE_DEVICES="; else cv="P1T_FIX6_GPU=1"; fi
  $L 15 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP $cv python3 $P/sync_lock_inversion_cpu.py > "$O/sync_lock_${c}_$ARM.log" 2>&1
  s "sync_lock_inversion $c" $? "$O/sync_lock_${c}_$ARM.log"
done
# ---- sync_late_release_cpu.py (the trace module built from the arm's dlpack_operand.cpp)
$L 15 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$B python3 $P/sync_late_release_cpu.py > "$O/sync_late_release_$ARM.log" 2>&1
s "sync_late_release" $? "$O/sync_late_release_$ARM.log"
# ---- core_cast_exit_gpu.py on the arm's real core (one GPU)
for m in bool index; do
  for ctl in "" "--control"; do
    t=${m}${ctl:+_control}
    $L 15 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP python3 $P/core_cast_exit_gpu.py $m $ctl > "$O/core_cast_exit_${t}_$ARM.log" 2>&1
    s "core_cast_exit_gpu $m $ctl" $? "$O/core_cast_exit_${t}_$ARM.log"
  done
done
