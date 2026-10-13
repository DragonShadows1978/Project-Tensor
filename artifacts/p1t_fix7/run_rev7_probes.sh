#!/usr/bin/env bash
# P1T-FIX7: every round-7 probe (artifacts/p1t_rev7/probes/) on one arm, each in a fresh process, every run under
# `flock -w 900 /tmp/forge-gpu.lock timeout <bound>` (GPU or not). One summary line per run; logs in <out_dir>
# (ABSOLUTE path required: the builds cd into the arm's root).
#   before = /tmp/p1t_fix7_before_tree (byte-identical copies of wt_fix6's 429446d sources, tests and binaries, plus
#            byte-identical probe files; make_before_tree.sh);
#   after  = this worktree and its build.
# Builds go to /tmp/p1t_fix7_rev7/<arm>/ (never /tmp/p1t_rev7_build, the reviewer's retained scratch). Unmodified probe
# sources run from the arm's tree, except the path-only copies of make_rev7_copies.py (build_cpu.sh, build_core_host.py,
# run_cpu_checks.py, argument_census_cpu.py, gpu_regressions.py: build dir, root and receipt paths only; diffs next to
# the copies). <bin>/tc_signal holds a byte-identical copy of the arm's built package (run_cpu_checks / census use it).
# Regeneration self-check: on the before arm, the core slices build_core_host takes must hash as the reviewer pinned
# them (artifacts/p1t_rev7/core_source_slices.json).
# Usage: run_rev7_probes.sh <arm> <arm_root> <out_dir>. No prior art claimed (run plumbing).
set -u
ARM=$1; ROOT=$2; O=$3
case $O in /*) ;; *) echo "out_dir must be absolute"; exit 2;; esac
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix7
P=$ROOT/artifacts/p1t_rev7/probes
B=/tmp/p1t_fix7_rev7/$ARM
L="flock -w 900 /tmp/forge-gpu.lock timeout"
mkdir -p "$O" "$B/tc_signal"
s() { local tag=$1 rc=$2 log=$3; echo "$ARM $tag exit=$rc :: $(grep -v '^\s*$' "$log" | tr '\n' ' ' | cut -c1-300)"; }
sha256sum $ROOT/tensor_cuda/tensor_cuda/_tensor_cuda*.so $ROOT/tc_signal/tc_signal/_tc_signal*.so > "$O/binaries_$ARM.txt"
python3 -I $W/artifacts/p1t_fix7/make_rev7_copies.py "$ROOT" "$B" "$O" > "$O/copies_$ARM.txt"
cp -r "$ROOT/tc_signal/tc_signal" "$B/tc_signal/"
sha256sum "$B"/tc_signal/tc_signal/_tc_signal*.so >> "$O/binaries_$ARM.txt"
# ---- builds (the reviewer's build_cpu.sh / build_core_host.py, path-only copies, from the arm's sources)
(cd "$ROOT" && $L 900 bash "$B/build_cpu_paths.sh") > "$O/build_cpu_$ARM.log" 2>&1; echo "$ARM build_cpu exit=$?"
(cd "$ROOT" && $L 900 python3 -I "$B/build_core_host_paths.py") > "$O/build_core_host_$ARM.log" 2>&1
echo "$ARM build_core_host exit=$?"
python3 -I - "$W/artifacts/p1t_rev7/core_source_slices.json" "$O/core_source_slices.json" > "$O/slices_selfcheck_$ARM.txt" <<'PY'
import json, sys
ref = json.load(open(sys.argv[1])); got = json.load(open(sys.argv[2]))
for r, g in zip(ref, got):
    print(g['path'], g['start_line'], g['end_line'], 'identical_to_reviewer_pin=', r['sha256'] == g['sha256'])
print('all_identical=', all(r['sha256'] == g['sha256'] for r, g in zip(ref, got)) and len(ref) == len(got))
PY
sha256sum "$B"/_p1t_rev6_trace*.so "$B"/_p1t_rev7_core*.so "$B"/fix2_harness "$B"/fix3_harness "$B"/cpu_native \
  "$B"/core_host.cpp >> "$O/binaries_$ARM.txt" 2>/dev/null
# ---- run_cpu_checks.py (14 checks, no visible GPU, their own per-job bounds)
$L 900 python3 -I "$B/run_cpu_checks_paths.py" > "$O/run_cpu_checks_$ARM.log" 2>&1; echo "$ARM run_cpu_checks exit=$?"
sed 's/^/  /' "$O/run_cpu_checks_$ARM.log" | cut -c1-200 | head -20
# ---- argument_census_cpu.py (no GPU work; the arm's add-on package copy and the regenerated core slice)
$L 300 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$B:$B/tc_signal python3 "$B/argument_census_cpu_paths.py" \
  > "$O/argument_census_$ARM.log" 2>&1
s "argument_census_cpu" $? "$O/argument_census_$ARM.log"
# ---- V7-01 add-on: knob_exit_cpu.py (no visible GPU, and GPU visible), each with --control
for c in nogpu gpu; do
  if [[ $c == nogpu ]]; then cv="CUDA_VISIBLE_DEVICES="; else cv="P1T_FIX7_GPU=1"; fi
  for ctl in "" "--control"; do
    t=knob_exit_cpu_${c}${ctl:+_control}
    $L 60 env $cv PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$ROOT/tc_signal python3 $P/knob_exit_cpu.py $ctl > "$O/$t.log" 2>&1
    s "$t" $? "$O/$t.log"
  done
done
# ---- V7-01 core, host slice: core_knob_exit_cpu.py (regenerated host module of the arm), with --control
for ctl in "" "--control"; do
  t=core_knob_exit_cpu${ctl:+_control}
  $L 60 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$B python3 $P/core_knob_exit_cpu.py $ctl > "$O/$t.log" 2>&1
  s "$t" $? "$O/$t.log"
done
# ---- V7-01 core, real module on the GPU: core_knob_exit_gpu.py (manages its own PYTHONPATH from its root)
for ctl in "" "--control"; do
  t=core_knob_exit_gpu${ctl:+_control}
  $L 120 env PYTHONDONTWRITEBYTECODE=1 python3 $P/core_knob_exit_gpu.py $ctl > "$O/$t.log" 2>&1
  s "$t" $? "$O/$t.log"
done
# ---- V7-02: query_diagnostic_cpu.py (trace module of the arm) and query_diagnostic_gpu.py (real module, GPU)
$L 60 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$B python3 $P/query_diagnostic_cpu.py > "$O/query_diagnostic_cpu.log" 2>&1
s "query_diagnostic_cpu" $? "$O/query_diagnostic_cpu.log"
$L 120 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$ROOT/tc_signal python3 $P/query_diagnostic_gpu.py > "$O/query_diagnostic_gpu.log" 2>&1
s "query_diagnostic_gpu" $? "$O/query_diagnostic_gpu.log"
# ---- gpu_regressions.py fix56 / semantics / multidevice (path-only copy: junit receipt -> out_dir)
for lane in fix56 semantics multidevice; do
  (cd "$ROOT" && $L 950 env PYTHONDONTWRITEBYTECODE=1 python3 "$B/gpu_regressions_paths.py" $lane) > "$O/gpu_regressions_$lane.log" 2>&1
  rc=$?; echo "$ARM gpu_regressions $lane exit=$rc :: $(tail -2 "$O/gpu_regressions_$lane.log" | tr '\n' ' ' | cut -c1-300)"
done
echo "$ARM done"
