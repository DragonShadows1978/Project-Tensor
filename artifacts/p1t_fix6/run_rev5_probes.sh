#!/usr/bin/env bash
# P1T-FIX6 (a copy of P1T-FIX5's runner, retargeted: before = acd42fa = wt_fix5's prebuilt binaries, after = this
# worktree; builds under /tmp/p1t_fix6_probes/<arm>; plus the order's Required-fix-1 trace deviation, see TRACE below).
# P1T-FIX5: every probe in the round-5 report's probe table (docs/P1T_REV5_REPORT.md "Probe list";
# artifacts/p1t_rev5/probes/) on one arm, sequentially, each in a fresh process, every run (CPU or GPU) under
# `flock -w 900 /tmp/forge-gpu.lock timeout <probe header timeout>`. Logs into <out_dir>; one summary line per run.
#   before = 1475fae: wt_fix4's prebuilt core + tc_signal (read-only), CPU probes built from wt_fix4's sources;
#   after  = this worktree's build, CPU probes built from its sources.
# Builds go to /tmp/p1t_fix5_probes/<arm>/ (never /tmp/p1t_rev5_*). Unmodified probe sources, except (all reported):
#   * path-only copies (make_probe_copies.py): device_error_sweep_single_gpu.py (root + basetemp), release_failures.py
#     (before arm: root = wt_fix4), build_cpu.sh (build dir);
#   * the trace module _p1t_rev5_trace: the reviewer recorded no build command (logs/p1t_rev5_trace_build.log is empty);
#     reconstructed from the reviewer's binary (no libcudart dependency: the probe stubs every CUDA call it needs) as a
#     plain g++ -shared build of addon_trace.cpp against the arm's tc_signal/src; the undefined cuda* symbols are listed
#     (must be none);
#   * handle_insert_oom_cpu.cpp and fill_grid_cpu.cpp embed product code (1475fae text), so as written they are
#     arm-independent: they run as written AND regenerated from the arm's sources (tc_signal/tests/fix5_regen.py), with
#     a self-check that regenerating from 1475fae reproduces the probe byte for byte;
#   * the shims /tmp/p1t_rev5_query_once.so and /tmp/p1t_rev5_fill_grid.so are the existing files at the paths the
#     probes hard-code, verified byte-identical to fresh builds of the probe sources (ledger 05:20), used read-only.
# Conditions: nogpu = CUDA_VISIBLE_DEVICES= (the reviewer's sandbox); gpu = the device visible.
# Usage: run_rev5_probes.sh <arm> <root> <out_dir> <part>; part = cpu | buildcpu | gpu | rf (each fits one call).
# No prior art claimed (run plumbing).
set -u
ARM=$1; ROOT=$2; O=$3; PART=$4
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix6
W4=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix4
P=$W/artifacts/p1t_rev5/probes
B=/tmp/p1t_fix6_probes/$ARM
L="flock -w 900 /tmp/forge-gpu.lock timeout"
PP=$ROOT/tensor_cuda/tensor_cuda:$ROOT/tensor_cuda:$ROOT/tc_signal
pybind_include=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src/include
cuda_root=/usr/local/cuda-12.6
read -r -a pyincs <<< "$(python3-config --includes)"
mkdir -p "$O" "$B" "$B/trace"
sha256sum $ROOT/tensor_cuda/tensor_cuda/_tensor_cuda*.so $ROOT/tc_signal/tc_signal/_tc_signal*.so > "$O/rev5_binaries_${ARM}_$PART.txt"
s() { local tag=$1 rc=$2 log=$3; echo "$ARM $tag exit=$rc :: $(grep -v '^\s*$' "$log" | tr '\n' ' ' | cut -c1-300)"; }
cd /tmp
if [[ $PART == cpu ]]; then
  python3 -I $W/artifacts/p1t_fix6/make_probe_copies.py $ARM $ROOT $B > "$O/probe_copies_$ARM.txt"
  # ---- builds
  timeout 180 g++ -std=c++17 -O1 -shared -fPIC $P/addon_trace.cpp -I$ROOT/tc_signal/src -I$ROOT/tc_signal/include \
    -I$pybind_include -I$cuda_root/include "${pyincs[@]}" -o "$B/trace/_p1t_rev5_trace$(python3-config --extension-suffix)" \
    > "$O/trace_build_$ARM.log" 2>&1; echo "$ARM build trace exit=$?"
  echo "undefined cuda symbols: [$(nm -D --undefined-only $B/trace/_p1t_rev5_trace*.so | grep -i ' cuda' | tr '\n' ' ')]" >> "$O/trace_build_$ARM.log"
  timeout 120 g++ -std=c++17 -O1 $P/handle_insert_oom_cpu.cpp -I$cuda_root/include -o $B/handle_oom > "$O/handle_oom_build_$ARM.log" 2>&1; echo "$ARM build handle_oom (as written) exit=$?"
  timeout 120 g++ -std=c++17 -O1 $P/fill_grid_cpu.cpp -o $B/fill_grid > "$O/fill_grid_build_$ARM.log" 2>&1; echo "$ARM build fill_grid (as written) exit=$?"
  python3 -I - "$P" "$ROOT" "$W4" "$B" "$W/tc_signal/tests" > "$O/regen_$ARM.txt" <<'PY'
import sys
from pathlib import Path
P, root, base, B, tests = sys.argv[1:6]
sys.path.insert(0, tests)
from fix5_regen import regen_handle_probe, regen_fill_probe
hp = (Path(P) / 'handle_insert_oom_cpu.cpp').read_text()
fp = (Path(P) / 'fill_grid_cpu.cpp').read_text()
src = lambda r, f: (Path(r) / f).read_text()
print('handle self-check (regen from 1475fae == probe):',
      regen_handle_probe(hp, src(base, 'tc_signal/src/conv.cu'), src(base, 'tc_signal/src/lstm.cu')) == hp)
print('fill self-check (regen from 1475fae == probe):', regen_fill_probe(fp, src(base, 'tensor_cuda/src/kernels.cu')) == fp)
h = regen_handle_probe(hp, src(root, 'tc_signal/src/conv.cu'), src(root, 'tc_signal/src/lstm.cu'))
f = regen_fill_probe(fp, src(root, 'tensor_cuda/src/kernels.cu'))
(Path(B) / 'handle_insert_oom_regen.cpp').write_text(h)
(Path(B) / 'fill_grid_regen.cpp').write_text(f)
print('regenerated from', root, ': handle identical to probe =', h == hp, '; fill identical to probe =', f == fp)
PY
  diff -u $P/handle_insert_oom_cpu.cpp $B/handle_insert_oom_regen.cpp > "$O/handle_insert_oom_regen_$ARM.diff"
  diff -u $P/fill_grid_cpu.cpp $B/fill_grid_regen.cpp > "$O/fill_grid_regen_$ARM.diff"
  timeout 120 g++ -std=c++17 -O1 $B/handle_insert_oom_regen.cpp -I$cuda_root/include -o $B/handle_oom_regen > "$O/handle_oom_regen_build_$ARM.log" 2>&1; echo "$ARM build handle_oom (regenerated) exit=$?"
  timeout 120 g++ -std=c++17 -O1 $B/fill_grid_regen.cpp -o $B/fill_grid_regen > "$O/fill_grid_regen_build_$ARM.log" 2>&1; echo "$ARM build fill_grid (regenerated) exit=$?"
  sha256sum $B/trace/_p1t_rev5_trace*.so $B/handle_oom $B/handle_oom_regen $B/fill_grid $B/fill_grid_regen $B/*_paths.* > "$O/probe_build_sha256_$ARM.txt" 2>/dev/null
  # ---- CPU probes
  for c in nogpu; do
    for ctl in "" "--control"; do
      t=full_${c}${ctl:+_control}
      $L 30 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP CUDA_VISIBLE_DEVICES= python3 $P/exit_gate.py full $ctl > "$O/rev5_exit_gate_${t}_$ARM.log" 2>&1; s "exit_gate full $c $ctl" $? "$O/rev5_exit_gate_${t}_$ARM.log"
    done
  done
  for ctl in "" "--control"; do
    t=trace${ctl:+_control}
    $L 30 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$B/trace python3 $P/exit_gate.py trace $ctl > "$O/rev5_exit_gate_${t}_$ARM.log" 2>&1; s "exit_gate trace $ctl" $? "$O/rev5_exit_gate_${t}_$ARM.log"
  done
  # ---- TRACE (order Required fix 1, a reported deviation): the round-5 trace child models the exit hook as
  # synchronize() alone. FIX6 removes synchronize()'s wait for other threads (V6-01), so that model fails as written.
  # The copy below differs only in what the hook calls: the trace module's real exit entry point (synchronize(), then
  # close_admission_and_wait, as tc_signal's atexit hook does), via one added binding `close` in a copy of addon_trace.cpp.
  python3 -I - "$P" "$B" > "$O/trace_exit_entry_copy_$ARM.txt" <<'PY'
import difflib, sys
from pathlib import Path
P, B = Path(sys.argv[1]), Path(sys.argv[2])
cpp = (P / 'addon_trace.cpp').read_text()
a = ' m.def("synchronize",&tcs::synchronize_and_release_all);\n'
assert cpp.count(a) == 1
cpp2 = cpp.replace(a, a + ' m.def("close",&tcs::close_admission_and_wait);  // FIX6 copy: the real exit entry point\n')
py = (P / 'addon_exit_cpu.py').read_text()
b = "    n.synchronize()\n    print('HOOK returned"
assert py.count(b) == 1
py2 = py.replace(b, "    n.synchronize(); n.close(5.0)  # FIX6 copy: tc_signal's real exit sequence\n    print('HOOK returned")
(B / 'addon_trace_exit_entry.cpp').write_text(cpp2)
(B / 'addon_exit_cpu_exit_entry.py').write_text(py2)
for name, x, y in (('addon_trace.cpp', cpp, cpp2), ('addon_exit_cpu.py', py, py2)):
    sys.stdout.write(''.join(difflib.unified_diff(x.splitlines(True), y.splitlines(True), name, name + ' (FIX6 copy)')))
PY
  mkdir -p $B/trace_exit_entry
  timeout 180 g++ -std=c++17 -O1 -shared -fPIC $B/addon_trace_exit_entry.cpp -I$ROOT/tc_signal/src -I$ROOT/tc_signal/include \
    -I$pybind_include -I$cuda_root/include "${pyincs[@]}" -o "$B/trace_exit_entry/_p1t_rev5_trace$(python3-config --extension-suffix)" \
    > "$O/trace_exit_entry_build_$ARM.log" 2>&1; echo "$ARM build trace (exit-entry copy) exit=$?"
  python3 -I - "$P/exit_gate.py" "$B/exit_gate_exit_entry.py" <<'PY'
import sys
from pathlib import Path
s = Path(sys.argv[1]).read_text()
a = "name={'full':'addon_exit_full_cpu.py','trace':'addon_exit_cpu.py','gpu':'addon_exit_gpu.py'}[mode]"
assert s.count(a) == 1
Path(sys.argv[2]).write_text(s.replace(a, "name={'trace':'addon_exit_cpu_exit_entry.py'}[mode]  # FIX6 copy (path only)"))
PY
  $L 30 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$B/trace_exit_entry python3 $B/exit_gate_exit_entry.py trace > "$O/rev5_exit_gate_trace_exit_entry_$ARM.log" 2>&1; s "exit_gate trace (copy: hook = synchronize + close, the real exit entry)" $? "$O/rev5_exit_gate_trace_exit_entry_$ARM.log"
  $L 15 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$B/trace python3 $P/query_error_cpu.py > "$O/rev5_query_error_cpu_$ARM.log" 2>&1; s "query_error_cpu" $? "$O/rev5_query_error_cpu_$ARM.log"
  $L 15 $B/handle_oom > "$O/rev5_handle_oom_written_$ARM.log" 2>&1; s "handle_insert_oom_cpu (as written: 1475fae slices)" $? "$O/rev5_handle_oom_written_$ARM.log"
  $L 15 $B/handle_oom_regen > "$O/rev5_handle_oom_regen_$ARM.log" 2>&1; s "handle_insert_oom_cpu (regenerated from $ARM)" $? "$O/rev5_handle_oom_regen_$ARM.log"
  $L 15 $B/fill_grid > "$O/rev5_fill_grid_written_$ARM.log" 2>&1; s "fill_grid_cpu (as written: baseline nblk)" $? "$O/rev5_fill_grid_written_$ARM.log"
  $L 15 $B/fill_grid_regen > "$O/rev5_fill_grid_regen_$ARM.log" 2>&1; s "fill_grid_cpu (regenerated from $ARM)" $? "$O/rev5_fill_grid_regen_$ARM.log"
  $L 30 env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=$PP python3 $P/core_import_query_error.py > "$O/rev5_core_import_query_error_$ARM.log" 2>&1; s "core_import_query_error (nogpu)" $? "$O/rev5_core_import_query_error_$ARM.log"
fi
if [[ $PART == buildcpu ]]; then
  python3 -I $W/artifacts/p1t_fix6/make_probe_copies.py $ARM $ROOT $B > "$O/probe_copies_buildcpu_$ARM.txt"
  (cd $ROOT && $L 1100 env PYTHONDONTWRITEBYTECODE=1 bash $B/build_cpu_paths.sh > "$O/rev5_build_cpu_$ARM.log" 2>&1; echo "$ARM build_cpu.sh (path copy, cwd $ROOT) exit=$?")
  ls -la $B/rev5_build >> "$O/rev5_build_cpu_$ARM.log" 2>&1
fi
if [[ $PART == gpu ]]; then
  E=(env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP)
  for mode in full gpu; do
    for ctl in "" "--control"; do
      t=${mode}_gpuvisible${ctl:+_control}
      $L 30 "${E[@]}" python3 $P/exit_gate.py $mode $ctl > "$O/rev5_exit_gate_${t}_$ARM.log" 2>&1; s "exit_gate $mode (GPU visible) $ctl" $? "$O/rev5_exit_gate_${t}_$ARM.log"
    done
  done
  $L 30 "${E[@]}" LD_PRELOAD=/tmp/p1t_rev5_query_once.so python3 $P/query_error_single_gpu.py > "$O/rev5_query_error_single_gpu_$ARM.log" 2>&1; s "query_error_single_gpu" $? "$O/rev5_query_error_single_gpu_$ARM.log"
  $L 30 "${E[@]}" LD_PRELOAD=/tmp/p1t_rev5_fill_grid.so python3 $P/fill_grid_single_gpu.py > "$O/rev5_fill_grid_single_gpu_$ARM.log" 2>&1; s "fill_grid_single_gpu" $? "$O/rev5_fill_grid_single_gpu_$ARM.log"
  $L 300 "${E[@]}" python3 $B/device_error_sweep_single_gpu_paths.py > "$O/rev5_device_error_sweep_$ARM.log" 2>&1; s "device_error_sweep_single_gpu (path copy, root $ROOT)" $? "$O/rev5_device_error_sweep_$ARM.log"
  $L 60 "${E[@]}" python3 $P/core_export_device.py > "$O/rev5_core_export_device_$ARM.log" 2>&1; s "core_export_device (3 = INCONCLUSIVE, one GPU)" $? "$O/rev5_core_export_device_$ARM.log"
  $L 30 "${E[@]}" python3 $P/sync_existing_error.py > "$O/rev5_sync_existing_error_$ARM.log" 2>&1; s "sync_existing_error" $? "$O/rev5_sync_existing_error_$ARM.log"
  $L 60 "${E[@]}" python3 $P/addon_lifetime.py > "$O/rev5_addon_lifetime_$ARM.log" 2>&1; r=$?
  echo "$ARM addon_lifetime exit=$r incorrect=$(grep -c '"correct": false' "$O/rev5_addon_lifetime_$ARM.log") correct=$(grep -c '"correct": true' "$O/rev5_addon_lifetime_$ARM.log") :: $(tail -1 "$O/rev5_addon_lifetime_$ARM.log")"
  $L 60 "${E[@]}" python3 $P/core_edges.py > "$O/rev5_core_edges_$ARM.log" 2>&1; s "core_edges" $? "$O/rev5_core_edges_$ARM.log"
  $L 60 "${E[@]}" python3 $P/core_device_drain.py > "$O/rev5_core_device_drain_$ARM.log" 2>&1; s "core_device_drain (3 = INCONCLUSIVE, one GPU)" $? "$O/rev5_core_device_drain_$ARM.log"
  for m in during bounded; do
    T0=$(date +%s.%N)
    $L 25 "${E[@]}" python3 $P/core_exit_overrun.py $m > "$O/rev5_core_exit_overrun_${m}_$ARM.log" 2>&1; r=$?
    echo "$ARM core_exit_overrun $m exit=$r wall_s=$(python3 -c "import time; print(round(time.time() - $T0, 1))") :: $(grep -v '^RESULT' "$O/rev5_core_exit_overrun_${m}_$ARM.log" | tr '\n' ' ' | cut -c1-200)"
  done
  $L 90 "${E[@]}" python3 $P/numerics_and_streams.py > "$O/rev5_numerics_and_streams_$ARM.log" 2>&1; s "numerics_and_streams" $? "$O/rev5_numerics_and_streams_$ARM.log"
  $L 90 "${E[@]}" python3 $P/cuda_errors_multidevice.py > "$O/rev5_cuda_errors_multidevice_$ARM.log" 2>&1; s "cuda_errors_multidevice (3 = INCONCLUSIVE, one GPU)" $? "$O/rev5_cuda_errors_multidevice_$ARM.log"
fi
if [[ $PART == rf ]]; then
  E=(env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PP)
  RF=$P/release_failures.py; [[ $ARM == before ]] && RF=$B/release_failures_paths.py
  $L 650 "${E[@]}" python3 $RF > "$O/rev5_release_failures_$ARM.log" 2>&1; r=$?
  echo "$ARM release_failures exit=$r :: $(grep -E 'passed|failed|skipped|error' "$O/rev5_release_failures_$ARM.log" | tail -2 | tr '\n' ' ')"
fi
