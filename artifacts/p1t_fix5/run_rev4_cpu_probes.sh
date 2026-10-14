#!/usr/bin/env bash
# P1T-FIX5: the round-4 reviewer probes that the round-5 probe table does NOT repeat (the CPU ones: host_checks,
# operand_census, real_metadata_plans, lifecycle.cpp + lifecycle_timeout.py, sync_slot_cpu.cpp, core_cpu.cpp,
# core_interop_cpu.cpp, and the FIX2/FIX3 native harness censuses), on one arm. Generated from P1T-FIX4's
# artifacts/p1t_fix4/run_rev4_probes.sh (its cpu part, unchanged logic) with W = this worktree, builds under
# /tmp/p1t_fix5_probes/<arm>/rev4, FIX5 copies of its two helpers, and ONE addition: lifecycle.cpp embeds core's
# lifecycle slice verbatim (1475fae), which FIX5 changed (dlpack_begin_shutdown re-takes the GIL explicitly), so the
# AFTER arm compiles the slice regenerated from its own bindings.cpp with FIX3's regen_lifecycle_slice.py (self-check:
# regenerating from wt_fix4 reproduces the probe; printed). The round-4 GPU probes are byte-identical to round 5's and
# run there. Usage: run_rev4_cpu_probes.sh <arm> <root> <out_dir>. No prior art claimed (run plumbing).
# P1T-FIX4: every probe in artifacts/p1t_rev4/probes/README.md on one arm, sequentially, each in a fresh process; every
# probe run (CPU or GPU) under `flock -w 900 /tmp/forge-gpu.lock timeout <README timeout>`. Logs into <out_dir>.
#   before = 62c8d9f: wt_fix3's prebuilt core + tc_signal (read-only) and CPU probes built from wt_fix3's sources;
#   after  = this worktree's build and CPU probes built from its sources.
# CPU probe builds use the reviewer's build_cpu.sh / build_core_interop_cpu.sh flags into /tmp/p1t_fix4_probes/<arm>/
# (not /tmp/p1t_rev4_build). Unmodified probe sources, except:
#   * path-only copies of host_checks / operand_census / real_metadata_plans (/ release_failures on the before arm):
#     make_probe_copies.py;
#   * core_interop_cpu import mode: run as written (62c8d9f slices, both arms) AND, on the after arm, with its
#     tensor_from_dlpack_capsule regenerated from this tree (regen_core_interop_import.py, self-checked);
#   * sync_slot_cpu.cpp: the reviewer recorded no compile command (empty logs/p1t_rev4_sync_slot_build.log); built with
#     build_cpu.sh's cpu_native flags against the arm's tc_signal/src (it #includes the real dlpack_operand.cpp).
# lifecycle.cpp is arm-independent (its slice, probe lines 11-165, is one verbatim block of both 62c8d9f's and this
# tree's bindings.cpp: checked and logged), built once per arm from the probe file.
# CPU probes run in two conditions: nogpu (CUDA_VISIBLE_DEVICES=, the reviewer's sandbox) and gpu (device visible),
# except core_interop import mode and core_import_query_error.py, which need nogpu by definition.
# Usage: run_rev4_probes.sh <arm> <root> <out_dir> <part>; part = cpu (builds + CPU probes), gpu (GPU probes but
# release_failures), rf (release_failures). Split so that each part runs synchronously within one foreground call.
# No prior art claimed (run plumbing).
set -u
ARM=$1; ROOT=$2; O=$3; PART=cpu
W4=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix4
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix5
P=$W/artifacts/p1t_rev4/probes
B=/tmp/p1t_fix5_probes/$ARM/rev4
L="flock -w 900 /tmp/forge-gpu.lock timeout"
PP=$ROOT/tensor_cuda/tensor_cuda:$ROOT/tensor_cuda:$ROOT/tc_signal
mkdir -p "$O" "$B"
pybind_include=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src/include
cuda_root=/usr/local/cuda-12.6
read -r -a pyincs <<< "$(python3-config --includes)"
read -r -a pylink <<< "$(python3-config --embed --ldflags)"
common=(-std=c++17 -O1 -g -I"$pybind_include" -I"$cuda_root/include" "${pyincs[@]}" "${pylink[@]}" -L"$cuda_root/lib64" -Wl,-rpath,"$cuda_root/lib64" -lcudart)
ubsan=(-fsanitize=undefined -fno-sanitize-recover=undefined)
sha256sum $ROOT/tensor_cuda/tensor_cuda/_tensor_cuda*.so $ROOT/tc_signal/tc_signal/_tc_signal*.so > "$O/rev4_binaries_$ARM.txt"
s() { local tag=$1 rc=$2 log=$3; echo "$ARM $tag exit=$rc :: $(grep -v '^\s*$' "$log" | tr '\n' ' ' | cut -c1-260)"; }
if [[ $PART == cpu ]]; then
# ---- builds (CPU only)
bld() { local name=$1; shift; timeout 180 g++ "$@" > "$B/${name}_build.log" 2>&1; echo "$ARM build $name exit=$?"; }
bld core_cpu $P/core_cpu.cpp -I$ROOT/tensor_cuda/include "${common[@]}" "${ubsan[@]}" -o $B/core_cpu
bld cpu_native $W/artifacts/p1t_rev3/probes/cpu_native.cpp -I$ROOT/tc_signal/src -I$ROOT/tc_signal/include "${common[@]}" -o $B/cpu_native
bld fix2_harness $ROOT/tc_signal/tests/fix2_native_harness.cpp -I$ROOT/tc_signal/src -I$ROOT/tc_signal/include "${common[@]}" "${ubsan[@]}" -o $B/fix2_harness
bld fix3_harness $ROOT/tc_signal/tests/fix3_native_harness.cpp -I$ROOT/tc_signal/src -I$ROOT/tc_signal/include "${common[@]}" "${ubsan[@]}" -o $B/fix3_harness
LIFE=$P/lifecycle.cpp
if [[ $ARM == after ]]; then
  python3 -I $W/artifacts/p1t_fix3/regen_lifecycle_slice.py $P/lifecycle.cpp $W4 $ROOT $B/lifecycle_regen.cpp > "$O/lifecycle_regen_$ARM.txt"
  diff -u $P/lifecycle.cpp $B/lifecycle_regen.cpp > "$O/lifecycle_regen_$ARM.diff"
  LIFE=$B/lifecycle_regen.cpp
fi
bld lifecycle -std=c++17 -shared -fPIC -O1 $LIFE -I$ROOT/tensor_cuda/include -I"$pybind_include" "${pyincs[@]}" -pthread -o "$B/_p1t_rev4_lifecycle$(python3-config --extension-suffix)"
bld sync_slot $P/sync_slot_cpu.cpp -I$ROOT/tc_signal/src -I$ROOT/tc_signal/include "${common[@]}" -o $B/sync_slot
ci=(-std=c++17 -O1 -Itensor_cuda/include -I"$pybind_include" -I"$cuda_root/include" "${pyincs[@]}" "${pylink[@]}" -L"$cuda_root/lib64" -Wl,-rpath,"$cuda_root/lib64" -lcudart)
(cd $ROOT && bld core_spy $P/core_interop_cpu.cpp "${ci[@]}" -DSPY -o $B/core_spy)
(cd $ROOT && bld core_import $P/core_interop_cpu.cpp "${ci[@]}" -o $B/core_import)
if [[ $ARM == after ]]; then
  python3 -I $W/artifacts/p1t_fix5/regen_core_interop_import.py $P/core_interop_cpu.cpp $ROOT $B/core_interop_import_regen.cpp > "$O/core_interop_import_regen_$ARM.txt"
  diff -u $P/core_interop_cpu.cpp $B/core_interop_import_regen.cpp > "$O/core_interop_import_regen_$ARM.diff"
  (cd $ROOT && bld core_import_regen $B/core_interop_import_regen.cpp "${ci[@]}" -o $B/core_import_regen)
fi
python3 -I $W/artifacts/p1t_fix5/make_rev4_probe_copies.py $ARM $ROOT $B > "$O/probe_copies_$ARM.txt"
python3 - "$P/lifecycle.cpp" "$ROOT/tensor_cuda/src/bindings.cpp" > "$O/lifecycle_slice_check_$ARM.txt" <<'PY'
import sys
probe = open(sys.argv[1]).read().splitlines(True)
src = open(sys.argv[2]).read()
print('lifecycle.cpp lines 11-165 one verbatim block of', sys.argv[2], ':', ''.join(probe[10:165]) in src)
PY
sha256sum $B/core_cpu $B/cpu_native $B/fix2_harness $B/fix3_harness $B/_p1t_rev4_lifecycle*.so $B/sync_slot $B/core_spy $B/core_import $B/core_import_regen $B/*_paths.py 2>/dev/null > "$O/probe_build_sha256_$ARM.txt"
cd /tmp
# ---- CPU probes, two conditions
for c in nogpu gpu; do
  CV=(); [[ $c == nogpu ]] && CV=(CUDA_VISIBLE_DEVICES=)
  E=(env PYTHONDONTWRITEBYTECODE=1 "${CV[@]}")
  (cd $B && $L 120 "${E[@]}" python3 host_checks_paths.py > "$O/rev4_host_checks_${ARM}_$c.log" 2>&1; echo "$ARM host_checks $c collector exit=$?"; cp host_checks.json "$O/rev4_host_checks_${ARM}_$c.json")
  $L 40 "${E[@]}" $B/fix2_harness ck_census > "$O/rev4_ck_census_${ARM}_$c.log" 2>&1; r=$?; echo "$ARM fix2_harness ck_census $c exit=$r :: $(tail -1 "$O/rev4_ck_census_${ARM}_$c.log" | cut -c1-120) ubsan_lines=$(grep -c 'runtime error' "$O/rev4_ck_census_${ARM}_$c.log")"
  $L 40 "${E[@]}" $B/fix3_harness accounting > "$O/rev4_accounting_${ARM}_$c.log" 2>&1; r=$?; echo "$ARM fix3_harness accounting $c exit=$r :: $(tail -1 "$O/rev4_accounting_${ARM}_$c.log" | cut -c1-120) ubsan_lines=$(grep -c 'runtime error' "$O/rev4_accounting_${ARM}_$c.log")"
  for m in control short default; do
    t=15; [[ $m == default ]] && t=25
    $L $t "${E[@]}" PYTHONPATH=$B python3 $P/lifecycle_timeout.py $m > "$O/rev4_lifecycle_${m}_${ARM}_$c.log" 2>&1; s "lifecycle_timeout $m $c" $? "$O/rev4_lifecycle_${m}_${ARM}_$c.log"
  done
  $L 40 "${E[@]}" PYTHONPATH=$ROOT/tc_signal python3 $B/operand_census_paths.py > "$O/rev4_operand_census_${ARM}_$c.log" 2>&1; s "operand_census $c" $? "$O/rev4_operand_census_${ARM}_$c.log"
  $L 40 "${E[@]}" PYTHONPATH=$ROOT/tc_signal python3 $B/real_metadata_plans_paths.py > "$O/rev4_real_metadata_plans_${ARM}_$c.log" 2>&1; s "real_metadata_plans $c" $? "$O/rev4_real_metadata_plans_${ARM}_$c.log"
  $L 15 "${E[@]}" $B/sync_slot > "$O/rev4_sync_slot_${ARM}_$c.log" 2>&1; s "sync_slot_cpu (rc 0 = defect trace reproduced) $c" $? "$O/rev4_sync_slot_${ARM}_$c.log"
  $L 15 "${E[@]}" $B/core_spy > "$O/rev4_core_spy_${ARM}_$c.log" 2>&1; s "core_interop SPY (rc 0 = defect trace reproduced; 62c8d9f slices) $c" $? "$O/rev4_core_spy_${ARM}_$c.log"
done
E=(env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=)
$L 15 "${E[@]}" $B/core_import > "$O/rev4_core_import_${ARM}_nogpu.log" 2>&1; s "core_interop import (rc 0 = defect reproduced; 62c8d9f slices) nogpu" $? "$O/rev4_core_import_${ARM}_nogpu.log"
if [[ $ARM == after ]]; then
  $L 15 "${E[@]}" $B/core_import_regen > "$O/rev4_core_import_regen_${ARM}_nogpu.log" 2>&1; s "core_interop import REGENERATED from this tree (rc 0 = defect; 1 = refused) nogpu" $? "$O/rev4_core_import_regen_${ARM}_nogpu.log"
fi
$L 30 "${E[@]}" PYTHONPATH=$PP python3 $P/core_import_query_error.py > "$O/rev4_core_import_query_error_$ARM.log" 2>&1; s "core_import_query_error (nogpu)" $? "$O/rev4_core_import_query_error_$ARM.log"
fi
