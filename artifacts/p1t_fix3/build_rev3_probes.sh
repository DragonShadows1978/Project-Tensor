#!/usr/bin/env bash
# P1T-FIX3: build the round-3 reviewer's CPU probes for one arm with the reviewer's exact compile commands
# (artifacts/p1t_rev3/probes/build_cpu.sh: same flags, same UBSan options), into /tmp/p1t_fix3_probes/<arm>/ instead of
# /tmp/p1t_rev3_build (outside this seat's area). The probe sources are compiled UNMODIFIED from this worktree's
# artifacts/p1t_rev3/probes/, except:
#   * lifecycle_slice.cpp: before = the committed file (5f9f955 slice); after = regenerated from <root> by
#     regen_lifecycle_slice.py (self-check: regenerating from 5f9f955 reproduces the committed file byte for byte);
#   * pending_bytes_adapt.cpp: the reviewer-authorized adaptation of the "entry" branch (make_pending_bytes_adapt.py),
#     built IN ADDITION to the unmodified pending_bytes.cpp.
# `#include "dlpack_operand.cpp"` and tc headers resolve through -I<root>/tc_signal/src and -I<root>/tensor_cuda/include
# (the reviewer's relative -I flags, made absolute for the arm). The reviewer's scratch tc_signal copy is not rebuilt:
# the arm's own built package is used (before: wt_fix2's prebuilt .so, read-only; after: this worktree's build).
# Usage: build_rev3_probes.sh <arm> <root> <log_dir>
# No prior art claimed (build plumbing).
set -u
ARM=$1; ROOT=$2; LOGS=$3
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix3
F2=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix2
P=$W/artifacts/p1t_rev3/probes
O=/tmp/p1t_fix3_probes/$ARM
mkdir -p "$O" "$LOGS"
pybind_include=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src/include
cuda_root=/usr/local/cuda-12.6
read -r -a pyincs <<< "$(python3-config --includes)"
read -r -a pylink <<< "$(python3-config --embed --ldflags)"
common=(-std=c++17 -O1 -g -I"$pybind_include" -I"$cuda_root/include" "${pyincs[@]}" "${pylink[@]}" -L"$cuda_root/lib64" -Wl,-rpath,"$cuda_root/lib64" -lcudart)
ubsan=(-fsanitize=undefined -fno-sanitize-recover=undefined)
if [[ $ARM == after ]]; then
  python3 -I $W/artifacts/p1t_fix3/regen_lifecycle_slice.py $P/lifecycle_slice.cpp $F2 $ROOT $O/lifecycle_slice.cpp > "$LOGS/lifecycle_slice_regen_$ARM.txt"
  diff -u $P/lifecycle_slice.cpp $O/lifecycle_slice.cpp > "$LOGS/lifecycle_slice_regen_$ARM.diff"
  SLICE=$O/lifecycle_slice.cpp
else
  SLICE=$P/lifecycle_slice.cpp
fi
python3 -I $W/artifacts/p1t_fix3/make_pending_bytes_adapt.py $P/pending_bytes.cpp $O/pending_bytes_adapt.cpp > "$LOGS/pending_bytes_adapt_$ARM.diff"
b() { local name=$1; shift; timeout 180 g++ "$@" > "$O/${name}_build.log" 2>&1; echo "$ARM build $name exit=$?"; }
b core_cpu $P/core_cpu.cpp -I$ROOT/tensor_cuda/include "${common[@]}" "${ubsan[@]}" -o $O/core_cpu
b cpu_native $P/cpu_native.cpp -I$ROOT/tc_signal/src -I$ROOT/tc_signal/include "${common[@]}" -o $O/cpu_native
b native_harness $ROOT/tc_signal/tests/fix2_native_harness.cpp -I$ROOT/tc_signal/src -I$ROOT/tc_signal/include "${common[@]}" "${ubsan[@]}" -o $O/native_harness
b pending_bytes $P/pending_bytes.cpp -I$ROOT/tc_signal/src -I$ROOT/tc_signal/include "${common[@]}" "${ubsan[@]}" -o $O/pending_bytes
b pending_bytes_adapt $O/pending_bytes_adapt.cpp -I$ROOT/tc_signal/src -I$ROOT/tc_signal/include "${common[@]}" "${ubsan[@]}" -o $O/pending_bytes_adapt
b lifecycle -std=c++17 -shared -fPIC -O1 $SLICE -I$ROOT/tensor_cuda/include -I"$pybind_include" "${pyincs[@]}" -pthread -o "$O/_p1t_rev3_lifecycle$(python3-config --extension-suffix)"
python3 -I $W/artifacts/p1t_fix3/make_probe_copies.py $ARM $ROOT $O
sha256sum $O/core_cpu $O/cpu_native $O/native_harness $O/pending_bytes $O/pending_bytes_adapt $O/_p1t_rev3_lifecycle*.so $SLICE $O/*_paths.py > "$LOGS/probe_build_sha256_$ARM.txt"
