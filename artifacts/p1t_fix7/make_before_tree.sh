#!/usr/bin/env bash
# P1T-FIX7: the "before" probe tree. The round-7 probes find their package through parents[3] of their own path; the
# 429446d build lives in wt_fix6 (read-only), which has no artifacts/p1t_rev7/. So this builds /tmp/p1t_fix7_before_tree
# holding byte-identical COPIES of wt_fix6's sources, tests and prebuilt binaries (tensor_cuda/{src,include,tensor_cuda,
# tests}, tc_signal/{src,include,tc_signal,tests}) plus byte-identical copies of the reviewer probe files it needs
# (every artifacts/p1t_rev*/probes directory) from this worktree. (The first run copied only rounds 7, 6 and round 3's
# cpu_native.cpp; the round-5 probes the tests' fix5_regen.py reads were added to the existing tree by the same cp,
# ledger 11:3x.)
# Hashes of both .so files and of every probe file are written to <out>/before_tree_sha256.txt; the .so hashes must
# equal 429446d's (core e7d4a3a3..., tc_signal ef44b9c4...). Nothing is written into wt_fix6.
# Usage: make_before_tree.sh <out_dir>. No prior art claimed (copy plumbing).
set -euo pipefail
O=$1
W6=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix6
W7=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix7
B=/tmp/p1t_fix7_before_tree
mkdir -p "$O"
if [[ -e $B ]]; then echo "refusing: $B exists (no deletes; use a fresh path)"; exit 2; fi
mkdir -p $B/tensor_cuda $B/tc_signal
for d in src include tensor_cuda tests; do cp -a $W6/tensor_cuda/$d $B/tensor_cuda/; done
for d in src include tc_signal tests; do cp -a $W6/tc_signal/$d $B/tc_signal/; done
# every reviewer probe directory (rounds 1-7; the tests' fix5_regen.py reads artifacts/p1t_rev5/probes)
for r in $W7/artifacts/p1t_rev*/probes; do n=$(basename $(dirname $r)); mkdir -p $B/artifacts/$n; cp -a $r $B/artifacts/$n/; done
git -C $W6 rev-parse HEAD > "$O/before_tree_sha256.txt"
git -C $W6 status --short >> "$O/before_tree_sha256.txt"
sha256sum $W6/tensor_cuda/tensor_cuda/_tensor_cuda*.so $W6/tc_signal/tc_signal/_tc_signal*.so \
  $B/tensor_cuda/tensor_cuda/_tensor_cuda*.so $B/tc_signal/tc_signal/_tc_signal*.so >> "$O/before_tree_sha256.txt"
(cd $B && find artifacts -type f | sort | xargs sha256sum) >> "$O/before_tree_sha256.txt"
(cd $W7 && find artifacts/p1t_rev*/probes -type f | sort | xargs sha256sum) > "$O/before_tree_probe_origin_sha256.txt"
diff <(cd $B && find artifacts -type f | sort | xargs sha256sum) "$O/before_tree_probe_origin_sha256.txt" \
  && echo "probe copies byte-identical: yes" >> "$O/before_tree_sha256.txt"
echo "before tree ready: $B"
