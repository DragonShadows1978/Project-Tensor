#!/usr/bin/env bash
# P1T-M3 G3: build one mutant copy (/tmp/p1t_m3_mut_<id>/tc_signal, made by make_mutants.py) and run G1 (own tests) and
# G2 (Voice's runner, --items T4 and --items T1,T2,T3,T4) against it. Receipts go to
# <worktree>/artifacts/p1t_m3/mutants/<id>/. Every GPU process runs under the shared GPU lock. Nothing is deleted.
# Usage: run_mutant.sh <id>
set -uo pipefail
ID="$1"
WT=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_m3
MROOT=/tmp/p1t_m3_mut_${ID}
OUT=$WT/artifacts/p1t_m3/mutants/${ID}
PP=$WT/tensor_cuda/tensor_cuda:$WT/tensor_cuda:$MROOT/tc_signal
VOICE=/mnt/ForgeRealm/Project-Voice
mkdir -p "$OUT"

echo "== build $(date)"
( cd "$MROOT/tc_signal" && PYBIND11_SOURCE_DIR=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src ./build.sh 89 ) \
  > "$OUT/build.log" 2>&1
echo "build exit=$?"
sha256sum "$MROOT"/tc_signal/tc_signal/_tc_signal*.so | tee "$OUT/so_sha256.txt"

echo "== which module $(date)"
flock -w 900 /tmp/forge-gpu.lock timeout 600 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PP" python3 -c \
  "import tc_signal; print('tc_signal from', tc_signal.__file__)" | tee "$OUT/which_module.txt"

echo "== G1 $(date)"
( cd "$MROOT" && flock -w 900 /tmp/forge-gpu.lock timeout 600 env PYTHONDONTWRITEBYTECODE=1 \
    TC_SIGNAL_TEST_RECEIPT="$OUT/g1_receipt.json" PYTHONPATH="$PP" \
    python3 -m pytest tc_signal/tests -q -p no:cacheprovider -rA ) > "$OUT/g1.log" 2>&1
echo "G1 pytest exit=$?"
tail -1 "$OUT/g1.log"

for ITEMS in T4 T1,T2,T3,T4; do
  TAG=$( [[ "$ITEMS" == T4 ]] && echo T4 || echo T1toT4 )
  echo "== G2 $ITEMS $(date)"
  ( cd "$VOICE" && flock -w 900 /tmp/forge-gpu.lock timeout 900 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PP" \
      python3 engine/tensor_requests/p1t_tests/run_p1t.py --adapter "$WT/artifacts/p1t_m3/tensorcuda_adapter.py" \
      --items "$ITEMS" --fixtures /mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures \
      --tolerances engine/tensor_requests/p1t_tests/tolerances_r2.json --out "$OUT/g2_${TAG}.json" ) \
    > "$OUT/g2_${TAG}.log" 2>&1
  echo "G2 $ITEMS runner exit=$?"
done
echo "== done $(date)"
