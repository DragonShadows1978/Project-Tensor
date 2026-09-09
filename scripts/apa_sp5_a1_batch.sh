#!/usr/bin/env bash
# APA-SP5 amendment 1: run a LIST of per-window cells, each as its own leased
# worker. Prior art: SP3/SP4G lead_gpu.sh (this repo); Unix flock(2).
#
# Each cell re-takes the flock, runs under its own timeout(1) worker rail, and
# releases with the registered 30 s cooldown, so a shared-card operator never
# waits more than one cell (~47 s + 30 s). Never kills anything.
#
# usage: apa_sp5_a1_batch.sh <arm> <w_first> <w_last>
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
[[ "$ROOT" == /mnt/ForgeRealm/Project-Tensor-wt-apa-sp5 ]] || exit 64
cd "$ROOT"
arm=${1:?arm}; first=${2:?first}; last=${3:?last}
fail=0
for ((w=first; w<=last; w++)); do
  printf -v ww '%02d' "$w"
  if [[ -e "artifacts/apa_sp5/windows_a1/ppl_${arm}_W1024_w${ww}.json" ]]; then
    echo "SKIP ppl_${arm}_W1024_w${ww} (exists)"; continue
  fi
  bash scripts/apa_sp5_lead_gpu.sh scripts/apa_sp5_a1_window.py \
      --arm "$arm" --window "$w" >/dev/null 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then echo "CELL_FAIL arm=$arm w=$w rc=$rc"; fail=1; else
    ppl=$(python3 -c "import json;print(round(json.load(open('artifacts/apa_sp5/windows_a1/ppl_${arm}_W1024_w${ww}.json'))['ppl'],3))" 2>/dev/null)
    echo "OK ppl_${arm}_W1024_w${ww} ppl=$ppl"
  fi
done
exit $fail
