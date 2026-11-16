#!/bin/bash
# Prior art: inherited forge-gpu-run contract (2026); ours: pre-lease CPU prefix and pin.
set -eu
export TMPDIR=/tmp/eng3f1_tmp PYTHONDONTWRITEBYTECODE=1
CUDA_VISIBLE_DEVICES= timeout 120s nice -n 19 python3 -B -P artifacts/eng3f1/tools/eng3f1_stage.py prefix g2
set +e
forge-gpu-run --min 30 --max 600 --label eng3f1-g2 -- timeout 570s nice -n 19 python3 -B -P artifacts/eng3f1/tools/eng3f1_stage.py run g2 > artifacts/eng3f1/staged/g2/GPU.log 2>&1
forge_exit=$?
set -e
python3 -B -P artifacts/eng3f1/tools/eng3f1_stage.py finish g2 "$forge_exit"
