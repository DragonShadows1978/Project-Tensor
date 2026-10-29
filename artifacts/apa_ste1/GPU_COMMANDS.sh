#!/usr/bin/env bash
# Lead only. Exact staged commands; never run by CPU seat.
set -euo pipefail
cd /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_ste1
export TMPDIR="$PWD/artifacts/apa_ste1/scratch"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export FORGE_GPU_RUN_LOG="$PWD/artifacts/apa_ste1/lead/governor.tsv"
finish_receipt() {
  local command_rc=$?
  trap - EXIT
  set +e
  CUDA_VISIBLE_DEVICES= timeout --foreground 120s python3 -B artifacts/apa_ste1/gpu_gate.py verify
  local verification_rc=$?
  if [[ "$command_rc" -ne 0 ]]; then
    exit "$command_rc"
  fi
  exit "$verification_rc"
}
trap finish_receipt EXIT
forge-gpu-run --min 300 --max 600 --label ste1-base-small -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which base --switch 0 --suite small --label base-small
forge-gpu-run --min 300 --max 600 --label ste1-arm-off-small -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which arm --switch 0 --suite small --label arm-off-small
CUDA_VISIBLE_DEVICES= timeout --foreground 60s python3 -B artifacts/apa_ste1/gpu_gate.py calibrate
forge-gpu-run --min 300 --max 600 --label ste1-arm-on-small -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which arm --switch 1 --suite small --label arm-on-small
forge-gpu-run --min 300 --max 600 --label ste1-arm-repeat-small -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which arm --switch 1 --suite small --label arm-repeat-small
forge-gpu-run --min 300 --max 600 --label ste1-arm-off-all-small -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which arm --switch 0 --suite small --label arm-off-all-small --all
forge-gpu-run --min 300 --max 600 --label ste1-arm-on-all-small -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which arm --switch 1 --suite small --label arm-on-all-small --all
forge-gpu-run --min 300 --max 600 --label ste1-base-grapa -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which base --switch 0 --suite grapa --label base-grapa
forge-gpu-run --min 300 --max 600 --label ste1-arm-off-grapa -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which arm --switch 0 --suite grapa --label arm-off-grapa
forge-gpu-run --min 300 --max 600 --label ste1-arm-on-grapa -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which arm --switch 1 --suite grapa --label arm-on-grapa
forge-gpu-run --min 300 --max 600 --label ste1-arm-repeat-grapa -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which arm --switch 1 --suite grapa --label arm-repeat-grapa
forge-gpu-run --min 300 --max 600 --label ste1-arm-off-all-grapa -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which arm --switch 0 --suite grapa --label arm-off-all-grapa --all
forge-gpu-run --min 300 --max 600 --label ste1-arm-on-all-grapa -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py capture --which arm --switch 1 --suite grapa --label arm-on-all-grapa --all
forge-gpu-run --min 300 --max 600 --label ste1-timing-0 -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py profile --switch 0
forge-gpu-run --min 300 --max 600 --label ste1-timing-1 -- env CUDA_VISIBLE_DEVICES=0 APA_STE1_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste1/gpu_gate.py profile --switch 1
