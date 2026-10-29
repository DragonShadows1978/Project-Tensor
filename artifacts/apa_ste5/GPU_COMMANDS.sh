#!/usr/bin/env bash
# LEAD ONLY: do not run on the CPU seat.
# Prior art: STE1/4 governed captures (2026); GNU timeout (2008+, unverified - lead to check).
set -euo pipefail
cd /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_ste1
export TMPDIR="$PWD/artifacts/apa_ste5/scratch"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p artifacts/apa_ste5/lead
export FORGE_GPU_RUN_LOG="$PWD/artifacts/apa_ste5/lead/governor.tsv"
CUDA_VISIBLE_DEVICES= timeout --foreground 120s python3 -B artifacts/apa_ste5/stage5.py --check
# 1. Base only, then G-G1 premise. STOP on any failure.
forge-gpu-run --min 300 --max 600 --label ste5v1-base-small -- env -u TC_APA_DK_STE CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config base --suite small
forge-gpu-run --min 300 --max 600 --label ste5v1-base-grapa -- env -u TC_APA_DK_STE CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config base --suite grapa
CUDA_VISIBLE_DEVICES= timeout --foreground 300s python3 -B artifacts/apa_ste5/verify5.py --premise --output artifacts/apa_ste5/lead/PREMISE.json
# 2. Arm configurations only after premise PASS.
forge-gpu-run --min 300 --max 600 --label ste5v1-off-small -- env CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config off --suite small
forge-gpu-run --min 300 --max 600 --label ste5v1-off-grapa -- env CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config off --suite grapa
forge-gpu-run --min 300 --max 600 --label ste5v1-unset-small -- env -u TC_APA_DK_STE CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config unset --suite small
forge-gpu-run --min 300 --max 600 --label ste5v1-unset-grapa -- env -u TC_APA_DK_STE CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config unset --suite grapa
forge-gpu-run --min 300 --max 600 --label ste5v1-on-small -- env CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config on --suite small
forge-gpu-run --min 300 --max 600 --label ste5v1-on-grapa -- env CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config on --suite grapa
forge-gpu-run --min 300 --max 600 --label ste5v1-off-all-small -- env CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config off-all --suite small
forge-gpu-run --min 300 --max 600 --label ste5v1-off-all-grapa -- env CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config off-all --suite grapa
forge-gpu-run --min 300 --max 600 --label ste5v1-on-all-small -- env CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config on-all --suite small
forge-gpu-run --min 300 --max 600 --label ste5v1-on-all-grapa -- env CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste5/capture5.py batch --config on-all --suite grapa
# 3. Timing: one h forward + g1 backward pair; warmup, then 2 samples.
forge-gpu-run --min 300 --max 600 --label ste5v1-timing-0 -- env CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=0 timeout --foreground 300s python3 -B artifacts/apa_ste5/timing5.py --switch 0
forge-gpu-run --min 300 --max 600 --label ste5v1-timing-1 -- env CUDA_VISIBLE_DEVICES=0 APA_STE5_LEAD_GPU=1 TC_DETERMINISTIC=1 TC_DET_EMBED_BWD=1 TC_APA_DK_STE=1 timeout --foreground 300s python3 -B artifacts/apa_ste5/timing5.py --switch 1
# 4. CPU verify recomputes gates, timing difference, then refreshes bundle only on PASS.
CUDA_VISIBLE_DEVICES= timeout --foreground 1800s python3 -B artifacts/apa_ste5/verify5.py --output artifacts/apa_ste5/lead/VERIFY5.json
