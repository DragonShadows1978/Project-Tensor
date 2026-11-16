#!/bin/bash
# Prior art: ENG3-F1 forge launcher (2026); ours: complete F2 environment.
set -eu
cd /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng3
export TMPDIR=/tmp/eng3f2_scratch PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
unset LD_PRELOAD ENG3_FILES ENG3_K ENG3_RECORD ENG3_OWN_TESTS ENG3F1_PACKAGE_PARENT ENG3F2_MODULE
export FORGE_GPU_DRY_RUN="${FORGE_GPU_DRY_RUN:-0}"
CUDA_VISIBLE_DEVICES= timeout 120s nice -n 19 python3 -B -P scripts/eng3f2_stage.py prefix g3-g2
unset CUDA_VISIBLE_DEVICES
set +e
forge-gpu-run --min 30 --max 1800 --label eng3f2-g3-g2 -- timeout 1750s nice -n 19 python3 -B -P scripts/eng3f2_stage.py run g3-g2 > artifacts/eng3f2/staged/g3-g2/GPU.log 2>&1
forge_exit=$?
set -e
timeout 30s python3 -B -P scripts/eng3f2_stage.py finish g3-g2 "$forge_exit"
