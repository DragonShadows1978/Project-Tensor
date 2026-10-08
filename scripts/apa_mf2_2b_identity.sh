#!/usr/bin/env bash
# APA-MF2.2b G3(ii): re-review probe identity_values set, after1 vs after2, separate processes
# (scripts/apa_mf2_2b_probe copy; real bf16 leaves + no bf16 pool, as adjudicated in MF2.2).
set -uo pipefail
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2
case ":${LD_LIBRARY_PATH:-}:${PATH}:${LD_PRELOAD:-}:" in *595.84*) echo "595.84 override present; refusing" >&2; exit 2;; esac
E=$W/artifacts/apa_mf2_2a/build
export APA_MF21REV_BEFORE_ENGINE=$W/artifacts/apa_mf2/build APA_MF21REV_PRODUCTION_ENGINE=$W/artifacts/apa_mf2_2/build_fd32b6d APA_MF21REV_AFTER_ENGINE=$W/artifacts/apa_mf2_1/build
export APA_MF2_2B_ENGINES="{\"after1\":\"$E\",\"after2\":\"$E\"}"
export APA_MF2_2_PKGS="{\"after1\":\"$W/artifacts/apa_mf2_2b/before/pkg\",\"after2\":\"$W/tensor_cuda\"}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 TC_ATTN_QTILE=0
export APA_MF2_2_BF16_LEAF=1 APA_MF2_2_SKIP_BF16_POOL=1 APA_MF2_2_IDENTITY_REVS=after1,after2
STAMP=$(date -u +%Y%m%dT%H%M%SZ)_mf2_2b_$$
export APA_MF21REV_REV=after2 APA_MF21REV_ENGINE=$E APA_MF21REV_PKG=$W/tensor_cuda APA_MF21REV_RUN=$STAMP PYTHONPATH=$E:$W/tensor_cuda
LOG=$W/logs/apa_mf2_2b_probe_identity_$STAMP.log
flock -w 900 /tmp/forge-gpu.lock timeout 600 python3 $W/scripts/apa_mf2_2b_probe/gpu_probe.py identity > "$LOG" 2>&1
echo "exit=$? log=$LOG"
