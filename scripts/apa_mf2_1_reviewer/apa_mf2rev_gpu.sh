#!/usr/bin/env bash
# Prior art: synchronous shell cell launchers (Project-Tensor 2026), taken;
# provenance-pinned probes and cell partition are this review's construction.
set -euo pipefail
# MF2.1 edit: root = the MF2.1 worktree; probe dir, receipts and logs renamed.
apa_review_root=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2
cd "$apa_review_root"
case "${1:-}" in
  leak|negative|identity|sliding|grapa|short|rule|census|extension|rule_definition) apa_review_cell="$1" ;;
  *) echo 'usage: bash scripts/apa_mf2rev_gpu.sh {leak|negative|identity|sliding|grapa|short|rule|census|extension}' >&2; exit 2 ;;
esac
case ":${LD_LIBRARY_PATH:-}:${PATH}:" in
  *nvidia-userspace-595.84*) echo 'RED: retired driver override present; refusing' >&2; exit 2 ;;
esac
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
export APA_MF2REV_ENGINE_DIR="${APA_MF2REV_ENGINE_DIR:-$apa_review_root/artifacts/apa_mf2_1/build}"
export PYTHONPATH="$apa_review_root/tensor_cuda:$APA_MF2REV_ENGINE_DIR:$apa_review_root/scripts/apa_mf2_1_reviewer:$apa_review_root"
apa_review_stamp="$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$apa_review_root/artifacts/apa_mf2_1/reviewer"
apa_review_log="$apa_review_root/logs/apa_mf2_1_reviewer_${apa_review_cell}_${apa_review_stamp}.log"
apa_review_receipt="$apa_review_root/artifacts/apa_mf2_1/reviewer/gpu_${apa_review_cell}_${apa_review_stamp}.json"
echo "Cell $apa_review_cell; log=$apa_review_log; receipt=$apa_review_receipt"
python3 "$apa_review_root/scripts/apa_mf2_1_reviewer/gpu_probe.py" "$apa_review_cell" --receipt "$apa_review_receipt" 2>&1 | tee "$apa_review_log"
