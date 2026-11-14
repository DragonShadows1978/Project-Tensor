#!/usr/bin/env bash
# Lead lane chain (2026-11-14): P1-S3's two tie lines, then ENG1 G5 (eng1_g5.sh), back to back so no other job
# queues between them. Each wrapper logs its own rcs. Prior art: none needed (sequencing only).
set -u
bash /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/lead/run_p1_s3_gpu_ties.sh /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod/lead/s3_gpu_1114b
echo "$(date -Is) ties wrapper rc=$?" >> /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng1/artifacts/eng1/lead/chain_1114.log
bash /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng1/artifacts/eng1/lead/eng1_g5.sh "$(date -Is -d '+8 hours')" /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng1/artifacts/eng1/lead/g5_1114
echo "$(date -Is) eng1_g5 wrapper rc=$?" >> /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng1/artifacts/eng1/lead/chain_1114.log
