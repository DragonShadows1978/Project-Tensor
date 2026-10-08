# APA-MF1 environment: this worktree's engine first; offline HF; NVML userspace.
export W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf1
export PYTHONPATH=$W/artifacts/apa_mf1/build:$W/tensor_cuda:$W
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=/mnt/ForgeRealm/engines/nvidia-userspace-595.84/root/usr/lib/x86_64-linux-gnu
export PATH=/mnt/ForgeRealm/engines/nvidia-userspace-595.84/root/usr/bin:$PATH
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
