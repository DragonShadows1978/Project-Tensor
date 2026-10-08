# APA-MF2 environment: this worktree's engine first; offline HF; NATIVE driver.
# Deliberately NOT copied from apa_mf1_env.sh: that file pins LD_LIBRARY_PATH/PATH
# to /mnt/ForgeRealm/engines/nvidia-userspace-595.84, which the order forbids
# (it breaks CUDA since the 10-07 reboot). Native driver 595.91.07 only.
export W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2
export PYTHONPATH=$W/artifacts/apa_mf2/build:$W/tensor_cuda:$W
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
case ":${LD_LIBRARY_PATH:-}:${PATH}:" in *nvidia-userspace-595.84*) echo "retired 595.84 override present; refusing" >&2; return 1 2>/dev/null || exit 1;; esac
