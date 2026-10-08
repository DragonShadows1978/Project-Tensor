# APA-MF2.1 environment: engine dir selectable (default: this order's build); offline HF;
# NATIVE driver only. Refuses the retired 595.84 override (it breaks CUDA since 10-07).
export W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2
export APA_MF2_1_ENGINE="${APA_MF2_1_ENGINE:-$W/artifacts/apa_mf2_1/build}"
export APA_MF2_1_PKG="${APA_MF2_1_PKG:-$W/tensor_cuda}"
export PYTHONPATH=$APA_MF2_1_ENGINE:$APA_MF2_1_PKG:$W
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
case ":${LD_LIBRARY_PATH:-}:${PATH}:" in *nvidia-userspace-595.84*) echo "retired 595.84 override present; refusing" >&2; return 1 2>/dev/null || exit 1;; esac
