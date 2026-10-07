# APA-MF2.2 environment: engine + package selectable (default: this order's build + this tree);
# offline HF; NATIVE driver only. Refuses the retired 595.84 override (it breaks CUDA).
export W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2
export APA_MF2_2_ENGINE="${APA_MF2_2_ENGINE:-$W/artifacts/apa_mf2_2/build}"
export APA_MF2_2_PKG="${APA_MF2_2_PKG:-$W/tensor_cuda}"
export PYTHONPATH=$APA_MF2_2_ENGINE:$APA_MF2_2_PKG:$W
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
case ":${LD_LIBRARY_PATH:-}:${PATH}:${LD_PRELOAD:-}:" in *595.84*) echo "retired 595.84 override present; refusing" >&2; return 1 2>/dev/null || exit 1;; esac
