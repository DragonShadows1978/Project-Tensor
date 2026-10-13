# Lead-run probes (not executed on this CPU-only review seat)
Use the reviewed core/add-on builds on PYTHONPATH; do not substitute production baseline core. Set PYTHONDONTWRITEBYTECODE=1. Run sequentially in the fork under the lead's normal GPU lock, each under timeout (60 seconds unless named otherwise).

- addon_lifetime.py: F01, whether the consumed capsule's last side-stream torch allocation is reclaimed/reused before the legacy-stream read. Incorrect output proves UAF; non-reuse or a clean finite run is inconclusive for universal ownership.
- core_edges.py: F03 rank/element/byte count refusal and real scalar/zero-size/offset/singleton-stride/used-capsule/stream edges. Current review predicts shape guards fail.
- core_device_drain.py: F05, last-use event on device 0 at deleter entry while device 1 is current. Two devices required; exit 3 means inconclusive, not pass.
- core_shutdown.py plus release_worker.cpp: F04, ordinary versus shutdown foreign-thread release of a REAL imported Storage. Run timeout 30 python ... --normal, then timeout 30 python ... . Missing completion or timeout fails. Compile helper:
  mkdir -p /tmp/p1t_rev_native
  timeout 60 g++ -shared -fPIC -std=c++17 -pthread -I/usr/include/python3.12 -I/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src/include -Itensor_cuda/include artifacts/p1t_rev/probes/release_worker.cpp -o /tmp/p1t_rev_native/p1t_rev_release_worker.so
- numerics_and_streams.py: residual finite determinism, both cuBLAS configurations, L=1/eps=0, NaN/Inf/rint signed zero and live nonlegacy producer ordering; timeout 90. It does not establish full numerical acceptance. Lead must rerun original core baseline comparison and Voice's unchanged 236-case runner to establish those claims.

Prior art: DLPack/array API data interchange (dmlc, 2017+; Consortium, 2021+), CPython 3.12 GIL/finalization docs, NVIDIA CUDA stream semantics and cuBLAS 12.6 reproducibility, ONNX operators. Taken: lifecycle/ordering/semantic invariants. Ours: small bounded adversarial scenarios; no new algorithm claim.
