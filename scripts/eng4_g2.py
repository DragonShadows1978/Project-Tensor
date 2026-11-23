"""ENG4 G2 witness: the ENG-MM1 call (A [4, 2, 3] x B [2, 3, 4], float32, CUDA), run under compute-sanitizer
memcheck by scripts/eng4_gpu_line.py. This process only makes the call and reports what happened.

Usage: PYTHONPATH=<package parent> python3 -B -P scripts/eng4_g2.py <out.json> [noop|pooled]
  noop   : the instrument preflight: one small CUDA tensor round trip and nothing else (memcheck must report 0 errors)
  pooled : the same ENG-MM1 call with tc.set_alloc_pooling(True) (cudaMallocAsync pool; recorded, not gated)
  (none) : the gated witness with the engine's default allocator (raw cudaMalloc; kernels.cu g_pool_transients=false)

Operands are seeded (seed 4401), nonzero. The forward's result (Build B) or its refusal (Build E) is recorded with a
host copy of the output and a synchronize, so any device fault is reported inside this process.

Prior art: compute-sanitizer memcheck (NVIDIA CUDA toolkit docs) for out-of-bounds global reads. Ours: the witness.
"""
import hashlib
import json
import sys

import numpy as np


def main():
    out, mode = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "witness")
    import tensor_cuda as tc
    rec = {"mode": mode, "module": tc._C.__file__,
           "module_sha256": hashlib.sha256(open(tc._C.__file__, "rb").read()).hexdigest(), "seed": 4401}
    rng = np.random.default_rng(4401)
    if mode == "noop":
        x = tc.tensor(np.arange(4, dtype=np.float32))
        rec["noop_roundtrip"] = x.numpy().tolist()
    else:
        if mode == "pooled":
            tc.set_alloc_pooling(True)
        a_np = rng.standard_normal((4, 2, 3)).astype(np.float32)
        b_np = rng.standard_normal((2, 3, 4)).astype(np.float32)
        a, b = tc.tensor(a_np), tc.tensor(b_np)
        rec["a_shape"], rec["b_shape"] = list(a.shape), list(b.shape)
        try:
            r = tc.matmul(a, b)
            host = r.numpy()
            tc.synchronize()
            rec["returned"] = {"shape": list(host.shape), "values": host.tolist()}
            # what a correct pairing would give for the first two batches (B has only two matrices)
            rec["matches_numpy_on_batches_0_1"] = bool(np.allclose(host[:2], a_np[:2] @ b_np, atol=1e-5))
        except Exception as e:  # noqa: BLE001 (recorded)
            rec["raised"] = {"type": type(e).__name__, "message": str(e)}
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "returned"}))


if __name__ == "__main__":
    main()
