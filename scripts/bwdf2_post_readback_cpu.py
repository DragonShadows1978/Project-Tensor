"""BWD-F2 L48: drive the staged K2 lines' readback (astype float32, then numpy, as rb() in scripts/bwdf2_step_k2.py)
and the post step's bitwise comparison on host tensors of each dtype the K2 rows read back, in each shape class the
rows produce (empty, one-element, [3, 1], [2, 3, 4], [4, 3]), on the CPU with Build F2's package.

  CUDA_VISIBLE_DEVICES= python3 -B -P scripts/bwdf2_post_readback_cpu.py <tensor_cuda package parent>
Prior art: none needed (a drive of this round's own code).
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tensor_cuda as tc  # noqa: E402
import bwdf2_gpu_post as P  # noqa: E402

rows = []
for dt in ("float32", "float16", "bfloat16"):
    for shape in ((0, 3), (1,), (3, 1), (2, 3, 4), (4, 3), (2, 0, 3)):
        v = (np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) * 0.5 + 0.25)
        t = tc.tensor(v, device="cpu", dtype=dt)
        r = np.ascontiguousarray(t.astype("float32").numpy())
        same = P.bits_equal(r, v.astype(np.float32))
        rows.append({"dtype": dt, "shape": list(shape), "readback_dtype": str(r.dtype), "readback_shape": list(r.shape),
                     "bits_equal": same})
        assert r.dtype == np.float32 and tuple(r.shape) == shape, rows[-1]
        if r.size > 1:
            assert same is None, rows[-1]       # exact values in every dtype (multiples of 0.25 below 2^7)
        else:
            assert same is not None and same.startswith("VACUOUS"), rows[-1]   # L19 refuses empty / constant
print(json.dumps(rows))
print("READBACK-DRIVE PASS", len(rows))
