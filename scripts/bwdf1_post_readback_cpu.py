"""BWD-F1 L48: drive the staged lines' readback (bwdf1_gpu_common.readback: astype float32, then numpy) and the case
verdict on host tensors of each dtype, empty, one-element and [4, 1] shapes, on the CPU with Build E's package.

  CUDA_VISIBLE_DEVICES= python3 -B -P scripts/bwdf1_post_readback_cpu.py <tensor_cuda package parent>

Prior art: none needed (a drive of this round's own code).
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tensor_cuda as tc  # noqa: E402
import bwdf1_gpu_common as G  # noqa: E402

rows = []
for dt in G.DTYPES:
    for shape in ((4, 0), (1,), (4, 1), (0,)):
        t = tc.tensor(np.zeros(shape, np.float32), device="cpu", dtype=dt)
        r = G.readback(t)
        nz = int(np.count_nonzero(np.ascontiguousarray(r).view(np.uint32)))
        row = {"result": "OK", "out_shape": list(r.shape), "want_shape": list(shape),
               "grads": {"a": {"shape": list(r.shape), "want": list(shape), "nonzero_bits": nz}}}
        rows.append({"dtype": dt, "shape": list(shape), "readback_dtype": str(r.dtype),
                     "readback_shape": list(r.shape), "verdict": G.case_verdict(row)})
        assert r.dtype == np.float32 and tuple(r.shape) == shape and rows[-1]["verdict"] == "PASS", rows[-1]
    t = tc.tensor(np.array([[-0.0], [0.0], [0.0], [0.0]], np.float32), device="cpu", dtype=dt)
    r = G.readback(t)
    v = G.case_verdict({"result": "OK", "out_shape": [4, 1], "want_shape": [4, 1], "grads": {"b": {
        "shape": [4, 1], "want": [4, 1], "nonzero_bits": int(np.count_nonzero(r.view(np.uint32)))}}})
    rows.append({"dtype": dt, "shape": [4, 1], "case": "-0.0 survives the widening", "verdict": v})
    assert v.startswith("FAIL"), v
print(json.dumps(rows))
print("READBACK-DRIVE PASS", len(rows))
