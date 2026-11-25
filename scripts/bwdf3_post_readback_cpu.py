"""BWD-F3 L48 (a copy of scripts/bwdf2_post_readback_cpu.py, extended): drive the staged lines' readback (astype
float32, then numpy, as rb() in scripts/bwdf3_step_k2.py and scripts/bwdf3_p2_cases.py) and the post step's bitwise
comparison on host tensors of each dtype the rows read back, in each shape class the rows produce (empty, one-element,
[3, 1], [2, 3, 4], [4, 3], [2, 0, 3], and P2's [3, 2, 4], [0, 2, 2], [6, 4, 1]), plus P2's int64 argmax readback and the
per-row npz files the P2 step writes and the post step reads back (`<row id>|<key>`), on the CPU with Build F3.

  CUDA_VISIBLE_DEVICES= python3 -B -P scripts/bwdf3_post_readback_cpu.py <tensor_cuda package parent>
Prior art: none needed (a drive of this round's own code).
"""
import json
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tensor_cuda as tc  # noqa: E402
import bwdf3_gpu_post as P  # noqa: E402

rows = []
for dt in ("float32", "float16", "bfloat16"):
    for shape in ((0, 3), (1,), (3, 1), (2, 3, 4), (4, 3), (2, 0, 3), (3, 2, 4), (0, 2, 2), (6, 4, 1)):
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
# P2's argmax readback: an int64 tensor, read with .numpy() and cast to float32
ai = tc.tensor(np.array([[3, 0, 6], [1, 5, 2]], np.int64), device="cpu", dtype="int64")
ar = ai.numpy().astype(np.float32)
assert ar.dtype == np.float32 and ar.tolist() == [[3.0, 0.0, 6.0], [1.0, 5.0, 2.0]], ar
rows.append({"dtype": "int64", "shape": [2, 3], "readback": ar.tolist()})
# P2's per-row npz round trip (step: np.savez(<out>/<rid>.npz, **arrays); post: "<rid>|<key>")
with tempfile.TemporaryDirectory() as d:
    a = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)
    np.savez(pathlib.Path(d) / "e2-rope_apply-sin-T-D-1.npz", out=a, ref=a.copy())
    np.savez(pathlib.Path(d) / "e1-rope_apply-empty-cuda-short-sin.npz", shape=np.array([0, 2, 2], np.float32))
    got = {}
    for f in sorted(pathlib.Path(d).glob("*.npz")):
        got.update({f"{f.stem}|{k}": v for k, v in np.load(f).items()})
    assert P.bits_equal(got["e2-rope_apply-sin-T-D-1|out"], got["e2-rope_apply-sin-T-D-1|ref"]) is None
    assert got["e1-rope_apply-empty-cuda-short-sin|shape"].tolist() == [0.0, 2.0, 2.0]
    rows.append({"npz_round_trip_keys": sorted(got)})
print(json.dumps(rows))
print("READBACK-DRIVE PASS", len(rows))
