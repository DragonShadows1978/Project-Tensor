"""ENG4 N1 recorder: what `tc._C.tensor` (the binding of C++ tensor_from_numpy, bindings.cpp) returns for every
listed NumPy dtype, on host ("cpu") operands, for one build in its own process (L25). The G1 test
(tensor_cuda/tests/test_eng4_matmul.py::test_g1_n1_listed_dtypes_keep_build_b_result) compares the build under test
with Build B's record.

Usage: CUDA_VISIBLE_DEVICES= PYTHONPATH=<package parent> python3 -B -P scripts/eng4_n1_record.py <label> <out.json>

Inputs are built by n1_inputs() from seed 4400 (recorded in the output); float rows include NaN, +-inf, -0.0 and a
subnormal so the bytes are nonzero and non-constant (L19). The record holds dtype, shape and the sha256 + hex of the
returned .numpy() bytes, and the tensor's own dtype string.

Prior art: no prior art known to me (a seeded record-and-compare of one entry point's results).
"""
import hashlib
import json
import os
import sys

import numpy as np

SEED = 4400


def n1_inputs():
    rng = np.random.default_rng(SEED)
    special = [np.nan, np.inf, -np.inf, -0.0, 0.0, 1.0, -2.5]
    f32 = np.concatenate([rng.standard_normal(17), special, [np.float32(1e-45)]]).astype(np.float32).reshape(5, 5)
    f16 = np.concatenate([rng.standard_normal(17), special, [6e-8]]).astype(np.float16).reshape(5, 5)
    i64 = rng.integers(-2 ** 62, 2 ** 62, size=(3, 4), dtype=np.int64)
    i64[0, 0], i64[0, 1] = np.iinfo(np.int64).min, np.iinfo(np.int64).max
    u8 = rng.integers(0, 256, size=(4, 6), dtype=np.uint8)
    u8[0, 0], u8[0, 1] = 0, 255
    bl = rng.integers(0, 2, size=(2, 7)).astype(np.bool_)
    bl[0, 0], bl[0, 1] = True, False
    return {"float32": f32, "float16": f16, "int64": i64, "uint8": u8, "bool": bl}


def record(tc):
    rows = {}
    for name, arr in n1_inputs().items():
        t = tc._C.tensor(arr, "cpu", False)
        out = t.numpy()
        raw = np.ascontiguousarray(out).tobytes()
        rows[name] = {"in_dtype": str(arr.dtype), "in_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                      "tensor_dtype": t.dtype, "shape": list(t.shape), "out_dtype": str(out.dtype),
                      "out_sha256": hashlib.sha256(raw).hexdigest(), "out_hex": raw.hex()}
    return rows


def main():
    label, out = sys.argv[1], sys.argv[2]
    import tensor_cuda as tc
    h = hashlib.sha256(open(tc._C.__file__, "rb").read()).hexdigest()
    rec = {"label": label, "seed": SEED, "module": tc._C.__file__, "module_sha256": h,
           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "rows": record(tc)}
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    for k, v in rec["rows"].items():
        print(label, k, v["tensor_dtype"], v["shape"], v["out_dtype"], v["out_sha256"][:16])


if __name__ == "__main__":
    main()
