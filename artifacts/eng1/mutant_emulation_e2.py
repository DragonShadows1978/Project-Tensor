#!/usr/bin/env python3
"""ENG1: NumPy emulation of the E2 mutants M05-M07 (GPU-only killers) scored with test_eng1.py's references and
tolerances. Prediction evidence only (class: reasoning + emulation); the kills are the lead's GPU lines.
M05: half path in the input dtype (pow, scale, multiply each rounded to the dtype) == the baseline arithmetic.
M06: float32 chain with exponent p instead of p - 1, one rounding.  M07: p * x^(p-1) rounded to the dtype before the
multiply by g. Usage: mutant_emulation_e2.py OUT.json
"""
import importlib.util, json, os, sys
from pathlib import Path
import numpy as np
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tensor_cuda"))
spec = importlib.util.spec_from_file_location("t_eng1", ROOT / "tensor_cuda/tests/test_eng1.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
def rnd(a, dtype):
    a = np.asarray(a, dtype=np.float32)
    if dtype == "float16":
        return a.astype(np.float16).astype(np.float32)
    b = a.view(np.uint32).astype(np.uint64)
    return ((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)
out = {}
for dtype in ("float16", "bfloat16"):
    for p in T.E2_P:
        x_all = rnd(np.array(T.E2_X), dtype).astype(np.float64)
        k = T.e2_scale_exponents(x_all, p)
        keep = k <= 24 if dtype == "float16" else np.ones(k.shape, dtype=bool)
        x, w = x_all[keep], np.power(2.0, -k[keep])
        ref = T.e2_reference_grad(x, w, p)
        x32, w32, p32 = x.astype(np.float32), rnd(w, dtype), np.float32(p)
        with np.errstate(over="ignore", invalid="ignore"):
            arms = {"M05": rnd(w32 * rnd(rnd(np.power(x32, np.float32(p - 1.0)), dtype) * p32, dtype), dtype),
                    "M06": rnd(w32 * (np.power(x32, p32) * p32), dtype),
                    "M07": rnd(w32 * rnd(np.power(x32, np.float32(p - 1.0)) * p32, dtype), dtype)}
        for m, got in arms.items():
            got = got.astype(np.float64)
            ok = bool(np.all(np.isfinite(got))) and bool(np.allclose(got, ref, rtol=T.E2_RTOL[dtype], atol=T.E2_ATOL[dtype]))
            out.setdefault(m, {})[f"test_e2_pow_grad[{dtype}-p{p:+g}]"] = "pass" if ok else "fail"
summary = {m: sorted(k for k, v in rows.items() if v == "fail") for m, rows in out.items()}
Path(sys.argv[1]).write_text(json.dumps(dict(per_id=out, predicted_failing=summary), indent=1) + "\n")
print(json.dumps({m: len(v) for m, v in summary.items()}), json.dumps(summary)[:900])
