#!/usr/bin/env python3
"""ENG1-REV2 Q1: test_eng1.py's e3_round (NumPy float16 cast; bfloat16 high-word round-to-nearest-even bit hack)
against PyTorch's float32->float16/bfloat16 conversions, on 2^20 seeded random float32 bit patterns plus every
float32 that is an exact bfloat16/float16 tie in a band of binades, subnormals, overflow edges, infinities, NaN.
Prior art: IEEE 754-2019 roundTiesToEven; PyTorch c10::BFloat16 round_to_nearest_even and c10::Half conversion.
Seeded sampling (numpy default_rng(20261110)). Ours: the comparison only."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, torch
REV2 = Path(__file__).resolve().parent
ROOT = REV2.parents[2]
sys.path.insert(0, str(ROOT / "artifacts/eng1/tool/arms/candidate/tensor_cuda"))
spec = importlib.util.spec_from_file_location("eng1_test", ROOT / "tensor_cuda/tests/test_eng1.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
rng = np.random.default_rng(20261110)
bits = rng.integers(0, 2 ** 32, size=2 ** 20, dtype=np.uint64).astype(np.uint32)
ties_bf = (np.arange(0x3f000000, 0x41000000, 0x10000, dtype=np.uint32) | 0x8000)            # exact bf16 ties
ties_bf = np.concatenate([ties_bf, ties_bf + 0x10000 * 0, (np.arange(0, 0x00800000, 0x10000, dtype=np.uint32) | 0x8000)])
f16_one = np.float32(1.0).view(np.uint32)
ties_f16 = (np.arange(0, 2 ** 12, dtype=np.uint32) << 13 | 0x1000) + np.uint32(0x38000000)  # 2^-15..: f16 ties
special = np.array([0, 0x80000000, 0x7f800000, 0xff800000, 0x7f7fffff, 0x7f7f8000, 0x7f7f7fff, 0x00000001,
                    0x477fe000, 0x477ff000, 0x477fefff, 0x33800000, 0x33000000, 0x33000001, 0x7fc00000, 0x7f800001],
                   dtype=np.uint32)
res = {}
with np.errstate(over="ignore", invalid="ignore"):
    for name, b in (("random", bits), ("bf16_ties", ties_bf), ("f16_ties", ties_f16), ("special", special)):
        x = b.view(np.float32)
        for dt, td in (("bfloat16", torch.bfloat16), ("float16", torch.float16)):
            mine = torch.from_numpy(x.copy()).to(td).to(torch.float32).numpy().astype(np.float64)
            theirs = M.e3_round(x, dt)
            same = np.array_equal(mine, theirs, equal_nan=True)
            nz = (mine != theirs) & ~(np.isnan(mine) & np.isnan(theirs))
            res[f"{name}/{dt}"] = dict(n=int(x.size), equal=bool(same), mismatches=int(nz.sum()),
                                       first=[hex(int(v)) for v in b[nz][:5]])
(REV2 / "q1_rounding_instrument.json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res))
