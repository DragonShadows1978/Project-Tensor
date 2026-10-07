"""APA-MF2 item-2 audit: execute the ADDITIVE consumer pattern of _causal_mask.

GRM core/gemma4_tc.py:740 (grad-enabled / no causal_softmax branch) and GRAPA
grapa/attention.py:82-84 ADD the bias to the scores and softmax:
    p = (sc + F._causal_mask(L, S, dev, sc.dtype)).softmax(-1)
This runs that exact expression on real Qwen3-1.7B L16 scores (4 query heads,
(L,S) in {(64,8192),(512,8192),(512,512)}, fp32/bf16/fp16), grad on (d sc of
sum(p * w), w random), with the before mask (MF1/fd32b6d functional.py, -1e4) and
the after mask (-inf): p and d sc must be bytewise equal and finite.
Prior art: none beyond the order's byte-identity method. Script: mine.
"""
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import apa_mf2_bk4emu as E  # noqa: E402
import tensor_cuda as tc  # noqa: E402
from tensor_cuda import functional as F  # noqa: E402

CAP = "/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_spq3/artifacts/apa_spq3/captures/Qwen3-1.7B"
spec = importlib.util.spec_from_file_location(
    "tensor_cuda._mf2_add_bef", os.path.join(ROOT, "artifacts/apa_mf2/before/mf1_AB/tensor_cuda__tensor_cuda__functional.py"))
BF = importlib.util.module_from_spec(spec); spec.loader.exec_module(BF)


def main():
    out = sys.argv[1]
    q = E.bf16_bits_to_f32(np.load(os.path.join(CAP, "L16_q.npy")))
    k = E.bf16_bits_to_f32(np.load(os.path.join(CAP, "L16_k.npy")))
    rows = []
    for L, S in ((64, 8192), (512, 8192), (512, 512)):
        qa = q[:4, S - L:S][None]; ka = k[np.arange(4) // 2, :S][None]
        w = np.random.default_rng(L + S).normal(0, 1, (1, 4, L, S)).astype(np.float32)
        for dt in ("float32", "bfloat16", "float16"):
            sc0 = tc.matmul(tc.tensor(qa, dtype=dt), tc.tensor(ka, dtype=dt), alpha=128 ** -0.5, trans_b=True)
            res = {}
            for arm, mask_fn in (("before", BF._causal_mask), ("after", F._causal_mask)):
                leaf = tc.tensor(sc0.float().numpy(), requires_grad=True)
                sc = leaf if dt == "float32" else leaf.astype(dt)
                p = (sc + mask_fn(L, S, "cuda", sc.dtype)).softmax(-1)
                (p.float() * tc.tensor(w)).sum().backward()
                res[arm] = [p.float().numpy(), leaf.grad.numpy()]
            eq = all(a.tobytes() == b.tobytes() for a, b in zip(res["before"], res["after"]))
            fin = all(np.isfinite(a).all() for a in res["after"])
            rows.append({"L": L, "S": S, "dtype": dt, "bytes_equal_p_and_dsc": eq, "after_finite": fin})
            print(json.dumps(rows[-1]), flush=True)
    r = {"pattern": "(sc + F._causal_mask(L,S,dev,sc.dtype)).softmax(-1), grad on", "rows": rows,
         "all_equal": all(x["bytes_equal_p_and_dsc"] for x in rows), "all_finite": all(x["after_finite"] for x in rows)}
    with open(out, "x") as f:
        json.dump(r, f, indent=1)
    print("all_equal", r["all_equal"], "all_finite", r["all_finite"], "WROTE", out)


if __name__ == "__main__":
    main()
