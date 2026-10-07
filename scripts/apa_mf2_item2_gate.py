"""APA-MF2 G3 (item 2): registered leak sweep + real-capture byte identity, before vs after.

usage: apa_mf2_item2_gate.py <out.json>
Leak sweep (registration census.leak): one masked key j=7 with score M (scale 1.0,
q=e0, K7=M e0), V nonzero only on key 7, shapes L3 S9 and L=S=9; paths SDPA grad-off
(causal_softmax), SDPA grad-on (composed softmax), apa_quant_attention refine 1.0 and
0.15. Arms: prod (fd32b6d quant.py + functional.py), mf1 (MF1 A+B), after (tree). All
before arms are loaded from files in artifacts/apa_mf2/before -- never the new code.
Criterion (after): rows that cannot see key 7 are exactly 0.0; all rows finite.
Cells whose refine-0.15 bulk score is non-finite (the quantizer computes |k|^2 in
fp32; |K| > ~1.8e19, or the reconstructed fp16 key overflows) are flagged
"nonfinite_bulk" (outside the order's "every finite score").
Byte identity on real captures: Qwen3-1.7B L16 and Qwen3-4B L21, 4 query heads
(KV expanded), (L,S) in {(64,8192),(512,8192),(512,512)}, fp32/bf16/fp16; SDPA
grad-off, SDPA grad-on (out + dq/dk/dv), apa_quant_attention p=1.0 (before = MF1
quant.py) and p=0.15 (before = item-1-only snapshot variants/quant_item1.py), both
with the before functional.py. Prior art: FlashAttention-2 (Dao 2023); PyTorch
SDPA boolean attn_mask. Script: mine.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import apa_mf2_bk4emu as E  # noqa: E402
import tensor_cuda as tc  # noqa: E402
from tensor_cuda import functional as F  # noqa: E402
from tensor_cuda import quant as Q  # noqa: E402

assert tc._C.__file__.startswith(ROOT) and Q.__file__.startswith(ROOT)
A = os.path.join(ROOT, "artifacts/apa_mf2")
CAP = "/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_spq3/artifacts/apa_spq3/captures"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MODS = {
    "prod_functional": load("tensor_cuda._mf2_prod_functional", f"{A}/before/prod_fd32b6d/tensor_cuda__tensor_cuda__functional.py"),
    "prod_quant": load("tensor_cuda._mf2_prod_quant", f"{A}/before/prod_fd32b6d/tensor_cuda__tensor_cuda__quant.py"),
    "mf1_functional": load("tensor_cuda._mf2_mf1_functional", f"{A}/before/mf1_AB/tensor_cuda__tensor_cuda__functional.py"),
    "mf1_quant": load("tensor_cuda._mf2_mf1_quant", f"{A}/before/mf1_AB/tensor_cuda__tensor_cuda__quant.py"),
    "item1_quant": load("tensor_cuda._mf2_item1_quant", f"{A}/variants/quant_item1.py"),
}


@contextlib.contextmanager
def functional(name):
    """name in {after, prod, mf1}: which functional.py serves F._causal_mask / SDPA."""
    if name == "after":
        yield
        return
    m = MODS[f"{name}_functional"]
    saved = (F._causal_mask, F.scaled_dot_product_attention)
    F._causal_mask, F.scaled_dot_product_attention = m._causal_mask, m.scaled_dot_product_attention
    try:
        yield
    finally:
        F._causal_mask, F.scaled_dot_product_attention = saved


def run_path(path, quant_fn, ts, scale):
    if path == "sdpa_nograd":
        with tc.no_grad():
            return F.scaled_dot_product_attention(*ts, is_causal=True, scale=scale), None
    if path == "sdpa_grad":
        leaves = [tc.tensor(t.float().numpy(), requires_grad=True) for t in ts]
        xs = [x if ts[0].dtype == "float32" else x.astype(ts[0].dtype) for x in leaves]
        o = F.scaled_dot_product_attention(*xs, is_causal=True, scale=scale)
        o.sum().backward()
        return o, [x.grad for x in leaves]
    p = 1.0 if path == "apa_p1" else 0.15
    with tc.no_grad():
        return quant_fn(*ts, bulk_bits=8, refine_percentile=p, is_causal=True, scale=scale), None


SWEEP = {"float32": (1e3, 9.9e3, 1e4, 1.0001e4, 2e4, 1e5, 1e6, 1e10, 1e20, 1e30),
         "bfloat16": (1e3, 1e4, 1e5, 1e10, 1e30, 3.3e38),
         "float16": (1e3, 1e4, 2e4, 6e4, 65504.0)}
QUANT = {"after": lambda: Q.apa_quant_attention, "mf1": lambda: MODS["mf1_quant"].apa_quant_attention,
         "prod": lambda: MODS["prod_quant"].apa_quant_attention}


def leak_sweep():
    rows = []
    D, key = 64, 7
    for dt, Ms in SWEEP.items():
        for M in Ms:
            for L, S in ((3, 9), (9, 9)):
                q = np.zeros((1, 1, L, D), np.float32); q[..., 0] = 1.0
                k = np.zeros((1, 1, S, D), np.float32); k[0, 0, key, 0] = M
                v = np.zeros((1, 1, S, D), np.float32); v[0, 0, key, :] = 1.0
                ts = [tc.tensor(a, dtype=dt) for a in (q, k, v)]
                blind = np.arange(L) < key - (S - L)
                R, CB, BND = Q._tables(D, 8, 1, True, "cuda")
                kq = Q._quantize_keys(ts[1], R, CB, BND).float().numpy()
                nonfinite_bulk = not np.isfinite(kq).all() or not np.isfinite(
                    np.einsum("bhld,bhsd->bhls", ts[0].float().numpy().astype(np.float64), kq.astype(np.float64))).all()
                for path in ("sdpa_nograd", "sdpa_grad", "apa_p1", "apa_p015"):
                    for arm in ("prod", "mf1", "after"):
                        with functional(arm):
                            o, _ = run_path(path, QUANT[arm](), ts, 1.0)
                        o = o.float().numpy()[0, 0]
                        rows.append({"dtype": dt, "M": M, "L": L, "S": S, "path": path, "arm": arm,
                                     "blind_rows_exact_zero": bool(np.all(o[blind] == 0.0)),
                                     "blind_rows_maxabs": float(np.nanmax(np.abs(o[blind]))) if np.isfinite(o[blind]).all() else float("nan"),
                                     "all_finite": bool(np.isfinite(o).all()),
                                     "nonfinite_bulk": bool(nonfinite_bulk and path == "apa_p015")})
    return rows


def reviewer_n1():
    q = np.zeros((1, 1, 3, 64), np.float32); q[..., 0] = 1
    k = np.zeros((1, 1, 9, 64), np.float32); k[0, 0, 7, 0] = 80008
    v = np.zeros_like(k); v[0, 0, 7, :] = 1
    out = {}
    for arm in ("prod", "mf1", "after"):
        with functional(arm):
            o = QUANT[arm]()(*(tc.tensor(a) for a in (q, k, v)), bulk_bits=8, refine_percentile=1.0,
                             is_causal=True).numpy()
        out[arm] = {"row0": o[0, 0, 0, :4].tolist(), "row0_exact_zero": bool(np.all(o[0, 0, 0] == 0.0)),
                    "rows12_maxdev_from_1": float(np.abs(o[0, 0, 1:] - 1.0).max())}
    return out


def byte_identity():
    rows = []
    for model, layer in (("Qwen3-1.7B", 16), ("Qwen3-4B", 21)):
        d = os.path.join(CAP, model)
        man = json.load(open(os.path.join(d, "manifest.json")))
        grp = man["config"]["num_attention_heads"] // man["config"]["num_key_value_heads"]
        q = E.bf16_bits_to_f32(np.load(os.path.join(d, f"L{layer:02d}_q.npy")))
        k = E.bf16_bits_to_f32(np.load(os.path.join(d, f"L{layer:02d}_k.npy")))
        v = E.bf16_bits_to_f32(np.load(os.path.join(d, f"L{layer:02d}_v.npy")))
        S0 = q.shape[1]
        hs = np.arange(4)
        for L, S in ((64, S0), (512, S0), (512, 512)):
            qa = q[hs, S - L:S][None]; ka = k[hs // grp, :S][None]; va = v[hs // grp, :S][None]
            for dt in ("float32", "bfloat16", "float16"):
                for path in ("sdpa_nograd", "sdpa_grad", "apa_p1", "apa_p015"):
                    res = {}
                    for arm in ("before", "after"):
                        ts = [tc.tensor(a, dtype=dt) for a in (qa, ka, va)]
                        if arm == "after":
                            fn = Q.apa_quant_attention
                        else:
                            fn = MODS["mf1_quant" if path == "apa_p1" else "item1_quant"].apa_quant_attention
                        with functional("after" if arm == "after" else "mf1"):
                            o, g = run_path(path, fn, ts, None)
                        res[arm] = [o.float().numpy()] + ([x.float().numpy() for x in g] if g else [])
                    eq = all(b.tobytes() == a.tobytes() for b, a in zip(res["before"], res["after"]))
                    fin = all(np.isfinite(a).all() for a in res["after"])
                    rows.append({"model": model, "layer": layer, "L": L, "S": S, "dtype": dt, "path": path,
                                 "tensors_compared": len(res["after"]), "bytes_equal": eq, "after_finite": fin})
                    print(json.dumps(rows[-1]), flush=True)
    return rows


def main():
    out = sys.argv[1]
    t0 = time.time()
    res = {"engine": tc._C.__file__, "reviewer_n1": reviewer_n1()}
    print(json.dumps(res["reviewer_n1"]), flush=True)
    res["leak"] = leak_sweep()
    res["byte_identity"] = byte_identity()
    lk = [r for r in res["leak"] if r["arm"] == "after"]
    dom = [r for r in lk if not r["nonfinite_bulk"]]
    res["summary"] = {
        "leak_after_cells": len(lk), "leak_after_in_domain": len(dom),
        "leak_after_in_domain_exact_zero_and_finite": sum(r["blind_rows_exact_zero"] and r["all_finite"] for r in dom),
        "leak_after_nonfinite_bulk_cells": [r for r in lk if r["nonfinite_bulk"]],
        "leak_before_cells_leaking": {a: sum(not r["blind_rows_exact_zero"] for r in res["leak"] if r["arm"] == a)
                                      for a in ("prod", "mf1")},
        "byte_identity_cells": len(res["byte_identity"]),
        "byte_identity_equal": sum(r["bytes_equal"] for r in res["byte_identity"]),
        "byte_identity_after_finite": sum(r["after_finite"] for r in res["byte_identity"]),
    }
    res["t_s"] = time.time() - t0
    with open(out, "x") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["summary"], indent=1), "\nWROTE", out, res["t_s"], flush=True)


if __name__ == "__main__":
    main()
