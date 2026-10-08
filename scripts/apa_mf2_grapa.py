"""APA-MF2 G3: GRAPA GQAAttention forward byte identity, before vs after mask, read-only import.

GRAPA (/mnt/ForgeRealm/GRAPA-Native-LLM) is imported read-only (PYTHONDONTWRITEBYTECODE=1);
nothing there is edited. Geometry: grapa.model.GRAPAConfig (d_model 448, 8 q / 2 kv heads,
head_dim 64, bulk 4). One GQAAttention module (one random init, seeded), the same inputs
x (B=1, L, 448) and RoPE tables F.rope_tables(L, 64, base=10000) for both arms.
Paths that reach F._causal_mask / F.scaled_dot_product_attention (grapa/attention.py):
  refine 1.0                  -> F.scaled_dot_product_attention(is_causal=True)   :150
  refine 1.0, force_selective -> apa_selective composed chain, ranking + cm      :82-84
Grad on (output + d x + d q_proj.weight) and off (output). L in {64, 128, 256}.
Before = MF1/fd32b6d functional.py loaded from file; after = tree.
Not executed (source reading only, reported): GRAPA's composed refine<1 causal branch
(attention.py:89-112). With kq present it is unreachable from GQAAttention (grad on ->
tc.apa_selective_train, grad off -> tc.apa_selective_attention, :68-77), and calling it
directly would bypass that router.
Prior art: GRAPA attention.py (project-local). Script: mine.
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
import tensor_cuda as tc  # noqa: E402
from tensor_cuda import functional as F  # noqa: E402

assert tc.__file__.startswith(ROOT) and tc._C.__file__.startswith(ROOT)
GRAPA = "/mnt/ForgeRealm/GRAPA-Native-LLM"
sys.path.insert(0, GRAPA)
from grapa.attention import GQAAttention  # noqa: E402
from grapa.model import GRAPAConfig  # noqa: E402
import grapa.attention as GA  # noqa: E402

assert GA.F is F and GA.tc is tc
BEF = os.path.join(ROOT, "artifacts/apa_mf2/before/mf1_AB/tensor_cuda__tensor_cuda__functional.py")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


BFN = load("tensor_cuda._mf2_grapa_bef_functional", BEF)


@contextlib.contextmanager
def arm_ctx(arm):
    saved = (F._causal_mask, F.scaled_dot_product_attention)
    try:
        if arm == "before":
            F._causal_mask, F.scaled_dot_product_attention = BFN._causal_mask, BFN.scaled_dot_product_attention
        yield
    finally:
        F._causal_mask, F.scaled_dot_product_attention = saved


def main():
    out = sys.argv[1]
    cfg = GRAPAConfig()
    np.random.seed(20261007)
    attn = GQAAttention(cfg)
    rows = []
    t0 = time.time()
    for L in (64, 128, 256):
        rng = np.random.default_rng(1000 + L)
        x_np = rng.normal(0, 1, (1, L, cfg.d_model)).astype(np.float32)
        cos, sin = F.rope_tables(L, cfg.head_dim, "cuda", base=cfg.rope_base)
        for force in (False, True):
            attn.force_selective = force
            for grad in (False, True):
                res = {}
                for arm in ("before", "after"):
                    with arm_ctx(arm):
                        if grad:
                            attn.zero_grad()
                            x = tc.tensor(x_np, requires_grad=True)
                            o = attn(x, cos, sin, refine_percentile=1.0)
                            o.sum().backward()
                            res[arm] = [o.numpy(), x.grad.numpy(), attn.q_proj.weight.grad.numpy().copy()]
                        else:
                            with tc.no_grad():
                                res[arm] = [attn(tc.tensor(x_np), cos, sin, refine_percentile=1.0).numpy()]
                eq = all(b.tobytes() == a.tobytes() for b, a in zip(res["before"], res["after"]))
                fin = all(np.isfinite(a).all() for a in res["after"])
                rows.append({"L": L, "force_selective": force, "grad": grad, "tensors": len(res["after"]),
                             "bytes_equal": eq, "after_finite": fin})
                print(json.dumps(rows[-1]), flush=True)
    res = {"grapa": GRAPA, "geometry": {k: getattr(cfg, k) for k in ("d_model", "n_heads", "n_kv_heads", "head_dim", "bulk_bits")},
           "rows": rows, "all_bytes_equal": all(r["bytes_equal"] for r in rows),
           "all_finite": all(r["after_finite"] for r in rows), "t_s": time.time() - t0}
    with open(out, "x") as f:
        json.dump(res, f, indent=1)
    print("all_bytes_equal", res["all_bytes_equal"], "all_finite", res["all_finite"], "WROTE", out, flush=True)


if __name__ == "__main__":
    main()
