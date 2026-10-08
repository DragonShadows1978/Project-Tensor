"""APA-MF2 G2 (item 1) on real captures: selection vs the BK4 kernel's own mask.

usage: apa_mf2_item1_real.py <model> <layer> <out.json>
Cells: L in {64,512,2048} (last L query rows of S=8192), bulk {4,8}, p 0.15, causal;
one query head per KV head (kq shared), the kernel's operands: q, k, v bf16 from the
capture, kq = quant._quantize_keys(bf16 K) (what apa_quant_attention computes).
Arms (same process, same operands):
  before = MF1 A+B quant.py + MF1 functional.py (artifacts/apa_mf2/before/mf1_AB)
  after  = tree (MF2)
Selection is captured at apa_quant_attention's tc.where condition (bf16 call).
fp32 item-1 cell = quant._bk4_selection on the fp32 upcasts of the same operands;
it must equal the after bf16 selection exactly (plumbing).
Per mismatch vs 'h': relative gap |a64 - thr_h| / max(1, |thr_h|) (a64 = exact fp64
|q.kq|*scale) against the registered 2e-5 tie band, and the row's visible count.
Prior art: BK4 kernels.cu (Project-Tensor 2026); FlashAttention-2 (Dao 2023)
bottom-right; MF1 harness pattern (apa_mf1_real.py). Script: mine.
"""
from __future__ import annotations

import contextlib
import hashlib
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
CAP = "/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_spq3/artifacts/apa_spq3/captures"
BEF = os.path.join(ROOT, "artifacts/apa_mf2/before/mf1_AB")
P = 0.15
BAND = 2e-5


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@contextlib.contextmanager
def arm(name):
    """'before' swaps in MF1 A+B quant + functional (files); 'after' is the tree."""
    if name == "after":
        yield Q.apa_quant_attention
        return
    bf = load("tensor_cuda._mf2_bef_functional", os.path.join(BEF, "tensor_cuda__tensor_cuda__functional.py"))
    bq = load("tensor_cuda._mf2_bef_quant", os.path.join(BEF, "tensor_cuda__tensor_cuda__quant.py"))
    saved = (F._causal_mask, F.scaled_dot_product_attention)
    F._causal_mask, F.scaled_dot_product_attention = bf._causal_mask, bf.scaled_dot_product_attention
    try:
        yield bq.apa_quant_attention
    finally:
        F._causal_mask, F.scaled_dot_product_attention = saved


def selection(fn, qt, kt, vt, bits):
    rec = []
    orig = tc.where

    def wh(c, x, y):
        rec.append(c.float().numpy()[0] != 0)
        return orig(c, x, y)

    tc.where = wh
    try:
        fn(qt, kt, vt, bulk_bits=bits, refine_percentile=P, is_causal=True)
    finally:
        tc.where = orig
    return rec[-1]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    model, layer, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    d = os.path.join(CAP, model)
    man = json.load(open(os.path.join(d, "manifest.json")))
    S = man["S"]; Hq = man["config"]["num_attention_heads"]; KVH = man["config"]["num_key_value_heads"]
    D = man["config"]["head_dim"]; grp = Hq // KVH
    files = {n: os.path.join(d, f"L{layer:02d}_{n}.npy") for n in "qkv"}
    q = E.bf16_bits_to_f32(np.load(files["q"]))
    k = E.bf16_bits_to_f32(np.load(files["k"]))
    v = E.bf16_bits_to_f32(np.load(files["v"]))
    scale = D ** -0.5
    z = Q._norm_ppf(1.0 - P)
    hs = np.arange(KVH) * grp
    res = {"model": model, "layer": layer, "S": S, "D": D, "heads": hs.tolist(), "z": z,
           "inputs_sha256": {n: sha(p) for n, p in files.items()}, "engine": tc._C.__file__,
           "engine_sha256": sha(tc._C.__file__), "cells": []}
    t0 = time.time()
    for L in (64, 512, 2048):
        vis = E.visible(L, S)
        cnt_row = vis.sum(-1)
        qt = tc.tensor(q[hs, S - L:][None], dtype="bfloat16")
        kt = tc.tensor(k[None], dtype="bfloat16")
        vt = tc.tensor(v[None], dtype="bfloat16")
        for bits in (4, 8):
            R, CB, BND = Q._tables(D, bits, KVH, True, "cuda")
            kq = Q._quantize_keys(kt, R, CB, BND)
            km, thr = {}, {}
            for var in ("h", "a"):
                r = tc._C.bp_kernel_4_diagnostic(qt, kt, kq, vt, float(scale), float(z), True, var)
                km[var] = r[3].numpy()[0] != 0
                thr[var] = r[2].numpy()[0]
            sel = {}
            for a in ("before", "after"):
                with arm(a) as fn:
                    sel[a] = selection(fn, qt, kt, vt, bits)
            helper = Q._bk4_selection(qt.float(), kq.float(), scale, z, True).numpy()[0] != 0
            q64 = qt.float().numpy()[0].astype(np.float64)
            kq64 = kq.float().numpy()[0].astype(np.float64)
            n_all = int(km["h"].size); n_vis = int(vis.sum()) * KVH
            cell = {"L": L, "bulk_bits": bits, "n_entries": n_all, "n_visible": n_vis,
                    "h_vs_a_mismatch": int((km["h"] != km["a"]).sum()),
                    "fp32_helper_equals_after_bf16_selection": bool(np.array_equal(helper, sel["after"]))}
            for a in ("before", "after"):
                for var in ("h", "a"):
                    mm = sel[a] != km[var]
                    n = int(mm.sum())
                    c = {"mismatch": n, "mismatch_visible": int((mm & vis[None]).sum()),
                         "agreement": 1.0 - n / n_all}
                    if var == "h" and n:
                        hh, ii, jj = np.nonzero(mm)
                        a64 = np.abs(np.einsum("nd,nd->n", q64[hh, ii], kq64[hh, jj]) * scale)
                        th = thr["h"][hh, ii].astype(np.float64)
                        gap = np.abs(a64 - th) / np.maximum(1.0, np.abs(th))
                        c.update({"max_rel_gap": float(gap.max()),
                                  "outside_tie_band": int((gap > BAND).sum()),
                                  "min_row_count": int(cnt_row[ii].min()),
                                  "invisible_selected": int((~vis[ii, jj] & sel[a][hh, ii, jj]).sum())})
                    cell[f"{a}_vs_{var}"] = c
            res["cells"].append(cell)
            print(json.dumps(cell), flush=True)
            del km, sel, helper, q64, kq64
    res["t_s"] = time.time() - t0
    with open(out, "x") as f:
        json.dump(res, f, indent=1)
    print("WROTE", out, res["t_s"], flush=True)


if __name__ == "__main__":
    main()
