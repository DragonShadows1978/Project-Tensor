"""APA-MF1 item 5: before / A / A+B on real Qwen3 captures (S=8192, bf16).

For one (model, layer): last L query rows (L in 64/512/2048) against all S keys,
bulk_bits 4 and 8, refine 0.15, causal. Arms: before (fd32b6d copy), A (snapshot
after fix A), A+B (tree). Same operands, same process, interleaved per cell.

Metrics per cell (all over VISIBLE (row, col) pairs, all query heads):
  eligible  = visible keys not zeroed by the stats mask (observed, masked_fill hook)
  refined   = visible keys selected for exact scoring (observed, tc.where hook)
  agree_fused = agreement of the code's selection with the fp64 fused-kernel rule
               (visible-count population mean/std of the code's own |bulk|)  [G3]
  out_rel_F / out_maxabs = output vs fp64 exact attention (||.||_F ratio, max-abs)
Kernel diag (one query head per KV head so kq is shared): tc._C.bp_kernel_4_diagnostic
variants 'a' (shipped apa_selective_fwd_train + a_mask) and 'h' (BK4) selection
masks vs each arm's quant.py mask, and vs the fp64 rule on fp64 bulk.

Prior art: fused-kernel rule kernels.cu bk4 (Project-Tensor); GRAPA _visible_counts;
FlashAttention-2 bottom-right causal (Dao, 2023). Harness: no prior art known to me.
"""
from __future__ import annotations

import argparse
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
import apa_mf1_ref as R  # noqa: E402
import tensor_cuda as tc  # noqa: E402
from tensor_cuda import quant as QAB  # noqa: E402

CAP = "/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_spq3/artifacts/apa_spq3/captures"
P = 0.15
G = 8


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@contextlib.contextmanager
def capture():
    rec = {}
    orig_mf, orig_where = tc.Tensor.masked_fill, tc.where

    def mf(self, m, v):
        rec["absr"] = self.float().numpy()
        rec["statmask"] = m.float().numpy()
        return orig_mf(self, m, v)

    def wh(c, x, y):
        rec["refine"] = c.float().numpy()
        return orig_where(c, x, y)

    tc.Tensor.masked_fill, tc.where = mf, wh
    try:
        yield rec
    finally:
        tc.Tensor.masked_fill, tc.where = orig_mf, orig_where


def fused_counts(absr, refine, vis, z, chunk=256):
    """Row-chunked fp64 fused rule vs the code's selection: (mismatch, ties)."""
    mism = ties = 0
    L = absr.shape[-2]
    for r0 in range(0, L, chunk):
        a = absr[..., r0:r0 + chunk, :].astype(np.float64)
        v = vis[r0:r0 + chunk]
        sel, thr = R.fused_rule_select(a, v, z)
        code = refine[..., r0:r0 + chunk, :] != 0
        mism += int(((code != sel) & v).sum())
        ties += int((R.ambiguous(a, thr, 1e-5) & v).sum())
    return mism, ties


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert tc.__file__.startswith(ROOT), tc.__file__
    arms = {
        "before": load_mod("tensor_cuda._mf1_before", os.path.join(ROOT, "artifacts/apa_mf1/before/tensor_cuda__tensor_cuda__quant.py")),
        "A": load_mod("tensor_cuda._mf1_A", os.path.join(ROOT, "artifacts/apa_mf1/variants/quant_A.py")),
        "AB": QAB,
    }
    d = os.path.join(CAP, a.model)
    man = json.load(open(os.path.join(d, "manifest.json")))
    S = man["S"]; Hq = man["config"]["num_attention_heads"]; KVH = man["config"]["num_key_value_heads"]
    Dh = man["config"]["head_dim"]; grp = Hq // KVH
    files = {n: os.path.join(d, f"L{a.layer:02d}_{n}.npy") for n in "qkv"}
    q = R.bf16_bits_to_f32(np.load(files["q"]))  # (Hq,S,D)
    k = R.bf16_bits_to_f32(np.load(files["k"]))  # (KVH,S,D)
    v = R.bf16_bits_to_f32(np.load(files["v"]))
    scale = Dh ** -0.5
    z = QAB._norm_ppf(1.0 - P)
    Lmax = 2048
    t0 = time.time()
    # fp64 exact attention for the last Lmax rows (absolute positions S-Lmax..S-1)
    ref = np.zeros((Hq, Lmax, Dh))
    for h in range(Hq):
        ref[h], _ = R.exact_attention(q[h, S - Lmax:], k[h // grp], v[h // grp], scale)
    t_ref = time.time() - t0
    result = {"model": a.model, "layer": a.layer, "S": S, "Hq": Hq, "KVH": KVH, "D": Dh, "z": z,
              "inputs_sha256": {n: sha(p) for n, p in files.items()}, "t_ref_s": t_ref,
              "engine": tc._C.__file__, "cells": []}
    kv_idx = np.arange(Hq) // grp
    for L in (64, 512, 2048):
        vis = R.visible_mask(L, S)
        n_vis_row = vis.sum(-1)
        refL = ref[:, Lmax - L:]
        for bits in (4, 8):
            for arm, mod in arms.items():
                acc = dict(n_vis=0, elig=0, refined=0, mism=0, ties=0, se=0.0, rn=0.0, maxabs=0.0)
                excl_rows_exact = True
                for h0 in range(0, Hq, G):
                    hs = np.arange(h0, min(Hq, h0 + G))
                    qt = tc.tensor(q[hs, S - L:][None], dtype="bfloat16")
                    kt = tc.tensor(k[kv_idx[hs]][None], dtype="bfloat16")
                    vt = tc.tensor(v[kv_idx[hs]][None], dtype="bfloat16")
                    with capture() as rec:
                        out = mod.apa_quant_attention(qt, kt, vt, bulk_bits=bits, refine_percentile=P, is_causal=True)
                    o = out.float().numpy()[0].astype(np.float64)
                    del out, qt, kt, vt
                    stat = np.broadcast_to(rec["statmask"] != 0, (L, S))
                    elig_row = (vis & ~stat).sum(-1)
                    excl = n_vis_row - elig_row
                    if not np.all(excl == (S - L if arm == "before" else 0)):
                        excl_rows_exact = False
                    ng = len(hs)
                    acc["n_vis"] += int(vis.sum()) * ng
                    acc["elig"] += int(elig_row.sum()) * ng
                    refm = rec["refine"][0]
                    acc["refined"] += int(((refm != 0) & vis).sum())
                    mm, tt = fused_counts(rec["absr"][0], refm, vis, z)
                    acc["mism"] += mm; acc["ties"] += tt
                    rr = refL[hs]
                    acc["se"] += float(((o - rr) ** 2).sum()); acc["rn"] += float((rr ** 2).sum())
                    acc["maxabs"] = max(acc["maxabs"], float(np.abs(o - rr).max()))
                    del rec
                cell = dict(L=L, bulk_bits=bits, arm=arm,
                            eligible_frac=acc["elig"] / acc["n_vis"], refined_frac=acc["refined"] / acc["n_vis"],
                            agree_fused=1.0 - acc["mism"] / acc["n_vis"], fused_mismatch=acc["mism"], fused_ties=acc["ties"],
                            out_rel_F=(acc["se"] / acc["rn"]) ** 0.5, out_maxabs=acc["maxabs"],
                            excluded_per_row_matches_prediction=excl_rows_exact, n_visible=acc["n_vis"])
                result["cells"].append(cell)
                print(json.dumps(cell), flush=True)
            # kernel diagnostic masks on one query head per KV head (kq shared)
            hs = np.arange(KVH) * grp
            qt = tc.tensor(q[hs, S - L:][None], dtype="bfloat16")
            kt = tc.tensor(k[None], dtype="bfloat16")
            vt = tc.tensor(v[None], dtype="bfloat16")
            Rt, CB, BND = QAB._tables(Dh, bits, KVH, True, "cuda")
            kq = QAB._quantize_keys(kt, Rt, CB, BND)
            kmask = {}
            for var in ("a", "h"):
                try:
                    r = tc._C.bp_kernel_4_diagnostic(qt, kt, kq, vt, float(scale), float(z), True, var)
                    kmask[var] = np.asarray(r[3].numpy())[0] != 0
                except Exception as e:  # noqa: BLE001
                    kmask[var] = repr(e)
            bulk64 = np.abs(np.einsum("hld,hsd->hls", qt.float().numpy()[0].astype(np.float64),
                                      kq.float().numpy()[0].astype(np.float64)) * scale)
            rule64, _ = R.fused_rule_select(bulk64, vis, z)
            del bulk64
            nv = int(vis.sum()) * KVH
            kd = dict(L=L, bulk_bits=bits, heads=hs.tolist(), kernel_vs_rule_fp64={}, arm_vs_kernel={})
            for var, km in kmask.items():
                if isinstance(km, str):
                    kd["kernel_vs_rule_fp64"][var] = km
                    continue
                kd["kernel_vs_rule_fp64"][var] = 1.0 - int(((km != rule64) & vis).sum()) / nv
            for arm, mod in arms.items():
                with capture() as rec:
                    mod.apa_quant_attention(qt, kt, vt, bulk_bits=bits, refine_percentile=P, is_causal=True)
                cm = rec["refine"][0] != 0
                kd["arm_vs_kernel"][arm] = {var: (1.0 - int(((cm != km) & vis).sum()) / nv) if not isinstance(km, str) else km
                                           for var, km in kmask.items()}
                del rec
            del rule64
            result.setdefault("kernel_diag", []).append(kd)
            print(json.dumps(kd), flush=True)
    result["t_total_s"] = time.time() - t0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "x") as f:
        json.dump(result, f, indent=1)
    print("WROTE", a.out, "t_total_s", result["t_total_s"])


if __name__ == "__main__":
    main()
