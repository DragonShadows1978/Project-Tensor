"""APA-MF2 G2 (item 1) synthetic census + review cases, before vs after.

usage: apa_mf2_item1_synth.py <out.json>
Cells: L in {1,3,64,513}, S in {L,L+1,4L+5}, D in {64,96,128}, causal and non-causal;
B=1, H=KVH=2, N(0,0.5) rounded to bf16, bulk 8, p 0.15 (72 cells).
Per cell:
  fp32 item-1 cell: quant._bk4_selection on fp32 upcasts of the kernel's bf16 operands
      vs BK4 'h' (G2a, exact) and 'a' (reported); before = MF1 A+B bf16 selection vs 'h'.
  R1-only floor: the bit-level BK4 transcription (scripts/apa_mf2_bk4emu.bk4_fp32) on the
      engine's SGEMM dots vs 'h' -- what is left if the statistics arithmetic were BK4's.
  bf16 call (G2b): apa_quant_attention bf16 selection vs 'h', mismatches inside the band.
  fp16 call (G2b): apa_quant_attention fp16 selection vs the fp64 oracle (rule_fp64) on the
      call's own operands, mismatches inside the band.
  fp32 call (diagnostic only, not gated: no kernel on identical operands): vs fp64 oracle.
Review cases B1/B2/B3: before vs after vs 'h'.
Prior art: BK4 kernels.cu; review witnesses (APA_MF1_REVIEW.md). Script: mine.
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

BEF = os.path.join(ROOT, "artifacts/apa_mf2/before/mf1_AB")
P = 0.15
Z = Q._norm_ppf(1.0 - P)
BAND = 2e-5


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


BQ = load("tensor_cuda._mf2_bef_quant", os.path.join(BEF, "tensor_cuda__tensor_cuda__quant.py"))
BF = load("tensor_cuda._mf2_bef_functional", os.path.join(BEF, "tensor_cuda__tensor_cuda__functional.py"))


@contextlib.contextmanager
def arm(name):
    if name == "after":
        yield Q
        return
    saved = (F._causal_mask, F.scaled_dot_product_attention)
    F._causal_mask, F.scaled_dot_product_attention = BF._causal_mask, BF.scaled_dot_product_attention
    try:
        yield BQ
    finally:
        F._causal_mask, F.scaled_dot_product_attention = saved


def selection(fn, *ts, causal, **kw):
    rec = []
    orig = tc.where

    def wh(c, x, y):
        rec.append(c.float().numpy() != 0)
        return orig(c, x, y)

    tc.where = wh
    try:
        fn(*ts, bulk_bits=8, refine_percentile=P, is_causal=causal, **kw)
    finally:
        tc.where = orig
    return rec[-1]


@contextlib.contextmanager
def supplied_kq(mod, kq):
    orig = mod._quantize_keys
    mod._quantize_keys = lambda *a, **k: kq
    try:
        yield
    finally:
        mod._quantize_keys = orig


def kmask(q, k, kq, v, scale, causal, var="h"):
    r = tc._C.bp_kernel_4_diagnostic(q, k, kq, v, float(scale), float(Z), bool(causal), var)
    return r[3].numpy() != 0, r[2].numpy()


def band_count(sel, ref, absb, th):
    mm = sel != ref
    inb = np.abs(absb - th) <= BAND * np.maximum(1.0, np.abs(th))
    return int(mm.sum()), int((mm & ~inb).sum())


def cell(L, S, D, causal):
    rng = np.random.default_rng(20261007 + 1000 * L + 10 * S + D)
    q, k, v = (E.to_bf16_f32(rng.normal(0, 0.5, (1, 2, n, D)).astype(np.float32)) for n in (L, S, S))
    scale = D ** -0.5
    qt, kt, vt = (tc.tensor(a, dtype="bfloat16") for a in (q, k, v))
    R, CB, BND = Q._tables(D, 8, 2, True, "cuda")
    kq = Q._quantize_keys(kt, R, CB, BND)
    kmh, thh = kmask(qt, kt, kq, vt, scale, causal, "h")
    kma, _ = kmask(qt, kt, kq, vt, scale, causal, "a")
    vis = E.visible(L, S, causal)
    helper = Q._bk4_selection(qt.float(), kq.float(), scale, Z, causal).numpy() != 0
    out = {"L": L, "S": S, "D": D, "causal": causal, "n": int(kmh.size),
           "h_vs_a": int((kmh != kma).sum()),
           "after_fp32_vs_h": int((helper != kmh).sum()), "after_fp32_vs_a": int((helper != kma).sum())}
    if out["after_fp32_vs_h"]:
        b, h_, i_, j_ = np.nonzero(helper != kmh)
        cnt = vis.sum(-1)
        out["after_mismatch_min_row_count"] = int(cnt[i_].min())
    with arm("after") as m:
        sel_bf = selection(m.apa_quant_attention, qt, kt, vt, causal=causal)
    with arm("before") as m:
        sel_before = selection(m.apa_quant_attention, qt, kt, vt, causal=causal)
    out["plumbing_bf16_call_equals_helper"] = bool(np.array_equal(np.broadcast_to(sel_bf, helper.shape), helper))
    out["before_bf16_vs_h"] = int((np.broadcast_to(sel_before, kmh.shape) != kmh).sum())
    # R1-only floor: BK4's own statistics arithmetic on the engine's SGEMM dots
    sg = tc.matmul(qt.float(), kq.float(), alpha=1.0, trans_b=True).numpy()[0]
    sel_t, _ = E.bk4_fp32(sg, vis, scale, Z)
    out["R1_only_floor_vs_h"] = int((sel_t != kmh[0]).sum())
    # G2b bf16 band vs kernel threshold
    q64 = q.astype(np.float64); kq64 = kq.float().numpy().astype(np.float64)
    absb = np.abs(np.einsum("bhld,bhsd->bhls", q64, kq64) * scale)
    n_mm, n_out = band_count(np.broadcast_to(sel_bf, kmh.shape), kmh, absb, thh[..., None])
    out["bf16_call_vs_h"] = n_mm; out["bf16_call_vs_h_outside_band"] = n_out
    # G2b fp16 call vs fp64 oracle on its own operands
    for dt in ("float16", "float32"):
        ts = [tc.tensor(a, dtype=dt) for a in (q, k, v)]
        kqd = Q._quantize_keys(ts[1], R, CB, BND).float().numpy().astype(np.float64)
        qd = ts[0].float().numpy().astype(np.float64)
        ab = np.abs(np.einsum("bhld,bhsd->bhls", qd, kqd) * scale)
        ref, th = E.rule_fp64(ab, vis, Z)
        s = np.broadcast_to(selection(Q.apa_quant_attention, *ts, causal=causal), ref.shape)
        n_mm, n_out = band_count(s, ref, ab, th)
        key = "fp16_call_vs_fp64" if dt == "float16" else "fp32_call_vs_fp64_DIAGNOSTIC"
        out[key] = n_mm; out[key + "_outside_band"] = n_out
    return out


def review_cases():
    res = {}
    D = 64
    # B1
    q = np.zeros((1, 1, 1, D), np.float32); q[..., 0:2] = 1
    kqa = np.zeros((1, 1, 2, D), np.float32); kqa[..., 0] = 1; kqa[0, 0, 1, 1] = 1 / 256
    v = np.zeros_like(kqa); v[0, 0, 1, :] = 1
    qt, kqt, vt = (tc.tensor(a, dtype="bfloat16") for a in (q, kqa, v))
    km, th = kmask(qt, kqt, kqt, vt, 0.125, True)
    res["B1"] = {"bk4_h": km.reshape(-1).tolist(), "bk4_thr": float(th.reshape(-1)[0])}
    for a in ("before", "after"):
        with arm(a) as m, supplied_kq(m, kqt):
            res["B1"][a] = selection(m.apa_quant_attention, qt, kqt, vt, causal=True, scale=0.125).reshape(-1).tolist()
    # B2 (non-causal, fp32 call isolates B2 from B1)
    q = np.zeros((1, 1, 1, D), np.float32); q[..., 0:2] = 1
    kqa = np.zeros((1, 1, 2, D), np.float32); kqa[..., 0] = 800.0; kqa[0, 0, 1, 1] = 0.00799560546875
    v = np.zeros_like(kqa); v[0, 0, 1, :] = 1
    qb, kqb, vb = (tc.tensor(a, dtype="bfloat16") for a in (q, kqa, v))
    km, th = kmask(qb, kqb, kqb, vb, 0.125, False)
    res["B2"] = {"bk4_h": km.reshape(-1).tolist(), "bk4_thr": float(th.reshape(-1)[0])}
    qt, kqt, vt = (tc.tensor(a) for a in (q, kqa, v))
    for a in ("before", "after"):
        with arm(a) as m, supplied_kq(m, kqt):
            res["B2"][a] = selection(m.apa_quant_attention, qt, kqt, vt, causal=False, scale=0.125).reshape(-1).tolist()
    # B3
    res["B3"] = {}
    for dt in ("float32", "bfloat16", "float16"):
        qz = tc.tensor(np.zeros((1, 1, 3, D), np.float32), dtype=dt)
        kz = tc.tensor(np.zeros((1, 1, 9, D), np.float32), dtype=dt)
        for a in ("before", "after"):
            with arm(a) as m:
                res["B3"][f"{a}_{dt}_selected"] = int(selection(m.apa_quant_attention, qz, kz, kz, causal=True).sum())
    qz = tc.tensor(np.zeros((1, 1, 3, D), np.float32), dtype="bfloat16")
    kz = tc.tensor(np.zeros((1, 1, 9, D), np.float32), dtype="bfloat16")
    res["B3"]["bk4_h_selected"] = int(kmask(qz, kz, kz, kz, 0.125, True)[0].sum())
    for c in ("B1", "B2"):
        res[c]["before_matches_h"] = res[c]["before"] == res[c]["bk4_h"]
        res[c]["after_matches_h"] = res[c]["after"] == res[c]["bk4_h"]
    res["B3"]["before_matches_h"] = all(res["B3"][f"before_{d}_selected"] == 24 for d in ("float32", "bfloat16", "float16"))
    res["B3"]["after_matches_h"] = all(res["B3"][f"after_{d}_selected"] == 24 for d in ("float32", "bfloat16", "float16"))
    return res


def main():
    out = sys.argv[1]
    t0 = time.time()
    res = {"engine": tc._C.__file__, "cells": [], "review_cases": review_cases()}
    print(json.dumps(res["review_cases"]), flush=True)
    for L in (1, 3, 64, 513):
        for S in (L, L + 1, 4 * L + 5):
            for D in (64, 96, 128):
                for causal in (True, False):
                    c = cell(L, S, D, causal)
                    res["cells"].append(c)
                    print(json.dumps(c), flush=True)
    res["t_s"] = time.time() - t0
    with open(out, "x") as f:
        json.dump(res, f, indent=1)
    print("WROTE", out, res["t_s"], flush=True)


if __name__ == "__main__":
    main()
