"""APA-MF2 design probe P0 (PRE-REGISTRATION, not a gate).

Question: can apa_quant_attention, composed from existing engine ops, reproduce the
BK4 kernel's selection bits exactly? Decompose any residual:
  R1 dot bits: BK4 'h' uses bf16 WMMA m16n16k16 with fp32 accumulate (bk3::scores);
     the engine's fp32 matmul is cublasSgemmStridedBatched (matmul.cu).
  R2 statistics order/fusion: BK4 sums sequentially (FADD) and squares with FFMA.
  R3 threshold arithmetic: BK4 fuses var and th with FFMA.
Method: extract BK4's own dot products through its threshold output -- with S=L=1,
scale=1, zthr=0 the kernel's th equals |dot| exactly (count=1, mean=|dot|,
th=fma(sqrt(var),0,mean)=mean). Then rebuild BK4's threshold/mask with the fp32
transcription (scripts/apa_mf2_bk4emu.py) from (a) extracted WMMA dots and (b) SGEMM
dots, and compare bitwise with the kernel's own thr/mask on a real cell.
Prior art: see scripts/apa_mf2_bk4emu.py. Dot-extraction-through-threshold trick:
no prior art known to me.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import apa_mf2_bk4emu as E  # noqa: E402
import tensor_cuda as tc  # noqa: E402
from tensor_cuda import quant as Q  # noqa: E402

assert tc._C.__file__.startswith(ROOT), tc._C.__file__
CAP = "/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_spq3/artifacts/apa_spq3/captures"


def kernel_dots(qb, kb, variant):
    """|dot| from the kernel itself for row pairs (qb[n], kb[n]); fp32 arrays of bf16 values."""
    N, D = qb.shape
    out = np.empty(N, np.float32)
    for s in range(0, N, 65535):
        e = min(N, s + 65535)
        n = e - s
        q = tc.tensor(np.ascontiguousarray(qb[s:e][None, :, None, :]), dtype="bfloat16")
        k = tc.tensor(np.ascontiguousarray(kb[s:e][None, :, None, :]), dtype="bfloat16")
        v = tc.tensor(np.zeros((1, n, 1, D), np.float32), dtype="bfloat16")
        r = tc._C.bp_kernel_4_diagnostic(q, k, k, v, 1.0, 0.0, True, variant)
        out[s:e] = r[2].numpy().reshape(-1)
    return out


def main():
    model, layer, L, bits = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    nkv = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    d = os.path.join(CAP, model)
    man = json.load(open(os.path.join(d, "manifest.json")))
    S = man["S"]; Hq = man["config"]["num_attention_heads"]; KVH = man["config"]["num_key_value_heads"]
    D = man["config"]["head_dim"]; grp = Hq // KVH
    q = E.bf16_bits_to_f32(np.load(os.path.join(d, f"L{layer:02d}_q.npy")))
    k = E.bf16_bits_to_f32(np.load(os.path.join(d, f"L{layer:02d}_k.npy")))
    v = E.bf16_bits_to_f32(np.load(os.path.join(d, f"L{layer:02d}_v.npy")))
    scale = D ** -0.5
    z = Q._norm_ppf(0.85)
    hs = np.arange(nkv) * grp
    qt = tc.tensor(q[hs, S - L:][None], dtype="bfloat16")
    kt = tc.tensor(k[None, :nkv], dtype="bfloat16")
    vt = tc.tensor(v[None, :nkv], dtype="bfloat16")
    R, CB, BND = Q._tables(D, bits, nkv, True, "cuda")
    kq = Q._quantize_keys(kt, R, CB, BND)               # bf16, the kernel's kq
    qf = qt.float().numpy()[0]; kqf = kq.float().numpy()[0]   # exact bf16 values
    vis = E.visible(L, S)
    res = {"model": model, "layer": layer, "L": L, "S": S, "bits": bits, "kv_heads": nkv}
    t0 = time.time()
    out = {}
    for var in ("h", "a"):
        r = tc._C.bp_kernel_4_diagnostic(qt, kt, kq, vt, float(scale), float(z), True, var)
        out[var] = (r[2].numpy()[0], r[3].numpy()[0] != 0)          # thr (H,L), mask (H,L,S)
    res["h_vs_a_mask_mismatch"] = int((out["h"][1] != out["a"][1]).sum())
    res["h_vs_a_thr_bits_differ"] = int((out["h"][0].view(np.uint32) != out["a"][0].view(np.uint32)).sum())
    # R1: dot bits, WMMA (h) vs SGEMM (engine fp32 matmul in the production shape)
    sg = tc.matmul(tc.tensor(qf[None]), tc.tensor(kqf[None]), alpha=1.0, trans_b=True).numpy()[0]   # (H,L,S)
    sg_scaled = tc.matmul(tc.tensor(qf[None]), tc.tensor(kqf[None]), alpha=float(scale), trans_b=True).numpy()[0]
    res["sgemm_alpha_equals_fl32(dot*scale)"] = bool(np.array_equal(
        sg_scaled.view(np.uint32), (sg * np.float32(scale)).view(np.uint32)))
    wm = np.empty_like(sg)
    for h in range(nkv):
        qq = np.repeat(qf[h], S, axis=0)                 # pairs (i, j) row-major
        kk = np.tile(kqf[h], (L, 1))
        wm[h] = kernel_dots(qq, kk, "h").reshape(L, S)
    wm_signed = np.copysign(wm, sg)                     # kernel gives |dot|; sign from SGEMM
    res["dots"] = {"n": int(sg.size),
                   "abs_wmma_vs_abs_sgemm_bits_differ": int((np.abs(sg).view(np.uint32) != wm.view(np.uint32)).sum())}
    if L <= 512:
        exact = np.einsum("hld,hsd->hls", qf.astype(np.float64), kqf.astype(np.float64))
        res["dots"]["abs_sgemm_vs_exact_rounded_differ"] = int((np.abs(sg) != np.abs(exact).astype(np.float32)).sum())
        res["dots"]["abs_wmma_vs_exact_rounded_differ"] = int((wm != np.abs(exact).astype(np.float32)).sum())
        del exact
    # model check: BK4 transcription on WMMA dots must reproduce the kernel's own thr/mask bits
    for name, dots in (("wmma_dots", wm_signed), ("sgemm_dots", sg)):
        sel, th = E.bk4_fp32(dots, vis, scale, z)
        res[f"transcription_{name}"] = {
            "thr_bits_differ_vs_h": int((th.view(np.uint32) != out["h"][0].view(np.uint32)).sum()),
            "mask_mismatch_vs_h": int((sel != out["h"][1]).sum()),
            "mask_mismatch_vs_a": int((sel != out["a"][1]).sum()),
        }
    # engine-op candidates on SGEMM fp32 bulk (what quant.py can compose)
    bulk = tc.tensor(sg_scaled[None])
    absb = bulk.abs()
    invis = tc.tensor((~vis).astype(np.float32))
    absb = absb.masked_fill(invis, 0.0)
    cnt = vis.sum(-1).astype(np.float32)[:, None]
    inv = tc.tensor(1.0 / cnt)
    cands = {}
    # (iii-a) tree sums (.sum), separate roundings (MF1-B style but fp32 bulk)
    mean = absb.sum([-1], True) * inv
    v_ = ((absb * absb).sum([-1], True) * inv - mean * mean).relu()
    cands["tree_sum_mulinv"] = mean + v_.pow(0.5) * z
    # (iii-b) sequential cumsum sums (BK4 order), squares via cumsum of fl(a*a), IEEE div
    sums = absb.cumsum(-1).numpy()[0][..., -1:]
    sqs = (absb * absb).cumsum(-1).numpy()[0][..., -1:]
    m32 = (sums.astype(np.float64) / cnt).astype(np.float32)
    q32 = (sqs.astype(np.float64) / cnt).astype(np.float32)
    var_a = np.maximum(E.fma32(-m32, m32, q32), np.float32(0))
    th_cs = E.fma32(np.sqrt(var_a.astype(np.float64)).astype(np.float32), np.full_like(m32, np.float32(z)), m32)
    absnp = np.abs(sg_scaled)
    for name, th in list(cands.items()):
        thn = th.numpy()[0]
        cands[name] = thn
    cands["cumsum_seq_noFMAsq_emuthr"] = th_cs
    res["engine_candidates"] = {}
    for name, thn in cands.items():
        sel = vis[None] & (absnp >= thn)
        mm = sel != out["h"][1]
        gap = np.abs(absnp - out["h"][0][..., None])
        res["engine_candidates"][name] = {
            "thr_bits_differ_vs_h": int((thn[..., 0].view(np.uint32) != out["h"][0].view(np.uint32)).sum()),
            "mask_mismatch_vs_h": int(mm.sum()),
            "max_rel_gap_of_mismatches": float((gap[mm] / np.maximum(1.0, out["h"][0][..., None].repeat(S, -1)[mm])).max()) if mm.any() else 0.0,
        }
    res["t_s"] = time.time() - t0
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
