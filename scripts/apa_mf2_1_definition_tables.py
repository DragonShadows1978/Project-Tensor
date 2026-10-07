"""APA-MF2.1 G4: definition-agreement tables (composed and BK4 kernels vs the fp64 definition).

usage: apa_mf2_1_definition_tables.py real <model> <layers,...> <out.json>
       apa_mf2_1_definition_tables.py synth <out.json>
       apa_mf2_1_definition_tables.py neartie <out.json>
Composed = the selection apa_quant_attention actually uses (last tc.where condition) on
the call's own quantized key (captured from quant._quantize_keys inside the call, or
supplied for the near-tie rule-isolation cells). Oracle = scripts/apa_mf2_1_defn.py
(fp64, centered two-pass, exact upcasts). Kernel = tc._C.bp_kernel_4_diagnostic
variants 'h' (BK4) and 'a' (shipped train kernel rule) on the SAME bf16 q, k, kq, v
(bf16 cells only; NON-gating). Band: | |b| - thr | <= 2e-5 |thr| (registered).
Real cells: MF2 geometry -- last L query rows of S=8192, one q head per kv head
(heads 0, g, 2g, ...), bulk {4,8}, p 0.15, causal.
Prior art: centered two-pass variance; Chan, Golub & LeVeque (1983) -- unverified, lead
to check. Harness: mine (MF2 apa_mf2_item1_real.py pattern).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import apa_mf2_1_defn as DEF  # noqa: E402
import tensor_cuda as tc  # noqa: E402
from tensor_cuda import quant as Q  # noqa: E402

assert tc._C.__file__.startswith(os.path.join(ROOT, "artifacts/apa_mf2_1/build")), tc._C.__file__
assert Q.__file__.startswith(os.path.join(ROOT, "tensor_cuda/tensor_cuda")), Q.__file__
CAP = "/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_spq3/artifacts/apa_spq3/captures"
P = 0.15
Z = Q._norm_ppf(1.0 - P)


@contextlib.contextmanager
def observe(kq=None):
    rec = {"where": []}
    ow, oq, oge = tc.where, Q._quantize_keys, tc.Tensor.ge

    def ge(self, other):          # observe the composed threshold (the helper's only .ge)
        rec["thr"] = other
        return oge(self, other)

    def wh(c, x, y):
        rec["where"].append(c)
        return ow(c, x, y)

    def qk(*a, **kw):
        r = kq if kq is not None else oq(*a, **kw)
        rec["kq"] = r
        return r

    tc.where, Q._quantize_keys, tc.Tensor.ge = wh, qk, ge
    try:
        yield rec
    finally:
        tc.where, Q._quantize_keys, tc.Tensor.ge = ow, oq, oge


def merge(acc, r):
    for k in ("n_entries", "n_visible", "mismatch", "in_band", "out_of_band", "invisible_selected"):
        acc[k] = acc.get(k, 0) + r[k]
    acc["max_rel_gap_mismatch"] = max(acc.get("max_rel_gap_mismatch", 0.0), r["max_rel_gap_mismatch"])
    for k in ("thr_rows", "thr_rows_rel_err_gt_band", "thr_gt_band_rows_cnt_le2", "thr_gt_band_rows_cnt_gt2"):
        if k in r:
            acc[k] = acc.get(k, 0) + r[k]
    if "thr_max_rel_err" in r:
        acc["thr_max_rel_err"] = max(acc.get("thr_max_rel_err", 0.0), r["thr_max_rel_err"])
    if "out_of_band_row_visible_counts" in r:
        o = acc.setdefault("out_of_band_row_visible_counts", {"le2": 0, "gt2": 0})
        o["le2"] += r["out_of_band_row_visible_counts"]["le2"]; o["gt2"] += r["out_of_band_row_visible_counts"]["gt2"]
        acc["max_rel_gap_out_of_band"] = max(acc.get("max_rel_gap_out_of_band", 0.0), r["max_rel_gap_out_of_band"])
    return acc


def thr_err(t_impl, t64, cnt=None):
    """Threshold accuracy (observation): |thr_impl - thr64| / |thr64| per row (absolute
    when thr64 == 0). Complements the entry tie band, which cannot see a threshold error
    on rows whose entries sit exactly ON the fp64 threshold (constant rows)."""
    e = np.abs(t_impl - t64) / np.where(t64 != 0, np.abs(t64), 1.0)
    r = {"thr_rows": int(e.size), "thr_max_rel_err": float(e.max()),
         "thr_rows_rel_err_gt_band": int((e > DEF.BAND).sum())}
    if cnt is not None:
        bad = e > DEF.BAND
        r["thr_gt_band_rows_cnt_le2"] = int((bad & (cnt <= 2)).sum())
        r["thr_gt_band_rows_cnt_gt2"] = int((bad & (cnt > 2)).sum())
    return r


def cell(qt, kt, vt, causal, scale, kq_supplied=None, kernels=True):
    """One call: composed / h / a vs fp64, accumulated over heads (fp64 per head)."""
    with observe(kq_supplied) as rec:
        out = Q.apa_quant_attention(qt, kt, vt, bulk_bits=cell.bits, refine_percentile=P,
                                    is_causal=causal, scale=scale)
    kq = rec["kq"]
    sel = rec["where"][-1].float().numpy() != 0
    thr_c = rec["thr"].numpy()[0, :, :, 0].astype(np.float64)
    km, kthr = {}, {}
    if kernels and qt.dtype == "bfloat16":
        for var in ("h", "a"):
            r_ = tc._C.bp_kernel_4_diagnostic(qt, kt, kq, vt, float(scale), float(Z), bool(causal), var)
            km[var] = r_[3].numpy() != 0
            kthr[var] = r_[2].numpy()[0].astype(np.float64)
    q64 = qt.float().numpy()[0].astype(np.float64)
    k64 = kq.float().numpy()[0].astype(np.float64)
    H = q64.shape[0]
    res = {"composed": {}, **{f"kernel_{v}": {} for v in km}}
    for h in range(H):
        b64 = (q64[h] @ k64[h].T) * float(scale)
        ref, thr, a, vis = DEF.definition_fp64(b64, causal, Z)
        merge(res["composed"], {**DEF.compare(np.broadcast_to(sel, (1, H) + ref.shape)[0, h], ref, a, thr, vis),
                                **thr_err(thr_c[h], thr[:, 0], vis.sum(-1))})
        for v in km:
            merge(res[f"kernel_{v}"], {**DEF.compare(km[v][0, h], ref, a, thr, vis), **thr_err(kthr[v][h], thr[:, 0], vis.sum(-1))})
    res["output_finite"] = bool(np.isfinite(out.float().numpy()).all())
    return res


cell.bits = 8


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def real(model, layers, out):
    d = os.path.join(CAP, model)
    man = json.load(open(os.path.join(d, "manifest.json")))
    S = man["S"]; Hq = man["config"]["num_attention_heads"]; KVH = man["config"]["num_key_value_heads"]
    D = man["config"]["head_dim"]; grp = Hq // KVH
    rows = []
    for layer in layers:
        files = {n: os.path.join(d, f"L{layer:02d}_{n}.npy") for n in "qkv"}
        for n, p in files.items():
            assert sha(p) == man["files"][os.path.basename(p)]["sha256"], "capture pin drift"
        arr = {n: (np.load(p).astype(np.uint32) << 16).view(np.float32) for n, p in files.items()}
        hs = np.arange(KVH) * grp
        for L in (64, 512, 2048):
            qt = tc.tensor(arr["q"][hs, S - L:][None], dtype="bfloat16")
            kt = tc.tensor(arr["k"][None], dtype="bfloat16")
            vt = tc.tensor(arr["v"][None], dtype="bfloat16")
            for bits in (4, 8):
                cell.bits = bits
                t0 = time.time()
                r = cell(qt, kt, vt, True, D ** -0.5)
                r.update({"model": model, "layer": layer, "L": L, "S": S, "bulk_bits": bits, "heads": hs.tolist(),
                          "t_s": time.time() - t0})
                rows.append(r)
                print(json.dumps(r), flush=True)
    cell.bits = 8
    json.dump({"kind": "real", "z": Z, "engine": tc._C.__file__, "rows": rows}, open(out, "x"), indent=1)


def synth(out):
    rows = []
    for L in (1, 3, 64, 513):
        for S in (L, L + 1, 4 * L + 5):
            for D in (64, 96, 128):
                for dt in ("float32", "bfloat16", "float16"):
                    for causal in (True, False):
                        rng = np.random.default_rng(20261007 + 1000 * L + 10 * S + D)
                        xs = [tc.tensor(rng.normal(0, 0.5, (1, 2, n, D)).astype(np.float32), dtype=dt) for n in (L, S, S)]
                        r = cell(*xs, causal, D ** -0.5)
                        r.update({"L": L, "S": S, "D": D, "dtype": dt, "causal": causal})
                        rows.append(r)
    json.dump({"kind": "synthetic", "z": Z, "rows": rows}, open(out, "x"), indent=1)
    print("synthetic cells", len(rows), flush=True)


def neartie(out):
    rows = []
    for D in (64, 96, 128):
        for L, S in ((1, 1), (1, 2), (1, 256), (1, 2048), (3, 9)):
            for causal in (False, True):
                q = np.zeros((1, 1, L, D), np.float32); q[..., :3] = 1
                k = np.zeros((1, 1, S, D), np.float32); k[..., :3] = [1, 107 / 256, 1 / 65536]
                qt, kt, vt = (tc.tensor(a, dtype="bfloat16") for a in (q, k, np.zeros_like(k)))
                r = cell(qt, kt, vt, causal, 1.0, kq_supplied=kt)
                r.update({"case": "review_F2_constant_row", "L": L, "S": S, "D": D, "causal": causal})
                rows.append(r)
    # witnesses: B1, B2, zero rows, count discriminator
    D = 64
    q = np.zeros((1, 1, 1, D), np.float32); q[..., 0:2] = 1
    kq = np.zeros((1, 1, 2, D), np.float32); kq[..., 0] = 1; kq[0, 0, 1, 1] = 1 / 256
    t = [tc.tensor(a, dtype="bfloat16") for a in (q, kq, kq)]
    rows.append({**cell(*t, True, 0.125, kq_supplied=t[1]), "case": "B1"})
    kq2 = np.zeros((1, 1, 2, D), np.float32); kq2[..., 0] = 800.0; kq2[0, 0, 1, 1] = 0.00799560546875
    t = [tc.tensor(a, dtype="bfloat16") for a in (q, kq2, kq2)]
    rows.append({**cell(*t, False, 0.125, kq_supplied=t[1]), "case": "B2"})
    for dt in ("float32", "bfloat16", "float16"):
        z3 = [tc.tensor(np.zeros((1, 1, n, 64), np.float32), dtype=dt) for n in (3, 9, 9)]
        rows.append({**cell(*z3, True, 0.125, kq_supplied=z3[1]), "case": "zero_rows", "dtype": dt})
    qc = np.zeros((1, 1, 3, 64), np.float32); qc[..., 0] = 1
    kc = np.zeros((1, 1, 9, 64), np.float32); kc[0, 0, :, 0] = [1, 1, 1, 1, 1, 1.2, 1.5, 1, 1]
    for dt in ("float32", "bfloat16"):
        t = [tc.tensor(a, dtype=dt) for a in (qc, kc, kc)]
        rows.append({**cell(*t, True, 1.0, kq_supplied=t[1]), "case": "count_divisor", "dtype": dt})
    json.dump({"kind": "near_tie", "z": Z, "rows": rows}, open(out, "x"), indent=1)
    for r in rows:
        print(json.dumps({k: v for k, v in r.items()}), flush=True)


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "real":
        real(sys.argv[2], [int(x) for x in sys.argv[3].split(",")], sys.argv[4])
    elif kind == "synth":
        synth(sys.argv[2])
    else:
        neartie(sys.argv[2])
