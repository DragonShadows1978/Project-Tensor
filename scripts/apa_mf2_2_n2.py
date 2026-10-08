"""APA-MF2.2 G3 (N2): definition agreement across the finite fp32 |bulk| range + real captures.

usage: apa_mf2_2_n2.py sweep <out.json>                (PYTHONPATH = engine:pkg:scripts)
       apa_mf2_2_n2.py real <model> <l1,l2,l3> <out.json>
       apa_mf2_2_n2.py table <before_sweep> <after_sweep> <before_real...> -- <after_real...>
Census: registration.json census.n2_sweep / census.real. Oracles (mine): apa_mf2_2_oracle
definition64 = fp64 statistics of the call's exact fp32 bulk; real cells ALSO the MF2.1
registered oracle apa_mf2_1_defn (fp64 GEMM of the call's own q/kq upcasts). Band 2e-5 relative.
Prior art: Blue (1978) / BLAS nrm2 scaling; Chan, Golub & LeVeque (1983) -- unverified, lead to
check. Harness: mine (MF2.1 apa_mf2_1_definition_tables.py pattern).
"""
import contextlib
import hashlib
import json
import os
import sys
import time

import numpy as np

import tensor_cuda as tc
from tensor_cuda import quant as Q
import apa_mf2_2_oracle as O

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(W, "tensor_cuda", "tests"))
Z = Q._norm_ppf(1.0 - 0.15)
CAP = "/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_spq3/artifacts/apa_spq3/captures"
KEYS = ("entries_visible", "mismatch", "in_band", "out_of_band", "invisible_selected")


def prov():
    return {"engine": tc._C.__file__, "engine_sha256": hashlib.sha256(open(tc._C.__file__, "rb").read()).hexdigest(),
            "quant": Q.__file__, "quant_sha256": hashlib.sha256(open(Q.__file__, "rb").read()).hexdigest()}


def acc(a, r):
    for k in KEYS:
        a[k] = a.get(k, 0) + r[k]
    a["max_rel_gap_mismatch"] = max(a.get("max_rel_gap_mismatch", 0.0), r["max_rel_gap_mismatch"])
    a["cells"] = a.get("cells", 0) + 1
    return a


@contextlib.contextmanager
def capture():
    rec = {}
    old = Q._definition_selection

    def spy(qf, kqf, scale, z, is_causal, bulk32=None):
        if bulk32 is None:
            bulk32 = tc.matmul(qf, kqf, alpha=scale, trans_b=True)
        rec.update(qf=qf, kqf=kqf, scale=scale, bulk32=bulk32)
        rec["sel"] = old(qf, kqf, scale, z, is_causal, bulk32=bulk32)
        return rec["sel"]

    Q._definition_selection = spy
    try:
        yield rec
    finally:
        Q._definition_selection = old


def sweep(out):
    import test_apa_mf2_2_n2 as T
    res = {"provenance": prov(), "z": Z, "isolated": {}, "full_api": {}, "witness": {}, "zero_rows": {},
           "constant_rows_all_selected": {}}
    for scale in T.SCALES:
        row = {}
        for pattern in T.PATTERNS:
            for S in T.SIZES:
                for causal in (False, True):
                    L = min(4, S) if causal else 4
                    rng = np.random.default_rng(S + 7 * L)
                    mag = T._base(pattern, L, S, rng) * scale
                    sign = np.where(rng.random((L, S)) < 0.5, -1.0, 1.0)
                    b32 = (sign * mag).astype(np.float32)[None, None]
                    sel = T._select(b32, causal)
                    r = O.mismatch(sel, b32, causal, Z)
                    acc(row, r)
                    if pattern == "constant":
                        k = f"{scale:g}"
                        ok = bool(np.array_equal(sel[0, 0] != 0, O.visible(L, S, causal)))
                        res["constant_rows_all_selected"][k] = res["constant_rows_all_selected"].get(k, True) and ok
        res["isolated"][f"{scale:g}"] = row
    rng = np.random.default_rng(5)
    b = rng.uniform(0.01, 1.0, (2, 2, 256)) * np.array([[1e30, 1e-30], [3e38, 1e-38]])[..., None]
    for causal in (False, True):
        sel = T._select(b.astype(np.float32)[None], causal)
        res["isolated"].setdefault("mixed_rows", {})
        acc(res["isolated"]["mixed_rows"], O.mismatch(sel, b.astype(np.float32)[None], causal, Z))
    for S in (1, 3, 256):
        for causal in (False, True):
            L = min(3, S) if causal else 3
            sel = T._select(np.zeros((1, 1, L, S), np.float32), causal)
            res["zero_rows"][f"S{S}/causal{causal}"] = bool(np.array_equal(sel[0, 0] != 0, O.visible(L, S, causal)))
    for mag in (1e20, 1e-30):
        for D in (64, 96, 128):
            q = np.zeros((1, 1, 1, D), np.float32); q[..., 0] = mag
            k = np.zeros((1, 1, 3, D), np.float32); k[0, 0, :, 0] = [1, 2, 3]
            qt, kt = tc.tensor(q), tc.tensor(k)
            with tc.no_grad():
                b32 = tc.matmul(qt, kt, alpha=1.0, trans_b=True)
                sel = Q._definition_selection(qt, kt, 1.0, Z, False, bulk32=b32).numpy()
            res["witness"][f"{mag:g}/D{D}"] = {"selection": (sel[0, 0, 0] != 0).tolist(),
                                              **O.mismatch(sel, b32.numpy(), False, Z)}
    for dt in ("float32", "bfloat16"):
        for mag in (1e-30, 1e-20, 1e-10, 1.0, 1e10, 1e20, 1e30):
            row = {}
            for S in (3, 256, 2048):
                for causal in (False, True):
                    rng = np.random.default_rng(S + int(causal))
                    q = (rng.normal(0, 1, (1, 1, 3, 64)) * mag).astype(np.float32)
                    k = rng.normal(0, 1, (1, 1, S, 64)).astype(np.float32)
                    v = rng.normal(0, 1, (1, 1, S, 64)).astype(np.float32)
                    xs = [tc.tensor(x, dtype=dt) for x in (q, k, v)]
                    with capture() as rec, tc.no_grad():
                        o = Q.apa_quant_attention(*xs, bulk_bits=4, refine_percentile=0.15, is_causal=causal, scale=1.0)
                    acc(row, O.mismatch(rec["sel"].numpy(), rec["bulk32"].numpy(), causal, Z))
                    row["output_finite"] = row.get("output_finite", True) and bool(np.isfinite(o.float().numpy()).all())
            res["full_api"][f"{dt}/{mag:g}"] = row
    json.dump(res, open(out, "x"), indent=1)
    print(json.dumps({"isolated_out_of_band": sum(r["out_of_band"] for r in res["isolated"].values()),
                      "full_api_out_of_band": sum(r["out_of_band"] for r in res["full_api"].values()),
                      "witness_out_of_band": sum(r["out_of_band"] for r in res["witness"].values()),
                      "zero_rows_ok": all(res["zero_rows"].values()),
                      "constant_rows_ok": all(res["constant_rows_all_selected"].values())}))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def real(model, layers, out):
    import apa_mf2_1_defn as DEF
    d = os.path.join(CAP, model)
    man = json.load(open(os.path.join(d, "manifest.json")))
    S = man["S"]; Hq = man["config"]["num_attention_heads"]; KVH = man["config"]["num_key_value_heads"]
    D = man["config"]["head_dim"]; grp = Hq // KVH
    rows = []
    for layer in layers:
        files = {n: os.path.join(d, f"L{layer:02d}_{n}.npy") for n in "qkv"}
        pins = {}
        for n, p in files.items():
            pins[n] = sha(p)
            assert pins[n] == man["files"][os.path.basename(p)]["sha256"], "capture pin drift"
        arr = {n: (np.load(p).astype(np.uint32) << 16).view(np.float32) for n, p in files.items()}
        hs = np.arange(KVH) * grp
        for L in (64, 512, 2048):
            qt = tc.tensor(arr["q"][hs, S - L:][None], dtype="bfloat16")
            kt = tc.tensor(arr["k"][None], dtype="bfloat16")
            vt = tc.tensor(arr["v"][None], dtype="bfloat16")
            for bits in (4, 8):
                t0 = time.time()
                with capture() as rec, tc.no_grad():
                    o = Q.apa_quant_attention(qt, kt, vt, bulk_bits=bits, refine_percentile=0.15, is_causal=True)
                sel = rec["sel"].numpy(); b32 = rec["bulk32"].numpy()
                q64 = rec["qf"].numpy()[0].astype(np.float64); k64 = rec["kqf"].numpy()[0].astype(np.float64)
                r1, r2 = {}, {}
                for h in range(len(hs)):
                    acc(r1, O.mismatch(sel[0, h], b32[0, h], True, Z))
                    b64 = (q64[h] @ k64[h].T) * float(rec["scale"])
                    ref, thr, a, vis = DEF.definition_fp64(b64, True, Z)
                    c = DEF.compare(sel[0, h] != 0, ref, a, thr, vis)
                    acc(r2, {"entries_visible": c["n_visible"], **{k: c[k] for k in KEYS[1:]},
                             "max_rel_gap_mismatch": c["max_rel_gap_mismatch"]})
                rows.append({"model": model, "layer": layer, "L": L, "S": S, "bits": bits, "heads": hs.tolist(),
                             "capture_sha256": pins,
                             "selection_sha256": hashlib.sha256(np.ascontiguousarray(sel != 0).tobytes()).hexdigest(),
                             "selected": int((sel != 0).sum()), "oracle_fp32bulk": r1, "oracle_mf2_1_gemm": r2,
                             "output_finite": bool(np.isfinite(o.float().numpy()).all()), "t_s": time.time() - t0})
                print(json.dumps({k: rows[-1][k] for k in ("model", "layer", "L", "bits", "selected")}
                                 | {"oob32": r1["out_of_band"], "oob64": r2["out_of_band"],
                                    "mm32": r1["mismatch"], "mm64": r2["mismatch"]}), flush=True)
    json.dump({"provenance": prov(), "z": Z, "rows": rows}, open(out, "x"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "sweep":
        sweep(sys.argv[2])
    elif sys.argv[1] == "real":
        real(sys.argv[2], [int(x) for x in sys.argv[3].split(",")], sys.argv[4])
