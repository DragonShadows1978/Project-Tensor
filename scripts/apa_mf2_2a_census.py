"""APA-MF2.2 Amendment 1, item 4: special-row census, ONE ENGINE PER PROCESS.

usage: apa_mf2_2a_census.py arm <name> <out.json>        (PYTHONPATH = engine:pkg)
       apa_mf2_2a_census.py compare <dir> <stamp> <out.json> [arms...]
Census (artifacts/apa_mf2_2a/registration.json item4_census). Special rows are built from FINITE
operands by fp32 overflow for attention paths (fp32/bf16: Q0=1e21 x K0=-1e18, scale 1; fp16: Q0=2 x
K0=-60000, scale 1e34) and from literal -inf scores for score-input paths. Every cell records the
output's finite flag, first values and a NaN-canonical sha256 (grad paths: grads too).
Prior art: differential testing (McKeeman 1998) -- unverified, lead to check; IEEE 754 special
arithmetic. Harness: mine.
"""
import contextlib
import hashlib
import json
import os
import sys

import numpy as np

os.environ["TC_ATTN_QTILE"] = "0"
ROWS = ("all_blocked", "all_masked", "neg_inf_first", "single_visible")
SIZES = (1, 2, 33, 1025)


def blocked_cols(row, S):
    if row in ("all_blocked", "all_masked"):
        return np.ones(S, bool)
    if row == "neg_inf_first":
        b = np.zeros(S, bool); b[0] = True; return b
    b = np.ones(S, bool); b[-1] = False; return b       # single_visible


def canon(a):
    a = np.asarray(a, np.float32)
    return hashlib.sha256(np.where(np.isnan(a), np.float32(np.nan), a).tobytes()).hexdigest()


def arm(name, out):
    import tensor_cuda as tc
    from tensor_cuda import functional as F
    from tensor_cuda import quant as Q
    eng = os.environ["APA_MF2_2A_ENGINE"]
    assert os.path.realpath(tc._C.__file__).startswith(os.path.realpath(eng) + "/"), tc._C.__file__
    rec = {"provenance": {"arm": name, "engine": tc._C.__file__,
                          "engine_sha256": hashlib.sha256(open(tc._C.__file__, "rb").read()).hexdigest(),
                          "package": tc.__file__}, "cells": {}}
    Z = Q._norm_ppf(0.85)

    def put(key, fn):
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001 -- a rejected operand is recorded, not hidden
            rec["cells"][key] = {"error": str(e).splitlines()[0][:160]}
            return
        cell = {}
        for nm, t in r.items():
            a = t.float().numpy()
            cell[nm] = {"finite": bool(np.isfinite(a).all()), "sha256": canon(a),
                        "values": [None if not np.isfinite(x) else float(x) for x in a.reshape(-1)[:4]]}
        rec["cells"][key] = cell

    def mk(a, dt, grad=False):
        a = np.asarray(a, np.float32)
        if dt == "bfloat16":
            t = tc.tensor(a, dtype=dt).detach()
            if grad:
                t.requires_grad = True
            return t
        return tc.tensor(a, dtype=dt, requires_grad=grad)

    def operands(dt, D, Lq, S, row, seed):
        rng = np.random.default_rng(seed)
        q0, k0, scale = (2.0, -60000.0, 1e34) if dt == "float16" else (1e21, -1e18, 1.0)
        q = np.zeros((1, 1, Lq, D), np.float32); q[..., 0] = q0
        k = np.zeros((1, 1, S, D), np.float32)
        if row != "all_masked":
            k[0, 0, blocked_cols(row, S), 0] = k0
        v = rng.normal(0, 1, (1, 1, S, D)).astype(np.float32)
        mask = np.zeros((1, 1, Lq, S), np.float32)
        if row == "all_masked":
            mask[...] = -np.inf
        return q, k, v, mask, scale

    @contextlib.contextmanager
    def fused_softmax(flag):
        old = F.USE_FUSED_SOFTMAX; F.USE_FUSED_SOFTMAX = flag
        try:
            yield
        finally:
            F.USE_FUSED_SOFTMAX = old

    for dt in ("float32", "float16", "bfloat16"):
        for D in (64, 96, 128):
            for Lq in (1, 3):
                for S in SIZES:
                    for row in ROWS:
                        seed = 1000 * S + 10 * D + Lq
                        q, k, v, mask, scale = operands(dt, D, Lq, S, row, seed)
                        base = f"{dt}/D{D}/Lq{Lq}/S{S}/{row}"
                        X = lambda grad=False: [mk(x, dt, grad) for x in (q, k, v)]  # noqa: E731
                        if row != "all_masked":
                            with tc.no_grad():
                                put("fused_sdpa/" + base, lambda: {"out": tc.fused_sdpa_noncausal(*X(), scale)})
                                put("int4_all/" + base, lambda: {"out": tc.apa_int4_sdpa_noncausal(*X(), scale, Z, True)})
                                put("int4_partial/" + base, lambda: {"out": tc.apa_int4_sdpa_noncausal(*X(), scale, Z, False)})
                                put("sdpa_infer_noncausal/" + base,
                                    lambda: {"out": F.scaled_dot_product_attention(*X(), scale=scale)})
                                if S >= Lq:
                                    with fused_softmax(False):
                                        put("sdpa_infer_causal/" + base,
                                            lambda: {"out": F.scaled_dot_product_attention(*X(), is_causal=True, scale=scale)})
                                    with fused_softmax(True):
                                        put("sdpa_infer_causal_fused/" + base,
                                            lambda: {"out": F.scaled_dot_product_attention(*X(), is_causal=True, scale=scale)})
                                for p in (1.0, 0.15):
                                    for causal in (False, True):
                                        if causal and S < Lq:
                                            continue
                                        put(f"apa_quant_p{p}_causal{causal}/" + base,
                                            lambda p=p, causal=causal: {"out": Q.apa_quant_attention(
                                                *X(), bulk_bits=4, refine_percentile=p, is_causal=causal, scale=scale)})
                            if S >= Lq:
                                def grad_causal():
                                    xs = X(True)
                                    o = F.scaled_dot_product_attention(*xs, is_causal=True, scale=scale)
                                    o.sum().backward()
                                    return {"out": o, **{"d" + n: x.grad for n, x in zip("qkv", xs)}}
                                put("sdpa_grad_causal/" + base, grad_causal)
                        else:
                            with tc.no_grad():
                                put("sdpa_infer_masked/" + base,
                                    lambda: {"out": F.scaled_dot_product_attention(*X(), attn_mask=mk(mask, dt), scale=scale)})

                            def grad_masked():
                                xs = X(True)
                                o = F.scaled_dot_product_attention(*xs, attn_mask=mk(mask, dt), scale=scale)
                                o.sum().backward()
                                return {"out": o, **{"d" + n: x.grad for n, x in zip("qkv", xs)}}
                            put("sdpa_grad_masked/" + base, grad_masked)
                # score-input paths: literal -inf scores
                if D == 64:
                    for S in SIZES:
                        for row in ("all_blocked", "neg_inf_first", "single_visible"):
                            sc = np.random.default_rng(S).normal(0, 1, (1, 1, Lq, S)).astype(np.float32)
                            sc[..., blocked_cols(row, S)] = -np.inf
                            base = f"{dt}/Lq{Lq}/S{S}/{row}"
                            with tc.no_grad():
                                put("eager_softmax/" + base, lambda: {"out": mk(sc, dt).softmax(-1)})
                                if S >= Lq:
                                    put("causal_softmax/" + base, lambda: {"out": tc.causal_softmax(mk(sc, dt))})
    with open(out, "x") as f:
        json.dump(rec, f)
    nonfin = sum(any(not c[n]["finite"] for n in c) for c in rec["cells"].values() if "error" not in c)
    print(name, "cells", len(rec["cells"]), "errors", sum("error" in c for c in rec["cells"].values()), "nonfinite", nonfin)


def compare(d, stamp, out, arms):
    A = {a: json.load(open(os.path.join(d, f"arm_{a}_{stamp}.json"))) for a in arms}
    keys = list(A["fd32b6d"]["cells"])
    for a in arms:
        assert list(A[a]["cells"]) == keys, a

    def status(c):
        if "error" in c:
            return "error"
        return "finite" if all(c[n]["finite"] for n in c) else "NONFINITE"

    res = {"stamp": stamp, "engines": {a: A[a]["provenance"] for a in arms}, "cells": len(keys)}
    for a in arms[1:]:
        new_nan = [k for k in keys if status(A["fd32b6d"]["cells"][k]) == "finite" and status(A[a]["cells"][k]) == "NONFINITE"]
        res[f"fd32b6d_finite_{a}_nonfinite"] = {"count": len(new_nan), "keys": new_nan}
    by = {}
    for k in keys:
        path, *_rest = k.split("/"); row = k.split("/")[-1]
        sig = tuple(status(A[a]["cells"][k]) for a in arms)
        by.setdefault(f"{path} | {row}", {}).setdefault(" / ".join(sig), 0)
        by[f"{path} | {row}"][" / ".join(sig)] += 1
    res["by_path_row"] = by
    if "after" in A and "mf2_2" in A:
        fdnan = [k for k in keys if status(A["fd32b6d"]["cells"][k]) == "NONFINITE"]
        changed = [k for k in fdnan if json.dumps(A["after"]["cells"][k], sort_keys=True) != json.dumps(A["mf2_2"]["cells"][k], sort_keys=True)]
        res["fd32b6d_nonfinite_cells"] = len(fdnan)
        res["fd32b6d_nonfinite_after_differs_from_mf2_2"] = {"count": len(changed), "keys": changed[:50]}
        fb = [k for k in keys if k.split("/")[0] in ("fused_sdpa", "int4_all", "int4_partial", "sdpa_infer_noncausal")
              and k.endswith("/all_blocked") and "error" not in A["after"]["cells"][k]]
        eq = [k for k in fb if A["after"]["cells"][k]["out"]["sha256"] == A["fd32b6d"]["cells"][k]["out"]["sha256"]]
        res["item3_fused_all_blocked_after_bytes_eq_fd32b6d"] = {"cells": len(fb), "equal": len(eq),
                                                                 "differ": [k for k in fb if k not in eq][:50]}
        other = [k for k in keys if k not in fb and json.dumps(A["after"]["cells"][k], sort_keys=True) != json.dumps(A["mf2_2"]["cells"][k], sort_keys=True)]
        res["after_differs_from_mf2_2_outside_fused_all_blocked"] = {"count": len(other), "keys": other[:50]}
        res["G2_pass"] = (res["fd32b6d_finite_after_nonfinite"]["count"] == 0 and len(changed) == 0
                          and len(eq) == len(fb))
    with open(out, "x") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) or "count" not in v else v["count"]) for k, v in res.items()
                      if k not in ("engines", "by_path_row")}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "arm":
        arm(sys.argv[2], sys.argv[3])
    else:
        compare(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
