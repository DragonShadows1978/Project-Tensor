"""APA-MF2.2c G2 table: the V7 sweep per package arm (one process each), both bands, plus the time of
the two moment sums at S = 2^20. usage: apa_mf2_2c_v7_table.py <arm> <out.json>. Mine."""
import collections, hashlib, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tensor_cuda", "tests"))
import tensor_cuda as tc
from tensor_cuda import quant as Q
import test_apa_mf2_2c_longkeys as T
arm, out = sys.argv[1], sys.argv[2]
res = {"arm": arm, "quant_sha256": hashlib.sha256(open(Q.__file__, "rb").read()).hexdigest(), "cells": [], "full_api": []}
for S in (1 << 10, 1 << 14, 1 << 17, (1 << 17) + 3, 1 << 20):
    for w in (2.0 ** -20, 2.0 ** -16, 2.0 ** -12, 2.0 ** -10, 1.0):
        for pat in ("sorted_steps", "steps", "uniform"):
            for causal in (False, True):
                b = T.rows(S, w, pat, 2, S)[None, None]
                sel = T.select(b, causal)
                r = T.O.mismatch(sel, b, causal, T.Z)
                res["cells"].append({"S": S, "width": w, "pattern": pat, "causal": causal, "mismatch": r["mismatch"],
                                     "out_of_band": r["out_of_band"], "review_band_out": T.review_band_errors(sel, b, causal)})
for dt in ("float32", "bfloat16"):
    for S in (1 << 14, 1 << 17, 1 << 20):
        for causal in (False, True):
            D, L = 64, 2
            b = T.rows(S, 2.0 ** -12, "sorted_steps", 1, S)[0]
            q = np.zeros((1, 1, L, D), np.float32); q[..., 0] = 1.0
            k = np.zeros((1, 1, S, D), np.float32); k[..., 0] = b
            with T.capture() as rec, tc.no_grad():
                Q.apa_quant_attention(*[tc.tensor(x, dtype=dt) for x in (q, k, k)], bulk_bits=4, refine_percentile=0.15,
                                      is_causal=causal, scale=1.0)
            r = T.O.mismatch(rec["sel"], rec["bulk32"], causal, T.Z)
            res["full_api"].append({"dtype": dt, "S": S, "causal": causal, "mismatch": r["mismatch"], "out_of_band": r["out_of_band"],
                                    "review_band_out": T.review_band_errors(rec["sel"], rec["bulk32"], causal)})
x = tc.tensor(np.random.default_rng(0).uniform(0, 1, (1, 1, 2, 1 << 20)).astype(np.float32))
f = getattr(Q, "_row_sum", lambda t: t.sum([-1], True))
for _ in range(3): f(x).numpy()
t0 = time.perf_counter()
for _ in range(20): f(x).numpy()
res["moment_sum_ms_S2^20_L2"] = (time.perf_counter() - t0) / 20 * 1e3
agg = collections.OrderedDict()
for c in res["cells"]:
    a = agg.setdefault(f"S={c['S']} w=2^{int(np.log2(c['width']))}", [0, 0, 0, 0]); a[0] += 1; a[1] += c["mismatch"]; a[2] += c["out_of_band"]; a[3] += c["review_band_out"]
res["by_S_width"] = agg
json.dump(res, open(out, "x"), indent=1)
print(json.dumps({"arm": arm, "cells_out_of_band": sum(c["out_of_band"] for c in res["cells"]), "cells_review_band_out": sum(c["review_band_out"] for c in res["cells"]),
                  "full_api_out_of_band": sum(c["out_of_band"] for c in res["full_api"]), "full_api_review_band_out": sum(c["review_band_out"] for c in res["full_api"]),
                  "moment_sum_ms": round(res["moment_sum_ms_S2^20_L2"], 3)}))
