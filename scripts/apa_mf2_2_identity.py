"""APA-MF2.2 G2: N1 rows and byte identity of the two fused non-causal kernels, one ENGINE PER
PROCESS (fd32b6d / MF2 048949d / MF2.1 / MF2.2; each with its own package).

usage: apa_mf2_2_identity.py arm <name> <out.json>     (PYTHONPATH = engine:pkg:scripts)
       apa_mf2_2_identity.py compare <dir> <stamp> <out.json>
Census (registration.json census.identity / n1_*): finite non-overflow inputs, TC_ATTN_QTILE=0.
Hashes are sha256 of the output's fp32 bytes (fp16 -> fp32 is a bijection on finite values).
Prior art: differential testing (McKeeman 1998, "Differential testing for software") --
unverified, lead to check; SHA-256 content pinning (FIPS 180-4). Harness: mine.
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
os.environ["TC_ATTN_QTILE"] = "0"


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def arm(name, out):
    import tensor_cuda as tc
    sys.path.insert(0, os.path.join(W, "tensor_cuda", "tests"))
    import test_apa_mf2_2_n1 as T          # same operand constructors as the pytest gate
    import apa_mf2_2_oracle as O
    rec = {"provenance": {"arm": name, "engine": tc._C.__file__, "engine_sha256": sha_file(tc._C.__file__),
                          "package": tc.__file__, "TC_ATTN_QTILE": os.environ["TC_ATTN_QTILE"]},
           "identity": {}, "n1": {}, "observations": {}}
    assert os.path.realpath(tc._C.__file__).startswith(os.path.realpath(os.environ["APA_MF2_2_ENGINE"]))

    def put(table, key, t, ref=None):
        a = t.float().numpy()
        r = {"sha256": hashlib.sha256(np.ascontiguousarray(a, np.float32).tobytes()).hexdigest(),
             "finite": bool(np.isfinite(a).all())}
        if ref is not None:
            r["max_err"] = float(np.abs(a.astype(np.float64)[0, 0] - ref).max()) if r["finite"] else None
        if table is rec["observations"]:
            r["values"] = [float(x) for x in a.reshape(-1)[:4]]
        table[key] = r

    # ---- identity census: finite non-overflow inputs
    for dt in ("float32", "float16"):
        for D in (64, 96, 128):
            for Lq in (1, 3, 17):
                for S in (1, 2, 3, 5, 9, 33, 200, 1025):
                    for sig in (0.2, 3.0):
                        rng = np.random.default_rng(int(1e6 * sig) + 1000 * D + 10 * Lq + S)
                        xs = [T._t(rng.normal(0, sig, (1, 2, n, D)), dt) for n in (Lq, S, S)]
                        for op in T.OPS:
                            put(rec["identity"], f"{dt}/{op}/D{D}/Lq{Lq}/S{S}/sig{sig}", T._run(op, *xs, D ** -0.5))
    for D in (64, 128):
        for S in (2, 33, 1025):
            rng = np.random.default_rng(D + S)
            xs = [T._t(rng.normal(0, 1, (1, 2, n, D)) * f, "float32") for n, f in ((3, 1e15), (S, 1e15), (S, 1.0))]
            for op in T.OPS:
                put(rec["identity"], f"wide/float32/{op}/D{D}/S{S}", T._run(op, *xs, D ** -0.5))
    # ---- N1: review witnesses + -inf-first sweep (the pytest census)
    for D in (64, 96, 128):
        for S in (2, 5):
            for op in ("sdpa", "int4_all", "int4_partial_fp16"):
                dt = "float16" if op == "int4_partial_fp16" else "float32"
                q = np.zeros((1, 1, 1, D), np.float32); k = np.zeros((1, 1, S, D), np.float32)
                v = np.ones_like(k); v[0, 0, 0] = 0
                q[..., 0] = 2 if dt == "float16" else 1e20
                k[0, 0, 0, 0] = -60000 if dt == "float16" else -1e20
                o = T._run("int4_partial" if op == "int4_partial_fp16" else op, *[T._t(x, dt) for x in (q, k, v)], 1.0)
                put(rec["n1"], f"witness/{op}/D{D}/S{S}", o, np.ones((1, D)))
    for dt in ("float32", "float16"):
        for op in T.OPS:
            for S in (2, 5, 33, 1025):
                for pattern in ("key0", "warp_first", "half"):
                    for D in (64, 96, 128):
                        for Lq in (1, 3):
                            xs, scale, ref, _ = T._sweep_operands(dt, op, S, D, Lq, pattern, 1000 * S + D + Lq)
                            put(rec["n1"], f"sweep/{dt}/{op}/S{S}/{pattern}/D{D}/Lq{Lq}", T._run(op, *xs, scale), ref)
    # ---- registered census: bf16-EXACT operand values through the fp32 kernels (both kernels
    # reject bf16 tensors); operands rounded fp32 -> bf16 -> fp32 on device, oracle on the rounded values
    for op in T.OPS:
        for S in (2, 5, 33, 1025):
            for pattern in ("key0", "warp_first", "half"):
                for D in (64, 128):
                    xs, scale, _, neg = T._sweep_operands("float32", op, S, D, 3, pattern, 7000 + S + D)
                    xs = [x.astype("bfloat16").float() for x in xs]
                    q64, k64, v64 = (x.numpy().astype(np.float64)[0, 0] for x in xs)
                    s64 = (q64 @ k64.T) * scale; s64[:, neg] = -np.inf
                    put(rec["n1"], f"sweep_bf16exact/float32/{op}/S{S}/{pattern}/D{D}/Lq3", T._run(op, *xs, scale),
                        O.softmax64(s64, v64))
    # ---- observations (not gated): F1 fused rows, fully blocked rows, bf16 rejection
    for op in ("sdpa", "int4_all"):
        for s in (-3.3e38, -3.402e38):
            q = np.zeros((1, 1, 1, 64), np.float32); q[..., 0] = 1
            k = np.zeros((1, 1, 5, 64), np.float32); k[0, 0, :, 0] = -T.FLT_MAX; k[0, 0, 0, 0] = s
            v = np.zeros_like(k); v[0, 0, 0] = 1
            put(rec["observations"], f"F1/{op}/{s}", T._run(op, *[T._t(x, "float32") for x in (q, k, v)], 1.0))
        q = np.zeros((1, 1, 1, 64), np.float32); q[..., 0] = 1e20
        k = np.zeros((1, 1, 5, 64), np.float32); k[0, 0, :, 0] = -1e20
        put(rec["observations"], f"fully_blocked_overflow/{op}",
            T._run(op, *[T._t(x, "float32") for x in (q, k, np.ones_like(k))], 1.0))
    try:
        T._run("sdpa", *[tc.tensor(np.zeros((1, 1, 1, 64), np.float32), dtype="bfloat16") for _ in range(3)], 1.0)
        rec["observations"]["bf16_fused_sdpa"] = "accepted"
    except Exception as e:  # noqa: BLE001
        rec["observations"]["bf16_fused_sdpa"] = "raised: " + str(e)
    with open(out, "x") as f:
        json.dump(rec, f, indent=1)
    print(name, "identity keys", len(rec["identity"]), "n1 keys", len(rec["n1"]),
          "n1 nonfinite", sum(not r["finite"] for r in rec["n1"].values()))


def compare(d, stamp, out):
    arms = {a: json.load(open(os.path.join(d, f"arm_{a}_{stamp}.json"))) for a in ("fd32b6d", "mf2", "mf2_1", "mf2_2")}
    new = arms["mf2_2"]
    res = {"stamp": stamp, "engines": {a: r["provenance"] for a, r in arms.items()}}
    for base in ("fd32b6d", "mf2", "mf2_1"):
        b = arms[base]
        assert b["provenance"]["engine_sha256"] != new["provenance"]["engine_sha256"]
        for table in ("identity", "n1"):
            assert b[table].keys() == new[table].keys()
            diff = [k for k in new[table] if new[table][k]["sha256"] != b[table][k]["sha256"]]
            res[f"{table}_vs_{base}"] = {"keys": len(new[table]), "equal": len(new[table]) - len(diff),
                                         "differ": len(diff), "differ_keys": diff[:40]}
    for a, r in arms.items():
        n1 = r["n1"]
        res[f"n1_{a}"] = {"keys": len(n1), "finite": sum(x["finite"] for x in n1.values()),
                          "nonfinite": sum(not x["finite"] for x in n1.values()),
                          "max_err_finite": max((x["max_err"] for x in n1.values() if x["finite"]), default=None),
                          "identity_nonfinite": sum(not x["finite"] for x in r["identity"].values())}
        res[f"observations_{a}"] = r["observations"]
    tol_ok = True
    for k, x in new["n1"].items():
        tol = 4e-3 if "float16" in k or "fp16" in k else 1e-5 * 3.5      # max |oracle| <= ~3.5 for N(0,1) V
        tol_ok &= x["finite"] and x["max_err"] <= tol
    res["G2a_mf2_2_n1_finite_and_correct"] = bool(tol_ok)
    res["G2b_identity_vs_mf2_and_fd32b6d"] = (res["identity_vs_mf2"]["differ"] == 0 and res["identity_vs_fd32b6d"]["differ"] == 0)
    res["PN1_n1_bytes_equal_mf2_and_fd32b6d"] = (res["n1_vs_mf2"]["differ"] == 0 and res["n1_vs_fd32b6d"]["differ"] == 0)
    with open(out, "x") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("observations") and k != "engines"}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "arm":
        arm(sys.argv[2], sys.argv[3])
    else:
        compare(sys.argv[2], sys.argv[3], sys.argv[4])
