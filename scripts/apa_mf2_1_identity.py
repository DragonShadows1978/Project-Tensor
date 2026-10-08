"""APA-MF2.1 G3(b): byte-identity arms, ONE ARM PER PROCESS (separate engine builds and
separate package imports -- never two runs of the new code).

usage: apa_mf2_1_identity.py <arm-name> <out.json>
The arm is selected by the environment (scripts/apa_mf2_1_identity.sh):
  before        MF2 engine (artifacts/apa_mf2/build) + MF2 package (artifacts/apa_mf2_1/before/pkg)
  after         MF2.1 engine (artifacts/apa_mf2_1/build) + tree package
  engine_iso    MF2 engine + tree package   (isolates the engine/sentinel change for
                apa_quant_attention p=0.15, whose selection changes by design in item 3)
Each output tensor is recorded as sha256(bytes) + shape + dtype + finiteness; the
compare script (apa_mf2_1_identity_compare.py) checks byte equality across arms.

Cells:
  real    Qwen3-1.7B L16, Qwen3-4B L21 captures (sha-pinned), (L,S) in {(1,8192),(16,16),
          (64,8192),(512,8192)}, H=2 (q heads 0,1; kv head per GQA group), fp32/bf16/fp16:
          sdpa_infer, sdpa_grad (out, dq, dk, dv), causal_softmax on the call's scores,
          apa p=1.0, apa p=0.15.
  reduce  reduce_max / reduce_min: trailing fast path and generic kernel, 3 dtypes,
          magnitudes 1 .. 3.3e38, rows mixing finite and +-inf (>= 1 finite per row).
  fused   fused_sdpa_noncausal (fp32/fp16) and apa_int4_sdpa_noncausal (refine_all 0/1),
          Lk in {1,2,3,9,200} (Lk < 4 exercises empty warps).
  grapa   GRAPA GQAAttention forward/backward, MF2 script geometry (read-only import).
  design  rows entirely beyond the old sentinels (expected to CHANGE; reported apart).
Prior art: none for the harness (mine); captures from APA-SPQ3.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import time

import numpy as np

import tensor_cuda as tc
from tensor_cuda import functional as F
from tensor_cuda import quant as Q

CAP = "/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_spq3/artifacts/apa_spq3/captures"
DTYPES = ("float32", "bfloat16", "float16")
REC = {}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def put(key, t):
    a = t.float().numpy() if hasattr(t, "float") else np.asarray(t)
    raw = t.numpy() if (hasattr(t, "dtype") and getattr(t, "dtype", None) == "float32") else a
    REC[key] = {"sha256": sha(np.ascontiguousarray(raw).tobytes() + str(t.dtype if hasattr(t, "dtype") else a.dtype).encode()),
                "shape": list(a.shape), "dtype": str(getattr(t, "dtype", a.dtype)),
                "finite": bool(np.isfinite(a).all())}


def file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def captures(model, layer, L, S, H=2):
    d = os.path.join(CAP, model)
    man = json.load(open(os.path.join(d, "manifest.json")))
    total = man["S"]
    group = man["config"]["num_attention_heads"] // man["config"]["num_key_value_heads"]
    out = []
    for kind in "qkv":
        p = os.path.join(d, f"L{layer:02d}_{kind}.npy")
        assert file_sha(p) == man["files"][os.path.basename(p)]["sha256"], "capture pin drift"
        a = np.load(p, mmap_mode="r")
        hs = np.arange(H) if kind == "q" else np.arange(H) // group
        sl = np.asarray(a[hs, total - L:total] if kind == "q" else a[hs, total - S:total])
        out.append((sl.astype(np.uint32) << 16).view(np.float32)[None])
    return out


def real():
    for model, layer in (("Qwen3-1.7B", 16), ("Qwen3-4B", 21)):
        for L, S in ((1, 8192), (16, 16), (64, 8192), (512, 8192)):
            arrs = captures(model, layer, L, S)
            for dt in DTYPES:
                base = f"real/{model}/L{layer}/{L}x{S}/{dt}"
                causal = L > 1
                xs = [tc.tensor(a, dtype=dt) for a in arrs]
                with tc.no_grad():
                    put(base + "/sdpa_infer", F.scaled_dot_product_attention(*xs, is_causal=causal))
                    sc = tc.matmul(xs[0], xs[1], alpha=128 ** -0.5, trans_b=True)
                    put(base + "/causal_softmax", tc.causal_softmax(sc))
                    del sc
                leaves = [tc.tensor(a, requires_grad=True) for a in arrs]
                ts = [x if dt == "float32" else x.astype(dt) for x in leaves]
                o = F.scaled_dot_product_attention(*ts, is_causal=causal)
                o.sum().backward()
                put(base + "/sdpa_grad/out", o)
                for n, x in zip("qkv", leaves):
                    put(base + f"/sdpa_grad/d{n}", x.grad)
                del o, leaves, ts
                with tc.no_grad():
                    for p in (1.0, 0.15):
                        put(base + f"/apa_p{p}", Q.apa_quant_attention(*xs, bulk_bits=8, refine_percentile=p,
                                                                        is_causal=causal))
                print(base, flush=True)


def reduce_sweep():
    rng = np.random.default_rng(20261007)
    shapes = [((7, 333), [-1]), ((333, 7), [0]), ((4, 5, 300), [1, 2]), ((300, 4, 5), [0]),
              ((64, 1), [-1]), ((3, 1025), [-1]), ((2, 3), [0, 1])]
    for dt in DTYPES:
        for (shape, axes) in shapes:
            for mag in (1.0, 1e3, 1e30, 3.3e38):
                if dt == "float16" and mag > 6e4:
                    continue
                a = (rng.uniform(-1, 1, shape) * mag).astype(np.float32)
                for mixed in (False, True):
                    b = a.copy()
                    if mixed:
                        flat = b.reshape(-1)
                        idx = rng.choice(flat.size, size=max(1, flat.size // 7), replace=False)
                        flat[idx] = np.where(rng.random(idx.size) < 0.5, -np.inf, np.inf)
                        # keep >= 1 finite value per reduced row: restore the first element of
                        # every output position along the reduced axes
                        b = flat.reshape(shape)
                        sl = [slice(None)] * len(shape)
                        for ax in axes:
                            sl[ax] = 0
                        b[tuple(sl)] = a[tuple(sl)]
                    t = tc.tensor(b, dtype=dt)
                    for op in ("max", "min"):
                        put(f"reduce/{dt}/{shape}/{axes}/{mag}/{'mixed' if mixed else 'finite'}/{op}",
                            getattr(t, op)(axes, False))


def design_rows():
    """Rows entirely beyond the old start values: EXPECTED to change (reported apart)."""
    lo = np.full((2, 300), -3.402e38, np.float32); lo[1] = -np.inf
    put("design/max_below_old_sentinel", tc.tensor(lo).max([-1], False))
    put("design/min_above_old_sentinel", tc.tensor(-lo).min([-1], False))
    w = np.array([[[[-3.3e38, 0.0], [-3.3e38, 0.0]]]], np.float32)
    with tc.no_grad():
        put("design/causal_softmax_-3.3e38", tc.causal_softmax(tc.tensor(w)))


def fused():
    rng = np.random.default_rng(5039)
    for dt in ("float32", "float16"):
        for Lq, Lk, D in ((7, 1, 64), (7, 2, 64), (5, 3, 128), (9, 9, 64), (64, 200, 128)):
            q, k, v = (rng.normal(0, 1, (1, 2, n, D)).astype(np.float32) for n in (Lq, Lk, Lk))
            xs = [tc.tensor(a, dtype=dt) for a in (q, k, v)]
            with tc.no_grad():
                put(f"fused/sdpa_noncausal/{dt}/{Lq}x{Lk}x{D}", tc.fused_sdpa_noncausal(*xs, D ** -0.5))
                for ra in (False, True):
                    put(f"fused/apa_int4/{dt}/{Lq}x{Lk}x{D}/refine_all={ra}",
                        tc.apa_int4_sdpa_noncausal(*xs, D ** -0.5, 1.0364, refine_all=ra))


def grapa():
    sys.path.insert(0, "/mnt/ForgeRealm/GRAPA-Native-LLM")
    from grapa.attention import GQAAttention
    from grapa.model import GRAPAConfig
    cfg = GRAPAConfig()
    np.random.seed(20261007)
    attn = GQAAttention(cfg)
    for L in (64, 128, 256):
        rng = np.random.default_rng(1000 + L)
        x_np = rng.normal(0, 1, (1, L, cfg.d_model)).astype(np.float32)
        cos, sin = F.rope_tables(L, cfg.head_dim, "cuda", base=cfg.rope_base)
        for force in (False, True):
            attn.force_selective = force
            with tc.no_grad():
                put(f"grapa/L{L}/force={force}/infer", attn(tc.tensor(x_np), cos, sin, refine_percentile=1.0))
            attn.zero_grad()
            x = tc.tensor(x_np, requires_grad=True)
            o = attn(x, cos, sin, refine_percentile=1.0)
            o.sum().backward()
            put(f"grapa/L{L}/force={force}/grad/out", o)
            put(f"grapa/L{L}/force={force}/grad/dx", x.grad)
            put(f"grapa/L{L}/force={force}/grad/dWq", attn.q_proj.weight.grad)


def main():
    arm, out = sys.argv[1], sys.argv[2]
    t0 = time.time()
    meta = {"arm": arm, "tensor_cuda": tc.__file__, "engine": tc._C.__file__,
            "engine_sha256": file_sha(tc._C.__file__), "functional_sha256": file_sha(F.__file__),
            "quant_sha256": file_sha(Q.__file__)}
    print(json.dumps(meta), flush=True)
    for fn in (reduce_sweep, design_rows, fused, grapa, real):
        fn()
        print("done", fn.__name__, len(REC), round(time.time() - t0, 1), flush=True)
    meta["t_s"] = time.time() - t0
    with open(out, "x") as f:
        json.dump({"meta": meta, "rec": REC}, f, indent=0)
    print("WROTE", out, len(REC), flush=True)


if __name__ == "__main__":
    main()
