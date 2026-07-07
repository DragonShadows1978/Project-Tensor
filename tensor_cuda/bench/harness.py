"""Phase 0.1 microbenchmark harness (KERNEL_OPT_IMPLEMENTATION_PLAN.md).

Timing protocol (per task spec): time.perf_counter bracketed by
tc.synchronize(), 10 warmup + 100 timed reps, median + IQR, fixed seeds.

Evidence class: kernel sweep (speed / memory shape / reconstruction error /
output deviation only — never model quality; see plan House Rules).

Usage:
    PYTHONPATH=/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 bench/harness.py --smoke
    PYTHONPATH=/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 bench/harness.py --full

Power constraint (plan House Rules): a single invocation must stay under the
10-minute GPU-draw bound. --smoke runs the smallest shapes only, one rep
count reduction, and is designed to finish in well under a minute. --full
runs the entire Phase-0.1 matrix and its wall time is NOT bounded by this
script — the caller is responsible for keeping any one run under 10 minutes
(use --kernels/--models/--only-shape to slice it if the full matrix would
run long).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import traceback
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc  # noqa: E402

from bench import geometries as G  # noqa: E402

SEED = 20260707
ARTIFACT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "artifacts", "kernel_opt",
)

# 12 GB card; leave headroom for the CUDA context + allocator pool + whatever
# else is resident (measured ~6.6 GB already used by another process at
# harness-authoring time). Shapes are pre-filtered against this budget and
# any skip is recorded, never silently capped (task spec).
VRAM_BUDGET_BYTES = 11.0 * (1024 ** 3)


def _free_bytes():
    """Best-effort free-VRAM read. Falls back to a conservative constant if
    the binding doesn't expose memory info (never used to silently proceed
    past a real OOM — actual kernel calls are still try/except-guarded)."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            timeout=5,
        )
        return int(out.decode().strip().splitlines()[0]) * 1024 * 1024
    except Exception:
        return None


def median_iqr(samples_ms):
    s = sorted(samples_ms)
    med = statistics.median(s)
    q1 = np.percentile(s, 25)
    q3 = np.percentile(s, 75)
    return med, (q3 - q1)


def timed(fn, warmup, reps):
    """time.perf_counter bracketed by tc.synchronize(), per task spec."""
    for _ in range(warmup):
        fn()
    tc.synchronize()
    samples = []
    for _ in range(reps):
        tc.synchronize()
        t0 = time.perf_counter()
        fn()
        tc.synchronize()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1e3)
    med, iqr = median_iqr(samples)
    return {"median_ms": med, "iqr_ms": iqr, "reps": reps}


def bytes_for_shape(*shapes_dtypes):
    total = 0
    for shape, itemsize in shapes_dtypes:
        n = 1
        for d in shape:
            n *= d
        total += n * itemsize
    return total


class SkipShape(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def check_budget(needed_bytes, margin=1.3):
    """Raise SkipShape if the shape's estimated working-set (with margin for
    transients: causal mask, bulk/rank score tensors, etc.) won't fit."""
    free = _free_bytes()
    budget = min(VRAM_BUDGET_BYTES, free) if free is not None else VRAM_BUDGET_BYTES
    needed = needed_bytes * margin
    if needed > budget:
        raise SkipShape(
            f"estimated {needed / 1e9:.2f} GB > budget {budget / 1e9:.2f} GB "
            f"(free={'unknown' if free is None else f'{free/1e9:.2f} GB'})"
        )


# ---------------------------------------------------------------------------
# Per-kernel benchmark functions. Each returns a dict entry or raises
# SkipShape (caught by the runner, recorded with a reason — never silent).
# ---------------------------------------------------------------------------

def bench_causal_softmax(shape, dtype, reps, warmup):
    """functional.scaled_dot_product_attention's fused causal-softmax path
    (tc.causal_softmax on (B, H*L, 1, S) folded scores) — see functional.py."""
    B, H, L, S, D = shape["B"], shape["H"], shape["L"], shape["S"], shape["D"]
    check_budget(bytes_for_shape(((B, H, L, S), 4), ((B, H, L, D), 4)))
    rng = np.random.default_rng(SEED)
    scores = (rng.standard_normal((B, H * L, 1, S)).astype(np.float32) * 0.1)
    scores_t = tc.tensor(scores)
    with tc.no_grad():
        def run():
            tc.causal_softmax(scores_t)
        r = timed(run, warmup, reps)
    del scores_t
    return r


def bench_apa_blend_softmax(shape, dtype, reps, warmup, sink):
    """tc.apa_blend_softmax / apa_blend_softmax_sink — the fused
    threshold+select+softmax kernel inside _cublas_blend_attention
    (GraftRepository/core/mistral7b_tc.py:_cublas_blend_attention)."""
    B, H, L, S = shape["B"], shape["H"], shape["L"], shape["S"]
    check_budget(bytes_for_shape(((B, H, L, S), 4), ((B, H, L, S), 4)))
    rng = np.random.default_rng(SEED)
    bulk = (rng.standard_normal((B, H, L, S)).astype(np.float32) * 0.2)
    rank = (bulk + rng.standard_normal(bulk.shape).astype(np.float32) * 0.05)
    bulk_t, rank_t = tc.tensor(bulk), tc.tensor(rank)
    zthr = 0.4
    with tc.no_grad():
        if sink:
            sinks = tc.tensor((rng.standard_normal(H) * 0.3).astype(np.float32))

            def run():
                tc.apa_blend_softmax_sink(bulk_t, rank_t, sinks, zthr)
        else:
            def run():
                tc.apa_blend_softmax(bulk_t, rank_t, zthr)
        r = timed(run, warmup, reps)
    del bulk_t, rank_t
    return r


def bench_apa_selective(shape, dtype, reps, warmup, sink):
    """tc.apa_selective_attention / apa_selective_attention_sink — the fused
    O(L) online-softmax selective-APA kernel (GQA-aware). Used at the
    near-full-context GPT-OSS shape per ledger 09:20 (blend fast path caps
    at fast_max_seq=4096, so large S falls to this kernel)."""
    B, H, KVH, L, S, D = (
        shape["B"], shape["H"], shape["KVH"], shape["L"], shape["S"], shape["D"]
    )
    check_budget(bytes_for_shape(
        ((B, H, L, D), 4), ((B, KVH, S, D), 4), ((B, KVH, S, D), 4), ((B, KVH, S, D), 4)
    ))
    rng = np.random.default_rng(SEED)
    q = (rng.standard_normal((B, H, L, D)) * 0.1).astype(np.float32)
    k = (rng.standard_normal((B, KVH, S, D)) * 0.1).astype(np.float32)
    v = (rng.standard_normal((B, KVH, S, D)) * 0.1).astype(np.float32)
    kq = (k + rng.standard_normal((B, KVH, S, D)) * 0.01).astype(np.float32)
    scale = 1.0 / math.sqrt(D)
    from tensor_cuda.quant import _norm_ppf
    z = _norm_ppf(1.0 - 0.15)
    q_t, k_t, kq_t, v_t = tc.tensor(q), tc.tensor(k), tc.tensor(kq), tc.tensor(v)
    with tc.no_grad():
        if sink:
            sinks = tc.tensor((rng.standard_normal(H) * 0.3).astype(np.float32))

            def run():
                tc.apa_selective_attention_sink(
                    q_t, k_t, kq_t, v_t, sinks, float(scale), float(z), True)
        else:
            def run():
                tc.apa_selective_attention(
                    q_t, k_t, kq_t, v_t, float(scale), float(z), True)
        r = timed(run, warmup, reps)
    del q_t, k_t, kq_t, v_t
    return r


def _make_q4_0(N, K, group, rng):
    q = rng.integers(0, 16, size=(N, K), dtype=np.uint8)
    s = (rng.random((N, K // group), dtype=np.float32) * 0.02 + 1e-4).astype(np.float16)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return packed, s


def bench_int4(shape, dtype, reps, warmup, fused):
    """tc.int4_linear / int4_linear_fused. M=1 rows == GEMV path (decode),
    M>1 == GEMM path (prefill) — same entry point, dispatch happens inside
    the kernel (kernels.cu int4_gemv_kernel / int4_gemm_fused_kernel)."""
    M, N, K = shape["M"], shape["N"], shape["K"]
    group = 128
    check_budget(bytes_for_shape(((M, K), 4), ((N, K), 1), ((N, K // group), 2)))
    rng = np.random.default_rng(SEED)
    packed, s = _make_q4_0(N, K, group, rng)
    x = rng.standard_normal((M, K)).astype(np.float32)
    x_t, packed_t, s_t = tc.tensor(x), tc.tensor(packed, dtype="uint8"), tc.tensor(s, dtype="float16")
    empty_z = tc.tensor(np.zeros((0,), np.float16), dtype="float16")
    fn = tc.int4_linear_fused if fused else tc.int4_linear
    with tc.no_grad():
        def run():
            fn(x_t, packed_t, s_t, empty_z, group)
        r = timed(run, warmup, reps)
    del x_t, packed_t, s_t
    return r


def bench_intn(shape, dtype, reps, warmup, bits, fused):
    """tc.intn_linear / intn_linear_fused at bits in {2,3}."""
    M, N, K = shape["M"], shape["N"], shape["K"]
    group = 64
    check_budget(bytes_for_shape(((M, K), 4), ((N, K), 1), ((N, K // group), 2)))
    rng = np.random.default_rng(SEED)
    from tensor_cuda.quantization import quantize_affine_per_group
    w = rng.standard_normal((N, K)).astype(np.float32) * 0.1
    q = quantize_affine_per_group(w, bits, group)
    x = rng.standard_normal((M, K)).astype(np.float32)
    x_t = tc.tensor(x, dtype="float32")
    packed_t = tc.tensor(q.packed, dtype="uint8")
    scales_t = tc.tensor(q.scales, dtype="float16")
    zeros_t = tc.tensor(q.zeros, dtype="float16")
    fn = tc.intn_linear_fused if fused else tc.intn_linear
    with tc.no_grad():
        def run():
            fn(x_t, packed_t, scales_t, zeros_t, bits, K, group)
        r = timed(run, warmup, reps)
    del x_t, packed_t, scales_t, zeros_t
    return r


def bench_mxfp4(shape, dtype, reps, warmup, expert):
    """tc.mxfp4_linear (GPT-OSS dense/packed expert weight, M=1 == GEMV,
    M>1 == GEMM) or tc.mxfp4_linear_expert (resident multi-expert gather)."""
    M, N, K = shape["M"], shape["N"], shape["K"]
    group = 32
    groups = K // group
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((M, K)).astype(np.float32) * 0.1
    if expert:
        E = shape["E"]
        check_budget(bytes_for_shape(((M, K), 4), ((E, N, groups, 16), 1), ((E, N, groups), 1)))
        blocks = rng.integers(0, 256, size=(E, N, groups, 16), dtype=np.uint8)
        scales = rng.integers(124, 131, size=(E, N, groups), dtype=np.uint8)
        x_t = tc.tensor(x, dtype="float32")
        blocks_t = tc.tensor(blocks, dtype="uint8")
        scales_t = tc.tensor(scales, dtype="uint8")
        eidx = E // 2
        with tc.no_grad():
            def run():
                tc.mxfp4_linear_expert(x_t, blocks_t, scales_t, eidx)
            r = timed(run, warmup, reps)
    else:
        check_budget(bytes_for_shape(((M, K), 4), ((N, groups, 16), 1), ((N, groups), 1)))
        blocks = rng.integers(0, 256, size=(N, groups, 16), dtype=np.uint8)
        scales = rng.integers(124, 131, size=(N, groups), dtype=np.uint8)
        x_t = tc.tensor(x, dtype="float32")
        blocks_t = tc.tensor(blocks, dtype="uint8")
        scales_t = tc.tensor(scales, dtype="uint8")
        with tc.no_grad():
            def run():
                tc.mxfp4_linear(x_t, blocks_t, scales_t)
            r = timed(run, warmup, reps)
    return r


def bench_rope(shape, dtype, reps, warmup):
    """tc.rope_apply — fused RoPE kernel."""
    B, H, L, D = shape["B"], shape["H"], shape["L"], shape["D"]
    T = L + 8
    check_budget(bytes_for_shape(((B, H, L, D), 4), ((T, D), 4), ((T, D), 4)))
    rng = np.random.default_rng(SEED)
    inv = 1.0 / (1e4 ** (np.arange(0, D, 2, np.float32) / D))
    ang = np.arange(T, dtype=np.float32)[:, None] * inv[None, :]
    emb = np.concatenate([ang, ang], -1)
    cs = tc.tensor(np.cos(emb))
    sn = tc.tensor(np.sin(emb))
    x = tc.tensor(rng.standard_normal((B, H, L, D)).astype(np.float32))
    with tc.no_grad():
        def run():
            tc.rope_apply(x, cs, sn, 0)
        r = timed(run, warmup, reps)
    return r


def bench_rms_norm(shape, dtype, reps, warmup):
    """tc.rms_norm — fused single-kernel RMSNorm."""
    B, L, Dh = shape["B"], shape["L"], shape["Dh"]
    check_budget(bytes_for_shape(((B, L, Dh), 4), ((Dh,), 4)))
    rng = np.random.default_rng(SEED)
    x = tc.tensor(rng.standard_normal((B, L, Dh)).astype(np.float32))
    w = tc.tensor(np.ones(Dh, dtype=np.float32))
    with tc.no_grad():
        def run():
            tc.rms_norm(x, w, 1e-6)
        r = timed(run, warmup, reps)
    return r


def bench_swiglu(shape, dtype, reps, warmup):
    """SwiGLU elementwise pair as composed today:
    silu(gate_proj(x)) * up_proj(x) (GraftRepository/core/mistral7b_tc.py:
    SwiGLU_TC.__call__). This benches the elementwise silu+mul pair alone
    (gate/up tensors pre-projected) — the two elementwise launches Phase 5
    targets for fusion, isolated from the surrounding GEMMs."""
    B, L, F = shape["B"], shape["L"], shape["F"]
    check_budget(bytes_for_shape(((B, L, F), 4), ((B, L, F), 4)))
    rng = np.random.default_rng(SEED)
    gate = tc.tensor(rng.standard_normal((B, L, F)).astype(np.float32) * 0.1)
    up = tc.tensor(rng.standard_normal((B, L, F)).astype(np.float32) * 0.1)
    with tc.no_grad():
        def run():
            gate.silu() * up
        r = timed(run, warmup, reps)
    return r


# ---------------------------------------------------------------------------
# Shape matrix construction (model geometries x decode/prefill shapes).
# ---------------------------------------------------------------------------

def build_shape_matrix(models, decode_s, prefill_l, include_gpt_oss_near_full):
    """Yields (kernel_family, label, shape_dict, extra) tuples."""
    entries = []

    for mname in models:
        geo = G.ALL_MODELS[mname]

        attn_keys = [k for k in geo if k.startswith("attn")]
        for ak in attn_keys:
            a: G.AttnGeometry = geo[ak]
            B = 1
            for S in decode_s:
                entries.append((
                    "causal_softmax", f"{mname}/{ak}/decode_S{S}",
                    {"B": B, "H": a.num_heads, "L": 1, "S": S, "D": a.head_dim}, {}
                ))
                entries.append((
                    "apa_blend_softmax", f"{mname}/{ak}/decode_S{S}",
                    {"B": B, "H": a.num_heads, "L": 1, "S": S}, {"sink": a.sinks}
                ))
                entries.append((
                    "apa_selective", f"{mname}/{ak}/decode_S{S}",
                    {"B": B, "H": a.num_heads, "KVH": a.num_kv_heads, "L": 1,
                     "S": S, "D": a.head_dim}, {"sink": a.sinks}
                ))
            for L in prefill_l:
                entries.append((
                    "causal_softmax", f"{mname}/{ak}/prefill_L{L}",
                    {"B": B, "H": a.num_heads, "L": L, "S": L, "D": a.head_dim}, {}
                ))
                entries.append((
                    "apa_blend_softmax", f"{mname}/{ak}/prefill_L{L}",
                    {"B": B, "H": a.num_heads, "L": L, "S": L}, {"sink": a.sinks}
                ))
                entries.append((
                    "apa_selective", f"{mname}/{ak}/prefill_L{L}",
                    {"B": B, "H": a.num_heads, "KVH": a.num_kv_heads, "L": L,
                     "S": L, "D": a.head_dim}, {"sink": a.sinks}
                ))
            entries.append((
                "rope", f"{mname}/{ak}/decode_S{decode_s[0]}",
                {"B": B, "H": a.num_heads, "L": 1, "D": a.head_dim}, {}
            ))
            entries.append((
                "rope", f"{mname}/{ak}/prefill_L{prefill_l[-1]}",
                {"B": B, "H": a.num_heads, "L": prefill_l[-1], "D": a.head_dim}, {}
            ))

        ffn: G.FFNGeometry = geo["ffn"]
        for L in (1,) + prefill_l:
            label = "decode_S1" if L == 1 else f"prefill_L{L}"
            entries.append((
                "rms_norm", f"{mname}/{label}",
                {"B": 1, "L": L, "Dh": ffn.hidden_dim}, {}
            ))
            entries.append((
                "swiglu", f"{mname}/{label}",
                {"B": 1, "L": L, "F": ffn.intermediate_dim}, {}
            ))

        for M, mlabel in ((1, "gemv_decode"), (512, "gemm_prefill_L512"), (2048, "gemm_prefill_L2048")):
            if M in prefill_l or M == 1:
                entries.append((
                    "int4", f"{mname}/{mlabel}",
                    {"M": M, "N": ffn.intermediate_dim, "K": ffn.hidden_dim}, {"fused": True}
                ))
                entries.append((
                    "int4_two_stage", f"{mname}/{mlabel}",
                    {"M": M, "N": ffn.intermediate_dim, "K": ffn.hidden_dim}, {"fused": False}
                ))
                for bits in (2, 3):
                    entries.append((
                        f"intn{bits}", f"{mname}/{mlabel}",
                        {"M": M, "N": ffn.intermediate_dim, "K": ffn.hidden_dim},
                        {"bits": bits, "fused": True}
                    ))

        if "moe" in geo:
            moe: G.MoEGeometry = geo["moe"]
            for M, mlabel in ((1, "gemv_decode"), (512, "gemm_prefill_L512")):
                if M in prefill_l or M == 1:
                    entries.append((
                        "mxfp4", f"{mname}/{mlabel}/gate_up",
                        {"M": M, "N": 2 * moe.intermediate_dim, "K": moe.hidden_dim}, {"expert": False}
                    ))
                    entries.append((
                        "mxfp4", f"{mname}/{mlabel}/down",
                        {"M": M, "N": moe.hidden_dim, "K": moe.intermediate_dim}, {"expert": False}
                    ))
                    entries.append((
                        "mxfp4_expert_resident", f"{mname}/{mlabel}/gate_up",
                        {"M": M, "N": 2 * moe.intermediate_dim, "K": moe.hidden_dim, "E": moe.num_experts},
                        {"expert": True}
                    ))

    if include_gpt_oss_near_full and "gpt_oss20b" in models:
        geo = G.ALL_MODELS["gpt_oss20b"]
        a = geo["attn_full_sink"]
        S = G.GPT_OSS_NEAR_FULL_S
        entries.append((
            "apa_selective", f"gpt_oss20b/attn_full_sink/near_full_S{S}",
            {"B": 1, "H": a.num_heads, "KVH": a.num_kv_heads, "L": 1, "S": S, "D": a.head_dim},
            {"sink": True}
        ))

    return entries


DISPATCH = {
    "causal_softmax": lambda shape, dtype, reps, warmup, extra: bench_causal_softmax(shape, dtype, reps, warmup),
    "apa_blend_softmax": lambda shape, dtype, reps, warmup, extra: bench_apa_blend_softmax(shape, dtype, reps, warmup, extra["sink"]),
    "apa_selective": lambda shape, dtype, reps, warmup, extra: bench_apa_selective(shape, dtype, reps, warmup, extra["sink"]),
    "int4": lambda shape, dtype, reps, warmup, extra: bench_int4(shape, dtype, reps, warmup, extra["fused"]),
    "int4_two_stage": lambda shape, dtype, reps, warmup, extra: bench_int4(shape, dtype, reps, warmup, extra["fused"]),
    "intn2": lambda shape, dtype, reps, warmup, extra: bench_intn(shape, dtype, reps, warmup, 2, extra["fused"]),
    "intn3": lambda shape, dtype, reps, warmup, extra: bench_intn(shape, dtype, reps, warmup, 3, extra["fused"]),
    "mxfp4": lambda shape, dtype, reps, warmup, extra: bench_mxfp4(shape, dtype, reps, warmup, extra["expert"]),
    "mxfp4_expert_resident": lambda shape, dtype, reps, warmup, extra: bench_mxfp4(shape, dtype, reps, warmup, extra["expert"]),
    "rope": lambda shape, dtype, reps, warmup, extra: bench_rope(shape, dtype, reps, warmup),
    "rms_norm": lambda shape, dtype, reps, warmup, extra: bench_rms_norm(shape, dtype, reps, warmup),
    "swiglu": lambda shape, dtype, reps, warmup, extra: bench_swiglu(shape, dtype, reps, warmup),
}


def run_matrix(entries, reps, warmup, dtype="float32", kernel_filter=None, verbose=True):
    results = {}
    skips = []
    for kernel, label, shape, extra in entries:
        if kernel_filter and kernel not in kernel_filter:
            continue
        key = f"{kernel}::{label}"
        try:
            r = DISPATCH[kernel](shape, dtype, reps, warmup, extra)
            r.update({"shape": shape, "dtype": dtype, "kernel": kernel, "label": label})
            results[key] = r
            if verbose:
                print(f"[OK]   {key:70s} median={r['median_ms']:.4f}ms iqr={r['iqr_ms']:.4f}ms")
        except SkipShape as e:
            entry = {"kernel": kernel, "label": label, "shape": shape, "dtype": dtype,
                      "skipped": True, "skip_reason": str(e.reason)}
            results[key] = entry
            skips.append(key)
            if verbose:
                print(f"[SKIP] {key:70s} reason={e.reason}")
        except Exception as e:  # noqa: BLE001 - record and continue, never abort the sweep
            tb = traceback.format_exc()
            entry = {"kernel": kernel, "label": label, "shape": shape, "dtype": dtype,
                      "skipped": True, "skip_reason": f"ERROR: {e}", "traceback": tb}
            results[key] = entry
            skips.append(key)
            if verbose:
                print(f"[ERR]  {key:70s} reason={e}")
    return results, skips


def print_summary_table(results):
    print("\n" + "=" * 100)
    print(f"{'kernel':22s} {'label':45s} {'median_ms':>12s} {'iqr_ms':>10s}")
    print("-" * 100)
    for key in sorted(results):
        r = results[key]
        if r.get("skipped"):
            print(f"{r['kernel']:22s} {r['label']:45s} {'SKIP':>12s} {r['skip_reason'][:40]}")
        else:
            print(f"{r['kernel']:22s} {r['label']:45s} {r['median_ms']:12.4f} {r['iqr_ms']:10.4f}")
    print("=" * 100)
    n_ok = sum(1 for r in results.values() if not r.get("skipped"))
    n_skip = sum(1 for r in results.values() if r.get("skipped"))
    print(f"{n_ok} ok, {n_skip} skipped, {len(results)} total\n")


def main():
    ap = argparse.ArgumentParser(description="Phase 0.1 kernel microbench harness")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true",
                       help="Smallest shapes only, reduced reps. Bounded to a few minutes.")
    mode.add_argument("--full", action="store_true",
                       help="Full Phase-0.1 matrix (decode S in {512,2048,8192}, "
                            "prefill L in {512,2048}, all models, GPT-OSS near-full-context).")
    ap.add_argument("--models", nargs="+", default=list(G.ALL_MODELS.keys()),
                     choices=list(G.ALL_MODELS.keys()))
    ap.add_argument("--kernels", nargs="+", default=None,
                     help="Restrict to these kernel families (see DISPATCH keys).")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--out", default=None, help="Output JSON path (default: auto-named under artifacts/kernel_opt/).")
    args = ap.parse_args()

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    if args.smoke:
        reps = args.reps or 20
        warmup = args.warmup or 5
        entries = build_shape_matrix(
            models=args.models, decode_s=(512,), prefill_l=(512,),
            include_gpt_oss_near_full=False,
        )
        tag = "smoke"
    else:
        reps = args.reps or 100
        warmup = args.warmup or 10
        entries = build_shape_matrix(
            models=args.models, decode_s=G.DECODE_S, prefill_l=G.PREFILL_L,
            include_gpt_oss_near_full=("gpt_oss20b" in args.models),
        )
        tag = "full"

    print(f"Mode={tag} models={args.models} reps={reps} warmup={warmup} "
          f"entries={len(entries)} kernel_filter={args.kernels}")

    t_start = time.time()
    results, skips = run_matrix(entries, reps, warmup, kernel_filter=args.kernels)
    wall = time.time() - t_start

    print_summary_table(results)
    print(f"Total wall time: {wall:.1f}s")

    out_path = args.out
    if out_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(ARTIFACT_DIR, f"kernel_microbench_{tag}_{ts}.json")

    artifact = {
        "artifact": os.path.basename(out_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": tag,
        "models": args.models,
        "seed": SEED,
        "reps": reps,
        "warmup": warmup,
        "total_wall_seconds": wall,
        "timing_method": "time.perf_counter bracketed by tc.synchronize()",
        "vram_budget_bytes": VRAM_BUDGET_BYTES,
        "num_entries": len(entries),
        "num_skipped": len(skips),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    main()
