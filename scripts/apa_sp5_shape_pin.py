#!/usr/bin/env python3
"""APA-SP5 item-2 kernel/shape pin: does the single-pass entry already
instantiate GPT-OSS-20B's shape (GQA H=64 / KVH=8, D=VD=64) WITH learned
attention sinks, and does it agree with a dense fp64 NumPy reference that
folds the sink into the softmax denominator only?

Prior art at this code site:
  - Online softmax running max + rescale: Milakov & Gimelshein (2018);
    FlashAttention-2 (Dao, 2023) for the fused tiled form.
  - Running-max selection criterion: BLASST (Yuan et al., arXiv 2512.12087,
    MLSys 2026) -- APA applies it to PRECISION (refine) rather than sparsity
    (drop); see /mnt/Shared/APA_SP_Prior_Art_Comparison_2026-09-06.md.
  - Weight-proportional quantization error: ThriftAttention (Sharratt,
    arXiv 2605.23081, 2026).
  - Learned attention sinks in the denominator: GPT-OSS model card
    (OpenAI, 2025); the port's sink_attention_tc implements the same.
  - Bulk key quantizer: TurboQuant-style rotation+codebook, tensor_cuda.quant.
Mine here: nothing algorithmic -- this is a shape/behaviour pin only.
"""
import json, os, sys, time
import numpy as np

ROOT = '/mnt/ForgeRealm/Project-Tensor-wt-apa-sp5'
os.environ['TC_APA_SP'] = '1'
sys.dont_write_bytecode = True
sys.path[:0] = [ROOT + '/artifacts/apa_sp5/build', ROOT + '/tensor_cuda']
import tensor_cuda as tc
from tensor_cuda.quant import _quantize_keys, _tables, _norm_ppf


def dense_ref(q, k, v, sinks, scale, causal, kvh):
    """fp64 dense reference. Sink logit joins the softmax denominator and
    contributes no value row (GPT-OSS convention, port sink_attention_tc)."""
    B, H, L, D = q.shape
    S = k.shape[2]
    g = H // kvh
    out = np.zeros((B, H, L, v.shape[3]), dtype=np.float64)
    for b in range(B):
        for h in range(H):
            kh = k[b, h // g].astype(np.float64)
            vh = v[b, h // g].astype(np.float64)
            for i in range(L):
                lim = (S - L + i + 1) if causal else S
                s = (q[b, h, i].astype(np.float64) @ kh[:lim].T) * scale
                if sinks is None:
                    m = s.max()
                    w = np.exp(s - m)
                    out[b, h, i] = (w @ vh[:lim]) / w.sum()
                else:
                    sk = float(sinks[h])
                    m = max(s.max(), sk)
                    w = np.exp(s - m)
                    out[b, h, i] = (w @ vh[:lim]) / (w.sum() + np.exp(sk - m))
    return out


def blend_ref(q, k, kq, v, sinks, scale, delta, causal, kvh):
    """Reference for the SINGLE-PASS RULE itself (running max over BULK
    scores, refine within delta). Exactness w.r.t. the rule, in fp64."""
    B, H, L, D = q.shape
    S = k.shape[2]
    g = H // kvh
    out = np.zeros((B, H, L, v.shape[3]), dtype=np.float64)
    frac = []
    for b in range(B):
        for h in range(H):
            kh = k[b, h // g].astype(np.float64)
            kqh = kq[b, h // g].astype(np.float64)
            vh = v[b, h // g].astype(np.float64)
            for i in range(L):
                lim = (S - L + i + 1) if causal else S
                bulk = (q[b, h, i].astype(np.float64) @ kqh[:lim].T) * scale
                exact = (q[b, h, i].astype(np.float64) @ kh[:lim].T) * scale
                run = np.maximum.accumulate(bulk)          # running max, BLASST-style
                refine = bulk >= run - delta
                score = np.where(refine, exact, bulk)
                frac.append(refine.mean())
                sk = float(sinks[h])
                m = max(score.max(), sk)
                w = np.exp(score - m)
                out[b, h, i] = (w @ vh[:lim]) / (w.sum() + np.exp(sk - m))
    return out, float(np.mean(frac))


def relf(a, b):
    d = np.linalg.norm(a.ravel() - b.ravel())
    n = np.linalg.norm(b.ravel())
    return float(d / n) if n else float(d)


def main():
    rng = np.random.default_rng(20260908)
    B, H, KVH, D, L, S = 1, 64, 8, 64, 48, 96
    scale = D ** -0.5
    res = {'shape': dict(B=B, H=H, KVH=KVH, D=D, L=L, S=S), 'scale': scale,
           'cases': []}
    for dtype in ('float32', 'bfloat16'):
        q = rng.normal(0, 1, (B, H, L, D)).astype(np.float32)
        k = rng.normal(0, 1, (B, KVH, S, D)).astype(np.float32)
        v = rng.normal(0, 1, (B, KVH, S, D)).astype(np.float32)
        sinks = rng.normal(0, 1, (H,)).astype(np.float32)
        qt = tc.tensor(q, dtype=dtype)
        kt = tc.tensor(k, dtype=dtype)
        vt = tc.tensor(v, dtype=dtype)
        st = tc.tensor(sinks, dtype=dtype)
        Rt, CB, BND = _tables(D, 4, KVH, True, qt.device.split(':')[0])
        kqt = _quantize_keys(kt, Rt, CB, BND)
        kq = kqt.float().numpy()
        # round-trip through the engine dtype so the reference sees the
        # exact bytes the kernel sees
        qn = qt.float().numpy(); kn = kt.float().numpy(); vn = vt.float().numpy()
        skn = st.float().numpy()

        # --- refine-all (delta huge) must equal dense attention with sinks
        big = 1e9
        t0 = time.perf_counter()
        out_all = tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, scale, big, True, st, False)
        tc.synchronize()
        wall_all = time.perf_counter() - t0
        ref = dense_ref(qn, kn, vn, skn, scale, True, KVH)
        a = out_all.float().numpy().astype(np.float64)
        res['cases'].append(dict(case='refine_all_vs_dense_sinks', dtype=dtype,
                                 max_abs=float(np.abs(a - ref).max()),
                                 rel_frobenius=relf(a, ref), wall_s=wall_all))

        # --- sinks ON vs sinks ZEROED: the sink must change the output
        zero = tc.tensor(np.zeros_like(sinks), dtype=dtype)
        out_zero = tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, scale, big, True, zero, False)
        out_none = tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, scale, big, True, None, False)
        az = out_zero.float().numpy().astype(np.float64)
        an = out_none.float().numpy().astype(np.float64)
        ref_zero = dense_ref(qn, kn, vn, np.zeros_like(skn), scale, True, KVH)
        ref_none = dense_ref(qn, kn, vn, None, scale, True, KVH)
        res['cases'].append(dict(case='sinks_zeroed_vs_dense', dtype=dtype,
                                 max_abs=float(np.abs(az - ref_zero).max()),
                                 rel_frobenius=relf(az, ref_zero)))
        res['cases'].append(dict(case='sinks_none_vs_dense_nosink', dtype=dtype,
                                 max_abs=float(np.abs(an - ref_none).max()),
                                 rel_frobenius=relf(an, ref_none)))
        res['cases'].append(dict(case='sink_effect_on_vs_zero', dtype=dtype,
                                 rel_frobenius=relf(a, az),
                                 max_abs=float(np.abs(a - az).max())))

        # --- a real delta: the SP kernel must match the fp64 rule reference
        delta = 3.0
        out_d, sel = tc._C.apa_selective_attention_sp(qt, kt, kqt, vt, scale, delta, True, st, True)
        bd = out_d.float().numpy().astype(np.float64)
        rref, frac = blend_ref(qn, kn, kq, vn, skn, scale, delta, True, KVH)
        selm = sel.numpy().astype(bool)
        elig2 = np.arange(S)[None, :] <= (S - L + np.arange(L))[:, None]
        elig = np.broadcast_to(elig2[None, None], selm.shape)
        res['cases'].append(dict(case='sp_delta_vs_fp64_rule', dtype=dtype, delta=delta,
                                 max_abs=float(np.abs(bd - rref).max()),
                                 rel_frobenius=relf(bd, rref),
                                 kernel_frac=float(selm[elig].mean()),
                                 reference_frac=frac,
                                 causal_leak=int((selm & ~elig).sum())))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
